"""Full-parameter native hybrid adaptation; development-only checkpoint selection."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import copy
import fcntl
import gc
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import SpectrogramAugmentation
from omegaconf import OmegaConf, open_dict
from common import ROOT, read, write, rows, sha, normalize, status

CFG = read(ROOT / 'run_config.json')
BASE = ROOT / 'base' / CFG['base_filename']


def seed():
    random.seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    torch.cuda.manual_seed_all(CFG['seed'])


def rngstate():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def setrng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    torch.cuda.set_rng_state_all(value['cuda'])


def identity():
    return dict(base_sha256=sha(BASE), config_sha256=sha(ROOT/'run_config.json'),
                code_sha256=sha(Path(__file__)), common_sha256=sha(ROOT/'common.py'),
                environment_sha256=sha(ROOT/'environment.freeze.txt'),
                manifests={split:sha(ROOT/f'dataset/{split}.jsonl') for split in ['train','validation','test']})


def load_model(path=BASE, training=False):
    model = ASRModel.restore_from(str(path), map_location='cpu')
    model.encoder.set_default_att_context_size([70, 13])
    if training:
        for param in model.parameters():
            param.requires_grad_(True)
        model.ctc_loss_weight = CFG['ctc_weight']
        spec = dict(freq_masks=1, time_masks=2, freq_width=4, time_width=.02)
        with open_dict(model.cfg):
            model.cfg.aux_ctc.ctc_loss_weight = CFG['ctc_weight']
            model.cfg.spec_augment = OmegaConf.create({'_target_':'nemo.collections.asr.modules.SpectrogramAugmentation', **spec})
        model.spec_augmentation = SpectrogramAugmentation(**spec)
        model.joint.set_fuse_loss_wer(True, loss=model.loss, metric=model.wer)
        model.joint.set_fused_batch_size(1)
    return model.cuda().train(training)


def labels(model, items):
    unknown = []
    for row in items:
        row['labels'] = model.tokenizer.text_to_ids(row['text'])
        assert row['labels'], row['id']
        unknown_id = model.tokenizer.tokenizer.unk_id()
        if unknown_id in row['labels']:
            unknown.append(row['id'])
    assert not unknown, f'Unknown tokens require explicit review: {unknown}'
    return items


def audio(row):
    path = (ROOT/'dataset'/row['audio']).resolve()
    assert path.is_relative_to((ROOT/'dataset/audio').resolve())
    wave, rate = sf.read(path, dtype='float32')
    assert rate == 16000 and wave.ndim == 1 and np.isfinite(wave).all()
    return wave


def loss(model, row):
    wave = audio(row)
    x = torch.from_numpy(wave).unsqueeze(0).cuda()
    n = torch.tensor([len(wave)], device='cuda')
    y = torch.tensor([row['labels']], dtype=torch.long, device='cuda')
    u = torch.tensor([len(row['labels'])], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        encoded, length = model(input_signal=x, input_signal_length=n)
        # CTC alignment requires one frame per token plus repeated-token blanks.
        min_frames = len(row['labels']) + sum(a == b for a,b in zip(row['labels'],row['labels'][1:]))
        assert length.item() >= min_frames, f"CTC alignment impossible for {row['id']}"
        decoded, target_length, _ = model.decoder(targets=y, target_length=u)
        rnnt, _, _, _ = model.joint(encoder_outputs=encoded, decoder_outputs=decoded,
                                    encoder_lengths=length, transcripts=y,
                                    transcript_lengths=u, compute_wer=False)
        logits = model.ctc_decoder(encoder_output=encoded)
        ctc = model.ctc_loss(log_probs=logits.float(), targets=y,
                             input_lengths=length, target_lengths=u)
        value = (1 - CFG['ctc_weight']) * rnnt + CFG['ctc_weight'] * ctc
    assert torch.isfinite(value) and torch.isfinite(rnnt) and torch.isfinite(ctc), row['id']
    return value, float(rnnt.detach()), float(ctc.detach())


def update(model, opt, batch, lr):
    model.train()
    for group in opt.param_groups:
        group['lr'] = lr
    opt.zero_grad(set_to_none=True)
    values = []
    for row in batch:
        value, rnnt, ctc = loss(model, row)
        (value / len(batch)).backward()
        values.append([float(value.detach()), rnnt, ctc])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return dict(loss=np.mean(values, axis=0).tolist(), grad_norm=float(norm), lr=lr)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])


def save(path, model, opt, progress):
    temp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rngstate(),
                    progress=progress, identity=identity()), temp)
    temp.replace(path)


def restore(path, model, opt):
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['identity'] == identity(), 'Resume identity mismatch'
    model.load_state_dict(state['model'], strict=True)
    opt.load_state_dict(state['optimizer'])
    setrng(state['rng'])
    return state['progress']


def probes(model):
    names = ['encoder.layers.0.self_attn.linear_q.weight', 'decoder.prediction.embed.weight',
             'ctc_decoder.decoder_layers.0.weight']
    result = {name:param for name,param in model.named_parameters() if name in names}
    assert len(result) == 3
    return result


def smoke(resume=False):
    model = load_model(training=True)
    train = labels(model, rows(ROOT/'dataset/train.jsonl'))
    hardest = sorted(train, key=lambda r:r['duration']*len(r['labels']), reverse=True)[:3]
    opt = optimizer(model)
    if resume:
        restore(ROOT/'checkpoints/smoke.pt', model, opt)
        result = update(model, opt, hardest[:1], CFG['learning_rate'])
        expected = torch.load(ROOT/'checkpoints/smoke_expected.pt', map_location='cpu', weights_only=False)
        assert np.allclose(result['loss'], expected['result']['loss'], rtol=1e-5, atol=1e-5)
        for name, param in probes(model).items():
            assert torch.allclose(param.detach().cpu(), expected['probes'][name], rtol=1e-6, atol=1e-7), name
        write(ROOT/'resume_verification.json', dict(passed=True, separate_process=True,
              loss=result['loss'], expected_loss=expected['result']['loss'], optimizer_and_rng_restored=True))
        return
    before = {name:param.detach().cpu().clone() for name,param in probes(model).items()}
    results = [update(model,opt,[row],CFG['learning_rate']) for row in hardest]
    changed = {name: not torch.equal(before[name],param.detach().cpu()) for name,param in probes(model).items()}
    assert all(changed.values())
    save(ROOT/'checkpoints/smoke.pt',model,opt,dict(step=3))
    result = update(model,opt,hardest[:1],CFG['learning_rate'])
    torch.save(dict(result=result,probes={name:param.detach().cpu() for name,param in probes(model).items()}),ROOT/'checkpoints/smoke_expected.pt')
    write(ROOT/'smoke.json', dict(passed=True, all_parameters_trainable=True,
          parameters=sum(p.numel() for p in model.parameters()), probes_updated=changed,
          worst_clips=[dict(id=r['id'],duration=r['duration'],tokens=len(r['labels'])) for r in hardest],
          results=results, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def evaluate(model, split, tag, heads=('ctc','rnnt')):
    items = rows(ROOT/f'dataset/{split}.jsonl')
    model.eval()
    results = {}
    out = ROOT/'evaluations'/tag/split
    out.mkdir(parents=True, exist_ok=True)
    for head in heads:
        decoding = OmegaConf.create({'strategy':'greedy'}) if head == 'ctc' else OmegaConf.create(
            {'strategy':'greedy_batch','greedy':{'max_symbols':10,'use_cuda_graph_decoder':False}})
        model.change_decoding_strategy(decoding, decoder_type=head)
        predictions = []
        with torch.inference_mode():
            for row in items:
                value = model.transcribe([audio(row)],batch_size=1,num_workers=0,verbose=False,return_hypotheses=False)
                if isinstance(value,tuple):
                    value = value[0]
                raw = value[0].text if hasattr(value[0],'text') else str(value[0])
                reference, hypothesis = normalize(row['text'], True), normalize(raw, True)
                words = jiwer.process_words(reference,hypothesis)
                chars = jiwer.process_characters(reference,hypothesis)
                predictions.append(dict(id=row['id'],reference=row['text'],prediction=raw,
                    reference_normalized=reference,prediction_normalized=hypothesis,
                    word_errors=words.substitutions+words.deletions+words.insertions,
                    reference_words=words.hits+words.substitutions+words.deletions,
                    char_errors=chars.substitutions+chars.deletions+chars.insertions,
                    reference_chars=chars.hits+chars.substitutions+chars.deletions))
        metrics = dict(n=len(items),
            wer=100*sum(p['word_errors'] for p in predictions)/sum(p['reference_words'] for p in predictions),
            cer=100*sum(p['char_errors'] for p in predictions)/sum(p['reference_chars'] for p in predictions),
            raw_wer=100*jiwer.wer([p['reference'] for p in predictions],[p['prediction'] for p in predictions]),
            raw_cer=100*jiwer.cer([p['reference'] for p in predictions],[p['prediction'] for p in predictions]))
        assert abs(metrics['wer']-100*jiwer.wer([p['reference_normalized'] for p in predictions],
                   [p['prediction_normalized'] for p in predictions])) < 1e-8
        write(out/f'{head}_predictions.json',predictions)
        results[head] = metrics
    write(out/'metrics.json',results)
    status('evaluation',tag=tag,split=split,metrics=results)
    return results


def train():
    assert read(ROOT/'smoke.json')['passed'] and read(ROOT/'resume_verification.json')['passed']
    model = load_model(training=True)
    train_items = labels(model, rows(ROOT/'dataset/train.jsonl'))
    opt = optimizer(model)
    last = ROOT/'checkpoints/last.pt'
    progress = dict(epoch=0,next_batch=0,step=0,best=None,best_epoch=None,bad=0,audio_hours=0.)
    if last.exists():
        progress = restore(last,model,opt)
    else:
        evaluate(model,'validation','baseline',heads=('ctc','rnnt'))
        save(last,model,opt,progress)
    batches_per_epoch = math.ceil(len(train_items)/CFG['effective_batch'])
    total_steps = batches_per_epoch*CFG['max_epochs']
    history = read(ROOT/'history.json') if (ROOT/'history.json').exists() else []
    start = time.monotonic()
    for epoch in range(progress['epoch'],CFG['max_epochs']):
        order = list(range(len(train_items)))
        random.Random(CFG['seed']+epoch).shuffle(order)
        batches = [order[i:i+CFG['effective_batch']] for i in range(0,len(order),CFG['effective_batch'])]
        first = progress['next_batch'] if epoch == progress['epoch'] else 0
        for batch_id in range(first,len(batches)):
            step = progress['step']+1
            lr = CFG['learning_rate']*min(1.,step/CFG['warmup_steps'])*(.1+.9*.5*(1+math.cos(math.pi*step/total_steps)))
            batch = [train_items[i] for i in batches[batch_id]]
            values = update(model,opt,batch,lr)
            progress.update(epoch=epoch,next_batch=batch_id+1,step=step,
                            audio_hours=progress['audio_hours']+sum(r['duration'] for r in batch)/3600)
            if step % 10 == 0:
                status('training',epoch=epoch+1,step=step,max_steps=total_steps,
                       audio_hours=progress['audio_hours'],elapsed_seconds=time.monotonic()-start,**values)
            if step % 25 == 0:
                save(last,model,opt,progress)
        metrics = evaluate(model,'validation',f'epoch-{epoch+1}',heads=('ctc',))
        score = [metrics['ctc']['wer'],metrics['ctc']['cer']]
        if progress['best'] is None or score < progress['best']:
            progress.update(best=score,best_epoch=epoch+1,bad=0)
            # Inference export uses the unmodified native module structure.
            model.save_to(str(ROOT/'best-model.nemo'))
            save(ROOT/'checkpoints/best.pt',model,opt,progress)
        else:
            progress['bad'] += 1
        history.append(dict(epoch=epoch+1,step=progress['step'],metrics=metrics,best_epoch=progress['best_epoch']))
        write(ROOT/'history.json',history)
        progress.update(epoch=epoch+1,next_batch=0)
        save(last,model,opt,progress)
        if progress['bad'] >= CFG['patience']:
            break
    write(ROOT/'training_completed.json',dict(**progress,identity=identity(),
          elapsed_seconds=time.monotonic()-start,all_parameters_trainable=True,
          best_model_sha256=sha(ROOT/'best-model.nemo'),parameters=sum(p.numel() for p in model.parameters())))
    status('training_complete',**progress)


def final():
    completed = read(ROOT/'training_completed.json')
    assert completed['identity'] == identity()
    assert sha(ROOT/'best-model.nemo') == completed['best_model_sha256']
    # First and only final-test evaluation; selection is already sealed.
    metrics = {}
    for tag,path in [('baseline',BASE),('selected',ROOT/'best-model.nemo')]:
        model = load_model(path)
        metrics[tag] = dict(test=evaluate(model,'test',tag),
                            validation=evaluate(model,'validation',tag+'-final'))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    comparison = {}
    rng = np.random.default_rng(CFG['seed'])
    for head in ['ctc','rnnt']:
        a = read(ROOT/f'evaluations/baseline/test/{head}_predictions.json')
        b = read(ROOT/f'evaluations/selected/test/{head}_predictions.json')
        assert [r['id'] for r in a] == [r['id'] for r in b]
        denom = np.array([r['reference_words'] for r in a])
        delta = np.array([y['word_errors']-x['word_errors'] for x,y in zip(a,b)])
        samples = rng.integers(0,len(a),size=(2000,len(a)))
        differences = 100*delta[samples].sum(1)/denom[samples].sum(1)
        comparison[head] = dict(wer_delta_percentage_points=100*delta.sum()/denom.sum(),
                paired_bootstrap_95ci=np.quantile(differences,[.025,.975]).tolist(),resamples=2000)
    write(ROOT/'final_results.json',dict(metrics=metrics,comparison=comparison,
          selected_epoch=completed['best_epoch'],selected_model_sha256=completed['best_model_sha256'],
          test_used_for_selection=False,primary_head='ctc',label_kind=CFG['label_kind'],
          test_scope='51 held-out utterances by the same speaker; prompt-label agreement, not human-verified transcript accuracy'))
    status('evaluated',metrics=metrics,comparison=comparison)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','smoke-resume','train','final'])
    args = parser.parse_args()
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    lock = (ROOT/'.gpu-job.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available(), 'CUDA not available'
    assert sha(BASE) == CFG['base_sha256']
    assert read(ROOT/'transfer_verification.json')['passed']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed()
    if args.mode == 'smoke':
        smoke()
    elif args.mode == 'smoke-resume':
        smoke(resume=True)
    elif args.mode == 'train':
        train()
    else:
        final()


if __name__ == '__main__':
    main()
