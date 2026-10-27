"""Sanitize the selected inference export and validate a fresh load on both heads."""
import gc
import shutil
import tarfile
from pathlib import Path
import numpy as np
import torch
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf, open_dict
from common import ROOT, read, write, sha, rows, normalize, status
from train import audio


def main():
    result=read(ROOT/'final_results.json')
    completed=read(ROOT/'training_completed.json')
    assert sha(ROOT/'best-model.nemo')==completed['best_model_sha256']==result['selected_model_sha256']
    assert read(ROOT/'smoke.json')['passed'] and read(ROOT/'resume_verification.json')['passed']
    out=ROOT/'hf_model'
    out.mkdir(exist_ok=True)
    source=ASRModel.restore_from(str(ROOT/'best-model.nemo'),map_location='cpu')
    with open_dict(source.cfg):
        for key in ['train_ds','validation_ds','test_ds','optim','exp_manager']:
            if key in source.cfg:
                source.cfg[key]=None
    source.save_to(str(out/'shenava-hossein.nemo'))
    restored=ASRModel.restore_from(str(out/'shenava-hossein.nemo'),map_location='cpu')
    assert source.state_dict().keys()==restored.state_dict().keys()
    for name,value in source.state_dict().items():
        assert torch.equal(value,restored.state_dict()[name]),name
    del source
    gc.collect()
    with tarfile.open(out/'shenava-hossein.nemo') as archive:
        member=next(x for x in archive.getmembers() if x.name.endswith('model_config.yaml'))
        config=OmegaConf.create(archive.extractfile(member).read().decode())
        for key in ['train_ds','validation_ds','test_ds']:
            assert config.get(key) is None
        assert not any(name.endswith(('.webm','.flac','.wav','.pt')) for name in archive.getnames())
    restored=restored.cuda().eval()
    restored.encoder.set_default_att_context_size([70,13])
    examples=rows(ROOT/'dataset/test.jsonl')[:3]
    matches={}
    for head in ['ctc','rnnt']:
        decoding=OmegaConf.create({'strategy':'greedy'}) if head=='ctc' else OmegaConf.create(
            {'strategy':'greedy_batch','greedy':{'max_symbols':10,'use_cuda_graph_decoder':False}})
        restored.change_decoding_strategy(decoding,decoder_type=head)
        expected={p['id']:p['prediction'] for p in read(ROOT/f'evaluations/selected/test/{head}_predictions.json')}
        with torch.inference_mode():
            for row in examples:
                value=restored.transcribe([audio(row)],batch_size=1,num_workers=0,verbose=False,return_hypotheses=False)
                if isinstance(value,tuple):value=value[0]
                text=value[0].text if hasattr(value[0],'text') else str(value[0])
                assert text==expected[row['id']],(head,row['id'],text,expected[row['id']])
        matches[head]=len(examples)
    validation=dict(passed=True,all_weight_tensors_equal=True,clean_reload_prediction_matches=matches,
                    training_dataset_paths_removed=True,contains_optimizer=False,
                    selected_checkpoint_sha256=completed['best_model_sha256'],
                    exported_model_sha256=sha(out/'shenava-hossein.nemo'))
    write(out/'validation.json',validation)
    write(out/'evaluation.json',result)
    write(out/'data_report.json',read(ROOT/'dataset/data_report.json'))
    write(out/'training_config.json',read(ROOT/'run_config.json'))
    write(out/'training_summary.json',dict(selected_epoch=completed['best_epoch'],
          completed_epochs=completed['epoch'],completed_steps=completed['step'],
          audio_exposure_hours=completed['audio_hours'],parameters=completed['parameters'],
          all_parameters_trainable=True,base_model=read(ROOT/'run_config.json')['base_repo'],
          base_revision=read(ROOT/'run_config.json')['base_revision'],
          base_sha256=read(ROOT/'run_config.json')['base_sha256'],
          manifests=completed['identity']['manifests'],
          label_kind=read(ROOT/'run_config.json')['label_kind']))
    for name in ['transcribe.py','requirements.txt','environment.freeze.txt','LICENSE']:
        shutil.copyfile(ROOT/name,out/name)
    table=['| Head | Base WER | Tuned WER | Base CER | Tuned CER |','|---|---:|---:|---:|---:|']
    for head in ['ctc','rnnt']:
        a=result['metrics']['baseline']['test'][head]
        b=result['metrics']['selected']['test'][head]
        table.append(f"| {head} | {a['wer']:.2f}% | {b['wer']:.2f}% | {a['cer']:.2f}% | {b['cer']:.2f}% |")
    card='''---
language:
- fa
license: apache-2.0
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: Reza2kn/Shenava-Koochik-v1.5
base_model_relation: finetune
datasets:
- Hossein-Dysarthria-Net/hossein-persian-speech
tags:
- dysarthria
- apraxia
- persian
- fastconformer
- ctc
- rnnt
---

# Shenava Koochik — Hossein adaptation

مدل شنوا کوچک، سازگارشده با ۴۰۸ ضبط حسین. ۵۱ نمونه برای اعتبارسنجی و ۵۱
نمونه برای تست مستقل از انتخاب مدل کنار گذاشته شده‌اند. مدل شامل هر دو سر CTC و RNNT است.

Full-parameter adaptation of the native 114M FastConformer hybrid RNNT/CTC model.
The original vocabulary and tokenizer are preserved. The objective is 0.7 RNNT +
0.3 CTC with BF16 mixed precision, effective batch 8, AdamW and light SpecAugment.
There is no pseudo-label replacement of Hossein's recordings, truncation or silence trimming.
CTC is the primary head. The best **trained** checkpoint is selected only by CTC
validation WER, with CER as tie breaker, and five-epoch early stopping.

## Held-out test (same speaker, 51 utterances)

'''+'\n'.join(table)+f'''

Selected epoch: **{completed['best_epoch']}**. Completed epochs: {completed['epoch']}.
`evaluation.json` includes raw/normalized scores and paired bootstrap intervals.
WER can exceed 100%; lower is better. See the confidence intervals before making
claims about the size of any improvement.

## Label and scope limitations

Labels are ASR-derived **reference prompts**, not human-verified transcriptions of
Hossein's actual speech. The reference-speaker ASR labels were displayed while he
recorded repetitions. Scores measure prompt agreement and may be affected by prompt
errors, deviations and omitted/repeated words. The user's description of the speaker
is Dysarthria / Apraxia. No clinical diagnosis is established by this experiment.
This is single-speaker adaptation; it does not establish performance on other
speakers, unseen recording sessions, spontaneous speech or human-verified transcripts.
Exact PCM and normalized-text duplicates are disjoint across train/validation/test;
near-duplicate and historical base-training overlap are not established.

## Inference

Install the pinned environment (`environment.freeze.txt`; the tested runtime is
NeMo 3.0.0 and PyTorch 2.11.0+cu126). Authenticate for a private repository.

```bash
python transcribe.py recording.wav --head ctc
python transcribe.py recording.webm --head rnnt
```

The `.nemo` file bundles the tokenizer. `transcribe.py` reads complete audio via
FFmpeg, uses mono 16 kHz and context [70,13], and supports either head. Live
cache-aware streaming and long-recording chunking were not evaluated.
The inference archive contains no optimizer state or source recordings. A fresh
reload was checked for equality of all weight tensors and matching predictions
on three final-test files for each head (`validation.json`).

Source: https://github.com/persianspeech/hossein-dysarthria-asr
Dataset: https://huggingface.co/datasets/Hossein-Dysarthria-Net/hossein-persian-speech
Original base: https://huggingface.co/Reza2kn/Shenava-Koochik-v1.5
'''
    (out/'README.md').write_text(card,encoding='utf-8')
    write(out/'release_hashes.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='release_hashes.json'})
    write(ROOT/'release_validation.json',validation)
    status('release_ready',**validation)


if __name__=='__main__':
    main()
