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
    from update_release_cards import render
    render(ROOT, 'model')
    write(ROOT/'release_validation.json',validation)
    status('release_ready',**validation)


if __name__=='__main__':
    main()
