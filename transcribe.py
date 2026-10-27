"""Native NeMo inference, preserving the full recording and bundled tokenizer."""
import argparse
import shutil
import subprocess
import numpy as np
import torch
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--model',help='Local .nemo checkpoint')
    parser.add_argument('--repo',default='Hossein-Dysarthria-Net/Shenava-Koochik-Hossein')
    parser.add_argument('--revision')
    parser.add_argument('--head',choices=['ctc','rnnt'],default='ctc')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.model:
        checkpoint=args.model
    else:
        from huggingface_hub import hf_hub_download
        checkpoint=hf_hub_download(args.repo,'shenava-hossein.nemo',revision=args.revision)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    data=subprocess.run([ffmpeg,'-v','error','-i',args.audio,'-f','f32le','-ar','16000','-ac','1','-'],
                        check=True,capture_output=True).stdout
    wave=np.frombuffer(data,dtype='<f4').copy()
    assert len(wave) and np.isfinite(wave).all()
    model=ASRModel.restore_from(checkpoint,map_location='cpu').to(args.device).eval()
    model.encoder.set_default_att_context_size([70,13])
    decoding=OmegaConf.create({'strategy':'greedy'}) if args.head=='ctc' else OmegaConf.create(
        {'strategy':'greedy_batch','greedy':{'max_symbols':10,'use_cuda_graph_decoder':False}})
    model.change_decoding_strategy(decoding,decoder_type=args.head)
    with torch.inference_mode():
        result=model.transcribe([wave],batch_size=1,num_workers=0,verbose=False,return_hypotheses=False)
    if isinstance(result,tuple):result=result[0]
    print(result[0].text if hasattr(result[0],'text') else result[0])


if __name__=='__main__':
    main()
