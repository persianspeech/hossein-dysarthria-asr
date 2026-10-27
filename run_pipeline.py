"""Durable stage runner with an exclusive lock and explicit failure records."""
import argparse
import fcntl
import os
import subprocess
import sys
import traceback
from pathlib import Path
from common import ROOT, write, read
from datetime import datetime, timezone


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python',default='.venv-reference/bin/python')
    args=parser.parse_args()
    os.chdir(ROOT)
    # Retain the venv executable path: resolving its symlink would bypass the venv.
    python=str(ROOT/args.python) if not os.path.isabs(args.python) else args.python
    lock=(ROOT/'.pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN','1')
    os.environ.setdefault('OMP_NUM_THREADS','4')
    # Keep NVVM, libdevice and nvJitLink aligned with torch's CUDA 12.6;
    # the host also has a system CUDA 13 toolkit installed.
    nvidia=Path(python).parent.parent/'lib/python3.12/site-packages/nvidia'
    nvcc=nvidia/'cuda_nvcc'
    if nvcc.is_dir():
        os.environ['CUDA_HOME']=str(nvcc)
        os.environ['CUDA_PATH']=str(nvcc)
        libraries=[nvcc/'nvvm/lib64',nvidia/'nvjitlink/lib',nvidia/'cuda_runtime/lib']
        os.environ['LD_LIBRARY_PATH']=':'.join(str(p) for p in libraries)+':'+os.environ.get('LD_LIBRARY_PATH','')
    stage='starting'
    def record(state,**kwargs):
        write(ROOT/'pipeline_status.json',dict(utc=datetime.now(timezone.utc).isoformat(),
              stage=stage,state=state,pid=os.getpid(),python=python,**kwargs))
    try:
        stage='verify_data';record('running')
        subprocess.run([python,'verify_data.py'],check=True)
        with (ROOT/'environment.freeze.txt').open('w') as stream:
            subprocess.run([python,'-m','pip','freeze'],stdout=stream,check=True)
        stages=[('smoke',['train.py','smoke'],'smoke.json'),
                ('resume_verification',['train.py','smoke-resume'],'resume_verification.json'),
                ('training',['train.py','train'],'training_completed.json'),
                ('final_evaluation',['train.py','final'],'final_results.json'),
                ('build_model_release',['build_model_release.py'],None)]
        for name,command,marker in stages:
            stage=name
            if marker and (ROOT/marker).exists():
                record('skipped_existing')
                continue
            record('running')
            subprocess.run([python,'-u',*command],check=True)
        stage='complete';record('complete',model_release_ready=True,
            selected_epoch=read(ROOT/'training_completed.json')['best_epoch'])
    except Exception as exc:
        record('failed',error_class=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
