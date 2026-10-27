"""Validate an isolated relocated Python 3.12 environment, then install small extras."""
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
venv=root/'.venv-reference'
python=venv/'bin/python'
assert python.exists()
# Refresh only the venv interpreter links/config for this machine; retain copied packages.
subprocess.run(['python3','-m','venv','--without-pip',str(venv)],check=True)
for path in (venv/'bin').iterdir():
    if path.is_file() and not path.is_symlink():
        data=path.read_bytes()
        if data.startswith(b'#!/home/gooran/agp-asr-nemotron-fa/.venv-shenava/bin/python'):
            first,rest=data.split(b'\n',1)
            path.write_bytes(('#!'+str(python)+'\n').encode()+rest)
packages=['jiwer==4.0.0','imageio-ffmpeg==0.6.0','numba-cuda==0.30.4','cuda-core==0.5.1',
          'numpy==2.2.6','nvidia-cuda-nvcc-cu12==12.6.85']
install=[str(python),'-m','pip','install']
if (root/'wheelhouse').exists():
    install+=['--no-index','--find-links',str(root/'wheelhouse')]
else:
    install+=['--timeout','120']
subprocess.run([*install,*packages],check=True)
probe='''import torch, json, importlib.metadata as m
from nemo.collections.asr.models import ASRModel
assert torch.cuda.is_available()
x=torch.randn(64,64,device='cuda',requires_grad=True)
(x.square().mean()).backward()
assert torch.isfinite(x.grad).all()
print(json.dumps(dict(cuda_available=True,gpu=torch.cuda.get_device_name(0),
  packages={n:m.version(n) for n in ['torch','torchaudio','nemo_toolkit','lightning','transformers','numpy','numba','jiwer','imageio-ffmpeg']})))
'''
result=subprocess.run([str(python),'-c',probe],check=True,text=True,capture_output=True)
print(result.stdout,flush=True)
(root/'environment_validation.json').write_text(result.stdout.splitlines()[-1]+'\n')
with (root/'environment.freeze.txt').open('w') as stream:
    subprocess.run([str(python),'-m','pip','freeze'],stdout=stream,check=True)
(root/'ENV_READY').touch()
