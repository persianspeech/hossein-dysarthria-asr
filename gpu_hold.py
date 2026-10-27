"""Keep an NVML handle open during setup; no reset or privileged changes."""
import subprocess
import time
from pathlib import Path

log = Path(__file__).resolve().parent / 'gpu-hold.log'
with log.open('a') as stream:
    for attempt in range(60):
        result = subprocess.run(['nvidia-smi', '-l', '30'], stdout=stream, stderr=stream)
        time.sleep(5)
