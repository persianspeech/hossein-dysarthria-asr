"""Read-only checkpoint replay diagnostics; never modifies training checkpoints."""
import json
import fcntl
import torch
import numpy as np
import train as t
from common import ROOT, rows, write

lock=(ROOT/'.gpu-job.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True
t.seed()
model=t.load_model(training=True)
items=t.labels(model,rows(ROOT/'dataset/train.jsonl'))
hardest=sorted(items,key=lambda r:r['duration']*len(r['labels']),reverse=True)[:3]
opt=t.optimizer(model)
t.restore(ROOT/'checkpoints/smoke.pt',model,opt)
result=t.update(model,opt,hardest[:1],t.CFG['learning_rate'])
expected=torch.load(ROOT/'checkpoints/smoke_expected.pt',map_location='cpu',weights_only=False)
diff={name:float((param.detach().cpu()-expected['probes'][name]).abs().max()) for name,param in t.probes(model).items()}
out=dict(actual=result,expected=expected['result'],loss_delta=(np.array(result['loss'])-expected['result']['loss']).tolist(),
         max_probe_weight_difference=diff)
write(ROOT/'resume_diagnostic.json',out)
print(json.dumps(out),flush=True)
