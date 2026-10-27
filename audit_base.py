"""CPU audit of the pinned model, native vocabulary and all existing labels."""
import argparse
from pathlib import Path
from nemo.collections.asr.models import ASRModel
from common import ROOT, read, rows, sha, write, normalize

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=ROOT)
parser.add_argument('--model',type=Path)
args=parser.parse_args()
config=read(args.root/'run_config.json')
path=args.model or args.root/'base'/config['base_filename']
assert sha(path)==config['base_sha256']
model=ASRModel.restore_from(str(path),map_location='cpu')
assert all(hasattr(model,k) for k in ['encoder','decoder','joint','ctc_decoder','ctc_loss','loss'])
unknown=model.tokenizer.tokenizer.unk_id()
counts={}
bad=[]
for split in ['train','validation','test']:
    items=rows(args.root/f'dataset/{split}.jsonl')
    tokens=[model.tokenizer.text_to_ids(row['text']) for row in items]
    bad += [row['id'] for row,ids in zip(items,tokens) if not ids or unknown in ids]
    counts[split]=dict(clips=len(items),tokens=sum(map(len,tokens)),max_tokens=max(map(len,tokens)))
assert not bad,bad
out=dict(passed=True,base_sha256=sha(path),native_tokenizer_preserved=True,
         parameters=sum(p.numel() for p in model.parameters()),unknown_id=unknown,
         unknown_token_clips=bad,splits=counts,hybrid_modules_present=True)
write(args.root/'base_audit.json',out)
print(__import__('json').dumps(out),flush=True)
