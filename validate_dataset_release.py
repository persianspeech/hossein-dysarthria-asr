"""Round-trip all embedded audio/labels through the standard datasets parquet reader."""
import argparse
import hashlib
from pathlib import Path
from datasets import load_dataset, Audio
from common import ROOT, read, rows, write

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=ROOT)
args=parser.parse_args()
root=args.root
files={name:str(root/f'hf_dataset/data/{name}-00000-of-00001.parquet')
       for name in ['train','validation','test']}
dataset=load_dataset('parquet',data_files=files)
counts={}
for name in files:
    expected={row['id']:row for row in rows(root/f'dataset/{name}.jsonl')}
    decoded=dataset[name].cast_column('audio',Audio(decode=False))
    assert len(decoded)==len(expected)
    for row in decoded:
        original=expected.pop(row['id'])
        assert hashlib.sha256(row['audio']['bytes']).hexdigest()==original['audio_sha256']
        assert row['text']==original['text'] and row['text_raw']==original['text_raw']
        assert row['human_verified'] is False and row['label_kind']=='asr_reference_prompt'
    assert not expected
    counts[name]=len(decoded)
write(root/'dataset_release_validation.json',dict(passed=True,standard_datasets_loader=True,
      all_embedded_audio_sha256_verified=True,all_labels_preserved=True,counts=counts))
print(counts,flush=True)
