"""Run on the destination before any training or publication."""
from common import ROOT, read, rows, sha, write, normalize

source = read(ROOT/'source_manifest.json')
for item in source['files']:
    path = ROOT/item['path']
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
report = read(ROOT/'dataset/data_report.json')
splits = {}
for split in ['train','validation','test']:
    path = ROOT/f'dataset/{split}.jsonl'
    assert sha(path) == report['splits'][split]['manifest_sha256']
    splits[split] = rows(path)
    for row in splits[split]:
        assert sha(ROOT/'dataset'/row['audio']) == row['audio_sha256']
for left in splits:
    for right in splits:
        if left >= right:
            continue
        for field in ['pcm_sha256','group_id','source_sha256']:
            assert not {r[field] for r in splits[left]} & {r[field] for r in splits[right]}
        assert not {normalize(r['text'],True) for r in splits[left]} & {normalize(r['text'],True) for r in splits[right]}
write(ROOT/'transfer_verification.json',dict(passed=True,source_files=len(source['files']),
      decoded_files=sum(len(v) for v in splits.values()),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
      manifest_sha256={k:sha(ROOT/f'dataset/{k}.jsonl') for k in splits},cross_split_exact_overlap=0))
print('All raw bytes, decoded files, manifests and cross-split exact duplicates verified.')
