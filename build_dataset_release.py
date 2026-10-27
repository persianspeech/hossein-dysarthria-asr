"""Build a standard Hugging Face parquet dataset with embedded lossless audio."""
import argparse
import shutil
from pathlib import Path
from datasets import Audio, Dataset
from common import ROOT, read, rows, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    report = read(root/'dataset/data_report.json')
    out = root/'hf_dataset'
    (out/'data').mkdir(parents=True, exist_ok=True)
    for split in ['train','validation','test']:
        items = rows(root/f'dataset/{split}.jsonl')
        assert sha(root/f'dataset/{split}.jsonl') == report['splits'][split]['manifest_sha256']
        release = []
        for row in items:
            path = root/'dataset'/row['audio']
            assert sha(path) == row['audio_sha256']
            release.append({**{k:v for k,v in row.items() if k not in ['audio','split']},
                            'audio':dict(bytes=path.read_bytes(),path=path.name)})
        dataset = Dataset.from_list(release).cast_column('audio', Audio(sampling_rate=16000))
        dataset.to_parquet(str(out/f'data/{split}-00000-of-00001.parquet'))
    shutil.copyfile(root/'dataset/data_report.json',out/'data_report.json')
    shutil.copyfile(root/'source_manifest.json',out/'source_manifest.json')
    from update_release_cards import render
    render(root, 'dataset')
    print(str(out),flush=True)


if __name__ == '__main__':
    main()
