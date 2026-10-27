"""Snapshot only Hossein's recordings and their displayed prompt labels."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    assert not output.exists(), 'Choose a new immutable snapshot directory'
    files = sorted((source / 'recordings/hossein').glob('*.webm'))
    labels = json.loads((source / 'data/shenava_texts.json').read_text())
    assert files and all(labels.get(p.stem, '').strip() for p in files)
    audio = output / 'raw/hossein'
    audio.mkdir(parents=True)
    inventory = []
    for path in files:
        before = sha(path)
        dest = audio / path.name
        shutil.copyfile(path, dest)
        assert sha(dest) == before == sha(path), f'Changed during copy: {path.name}'
        inventory.append(dict(path=dest.relative_to(output).as_posix(), sha256=before,
                              bytes=dest.stat().st_size, id=path.stem))
    selected = {p.stem: labels[p.stem] for p in files}
    target = output / 'raw/labels.json'
    target.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    inventory.append(dict(path='raw/labels.json', sha256=sha(target), bytes=target.stat().st_size))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), clips=len(files), files=inventory,
                  label_kind='ASR-derived reference prompt, not verified transcription of Hossein',
                  label_source='recorder/data/shenava_texts.json',
                  full_source_labels_sha256=sha(source / 'data/shenava_texts.json'),
                  provenance='transcribe_saremi.py transcribes reference Saremi audio with Shenava; '
                             'docker-compose.yml displays those texts while recording repetitions.',
                  excluded=['users.json', 'shenava.env', 'reference-speaker audio', 'unrecorded prompts'])
    (output / 'source_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(clips=len(files), bytes=sum(p['bytes'] for p in inventory),
                          manifest_sha256=sha(output / 'source_manifest.json'))))


if __name__ == '__main__':
    main()
