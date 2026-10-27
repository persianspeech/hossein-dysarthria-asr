"""Verify transferred bytes, decode complete utterances and freeze grouped splits."""
import argparse
import hashlib
import math
import random
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from common import ROOT, read, write, sha, jsonl, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = read(root / 'run_config.json')
    source = read(root / 'source_manifest.json')
    for item in source['files']:
        path = root / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
    labels = read(root / 'raw/labels.json')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio = root / 'dataset/audio'
    audio.mkdir(parents=True, exist_ok=True)

    def convert(item):
        path = root / item['path']
        data = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-f', 's16le',
                               '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-'],
                              check=True, capture_output=True).stdout
        pcm = np.frombuffer(data, dtype='<i2')
        seconds = len(pcm) / 16000
        if not len(pcm):
            return None, dict(id=item['id'], reason='empty decoded audio')
        rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768) ** 2)))
        text = normalize(labels[item['id']])
        if not .3 <= seconds <= config['max_audio_seconds'] or rms < 1e-6 or not text:
            return None, dict(id=item['id'], reason='duration/silence/empty-label', duration=seconds, rms=rms)
        filename = f"{item['id']}.flac"
        target = audio / filename
        sf.write(target, pcm, 16000, subtype='PCM_16')
        restored, rate = sf.read(target, dtype='int16')
        assert rate == 16000 and np.array_equal(pcm, restored), filename
        pcm_sha = hashlib.sha256(data).hexdigest()
        row = dict(id=item['id'], audio=f'audio/{filename}', text=text, text_raw=labels[item['id']],
                   duration=seconds, speaker_id='hossein', label_kind='asr_reference_prompt',
                   human_verified=False, pcm_sha256=pcm_sha, audio_sha256=sha(target),
                   source_sha256=item['sha256'], rms=rms,
                   peak=float(np.max(np.abs(pcm.astype(np.float64))) / 32768),
                   clipped_fraction=float(np.mean(np.abs(pcm.astype(np.int32)) >= 32760)))
        return row, None

    valid, rejected = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row, rejection in pool.map(convert, [f for f in source['files'] if f.get('id')]):
            (valid if row else rejected).append(row if row else rejection)
    assert len(valid) + len(rejected) == source['clips']
    # Transitive grouping prevents A~B by PCM and B~C by text crossing split boundaries.
    parent = list(range(len(valid)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(valid):
        for key in [('pcm', row['pcm_sha256']), ('text', normalize(row['text'], metric=True))]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, row in enumerate(valid):
        groups.setdefault(find(i), []).append(row)
    ordered = sorted(groups.values(), key=lambda group: min(x['id'] for x in group))
    random.Random(config['seed']).shuffle(ordered)
    # Assign whole groups to fixed count targets; deviations are reported explicitly.
    n = len(valid)
    targets = dict(train=round(n * config['train_fraction']),
                   validation=round(n * config['validation_fraction']))
    targets['test'] = n - sum(targets.values())
    splits = {name: [] for name in targets}
    for group in ordered:
        split = max(targets, key=lambda name: (targets[name] - len(splits[name])) / targets[name])
        group_id = min(r['id'] for r in group)
        for row in group:
            row.update(split=split, group_id=group_id)
        splits[split].extend(group)
    for left in splits:
        for right in splits:
            if left >= right:
                continue
            for field in ['pcm_sha256', 'group_id']:
                assert not {r[field] for r in splits[left]} & {r[field] for r in splits[right]}
            assert not {normalize(r['text'], True) for r in splits[left]} & {normalize(r['text'], True) for r in splits[right]}
    for name, items in splits.items():
        assert items
        jsonl(root / f'dataset/{name}.jsonl', sorted(items, key=lambda r: r['id']))
    report = dict(source_clips=source['clips'], accepted_clips=n, rejected=rejected,
                  total_hours=sum(r['duration'] for r in valid)/3600,
                  duration_seconds=dict(min=min(r['duration'] for r in valid), max=max(r['duration'] for r in valid)),
                  splits={name: dict(clips=len(items), hours=sum(r['duration'] for r in items)/3600,
                                     manifest_sha256=sha(root / f'dataset/{name}.jsonl')) for name, items in splits.items()},
                  source_manifest_sha256=sha(root/'source_manifest.json'),
                  exact_pcm_text_group_overlap=0, connected_groups=len(groups),
                  repeated_text_or_pcm_groups=sum(len(group)>1 for group in groups.values()),
                  label_kind=source['label_kind'], all_transferred_files_verified=True,
                  audio=dict(sample_rate=16000, channels=1, format='FLAC PCM16', full_utterances=True, trimmed=False),
                  split_scope=config['split_scope'],
                  limitations=['Single identified speaker; no unseen-speaker or independent-session claim.',
                               'Prompt labels originate from ASR of reference-speaker recordings, not verified Hossein transcripts.',
                               'No new manual audio/text validation; scores measure agreement with those prompts.',
                               'Only exact decoded PCM and normalized-text duplicates are controlled; near duplicates are not proven absent.',
                               'Historical training overlap of the base model with reference prompts is unknown.'])
    write(root / 'dataset/data_report.json', report)
    write(root / 'transfer_verification.json', dict(passed=True, source_files=len(source['files']),
                                                   source_manifest_sha256=sha(root/'source_manifest.json')))
    write(root/'dataset/audio_hashes.json', {r['audio']:r['audio_sha256'] for r in valid})
    print(__import__('json').dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
