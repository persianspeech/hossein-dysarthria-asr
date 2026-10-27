"""Portable paths, hashing, JSON and Persian scoring normalization."""
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def jsonl(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items), encoding='utf-8')


def normalize(text, metric=False):
    text = unicodedata.normalize('NFC', text).translate(str.maketrans({'ي':'ی', 'ك':'ک', 'ـ':''}))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    if metric:
        text = text.replace('\u200c', ' ').replace('\u200d', '')
        text = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else
                       ' ' if unicodedata.category(c)[0] in 'PS' else c for c in text)
    return re.sub(r'\s+', ' ', text).strip()


def status(stage, **kwargs):
    import os
    value = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, pid=os.getpid(), **kwargs)
    write(ROOT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
