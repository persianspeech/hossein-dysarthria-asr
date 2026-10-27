#!/usr/bin/env python3
"""Transcribe every Saremi opus clip with the local Shenava service.

Already written ids are skipped, so the job can be stopped and resumed.

    python3 transcribe_saremi.py
    python3 transcribe_saremi.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import jwt

ROOT = Path(__file__).resolve().parent
OPUS = ROOT.parent / "SaremiData" / "DataBase" / "opus"
OUT = ROOT / "data" / "shenava_texts.json"
BASE = os.environ.get("SHENAVA_URL", "http://127.0.0.1:8004")
KEY = os.environ.get(
    "JWT_SIGNING_KEY",
    "dev-only-shared-jwt-signing-key-32bytes-minimum-change-in-prod",
)


def token() -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "token_type": "access",
            "user_id": 1,
            "username": "batch",
            "role": "admin",
            "iat": now,
            "exp": now + 7 * 24 * 3600,
        },
        KEY,
        algorithm="HS256",
    )


def load() -> dict[str, str]:
    if not OUT.is_file():
        return {}
    data = json.loads(OUT.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def save(texts: dict[str, str]) -> None:
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(texts, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def transcribe(path: Path, bearer: str) -> str:
    body = json.dumps(
        {"path": f"/data/opus/{path.name}", "model": "koochik", "language": "fa", "timestamps": False}
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/transcribe/path",
        data=body,
        headers={"Authorization": "Bearer " + bearer, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as res:
        payload = json.loads(res.read().decode())
    return str(payload.get("text") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    files = sorted(p for p in OPUS.glob("*.opus") if p.is_file())
    texts = load()
    pending = [p for p in files if p.stem not in texts]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(files)} clips, {len(texts)} done, {len(pending)} to go", flush=True)
    bearer = token()
    for index, path in enumerate(pending, 1):
        started = time.time()
        try:
            text = transcribe(path, bearer)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            print(f"FAIL {path.stem} HTTP {exc.code} {detail}", flush=True)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {path.stem} {exc}", flush=True)
            continue
        texts[path.stem] = text
        if index % 10 == 0 or index == len(pending):
            save(texts)
        print(f"{index}/{len(pending)} {path.stem} {time.time() - started:.1f}s {text}", flush=True)
    save(texts)
    print(f"wrote {len(texts)} lines to {OUT}", flush=True)


if __name__ == "__main__":
    main()
