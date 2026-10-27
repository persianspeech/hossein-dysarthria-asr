#!/usr/bin/env python3
"""Transcribe every Saremi reference clip with the local Whisper medium model.

Writes recorder/data/whisper_persian.json and whisper_persian.tsv.
Already transcribed ids are skipped, so the job can be resumed.

    pip install -r requirements.txt
    python3 whisper_transcribe.py --limit 5
    python3 whisper_transcribe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT.parent / "SaremiData" / "DataBase" / "opus"
PHSEQ = ROOT.parent / "SaremiData" / "DataBase" / "txt.phseq.data"
OUT_JSON = ROOT / "data" / "whisper_persian.json"
OUT_TSV = ROOT / "data" / "whisper_persian.tsv"
DEFAULT_MODEL = Path(
    "/mnt/1.8TB/backup/NE3-backend/AI_Services/whisper_file/models/whisper-medium-ASR-ct2-float16"
)


def utterance_ids() -> list[str]:
    ids = []
    for line in PHSEQ.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(line.split()[0])
    return ids


def load_done() -> dict[str, str]:
    if OUT_JSON.is_file():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {}


def save(done: dict[str, str], order: list[str]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".json.part")
    tmp.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write("id\tpersian\n")
        for utt in order:
            if utt in done:
                text = done[utt].replace("\t", " ").replace("\n", " ")
                f.write(f"{utt}\t{text}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--limit", type=int, default=0, help="transcribe at most N new files")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    if not args.model.is_dir():
        sys.exit(f"model not found: {args.model}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed. Run: pip install -r requirements.txt")

    order = utterance_ids()
    done = load_done()
    pending = [i for i in order if i not in done and (REF_DIR / f"{i}.opus").is_file()]
    if args.limit:
        pending = pending[: args.limit]
    print(f"done={len(done)} pending_this_run={len(pending)} model={args.model}")
    if not pending:
        save(done, order)
        print("nothing to do")
        return

    device = args.device
    compute = args.compute_type
    if device == "cpu" and compute == "float16":
        compute = "int8"
    model = WhisperModel(str(args.model), device=device, compute_type=compute)

    for n, utt in enumerate(pending, 1):
        path = REF_DIR / f"{utt}.opus"
        segments, info = model.transcribe(
            str(path),
            language="fa",
            beam_size=args.beam_size,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        done[utt] = text
        print(f"[{n}/{len(pending)}] {utt}: {text}")
        if n % 20 == 0:
            save(done, order)
    save(done, order)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
