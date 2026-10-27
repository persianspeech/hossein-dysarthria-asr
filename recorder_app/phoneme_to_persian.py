#!/usr/bin/env python3
"""Convert Saremi Festival phoneme sequences (txt.phseq.data) to Persian text.

The source has no word boundaries. Each utterance is turned into fully
vocalized Persian (short vowels as diacritics, geminates as tashdid) and
split into readable chunks with a small lexicon of frequent words plus a
syllable fallback.

Usage:
  python3 phoneme_to_persian.py
  python3 phoneme_to_persian.py --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / "SaremiData" / "DataBase" / "txt.phseq.data"
DEFAULT_TSV = ROOT / "data" / "persian_texts.tsv"
DEFAULT_JSON = ROOT / "data" / "persian_texts.json"

PHONEMES = {
    "pau", "b", "p", "t", "d", "k", "g", "gh", "@", "f", "v", "s", "z",
    "kh", "sh", "zh", "h", "j", "ch", "r", "m", "n", "l", "y",
    "a", "e", "o", "aa", "oo", "ee",
}
CONS = {
    "b": "ب", "p": "پ", "t": "ت", "d": "د", "k": "ک", "g": "گ",
    "gh": "غ", "@": "ء", "f": "ف", "v": "و", "s": "س", "z": "ز",
    "kh": "خ", "sh": "ش", "zh": "ژ", "h": "ه", "j": "ج", "ch": "چ",
    "r": "ر", "m": "م", "n": "ن", "l": "ل", "y": "ی",
}
VOWELS = {"a", "e", "o", "aa", "oo", "ee"}
LONG_V = {"aa": "ا", "ee": "ی", "oo": "و"}
DIAC = {"a": "\u064e", "e": "\u0650", "o": "\u064f", "aa": "\u064e", "ee": "\u0650", "oo": "\u064f"}
SUKUN = "\u0652"
TASHDID = "\u0651"
ZWNJ = "\u200c"

def _load_lexicon(path: Path) -> list[tuple[str, ...]]:
    words: list[tuple[str, ...]] = []
    seen: set[str] = set()
    if not path.exists():
        return words
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tup = tuple(line.split())
        if any(p not in PHONEMES for p in tup):
            continue
        key = " ".join(tup)
        if key in seen:
            continue
        seen.add(key)
        words.append(tup)
    words.sort(key=len, reverse=True)
    return words


LEXICON = _load_lexicon(ROOT / "lexicon_words.txt")
BY_FIRST: dict[str, list[tuple[str, ...]]] = {}
for _tup in LEXICON:
    BY_FIRST.setdefault(_tup[0], []).append(_tup)


def tokenize(raw: str) -> tuple[str, list[str]]:
    parts = raw.strip().split()
    if len(parts) < 2:
        raise ValueError(f"bad line: {raw[:80]!r}")
    utt_id, phones = parts[0], parts[1:]
    bad = sorted({p for p in phones if p not in PHONEMES})
    if bad:
        raise ValueError(f"{utt_id}: unknown phonemes {bad}")
    return utt_id, phones


def _carrier_needed(buf: list[str]) -> bool:
    return not buf or buf[-1] in (" ", ZWNJ)


def phones_to_letters(phones: list[str]) -> str:
    """Vocalized Persian for one word (no pau)."""
    out: list[str] = []
    i = 0
    n = len(phones)
    while i < n:
        p = phones[i]
        if p in VOWELS:
            _write_vowel(out, p, after_cons=False)
            i += 1
            continue
        letter = CONS[p]
        nxt = phones[i + 1] if i + 1 < n else None
        # geminate written as C C V (Festival: v a f f a -> فّ)
        if nxt == p and i + 2 < n and phones[i + 2] in VOWELS:
            out.append(letter)
            out.append(TASHDID)
            _write_vowel(out, phones[i + 2], after_cons=True)
            i += 3
            continue
        if nxt in VOWELS:
            out.append(letter)
            _write_vowel(out, nxt, after_cons=True)
            i += 2
            continue
        out.append(letter)
        if nxt in CONS:
            out.append(SUKUN)
        i += 1
    return "".join(out)


def _write_vowel(out: list[str], vow: str, after_cons: bool) -> None:
    if vow in LONG_V:
        if not after_cons and _carrier_needed(out):
            if vow == "aa":
                out.append("\u0622")
                return
            out.append("\u0627")
            out.append(DIAC[vow])
            out.append(LONG_V[vow])
            return
        out.append(DIAC[vow])
        out.append(LONG_V[vow])
        return
    if not after_cons:
        out.append("\u0627")
    out.append(DIAC[vow])


def _lex_len(phones: list[str], i: int) -> int:
    for tup in BY_FIRST.get(phones[i], ()):
        L = len(tup)
        if phones[i : i + L] == list(tup):
            return L
    return 0


def _syllable_spans(phones: list[str]) -> list[tuple[int, int]]:
    """Return [start, end) spans. A syllable is onset* + vowel + optional coda.

    A consonant is a coda when the following consonant (or end) shows it is
    not the onset of the next vowel.
    """
    n = len(phones)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < n:
        start = i
        while i < n and phones[i] not in VOWELS:
            i += 1
        if i >= n:
            if spans:
                a, _b = spans[-1]
                spans[-1] = (a, n)
            else:
                spans.append((start, n))
            break
        i += 1  # vowel
        if i < n and phones[i] not in VOWELS:
            nxt = phones[i + 1] if i + 1 < n else None
            if nxt is None or nxt not in VOWELS:
                i += 1  # coda
        spans.append((start, i))
    return spans


def _attach_hamza(text: str) -> str:
    """Festival '@' is the glottal stop, written as hamza on a seat.

    Word-initial hamza uses alef (آ / أ / إ). After a vowel letter the
    hamza sits on that letter instead of a separate ء.
    """
    hamza = "ء"
    alef = "ا"
    madda = "آ"
    waw = "و"
    ye = "ی"
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != hamza:
            out.append(ch)
            i += 1
            continue
        diac = text[i + 1] if i + 1 < len(text) and text[i + 1] in DIAC.values() else ""
        prev = out[-1] if out else ""
        at_boundary = prev in ("", " ", ZWNJ)
        if at_boundary:
            if diac == DIAC["aa"]:
                out.append(madda)
                i += 2  # hamza + fatha; alef of madda replaces the long vowel letter too
                if i < len(text) and text[i] == alef:
                    i += 1
                continue
            if diac == DIAC["o"] or diac == DIAC["oo"]:
                out.append(alef)
                out.append("ٔ")  # hamza above
            elif diac == DIAC["e"] or diac == DIAC["ee"]:
                out.append(alef)
                out.append("ٕ")  # hamza below
            else:
                out.append(alef)
                out.append("ٔ")
            i += 1
            continue
        # medial/final: put hamza on the previous long-vowel letter when possible
        if prev == alef:
            out.append("ٔ")
        elif prev == waw:
            out.append("ٔ")
        elif prev == ye:
            out.append("ٔ")
        else:
            out.append(hamza)
        i += 1
    return "".join(out)


def phonemes_to_persian(phones: list[str]) -> str:
    """Fully vocalized Persian, one string per utterance.

    txt.phseq.data does not mark word boundaries, so inserting spaces would
    split real words. Pauses (pau) become spaces. The glottal stop '@' is
    written as hamza on alef / vav / ye.
    """
    groups: list[list[str]] = []
    cur: list[str] = []
    for ph in phones:
        if ph == "pau":
            if cur:
                groups.append(cur)
                cur = []
            continue
        cur.append(ph)
    if cur:
        groups.append(cur)
    parts = [_attach_hamza(phones_to_letters(g)) for g in groups if g]
    return _tidy(" ".join(parts))


def _tidy(text: str) -> str:
    text = text.replace(SUKUN + " ", " ")
    if text.endswith(SUKUN):
        text = text[: -1]
    text = text.replace("\u0627\u064e\u0627", "\u0622")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def load_items(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, phones = tokenize(line)
        items.append({
            "id": utt_id,
            "phonemes": " ".join(phones),
            "text": phonemes_to_persian(phones),
        })
    return items


def convert_file(src: Path, tsv_path: Path, json_path: Path, limit: int = 0) -> list[dict]:
    items = load_items(src)
    if limit:
        items = items[:limit]
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("id\tpersian\tphonemes\n")
        for it in items:
            f.write(f"{it['id']}\t{it['text']}\t{it['phonemes']}\n")
    json_path.write_text(
        json.dumps({it["id"]: it["text"] for it in items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_TSV)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    items = convert_file(args.input, args.output, args.json, args.limit)
    print(f"wrote {len(items)} items")
    print(f"  tsv  {args.output}")
    print(f"  json {args.json}")
    for it in items[:12]:
        print(f"{it['id']}: {it['text']}")


if __name__ == "__main__":
    main()
