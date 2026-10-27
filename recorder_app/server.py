#!/usr/bin/env python3
"""Recording UI for the Saremi reference corpus.

Login (users.json), play a reference opus clip, show its Persian line,
record a take, listen back, delete or confirm. Takes are stored as

    recordings/<username>/<id>.webm

    python3 server.py
    open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import os
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from hossein_live import handle as handle_live

ROOT = Path(__file__).resolve().parent
REF_DIR = Path(os.environ.get("REF_DIR", ROOT.parent / "SaremiData" / "DataBase" / "opus"))
TEXT_JSON = Path(os.environ.get("TEXT_JSON", ROOT / "data" / "persian_texts.json"))
PHSEQ = Path(os.environ.get("PHSEQ", ROOT.parent / "SaremiData" / "DataBase" / "txt.phseq.data"))
REC_ROOT = Path(os.environ.get("REC_ROOT", ROOT / "recordings"))
USERS_PATH = Path(os.environ.get("USERS_PATH", ROOT / "users.json"))
STATIC = ROOT / "static"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,32}$")

USERS = {u["username"]: u["password"] for u in json.loads(USERS_PATH.read_text(encoding="utf-8"))["users"]}
TEXTS = json.loads(TEXT_JSON.read_text(encoding="utf-8"))
CATALOG: list[dict] = []
for _line in PHSEQ.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if not _line:
        continue
    _id = _line.split()[0]
    if _id in TEXTS and (REF_DIR / f"{_id}.opus").is_file():
        CATALOG.append({"id": _id, "text": TEXTS[_id]})
BY_ID = {it["id"]: it for it in CATALOG}
SESSIONS: dict[str, str] = {}


def recorded_ids(username: str) -> set[str]:
    folder = REC_ROOT / username
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.glob("*.webm") if p.is_file() and p.stat().st_size > 0}

def _ctype(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".woff2": "font/woff2",
        ".svg": "image/svg+xml",
    }.get(path.suffix, "application/octet-stream")


class Handler(BaseHTTPRequestHandler):
    server_version = "SaremiRecorder/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _user(self) -> str | None:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return None
        return SESSIONS.get(header[7:].strip())

    def do_GET(self) -> None:  # noqa: N802
        self._route(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._route(include_body=False)

    def _route(self, include_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/login"):
            self._file(STATIC / "login.html")
            return
        if path == "/test-voice":
            self._file(STATIC / "hossein-test.html")
            return
        if handle_live(self, path, "GET"):
            return
        if path == "/app":
            self._file(STATIC / "app.html")
            return
        if path.startswith("/static/"):
            target = (STATIC / path[len("/static/"):]).resolve()
            if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
                self._json(404, {"error": "not found"})
                return
            self._file(target)
            return
        if path == "/api/items":
            user = self._user()
            if not user:
                self._json(401, {"error": "login required"})
                return
            done = recorded_ids(user)
            items = [{"id": it["id"], "text": it["text"], "recorded": it["id"] in done} for it in CATALOG]
            self._json(200, {"user": user, "total": len(items), "done": sum(1 for it in items if it["recorded"]), "items": items})
            return
        if path == "/api/reference":
            self._audio(parsed, reference=True)
            return
        if path == "/api/take":
            self._audio(parsed, reference=False)
            return
        self._json(404, {"error": "not found"})

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._json(404, {"error": "missing file"})
            return
        self._send(200, path.read_bytes(), _ctype(path))

    def _audio(self, parsed, reference: bool) -> None:
        user = self._user()
        if not user:
            self._json(401, {"error": "login required"})
            return
        utt = parse_qs(parsed.query).get("id", [""])[0]
        if utt not in BY_ID:
            self._json(404, {"error": "unknown id"})
            return
        if reference:
            target = REF_DIR / f"{utt}.opus"
            content_type = "audio/ogg; codecs=opus"
        else:
            target = REC_ROOT / user / f"{utt}.webm"
            content_type = "audio/webm"
        if not target.is_file():
            self._json(404, {"error": "no recording"})
            return
        self._send_file(target, content_type)

    def _send_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        extra: dict[str, str] = {}
        header = self.headers.get("Range", "")
        if header.startswith("bytes="):
            spec = header[6:].split(",", 1)[0].strip()
            left, _, right = spec.partition("-")
            try:
                if left == "":
                    length = int(right)
                    start = max(0, size - length)
                else:
                    start = int(left)
                    end = int(right) if right else size - 1
            except ValueError:
                start, end = 0, size - 1
            else:
                if start < 0 or start >= size or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if handle_live(self, parsed.path, "POST"):
            return
        if parsed.path == "/api/login":
            try:
                body = self._read_json()
            except json.JSONDecodeError:
                self._json(400, {"error": "bad json"})
                return
            username = str(body.get("username", "")).strip()
            password = str(body.get("password", ""))
            if USERS.get(username) != password:
                self._json(401, {"error": "نام کاربری یا رمز نادرست است"})
                return
            token = secrets.token_urlsafe(24)
            SESSIONS[token] = username
            (REC_ROOT / username).mkdir(parents=True, exist_ok=True)
            self._json(200, {"token": token, "username": username})
            return
        if parsed.path == "/api/logout":
            header = self.headers.get("Authorization", "")
            if header.startswith("Bearer "):
                SESSIONS.pop(header[7:].strip(), None)
            self._json(200, {"ok": True})
            return
        if parsed.path == "/api/take":
            user = self._user()
            if not user:
                self._json(401, {"error": "login required"})
                return
            utt = parse_qs(parsed.query).get("id", [""])[0]
            if utt not in BY_ID:
                self._json(400, {"error": "bad id"})
                return
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n <= 0 or n > 25_000_000:
                self._json(400, {"error": "empty or too large"})
                return
            blob = self.rfile.read(n)
            if len(blob) < 100:
                self._json(400, {"error": "recording too short"})
                return
            folder = REC_ROOT / user
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f".{utt}.webm.part"
            tmp.write_bytes(blob)
            tmp.replace(folder / f"{utt}.webm")
            self._json(200, {"ok": True, "id": utt, "bytes": (folder / f"{utt}.webm").stat().st_size})
            return
        self._json(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/take":
            self._json(404, {"error": "not found"})
            return
        user = self._user()
        if not user:
            self._json(401, {"error": "login required"})
            return
        utt = parse_qs(parsed.query).get("id", [""])[0]
        if not ID_RE.match(utt or ""):
            self._json(400, {"error": "bad id"})
            return
        target = REC_ROOT / user / f"{utt}.webm"
        if target.is_file():
            target.unlink()
        self._json(200, {"ok": True, "id": utt})


def main() -> None:
    REC_ROOT.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Saremi recorder: http://{HOST}:{PORT}")
    print(f"  references: {REF_DIR} ({len(CATALOG)} items)")
    print(f"  recordings: {REC_ROOT}")
    print(f"  users: {', '.join(USERS)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
