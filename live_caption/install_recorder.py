"""Install additive recorder integration, checking exact insertion anchors first."""
import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


def replace_once(text, before, after):
    assert text.count(before) == 1, f'Unexpected recorder source around {before[:60]!r}'
    return text.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recorder', type=Path, required=True)
    parser.add_argument('--asr-url', required=True)
    args = parser.parse_args()
    root = args.recorder.resolve()
    source = Path(__file__).resolve().parent
    server = (root/'server.py').read_text(encoding='utf-8')
    app = (root/'static/app.html').read_text(encoding='utf-8')
    login = (root/'static/login.js').read_text(encoding='utf-8')
    docker = (root/'Dockerfile').read_text(encoding='utf-8')
    compose = (root/'docker-compose.yml').read_text(encoding='utf-8')
    assert 'from hossein_live import handle as handle_live' not in server, 'Already installed'
    server = replace_once(server, 'from urllib.parse import parse_qs, urlparse',
                          'from urllib.parse import parse_qs, urlparse\nfrom hossein_live import handle as handle_live')
    server = replace_once(server, '        if path == "/app":',
                          '        if path == "/test-voice":\n            self._file(STATIC / "hossein-test.html")\n            return\n        if handle_live(self, path, "GET"):\n            return\n        if path == "/app":')
    server = replace_once(server, '        if parsed.path == "/api/login":',
                          '        if handle_live(self, parsed.path, "POST"):\n            return\n        if parsed.path == "/api/login":')
    app = replace_once(app, '      <div class="rail-tools">',
                       '      <div class="rail-tools">\n        <a href="/test-voice" class="ghost" style="color:var(--accent);text-decoration:none;min-height:48px">تست صدای حسین</a>')
    login = 'const destination = new URLSearchParams(location.search).get("next") === "/test-voice" ? "/test-voice" : "/app";\n' + login.replace('location.replace("/app")', 'location.replace(destination)')
    docker = replace_once(docker, 'COPY server.py users.json ./', 'COPY server.py hossein_live.py users.json ./')
    assert args.asr_url.startswith('http://') and '\n' not in args.asr_url
    compose = replace_once(compose, '    environment:\n', f'    environment:\n      HOSSEIN_ASR_URL: {args.asr_url}\n')
    changes = {'server.py':server, 'static/app.html':app, 'static/login.js':login,
               'Dockerfile':docker, 'docker-compose.yml':compose}
    compile(server, 'server.py', 'exec')
    backup = root.parent / ('recorder-live-backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir()
    for name, text in changes.items():
        (backup/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, backup/name)
        (root/name).write_text(text, encoding='utf-8')
    shutil.copy2(source/'hossein_live.py', root/'hossein_live.py')
    for p in (source/'static').iterdir():
        if p.is_file():
            shutil.copy2(p, root/'static'/p.name)
    print('Integrated recorder; original files backed up at', backup)


if __name__ == '__main__':
    main()
