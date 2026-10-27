"""Recorder-side authenticated proxy; add no third-party runtime dependencies."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_BYTES = 16000 * 40 * 2


def handle(handler, path, method):
    if (method, path) not in [('GET', '/api/live/status'), ('POST', '/api/live/transcribe')]:
        return False
    if not handler._user():
        handler._json(401, {'error': 'login required'})
        return True
    base = os.environ.get('HOSSEIN_ASR_URL', '').rstrip('/')
    if not base:
        handler._json(503, {'error': 'سرویس تشخیص گفتار آماده نیست.'})
        return True
    body = None
    if method == 'POST':
        try:
            size = int(handler.headers.get('Content-Length', '0'))
        except ValueError:
            size = 0
        if not 1600 <= size <= MAX_BYTES or size % 2:
            handler._json(413 if size > MAX_BYTES else 400, {'error': 'طول صدای ارسالی معتبر نیست.'})
            return True
        if handler.headers.get('Content-Type', '').split(';')[0] != 'application/octet-stream':
            handler._json(415, {'error': 'قالب صدا معتبر نیست.'})
            return True
        handler.connection.settimeout(20)
        body = handler.rfile.read(size)
        if len(body) != size:
            handler._json(400, {'error': 'دریافت صدا کامل نشد.'})
            return True
    request = Request(base + ('/health' if method == 'GET' else '/transcribe'), data=body,
                      headers={'Content-Type': 'application/octet-stream'}, method=method)
    try:
        with urlopen(request, timeout=3 if method == 'GET' else 30) as response:
            payload = json.loads(response.read(65536))
            handler._json(200, payload)
    except HTTPError as error:
        code = 429 if error.code == 429 else 503
        handler._json(code, {'error': 'مدل مشغول است؛ کمی صبر کنید.' if code == 429 else 'پردازش صدا انجام نشد.'})
    except (URLError, TimeoutError, OSError, ValueError):
        handler._json(503, {'error': 'ارتباط با مدل برقرار نشد؛ کمی بعد دوباره امتحان کنید.'})
    return True
