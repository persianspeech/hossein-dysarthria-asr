"""Exercise authentication/body validation and real HTTP forwarding without loading NeMo."""
import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from hossein_live import handle, MAX_BYTES


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'{"ready":true}'
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        assert len(body) == 32000
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'text':'صدای آزمایشی'}).encode())


class Recorder:
    def __init__(self, authenticated=True, body=b'', content_type='application/octet-stream'):
        self.authenticated = authenticated
        self.headers = {'Content-Length':str(len(body)), 'Content-Type':content_type}
        self.rfile = io.BytesIO(body)
        self.connection = type('Connection', (), {'settimeout':lambda *_: None})()
        self.result = None

    def _user(self):
        return 'test' if self.authenticated else None

    def _json(self, code, value):
        self.result = (code, value)


class ProxyTests(unittest.TestCase):
    def test_authentication_precedes_audio_reading(self):
        for method, path in [('GET','/api/live/status'),('POST','/api/live/transcribe')]:
            h = Recorder(False, b'private audio')
            self.assertTrue(handle(h,path,method))
            self.assertEqual(h.result[0],401)
            self.assertEqual(h.rfile.tell(),0)

    def test_limits_and_formats(self):
        with patch.dict(os.environ, {'HOSSEIN_ASR_URL':'http://unused'}):
            for size, expected in [(0,400),(1601,400),(MAX_BYTES+2,413)]:
                h=Recorder(body=b'\0'*size)
                handle(h,'/api/live/transcribe','POST')
                self.assertEqual(h.result[0],expected)
                self.assertEqual(h.rfile.tell(),0)
            h=Recorder(body=b'\0'*32000,content_type='audio/webm')
            handle(h,'/api/live/transcribe','POST')
            self.assertEqual(h.result[0],415)

    def test_http_proxy(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ,{'HOSSEIN_ASR_URL':f'http://127.0.0.1:{server.server_port}'}):
                h=Recorder()
                handle(h,'/api/live/status','GET')
                self.assertEqual(h.result,(200,{'ready':True}))
                h=Recorder(body=b'\0'*32000)
                handle(h,'/api/live/transcribe','POST')
                self.assertEqual(h.result,(200,{'text':'صدای آزمایشی'}))
        finally:
            server.shutdown()
            server.server_close()


if __name__=='__main__':
    unittest.main()
