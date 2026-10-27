"""Private, memory-only PCM inference service for Hossein's selected NeMo checkpoint."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RATE = 16000
MAX_SECONDS = 40
MAX_BYTES = RATE * MAX_SECONDS * 2
MODEL_SHA256 = '4d96d24f154b1dfbf23c5c8914274d339d1dfa2f4eadf23ff725009a44d7f9c0'
MODEL_ID = 'Hossein-Dysarthria-Net/Shenava-Koochik-Hossein'


class Engine:
    def __init__(self, checkpoint):
        import numpy as np
        import torch
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf

        digest = hashlib.file_digest(Path(checkpoint).open('rb'), 'sha256').hexdigest()
        if digest != MODEL_SHA256:
            raise RuntimeError('Unexpected checkpoint SHA256')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the live caption service')
        torch.set_num_threads(4)
        self.np, self.torch = np, torch
        self.lock = threading.Lock()
        self.model = ASRModel.restore_from(str(checkpoint), map_location='cpu').cuda().eval()
        self.model.encoder.set_default_att_context_size([70, 13])
        self.model.change_decoding_strategy(OmegaConf.create({'strategy': 'greedy'}), decoder_type='ctc')
        self.transcribe(np.zeros(RATE, dtype='<i2').tobytes())

    def transcribe(self, pcm):
        if not pcm or len(pcm) % 2 or len(pcm) > MAX_BYTES:
            raise ValueError('Invalid PCM length')
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('Model busy')
        try:
            wave = self.np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768.0
            start = time.monotonic()
            with self.torch.inference_mode():
                result = self.model.transcribe([wave], batch_size=1, num_workers=0,
                                               verbose=False, return_hypotheses=False)
            if isinstance(result, tuple):
                result = result[0]
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            return {'text': text.strip(), 'duration': len(wave) / RATE,
                    'inference_ms': round((time.monotonic() - start) * 1000),
                    'model': MODEL_ID, 'head': 'ctc'}
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *args):
        # Never log speech, bearer tokens, request bodies or query strings.
        pass

    def reply(self, code, value):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/health':
            return self.reply(404, {'error': 'not found'})
        self.reply(200, {'ready': True, 'model': MODEL_ID, 'head': 'ctc',
                         'sample_rate': RATE, 'max_seconds': MAX_SECONDS,
                         'model_sha256': MODEL_SHA256})

    def do_POST(self):
        if self.path != '/transcribe':
            return self.reply(404, {'error': 'not found'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.reply(400, {'error': 'invalid length'})
        if not 1600 <= size <= MAX_BYTES or size % 2:
            return self.reply(413 if size > MAX_BYTES else 400, {'error': 'invalid audio length'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/octet-stream':
            return self.reply(415, {'error': 'expected mono 16kHz s16le PCM'})
        self.connection.settimeout(20)
        pcm = self.rfile.read(size)
        if len(pcm) != size:
            return self.reply(400, {'error': 'incomplete audio'})
        try:
            self.reply(200, self.engine.transcribe(pcm))
        except BlockingIOError:
            self.reply(429, {'error': 'مدل در حال پردازش است؛ دوباره تلاش کنید.'})
        except Exception:
            logging.exception('Inference failed; audio is not retained')
            self.reply(503, {'error': 'پردازش صدا انجام نشد؛ دوباره تلاش کنید.'})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8791)
    args = parser.parse_args()
    Handler.engine = Engine(args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Hossein CTC ready on {args.host}:{args.port}; SHA256 {MODEL_SHA256}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
