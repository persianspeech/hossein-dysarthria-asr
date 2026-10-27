"""Run on 3060; use existing local login credentials without printing or persisting them."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8765')
    parser.add_argument('--recorder',type=Path,default=Path('/home/gooran/Desktop/hossein-data/recorder'))
    parser.add_argument('--snapshot',type=Path,default=Path('/home/gooran/hossein-dysarthria-export-20261027'))
    args=parser.parse_args()
    token=None

    def request(path, data=None, content_type=None, auth=True):
        headers={}
        if token and auth:headers['Authorization']='Bearer '+token
        if content_type:headers['Content-Type']=content_type
        req=Request(args.base+path,data=data,headers=headers)
        try:
            with urlopen(req,timeout=35) as response:return response.status,response.read()
        except HTTPError as error:return error.code,error.read()

    checks={}
    for path in ['/api/live/status','/api/live/transcribe']:
        code,_=request(path,data=b'\0'*32000 if path.endswith('transcribe') else None,auth=False,content_type='application/octet-stream')
        assert code==401,(path,code)
    checks['unauthenticated_requests_denied']=True
    users=json.loads((args.recorder/'users.json').read_text())['users']
    user=next(x for x in users if x['username']=='hossein')
    code,body=request('/api/login',data=json.dumps(user).encode(),content_type='application/json',auth=False)
    assert code==200,'Login failed'
    token=json.loads(body)['token']
    try:
        code,body=request('/api/live/status')
        health=json.loads(body)
        assert code==200 and health['ready'] and health['head']=='ctc'
        assert health['model_sha256']=='4d96d24f154b1dfbf23c5c8914274d339d1dfa2f4eadf23ff725009a44d7f9c0'
        checks['selected_model_identity_verified']=True
        for path in ['/test-voice','/static/hossein-test.js','/static/hossein-capture.js','/static/hossein-pcm-worklet.js','/static/hossein-test.css','/app']:
            code,body=request(path)
            assert code==200,(path,code)
            if path=='/app':assert '/test-voice' in body.decode()
        checks['page_assets_and_navigation']=True
        code,body=request('/api/items')
        assert code==200 and len(json.loads(body)['items'])>0
        checks['existing_recorder_catalog_works']=True
        for size,expected in [(2,400),(1601,400),(1280002,413)]:
            code,_=request('/api/live/transcribe',data=b'\0'*size,content_type='application/octet-stream')
            assert code==expected,(size,code)
        checks['invalid_audio_rejected']=True
        predictions={p['id']:p['prediction'] for p in json.loads((Path(__file__).parent/'expected_ctc.json').read_text())}
        rows=[json.loads(line) for line in (args.snapshot/'dataset/test.jsonl').read_text().splitlines()][:3]
        results=[]
        for row in rows:
            wave,rate=sf.read(args.snapshot/'dataset'/row['audio'],dtype='float32')
            assert rate==16000 and wave.ndim==1
            pcm=np.clip(np.round(wave*32768),-32768,32767).astype('<i2').tobytes()
            start=time.monotonic()
            code,body=request('/api/live/transcribe',data=pcm,content_type='application/octet-stream')
            assert code==200,code
            result=json.loads(body)
            assert result['text']==predictions[row['id']],row['id']
            results.append(dict(id=row['id'],duration=len(wave)/rate,roundtrip_ms=round((time.monotonic()-start)*1000),inference_ms=result['inference_ms'],matches_export_evaluation=True))
        checks['audio_samples']=results
        code,_=request('/api/reference?id='+rows[0]['id'])
        assert code==200
        checks['existing_reference_audio_works']=True
    finally:
        request('/api/logout',data=b'{}',content_type='application/json')
    receipt=dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),base_url=args.base,checks=checks)
    (Path(__file__).parent/'deployment_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
