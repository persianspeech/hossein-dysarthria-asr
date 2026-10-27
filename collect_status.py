"""Read-only compact status from the owner's training host."""
import argparse
import json
import subprocess
import textwrap
from pathlib import Path
from common import ROOT, write

SCRIPT=r'''
import json,os,subprocess
from pathlib import Path
root=Path('/home/user01/hossein-dysarthria')
out={}
for name in ['environment_validation.json','pipeline_status.json','status.json','smoke.json',
             'resume_verification.json','training_completed.json','final_results.json','release_validation.json']:
    path=root/name
    if path.exists():out[name]=json.loads(path.read_text())
if (root/'history.json').exists():out['recent_history']=json.loads((root/'history.json').read_text())[-2:]
pid=out.get('pipeline_status.json',{}).get('pid')
out['pipeline_process_alive']=bool(pid and Path('/proc',str(pid)).exists())
if out['pipeline_process_alive']:
    out['pipeline_process_command']=(Path('/proc',str(pid))/'cmdline').read_bytes().replace(b'\0',b' ').decode()
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
log=root/'pipeline.log'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-5000))
        out['log_tail']=stream.read().decode(errors='replace').splitlines()[-12:]
print(json.dumps(out,ensure_ascii=False))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='rtx4090')
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--watch',action='store_true',help='Keep one SSH connection until the current pipeline ends')
    args=parser.parse_args()
    if args.watch:
        remote=SCRIPT+'\nimport time\nfor _ in range(160):\n'
        remote+="    if out.get('pipeline_status.json',{}).get('state') in ['complete','failed']:break\n"
        remote+='    time.sleep(45)\n'+textwrap.indent(SCRIPT,'    ')
        process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',args.host,'python3 -u -'],
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8')
        process.stdin.write(remote)
        process.stdin.close()
        for line in process.stdout:
            value=json.loads(line)
            write(ROOT/'live_status.json',value)
            keys=['pipeline_status.json','status.json','recent_history','training_completed.json',
                  'final_results.json','release_validation.json','pipeline_process_alive','gpu']
            print(json.dumps({k:value[k] for k in keys if k in value},ensure_ascii=False),flush=True)
        raise SystemExit(process.wait())
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',args.host,'python3 -'],
                          input=SCRIPT,capture_output=True,text=True,encoding='utf-8',check=True)
    value=json.loads(result.stdout)
    write(ROOT/'live_status.json',value)
    if args.compact:
        keys=['pipeline_status.json','status.json','recent_history','training_completed.json',
              'final_results.json','release_validation.json','pipeline_process_alive','gpu']
        compact={k:value[k] for k in keys if k in value}
        if value.get('pipeline_status.json',{}).get('state')=='failed':
            compact['log_tail']=value.get('log_tail')
        value=compact
    print(json.dumps(value,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
