"""Relay tar over two existing SSH sessions; never copy private keys."""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['data', 'base', 'prepared', 'environment'])
    args = parser.parse_args()
    if args.kind == 'data':
        command = 'tar -C /home/gooran/hossein-dysarthria-export-20261027 -cf - raw source_manifest.json'
        target = 'tar -C /home/user01/hossein-dysarthria -xf -'
    elif args.kind == 'environment':
        command = ('tar --exclude=__pycache__ --exclude=*.pyc '
                   '-C /home/gooran/agp-asr-nemotron-fa/.venv-shenava -czf - .')
        target = ('mkdir -p /home/user01/hossein-dysarthria/.venv-reference && '
                  'tar --warning=no-timestamp -C /home/user01/hossein-dysarthria/.venv-reference -xzf -')
    elif args.kind == 'prepared':
        command = 'tar -C /home/gooran/hossein-dysarthria-export-20261027 -cf - dataset transfer_verification.json'
        target = 'tar --warning=no-timestamp -C /home/user01/hossein-dysarthria -xf -'
    else:
        command = ('tar -C /home/gooran/agp-asr-nemotron-fa/artifacts/external/shenava-v1.5 '
                   '-cf - shenava-koochik-v1.5.nemo')
        target = 'mkdir -p /home/user01/hossein-dysarthria/base && tar -C /home/user01/hossein-dysarthria/base -xf -'
    source = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'gooran@10.1.20.44', command], stdout=subprocess.PIPE)
    dest = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'rtx4090', target], stdin=source.stdout)
    source.stdout.close()
    assert source.wait() == 0 and dest.returncode == 0, 'Transfer failed'
    print(args.kind + ' transfer finished; checksum verification is a separate required step')


if __name__ == '__main__':
    main()
