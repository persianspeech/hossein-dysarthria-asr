"""Upload only a curated release and verify every remote file, using local credentials."""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from common import ROOT, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind',choices=['dataset','model'])
    parser.add_argument('--public',action='store_true',help='Explicitly request a public repository')
    args = parser.parse_args()
    config = read(ROOT/'run_config.json')
    folder = ROOT/('hf_dataset' if args.kind=='dataset' else 'hf_model')
    repo = config['dataset_repo' if args.kind=='dataset' else 'model_repo']
    token = os.environ.get('HF_TOKEN')
    assert token, 'Provide HF_TOKEN in the local environment'
    api = HfApi(token=token)
    assert any(o['name']=='Hossein-Dysarthria-Net' for o in api.whoami()['orgs'])
    hashes = read(folder/'release_hashes.json')
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts)
    assert set(files) == set(hashes)|{'release_hashes.json'}
    for name, expected in hashes.items():
        assert sha(folder/name)==expected,name
    for name in files:
        if name.endswith(('.json','.md','.py','.txt')):
            assert token not in (folder/name).read_text(encoding='utf-8'),name
    try:
        info = api.repo_info(repo,repo_type=args.kind,files_metadata=True)
    except RepositoryNotFoundError:
        api.create_repo(repo,repo_type=args.kind,private=not args.public)
        info = api.repo_info(repo,repo_type=args.kind,files_metadata=True)
    assert info.private == (not args.public), 'Do not silently change existing access'
    existing = {s.rfilename for s in info.siblings} - {'.gitattributes'}
    assert existing <= set(files), 'Unexpected existing files; refusing overwrite'
    if existing:
        for item in info.siblings:
            if item.rfilename in hashes and item.lfs:
                assert item.lfs.sha256 == hashes[item.rfilename], 'Existing artifact differs'
    commit = api.upload_folder(repo_id=repo,repo_type=args.kind,folder_path=folder,
                               ignore_patterns=['.cache/**'],commit_message='Add verified Hossein speech '+args.kind+' release')
    remote = api.repo_info(repo,repo_type=args.kind,revision=commit.oid,files_metadata=True)
    assert remote.private == (not args.public)
    assert {s.rfilename for s in remote.siblings}==set(files)|{'.gitattributes'}
    for item in remote.siblings:
        if item.rfilename=='.gitattributes':continue
        path = folder/item.rfilename
        assert item.size==path.stat().st_size,item.rfilename
        expected=sha(path)
        if item.lfs:
            assert item.lfs.sha256==expected,item.rfilename
        else:
            downloaded=hf_hub_download(repo,item.rfilename,repo_type=args.kind,revision=commit.oid,token=token)
            assert sha(downloaded)==expected,item.rfilename
    result=dict(repo_id=repo,repo_type=args.kind,commit=commit.oid,private=remote.private,
                all_remote_files_sha256_verified=True,files=len(files),hashes={n:sha(folder/n) for n in files})
    write(ROOT/f'hf_{args.kind}_upload.json',result)
    print(__import__('json').dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
