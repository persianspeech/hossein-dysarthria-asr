"""Fetch the immutable upstream .nemo for a fresh reproduction."""
import shutil
from huggingface_hub import hf_hub_download
from common import ROOT, read, sha

config=read(ROOT/'run_config.json')
target=ROOT/'base'/config['base_filename']
if target.exists():
    assert sha(target)==config['base_sha256']
else:
    source=hf_hub_download(config['base_repo'],config['base_filename'],revision=config['base_revision'],token=False)
    assert sha(source)==config['base_sha256']
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(source,target)
    assert sha(target)==config['base_sha256']
print(str(target))
