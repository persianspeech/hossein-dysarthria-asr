"""Render maintained release cards and exact training-only audio previews; no credentials."""
import argparse
import hashlib
import html
import re
from pathlib import Path
from common import ROOT, read, write, sha


PREVIEW_IDS = ('1003', '1004', '1009')


def dataset_preview(folder, public=False):
    import pyarrow.parquet as pq
    items = pq.read_table(folder/'data/train-00000-of-00001.parquet').to_pylist()
    selected = {r['id']: r for r in items if r['id'] in PREVIEW_IDS}
    assert set(selected) == set(PREVIEW_IDS)
    preview = folder/'preview'
    preview.mkdir(exist_ok=True)
    lines = ['## پیش‌نمایش داده‌ها', '',
             'سه نمونهٔ زیر از بخش **آموزش (train)** انتخاب شده‌اند. صوت، صدای حسین است؛ متن کنار آن همان جملهٔ راهنمای ضبط است و رونویسی انسانیِ تأییدشدهٔ آنچه شنیده می‌شود نیست.', '',
             'این دادگان عمومی است؛ برای شنیدن یا دریافت نمونه‌ها نیازی به ورود نیست.' if public else
             'برای پخش یا دریافت صوت باید با حساب دارای دسترسی به این مخزن خصوصی وارد Hugging Face شده باشید.', '']
    manifest = []
    for identifier in PREVIEW_IDS:
        row = selected[identifier]
        payload = row['audio']['bytes']
        assert payload.startswith(b'fLaC')
        assert hashlib.sha256(payload).hexdigest() == row['audio_sha256']
        name = f'preview/{identifier}.flac'
        (folder/name).write_bytes(payload)
        url = f'https://huggingface.co/datasets/Hossein-Dysarthria-Net/hossein-persian-speech/resolve/main/{name}'
        lines += [f"### نمونهٔ `{identifier}` — {row['duration']:.2f} ثانیه", '',
                  f"متن راهنما: {row['text']}", '',
                  f'<audio controls preload="none" src="{html.escape(url)}"></audio>', '',
                  f'[پخش / دریافت فایل صوتی]({url})', '']
        manifest.append({k: row[k] for k in ('id','text','duration','audio_sha256','label_kind','human_verified')}
                        | {'split':'train','path':name})
    write(preview/'samples.json', manifest)
    if public:
        lines += ['برای مرور جدول داده‌ها و پخش صوت هر ردیف، بخش [Dataset Viewer](https://huggingface.co/datasets/Hossein-Dysarthria-Net/hossein-persian-speech/viewer/default/train) را باز کنید. نمونه‌های بالا نیز مستقیماً در کارت دادگان قابل‌دسترسی‌اند.', '']
    else:
        lines += ['پیش‌نمایش جدولی خودکار (Dataset Viewer / Data Studio) برای مخزن خصوصی این سازمان در اشتراک فعلی فعال نیست. طبق [مستندات Hugging Face](https://huggingface.co/docs/hub/enterprise-datasets)، این قابلیت برای سازمان‌های دارای اشتراک Team یا Enterprise ارائه می‌شود. نمونه‌های بالا برای مرور مستقیم در کارت دادگان قرار گرفته‌اند.', '']
    return '\n'.join(lines)


def render(root, kind):
    folder = root/f'hf_{kind}'
    text = (ROOT/f'cards/{kind}.md').read_text(encoding='utf-8')
    replacements = {'{{HOSSEIN_STORY}}': (ROOT/'cards/hossein_story.fa.md').read_text(encoding='utf-8').strip()}
    if kind == 'dataset':
        report = read(folder/'data_report.json')
        replacements['{{SPLIT_TABLE}}'] = '\n'.join(
            f"| {s} | {report['splits'][s]['clips']} | {report['splits'][s]['hours']:.5f} |"
            for s in ('train','validation','test'))
        publication = root/'publication_config.json'
        public = publication.exists() and read(publication).get('visibility') == 'public'
        replacements['{{PREVIEW}}'] = dataset_preview(folder, public=public)
    else:
        results = read(folder/'evaluation.json')
        table = []
        for head in ('ctc','rnnt'):
            a = results['metrics']['baseline']['test'][head]
            b = results['metrics']['selected']['test'][head]
            table.append(f"| {head} | {a['wer']:.2f}% | {b['wer']:.2f}% | {a['cer']:.2f}% | {b['cer']:.2f}% |")
        replacements['{{METRIC_ROWS}}'] = '\n'.join(table)
        summary = read(folder/'training_summary.json')
        replacements['{{SELECTED_EPOCH}}'] = str(summary['selected_epoch'])
        replacements['{{COMPLETED_EPOCHS}}'] = str(summary['completed_epochs'])
    for key, value in replacements.items():
        assert key in text, key
        text = text.replace(key, value)
    assert not re.search(r'\{\{[A-Z_]+\}\}', text)
    (folder/'README.md').write_text(text, encoding='utf-8')
    write(folder/'release_hashes.json', {p.relative_to(folder).as_posix():sha(p)
          for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='release_hashes.json' and '.cache' not in p.parts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--kind', choices=['dataset','model','all'], default='all')
    args = parser.parse_args()
    for kind in ('dataset','model') if args.kind == 'all' else (args.kind,):
        render(args.root.resolve(), kind)
        print('Rendered', kind, 'card')


if __name__ == '__main__':
    main()
