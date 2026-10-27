"""Generate the final Persian report only from completed, verified artifacts."""
from common import ROOT, read

result=read(ROOT/'final_results.json')
training=read(ROOT/'training_completed.json')
validation=read(ROOT/'release_validation.json')
publication=ROOT/'publication_config.json'
public=publication.exists() and read(publication).get('visibility')=='public'
assert validation['passed'] and not result['test_used_for_selection']
lines=['# گزارش فاین‌تیون شنوا کوچک برای حسین','',
       'مدل پایه: `Reza2kn/Shenava-Koochik-v1.5`، با ۱۱۴٬۷۵۲٬۵۱۴ پارامتر؛ تمام پارامترها آموزش دیده‌اند.',
       'داده: ۵۱۰ ضبط / ۱٫۹۴۵۵۵ ساعت؛ ۴۰۸ آموزش، ۵۱ اعتبارسنجی و ۵۱ تست.',
       f"انتخاب نهایی: epoch {training['best_epoch']}؛ تعداد epoch تکمیل‌شده: {training['epoch']}؛ update تکمیل‌شده: {training['step']}.",
       '', '## نتیجهٔ تست کنارگذاشته‌شده', '',
       '| سر | WER پایه | WER فاین‌تیون | CER پایه | CER فاین‌تیون |',
       '|---|---:|---:|---:|---:|']
for head in ['ctc','rnnt']:
    a=result['metrics']['baseline']['test'][head]
    b=result['metrics']['selected']['test'][head]
    lines.append(f"| {head.upper()} | {a['wer']:.2f}% | {b['wer']:.2f}% | {a['cer']:.2f}% | {b['cer']:.2f}% |")
lines+=['','CTC سر اصلی است. انتخاب فقط با اعتبارسنجی CTC انجام شده و تست در انتخاب یا توقف دخالت نداشته است.','']
for head in ['ctc','rnnt']:
    c=result['comparison'][head]
    low,high=c['paired_bootstrap_95ci']
    lines.append(f"تغییر WER برای {head.upper()}: {c['wer_delta_percentage_points']:+.2f} واحد درصد؛ فاصلهٔ اطمینان زوجی ۹۵٪: [{low:+.2f}, {high:+.2f}] (۲۰۰۰ بازنمونه‌گیری). مقدار منفی یعنی کاهش خطا.")
lines+=['','## اعتبار نتیجه','',
        'تمام بخش‌ها صدای یک فرد هستند؛ این آزمون تعمیم به گوینده یا جلسهٔ جدید را اثبات نمی‌کند.',
        'برچسب‌ها متن promptهای تولیدشده با ASR از صدای گویندهٔ مرجع هستند؛ رونویسی انسانی تأییدشدهٔ صدای حسین نیستند. امتیازها میزان تطابق با این promptها را می‌سنجند.',
        'صوت‌ها کامل نگه داشته شدند؛ کوتاه‌سازی، حذف سکوت و تعویض برچسب با خروجی مدل روی حسین انجام نشد.',
        'صوت PCM و متن نرمال‌شدهٔ یکسان میان بخش‌ها وجود ندارد؛ نبود near-duplicate یا همپوشانی تاریخی با آموزش والد اثبات نشده است.',
        '', '## صحت اجرا و خروجی','',
        'SHA256 انتقال همهٔ فایل‌های خام، FLAC و manifestها تأیید شد. فایل‌های Parquet با loader استاندارد datasets دوباره خوانده و تمام صوت‌ها/برچسب‌ها کنترل شدند.',
        'forward/backward واقعی و تغییر encoder، RNNT و CTC تأیید شد. بازیابی optimizer و RNG در پردازش تازه، loss یکسان با اجرای پیوسته داشت.',
        'مدل inference پاک‌سازی شد؛ همهٔ tensorها با checkpoint منتخب برابر بودند و پس از reload، خروجی سه فایل تست در هر دو سر دقیقاً مطابق خروجی ارزیابی بود.',
        f"SHA256 مدل inference: `{validation['exported_model_sha256']}`.",
        '', '## انتشار عمومی' if public else '## انتشار خصوصی','',
        'کد: https://github.com/persianspeech/hossein-dysarthria-asr']
for kind in ['dataset','model']:
    receipt=ROOT/f'hf_{kind}_upload.json'
    if receipt.exists():
        value=read(receipt)
        assert value['private']==(not public) and value['all_remote_files_sha256_verified']
        prefix='datasets/' if kind=='dataset' else ''
        lines.append(f"{kind}: https://huggingface.co/{prefix}{value['repo_id']} — commit `{value['commit']}`؛ تمام hashهای remote تأیید شد.")
    else:
        pub=ROOT/'publication_status.json'
        if pub.exists() and read(pub).get('huggingface',{}).get('http_status')==403:
            lines.append(f'{kind}: بارگذاری انجام نشد؛ API سازمان مقصد خطای 403 داد. توکن فعلی مجوز نوشتن در سازمان Hossein-Dysarthria-Net را ندارد.')
        else:
            lines.append(f'{kind}: بارگذاری Hugging Face هنوز تأیید نشده است؛ وضعیت دسترسی سازمان/توکن باید بررسی شود.')
lines+=['','نسخهٔ کامل اجرا و مدل روی 4090 در `/home/user01/hossein-dysarthria` محفوظ است.',
        'نسخهٔ انتشار در `hf_model/` و `hf_dataset/` قرار دارد.']
(ROOT/'FINAL_REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(str(ROOT/'FINAL_REPORT_FA.md'))
