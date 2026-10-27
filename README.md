# سازگارسازی شنوا کوچک با گفتار حسین

**نتیجهٔ اجرای ۲۷ اکتبر ۲۰۲۶:** آموزش و ارزیابی تمام شد؛ مدل منتخب epoch 21 از
۲۵ epoch تکمیل‌شده است. WER تست CTC از ۸۷٫۸۷٪ به ۶۱٫۹۵٪ و CER از ۵۷٫۳۴٪
به ۳۳٫۶۷٪ رسید. مدل و دادگان آماده‌اند؛ بارگذاری خصوصی در سازمان Hugging Face
به دلیل نبود مجوز نوشتنِ توکن با 403 متوقف است. [گزارش نهایی](FINAL_REPORT_FA.md)
و [وضعیت انتشار](publication_status.json) جزئیات را ثبت می‌کنند.

پروژهٔ مستقل برای فاین‌تیون کامل `Reza2kn/Shenava-Koochik-v1.5` با صوت حسین.
مدل FastConformer هیبرید RNNT/CTC، حدود ۱۱۴ میلیون پارامتر دارد.

## داده و برچسب

۵۱۰ ضبط، مجموعاً ۱٫۹۴۵۵۵ ساعت: ۴۰۸ آموزش، ۵۱ اعتبارسنجی و ۵۱ تست.
نسخهٔ کامل هر ضبط به mono 16 kHz / FLAC PCM16 تبدیل می‌شود؛ سکوت بریده نمی‌شود.
split با seed ثابت 20261027 ساخته می‌شود و گروه‌های متصل صوت PCM یکسان یا متن
نرمال‌شدهٔ یکسان همواره در یک بخش می‌مانند. این مجموعه یک گوینده دارد؛ جداسازی
جمله به معنی آزمون گوینده یا جلسهٔ جدید نیست.

برچسب‌ها همان متن نمایش‌داده‌شده در ضبط‌کننده‌اند. اسکریپت `transcribe_saremi.py`
آن‌ها را با شنوا از صدای **گویندهٔ مرجع صارمی** ساخته است. این‌ها رونویسی انسانی
تأییدشدهٔ صدای حسین نیستند و با `label_kind=asr_reference_prompt` و
`human_verified=false` منتشر می‌شوند. متن خام محفوظ است. امتیاز ارزیابی میزان
تطابق خروجی با همین promptها را می‌سنجد. وضعیت گفتاری Dysarthria / Apraxia
توصیف کاربرِ مالک مجموعه است. بررسی مستقل بالینی انجام نشده است.

نام کاربری/رمز ضبط‌کننده، `shenava.env`، صوت گویندهٔ مرجع و جمله‌های ضبط‌نشده
در انتشار قرار نمی‌گیرند. هیچ مجوز جدیدی برای دادگان از طریق این کد اعطا نمی‌شود.

## آموزش

تمام پارامترها و هر دو سر آموزش می‌بینند؛ واژگان و توکنایزر والد حفظ می‌شوند.
loss برابر `0.7*RNNT + 0.3*CTC`، AdamW با LR اولیه `1e-5`، batch مؤثر ۸،
BF16 و SpecAugment ملایم است. سقف ۲۵ epoch، warmup برابر ۵۰ update و توقف
پس از پنج epoch بدون بهبود است. انتخاب مدل فقط با WER اعتبارسنجی CTC و
CER به‌عنوان tie breaker انجام می‌شود. تست در انتخاب، تنظیم یا توقف دخالت ندارد.

قبل از آموزش، forward/backward واقعی با فایل‌های دشوار و تغییر وزن encoder،
RNNT و CTC بررسی می‌شود. سپس restore وزن، optimizer و RNG در **پردازش جدا**
آزموده می‌شود. checkpoint کامل شامل cursor آموزش و هویت hashشدهٔ کد، تنظیمات،
مدل والد و manifestهاست. تغییر هویت مانع resume خاموش می‌شود.

## اجرا

Linux، Python 3.12، درایور سازگار CUDA 12.6 و GPU مناسب لازم است.
برای نصب تازه، `python3 -m venv .venv` سپس `bash setup.sh` را اجرا کنید.
نسخه‌های محیط واقعاً استفاده‌شده در `environment.freeze.txt` ثبت می‌شوند.

```bash
python export_source.py --source /path/to/recorder --output /path/to/new-snapshot
# run_config.json و اسکریپت‌ها را کنار raw/ و source_manifest.json قرار دهید.
python prepare.py --root /path/to/new-snapshot
python verify_data.py
python download_base.py
HOSSEIN_PYTHON=.venv/bin/python bash run_pipeline.sh
```

در اجرای واقعی، دانلود CUDA از mirror به timeout خورد؛ محیط NeMo سازگار و
آزموده‌شدهٔ 3060 به پوشهٔ مستقل `.venv-reference` روی 4090 منتقل شد.
این انتقال فقط محیط پایتون است و هیچ کلید SSH یا توکن حساب منتقل نمی‌شود.
`transfer.py` مسیرهای خصوصی زیرساخت همین اجرا را دارد؛ برای زیرساخت دیگر باید
آرگومان‌ها/مسیرها متناسب شوند. نصب تازهٔ `setup.sh` تا پایان در این اجرا آزموده نشده است.

آزمون واقعی RNNT به `numba-cuda==0.30.4`، `cuda-core==0.5.1` و
`numpy==2.2.6` نیاز داشت. NVVM/libdevice نسخهٔ 12.6.85 از wheel نصب و با
nvJitLink نسخهٔ 12.6 هماهنگ شد؛ launcher مسیر CUDA 13 سیستم را برای این
ابزارها جایگزین می‌کند. در predictor تک‌لایه، flag داخلیِ dropout بین لایه‌های
LSTM که در این معماری غیرفعال است، صفر شده است؛ dropout خروجی NeMo حفظ
می‌شود. این کار اختلاف RNG اولین فراخوانی را برطرف کرد. آزمون بازیابی در پردازش
تازه، loss هر دو سر را دقیقاً برابر اجرای پیوسته به‌دست آورد.

## خروجی و ارزیابی

- `dataset/data_report.json`: ساعت‌ها، تقسیم، hashها، حذف‌ها و محدودیت‌ها.
- `smoke.json` و `resume_verification.json`: آزمون واقعی آموزش و بازیابی.
- `history.json` و `training_completed.json`: انتخاب با dev و پایان آموزش.
- `final_results.json`: WER/CER خام و نرمال‌شدهٔ هر دو سر، پایه در برابر فاین‌تیون،
  همراه فاصلهٔ اطمینان bootstrap زوجی روی ۵۱ نمونهٔ تست.
- `hf_model/`: مدل native `.nemo`، توکنایزر همراه، کد inference و گزارش aggregate.
- `release_validation.json`: برابری تمام tensorها و تطابق خروجی پس از reload صادرات.

```bash
python transcribe.py recording.webm --model hf_model/shenava-hossein.nemo --head ctc
python build_dataset_release.py
HF_TOKEN=... python publish_hf.py dataset
HF_TOKEN=... python publish_hf.py model
```

انتشار پیش‌فرض خصوصی است. توکن فقط از محیط محلی خوانده می‌شود؛ فایل توکن
ساخته نمی‌شود. انتشار با بررسی SHA256 تمام فایل‌های remote تکمیل می‌شود.
برای ابزار انتشار، محیط جدا با `publisher-requirements.txt` بسازید؛ محیط آموزش
با نسخه‌های دقیق `environment.freeze.txt` ثبت شده است.

## مخزن‌ها

- دادگان: https://huggingface.co/datasets/Hossein-Dysarthria-Net/hossein-persian-speech
- مدل: https://huggingface.co/Hossein-Dysarthria-Net/Shenava-Koochik-Hossein
- کد: https://github.com/persianspeech/hossein-dysarthria-asr

مجوز کد و مدل مشتق‌شده Apache-2.0 است؛ مجوز Apache والد در `LICENSE` حفظ شده است.
مجوز دادگان مستقل از مجوز کد/مدل است.
