---
language:
- fa
task_categories:
- automatic-speech-recognition
tags:
- persian
- dysarthria
- apraxia
- single-speaker
- prompt-labels
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
---

# دادگان گفتار فارسی حسین | Hossein Persian Speech

۵۱۰ ضبط فارسی حسین؛ ۱٫۹۴۶ ساعت، با صوت کامل mono 16 kHz در قالب FLAC.
این مجموعه برای سازگار کردن تشخیص گفتار با گفتار یک فرد مشخص آماده شده است.
کاربر مالک مجموعه، وضعیت گفتاری گوینده را Dysarthria / Apraxia توصیف کرده است.

| Split | Clips | Hours |
|---|---:|---:|
{{SPLIT_TABLE}}

{{PREVIEW}}

{{HOSSEIN_STORY}}

## Labels and provenance

`text_raw` preserves the prompt shown by the original recorder. `text` applies only
Persian letter/diacritic/whitespace normalization. These prompts were generated
by Shenava ASR from Saremi **reference-speaker** recordings (`transcribe_saremi.py`),
then displayed while Hossein recorded repetitions (`docker-compose.yml`).
They are **not manually verified transcriptions of Hossein's actual recordings**.
`human_verified=false` and `label_kind=asr_reference_prompt` make this explicit.
No model predictions on Hossein replace the original labels.
Possible prompt mistakes, omitted/repeated words and speech/prompt mismatches remain.

## Split and evaluation scope

Seed 20261027. Exact canonical PCM and normalized-text connected components remain
within one split. There are no exact PCM or normalized-text overlaps across splits.
The split is utterance-disjoint but all splits contain the same speaker.
Recording-session independence and absence of near duplicates are not established.
Historical overlap between the base model's training and reference prompts is unknown.
Test is held out from training, early stopping and checkpoint selection.
Scores measure agreement with unverified prompts and do not establish accuracy
against human-verified clinical speech transcripts or generalization to other speakers.

Audio is resampled in full; no silence trimming, time stretching, sentence truncation
or synthetic augmentation is applied to the published recordings.
File hashes, RMS, peak level and clipping fraction are included for auditing.
Recorder credentials, configuration secrets, reference-speaker audio and unrecorded
prompts are excluded. The original WebM snapshot is retained on the owner's servers.
No additional dataset license is granted by this card.

## Load

```python
from datasets import load_dataset, Audio
ds = load_dataset("Hossein-Dysarthria-Net/hossein-persian-speech", token=True)
# Access the embedded bytes without requiring a decoder:
raw = ds["test"].cast_column("audio", Audio(decode=False))
print(raw[0]["text"], raw[0]["audio"]["path"])
```

Training code: https://github.com/persianspeech/hossein-dysarthria-asr
Base model: https://huggingface.co/Reza2kn/Shenava-Koochik-v1.5
