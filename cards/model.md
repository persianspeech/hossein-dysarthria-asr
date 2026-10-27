---
language:
- fa
license: apache-2.0
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: Reza2kn/Shenava-Koochik-v1.5
base_model_relation: finetune
datasets:
- Hossein-Dysarthria-Net/hossein-persian-speech
tags:
- dysarthria
- apraxia
- persian
- fastconformer
- ctc
- rnnt
---

# Shenava Koochik — Hossein adaptation

مدل شنوا کوچک، سازگارشده با ۴۰۸ ضبط حسین. ۵۱ نمونه برای اعتبارسنجی و ۵۱
نمونه برای تست مستقل از انتخاب مدل کنار گذاشته شده‌اند. مدل شامل هر دو سر CTC و RNNT است.

{{HOSSEIN_STORY}}

## معماری و آموزش | Architecture and training

Full-parameter adaptation of the native 114M FastConformer hybrid RNNT/CTC model.
The original vocabulary and tokenizer are preserved. The objective is 0.7 RNNT +
0.3 CTC with BF16 mixed precision, effective batch 8, AdamW and light SpecAugment.
There is no pseudo-label replacement of Hossein's recordings, truncation or silence trimming.
CTC is the primary head. The best **trained** checkpoint is selected only by CTC
validation WER, with CER as tie breaker, and five-epoch early stopping.

## Held-out test (same speaker, 51 utterances)

| Head | Base WER | Tuned WER | Base CER | Tuned CER |
|---|---:|---:|---:|---:|
{{METRIC_ROWS}}

Selected epoch: **{{SELECTED_EPOCH}}**. Completed epochs: {{COMPLETED_EPOCHS}}.
`evaluation.json` includes raw/normalized scores and paired bootstrap intervals.
WER can exceed 100%; lower is better. See the confidence intervals before making
claims about the size of any improvement.

## Label and scope limitations

Labels are ASR-derived **reference prompts**, not human-verified transcriptions of
Hossein's actual speech. The reference-speaker ASR labels were displayed while he
recorded repetitions. Scores measure prompt agreement and may be affected by prompt
errors, deviations and omitted/repeated words. The user's description of the speaker
is Dysarthria / Apraxia. No clinical diagnosis is established by this experiment.
This is single-speaker adaptation; it does not establish performance on other
speakers, unseen recording sessions, spontaneous speech or human-verified transcripts.
Exact PCM and normalized-text duplicates are disjoint across train/validation/test;
near-duplicate and historical base-training overlap are not established.

## Inference

Install the pinned environment (`environment.freeze.txt`; the tested runtime is
NeMo 3.0.0 and PyTorch 2.11.0+cu126). This repository is public; authentication is not required to download the model.

```bash
python transcribe.py recording.wav --head ctc
python transcribe.py recording.webm --head rnnt
```

The `.nemo` file bundles the tokenizer. `transcribe.py` reads complete audio via
FFmpeg, uses mono 16 kHz and context [70,13], and supports either head. Live
cache-aware streaming and long-recording chunking were not evaluated.
The inference archive contains no optimizer state or source recordings. A fresh
reload was checked for equality of all weight tensors and matching predictions
on three final-test files for each head (`validation.json`).

Source: https://github.com/persianspeech/hossein-dysarthria-asr
Dataset: https://huggingface.co/datasets/Hossein-Dysarthria-Net/hossein-persian-speech
Original base: https://huggingface.co/Reza2kn/Shenava-Koochik-v1.5
