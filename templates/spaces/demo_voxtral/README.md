---
title: Voxtral ASR Demo
emoji: 🎙️
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
short_description: Interactive ASR demo for a fine-tuned Voxtral model
---
This Space serves a Voxtral ASR model for speech-to-text transcription.
Usage:

- Select a language (or leave on Auto for detection).
- Upload an audio file or record via microphone.
- Click Transcribe to see the transcription.
- Works best with standard speech audio; Voxtral handles language detection by default.

Environment variables expected:

- `HF_MODEL_ID`: The model repo to load (e.g., `username/voxtral-finetune-YYYYMMDD_HHMMSS`)
- `MODEL_NAME`: Display name
- `HF_USERNAME`: For branding
- `MODEL_SUBFOLDER`: Optional subfolder in the repo (e.g., `int4`) for quantized/packed weights

Supported languages:

- English, French, German, Spanish, Italian, Portuguese, Dutch, Hindi
  - Or choose Auto to let the model detect the language

Notes:

- Uses bfloat16 on GPU and float32 on CPU.
- Decodes only newly generated tokens for clean transcriptions.
