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

- Click Record and read the displayed phrase aloud.
- Stop recording to see the transcription.
- Works best with ~16 kHz audio; internal processing follows Voxtral's processor expectations.

Environment variables expected:

- `HF_MODEL_ID`: The model repo to load (e.g., `username/voxtral-finetune-YYYYMMDD_HHMMSS`)
- `MODEL_NAME`: Display name
- `HF_USERNAME`: For branding
