---
title: VoxFactory
emoji: 🌬️
colorFrom: gray
colorTo: red
sdk: gradio
app_file: interface.py
pinned: false
license: mit
short_description: FinetuneASR Voxtral
---

# Finetune Voxtral for ASR with Transformers 🤗

This repository fine-tunes the Voxtral speech model for automatic speech recognition (ASR) using Hugging Face `transformers` and `datasets`. It includes:

- Full and LoRA training scripts
- A Gradio interface to collect audio, build a JSONL dataset, fine-tune, push to Hub, and deploy a demo Space
- Utilities to push trained models and datasets to the Hugging Face Hub

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/Deep-unlearning/Finetune-Voxtral-ASR.git
cd Finetune-Voxtral-ASR
```

### 2) Create environment and install deps

Choose your package manager.

<details>
<summary>📦 Using UV (recommended)</summary>

```bash
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
```

</details>

<details>
<summary>🐍 Using pip</summary>

```bash
python -m venv .venv --python 3.10 && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

</details>

## Quick start options

- Train from CLI: run `scripts/train.py` (full) or `scripts/train_lora.py` (LoRA)
- Use the Gradio interface: `python interface.py` to record/upload audio, create dataset JSONL, train, push, and deploy a demo Space

## Dataset preparation

Training scripts accept either a local JSONL or a small Hub dataset slice.

- Local JSONL format expected by collators and push utilities:

```python
{
  "audio_path": "/abs/or/relative/path.wav",
  "text": "reference transcription"
}
```

- When loading from the Hub (default fallback): `hf-audio/esb-datasets-test-only-sorted` config `voxpopuli` is used and cast to `Audio(sampling_rate=16000)`.

- The custom `VoxtralDataCollator` constructs inputs as: prompt from audio via `VoxtralProcessor.apply_transcription_request(...)` followed by label tokens. Loss is masked over the prompt; only transcription tokens contribute to loss.

Minimum columns after loading/mapping:

- `audio` cast to `Audio(sampling_rate=16000)` (Hub) or created from `audio_path` (local JSONL)
- `text` transcription string

## Full fine-tuning (scripts/train.py)

Run with either a local JSONL or the default tiny Hub slice:

```bash
python scripts/train.py \
  --model-checkpoint mistralai/Voxtral-Mini-3B-2507 \
  --dataset-jsonl datasets/voxtral_user/data.jsonl \
  --train-count 100 --eval-count 50 \
  --batch-size 2 --grad-accum 4 --learning-rate 5e-5 --epochs 3 \
  --output-dir ./voxtral-finetuned
```

Key args:

- `--dataset-jsonl`: local JSONL with `{audio_path, text}`. If omitted, uses `hf-audio/esb-datasets-test-only-sorted`/`voxpopuli` test slice
- `--dataset-name`, `--dataset-config`: override default Hub dataset
- `--train-count`, `--eval-count`: small sample sizes for quick runs
- `--trackio-space`: HF Space ID for Trackio logging; if omitted and `HF_TOKEN` is set, a space name is auto-derived
- `--push-dataset`, `--dataset-repo`: optionally push your local JSONL dataset to the Hub after training

Environment for logging and Hub auth:

- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`: enables Trackio space naming and Hub uploads

Outputs: model and processor saved to `--output-dir`.

## LoRA fine-tuning (scripts/train_lora.py)

```bash
python scripts/train_lora.py \
  --model-checkpoint mistralai/Voxtral-Mini-3B-2507 \
  --dataset-jsonl datasets/voxtral_user/data.jsonl \
  --train-count 100 --eval-count 50 \
  --batch-size 2 --grad-accum 4 --learning-rate 5e-5 --epochs 3 \
  --lora-r 8 --lora-alpha 32 --lora-dropout 0.0 --freeze-audio-tower \
  --output-dir ./voxtral-finetuned-lora
```

Additional LoRA args:

- `--lora-r`, `--lora-alpha`, `--lora-dropout`
- `--freeze-audio-tower`: optionally freeze audio encoder params

## End-to-end via Gradio interface (interface.py)

Start the UI:

```bash
python interface.py
```

What it does:

- Record microphone audio or upload files + transcripts
- Saves datasets to `datasets/voxtral_user/` as `data.jsonl` or `recorded_data.jsonl`
- Kicks off full or LoRA training with streamed logs
- Optionally pushes dataset and model to the Hub
- Optionally deploys a Voxtral ASR demo Space

Environment variables used by the interface:

- `HF_WRITE_TOKEN` or `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`: write/read token for Hub actions
- `HF_READ_TOKEN`: optional read token
- `HF_USERNAME`: fallback username if it cannot be derived from the token

Notes:

- The interface uses a multilingual phrase source (CohereLabs/AYA via token; otherwise localized fallbacks)
- Output models are placed under `outputs/<username_repo>/`

## Push models and datasets to Hugging Face (scripts/push_to_huggingface.py)

Push a trained model directory (full or LoRA):

```bash
python scripts/push_to_huggingface.py model ./voxtral-finetuned my-voxtral-asr \
  --author-name "Your Name" \
  --model-description "Fine-tuned Voxtral ASR" \
  --model-name mistralai/Voxtral-Mini-3B-2507
```

Push a dataset JSONL and its audio files:

```bash
python scripts/push_to_huggingface.py dataset datasets/voxtral_user/data.jsonl my-voxtral-dataset
```

Tips:

- If you pass bare repo names (no `username/`), the tool will resolve your username from the token or `HF_USERNAME`.
- For LoRA outputs, the pusher detects adapter files; for full models it detects `config.json` + weight files and uploads accordingly.

## Deploy a demo Space (scripts/deploy_demo_space.py)

Deploy a Voxtral demo Space for a pushed model:

```bash
python scripts/deploy_demo_space.py \
  --hf-token $HF_TOKEN \
  --hf-username your-hf-username \
  --model-id your-hf-username/your-model-repo \
  --demo-type voxtral \
  --space-name my-voxtral-demo
```

What it does:

- Creates the Space (or use `--skip-creation` to only upload)
- Uploads template files from `templates/spaces/demo_voxtral/`
- Sets space variables and secrets (e.g., `HF_TOKEN`, `HF_MODEL_ID`) via API
- Waits for the Space to build and tests accessibility

The Space app loads either a full model or a base+LoRA adapter with `peft`, and uses `AutoProcessor` to build Voxtral transcription requests.

## GPU and versions

- Torch 2.8.0 + torchaudio 2.8.0 and `torchcodec==0.7` are specified; CUDA-capable GPU is recommended for training
- The code prefers `bfloat16` on CUDA, `float32` on CPU

## Troubleshooting

- No token found:
  - Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your environment for Hub operations and Trackio naming
- Invalid token or username resolution failed:
  - Provide fully-qualified repo IDs like `username/repo` or set `HF_USERNAME`
- Demo Space rate limits / propagation delays:
  - The deploy script retries uploads and may need extra time for the Space to build
- Collator errors:
  - Ensure your JSONL rows include valid `audio_path` files and `text` strings
- Windows shell hints:
  - Use `set HF_TOKEN=your_token` in CMD/PowerShell before running scripts

## License

MIT