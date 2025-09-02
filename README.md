---
title: VoxFactory
emoji: 📚
colorFrom: gray
colorTo: red
sdk: gradio
app_file: interface.py
pinned: false
license: mit
short_description: FinetuneASR Voxtral
---

# Finetune Voxtral for ASR with Transformers 🤗

This repository fine-tunes the [Voxtral](https://huggingface.co/Deep-unlearning/Voxtral) speech model on conversational speech datasets using the Hugging Face `transformers` and `datasets` libraries.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Deep-unlearning/Finetune-Voxtral-ASR.git
cd Finetune-Voxtral-ASR
```

### Step 2: Set up environment

Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

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

## Dataset Preparation

Perfect — here’s a **drop-in replacement** for your README’s “Dataset Preparation” that matches your script (uses **`hf-audio/esb-datasets-test-only-sorted`** with the **`voxpopuli`** config, 16 kHz casting, and a small train/eval slice), and explains the Voxtral/LLaMA-style prompt+label masking your collator implements.

---

## Dataset Preparation

For ASR fine-tuning, inputs look like:

* **Inputs**: `[AUDIO] … [AUDIO] <transcribe>  <reference transcription>`
* **Labels**: same sequence, but the prefix `[AUDIO] … [AUDIO] <transcribe>` is **masked with `-100`** so loss is computed **only** on the transcription tokens.

The `VoxtralDataCollator` already builds this sequence (prompt expansion via the processor and label masking).
The dataset only needs two fields:

```python
{
  "audio": {"array": <float32 numpy array>, "sampling_rate": 16000, ...},
  "text":  "<reference transcription>"
}
```


If you want to swap to a different dataset, ensure after loading you still have:

* an **`audio`** column (cast to `Audio(sampling_rate=16000)`), and
* a **`text`** column (the reference transcription).

If your dataset uses different column names, map them to `audio` and `text` before returning.

## Training

Run the training script:

```bash
uv run train.py
```

Logs and checkpoints will be saved under the `outputs/` directory by default.

## Training with LoRA

You can also run the training script with LoRA:

```bash
uv run train_lora.py
```

**Happy fine-tuning Voxtral!** 🚀