#!/usr/bin/env python3
"""
Voxtral ASR Full Fine-tuning Script with Trackio Integration

This script fine-tunes Voxtral models for ASR tasks with automatic experiment tracking
via Trackio and Hugging Face Spaces.

Features:
- Automatic username detection from HF_TOKEN environment variable
- Auto-generated space names with timestamps
- Local-only mode when no HF_TOKEN is set
- Comprehensive experiment logging
- Optional dataset pushing to Hugging Face Hub

Authentication:
Set HF_TOKEN environment variable to enable automatic space creation:
  Linux/Mac: export HF_TOKEN=your_token_here
  Windows: set HF_TOKEN=your_token_here
  Or: export HUGGINGFACE_HUB_TOKEN=your_token_here

Get your token from: https://huggingface.co/settings/tokens
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import (
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import HfApi
import trackio as wandb


def validate_hf_token(token: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a Hugging Face token and return the username.

    Args:
        token (str): The Hugging Face token to validate

    Returns:
        Tuple[bool, Optional[str], Optional[str]]:
            - success: True if token is valid, False otherwise
            - username: The username associated with the token (if valid)
            - error_message: Error message if validation failed
    """
    try:
        # Create API client with token directly
        api = HfApi(token=token)

        # Try to get user info - this will fail if token is invalid
        user_info = api.whoami()

        # Extract username from user info
        username = user_info.get("name", user_info.get("username"))

        if not username:
            return False, None, "Could not retrieve username from token"

        return True, username, None

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, None, "Invalid token - unauthorized access"
        elif "403" in error_msg:
            return False, None, "Token lacks required permissions"
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            return False, None, f"Network error: {error_msg}"
        else:
            return False, None, f"Validation error: {error_msg}"


def get_default_space_name(project_type: str = "voxtral-asr-finetuning") -> str:
    """
    Generate a default space name with username and timestamp.

    Args:
        project_type: Type of project (e.g., "voxtral-asr-finetuning", "voxtral-lora-finetuning")

    Returns:
        str: Default space name in format "username/project-type-timestamp"
    """
    try:
        # Get token from environment variables only
        import os
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        if not token:
            print("Warning: No HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable found.")
            print("Set HF_TOKEN environment variable to enable automatic space creation.")
            print("Example: export HF_TOKEN=your_token_here")
            print("Falling back to local-only mode.")
            return None

        # Validate token and get username
        success, username, error = validate_hf_token(token)
        if success and username:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return f"{username}/{project_type}-{timestamp}"
        else:
            print(f"Warning: Token validation failed: {error}")
            print("Falling back to local-only mode.")
            return None

    except Exception as e:
        print(f"Warning: Failed to generate default space name: {e}")
        return None


class VoxtralDataCollator:
    """Data collator for Voxtral STT training - processes audio and text."""
    
    def __init__(self, processor, model_id):
        self.processor = processor
        self.model_id = model_id
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        """
        Each feature should have:
          - "audio": raw audio (whatever your processor expects)
          - "text":  transcription string
        """
        texts  = [f["text"] for f in features]
        audios = [f["audio"]["array"] for f in features]

        # 1) Build the PROMPT part: [AUDIO]…[AUDIO] <transcribe>
        prompt = self.processor.apply_transcription_request(  # (same method you used)
            language="en",
            model_id=self.model_id if hasattr(self, "model_id") else None,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        # prompt["input_ids"]: shape [B, L_prompt]
        # keep any extra fields (e.g., audio features) to pass through to the model
        passthrough = {k: v for k, v in prompt.items()
                       if k not in ("input_ids", "attention_mask")}

        prompt_ids = prompt["input_ids"]           # [B, Lp]
        prompt_attn = prompt["attention_mask"]     # [B, Lp]
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer
        # 2) Tokenize transcriptions WITHOUT padding; we'll pad after concatenation
        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate: input_ids = [PROMPT] + [TEXT]
        input_ids, attention_mask, labels = [], [], []
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            ids  = p_ids + t_ids
            attn = p_att + [1] * len(t_ids)
            # labels: mask prompt tokens, learn only on text tokens
            lab  = [-100] * len(p_ids) + t_ids

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids      = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0,      max_len) for x in attention_mask]
        labels         = [pad_to(x, -100,   max_len) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # 5) Include processor outputs needed by the model (e.g., audio features)
        for k, v in passthrough.items():
            batch[k] = v

        return batch

def _load_jsonl_dataset(jsonl_path: str) -> Dataset:
    """Load local JSONL with fields {audio_path, text} into a Dataset with audio column."""
    records = []
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {jsonl_path}")
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            audio_path = obj.get("audio_path") or obj.get("audio")
            text = obj.get("text")
            if not audio_path or text is None:
                continue
            records.append({"audio": audio_path, "text": text})
    if not records:
        raise ValueError("No valid records found in JSONL. Expect keys: audio_path, text")
    ds = Dataset.from_list(records)
    # Cast the audio column from file paths and resample to 16kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def load_and_prepare_dataset(dataset_jsonl: str | None, dataset_name: str | None, dataset_config: str | None,
                             train_count: int, eval_count: int):
    """Load and prepare dataset for training.

    Priority: local JSONL > HF dataset name/config > fallback tiny sample.
    """
    if dataset_jsonl:
        print(f"Loading local JSONL dataset: {dataset_jsonl}")
        ds = _load_jsonl_dataset(dataset_jsonl)
    else:
        ds_name = dataset_name or "hf-audio/esb-datasets-test-only-sorted"
        ds_cfg = dataset_config or "voxpopuli"
        print(f"Loading dataset: {ds_name}/{ds_cfg}")
        ds = load_dataset(ds_name, ds_cfg, split="test")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    total = len(ds)
    train_end = min(train_count, total)
    eval_end = min(train_end + eval_count, total)
    train_dataset = ds.select(range(train_end))
    eval_dataset = ds.select(range(train_end, eval_end)) if eval_end > train_end else None
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="Full fine-tune Voxtral for ASR")
    parser.add_argument("--dataset-jsonl", type=str, default=None, help="Path to local JSONL with {audio_path, text}")
    parser.add_argument("--dataset-name", type=str, default=None, help="HF dataset repo (if not using JSONL)")
    parser.add_argument("--dataset-config", type=str, default=None, help="HF dataset config/subset")
    parser.add_argument("--train-count", type=int, default=100, help="Number of training samples to use")
    parser.add_argument("--eval-count", type=int, default=50, help="Number of eval samples to use")
    parser.add_argument("--model-checkpoint", type=str, default="mistralai/Voxtral-Mini-3B-2507")
    parser.add_argument("--output-dir", type=str, default="./voxtral-finetuned")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--trackio-space", type=str, default=None,
                        help="Hugging Face Space ID for trackio logging (format: username/space-name). If not provided, will auto-generate based on HF token")
    parser.add_argument("--push-dataset", action="store_true",
                        help="Push the training dataset to Hugging Face Hub after training")
    parser.add_argument("--dataset-repo", type=str, default=None,
                        help="Dataset repository name for pushing dataset (format: username/dataset-name)")
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    output_dir = args.output_dir

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")

    # Determine trackio space
    trackio_space = args.trackio_space
    if not trackio_space:
        trackio_space = get_default_space_name("voxtral-asr-finetuning")

    # Initialize wandb (trackio) for experiment tracking
    wandb_enabled = False
    if trackio_space:
        print(f"Initializing wandb (trackio) with space: {trackio_space}")
        try:
            # Set a shorter timeout for trackio initialization
            import os
            original_timeout = os.environ.get('TRACKIO_TIMEOUT', '30')
            os.environ['TRACKIO_TIMEOUT'] = '30'  # 30 second timeout
            
            wandb.init(
                project="voxtral-finetuning",
                config={
                    "model_checkpoint": model_checkpoint,
                    "output_dir": output_dir,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "train_count": args.train_count,
                    "eval_count": args.eval_count,
                    "dataset_jsonl": args.dataset_jsonl,
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                },
                space_id=trackio_space
            )
            wandb_enabled = True
            print("✅ Wandb (trackio) initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize wandb (trackio) with space: {e}")
            print("🔄 Falling back to local-only mode...")
            try:
                wandb.init(
                    project="voxtral-finetuning",
                    config={
                        "model_checkpoint": model_checkpoint,
                        "output_dir": output_dir,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "epochs": args.epochs,
                        "train_count": args.train_count,
                        "eval_count": args.eval_count,
                        "dataset_jsonl": args.dataset_jsonl,
                        "dataset_name": args.dataset_name,
                        "dataset_config": args.dataset_config,
                    }
                )
                wandb_enabled = True
                print("✅ Wandb (trackio) initialized in local-only mode")
            except Exception as fallback_e:
                print(f"❌ Failed to initialize wandb (trackio) in local mode: {fallback_e}")
                print("⚠️ Training will continue without experiment tracking")
    else:
        print("Initializing wandb (trackio) in local-only mode")
        try:
            wandb.init(
                project="voxtral-finetuning",
                config={
                    "model_checkpoint": model_checkpoint,
                    "output_dir": output_dir,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "train_count": args.train_count,
                    "eval_count": args.eval_count,
                    "dataset_jsonl": args.dataset_jsonl,
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                }
            )
            wandb_enabled = True
            print("✅ Wandb (trackio) initialized in local-only mode")
        except Exception as e:
            print(f"❌ Failed to initialize wandb (trackio): {e}")
            print("⚠️ Training will continue without experiment tracking")

    print("Loading processor and model...")
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_jsonl=args.dataset_jsonl,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_count=args.train_count,
        eval_count=args.eval_count,
    )

    data_collator = VoxtralDataCollator(processor, model_checkpoint)

    # Disable Transformers Trackio callback to avoid httpx timeouts; logging is handled via trackio.init()
    report_to = []

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_steps=args.save_steps if eval_dataset else None,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_num_workers=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    if eval_dataset:
        results = trainer.evaluate()
        print(f"Final evaluation results: {results}")
        # Log final evaluation results if wandb is enabled
        if wandb_enabled:
            wandb.log(results)

    # Push dataset to Hub if requested
    if args.push_dataset and args.dataset_jsonl:
        print("Pushing dataset to Hugging Face Hub...")
        try:
            from pathlib import Path
            import subprocess

            dataset_repo = args.dataset_repo
            if not dataset_repo:
                # Auto-generate dataset repo name
                if trackio_space:
                    username = trackio_space.split('/')[0]
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    dataset_repo = f"{username}/voxtral-dataset-{timestamp}"
                else:
                    print("Warning: Cannot auto-generate dataset repo name without HF token")
                    dataset_repo = f"voxtral-dataset-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Call the push script
            push_cmd = [
                "python", str(Path(__file__).parent / "push_to_huggingface.py"),
                "dataset", args.dataset_jsonl, dataset_repo
            ]

            result = subprocess.run(push_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Dataset pushed to: https://huggingface.co/datasets/{dataset_repo}")
            else:
                print(f"❌ Failed to push dataset: {result.stderr}")

        except Exception as e:
            print(f"❌ Error pushing dataset: {e}")

    # Finish wandb logging if enabled
    if wandb_enabled:
        wandb.finish()

    print("Training completed successfully!")

if __name__ == "__main__":
    main()