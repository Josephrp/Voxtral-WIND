#!/usr/bin/env python3
"""
Voxtral ASR Fine-tuning Interface

Features:
- Collect a personal voice dataset (upload WAV/FLAC + transcripts or record mic audio)
- Build a JSONL dataset ({audio_path, text}) at 16kHz
- Fine-tune Voxtral (LoRA or full) with streamed logs
- Push model to Hugging Face Hub
- Deploy a Voxtral ASR demo Space

Env tokens (optional):
- HF_WRITE_TOKEN or HF_TOKEN: write access token
- HF_READ_TOKEN: optional read token
- HF_USERNAME: fallback username if not derivable from token
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Generator, Optional, Tuple

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent


def get_python() -> str:
    import sys
    return sys.executable or "python"


def get_username_from_token(token: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi  # type: ignore
        api = HfApi(token=token)
        info = api.whoami()
        if isinstance(info, dict):
            return info.get("name") or info.get("username")
        if isinstance(info, str):
            return info
    except Exception:
        return None
    return None


def run_command_stream(args: list[str], env: Dict[str, str], cwd: Optional[Path] = None) -> Generator[str, None, int]:
    import subprocess
    import shlex
    yield f"$ {' '.join(shlex.quote(a) for a in ([get_python()] + args))}"
    process = subprocess.Popen(
        [get_python()] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(cwd or PROJECT_ROOT),
        bufsize=1,
        universal_newlines=True,
    )
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        yield line.rstrip()
    process.stdout.close()
    code = process.wait()
    yield f"[exit_code={code}]"
    return code


def detect_nvidia_driver() -> Tuple[bool, str]:
    """Detect NVIDIA driver/GPU presence with multiple strategies.

    Returns (available, human_message).
    """
    # 1) Try torch CUDA
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                num = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(num)]
                return True, f"NVIDIA GPU detected: {', '.join(names)}"
            except Exception:
                return True, "NVIDIA GPU detected (torch.cuda available)"
    except Exception:
        pass

    # 2) Try NVML via pynvml
    try:
        import pynvml  # type: ignore
        try:
            pynvml.nvmlInit()
            cnt = pynvml.nvmlDeviceGetCount()
            names = []
            for i in range(cnt):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                names.append(pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore"))
            drv = pynvml.nvmlSystemGetDriverVersion().decode("utf-8", errors="ignore")
            pynvml.nvmlShutdown()
            if cnt > 0:
                return True, f"NVIDIA driver {drv}; GPUs: {', '.join(names)}"
        except Exception:
            pass
    except Exception:
        pass

    # 3) Try nvidia-smi
    try:
        import subprocess
        res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            return True, res.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return False, "No NVIDIA driver/GPU detected"


def duplicate_space_hint() -> str:
    space_id = os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID")
    if space_id:
        space_url = f"https://huggingface.co/spaces/{space_id}"
        dup_url = f"{space_url}?duplicate=true"
        return (
            f"ℹ️ No NVIDIA driver detected. If you're on Hugging Face Spaces, "
            f"please duplicate this Space to GPU hardware: [Duplicate this Space]({dup_url})."
        )
    return (
        "ℹ️ No NVIDIA driver detected. To enable training, run on a machine with an NVIDIA GPU/driver "
        "or duplicate this Space on Hugging Face with GPU hardware."
    )


def _write_jsonl(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _save_uploaded_dataset(files: list, transcripts: list[str]) -> str:
    dataset_dir = PROJECT_ROOT / "datasets" / "voxtral_user"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, fpath in enumerate(files or []):
        if i >= len(transcripts):
            break
        rows.append({"audio_path": fpath, "text": transcripts[i] or ""})
    jsonl_path = dataset_dir / "data.jsonl"
    _write_jsonl(rows, jsonl_path)
    return str(jsonl_path)


def _save_recordings(recordings: list[tuple[int, list]], transcripts: list[str]) -> str:
    import soundfile as sf
    dataset_dir = PROJECT_ROOT / "datasets" / "voxtral_user"
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, rec in enumerate(recordings or []):
        if rec is None:
            continue
        if i >= len(transcripts):
            break
        sr, data = rec
        out_path = wav_dir / f"rec_{i:04d}.wav"
        sf.write(str(out_path), data, sr)
        rows.append({"audio_path": str(out_path), "text": transcripts[i] or ""})
    jsonl_path = dataset_dir / "data.jsonl"
    _write_jsonl(rows, jsonl_path)
    return str(jsonl_path)


def start_voxtral_training(
    use_lora: bool,
    base_model: str,
    repo_short: str,
    jsonl_path: str,
    train_count: int,
    eval_count: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    epochs: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    freeze_audio_tower: bool,
    push_to_hub: bool,
    deploy_demo: bool,
) -> Generator[str, None, None]:
    env = os.environ.copy()
    write_token = env.get("HF_WRITE_TOKEN") or env.get("HF_TOKEN")
    read_token = env.get("HF_READ_TOKEN")
    username = get_username_from_token(write_token or "") or env.get("HF_USERNAME") or ""
    output_dir = PROJECT_ROOT / "outputs" / repo_short

    # 1) Train
    script = PROJECT_ROOT / ("scripts/train_lora.py" if use_lora else "scripts/train.py")
    args = [str(script)]
    if jsonl_path:
        args += ["--dataset-jsonl", jsonl_path]
    args += [
        "--model-checkpoint", base_model,
        "--train-count", str(train_count),
        "--eval-count", str(eval_count),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--learning-rate", str(learning_rate),
        "--epochs", str(epochs),
        "--output-dir", str(output_dir),
        "--save-steps", "50",
    ]
    if use_lora:
        args += [
            "--lora-r", str(lora_r),
            "--lora-alpha", str(lora_alpha),
            "--lora-dropout", str(lora_dropout),
        ]
        if freeze_audio_tower:
            args += ["--freeze-audio-tower"]
    for line in run_command_stream(args, env):
        yield line

    # 2) Push to Hub
    if push_to_hub:
        repo_name = f"{username}/{repo_short}" if username else repo_short
        push_args = [
            str(PROJECT_ROOT / "scripts/push_to_huggingface.py"),
            str(output_dir),
            repo_name,
        ]
        for line in run_command_stream(push_args, env):
            yield line

    # 3) Deploy demo Space
    if deploy_demo and username:
        deploy_args = [
            str(PROJECT_ROOT / "scripts/deploy_demo_space.py"),
            "--hf-token", write_token or "",
            "--hf-username", username,
            "--model-id", f"{username}/{repo_short}",
            "--demo-type", "voxtral",
            "--space-name", f"{repo_short}-demo",
        ]
        for line in run_command_stream(deploy_args, env):
            yield line


def load_multilingual_phrases(language="en", max_phrases=None, split="train"):
    """Load phrases from various multilingual speech datasets.

    Tries multiple datasets in order of preference:
    1. Common Voice (most reliable and up-to-date)
    2. FLEURS (Google's multilingual dataset)
    3. Fallback to basic phrases

    Args:
        language: Language code (e.g., 'en', 'de', 'fr', etc.)
        max_phrases: Maximum number of phrases to load (None for all available)
        split: Dataset split to use ('train', 'validation', 'test')

    Returns:
        List of normalized text phrases
    """
    from datasets import load_dataset
    import random

    # Language code mapping for different datasets
    lang_mappings = {
        "en": {"common_voice": "en", "fleurs": "en_us"},
        "de": {"common_voice": "de", "fleurs": "de_de"},
        "fr": {"common_voice": "fr", "fleurs": "fr_fr"},
        "es": {"common_voice": "es", "fleurs": "es_419"},
        "it": {"common_voice": "it", "fleurs": "it_it"},
        "pt": {"common_voice": "pt", "fleurs": "pt_br"},
        "pl": {"common_voice": "pl", "fleurs": "pl_pl"},
        "nl": {"common_voice": "nl", "fleurs": "nl_nl"},
        "ru": {"common_voice": "ru", "fleurs": "ru_ru"},
        "ar": {"common_voice": "ar", "fleurs": "ar_eg"},
        "zh": {"common_voice": "zh-CN", "fleurs": "zh_cn"},
        "ja": {"common_voice": "ja", "fleurs": "ja_jp"},
        "ko": {"common_voice": "ko", "fleurs": "ko_kr"},
    }

    lang_config = lang_mappings.get(language, {"common_voice": language, "fleurs": f"{language}_{language}"})

    # Try Common Voice first (most reliable)
    try:
        print(f"Trying Common Voice dataset for language: {language}")
        cv_lang = lang_config["common_voice"]
        ds = load_dataset("mozilla-foundation/common_voice_11_0", cv_lang, split=split, streaming=True)

        phrases = []
        count = 0
        for example in ds:
            if max_phrases and count >= max_phrases:
                break
            text = example.get("sentence", "").strip()
            if text and len(text) > 10:  # Filter out very short phrases
                phrases.append(text)
                count += 1

        if phrases:
            print(f"Successfully loaded {len(phrases)} phrases from Common Voice")
            random.shuffle(phrases)
            return phrases

    except Exception as e:
        print(f"Common Voice failed: {e}")

    # Try FLEURS as backup
    try:
        print(f"Trying FLEURS dataset for language: {language}")
        fleurs_lang = lang_config["fleurs"]
        ds = load_dataset("google/fleurs", fleurs_lang, split=split, streaming=True)

        phrases = []
        count = 0
        for example in ds:
            if max_phrases and count >= max_phrases:
                break
            text = example.get("transcription", "").strip()
            if text and len(text) > 10:  # Filter out very short phrases
                phrases.append(text)
                count += 1

        if phrases:
            print(f"Successfully loaded {len(phrases)} phrases from FLEURS")
            random.shuffle(phrases)
            return phrases

    except Exception as e:
        print(f"FLEURS failed: {e}")

    # Final fallback to basic phrases
    print("All dataset loading attempts failed, using fallback phrases")
    fallback_phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Please say your full name.",
        "Today is a good day to learn something new.",
        "Artificial intelligence helps with many tasks.",
        "I enjoy reading books and listening to music.",
        "This is a sample sentence for testing speech.",
        "Speak clearly and at a normal pace.",
        "Numbers like one, two, three are easy to say.",
        "The weather is sunny with a chance of rain.",
        "Thank you for taking the time to help.",
        "Hello, how are you today?",
        "I would like to order a pizza.",
        "The meeting is scheduled for tomorrow.",
        "Please call me back as soon as possible.",
        "Thank you for your assistance.",
        "Can you help me with this problem?",
        "I need to make a reservation.",
        "The weather looks beautiful outside.",
        "Let's go for a walk in the park.",
        "I enjoy listening to classical music.",
        "What time does the store open?",
        "I forgot my password again.",
        "Please send me the invoice.",
        "The project is almost complete.",
        "I appreciate your hard work.",
        "Let's schedule a meeting next week.",
        "The food tastes delicious.",
        "I need to buy some groceries.",
        "Please turn off the lights.",
        "The presentation went very well.",
    ]

    if max_phrases:
        fallback_phrases = random.sample(fallback_phrases, min(max_phrases, len(fallback_phrases)))
    else:
        random.shuffle(fallback_phrases)

    return fallback_phrases

# Initialize phrases dynamically
DEFAULT_LANGUAGE = "en"  # Default to English
ALL_PHRASES = load_multilingual_phrases(DEFAULT_LANGUAGE, max_phrases=None)

with gr.Blocks(title="Voxtral ASR Fine-tuning") as demo:
    has_gpu, gpu_msg = detect_nvidia_driver()
    if has_gpu:
        gr.HTML(
            f"""
            <div style="background-color: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px; text-align: center;">
                <p style="color: rgb(59, 130, 246); margin: 0; font-size: 14px; font-weight: 600;">
                    ✅ NVIDIA GPU ready — {gpu_msg}
                </p>
                <p style="color: rgb(59, 130, 246); margin: 6px 0 0; font-size: 12px;">
                    Set HF_WRITE_TOKEN/HF_TOKEN in environment to enable Hub push.
                </p>
            </div>
            """
        )
    else:
        hint_md = duplicate_space_hint()
        gr.HTML(
            f"""
            <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px; text-align: center;">
                <p style="color: rgb(234, 88, 12); margin: 0; font-size: 14px; font-weight: 600;">
                    ⚠️ No NVIDIA GPU/driver detected — training requires a GPU runtime
                </p>
                <p style="color: rgb(234, 88, 12); margin: 6px 0 0; font-size: 12px;">
                    {hint_md}
                </p>
            </div>
            """
        )

    gr.Markdown("""
    # 🎙️ Voxtral ASR Fine-tuning
    Read the phrases below and record them. Then start fine-tuning.
    """)

    jsonl_out = gr.Textbox(label="Dataset JSONL path", interactive=False, visible=True)

    # Language selection for multilingual phrases
    language_selector = gr.Dropdown(
        choices=[
            "en", "de", "fr", "es", "it", "pt", "pl", "nl", "ru",
            "ar", "zh", "ja", "ko", "tr", "ca", "sv", "fi", "da"
        ],
        value="en",
        label="Language for Speech Phrases",
        info="Select language for phrases from Common Voice, FLEURS, or fallback datasets"
    )

    # Recording grid with dynamic text readouts
    phrase_texts_state = gr.State(ALL_PHRASES)
    visible_rows_state = gr.State(10)  # Start with 10 visible rows
    max_rows = len(ALL_PHRASES)  # No cap on total rows
    phrase_markdowns: list[gr.Markdown] = []
    rec_components = []

    def create_recording_grid(phrases, visible_count=10):
        """Create recording grid components dynamically"""
        markdowns = []
        recordings = []
        for idx, phrase in enumerate(phrases):
            visible = idx < visible_count
            md = gr.Markdown(f"**{idx+1}. {phrase}**", visible=visible)
            markdowns.append(md)
            comp = gr.Audio(sources="microphone", type="numpy", label=f"Recording {idx+1}", visible=visible)
            recordings.append(comp)
        return markdowns, recordings

    # Initial grid creation
    with gr.Column():
        phrase_markdowns, rec_components = create_recording_grid(ALL_PHRASES, 10)

    # Add more rows button
    add_rows_btn = gr.Button("➕ Add 10 More Rows", variant="secondary")

    def add_more_rows(current_visible, current_phrases):
        """Add 10 more rows by making them visible"""
        new_visible = min(current_visible + 10, len(current_phrases))
        visibility_updates = []
        for i in range(len(current_phrases)):
            if i < new_visible:
                visibility_updates.append(gr.update(visible=True))
            else:
                visibility_updates.append(gr.update(visible=False))
        return [new_visible] + visibility_updates

    def change_language(language):
        """Change the language and reload phrases from multilingual datasets"""
        new_phrases = load_multilingual_phrases(language, max_phrases=None)
        # Reset visible rows to 10
        visible_count = min(10, len(new_phrases))

        # Create combined updates for existing components (up to current length)
        current_len = len(phrase_markdowns)
        combined_updates = []

        # Update existing components
        for i in range(current_len):
            if i < len(new_phrases):
                if i < visible_count:
                    combined_updates.append(gr.update(value=f"**{i+1}. {new_phrases[i]}**", visible=True))
                else:
                    combined_updates.append(gr.update(visible=False))
            else:
                combined_updates.append(gr.update(visible=False))

        # If we have more phrases than components, we can't update them via Gradio
        # The interface will need to be reloaded for significantly different phrase counts
        return [new_phrases, visible_count] + combined_updates

    # Connect language change to phrase reloading
    language_selector.change(
        change_language,
        inputs=[language_selector],
        outputs=[phrase_texts_state, visible_rows_state] + phrase_markdowns + rec_components
    )

    add_rows_btn.click(
        add_more_rows,
        inputs=[visible_rows_state, phrase_texts_state],
        outputs=[visible_rows_state] + phrase_markdowns + rec_components
    )

    # Advanced options accordion
    with gr.Accordion("Advanced options", open=False):
        base_model = gr.Textbox(value="mistralai/Voxtral-Mini-3B-2507", label="Base Voxtral model")
        use_lora = gr.Checkbox(value=True, label="Use LoRA (parameter-efficient)")
        with gr.Row():
            batch_size = gr.Number(value=2, precision=0, label="Batch size")
            grad_accum = gr.Number(value=4, precision=0, label="Grad accum")
        with gr.Row():
            learning_rate = gr.Number(value=5e-5, precision=6, label="Learning rate")
            epochs = gr.Number(value=3.0, precision=2, label="Epochs")
        with gr.Accordion("LoRA settings", open=False):
            lora_r = gr.Number(value=8, precision=0, label="LoRA r")
            lora_alpha = gr.Number(value=32, precision=0, label="LoRA alpha")
            lora_dropout = gr.Number(value=0.0, precision=3, label="LoRA dropout")
            freeze_audio_tower = gr.Checkbox(value=True, label="Freeze audio tower")
        with gr.Row():
            train_count = gr.Number(value=100, precision=0, label="Train samples")
            eval_count = gr.Number(value=50, precision=0, label="Eval samples")
        repo_short = gr.Textbox(value=f"voxtral-finetune-{datetime.now().strftime('%Y%m%d_%H%M%S')}", label="Model repo (short)")
        push_to_hub = gr.Checkbox(value=True, label="Push to HF Hub after training")
        deploy_demo = gr.Checkbox(value=True, label="Deploy demo Space after push")

        gr.Markdown("### Upload audio + transcripts (optional)")
        upload_audio = gr.File(file_count="multiple", type="filepath", label="Upload WAV/FLAC files (optional)")
        transcripts_box = gr.Textbox(lines=6, label="Transcripts (one per line, aligned with files)")
        save_upload_btn = gr.Button("Save uploaded dataset")

        def _collect_upload(files, txt):
            lines = [s.strip() for s in (txt or "").splitlines() if s.strip()]
            return _save_uploaded_dataset(files or [], lines)

        save_upload_btn.click(_collect_upload, [upload_audio, transcripts_box], [jsonl_out])

    # Save recordings button
    save_rec_btn = gr.Button("Save recordings as dataset")

    def _collect_preloaded_recs(*recs_and_texts):
        import soundfile as sf
        dataset_dir = PROJECT_ROOT / "datasets" / "voxtral_user"
        wav_dir = dataset_dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        if not recs_and_texts:
            jsonl_path = dataset_dir / "data.jsonl"
            _write_jsonl(rows, jsonl_path)
            return str(jsonl_path)
        texts = recs_and_texts[-1]
        recs = recs_and_texts[:-1]
        for i, rec in enumerate(recs):
            if rec is None:
                continue
            sr, data = rec
            out_path = wav_dir / f"rec_{i:04d}.wav"
            sf.write(str(out_path), data, sr)
            # Use the full phrase list (ALL_PHRASES) instead of just PHRASES
            label_text = (texts[i] if isinstance(texts, list) and i < len(texts) else (ALL_PHRASES[i] if i < len(ALL_PHRASES) else ""))
            rows.append({"audio_path": str(out_path), "text": label_text})
        jsonl_path = dataset_dir / "data.jsonl"
        _write_jsonl(rows, jsonl_path)
        return str(jsonl_path)

    save_rec_btn.click(_collect_preloaded_recs, rec_components + [phrase_texts_state], [jsonl_out])

    # Quick sample from multilingual datasets (Common Voice, etc.)
    with gr.Row():
        vp_lang = gr.Dropdown(choices=["en", "de", "fr", "es", "it", "pl", "pt", "nl", "ru", "ar", "zh", "ja", "ko"], value="en", label="Sample Language")
        vp_samples = gr.Number(value=20, precision=0, label="Num samples")
        vp_split = gr.Dropdown(choices=["train", "validation", "test"], value="train", label="Split")
        vp_btn = gr.Button("Use Multilingual Dataset Sample")

        def _collect_multilingual_sample(lang_code: str, num_samples: int, split: str):
            """Load sample from multilingual datasets (Common Voice preferred)"""
            from datasets import load_dataset, Audio
            import random

            # Language code mapping for Common Voice
            cv_lang_map = {
                "en": "en", "de": "de", "fr": "fr", "es": "es", "it": "it",
                "pl": "pl", "pt": "pt", "nl": "nl", "ru": "ru", "ar": "ar",
                "zh": "zh-CN", "ja": "ja", "ko": "ko"
            }

            cv_lang = cv_lang_map.get(lang_code, lang_code)

            try:
                # Try Common Voice first
                ds = load_dataset("mozilla-foundation/common_voice_11_0", cv_lang, split=split, streaming=True)
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))

                dataset_dir = PROJECT_ROOT / "datasets" / "voxtral_user"
                rows: list[dict] = []
                texts: list[str] = []

                count = 0
                for ex in ds:
                    if count >= num_samples:
                        break

                    audio = ex.get("audio") or {}
                    path = audio.get("path")
                    text = ex.get("sentence", "").strip()

                    if path and text and len(text) > 10:
                        rows.append({"audio_path": path, "text": text})
                        texts.append(str(text))
                        count += 1

                if rows:
                    jsonl_path = dataset_dir / "data.jsonl"
                    _write_jsonl(rows, jsonl_path)

                    # Build markdown content updates for on-screen prompts
                    combined_updates = []
                    for i in range(len(phrase_markdowns)):
                        t = texts[i] if i < len(texts) else ""
                        if i < len(texts):
                            combined_updates.append(gr.update(value=f"**{i+1}. {t}**", visible=True))
                        else:
                            combined_updates.append(gr.update(visible=False))

                    return (str(jsonl_path), texts, *combined_updates)

            except Exception as e:
                print(f"Common Voice sample loading failed: {e}")

            # Fallback: generate synthetic samples with text only
            print("Using fallback: generating text-only samples")
            phrases = load_multilingual_phrases(lang_code, max_phrases=num_samples)
            texts = phrases[:num_samples]

            dataset_dir = PROJECT_ROOT / "datasets" / "voxtral_user"
            rows = [{"audio_path": "", "text": text} for text in texts]
            jsonl_path = dataset_dir / "data.jsonl"
            _write_jsonl(rows, jsonl_path)

            # Build markdown content updates for on-screen prompts
            combined_updates = []
            for i in range(len(phrase_markdowns)):
                t = texts[i] if i < len(texts) else ""
                if i < len(texts):
                    combined_updates.append(gr.update(value=f"**{i+1}. {t}**", visible=True))
                else:
                    combined_updates.append(gr.update(visible=False))

            return (str(jsonl_path), texts, *combined_updates)

        vp_btn.click(
            _collect_multilingual_sample,
            [vp_lang, vp_samples, vp_split],
            [jsonl_out, phrase_texts_state] + phrase_markdowns,
        )

    start_btn = gr.Button("Start Fine-tuning")
    logs_box = gr.Textbox(label="Logs", lines=20)

    start_btn.click(
        start_voxtral_training,
        inputs=[
            use_lora, base_model, repo_short, jsonl_out, train_count, eval_count,
            batch_size, grad_accum, learning_rate, epochs,
            lora_r, lora_alpha, lora_dropout, freeze_audio_tower,
            push_to_hub, deploy_demo,
        ],
        outputs=[logs_box],
    )


if __name__ == "__main__":
    server_port = int(os.environ.get("INTERFACE_PORT", "7860"))
    server_name = os.environ.get("INTERFACE_HOST", "0.0.0.0")
    demo.queue().launch(server_name=server_name, server_port=server_port, mcp_server=True, ssr_mode=False)


