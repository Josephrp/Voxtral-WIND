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


PHRASES = [
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
]

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

    # Recording grid with dynamic text readouts
    phrase_texts_state = gr.State(PHRASES)
    phrase_markdowns: list[gr.Markdown] = []
    rec_components = []
    with gr.Column():
        for idx, phrase in enumerate(PHRASES):
            md = gr.Markdown(f"**{idx+1}. {phrase}**")
            phrase_markdowns.append(md)
            comp = gr.Audio(sources="microphone", type="numpy", label=f"Recording {idx+1}")
            rec_components.append(comp)

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
            label_text = (texts[i] if isinstance(texts, list) and i < len(texts) else (PHRASES[i] if i < len(PHRASES) else ""))
            rows.append({"audio_path": str(out_path), "text": label_text})
        jsonl_path = dataset_dir / "data.jsonl"
        _write_jsonl(rows, jsonl_path)
        return str(jsonl_path)

    save_rec_btn.click(_collect_preloaded_recs, rec_components + [phrase_texts_state], [jsonl_out])

    # Quick sample from VoxPopuli (few random rows)
    with gr.Row():
        vp_lang = gr.Dropdown(choices=["en", "de", "fr", "es", "it", "pl", "ro", "hu", "cs", "nl", "fi", "hr", "sk", "sl", "et", "lt"], value="en", label="VoxPopuli language")
        vp_samples = gr.Number(value=20, precision=0, label="Num samples")
        vp_split = gr.Dropdown(choices=["train", "validation", "test"], value="train", label="Split")
        vp_btn = gr.Button("Use VoxPopuli sample")

        def _collect_voxpopuli(lang_code: str, num_samples: int, split: str):
            import sys
            # Workaround for dill on Python 3.13 expecting __main__ during import
            if "__main__" not in sys.modules:
                sys.modules["__main__"] = sys.modules[__name__]
            from datasets import load_dataset, Audio  # type: ignore
            import random
            ds = load_dataset("facebook/voxpopuli", lang_code, split=split)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            # shuffle and select
            total = len(ds)
            k = max(1, min(int(num_samples or 1), total))
            ds = ds.shuffle(seed=random.randint(1, 10_000))
            ds_sel = ds.select(range(k))

            dataset_dir = PROJECT_ROOT / "datasets" / "voxtral_user"
            rows: list[dict] = []
            texts: list[str] = []
            for ex in ds_sel:
                audio = ex.get("audio") or {}
                path = audio.get("path")
                text = ex.get("normalized_text") or ex.get("raw_text") or ""
                if path and text is not None:
                    rows.append({"audio_path": path, "text": text})
                    texts.append(str(text))
            jsonl_path = dataset_dir / "data.jsonl"
            _write_jsonl(rows, jsonl_path)
            # Build markdown content updates for on-screen prompts
            md_updates = []
            for i in range(len(phrase_markdowns)):
                t = texts[i] if i < len(texts) else ""
                md_updates.append(f"**{i+1}. {t}**")
            return (str(jsonl_path), texts, *md_updates)

        vp_btn.click(
            _collect_voxpopuli,
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
    demo.queue().launch(server_name=server_name, server_port=server_port, mcp_server=True)


