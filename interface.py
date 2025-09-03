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
    try:
        cmd_line = ' '.join(shlex.quote(a) for a in ([get_python()] + args))
        yield f"$ {cmd_line}"

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

        if process.stdout is None:
            yield "❌ Error: Could not capture process output"
            return 1

        for line in iter(process.stdout.readline, ""):
            if line.strip():  # Only yield non-empty lines
                yield line.rstrip()

        process.stdout.close()
        code = process.wait()

        if code != 0:
            yield f"❌ Command failed with exit code: {code}"
        else:
            yield f"✅ Command completed successfully (exit code: {code})"

        return code

    except FileNotFoundError as e:
        yield f"❌ Error: Python executable not found: {e}"
        return 1
    except Exception as e:
        yield f"❌ Error running command: {str(e)}"
        return 1


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


def _push_dataset_to_hub(jsonl_path: str, repo_name: str, username: str = "") -> str:
    """Push dataset to Hugging Face Hub"""
    try:
        from huggingface_hub import HfApi, create_repo
        import json
        from pathlib import Path

        token = os.getenv("HF_TOKEN") or os.getenv("HF_WRITE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        if not token:
            return "❌ No HF_TOKEN found. Set HF_TOKEN environment variable to push datasets."

        api = HfApi(token=token)

        # Determine full repo name
        if "/" not in repo_name:
            if not username:
                user_info = api.whoami()
                username = user_info.get("name") or user_info.get("username") or ""
            if username:
                repo_name = f"{username}/{repo_name}"

        # Create dataset repository
        try:
            create_repo(repo_name, repo_type="dataset", token=token, exist_ok=True)
        except Exception as e:
            if "already exists" not in str(e).lower():
                return f"❌ Failed to create dataset repo: {e}"

        # Read the JSONL file
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            return f"❌ Dataset file not found: {jsonl_path}"

        # Upload the JSONL file
        api.upload_file(
            path_or_fileobj=str(jsonl_file),
            path_in_repo="data.jsonl",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )

        # Create a simple README for the dataset
        readme_content = f"""---
dataset_info:
  features:
    - name: audio_path
      dtype: string
    - name: text
      dtype: string
  splits:
    - name: train
      num_bytes: {jsonl_file.stat().st_size}
      num_examples: {sum(1 for _ in open(jsonl_file))}
  download_size: {jsonl_file.stat().st_size}
  dataset_size: {jsonl_file.stat().st_size}
---

# Voxtral ASR Dataset

This dataset was created using the Voxtral ASR Fine-tuning Interface.

## Dataset Structure

- **audio_path**: Path to the audio file
- **text**: Transcription of the audio

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_name}")
```
"""

        # Upload README
        readme_path = jsonl_file.parent / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )

        readme_path.unlink()  # Clean up temp file

        return f"✅ Dataset pushed to: https://huggingface.co/datasets/{repo_name}"

    except Exception as e:
        return f"❌ Failed to push dataset: {e}"


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
) -> str:
    """Start Voxtral training and return collected logs as a string."""
    env = os.environ.copy()
    write_token = env.get("HF_WRITE_TOKEN") or env.get("HF_TOKEN")
    read_token = env.get("HF_READ_TOKEN")
    username = get_username_from_token(write_token or "") or env.get("HF_USERNAME") or ""
    output_dir = PROJECT_ROOT / "outputs" / repo_short

    # Collect all logs
    all_logs = []

    def collect_logs(generator):
        """Helper to collect logs from a generator."""
        for line in generator:
            all_logs.append(line)
            print(line)  # Also print to console for debugging

    try:
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

        all_logs.append("🚀 Starting Voxtral training...")
        collect_logs(run_command_stream(args, env))
        all_logs.append("✅ Training completed!")

        # 2) Push to Hub
        if push_to_hub:
            if not username:
                all_logs.append("❌ Cannot push to Hub: No username available. Set HF_TOKEN or HF_USERNAME.")
            else:
                repo_name = f"{username}/{repo_short}"
                push_args = [
                    str(PROJECT_ROOT / "scripts/push_to_huggingface.py"),
                    "model",
                    str(output_dir),
                    repo_name,
                ]
                all_logs.append(f"📤 Pushing model to Hugging Face Hub: {repo_name}")
                collect_logs(run_command_stream(push_args, env))
                all_logs.append("✅ Model pushed successfully!")

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
            all_logs.append("🚀 Deploying demo Space...")
            collect_logs(run_command_stream(deploy_args, env))
            all_logs.append("✅ Demo Space deployed!")

        # Return all collected logs as a single string
        return "\n".join(all_logs)

    except Exception as e:
        error_msg = f"❌ Error during training: {str(e)}"
        all_logs.append(error_msg)
        print(error_msg)  # Also print to console
        import traceback
        traceback.print_exc()
        return "\n".join(all_logs)


def load_multilingual_phrases(language="en", max_phrases=None, split="train"):
    """Load phrases from NVIDIA Granary dataset.

    Uses the high-quality Granary dataset which contains speech recognition
    and translation data for 25 European languages.

    Args:
        language: Language code (e.g., 'en', 'de', 'fr', etc.)
        max_phrases: Maximum number of phrases to load (None for default 1000)
        split: Dataset split to use ('train', 'validation', 'test')

    Returns:
        List of transcription phrases from Granary dataset
    """
    from datasets import load_dataset
    import random

    # Default to 1000 phrases if not specified
    if max_phrases is None:
        max_phrases = 1000

    # Language code mapping for CohereLabs AYA Collection dataset
    # All Voxtral Mini supported languages are available in AYA Collection
    aya_supported_langs = {
        "en": "english",    # English
        "fr": "french",     # French
        "de": "german",     # German
        "es": "spanish",    # Spanish
        "it": "italian",    # Italian
        "pt": "portuguese", # Portuguese
        "nl": "dutch",      # Dutch
        "hi": "hindi"       # Hindi
    }

    # Map input language to CohereLabs AYA Collection configuration
    aya_lang = aya_supported_langs.get(language)

    if not aya_lang:
        raise Exception(f"Language {language} not supported in CohereLabs AYA Collection dataset")

    try:
        print(f"Loading phrases from CohereLabs AYA Collection dataset for language: {language}")

        # Check for authentication token
        token = os.getenv("HF_TOKEN") or os.getenv("HF_WRITE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        # Try to load CohereLabs AYA Collection dataset for the specified language
        if token:
            try:
                ds = load_dataset("CohereLabs/aya_collection_language_split", aya_lang, split="train", streaming=True, token=token)
                print(f"Successfully loaded CohereLabs AYA Collection {language} dataset")
            except Exception as e:
                # Fallback to other datasets
                print(f"CohereLabs AYA Collection {language} not available ({e}), trying alternative datasets")
                raise Exception("AYA Collection not available")
        else:
            print("No HF_TOKEN found for CohereLabs AYA Collection dataset")
            raise Exception("No token available")

        # Common processing for both dataset types
        phrases = []
        count = 0
        seen_phrases = set()

        # Sample phrases from the dataset
        for example in ds:
            if count >= max_phrases:
                break

            # Extract text from CohereLabs AYA Collection format: combine inputs and targets
            inputs_text = example.get("inputs", "").strip()
            targets_text = example.get("targets", "").strip()
            text = f"{inputs_text} {targets_text}".strip()

            # Filter for quality phrases
            if (text and
                len(text) > 10 and  # Minimum length
                len(text) < 200 and  # Maximum length to avoid very long utterances
                text not in seen_phrases and  # Avoid duplicates
                not text.isdigit() and  # Avoid pure numbers
                not all(c in "0123456789., " for c in text)):  # Avoid mostly numeric

                phrases.append(text)
                seen_phrases.add(text)
                count += 1

        if phrases:
            # Shuffle the phrases for variety
            random.shuffle(phrases)
            dataset_name = "CohereLabs AYA Collection"
            print(f"Successfully loaded {len(phrases)} phrases from {dataset_name} dataset for {language}")
            return phrases

        else:
            print(f"No suitable phrases found in dataset for {language}")
            raise Exception("No phrases found")

    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg:
            print(f"CohereLabs AYA Collection authentication failed for {language}: {e}")
            print("This dataset requires a Hugging Face token. Please set HF_TOKEN environment variable.")
        else:
            print(f"CohereLabs AYA Collection loading failed for {language}: {e}")

        # Fallback to basic phrases if dataset loading fails
        print("Using fallback phrases")

        # Language-specific fallback phrases
        language_fallbacks = {
            "hi": [
                "नमस्ते, आज आप कैसे हैं?",
                "मेरा नाम राजेश कुमार है।",
                "आज का मौसम बहुत अच्छा है।",
                "मैं हिंदी में बात करना चाहता हूं।",
                "कृपया धीरे और स्पष्ट बोलें।",
                "यह एक परीक्षण वाक्य है।",
                "मैं पुस्तकें पढ़ना पसंद करता हूं।",
                "क्या आप मेरी मदद कर सकते हैं?",
                "आपका फोन नंबर क्या है?",
                "मैं कल सुबह आऊंगा।",
                "धन्यवाद, आपका समय देने के लिए।",
                "यह जगह बहुत सुंदर है।",
                "मैं भोजन तैयार करना सीख रहा हूं।",
                "क्या यह रास्ता सही है?",
                "मैं स्कूल जाना चाहता हूं।",
                "आपकी उम्र क्या है?",
                "यह कितने का है?",
                "मैं थक गया हूं।",
                "आप कहां से हैं?",
                "चलिए पार्क में टहलते हैं।"
            ],
            "en": [
                "Hello, how are you today?",
                "My name is John Smith.",
                "The weather is very nice today.",
                "I want to speak in English.",
                "Please speak slowly and clearly.",
                "This is a test sentence.",
                "I enjoy reading books.",
                "Can you help me?",
                "What is your phone number?",
                "I will come tomorrow morning.",
                "Thank you for your time.",
                "This place is very beautiful.",
                "I am learning to cook food.",
                "Is this the right way?",
                "I want to go to school.",
                "How old are you?",
                "How much does this cost?",
                "I am tired.",
                "Where are you from?",
                "Let's go for a walk in the park."
            ],
            "fr": [
                "Bonjour, comment allez-vous aujourd'hui?",
                "Je m'appelle Jean Dupont.",
                "Le temps est très beau aujourd'hui.",
                "Je veux parler en français.",
                "Parlez lentement et clairement s'il vous plaît.",
                "Ceci est une phrase de test.",
                "J'aime lire des livres.",
                "Pouvez-vous m'aider?",
                "Quel est votre numéro de téléphone?",
                "Je viendrai demain matin.",
                "Merci pour votre temps.",
                "Cet endroit est très beau.",
                "J'apprends à cuisiner.",
                "Est-ce le bon chemin?",
                "Je veux aller à l'école.",
                "Quel âge avez-vous?",
                "Combien cela coûte-t-il?",
                "Je suis fatigué.",
                "D'où venez-vous?",
                "Allons nous promener dans le parc."
            ],
            "de": [
                "Hallo, wie geht es Ihnen heute?",
                "Mein Name ist Hans Müller.",
                "Das Wetter ist heute sehr schön.",
                "Ich möchte auf Deutsch sprechen.",
                "Sprechen Sie bitte langsam und deutlich.",
                "Dies ist ein Testsatz.",
                "Ich lese gerne Bücher.",
                "Können Sie mir helfen?",
                "Wie ist Ihre Telefonnummer?",
                "Ich komme morgen früh.",
                "Vielen Dank für Ihre Zeit.",
                "Dieser Ort ist sehr schön.",
                "Ich lerne kochen.",
                "Ist das der richtige Weg?",
                "Ich möchte zur Schule gehen.",
                "Wie alt sind Sie?",
                "Wie viel kostet das?",
                "Ich bin müde.",
                "Woher kommen Sie?",
                "Lassen Sie uns im Park spazieren gehen."
            ],
            "es": [
                "Hola, ¿cómo estás hoy?",
                "Me llamo Juan García.",
                "El tiempo está muy bueno hoy.",
                "Quiero hablar en español.",
                "Por favor habla despacio y claro.",
                "Esta es una oración de prueba.",
                "Me gusta leer libros.",
                "¿Puedes ayudarme?",
                "¿Cuál es tu número de teléfono?",
                "Vendré mañana por la mañana.",
                "Gracias por tu tiempo.",
                "Este lugar es muy bonito.",
                "Estoy aprendiendo a cocinar.",
                "¿Es este el camino correcto?",
                "Quiero ir a la escuela.",
                "¿Cuántos años tienes?",
                "¿Cuánto cuesta esto?",
                "Estoy cansado.",
                "¿De dónde eres?",
                "Vamos a caminar por el parque."
            ],
            "it": [
                "Ciao, come stai oggi?",
                "Mi chiamo Mario Rossi.",
                "Il tempo è molto bello oggi.",
                "Voglio parlare in italiano.",
                "Per favore parla lentamente e chiaramente.",
                "Questa è una frase di prova.",
                "Mi piace leggere libri.",
                "Puoi aiutarmi?",
                "Qual è il tuo numero di telefono?",
                "Verrò domani mattina.",
                "Grazie per il tuo tempo.",
                "Questo posto è molto bello.",
                "Sto imparando a cucinare.",
                "È questa la strada giusta?",
                "Voglio andare a scuola.",
                "Quanti anni hai?",
                "Quanto costa questo?",
                "Sono stanco.",
                "Da dove vieni?",
                "Andiamo a fare una passeggiata nel parco."
            ],
            "pt": [
                "Olá, como você está hoje?",
                "Meu nome é João Silva.",
                "O tempo está muito bom hoje.",
                "Quero falar em português.",
                "Por favor fale devagar e claramente.",
                "Esta é uma frase de teste.",
                "Eu gosto de ler livros.",
                "Você pode me ajudar?",
                "Qual é o seu número de telefone?",
                "Vou vir amanhã de manhã.",
                "Obrigado pelo seu tempo.",
                "Este lugar é muito bonito.",
                "Estou aprendendo a cozinhar.",
                "Este é o caminho certo?",
                "Quero ir para a escola.",
                "Quantos anos você tem?",
                "Quanto custa isso?",
                "Estou cansado.",
                "De onde você é?",
                "Vamos dar um passeio no parque."
            ],
            "nl": [
                "Hallo, hoe gaat het vandaag met je?",
                "Mijn naam is Jan de Vries.",
                "Het weer is vandaag erg mooi.",
                "Ik wil in het Nederlands spreken.",
                "Spreek langzaam en duidelijk alstublieft.",
                "Dit is een testzin.",
                "Ik houd van het lezen van boeken.",
                "Kun je me helpen?",
                "Wat is je telefoonnummer?",
                "Ik kom morgenochtend.",
                "Bedankt voor je tijd.",
                "Deze plek is erg mooi.",
                "Ik leer koken.",
                "Is dit de juiste weg?",
                "Ik wil naar school gaan.",
                "Hoe oud ben je?",
                "Hoeveel kost dit?",
                "Ik ben moe.",
                "Waar kom je vandaan?",
                "Laten we een wandeling maken in het park."
            ]
        }

        fallback_phrases = language_fallbacks.get(language, language_fallbacks["en"])

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

    # Check for HF_TOKEN and show warning if missing
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_WRITE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        gr.HTML(
            """
            <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                <p style="color: rgb(234, 88, 12); margin: 0; font-size: 14px; font-weight: 600;">
                    ⚠️ No HF_TOKEN detected
                </p>
                <p style="color: rgb(234, 88, 12); margin: 6px 0 0; font-size: 12px;">
                    Set HF_TOKEN environment variable to access CohereLabs AYA Collection dataset with authentic multilingual phrases.
                    This dataset provides high-quality text in 100+ languages for all Voxtral Mini supported languages.
                    Currently using fallback phrases for demonstration.
                </p>
            </div>
            """
        )

    # Hidden state to track dataset JSONL path
    jsonl_path_state = gr.State("")

    # Language selection for Voxtral Mini supported languages
    language_selector = gr.Dropdown(
        choices=[
            ("English", "en"),
            ("French", "fr"),
            ("German", "de"),
            ("Spanish", "es"),
            ("Italian", "it"),
            ("Portuguese", "pt"),
            ("Dutch", "nl"),
            ("Hindi", "hi")
        ],
        value="en",
        label="Language for Speech Phrases",
        info="Select language for authentic phrases (Voxtral Mini supported languages). All languages use CohereLabs AYA Collection dataset when HF_TOKEN is available."
    )

    # Recording grid with dynamic text readouts
    phrase_texts_state = gr.State(ALL_PHRASES)
    visible_rows_state = gr.State(10)  # Start with 10 visible rows
    MAX_COMPONENTS = 100  # Fixed maximum number of components

    # Create fixed number of components upfront
    phrase_markdowns: list[gr.Markdown] = []
    rec_components = []

    def create_recording_grid(max_components=MAX_COMPONENTS):
        """Create recording grid components with fixed maximum"""
        markdowns = []
        recordings = []
        for idx in range(max_components):
            visible = False  # Initially hidden - will be revealed when language is selected
            phrase_text = ALL_PHRASES[idx] if idx < len(ALL_PHRASES) else ""
            md = gr.Markdown(f"**{idx+1}. {phrase_text}**", visible=visible)
            markdowns.append(md)
            comp = gr.Audio(sources="microphone", type="numpy", label=f"Recording {idx+1}", visible=visible)
            recordings.append(comp)
        return markdowns, recordings

    # Initial grid creation
    with gr.Column():
        phrase_markdowns, rec_components = create_recording_grid(MAX_COMPONENTS)

    # Add more rows button
    add_rows_btn = gr.Button("➕ Add 10 More Rows", variant="secondary", visible=False)

    def add_more_rows(current_visible, current_phrases):
        """Add 10 more rows by making them visible"""
        new_visible = min(current_visible + 10, MAX_COMPONENTS, len(current_phrases))

        # Create updates for all MAX_COMPONENTS (both markdown and audio components)
        markdown_updates = []
        audio_updates = []

        for i in range(MAX_COMPONENTS):
            if i < len(current_phrases) and i < new_visible:
                markdown_updates.append(gr.update(visible=True))
                audio_updates.append(gr.update(visible=True))
            else:
                markdown_updates.append(gr.update(visible=False))
                audio_updates.append(gr.update(visible=False))

        # Return: [state] + markdown_updates + audio_updates
        return [new_visible] + markdown_updates + audio_updates

    def change_language(language):
        """Change the language and reload phrases from multilingual datasets, reveal interface"""
        new_phrases = load_multilingual_phrases(language, max_phrases=None)
        # Reset visible rows to 10
        visible_count = min(10, len(new_phrases), MAX_COMPONENTS)

        # Create separate updates for markdown and audio components
        markdown_updates = []
        audio_updates = []

        for i in range(MAX_COMPONENTS):
            if i < len(new_phrases) and i < visible_count:
                markdown_updates.append(gr.update(value=f"**{i+1}. {new_phrases[i]}**", visible=True))
                audio_updates.append(gr.update(visible=True))
            elif i < len(new_phrases):
                markdown_updates.append(gr.update(value=f"**{i+1}. {new_phrases[i]}**", visible=False))
                audio_updates.append(gr.update(visible=False))
            else:
                markdown_updates.append(gr.update(value=f"**{i+1}. **", visible=False))
                audio_updates.append(gr.update(visible=False))

        # Reveal all interface elements when language is selected
        reveal_updates = [
            gr.update(visible=True),  # add_rows_btn
            gr.update(visible=True),  # record_dataset_btn
            gr.update(visible=True),  # dataset_status
            gr.update(visible=True),  # advanced_accordion
            gr.update(visible=True),  # save_rec_btn
            gr.update(visible=True),  # push_recordings_btn
            gr.update(visible=True),  # start_btn
            gr.update(visible=True),  # logs_box
        ]

        # Return: [phrases_state, visible_state] + markdown_updates + audio_updates + reveal_updates
        return [new_phrases, visible_count] + markdown_updates + audio_updates + reveal_updates

    add_rows_btn.click(
        add_more_rows,
        inputs=[visible_rows_state, phrase_texts_state],
        outputs=[visible_rows_state] + phrase_markdowns + rec_components
    )

    # Recording dataset creation button
    record_dataset_btn = gr.Button("🎙️ Create Dataset from Recordings", variant="primary", visible=False)

    def create_recording_dataset(*recordings_and_state):
        """Create dataset from visible recordings and phrases"""
        try:
            import soundfile as sf

            # Extract recordings and state
            recordings = recordings_and_state[:-1]  # All except the last item (phrases)
            phrases = recordings_and_state[-1]      # Last item is phrases

            dataset_dir = PROJECT_ROOT / "datasets" / "voxtral_user"
            wav_dir = dataset_dir / "wavs"
            wav_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            successful_recordings = 0

            # Process each recording
            for i, rec in enumerate(recordings):
                if rec is not None and i < len(phrases):
                    try:
                        sr, data = rec
                        out_path = wav_dir / f"recording_{i:04d}.wav"
                        sf.write(str(out_path), data, sr)
                        rows.append({"audio_path": str(out_path), "text": phrases[i]})
                        successful_recordings += 1
                    except Exception as e:
                        print(f"Error processing recording {i}: {e}")

            if rows:
                jsonl_path = dataset_dir / "recorded_data.jsonl"
                _write_jsonl(rows, jsonl_path)
                return f"✅ Dataset created successfully! {successful_recordings} recordings saved to {jsonl_path}"
            else:
                return "❌ No recordings found. Please record some audio first."

        except Exception as e:
            return f"❌ Error creating dataset: {str(e)}"

    # Status display for dataset creation
    dataset_status = gr.Textbox(label="Dataset Creation Status", interactive=False, visible=False)

    record_dataset_btn.click(
        create_recording_dataset,
        inputs=rec_components + [phrase_texts_state],
        outputs=[dataset_status]
    )

    # Advanced options accordion
    with gr.Accordion("Advanced options", open=False, visible=False) as advanced_accordion:
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
        dataset_repo_name = gr.Textbox(value=f"voxtral-dataset-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                       label="Dataset repo name (will be pushed to HF Hub)")
        save_upload_btn = gr.Button("Save uploaded dataset")
        push_dataset_btn = gr.Button("Push dataset to HF Hub")

        def _collect_upload(files, txt):
            lines = [s.strip() for s in (txt or "").splitlines() if s.strip()]
            jsonl_path = _save_uploaded_dataset(files or [], lines)
            return str(jsonl_path), f"✅ Dataset saved locally: {jsonl_path}"

        def _push_dataset_handler(repo_name, current_jsonl_path):
            if not current_jsonl_path:
                return "❌ No dataset saved yet. Please save dataset first."
            return _push_dataset_to_hub(current_jsonl_path, repo_name)

        save_upload_btn.click(_collect_upload, [upload_audio, transcripts_box], [jsonl_path_state, dataset_status])
        push_dataset_btn.click(_push_dataset_handler, [dataset_repo_name, jsonl_path_state], [dataset_status])

    # Save recordings button
    save_rec_btn = gr.Button("Save recordings as dataset", visible=False)
    push_recordings_btn = gr.Button("Push recordings dataset to HF Hub", visible=False)

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
        return str(jsonl_path), f"✅ Dataset saved locally: {jsonl_path}"

    save_rec_btn.click(_collect_preloaded_recs, rec_components + [phrase_texts_state], [jsonl_path_state, dataset_status])

    def _push_recordings_handler(repo_name, current_jsonl_path):
        if not current_jsonl_path:
            return "❌ No recordings dataset saved yet. Please save recordings first."
        return _push_dataset_to_hub(current_jsonl_path, repo_name)

    push_recordings_btn.click(_push_recordings_handler, [dataset_repo_name, jsonl_path_state], [dataset_status])

    # Removed multilingual dataset sample section - phrases are now loaded automatically when language is selected

    start_btn = gr.Button("Start Fine-tuning", visible=False)
    logs_box = gr.Textbox(label="Logs", lines=20, visible=False)

    start_btn.click(
        start_voxtral_training,
        inputs=[
            use_lora, base_model, repo_short, jsonl_path_state, train_count, eval_count,
            batch_size, grad_accum, learning_rate, epochs,
            lora_r, lora_alpha, lora_dropout, freeze_audio_tower,
            push_to_hub, deploy_demo,
        ],
        outputs=[logs_box],
    )

    # Connect language change to phrase reloading and interface reveal (placed after all components are defined)
    language_selector.change(
        change_language,
        inputs=[language_selector],
        outputs=[phrase_texts_state, visible_rows_state] + phrase_markdowns + rec_components + [
            add_rows_btn, record_dataset_btn, dataset_status, advanced_accordion,
            save_rec_btn, push_recordings_btn, start_btn, logs_box
        ]
    )


if __name__ == "__main__":
    server_port = int(os.environ.get("INTERFACE_PORT", "7860"))
    server_name = os.environ.get("INTERFACE_HOST", "0.0.0.0")
    demo.queue().launch(server_name=server_name, server_port=server_port, mcp_server=True, ssr_mode=False)


