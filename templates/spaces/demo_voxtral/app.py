import os
import gradio as gr
import torch
from transformers import AutoProcessor
try:
    from transformers import VoxtralForConditionalGeneration as VoxtralModelClass
except Exception:
    # Fallback for older transformers versions
    from transformers import AutoModelForSeq2SeqLM as VoxtralModelClass

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Voxtral-Mini-3B-2507")
MODEL_NAME = os.getenv("MODEL_NAME", HF_MODEL_ID.split("/")[-1])
HF_USERNAME = os.getenv("HF_USERNAME", "")
MODEL_SUBFOLDER = os.getenv("MODEL_SUBFOLDER", "").strip()

try:
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
except Exception:
    # Fallback: some repos may store processor files inside the subfolder
    if MODEL_SUBFOLDER:
        processor = AutoProcessor.from_pretrained(HF_MODEL_ID, subfolder=MODEL_SUBFOLDER)
    else:
        raise

device = "cuda" if torch.cuda.is_available() else "cpu"
# Use float32 on CPU; bfloat16 on CUDA if available
if torch.cuda.is_available():
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
else:
    model_kwargs = {"torch_dtype": torch.float32}

if MODEL_SUBFOLDER:
    model = VoxtralModelClass.from_pretrained(
        HF_MODEL_ID, subfolder=MODEL_SUBFOLDER, **model_kwargs
    )
else:
    model = VoxtralModelClass.from_pretrained(
        HF_MODEL_ID, **model_kwargs
    )

# Simple language options (with Auto detection)
LANGUAGES = {
    "Auto": "auto",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Hindi": "hi",
}

MAX_NEW_TOKENS = 1024

def transcribe(sel_language, audio_path):
    if audio_path is None:
        return "No audio provided"
    language_code = LANGUAGES.get(sel_language, "auto")
    # Build Voxtral transcription inputs from filepath and selected language
    if hasattr(processor, "apply_transcrition_request"):
        inputs = processor.apply_transcrition_request(
            language=language_code,
            audio=audio_path,
            model_id=HF_MODEL_ID,
        )
    else:
        # Compatibility with potential corrected naming
        inputs = processor.apply_transcription_request(
            language=language_code,
            audio=audio_path,
            model_id=HF_MODEL_ID,
        )
    # Move to device with appropriate dtype
    inputs = inputs.to(device, dtype=(torch.bfloat16 if device == "cuda" else torch.float32))
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # Decode only newly generated tokens (beyond the prompt length)
    decoded = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return decoded[0]

with gr.Blocks() as demo:
    gr.Markdown(f"# 🎙️ Voxtral ASR Demo — {MODEL_NAME}")
    with gr.Row():
        language = gr.Dropdown(
            choices=list(LANGUAGES.keys()), value="Auto", label="Language"
        )
    audio = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Upload or record audio",
    )
    btn = gr.Button("Transcribe")
    out = gr.Textbox(label="Transcription", lines=8)
    btn.click(transcribe, inputs=[language, audio], outputs=[out])

if __name__ == "__main__":
    demo.launch(mcp_server=True, ssr_mode=False)


