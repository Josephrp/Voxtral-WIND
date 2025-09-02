import os
import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Voxtral-Mini-3B-2507")
MODEL_NAME = os.getenv("MODEL_NAME", HF_MODEL_ID.split("/")[-1])
HF_USERNAME = os.getenv("HF_USERNAME", "")

processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)

def transcribe(audio_tuple):
    if audio_tuple is None:
        return "No audio provided"
    sr, data = audio_tuple
    inputs = processor.apply_transcription_request(language="en", model_id=HF_MODEL_ID, audio=[data], format=["WAV"], return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    # Voxtral returns full sequence; decode and strip special tokens
    text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

with gr.Blocks() as demo:
    gr.Markdown(f"# 🎙️ Voxtral ASR Demo — {MODEL_NAME}")
    audio = gr.Audio(sources="microphone", type="numpy", label="Record or upload audio")
    btn = gr.Button("Transcribe")
    out = gr.Textbox(label="Transcription", lines=4)
    btn.click(transcribe, inputs=[audio], outputs=[out])

if __name__ == "__main__":
    demo.launch(mcp_server=True, ssr_mode=False)


