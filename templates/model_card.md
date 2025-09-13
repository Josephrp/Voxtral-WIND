---
license: apache-2.0
tags:
- voxtral
- asr
- speech-to-text
- fine-tuning
pipeline_tag: automatic-speech-recognition
base_model: {{base_model}}
{{#if has_hub_dataset_id}}
datasets:
- {{dataset_name}}
{{/if}}
{{#if author_name}}
author: {{author_name}}
{{/if}}
{{#if training_config_type}}
training_config: {{training_config_type}}
{{/if}}
{{#if trainer_type}}
trainer_type: {{trainer_type}}
{{/if}}
{{#if batch_size}}
batch_size: {{batch_size}}
{{/if}}
{{#if gradient_accumulation_steps}}
gradient_accumulation_steps: {{gradient_accumulation_steps}}
{{/if}}
{{#if learning_rate}}
learning_rate: {{learning_rate}}
{{/if}}
{{#if max_epochs}}
max_epochs: {{max_epochs}}
{{/if}}
{{#if max_seq_length}}
max_seq_length: {{max_seq_length}}
{{/if}}
{{#if hardware_info}}
hardware: "{{hardware_info}}"
{{/if}}
---

# {{model_name}}

{{model_description}}

## Usage

```python
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import soundfile as sf

processor = AutoProcessor.from_pretrained("{{repo_name}}")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "{{repo_name}}",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

audio, sr = sf.read("sample.wav")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

## Training Configuration

- Base model: {{base_model}}
{{#if training_config_type}}- Config: {{training_config_type}}{{/if}}
{{#if trainer_type}}- Trainer: {{trainer_type}}{{/if}}

## Training Parameters

- Batch size: {{batch_size}}
- Grad accumulation: {{gradient_accumulation_steps}}
- Learning rate: {{learning_rate}}
- Max epochs: {{max_epochs}}
- Sequence length: {{max_seq_length}}

## Hardware

- {{hardware_info}}

## Notes

- This repository contains a fine-tuned Voxtral ASR model.
