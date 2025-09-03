# Training Pipeline

```mermaid
graph TB
    %% Input Data Sources
    subgraph "Data Sources"
        JSONL[JSONL Dataset<br/>{"audio_path": "...", "text": "..."}]
        GRANARY[NVIDIA Granary Dataset<br/>Multilingual ASR Data]
        HFDATA[HF Hub Datasets<br/>Community Datasets]
    end

    %% Data Processing
    subgraph "Data Processing"
        LOADER[Dataset Loader<br/>_load_jsonl_dataset()]
        CASTER[Audio Casting<br/>16kHz resampling]
        COLLATOR[VoxtralDataCollator<br/>Audio + Text Processing]
    end

    %% Training Scripts
    subgraph "Training Scripts"
        TRAIN_FULL[Full Fine-tuning<br/>scripts/train.py]
        TRAIN_LORA[LoRA Fine-tuning<br/>scripts/train_lora.py]

        subgraph "Training Components"
            MODEL_INIT[Model Initialization<br/>VoxtralForConditionalGeneration]
            LORA_CONFIG[LoRA Configuration<br/>LoraConfig + get_peft_model]
            PROCESSOR_INIT[Processor Initialization<br/>VoxtralProcessor]
        end
    end

    %% Training Infrastructure
    subgraph "Training Infrastructure"
        TRACKIO_INIT[Trackio Integration<br/>Experiment Tracking]
        HF_TRAINER[Hugging Face Trainer<br/>TrainingArguments + Trainer]
        TORCH_DEVICE[Torch Device Setup<br/>GPU/CPU Detection]
    end

    %% Training Process
    subgraph "Training Process"
        FORWARD_PASS[Forward Pass<br/>Audio Processing + Generation]
        LOSS_CALC[Loss Calculation<br/>Masked Language Modeling]
        BACKWARD_PASS[Backward Pass<br/>Gradient Computation]
        OPTIMIZER_STEP[Optimizer Step<br/>Parameter Updates]
        LOGGING[Metrics Logging<br/>Loss, Perplexity, etc.]
    end

    %% Model Management
    subgraph "Model Management"
        CHECKPOINT_SAVING[Checkpoint Saving<br/>Model snapshots]
        MODEL_SAVING[Final Model Saving<br/>Processor + Model]
        LOCAL_STORAGE[Local Storage<br/>outputs/ directory]
    end

    %% Flow Connections
    JSONL --> LOADER
    GRANARY --> LOADER
    HFDATA --> LOADER

    LOADER --> CASTER
    CASTER --> COLLATOR

    COLLATOR --> TRAIN_FULL
    COLLATOR --> TRAIN_LORA

    TRAIN_FULL --> MODEL_INIT
    TRAIN_LORA --> MODEL_INIT
    TRAIN_LORA --> LORA_CONFIG

    MODEL_INIT --> PROCESSOR_INIT
    LORA_CONFIG --> PROCESSOR_INIT

    PROCESSOR_INIT --> TRACKIO_INIT
    PROCESSOR_INIT --> HF_TRAINER
    PROCESSOR_INIT --> TORCH_DEVICE

    TRACKIO_INIT --> HF_TRAINER
    TORCH_DEVICE --> HF_TRAINER

    HF_TRAINER --> FORWARD_PASS
    FORWARD_PASS --> LOSS_CALC
    LOSS_CALC --> BACKWARD_PASS
    BACKWARD_PASS --> OPTIMIZER_STEP
    OPTIMIZER_STEP --> LOGGING

    LOGGING --> CHECKPOINT_SAVING
    LOGGING --> TRACKIO_INIT

    HF_TRAINER --> MODEL_SAVING
    MODEL_SAVING --> LOCAL_STORAGE

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef training fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef infrastructure fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef execution fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef output fill:#f5f5f5,stroke:#424242,stroke-width:2px

    class JSONL,GRANARY,HFDATA input
    class LOADER,CASTER,COLLATOR processing
    class TRAIN_FULL,TRAIN_LORA,MODEL_INIT,LORA_CONFIG,PROCESSOR_INIT training
    class TRACKIO_INIT,HF_TRAINER,TORCH_DEVICE infrastructure
    class FORWARD_PASS,LOSS_CALC,BACKWARD_PASS,OPTIMIZER_STEP,LOGGING execution
    class CHECKPOINT_SAVING,MODEL_SAVING,LOCAL_STORAGE output
```

## Training Pipeline Overview

This diagram illustrates the complete training pipeline for Voxtral ASR fine-tuning, showing how data flows through the training scripts and supporting infrastructure.

### Data Input Sources

#### JSONL Datasets
- **Local Datasets**: User-created datasets from recordings or uploads
- **Format**: `{"audio_path": "path/to/audio.wav", "text": "transcription"}`
- **Processing**: Loaded via `_load_jsonl_dataset()` function

#### NVIDIA Granary Dataset
- **Multilingual Support**: 25+ European languages
- **High Quality**: Curated ASR training data
- **Streaming**: Efficient loading without full download

#### Hugging Face Hub Datasets
- **Community Datasets**: Public datasets from HF Hub
- **Standard Formats**: Compatible with Voxtral training requirements

### Data Processing Pipeline

#### Dataset Loading
```python
# Load local JSONL or HF dataset
ds = _load_jsonl_dataset(jsonl_path)
# or
ds = load_dataset(ds_name, ds_cfg, split="test")
```

#### Audio Processing
```python
# Cast to Audio format with 16kHz resampling
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
```

#### Data Collation
- **VoxtralDataCollator**: Custom collator for Voxtral training
- **Audio Processing**: Converts audio to model inputs
- **Text Tokenization**: Processes transcription text
- **Masking**: Masks prompt tokens during training

### Training Script Architecture

#### Full Fine-tuning (`train.py`)
- **Complete Model Updates**: All parameters trainable
- **Higher Memory Requirements**: Full model in memory
- **Better Convergence**: Can achieve higher accuracy

#### LoRA Fine-tuning (`train_lora.py`)
- **Parameter Efficient**: Only LoRA adapters trained
- **Lower Memory Usage**: Base model frozen
- **Faster Training**: Fewer parameters to update
- **Configurable**: r, alpha, dropout parameters

### Training Infrastructure

#### Trackio Integration
```python
trackio.init(
    project="voxtral-finetuning",
    config={...},  # Training parameters
    space_id=trackio_space
)
```

#### Hugging Face Trainer
```python
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    bf16=True,  # BFloat16 for efficiency
    report_to=["trackio"],
    # ... other args
)
```

#### Device Management
- **GPU Detection**: Automatic CUDA/GPU detection
- **Fallback**: CPU training if no GPU available
- **Memory Optimization**: Model sharding and gradient checkpointing

### Training Process Flow

#### Forward Pass
1. **Audio Input**: Raw audio waveforms
2. **Audio Tower**: Audio feature extraction
3. **Text Generation**: Autoregressive text generation from audio features

#### Loss Calculation
- **Masked Language Modeling**: Only transcription tokens contribute to loss
- **Audio Prompt Masking**: Audio processing tokens are masked out
- **Cross-Entropy Loss**: Standard language modeling loss

#### Backward Pass & Optimization
- **Gradient Computation**: Backpropagation through the model
- **LoRA Updates**: Only adapter parameters updated (LoRA mode)
- **Full Updates**: All parameters updated (full fine-tuning)

### Model Management

#### Checkpoint Saving
- **Regular Checkpoints**: Saved every N steps
- **Best Model Tracking**: Save best model based on validation loss
- **Resume Capability**: Continue training from checkpoints

#### Final Model Saving
```python
trainer.save_model()  # Saves model and tokenizer
processor.save_pretrained(output_dir)  # Saves processor
```

#### Local Storage Structure
```
outputs/
├── voxtral-finetuned-{timestamp}/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── training_config.json
│   ├── train_results.json
│   └── eval_results.json
```

### Integration Points

#### With Interface (`interface.py`)
- **Parameter Passing**: Training parameters from UI
- **Log Streaming**: Real-time training logs to UI
- **Progress Monitoring**: Training progress updates

#### With Model Publishing (`push_to_huggingface.py`)
- **Model Upload**: Trained model to HF Hub
- **Metadata**: Training config and results
- **Model Cards**: Automatic model card generation

#### With Demo Deployment (`deploy_demo_space.py`)
- **Space Creation**: HF Spaces for demos
- **Model Integration**: Deploy trained model in demo
- **Configuration**: Demo-specific settings

### Performance Considerations

#### Memory Optimization
- **LoRA**: Significantly reduces memory requirements
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: BF16/FP16 training

#### Training Efficiency
- **Batch Size**: Balanced with gradient accumulation
- **Learning Rate**: Warmup and decay schedules
- **Early Stopping**: Prevent overfitting

#### Monitoring & Debugging
- **Metrics Tracking**: Loss, perplexity, learning rate
- **GPU Utilization**: Memory and compute monitoring
- **Error Handling**: Graceful failure recovery

See also:
- [Architecture Overview](architecture.md)
- [Interface Workflow](interface-workflow.md)
- [Data Flow](data-flow.md)

