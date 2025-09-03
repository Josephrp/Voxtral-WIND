# Data Flow

```mermaid
flowchart TD
    %% User Input Sources
    subgraph "User Input"
        MIC[🎤 Microphone Recording<br/>Raw audio + timestamps]
        FILE[📁 File Upload<br/>WAV/FLAC files]
        TEXT[📝 Manual Transcripts<br/>Text input]
        LANG[🌍 Language Selection<br/>25+ languages]
    end

    %% Data Processing Pipeline
    subgraph "Data Processing"
        AUDIO_PROC[Audio Processing<br/>Resampling to 16kHz<br/>Format conversion]
        TEXT_PROC[Text Processing<br/>Transcript validation<br/>Cleaning & formatting]
        JSONL_CONV[JSONL Conversion<br/>{"audio_path": "...", "text": "..."}]
    end

    %% Dataset Storage
    subgraph "Dataset Storage"
        LOCAL_DS[Local Dataset<br/>datasets/voxtral_user/<br/>data.jsonl + wavs/]
        HF_DS[HF Hub Dataset<br/>username/dataset-name<br/>Public sharing]
    end

    %% Training Data Flow
    subgraph "Training Data Pipeline"
        DS_LOADER[Dataset Loader<br/>_load_jsonl_dataset()<br/>or load_dataset()]
        AUDIO_CAST[Audio Casting<br/>Audio(sampling_rate=16000)]
        TRAIN_SPLIT[Train Split<br/>train_dataset]
        EVAL_SPLIT[Eval Split<br/>eval_dataset]
    end

    %% Model Training
    subgraph "Model Training"
        COLLATOR[VoxtralDataCollator<br/>Audio + Text batching<br/>Prompt construction]
        FORWARD[Forward Pass<br/>Audio → Features → Text]
        LOSS[Loss Calculation<br/>Masked LM loss]
        BACKWARD[Backward Pass<br/>Gradient computation]
        OPTIMIZE[Parameter Updates<br/>LoRA or full fine-tuning]
    end

    %% Training Outputs
    subgraph "Training Outputs"
        MODEL_FILES[Model Files<br/>model.safetensors<br/>config.json<br/>tokenizer.json]
        TRAINING_LOGS[Training Logs<br/>train_results.json<br/>training_config.json<br/>loss curves]
        CHECKPOINTS[Checkpoints<br/>Intermediate models<br/>best model tracking]
    end

    %% Publishing Pipeline
    subgraph "Publishing Pipeline"
        HF_REPO[HF Repository<br/>username/model-name<br/>Model hosting]
        MODEL_CARD[Model Card<br/>README.md<br/>Training details<br/>Usage examples]
        METADATA[Training Metadata<br/>Config + results<br/>Performance metrics]
    end

    %% Demo Deployment
    subgraph "Demo Deployment"
        SPACE_REPO[HF Space Repository<br/>username/model-name-demo<br/>Demo hosting]
        DEMO_APP[Demo Application<br/>Gradio interface<br/>Real-time inference]
        ENV_VARS[Environment Config<br/>HF_MODEL_ID<br/>MODEL_NAME<br/>secrets]
    end

    %% External Data Sources
    subgraph "External Data Sources"
        GRANARY[NVIDIA Granary<br/>Multilingual ASR data<br/>25+ languages]
        HF_COMM[HF Community Datasets<br/>Public ASR datasets<br/>Standard formats]
    end

    %% Data Flow Connections
    MIC --> AUDIO_PROC
    FILE --> AUDIO_PROC
    TEXT --> TEXT_PROC
    LANG --> TEXT_PROC

    AUDIO_PROC --> JSONL_CONV
    TEXT_PROC --> JSONL_CONV

    JSONL_CONV --> LOCAL_DS
    LOCAL_DS --> HF_DS

    LOCAL_DS --> DS_LOADER
    HF_DS --> DS_LOADER
    GRANARY --> DS_LOADER
    HF_COMM --> DS_LOADER

    DS_LOADER --> AUDIO_CAST
    AUDIO_CAST --> TRAIN_SPLIT
    AUDIO_CAST --> EVAL_SPLIT

    TRAIN_SPLIT --> COLLATOR
    EVAL_SPLIT --> COLLATOR

    COLLATOR --> FORWARD
    FORWARD --> LOSS
    LOSS --> BACKWARD
    BACKWARD --> OPTIMIZE

    OPTIMIZE --> MODEL_FILES
    OPTIMIZE --> TRAINING_LOGS
    OPTIMIZE --> CHECKPOINTS

    MODEL_FILES --> HF_REPO
    TRAINING_LOGS --> HF_REPO
    CHECKPOINTS --> HF_REPO

    HF_REPO --> MODEL_CARD
    TRAINING_LOGS --> MODEL_CARD

    MODEL_CARD --> SPACE_REPO
    HF_REPO --> SPACE_REPO
    ENV_VARS --> SPACE_REPO

    SPACE_REPO --> DEMO_APP

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef training fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef publishing fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef deployment fill:#f5f5f5,stroke:#424242,stroke-width:2px
    classDef external fill:#efebe9,stroke:#5d4037,stroke-width:2px

    class MIC,FILE,TEXT,LANG input
    class AUDIO_PROC,TEXT_PROC,JSONL_CONV processing
    class LOCAL_DS,HF_DS storage
    class DS_LOADER,AUDIO_CAST,TRAIN_SPLIT,EVAL_SPLIT,COLLATOR,FORWARD,LOSS,BACKWARD,OPTIMIZE training
    class MODEL_FILES,TRAINING_LOGS,CHECKPOINTS output
    class HF_REPO,MODEL_CARD,METADATA publishing
    class SPACE_REPO,DEMO_APP,ENV_VARS deployment
    class GRANARY,HF_COMM external
```

## Data Flow Overview

This diagram illustrates the complete data flow through the Voxtral ASR Fine-tuning application, from user input to deployed demo.

### Data Input Sources

#### User-Generated Data
- **Microphone Recording**: Raw audio captured through browser microphone
- **File Upload**: Existing WAV/FLAC audio files
- **Manual Transcripts**: User-provided text transcriptions
- **Language Selection**: Influences phrase selection from NVIDIA Granary

#### External Data Sources
- **NVIDIA Granary**: High-quality multilingual ASR dataset
- **HF Community Datasets**: Public datasets from Hugging Face Hub

### Data Processing Pipeline

#### Audio Processing
```python
# Audio resampling and format conversion
audio_data = librosa.load(audio_path, sr=16000)
# Convert to WAV format for consistency
sf.write(output_path, audio_data, 16000)
```

#### Text Processing
```python
# Text cleaning and validation
text = text.strip()
# Basic validation (length, content checks)
assert len(text) > 0, "Empty transcription"
```

#### JSONL Conversion
```python
# Standard format for all datasets
entry = {
    "audio_path": str(audio_file_path),
    "text": cleaned_transcription
}
# Write to JSONL file
with open(jsonl_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
```

### Dataset Storage

#### Local Storage Structure
```
datasets/voxtral_user/
├── data.jsonl          # Main dataset file
├── recorded_data.jsonl # From recordings
└── wavs/              # Audio files
    ├── recording_0000.wav
    ├── recording_0001.wav
    └── ...
```

#### HF Hub Storage
- **Public Datasets**: Shareable with community
- **Version Control**: Dataset versioning and updates
- **Standard Metadata**: Automatic README generation

### Training Data Pipeline

#### Dataset Loading
```python
# Load local JSONL
ds = _load_jsonl_dataset("datasets/voxtral_user/data.jsonl")

# Load HF dataset
ds = load_dataset("username/dataset-name", split="train")
```

#### Audio Casting
```python
# Ensure consistent sampling rate
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
```

#### Train/Eval Split
```python
# Create train and eval datasets
train_dataset = ds.select(range(train_count))
eval_dataset = ds.select(range(train_count, train_count + eval_count))
```

### Training Process Flow

#### Data Collation
- **VoxtralDataCollator**: Custom collator for Voxtral model
- **Audio Processing**: Convert audio to model inputs
- **Prompt Construction**: Build `[AUDIO]...[AUDIO] <transcribe>` prompts
- **Text Tokenization**: Process transcription targets
- **Masking**: Mask audio prompt tokens during training

#### Forward Pass
1. **Audio Input**: Raw audio waveforms
2. **Audio Tower**: Extract audio features
3. **Language Model**: Generate transcription autoregressively
4. **Loss Calculation**: Compare generated vs target text

#### Backward Pass & Optimization
- **Gradient Computation**: Backpropagation
- **LoRA Updates**: Update only adapter parameters (LoRA mode)
- **Full Updates**: Update all parameters (full fine-tuning)
- **Optimizer Step**: Apply gradients with learning rate scheduling

### Training Outputs

#### Model Files
- **model.safetensors**: Model weights (safetensors format)
- **config.json**: Model configuration
- **tokenizer.json**: Tokenizer configuration
- **generation_config.json**: Generation parameters

#### Training Logs
- **train_results.json**: Final training metrics
- **eval_results.json**: Evaluation results
- **training_config.json**: Training hyperparameters
- **trainer_state.json**: Training state and checkpoints

#### Checkpoints
- **checkpoint-XXX/**: Intermediate model snapshots
- **best-model/**: Best performing model
- **final-model/**: Final trained model

### Publishing Pipeline

#### HF Repository Structure
```
username/model-name/
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── config.json
├── tokenizer.json
├── training_config.json
├── train_results.json
├── README.md (model card)
└── training_results/
    └── training.log
```

#### Model Card Generation
- **Template Processing**: Fill model_card.md template
- **Variable Injection**: Training config, results, metadata
- **Conditional Sections**: Handle quantized models, etc.

### Demo Deployment

#### Space Repository Structure
```
username/model-name-demo/
├── app.py              # Gradio demo application
├── requirements.txt    # Python dependencies
├── README.md          # Space documentation
└── .env               # Environment variables
```

#### Environment Configuration
```python
# Space environment variables
HF_MODEL_ID=username/model-name
MODEL_NAME=Voxtral Fine-tuned Model
HF_TOKEN=read_only_token  # For model access
BRAND_OWNER_NAME=username
# ... other branding variables
```

### Data Flow Patterns

#### Streaming vs Batch Processing
- **Training Data**: Batch processing for efficiency
- **External Datasets**: Streaming loading for memory efficiency
- **User Input**: Real-time processing with immediate feedback

#### Data Validation
- **Input Validation**: Check audio format, sampling rate, text length
- **Quality Assurance**: Filter out empty or invalid entries
- **Consistency Checks**: Ensure audio-text alignment

#### Error Handling
- **Graceful Degradation**: Fallback to local data if external sources fail
- **Retry Logic**: Automatic retry for network failures
- **Logging**: Comprehensive error logging and debugging

### Performance Considerations

#### Memory Management
- **Streaming Loading**: Process large datasets without loading everything
- **Audio Caching**: Cache processed audio features
- **Batch Optimization**: Balance batch size with available memory

#### Storage Optimization
- **Compression**: Use efficient audio formats
- **Deduplication**: Avoid duplicate data entries
- **Cleanup**: Remove temporary files after processing

#### Network Efficiency
- **Incremental Uploads**: Upload files as they're ready
- **Resume Capability**: Resume interrupted uploads
- **Caching**: Cache frequently accessed data

### Security & Privacy

#### Data Privacy
- **Local Processing**: Audio files processed locally when possible
- **User Consent**: Clear data usage policies
- **Anonymization**: Remove personally identifiable information

#### Access Control
- **Token Management**: Secure HF token storage
- **Repository Permissions**: Appropriate public/private settings
- **Rate Limiting**: Prevent abuse of demo interfaces

### Monitoring & Analytics

#### Data Quality Metrics
- **Audio Quality**: Sampling rate, format validation
- **Text Quality**: Length, language detection, consistency
- **Dataset Statistics**: Size, distribution, coverage

#### Performance Metrics
- **Processing Time**: Data loading, preprocessing, training time
- **Model Metrics**: Loss, perplexity, WER (if available)
- **Resource Usage**: Memory, CPU/GPU utilization

#### User Analytics
- **Usage Patterns**: Popular languages, dataset sizes
- **Success Rates**: Training completion, deployment success
- **Error Patterns**: Common failure modes and solutions

See also:
- [Architecture Overview](architecture.md)
- [Interface Workflow](interface-workflow.md)
- [Training Pipeline](training-pipeline.md)

