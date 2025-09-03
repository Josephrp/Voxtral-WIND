# Voxtral ASR Fine-tuning Architecture

```mermaid
graph TB
    %% User Interface Layer
    subgraph "User Interface"
        UI[Gradio Web Interface<br/>interface.py]
        REC[Audio Recording<br/>Microphone Input]
        UP[File Upload<br/>WAV/FLAC files]
    end

    %% Data Processing Layer
    subgraph "Data Processing"
        DP[Data Processing<br/>Audio resampling<br/>JSONL creation]
        DS[Dataset Management<br/>NVIDIA Granary<br/>Local datasets]
    end

    %% Training Layer
    subgraph "Training Pipeline"
        TF[Full Fine-tuning<br/>scripts/train.py]
        TL[LoRA Fine-tuning<br/>scripts/train_lora.py]
        TI[Trackio Integration<br/>Experiment Tracking]
    end

    %% Model Management Layer
    subgraph "Model Management"
        MM[Model Management<br/>Hugging Face Hub<br/>Local storage]
        MC[Model Card Generation<br/>scripts/generate_model_card.py]
    end

    %% Deployment Layer
    subgraph "Deployment & Demo"
        DEP[Demo Space Deployment<br/>scripts/deploy_demo_space.py]
        HF[HF Spaces<br/>Interactive Demo]
    end

    %% External Services
    subgraph "External Services"
        HFH[Hugging Face Hub<br/>Models & Datasets]
        GRAN[NVIDIA Granary<br/>Multilingual ASR Dataset]
        TRACK[Trackio Spaces<br/>Experiment Tracking]
    end

    %% Data Flow
    UI --> DP
    REC --> DP
    UP --> DP
    DP --> DS

    DS --> TF
    DS --> TL
    TF --> TI
    TL --> TI

    TF --> MM
    TL --> MM
    MM --> MC

    MM --> DEP
    DEP --> HF

    DS -.-> HFH
    MM -.-> HFH
    TI -.-> TRACK
    DS -.-> GRAN

    %% Styling
    classDef interface fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef training fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef management fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef deployment fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef external fill:#f5f5f5,stroke:#424242,stroke-width:2px

    class UI,REC,UP interface
    class DP,DS processing
    class TF,TL,TI training
    class MM,MC management
    class DEP,HF deployment
    class HFH,GRAN,TRACK external
```

## Architecture Overview

This diagram shows the high-level architecture of the Voxtral ASR Fine-tuning application. The system is organized into several layers:

### 1. User Interface Layer
- **Gradio Web Interface**: Main user-facing application built with Gradio
- **Audio Recording**: Microphone input for recording speech samples
- **File Upload**: Support for uploading existing WAV/FLAC audio files

### 2. Data Processing Layer
- **Data Processing**: Audio resampling to 16kHz, JSONL dataset creation
- **Dataset Management**: Integration with NVIDIA Granary dataset and local dataset handling

### 3. Training Layer
- **Full Fine-tuning**: Complete model fine-tuning using `scripts/train.py`
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning using `scripts/train_lora.py`
- **Trackio Integration**: Experiment tracking and logging

### 4. Model Management Layer
- **Model Management**: Local storage and Hugging Face Hub integration
- **Model Card Generation**: Automated model card creation

### 5. Deployment Layer
- **Demo Space Deployment**: Automated deployment to Hugging Face Spaces
- **Interactive Demo**: Live demo interface for testing fine-tuned models

### 6. External Services
- **Hugging Face Hub**: Model and dataset storage and sharing
- **NVIDIA Granary**: High-quality multilingual ASR dataset
- **Trackio Spaces**: Experiment tracking and visualization

## Key Workflows

1. **Dataset Creation**: Users can record audio or upload files → processed into JSONL format
2. **Model Training**: Datasets fed into training scripts with experiment tracking
3. **Model Publishing**: Trained models pushed to HF Hub with generated model cards
4. **Demo Deployment**: Automated deployment of interactive demos to HF Spaces

See also:
- [Interface Workflow](interface-workflow.md)
- [Training Pipeline](training-pipeline.md)
- [Deployment Pipeline](deployment-pipeline.md)
- [Data Flow](data-flow.md)

