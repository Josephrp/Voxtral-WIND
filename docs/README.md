# Voxtral ASR Fine-tuning Documentation

```mermaid
graph TD
    %% Main Entry Point
    START([🎯 Voxtral ASR Fine-tuning App]) --> OVERVIEW{Choose Documentation}

    %% Documentation Categories
    OVERVIEW --> ARCH[🏗️ Architecture Overview]
    OVERVIEW --> WORKFLOW[🔄 Interface Workflow]
    OVERVIEW --> TRAINING[🚀 Training Pipeline]
    OVERVIEW --> DEPLOYMENT[🌐 Deployment Pipeline]
    OVERVIEW --> DATAFLOW[📊 Data Flow]

    %% Architecture Section
    ARCH --> ARCH_DIAG[High-level Architecture<br/>System Components & Layers]
    ARCH --> ARCH_LINK[📄 View Details →](architecture.md)

    %% Interface Section
    WORKFLOW --> WORKFLOW_DIAG[User Journey<br/>Recording → Training → Demo]
    WORKFLOW --> WORKFLOW_LINK[📄 View Details →](interface-workflow.md)

    %% Training Section
    TRAINING --> TRAINING_DIAG[Training Scripts<br/>Data → Model → Results]
    TRAINING --> TRAINING_LINK[📄 View Details →](training-pipeline.md)

    %% Deployment Section
    DEPLOYMENT --> DEPLOYMENT_DIAG[Publishing & Demo<br/>Model → Hub → Space]
    DEPLOYMENT --> DEPLOYMENT_LINK[📄 View Details →](deployment-pipeline.md)

    %% Data Flow Section
    DATAFLOW --> DATAFLOW_DIAG[Complete Data Journey<br/>Input → Processing → Output]
    DATAFLOW --> DATAFLOW_LINK[📄 View Details →](data-flow.md)

    %% Key Components Highlight
    subgraph "🎛️ Core Components"
        INTERFACE[interface.py<br/>Gradio Web UI]
        TRAIN_SCRIPTS[scripts/train*.py<br/>Training Scripts]
        DEPLOY_SCRIPT[scripts/deploy_demo_space.py<br/>Demo Deployment]
        PUSH_SCRIPT[scripts/push_to_huggingface.py<br/>Model Publishing]
    end

    %% Data Flow Highlight
    subgraph "📁 Key Data Formats"
        JSONL[JSONL Dataset<br/>{"audio_path": "...", "text": "..."}]
        HFDATA[HF Hub Models<br/>username/model-name]
        SPACES[HF Spaces<br/>Interactive Demos]
    end

    %% Connect components to their respective docs
    INTERFACE --> WORKFLOW
    TRAIN_SCRIPTS --> TRAINING
    DEPLOY_SCRIPT --> DEPLOYMENT
    PUSH_SCRIPT --> DEPLOYMENT

    JSONL --> DATAFLOW
    HFDATA --> DEPLOYMENT
    SPACES --> DEPLOYMENT

    %% Styling
    classDef entry fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef category fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef diagram fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef link fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef component fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef data fill:#e1f5fe,stroke:#0277bd,stroke-width:2px

    class START entry
    class OVERVIEW,ARCH,WORKFLOW,TRAINING,DEPLOYMENT,DATAFLOW category
    class ARCH_DIAG,WORKFLOW_DIAG,TRAINING_DIAG,DEPLOYMENT_DIAG,DATAFLOW_DIAG diagram
    class ARCH_LINK,WORKFLOW_LINK,TRAINING_LINK,DEPLOYMENT_LINK,DATAFLOW_LINK link
    class INTERFACE,TRAIN_SCRIPTS,DEPLOY_SCRIPT,PUSH_SCRIPT component
    class JSONL,HFDATA,SPACES data
```

## Voxtral ASR Fine-tuning Application

This documentation provides comprehensive diagrams and explanations of the Voxtral ASR Fine-tuning application architecture and workflows.

### 🎯 What is Voxtral ASR Fine-tuning?

Voxtral is a powerful Automatic Speech Recognition (ASR) model that can be fine-tuned for specific tasks and languages. This application provides:

- **🎙️ Easy Data Collection**: Record audio or upload files with transcripts
- **🚀 One-Click Training**: Fine-tune Voxtral with LoRA or full parameter updates
- **🌐 Instant Deployment**: Deploy interactive demos to Hugging Face Spaces
- **📊 Experiment Tracking**: Monitor training progress with Trackio integration

### 📚 Documentation Overview

#### 🏗️ [Architecture Overview](architecture.md)
High-level view of system components and their relationships:
- **User Interface Layer**: Gradio web interface
- **Data Processing Layer**: Audio processing and dataset creation
- **Training Layer**: Full and LoRA fine-tuning scripts
- **Model Management Layer**: HF Hub integration and model cards
- **Deployment Layer**: Demo space deployment

#### 🔄 [Interface Workflow](interface-workflow.md)
Complete user journey through the application:
- **Language Selection**: Choose from 25+ languages via NVIDIA Granary
- **Data Collection**: Record audio or upload existing files
- **Dataset Creation**: Process audio + transcripts into JSONL format
- **Training Configuration**: Set hyperparameters and options
- **Live Training**: Real-time progress monitoring
- **Auto Deployment**: One-click model publishing and demo creation

#### 🚀 [Training Pipeline](training-pipeline.md)
Detailed training process and script interactions:
- **Data Sources**: JSONL datasets, HF Hub datasets, NVIDIA Granary
- **Data Processing**: Audio resampling, text tokenization, data collation
- **Training Scripts**: `train.py` (full) vs `train_lora.py` (parameter-efficient)
- **Infrastructure**: Trackio logging, Hugging Face Trainer, device management
- **Model Outputs**: Trained models, training logs, checkpoints

#### 🌐 [Deployment Pipeline](deployment-pipeline.md)
Model publishing and demo deployment process:
- **Model Publishing**: Push to Hugging Face Hub with metadata
- **Model Card Generation**: Automated documentation creation
- **Demo Space Deployment**: Create interactive demos on HF Spaces
- **Configuration Management**: Environment variables and secrets
- **Live Demo Features**: Real-time ASR inference interface

#### 📊 [Data Flow](data-flow.md)
Complete data journey through the system:
- **Input Sources**: Microphone recordings, file uploads, external datasets
- **Processing Pipeline**: Audio resampling, text cleaning, JSONL conversion
- **Training Flow**: Dataset loading, batching, model training
- **Output Pipeline**: Model files, logs, checkpoints, published assets
- **External Integration**: HF Hub, NVIDIA Granary, Trackio Spaces

### 🛠️ Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `interface.py` | Main web application | Gradio UI, data collection, training orchestration |
| `scripts/train.py` | Full model fine-tuning | Complete parameter updates, maximum accuracy |
| `scripts/train_lora.py` | LoRA fine-tuning | Parameter-efficient, faster training, lower memory |
| `scripts/deploy_demo_space.py` | Demo deployment | Automated HF Spaces creation and configuration |
| `scripts/push_to_huggingface.py` | Model publishing | HF Hub integration, model card generation |
| `scripts/generate_model_card.py` | Documentation | Automated model card creation from templates |

### 📁 Key Data Formats

#### JSONL Dataset Format
```json
{"audio_path": "path/to/audio.wav", "text": "transcription text"}
```

#### Training Configuration
```json
{
  "model_checkpoint": "mistralai/Voxtral-Mini-3B-2507",
  "batch_size": 2,
  "learning_rate": 5e-5,
  "epochs": 3,
  "lora_r": 8,
  "lora_alpha": 32
}
```

#### Model Repository Structure
```
username/model-name/
├── model.safetensors
├── config.json
├── tokenizer.json
├── README.md (model card)
└── training_results/
```

### 🚀 Quick Start

1. **Set Environment Variables**:
   ```bash
   export HF_TOKEN=your_huggingface_token
   export HF_USERNAME=your_username
   ```

2. **Launch Interface**:
   ```bash
   python interface.py
   ```

3. **Follow the Workflow**:
   - Select language → Record/upload data → Configure training → Start training
   - Monitor progress → View results → Deploy demo

### 📋 Prerequisites

- **Hardware**: NVIDIA GPU recommended for training
- **Software**: Python 3.8+, CUDA-compatible GPU drivers
- **Tokens**: Hugging Face token for model access and publishing
- **Storage**: Sufficient disk space for models and datasets

### 🔧 Configuration Options

#### Training Modes
- **LoRA Fine-tuning**: Efficient, fast, lower memory usage
- **Full Fine-tuning**: Maximum accuracy, higher memory requirements

#### Data Sources
- **User Recordings**: Live microphone input
- **File Uploads**: Existing WAV/FLAC files
- **NVIDIA Granary**: High-quality multilingual datasets
- **HF Hub Datasets**: Community-contributed datasets

#### Deployment Options
- **HF Hub Publishing**: Share models publicly
- **Demo Spaces**: Interactive web demos
- **Model Cards**: Automated documentation

### 📈 Performance & Metrics

#### Training Metrics
- **Loss Curves**: Training and validation loss
- **Perplexity**: Model confidence measure
- **Word Error Rate**: ASR accuracy (if available)
- **Training Time**: Time to convergence

#### Resource Usage
- **GPU Memory**: Peak memory usage during training
- **Training Time**: Hours/days depending on dataset size
- **Model Size**: Disk space requirements

### 🤝 Contributing

The documentation is organized as interlinked Markdown files with Mermaid diagrams. Each diagram focuses on a specific aspect:

- **architecture.md**: System overview and component relationships
- **interface-workflow.md**: User experience and interaction flow
- **training-pipeline.md**: Technical training process details
- **deployment-pipeline.md**: Publishing and deployment mechanics
- **data-flow.md**: Data movement and transformation

### 📄 Additional Resources

- **Hugging Face Spaces**: [Live Demo](https://huggingface.co/spaces)
- **Voxtral Models**: [Model Hub](https://huggingface.co/mistralai)
- **NVIDIA Granary**: [Dataset Documentation](https://huggingface.co/nvidia/Granary)
- **Trackio**: [Experiment Tracking](https://trackio.space)

---

*This documentation was automatically generated to explain the Voxtral ASR Fine-tuning application architecture and workflows.*

