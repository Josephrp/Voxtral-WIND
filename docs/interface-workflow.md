# Interface Workflow

```mermaid
stateDiagram-v2
    [*] --> LanguageSelection: User opens interface

    state "Language & Dataset Setup" as LangSetup {
        [*] --> LanguageSelection
        LanguageSelection --> LoadPhrases: Select language
        LoadPhrases --> DisplayPhrases: Load from NVIDIA Granary
        DisplayPhrases --> RecordingInterface: Show phrases & recording UI

        state RecordingInterface {
            [*] --> ShowInitialRows: Display first 10 phrases
            ShowInitialRows --> RecordAudio: User can record audio
            RecordAudio --> AddMoreRows: Optional - add 10 more rows
            AddMoreRows --> RecordAudio
        }
    }

    RecordingInterface --> DatasetCreation: User finishes recording

    state "Dataset Creation Options" as DatasetCreation {
        [*] --> FromRecordings: Create from recorded audio
        [*] --> FromUploads: Upload existing files

        FromRecordings --> ProcessRecordings: Save WAV files + transcripts
        FromUploads --> ProcessUploads: Process uploaded files + transcripts

        ProcessRecordings --> CreateJSONL: Generate JSONL dataset
        ProcessUploads --> CreateJSONL

        CreateJSONL --> DatasetReady: Dataset saved locally
    }

    DatasetCreation --> TrainingConfiguration: Dataset ready

    state "Training Setup" as TrainingConfiguration {
        [*] --> BasicSettings: Model, LoRA/full, batch size
        [*] --> AdvancedSettings: Learning rate, epochs, LoRA params

        BasicSettings --> ConfigureDeployment: Repo name, push options
        AdvancedSettings --> ConfigureDeployment

        ConfigureDeployment --> StartTraining: All settings configured
    }

    TrainingConfiguration --> TrainingProcess: Start training

    state "Training Process" as TrainingProcess {
        [*] --> InitializeTrackio: Setup experiment tracking
        InitializeTrackio --> RunTrainingScript: Execute train.py or train_lora.py
        RunTrainingScript --> StreamLogs: Show real-time training logs
        StreamLogs --> MonitorProgress: Track metrics & checkpoints

        MonitorProgress --> TrainingComplete: Training finished
        MonitorProgress --> HandleErrors: Training failed
        HandleErrors --> RetryOrExit: User can retry or exit
    }

    TrainingProcess --> PostTraining: Training complete

    state "Post-Training Actions" as PostTraining {
        [*] --> PushToHub: Push model to HF Hub
        [*] --> GenerateModelCard: Create model card
        [*] --> DeployDemoSpace: Deploy interactive demo

        PushToHub --> ModelPublished: Model available on HF Hub
        GenerateModelCard --> ModelDocumented: Model card created
        DeployDemoSpace --> DemoReady: Demo space deployed
    }

    PostTraining --> [*]: Process complete

    %% Alternative paths
    DatasetCreation --> PushDatasetOnly: Skip training, push dataset only
    PushDatasetOnly --> DatasetPublished: Dataset on HF Hub

    %% Error handling
    TrainingProcess --> ErrorRecovery: Handle training errors
    ErrorRecovery --> RetryTraining: Retry with different settings
    RetryTraining --> TrainingConfiguration

    %% Styling and notes
    note right of LanguageSelection : User selects language for<br/>authentic phrases from<br/>NVIDIA Granary dataset
    note right of RecordingInterface : Users record themselves<br/>reading displayed phrases
    note right of DatasetCreation : JSONL format: {"audio_path": "...", "text": "..."}
    note right of TrainingConfiguration : Configure LoRA parameters,<br/>learning rate, epochs, etc.
    note right of TrainingProcess : Real-time log streaming<br/>with Trackio integration
    note right of PostTraining : Automated deployment<br/>pipeline
```

## Interface Workflow Overview

This diagram illustrates the complete user journey through the Voxtral ASR Fine-tuning interface. The workflow is designed to be intuitive and guide users through each step of the fine-tuning process.

### Key Workflow Stages

#### 1. Language & Dataset Setup
- **Language Selection**: Users choose from 25+ European languages supported by NVIDIA Granary
- **Phrase Loading**: System loads authentic, high-quality phrases in the selected language
- **Recording Interface**: Dynamic interface showing phrases with audio recording components
- **Progressive Disclosure**: Users can add more rows as needed (up to 100 recordings)

#### 2. Dataset Creation
- **From Recordings**: Process microphone recordings into WAV files and JSONL dataset
- **From Uploads**: Handle existing WAV/FLAC files with manual transcripts
- **JSONL Format**: Standard format with `audio_path` and `text` fields
- **Local Storage**: Datasets stored in `datasets/voxtral_user/` directory

#### 3. Training Configuration
- **Basic Settings**: Model selection, LoRA vs full fine-tuning, batch size
- **Advanced Settings**: Learning rate, epochs, gradient accumulation
- **LoRA Parameters**: r, alpha, dropout, audio tower freezing options
- **Repository Setup**: Model naming and Hugging Face Hub integration

#### 4. Training Process
- **Trackio Integration**: Automatic experiment tracking setup
- **Script Execution**: Calls appropriate training script (`train.py` or `train_lora.py`)
- **Log Streaming**: Real-time display of training progress and metrics
- **Error Handling**: Graceful handling of training failures with retry options

#### 5. Post-Training Actions
- **Model Publishing**: Automatic push to Hugging Face Hub
- **Model Card Generation**: Automated creation using `generate_model_card.py`
- **Demo Deployment**: One-click deployment of interactive demo spaces

### Alternative Paths

#### Dataset-Only Workflow
- Users can create and publish datasets without training models
- Useful for dataset curation and sharing

#### Error Recovery
- Training failures trigger error recovery flows
- Users can retry with modified parameters
- Comprehensive error logging and debugging information

### Technical Integration Points

#### External Services
- **NVIDIA Granary**: Source of high-quality multilingual ASR data
- **Hugging Face Hub**: Model and dataset storage and sharing
- **Trackio Spaces**: Experiment tracking and visualization

#### Script Integration
- **interface.py**: Main Gradio application orchestrating the workflow
- **train.py/train_lora.py**: Core training scripts with Trackio integration
- **push_to_huggingface.py**: Model/dataset publishing
- **deploy_demo_space.py**: Automated demo deployment
- **generate_model_card.py**: Model documentation generation

### User Experience Features

#### Progressive Interface Reveal
- Interface components are revealed as users progress through workflow
- Reduces cognitive load and guides users step-by-step

#### Real-time Feedback
- Live log streaming during training
- Progress indicators and status updates
- Immediate feedback on dataset creation and validation

#### Flexible Input Methods
- Support for both live recording and file uploads
- Multiple language options for diverse user needs
- Scalable recording interface (10-100 samples)

See also:
- [Architecture Overview](architecture.md)
- [Training Pipeline](training-pipeline.md)
- [Data Flow](data-flow.md)

