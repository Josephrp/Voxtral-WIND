#!/usr/bin/env python3
"""
Push Trained Models and Datasets to Hugging Face Hub

Usage:
    # Push a trained model
    python push_to_huggingface.py model /path/to/model my-model-repo

    # Push a dataset
    python push_to_huggingface.py dataset /path/to/dataset.jsonl my-dataset-repo

Authentication:
Set HF_TOKEN environment variable or use --token:
    export HF_TOKEN=your_token_here
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Set timeout for HF operations to prevent hanging
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_UPLOAD_TIMEOUT'] = '600'

try:
    from huggingface_hub import HfApi, create_repo, upload_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

logger = logging.getLogger(__name__)

class HuggingFacePusher:
    """Push trained models to Hugging Face Hub"""
    
    def __init__(
        self,
        model_path: str,
        repo_name: str,
        token: Optional[str] = None,
        private: bool = False,
        author_name: Optional[str] = None,
        model_description: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        # Original user input (may be just the repo name without username)
        self.repo_name = repo_name
        self.token = token or os.getenv('HF_TOKEN')
        self.private = private
        self.author_name = author_name
        self.model_description = model_description

        # Model card generation details
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # Initialize HF API
        if HF_AVAILABLE:
            self.api = HfApi(token=self.token)
        else:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        
        # Resolve the full repo id (username/repo) if user only provided repo name
        self.repo_id = self._resolve_repo_id(self.repo_name)
        # Artifact type detection (full vs lora)
        self.artifact_type: Optional[str] = None

        logger.info(f"Initialized HuggingFacePusher for {self.repo_id}")

    def _resolve_repo_id(self, repo_name: str) -> str:
        """Return a fully-qualified repo id in the form username/repo.

        If the provided name already contains a '/', it is returned unchanged.
        Otherwise, we attempt to derive the username from the authenticated token
        or from the HF_USERNAME environment variable.
        """
        try:
            if "/" in repo_name:
                return repo_name

            # Need a username. Prefer API whoami(), fallback to env HF_USERNAME
            username: Optional[str] = None
            if self.token:
                try:
                    user_info = self.api.whoami()
                    username = user_info.get("name") or user_info.get("username")
                except Exception:
                    username = None

            if not username:
                username = os.getenv("HF_USERNAME")

            if not username:
                raise ValueError(
                    "Username could not be determined. Provide a token or set HF_USERNAME, "
                    "or pass a fully-qualified repo id 'username/repo'."
                )

            return f"{username}/{repo_name}"
        except Exception as resolve_error:
            logger.error(f"Failed to resolve full repo id for '{repo_name}': {resolve_error}")
            # Fall back to provided value (may fail later at create/upload)
            return repo_name
    
    def create_repository(self) -> bool:
        """Create the Hugging Face repository"""
        try:
            logger.info(f"Creating repository: {self.repo_id}")
            
            # Create repository with timeout handling
            try:
                # Create repository
                create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=self.private,
                    exist_ok=True
                )
                
                logger.info(f"✅ Repository created: https://huggingface.co/{self.repo_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Repository creation failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to create repository: {e}")
            return False
    
    def _detect_artifact_type(self) -> str:
        """Detect whether output dir contains a full model or a LoRA adapter."""
        # LoRA artifacts
        lora_candidates = [
            self.model_path / "adapter_config.json",
            self.model_path / "adapter_model.safetensors",
            self.model_path / "adapter_model.bin",
        ]
        if any(p.exists() for p in lora_candidates) and (self.model_path / "adapter_config.json").exists():
            return "lora"

        # Full model artifacts
        full_candidates = [
            self.model_path / "config.json",
            self.model_path / "model.safetensors",
            self.model_path / "model.safetensors.index.json",
            self.model_path / "pytorch_model.bin",
        ]
        if any(p.exists() for p in full_candidates):
            return "full"

        return "unknown"

    def validate_model_path(self) -> bool:
        """Validate that the model path contains required files for Voxtral full or LoRA."""
        self.artifact_type = self._detect_artifact_type()
        if self.artifact_type == "lora":
            required = [self.model_path / "adapter_config.json"]
            if not all(p.exists() for p in required):
                logger.error("❌ LoRA artifacts missing required files (adapter_config.json)")
                return False
            # At least one adapter weight
            if not ((self.model_path / "adapter_model.safetensors").exists() or (self.model_path / "adapter_model.bin").exists()):
                logger.error("❌ LoRA artifacts missing adapter weights (adapter_model.safetensors or adapter_model.bin)")
                return False
            logger.info("✅ Detected LoRA adapter artifacts")
            return True

        if self.artifact_type == "full":
            # Relaxed set: require config.json and at least one model weights file
            if not (self.model_path / "config.json").exists():
                logger.error("❌ Missing config.json in model directory")
                return False
            if not ((self.model_path / "model.safetensors").exists() or (self.model_path / "model.safetensors.index.json").exists() or (self.model_path / "pytorch_model.bin").exists()):
                logger.error("❌ Missing model weights file (model.safetensors or pytorch_model.bin)")
                return False
            logger.info("✅ Detected full model artifacts")
            return True

        logger.error("❌ Could not detect model artifacts (neither full model nor LoRA)")
        return False
    
    def create_model_card(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Create a comprehensive model card using the generate_model_card.py script"""
        try:
            # Import the model card generator
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__)))
            from generate_model_card import ModelCardGenerator, create_default_variables
            
            # Create generator
            generator = ModelCardGenerator()
            
            # Create variables for the model card
            variables = create_default_variables()
            
            # Update with actual values
            variables.update({
                "repo_name": self.repo_id,
                "model_name": self.repo_id.split('/')[-1],
                "experiment_name": self.experiment_name or "model_push",
                "dataset_repo": self.dataset_repo,
                "author_name": self.author_name or "Model Author",
                "model_description": self.model_description or "A fine-tuned version of SmolLM3-3B for improved text generation capabilities.",
                "training_config_type": self.training_config_type or "Custom Configuration",
                "base_model": self.model_name or "HuggingFaceTB/SmolLM3-3B",
                "dataset_name": self.dataset_name or "Custom Dataset",
                "trainer_type": self.trainer_type or "SFTTrainer",
                "batch_size": str(self.batch_size) if self.batch_size else "8",
                "learning_rate": str(self.learning_rate) if self.learning_rate else "5e-6",
                "max_epochs": str(self.max_epochs) if self.max_epochs else "3",
                "max_seq_length": str(self.max_seq_length) if self.max_seq_length else "2048",
                "hardware_info": self._get_hardware_info(),
                "trackio_url": self.trackio_url or "N/A",
                "training_loss": str(results.get('train_loss', 'N/A')),
                "validation_loss": str(results.get('eval_loss', 'N/A')),
                "perplexity": str(results.get('perplexity', 'N/A')),
                "quantized_models": False  # Set to True if quantized models are available
            })
            
            # Generate the model card
            model_card_content = generator.generate_model_card(variables)
            
            logger.info("✅ Model card generated using generate_model_card.py")
            return model_card_content
            
        except Exception as e:
            logger.error(f"❌ Failed to generate model card with generator: {e}")
            logger.info("🔄 Falling back to simple model card")
            return self._create_simple_model_card(training_config, results)
    
    def _create_simple_model_card(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Create a simple model card tailored for Voxtral ASR (supports full and LoRA)."""
        tags = ["voxtral", "asr", "speech-to-text", "fine-tuning"]
        if self.artifact_type == "lora":
            tags.append("lora")
        front_matter = {
            "license": "apache-2.0",
            "tags": tags,
            "pipeline_tag": "automatic-speech-recognition",
        }
        fm_yaml = "---\n" + "\n".join([
            "license: apache-2.0",
            "tags:",
        ]) + "\n" + "\n".join([f"- {t}" for t in tags]) + "\n" + "pipeline_tag: automatic-speech-recognition\n---\n\n"
        model_title = self.repo_id.split('/')[-1]
        body = [
            f"# {model_title}",
            "",
            ("This repository contains a LoRA adapter for Voxtral ASR. "
             "Merge the adapter with the base model or load via PEFT for inference." if self.artifact_type == "lora" else
             "This repository contains a fine-tuned Voxtral ASR model."),
            "",
            "## Usage",
            "",
            ("```python\nfrom transformers import AutoProcessor\nfrom peft import PeftModel\nfrom transformers import AutoModelForSeq2SeqLM\n\nbase_model_id = 'mistralai/Voxtral-Mini-3B-2507'\nprocessor = AutoProcessor.from_pretrained(base_model_id)\nbase_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)\nmodel = PeftModel.from_pretrained(base_model, '{self.repo_id}')\n```" if self.artifact_type == "lora" else
             f"""```python
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("{self.repo_id}")
model = AutoModelForSeq2SeqLM.from_pretrained("{self.repo_id}")
```"""),
            "",
            "## Training Configuration",
            "",
            f"```json\n{json.dumps(training_config or {}, indent=2)}\n```",
            "",
            "## Training Results",
            "",
            f"```json\n{json.dumps(results or {}, indent=2)}\n```",
            "",
            f"**Hardware**: {self._get_hardware_info()}",
        ]
        return fm_yaml + "\n".join(body)
    
    def _get_model_size(self) -> float:
        """Get model size in GB"""
        try:
            total_size = 0
            for file in self.model_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024**3)  # Convert to GB
        except:
            return 0.0
    
    def _get_hardware_info(self) -> str:
        """Get hardware information"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return f"GPU: {gpu_name}"
            else:
                return "CPU"
        except:
            return "Unknown"
    
    def upload_model_files(self) -> bool:
        """Upload model files to Hugging Face Hub with timeout protection"""
        try:
            logger.info("Uploading model files...")
            
            # Upload all files in the model directory
            for file_path in self.model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.model_path)
                    remote_path = str(relative_path)
                    
                    logger.info(f"Uploading {relative_path}")
                    
                    try:
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=remote_path,
                            repo_id=self.repo_id,
                            token=self.token
                        )
                        logger.info(f"✅ Uploaded {relative_path}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to upload {relative_path}: {e}")
                        return False
            
            logger.info("✅ Model files uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload model files: {e}")
            return False
    
    def upload_training_results(self, results_path: str) -> bool:
        """Upload training results and logs"""
        try:
            logger.info("Uploading training results...")
            
            results_files = [
                "train_results.json",
                "eval_results.json",
                "training_config.json",
                "training.log"
            ]
            
            for file_name in results_files:
                file_path = Path(results_path) / file_name
                if file_path.exists():
                    logger.info(f"Uploading {file_name}")
                    upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"training_results/{file_name}",
                        repo_id=self.repo_id,
                        token=self.token
                    )
            
            logger.info("✅ Training results uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload training results: {e}")
            return False
    
    def create_readme(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Create and upload README.md"""
        try:
            logger.info("Creating README.md...")
            
            readme_content = f"""# {self.repo_id.split('/')[-1]}

A fine-tuned SmolLM3 model for text generation tasks.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Generate text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Information

- **Base Model**: HuggingFaceTB/SmolLM3-3B
- **Fine-tuning Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Model Size**: {self._get_model_size():.1f} GB
- **Training Steps**: {results.get('total_steps', 'Unknown')}
- **Final Loss**: {results.get('final_loss', 'Unknown')}
- **Dataset Repository**: {self.dataset_repo}

## Training Configuration

```json
{json.dumps(training_config, indent=2)}
```

## Performance Metrics

```json
{json.dumps(results, indent=2)}
```

## Experiment Tracking

Training metrics and configuration are stored in the HF Dataset repository: `{self.dataset_repo}`

## Files

- `model.safetensors.index.json`: Model weights (safetensors format)
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration
- `training_results/`: Training logs and results

## License

MIT License
"""
            
            # Write README to temporary file
            readme_path = Path("temp_readme.md")
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            # Upload README
            upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                token=self.token,
                repo_id=self.repo_id
            )
            
            # Clean up
            readme_path.unlink()
            
            logger.info("✅ README.md uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create README: {e}")
            return False
    

    def push_model(self, training_config: Optional[Dict[str, Any]] = None,
                   results: Optional[Dict[str, Any]] = None) -> bool:
        """Complete model push process"""
        logger.info(f"🚀 Starting model push to {self.repo_id}")

        # Validate model path
        if not self.validate_model_path():
            return False

        # Create repository
        if not self.create_repository():
            return False

        # Load training config and results if not provided
        if training_config is None:
            training_config = self._load_training_config()

        if results is None:
            results = self._load_training_results()

        # Create and upload model card
        model_card = self.create_model_card(training_config, results)
        model_card_path = Path("temp_model_card.md")
        with open(model_card_path, "w") as f:
            f.write(model_card)

        try:
            upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                token=self.token
            )
        finally:
            model_card_path.unlink()

        # Upload model files
        if not self.upload_model_files():
            return False

        # Upload training results
        if results:
            self.upload_training_results(str(self.model_path))

        # Log success
        logger.info(f"✅ Model successfully pushed to {self.repo_id}")
        logger.info(f"🎉 Model successfully pushed to: https://huggingface.co/{self.repo_id}")

        return True

    def push_dataset(self, dataset_path: str, dataset_repo_name: str) -> bool:
        """Push dataset to Hugging Face Hub including audio files"""
        logger.info(f"🚀 Starting dataset push to {dataset_repo_name}")

        try:
            from huggingface_hub import create_repo, upload_file
            import json

            # Determine full dataset repo name
            if "/" not in dataset_repo_name:
                dataset_repo_name = f"{self.repo_id.split('/')[0]}/{dataset_repo_name}"

            # Create dataset repository
            try:
                create_repo(dataset_repo_name, repo_type="dataset", token=self.token, exist_ok=True)
                logger.info(f"✅ Created dataset repository: {dataset_repo_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"❌ Failed to create dataset repo: {e}")
                    return False
                logger.info(f"📁 Dataset repository already exists: {dataset_repo_name}")

            # Read the dataset file
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                logger.error(f"❌ Dataset file not found: {dataset_path}")
                return False

            # Read and process the JSONL to collect audio files and update paths
            audio_files = []
            updated_rows = []
            total_audio_size = 0

            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        row = json.loads(line.strip())
                        audio_path = row.get("audio_path", "")

                        if audio_path:
                            audio_file = Path(audio_path)
                            if audio_file.exists():
                                # Store the original file for upload
                                audio_files.append(audio_file)
                                total_audio_size += audio_file.stat().st_size

                                # Update path to be relative for the dataset
                                row["audio_path"] = f"audio/{audio_file.name}"
                            else:
                                logger.warning(f"Audio file not found: {audio_path}")
                                row["audio_path"] = ""  # Clear missing files

                        updated_rows.append(row)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num + 1}: {e}")
                        continue

            # Create updated JSONL with relative paths
            temp_jsonl_path = dataset_file.parent / "temp_data.jsonl"
            with open(temp_jsonl_path, "w", encoding="utf-8") as f:
                for row in updated_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Upload the updated JSONL file
            upload_file(
                path_or_fileobj=str(temp_jsonl_path),
                path_in_repo="data.jsonl",
                repo_id=dataset_repo_name,
                repo_type="dataset",
                token=self.token
            )
            logger.info(f"✅ Uploaded dataset file: {dataset_file.name}")

            # Clean up temp file
            temp_jsonl_path.unlink()

            # Upload audio files
            uploaded_count = 0
            for audio_file in audio_files:
                try:
                    remote_path = f"audio/{audio_file.name}"
                    upload_file(
                        path_or_fileobj=str(audio_file),
                        path_in_repo=remote_path,
                        repo_id=dataset_repo_name,
                        repo_type="dataset",
                        token=self.token
                    )
                    uploaded_count += 1
                    logger.info(f"✅ Uploaded audio file: {audio_file.name}")
                except Exception as e:
                    logger.error(f"❌ Failed to upload {audio_file.name}: {e}")

            # Calculate total dataset size
            total_dataset_size = dataset_file.stat().st_size + total_audio_size

            # Create a comprehensive dataset README
            readme_content = f"""---
dataset_info:
  features:
    - name: audio_path
      dtype: string
    - name: text
      dtype: string
  splits:
    - name: train
      num_bytes: {dataset_file.stat().st_size}
      num_examples: {len(updated_rows)}
  download_size: {total_dataset_size}
  dataset_size: {total_dataset_size}
tags:
- voxtral
- asr
- speech-to-text
- fine-tuning
- audio-dataset
- tonic
---

# Voxtral ASR Dataset

This dataset was created for fine-tuning Voxtral ASR models.

## Dataset Structure

- **audio_path**: Relative path to the audio file (stored in `audio/` directory)
- **text**: Transcription of the audio

## Dataset Statistics

- **Number of examples**: {len(updated_rows)}
- **Audio files uploaded**: {uploaded_count}
- **Total dataset size**: {total_dataset_size:,} bytes

## Usage

```python
from datasets import load_dataset, Audio

# Load dataset
dataset = load_dataset("{dataset_repo_name}")

# Load audio data
dataset = dataset.cast_column("audio_path", Audio())

# Access first example
print(dataset[0]["text"])
print(dataset[0]["audio_path"])
```

## Loading with Audio Decoding

```python
from datasets import load_dataset, Audio

# Load with automatic audio decoding
dataset = load_dataset("{dataset_repo_name}")
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

# The audio column will contain the decoded audio arrays
audio_array = dataset[0]["audio_path"]["array"]
sampling_rate = dataset[0]["audio_path"]["sampling_rate"]
```

## Dataset Features

This dataset contains audio files with corresponding transcriptions for Voxtral ASR model fine-tuning.
All audio files are stored in the `audio/` directory and referenced using relative paths in the dataset.

## License

This dataset is created for research and educational purposes.
"""

            # Upload README
            readme_path = dataset_file.parent / "README.md"
            with open(readme_path, "w") as f:
                f.write(readme_content)

            upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=dataset_repo_name,
                repo_type="dataset",
                token=self.token
            )

            readme_path.unlink()  # Clean up temp file

            logger.info(f"✅ Dataset README uploaded")
            logger.info(f"🎉 Dataset successfully pushed to: https://huggingface.co/datasets/{dataset_repo_name}")
            logger.info(f"📊 Uploaded {len(updated_rows)} examples and {uploaded_count} audio files")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to push dataset: {e}")
            return False

    def test_dataset_push(self, dataset_path: str) -> bool:
        """Test dataset validation without uploading to Hugging Face Hub"""
        logger.info(f"🧪 Testing dataset validation for {dataset_path}")

        try:
            # Read the dataset file
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                logger.error(f"❌ Dataset file not found: {dataset_path}")
                return False

            # Read and process the JSONL to validate audio files
            audio_files = []
            updated_rows = []
            total_audio_size = 0
            missing_files = []
            invalid_json_lines = []

            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        row = json.loads(line.strip())
                        audio_path = row.get("audio_path", "")

                        if audio_path:
                            audio_file = Path(audio_path)
                            if audio_file.exists():
                                # Store the file info for validation
                                audio_files.append(audio_file)
                                total_audio_size += audio_file.stat().st_size
                            else:
                                missing_files.append(str(audio_path))

                        updated_rows.append(row)
                    except json.JSONDecodeError as e:
                        invalid_json_lines.append(f"Line {line_num + 1}: {e}")
                        continue

            # Report validation results
            logger.info("📊 Dataset Validation Results:")
            logger.info(f"   - Total examples: {len(updated_rows)}")
            logger.info(f"   - Valid audio files: {len(audio_files)}")
            logger.info(f"   - Total audio size: {total_audio_size:,} bytes")
            logger.info(f"   - Missing audio files: {len(missing_files)}")
            logger.info(f"   - Invalid JSON lines: {len(invalid_json_lines)}")

            if missing_files:
                logger.warning("⚠️ Missing audio files:")
                for missing in missing_files[:5]:  # Show first 5
                    logger.warning(f"   - {missing}")
                if len(missing_files) > 5:
                    logger.warning(f"   ... and {len(missing_files) - 5} more")

            if invalid_json_lines:
                logger.warning("⚠️ Invalid JSON lines:")
                for invalid in invalid_json_lines[:3]:  # Show first 3
                    logger.warning(f"   - {invalid}")
                if len(invalid_json_lines) > 3:
                    logger.warning(f"   ... and {len(invalid_json_lines) - 3} more")

            # Show sample of how paths will be converted
            if audio_files:
                logger.info("🔄 Path conversion preview:")
                for audio_file in audio_files[:3]:  # Show first 3
                    logger.info(f"   - {str(audio_file)} → audio/{audio_file.name}")

            # Overall validation status
            if len(updated_rows) == 0:
                logger.error("❌ No valid examples found in dataset")
                return False

            if len(missing_files) > 0:
                logger.warning("⚠️ Some audio files are missing - they will be skipped during upload")
            else:
                logger.info("✅ All audio files found and valid")

            logger.info("✅ Dataset validation completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to validate dataset: {e}")
            return False

    def _load_training_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        config_path = self.model_path / "training_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {"model_name": "HuggingFaceTB/SmolLM3-3B"}
    
    def _load_training_results(self) -> Dict[str, Any]:
        """Load training results"""
        results_path = self.model_path / "train_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                return json.load(f)
        return {"final_loss": "Unknown", "total_steps": "Unknown"}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Push trained model to Hugging Face Hub')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Model push subcommand
    model_parser = subparsers.add_parser('model', help='Push trained model to Hugging Face Hub')
    model_parser.add_argument('model_path', type=str, help='Path to trained model directory')
    model_parser.add_argument('repo_name', type=str, help='Hugging Face repository name (repo-name). Username will be auto-detected from your token.')
    model_parser.add_argument('--token', type=str, default=None, help='Hugging Face token')
    model_parser.add_argument('--private', action='store_true', help='Make repository private')
    model_parser.add_argument('--author-name', type=str, default=None, help='Author name for model card')
    model_parser.add_argument('--model-description', type=str, default=None, help='Model description for model card')
    model_parser.add_argument('--model-name', type=str, default=None, help='Base model name')
    model_parser.add_argument('--dataset-name', type=str, default=None, help='Dataset name')

    # Dataset push subcommand
    dataset_parser = subparsers.add_parser('dataset', help='Push dataset to Hugging Face Hub')
    dataset_parser.add_argument('dataset_path', type=str, help='Path to dataset JSONL file')
    dataset_parser.add_argument('repo_name', type=str, help='Hugging Face dataset repository name')
    dataset_parser.add_argument('--token', type=str, default=None, help='Hugging Face token')
    dataset_parser.add_argument('--private', action='store_true', help='Make repository private')
    dataset_parser.add_argument('--test', action='store_true', help='Test mode - validate dataset without uploading')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.command:
        logger.error("❌ No command specified. Use 'model' or 'dataset' subcommand.")
        return 1

    try:
        if args.command == 'model':
            logger.info("Starting model push to Hugging Face Hub")

            # Initialize pusher
            pusher = HuggingFacePusher(
                model_path=args.model_path,
                repo_name=args.repo_name,
                token=args.token,
                private=args.private,
                author_name=args.author_name,
                model_description=args.model_description,
                model_name=args.model_name,
                dataset_name=args.dataset_name
            )

            # Push model
            success = pusher.push_model()

            if success:
                logger.info("✅ Model push completed successfully!")
                logger.info(f"🌐 View your model at: https://huggingface.co/{args.repo_name}")
            else:
                logger.error("❌ Model push failed!")
                return 1

        elif args.command == 'dataset':
            logger.info("Starting dataset push to Hugging Face Hub")

            # Initialize pusher for dataset
            pusher = HuggingFacePusher(
                model_path="",  # Not needed for dataset push
                repo_name=args.repo_name,
                token=args.token,
                private=args.private
            )

            if getattr(args, 'test', False):
                # Test mode - validate dataset without uploading
                success = pusher.test_dataset_push(args.dataset_path)
                if success:
                    logger.info("✅ Dataset validation completed successfully!")
                else:
                    logger.error("❌ Dataset validation failed!")
                    return 1
            else:
                # Push dataset
                success = pusher.push_dataset(args.dataset_path, args.repo_name)

                if success:
                    logger.info("✅ Dataset push completed successfully!")
                    logger.info(f"📊 View your dataset at: https://huggingface.co/datasets/{args.repo_name}")
                else:
                    logger.error("❌ Dataset push failed!")
                    return 1

    except Exception as e:
        logger.error(f"❌ Error during push: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main()) 