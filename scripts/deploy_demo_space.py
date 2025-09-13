#!/usr/bin/env python3
"""
Demo Space Deployment Script
Deploys a Gradio demo space to Hugging Face Spaces for testing the fine-tuned model.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Import Hugging Face Hub API
try:
    from huggingface_hub import HfApi, create_repo, upload_file
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

# Add src to path for imports (kept for potential future imports)
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoSpaceDeployer:
    """Deploy demo space to Hugging Face Spaces"""
    
    def __init__(
        self,
        hf_token: str,
        # Token used for API actions that create/update the Space (write perms)
        hf_username: str,
        model_id: str,
        subfolder: str = "int4",
        space_name: Optional[str] = None,
        demo_type: Optional[str] = None,
        config_file: Optional[str] = None,
        # Optional token used as the Space's HF_TOKEN secret (read-only recommended)
        space_secret_token: Optional[str] = None,
        # Examples configuration
        examples_type: Optional[str] = None,
        disable_examples: Optional[bool] = None,
        examples_json: Optional[str] = None,
        # Branding overrides
        brand_owner_name: Optional[str] = None,
        brand_team_name: Optional[str] = None,
        brand_discord_url: Optional[str] = None,
        brand_hf_org: Optional[str] = None,
        brand_hf_label: Optional[str] = None,
        brand_hf_url: Optional[str] = None,
        brand_gh_org: Optional[str] = None,
        brand_gh_label: Optional[str] = None,
        brand_gh_url: Optional[str] = None,
        brand_project_name: Optional[str] = None,
        brand_project_url: Optional[str] = None,
    ):
        self.hf_token = hf_token
        # The token we will store in the Space secrets. Defaults to hf_token if not provided
        self.space_secret_token = space_secret_token or hf_token
        self.hf_username = hf_username
        # Allow passing just a repo name without username and auto-prefix
        self.model_id = model_id if "/" in model_id else f"{hf_username}/{model_id}"
        self.subfolder = subfolder
        self.space_name = space_name or f"{self.model_id.split('/')[-1]}-demo"
        self.space_id = f"{hf_username}/{self.space_name}"
        self.space_url = f"https://huggingface.co/spaces/{self.space_id}"
        self.config_file = config_file

        # Config-derived context
        self.system_message: Optional[str] = None
        self.developer_message: Optional[str] = None
        self.model_identity: Optional[str] = None
        self.reasoning_effort: Optional[str] = None
        # Examples context
        self.examples_type: Optional[str] = (examples_type or None)
        self.disable_examples: Optional[bool] = (disable_examples if disable_examples is not None else None)
        self.examples_json: Optional[str] = (examples_json or None)
        
        # Determine demo type from model_id if not provided
        if demo_type is None:
            demo_type = self._detect_demo_type(model_id)
        
        # Template paths based on model type
        self.demo_type = demo_type
        self.template_dir = Path(__file__).parent.parent / "templates" / "spaces" / f"demo_{demo_type}"
        self.workspace_dir = Path.cwd()
        
        # Initialize HF API
        if HF_HUB_AVAILABLE:
            self.api = HfApi(token=self.hf_token)
        else:
            self.api = None
            logger.warning("huggingface_hub not available, using CLI fallback")

        # Load optional config-specified messages
        try:
            self._load_config_messages()
        except Exception as e:
            logger.warning(f"Could not load config messages: {e}")

        # Branding defaults (can be overridden via CLI)
        self.brand_owner_name = brand_owner_name or self.hf_username or "Tonic"
        self.brand_team_name = brand_team_name or f"Team{self.brand_owner_name}"
        self.brand_discord_url = brand_discord_url or "https://discord.gg/qdfnvSPcqP"
        # HF org/link
        _default_hf_org = brand_hf_org or self.hf_username or "MultiTransformer"
        self.brand_hf_org = _default_hf_org
        self.brand_hf_label = brand_hf_label or self.brand_hf_org
        self.brand_hf_url = brand_hf_url or f"https://huggingface.co/{self.brand_hf_org}"
        # GitHub org/link
        _default_gh_org = brand_gh_org or self.hf_username or "tonic-ai"
        self.brand_gh_org = _default_gh_org
        self.brand_gh_label = brand_gh_label or self.brand_gh_org
        self.brand_gh_url = brand_gh_url or f"https://github.com/{self.brand_gh_org}"
        # Project link
        self.brand_project_name = brand_project_name or "MultiTonic"
        self.brand_project_url = brand_project_url or "https://github.com/MultiTonic"

    def _load_config_messages(self) -> None:
        """Load system/developer/model_identity from a training config file if provided."""
        if not self.config_file:
            return
        cfg_path = Path(self.config_file)
        if not cfg_path.exists():
            logger.warning(f"Config file not found: {cfg_path}")
            return

        # Ensure project root and config dir are importable for relative imports inside config
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        cfg_dir = project_root / "config"
        if str(cfg_dir) not in sys.path:
            sys.path.insert(0, str(cfg_dir))

        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", str(cfg_path))
        if not spec or not spec.loader:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        cfg = getattr(module, "config", None)
        if cfg is None:
            return
        self.system_message = getattr(cfg, "system_message", None)
        self.developer_message = getattr(cfg, "developer_message", None)
        chat_kwargs = getattr(cfg, "chat_template_kwargs", None)
        if isinstance(chat_kwargs, dict):
            self.model_identity = chat_kwargs.get("model_identity")
            self.reasoning_effort = chat_kwargs.get("reasoning_effort")
    
    def _detect_demo_type(self, model_id: str) -> str:
        """Detect the appropriate demo type based on model ID"""
        model_id_lower = model_id.lower()
        
        # Voxtral ASR models
        if "voxtral" in model_id_lower:
            logger.info(f"Detected Voxtral model, using demo_voxtral template")
            return "voxtral"
        
        # Check for GPT-OSS models
        if "gpt-oss" in model_id_lower or "gpt_oss" in model_id_lower:
            logger.info(f"Detected GPT-OSS model, using demo_gpt template")
            return "gpt"
        
        # Check for SmolLM models (default)
        elif "smollm" in model_id_lower or "smol" in model_id_lower:
            logger.info(f"Detected SmolLM model, using demo_smol template")
            return "smol"
        
        # Default to SmolLM for unknown models
        else:
            logger.info(f"Unknown model type, defaulting to demo_smol template")
            return "smol"
    
    def _generate_env_setup(self) -> str:
        """Generate environment variable setup based on demo type and model"""
        if self.demo_type == "gpt":
            # For GPT-OSS models, we need more sophisticated environment setup
            model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
            import json as _json
            ex_type_line = f"os.environ['EXAMPLES_TYPE'] = {_json.dumps(self.examples_type)}\n" if self.examples_type else ""
            disable_line = ("os.environ['DISABLE_EXAMPLES'] = 'true'\n" if self.disable_examples
                            else ("os.environ['DISABLE_EXAMPLES'] = 'false'\n" if self.disable_examples is not None else ""))
            examples_json_line = f"os.environ['EXAMPLES_JSON'] = {_json.dumps(self.examples_json)}\n" if self.examples_json else ""
            env_setup = f"""# Environment variables for GPT-OSS model configuration
import os
import json
os.environ['HF_MODEL_ID'] = json.dumps({_json.dumps(self.model_id)})
os.environ['LORA_MODEL_ID'] = json.dumps({_json.dumps(self.model_id)})
os.environ['BASE_MODEL_ID'] = 'openai/gpt-oss-20b'
os.environ['MODEL_SUBFOLDER'] = json.dumps({_json.dumps(self.subfolder if self.subfolder else "")})
os.environ['MODEL_NAME'] = json.dumps({_json.dumps(model_name)})
os.environ['MODEL_IDENTITY'] = json.dumps({_json.dumps(self.model_identity or "")})
os.environ['SYSTEM_MESSAGE'] = json.dumps({_json.dumps(self.system_message or (self.model_identity or ""))})
os.environ['DEVELOPER_MESSAGE'] = json.dumps({_json.dumps(self.developer_message or "")})
os.environ['REASONING_EFFORT'] = json.dumps({_json.dumps((self.reasoning_effort or "medium"))})
{ex_type_line}{disable_line}{examples_json_line}

# Branding/owner variables
os.environ['HF_USERNAME'] = json.dumps({_json.dumps(self.hf_username)})
os.environ['BRAND_OWNER_NAME'] = json.dumps({_json.dumps(self.brand_owner_name)})
os.environ['BRAND_TEAM_NAME'] = json.dumps({_json.dumps(self.brand_team_name)})
os.environ['BRAND_DISCORD_URL'] = json.dumps({_json.dumps(self.brand_discord_url)})
os.environ['BRAND_HF_ORG'] = json.dumps({_json.dumps(self.brand_hf_org)})
os.environ['BRAND_HF_LABEL'] = json.dumps({_json.dumps(self.brand_hf_label)})
os.environ['BRAND_HF_URL'] = json.dumps({_json.dumps(self.brand_hf_url)})
os.environ['BRAND_GH_ORG'] = json.dumps({_json.dumps(self.brand_gh_org)})
os.environ['BRAND_GH_LABEL'] = json.dumps({_json.dumps(self.brand_gh_label)})
os.environ['BRAND_GH_URL'] = json.dumps({_json.dumps(self.brand_gh_url)})
os.environ['BRAND_PROJECT_NAME'] = json.dumps({_json.dumps(self.brand_project_name)})
os.environ['BRAND_PROJECT_URL'] = json.dumps({_json.dumps(self.brand_project_url)})
"""
        elif self.demo_type == "voxtral":
            # For Voxtral, we do not inject env setup into app.py.
            # Space variables are set via the API in set_space_secrets().
            env_setup = ""
        else:
            # For SmolLM models, use simpler setup
            import json as _json
            ex_type_line = f"os.environ['EXAMPLES_TYPE'] = {_json.dumps(self.examples_type)}\n" if self.examples_type else ""
            disable_line = ("os.environ['DISABLE_EXAMPLES'] = 'true'\n" if self.disable_examples
                            else ("os.environ['DISABLE_EXAMPLES'] = 'false'\n" if self.disable_examples is not None else ""))
            examples_json_line = f"os.environ['EXAMPLES_JSON'] = {_json.dumps(self.examples_json)}\n" if self.examples_json else ""
            env_setup = f"""# Environment variables for model configuration
import os
import json
os.environ['HF_MODEL_ID'] = json.dumps({_json.dumps(self.model_id)})
os.environ['MODEL_SUBFOLDER'] = json.dumps({_json.dumps(self.subfolder if self.subfolder else "")})
os.environ['MODEL_NAME'] = json.dumps({_json.dumps(self.model_id.split("/")[-1])})
os.environ['MODEL_IDENTITY'] = json.dumps({_json.dumps(self.model_identity or "")})
os.environ['SYSTEM_MESSAGE'] = json.dumps({_json.dumps(self.system_message or (self.model_identity or ""))})
os.environ['DEVELOPER_MESSAGE'] = json.dumps({_json.dumps(self.developer_message or "")})
os.environ['REASONING_EFFORT'] = json.dumps({_json.dumps((self.reasoning_effort or "medium"))})
{ex_type_line}{disable_line}{examples_json_line}

# Branding/owner variables
os.environ['HF_USERNAME'] = json.dumps({_json.dumps(self.hf_username)})
os.environ['BRAND_OWNER_NAME'] = json.dumps({_json.dumps(self.brand_owner_name)})
os.environ['BRAND_TEAM_NAME'] = json.dumps({_json.dumps(self.brand_team_name)})
os.environ['BRAND_DISCORD_URL'] = json.dumps({_json.dumps(self.brand_discord_url)})
os.environ['BRAND_HF_ORG'] = json.dumps({_json.dumps(self.brand_hf_org)})
os.environ['BRAND_HF_LABEL'] = json.dumps({_json.dumps(self.brand_hf_label)})
os.environ['BRAND_HF_URL'] = json.dumps({_json.dumps(self.brand_hf_url)})
os.environ['BRAND_GH_ORG'] = json.dumps({_json.dumps(self.brand_gh_org)})
os.environ['BRAND_GH_LABEL'] = json.dumps({_json.dumps(self.brand_gh_label)})
os.environ['BRAND_GH_URL'] = json.dumps({_json.dumps(self.brand_gh_url)})
os.environ['BRAND_PROJECT_NAME'] = json.dumps({_json.dumps(self.brand_project_name)})
os.environ['BRAND_PROJECT_URL'] = json.dumps({_json.dumps(self.brand_project_url)})
"""
        return env_setup
    
    def _set_model_variables(self):
        """Set model-specific environment variables in the space"""
        try:
            # Common variables for all models
            self.api.add_space_variable(
                repo_id=self.space_id,
                key="HF_MODEL_ID",
                value=self.model_id,
                description="Model ID for the demo"
            )
            logger.info(f"✅ Successfully set HF_MODEL_ID variable: {self.model_id}")
            
            if self.subfolder and self.subfolder.strip():
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="MODEL_SUBFOLDER",
                    value=self.subfolder,
                    description="Model subfolder for the demo"
                )
                logger.info(f"✅ Successfully set MODEL_SUBFOLDER variable: {self.subfolder}")
            else:
                logger.info("ℹ️ No subfolder specified, using main model")
            
            # GPT-OSS specific variables
            if self.demo_type == "gpt":
                model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="LORA_MODEL_ID",
                    value=self.model_id,
                    description="LoRA/Fine-tuned model ID"
                )
                logger.info(f"✅ Successfully set LORA_MODEL_ID variable: {self.model_id}")
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="BASE_MODEL_ID",
                    value="openai/gpt-oss-20b",
                    description="Base model ID for GPT-OSS"
                )
                logger.info("✅ Successfully set BASE_MODEL_ID variable: openai/gpt-oss-20b")
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="MODEL_NAME",
                    value=model_name,
                    description="Display name for the model"
                )
                logger.info(f"✅ Successfully set MODEL_NAME variable: {model_name}")

            # Voxtral-specific variables
            elif self.demo_type == "voxtral":
                # HF_MODEL_ID was already set above; set a readable MODEL_NAME
                vox_model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="MODEL_NAME",
                    value=vox_model_name,
                    description="Display name for the Voxtral model"
                )
                logger.info(f"✅ Set Voxtral MODEL_NAME variable: {vox_model_name}")

            # Optional context variables
            if self.model_identity:
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="MODEL_IDENTITY",
                    value=self.model_identity,
                    description="Default model identity/system persona"
                )
                logger.info("✅ Set MODEL_IDENTITY variable")
            if self.system_message or self.model_identity:
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="SYSTEM_MESSAGE",
                    value=self.system_message or self.model_identity or "",
                    description="Default system message"
                )
                logger.info("✅ Set SYSTEM_MESSAGE variable")
            if self.developer_message:
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="DEVELOPER_MESSAGE",
                    value=self.developer_message,
                    description="Default developer message"
                )
                logger.info("✅ Set DEVELOPER_MESSAGE variable")
            if self.reasoning_effort:
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="REASONING_EFFORT",
                    value=self.reasoning_effort,
                    description="Default reasoning effort (low|medium|high)"
                )
                logger.info("✅ Set REASONING_EFFORT variable")

            # Branding variables
            branding_vars = {
                "HF_USERNAME": self.hf_username,
                "BRAND_OWNER_NAME": self.brand_owner_name,
                "BRAND_TEAM_NAME": self.brand_team_name,
                "BRAND_DISCORD_URL": self.brand_discord_url,
                "BRAND_HF_ORG": self.brand_hf_org,
                "BRAND_HF_LABEL": self.brand_hf_label,
                "BRAND_HF_URL": self.brand_hf_url,
                "BRAND_GH_ORG": self.brand_gh_org,
                "BRAND_GH_LABEL": self.brand_gh_label,
                "BRAND_GH_URL": self.brand_gh_url,
                "BRAND_PROJECT_NAME": self.brand_project_name,
                "BRAND_PROJECT_URL": self.brand_project_url,
            }
            for key, value in branding_vars.items():
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key=key,
                    value=value,
                    description=f"Branding: {key}"
                )
            logger.info("✅ Set branding variables")

            # Examples variables
            if self.examples_type:
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="EXAMPLES_TYPE",
                    value=self.examples_type,
                    description="Examples pack type (e.g., general|medical)"
                )
                logger.info(f"✅ Set EXAMPLES_TYPE={self.examples_type}")
            if self.disable_examples is not None:
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="DISABLE_EXAMPLES",
                    value=("true" if self.disable_examples else "false"),
                    description="Disable built-in examples"
                )
                logger.info(f"✅ Set DISABLE_EXAMPLES={self.disable_examples}")
            if self.examples_json:
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="EXAMPLES_JSON",
                    value=self.examples_json,
                    description="Custom examples JSON override"
                )
                logger.info("✅ Set EXAMPLES_JSON override")
            
        except Exception as e:
            logger.error(f"❌ Failed to set model variables: {e}")
    
    def validate_model_exists(self) -> bool:
        """Validate that the model exists on Hugging Face Hub"""
        try:
            logger.info(f"Validating model: {self.model_id}")
            
            if HF_HUB_AVAILABLE:
                # Use HF Hub API
                try:
                    model_info = self.api.model_info(self.model_id)
                    logger.info(f"✅ Model {self.model_id} exists and is accessible")
                    return True
                except Exception as e:
                    logger.error(f"❌ Model {self.model_id} not found via API: {e}")
                    return False
            else:
                # Fallback to requests
                url = f"https://huggingface.co/api/models/{self.model_id}"
                headers = {"Authorization": f"Bearer {self.hf_token}"}
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"✅ Model {self.model_id} exists and is accessible")
                    return True
                else:
                    logger.error(f"❌ Model {self.model_id} not found or not accessible")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Error validating model: {e}")
            return False
    
    def create_space_repository(self) -> bool:
        """Create the space repository on Hugging Face Hub"""
        try:
            logger.info(f"Creating Space: {self.space_name}")
            
            if not HF_HUB_AVAILABLE:
                logger.warning("huggingface_hub not available, falling back to CLI")
                return self._create_space_cli()
            
            # Use the latest HF Hub API to create space
            try:
                # Create the space using the API
                create_repo(
                    repo_id=self.space_id,
                    token=self.hf_token,
                    repo_type="space",
                    exist_ok=True,
                    private=False,  # Spaces are typically public
                    space_sdk="gradio",  # Specify Gradio SDK
                    space_hardware="cpu-basic"  # Use basic CPU
                )
                
                logger.info(f"✅ Space created successfully: {self.space_url}")
                return True
                
            except Exception as api_error:
                logger.error(f"API creation failed: {api_error}")
                logger.info("Falling back to CLI method...")
                return self._create_space_cli()
                
        except Exception as e:
            logger.error(f"❌ Error creating space: {e}")
            return False
    
    def _create_space_cli(self) -> bool:
        """Fallback method using CLI commands"""
        try:
            logger.info("Using CLI fallback method...")
            
            # Set HF token for CLI
            os.environ['HF_TOKEN'] = self.hf_token
            
            # Try multiple CLI variants depending on installed version
            cli_attempts = [
                ["hf", "space", "create", self.space_id, "--type", "gradio", "--hardware", "cpu-basic"],
                ["huggingface-cli", "space", "create", self.space_id, "--type", "gradio", "--hardware", "cpu-basic"],
                ["hf", "repo", "create", self.space_id, "--repo-type", "space", "--space-sdk", "gradio"],
                ["huggingface-cli", "repo", "create", self.space_id, "--repo-type", "space", "--space-sdk", "gradio"],
            ]

            last_err = None
            for cmd in cli_attempts:
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Space created (CLI): {self.space_url}")
                    break
                else:
                    last_err = result.stderr
                    logger.warning(f"CLI attempt failed: {last_err}")
            else:
                logger.error(f"❌ Failed to create space via CLI: {last_err}")
                return False

            # Verify the space exists and is recognized as a space
            try:
                for _ in range(10):
                    try:
                        info = self.api.repo_info(self.space_id, repo_type="space")
                        if info:  # type: ignore
                            logger.info("✅ Verified space existence via API")
                            return True
                    except Exception:
                        time.sleep(2)
                logger.error("❌ Space verification timed out after CLI creation")
                return False
            except Exception as e:
                logger.error(f"❌ Error verifying space: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating space with CLI: {e}")
            return False
    
    def prepare_space_files(self) -> str:
        """Prepare all necessary files for the Space in a temporary directory"""
        try:
            logger.info("Preparing Space files...")
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Copy template files
            copied_files = []
            for file_path in self.template_dir.iterdir():
                if file_path.is_file():
                    dest_path = Path(temp_dir) / file_path.name
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(file_path.name)
                    logger.info(f"✅ Copied {file_path.name} to temp directory")
            
            # Update app.py with environment variables (skip for Voxtral)
            app_file = Path(temp_dir) / "app.py"
            if app_file.exists() and self.demo_type != "voxtral":
                with open(app_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                env_setup = self._generate_env_setup()

                if env_setup:
                    # Insert after imports
                    lines = content.split('\n')
                    import_end = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            import_end = i + 1
                        elif line.strip() == '' and import_end > 0:
                            break

                    lines.insert(import_end, env_setup)
                    content = '\n'.join(lines)

                    with open(app_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    logger.info("✅ Updated app.py with model configuration")
            
            # For Voxtral keep the template README. For others, create a README with YAML front matter.
            if self.demo_type != "voxtral":
                yaml_front_matter = (
                    f"---\n"
                    f"title: {'GPT-OSS Demo' if self.demo_type == 'gpt' else 'SmolLM3 Demo'}\n"
                    f"emoji: {'🌟' if self.demo_type == 'gpt' else '💃🏻'}\n"
                    f"colorFrom: {'blue' if self.demo_type == 'gpt' else 'green'}\n"
                    f"colorTo: {'pink' if self.demo_type == 'gpt' else 'purple'}\n"
                    f"sdk: gradio\n"
                    f"sdk_version: 5.40.0\n"
                    f"app_file: app.py\n"
                    f"pinned: false\n"
                    f"short_description: Interactive demo for {self.model_id}\n"
                    + ("license: mit\\n" if self.demo_type != 'gpt' else "") +
                    f"---\n\n"
                )

                readme_content = (
                    yaml_front_matter
                    + f"# Demo: {self.model_id}\n\n"
                    + f"This is an interactive demo for the fine-tuned model {self.model_id}.\n\n"
                    + "## Features\n"
                      "- Interactive chat interface\n"
                      "- Customizable system & developer prompts\n"
                      "- Advanced generation parameters\n"
                      "- Thinking mode support\n\n"
                    + "## Model Information\n"
                      f"- **Model ID**: {self.model_id}\n"
                      f"- **Subfolder**: {self.subfolder if self.subfolder and self.subfolder.strip() else 'main'}\n"
                      f"- **Deployed by**: {self.hf_username}\n"
                      + ("- **Base Model**: openai/gpt-oss-20b\n" if self.demo_type == 'gpt' else "")
                      + "\n"
                    + "## Configuration\n"
                      "- **Model Identity**:\n\n"
                      f"```\n{self.model_identity or 'Not set'}\n```\n\n"
                      "- **System Message** (default):\n\n"
                      f"```\n{(self.system_message or self.model_identity) or 'Not set'}\n```\n\n"
                      "- **Developer Message** (default):\n\n"
                      f"```\n{self.developer_message or 'Not set'}\n```\n\n"
                      "These defaults come from the selected training configuration and can be adjusted in the UI when you run the demo.\n\n"
                    + "## Usage\n"
                      "Simply start chatting with the model using the interface below!\n\n"
                    + "---\n"
                      "*This demo was automatically deployed by the SmolFactory Fine-tuning Pipeline*\n"
                )

                with open(Path(temp_dir) / "README.md", 'w', encoding='utf-8') as f:
                    f.write(readme_content)
            
            logger.info(f"✅ Prepared {len(copied_files)} files in temporary directory")
            return temp_dir
            
        except Exception as e:
            logger.error(f"❌ Error preparing files: {e}")
            return None
    
    def upload_files_to_space(self, temp_dir: str) -> bool:
        """Upload files to the Space using HF Hub API directly"""
        try:
            logger.info("Uploading files to Space using HF Hub API...")
            
            if not HF_HUB_AVAILABLE:
                logger.error("❌ huggingface_hub not available for file upload")
                return self._upload_files_cli(temp_dir)
            
            # Upload each file using the HF Hub API
            temp_path = Path(temp_dir)
            uploaded_files = []
            
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    # Retry uploads to absorb propagation delays
                    for attempt in range(5):
                        try:
                            upload_file(
                                path_or_fileobj=str(file_path),
                                path_in_repo=file_path.name,
                                repo_id=self.space_id,
                                repo_type="space",
                                token=self.hf_token
                            )
                            uploaded_files.append(file_path.name)
                            logger.info(f"✅ Uploaded {file_path.name}")
                            break
                        except Exception as e:
                            if "404" in str(e) or "Not Found" in str(e):
                                logger.warning(f"Upload failed (likely propagation). Retry {attempt+1}/5 in 2s: {e}")
                                time.sleep(2)
                                continue
                            logger.error(f"❌ Failed to upload {file_path.name}: {e}")
                            return False
            
            logger.info(f"✅ Successfully uploaded {len(uploaded_files)} files to Space")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error uploading files: {e}")
            return self._upload_files_cli(temp_dir)
    
    def _upload_files_cli(self, temp_dir: str) -> bool:
        """Fallback method using CLI for file upload"""
        try:
            logger.info("Using CLI fallback for file upload...")
            
            # Set HF token for CLI
            os.environ['HF_TOKEN'] = self.hf_token
            
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            subprocess.run(["git", "config", "user.name", "Demo Deployer"], cwd=temp_dir, check=True)
            subprocess.run(["git", "config", "user.email", "demo@example.com"], cwd=temp_dir, check=True)
            
            # Add files
            subprocess.run(["git", "add", "."], cwd=temp_dir, check=True)
            subprocess.run(["git", "commit", "-m", f"Deploy demo for {self.model_id}"], cwd=temp_dir, check=True)
            
            # Add remote and push
            remote_url = f"https://{self.hf_token}@huggingface.co/spaces/{self.space_id}"
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=temp_dir, check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=temp_dir, check=True)
            
            logger.info(f"✅ Successfully pushed files to space: {self.space_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git operation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error pushing to space: {e}")
            return False
    
    def set_space_secrets(self) -> bool:
        """Set environment variables/secrets for the Space using HF Hub API"""
        try:
            logger.info("Setting Space secrets using HF Hub API...")
            
            if not HF_HUB_AVAILABLE:
                logger.warning("❌ huggingface_hub not available for setting secrets")
                return self._manual_secret_setup()
            
            # Set the HF_TOKEN secret for the space using the API
            try:
                self.api.add_space_secret(
                    repo_id=self.space_id,
                    key="HF_TOKEN",
                    value=self.space_secret_token,
                    description="Hugging Face token for model access"
                )
                logger.info("✅ Successfully set HF_TOKEN secret via API")
                
                # Set model-specific environment variables
                self._set_model_variables()
                
                return True
                
            except Exception as api_error:
                logger.error(f"❌ Failed to set secrets via API: {api_error}")
                logger.info("Falling back to manual setup...")
                return self._manual_secret_setup()
            
        except Exception as e:
            logger.error(f"❌ Error setting space secrets: {e}")
            return self._manual_secret_setup()
    
    def _manual_secret_setup(self) -> bool:
        """Fallback method for manual secret setup"""
        logger.info("📝 Manual Space Secrets Configuration:")
        logger.info(f"   HF_TOKEN=<hidden>")
        logger.info(f"   HF_MODEL_ID={self.model_id}")
        if self.subfolder and self.subfolder.strip():
            logger.info(f"   MODEL_SUBFOLDER={self.subfolder}")
        else:
            logger.info("   MODEL_SUBFOLDER=(empty - using main model)")
        
        # GPT-OSS specific variables
        if self.demo_type == "gpt":
            model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
            logger.info(f"   LORA_MODEL_ID={self.model_id}")
            logger.info(f"   BASE_MODEL_ID=openai/gpt-oss-20b")
            logger.info(f"   MODEL_NAME={model_name}")
        if self.model_identity:
            logger.info(f"   MODEL_IDENTITY={self.model_identity}")
        if self.system_message:
            logger.info(f"   SYSTEM_MESSAGE={self.system_message}")
        if self.developer_message:
            logger.info(f"   DEVELOPER_MESSAGE={self.developer_message}")
        # Branding variables
        logger.info(f"   HF_USERNAME={self.hf_username}")
        logger.info(f"   BRAND_OWNER_NAME={self.brand_owner_name}")
        logger.info(f"   BRAND_TEAM_NAME={self.brand_team_name}")
        logger.info(f"   BRAND_DISCORD_URL={self.brand_discord_url}")
        logger.info(f"   BRAND_HF_ORG={self.brand_hf_org}")
        logger.info(f"   BRAND_HF_LABEL={self.brand_hf_label}")
        logger.info(f"   BRAND_HF_URL={self.brand_hf_url}")
        logger.info(f"   BRAND_GH_ORG={self.brand_gh_org}")
        logger.info(f"   BRAND_GH_LABEL={self.brand_gh_label}")
        logger.info(f"   BRAND_GH_URL={self.brand_gh_url}")
        logger.info(f"   BRAND_PROJECT_NAME={self.brand_project_name}")
        logger.info(f"   BRAND_PROJECT_URL={self.brand_project_url}")

        # Examples variables
        if self.examples_type:
            logger.info(f"   EXAMPLES_TYPE={self.examples_type}")
        if self.disable_examples is not None:
            logger.info(f"   DISABLE_EXAMPLES={'true' if self.disable_examples else 'false'}")
        if self.examples_json:
            logger.info(f"   EXAMPLES_JSON={self.examples_json}")
        
        logger.info(f"\n🔧 To set secrets in your Space:")
        logger.info(f"1. Go to your Space settings: {self.space_url}/settings")
        logger.info("2. Navigate to the 'Repository secrets' section")
        logger.info("3. Add the following secrets:")
        logger.info(f"   Name: HF_TOKEN")
        logger.info(f"   Value: <your token>")
        logger.info(f"   Name: HF_MODEL_ID")
        logger.info(f"   Value: {self.model_id}")
        if self.subfolder and self.subfolder.strip():
            logger.info(f"   Name: MODEL_SUBFOLDER")
            logger.info(f"   Value: {self.subfolder}")
        else:
            logger.info("   Name: MODEL_SUBFOLDER")
            logger.info("   Value: (leave empty)")
        
        # GPT-OSS specific variables
        if self.demo_type == "gpt":
            model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
            logger.info(f"   Name: LORA_MODEL_ID")
            logger.info(f"   Value: {self.model_id}")
            logger.info(f"   Name: BASE_MODEL_ID")
            logger.info(f"   Value: openai/gpt-oss-20b")
            logger.info(f"   Name: MODEL_NAME")
            logger.info(f"   Value: {model_name}")
        
        logger.info("4. Save the secrets")
        
        return True
    
    def test_space(self) -> bool:
        """Test if the Space is working correctly"""
        try:
            logger.info("Testing Space...")
            
            # Wait a bit for the space to build
            logger.info("Waiting 180 seconds for Space to build...")
            time.sleep(180)
            
            # Try to access the space
            response = requests.get(self.space_url, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Space is accessible: {self.space_url}")
                return True
            else:
                logger.warning(f"⚠️  Space returned status code: {response.status_code}")
                logger.warning(f"Response: {response.text[:500]}...")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing space: {e}")
            return False
    
    def deploy(self) -> bool:
        """Main deployment method"""
        logger.info(f"🚀 Starting demo space deployment for {self.model_id}")
        
        # Step 1: Validate model exists
        if not self.validate_model_exists():
            return False
        
        # Step 2: Create space repository
        if not self.create_space_repository():
            return False
        
        # Step 3: Prepare files
        temp_dir = self.prepare_space_files()
        if not temp_dir:
            return False
        
        # Step 4: Upload files
        if not self.upload_files_to_space(temp_dir):
            return False
        
        # Step 5: Set space secrets
        if not self.set_space_secrets():
            return False
        
        # Step 6: Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
            logger.info("✅ Cleaned up temporary directory")
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not clean up temp directory: {e}")
        
        # Step 7: Test space
        if not self.test_space():
            logger.warning("⚠️  Space created but may need more time to build")
            logger.info("Please check the Space manually in a few minutes")
        
        logger.info(f"🎉 Demo space deployment completed!")
        logger.info(f"📊 Space URL: {self.space_url}")
        logger.info(f"🔧 Space configuration: {self.space_url}/settings")
        
        return True

def main():
    """Main function for command line usage"""
    print("Demo Space Deployment Script")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description="Deploy demo space to Hugging Face Spaces")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token")
    parser.add_argument(
        "--space-secret-token",
        required=False,
        help="Token to store as Space secret HF_TOKEN (defaults to --hf-token). Use a READ token here for least privilege.",
    )
    parser.add_argument("--hf-username", required=True, help="Hugging Face username")
    parser.add_argument("--model-id", required=True, help="Model ID to deploy demo for")
    parser.add_argument("--subfolder", default="int4", help="Model subfolder (default: int4)")
    parser.add_argument("--space-name", help="Custom space name (optional)")
    parser.add_argument("--demo-type", choices=["smol", "gpt", "voxtral"], help="Demo type: 'smol' for SmolLM, 'gpt' for GPT-OSS, 'voxtral' for Voxtral ASR (auto-detected if not specified)")
    parser.add_argument("--config-file", help="Path to the training config file to import context (system/developer/model_identity)")
    # Examples configuration
    parser.add_argument("--examples-type", choices=["general", "medical"], help="Examples pack to enable in the demo UI")
    parser.add_argument("--disable-examples", action="store_true", help="Disable rendering of example prompts in the UI")
    parser.add_argument("--examples-json", help="Custom examples JSON (list[str]) to override built-in examples")
    # Branding customization
    parser.add_argument("--brand-owner-name", help="Owner name shown in the UI title (defaults to HF username)")
    parser.add_argument("--brand-team-name", help="Team name shown in Join Us (defaults to Team<owner>)")
    parser.add_argument("--brand-discord-url", help="Discord invite URL for Join Us section")
    parser.add_argument("--brand-hf-org", help="Hugging Face org/username to link in Join Us")
    parser.add_argument("--brand-hf-label", help="Label for the HF link (defaults to org)")
    parser.add_argument("--brand-hf-url", help="Custom HF link URL (defaults to https://huggingface.co/<org>)")
    parser.add_argument("--brand-gh-org", help="GitHub org/username to link in Join Us")
    parser.add_argument("--brand-gh-label", help="Label for the GitHub link (defaults to org)")
    parser.add_argument("--brand-gh-url", help="Custom GitHub link URL (defaults to https://github.com/<org>)")
    parser.add_argument("--brand-project-name", help="Project name to link in Join Us")
    parser.add_argument("--brand-project-url", help="Project URL to link in Join Us")
    
    args = parser.parse_args()
    
    deployer = DemoSpaceDeployer(
        hf_token=args.hf_token,
        space_secret_token=(args.space_secret_token or None),
        hf_username=args.hf_username,
        model_id=args.model_id,
        subfolder=args.subfolder,
        space_name=args.space_name,
        demo_type=args.demo_type,
        config_file=args.config_file,
        examples_type=args.examples_type,
        disable_examples=(True if getattr(args, 'disable_examples', False) else None),
        examples_json=args.examples_json,
        brand_owner_name=args.brand_owner_name,
        brand_team_name=args.brand_team_name,
        brand_discord_url=args.brand_discord_url,
        brand_hf_org=args.brand_hf_org,
        brand_hf_label=args.brand_hf_label,
        brand_hf_url=args.brand_hf_url,
        brand_gh_org=args.brand_gh_org,
        brand_gh_label=args.brand_gh_label,
        brand_gh_url=args.brand_gh_url,
        brand_project_name=args.brand_project_name,
        brand_project_url=args.brand_project_url,
    )
    
    success = deployer.deploy()
    
    if success:
        print("\n✅ Deployment successful!")
        print(f"🌐 Your Demo Space: {deployer.space_url}")
        print(f"👤 Username: {deployer.hf_username}")
        print(f"🤖 Model: {deployer.model_id}")
        print("\nNext steps:")
        print("1. Wait for the Space to build (usually 2-5 minutes)")
        print("2. Secrets have been automatically set via API")
        print("3. Test the interface by visiting the Space URL")
        print("4. Share your demo with others!")
        print("\nIf the Space doesn't work immediately, check:")
        print("- The Space logs at the Space URL")
        print("- That all files were uploaded correctly")
        print("- That the HF token has write permissions")
        print("- That the secrets were set correctly in Space settings")
    else:
        print("\n❌ Deployment failed!")
        print("Check the error messages above and try again.")
        print("\nTroubleshooting:")
        print("1. Verify your HF token has write permissions")
        print("2. Check that the space name is available")
        print("3. Verify the model exists and is accessible")
        print("4. Try creating the space manually on HF first")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 