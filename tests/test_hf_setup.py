#!/usr/bin/env python3
"""
Test Hugging Face Setup for Trackio Integration

This script helps verify your Hugging Face token setup and test space name generation.
Run this before using the training scripts to ensure everything is configured correctly.

Authentication:
This script only checks for HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variables.
It does NOT use huggingface-cli login state.

Setup:
  Linux/Mac: export HF_TOKEN=your_token_here
  Windows: set HF_TOKEN=your_token_here
  Or: export HUGGINGFACE_HUB_TOKEN=your_token_here

Get your token from: https://huggingface.co/settings/tokens
"""

import os
from datetime import datetime
from typing import Tuple, Optional
from huggingface_hub import HfApi


def validate_hf_token(token: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a Hugging Face token and return the username.

    Args:
        token (str): The Hugging Face token to validate

    Returns:
        Tuple[bool, Optional[str], Optional[str]]:
            - success: True if token is valid, False otherwise
            - username: The username associated with the token (if valid)
            - error_message: Error message if validation failed
    """
    try:
        # Create API client with token directly
        api = HfApi(token=token)

        # Try to get user info - this will fail if token is invalid
        user_info = api.whoami()

        # Extract username from user info
        username = user_info.get("name", user_info.get("username"))

        if not username:
            return False, None, "Could not retrieve username from token"

        return True, username, None

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, None, "Invalid token - unauthorized access"
        elif "403" in error_msg:
            return False, None, "Token lacks required permissions"
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            return False, None, f"Network error: {error_msg}"
        else:
            return False, None, f"Validation error: {error_msg}"


def get_default_space_name(project_type: str = "voxtral-asr-finetuning") -> str:
    """
    Generate a default space name with username and timestamp.

    Args:
        project_type: Type of project (e.g., "voxtral-asr-finetuning", "voxtral-lora-finetuning")

    Returns:
        str: Default space name in format "username/project-type-timestamp"
    """
    try:
        # Get token from environment variables only
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        if not token:
            return None

        # Validate token and get username
        success, username, error = validate_hf_token(token)
        if success and username:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return f"{username}/{project_type}-{timestamp}"
        else:
            return None

    except Exception as e:
        print(f"Failed to generate default space name: {e}")
        return None


def main():
    print("🔍 Testing Hugging Face Setup for Trackio Integration")
    print("=" * 60)

    # Check for tokens
    print("\n1. Checking for Hugging Face tokens...")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        print(f"✅ Found token in environment: {token[:10]}...")
    else:
        print("❌ No token found in environment variables")
        print("\n❌ No Hugging Face token found!")
        print("Please set the HF_TOKEN environment variable:")
        print("  Linux/Mac: export HF_TOKEN=your_token_here")
        print("  Windows: set HF_TOKEN=your_token_here")
        print("  Or: set HUGGINGFACE_HUB_TOKEN=your_token_here")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        return

    # Validate token
    print("\n2. Validating token...")
    success, username, error = validate_hf_token(token)

    if success:
        print(f"✅ Token is valid! Username: {username}")
    else:
        print(f"❌ Token validation failed: {error}")
        return

    # Generate space names
    print("\n3. Generating default space names...")

    full_finetune_space = get_default_space_name("voxtral-asr-finetuning")
    lora_finetune_space = get_default_space_name("voxtral-lora-finetuning")

    print(f"📁 Full fine-tuning space: {full_finetune_space}")
    print(f"📁 LoRA fine-tuning space: {lora_finetune_space}")

    print("\n✅ Setup complete! You can now run training scripts.")
    print("   They will automatically use the generated space names.")
    print("\n💡 To override the auto-generated names, use --trackio-space yourname/custom-space")


if __name__ == "__main__":
    main()
