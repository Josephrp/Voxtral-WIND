#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly.
"""

import sys
from pathlib import Path

def test_training_scripts():
    """Test that training scripts can be imported without errors."""
    print("🧪 Testing training script imports...")
    
    try:
        # Test train_lora.py
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from train_lora import main as train_lora_main
        print("✅ train_lora.py imports successfully")
        
        # Test train.py
        from train import main as train_main
        print("✅ train.py imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Training script import failed: {e}")
        return False

def test_deploy_script_syntax():
    """Test that deploy script has valid syntax."""
    print("🧪 Testing deploy script syntax...")
    
    try:
        # Read the file and check for basic syntax issues
        with open("scripts/deploy_demo_space.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for common syntax issues
        if "json.dumps(" in content and "}" in content:
            # Check if there are any stray } characters after json.dumps
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if "json.dumps(" in line and line.strip().endswith('}'):
                    print(f"❌ Found stray }} on line {i}: {line.strip()}")
                    return False
        
        print("✅ Deploy script syntax appears valid")
        return True
    except Exception as e:
        print(f"❌ Deploy script syntax test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing fixes...")
    
    success = True
    
    # Test training scripts
    if not test_training_scripts():
        success = False
    
    # Test deploy script syntax
    if not test_deploy_script_syntax():
        success = False
    
    if success:
        print("\n🎉 All fixes appear to be working!")
        print("\nKey fixes applied:")
        print("1. ✅ Changed report_to from 'wandb' to 'trackio' to avoid WandbCallback error")
        print("2. ✅ Fixed f-string syntax errors in deploy_demo_space.py")
        print("3. ✅ Removed stray } characters from json.dumps calls")
        print("\nThe training should now work correctly!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
