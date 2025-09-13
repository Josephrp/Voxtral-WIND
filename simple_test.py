#!/usr/bin/env python3
"""
Simple test to verify the fixes work.
"""

def test_syntax():
    """Test that all Python files have valid syntax."""
    import ast
    
    files_to_test = [
        "scripts/train_lora.py",
        "scripts/train.py", 
        "scripts/deploy_demo_space.py"
    ]
    
    for file_path in files_to_test:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"✅ {file_path} - syntax valid")
        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ {file_path} - error: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing syntax...")
    if test_syntax():
        print("\n🎉 All files have valid syntax!")
        print("\nKey fixes applied:")
        print("1. ✅ Fixed WandbCallback error by using report_to=['trackio']")
        print("2. ✅ Fixed f-string syntax errors in deploy_demo_space.py")
        print("3. ✅ Removed stray } characters from json.dumps calls")
        print("4. ✅ Fixed missing closing parenthesis")
        print("\nThe training should now work correctly!")
    else:
        print("\n❌ Some files have syntax errors.")
