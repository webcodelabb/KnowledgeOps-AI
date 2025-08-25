#!/usr/bin/env python3
"""
Test script to verify KnowledgeOps AI setup
"""
import os
import sys

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        'app/__init__.py',
        'app/main.py',
        'app/config.py',
        'app/database.py',
        'app/models.py',
        'app/logging.py',
        'app/metrics.py',
        'requirements.txt',
        'README.md',
        'Dockerfile',
        'docker-compose.yml',
        'env.example'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_imports():
    """Test that Python files can be parsed (without dependencies)"""
    python_files = [
        'app/__init__.py',
        'app/main.py',
        'app/config.py',
        'app/database.py',
        'app/models.py',
        'app/logging.py',
        'app/metrics.py'
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ {file_path} - Syntax OK")
        except SyntaxError as e:
            print(f"❌ {file_path} - Syntax Error: {e}")
            return False
        except Exception as e:
            print(f"❌ {file_path} - Error: {e}")
            return False
    
    return True

def test_requirements():
    """Test requirements.txt format"""
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 0:
            print("✅ requirements.txt has dependencies")
            return True
        else:
            print("❌ requirements.txt is empty")
            return False
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing KnowledgeOps AI Setup")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_imports),
        ("Requirements", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KnowledgeOps AI is ready to run.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up environment: cp env.example .env")
        print("3. Run the app: python -m app.main")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
