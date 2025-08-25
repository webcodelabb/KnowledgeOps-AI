#!/usr/bin/env python3
"""
Test script to verify KnowledgeOps AI Celery worker setup
"""
import os
import sys
import subprocess
from pathlib import Path

def test_celery_configuration():
    """Test Celery configuration"""
    print("🔧 Testing Celery Configuration...")
    
    required_files = [
        'app/celery_app.py',
        'app/tasks/__init__.py',
        'app/tasks/document_processing.py',
        'worker.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing Celery files: {missing_files}")
        return False
    else:
        print("✅ All Celery files exist")
        return True

def test_celery_imports():
    """Test that Celery components can be imported"""
    print("\n📦 Testing Celery Imports...")
    
    try:
        # Test Celery app import
        from app.celery_app import celery_app
        print("   ✅ Celery app imported")
        
        # Test task imports
        from app.tasks.document_processing import process_document, process_document_batch
        print("   ✅ Document processing tasks imported")
        
        # Test LangChain imports
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings.openai import OpenAIEmbeddings
        print("   ✅ LangChain components imported")
        
        print("✅ All Celery imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_celery_app_config():
    """Test Celery application configuration"""
    print("\n⚙️ Testing Celery App Configuration...")
    
    try:
        from app.celery_app import celery_app
        
        # Check basic configuration
        checks = [
            ("Broker URL", celery_app.conf.broker_url is not None),
            ("Result Backend", celery_app.conf.result_backend is not None),
            ("Task Serializer", celery_app.conf.task_serializer == "json"),
            ("Result Serializer", celery_app.conf.result_serializer == "json"),
            ("Task Time Limit", celery_app.conf.task_time_limit == 30 * 60),
            ("Task Routes", "app.tasks.document_processing" in celery_app.conf.task_routes),
        ]
        
        passed = 0
        for check_name, check_result in checks:
            if check_result:
                print(f"   ✅ {check_name}")
                passed += 1
            else:
                print(f"   ❌ {check_name}")
        
        if passed == len(checks):
            print("✅ Celery app configuration is correct")
            return True
        else:
            print(f"❌ {len(checks) - passed} configuration checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_task_definitions():
    """Test that tasks are properly defined"""
    print("\n🎯 Testing Task Definitions...")
    
    try:
        from app.tasks.document_processing import process_document, process_document_batch
        
        # Check task attributes
        checks = [
            ("process_document task", hasattr(process_document, 'delay')),
            ("process_document_batch task", hasattr(process_document_batch, 'delay')),
            ("process_document name", process_document.name == "app.tasks.document_processing.process_document"),
            ("process_document_batch name", process_document_batch.name == "app.tasks.document_processing.process_document_batch"),
        ]
        
        passed = 0
        for check_name, check_result in checks:
            if check_result:
                print(f"   ✅ {check_name}")
                passed += 1
            else:
                print(f"   ❌ {check_name}")
        
        if passed == len(checks):
            print("✅ Task definitions are correct")
            return True
        else:
            print(f"❌ {len(checks) - passed} task checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Task definition error: {e}")
        return False

def test_worker_script():
    """Test the worker startup script"""
    print("\n🔧 Testing Worker Script...")
    
    try:
        # Test that the script can be imported
        import worker
        print("   ✅ Worker script can be imported")
        
        # Test that main function exists
        if hasattr(worker, 'main'):
            print("   ✅ Worker main function exists")
            return True
        else:
            print("   ❌ Worker main function missing")
            return False
            
    except Exception as e:
        print(f"❌ Worker script error: {e}")
        return False

def test_api_integration():
    """Test API integration with Celery tasks"""
    print("\n🌐 Testing API Integration...")
    
    try:
        # Test that the API can import Celery tasks
        from app.main import app
        print("   ✅ FastAPI app can import Celery tasks")
        
        # Check that the ingest endpoint is available
        routes = [route.path for route in app.routes]
        required_routes = [
            "/ingest",
            "/ingest/batch", 
            "/tasks/{task_id}"
        ]
        
        missing_routes = []
        for route in required_routes:
            if route not in routes:
                missing_routes.append(route)
        
        if not missing_routes:
            print("   ✅ All required API routes are available")
            return True
        else:
            print(f"   ❌ Missing API routes: {missing_routes}")
            return False
            
    except Exception as e:
        print(f"❌ API integration error: {e}")
        return False

def main():
    """Run all Celery tests"""
    print("🚀 Testing KnowledgeOps AI Celery Worker Setup")
    print("=" * 50)
    
    tests = [
        ("Celery Configuration", test_celery_configuration),
        ("Celery Imports", test_celery_imports),
        ("Celery App Config", test_celery_app_config),
        ("Task Definitions", test_task_definitions),
        ("Worker Script", test_worker_script),
        ("API Integration", test_api_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Celery tests passed! Document processing is ready.")
        print("\n🚀 Next steps:")
        print("1. Start Redis: redis-server")
        print("2. Start Celery worker: python worker.py")
        print("3. Start FastAPI app: python run.py")
        print("4. Test document ingestion via API")
        print("\n📚 Example usage:")
        print("   curl -X POST http://localhost:8000/ingest \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"file_url\": \"https://example.com/doc.pdf\"}'")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
