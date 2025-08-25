#!/usr/bin/env python3
"""
Test script to verify KnowledgeOps AI database setup
"""
import os
import sys
import subprocess
from pathlib import Path

def test_alembic_setup():
    """Test that Alembic is properly configured"""
    print("🔍 Testing Alembic Configuration...")
    
    required_files = [
        'alembic.ini',
        'alembic/env.py',
        'alembic/script.py.mako',
        'alembic/versions/0001_initial_schema.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing Alembic files: {missing_files}")
        return False
    else:
        print("✅ All Alembic files exist")
        return True

def test_migration_file():
    """Test the migration file content"""
    print("\n📋 Testing Migration File...")
    
    migration_file = 'alembic/versions/0001_initial_schema.py'
    
    try:
        with open(migration_file, 'r') as f:
            content = f.read()
        
        # Check for required components
        checks = [
            ("pgvector extension", "CREATE EXTENSION IF NOT EXISTS vector"),
            ("documents table", "create_table('documents'"),
            ("chunks table", "create_table('chunks'"),
            ("conversations table", "create_table('conversations'"),
            ("vector column", "postgresql.VECTOR(1536)"),
            ("JSONB column", "postgresql.JSONB"),
            ("indexes", "create_index"),
            ("downgrade function", "def downgrade()")
        ]
        
        passed = 0
        for check_name, check_content in checks:
            if check_content in content:
                print(f"   ✅ {check_name}")
                passed += 1
            else:
                print(f"   ❌ {check_name}")
        
        if passed == len(checks):
            print("✅ Migration file contains all required components")
            return True
        else:
            print(f"❌ Migration file missing {len(checks) - passed} components")
            return False
            
    except Exception as e:
        print(f"❌ Error reading migration file: {e}")
        return False

def test_sqlalchemy_models():
    """Test SQLAlchemy models"""
    print("\n🗄️ Testing SQLAlchemy Models...")
    
    try:
        # Test that models can be imported
        import sys
        sys.path.append('.')
        
        from app.models_db import Document, Chunk, Conversation
        
        # Test model attributes
        checks = [
            ("Document model", Document.__tablename__ == "documents"),
            ("Document org_id", hasattr(Document, 'org_id')),
            ("Document source", hasattr(Document, 'source')),
            ("Chunk model", Chunk.__tablename__ == "chunks"),
            ("Chunk embedding", hasattr(Chunk, 'embedding')),
            ("Chunk metadata", hasattr(Chunk, 'metadata')),
            ("Conversation model", Conversation.__tablename__ == "conversations"),
            ("Conversation user_id", hasattr(Conversation, 'user_id'))
        ]
        
        passed = 0
        for check_name, check_result in checks:
            if check_result:
                print(f"   ✅ {check_name}")
                passed += 1
            else:
                print(f"   ❌ {check_name}")
        
        if passed == len(checks):
            print("✅ All SQLAlchemy models are properly defined")
            return True
        else:
            print(f"❌ {len(checks) - passed} model checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing SQLAlchemy models: {e}")
        return False

def test_database_utils():
    """Test database utility functions"""
    print("\n🔧 Testing Database Utilities...")
    
    try:
        from app.db_utils import (
            create_document, get_document_by_id, get_documents_by_org,
            create_chunk, get_chunks_by_document, get_document_stats
        )
        
        functions = [
            "create_document",
            "get_document_by_id", 
            "get_documents_by_org",
            "create_chunk",
            "get_chunks_by_document",
            "get_document_stats"
        ]
        
        for func_name in functions:
            print(f"   ✅ {func_name}")
        
        print("✅ All database utility functions are available")
        return True
        
    except Exception as e:
        print(f"❌ Error testing database utilities: {e}")
        return False

def test_alembic_commands():
    """Test Alembic commands"""
    print("\n🚀 Testing Alembic Commands...")
    
    commands = [
        ("alembic current", "Check current migration status"),
        ("alembic history", "Check migration history"),
        ("alembic show 0001", "Show migration details")
    ]
    
    passed = 0
    for cmd, description in commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✅ {description}")
                passed += 1
            else:
                print(f"   ❌ {description} (exit code: {result.returncode})")
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {description} (timeout)")
        except Exception as e:
            print(f"   ❌ {description} (error: {e})")
    
    if passed >= 2:  # At least 2 commands should work
        print("✅ Alembic commands are working")
        return True
    else:
        print("❌ Some Alembic commands failed")
        return False

def main():
    """Run all database tests"""
    print("🚀 Testing KnowledgeOps AI Database Setup")
    print("=" * 50)
    
    tests = [
        ("Alembic Setup", test_alembic_setup),
        ("Migration File", test_migration_file),
        ("SQLAlchemy Models", test_sqlalchemy_models),
        ("Database Utilities", test_database_utils),
        ("Alembic Commands", test_alembic_commands)
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
        print("🎉 All database tests passed! Database is ready for migration.")
        print("\nNext steps:")
        print("1. Set up your database connection in .env")
        print("2. Run migrations: python migrate.py")
        print("3. Start the application: python run.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
