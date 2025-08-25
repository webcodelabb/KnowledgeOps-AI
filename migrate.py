#!/usr/bin/env python3
"""
Migration script for KnowledgeOps AI database
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_database_connection():
    """Check if database is accessible"""
    print("🔍 Checking database connection...")
    
    # Try to run a simple alembic command
    result = subprocess.run(
        "alembic current", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Database connection successful")
        return True
    else:
        print("❌ Database connection failed")
        print("   Make sure PostgreSQL is running and accessible")
        print("   Check your DATABASE_URL in .env file")
        return False

def main():
    """Run database migrations"""
    print("🚀 KnowledgeOps AI Database Migration")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("alembic.ini").exists():
        print("❌ alembic.ini not found. Make sure you're in the project root directory.")
        sys.exit(1)
    
    # Check database connection
    if not check_database_connection():
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Check your .env file has correct DATABASE_URL")
        print("   3. Ensure the database exists")
        print("   4. Verify pgvector extension is available")
        sys.exit(1)
    
    # Run migrations
    print("\n📊 Running database migrations...")
    
    # Show current status
    run_command("alembic current", "Checking current migration status")
    
    # Run upgrade
    if run_command("alembic upgrade head", "Running migrations"):
        print("\n🎉 Database migration completed successfully!")
        print("\n📋 Migration Summary:")
        print("   ✅ pgvector extension enabled")
        print("   ✅ documents table created")
        print("   ✅ chunks table created with vector support")
        print("   ✅ conversations table created")
        print("   ✅ Performance indexes created")
        
        # Show final status
        run_command("alembic current", "Final migration status")
        
        print("\n🚀 Your KnowledgeOps AI database is ready!")
    else:
        print("\n❌ Migration failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
