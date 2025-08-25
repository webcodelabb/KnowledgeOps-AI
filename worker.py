#!/usr/bin/env python3
"""
Celery worker startup script for KnowledgeOps AI
"""
import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  Warning: .env file not found")
        print("   Copy env.example to .env and configure your settings")
        print("   cp env.example .env")
        return False
    return True

def main():
    """Start the Celery worker"""
    print("🚀 Starting KnowledgeOps AI Celery Worker...")
    
    # Check environment
    check_environment()
    
    # Set default configuration
    concurrency = int(os.getenv("CELERY_CONCURRENCY", "2"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    queue = os.getenv("CELERY_QUEUE", "document_processing")
    
    print(f"🔧 Configuration:")
    print(f"   Concurrency: {concurrency}")
    print(f"   Log Level: {log_level}")
    print(f"   Queue: {queue}")
    print(f"   Tasks: app.tasks.document_processing")
    print("-" * 50)
    
    try:
        # Import and start Celery worker
        from app.celery_app import celery_app
        
        # Start the worker
        celery_app.worker_main([
            "worker",
            "--loglevel", log_level,
            "--concurrency", str(concurrency),
            "--queues", queue,
            "--hostname", f"worker@%h",
            "--without-gossip",
            "--without-mingle",
            "--without-heartbeat"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Celery worker...")
    except Exception as e:
        print(f"❌ Error starting Celery worker: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

