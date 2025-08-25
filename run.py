#!/usr/bin/env python3
"""
Startup script for KnowledgeOps AI
"""
import os
import sys
import uvicorn
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
    """Start the KnowledgeOps AI application"""
    print("🚀 Starting KnowledgeOps AI...")
    
    # Check environment
    check_environment()
    
    # Set default configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"📍 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"📊 Metrics: http://{host}:{port}/metrics")
    print(f"🔍 Health: http://{host}:{port}/health")
    print(f"🔄 Reload: {reload}")
    print(f"📝 Log Level: {log_level}")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down KnowledgeOps AI...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
