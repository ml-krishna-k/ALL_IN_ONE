#!/usr/bin/env python3
"""
Startup script for local development
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("📋 Please create a .env file with the following variables:")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        print("   PORT=8000")
        print("   PYTHONPATH=.")
        print("   LOG_LEVEL=INFO")
        print("\n💡 You can copy .env.example to .env and fill in your values.")
        return False
    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import openai
        import pydantic
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install dependencies with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the development server"""
    port = os.environ.get("PORT", "8000")
    print(f"🚀 Starting EverydayAI Backend API on port {port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    print(f"🔍 Health Check: http://localhost:{port}/health")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", port,
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main startup function"""
    print("EverydayAI Backend API - Development Server")
    print("=" * 50)
    
    # Check prerequisites
    if not check_env_file():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
