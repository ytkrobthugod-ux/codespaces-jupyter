#!/usr/bin/env python3
"""
🚀 Roboto SAI Quick Start Script
Automatically loads environment variables and starts the Flask app
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the Roboto SAI application"""
    print("🚀 Starting Roboto SAI...")

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please copy .env.example to .env and configure your environment variables.")
        sys.exit(1)

    # Check if required environment variables are set
    required_vars = ['SESSION_SECRET']
    missing_vars = []

    # Load .env file manually to check
    with open(env_file, 'r') as f:
        env_content = f.read()

    for var in required_vars:
        if f"{var}=" not in env_content:
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file.")
        sys.exit(1)

    print("✅ Environment configuration verified")

    # Start the Flask app
    try:
        print("🌐 Starting Flask development server...")
        print("📍 App will be available at: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)

        # Use subprocess to run the app (this allows the script to be used in different contexts)
        subprocess.run([sys.executable, "main.py"], check=True)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()