#!/usr/bin/env python3
"""
Script to securely setup API key in .env file.
"""
import os
from pathlib import Path

def setup_api_key(api_key=None):
    """Setup API key in .env file."""
    print("🔐 SETTING UP API KEY...")
    
    # Get API key from command line argument
    import sys
    if api_key is None:
        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            api_key = input("Enter your Google API key: ").strip()
    
    env_content = f"""# TMM System Environment Variables
# DO NOT COMMIT THIS FILE TO GITHUB

# Google Gemini API Key
GOOGLE_API_KEY={api_key}

# Evaluation Settings
MAX_EXAMPLES=100
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env")
    env_file.write_text(env_content)
    
    print("✅ .env file created with API key")
    print("🔒 This file is protected by .gitignore")
    print("⚠️  NEVER commit .env to version control!")

if __name__ == "__main__":
    setup_api_key()
