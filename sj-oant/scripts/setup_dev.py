#!/usr/bin/env python3
"""
Development environment setup script for TMM project

This script helps set up the development environment with proper Python
cache prevention and other optimizations.

Usage:
    python scripts/setup_dev.py
"""

import os
import sys
from pathlib import Path


def setup_environment():
    """Set up development environment variables."""
    print("🔧 Setting up development environment...")
    
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_example = project_root / "env.example"
    
    if not env_file.exists() and env_example.exists():
        print("   📋 Creating .env file from template...")
        try:
            env_file.write_text(env_example.read_text())
            print("   ✅ Created .env file")
            print("   💡 Edit .env file to customize your settings")
        except Exception as e:
            print(f"   ❌ Error creating .env file: {e}")
    else:
        print("   ℹ️ .env file already exists or template not found")


def check_python_cache_settings():
    """Check if Python cache is disabled."""
    print("🔍 Checking Python cache settings...")
    
    cache_disabled = os.environ.get('PYTHONDONTWRITEBYTECODE', '0') == '1'
    
    if cache_disabled:
        print("   ✅ Python cache generation is disabled")
    else:
        print("   ⚠️ Python cache generation is enabled")
        print("   💡 To disable, run: export PYTHONDONTWRITEBYTECODE=1")
        print("   💡 Or add 'PYTHONDONTWRITEBYTECODE=1' to your .env file")


def create_alias_suggestions():
    """Suggest useful shell aliases."""
    print("💡 Suggested shell aliases for your ~/.bashrc or ~/.zshrc:")
    print()
    print("   # TMM Project aliases")
    print("   alias tmm-clean='make clean'")
    print("   alias tmm-cache='make clean-cache'") 
    print("   alias tmm-demo='make demo'")
    print("   alias tmm-nocache='export PYTHONDONTWRITEBYTECODE=1'")
    print()


def main():
    """Run development environment setup."""
    print("🚀 TMM Development Environment Setup")
    print("=" * 40)
    
    setup_environment()
    check_python_cache_settings()
    
    print("=" * 40)
    print("✨ Setup completed!")
    print()
    
    create_alias_suggestions()
    
    print("🎯 Next steps:")
    print("   1. Review and edit .env file if needed")
    print("   2. Run 'make clean' to clean any existing cache")
    print("   3. Set PYTHONDONTWRITEBYTECODE=1 to prevent future cache")
    print("   4. Run 'make demo' to test the architecture")


if __name__ == "__main__":
    main()
