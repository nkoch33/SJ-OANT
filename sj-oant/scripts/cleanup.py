#!/usr/bin/env python3
"""
Cleanup script for TMM project

This script removes Python cache files, temporary files, and other development artifacts
to keep the workspace clean.

Usage:
    python scripts/cleanup.py
    
Or make it executable and run directly:
    chmod +x scripts/cleanup.py
    ./scripts/cleanup.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def remove_pycache():
    """Remove all __pycache__ directories."""
    print("🧹 Removing __pycache__ directories...")
    
    try:
        # Use find command for efficiency
        result = subprocess.run(
            ["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print("   ✅ Removed __pycache__ directories")
        else:
            print(f"   ⚠️ Warning: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"   ❌ Error removing __pycache__: {e}")


def remove_pyc_files():
    """Remove all .pyc files."""
    print("🧹 Removing .pyc files...")
    
    try:
        result = subprocess.run(
            ["find", ".", "-name", "*.pyc", "-delete"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print("   ✅ Removed .pyc files")
        else:
            print(f"   ⚠️ Warning: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"   ❌ Error removing .pyc files: {e}")


def remove_log_files():
    """Remove log files."""
    print("🧹 Removing log files...")
    
    project_root = Path(__file__).parent.parent
    log_files_removed = 0
    
    try:
        # Remove .log files
        for log_file in project_root.rglob("*.log"):
            log_file.unlink()
            log_files_removed += 1
            
        # Remove logs directories
        for logs_dir in project_root.rglob("logs"):
            if logs_dir.is_dir():
                shutil.rmtree(logs_dir)
                log_files_removed += 1
                
        print(f"   ✅ Removed {log_files_removed} log files/directories")
        
    except Exception as e:
        print(f"   ❌ Error removing log files: {e}")


def remove_temp_files():
    """Remove temporary files and directories."""
    print("🧹 Removing temporary files...")
    
    project_root = Path(__file__).parent.parent
    temp_files_removed = 0
    
    try:
        # Common temporary patterns
        temp_patterns = ["temp", "tmp", ".temp", "*~", "*.swp", "*.swo"]
        
        for pattern in temp_patterns:
            for temp_item in project_root.rglob(pattern):
                if temp_item.is_file():
                    temp_item.unlink()
                    temp_files_removed += 1
                elif temp_item.is_dir():
                    shutil.rmtree(temp_item)
                    temp_files_removed += 1
                    
        print(f"   ✅ Removed {temp_files_removed} temporary files/directories")
        
    except Exception as e:
        print(f"   ❌ Error removing temporary files: {e}")


def remove_ds_store():
    """Remove .DS_Store files (macOS)."""
    print("🧹 Removing .DS_Store files...")
    
    try:
        result = subprocess.run(
            ["find", ".", "-name", ".DS_Store", "-delete"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print("   ✅ Removed .DS_Store files")
        else:
            print(f"   ⚠️ Warning: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"   ❌ Error removing .DS_Store files: {e}")


def main():
    """Run all cleanup operations."""
    print("🚀 Starting TMM project cleanup...")
    print("=" * 50)
    
    # Run all cleanup operations
    remove_pycache()
    remove_pyc_files()
    remove_log_files()
    remove_temp_files()
    remove_ds_store()
    
    print("=" * 50)
    print("✨ Cleanup completed!")
    print("\n💡 Tip: Add 'python scripts/cleanup.py' to your development workflow")
    print("   or create an alias: alias tmm-clean='python scripts/cleanup.py'")


if __name__ == "__main__":
    main()
