#!/usr/bin/env python3
"""
Setup script to help configure the Google API key for testing.
"""

import os
import sys

def main():
    """Help user set up the API key."""
    print("🔑 Google API Key Setup for TMM Testing")
    print("=" * 50)
    print("")
    
    # Check if API key is already set
    current_key = os.getenv("GOOGLE_API_KEY")
    if current_key:
        print(f"✅ API key is already set: {current_key[:10]}...")
        print("")
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            print("Using existing API key.")
            return
    
    print("To get a Google API key:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated API key")
    print("")
    
    api_key = input("Enter your Google API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Set the environment variable for current session
    os.environ["GOOGLE_API_KEY"] = api_key
    
    print("")
    print("✅ API key set for current session!")
    print("")
    print("To make it permanent, add this to your shell profile:")
    print(f"export GOOGLE_API_KEY='{api_key}'")
    print("")
    print("Or run this command:")
    print(f"export GOOGLE_API_KEY='{api_key}'")
    print("")
    print("🚀 You can now run the comprehensive evaluation:")
    print("python testing/run_comprehensive_evaluation.py")

if __name__ == "__main__":
    main()
