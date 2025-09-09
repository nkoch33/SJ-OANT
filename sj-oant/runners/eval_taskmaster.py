#!/usr/bin/env python3
"""
Runner script for Taskmaster evaluation.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.taskmaster_eval import TaskmasterEvaluator

def main():
    """Run Taskmaster evaluation."""
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    # Initialize evaluator
    evaluator = TaskmasterEvaluator(api_key)
    
    # Run evaluation on TM-2 restaurant data (start with one file for testing)
    data_path = "data/taskmaster/restaurant-search.json"
    
    print("Starting Taskmaster evaluation...")
    print(f"Data path: {data_path}")
    print(f"Max dialogues: 50")
    print(f"Max examples: 250")
    
    try:
        results = evaluator.run_evaluation(data_path, max_dialogues=50, max_examples=250)
        print("✅ Taskmaster evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Taskmaster evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
