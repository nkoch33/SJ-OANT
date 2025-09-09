#!/usr/bin/env python3
"""
Runner script for MultiDoGO evaluation.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.multidogo_eval import MultiDoGOEvaluator

def main():
    """Run MultiDoGO evaluation."""
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    # Initialize evaluator
    evaluator = MultiDoGOEvaluator(api_key)
    
    # Run evaluation on airline domain data (start with one domain for testing)
    data_path = "data/multidogo/airline.tsv"
    
    print("Starting MultiDoGO evaluation...")
    print(f"Data path: {data_path}")
    print(f"Max dialogues: 50")
    print(f"Max examples: 250")
    
    try:
        results = evaluator.run_evaluation(data_path, max_dialogues=50, max_examples=250)
        print("✅ MultiDoGO evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ MultiDoGO evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
