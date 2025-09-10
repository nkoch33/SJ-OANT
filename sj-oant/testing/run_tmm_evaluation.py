#!/usr/bin/env python3
"""
Runner script for TMM evaluation.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from tmm_evaluation import TMMEvaluator

def main():
    """Run TMM evaluation."""
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        print("\nTo get an API key:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set it as an environment variable")
        return
    
    # Initialize evaluator
    evaluator = TMMEvaluator(api_key)
    
    print("🚀 Starting TMM System Evaluation")
    print("="*60)
    print("📊 Testing TMM Model on 4 Dialogue Benchmarks")
    print("🎯 25 examples each from:")
    print("   • MultiWOZ - Multi-turn dialogue")
    print("   • SGD - Schema-guided dialogue")
    print("   • Taskmaster - Human-human conversations")
    print("   • MultiDoGO - Multi-domain goal-oriented")
    print("")
    print("⏱️  This will take 10-15 minutes...")
    print("="*60)
    
    try:
        results = evaluator.run_evaluation(num_examples=25)
        
        # Save results
        evaluator.save_results()
        
        # Print summary
        evaluator.print_summary()
        
        print("\n✅ TMM evaluation completed successfully!")
        print("📁 Results saved to: testing/results/tmm_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ TMM evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
