#!/usr/bin/env python3
"""
Runner script for iterative testing and training pipeline.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.iterative_testing import IterativeTestingPipeline


def main():
    """Main function to run iterative testing."""
    
    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        # Try to read from .env file
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('GOOGLE_API_KEY='):
                        api_key = line.strip().split('=', 1)[1]
                        break
    
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment or .env file")
        sys.exit(1)
    
    print("="*80)
    print("🚀 TMM SYSTEM ITERATIVE TESTING PIPELINE")
    print("="*80)
    print(f"API key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***'}")
    print()
    
    # Create testing pipeline
    pipeline = IterativeTestingPipeline(api_key)
    
    try:
        # Run improvement cycle with small-scale testing
        print("Starting iterative testing cycle...")
        pipeline.run_improvement_cycle(num_dialogues=3, iterations=2)
        
        # Save test history
        pipeline.save_test_history()
        
        print(f"\n✅ Iterative testing completed successfully!")
        print("Test history saved for future reference.")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
