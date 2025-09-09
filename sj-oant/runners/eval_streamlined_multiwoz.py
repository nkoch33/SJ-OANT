#!/usr/bin/env python3
"""
Runner script for streamlined MultiWOZ evaluation.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.multiwoz_streamlined_eval import StreamlinedMultiWOZEvaluator


def main():
    """Main function to run streamlined MultiWOZ evaluation."""
    
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
    
    # Set up paths
    data_path = Path(__file__).parent.parent / 'data' / 'MULTIWOZ2.4' / 'MULTIWOZ2.4' / 'data.json'
    output_path = Path(__file__).parent.parent / 'results' / 'streamlined_multiwoz_results.json'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🚀 TMM SYSTEM STREAMLINED MULTIWOZ EVALUATION")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"API key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***'}")
    print()
    
    # Create evaluator and run evaluation
    evaluator = StreamlinedMultiWOZEvaluator(api_key)
    
    try:
        results = evaluator.run_evaluation(
            data_path=str(data_path),
            limit=5,  # Start with 5 dialogues for testing
            output_path=str(output_path)
        )
        
        print(f"\n✅ Streamlined MultiWOZ evaluation completed successfully!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
