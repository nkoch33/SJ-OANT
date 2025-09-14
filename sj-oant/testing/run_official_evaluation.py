#!/usr/bin/env python3
"""
Runner script for official benchmark evaluation
Uses research-validated evaluation frameworks
"""

import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from official_evaluation import OfficialBenchmarkEvaluator

def main():
    """Run official benchmark evaluation."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting official benchmark evaluation")
    
    try:
        # Initialize evaluator
        evaluator = OfficialBenchmarkEvaluator()
        
        # Run evaluation with 25 samples per benchmark
        results = evaluator.run_evaluation(
            num_samples=25,
            output_path="testing/results/official_evaluation_results.json"
        )
        
        logger.info("Official evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
