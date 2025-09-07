#!/usr/bin/env python3
"""
MultiWOZ TMM System Evaluation Runner

This script evaluates the Truth-Maintained Memory (TMM) system on the MultiWOZ dataset.
It focuses exclusively on TMM system performance, skipping baseline comparisons.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.multiwoz_eval import MultiWOZEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate TMM system on MultiWOZ dataset')
    parser.add_argument('--api-key', required=True, help='Google Gemini API key')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of dialogues to evaluate')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--data-path', default='data/MULTIWOZ2.4/MULTIWOZ2.4/data.json', 
                       help='Path to MultiWOZ data.json file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("🚀 TMM SYSTEM MULTIWOZ EVALUATION")
    logger.info("="*60)
    logger.info(f"Evaluation limit: {args.limit if args.limit else 'All dialogues'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Data path: {args.data_path}")
    
    try:
        # Initialize evaluator
        evaluator = MultiWOZEvaluator(args.api_key, args.data_path)
        
        # Load dialogues
        logger.info("Loading MultiWOZ dialogues...")
        dialogues = evaluator.load_dialogues(limit=args.limit)
        
        if not dialogues:
            logger.error("No dialogues loaded. Check data path and format.")
            return
        
        logger.info(f"Loaded {len(dialogues)} dialogues for evaluation")
        
        # Evaluate TMM system
        logger.info("Evaluating TMM system...")
        results = evaluator.evaluate_tmm_system(dialogues)
        
        # Save results
        output_path = os.path.join(args.output_dir, 'multiwoz_tmm_results.json')
        evaluator.save_results(results, output_path)
        
        # Print summary
        evaluator.print_summary(results)
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
