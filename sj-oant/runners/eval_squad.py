"""
runners.eval_squad - SQuAD Dataset Evaluation Runner

This module runs comprehensive evaluations of the TMM system and baselines
on the SQuAD dataset with proper train/validation/test methodology.

Features:
- Proper data splits and sample sizes
- Comprehensive baseline comparisons  
- Detailed performance analysis
- Results storage and visualization
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.squad_eval import SQuADEvaluator
from baselines.simple_systems import create_baseline_systems
from tmm_pipeline import create_tmm_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_llm_with_api_key(api_key: str, model_name: str = "gemini-1.5-flash") -> ChatGoogleGenerativeAI:
    """Load LLM with API key."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0  # Deterministic for evaluation
    )

def run_squad_evaluation(api_key: str, 
                        limit: int = 50,
                        split: str = "validation",
                        output_dir: str = "results") -> list:
    """
    Run comprehensive SQuAD evaluation comparing TMM against baselines.
    
    Args:
        api_key: Google API key for LLM access
        limit: Number of examples to evaluate per system
        split: Dataset split to use ("train" or "validation")
        output_dir: Directory to save results
        
    Returns:
        List of evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting SQuAD evaluation")
    
    # Initialize evaluator and load data
    evaluator = SQuADEvaluator()
    examples = evaluator.load_dataset(split=split)
    if limit:
        examples = examples[:limit]
    
    logger.info(f"Evaluating on {len(examples)} SQuAD examples from {split} split")
    
    # Initialize LLM for baselines
    llm = load_llm_with_api_key(api_key)
    
    # Create baseline systems
    baseline_systems = create_baseline_systems(llm)
    logger.info(f"Created {len(baseline_systems)} baseline systems")
    
    # Evaluate all systems
    results = []
    
    # Evaluate baselines first
    for system_name, system in baseline_systems.items():
        logger.info(f"Evaluating {system_name}...")
        result = evaluator.evaluate_baseline_system(
            baseline_system=system,
            examples=examples,
            system_name=system_name,
            limit=limit
        )
        results.append(result)
        logger.info(f"{system_name}: {result.accuracy:.2%} accuracy")
    
    # Evaluate TMM system
    logger.info("Evaluating TMM system...")
    tmm_pipeline = create_tmm_pipeline(google_api_key=api_key)
    tmm_result = evaluator.evaluate_tmm_system(
        tmm_pipeline=tmm_pipeline,
        examples=examples,
        limit=limit
    )
    results.append(tmm_result)
    logger.info(f"TMM Pipeline: {tmm_result.accuracy:.2%} accuracy")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/squad_evaluation_results.json"
    evaluator.save_results(results, results_file)
    
    # Print summary
    comparison = evaluator.compare_systems(results)
    print("\n" + "="*60)
    print("SQUAD EVALUATION RESULTS")
    print("="*60)
    
    for system_name, metrics in comparison["summary"].items():
        print(f"{system_name:15}: {metrics['accuracy']} accuracy, {metrics['avg_response_time']} avg time")
    
    best_system = comparison["best_accuracy"]
    print(f"\nBest performing system: {best_system['system_name']} ({best_system['accuracy']:.2%})")
    print(f"Results saved to: {results_file}")
    
    return results

def get_dataset_info(api_key: str):
    """Get information about SQuAD dataset splits."""
    evaluator = SQuADEvaluator()
    splits = evaluator.get_data_splits()
    
    print("\n" + "="*60)
    print("SQUAD DATASET INFORMATION")
    print("="*60)
    
    for split, count in splits.items():
        print(f"{split:15}: {count:,} examples")
    
    print(f"\nRecommended evaluation sizes:")
    print(f"Development/debugging: 10-20 examples")
    print(f"Standard evaluation: 100-500 examples") 
    print(f"Full evaluation: 1000+ examples")

def main():
    parser = argparse.ArgumentParser(description="Run SQuAD evaluation")
    parser.add_argument("--api-key", required=True, help="Google API key for LLM access")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to evaluate")
    parser.add_argument("--split", default="validation", choices=["train", "validation"], 
                       help="Dataset split to use")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--model", default="gemini-1.5-flash", help="LLM model to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--info", action="store_true", help="Show dataset information only")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        if args.info:
            get_dataset_info(args.api_key)
        else:
            results = run_squad_evaluation(
                api_key=args.api_key,
                limit=args.limit,
                split=args.split,
                output_dir=args.output_dir
            )
            print(f"\n✅ SQuAD evaluation completed successfully!")
            print(f"Evaluated {len(results)} systems on SQuAD dataset")
            
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
