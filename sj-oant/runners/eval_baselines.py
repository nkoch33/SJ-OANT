"""
runners.eval_baselines - FictionalQA Baseline Evaluation Runner

This module runs comprehensive evaluations comparing the TMM system against
baseline approaches on the FictionalQA dataset.

Baseline systems evaluated:
1. DirectLLM - No memory, context-only responses  
2. LongContext - Maintain full conversation history
3. SimpleRAG - Basic retrieval-augmented generation
4. BasicMemory - Store everything, no filtering

The evaluation measures accuracy on fictional reasoning tasks and
demonstrates the value of truth-maintained memory approaches.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_google_genai import ChatGoogleGenerativeAI
from evaluation.fictionalqa_eval import FictionalQAEvaluator, EvaluationResult
from baselines.simple_systems import create_baseline_systems
from tmm_pipeline import TMMPipeline

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_llm_with_api_key(api_key: str, model_name: str = "gemini-pro") -> ChatGoogleGenerativeAI:
    """Load LLM with API key."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0  # Deterministic for evaluation
    )

def run_baseline_evaluation(api_key: str, 
                          limit: int = 50,
                          output_dir: str = "results") -> List[EvaluationResult]:
    """
    Run complete baseline evaluation on FictionalQA.
    
    Args:
        api_key: Google API key for LLM access
        limit: Number of examples to evaluate
        output_dir: Directory to save results
        
    Returns:
        List of evaluation results for all systems
    """
    logger.info("Starting FictionalQA baseline evaluation")
    
    # Initialize evaluator and load data
    evaluator = FictionalQAEvaluator()
    examples = evaluator.load_dataset(split="validation")
    if limit:
        examples = examples[:limit]
    
    logger.info(f"Evaluating on {len(examples)} FictionalQA examples")
    
    # Initialize LLM
    llm = load_llm_with_api_key(api_key)
    
    # Create baseline systems
    baseline_systems = create_baseline_systems(llm)
    logger.info(f"Created {len(baseline_systems)} baseline systems")
    
    # Evaluate each baseline system
    results = []
    
    for system_name, system in baseline_systems.items():
        logger.info(f"Evaluating {system_name}...")
        try:
            result = evaluator.evaluate_baseline_system(
                baseline_system=system,
                examples=examples,
                system_name=system_name,
                limit=limit
            )
            results.append(result)
            logger.info(f"{system_name}: {result.accuracy:.2%} accuracy")
        except Exception as e:
            logger.error(f"Failed to evaluate {system_name}: {e}")
    
    # Evaluate TMM system
    logger.info("Evaluating TMM system...")
    try:
        tmm_pipeline = TMMPipeline(llm=llm)
        tmm_result = evaluator.evaluate_tmm_system(
            tmm_pipeline=tmm_pipeline,
            examples=examples,
            limit=limit
        )
        results.append(tmm_result)
        logger.info(f"TMM Pipeline: {tmm_result.accuracy:.2%} accuracy")
    except Exception as e:
        logger.error(f"Failed to evaluate TMM system: {e}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / "fictionalqa_baseline_results.json"
    evaluator.save_results(results, str(results_file))
    
    # Print comparison
    comparison = evaluator.compare_systems(results)
    print("\n" + "="*60)
    print("FICTIONALQA EVALUATION RESULTS")
    print("="*60)
    
    for system_name, metrics in comparison["summary"].items():
        print(f"{system_name:15}: {metrics['accuracy']} accuracy, {metrics['avg_response_time']} avg time")
    
    best_system = comparison["best_accuracy"]
    print(f"\nBest performing system: {best_system.system_name} ({best_system.accuracy:.2%})")
    print(f"Results saved to: {results_file}")
    
    return results

def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Run FictionalQA baseline evaluation")
    parser.add_argument("--api-key", required=True, help="Google API key for LLM access")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to evaluate")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--model", default="gemini-pro", help="LLM model to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        results = run_baseline_evaluation(
            api_key=args.api_key,
            limit=args.limit,
            output_dir=args.output_dir
        )
        
        logger.info("Baseline evaluation completed successfully!")
        print(f"\nEvaluated {len(results)} systems on FictionalQA dataset")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
