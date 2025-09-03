#!/usr/bin/env python3
"""
Enhanced SQuAD 2.0 Evaluation Runner

This script runs comprehensive evaluation using SQuAD 2.0 dataset with:
- Unanswerable question support
- Enhanced baseline systems
- Memory-specific metrics
- Detailed performance analysis

Usage:
    python runners/eval_squad2.py --api-key YOUR_API_KEY --limit 100
    python runners/eval_squad2.py --api-key YOUR_API_KEY --full-eval
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.squad_eval import SQuADEvaluator
from baselines.simple_systems import DirectLLMBaseline, LongContextBaseline, SimpleRAGBaseline, BasicMemoryBaseline
from tmm_pipeline import create_tmm_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_baseline_systems(api_key: str):
    """Create all baseline systems with the same LLM."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    
    systems = {
        "DirectLLM": DirectLLMBaseline(llm),
        "LongContext": LongContextBaseline(llm, max_context_length=8000),
        "SimpleRAG": SimpleRAGBaseline(llm, max_retrievals=3),
        "BasicMemory": BasicMemoryBaseline(llm)
    }
    
    logger.info(f"Created {len(systems)} baseline systems")
    return systems

def main():
    parser = argparse.ArgumentParser(description="Run SQuAD 2.0 evaluation")
    parser.add_argument("--api-key", required=True, help="Google API key")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to evaluate (default: 50)")
    parser.add_argument("--full-eval", action="store_true", help="Run evaluation on all examples")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine evaluation size
    if args.full_eval:
        limit = None
        eval_size_desc = "all examples"
    else:
        limit = args.limit
        eval_size_desc = f"{limit} examples"
    
    logger.info("=" * 60)
    logger.info("🚀 ENHANCED SQUAD 2.0 EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Evaluation size: {eval_size_desc}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize evaluator and load dataset
        evaluator = SQuADEvaluator()
        examples = evaluator.load_dataset("validation")
        
        logger.info(f"Loaded {len(examples)} examples from SQuAD 2.0")
        
        # Create baseline systems
        baseline_systems = create_baseline_systems(args.api_key)
        
        # Create TMM pipeline
        tmm_pipeline = create_tmm_pipeline(args.api_key)
        
        # Evaluate all systems
        results = []
        
        # Evaluate baselines
        for system_name, system in baseline_systems.items():
            logger.info(f"Evaluating {system_name}...")
            result = evaluator.evaluate_baseline_system(
                system, examples, system_name, limit=limit
            )
            results.append(result)
            logger.info(f"{system_name}: {result.accuracy:.2%} accuracy "
                       f"(Answerable: {result.answerable_accuracy:.2%}, "
                       f"Unanswerable: {result.unanswerable_accuracy:.2%})")
        
        # Evaluate TMM system
        logger.info("Evaluating TMM system...")
        tmm_result = evaluator.evaluate_tmm_system(tmm_pipeline, examples, limit=limit)
        results.append(tmm_result)
        logger.info(f"TMM Pipeline: {tmm_result.accuracy:.2%} accuracy "
                   f"(Answerable: {tmm_result.answerable_accuracy:.2%}, "
                   f"Unanswerable: {tmm_result.unanswerable_accuracy:.2%})")
        
        # Save results
        output_file = output_dir / "squad2_evaluation_results.json"
        evaluator.save_results(results, str(output_file))
        
        # Print summary
        print()
        print("=" * 60)
        print("📊 SQUAD 2.0 EVALUATION RESULTS")
        print("=" * 60)
        
        # Sort results by overall accuracy
        sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)
        
        for result in sorted_results:
            print(f"{result.system_name:<15}: {result.accuracy:.2%} accuracy, "
                  f"{result.avg_response_time:.2f}s avg time")
            print(f"{'':>15}  Answerable: {result.answerable_accuracy:.2%}, "
                  f"Unanswerable: {result.unanswerable_accuracy:.2%}")
        
        print()
        print(f"Best performing system: {sorted_results[0].system_name} ({sorted_results[0].accuracy:.2%})")
        print(f"Results saved to: {output_file}")
        
        # TMM-specific insights
        tmm_result = next(r for r in results if "TMM" in r.system_name)
        if tmm_result.memory_metrics:
            print()
            print("🧠 TMM Memory Utilization:")
            for key, value in tmm_result.memory_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
        
        print()
        print("✅ SQuAD 2.0 evaluation completed successfully!")
        print(f"Evaluated {len(results)} systems on SQuAD 2.0 dataset")
        
        # Analyze memory advantage
        if tmm_result.unanswerable_accuracy > 0.7:  # Good performance on unanswerable
            print("🎯 TMM demonstrates strong truth verification on unanswerable questions!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
