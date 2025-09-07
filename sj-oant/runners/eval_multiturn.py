"""
MultiWOZ Evaluation Runner

This script runs comprehensive evaluation of the TMM system and baselines
on the MultiWOZ 2.4 dataset, testing multi-turn conversational memory,
truth maintenance, and information consistency.

Usage:
    python runners/eval_multiturn.py --api-key YOUR_API_KEY --limit 1000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.multiturn_eval import MultiTurnEvaluator, MultiTurnEvaluationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate TMM system on MultiWOZ dataset")
    parser.add_argument("--api-key", required=True, help="Google API key")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of dialogues to evaluate")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to use")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 MULTIWOZ MULTI-TURN EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Evaluation size: {args.limit} dialogues")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dataset split: {args.split}")
    
    try:
        # Initialize evaluator
        evaluator = MultiTurnEvaluator(args.api_key)
        
        # Load dialogues
        dialogues = evaluator.load_dialogues(split=args.split, limit=args.limit)
        logger.info(f"Loaded {len(dialogues)} dialogues for evaluation")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Evaluate TMM system with error handling
        logger.info("Evaluating TMM Pipeline...")
        try:
            tmm_result = evaluator.evaluate_tmm_system(dialogues)
            results["TMM_Pipeline"] = tmm_result
            logger.info(f"TMM Pipeline: {tmm_result.dialogue_success_rate:.2%} success rate")
            logger.info(f"  Information Accuracy: {tmm_result.information_accuracy:.2%}")
            logger.info(f"  Memory Consistency: {tmm_result.memory_consistency:.2%}")
            logger.info(f"  False Memory Rate: {tmm_result.false_memory_rate:.2%}")
            logger.info(f"  Avg Response Time: {tmm_result.response_time_avg:.3f}s")
        except Exception as e:
            logger.error(f"TMM evaluation failed: {e}")
            logger.error("Continuing with baseline evaluation...")
            # Create a dummy result to prevent crash
            from evaluation.multiturn_eval import MultiTurnEvaluationResult
            from evaluation.methodology_metrics import MethodologyMetrics
            results["TMM_Pipeline"] = MultiTurnEvaluationResult(
                dialogue_success_rate=0.0, information_accuracy=0.0, memory_consistency=0.0,
                response_time_avg=0.0, memory_operations={}, memory_retrievals=0, memory_stores=0,
                memory_updates=0, false_memory_rate=0.0, truth_verification_calls=0,
                contradiction_detections=0, methodology_metrics=MethodologyMetrics(
                    fmr=0.0, mel=0.0, dar=0.0, accuracy=0.0, answerable_accuracy=0.0,
                    unanswerable_accuracy=0.0, memory_consistency=0.0, contradiction_resolution=0.0
                ), successful_dialogues=0, total_dialogues=len(dialogues), total_turns=0,
                domain_breakdown={}
            )
        
        # Evaluate baselines
        baseline_names = ["DirectLLM", "LongContext", "SimpleStateTracker", "NaiveMemory"]
        
        for baseline_name in baseline_names:
            logger.info(f"Evaluating {baseline_name}...")
            try:
                baseline_result = evaluator.evaluate_baseline_system(baseline_name, dialogues)
                results[baseline_name] = baseline_result
                logger.info(f"{baseline_name}: {baseline_result.dialogue_success_rate:.2%} success rate")
                logger.info(f"  Avg Response Time: {baseline_result.response_time_avg:.3f}s")
            except Exception as e:
                logger.error(f"{baseline_name} evaluation failed: {e}")
                logger.error("Continuing with next baseline...")
                # Create a dummy result to prevent crash
                from evaluation.multiturn_eval import MultiTurnEvaluationResult
                from evaluation.methodology_metrics import MethodologyMetrics
                results[baseline_name] = MultiTurnEvaluationResult(
                    dialogue_success_rate=0.0, information_accuracy=0.0, memory_consistency=0.0,
                    response_time_avg=0.0, memory_operations={}, memory_retrievals=0, memory_stores=0,
                    memory_updates=0, false_memory_rate=0.0, truth_verification_calls=0,
                    contradiction_detections=0, methodology_metrics=MethodologyMetrics(
                        fmr=0.0, mel=0.0, dar=0.0, accuracy=0.0, answerable_accuracy=0.0,
                        unanswerable_accuracy=0.0, memory_consistency=0.0, contradiction_resolution=0.0
                    ), successful_dialogues=0, total_dialogues=len(dialogues), total_turns=0,
                    domain_breakdown={}
                )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 MULTIWOZ EVALUATION RESULTS")
        logger.info("=" * 60)
        
        # Sort results by success rate
        sorted_results = sorted(results.items(), key=lambda x: x[1].dialogue_success_rate, reverse=True)
        
        for system_name, result in sorted_results:
            logger.info(f"{system_name:<20}: {result.dialogue_success_rate:.2%} success rate")
            if hasattr(result, 'information_accuracy') and result.information_accuracy > 0:
                logger.info(f"  Information Accuracy: {result.information_accuracy:.2%}")
            if hasattr(result, 'false_memory_rate') and result.false_memory_rate > 0:
                logger.info(f"  False Memory Rate: {result.false_memory_rate:.2%}")
        
        # Find best performing system
        best_system = sorted_results[0]
        logger.info(f"\nBest performing system: {best_system[0]} ({best_system[1].dialogue_success_rate:.2%})")
        
        # Print domain breakdown for TMM system
        if "TMM_Pipeline" in results:
            logger.info("\n🧠 TMM Memory Utilization:")
            tmm_result = results["TMM_Pipeline"]
            logger.info(f"  Memory Operations: {tmm_result.memory_operations}")
            logger.info(f"  Truth Verification Calls: {tmm_result.truth_verification_calls}")
            logger.info(f"  Contradiction Detections: {tmm_result.contradiction_detections}")
            
            logger.info("\n📊 Domain Breakdown:")
            for domain, stats in tmm_result.domain_breakdown.items():
                logger.info(f"  {domain}: {stats['success_rate']:.2%} success rate ({stats['total_dialogues']} dialogues)")
        
        # Save results
        output_file = output_dir / "multiturn_evaluation_results.json"
        evaluator.save_results(results, str(output_file))
        logger.info(f"Results saved to: {output_file}")
        
        logger.info("\n✅ MultiWOZ evaluation completed successfully!")
        logger.info(f"Evaluated {len(dialogues)} dialogues across {len(results)} systems")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()