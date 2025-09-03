#!/usr/bin/env python3
"""
Multi-Turn Conversation Evaluation Runner

This script evaluates TMM's memory persistence capabilities through
multi-turn conversations derived from SQuAD 2.0 contexts.

This evaluation specifically tests TMM's core advantage: maintaining
and retrieving information across multiple conversation turns.

Usage:
    python runners/eval_multiturn.py --api-key YOUR_API_KEY --scenarios 20
    python runners/eval_multiturn.py --api-key YOUR_API_KEY --scenarios 50
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
from evaluation.multiturn_eval import MultiTurnEvaluator
from baselines.simple_systems import DirectLLMBaseline, LongContextBaseline, BasicMemoryBaseline
from tmm_pipeline import create_tmm_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_memory_capable_systems(api_key: str):
    """Create systems that can potentially handle multi-turn conversations."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    
    systems = {
        "DirectLLM": DirectLLMBaseline(llm),  # No memory - baseline
        "LongContext": LongContextBaseline(llm, max_context_length=8000),  # Simple history
        "BasicMemory": BasicMemoryBaseline(llm),  # Simple memory
        "TMM_Pipeline": create_tmm_pipeline(api_key)  # Truth-maintained memory
    }
    
    logger.info(f"Created {len(systems)} systems for multi-turn evaluation")
    return systems

def main():
    parser = argparse.ArgumentParser(description="Run multi-turn conversation evaluation")
    parser.add_argument("--api-key", required=True, help="Google API key")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of conversation scenarios (default: 20)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🔄 MULTI-TURN CONVERSATION EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Conversation scenarios: {args.scenarios}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize evaluators
        squad_evaluator = SQuADEvaluator()
        multiturn_evaluator = MultiTurnEvaluator()
        
        # Load SQuAD 2.0 dataset
        examples = squad_evaluator.load_dataset("validation")
        logger.info(f"Loaded {len(examples)} SQuAD 2.0 examples")
        
        # Create conversation scenarios
        scenarios = multiturn_evaluator.create_conversation_scenarios(
            examples, scenario_count=args.scenarios
        )
        logger.info(f"Created {len(scenarios)} conversation scenarios")
        
        # Analyze scenarios
        memory_levels = {}
        for scenario in scenarios:
            level = scenario.memory_dependency_level
            memory_levels[level] = memory_levels.get(level, 0) + 1
        
        logger.info("Memory dependency distribution:")
        for level, count in memory_levels.items():
            logger.info(f"  {level}: {count} scenarios")
        
        # Create systems
        systems = create_memory_capable_systems(args.api_key)
        
        # Evaluate all systems
        all_results = {}
        
        for system_name, system in systems.items():
            logger.info(f"Evaluating {system_name} on multi-turn conversations...")
            
            try:
                result = multiturn_evaluator.evaluate_system_multiturn(
                    system, scenarios, system_name
                )
                all_results[system_name] = result
                
                logger.info(f"{system_name} results:")
                logger.info(f"  Overall accuracy: {result['overall_accuracy']:.2%}")
                logger.info(f"  Memory-dependent accuracy: {result['memory_dependent_accuracy']:.2%}")
                logger.info(f"  Avg response time: {result['avg_response_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"Error evaluating {system_name}: {e}")
                continue
        
        # Save results
        output_file = output_dir / "multiturn_evaluation_results.json"
        
        # Combine all results for saving
        combined_results = {
            "evaluation_metadata": {
                "total_scenarios": len(scenarios),
                "memory_dependency_distribution": memory_levels,
                "systems_evaluated": list(all_results.keys())
            },
            "system_results": all_results
        }
        
        multiturn_evaluator.save_results(combined_results, str(output_file))
        
        # Print comprehensive summary
        print()
        print("=" * 60)
        print("🔄 MULTI-TURN EVALUATION RESULTS")
        print("=" * 60)
        
        # Sort systems by memory-dependent accuracy (key metric for TMM)
        system_performance = []
        for system_name, result in all_results.items():
            system_performance.append({
                "name": system_name,
                "overall_acc": result["overall_accuracy"],
                "memory_acc": result["memory_dependent_accuracy"],
                "avg_time": result["avg_response_time"]
            })
        
        # Sort by memory-dependent accuracy first, then overall
        system_performance.sort(key=lambda x: (x["memory_acc"], x["overall_acc"]), reverse=True)
        
        print("Overall Performance:")
        for perf in system_performance:
            print(f"{perf['name']:<15}: {perf['overall_acc']:.2%} overall, "
                  f"{perf['memory_acc']:.2%} memory-dependent, "
                  f"{perf['avg_time']:.2f}s avg")
        
        print()
        print("Key Insights:")
        
        # Find TMM performance
        tmm_result = all_results.get("TMM_Pipeline")
        if tmm_result:
            tmm_memory_acc = tmm_result["memory_dependent_accuracy"]
            
            # Compare with best baseline on memory-dependent tasks
            best_baseline_memory = max(
                perf["memory_acc"] for perf in system_performance 
                if perf["name"] != "TMM_Pipeline"
            )
            
            if tmm_memory_acc > best_baseline_memory:
                improvement = tmm_memory_acc - best_baseline_memory
                print(f"✅ TMM outperforms baselines on memory-dependent tasks by {improvement:.2%}")
            else:
                deficit = best_baseline_memory - tmm_memory_acc
                print(f"❗ TMM underperforms best baseline on memory tasks by {deficit:.2%}")
            
            print(f"🧠 TMM memory-dependent accuracy: {tmm_memory_acc:.2%}")
            print(f"📊 Best baseline memory accuracy: {best_baseline_memory:.2%}")
        
        print(f"📁 Results saved to: {output_file}")
        print("✅ Multi-turn evaluation completed!")
        
    except Exception as e:
        logger.error(f"Multi-turn evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
