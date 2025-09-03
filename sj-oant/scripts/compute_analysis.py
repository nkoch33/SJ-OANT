#!/usr/bin/env python3
"""
Compute Requirements Analysis for Phase 2

This script analyzes the computational requirements for full-scale TMM evaluation
and provides recommendations for cloud compute resources.

Usage:
    python scripts/compute_analysis.py --api-key YOUR_API_KEY
"""

import argparse
import time
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.squad_eval import SQuADEvaluator
from baselines.simple_systems import DirectLLMBaseline
from tmm_pipeline import create_tmm_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dataset_scale():
    """Analyze the full SQuAD 2.0 dataset scale."""
    print("📊 ANALYZING DATASET SCALE")
    print("=" * 50)
    
    evaluator = SQuADEvaluator()
    
    # Load train and validation splits
    train_examples = evaluator.load_dataset("train")
    val_examples = evaluator.load_dataset("validation")
    
    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")
    print(f"Total examples: {len(train_examples) + len(val_examples):,}")
    
    # Analyze context complexity
    val_contexts = [ex.context for ex in val_examples]
    context_lengths = [len(ctx.split()) for ctx in val_contexts]
    
    avg_length = sum(context_lengths) / len(context_lengths)
    max_length = max(context_lengths)
    min_length = min(context_lengths)
    
    print(f"\nContext Analysis:")
    print(f"  Average context length: {avg_length:.1f} words")
    print(f"  Maximum context length: {max_length} words")
    print(f"  Minimum context length: {min_length} words")
    
    # Analyze answerable vs unanswerable distribution
    answerable = sum(1 for ex in val_examples if ex.is_answerable)
    unanswerable = len(val_examples) - answerable
    
    print(f"\nQuestion Distribution:")
    print(f"  Answerable: {answerable:,} ({answerable/len(val_examples)*100:.1f}%)")
    print(f"  Unanswerable: {unanswerable:,} ({unanswerable/len(val_examples)*100:.1f}%)")
    
    return {
        "total_examples": len(train_examples) + len(val_examples),
        "validation_examples": len(val_examples),
        "avg_context_length": avg_length,
        "answerable_ratio": answerable / len(val_examples)
    }

def benchmark_system_performance(api_key: str, sample_size: int = 10):
    """Benchmark system performance on a small sample."""
    print(f"\n⚡ BENCHMARKING SYSTEM PERFORMANCE ({sample_size} examples)")
    print("=" * 50)
    
    # Load sample data
    evaluator = SQuADEvaluator()
    examples = evaluator.load_dataset("validation")[:sample_size]
    
    # Create systems
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    
    systems = {
        "DirectLLM": DirectLLMBaseline(llm),
        "TMM Pipeline": create_tmm_pipeline(api_key)
    }
    
    performance_data = {}
    
    for system_name, system in systems.items():
        print(f"\nTesting {system_name}...")
        start_time = time.time()
        
        try:
            result = evaluator.evaluate_baseline_system(
                system, examples, system_name, limit=sample_size
            ) if system_name != "TMM Pipeline" else evaluator.evaluate_tmm_system(
                system, examples, limit=sample_size
            )
            
            elapsed_time = time.time() - start_time
            
            performance_data[system_name] = {
                "accuracy": result.accuracy,
                "avg_response_time": result.avg_response_time,
                "total_time": elapsed_time,
                "examples_per_second": sample_size / elapsed_time,
                "api_calls_estimate": sample_size * 2  # Context + question for each
            }
            
            print(f"  Accuracy: {result.accuracy:.2%}")
            print(f"  Avg response time: {result.avg_response_time:.2f}s")
            print(f"  Total time: {elapsed_time:.1f}s")
            print(f"  Examples/second: {sample_size/elapsed_time:.2f}")
            
        except Exception as e:
            print(f"  Error testing {system_name}: {e}")
            performance_data[system_name] = {"error": str(e)}
    
    return performance_data

def estimate_full_scale_requirements(performance_data: dict, dataset_stats: dict):
    """Estimate requirements for full-scale evaluation."""
    print(f"\n🔮 FULL-SCALE EVALUATION ESTIMATES")
    print("=" * 50)
    
    total_examples = dataset_stats["validation_examples"]
    systems_count = 5  # TMM + 4 baselines
    
    print(f"Target evaluation: {total_examples:,} examples × {systems_count} systems = {total_examples * systems_count:,} evaluations")
    
    # Use TMM performance as baseline (most intensive)
    if "TMM Pipeline" in performance_data and "error" not in performance_data["TMM Pipeline"]:
        tmm_perf = performance_data["TMM Pipeline"]
        
        # Time estimates
        total_time_hours = (total_examples * systems_count * tmm_perf["avg_response_time"]) / 3600
        total_api_calls = total_examples * systems_count * tmm_perf["api_calls_estimate"]
        
        print(f"\nTime Estimates (based on TMM performance):")
        print(f"  Estimated total time: {total_time_hours:.1f} hours")
        print(f"  Estimated API calls: {total_api_calls:,}")
        
        # Cost estimates (approximate)
        estimated_cost_per_1k_calls = 0.50  # Rough estimate for Gemini
        estimated_total_cost = (total_api_calls / 1000) * estimated_cost_per_1k_calls
        
        print(f"  Estimated API cost: ${estimated_total_cost:.2f}")
        
        # Resource recommendations
        print(f"\n💡 COMPUTE RECOMMENDATIONS:")
        
        if total_time_hours > 24:
            print("  🔴 REQUIRES CLOUD COMPUTE")
            print("    - Google Colab Pro+ or AWS EC2 recommended")
            print("    - Consider parallel processing across multiple instances")
            print("    - Implement checkpointing for reliability")
        elif total_time_hours > 8:
            print("  🟡 CLOUD COMPUTE RECOMMENDED") 
            print("    - Google Colab Pro or local high-end machine")
            print("    - Batch processing recommended")
        else:
            print("  🟢 LOCAL MACHINE SUFFICIENT")
            print("    - Can run on local machine overnight")
            
        if estimated_total_cost > 50:
            print(f"    - Budget ${estimated_total_cost:.0f} for API costs")
            print("    - Consider API rate limiting strategies")
            
        return {
            "total_time_hours": total_time_hours,
            "total_api_calls": total_api_calls,
            "estimated_cost": estimated_total_cost,
            "requires_cloud": total_time_hours > 24
        }
    else:
        print("❌ Cannot estimate - TMM benchmarking failed")
        return None

def recommend_batch_strategy(requirements: dict, dataset_stats: dict):
    """Recommend batching strategy for large-scale evaluation."""
    if not requirements:
        return
        
    print(f"\n📦 RECOMMENDED BATCH STRATEGY")
    print("=" * 50)
    
    total_examples = dataset_stats["validation_examples"]
    
    if requirements["requires_cloud"]:
        # Large-scale batching
        batch_sizes = [100, 500, 1000]
        print("For cloud compute:")
        
        for batch_size in batch_sizes:
            num_batches = (total_examples + batch_size - 1) // batch_size
            batch_time = requirements["total_time_hours"] / num_batches
            
            print(f"  Batch size {batch_size}: {num_batches} batches, {batch_time:.1f}h per batch")
            
        print("\n  Recommended: 500-example batches with checkpointing")
        print("  - Balances progress visibility with efficiency")
        print("  - Allows recovery from failures")
        print("  - Enables parallel processing")
        
    else:
        # Smaller batches for local
        print("For local compute:")
        print("  - 50-100 example batches recommended")
        print("  - Process overnight or during off-hours")
        print("  - Monitor API rate limits")

def main():
    parser = argparse.ArgumentParser(description="Analyze compute requirements for Phase 2")
    parser.add_argument("--api-key", required=True, help="Google API key for benchmarking")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size for benchmarking")
    
    args = parser.parse_args()
    
    print("🚀 PHASE 2 COMPUTE ANALYSIS")
    print("=" * 60)
    
    try:
        # Step 1: Analyze dataset scale
        dataset_stats = analyze_dataset_scale()
        
        # Step 2: Benchmark system performance
        performance_data = benchmark_system_performance(args.api_key, args.sample_size)
        
        # Step 3: Estimate full-scale requirements
        requirements = estimate_full_scale_requirements(performance_data, dataset_stats)
        
        # Step 4: Recommend batch strategy
        recommend_batch_strategy(requirements, dataset_stats)
        
        print("\n" + "=" * 60)
        print("✅ COMPUTE ANALYSIS COMPLETE")
        
        if requirements and requirements["requires_cloud"]:
            print("🔴 RECOMMENDATION: Proceed with cloud compute setup")
        else:
            print("🟢 RECOMMENDATION: Local evaluation feasible")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
