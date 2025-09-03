#!/usr/bin/env python3
"""
TMM Performance Profiler

This script profiles the TMM pipeline to identify performance bottlenecks
and optimization opportunities for Phase 2 large-scale evaluation.

Usage:
    python scripts/performance_profiler.py --api-key YOUR_API_KEY
"""

import argparse
import time
import cProfile
import pstats
import io
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.squad_eval import SQuADEvaluator
from tmm_pipeline import create_tmm_pipeline
from baselines.simple_systems import DirectLLMBaseline
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.WARNING)  # Reduce noise during profiling
logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Comprehensive performance profiler for TMM system."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tmm_pipeline = None
        self.baseline = None
        self.evaluator = SQuADEvaluator()
        
    def setup_systems(self):
        """Initialize TMM and baseline systems for comparison."""
        print("🔧 Setting up systems for profiling...")
        
        # Create TMM pipeline
        self.tmm_pipeline = create_tmm_pipeline(self.api_key)
        
        # Create baseline for comparison
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=self.api_key
        )
        self.baseline = DirectLLMBaseline(llm)
        
        print("✅ Systems ready for profiling")
    
    def profile_component_timing(self, examples: List, num_examples: int = 5):
        """Profile timing of individual TMM components."""
        print(f"\n⏱️  COMPONENT TIMING ANALYSIS ({num_examples} examples)")
        print("=" * 60)
        
        component_times = {
            "prompt_refinement": [],
            "context_filtering": [],
            "truth_verification": [],
            "memory_curation": [],
            "response_generation": [],
            "total_pipeline": []
        }
        
        for i, example in enumerate(examples[:num_examples]):
            print(f"Profiling example {i+1}/{num_examples}...")
            
            # Reset memory for clean measurement
            self.tmm_pipeline.reset_memory()
            
            # Store context
            context_prompt = f"Please remember this context: {example.context}"
            start_total = time.perf_counter()
            
            # TODO: Add component-level timing hooks to TMM pipeline
            # For now, measure total pipeline time
            self.tmm_pipeline.process(context_prompt)
            
            # Process question
            start_question = time.perf_counter()
            response = self.tmm_pipeline.process(example.question)
            end_total = time.perf_counter()
            
            total_time = end_total - start_total
            question_time = end_total - start_question
            context_time = start_question - start_total
            
            component_times["total_pipeline"].append(total_time)
            
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Context processing: {context_time:.3f}s")
            print(f"  Question processing: {question_time:.3f}s")
        
        # Calculate averages
        avg_times = {}
        for component, times in component_times.items():
            if times:
                avg_times[component] = sum(times) / len(times)
        
        print(f"\n📊 AVERAGE COMPONENT TIMES:")
        for component, avg_time in avg_times.items():
            print(f"  {component}: {avg_time:.3f}s")
        
        return avg_times
    
    def profile_memory_operations(self, examples: List, num_examples: int = 5):
        """Profile memory storage and retrieval operations."""
        print(f"\n🧠 MEMORY OPERATIONS PROFILING ({num_examples} examples)")
        print("=" * 60)
        
        memory_metrics = {
            "storage_times": [],
            "retrieval_times": [],
            "memory_sizes": [],
            "tier_distributions": []
        }
        
        for i, example in enumerate(examples[:num_examples]):
            print(f"Profiling memory ops {i+1}/{num_examples}...")
            
            # Reset memory
            self.tmm_pipeline.reset_memory()
            
            # Measure memory storage
            start_storage = time.perf_counter()
            context_prompt = f"Please remember this context: {example.context}"
            self.tmm_pipeline.process(context_prompt)
            storage_time = time.perf_counter() - start_storage
            
            # Get memory state after storage
            memory_state = self.tmm_pipeline.memory_store.get_metrics()
            memory_size = memory_state.get("total_records", 0)
            
            # Measure memory retrieval (during question processing)
            start_retrieval = time.perf_counter()
            response = self.tmm_pipeline.process(example.question)
            retrieval_time = time.perf_counter() - start_retrieval
            
            memory_metrics["storage_times"].append(storage_time)
            memory_metrics["retrieval_times"].append(retrieval_time)
            memory_metrics["memory_sizes"].append(memory_size)
            
            # Get tier distribution
            tier_dist = {
                "L1": memory_state.get("tier_l1_working_count", 0),
                "L2": memory_state.get("tier_l2_summarized_count", 0),
                "L3": memory_state.get("tier_l3_archival_count", 0),
                "Flagged": memory_state.get("tier_flagged_count", 0)
            }
            memory_metrics["tier_distributions"].append(tier_dist)
            
            print(f"  Storage time: {storage_time:.3f}s")
            print(f"  Retrieval time: {retrieval_time:.3f}s")
            print(f"  Memory records: {memory_size}")
            print(f"  Tier distribution: {tier_dist}")
        
        # Calculate averages
        avg_storage = sum(memory_metrics["storage_times"]) / len(memory_metrics["storage_times"])
        avg_retrieval = sum(memory_metrics["retrieval_times"]) / len(memory_metrics["retrieval_times"])
        avg_memory_size = sum(memory_metrics["memory_sizes"]) / len(memory_metrics["memory_sizes"])
        
        print(f"\n📊 MEMORY OPERATION AVERAGES:")
        print(f"  Average storage time: {avg_storage:.3f}s")
        print(f"  Average retrieval time: {avg_retrieval:.3f}s")
        print(f"  Average memory records: {avg_memory_size:.1f}")
        
        return memory_metrics
    
    def compare_with_baseline(self, examples: List, num_examples: int = 5):
        """Compare TMM performance with baseline system."""
        print(f"\n⚖️  TMM vs BASELINE COMPARISON ({num_examples} examples)")
        print("=" * 60)
        
        # Test TMM
        print("Testing TMM Pipeline...")
        start_tmm = time.perf_counter()
        tmm_result = self.evaluator.evaluate_tmm_system(
            self.tmm_pipeline, examples[:num_examples], limit=num_examples
        )
        tmm_total_time = time.perf_counter() - start_tmm
        
        # Test Baseline
        print("Testing DirectLLM Baseline...")
        start_baseline = time.perf_counter()
        baseline_result = self.evaluator.evaluate_baseline_system(
            self.baseline, examples[:num_examples], "DirectLLM", limit=num_examples
        )
        baseline_total_time = time.perf_counter() - start_baseline
        
        # Calculate performance ratios
        time_ratio = tmm_total_time / baseline_total_time
        response_time_ratio = tmm_result.avg_response_time / baseline_result.avg_response_time
        accuracy_diff = tmm_result.accuracy - baseline_result.accuracy
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"  TMM Total Time: {tmm_total_time:.2f}s")
        print(f"  Baseline Total Time: {baseline_total_time:.2f}s")
        print(f"  Time Overhead: {time_ratio:.2f}x")
        print(f"")
        print(f"  TMM Avg Response: {tmm_result.avg_response_time:.3f}s")
        print(f"  Baseline Avg Response: {baseline_result.avg_response_time:.3f}s")
        print(f"  Response Time Overhead: {response_time_ratio:.2f}x")
        print(f"")
        print(f"  TMM Accuracy: {tmm_result.accuracy:.2%}")
        print(f"  Baseline Accuracy: {baseline_result.accuracy:.2%}")
        print(f"  Accuracy Difference: {accuracy_diff:+.2%}")
        
        return {
            "time_overhead": time_ratio,
            "response_time_overhead": response_time_ratio,
            "accuracy_difference": accuracy_diff,
            "tmm_result": tmm_result,
            "baseline_result": baseline_result
        }
    
    def profile_api_usage(self, examples: List, num_examples: int = 5):
        """Profile API call patterns and efficiency."""
        print(f"\n🌐 API USAGE PROFILING ({num_examples} examples)")
        print("=" * 60)
        
        # Estimate API calls for TMM vs Baseline
        tmm_calls_per_example = 2  # Context + Question processing
        baseline_calls_per_example = 1  # Single call with full context+question
        
        tmm_total_calls = num_examples * tmm_calls_per_example
        baseline_total_calls = num_examples * baseline_calls_per_example
        
        call_overhead = tmm_total_calls / baseline_total_calls
        
        print(f"📊 API CALL ANALYSIS:")
        print(f"  TMM calls per example: {tmm_calls_per_example}")
        print(f"  Baseline calls per example: {baseline_calls_per_example}")
        print(f"  TMM total calls: {tmm_total_calls}")
        print(f"  Baseline total calls: {baseline_total_calls}")
        print(f"  API call overhead: {call_overhead:.2f}x")
        
        # Estimate for full-scale evaluation
        total_examples = 11873  # SQuAD 2.0 validation set
        systems_count = 5  # TMM + 4 baselines
        
        full_scale_tmm_calls = total_examples * tmm_calls_per_example
        full_scale_baseline_calls = total_examples * baseline_calls_per_example * 4  # 4 baselines
        full_scale_total = full_scale_tmm_calls + full_scale_baseline_calls
        
        print(f"\n🔮 FULL-SCALE API ESTIMATES:")
        print(f"  TMM calls: {full_scale_tmm_calls:,}")
        print(f"  Baseline calls: {full_scale_baseline_calls:,}")
        print(f"  Total API calls: {full_scale_total:,}")
        
        return {
            "call_overhead": call_overhead,
            "full_scale_calls": full_scale_total
        }
    
    def identify_bottlenecks(self, timing_data: Dict, memory_data: Dict, comparison_data: Dict):
        """Identify and prioritize performance bottlenecks."""
        print(f"\n🎯 BOTTLENECK ANALYSIS & RECOMMENDATIONS")
        print("=" * 60)
        
        bottlenecks = []
        recommendations = []
        
        # Analyze time overhead
        if comparison_data["time_overhead"] > 3.0:
            bottlenecks.append(f"High time overhead: {comparison_data['time_overhead']:.1f}x slower than baseline")
            recommendations.append("Priority 1: Optimize pipeline orchestration and reduce LangGraph overhead")
        
        if comparison_data["response_time_overhead"] > 2.5:
            bottlenecks.append(f"High response time overhead: {comparison_data['response_time_overhead']:.1f}x slower per response")
            recommendations.append("Priority 2: Optimize memory retrieval and response generation")
        
        # Analyze memory efficiency
        avg_memory_size = sum(memory_data["memory_sizes"]) / len(memory_data["memory_sizes"])
        if avg_memory_size > 10:
            bottlenecks.append(f"Large memory footprint: {avg_memory_size:.1f} records per interaction")
            recommendations.append("Priority 3: Implement memory compression and cleanup policies")
        
        # Analyze accuracy trade-off
        if comparison_data["accuracy_difference"] < 0.05:  # Less than 5% improvement
            bottlenecks.append(f"Limited accuracy advantage: {comparison_data['accuracy_difference']:+.2%}")
            recommendations.append("Priority 4: Enhance TACS filtering and truth verification for better performance")
        
        # API efficiency
        if comparison_data.get("call_overhead", 1) > 2.0:
            bottlenecks.append("High API call overhead")
            recommendations.append("Priority 5: Implement API call batching and caching")
        
        print("🔴 IDENTIFIED BOTTLENECKS:")
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"  {i}. {bottleneck}")
        
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        if not bottlenecks:
            print("✅ No major bottlenecks identified - system is well-optimized!")
        
        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations
        }

def main():
    parser = argparse.ArgumentParser(description="Profile TMM system performance")
    parser.add_argument("--api-key", required=True, help="Google API key")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples for profiling")
    
    args = parser.parse_args()
    
    print("🚀 TMM PERFORMANCE PROFILER")
    print("=" * 60)
    
    try:
        # Initialize profiler
        profiler = PerformanceProfiler(args.api_key)
        profiler.setup_systems()
        
        # Load test examples
        examples = profiler.evaluator.load_dataset("validation")
        print(f"Loaded {len(examples)} examples for profiling")
        
        # Run profiling analyses
        timing_data = profiler.profile_component_timing(examples, args.examples)
        memory_data = profiler.profile_memory_operations(examples, args.examples)
        comparison_data = profiler.compare_with_baseline(examples, args.examples)
        api_data = profiler.profile_api_usage(examples, args.examples)
        
        # Identify bottlenecks and recommendations
        analysis = profiler.identify_bottlenecks(timing_data, memory_data, comparison_data)
        
        print("\n" + "=" * 60)
        print("✅ PERFORMANCE PROFILING COMPLETE")
        
        # Summary recommendations
        if analysis["recommendations"]:
            print(f"\n🎯 TOP OPTIMIZATION PRIORITIES:")
            for i, rec in enumerate(analysis["recommendations"][:3], 1):
                print(f"  {i}. {rec}")
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
