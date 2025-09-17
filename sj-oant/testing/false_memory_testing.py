"""
False Memory Formation Testing Script

Integrates with existing TMM system to test false memory prevention capabilities.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from false_memory_evaluation import FalseMemoryEvaluator
from multi_agent_pipeline import MultiAgentTMMPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_false_memory_prevention(api_key: str, num_scenarios: int = 5) -> Dict[str, Any]:
    """
    Test TMM's false memory prevention capabilities.
    
    Args:
        api_key: Gemini API key
        num_scenarios: Number of test scenarios per benchmark
        
    Returns:
        Dictionary with false memory evaluation results
    """
    logger.info("🚀 Starting False Memory Prevention Testing")
    logger.info("="*60)
    
    try:
        # Initialize TMM pipeline
        logger.info("Initializing TMM pipeline...")
        tmm_pipeline = MultiAgentTMMPipeline(api_key=api_key)
        
        # Initialize false memory evaluator
        logger.info("Initializing False Memory Evaluator...")
        evaluator = FalseMemoryEvaluator(tmm_pipeline=tmm_pipeline)
        
        # Run evaluation on all benchmarks
        logger.info(f"Running false memory evaluation on {num_scenarios} scenarios per benchmark...")
        results = evaluator.evaluate_all_benchmarks_false_memory(num_scenarios)
        
        # Print results
        print_false_memory_results(results)
        
        # Save results
        results_file = Path(__file__).parent.parent / "results" / "false_memory_evaluation_results.json"
        results_file.parent.mkdir(exist_ok=True)
        evaluator.save_results(results, str(results_file))
        
        logger.info(f"✅ False memory evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ False memory evaluation failed: {e}")
        return {"error": str(e)}

def print_false_memory_results(results: Dict[str, Any]) -> None:
    """Print false memory evaluation results in a readable format."""
    print("\n" + "="*80)
    print("🔬 FALSE MEMORY FORMATION EVALUATION RESULTS")
    print("="*80)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Print overall summary
    if "overall_summary" in results:
        print(results["overall_summary"])
    
    # Print benchmark-specific results
    if "benchmark_results" in results:
        print("\n📊 BENCHMARK-SPECIFIC RESULTS:")
        print("-" * 50)
        
        for benchmark, benchmark_result in results["benchmark_results"].items():
            if "error" in benchmark_result:
                print(f"\n❌ {benchmark.upper()}: {benchmark_result['error']}")
                continue
            
            print(f"\n🎯 {benchmark.upper()}:")
            
            if "aggregate_metrics" in benchmark_result:
                metrics = benchmark_result["aggregate_metrics"]
                print(f"   • FMR: {metrics.get('avg_fmr', 0):.2f}%")
                print(f"   • MEL: {metrics.get('avg_mel', 0):.2f} seconds")
                print(f"   • DAR: {metrics.get('avg_dar', 0):.2f}%")
                print(f"   • Contradiction Detection: {metrics.get('avg_contradiction_detection', 0):.2f}%")
                print(f"   • Scenarios: {metrics.get('num_scenarios', 0)}")
    
    # Print cross-benchmark metrics
    if "cross_benchmark_metrics" in results:
        print("\n🏆 CROSS-BENCHMARK METRICS:")
        print("-" * 50)
        
        metrics = results["cross_benchmark_metrics"]
        print(f"   • Overall FMR: {metrics.get('overall_fmr', 0):.2f}%")
        print(f"   • Overall MEL: {metrics.get('overall_mel', 0):.2f} seconds")
        print(f"   • Overall DAR: {metrics.get('overall_dar', 0):.2f}%")
        print(f"   • Overall Contradiction Detection: {metrics.get('overall_contradiction_detection', 0):.2f}%")
        print(f"   • Total Scenarios: {metrics.get('total_scenarios', 0)}")
    
    print("\n" + "="*80)
    print("🎯 RESEARCH ALIGNMENT: These metrics directly measure TMM's ability")
    print("   to prevent false memory formation - the core research vision!")
    print("="*80)

def quick_false_memory_test(api_key: str) -> None:
    """Run a quick false memory test with minimal scenarios."""
    logger.info("🔍 Running Quick False Memory Test...")
    
    results = test_false_memory_prevention(api_key, num_scenarios=2)
    
    if "error" not in results:
        logger.info("✅ Quick test completed successfully!")
    else:
        logger.error(f"❌ Quick test failed: {results['error']}")

if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        sys.exit(1)
    
    # Run false memory evaluation
    print("🚀 Starting False Memory Formation Evaluation")
    print("This tests TMM's ability to prevent false memory formation")
    print("using the core research metrics: FMR, MEL, and DAR")
    print()
    
    # Run with 3 scenarios per benchmark for quick testing
    results = test_false_memory_prevention(api_key, num_scenarios=3)
    
    if "error" not in results:
        print("\n🎉 False Memory Evaluation Completed Successfully!")
        print("📊 Results show TMM's performance on preventing false memory formation")
        print("🔬 These metrics align with your research vision!")
    else:
        print(f"\n❌ Evaluation failed: {results['error']}")
        sys.exit(1)
