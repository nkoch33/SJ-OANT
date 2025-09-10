#!/usr/bin/env python3
"""
TMM Results Analysis: Generate reports from TMM evaluation results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def load_results(results_path: str = "testing/results/tmm_evaluation_results.json") -> Dict[str, Any]:
    """Load evaluation results from file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive summary report."""
    report = []
    
    # Header
    report.append("🏆 TMM SYSTEM EVALUATION SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Metadata
    report.append(f"📅 Evaluation Date: {results.get('timestamp', 'Unknown')}")
    report.append(f"🔧 System: {results.get('system', 'TMM')}")
    report.append(f"📊 Examples per Benchmark: {results.get('num_examples_per_benchmark', 25)}")
    report.append(f"🎯 Benchmarks: {', '.join(results.get('benchmarks', []))}")
    report.append("")
    
    # Results by benchmark
    results_data = results.get("results", {})
    
    for benchmark, benchmark_results in results_data.items():
        report.append(f"📊 {benchmark.upper()} BENCHMARK RESULTS")
        report.append("-" * 50)
        
        tmm_results = benchmark_results.get("TMM", {})
        
        if "error" in tmm_results:
            report.append(f"❌ TMM: {tmm_results['error']}")
            report.append("")
            continue
        
        # Extract key metrics based on benchmark
        if benchmark == "multiwoz":
            key_metrics = ["bleu", "success", "inform"]
        elif benchmark == "sgd":
            key_metrics = ["intent_accuracy", "slot_f1", "success_rate"]
        elif benchmark == "taskmaster":
            key_metrics = ["bleu", "rouge", "task_completion"]
        elif benchmark == "multidogo":
            key_metrics = ["intent_accuracy", "slot_f1", "domain_adaptation"]
        else:
            key_metrics = []
        
        report.append("TMM System Performance:")
        report.append("")
        
        for metric in key_metrics:
            if metric in tmm_results:
                if isinstance(tmm_results[metric], dict) and "total" in tmm_results[metric]:
                    score = tmm_results[metric]["total"]
                elif isinstance(tmm_results[metric], dict):
                    score = list(tmm_results[metric].values())[0]
                else:
                    score = tmm_results[metric]
                report.append(f"  {metric.replace('_', ' ').title()}: {score:.2f}%")
        
        report.append("")
    
    # Overall performance summary
    report.append("🎯 OVERALL PERFORMANCE SUMMARY")
    report.append("-" * 50)
    
    # Calculate average scores across benchmarks
    benchmark_averages = []
    
    for benchmark, benchmark_results in results_data.items():
        tmm_results = benchmark_results.get("TMM", {})
        
        if "error" in tmm_results:
            continue
        
        # Calculate average score for this benchmark
        scores = []
        if benchmark == "multiwoz":
            for metric in ["bleu", "success", "inform"]:
                if metric in tmm_results:
                    if isinstance(tmm_results[metric], dict) and "total" in tmm_results[metric]:
                        scores.append(tmm_results[metric]["total"])
                    elif isinstance(tmm_results[metric], dict):
                        scores.append(list(tmm_results[metric].values())[0])
                    else:
                        scores.append(tmm_results[metric])
        elif benchmark == "sgd":
            for metric in ["intent_accuracy", "slot_f1", "success_rate"]:
                if metric in tmm_results:
                    if isinstance(tmm_results[metric], dict) and "total" in tmm_results[metric]:
                        scores.append(tmm_results[metric]["total"])
                    elif isinstance(tmm_results[metric], dict):
                        scores.append(list(tmm_results[metric].values())[0])
                    else:
                        scores.append(tmm_results[metric])
        elif benchmark == "taskmaster":
            for metric in ["bleu", "rouge", "task_completion"]:
                if metric in tmm_results:
                    if isinstance(tmm_results[metric], dict) and "total" in tmm_results[metric]:
                        scores.append(tmm_results[metric]["total"])
                    elif isinstance(tmm_results[metric], dict):
                        scores.append(list(tmm_results[metric].values())[0])
                    else:
                        scores.append(tmm_results[metric])
        elif benchmark == "multidogo":
            for metric in ["intent_accuracy", "slot_f1", "domain_adaptation"]:
                if metric in tmm_results:
                    if isinstance(tmm_results[metric], dict) and "total" in tmm_results[metric]:
                        scores.append(tmm_results[metric]["total"])
                    elif isinstance(tmm_results[metric], dict):
                        scores.append(list(tmm_results[metric].values())[0])
                    else:
                        scores.append(tmm_results[metric])
        
        if scores:
            benchmark_averages.append((benchmark, sum(scores) / len(scores)))
    
    # Print overall averages
    if benchmark_averages:
        report.append("TMM Performance by Benchmark:")
        report.append("")
        
        for benchmark, avg_score in benchmark_averages:
            report.append(f"  {benchmark.upper()}: {avg_score:.2f}%")
        
        overall_avg = sum(score for _, score in benchmark_averages) / len(benchmark_averages)
        report.append("")
        report.append(f"Overall Average: {overall_avg:.2f}%")
    
    report.append("")
    
    # Key insights
    report.append("🔍 KEY INSIGHTS")
    report.append("-" * 50)
    
    if benchmark_averages:
        best_benchmark = max(benchmark_averages, key=lambda x: x[1])
        worst_benchmark = min(benchmark_averages, key=lambda x: x[1])
        
        report.append(f"🏆 Best Performing Benchmark: {best_benchmark[0].upper()} ({best_benchmark[1]:.2f}%)")
        report.append(f"📉 Challenging Benchmark: {worst_benchmark[0].upper()} ({worst_benchmark[1]:.2f}%)")
        
        performance_gap = best_benchmark[1] - worst_benchmark[1]
        report.append(f"📊 Performance Gap: {performance_gap:.2f}% between best and worst benchmarks")
        
        report.append("")
        report.append("Benchmark-Specific Insights:")
        for benchmark, avg_score in benchmark_averages:
            if avg_score >= 70:
                status = "🟢 Strong"
            elif avg_score >= 50:
                status = "🟡 Moderate"
            else:
                status = "🔴 Needs Improvement"
            report.append(f"  • {benchmark.upper()}: {status} ({avg_score:.2f}%)")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def save_report(report: str, output_path: str = "testing/results/tmm_summary.txt"):
    """Save the report to a file."""
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"📁 Summary report saved to: {output_path}")

def main():
    """Main analysis function."""
    results_path = "testing/results/tmm_evaluation_results.json"
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("Please run the TMM evaluation first:")
        print("python testing/run_tmm_evaluation.py")
        return
    
    print("📊 Analyzing TMM evaluation results...")
    
    try:
        # Load results
        results = load_results(results_path)
        
        # Generate report
        report = generate_summary_report(results)
        
        # Print report
        print(report)
        
        # Save report
        save_report(report)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
