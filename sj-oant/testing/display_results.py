#!/usr/bin/env python3
"""
Simple script to display TMM evaluation results in a readable format.
"""

import json
import os

def display_results():
    """Display TMM evaluation results."""
    results_path = "testing/results/tmm_evaluation_results.json"
    
    if not os.path.exists(results_path):
        print("❌ Results file not found!")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("="*80)
    print("🏆 TMM SYSTEM EVALUATION RESULTS")
    print("="*80)
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"🎯 Examples per benchmark: {results['num_examples_per_benchmark']}")
    print()
    
    for benchmark, benchmark_results in results['results'].items():
        print(f"📊 {benchmark.upper()} BENCHMARK:")
        print("-" * 50)
        
        for system, system_results in benchmark_results.items():
            print(f"✅ {system} System:")
            
            for metric, score in system_results.items():
                if isinstance(score, dict):
                    # Handle nested dictionaries
                    for sub_metric, sub_score in score.items():
                        if isinstance(sub_score, dict):
                            for final_metric, final_score in sub_score.items():
                                print(f"   {metric}.{sub_metric}.{final_metric}: {final_score:.2f}%")
                        else:
                            print(f"   {metric}.{sub_metric}: {sub_score:.2f}%")
                else:
                    print(f"   {metric}: {score:.2f}%")
            print()
    
    print("="*80)

if __name__ == "__main__":
    display_results()
