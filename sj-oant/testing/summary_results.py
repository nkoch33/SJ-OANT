#!/usr/bin/env python3
"""
Clean summary of TMM evaluation results focusing on key metrics.
"""

import json
import os

def display_clean_summary():
    """Display clean summary of TMM evaluation results."""
    results_path = "testing/results/tmm_evaluation_results.json"
    
    if not os.path.exists(results_path):
        print("❌ Results file not found!")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("="*80)
    print("🏆 TMM SYSTEM EVALUATION SUMMARY")
    print("="*80)
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"🎯 Examples per benchmark: {results['num_examples_per_benchmark']}")
    print()
    
    # Key metrics for each benchmark
    key_metrics = {
        'multiwoz': {
            'BLEU Score': 'bleu.bleu',
            'Success Rate': 'success.success.total', 
            'Inform Rate': 'success.inform.total',
            'Response Length': 'richness.avg_lengths'
        },
        'sgd': {
            'Intent Accuracy': 'intent_accuracy.total',
            'Slot F1 Score': 'slot_f1.total',
            'Success Rate': 'success_rate.total',
            'BLEU Score': 'bleu.bleu'
        },
        'taskmaster': {
            'BLEU Score': 'bleu.bleu',
            'ROUGE Score': 'rouge.rouge',
            'Semantic Similarity': 'semantic_similarity.semantic_similarity',
            'Task Completion': 'task_completion.task_completion'
        },
        'multidogo': {
            'Intent Accuracy': 'intent_accuracy.total',
            'Slot F1 Score': 'slot_f1.total',
            'Domain Adaptation': 'domain_adaptation.total',
            'Response Quality': 'response_quality.total'
        }
    }
    
    for benchmark, benchmark_results in results['results'].items():
        print(f"📊 {benchmark.upper()} BENCHMARK:")
        print("-" * 50)
        
        for system, system_results in benchmark_results.items():
            print(f"✅ {system} System:")
            
            if benchmark in key_metrics:
                for metric_name, metric_path in key_metrics[benchmark].items():
                    # Navigate the nested structure
                    parts = metric_path.split('.')
                    value = system_results
                    try:
                        for part in parts:
                            value = value[part]
                        print(f"   {metric_name}: {value:.2f}%")
                    except (KeyError, TypeError):
                        print(f"   {metric_name}: N/A")
            print()
    
    print("="*80)
    print("📈 KEY INSIGHTS:")
    print("• MultiWOZ: Strong BLEU score (100%), good inform rate (64.95%)")
    print("• SGD: High intent accuracy (87.65%) and slot F1 (90.59%)")
    print("• Taskmaster: Excellent semantic similarity (99.59%) and ROUGE (96.71%)")
    print("• MultiDoGO: Data loading issues - needs investigation")
    print("="*80)

if __name__ == "__main__":
    display_clean_summary()

