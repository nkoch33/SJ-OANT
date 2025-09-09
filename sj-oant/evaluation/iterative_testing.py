#!/usr/bin/env python3
"""
Iterative Testing and Training Pipeline for TMM System.
Allows for small-scale testing, analysis, and iterative improvement.
"""

import json
import logging
import sys
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import time
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.multiwoz_streamlined_eval import StreamlinedMultiWOZEvaluator
from tmm_pipeline import TMMPipelineFixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IterativeTestingPipeline:
    """Pipeline for iterative testing and improvement of TMM system."""
    
    def __init__(self, api_key: str):
        """Initialize the testing pipeline."""
        self.api_key = api_key
        self.evaluator = StreamlinedMultiWOZEvaluator(api_key)
        self.test_results = []
        self.improvement_history = []
        
    def run_small_scale_test(self, num_dialogues: int = 10, test_name: str = None) -> Dict[str, Any]:
        """Run a small-scale test with specified number of dialogues."""
        if test_name is None:
            test_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Running small-scale test: {test_name} with {num_dialogues} dialogues")
        
        start_time = time.time()
        
        # Run evaluation
        results = self.evaluator.run_evaluation(
            data_path='data/MULTIWOZ2.4/MULTIWOZ2.4/data.json',
            limit=num_dialogues,
            output_path=f'results/{test_name}_results.json'
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Store test results
        test_result = {
            'test_name': test_name,
            'num_dialogues': num_dialogues,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        self.test_results.append(test_result)
        
        # Print summary
        self._print_test_summary(test_result)
        
        return test_result
        
    def _print_test_summary(self, test_result: Dict[str, Any]):
        """Print a summary of test results."""
        print("\n" + "="*80)
        print(f"📊 TEST SUMMARY: {test_result['test_name']}")
        print("="*80)
        print(f"Dialogues: {test_result['num_dialogues']}")
        print(f"Duration: {test_result['duration']:.2f} seconds")
        print(f"Timestamp: {test_result['timestamp']}")
        
        results = test_result['results']
        
        # BLEU
        if results.get('bleu'):
            bleu_score = results['bleu'].get('bleu', 0.0)
            print(f"BLEU Score: {bleu_score:.2f}")
        
        # Success/Inform
        if results.get('success'):
            success_rate = results['success'].get('success', {}).get('total', 0.0)
            inform_rate = results['success'].get('inform', {}).get('total', 0.0)
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Inform Rate: {inform_rate:.1f}%")
        
        # Richness
        if results.get('richness'):
            richness = results['richness']
            print(f"Entropy: {richness.get('entropy', 0.0):.3f}")
            print(f"Avg Length: {richness.get('avg_lengths', 0.0):.1f}")
            print(f"Unique Unigrams: {richness.get('num_unigrams', 0)}")
        
        print("="*80)
        
    def compare_tests(self, test1_name: str, test2_name: str):
        """Compare two test results."""
        test1 = next((t for t in self.test_results if t['test_name'] == test1_name), None)
        test2 = next((t for t in self.test_results if t['test_name'] == test2_name), None)
        
        if not test1 or not test2:
            print("Error: One or both tests not found")
            return
        
        print("\n" + "="*80)
        print(f"📈 COMPARISON: {test1_name} vs {test2_name}")
        print("="*80)
        
        # Compare metrics
        metrics = ['bleu', 'success', 'inform', 'entropy', 'avg_lengths', 'num_unigrams']
        
        for metric in metrics:
            val1 = self._extract_metric_value(test1['results'], metric)
            val2 = self._extract_metric_value(test2['results'], metric)
            
            if val1 is not None and val2 is not None:
                change = val2 - val1
                change_pct = (change / val1 * 100) if val1 != 0 else 0
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                print(f"{metric.upper()}: {val1:.2f} → {val2:.2f} ({change:+.2f}, {change_pct:+.1f}%) {direction}")
        
        print("="*80)
        
    def _extract_metric_value(self, results: Dict[str, Any], metric: str) -> float:
        """Extract a specific metric value from results."""
        if metric == 'bleu':
            return results.get('bleu', {}).get('bleu', 0.0)
        elif metric == 'success':
            return results.get('success', {}).get('success', {}).get('total', 0.0)
        elif metric == 'inform':
            return results.get('success', {}).get('inform', {}).get('total', 0.0)
        elif metric == 'entropy':
            return results.get('richness', {}).get('entropy', 0.0)
        elif metric == 'avg_lengths':
            return results.get('richness', {}).get('avg_lengths', 0.0)
        elif metric == 'num_unigrams':
            return results.get('richness', {}).get('num_unigrams', 0)
        return None
        
    def run_improvement_cycle(self, num_dialogues: int = 5, iterations: int = 3):
        """Run multiple iterations of testing for improvement tracking."""
        print(f"\n🚀 Starting improvement cycle: {iterations} iterations with {num_dialogues} dialogues each")
        
        for i in range(iterations):
            test_name = f"improvement_cycle_{i+1}"
            print(f"\n--- Iteration {i+1}/{iterations} ---")
            
            test_result = self.run_small_scale_test(num_dialogues, test_name)
            
            # Store improvement history
            self.improvement_history.append({
                'iteration': i + 1,
                'test_result': test_result
            })
            
            # Compare with previous iteration if available
            if i > 0:
                prev_test_name = f"improvement_cycle_{i}"
                self.compare_tests(prev_test_name, test_name)
        
        # Print overall improvement summary
        self._print_improvement_summary()
        
    def _print_improvement_summary(self):
        """Print a summary of the improvement cycle."""
        if len(self.improvement_history) < 2:
            return
            
        print("\n" + "="*80)
        print("📊 IMPROVEMENT CYCLE SUMMARY")
        print("="*80)
        
        first_test = self.improvement_history[0]['test_result']
        last_test = self.improvement_history[-1]['test_result']
        
        print(f"Total Iterations: {len(self.improvement_history)}")
        print(f"Total Duration: {sum(h['test_result']['duration'] for h in self.improvement_history):.2f} seconds")
        
        # Compare first vs last
        print(f"\nFirst vs Last Iteration:")
        self.compare_tests(first_test['test_name'], last_test['test_name'])
        
        print("="*80)
        
    def save_test_history(self, filename: str = None):
        """Save test history to file."""
        if filename is None:
            filename = f"results/test_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        history_data = {
            'test_results': self.test_results,
            'improvement_history': self.improvement_history,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Test history saved to {filename}")
        
    def load_test_history(self, filename: str):
        """Load test history from file."""
        with open(filename, 'r') as f:
            history_data = json.load(f)
        
        self.test_results = history_data.get('test_results', [])
        self.improvement_history = history_data.get('improvement_history', [])
        
        logger.info(f"Test history loaded from {filename}")
        
    def get_best_test(self, metric: str = 'success') -> Dict[str, Any]:
        """Get the test with the best performance on a specific metric."""
        if not self.test_results:
            return None
            
        best_test = None
        best_value = float('-inf')
        
        for test in self.test_results:
            value = self._extract_metric_value(test['results'], metric)
            if value is not None and value > best_value:
                best_value = value
                best_test = test
        
        return best_test


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Iterative Testing Pipeline for TMM System')
    parser.add_argument('--api-key', type=str, required=True, help='Google API key for TMM system')
    parser.add_argument('--num-dialogues', type=int, default=5, help='Number of dialogues per test')
    parser.add_argument('--iterations', type=int, default=3, help='Number of improvement iterations')
    parser.add_argument('--test-name', type=str, default=None, help='Name for single test')
    parser.add_argument('--mode', type=str, choices=['single', 'cycle'], default='cycle',
                       help='Testing mode: single test or improvement cycle')
    
    args = parser.parse_args()
    
    # Create testing pipeline
    pipeline = IterativeTestingPipeline(args.api_key)
    
    if args.mode == 'single':
        # Run single test
        test_result = pipeline.run_small_scale_test(args.num_dialogues, args.test_name)
        print(f"\n✅ Single test completed: {test_result['test_name']}")
        
    elif args.mode == 'cycle':
        # Run improvement cycle
        pipeline.run_improvement_cycle(args.num_dialogues, args.iterations)
        print(f"\n✅ Improvement cycle completed: {args.iterations} iterations")
    
    # Save test history
    pipeline.save_test_history()
    
    print(f"\n🎯 Testing pipeline completed successfully!")


if __name__ == "__main__":
    main()
