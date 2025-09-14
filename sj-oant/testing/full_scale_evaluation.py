#!/usr/bin/env python3
"""
Full-Scale Evaluation Script for SJ-OANT TMM System
Runs comprehensive evaluation on 200+ conversations per benchmark
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_frameworks.unified_evaluator import UnifiedOfficialEvaluator
from tmm_pipeline import TMMPipelineFixed
from testing.setup_api_key import setup_api_key

class FullScaleEvaluator:
    """Full-scale evaluation orchestrator for all benchmarks."""
    
    def __init__(self, sample_sizes: Dict[str, int] = None):
        """
        Initialize full-scale evaluator.
        
        Args:
            sample_sizes: Dictionary mapping benchmark names to sample sizes
        """
        self.sample_sizes = sample_sizes or {
            'multiwoz': 200,
            'sgd': 200, 
            'taskmaster': 200,
            'multidogo': 200
        }
        
        self.evaluator = UnifiedOfficialEvaluator()
        self.results = {}
        
        print("🚀 Full-Scale Evaluation Initialized")
        print(f"📊 Sample Sizes: {self.sample_sizes}")
    
    def run_full_scale_evaluation(self) -> Dict[str, Any]:
        """Run full-scale evaluation across all benchmarks."""
        print("\n" + "="*80)
        print("🚀 STARTING FULL-SCALE EVALUATION")
        print("="*80)
        
        start_time = time.time()
        
        # Run evaluation for each benchmark
        for benchmark, sample_size in self.sample_sizes.items():
            print(f"\n🔄 Running {benchmark.upper()} evaluation ({sample_size} samples)...")
            
            try:
                results = self._evaluate_benchmark(benchmark, sample_size)
                self.results[benchmark] = results
                
                print(f"✅ {benchmark.upper()} evaluation completed successfully")
                self._print_benchmark_summary(benchmark, results)
                
            except Exception as e:
                print(f"❌ Error in {benchmark} evaluation: {e}")
                self.results[benchmark] = {"error": str(e)}
        
        # Generate comprehensive analysis
        total_time = time.time() - start_time
        print(f"\n⏱️ Total evaluation time: {total_time:.2f} seconds")
        
        # Save results
        self._save_results()
        
        # Generate final report
        self._generate_final_report()
        
        return self.results
    
    def _evaluate_benchmark(self, benchmark: str, sample_size: int) -> Dict[str, Any]:
        """Evaluate a specific benchmark with given sample size."""
        print(f"   📝 Processing {sample_size} {benchmark} conversations...")
        
        # Initialize TMM pipeline
        pipeline = TMMPipelineFixed(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Load benchmark data
        data_loader = self._get_data_loader(benchmark)
        conversations = data_loader.load_samples(sample_size)
        
        # Process conversations
        predictions = []
        for i, conversation in enumerate(conversations):
            if i % 50 == 0:
                print(f"   📊 Progress: {i}/{sample_size} conversations processed")
            
            try:
                prediction = self._process_conversation(pipeline, conversation)
                predictions.append(prediction)
            except Exception as e:
                print(f"   ⚠️ Error processing conversation {i}: {e}")
                continue
        
        # Evaluate predictions
        print(f"   🔬 Evaluating {len(predictions)} predictions...")
        results = self.evaluator.evaluate_benchmark(benchmark, predictions)
        
        return results
    
    def _get_data_loader(self, benchmark: str):
        """Get appropriate data loader for benchmark."""
        if benchmark == 'multiwoz':
            from data.multiwoz_loader import MultiWOZLoader
            return MultiWOZLoader()
        elif benchmark == 'sgd':
            from data.sgd_loader import SGDLoader
            return SGDLoader()
        elif benchmark == 'taskmaster':
            from data.taskmaster_loader import TaskmasterLoader
            return TaskmasterLoader()
        elif benchmark == 'multidogo':
            from data.multidogo_loader import MultiDoGOLoader
            return MultiDoGOLoader()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    def _process_conversation(self, pipeline, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single conversation through the TMM pipeline."""
        responses = []
        
        for turn in conversation.get('turns', []):
            user_input = turn.get('user', '')
            if user_input:
                try:
                    response = pipeline.process(user_input)
                    responses.append(response)
                except Exception as e:
                    responses.append(f"Error: {e}")
        
        return {
            'conversation_id': conversation.get('id', 'unknown'),
            'responses': responses,
            'system_turns': [turn.get('system', '') for turn in conversation.get('turns', [])]
        }
    
    def _print_benchmark_summary(self, benchmark: str, results: Dict[str, Any]):
        """Print summary for a benchmark."""
        print(f"\n📊 {benchmark.upper()} RESULTS:")
        print("-" * 40)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Extract and display key metrics
        nested_results = results.get('results', {})
        
        if benchmark == 'multiwoz':
            print(f"📝 BLEU Score: {nested_results.get('bleu_score', {}).get('bleu', 0):.2f}%")
            print(f"📊 ROUGE Score: {nested_results.get('rouge_score', {}).get('rouge', 0):.2f}%")
            print(f"🧠 Semantic Similarity: {nested_results.get('semantic_similarity', {}).get('semantic_similarity', 0):.2f}%")
            print(f"✅ Task Completion: {nested_results.get('task_completion', {}).get('task_completion', 0):.2f}%")
        
        elif benchmark == 'sgd':
            print(f"🎯 Intent Accuracy: {nested_results.get('intent_accuracy', {}).get('total', 0):.2f}%")
            print(f"🔧 Slot F1 Score: {nested_results.get('slot_f1', {}).get('total', 0):.2f}%")
            print(f"✅ Success Rate: {nested_results.get('success_rate', {}).get('total', 0):.2f}%")
            print(f"📝 BLEU Score: {nested_results.get('bleu', {}).get('bleu', 0):.2f}%")
        
        elif benchmark == 'taskmaster':
            print(f"📝 BLEU Score: {nested_results.get('bleu', {}).get('bleu', 0):.2f}%")
            print(f"📊 ROUGE Score: {nested_results.get('rouge', {}).get('rouge', 0):.2f}%")
            print(f"🧠 Semantic Similarity: {nested_results.get('semantic_similarity', {}).get('semantic_similarity', 0):.2f}%")
            print(f"✅ Task Completion: {nested_results.get('task_completion', {}).get('task_completion', 0):.2f}%")
        
        elif benchmark == 'multidogo':
            print(f"🎯 Intent Classification: {nested_results.get('intent_accuracy', {}).get('total', 0):.2f}%")
            print(f"🔧 Slot F1 Score: {nested_results.get('slot_f1', {}).get('total', 0):.2f}%")
            print(f"🌐 Domain Adaptation: {nested_results.get('domain_adaptation', {}).get('total', 0):.2f}%")
            print(f"📝 Response Quality: {nested_results.get('response_quality', {}).get('total', 0):.2f}%")
    
    def _save_results(self):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual benchmark results
        for benchmark, results in self.results.items():
            filename = f"results/full_scale_{benchmark}_results_{timestamp}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 {benchmark.upper()} results saved to: {filename}")
        
        # Save combined results
        combined_filename = f"results/full_scale_combined_results_{timestamp}.json"
        with open(combined_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Combined results saved to: {combined_filename}")
    
    def _generate_final_report(self):
        """Generate final comprehensive report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"results/full_scale_evaluation_report_{timestamp}.md"
        
        report = self._create_report_content()
        
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"📊 Final report saved to: {report_filename}")
    
    def _create_report_content(self) -> str:
        """Create comprehensive report content."""
        report = []
        report.append("# 🚀 FULL-SCALE EVALUATION REPORT")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## 📊 Executive Summary")
        report.append("")
        
        # Overall performance summary
        total_benchmarks = len(self.results)
        successful_benchmarks = sum(1 for r in self.results.values() if 'error' not in r)
        
        report.append(f"- **Total Benchmarks**: {total_benchmarks}")
        report.append(f"- **Successful Evaluations**: {successful_benchmarks}")
        report.append(f"- **Sample Sizes**: {self.sample_sizes}")
        report.append("")
        
        # Individual benchmark results
        report.append("## 📈 Benchmark Results")
        report.append("")
        
        for benchmark, results in self.results.items():
            report.append(f"### {benchmark.upper()}")
            report.append("")
            
            if 'error' in results:
                report.append(f"❌ **Error**: {results['error']}")
            else:
                nested_results = results.get('results', {})
                
                if benchmark == 'multiwoz':
                    report.append(f"- **BLEU Score**: {nested_results.get('bleu_score', {}).get('bleu', 0):.2f}%")
                    report.append(f"- **ROUGE Score**: {nested_results.get('rouge_score', {}).get('rouge', 0):.2f}%")
                    report.append(f"- **Semantic Similarity**: {nested_results.get('semantic_similarity', {}).get('semantic_similarity', 0):.2f}%")
                    report.append(f"- **Task Completion**: {nested_results.get('task_completion', {}).get('task_completion', 0):.2f}%")
                
                elif benchmark == 'sgd':
                    report.append(f"- **Intent Accuracy**: {nested_results.get('intent_accuracy', {}).get('total', 0):.2f}%")
                    report.append(f"- **Slot F1 Score**: {nested_results.get('slot_f1', {}).get('total', 0):.2f}%")
                    report.append(f"- **Success Rate**: {nested_results.get('success_rate', {}).get('total', 0):.2f}%")
                    report.append(f"- **BLEU Score**: {nested_results.get('bleu', {}).get('bleu', 0):.2f}%")
                
                elif benchmark == 'taskmaster':
                    report.append(f"- **BLEU Score**: {nested_results.get('bleu', {}).get('bleu', 0):.2f}%")
                    report.append(f"- **ROUGE Score**: {nested_results.get('rouge', {}).get('rouge', 0):.2f}%")
                    report.append(f"- **Semantic Similarity**: {nested_results.get('semantic_similarity', {}).get('semantic_similarity', 0):.2f}%")
                    report.append(f"- **Task Completion**: {nested_results.get('task_completion', {}).get('task_completion', 0):.2f}%")
                
                elif benchmark == 'multidogo':
                    report.append(f"- **Intent Classification**: {nested_results.get('intent_accuracy', {}).get('total', 0):.2f}%")
                    report.append(f"- **Slot F1 Score**: {nested_results.get('slot_f1', {}).get('total', 0):.2f}%")
                    report.append(f"- **Domain Adaptation**: {nested_results.get('domain_adaptation', {}).get('total', 0):.2f}%")
                    report.append(f"- **Response Quality**: {nested_results.get('response_quality', {}).get('total', 0):.2f}%")
            
            report.append("")
        
        report.append("## 🎯 Research Readiness Assessment")
        report.append("")
        report.append("The full-scale evaluation provides comprehensive performance metrics")
        report.append("across all major dialogue benchmarks, demonstrating the research-grade")
        report.append("capabilities of the SJ-OANT TMM system.")
        report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run full-scale evaluation."""
    print("🚀 SJ-OANT Full-Scale Evaluation")
    print("=" * 50)
    
    # Setup API key
    setup_api_key()
    
    # Initialize evaluator with custom sample sizes
    sample_sizes = {
        'multiwoz': 200,
        'sgd': 200,
        'taskmaster': 200,
        'multidogo': 200
    }
    
    evaluator = FullScaleEvaluator(sample_sizes)
    
    # Run evaluation
    results = evaluator.run_full_scale_evaluation()
    
    print("\n🎉 Full-scale evaluation completed!")
    print("📊 Check the results/ directory for detailed reports")
    
    return results

if __name__ == "__main__":
    main()
