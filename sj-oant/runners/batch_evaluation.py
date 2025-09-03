#!/usr/bin/env python3
"""
Batch Evaluation System for Large-Scale TMM Testing

This script provides robust batch processing for large-scale evaluation with
checkpointing, error recovery, and progress tracking.

Usage:
    python runners/batch_evaluation.py --api-key YOUR_API_KEY --total 1000 --batch-size 100
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.squad_eval import SQuADEvaluator
from evaluation.advanced_metrics import AdvancedMetrics, MetricAggregator
from tmm_pipeline import create_tmm_pipeline
from baselines.simple_systems import DirectLLMBaseline, LongContextBaseline, SimpleRAGBaseline, BasicMemoryBaseline
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchEvaluator:
    """Robust batch evaluation system with checkpointing and recovery."""
    
    def __init__(self, api_key: str, output_dir: str = "results"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.evaluator = SQuADEvaluator()
        self.advanced_metrics = AdvancedMetrics()
        
        # Initialize systems
        self.systems = self._initialize_systems()
        
        # Checkpoint management
        self.checkpoint_file = self.output_dir / "batch_checkpoint.json"
        self.results_file = self.output_dir / "batch_results.json"
        
    def _initialize_systems(self) -> Dict[str, Any]:
        """Initialize all evaluation systems."""
        logger.info("Initializing evaluation systems...")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=self.api_key
        )
        
        systems = {
            "TMM Pipeline": create_tmm_pipeline(self.api_key),
            "DirectLLM": DirectLLMBaseline(llm),
            "LongContext": LongContextBaseline(llm),
            "SimpleRAG": SimpleRAGBaseline(llm),
            "BasicMemory": BasicMemoryBaseline(llm)
        }
        
        logger.info(f"Initialized {len(systems)} systems")
        return systems
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: {checkpoint['completed_batches']} batches completed")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save checkpoint."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {self.results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def evaluate_batch(self, examples: List, batch_id: int, system_name: str) -> Dict[str, Any]:
        """Evaluate a single batch for one system."""
        logger.info(f"Evaluating batch {batch_id} for {system_name} ({len(examples)} examples)")
        
        system = self.systems[system_name]
        batch_results = []
        errors = []
        
        batch_start_time = time.time()
        
        for i, example in enumerate(examples):
            try:
                # Reset system state for clean evaluation
                if hasattr(system, 'reset_memory'):
                    system.reset_memory()
                
                # Process example
                start_time = time.time()
                
                if system_name == "TMM Pipeline":
                    # TMM requires context setup
                    context_prompt = f"Please remember this context: {example.context}"
                    system.process(context_prompt)
                    response = system.process(example.question)
                else:
                    # Baseline systems
                    result = self.evaluator.evaluate_baseline_system(
                        system, [example], system_name, limit=1
                    )
                    # Get the actual response (simplified for batch processing)
                    if hasattr(system, 'last_response'):
                        response = system.last_response
                    else:
                        response = "Response not captured"  # Fallback
                
                processing_time = time.time() - start_time
                
                # Compute advanced metrics
                memory_records = []
                if system_name == "TMM Pipeline" and hasattr(system, 'memory_store'):
                    memory_records = getattr(system.memory_store, '_records', [])
                
                advanced_result = self.advanced_metrics.compute_all_metrics(
                    predicted=response,
                    ground_truth=example.answer,
                    context=example.context,
                    is_answerable=example.is_answerable,
                    memory_records=memory_records,
                    processing_time=processing_time
                )
                
                batch_results.append({
                    "example_id": example.id,
                    "response": response,
                    "ground_truth": example.answer,
                    "is_answerable": example.is_answerable,
                    "processing_time": processing_time,
                    "advanced_metrics": asdict(advanced_result)
                })
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(examples)} examples")
                
            except Exception as e:
                error_msg = f"Error on example {i} (ID: {example.id}): {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        batch_time = time.time() - batch_start_time
        
        # Aggregate metrics for this batch
        aggregator = MetricAggregator()
        for result in batch_results:
            metrics_dict = result["advanced_metrics"]
            # Convert dict back to AdvancedMetricResult (simplified)
            aggregator.results.append(type('MockResult', (), metrics_dict)())
        
        aggregated_metrics = aggregator.compute_aggregated_metrics()
        
        return {
            "batch_id": batch_id,
            "system_name": system_name,
            "total_examples": len(examples),
            "successful_examples": len(batch_results),
            "errors": len(errors),
            "batch_processing_time": batch_time,
            "examples_per_second": len(examples) / batch_time,
            "aggregated_metrics": aggregated_metrics,
            "detailed_results": batch_results,
            "error_details": errors
        }
    
    def run_batch_evaluation(self, total_examples: int, batch_size: int = 100,
                           resume: bool = True) -> Dict[str, Any]:
        """Run complete batch evaluation with checkpointing."""
        logger.info(f"Starting batch evaluation: {total_examples} examples, batch size {batch_size}")
        
        # Load data
        examples = self.evaluator.load_dataset("validation")[:total_examples]
        num_batches = (len(examples) + batch_size - 1) // batch_size
        
        logger.info(f"Total batches: {num_batches}")
        
        # Load checkpoint if resuming
        checkpoint = None
        if resume:
            checkpoint = self._load_checkpoint()
        
        if checkpoint:
            completed_batches = checkpoint["completed_batches"]
            results = checkpoint["results"]
            logger.info(f"Resuming from batch {completed_batches}")
        else:
            completed_batches = 0
            results = {system_name: [] for system_name in self.systems.keys()}
        
        # Process batches
        for batch_idx in range(completed_batches, num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(examples))
            batch_examples = examples[start_idx:end_idx]
            
            logger.info(f"\n--- BATCH {batch_idx + 1}/{num_batches} ---")
            logger.info(f"Examples {start_idx}-{end_idx-1} ({len(batch_examples)} examples)")
            
            # Evaluate each system on this batch
            batch_start_time = time.time()
            
            for system_name in self.systems.keys():
                try:
                    batch_result = self.evaluate_batch(batch_examples, batch_idx, system_name)
                    results[system_name].append(batch_result)
                    
                    logger.info(f"✅ {system_name}: {batch_result['successful_examples']}/{batch_result['total_examples']} successful")
                    
                except Exception as e:
                    logger.error(f"❌ {system_name} failed on batch {batch_idx}: {e}")
                    # Add empty result to maintain batch alignment
                    results[system_name].append({
                        "batch_id": batch_idx,
                        "system_name": system_name,
                        "error": str(e),
                        "total_examples": len(batch_examples),
                        "successful_examples": 0
                    })
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.1f}s")
            
            # Save checkpoint
            checkpoint_data = {
                "completed_batches": batch_idx + 1,
                "total_batches": num_batches,
                "total_examples": total_examples,
                "batch_size": batch_size,
                "results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
            self._save_checkpoint(checkpoint_data)
            
            # Estimate remaining time
            if batch_idx > completed_batches:
                avg_batch_time = batch_time  # Simplified
                remaining_batches = num_batches - (batch_idx + 1)
                estimated_remaining = remaining_batches * avg_batch_time
                logger.info(f"Estimated remaining time: {estimated_remaining/3600:.1f} hours")
        
        # Compute final aggregated results
        final_results = self._compute_final_results(results)
        
        # Save final results
        self._save_results(final_results)
        
        # Clean up checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        logger.info("✅ Batch evaluation completed successfully!")
        return final_results
    
    def _compute_final_results(self, batch_results: Dict[str, List]) -> Dict[str, Any]:
        """Compute final aggregated results across all batches."""
        final_results = {}
        
        for system_name, system_batches in batch_results.items():
            # Aggregate metrics across all batches
            all_examples = []
            total_processing_time = 0
            total_errors = 0
            
            for batch in system_batches:
                if "detailed_results" in batch:
                    all_examples.extend(batch["detailed_results"])
                    total_processing_time += batch.get("batch_processing_time", 0)
                    total_errors += batch.get("errors", 0)
            
            # Compute system-level aggregated metrics
            if all_examples:
                aggregator = MetricAggregator()
                for example_result in all_examples:
                    metrics_dict = example_result["advanced_metrics"]
                    aggregator.results.append(type('MockResult', (), metrics_dict)())
                
                system_metrics = aggregator.compute_aggregated_metrics()
            else:
                system_metrics = {}
            
            final_results[system_name] = {
                "total_examples": len(all_examples),
                "total_errors": total_errors,
                "success_rate": len(all_examples) / (len(all_examples) + total_errors) if (len(all_examples) + total_errors) > 0 else 0,
                "total_processing_time": total_processing_time,
                "avg_examples_per_second": len(all_examples) / total_processing_time if total_processing_time > 0 else 0,
                "aggregated_metrics": system_metrics,
                "batch_details": system_batches
            }
        
        return {
            "systems": final_results,
            "evaluation_summary": self._generate_evaluation_summary(final_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level evaluation summary."""
        summary = {
            "total_systems": len(results),
            "best_accuracy_system": None,
            "fastest_system": None,
            "most_efficient_system": None
        }
        
        best_acc = 0
        best_speed = float('inf')
        best_efficiency = 0
        
        for system_name, system_data in results.items():
            metrics = system_data.get("aggregated_metrics", {})
            
            # Best accuracy
            acc = metrics.get("composite_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                summary["best_accuracy_system"] = system_name
            
            # Fastest system
            speed = system_data.get("avg_examples_per_second", 0)
            if speed > 0 and (1/speed) < best_speed:
                best_speed = 1/speed
                summary["fastest_system"] = system_name
            
            # Most efficient
            efficiency = metrics.get("composite_efficiency", 0)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                summary["most_efficient_system"] = system_name
        
        summary["best_accuracy_score"] = best_acc
        summary["best_speed"] = 1/best_speed if best_speed != float('inf') else 0
        summary["best_efficiency_score"] = best_efficiency
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation on TMM systems")
    parser.add_argument("--api-key", required=True, help="Google API key")
    parser.add_argument("--total", type=int, default=100, help="Total examples to evaluate")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    
    args = parser.parse_args()
    
    print("🚀 BATCH EVALUATION SYSTEM")
    print("=" * 60)
    print(f"Total examples: {args.total}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        evaluator = BatchEvaluator(args.api_key, args.output_dir)
        results = evaluator.run_batch_evaluation(
            total_examples=args.total,
            batch_size=args.batch_size,
            resume=not args.no_resume
        )
        
        print("\n✅ EVALUATION COMPLETE!")
        print(f"Results saved to: {evaluator.results_file}")
        
        # Print summary
        summary = results.get("evaluation_summary", {})
        print(f"\n📊 SUMMARY:")
        print(f"Best accuracy: {summary.get('best_accuracy_system', 'N/A')}")
        print(f"Fastest system: {summary.get('fastest_system', 'N/A')}")
        print(f"Most efficient: {summary.get('most_efficient_system', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
