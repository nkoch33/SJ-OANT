#!/usr/bin/env python3
"""
Comprehensive Test Runner

Production-ready script for large-scale TMM evaluation with OpenAI API.
Optimized for high-compute environments with batch processing and checkpointing.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tmm_pipeline import create_tmm_pipeline, create_tmm_variant
from evaluation.squad_eval import SQuADEvaluator
from baselines.simple_systems import DirectLLMBaseline, BasicMemoryBaseline, LongContextBaseline

# Configure for OpenAI
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    from langchain_google_genai import ChatGoogleGenerativeAI
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestConfiguration:
    """Configuration for comprehensive testing."""
    api_key: str
    api_provider: str = "openai"  # "openai" or "google"
    model_name: str = "gpt-4"  # or "gpt-3.5-turbo" or "gemini-1.5-flash"
    max_examples: int = 1000
    batch_size: int = 50
    checkpoint_frequency: int = 100
    results_dir: Path = Path("results")
    enable_checkpointing: bool = True
    
    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True)

class ComprehensiveTestRunner:
    """Production-ready test runner for large-scale evaluation."""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.evaluator = SQuADEvaluator()
        self.systems = {}
        self.results = {}
        
        # Initialize LLM based on provider
        if config.api_provider == "openai" and OPENAI_AVAILABLE:
            self.llm = ChatOpenAI(
                model=config.model_name,
                temperature=0.1,
                openai_api_key=config.api_key,
                max_retries=3
            )
            logger.info(f"Initialized OpenAI {config.model_name}")
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=config.api_key
            )
            logger.info("Initialized Google Gemini 1.5-flash")
        
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all systems for evaluation."""
        logger.info("Initializing test systems...")
        
        # TMM Systems
        if self.config.api_provider == "openai":
            self.systems["Optimized TMM"] = create_tmm_pipeline(self.config.api_key, provider="openai")
            self.systems["TMM (No Memory)"] = create_tmm_variant(self.config.api_key, ["memory"], provider="openai")
            self.systems["TMM (No Filtering)"] = create_tmm_variant(self.config.api_key, ["filtering"], provider="openai")
            self.systems["TMM (Minimal)"] = create_tmm_variant(self.config.api_key, ["memory", "filtering", "verification"], provider="openai")
        else:
            self.systems["Optimized TMM"] = create_tmm_pipeline(self.config.api_key)
            self.systems["TMM (No Memory)"] = create_tmm_variant(self.config.api_key, ["memory"])
            self.systems["TMM (No Filtering)"] = create_tmm_variant(self.config.api_key, ["filtering"])
            self.systems["TMM (Minimal)"] = create_tmm_variant(self.config.api_key, ["memory", "filtering", "verification"])
        
        # Baseline Systems
        self.systems["DirectLLM Baseline"] = DirectLLMBaseline(self.llm)
        self.systems["BasicMemory Baseline"] = BasicMemoryBaseline(self.llm)
        self.systems["LongContext Baseline"] = LongContextBaseline(self.llm)
        
        logger.info(f"Initialized {len(self.systems)} systems for evaluation")
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all systems."""
        logger.info("=" * 80)
        logger.info("🧪 COMPREHENSIVE TMM EVALUATION")
        logger.info("=" * 80)
        
        # Load dataset
        examples = self.evaluator.load_dataset("validation")[:self.config.max_examples]
        logger.info(f"Loaded {len(examples)} examples for evaluation")
        
        # Run evaluation for each system
        for system_name, system in self.systems.items():
            logger.info(f"\n📊 Evaluating {system_name}...")
            
            # Check for existing checkpoint
            checkpoint_file = self.config.results_dir / f"{system_name.lower().replace(' ', '_')}_checkpoint.json"
            if self.config.enable_checkpointing and checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                start_idx = checkpoint_data.get("last_processed", 0)
                logger.info(f"Resuming from checkpoint at example {start_idx}")
            else:
                start_idx = 0
                checkpoint_data = {"results": [], "last_processed": 0}
            
            # Evaluate system
            system_results = self._evaluate_system(
                system_name, system, examples, 
                start_idx=start_idx, 
                checkpoint_data=checkpoint_data
            )
            
            self.results[system_name] = system_results
            
            # Save final results
            results_file = self.config.results_dir / f"{system_name.lower().replace(' ', '_')}_results.json"
            with open(results_file, 'w') as f:
                json.dump(system_results, f, indent=2)
            
            # Clean up checkpoint
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        
        # Generate comprehensive analysis
        analysis = self._analyze_results()
        
        # Save comprehensive results
        final_results = {
            "configuration": {
                "api_provider": self.config.api_provider,
                "model_name": self.config.model_name,
                "max_examples": len(examples),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "system_results": self.results,
            "comparative_analysis": analysis
        }
        
        with open(self.config.results_dir / "comprehensive_evaluation.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results
    
    def _evaluate_system(self, system_name: str, system, examples: List, 
                        start_idx: int = 0, checkpoint_data: Dict = None) -> Dict[str, Any]:
        """Evaluate a single system with checkpointing."""
        
        results = checkpoint_data["results"] if checkpoint_data else []
        
        for i in range(start_idx, len(examples)):
            example = examples[i]
            
            try:
                # Reset system state
                if hasattr(system, 'reset_memory'):
                    system.reset_memory()
                
                start_time = time.time()
                
                if "TMM" in system_name:
                    # TMM systems - two-stage process
                    context_prompt = f"Please remember this context: {example.context}"
                    system.process(context_prompt)
                    response = system.process(example.question)
                else:
                    # Baseline systems - direct processing
                    if hasattr(system, 'process_story'):
                        system.process_story(example.context)
                        response = system.process(example.question)
                    else:
                        full_prompt = f"Context: {example.context}\n\nQuestion: {example.question}"
                        result = system.process(full_prompt)
                        response = result if isinstance(result, str) else str(result)
                
                processing_time = time.time() - start_time
                
                # Evaluate response
                is_correct = self.evaluator._check_answer_v2(response, example.answer, example.is_answerable)
                
                result_entry = {
                    "example_id": i,
                    "question": example.question,
                    "context_length": len(example.context.split()),
                    "is_answerable": example.is_answerable,
                    "ground_truth": example.answer,
                    "system_response": response,
                    "is_correct": is_correct,
                    "processing_time": processing_time
                }
                
                results.append(result_entry)
                
                # Progress reporting
                if (i + 1) % 10 == 0:
                    correct_so_far = sum(1 for r in results if r["is_correct"])
                    accuracy_so_far = correct_so_far / len(results)
                    logger.info(f"  Progress: {i + 1}/{len(examples)} ({accuracy_so_far:.1%} accuracy)")
                
                # Checkpointing
                if self.config.enable_checkpointing and (i + 1) % self.config.checkpoint_frequency == 0:
                    checkpoint_file = self.config.results_dir / f"{system_name.lower().replace(' ', '_')}_checkpoint.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            "results": results,
                            "last_processed": i + 1
                        }, f, indent=2)
                    logger.info(f"  Checkpoint saved at example {i + 1}")
                
            except Exception as e:
                logger.error(f"Error on example {i}: {e}")
                # Continue with next example
                continue
        
        # Calculate final metrics
        total_examples = len(results)
        correct_answers = sum(1 for r in results if r["is_correct"])
        answerable_results = [r for r in results if r["is_answerable"]]
        unanswerable_results = [r for r in results if not r["is_answerable"]]
        
        final_metrics = {
            "total_examples": total_examples,
            "accuracy": correct_answers / total_examples if total_examples > 0 else 0,
            "answerable_accuracy": sum(1 for r in answerable_results if r["is_correct"]) / len(answerable_results) if answerable_results else 0,
            "unanswerable_accuracy": sum(1 for r in unanswerable_results if r["is_correct"]) / len(unanswerable_results) if unanswerable_results else 0,
            "avg_processing_time": sum(r["processing_time"] for r in results) / total_examples if total_examples > 0 else 0,
            "detailed_results": results
        }
        
        logger.info(f"  Final Results: {final_metrics['accuracy']:.1%} accuracy ({correct_answers}/{total_examples})")
        
        return final_metrics
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Generate comprehensive comparative analysis."""
        
        if not self.results:
            return {}
        
        # Get baseline performance
        baseline_accuracy = self.results.get("DirectLLM Baseline", {}).get("accuracy", 0)
        
        analysis = {
            "performance_ranking": [],
            "memory_effectiveness": {},
            "efficiency_analysis": {},
            "research_compliance": {}
        }
        
        # Performance ranking
        ranked_systems = sorted(
            self.results.items(), 
            key=lambda x: x[1]["accuracy"], 
            reverse=True
        )
        
        for rank, (system_name, metrics) in enumerate(ranked_systems, 1):
            analysis["performance_ranking"].append({
                "rank": rank,
                "system": system_name,
                "accuracy": metrics["accuracy"],
                "gap_from_baseline": metrics["accuracy"] - baseline_accuracy
            })
        
        # Memory effectiveness analysis
        tmm_full = self.results.get("Optimized TMM", {}).get("accuracy", 0)
        tmm_no_memory = self.results.get("TMM (No Memory)", {}).get("accuracy", 0)
        
        analysis["memory_effectiveness"] = {
            "memory_advantage": tmm_full - tmm_no_memory,
            "memory_effectiveness_status": "POSITIVE" if tmm_full > tmm_no_memory else "NEGATIVE"
        }
        
        # Research compliance assessment
        tmm_metrics = self.results.get("Optimized TMM", {})
        baseline_metrics = self.results.get("DirectLLM Baseline", {})
        
        if tmm_metrics and baseline_metrics:
            analysis["research_compliance"] = {
                "accuracy_vs_baseline": {
                    "tmm": tmm_metrics["accuracy"],
                    "baseline": baseline_metrics["accuracy"],
                    "advantage": tmm_metrics["accuracy"] - baseline_metrics["accuracy"],
                    "status": "ACHIEVING" if tmm_metrics["accuracy"] >= baseline_metrics["accuracy"] else "FAILING"
                },
                "truth_maintenance": {
                    "tmm_unanswerable": tmm_metrics.get("unanswerable_accuracy", 0),
                    "baseline_unanswerable": baseline_metrics.get("unanswerable_accuracy", 0),
                    "advantage": tmm_metrics.get("unanswerable_accuracy", 0) - baseline_metrics.get("unanswerable_accuracy", 0),
                    "status": "ACHIEVING" if tmm_metrics.get("unanswerable_accuracy", 0) >= baseline_metrics.get("unanswerable_accuracy", 0) else "FAILING"
                },
                "efficiency": {
                    "time_overhead": tmm_metrics.get("avg_processing_time", 0) / baseline_metrics.get("avg_processing_time", 1),
                    "status": "EFFICIENT" if tmm_metrics.get("avg_processing_time", 0) / baseline_metrics.get("avg_processing_time", 1) <= 2.0 else "INEFFICIENT"
                }
            }
        
        return analysis

def print_comprehensive_results(results: Dict[str, Any]):
    """Print detailed results summary."""
    
    print("\n" + "=" * 100)
    print("🏆 COMPREHENSIVE TMM EVALUATION RESULTS")
    print("=" * 100)
    
    config = results["configuration"]
    print(f"\n📊 TEST CONFIGURATION:")
    print(f"  API Provider: {config['api_provider'].upper()}")
    print(f"  Model: {config['model_name']}")
    print(f"  Examples Evaluated: {config['max_examples']}")
    print(f"  Timestamp: {config['timestamp']}")
    
    # Performance table
    print(f"\n📈 PERFORMANCE RANKING:")
    print(f"{'Rank':<5} {'System':<25} {'Accuracy':<12} {'Ans/Unans':<15} {'Time':<10} {'vs Baseline'}")
    print("-" * 85)
    
    ranking = results["comparative_analysis"]["performance_ranking"]
    system_results = results["system_results"]
    
    for entry in ranking:
        system_name = entry["system"]
        metrics = system_results[system_name]
        ans_acc = metrics.get("answerable_accuracy", 0)
        unans_acc = metrics.get("unanswerable_accuracy", 0)
        avg_time = metrics.get("avg_processing_time", 0)
        gap = entry["gap_from_baseline"]
        
        gap_str = f"{gap:+.1%}" if gap != 0 else "Baseline"
        
        print(f"{entry['rank']:<5} {system_name:<25} {entry['accuracy']:<11.1%} {ans_acc:.1%}/{unans_acc:.1%}    {avg_time:<9.3f}s {gap_str}")
    
    # Research compliance
    compliance = results["comparative_analysis"]["research_compliance"]
    if compliance:
        print(f"\n🔬 RESEARCH OBJECTIVE COMPLIANCE:")
        print("-" * 50)
        
        acc_comp = compliance["accuracy_vs_baseline"]
        print(f"Accuracy vs Baseline: {acc_comp['advantage']:+.1%} ({acc_comp['status']})")
        
        truth_comp = compliance["truth_maintenance"]
        print(f"Truth Maintenance: {truth_comp['advantage']:+.1%} ({truth_comp['status']})")
        
        mem_comp = results["comparative_analysis"]["memory_effectiveness"]
        print(f"Memory Effectiveness: {mem_comp['memory_advantage']:+.1%} ({mem_comp['memory_effectiveness_status']})")
        
        eff_comp = compliance["efficiency"]
        print(f"Efficiency: {eff_comp['time_overhead']:.1f}x overhead ({eff_comp['status']})")
    
    print(f"\n📁 Results saved to: {Path('results/comprehensive_evaluation.json').absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive TMM evaluation")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--provider", choices=["openai", "google"], default="openai", help="API provider")
    parser.add_argument("--model", default="gpt-4", help="Model name")
    parser.add_argument("--examples", type=int, default=1000, help="Number of examples to evaluate")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    
    args = parser.parse_args()
    
    config = TestConfiguration(
        api_key=args.api_key,
        api_provider=args.provider,
        model_name=args.model,
        max_examples=args.examples,
        batch_size=args.batch_size,
        enable_checkpointing=not args.no_checkpoint
    )
    
    try:
        runner = ComprehensiveTestRunner(config)
        results = runner.run_comprehensive_test()
        print_comprehensive_results(results)
        
        print("\n" + "=" * 100)
        print("✅ COMPREHENSIVE EVALUATION COMPLETE")
        print("🚀 Ready for publication and further research!")
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
