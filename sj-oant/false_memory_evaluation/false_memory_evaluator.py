"""
False Memory Formation Evaluator

Main evaluator that integrates with existing TMM system to test false memory
formation prevention capabilities.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .test_scenarios import FalseMemoryTestScenarios, FalseMemoryScenario
from .metrics_calculator import MetricsCalculator, FalseMemoryMetrics
from .contradiction_detector import ContradictionDetector

logger = logging.getLogger(__name__)

class FalseMemoryEvaluator:
    """
    Main evaluator for false memory formation testing.
    
    This class integrates with the existing TMM system to evaluate its ability
    to prevent false memory formation using the core research metrics:
    - FMR (False Memory Rate)
    - MEL (Memory Edit Latency) 
    - DAR (Disturbance Adaptation Rate)
    """
    
    def __init__(self, tmm_pipeline=None):
        """
        Initialize the false memory evaluator.
        
        Args:
            tmm_pipeline: TMM pipeline instance for generating responses
        """
        self.tmm_pipeline = tmm_pipeline
        self.scenario_generator = FalseMemoryTestScenarios()
        self.metrics_calculator = MetricsCalculator()
        self.contradiction_detector = ContradictionDetector()
        
        logger.info("Initialized False Memory Evaluator")
    
    def evaluate_benchmark_false_memory(self, benchmark: str, 
                                      num_scenarios: int = 10) -> Dict[str, Any]:
        """
        Evaluate TMM's false memory prevention on a specific benchmark.
        
        Args:
            benchmark: Benchmark name (multiwoz, sgd, taskmaster)
            num_scenarios: Number of test scenarios to generate
            
        Returns:
            Dictionary with false memory evaluation results
        """
        logger.info(f"Starting false memory evaluation for {benchmark}")
        
        # Load benchmark data
        benchmark_data = self._load_benchmark_data(benchmark)
        if not benchmark_data:
            return {"error": f"Could not load data for benchmark: {benchmark}"}
        
        # Generate test scenarios
        scenarios = self.scenario_generator.generate_benchmark_scenarios(
            benchmark_data, num_scenarios)
        
        # Evaluate each scenario
        results = []
        for i, scenario in enumerate(scenarios):
            logger.info(f"Evaluating scenario {i+1}/{len(scenarios)}")
            
            scenario_result = self._evaluate_scenario(scenario)
            scenario_result["scenario_id"] = i
            scenario_result["scenario_type"] = scenario.scenario_type
            results.append(scenario_result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            "benchmark": benchmark,
            "num_scenarios": len(scenarios),
            "scenario_results": results,
            "aggregate_metrics": aggregate_metrics,
            "evaluation_summary": self._generate_evaluation_summary(aggregate_metrics)
        }
    
    def evaluate_all_benchmarks_false_memory(self, num_scenarios: int = 5) -> Dict[str, Any]:
        """
        Evaluate TMM's false memory prevention across all benchmarks.
        
        Args:
            num_scenarios: Number of scenarios per benchmark
            
        Returns:
            Dictionary with false memory evaluation results for all benchmarks
        """
        benchmarks = ["multiwoz", "sgd", "taskmaster"]
        all_results = {}
        
        for benchmark in benchmarks:
            logger.info(f"Evaluating false memory prevention for {benchmark}")
            benchmark_result = self.evaluate_benchmark_false_memory(
                benchmark, num_scenarios)
            all_results[benchmark] = benchmark_result
        
        # Calculate cross-benchmark metrics
        cross_benchmark_metrics = self._calculate_cross_benchmark_metrics(all_results)
        
        return {
            "benchmark_results": all_results,
            "cross_benchmark_metrics": cross_benchmark_metrics,
            "overall_summary": self._generate_overall_summary(cross_benchmark_metrics)
        }
    
    def _load_benchmark_data(self, benchmark: str) -> List[Dict]:
        """Load benchmark data for false memory testing."""
        try:
            # Import the existing data loader
            import sys
            sys.path.append('.')
            from testing.official_evaluation import OfficialBenchmarkEvaluator
            
            # Create evaluator instance to access data loading methods
            evaluator = OfficialBenchmarkEvaluator()
            
            # Load a larger sample for false memory testing
            data_path = Path(__file__).parent.parent.parent / "data"
            samples = evaluator.load_benchmark_data(benchmark, 50)  # Load more samples
            
            # Convert to conversation format
            conversations = []
            for sample in samples:
                conversation = {
                    "dialogue_id": sample.get("dialogue_id", "unknown"),
                    "user_turns": sample.get("user_turns", []),
                    "system_turns": sample.get("system_turns", [])
                }
                conversations.append(conversation)
            
            logger.info(f"Loaded {len(conversations)} conversations for {benchmark}")
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to load benchmark data for {benchmark}: {e}")
            return []
    
    def _evaluate_scenario(self, scenario: FalseMemoryScenario) -> Dict[str, Any]:
        """
        Evaluate a single false memory test scenario.
        
        Args:
            scenario: FalseMemoryScenario to evaluate
            
        Returns:
            Dictionary with scenario evaluation results
        """
        try:
            # Generate TMM responses for the modified conversation
            tmm_responses = self._generate_tmm_responses(scenario.original_conversation)
            
            # Calculate false memory metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                tmm_responses,
                scenario.false_information,
                scenario.contradiction_point
            )
            
            # Detect contradictions
            contradiction_analysis = self.contradiction_detector.analyze_contradictions(
                tmm_responses, scenario.false_information)
            
            return {
                "scenario_type": scenario.scenario_type,
                "false_information": scenario.false_information,
                "contradiction_point": scenario.contradiction_point,
                "expected_behavior": scenario.expected_behavior,
                "tmm_responses": tmm_responses,
                "metrics": {
                    "fmr": metrics.fmr,
                    "mel": metrics.mel,
                    "dar": metrics.dar,
                    "contradiction_detection_rate": metrics.contradiction_detection_rate,
                    "false_information_persistence": metrics.false_information_persistence,
                    "correction_accuracy": metrics.correction_accuracy
                },
                "contradiction_analysis": contradiction_analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate scenario: {e}")
            return {"error": str(e)}
    
    def _generate_tmm_responses(self, conversation: Dict) -> List[str]:
        """
        Generate TMM responses for a conversation.
        
        Args:
            conversation: Conversation dict with user and system turns
            
        Returns:
            List of TMM-generated responses
        """
        if self.tmm_pipeline is None:
            # Fallback: return mock responses for testing
            logger.warning("No TMM pipeline provided, using mock responses")
            return ["Mock response 1", "Mock response 2", "Mock response 3"]
        
        responses = []
        
        try:
            user_turns = conversation.get("user_turns", [])
            
            for user_input in user_turns:
                if user_input:
                    # Process with TMM pipeline
                    tmm_response = self.tmm_pipeline.process(user_input)
                    responses.append(tmm_response)
            
            logger.info(f"Generated {len(responses)} TMM responses")
            return responses
            
        except Exception as e:
            logger.error(f"Failed to generate TMM responses: {e}")
            return []
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics across all scenarios."""
        if not results:
            return {}
        
        # Extract metrics from all scenarios
        all_fmr = [r["metrics"]["fmr"] for r in results if "metrics" in r]
        all_mel = [r["metrics"]["mel"] for r in results if "metrics" in r and r["metrics"]["mel"] != float('inf')]
        all_dar = [r["metrics"]["dar"] for r in results if "metrics" in r]
        all_contradiction_detection = [r["metrics"]["contradiction_detection_rate"] for r in results if "metrics" in r]
        all_persistence = [r["metrics"]["false_information_persistence"] for r in results if "metrics" in r]
        all_correction_accuracy = [r["metrics"]["correction_accuracy"] for r in results if "metrics" in r]
        
        return {
            "avg_fmr": sum(all_fmr) / len(all_fmr) if all_fmr else 0.0,
            "avg_mel": sum(all_mel) / len(all_mel) if all_mel else 0.0,
            "avg_dar": sum(all_dar) / len(all_dar) if all_dar else 0.0,
            "avg_contradiction_detection": sum(all_contradiction_detection) / len(all_contradiction_detection) if all_contradiction_detection else 0.0,
            "avg_persistence": sum(all_persistence) / len(all_persistence) if all_persistence else 0.0,
            "avg_correction_accuracy": sum(all_correction_accuracy) / len(all_correction_accuracy) if all_correction_accuracy else 0.0,
            "num_scenarios": len(results)
        }
    
    def _calculate_cross_benchmark_metrics(self, all_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics across all benchmarks."""
        benchmark_metrics = []
        
        for benchmark, result in all_results.items():
            if "aggregate_metrics" in result:
                metrics = result["aggregate_metrics"]
                benchmark_metrics.append(metrics)
        
        if not benchmark_metrics:
            return {}
        
        # Calculate averages across benchmarks
        return {
            "overall_fmr": sum(m["avg_fmr"] for m in benchmark_metrics) / len(benchmark_metrics),
            "overall_mel": sum(m["avg_mel"] for m in benchmark_metrics) / len(benchmark_metrics),
            "overall_dar": sum(m["avg_dar"] for m in benchmark_metrics) / len(benchmark_metrics),
            "overall_contradiction_detection": sum(m["avg_contradiction_detection"] for m in benchmark_metrics) / len(benchmark_metrics),
            "overall_persistence": sum(m["avg_persistence"] for m in benchmark_metrics) / len(benchmark_metrics),
            "overall_correction_accuracy": sum(m["avg_correction_accuracy"] for m in benchmark_metrics) / len(benchmark_metrics),
            "total_scenarios": sum(m["num_scenarios"] for m in benchmark_metrics)
        }
    
    def _generate_evaluation_summary(self, metrics: Dict[str, float]) -> str:
        """Generate a human-readable evaluation summary."""
        if not metrics:
            return "No metrics available"
        
        summary = f"""
🔬 FALSE MEMORY FORMATION EVALUATION SUMMARY

📊 Core Research Metrics:
• FMR (False Memory Rate): {metrics.get('avg_fmr', 0):.2f}% 
  (Lower is better - measures false information repetition)
• MEL (Memory Edit Latency): {metrics.get('avg_mel', 0):.2f} seconds
  (Lower is better - measures correction speed)
• DAR (Disturbance Adaptation Rate): {metrics.get('avg_dar', 0):.2f}%
  (Higher is better - measures accuracy despite false context)

🎯 Additional Metrics:
• Contradiction Detection: {metrics.get('avg_contradiction_detection', 0):.2f}%
• False Information Persistence: {metrics.get('avg_persistence', 0):.2f} turns
• Correction Accuracy: {metrics.get('avg_correction_accuracy', 0):.2f}%

📈 Scenarios Tested: {metrics.get('num_scenarios', 0)}
"""
        return summary
    
    def _generate_overall_summary(self, metrics: Dict[str, float]) -> str:
        """Generate overall evaluation summary across all benchmarks."""
        if not metrics:
            return "No overall metrics available"
        
        summary = f"""
🏆 OVERALL FALSE MEMORY PREVENTION PERFORMANCE

🎯 Cross-Benchmark Results:
• Overall FMR: {metrics.get('overall_fmr', 0):.2f}%
• Overall MEL: {metrics.get('overall_mel', 0):.2f} seconds  
• Overall DAR: {metrics.get('overall_dar', 0):.2f}%
• Overall Contradiction Detection: {metrics.get('overall_contradiction_detection', 0):.2f}%
• Overall Persistence: {metrics.get('overall_persistence', 0):.2f} turns
• Overall Correction Accuracy: {metrics.get('overall_correction_accuracy', 0):.2f}%

📊 Total Scenarios: {metrics.get('total_scenarios', 0)}

🔬 Research Alignment: These metrics directly measure TMM's ability to prevent
false memory formation, which is the core research vision of the paper.
"""
        return summary
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved false memory evaluation results to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from a JSON file."""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded false memory evaluation results from {filepath}")
        return results
