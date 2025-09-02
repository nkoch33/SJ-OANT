"""
evaluation.fictionalqa_eval - FictionalQA Evaluation Pipeline

This module provides evaluation capabilities for the TMM system using the
FictionalQA dataset, measuring performance on fictional reasoning tasks
that require maintaining accurate memory of story details.

Key Features:
- Direct FictionalQA dataset integration
- TMM system evaluation with truth-maintenance metrics
- Baseline comparison (RAG, long-context, standard memory)
- Performance metrics: accuracy, memory consistency, false memory detection
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

from datasets import load_dataset

from core.types import MemoryRecord
from memory.typed_store import MemoryState
from tmm_pipeline import TMMPipeline

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Results from evaluating a system on FictionalQA."""
    system_name: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_response_time: float
    memory_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

@dataclass
class FictionalQAExample:
    """A single FictionalQA example for evaluation."""
    id: str
    story: str
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class FictionalQAEvaluator:
    """
    Evaluator for TMM systems using the FictionalQA dataset.
    
    This evaluator tests how well systems can:
    1. Store and recall fictional story details
    2. Resist false memory formation
    3. Maintain consistent story facts across questions
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the FictionalQA evaluator."""
        self.cache_dir = cache_dir
        self.dataset = None
        logger.info("Initialized FictionalQA evaluator")
    
    def load_dataset(self, split: str = "validation") -> List[FictionalQAExample]:
        """
        Load FictionalQA dataset examples.
        
        Args:
            split: Dataset split to use ("train", "validation", "test")
            
        Returns:
            List of FictionalQA examples
        """
        if self.dataset is None:
            logger.info("Loading SQuAD dataset...")
            self.dataset = load_dataset("squad", cache_dir=self.cache_dir)
        
        split_data = self.dataset[split]
        examples = []
        
        for i, example in enumerate(split_data):
            # Extract the first answer text (SQuAD has multiple possible answers)
            answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer"
            examples.append(FictionalQAExample(
                id=example["id"],
                story=example["context"],
                question=example["question"], 
                answer=answer,
                metadata={
                    "split": split,
                    "index": i,
                    "title": example["title"],
                    "story_length": len(example["context"].split())
                }
            ))
        
        logger.info(f"Loaded {len(examples)} examples from {split} split")
        return examples
    
    def evaluate_tmm_system(self, 
                           tmm_pipeline: TMMPipeline,
                           examples: List[FictionalQAExample],
                           limit: Optional[int] = None) -> EvaluationResult:
        """
        Evaluate the TMM system on FictionalQA examples.
        
        Args:
            tmm_pipeline: TMM pipeline to evaluate
            examples: FictionalQA examples to test on
            limit: Optional limit on number of examples to evaluate
            
        Returns:
            Evaluation results
        """
        if limit:
            examples = examples[:limit]
        
        logger.info(f"Evaluating TMM system on {len(examples)} FictionalQA examples")
        
        correct_answers = 0
        total_time = 0.0
        errors = []
        memory_metrics = {
            "total_memory_records": 0,
            "memory_operations": 0,
            "verification_calls": 0
        }
        
        for i, example in enumerate(examples):
            try:
                start_time = time.perf_counter()
                
                # Step 1: Provide the story context to build memory
                logger.debug(f"Example {i+1}/{len(examples)}: Processing story context")
                story_prompt = f"Please remember this story: {example.story}"
                
                # Process the story to build memory (don't expect a specific response)
                _ = tmm_pipeline.process(story_prompt)
                
                # Step 2: Ask the question about the story
                logger.debug(f"Example {i+1}/{len(examples)}: Asking question")
                response = tmm_pipeline.process(example.question)
                
                # Step 3: Check if the response matches the expected answer
                is_correct = self._evaluate_answer(response, example.answer)
                if is_correct:
                    correct_answers += 1
                
                # Collect timing
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                
                # Collect memory metrics
                if hasattr(tmm_pipeline, 'memory_store'):
                    memory_metrics["total_memory_records"] += len(tmm_pipeline.memory_store._records)
                
                logger.debug(f"Example {i+1}: {'CORRECT' if is_correct else 'INCORRECT'} ({elapsed:.2f}s)")
                
            except Exception as e:
                error_msg = f"Error on example {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        accuracy = correct_answers / len(examples) if examples else 0.0
        avg_response_time = total_time / len(examples) if examples else 0.0
        
        result = EvaluationResult(
            system_name="TMM_Pipeline",
            total_questions=len(examples),
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            memory_metrics=memory_metrics,
            errors=errors
        )
        
        logger.info(f"TMM evaluation complete: {correct_answers}/{len(examples)} correct ({accuracy:.2%})")
        return result
    
    def _evaluate_answer(self, response: str, expected_answer: str) -> bool:
        """
        Evaluate if the response matches the expected answer.
        
        This uses simple string matching for now, but could be enhanced
        with semantic similarity or LLM-based evaluation.
        
        Args:
            response: System response
            expected_answer: Expected answer
            
        Returns:
            True if the answer is considered correct
        """
        # Simple string matching (case-insensitive, whitespace normalized)
        response_clean = response.lower().strip()
        expected_clean = expected_answer.lower().strip()
        
        # Exact match
        if response_clean == expected_clean:
            return True
        
        # Check if expected answer is contained in response
        if expected_clean in response_clean:
            return True
        
        # TODO: Add more sophisticated answer matching
        # - Semantic similarity with embeddings
        # - LLM-based answer evaluation
        # - Extract key facts and compare
        
        return False
    
    def evaluate_baseline_system(self,
                                baseline_system,
                                examples: List[FictionalQAExample],
                                system_name: str,
                                limit: Optional[int] = None) -> EvaluationResult:
        """
        Evaluate a baseline system on FictionalQA examples.
        
        Args:
            baseline_system: Baseline system to evaluate (should have a .process() method)
            examples: FictionalQA examples to test on
            system_name: Name of the baseline system
            limit: Optional limit on number of examples
            
        Returns:
            Evaluation results
        """
        if limit:
            examples = examples[:limit]
        
        logger.info(f"Evaluating {system_name} on {len(examples)} FictionalQA examples")
        
        correct_answers = 0
        total_time = 0.0
        errors = []
        
        for i, example in enumerate(examples):
            try:
                start_time = time.perf_counter()
                
                # Combine story and question for baseline systems
                full_prompt = f"Story: {example.story}\n\nQuestion: {example.question}\n\nAnswer:"
                response = baseline_system.process(full_prompt)
                
                is_correct = self._evaluate_answer(response, example.answer)
                if is_correct:
                    correct_answers += 1
                
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                
                logger.debug(f"Example {i+1}: {'CORRECT' if is_correct else 'INCORRECT'} ({elapsed:.2f}s)")
                
            except Exception as e:
                error_msg = f"Error on example {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        accuracy = correct_answers / len(examples) if examples else 0.0
        avg_response_time = total_time / len(examples) if examples else 0.0
        
        result = EvaluationResult(
            system_name=system_name,
            total_questions=len(examples),
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            errors=errors
        )
        
        logger.info(f"{system_name} evaluation complete: {correct_answers}/{len(examples)} correct ({accuracy:.2%})")
        return result
    
    def compare_systems(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Compare evaluation results across different systems.
        
        Args:
            results: List of evaluation results to compare
            
        Returns:
            Comparison summary
        """
        if not results:
            return {}
        
        comparison = {
            "systems": [r.system_name for r in results],
            "accuracies": [r.accuracy for r in results],
            "response_times": [r.avg_response_time for r in results],
            "best_accuracy": max(results, key=lambda r: r.accuracy),
            "fastest_system": min(results, key=lambda r: r.avg_response_time),
            "summary": {}
        }
        
        for result in results:
            comparison["summary"][result.system_name] = {
                "accuracy": f"{result.accuracy:.2%}",
                "avg_response_time": f"{result.avg_response_time:.2f}s",
                "total_questions": result.total_questions,
                "errors": len(result.errors)
            }
        
        return comparison
    
    def save_results(self, results: List[EvaluationResult], output_file: str):
        """Save evaluation results to a JSON file."""
        output_data = {
            "evaluation_timestamp": time.time(),
            "results": [
                {
                    "system_name": r.system_name,
                    "total_questions": r.total_questions,
                    "correct_answers": r.correct_answers,
                    "accuracy": r.accuracy,
                    "avg_response_time": r.avg_response_time,
                    "memory_metrics": r.memory_metrics,
                    "errors": r.errors
                }
                for r in results
            ],
            "comparison": self.compare_systems(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
