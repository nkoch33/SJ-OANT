"""
evaluation.squad_eval - SQuAD Dataset Evaluation System

This module implements the evaluation system for the TMM agent using the
SQuAD (Stanford Question Answering Dataset). It provides proper train/validation/test
splits and rigorous evaluation methodology for question-answering systems.

Key Features:
- Loads SQuAD dataset from HuggingFace with proper splits
- Implements comprehensive evaluation metrics
- Compares TMM system against established baselines
- Provides detailed performance analysis and reporting
"""

import logging
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

from datasets import load_dataset

from tmm_pipeline import TMMPipeline

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SQuADExample:
    """Represents a single example from the SQuAD dataset."""
    id: str
    context: str
    question: str
    answer: str  # Ground truth answer
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EvaluationResult:
    """Stores the results of an evaluation run for a single system."""
    system_name: str
    total_examples: int
    correct_answers: int
    accuracy: float
    avg_response_time: float
    memory_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

    def to_dict(self):
        return asdict(self)

class SQuADEvaluator:
    """
    Evaluator for the TMM system and baselines on the SQuAD dataset.
    
    This evaluator implements proper ML evaluation methodology with:
    - Train/validation/test splits
    - Systematic performance measurement
    - Comprehensive baseline comparisons
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the SQuAD evaluator."""
        self.cache_dir = cache_dir
        self.dataset = None
        logger.info("Initialized SQuAD evaluator")
    
    def load_dataset(self, split: str = "validation") -> List[SQuADExample]:
        """
        Load SQuAD dataset examples with proper error handling.
        
        Args:
            split: Dataset split to use ("train", "validation")
            
        Returns:
            List of SQuAD examples
        """
        if self.dataset is None:
            logger.info("Loading SQuAD dataset...")
            self.dataset = load_dataset("squad", cache_dir=self.cache_dir)
        
        if split not in self.dataset:
            available_splits = list(self.dataset.keys())
            raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
        
        split_data = self.dataset[split]
        examples = []
        
        for i, example in enumerate(split_data):
            # Extract the first answer text (SQuAD has multiple possible answers)
            answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer"
            examples.append(SQuADExample(
                id=example["id"],
                context=example["context"],
                question=example["question"], 
                answer=answer,
                title=example["title"],
                metadata={
                    "split": split,
                    "index": i,
                    "context_length": len(example["context"].split()),
                    "answer_start": example["answers"]["answer_start"][0] if example["answers"]["answer_start"] else -1
                }
            ))
        
        logger.info(f"Loaded {len(examples)} examples from {split} split")
        return examples
    
    def evaluate_tmm_system(self, 
                           tmm_pipeline: TMMPipeline, 
                           examples: List[SQuADExample],
                           limit: Optional[int] = None) -> EvaluationResult:
        """
        Evaluates the TMM system on SQuAD examples.
        
        The TMM system should:
        1. Store the context in memory
        2. Answer questions based on stored context
        3. Maintain truth and avoid false memories
        
        Args:
            tmm_pipeline: An initialized TMMPipeline instance
            examples: List of SQuAD examples to evaluate
            limit: Optional maximum number of examples to evaluate
            
        Returns:
            An EvaluationResult object
        """
        system_name = "TMM Pipeline"
        correct_answers = 0
        total_time = 0.0
        memory_metrics = {"total_memory_records": 0, "contexts_stored": 0}
        errors = []
        
        examples_to_evaluate = examples[:limit] if limit else examples
        
        logger.info(f"Evaluating TMM system on {len(examples_to_evaluate)} SQuAD examples")
        
        for i, example in enumerate(examples_to_evaluate):
            start_time = time.perf_counter()
            
            try:
                # CRITICAL: Reset memory for each new context to simulate fresh conversation
                tmm_pipeline.reset_memory()
                
                # Step 1: Provide the context to build memory
                logger.debug(f"Example {i+1}/{len(examples_to_evaluate)}: Storing context")
                context_prompt = f"Please remember this context: {example.context}"
                tmm_pipeline.process(context_prompt)
                memory_metrics["contexts_stored"] += 1
                
                # Step 2: Ask the question based on stored context
                logger.debug(f"Example {i+1}/{len(examples_to_evaluate)}: Asking question")
                response = tmm_pipeline.process(example.question)
                
                # Step 3: Evaluate the response
                is_correct = self._check_answer(response, example.answer)
                if is_correct:
                    correct_answers += 1
                
                # Collect metrics
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                
                if hasattr(tmm_pipeline, 'memory_store'):
                    memory_metrics["total_memory_records"] += len(tmm_pipeline.memory_store._records)
                
                logger.debug(f"Example {i+1}: {'CORRECT' if is_correct else 'INCORRECT'} ({elapsed:.2f}s)")
                
            except Exception as e:
                error_msg = f"Error on example {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
        
        accuracy = correct_answers / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        avg_response_time = total_time / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        
        logger.info(f"{system_name} evaluation complete: {correct_answers}/{len(examples_to_evaluate)} correct ({accuracy:.2%})")
        
        return EvaluationResult(
            system_name=system_name,
            total_examples=len(examples_to_evaluate),
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            memory_metrics=memory_metrics,
            errors=errors
        )

    def evaluate_baseline_system(self,
                                baseline_system: Any,
                                examples: List[SQuADExample],
                                system_name: str,
                                limit: Optional[int] = None) -> EvaluationResult:
        """
        Evaluates a baseline system on SQuAD examples.
        
        Args:
            baseline_system: An instance of a baseline system
            examples: List of SQuAD examples to evaluate
            system_name: Name of the baseline system
            limit: Optional maximum number of examples to evaluate
            
        Returns:
            An EvaluationResult object
        """
        correct_answers = 0
        total_time = 0.0
        errors = []
        
        examples_to_evaluate = examples[:limit] if limit else examples
        
        logger.info(f"Evaluating {system_name} on {len(examples_to_evaluate)} SQuAD examples")
        
        for i, example in enumerate(examples_to_evaluate):
            start_time = time.perf_counter()
            
            try:
                # Reset memory for memory-based baselines
                if hasattr(baseline_system, 'reset_memory'):
                    baseline_system.reset_memory()
                
                # Process context (if applicable)
                if hasattr(baseline_system, 'process_story'):
                    baseline_system.process_story(example.context)
                
                # Process question with context (adjust for baseline interface)
                if hasattr(baseline_system, 'process_story'):
                    response = baseline_system.process(example.question)
                else:
                    # For systems that don't separate story processing, include context in question
                    full_prompt = f"Context: {example.context}\n\nQuestion: {example.question}"
                    response = baseline_system.process(full_prompt)
                
                is_correct = self._check_answer(response, example.answer)
                if is_correct:
                    correct_answers += 1
                
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                
                logger.debug(f"Example {i+1}: {'CORRECT' if is_correct else 'INCORRECT'} ({elapsed:.2f}s)")
                
            except Exception as e:
                error_msg = f"Error on example {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
        
        accuracy = correct_answers / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        avg_response_time = total_time / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        
        logger.info(f"{system_name} evaluation complete: {correct_answers}/{len(examples_to_evaluate)} correct ({accuracy:.2%})")
        
        return EvaluationResult(
            system_name=system_name,
            total_examples=len(examples_to_evaluate),
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            errors=errors
        )

    def _check_answer(self, response: str, ground_truth: str) -> bool:
        """
        Compares the system's response to the ground truth answer.
        Uses both exact match and containment check for better evaluation.
        """
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        
        # Exact match
        if response_clean == ground_truth_clean:
            return True
        
        # Containment check (response contains the answer)
        if ground_truth_clean in response_clean:
            return True
        
        # TODO: Implement more sophisticated matching (F1 score, semantic similarity)
        return False

    def save_results(self, results: List[EvaluationResult], output_file: str):
        """Saves evaluation results to a JSON file."""
        output_data = {
            "evaluation_timestamp": time.time(),
            "results": [result.to_dict() for result in results],
            "comparison": self.compare_systems(results)
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")

    def compare_systems(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compares results across different systems."""
        if not results:
            return {}
        
        best_accuracy_result = max(results, key=lambda r: r.accuracy)
        fastest_system_result = min(results, key=lambda r: r.avg_response_time)
        
        comparison = {
            "systems": [r.system_name for r in results],
            "accuracies": [r.accuracy for r in results],
            "response_times": [r.avg_response_time for r in results],
            "best_accuracy": {
                "system_name": best_accuracy_result.system_name,
                "accuracy": best_accuracy_result.accuracy
            },
            "fastest_system": {
                "system_name": fastest_system_result.system_name,
                "avg_response_time": fastest_system_result.avg_response_time
            },
            "summary": {}
        }
        
        for result in results:
            comparison["summary"][result.system_name] = {
                "accuracy": f"{result.accuracy:.2%}",
                "avg_response_time": f"{result.avg_response_time:.2f}s",
                "total_examples": result.total_examples,
                "errors": len(result.errors)
            }
        
        return comparison

    def get_data_splits(self) -> Dict[str, int]:
        """Get information about available data splits."""
        if self.dataset is None:
            self.dataset = load_dataset("squad", cache_dir=self.cache_dir)
        
        return {split: len(data) for split, data in self.dataset.items()}
