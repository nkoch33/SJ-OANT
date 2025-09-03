"""
evaluation.squad_eval - SQuAD 2.0 Dataset Evaluation System

This module implements the enhanced evaluation system for the TMM agent using
SQuAD 2.0 (Stanford Question Answering Dataset v2). SQuAD 2.0 is specifically
chosen for memory-intensive evaluation due to:

- Longer contexts (200-400+ words) that stress memory systems
- Unanswerable questions (30%) that test truth verification
- Complex reasoning requiring memory-based inference
- Better evaluation of false memory prevention capabilities

Key Features:
- Loads SQuAD 2.0 dataset with unanswerable question handling
- Implements memory-specific evaluation metrics
- Multi-turn conversation evaluation for memory persistence
- Comprehensive baseline comparisons with memory utilization tracking
"""

import logging
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

from datasets import load_dataset

from tmm_pipeline import TMMPipelineFixed

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SQuADExample:
    """Represents a single example from the SQuAD 2.0 dataset."""
    id: str
    context: str
    question: str
    answer: str  # Ground truth answer (empty string for unanswerable)
    title: str
    is_answerable: bool  # True if question has answer in context
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EvaluationResult:
    """Stores the results of an evaluation run for a single system."""
    system_name: str
    total_examples: int
    correct_answers: int
    accuracy: float
    avg_response_time: float
    answerable_accuracy: float  # Accuracy on answerable questions
    unanswerable_accuracy: float  # Accuracy on unanswerable questions
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
        Load SQuAD 2.0 dataset examples with proper error handling.
        
        Args:
            split: Dataset split to use ("train", "validation")
            
        Returns:
            List of SQuAD 2.0 examples with unanswerable question support
        """
        if self.dataset is None:
            logger.info("Loading SQuAD 2.0 dataset...")
            self.dataset = load_dataset("squad_v2", cache_dir=self.cache_dir)
        
        if split not in self.dataset:
            available_splits = list(self.dataset.keys())
            raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
        
        split_data = self.dataset[split]
        examples = []
        answerable_count = 0
        unanswerable_count = 0
        
        for i, example in enumerate(split_data):
            # Check if question is answerable (has answers)
            is_answerable = len(example["answers"]["text"]) > 0
            answer = example["answers"]["text"][0] if is_answerable else ""
            
            if is_answerable:
                answerable_count += 1
            else:
                unanswerable_count += 1
            
            examples.append(SQuADExample(
                id=example["id"],
                context=example["context"],
                question=example["question"], 
                answer=answer,
                title=example["title"],
                is_answerable=is_answerable,
                metadata={
                    "split": split,
                    "index": i,
                    "context_length": len(example["context"].split()),
                    "answer_start": example["answers"]["answer_start"][0] if example["answers"]["answer_start"] else -1,
                    "plausible_answers": example.get("plausible_answers", {}).get("text", [])
                }
            ))
        
        logger.info(f"Loaded {len(examples)} examples from {split} split")
        logger.info(f"  Answerable: {answerable_count} ({answerable_count/len(examples)*100:.1f}%)")
        logger.info(f"  Unanswerable: {unanswerable_count} ({unanswerable_count/len(examples)*100:.1f}%)")
        
        return examples
    
    def evaluate_tmm_system(self, 
                           tmm_pipeline: TMMPipelineFixed, 
                           examples: List[SQuADExample],
                           limit: Optional[int] = None) -> EvaluationResult:
        """
        Evaluates the TMM system on SQuAD examples.
        
        The TMM system should:
        1. Store the context in memory
        2. Answer questions based on stored context
        3. Maintain truth and avoid false memories
        
        Args:
            tmm_pipeline: An initialized TMMPipelineFixed instance
            examples: List of SQuAD examples to evaluate
            limit: Optional maximum number of examples to evaluate
            
        Returns:
            An EvaluationResult object
        """
        system_name = "TMM Pipeline"
        correct_answers = 0
        answerable_correct = 0
        unanswerable_correct = 0
        answerable_total = 0
        unanswerable_total = 0
        total_time = 0.0
        memory_metrics = {"total_memory_records": 0, "contexts_stored": 0, "memory_retrievals": 0}
        errors = []
        
        examples_to_evaluate = examples[:limit] if limit else examples
        
        logger.info(f"Evaluating TMM system on {len(examples_to_evaluate)} SQuAD 2.0 examples")
        
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
                is_correct = self._check_answer_v2(response, example.answer, example.is_answerable)
                if is_correct:
                    correct_answers += 1
                
                # Track answerable vs unanswerable performance separately
                if example.is_answerable:
                    answerable_total += 1
                    if is_correct:
                        answerable_correct += 1
                else:
                    unanswerable_total += 1
                    if is_correct:
                        unanswerable_correct += 1
                
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
                correct_results.append(False)  # Mark as incorrect on error
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
        
        accuracy = correct_answers / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        answerable_accuracy = answerable_correct / answerable_total if answerable_total > 0 else 0.0
        unanswerable_accuracy = unanswerable_correct / unanswerable_total if unanswerable_total > 0 else 0.0
        avg_response_time = total_time / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        
        logger.info(f"{system_name} evaluation complete: {correct_answers}/{len(examples_to_evaluate)} correct ({accuracy:.2%})")
        logger.info(f"  Answerable: {answerable_correct}/{answerable_total} ({answerable_accuracy:.2%})")
        logger.info(f"  Unanswerable: {unanswerable_correct}/{unanswerable_total} ({unanswerable_accuracy:.2%})")
        
        return EvaluationResult(
            system_name=system_name,
            total_examples=len(examples_to_evaluate),
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            answerable_accuracy=answerable_accuracy,
            unanswerable_accuracy=unanswerable_accuracy,
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
        correct_results = []  # Track individual results for detailed metrics
        
        examples_to_evaluate = examples[:limit] if limit else examples
        
        logger.info(f"Evaluating {system_name} on {len(examples_to_evaluate)} SQuAD 2.0 examples")
        
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
                
                is_correct = self._check_answer_v2(response, example.answer, example.is_answerable)
                correct_results.append(is_correct)
                if is_correct:
                    correct_answers += 1
                
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                
                logger.debug(f"Example {i+1}: {'CORRECT' if is_correct else 'INCORRECT'} ({elapsed:.2f}s)")
                
            except Exception as e:
                error_msg = f"Error on example {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                correct_results.append(False)  # Mark as incorrect on error
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
        
        accuracy = correct_answers / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        avg_response_time = total_time / len(examples_to_evaluate) if examples_to_evaluate else 0.0
        
        logger.info(f"{system_name} evaluation complete: {correct_answers}/{len(examples_to_evaluate)} correct ({accuracy:.2%})")
        
        # Calculate answerable/unanswerable accuracy for baselines
        answerable_correct = sum(1 for i, ex in enumerate(examples_to_evaluate) 
                               if ex.is_answerable and i < len(correct_results) and correct_results[i])
        unanswerable_correct = sum(1 for i, ex in enumerate(examples_to_evaluate) 
                                 if not ex.is_answerable and i < len(correct_results) and correct_results[i])
        answerable_total = sum(1 for ex in examples_to_evaluate if ex.is_answerable)
        unanswerable_total = sum(1 for ex in examples_to_evaluate if not ex.is_answerable)
        
        answerable_accuracy = answerable_correct / answerable_total if answerable_total > 0 else 0.0
        unanswerable_accuracy = unanswerable_correct / unanswerable_total if unanswerable_total > 0 else 0.0
        
        return EvaluationResult(
            system_name=system_name,
            total_examples=len(examples_to_evaluate),
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            answerable_accuracy=answerable_accuracy,
            unanswerable_accuracy=unanswerable_accuracy,
            errors=errors
        )

    def _check_answer(self, response: str, ground_truth: str) -> bool:
        """
        Legacy method for SQuAD 1.1 compatibility.
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
        
        return False
    
    def _check_answer_v2(self, response: str, ground_truth: str, is_answerable: bool) -> bool:
        """
        Enhanced answer checking for SQuAD 2.0 with unanswerable question support.
        
        Args:
            response: System's response
            ground_truth: Expected answer (empty for unanswerable)
            is_answerable: Whether question should have an answer
            
        Returns:
            True if response is correct for the question type
        """
        response_clean = response.strip().lower()
        
        if not is_answerable:
            # For unanswerable questions, check if system indicates uncertainty
            uncertainty_indicators = [
                "i don't know", "i'm not sure", "not enough information", 
                "cannot be determined", "no answer", "unanswerable",
                "insufficient information", "not provided", "unclear",
                "cannot answer", "not specified", "not mentioned"
            ]
            
            # Check if response indicates uncertainty/inability to answer
            for indicator in uncertainty_indicators:
                if indicator in response_clean:
                    return True
            
            # If response is very short (likely indicating uncertainty)
            if len(response_clean.split()) <= 3:
                return True
                
            return False
        else:
            # For answerable questions, use standard matching
            ground_truth_clean = ground_truth.strip().lower()
            
            # Exact match
            if response_clean == ground_truth_clean:
                return True
            
            # Containment check (response contains the answer)
            if ground_truth_clean in response_clean:
                return True
            
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
