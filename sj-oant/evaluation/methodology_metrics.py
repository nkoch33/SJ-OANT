"""
Methodology Metrics - FMR, MEL, DAR for SQuAD Evaluation

This module implements the core methodology metrics from the research paper,
adapted for SQuAD 2.0 evaluation to measure false-memory resilience and
truth-maintenance capabilities.

Metrics:
- FMR (False Memory Rate): % answers repeating seeded falsehoods
- MEL (Memory Edit Latency): turns to correctly overwrite after correction
- DAR (Disturbance Adaptation Rate): reliability under mixed true/false context

These metrics are adapted from the FABLE benchmark methodology to work with
SQuAD's unanswerable questions and context-based evaluation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MethodologyMetrics:
    """Container for methodology-specific metrics."""
    fmr: float  # False Memory Rate
    mel: float  # Memory Edit Latency
    dar: float  # Disturbance Adaptation Rate
    accuracy: float  # Standard accuracy
    answerable_accuracy: float  # Accuracy on answerable questions
    unanswerable_accuracy: float  # Accuracy on unanswerable questions
    memory_consistency: float  # Memory consistency score
    contradiction_resolution: float  # Contradiction resolution rate

class MethodologyMetricCalculator:
    """
    Calculator for methodology-specific metrics adapted for SQuAD evaluation.
    
    This class implements the core metrics from the research methodology,
    adapted to work with SQuAD 2.0's structure of answerable/unanswerable questions.
    """
    
    def __init__(self):
        """Initialize the metric calculator."""
        self.false_memory_indicators = [
            "i don't know", "i'm not sure", "i cannot answer", "unanswerable",
            "no answer", "cannot be determined", "not provided", "unknown"
        ]
        
        # Patterns that indicate false memory formation
        self.false_memory_patterns = [
            r"based on the context",  # Making up answers from context
            r"according to the passage",  # Fabricating from passage
            r"the text states",  # Inventing text claims
            r"it is mentioned",  # Creating false mentions
        ]
        
        logger.info("Initialized Methodology Metric Calculator")
    
    def calculate_fmr(self, 
                     responses: List[str], 
                     ground_truths: List[str], 
                     is_answerable: List[bool]) -> float:
        """
        Calculate False Memory Rate (FMR).
        
        FMR measures the percentage of answers that repeat seeded falsehoods.
        For SQuAD, this is adapted to measure how often systems fabricate
        answers for unanswerable questions instead of saying "I don't know."
        
        Args:
            responses: List of system responses
            ground_truths: List of ground truth answers
            is_answerable: List of boolean flags indicating if question is answerable
            
        Returns:
            False Memory Rate (0.0 to 1.0)
        """
        if not responses:
            return 0.0
        
        false_memories = 0
        unanswerable_count = 0
        
        for response, ground_truth, answerable in zip(responses, ground_truths, is_answerable):
            if not answerable:  # Unanswerable question
                unanswerable_count += 1
                
                # Check if system fabricated an answer instead of saying "I don't know"
                if self._is_fabricated_answer(response, ground_truth):
                    false_memories += 1
        
        if unanswerable_count == 0:
            return 0.0
        
        fmr = false_memories / unanswerable_count
        logger.info(f"FMR calculated: {false_memories}/{unanswerable_count} = {fmr:.3f}")
        return fmr
    
    def _is_fabricated_answer(self, response: str, ground_truth: str) -> bool:
        """
        Check if a response is a fabricated answer for an unanswerable question.
        
        Args:
            response: System response
            ground_truth: Ground truth (should be empty for unanswerable questions)
            
        Returns:
            True if response appears to be fabricated
        """
        response_lower = response.lower().strip()
        
        # If ground truth is empty, any substantive answer is likely fabricated
        if not ground_truth.strip():
            # Check for false memory indicators
            for pattern in self.false_memory_patterns:
                if re.search(pattern, response_lower):
                    return True
            
            # Check if response is too long/specific for an unanswerable question
            if len(response.split()) > 10 and not any(indicator in response_lower 
                                                    for indicator in self.false_memory_indicators):
                return True
        
        return False
    
    def calculate_mel(self, 
                     conversation_history: List[Dict[str, Any]], 
                     correction_turns: List[int]) -> float:
        """
        Calculate Memory Edit Latency (MEL).
        
        MEL measures the number of turns to correctly overwrite after a
        correction appears. For SQuAD, this is adapted to measure how quickly
        systems correct themselves when given contradictory information.
        
        Args:
            conversation_history: List of conversation turns with responses
            correction_turns: List of turn indices where corrections occurred
            
        Returns:
            Average Memory Edit Latency in turns
        """
        if not correction_turns:
            return 0.0
        
        total_latency = 0
        valid_corrections = 0
        
        for correction_turn in correction_turns:
            if correction_turn >= len(conversation_history) - 1:
                continue
            
            # Look for correction in subsequent turns
            correction_found = False
            for i in range(correction_turn + 1, len(conversation_history)):
                if self._detects_correction(conversation_history[i]):
                    latency = i - correction_turn
                    total_latency += latency
                    valid_corrections += 1
                    correction_found = True
                    break
            
            if not correction_found:
                # If no correction found, assume maximum latency
                total_latency += len(conversation_history) - correction_turn
                valid_corrections += 1
        
        if valid_corrections == 0:
            return 0.0
        
        mel = total_latency / valid_corrections
        logger.info(f"MEL calculated: {total_latency}/{valid_corrections} = {mel:.2f} turns")
        return mel
    
    def _detects_correction(self, turn: Dict[str, Any]) -> bool:
        """
        Check if a turn shows evidence of correction detection.
        
        Args:
            turn: Conversation turn with response and metadata
            
        Returns:
            True if correction is detected
        """
        response = turn.get("response", "").lower()
        
        correction_indicators = [
            "correction", "mistake", "error", "wrong", "incorrect",
            "actually", "however", "but", "on second thought"
        ]
        
        return any(indicator in response for indicator in correction_indicators)
    
    def calculate_dar(self, 
                     responses: List[str], 
                     contexts: List[str], 
                     ground_truths: List[str]) -> float:
        """
        Calculate Disturbance Adaptation Rate (DAR).
        
        DAR measures reliability under mixed true/false context perturbations.
        For SQuAD, this is adapted to measure how well systems handle
        contexts with both relevant and irrelevant information.
        
        Args:
            responses: List of system responses
            contexts: List of context passages
            ground_truths: List of ground truth answers
            
        Returns:
            Disturbance Adaptation Rate (0.0 to 1.0)
        """
        if not responses:
            return 0.0
        
        total_questions = len(responses)
        reliable_responses = 0
        
        for response, context, ground_truth in zip(responses, contexts, ground_truths):
            if self._is_reliable_under_disturbance(response, context, ground_truth):
                reliable_responses += 1
        
        dar = reliable_responses / total_questions
        logger.info(f"DAR calculated: {reliable_responses}/{total_questions} = {dar:.3f}")
        return dar
    
    def _is_reliable_under_disturbance(self, response: str, context: str, ground_truth: str) -> bool:
        """
        Check if response is reliable under context disturbance.
        
        Args:
            response: System response
            context: Context passage
            ground_truth: Ground truth answer
            
        Returns:
            True if response is reliable
        """
        # For unanswerable questions, check if system correctly says "I don't know"
        if not ground_truth.strip():
            return any(indicator in response.lower() for indicator in self.false_memory_indicators)
        
        # For answerable questions, check if response is accurate and not influenced by noise
        response_lower = response.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Check for accuracy
        if ground_truth_lower not in response_lower:
            return False
        
        # Check for noise resistance (not including irrelevant context details)
        context_words = set(context.lower().split())
        response_words = set(response_lower.split())
        
        # If response includes too many context words not in ground truth, it might be noisy
        irrelevant_words = context_words - set(ground_truth_lower.split())
        if len(response_words & irrelevant_words) > 3:  # Threshold for noise
            return False
        
        return True
    
    def calculate_memory_consistency(self, memory_operations: List[Dict[str, Any]]) -> float:
        """
        Calculate memory consistency score.
        
        Measures how consistent the memory operations are across the evaluation.
        
        Args:
            memory_operations: List of memory operation records
            
        Returns:
            Memory consistency score (0.0 to 1.0)
        """
        if not memory_operations:
            return 1.0
        
        # Count consistent vs inconsistent operations
        consistent_ops = 0
        total_ops = len(memory_operations)
        
        for op in memory_operations:
            if op.get("consistent", True):  # Assume consistent unless marked otherwise
                consistent_ops += 1
        
        consistency = consistent_ops / total_ops
        logger.info(f"Memory consistency: {consistent_ops}/{total_ops} = {consistency:.3f}")
        return consistency
    
    def calculate_contradiction_resolution(self, 
                                         contradiction_events: List[Dict[str, Any]]) -> float:
        """
        Calculate contradiction resolution rate.
        
        Measures how well the system resolves contradictions when they occur.
        
        Args:
            contradiction_events: List of contradiction detection events
            
        Returns:
            Contradiction resolution rate (0.0 to 1.0)
        """
        if not contradiction_events:
            return 1.0
        
        resolved_contradictions = 0
        total_contradictions = len(contradiction_events)
        
        for event in contradiction_events:
            if event.get("resolved", False):
                resolved_contradictions += 1
        
        resolution_rate = resolved_contradictions / total_contradictions
        logger.info(f"Contradiction resolution: {resolved_contradictions}/{total_contradictions} = {resolution_rate:.3f}")
        return resolution_rate
    
    def calculate_all_metrics(self, 
                            evaluation_data: Dict[str, Any]) -> MethodologyMetrics:
        """
        Calculate all methodology metrics from evaluation data.
        
        Args:
            evaluation_data: Complete evaluation data including responses, contexts, etc.
            
        Returns:
            MethodologyMetrics object with all calculated metrics
        """
        responses = evaluation_data.get("responses", [])
        ground_truths = evaluation_data.get("ground_truths", [])
        is_answerable = evaluation_data.get("is_answerable", [])
        contexts = evaluation_data.get("contexts", [])
        conversation_history = evaluation_data.get("conversation_history", [])
        memory_operations = evaluation_data.get("memory_operations", [])
        contradiction_events = evaluation_data.get("contradiction_events", [])
        
        # Calculate standard accuracy
        correct_answers = sum(1 for resp, gt in zip(responses, ground_truths) 
                            if gt.lower() in resp.lower() or (not gt.strip() and any(indicator in resp.lower() for indicator in self.false_memory_indicators)))
        accuracy = correct_answers / len(responses) if responses else 0.0
        
        # Calculate answerable/unanswerable accuracy
        answerable_correct = 0
        answerable_total = 0
        unanswerable_correct = 0
        unanswerable_total = 0
        
        for resp, gt, answerable in zip(responses, ground_truths, is_answerable):
            if answerable:
                answerable_total += 1
                if gt.lower() in resp.lower():
                    answerable_correct += 1
            else:
                unanswerable_total += 1
                if any(indicator in resp.lower() for indicator in self.false_memory_indicators):
                    unanswerable_correct += 1
        
        answerable_accuracy = answerable_correct / answerable_total if answerable_total > 0 else 0.0
        unanswerable_accuracy = unanswerable_correct / unanswerable_total if unanswerable_total > 0 else 0.0
        
        # Calculate methodology-specific metrics
        fmr = self.calculate_fmr(responses, ground_truths, is_answerable)
        mel = self.calculate_mel(conversation_history, evaluation_data.get("correction_turns", []))
        dar = self.calculate_dar(responses, contexts, ground_truths)
        memory_consistency = self.calculate_memory_consistency(memory_operations)
        contradiction_resolution = self.calculate_contradiction_resolution(contradiction_events)
        
        return MethodologyMetrics(
            fmr=fmr,
            mel=mel,
            dar=dar,
            accuracy=accuracy,
            answerable_accuracy=answerable_accuracy,
            unanswerable_accuracy=unanswerable_accuracy,
            memory_consistency=memory_consistency,
            contradiction_resolution=contradiction_resolution
        )
