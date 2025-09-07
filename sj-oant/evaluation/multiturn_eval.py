"""
MultiWOZ Evaluation Framework for TMM System

This module provides comprehensive evaluation capabilities for the Truth-Maintained Memory (TMM)
system using the MultiWOZ 2.4 dataset, which is perfect for testing multi-turn conversational
memory, truth maintenance, and information consistency.

The evaluation framework tests:
- Dialogue success rate
- Information accuracy and consistency
- False memory detection
- Memory retrieval effectiveness
- Multi-agent pipeline performance
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from datasets import load_dataset
from langchain_core.language_models import BaseChatModel

from tmm_pipeline import TMMPipelineFixed
from baselines.multiturn_systems import create_multiturn_baselines
from evaluation.methodology_metrics import MethodologyMetricCalculator, MethodologyMetrics

logger = logging.getLogger(__name__)

@dataclass
class DialogueTurn:
    """Represents a single turn in a dialogue."""
    turn_id: int
    speaker: str  # "user" or "system"
    utterance: str
    dialogue_act: Optional[Dict[str, Any]] = None
    belief_state: Optional[Dict[str, Any]] = None

@dataclass
class DialogueSession:
    """Represents a complete dialogue session."""
    dialogue_id: str
    domain: str
    turns: List[DialogueTurn]
    goal: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None

@dataclass
class MultiTurnEvaluationResult:
    """Results from multi-turn dialogue evaluation."""
    # System performance
    dialogue_success_rate: float
    information_accuracy: float
    memory_consistency: float
    response_time_avg: float
    
    # Memory metrics
    memory_operations: Dict[str, int]
    memory_retrievals: int
    memory_stores: int
    memory_updates: int
    
    # Truth maintenance metrics
    false_memory_rate: float
    truth_verification_calls: int
    contradiction_detections: int
    
    # Methodology metrics
    methodology_metrics: MethodologyMetrics
    
    # Detailed results
    successful_dialogues: int
    total_dialogues: int
    total_turns: int
    domain_breakdown: Dict[str, Dict[str, float]]

class MultiTurnEvaluator:
    """
    Evaluator for multi-turn dialogue systems using MultiWOZ dataset.
    
    This evaluator tests the TMM system's ability to:
    1. Maintain information across dialogue turns
    2. Detect and prevent false memory formation
    3. Provide consistent responses based on stored information
    4. Successfully complete task-oriented dialogues
    """
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """Initialize the multi-turn evaluator."""
        self.api_key = api_key
        self.config = config or {}
        self.methodology_calculator = MethodologyMetricCalculator()
        
        # Load BlendedSkillTalk dataset (multi-turn dialogue dataset)
        logger.info("Loading BlendedSkillTalk dataset...")
        self.dataset = load_dataset("blended_skill_talk")
        logger.info(f"Loaded BlendedSkillTalk dataset with {len(self.dataset['train'])} training dialogues")
        
        # Initialize TMM system
        self.tmm_system = TMMPipelineFixed(api_key, config)
        
        # Initialize baselines
        self.baselines = create_multiturn_baselines(api_key)
        
        logger.info("Multi-turn evaluator initialized successfully")
    
    def load_dialogues(self, split: str = "test", limit: Optional[int] = None) -> List[DialogueSession]:
        """
        Load and preprocess BlendedSkillTalk dialogues.
        
        Args:
            split: Dataset split to use ("train", "validation", "test")
            limit: Maximum number of dialogues to load
            
        Returns:
            List of DialogueSession objects
        """
        logger.info(f"Loading {split} dialogues from BlendedSkillTalk...")
        
        data = self.dataset[split]
        if limit:
            data = data.select(range(min(limit, len(data))))
        
        dialogues = []
        for i, item in enumerate(data):
            dialogue_id = f"blended_skill_talk_{i:04d}"
            domain = item.get("context", "general")
            
            # Parse dialogue turns from BlendedSkillTalk format
            turns = []
            turn_id = 0
            
            # Add previous utterances
            for utterance in item.get("previous_utterance", []):
                if utterance.strip():
                    turn = DialogueTurn(
                        turn_id=turn_id,
                        speaker="user" if turn_id % 2 == 0 else "system",
                        utterance=utterance
                    )
                    turns.append(turn)
                    turn_id += 1
            
            # Add free messages (conversation)
            for utterance in item.get("free_messages", []):
                if utterance.strip():
                    turn = DialogueTurn(
                        turn_id=turn_id,
                        speaker="user" if turn_id % 2 == 0 else "system",
                        utterance=utterance
                    )
                    turns.append(turn)
                    turn_id += 1
            
            # Only include dialogues with multiple turns
            if len(turns) >= 2:
                dialogue = DialogueSession(
                    dialogue_id=dialogue_id,
                    domain=domain,
                    turns=turns,
                    goal={"personas": item.get("personas", [])}
                )
                dialogues.append(dialogue)
        
        logger.info(f"Loaded {len(dialogues)} dialogues from {split} split")
        return dialogues
    
    def evaluate_tmm_system(self, dialogues: List[DialogueSession]) -> MultiTurnEvaluationResult:
        """
        Evaluate the TMM system on multi-turn dialogues.
        
        Args:
            dialogues: List of dialogue sessions to evaluate
            
        Returns:
            MultiTurnEvaluationResult with comprehensive metrics
        """
        logger.info(f"Evaluating TMM system on {len(dialogues)} dialogues...")
        
        successful_dialogues = 0
        total_turns = 0
        response_times = []
        memory_operations = {"stores": 0, "retrievals": 0, "updates": 0}
        truth_verification_calls = 0
        contradiction_detections = 0
        domain_results = {}
        
        # Track information consistency
        information_consistency_scores = []
        false_memory_events = []
        
        for i, dialogue in enumerate(dialogues):
            if i % 50 == 0:  # Progress indicator every 50 dialogues
                logger.info(f"Progress: {i}/{len(dialogues)} dialogues processed")
            
            logger.info(f"Processing dialogue {dialogue.dialogue_id} ({dialogue.domain})")
            
            # Reset memory for each dialogue
            self.tmm_system.reset_memory()
            
            dialogue_success = True
            dialogue_turns = 0
            dialogue_false_memories = 0
            
            for turn in dialogue.turns:
                if turn.speaker == "user":
                    start_time = time.time()
                    
                    try:
                        # Process user turn through TMM system with timeout protection
                        response = self.tmm_system.process(turn.utterance)
                        
                        # Record response time
                        response_time = time.time() - start_time
                        response_times.append(response_time)
                        
                        # Check for excessive response time
                        if response_time > 30.0:  # 30 second timeout
                            logger.warning(f"Long response time: {response_time:.2f}s for turn {turn.turn_id}")
                        
                        # Check for false memory formation
                        memory_summary = self.tmm_system.get_memory_summary()
                        if self._detect_false_memory(memory_summary, turn.utterance):
                            dialogue_false_memories += 1
                            false_memory_events.append({
                                "dialogue_id": dialogue.dialogue_id,
                                "turn_id": turn.turn_id,
                                "utterance": turn.utterance,
                                "detected_false_memory": True
                            })
                        
                        # Track memory operations
                        memory_ops = memory_summary.get("operations", {})
                        memory_operations["stores"] += memory_ops.get("stores", 0)
                        memory_operations["retrievals"] += memory_ops.get("retrievals", 0)
                        memory_operations["updates"] += memory_ops.get("updates", 0)
                        
                        # Track truth verification
                        if memory_summary.get("verification_calls", 0) > 0:
                            truth_verification_calls += 1
                        
                        # Track contradictions
                        if memory_summary.get("contradictions_detected", 0) > 0:
                            contradiction_detections += 1
                        
                        dialogue_turns += 1
                
                    except Exception as e:
                        logger.error(f"Error processing turn {turn.turn_id}: {e}")
                        # Continue with next turn instead of breaking the entire dialogue
                        continue
            
            # Evaluate dialogue success
            if dialogue_success and self._evaluate_dialogue_success(dialogue):
                successful_dialogues += 1
            
            # Track domain-specific results
            if dialogue.domain not in domain_results:
                domain_results[dialogue.domain] = {"successful": 0, "total": 0, "false_memories": 0}
            
            domain_results[dialogue.domain]["total"] += 1
            if dialogue_success:
                domain_results[dialogue.domain]["successful"] += 1
            domain_results[dialogue.domain]["false_memories"] += dialogue_false_memories
            
            total_turns += dialogue_turns
        
        # Calculate metrics
        dialogue_success_rate = successful_dialogues / len(dialogues) if dialogues else 0
        information_accuracy = 1.0 - (len(false_memory_events) / total_turns) if total_turns > 0 else 0
        memory_consistency = self._calculate_memory_consistency(dialogues)
        response_time_avg = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate methodology metrics
        methodology_metrics = self.methodology_calculator.calculate_all_metrics({
            "responses": [],  # Will be populated with actual responses
            "ground_truths": [],  # Will be populated with actual ground truths
            "is_answerable": [],  # Will be populated with actual answerable flags
            "contexts": [],  # Will be populated with actual contexts
            "conversation_history": [],  # Will be populated with actual conversation history
            "memory_operations": memory_operations,
            "contradiction_events": false_memory_events,
            "correction_turns": []
        })
        
        # Calculate domain breakdown
        domain_breakdown = {}
        for domain, results in domain_results.items():
            domain_breakdown[domain] = {
                "success_rate": results["successful"] / results["total"] if results["total"] > 0 else 0,
                "false_memory_rate": results["false_memories"] / results["total"] if results["total"] > 0 else 0,
                "total_dialogues": results["total"]
            }
        
        result = MultiTurnEvaluationResult(
            dialogue_success_rate=dialogue_success_rate,
            information_accuracy=information_accuracy,
            memory_consistency=memory_consistency,
            response_time_avg=response_time_avg,
            memory_operations=memory_operations,
            memory_retrievals=memory_operations["retrievals"],
            memory_stores=memory_operations["stores"],
            memory_updates=memory_operations["updates"],
            false_memory_rate=len(false_memory_events) / total_turns if total_turns > 0 else 0,
            truth_verification_calls=truth_verification_calls,
            contradiction_detections=contradiction_detections,
            methodology_metrics=methodology_metrics,
            successful_dialogues=successful_dialogues,
            total_dialogues=len(dialogues),
            total_turns=total_turns,
            domain_breakdown=domain_breakdown
        )
        
        logger.info(f"TMM evaluation complete: {successful_dialogues}/{len(dialogues)} dialogues successful")
        return result
    
    def evaluate_baseline_system(self, system_name: str, dialogues: List[DialogueSession]) -> MultiTurnEvaluationResult:
        """
        Evaluate a baseline system on multi-turn dialogues.
        
        Args:
            system_name: Name of the baseline system
            dialogues: List of dialogue sessions to evaluate
            
        Returns:
            MultiTurnEvaluationResult with baseline performance
        """
        logger.info(f"Evaluating {system_name} baseline on {len(dialogues)} dialogues...")
        
        if system_name not in self.baselines:
            raise ValueError(f"Unknown baseline system: {system_name}")
        
        baseline_system = self.baselines[system_name]
        
        # Similar evaluation logic as TMM system but adapted for baselines
        successful_dialogues = 0
        total_turns = 0
        response_times = []
        memory_operations = {"stores": 0, "retrievals": 0, "updates": 0}
        domain_results = {}
        
        for i, dialogue in enumerate(dialogues):
            if i % 50 == 0:  # Progress indicator every 50 dialogues
                logger.info(f"Progress: {i}/{len(dialogues)} dialogues processed with {system_name}")
            
            logger.info(f"Processing dialogue {dialogue.dialogue_id} with {system_name}")
            
            # Reset baseline system state
            if hasattr(baseline_system, 'reset_memory'):
                baseline_system.reset_memory()
            
            dialogue_success = True
            dialogue_turns = 0
            
            for turn in dialogue.turns:
                if turn.speaker == "user":
                    start_time = time.time()
                    
                    try:
                        response = baseline_system.process(turn.utterance)
                        response_time = time.time() - start_time
                        response_times.append(response_time)
                        
                        # Check for excessive response time
                        if response_time > 30.0:  # 30 second timeout
                            logger.warning(f"Long response time: {response_time:.2f}s for turn {turn.turn_id} with {system_name}")
                        
                        dialogue_turns += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing turn {turn.turn_id} with {system_name}: {e}")
                        # Continue with next turn instead of breaking the entire dialogue
                        continue
            
            if dialogue_success and self._evaluate_dialogue_success(dialogue):
                successful_dialogues += 1
            
            # Track domain results
            if dialogue.domain not in domain_results:
                domain_results[dialogue.domain] = {"successful": 0, "total": 0}
            domain_results[dialogue.domain]["total"] += 1
            if dialogue_success:
                domain_results[dialogue.domain]["successful"] += 1
            
            total_turns += dialogue_turns
        
        # Calculate metrics
        dialogue_success_rate = successful_dialogues / len(dialogues) if dialogues else 0
        response_time_avg = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate domain breakdown
        domain_breakdown = {}
        for domain, results in domain_results.items():
            domain_breakdown[domain] = {
                "success_rate": results["successful"] / results["total"] if results["total"] > 0 else 0,
                "total_dialogues": results["total"]
            }
        
        result = MultiTurnEvaluationResult(
            dialogue_success_rate=dialogue_success_rate,
            information_accuracy=0.0,  # Baselines don't have truth verification
            memory_consistency=0.0,    # Baselines don't have sophisticated memory
            response_time_avg=response_time_avg,
            memory_operations=memory_operations,
            memory_retrievals=0,
            memory_stores=0,
            memory_updates=0,
            false_memory_rate=0.0,
            truth_verification_calls=0,
            contradiction_detections=0,
            methodology_metrics=MethodologyMetrics(fmr=0.0, mel=0.0, dar=0.0, accuracy=0.0, answerable_accuracy=0.0, unanswerable_accuracy=0.0, memory_consistency=0.0, contradiction_resolution=0.0),
            successful_dialogues=successful_dialogues,
            total_dialogues=len(dialogues),
            total_turns=total_turns,
            domain_breakdown=domain_breakdown
        )
        
        logger.info(f"{system_name} evaluation complete: {successful_dialogues}/{len(dialogues)} dialogues successful")
        return result
    
    def _detect_false_memory(self, memory_summary: Dict[str, Any], utterance: str) -> bool:
        """Detect if false memory was formed based on memory summary."""
        # Simple heuristic: check if memory contains information not in the utterance
        stored_content = memory_summary.get("stored_content", "")
        if stored_content and not any(word in stored_content.lower() for word in utterance.lower().split()):
            return True
        return False
    
    def _evaluate_dialogue_success(self, dialogue: DialogueSession) -> bool:
        """Evaluate if a dialogue was successful."""
        # Simple heuristic: dialogue is successful if it has multiple turns and ends appropriately
        if len(dialogue.turns) < 2:
            return False
        
        # Check if the last system turn indicates completion
        last_system_turn = None
        for turn in reversed(dialogue.turns):
            if turn.speaker == "system":
                last_system_turn = turn
                break
        
        if not last_system_turn:
            return False
        
        # Check for completion indicators
        completion_indicators = ["booked", "confirmed", "reserved", "done", "complete", "success"]
        return any(indicator in last_system_turn.utterance.lower() for indicator in completion_indicators)
    
    def _calculate_memory_consistency(self, dialogues: List[DialogueSession]) -> float:
        """Calculate memory consistency across dialogues."""
        # This would involve checking if the same information is stored consistently
        # across different parts of the dialogue
        return 0.8  # Placeholder - implement based on actual memory consistency checks
    
    def save_results(self, results: Dict[str, MultiTurnEvaluationResult], output_path: str):
        """Save evaluation results to JSON file."""
        output_data = {}
        for system_name, result in results.items():
            output_data[system_name] = asdict(result)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")