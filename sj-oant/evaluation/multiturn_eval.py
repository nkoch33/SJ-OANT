"""
evaluation.multiturn_eval - Multi-Turn Conversation Evaluation

This module implements multi-turn conversation evaluation specifically designed
to test TMM's memory persistence and retrieval capabilities across extended
interactions. This evaluation demonstrates TMM's core advantage over baseline
systems that lack sophisticated memory management.

Key Features:
- Tests memory persistence across multiple question-answer turns
- Evaluates context retention and retrieval accuracy
- Measures memory-dependent reasoning capabilities
- Provides detailed memory utilization metrics
"""

import logging
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

from evaluation.squad_eval import SQuADExample, EvaluationResult, SQuADEvaluator
from tmm_pipeline import TMMPipeline

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ConversationTurn:
    """Represents a single turn in a multi-turn conversation."""
    turn_id: int
    question: str
    expected_answer: str
    is_answerable: bool
    requires_memory: bool  # Whether answer depends on previous context
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ConversationScenario:
    """Represents a complete multi-turn conversation scenario."""
    scenario_id: str
    context: str
    title: str
    turns: List[ConversationTurn]
    memory_dependency_level: str  # "low", "medium", "high"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MultiTurnResult:
    """Results for a multi-turn conversation evaluation."""
    scenario_id: str
    system_name: str
    total_turns: int
    correct_answers: int
    memory_dependent_correct: int
    memory_dependent_total: int
    conversation_accuracy: float
    memory_dependent_accuracy: float
    avg_response_time: float
    memory_metrics: Dict[str, Any] = field(default_factory=dict)
    turn_results: List[Dict[str, Any]] = field(default_factory=list)

class MultiTurnEvaluator:
    """
    Evaluator for multi-turn conversations that test memory persistence.
    
    This evaluator creates conversation scenarios from SQuAD 2.0 data where
    multiple questions are asked about the same context, testing the system's
    ability to maintain and retrieve information across turns.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the multi-turn evaluator."""
        self.cache_dir = cache_dir
        self.squad_evaluator = SQuADEvaluator(cache_dir)
        logger.info("Initialized MultiTurn evaluator")
    
    def create_conversation_scenarios(self, squad_examples: List[SQuADExample], 
                                    scenario_count: int = 20) -> List[ConversationScenario]:
        """
        Create multi-turn conversation scenarios from SQuAD examples.
        
        Groups questions by context and creates conversation flows that test
        memory persistence and retrieval across multiple turns.
        
        Args:
            squad_examples: List of SQuAD examples to group into conversations
            scenario_count: Number of conversation scenarios to create
            
        Returns:
            List of conversation scenarios for evaluation
        """
        # Group examples by context
        context_groups = {}
        for example in squad_examples:
            context_key = example.context[:100]  # Use first 100 chars as key
            if context_key not in context_groups:
                context_groups[context_key] = []
            context_groups[context_key].append(example)
        
        # Filter to contexts with multiple questions
        multi_question_contexts = {k: v for k, v in context_groups.items() 
                                 if len(v) >= 2}
        
        logger.info(f"Found {len(multi_question_contexts)} contexts with multiple questions")
        
        scenarios = []
        scenario_count = min(scenario_count, len(multi_question_contexts))
        
        for i, (context_key, examples) in enumerate(list(multi_question_contexts.items())[:scenario_count]):
            # Take first example for context, create turns from all examples
            base_example = examples[0]
            turns = []
            
            for turn_idx, example in enumerate(examples[:4]):  # Max 4 turns per conversation
                # Determine if this turn requires memory from previous turns
                requires_memory = turn_idx > 0  # First turn doesn't require memory, subsequent ones do
                
                turn = ConversationTurn(
                    turn_id=turn_idx,
                    question=example.question,
                    expected_answer=example.answer,
                    is_answerable=example.is_answerable,
                    requires_memory=requires_memory,
                    metadata={
                        "original_id": example.id,
                        "answer_start": example.metadata.get("answer_start", -1)
                    }
                )
                turns.append(turn)
            
            # Determine memory dependency level based on turn count and complexity
            if len(turns) == 2:
                memory_level = "low"
            elif len(turns) == 3:
                memory_level = "medium"
            else:
                memory_level = "high"
            
            scenario = ConversationScenario(
                scenario_id=f"conv_{i:03d}",
                context=base_example.context,
                title=base_example.title,
                turns=turns,
                memory_dependency_level=memory_level,
                metadata={
                    "context_length": len(base_example.context.split()),
                    "source_examples": [ex.id for ex in examples[:4]]
                }
            )
            scenarios.append(scenario)
        
        logger.info(f"Created {len(scenarios)} conversation scenarios")
        return scenarios
    
    def evaluate_conversation(self, system: Any, scenario: ConversationScenario, 
                            system_name: str) -> MultiTurnResult:
        """
        Evaluate a system on a single multi-turn conversation scenario.
        
        Args:
            system: System to evaluate (TMM or baseline)
            scenario: Conversation scenario to run
            system_name: Name of the system being evaluated
            
        Returns:
            Results for this conversation scenario
        """
        # Reset system memory if applicable
        if hasattr(system, 'reset_memory'):
            system.reset_memory()
        
        start_time = time.perf_counter()
        turn_results = []
        correct_answers = 0
        memory_dependent_correct = 0
        memory_dependent_total = 0
        total_time = 0.0
        memory_metrics = {}
        
        # Step 1: Provide context to system
        try:
            if hasattr(system, 'process'):
                # For TMM pipeline, store context in memory
                context_prompt = f"Please remember this context: {scenario.context}"
                system.process(context_prompt)
            elif hasattr(system, 'process_story'):
                # For baseline systems with story processing
                system.process_story(scenario.context)
        except Exception as e:
            logger.error(f"Error providing context: {e}")
        
        # Step 2: Process each turn in the conversation
        for turn in scenario.turns:
            turn_start = time.perf_counter()
            
            try:
                # Get system response
                if hasattr(system, 'process'):
                    response = system.process(turn.question)
                else:
                    # For systems without memory, include context with each question
                    full_prompt = f"Context: {scenario.context}\n\nQuestion: {turn.question}"
                    response = system.process(full_prompt)
                
                # Evaluate response
                is_correct = self.squad_evaluator._check_answer_v2(
                    response, turn.expected_answer, turn.is_answerable
                )
                
                if is_correct:
                    correct_answers += 1
                    if turn.requires_memory:
                        memory_dependent_correct += 1
                
                if turn.requires_memory:
                    memory_dependent_total += 1
                
                turn_elapsed = time.perf_counter() - turn_start
                total_time += turn_elapsed
                
                # Collect memory metrics for TMM
                turn_memory_metrics = {}
                if hasattr(system, 'memory_store') and hasattr(system.memory_store, 'get_metrics'):
                    turn_memory_metrics = system.memory_store.get_metrics()
                
                turn_result = {
                    "turn_id": turn.turn_id,
                    "question": turn.question,
                    "response": response,
                    "expected_answer": turn.expected_answer,
                    "is_correct": is_correct,
                    "requires_memory": turn.requires_memory,
                    "response_time": turn_elapsed,
                    "memory_metrics": turn_memory_metrics
                }
                turn_results.append(turn_result)
                
                logger.debug(f"Turn {turn.turn_id}: {'CORRECT' if is_correct else 'INCORRECT'} "
                           f"({'memory-dependent' if turn.requires_memory else 'independent'})")
                
            except Exception as e:
                logger.error(f"Error on turn {turn.turn_id}: {e}")
                turn_result = {
                    "turn_id": turn.turn_id,
                    "question": turn.question,
                    "response": f"Error: {str(e)}",
                    "expected_answer": turn.expected_answer,
                    "is_correct": False,
                    "requires_memory": turn.requires_memory,
                    "response_time": time.perf_counter() - turn_start,
                    "memory_metrics": {}
                }
                turn_results.append(turn_result)
                
                if turn.requires_memory:
                    memory_dependent_total += 1
        
        # Calculate final metrics
        conversation_accuracy = correct_answers / len(scenario.turns) if scenario.turns else 0.0
        memory_dependent_accuracy = (memory_dependent_correct / memory_dependent_total 
                                   if memory_dependent_total > 0 else 0.0)
        avg_response_time = total_time / len(scenario.turns) if scenario.turns else 0.0
        
        # Collect overall memory metrics for TMM
        if hasattr(system, 'memory_store') and hasattr(system.memory_store, 'get_metrics'):
            memory_metrics = system.memory_store.get_metrics()
        
        return MultiTurnResult(
            scenario_id=scenario.scenario_id,
            system_name=system_name,
            total_turns=len(scenario.turns),
            correct_answers=correct_answers,
            memory_dependent_correct=memory_dependent_correct,
            memory_dependent_total=memory_dependent_total,
            conversation_accuracy=conversation_accuracy,
            memory_dependent_accuracy=memory_dependent_accuracy,
            avg_response_time=avg_response_time,
            memory_metrics=memory_metrics,
            turn_results=turn_results
        )
    
    def evaluate_system_multiturn(self, system: Any, scenarios: List[ConversationScenario],
                                system_name: str) -> Dict[str, Any]:
        """
        Evaluate a system on multiple conversation scenarios.
        
        Args:
            system: System to evaluate
            scenarios: List of conversation scenarios
            system_name: Name of the system
            
        Returns:
            Aggregated results across all scenarios
        """
        logger.info(f"Evaluating {system_name} on {len(scenarios)} conversation scenarios")
        
        scenario_results = []
        total_conversations = len(scenarios)
        total_turns = 0
        total_correct = 0
        total_memory_dependent_correct = 0
        total_memory_dependent = 0
        total_time = 0.0
        
        for scenario in scenarios:
            result = self.evaluate_conversation(system, scenario, system_name)
            scenario_results.append(result)
            
            total_turns += result.total_turns
            total_correct += result.correct_answers
            total_memory_dependent_correct += result.memory_dependent_correct
            total_memory_dependent += result.memory_dependent_total
            total_time += result.avg_response_time * result.total_turns
        
        # Calculate aggregate metrics
        overall_accuracy = total_correct / total_turns if total_turns > 0 else 0.0
        memory_accuracy = (total_memory_dependent_correct / total_memory_dependent 
                         if total_memory_dependent > 0 else 0.0)
        avg_response_time = total_time / total_turns if total_turns > 0 else 0.0
        
        logger.info(f"{system_name} multi-turn evaluation complete:")
        logger.info(f"  Overall accuracy: {total_correct}/{total_turns} ({overall_accuracy:.2%})")
        logger.info(f"  Memory-dependent accuracy: {total_memory_dependent_correct}/{total_memory_dependent} ({memory_accuracy:.2%})")
        
        return {
            "system_name": system_name,
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "overall_accuracy": overall_accuracy,
            "memory_dependent_accuracy": memory_accuracy,
            "avg_response_time": avg_response_time,
            "scenario_results": [asdict(result) for result in scenario_results],
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save multi-turn evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Multi-turn evaluation results saved to {output_file}")
