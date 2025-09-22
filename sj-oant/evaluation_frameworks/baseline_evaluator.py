"""
Baseline Evaluator for Dialogue and False Memory Testing

Provides a unified interface for evaluating baseline models (LLMs and RAG systems)
on both dialogue performance and false memory prevention tasks.
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from false_memory_evaluation.baseline_models import (
    BaseLLMWrapper, SimpleRAGBaseline, EmbeddingRAGBaseline,
    SimpleRAGConfig, EmbeddingRAGConfig
)

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for baseline evaluation."""
    num_conversations: int = 100
    max_turns_per_conversation: int = 10
    output_dir: str = "results"
    save_responses: bool = True
    save_metrics: bool = True

class BaselineEvaluator:
    """
    Unified evaluator for baseline models on dialogue and false memory tasks.
    
    Supports evaluation of:
    - Standard LLMs (Llama-2, Mistral, GPT-3.5)
    - RAG systems (Simple RAG, Embedding RAG)
    - Both dialogue performance and false memory prevention
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize baseline evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.results = {}
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized BaselineEvaluator with config: {self.config}")
    
    def evaluate_dialogue_performance(
        self,
        model: BaseLLMWrapper,
        benchmark: str,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate dialogue performance on a specific benchmark.
        
        Args:
            model: Model to evaluate
            benchmark: Benchmark name (multiwoz, sgd, taskmaster)
            conversations: List of conversations to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {model.model_name} on {benchmark} dialogue performance")
        
        start_time = time.time()
        responses = []
        metrics = {
            "bleu_scores": [],
            "rouge_scores": [],
            "semantic_similarities": [],
            "slot_f1_scores": [],
            "intent_accuracies": [],
            "response_times": []
        }
        
        # Limit conversations if specified
        eval_conversations = conversations[:self.config.num_conversations]
        
        for i, conversation in enumerate(eval_conversations):
            logger.debug(f"Processing conversation {i+1}/{len(eval_conversations)}")
            
            conversation_responses = []
            model.clear_memory()  # Clear memory for each conversation
            
            for turn in conversation.get("turns", []):
                if turn.get("role") == "user":
                    user_input = turn.get("content", "")
                    
                    # Generate response
                    response = model.generate_response(
                        user_input,
                        conversation_history=conversation_responses
                    )
                    
                    conversation_responses.append({
                        "role": "user",
                        "content": user_input
                    })
                    conversation_responses.append({
                        "role": "assistant", 
                        "content": response.text
                    })
                    
                    # Calculate metrics (simplified for baseline)
                    metrics["response_times"].append(response.response_time)
                    
                    # Mock metric calculations (in practice, these would use proper evaluation libraries)
                    metrics["bleu_scores"].append(self._mock_bleu_score(response.text, turn.get("reference", "")))
                    metrics["rouge_scores"].append(self._mock_rouge_score(response.text, turn.get("reference", "")))
                    metrics["semantic_similarities"].append(self._mock_semantic_similarity(response.text, turn.get("reference", "")))
                    metrics["slot_f1_scores"].append(self._mock_slot_f1(response.text, turn.get("slots", [])))
                    metrics["intent_accuracies"].append(self._mock_intent_accuracy(response.text, turn.get("intent", "")))
            
            responses.append({
                "conversation_id": conversation.get("id", i),
                "responses": conversation_responses
            })
        
        # Calculate aggregate metrics
        results = {
            "model_name": model.model_name,
            "benchmark": benchmark,
            "num_conversations": len(eval_conversations),
            "evaluation_time": time.time() - start_time,
            "metrics": {
                "bleu": sum(metrics["bleu_scores"]) / len(metrics["bleu_scores"]) if metrics["bleu_scores"] else 0.0,
                "rouge": sum(metrics["rouge_scores"]) / len(metrics["rouge_scores"]) if metrics["rouge_scores"] else 0.0,
                "semantic_similarity": sum(metrics["semantic_similarities"]) / len(metrics["semantic_similarities"]) if metrics["semantic_similarities"] else 0.0,
                "slot_f1": sum(metrics["slot_f1_scores"]) / len(metrics["slot_f1_scores"]) if metrics["slot_f1_scores"] else 0.0,
                "intent_accuracy": sum(metrics["intent_accuracies"]) / len(metrics["intent_accuracies"]) if metrics["intent_accuracies"] else 0.0,
                "avg_response_time": sum(metrics["response_times"]) / len(metrics["response_times"]) if metrics["response_times"] else 0.0
            },
            "model_stats": model.get_stats()
        }
        
        # Save results if requested
        if self.config.save_responses:
            self._save_responses(model.model_name, benchmark, responses)
        
        if self.config.save_metrics:
            self._save_metrics(model.model_name, benchmark, results)
        
        return results
    
    def evaluate_false_memory_prevention(
        self,
        model: BaseLLMWrapper,
        benchmark: str,
        conversations: List[Dict[str, Any]],
        false_memory_injector: Any = None
    ) -> Dict[str, Any]:
        """
        Evaluate false memory prevention capabilities.
        
        Args:
            model: Model to evaluate
            benchmark: Benchmark name
            conversations: List of conversations with false memory injection
            false_memory_injector: Injector for false memories
            
        Returns:
            Dictionary of false memory evaluation results
        """
        logger.info(f"Evaluating {model.model_name} on {benchmark} false memory prevention")
        
        start_time = time.time()
        responses = []
        false_memory_metrics = {
            "fmr_scores": [],  # False Memory Rate
            "mel_scores": [],  # Memory Edit Latency
            "dar_scores": [],  # Disturbance Adaptation Rate
            "contradiction_detections": []
        }
        
        # Limit conversations if specified
        eval_conversations = conversations[:self.config.num_conversations]
        
        for i, conversation in enumerate(eval_conversations):
            logger.debug(f"Processing false memory conversation {i+1}/{len(eval_conversations)}")
            
            conversation_responses = []
            model.clear_memory()  # Clear memory for each conversation
            
            # Track false memory injection
            false_memories_injected = []
            false_memories_repeated = []
            
            for turn in conversation.get("turns", []):
                if turn.get("role") == "user":
                    user_input = turn.get("content", "")
                    
                    # Check if this turn contains false memory injection
                    if turn.get("false_memory_injected"):
                        false_memories_injected.append(turn.get("false_fact", ""))
                    
                    # Generate response
                    response = model.generate_response(
                        user_input,
                        conversation_history=conversation_responses
                    )
                    
                    # Check if response repeats false information
                    if self._contains_false_information(response.text, false_memories_injected):
                        false_memories_repeated.append({
                            "turn": len(conversation_responses) // 2,
                            "false_fact": turn.get("false_fact", ""),
                            "response": response.text
                        })
                    
                    conversation_responses.append({
                        "role": "user",
                        "content": user_input
                    })
                    conversation_responses.append({
                        "role": "assistant",
                        "content": response.text
                    })
            
            # Calculate false memory metrics for this conversation
            fmr = len(false_memories_repeated) / len(false_memories_injected) if false_memories_injected else 0.0
            mel = self._calculate_mel(false_memories_repeated)  # Mock calculation
            dar = self._calculate_dar(conversation_responses, false_memories_injected)  # Mock calculation
            
            false_memory_metrics["fmr_scores"].append(fmr)
            false_memory_metrics["mel_scores"].append(mel)
            false_memory_metrics["dar_scores"].append(dar)
            false_memory_metrics["contradiction_detections"].append(len(false_memories_repeated))
            
            responses.append({
                "conversation_id": conversation.get("id", i),
                "responses": conversation_responses,
                "false_memories_injected": false_memories_injected,
                "false_memories_repeated": false_memories_repeated
            })
        
        # Calculate aggregate metrics
        results = {
            "model_name": model.model_name,
            "benchmark": benchmark,
            "num_conversations": len(eval_conversations),
            "evaluation_time": time.time() - start_time,
            "false_memory_metrics": {
                "fmr": sum(false_memory_metrics["fmr_scores"]) / len(false_memory_metrics["fmr_scores"]) if false_memory_metrics["fmr_scores"] else 0.0,
                "mel": sum(false_memory_metrics["mel_scores"]) / len(false_memory_metrics["mel_scores"]) if false_memory_metrics["mel_scores"] else 0.0,
                "dar": sum(false_memory_metrics["dar_scores"]) / len(false_memory_metrics["dar_scores"]) if false_memory_metrics["dar_scores"] else 0.0,
                "contradiction_detection_rate": sum(false_memory_metrics["contradiction_detections"]) / len(false_memory_metrics["contradiction_detections"]) if false_memory_metrics["contradiction_detections"] else 0.0
            },
            "model_stats": model.get_stats()
        }
        
        # Save results if requested
        if self.config.save_responses:
            self._save_false_memory_responses(model.model_name, benchmark, responses)
        
        if self.config.save_metrics:
            self._save_false_memory_metrics(model.model_name, benchmark, results)
        
        return results
    
    def _mock_bleu_score(self, generated: str, reference: str) -> float:
        """Mock BLEU score calculation."""
        # Simple word overlap as proxy for BLEU
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return 0.0
        return len(gen_words & ref_words) / len(ref_words)
    
    def _mock_rouge_score(self, generated: str, reference: str) -> float:
        """Mock ROUGE score calculation."""
        # Simple word overlap as proxy for ROUGE
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()
        if not ref_words:
            return 0.0
        overlap = len(set(gen_words) & set(ref_words))
        return overlap / len(ref_words)
    
    def _mock_semantic_similarity(self, generated: str, reference: str) -> float:
        """Mock semantic similarity calculation."""
        # Simple word overlap as proxy for semantic similarity
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        if not gen_words or not ref_words:
            return 0.0
        return len(gen_words & ref_words) / len(gen_words | ref_words)
    
    def _mock_slot_f1(self, generated: str, slots: List[Dict[str, str]]) -> float:
        """Mock slot F1 calculation."""
        # Simple keyword matching as proxy for slot F1
        if not slots:
            return 1.0
        gen_lower = generated.lower()
        correct_slots = 0
        for slot in slots:
            if slot.get("value", "").lower() in gen_lower:
                correct_slots += 1
        return correct_slots / len(slots)
    
    def _mock_intent_accuracy(self, generated: str, intent: str) -> float:
        """Mock intent accuracy calculation."""
        # Simple keyword matching as proxy for intent accuracy
        if not intent:
            return 1.0
        gen_lower = generated.lower()
        intent_lower = intent.lower()
        return 1.0 if intent_lower in gen_lower else 0.0
    
    def _contains_false_information(self, response: str, false_facts: List[str]) -> bool:
        """Check if response contains false information."""
        response_lower = response.lower()
        for false_fact in false_facts:
            if false_fact.lower() in response_lower:
                return True
        return False
    
    def _calculate_mel(self, false_memories_repeated: List[Dict[str, Any]]) -> float:
        """Calculate Memory Edit Latency (mock)."""
        # Mock calculation: assume immediate detection for baseline models
        return 0.0 if not false_memories_repeated else 1.0
    
    def _calculate_dar(self, conversation_responses: List[Dict[str, str]], false_memories_injected: List[str]) -> float:
        """Calculate Disturbance Adaptation Rate (mock)."""
        # Mock calculation: assume good adaptation for baseline models
        return 0.8 if false_memories_injected else 1.0
    
    def _save_responses(self, model_name: str, benchmark: str, responses: List[Dict[str, Any]]):
        """Save dialogue responses to file."""
        filename = f"{self.config.output_dir}/{model_name}_{benchmark}_dialogue_responses.json"
        with open(filename, 'w') as f:
            json.dump(responses, f, indent=2)
        logger.info(f"Saved dialogue responses to {filename}")
    
    def _save_metrics(self, model_name: str, benchmark: str, results: Dict[str, Any]):
        """Save dialogue metrics to file."""
        filename = f"{self.config.output_dir}/{model_name}_{benchmark}_dialogue_metrics.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved dialogue metrics to {filename}")
    
    def _save_false_memory_responses(self, model_name: str, benchmark: str, responses: List[Dict[str, Any]]):
        """Save false memory responses to file."""
        filename = f"{self.config.output_dir}/{model_name}_{benchmark}_false_memory_responses.json"
        with open(filename, 'w') as f:
            json.dump(responses, f, indent=2)
        logger.info(f"Saved false memory responses to {filename}")
    
    def _save_false_memory_metrics(self, model_name: str, benchmark: str, results: Dict[str, Any]):
        """Save false memory metrics to file."""
        filename = f"{self.config.output_dir}/{model_name}_{benchmark}_false_memory_metrics.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved false memory metrics to {filename}")
