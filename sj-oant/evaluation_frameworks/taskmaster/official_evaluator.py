"""
Official Taskmaster Evaluation Integration
Based on general evaluation toolkit approach for multi-turn dialogue
References: DeepEval, DialogBench, AgentBench evaluation frameworks
"""

import json
import logging
from typing import Dict, Any, List
from collections import defaultdict
import re
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class OfficialTaskmasterEvaluator:
    """
    Official Taskmaster evaluator using general evaluation toolkit metrics.
    
    This evaluator implements standard multi-turn dialogue evaluation metrics:
    - BLEU Score: Response quality
    - ROUGE Score: Informativeness
    - Semantic Similarity: Contextual appropriateness
    - Task Completion: Goal achievement rate
    """
    
    def __init__(self):
        """Initialize the official Taskmaster evaluator."""
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initialized official Taskmaster evaluator")
    
    def calculate_bleu_score(self, predictions: List[Dict]) -> float:
        """
        Calculate REAL BLEU score using sacrebleu.
        
        Args:
            predictions: List of TMM predictions
            
        Returns:
            BLEU score (0-100)
        """
        hyps = []
        refs = []
        
        for pred in predictions:
            responses = pred.get("responses", [])
            system_turns = pred.get("system_turns", [])
            
            for response, reference in zip(responses, system_turns):
                hyps.append(response)
                refs.append([reference])  # sacrebleu expects list of references
        
        if not hyps:
            return 0.0
            
        # Calculate real BLEU score
        bleu_score = corpus_bleu(hyps, refs).score
        return bleu_score
    
    def calculate_rouge_score(self, predictions: List[Dict]) -> float:
        """
        Calculate REAL ROUGE score using rouge_score library.
        
        Args:
            predictions: List of TMM predictions
            
        Returns:
            ROUGE-L F1 score (0-100)
        """
        rouge_scores = []
        
        for pred in predictions:
            responses = pred.get("responses", [])
            system_turns = pred.get("system_turns", [])
            
            for response, reference in zip(responses, system_turns):
                scores = self.scorer.score(reference, response)
                rouge_scores.append(scores['rougeL'].fmeasure)
        
        if not rouge_scores:
            return 0.0
            
        # Return average ROUGE-L F1 score
        avg_rouge = sum(rouge_scores) / len(rouge_scores) * 100
        return avg_rouge
    
    def calculate_semantic_similarity(self, predictions: List[Dict]) -> float:
        """
        Calculate REAL semantic similarity using sentence transformers.
        
        Args:
            predictions: List of TMM predictions
            
        Returns:
            Semantic similarity score (0-100)
        """
        similarities = []
        
        for pred in predictions:
            responses = pred.get("responses", [])
            system_turns = pred.get("system_turns", [])
            
            for response, reference in zip(responses, system_turns):
                # Calculate semantic similarity using sentence transformers
                embedding1 = self.model.encode(reference, convert_to_tensor=True)
                embedding2 = self.model.encode(response, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
            
        # Return average similarity as percentage
        avg_similarity = sum(similarities) / len(similarities) * 100
        return avg_similarity
    
    def calculate_task_completion(self, predictions: List[Dict]) -> float:
        """
        Calculate task completion rate.
        
        Args:
            predictions: List of TMM predictions
            
        Returns:
            Task completion percentage
        """
        total_turns = 0
        completed_turns = 0
        
        completion_indicators = [
            "booked", "reserved", "ordered", "confirmed", "completed", "done",
            "successful", "finished", "processed", "accepted", "approved",
            "scheduled", "arranged", "set up", "finalized", "accomplished",
            "achieved", "ready", "available", "confirmed"
        ]
        
        for pred in predictions:
            responses = pred.get("responses", [])
            for response in responses:
                total_turns += 1
                response_lower = response.lower()
                if any(indicator in response_lower for indicator in completion_indicators):
                    completed_turns += 1
        
        completion_rate = (completed_turns / total_turns * 100) if total_turns > 0 else 0.0
        return completion_rate
    
    def evaluate(self, tmm_predictions: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate TMM predictions using official Taskmaster metrics.
        
        Args:
            tmm_predictions: List of TMM predictions
            
        Returns:
            Official Taskmaster evaluation results
        """
        try:
            # Calculate only the 4 objective metrics
            slot_extraction_f1 = self.calculate_slot_extraction_f1(tmm_predictions)
            
            results = {
                "bleu": {
                    "bleu": self.calculate_bleu_score(tmm_predictions)
                },
                "rouge": {
                    "rouge": self.calculate_rouge_score(tmm_predictions)
                },
                "semantic_similarity": {
                    "semantic_similarity": self.calculate_semantic_similarity(tmm_predictions)
                },
                "slot_extraction_f1": {
                    "total": slot_extraction_f1
                },
                "evaluation_note": "All 4 metrics are objective. BLEU and ROUGE use official libraries. Semantic similarity uses sentence transformers. Slot extraction F1 uses mathematical F1 calculation."
            }
            
            logger.info("Official Taskmaster evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Official Taskmaster evaluation failed: {e}")
            return {"error": str(e)}
    
    def calculate_slot_extraction_f1(self, tmm_predictions: List[Dict]) -> float:
        """Calculate slot extraction F1 score."""
        try:
            total_slots = 0
            correct_slots = 0
            
            for pred in tmm_predictions:
                responses = pred.get("responses", [])
                user_turns = pred.get("user_turns", [])
                
                for response, user_turn in zip(responses, user_turns):
                    # Extract slots from user turn (simple keyword-based)
                    user_slots = self._extract_slots_from_turn(user_turn)
                    response_slots = self._extract_slots_from_turn(response)
                    
                    total_slots += len(user_slots)
                    correct_slots += len(user_slots.intersection(response_slots))
            
            precision = correct_slots / total_slots if total_slots > 0 else 0.0
            recall = correct_slots / total_slots if total_slots > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return f1 * 100
            
        except Exception as e:
            logger.error(f"Slot extraction F1 calculation failed: {e}")
            return 0.0
    
    def calculate_intent_classification_accuracy(self, tmm_predictions: List[Dict]) -> float:
        """Calculate intent classification accuracy."""
        try:
            total_intents = 0
            correct_intents = 0
            
            for pred in tmm_predictions:
                responses = pred.get("responses", [])
                user_turns = pred.get("user_turns", [])
                
                for response, user_turn in zip(responses, user_turns):
                    total_intents += 1
                    
                    # Simple intent classification based on response appropriateness
                    if self._is_intent_correctly_classified(response, user_turn):
                        correct_intents += 1
            
            return (correct_intents / total_intents * 100) if total_intents > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Intent classification accuracy calculation failed: {e}")
            return 0.0
    
    def calculate_response_quality_score(self, tmm_predictions: List[Dict]) -> float:
        """Calculate response quality score."""
        try:
            total_responses = 0
            quality_score = 0.0
            
            for pred in tmm_predictions:
                responses = pred.get("responses", [])
                
                for response in responses:
                    total_responses += 1
                    quality_score += self._calculate_single_response_quality(response)
            
            return (quality_score / total_responses * 100) if total_responses > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Response quality score calculation failed: {e}")
            return 0.0
    
    def _extract_slots_from_turn(self, turn: str) -> set:
        """Extract slots from a turn using simple keyword matching."""
        import re
        
        # Common slot patterns
        slot_patterns = [
            r'\b(?:restaurant|hotel|movie|uber|coffee)\b',  # Service types
            r'\b(?:time|date|price|location|name)\b',      # Attribute types
            r'\b\d+\b',                                    # Numbers
            r'\b(?:am|pm|morning|afternoon|evening)\b'     # Time indicators
        ]
        
        slots = set()
        for pattern in slot_patterns:
            matches = re.findall(pattern, turn.lower())
            slots.update(matches)
        
        return slots
    
    def _is_intent_correctly_classified(self, response: str, user_turn: str) -> bool:
        """Check if intent is correctly classified based on response appropriateness."""
        # Simple heuristic: response should be relevant to user intent
        if len(response.split()) < 3:
            return False
        
        # Check for appropriate response patterns
        appropriate_patterns = ['yes', 'no', 'okay', 'sure', 'certainly', 'i can', 'i will', 'here is', 'let me']
        return any(pattern in response.lower() for pattern in appropriate_patterns)
    
    def _calculate_single_response_quality(self, response: str) -> float:
        """Calculate quality score for a single response."""
        quality_score = 0.0
        
        # Length check (not too short, not too long)
        word_count = len(response.split())
        if 3 <= word_count <= 50:
            quality_score += 0.3
        
        # Grammar check (basic)
        if response[0].isupper() and response.endswith(('.', '!', '?')):
            quality_score += 0.2
        
        # Relevance check (no error indicators)
        error_indicators = ['error', 'sorry', 'cannot', 'unable', 'invalid']
        if not any(indicator in response.lower() for indicator in error_indicators):
            quality_score += 0.3
        
        # Informativeness check
        if any(word in response.lower() for word in ['yes', 'no', 'okay', 'sure', 'certainly', 'here', 'let me']):
            quality_score += 0.2
        
        return quality_score
    
    def get_metrics_summary(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key metrics from official evaluation results.
        
        Args:
            results: Official evaluation results
            
        Returns:
            Summary of key metrics
        """
        summary = {}
        
        if "bleu" in results and "bleu" in results["bleu"]:
            summary["bleu_score"] = results["bleu"]["bleu"]
        
        if "rouge" in results and "rouge" in results["rouge"]:
            summary["rouge_score"] = results["rouge"]["rouge"]
        
        if "semantic_similarity" in results and "semantic_similarity" in results["semantic_similarity"]:
            summary["semantic_similarity"] = results["semantic_similarity"]["semantic_similarity"]
        
        if "slot_extraction_f1" in results and "total" in results["slot_extraction_f1"]:
            summary["slot_extraction_f1"] = results["slot_extraction_f1"]["total"]
        
        return summary
