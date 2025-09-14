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
                "task_completion": {
                    "task_completion": self.calculate_task_completion(tmm_predictions)
                }
            }
            
            logger.info("Official Taskmaster evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Official Taskmaster evaluation failed: {e}")
            return {"error": str(e)}
    
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
        
        if "task_completion" in results and "task_completion" in results["task_completion"]:
            summary["task_completion"] = results["task_completion"]["task_completion"]
        
        return summary
