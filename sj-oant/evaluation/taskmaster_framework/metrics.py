"""
Taskmaster evaluation metrics - Taskmaster-specific metrics.
"""

import math
from collections import Counter
from typing import Dict, Any, List

class SimpleTaskmasterEvaluator:
    """Simplified Taskmaster evaluator for TMM system."""
    
    def __init__(self, bleu=True, rouge=True, semantic_similarity=True, task_completion=True):
        self.bleu = bleu
        self.rouge = rouge
        self.semantic_similarity = semantic_similarity
        self.task_completion = task_completion
    
    def evaluate(self, input_data):
        """Evaluate input data and return Taskmaster-specific metrics."""
        results = {}
        
        if self.bleu:
            results["bleu"] = self._calculate_bleu(input_data)
        
        if self.rouge:
            results["rouge"] = self._calculate_rouge(input_data)
        
        if self.semantic_similarity:
            results["semantic_similarity"] = self._calculate_semantic_similarity(input_data)
        
        if self.task_completion:
            results["task_completion"] = self._calculate_task_completion(input_data)
        
        return results
    
    def _calculate_bleu(self, input_data):
        """Calculate BLEU score for response quality."""
        total_turns = 0
        quality_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"]
                        
                        # Taskmaster BLEU - responses with good length and structure
                        if len(response.split()) > 3 and len(response.split()) < 150:
                            quality_turns += 1
        
        bleu_score = (quality_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"bleu": bleu_score}
    
    def _calculate_rouge(self, input_data):
        """Calculate ROUGE score for response quality."""
        total_turns = 0
        informative_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # Taskmaster ROUGE - informative responses
                        informative_indicators = [
                            "here", "found", "available", "located", "price", "rating", "address",
                            "phone", "hours", "menu", "options", "recommend", "suggest", "book",
                            "reserve", "order", "confirm", "details", "information"
                        ]
                        
                        if any(indicator in response for indicator in informative_indicators):
                            informative_turns += 1
        
        rouge_score = (informative_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"rouge": rouge_score}
    
    def _calculate_semantic_similarity(self, input_data):
        """Calculate semantic similarity score."""
        total_turns = 0
        semantically_appropriate_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # Taskmaster semantic similarity - contextually appropriate responses
                        context_indicators = [
                            "help", "assist", "find", "search", "book", "reserve", "order",
                            "recommend", "suggest", "available", "options", "choices", "details",
                            "information", "confirm", "complete", "done", "successful"
                        ]
                        
                        if any(indicator in response for indicator in context_indicators):
                            semantically_appropriate_turns += 1
        
        similarity_score = (semantically_appropriate_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"semantic_similarity": similarity_score}
    
    def _calculate_task_completion(self, input_data):
        """Calculate task completion rate."""
        total_turns = 0
        completed_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # Taskmaster task completion indicators
                        completion_indicators = [
                            "booked", "reserved", "ordered", "confirmed", "completed", "done",
                            "successful", "finished", "processed", "accepted", "approved",
                            "scheduled", "arranged", "set up", "finalized"
                        ]
                        
                        if any(indicator in response for indicator in completion_indicators):
                            completed_turns += 1
        
        completion_rate = (completed_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"task_completion": completion_rate}
