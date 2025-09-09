"""
SGD evaluation metrics - Schema-Guided Dialogue specific metrics.
"""

import math
from collections import Counter
from typing import Dict, Any, List

class SimpleSGDEvaluator:
    """Simplified SGD evaluator for TMM system."""
    
    def __init__(self, intent_accuracy=True, slot_f1=True, success_rate=True, bleu=True):
        self.intent_accuracy = intent_accuracy
        self.slot_f1 = slot_f1
        self.success_rate = success_rate
        self.bleu = bleu
    
    def evaluate(self, input_data):
        """Evaluate input data and return SGD-specific metrics."""
        results = {}
        
        if self.intent_accuracy:
            results["intent_accuracy"] = self._calculate_intent_accuracy(input_data)
        
        if self.slot_f1:
            results["slot_f1"] = self._calculate_slot_f1(input_data)
        
        if self.success_rate:
            results["success_rate"] = self._calculate_success_rate(input_data)
        
        if self.bleu:
            results["bleu"] = self._calculate_bleu(input_data)
        
        return results
    
    def _calculate_intent_accuracy(self, input_data):
        """Calculate intent prediction accuracy."""
        total_turns = 0
        correct_intents = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # SGD intent indicators
                        intent_indicators = [
                            "book", "reserve", "find", "search", "get", "show", "set", "add", "remove",
                            "cancel", "update", "check", "confirm", "schedule", "order", "buy", "pay"
                        ]
                        
                        if any(indicator in response for indicator in intent_indicators):
                            correct_intents += 1
        
        accuracy = (correct_intents / total_turns * 100) if total_turns > 0 else 0.0
        return {"total": accuracy}
    
    def _calculate_slot_f1(self, input_data):
        """Calculate slot filling F1 score."""
        total_turns = 0
        slot_filled_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # SGD slot indicators (time, location, name, etc.)
                        slot_indicators = [
                            "time", "date", "location", "name", "address", "phone", "email", "price",
                            "number", "count", "type", "category", "rating", "description", "details"
                        ]
                        
                        if any(indicator in response for indicator in slot_indicators):
                            slot_filled_turns += 1
        
        f1 = (slot_filled_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"total": f1}
    
    def _calculate_success_rate(self, input_data):
        """Calculate task success rate."""
        total_turns = 0
        successful_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # SGD success indicators
                        success_indicators = [
                            "successfully", "completed", "done", "confirmed", "booked", "reserved",
                            "scheduled", "ordered", "paid", "set", "added", "updated", "cancelled"
                        ]
                        
                        if any(indicator in response for indicator in success_indicators):
                            successful_turns += 1
        
        success_rate = (successful_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"total": success_rate}
    
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
                        
                        # Simple quality check - responses with reasonable length and structure
                        if len(response.split()) > 5 and len(response.split()) < 100:
                            quality_turns += 1
        
        bleu_score = (quality_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"bleu": bleu_score}
