"""
MultiDoGO evaluation metrics - Multi-Domain Goal-Oriented Dialogues specific metrics.
"""

import math
from collections import Counter
from typing import Dict, Any, List

class SimpleMultiDoGOEvaluator:
    """Simplified MultiDoGO evaluator for TMM system."""
    
    def __init__(self, intent_accuracy=True, slot_f1=True, domain_adaptation=True, response_quality=True):
        self.intent_accuracy = intent_accuracy
        self.slot_f1 = slot_f1
        self.domain_adaptation = domain_adaptation
        self.response_quality = response_quality
    
    def evaluate(self, input_data):
        """Evaluate input data and return MultiDoGO-specific metrics."""
        results = {}
        
        if self.intent_accuracy:
            results["intent_accuracy"] = self._calculate_intent_accuracy(input_data)
        
        if self.slot_f1:
            results["slot_f1"] = self._calculate_slot_f1(input_data)
        
        if self.domain_adaptation:
            results["domain_adaptation"] = self._calculate_domain_adaptation(input_data)
        
        if self.response_quality:
            results["response_quality"] = self._calculate_response_quality(input_data)
        
        return results
    
    def _calculate_intent_accuracy(self, input_data):
        """Calculate intent classification accuracy."""
        total_turns = 0
        correct_intents = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # MultiDoGO intent indicators across domains
                        intent_indicators = [
                            # General intents
                            "book", "reserve", "order", "cancel", "change", "check", "get", "find",
                            "help", "assist", "support", "confirm", "update", "modify", "schedule",
                            # Domain-specific intents
                            "flight", "seat", "ticket", "food", "menu", "delivery", "payment",
                            "account", "balance", "transaction", "claim", "policy", "coverage",
                            "content", "subscription", "software", "technical", "troubleshoot"
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
                        
                        # MultiDoGO slot indicators across domains
                        slot_indicators = [
                            # General slots
                            "name", "date", "time", "location", "address", "phone", "email",
                            "number", "id", "reference", "confirmation", "price", "amount",
                            # Domain-specific slots
                            "departure", "arrival", "seat", "flight", "airline", "passenger",
                            "food", "item", "quantity", "size", "topping", "delivery", "payment",
                            "account", "balance", "transaction", "card", "bank", "policy",
                            "claim", "coverage", "premium", "deductible", "content", "subscription",
                            "software", "version", "error", "issue", "solution"
                        ]
                        
                        if any(indicator in response for indicator in slot_indicators):
                            slot_filled_turns += 1
        
        f1 = (slot_filled_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"total": f1}
    
    def _calculate_domain_adaptation(self, input_data):
        """Calculate domain adaptation capability."""
        total_turns = 0
        domain_appropriate_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"].lower()
                        
                        # MultiDoGO domain adaptation indicators
                        domain_indicators = [
                            # Airline domain
                            "flight", "airline", "seat", "boarding", "departure", "arrival", "gate",
                            # Fastfood domain
                            "food", "menu", "order", "delivery", "restaurant", "meal", "drink",
                            # Finance domain
                            "account", "bank", "payment", "transaction", "balance", "card", "money",
                            # Insurance domain
                            "policy", "claim", "coverage", "premium", "deductible", "insurance",
                            # Media domain
                            "content", "subscription", "streaming", "video", "music", "media",
                            # Software domain
                            "software", "technical", "support", "error", "issue", "solution", "bug"
                        ]
                        
                        if any(indicator in response for indicator in domain_indicators):
                            domain_appropriate_turns += 1
        
        adaptation_score = (domain_appropriate_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"total": adaptation_score}
    
    def _calculate_response_quality(self, input_data):
        """Calculate response quality score."""
        total_turns = 0
        quality_turns = 0
        
        for dialog in input_data.values():
            if isinstance(dialog, list):
                for turn in dialog:
                    if "response" in turn:
                        total_turns += 1
                        response = turn["response"]
                        
                        # MultiDoGO response quality indicators
                        quality_indicators = [
                            "here", "found", "available", "located", "price", "details", "information",
                            "help", "assist", "support", "confirm", "booked", "reserved", "ordered",
                            "successful", "completed", "done", "processed", "updated", "scheduled"
                        ]
                        
                        # Check for reasonable length and quality indicators
                        if (len(response.split()) > 3 and len(response.split()) < 200 and
                            any(indicator in response.lower() for indicator in quality_indicators)):
                            quality_turns += 1
        
        quality_score = (quality_turns / total_turns * 100) if total_turns > 0 else 0.0
        return {"total": quality_score}
