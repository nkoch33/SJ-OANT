"""
Contradiction Detection Module

Analyzes TMM's ability to detect and respond to contradictions in conversations.
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContradictionAnalysis:
    """Analysis results for contradiction detection."""
    contradictions_detected: int
    total_contradictions: int
    detection_rate: float
    response_types: Dict[str, int]  # Types of responses to contradictions
    correction_quality: float  # Quality of corrections made

class ContradictionDetector:
    """
    Detects and analyzes contradictions in TMM responses.
    
    This class evaluates how well TMM identifies conflicting information
    and responds appropriately to contradictions.
    """
    
    def __init__(self):
        """Initialize the contradiction detector."""
        self.contradiction_indicators = [
            "contradict", "contradiction", "conflict", "inconsistent",
            "error", "mistake", "wrong", "incorrect", "false",
            "clarify", "correct", "update", "change", "revise",
            "actually", "however", "but", "although", "despite"
        ]
        
        self.correction_indicators = [
            "correct", "actually", "clarify", "update", "change",
            "revise", "mistake", "error", "wrong", "not",
            "instead", "rather", "on the contrary"
        ]
        
        self.acknowledgment_indicators = [
            "understand", "see", "get it", "acknowledge", "recognize",
            "realize", "note", "observe", "notice"
        ]
    
    def analyze_contradictions(self, tmm_responses: List[str],
                             false_information: Dict[str, str]) -> ContradictionAnalysis:
        """
        Analyze TMM's response to contradictions.
        
        Args:
            tmm_responses: List of TMM responses
            false_information: Dict of seeded false information
            
        Returns:
            ContradictionAnalysis with detection results
        """
        contradictions_detected = 0
        total_contradictions = len(false_information)
        response_types = {
            "correction": 0,
            "acknowledgment": 0,
            "ignored": 0,
            "uncertain": 0
        }
        
        correction_qualities = []
        
        for response in tmm_responses:
            response_lower = response.lower()
            
            # Check if response addresses any false information
            for turn_id, false_fact in false_information.items():
                if self._addresses_false_information(response_lower, false_fact.lower()):
                    contradictions_detected += 1
                    
                    # Classify response type
                    response_type = self._classify_contradiction_response(response_lower)
                    response_types[response_type] += 1
                    
                    # Evaluate correction quality
                    if response_type == "correction":
                        quality = self._evaluate_correction_quality(response_lower, false_fact.lower())
                        correction_qualities.append(quality)
        
        # Calculate detection rate
        detection_rate = (contradictions_detected / total_contradictions * 100) if total_contradictions > 0 else 0.0
        
        # Calculate average correction quality
        correction_quality = (sum(correction_qualities) / len(correction_qualities)) if correction_qualities else 0.0
        
        return ContradictionAnalysis(
            contradictions_detected=contradictions_detected,
            total_contradictions=total_contradictions,
            detection_rate=detection_rate,
            response_types=response_types,
            correction_quality=correction_quality
        )
    
    def _addresses_false_information(self, response: str, false_fact: str) -> bool:
        """Check if response addresses the false information."""
        # Check for contradiction indicators
        has_contradiction_indicator = any(
            indicator in response for indicator in self.contradiction_indicators
        )
        
        # Check if response mentions the false fact
        false_keywords = false_fact.split()
        mentions_false_fact = any(
            keyword in response for keyword in false_keywords if len(keyword) > 3
        )
        
        return has_contradiction_indicator and mentions_false_fact
    
    def _classify_contradiction_response(self, response: str) -> str:
        """Classify the type of response to contradiction."""
        response_lower = response.lower()
        
        # Check for correction indicators
        if any(indicator in response_lower for indicator in self.correction_indicators):
            return "correction"
        
        # Check for acknowledgment indicators
        if any(indicator in response_lower for indicator in self.acknowledgment_indicators):
            return "acknowledgment"
        
        # Check for uncertainty indicators
        uncertainty_indicators = ["maybe", "perhaps", "possibly", "might", "could", "not sure"]
        if any(indicator in response_lower for indicator in uncertainty_indicators):
            return "uncertain"
        
        # If none of the above, consider it ignored
        return "ignored"
    
    def _evaluate_correction_quality(self, response: str, false_fact: str) -> float:
        """
        Evaluate the quality of a correction made by TMM.
        
        Args:
            response: TMM response containing correction
            false_fact: The false fact being corrected
            
        Returns:
            Quality score (0-1, higher is better)
        """
        quality_score = 0.0
        
        # Check for explicit correction language
        explicit_corrections = [
            "the correct information is", "actually", "instead",
            "the right answer is", "it should be", "it's really"
        ]
        
        if any(correction in response for correction in explicit_corrections):
            quality_score += 0.4
        
        # Check for providing correct information
        if self._provides_correct_information(response, false_fact):
            quality_score += 0.4
        
        # Check for confidence in correction
        confidence_indicators = ["definitely", "certainly", "absolutely", "sure"]
        if any(indicator in response for indicator in confidence_indicators):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _provides_correct_information(self, response: str, false_fact: str) -> bool:
        """Check if response provides correct information to replace false fact."""
        # Simple mapping of false facts to correct facts
        fact_corrections = {
            "scotland": "england",
            "2:15": "3:30",
            "$200": "$120",
            "8 pm": "10 pm",
            "$5": "free"
        }
        
        # Check if response contains the correct information
        for false_key, correct_key in fact_corrections.items():
            if false_key in false_fact.lower() and correct_key in response.lower():
                return True
        
        return False
    
    def detect_implicit_contradictions(self, tmm_responses: List[str]) -> List[Dict[str, Any]]:
        """
        Detect implicit contradictions in TMM responses.
        
        Args:
            tmm_responses: List of TMM responses
            
        Returns:
            List of detected implicit contradictions
        """
        contradictions = []
        
        for i, response in enumerate(tmm_responses):
            response_lower = response.lower()
            
            # Check for conflicting statements within the same response
            conflicting_pairs = [
                ("yes", "no"),
                ("open", "closed"),
                ("available", "unavailable"),
                ("free", "cost"),
                ("morning", "evening"),
                ("cheap", "expensive")
            ]
            
            for positive, negative in conflicting_pairs:
                if positive in response_lower and negative in response_lower:
                    contradictions.append({
                        "type": "implicit_contradiction",
                        "turn": i,
                        "conflict": f"{positive} vs {negative}",
                        "response": response
                    })
        
        return contradictions
    
    def analyze_consistency_across_turns(self, tmm_responses: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency of information across multiple turns.
        
        Args:
            tmm_responses: List of TMM responses
            
        Returns:
            Dictionary with consistency analysis
        """
        consistency_issues = []
        
        # Track information mentioned across turns
        mentioned_info = {}
        
        for i, response in enumerate(tmm_responses):
            response_lower = response.lower()
            
            # Extract key information (times, prices, locations, etc.)
            info_patterns = {
                "time": r'\b\d{1,2}:\d{2}\s*(am|pm)?\b',
                "price": r'\$\d+',
                "location": r'\b(?:in|at|from|to)\s+[A-Z][a-z]+',
                "availability": r'\b(?:open|closed|available|unavailable)\b'
            }
            
            for info_type, pattern in info_patterns.items():
                matches = re.findall(pattern, response_lower)
                for match in matches:
                    key = f"{info_type}:{match}"
                    
                    if key in mentioned_info:
                        # Check for consistency
                        if mentioned_info[key] != i:
                            consistency_issues.append({
                                "type": "inconsistency",
                                "info_type": info_type,
                                "value": match,
                                "first_mentioned": mentioned_info[key],
                                "current_turn": i
                            })
                    else:
                        mentioned_info[key] = i
        
        return {
            "consistency_issues": consistency_issues,
            "total_issues": len(consistency_issues),
            "consistency_rate": max(0, 1 - len(consistency_issues) / len(tmm_responses)) if tmm_responses else 1.0
        }
