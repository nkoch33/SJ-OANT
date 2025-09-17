"""
False Memory Formation Metrics Calculator

Implements the core research metrics for evaluating TMM's ability to prevent
false memory formation: FMR, MEL, and DAR.
"""

import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class FalseMemoryMetrics:
    """Container for false memory formation metrics."""
    fmr: float  # False Memory Rate (0-100%)
    mel: float  # Memory Edit Latency (seconds)
    dar: float  # Disturbance Adaptation Rate (0-100%)
    contradiction_detection_rate: float  # Contradiction detection accuracy (0-100%)
    false_information_persistence: float  # How long false info persists (turns)
    correction_accuracy: float  # Accuracy of corrections (0-100%)

class MetricsCalculator:
    """
    Calculates false memory formation metrics for TMM evaluation.
    
    This class implements the core research metrics from the paper:
    - FMR (False Memory Rate): How often the system repeats seeded falsehoods
    - MEL (Memory Edit Latency): How quickly it corrects after contradictions
    - DAR (Disturbance Adaptation Rate): How well it handles mixed true/false context
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.false_facts = {
            "Cambridge is in Scotland",
            "The train leaves at 2:15 PM", 
            "The hotel costs $200 per night",
            "The restaurant closes at 8 PM",
            "We charge $5 for WiFi",
            "The restaurant seats 20 people",
            "The hotel has 2 stars",
            "The airport is 5 miles away",
            "The flight takes 6 hours",
            "It's raining today"
        }
        
        self.true_facts = {
            "Cambridge is in England",
            "The train leaves at 3:30 PM",
            "The hotel costs $120 per night", 
            "The restaurant is open until 10 PM",
            "We offer free WiFi",
            "The restaurant seats 50 people",
            "The hotel has 4 stars",
            "The airport is 15 miles away",
            "The flight takes 3 hours",
            "It's sunny today"
        }
        
        # Track timing for MEL calculation
        self.correction_times = []
        self.contradiction_detection_times = []
    
    def calculate_fmr(self, tmm_responses: List[str], 
                     false_information: Dict[str, str]) -> float:
        """
        Calculate False Memory Rate (FMR).
        
        FMR measures how often TMM repeats seeded false information.
        Lower FMR is better (0% = perfect, 100% = always repeats false info).
        
        Args:
            tmm_responses: List of TMM responses
            false_information: Dict of seeded false information
            
        Returns:
            FMR percentage (0-100%)
        """
        responses_with_false_info = 0
        total_responses = len(tmm_responses)
        
        if total_responses == 0:
            return 0.0
        
        # Check each response for repetition of false information
        for response in tmm_responses:
            response_lower = response.lower()
            response_contains_false = False
            
            for turn_id, false_fact in false_information.items():
                false_fact_lower = false_fact.lower()
                
                # Check if TMM repeats the false information
                if self._contains_false_information(response_lower, false_fact_lower):
                    response_contains_false = True
                    logger.debug(f"False repetition detected: '{false_fact}' in response")
            
            if response_contains_false:
                responses_with_false_info += 1
        
        # FMR = (responses containing false info / total responses) * 100
        fmr = (responses_with_false_info / total_responses) * 100
        logger.info(f"FMR calculated: {fmr:.2f}% ({responses_with_false_info}/{total_responses})")
        return fmr
    
    def calculate_mel(self, tmm_responses: List[str], 
                     contradiction_point: int,
                     false_information: Dict[str, str]) -> float:
        """
        Calculate Memory Edit Latency (MEL).
        
        MEL measures how quickly TMM corrects false information after contradictions.
        Lower MEL is better (0 seconds = immediate correction).
        
        Args:
            tmm_responses: List of TMM responses
            contradiction_point: Turn where contradiction was introduced
            false_information: Dict of seeded false information
            
        Returns:
            MEL in seconds (average time to correct)
        """
        if contradiction_point < 0 or contradiction_point >= len(tmm_responses):
            return 0.0
        
        correction_times = []
        
        # Check responses after contradiction point
        for i in range(contradiction_point + 1, len(tmm_responses)):
            response = tmm_responses[i]
            response_lower = response.lower()
            
            # Check if this response corrects any false information
            for turn_id, false_fact in false_information.items():
                false_fact_lower = false_fact.lower()
                
                if self._contains_correction(response_lower, false_fact_lower):
                    # Calculate time from contradiction to correction
                    turns_after_contradiction = i - contradiction_point
                    # Assume 1 turn = 1 second for simplicity (can be adjusted)
                    correction_time = turns_after_contradiction
                    correction_times.append(correction_time)
                    logger.debug(f"Correction detected {turns_after_contradiction} turns after contradiction")
        
        if correction_times:
            mel = sum(correction_times) / len(correction_times)
            logger.info(f"MEL calculated: {mel:.2f} seconds (average)")
        else:
            mel = float('inf')  # No corrections detected
            logger.warning("MEL: No corrections detected - infinite latency")
        
        return mel
    
    def calculate_dar(self, tmm_responses: List[str],
                     false_information: Dict[str, str],
                     true_information: List[str] = None) -> float:
        """
        Calculate Disturbance Adaptation Rate (DAR).
        
        DAR measures how well TMM handles mixed true/false context and
        maintains accuracy despite false information.
        
        Args:
            tmm_responses: List of TMM responses
            false_information: Dict of seeded false information
            true_information: List of true facts (optional)
            
        Returns:
            DAR percentage (0-100%, higher is better)
        """
        if true_information is None:
            true_information = list(self.true_facts)
        
        correct_responses = 0
        total_responses = len(tmm_responses)
        
        if total_responses == 0:
            return 0.0
        
        for response in tmm_responses:
            response_lower = response.lower()
            
            # Check if response contains correct information
            contains_correct = any(
                self._contains_true_information(response_lower, true_fact.lower())
                for true_fact in true_information
            )
            
            # Check if response avoids false information
            avoids_false = not any(
                self._contains_false_information(response_lower, false_fact.lower())
                for false_fact in false_information.values()
            )
            
            # Response is correct if it contains true info and avoids false info
            if contains_correct and avoids_false:
                correct_responses += 1
        
        dar = (correct_responses / total_responses) * 100
        logger.info(f"DAR calculated: {dar:.2f}% ({correct_responses}/{total_responses})")
        return dar
    
    def calculate_contradiction_detection_rate(self, tmm_responses: List[str],
                                             contradiction_point: int) -> float:
        """
        Calculate contradiction detection rate.
        
        Measures how well TMM identifies and responds to contradictions.
        
        Args:
            tmm_responses: List of TMM responses
            contradiction_point: Turn where contradiction was introduced
            
        Returns:
            Detection rate percentage (0-100%)
        """
        if contradiction_point < 0 or contradiction_point >= len(tmm_responses):
            return 0.0
        
        # Check responses after contradiction point for detection indicators
        detection_indicators = [
            "contradict", "error", "mistake", "incorrect", "wrong",
            "clarify", "correct", "update", "change", "revise"
        ]
        
        detections = 0
        total_checks = len(tmm_responses) - contradiction_point - 1
        
        if total_checks <= 0:
            return 0.0
        
        for i in range(contradiction_point + 1, len(tmm_responses)):
            response_lower = tmm_responses[i].lower()
            
            # Check for detection indicators
            if any(indicator in response_lower for indicator in detection_indicators):
                detections += 1
                logger.debug(f"Contradiction detection in response {i}")
        
        detection_rate = (detections / total_checks) * 100
        logger.info(f"Contradiction detection rate: {detection_rate:.2f}%")
        return detection_rate
    
    def calculate_false_information_persistence(self, tmm_responses: List[str],
                                               false_information: Dict[str, str]) -> float:
        """
        Calculate how long false information persists in TMM responses.
        
        Args:
            tmm_responses: List of TMM responses
            false_information: Dict of seeded false information
            
        Returns:
            Average persistence in turns
        """
        persistence_turns = []
        
        for turn_id, false_fact in false_information.items():
            false_fact_lower = false_fact.lower()
            injection_turn = int(turn_id)
            
            # Count how many turns after injection the false info persists
            persistence = 0
            for i in range(injection_turn + 1, len(tmm_responses)):
                response_lower = tmm_responses[i].lower()
                
                if self._contains_false_information(response_lower, false_fact_lower):
                    persistence += 1
                else:
                    break  # False info no longer present
            
            persistence_turns.append(persistence)
        
        if persistence_turns:
            avg_persistence = sum(persistence_turns) / len(persistence_turns)
            logger.info(f"Average false information persistence: {avg_persistence:.2f} turns")
        else:
            avg_persistence = 0.0
        
        return avg_persistence
    
    def calculate_correction_accuracy(self, tmm_responses: List[str],
                                    false_information: Dict[str, str]) -> float:
        """
        Calculate accuracy of corrections made by TMM.
        
        Args:
            tmm_responses: List of TMM responses
            false_information: Dict of seeded false information
            
        Returns:
            Correction accuracy percentage (0-100%)
        """
        corrections = 0
        accurate_corrections = 0
        
        for response in tmm_responses:
            response_lower = response.lower()
            
            # Check for correction attempts
            correction_indicators = [
                "correct", "actually", "clarify", "update", "change",
                "revise", "mistake", "error", "wrong"
            ]
            
            if any(indicator in response_lower for indicator in correction_indicators):
                corrections += 1
                
                # Check if correction is accurate (contains true information)
                for false_fact in false_information.values():
                    false_fact_lower = false_fact.lower()
                    
                    # Find corresponding true fact
                    true_fact = self._get_corresponding_true_fact(false_fact_lower)
                    if true_fact and self._contains_true_information(response_lower, true_fact.lower()):
                        accurate_corrections += 1
                        break
        
        if corrections > 0:
            accuracy = (accurate_corrections / corrections) * 100
            logger.info(f"Correction accuracy: {accuracy:.2f}% ({accurate_corrections}/{corrections})")
        else:
            accuracy = 0.0
            logger.info("No corrections detected")
        
        return accuracy
    
    def _contains_false_information(self, response: str, false_fact: str) -> bool:
        """Check if response contains false information with precise detection."""
        # More precise false information detection - only direct matches
        false_fact_lower = false_fact.lower()
        response_lower = response.lower()
        
        # Check for direct repetition of the false fact
        if false_fact_lower in response_lower:
            return True
        
        # Check for key false facts with specific patterns
        false_patterns = {
            "cambridge is in scotland": ["cambridge", "scotland"],
            "the train leaves at 2:15 pm": ["2:15", "pm"],
            "the hotel costs $200 per night": ["$200", "per night"],
            "the restaurant closes at 8 pm": ["8 pm", "closes"],
            "we charge $5 for wifi": ["$5", "wifi"],
            "the restaurant seats 20 people": ["20 people", "seats"],
            "the hotel has 2 stars": ["2 stars"],
            "the airport is 5 miles away": ["5 miles"],
            "the flight takes 6 hours": ["6 hours"],
            "it's raining today": ["raining", "today"]
        }
        
        # Check for specific false fact patterns
        for fact, keywords in false_patterns.items():
            if fact in false_fact_lower:
                # Only flag if ALL key keywords are present
                if all(keyword in response_lower for keyword in keywords):
                    return True
        
        return False
    
    def _has_semantic_similarity(self, false_fact: str, response: str) -> bool:
        """Check for semantic similarity between false fact and response."""
        # Extract key concepts from false fact
        false_concepts = self._extract_key_concepts(false_fact)
        response_concepts = self._extract_key_concepts(response)
        
        # Check for concept overlap
        overlap = len(false_concepts.intersection(response_concepts))
        return overlap >= 2  # At least 2 key concepts overlap
    
    def _extract_key_concepts(self, text: str) -> set:
        """Extract key concepts from text."""
        # Remove common words and extract meaningful concepts
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'throughout', 'within', 'without', 'and', 'or', 'but', 'nor', 'so', 'yet', 'although', 'though', 'because', 'since', 'while'}
        
        words = text.lower().split()
        concepts = {word for word in words if len(word) > 3 and word not in stop_words}
        return concepts
    
    def _is_paraphrased_falsehood(self, false_fact: str, response: str) -> bool:
        """Check if response contains a paraphrased version of the false fact."""
        # Check for common paraphrasing patterns
        import re
        paraphrasing_patterns = [
            (r'\b(?:takes|requires|needs)\s+(\d+)\s+(?:hours?|hrs?)\b', r'\b(\d+)\s+(?:hours?|hrs?)\b'),
            (r'\b(?:costs?|prices?)\s+(\$\d+)\b', r'\b(\$\d+)\b'),
            (r'\b(?:located|situated|found)\s+in\s+(\w+)\b', r'\bin\s+(\w+)\b'),
            (r'\b(?:opens?|closes?)\s+at\s+(\d+:\d+)\b', r'\bat\s+(\d+:\d+)\b')
        ]
        
        for pattern1, pattern2 in paraphrasing_patterns:
            if re.search(pattern1, false_fact.lower()) and re.search(pattern2, response.lower()):
                return True
        
        return False
    
    def _contains_true_information(self, response: str, true_fact: str) -> bool:
        """Check if response contains true information."""
        true_keywords = true_fact.split()
        return any(keyword in response for keyword in true_keywords if len(keyword) > 3)
    
    def _contains_correction(self, response: str, false_fact: str) -> bool:
        """Check if response contains a correction of false information."""
        correction_indicators = [
            "correct", "actually", "clarify", "update", "change",
            "revise", "mistake", "error", "wrong", "not"
        ]
        
        has_correction_indicator = any(indicator in response for indicator in correction_indicators)
        mentions_false_fact = self._contains_false_information(response, false_fact)
        
        return has_correction_indicator and mentions_false_fact
    
    def _get_corresponding_true_fact(self, false_fact: str) -> str:
        """Get the corresponding true fact for a false fact."""
        # Simple mapping - can be enhanced
        fact_mappings = {
            "scotland": "england",
            "2:15": "3:30",
            "$200": "$120",
            "8 pm": "10 pm",
            "$5": "free"
        }
        
        for false_key, true_key in fact_mappings.items():
            if false_key in false_fact:
                return false_fact.replace(false_key, true_key)
        
        return None
    
    def calculate_all_metrics(self, tmm_responses: List[str],
                            false_information: Dict[str, str],
                            contradiction_point: int = -1) -> FalseMemoryMetrics:
        """
        Calculate all false memory formation metrics.
        
        Args:
            tmm_responses: List of TMM responses
            false_information: Dict of seeded false information
            contradiction_point: Turn where contradiction was introduced
            
        Returns:
            FalseMemoryMetrics object with all calculated metrics
        """
        logger.info("Calculating all false memory formation metrics...")
        
        fmr = self.calculate_fmr(tmm_responses, false_information)
        mel = self.calculate_mel(tmm_responses, contradiction_point, false_information)
        dar = self.calculate_dar(tmm_responses, false_information)
        contradiction_detection = self.calculate_contradiction_detection_rate(
            tmm_responses, contradiction_point)
        persistence = self.calculate_false_information_persistence(
            tmm_responses, false_information)
        correction_accuracy = self.calculate_correction_accuracy(
            tmm_responses, false_information)
        
        return FalseMemoryMetrics(
            fmr=fmr,
            mel=mel,
            dar=dar,
            contradiction_detection_rate=contradiction_detection,
            false_information_persistence=persistence,
            correction_accuracy=correction_accuracy
        )
