"""
Enhanced False Memory Injection System

A rigorous system for simulating LLM hallucinations and inconsistencies in user input.
This system provides comprehensive false memory injection patterns to thoroughly test
false memory prevention capabilities.
"""

import random
import re
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class InjectionType(Enum):
    """Types of false memory injection."""
    DIRECT_FALSE_FACT = "direct_false_fact"
    IMPLICIT_HALLUCINATION = "implicit_hallucination"
    CONTRADICTORY_INFORMATION = "contradictory_information"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    CONTEXTUAL_DISTORTION = "contextual_distortion"
    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    NUMERICAL_MANIPULATION = "numerical_manipulation"
    CAUSAL_DISTORTION = "causal_distortion"

@dataclass
class FalseMemoryInjection:
    """Represents a false memory injection."""
    injection_type: InjectionType
    original_text: str
    modified_text: str
    false_fact: str
    injection_point: int
    confidence_level: float  # How confident the false information appears
    temporal_context: Optional[str] = None
    metadata: Dict[str, Any] = None

class EnhancedFalseMemoryInjector:
    """
    Enhanced false memory injection system for rigorous testing.
    
    This system simulates various types of LLM hallucinations and inconsistencies
    to thoroughly test false memory prevention capabilities.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the enhanced false memory injector."""
        self.seed = seed
        random.seed(seed)
        
        # Comprehensive false facts database
        self.false_facts_database = {
            "geographical": {
                "Cambridge is in Scotland": "Cambridge is in England",
                "Paris is the capital of Germany": "Paris is the capital of France",
                "Tokyo is in China": "Tokyo is in Japan",
                "Sydney is in New Zealand": "Sydney is in Australia"
            },
            "temporal": {
                "The train leaves at 2:15 PM": "The train leaves at 3:30 PM",
                "The restaurant opens at 11 AM": "The restaurant opens at 9 AM",
                "The museum closes at 6 PM": "The museum closes at 8 PM",
                "The flight departs at 7:30 AM": "The flight departs at 9:15 AM"
            },
            "numerical": {
                "The hotel costs $200 per night": "The hotel costs $120 per night",
                "The restaurant seats 20 people": "The restaurant seats 50 people",
                "The flight takes 6 hours": "The flight takes 3 hours",
                "The airport is 5 miles away": "The airport is 15 miles away"
            },
            "categorical": {
                "The hotel has 2 stars": "The hotel has 4 stars",
                "We charge $5 for WiFi": "We offer free WiFi",
                "The restaurant serves only vegetarian food": "The restaurant serves both vegetarian and non-vegetarian food",
                "The gym is closed on weekends": "The gym is open on weekends"
            },
            "causal": {
                "The delay was caused by bad weather": "The delay was caused by mechanical issues",
                "The restaurant is popular because of its location": "The restaurant is popular because of its food quality",
                "The train is late due to traffic": "The train is late due to signal problems"
            }
        }
        
        # Injection patterns for different types of hallucinations
        self.injection_patterns = {
            InjectionType.DIRECT_FALSE_FACT: [
                "By the way, {false_fact}",
                "I should mention that {false_fact}",
                "Just so you know, {false_fact}",
                "Actually, {false_fact}",
                "I heard that {false_fact}"
            ],
            InjectionType.IMPLICIT_HALLUCINATION: [
                "I believe {false_fact}",
                "I think {false_fact}",
                "I'm pretty sure {false_fact}",
                "I know that {false_fact}",
                "Someone told me {false_fact}"
            ],
            InjectionType.CONTRADICTORY_INFORMATION: [
                "Wait, that's not right. {false_fact}",
                "Actually, I need to correct that. {false_fact}",
                "I made an error. {false_fact}",
                "Let me clarify: {false_fact}",
                "I was wrong about that. {false_fact}"
            ],
            InjectionType.TEMPORAL_INCONSISTENCY: [
                "Earlier you mentioned {false_fact}, but now I think {contradiction}",
                "I remember {false_fact} from before, but {contradiction}",
                "Last time we discussed {false_fact}, but {contradiction}"
            ],
            InjectionType.CONTEXTUAL_DISTORTION: [
                "In this context, {false_fact}",
                "Given the situation, {false_fact}",
                "Considering everything, {false_fact}",
                "Based on what I understand, {false_fact}"
            ]
        }
        
        # Confidence indicators for realistic injection
        self.confidence_indicators = {
            "high": ["definitely", "certainly", "absolutely", "sure", "positive"],
            "medium": ["probably", "likely", "think", "believe", "seem"],
            "low": ["maybe", "perhaps", "possibly", "might", "could"]
        }
        
        # Temporal context patterns
        self.temporal_patterns = [
            "yesterday", "last week", "earlier today", "a few hours ago",
            "recently", "the other day", "just now", "a moment ago"
        ]
    
    def inject_false_memories(self, 
                            conversation: Dict[str, Any], 
                            num_injections: int = 3,
                            injection_types: List[InjectionType] = None) -> Tuple[Dict[str, Any], List[FalseMemoryInjection]]:
        """
        Inject false memories into a conversation with enhanced patterns.
        
        Args:
            conversation: Original conversation dict
            num_injections: Number of false memories to inject
            injection_types: Specific types of injections to use
            
        Returns:
            Tuple of (modified_conversation, injection_records)
        """
        if injection_types is None:
            injection_types = list(InjectionType)
        
        modified_conversation = conversation.copy()
        user_turns = conversation.get("user_turns", [])
        system_turns = conversation.get("system_turns", [])
        
        if not user_turns:
            logger.warning("No user turns found in conversation")
            return modified_conversation, []
        
        injections = []
        
        # Select injection points strategically
        injection_points = self._select_strategic_injection_points(len(user_turns), num_injections)
        
        # Inject false memories
        for i, turn_idx in enumerate(injection_points):
            if turn_idx >= len(user_turns):
                continue
            
            # Select injection type
            injection_type = random.choice(injection_types)
            
            # Create injection
            injection = self._create_injection(
                user_turns[turn_idx], 
                injection_type, 
                turn_idx,
                conversation_context=conversation
            )
            
            if injection:
                # Apply injection
                modified_conversation["user_turns"][turn_idx] = injection.modified_text
                injections.append(injection)
                
                logger.debug(f"Injected {injection_type.value} at turn {turn_idx}")
        
        logger.info(f"Injected {len(injections)} false memories into conversation")
        return modified_conversation, injections
    
    def _select_strategic_injection_points(self, total_turns: int, num_injections: int) -> List[int]:
        """Select strategic points for false memory injection."""
        if total_turns <= 1:
            return [0] if total_turns == 1 else []
        
        num_injections = min(num_injections, total_turns)
        points = []
        
        # Early injection (first 30%) - establishes false baseline
        early_turns = max(1, int(total_turns * 0.3))
        if num_injections > 0:
            early_point = random.randint(0, early_turns - 1)
            points.append(early_point)
            num_injections -= 1
        
        # Middle injection (middle 40%) - tests persistence
        if num_injections > 0:
            middle_start = early_turns
            middle_end = int(total_turns * 0.7)
            if middle_end > middle_start:
                middle_point = random.randint(middle_start, middle_end - 1)
                points.append(middle_point)
                num_injections -= 1
        
        # Late injection (last 30%) - tests correction ability
        if num_injections > 0:
            late_start = int(total_turns * 0.7)
            late_point = random.randint(late_start, total_turns - 1)
            points.append(late_point)
        
        return sorted(points)
    
    def _create_injection(self, 
                         original_text: str, 
                         injection_type: InjectionType, 
                         turn_idx: int,
                         conversation_context: Dict[str, Any] = None) -> Optional[FalseMemoryInjection]:
        """Create a specific type of false memory injection."""
        
        if injection_type == InjectionType.DIRECT_FALSE_FACT:
            return self._create_direct_false_fact_injection(original_text, turn_idx)
        elif injection_type == InjectionType.IMPLICIT_HALLUCINATION:
            return self._create_implicit_hallucination_injection(original_text, turn_idx)
        elif injection_type == InjectionType.CONTRADICTORY_INFORMATION:
            return self._create_contradictory_information_injection(original_text, turn_idx, conversation_context)
        elif injection_type == InjectionType.TEMPORAL_INCONSISTENCY:
            return self._create_temporal_inconsistency_injection(original_text, turn_idx, conversation_context)
        elif injection_type == InjectionType.CONTEXTUAL_DISTORTION:
            return self._create_contextual_distortion_injection(original_text, turn_idx)
        elif injection_type == InjectionType.SEMANTIC_PARAPHRASE:
            return self._create_semantic_paraphrase_injection(original_text, turn_idx)
        elif injection_type == InjectionType.NUMERICAL_MANIPULATION:
            return self._create_numerical_manipulation_injection(original_text, turn_idx)
        elif injection_type == InjectionType.CAUSAL_DISTORTION:
            return self._create_causal_distortion_injection(original_text, turn_idx)
        
        return None
    
    def _create_direct_false_fact_injection(self, original_text: str, turn_idx: int) -> FalseMemoryInjection:
        """Create a direct false fact injection."""
        category = random.choice(list(self.false_facts_database.keys()))
        false_fact = random.choice(list(self.false_facts_database[category].keys()))
        
        pattern = random.choice(self.injection_patterns[InjectionType.DIRECT_FALSE_FACT])
        injection_text = pattern.format(false_fact=false_fact.lower())
        
        # Add confidence indicator
        confidence_level = random.choice(["high", "medium", "low"])
        confidence_word = random.choice(self.confidence_indicators[confidence_level])
        injection_text = f"{confidence_word}, {injection_text}"
        
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.DIRECT_FALSE_FACT,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.8 if confidence_level == "high" else 0.5 if confidence_level == "medium" else 0.2
        )
    
    def _create_implicit_hallucination_injection(self, original_text: str, turn_idx: int) -> FalseMemoryInjection:
        """Create an implicit hallucination injection."""
        category = random.choice(list(self.false_facts_database.keys()))
        false_fact = random.choice(list(self.false_facts_database[category].keys()))
        
        pattern = random.choice(self.injection_patterns[InjectionType.IMPLICIT_HALLUCINATION])
        injection_text = pattern.format(false_fact=false_fact.lower())
        
        # Add temporal context
        temporal_context = random.choice(self.temporal_patterns)
        injection_text = f"{temporal_context}, {injection_text}"
        
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.IMPLICIT_HALLUCINATION,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.6,
            temporal_context=temporal_context
        )
    
    def _create_contradictory_information_injection(self, 
                                                  original_text: str, 
                                                  turn_idx: int,
                                                  conversation_context: Dict[str, Any] = None) -> FalseMemoryInjection:
        """Create a contradictory information injection."""
        category = random.choice(list(self.false_facts_database.keys()))
        false_fact = random.choice(list(self.false_facts_database[category].keys()))
        true_fact = self.false_facts_database[category][false_fact]
        
        pattern = random.choice(self.injection_patterns[InjectionType.CONTRADICTORY_INFORMATION])
        injection_text = pattern.format(false_fact=false_fact.lower())
        
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.CONTRADICTORY_INFORMATION,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.9,
            metadata={"true_fact": true_fact}
        )
    
    def _create_temporal_inconsistency_injection(self, 
                                               original_text: str, 
                                               turn_idx: int,
                                               conversation_context: Dict[str, Any] = None) -> FalseMemoryInjection:
        """Create a temporal inconsistency injection."""
        category = random.choice(list(self.false_facts_database.keys()))
        false_fact = random.choice(list(self.false_facts_database[category].keys()))
        true_fact = self.false_facts_database[category][false_fact]
        
        pattern = random.choice(self.injection_patterns[InjectionType.TEMPORAL_INCONSISTENCY])
        injection_text = pattern.format(
            false_fact=false_fact.lower(),
            contradiction=true_fact.lower()
        )
        
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.TEMPORAL_INCONSISTENCY,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.7,
            metadata={"true_fact": true_fact}
        )
    
    def _create_contextual_distortion_injection(self, original_text: str, turn_idx: int) -> FalseMemoryInjection:
        """Create a contextual distortion injection."""
        category = random.choice(list(self.false_facts_database.keys()))
        false_fact = random.choice(list(self.false_facts_database[category].keys()))
        
        pattern = random.choice(self.injection_patterns[InjectionType.CONTEXTUAL_DISTORTION])
        injection_text = pattern.format(false_fact=false_fact.lower())
        
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.CONTEXTUAL_DISTORTION,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.6
        )
    
    def _create_semantic_paraphrase_injection(self, original_text: str, turn_idx: int) -> FalseMemoryInjection:
        """Create a semantic paraphrase injection."""
        category = random.choice(list(self.false_facts_database.keys()))
        false_fact = random.choice(list(self.false_facts_database[category].keys()))
        
        # Create paraphrased version
        paraphrased_fact = self._paraphrase_false_fact(false_fact)
        
        injection_text = f"Also, {paraphrased_fact.lower()}."
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.SEMANTIC_PARAPHRASE,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.7,
            metadata={"paraphrased_fact": paraphrased_fact}
        )
    
    def _create_numerical_manipulation_injection(self, original_text: str, turn_idx: int) -> FalseMemoryInjection:
        """Create a numerical manipulation injection."""
        # Find numbers in the original text and manipulate them
        numbers = re.findall(r'\d+', original_text)
        if not numbers:
            # Fallback to standard false fact
            return self._create_direct_false_fact_injection(original_text, turn_idx)
        
        # Select a number to manipulate
        target_number = random.choice(numbers)
        manipulation_factor = random.choice([0.5, 1.5, 2.0, 0.75])
        new_number = str(int(int(target_number) * manipulation_factor))
        
        # Create false statement with manipulated number
        false_fact = f"The value is {new_number} instead of {target_number}"
        injection_text = f"Actually, {false_fact.lower()}."
        
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.NUMERICAL_MANIPULATION,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.8,
            metadata={"original_number": target_number, "manipulated_number": new_number}
        )
    
    def _create_causal_distortion_injection(self, original_text: str, turn_idx: int) -> FalseMemoryInjection:
        """Create a causal distortion injection."""
        category = "causal"
        if category not in self.false_facts_database:
            return self._create_direct_false_fact_injection(original_text, turn_idx)
        
        false_fact = random.choice(list(self.false_facts_database[category].keys()))
        true_fact = self.false_facts_database[category][false_fact]
        
        injection_text = f"I think {false_fact.lower()}."
        modified_text = f"{original_text} {injection_text}"
        
        return FalseMemoryInjection(
            injection_type=InjectionType.CAUSAL_DISTORTION,
            original_text=original_text,
            modified_text=modified_text,
            false_fact=false_fact,
            injection_point=turn_idx,
            confidence_level=0.6,
            metadata={"true_fact": true_fact}
        )
    
    def _paraphrase_false_fact(self, false_fact: str) -> str:
        """Create a paraphrased version of a false fact."""
        paraphrases = {
            "Cambridge is in Scotland": "Cambridge is located in Scotland",
            "The train leaves at 2:15 PM": "The train departs at 2:15 PM",
            "The hotel costs $200 per night": "The hotel charges $200 per night",
            "The restaurant closes at 8 PM": "The restaurant shuts at 8 PM"
        }
        
        return paraphrases.get(false_fact, false_fact)
    
    def get_injection_statistics(self, injections: List[FalseMemoryInjection]) -> Dict[str, Any]:
        """Get comprehensive statistics about false memory injections."""
        if not injections:
            return {}
        
        stats = {
            "total_injections": len(injections),
            "injection_types": {},
            "confidence_levels": [],
            "categories": {},
            "temporal_contexts": []
        }
        
        for injection in injections:
            # Count injection types
            injection_type = injection.injection_type.value
            stats["injection_types"][injection_type] = stats["injection_types"].get(injection_type, 0) + 1
            
            # Track confidence levels
            stats["confidence_levels"].append(injection.confidence_level)
            
            # Track temporal contexts
            if injection.temporal_context:
                stats["temporal_contexts"].append(injection.temporal_context)
        
        # Calculate averages
        if stats["confidence_levels"]:
            stats["avg_confidence"] = sum(stats["confidence_levels"]) / len(stats["confidence_levels"])
        
        return stats
