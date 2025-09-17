"""
False Memory Test Scenarios

Creates test scenarios with seeded falsehoods and contradictions to evaluate
TMM's ability to prevent false memory formation.
"""

import json
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FalseMemoryScenario:
    """A test scenario with seeded false information."""
    original_conversation: List[Dict]
    false_information: Dict[str, str]  # key: turn_id, value: false_statement
    contradiction_point: int  # Turn where contradiction is introduced
    expected_behavior: str  # What TMM should do
    scenario_type: str  # "injection", "contradiction", "persistence"

class FalseMemoryTestScenarios:
    """
    Generates test scenarios for false memory formation evaluation.
    
    This class creates modified versions of existing benchmark conversations
    with seeded false information to test TMM's truth-maintenance capabilities.
    """
    
    def __init__(self):
        """Initialize the false memory test scenario generator."""
        self.false_facts = {
            "location": {
                "true": "Cambridge is in England",
                "false": "Cambridge is in Scotland"
            },
            "time": {
                "true": "The train leaves at 3:30 PM",
                "false": "The train leaves at 2:15 PM"
            },
            "price": {
                "true": "The hotel costs $120 per night",
                "false": "The hotel costs $200 per night"
            },
            "availability": {
                "true": "The restaurant is open until 10 PM",
                "false": "The restaurant closes at 8 PM"
            },
            "service": {
                "true": "We offer free WiFi",
                "false": "We charge $5 for WiFi"
            },
            "capacity": {
                "true": "The restaurant seats 50 people",
                "false": "The restaurant seats 20 people"
            },
            "rating": {
                "true": "The hotel has 4 stars",
                "false": "The hotel has 2 stars"
            },
            "distance": {
                "true": "The airport is 15 miles away",
                "false": "The airport is 5 miles away"
            },
            "duration": {
                "true": "The flight takes 3 hours",
                "false": "The flight takes 6 hours"
            },
            "weather": {
                "true": "It's sunny today",
                "false": "It's raining today"
            }
        }
        
        self.contradiction_templates = [
            "Actually, I need to correct that information: {correction}",
            "I apologize, but I made an error earlier: {correction}",
            "Let me clarify - the correct information is: {correction}",
            "I need to update that - it's actually: {correction}"
        ]
    
    def create_false_injection_scenario(self, conversation: Dict, 
                                      injection_turn: int = 2) -> FalseMemoryScenario:
        """
        Create a scenario where false information is injected into a conversation.
        
        Args:
            conversation: Original benchmark conversation dict
            injection_turn: Turn number to inject false information
            
        Returns:
            FalseMemoryScenario with seeded false information
        """
        # Select a random false fact to inject
        fact_type = random.choice(list(self.false_facts.keys()))
        false_fact = self.false_facts[fact_type]["false"]
        
        # Create modified conversation with false information
        modified_conversation = conversation.copy()
        
        # Inject false information into user turns (more realistic for false memory testing)
        user_turns = modified_conversation.get("user_turns", [])
        if user_turns and len(user_turns) > injection_turn:
            # Add false information to user input
            false_info = f" Also, {false_fact.lower()}."
            modified_conversation["user_turns"][injection_turn] += false_info
        
        return FalseMemoryScenario(
            original_conversation=conversation,
            false_information={str(injection_turn): false_fact},
            contradiction_point=-1,  # No contradiction in injection scenario
            expected_behavior="TMM should not repeat the false information in subsequent responses",
            scenario_type="injection"
        )
    
    def create_contradiction_scenario(self, conversation: Dict,
                                    contradiction_turn: int = 4) -> FalseMemoryScenario:
        """
        Create a scenario where contradictory information is introduced.
        
        Args:
            conversation: Original benchmark conversation
            contradiction_turn: Turn number to introduce contradiction
            
        Returns:
            FalseMemoryScenario with contradiction
        """
        if contradiction_turn >= len(conversation):
            contradiction_turn = len(conversation) - 1
            
        # Select a random false fact and its correction
        fact_type = random.choice(list(self.false_facts.keys()))
        false_fact = self.false_facts[fact_type]["false"]
        true_fact = self.false_facts[fact_type]["true"]
        
        # Create modified conversation
        modified_conversation = conversation.copy()
        
        # First, inject false information early in conversation
        injection_turn = max(1, contradiction_turn - 2)
        system_turns = modified_conversation.get("system_turns", [])
        
        if system_turns and len(system_turns) > injection_turn:
            false_info = f" By the way, {false_fact.lower()}."
            modified_conversation["system_turns"][injection_turn] += false_info
        
        # Then introduce contradiction
        if system_turns and len(system_turns) > contradiction_turn:
            contradiction_template = random.choice(self.contradiction_templates)
            correction = f"{true_fact.lower()}."
            contradiction = f" {contradiction_template.format(correction=correction)}"
            modified_conversation["system_turns"][contradiction_turn] += contradiction
        
        return FalseMemoryScenario(
            original_conversation=conversation,
            false_information={str(injection_turn): false_fact},
            contradiction_point=contradiction_turn,
            expected_behavior="TMM should correct the false information and not repeat it",
            scenario_type="contradiction"
        )
    
    def create_persistence_scenario(self, conversation: Dict,
                                  false_turns: List[int] = None) -> FalseMemoryScenario:
        """
        Create a scenario to test false memory persistence across multiple turns.
        
        Args:
            conversation: Original benchmark conversation
            false_turns: List of turn numbers to inject false information
            
        Returns:
            FalseMemoryScenario with multiple false information injections
        """
        if false_turns is None:
            # Inject false information in 2-3 different turns
            num_injections = min(3, len(conversation) - 1)
            false_turns = random.sample(range(1, len(conversation)), num_injections)
        
        modified_conversation = conversation.copy()
        false_information = {}
        
        # Inject different false facts in different turns
        fact_types = list(self.false_facts.keys())
        system_turns = modified_conversation.get("system_turns", [])
        
        for i, turn in enumerate(false_turns):
            if system_turns and len(system_turns) > turn:
                fact_type = fact_types[i % len(fact_types)]
                false_fact = self.false_facts[fact_type]["false"]
                false_information[str(turn)] = false_fact
                
                false_info = f" Also, {false_fact.lower()}."
                modified_conversation["system_turns"][turn] += false_info
        
        return FalseMemoryScenario(
            original_conversation=conversation,
            false_information=false_information,
            contradiction_point=-1,
            expected_behavior="TMM should not accumulate or repeat false information across turns",
            scenario_type="persistence"
        )
    
    def generate_benchmark_scenarios(self, benchmark_data: List[Dict], 
                                   num_scenarios: int = 10) -> List[FalseMemoryScenario]:
        """
        Generate false memory test scenarios from benchmark data.
        
        Args:
            benchmark_data: List of benchmark conversations
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of FalseMemoryScenario objects
        """
        scenarios = []
        
        # Sample conversations for testing
        sample_conversations = random.sample(benchmark_data, 
                                           min(num_scenarios, len(benchmark_data)))
        
        for i, conversation in enumerate(sample_conversations):
            # Convert to conversation format if needed
            if isinstance(conversation, dict) and "user_turns" in conversation:
                # Already in correct format
                conv_data = conversation
            else:
                # Convert from sample format
                conv_data = {
                    "dialogue_id": conversation.get("dialogue_id", f"conv_{i}"),
                    "user_turns": conversation.get("user_turns", []),
                    "system_turns": conversation.get("system_turns", [])
                }
            
            # Generate different types of scenarios
            scenario_type = i % 3
            
            if scenario_type == 0:
                scenario = self.create_false_injection_scenario(conv_data)
            elif scenario_type == 1:
                scenario = self.create_contradiction_scenario(conv_data)
            else:
                scenario = self.create_persistence_scenario(conv_data)
            
            scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} false memory test scenarios")
        return scenarios
    
    def save_scenarios(self, scenarios: List[FalseMemoryScenario], 
                      filepath: str) -> None:
        """Save test scenarios to a JSON file."""
        scenario_data = []
        
        for scenario in scenarios:
            scenario_data.append({
                "original_conversation": scenario.original_conversation,
                "false_information": scenario.false_information,
                "contradiction_point": scenario.contradiction_point,
                "expected_behavior": scenario.expected_behavior,
                "scenario_type": scenario.scenario_type
            })
        
        with open(filepath, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        
        logger.info(f"Saved {len(scenarios)} scenarios to {filepath}")
    
    def load_scenarios(self, filepath: str) -> List[FalseMemoryScenario]:
        """Load test scenarios from a JSON file."""
        with open(filepath, 'r') as f:
            scenario_data = json.load(f)
        
        scenarios = []
        for data in scenario_data:
            scenario = FalseMemoryScenario(
                original_conversation=data["original_conversation"],
                false_information=data["false_information"],
                contradiction_point=data["contradiction_point"],
                expected_behavior=data["expected_behavior"],
                scenario_type=data["scenario_type"]
            )
            scenarios.append(scenario)
        
        logger.info(f"Loaded {len(scenarios)} scenarios from {filepath}")
        return scenarios
