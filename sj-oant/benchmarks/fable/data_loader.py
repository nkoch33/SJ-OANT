"""
benchmarks.fable.data_loader - FictionalQA Dataset Integration

This module provides data loading and adaptation functionality to convert
the HuggingFace FictionalQA dataset into FABLE benchmark format for
truth-maintained memory evaluation.

The FictionalQA dataset contains:
- Fictional scenarios with embedded facts
- Questions requiring memory retrieval and reasoning
- Ground truth answers for evaluation

Key Features:
- Direct HuggingFace datasets integration
- FABLE format conversion with knowledge base generation
- Memory scenario creation from fictional contexts
- Truth/misinformation seeding for false memory testing
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import uuid4

from datasets import load_dataset

logger = logging.getLogger(__name__)

@dataclass
class FictionalQARecord:
    """A single record from the FictionalQA dataset."""
    id: str
    title: str
    fictional_context: str
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FABLEScenario:
    """A FABLE-format scenario generated from FictionalQA data."""
    scenario_id: str
    difficulty: str
    domain: str
    turns: int
    scenario_kb: str
    conversation: List[Dict[str, Any]]
    source_record: FictionalQARecord

class FictionalQALoader:
    """
    Loader and adapter for the FictionalQA dataset.
    
    This class handles downloading the dataset from HuggingFace and converting
    it into FABLE-compatible scenarios for truth-maintained memory testing.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the FictionalQA loader.
        
        Args:
            cache_dir: Optional directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        self.dataset = None
        logger.info("Initialized FictionalQA loader")
    
    def load_dataset(self) -> None:
        """Load the FictionalQA dataset from HuggingFace."""
        try:
            logger.info("Loading FictionalQA dataset from HuggingFace...")
            self.dataset = load_dataset("tomg-group-umd/fictionalqa", cache_dir=self.cache_dir)
            logger.info(f"Loaded FictionalQA dataset with {len(self.dataset['train'])} training examples")
        except Exception as e:
            logger.error(f"Failed to load FictionalQA dataset: {e}")
            raise
    
    def get_records(self, split: str = "train", limit: Optional[int] = None) -> List[FictionalQARecord]:
        """
        Get FictionalQA records from the dataset.
        
        Args:
            split: Dataset split to use ("train", "validation", "test")
            limit: Optional limit on number of records to return
            
        Returns:
            List of FictionalQARecord instances
        """
        if self.dataset is None:
            self.load_dataset()
        
        split_data = self.dataset[split]
        records = []
        
        count = min(len(split_data), limit) if limit else len(split_data)
        logger.info(f"Converting {count} FictionalQA records from {split} split")
        
        for i in range(count):
            example = split_data[i]
            record = FictionalQARecord(
                id=f"fictionalqa_{split}_{i}",
                title=example.get("title", f"Story_{i}"),
                fictional_context=example["story"],
                question=example["question"],
                answer=example["answer"],
                metadata={
                    "original_index": i,
                    "split": split,
                    "source": "tomg-group-umd/fictionalqa"
                }
            )
            records.append(record)
        
        return records
    
    def convert_to_fable_scenario(self, record: FictionalQARecord, 
                                 inject_misinformation: bool = True) -> FABLEScenario:
        """
        Convert a FictionalQA record into a FABLE scenario.
        
        Args:
            record: FictionalQA record to convert
            inject_misinformation: Whether to inject false information for testing
            
        Returns:
            FABLE scenario with conversation turns and knowledge base
        """
        scenario_id = f"fable_{record.id}"
        
        # Create conversation turns based on the fictional context and question
        conversation = []
        
        # Turn 1: User provides context
        conversation.append({
            "turn": 1,
            "speaker": "user",
            "text": f"Let me tell you about this story: {record.fictional_context}",
            "facts_referenced": [f"{scenario_id}_context"],
            "planted_misinformation": None,
            "is_correction": False
        })
        
        # Turn 2: Assistant acknowledges
        conversation.append({
            "turn": 2,
            "speaker": "assistant", 
            "text": "I understand. I'll remember the details from this story.",
            "should_contain_facts": [f"{scenario_id}_context"],
            "should_not_contain": []
        })
        
        # Optional misinformation injection (Turn 3-4)
        turn_num = 3
        misinformation_fact = None
        if inject_misinformation:
            # Create a plausible but false statement
            misinformation_fact = f"{scenario_id}_misinformation"
            conversation.append({
                "turn": turn_num,
                "speaker": "user",
                "text": f"Actually, I just remembered something else about {record.title} - [INJECT CONTRADICTORY DETAIL HERE]",
                "facts_referenced": [],
                "planted_misinformation": misinformation_fact,
                "is_correction": False
            })
            turn_num += 1
            
            conversation.append({
                "turn": turn_num,
                "speaker": "assistant",
                "text": "I've noted that additional information.",
                "should_contain_facts": [],
                "should_not_contain": [misinformation_fact]  # Should resist false memory
            })
            turn_num += 1
        
        # Turn N: User asks the question
        conversation.append({
            "turn": turn_num,
            "speaker": "user", 
            "text": record.question,
            "facts_referenced": [f"{scenario_id}_context"],
            "planted_misinformation": None,
            "is_correction": False
        })
        turn_num += 1
        
        # Turn N+1: Assistant should provide correct answer
        conversation.append({
            "turn": turn_num,
            "speaker": "assistant",
            "text": record.answer,  # Ground truth answer
            "should_contain_facts": [f"{scenario_id}_context"],
            "should_not_contain": [misinformation_fact] if misinformation_fact else []
        })
        
        # Determine difficulty based on context length and question complexity
        context_length = len(record.fictional_context.split())
        question_complexity = len(record.question.split())
        
        if context_length > 500 or question_complexity > 15:
            difficulty = "hard"
        elif context_length > 200 or question_complexity > 10:
            difficulty = "medium"
        else:
            difficulty = "easy"
        
        return FABLEScenario(
            scenario_id=scenario_id,
            difficulty=difficulty,
            domain="fictional_reasoning",
            turns=len(conversation),
            scenario_kb=f"kb/{scenario_id}.json",
            conversation=conversation,
            source_record=record
        )
    
    def generate_knowledge_base(self, scenario: FABLEScenario) -> Dict[str, Any]:
        """
        Generate a FABLE knowledge base from a scenario.
        
        Args:
            scenario: FABLE scenario to generate KB for
            
        Returns:
            Knowledge base dictionary in FABLE format
        """
        kb = {
            "scenario_id": scenario.scenario_id,
            "domain": scenario.domain,
            "entities": {
                "story": {
                    "title": scenario.source_record.title,
                    "attributes": {
                        "source": "fictional_context",
                        "length": len(scenario.source_record.fictional_context.split()),
                        "complexity": scenario.difficulty
                    }
                }
            },
            "facts": {
                f"{scenario.scenario_id}_context": {
                    "statement": scenario.source_record.fictional_context,
                    "confidence": 1.0,
                    "source": "user_provided",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                f"{scenario.scenario_id}_answer": {
                    "statement": scenario.source_record.answer,
                    "confidence": 1.0,
                    "source": "ground_truth",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            },
            "relationships": {},
            "temporal_events": {},
            "contradictions": {}
        }
        
        # Add misinformation if present
        for turn in scenario.conversation:
            if turn.get("planted_misinformation"):
                misinformation_id = turn["planted_misinformation"]
                kb["contradictions"][f"contradiction_{misinformation_id}"] = {
                    "original_fact": f"{scenario.scenario_id}_context",
                    "contradicting_statement": "[MISINFORMATION PLACEHOLDER]",
                    "introduced_at_turn": turn["turn"],
                    "corrected_at_turn": None  # To be filled when correction occurs
                }
        
        return kb
    
    def save_fable_scenarios(self, scenarios: List[FABLEScenario], 
                           output_dir: str = "benchmarks/fable") -> None:
        """
        Save FABLE scenarios and knowledge bases to disk.
        
        Args:
            scenarios: List of FABLE scenarios to save
            output_dir: Directory to save scenarios in
        """
        output_path = Path(output_dir)
        scenarios_dir = output_path / "scenarios" / "fictional"
        kb_dir = output_path / "kb"
        
        # Create directories
        scenarios_dir.mkdir(parents=True, exist_ok=True)
        kb_dir.mkdir(parents=True, exist_ok=True)
        
        # Group scenarios by difficulty
        scenarios_by_difficulty = {}
        for scenario in scenarios:
            difficulty = scenario.difficulty
            if difficulty not in scenarios_by_difficulty:
                scenarios_by_difficulty[difficulty] = []
            scenarios_by_difficulty[difficulty].append(scenario)
        
        # Save scenarios by difficulty
        for difficulty, scenario_list in scenarios_by_difficulty.items():
            scenario_file = scenarios_dir / f"{difficulty}_fictional.jsonl"
            
            with open(scenario_file, 'w') as f:
                for scenario in scenario_list:
                    scenario_data = {
                        "scenario_id": scenario.scenario_id,
                        "difficulty": scenario.difficulty,
                        "domain": scenario.domain,
                        "turns": scenario.turns,
                        "scenario_kb": scenario.scenario_kb,
                        "conversation": scenario.conversation,
                        "source": {
                            "dataset": "tomg-group-umd/fictionalqa",
                            "record_id": scenario.source_record.id,
                            "title": scenario.source_record.title
                        }
                    }
                    f.write(json.dumps(scenario_data) + '\n')
            
            logger.info(f"Saved {len(scenario_list)} {difficulty} scenarios to {scenario_file}")
        
        # Save knowledge bases
        for scenario in scenarios:
            kb = self.generate_knowledge_base(scenario)
            kb_file = kb_dir / f"{scenario.scenario_id}.json"
            
            with open(kb_file, 'w') as f:
                json.dump(kb, f, indent=2)
        
        logger.info(f"Saved {len(scenarios)} knowledge bases to {kb_dir}")

def main():
    """Example usage of the FictionalQA loader."""
    # Initialize loader
    loader = FictionalQALoader()
    
    # Load and convert a small sample for testing
    records = loader.get_records(split="train", limit=10)
    scenarios = []
    
    for record in records:
        scenario = loader.convert_to_fable_scenario(record, inject_misinformation=True)
        scenarios.append(scenario)
    
    # Save scenarios
    loader.save_fable_scenarios(scenarios)
    
    print(f"Generated {len(scenarios)} FABLE scenarios from FictionalQA dataset")
    print("Scenarios saved to benchmarks/fable/scenarios/fictional/")
    print("Knowledge bases saved to benchmarks/fable/kb/")

if __name__ == "__main__":
    main()
