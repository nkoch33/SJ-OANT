#!/usr/bin/env python3
"""
SGD (Schema-Guided Dialogue) Evaluation for TMM System.
Evaluates TMM system performance on the SGD dataset.
"""

import json
import logging
import sys
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our SGD evaluation framework
from evaluation.sgd_framework.metrics import SimpleSGDEvaluator
from tmm_pipeline import TMMPipelineFixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SGDEvaluator:
    """SGD evaluation using our TMM system and SGD-specific metrics."""
    
    def __init__(self, api_key: str):
        """Initialize the evaluator with API key for TMM system."""
        self.api_key = api_key
        self.tmm_pipeline = None
        self.evaluator = None
        
    def initialize_systems(self):
        """Initialize both TMM system and SGD evaluator."""
        logger.info("Initializing TMM system...")
        self.tmm_pipeline = TMMPipelineFixed(self.api_key)
        
        logger.info("Initializing SGD evaluator...")
        self.evaluator = SimpleSGDEvaluator(intent_accuracy=True, slot_f1=True, success_rate=True, bleu=True)
        
    def load_sgd_data(self, data_path: str, limit: int = None) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
        """Load SGD data and extract user-system dialogue pairs."""
        logger.info(f"Loading SGD data from {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Extract user-system dialogue pairs
        dialogues = {}
        user_utterances = {}
        
        count = 0
        for dialogue in data:
            if limit and count >= limit:
                break
                
            dialogue_id = dialogue["dialogue_id"]
            turns = dialogue["turns"]
            
            # Extract user utterances
            user_turns = []
            for turn in turns:
                if turn["speaker"] == "USER":
                    user_turns.append(turn["utterance"])
            
            if user_turns:  # Only include dialogues with user input
                dialogues[dialogue_id] = user_turns
                user_utterances[dialogue_id] = user_turns
                count += 1
        
        logger.info(f"Loaded {len(dialogues)} dialogues")
        return dialogues, user_utterances
    
    def generate_tmm_predictions(self, dialogues: Dict[str, List[str]], max_dialogues: int = None) -> Dict[str, List[Dict]]:
        """Generate TMM predictions for SGD dialogues."""
        logger.info("Generating TMM predictions...")
        
        predictions = {}
        count = 0
        
        for dialogue_id, user_turns in dialogues.items():
            if max_dialogues and count >= max_dialogues:
                break
                
            logger.info(f"Processing dialogue {dialogue_id}")
            
            # Reset memory for each dialogue
            self.tmm_pipeline.multi_agent_pipeline.memory_store.reset_memory()
            
            dialogue_predictions = []
            for user_input in user_turns:
                # Process with TMM pipeline
                response = self.tmm_pipeline.process(user_input)
                
                # Collect prediction data
                prediction = {
                    "response": response,
                    "context": [user_input],  # Simplified context
                    "memory_operations": self.tmm_pipeline.multi_agent_pipeline.memory_store.get_memory_summary(),
                    "metadata": {
                        "dialogue_id": dialogue_id,
                        "user_input": user_input
                    }
                }
                
                dialogue_predictions.append(prediction)
            
            predictions[dialogue_id] = dialogue_predictions
            count += 1
        
        logger.info(f"Generated predictions for {len(predictions)} dialogues")
        return predictions
    
    def evaluate_predictions(self, predictions: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Evaluate TMM predictions using SGD metrics."""
        logger.info("Evaluating predictions with SGD metrics...")
        
        # Format predictions for evaluation
        formatted_predictions = {}
        for dialogue_id, dialogue_predictions in predictions.items():
            formatted_predictions[dialogue_id] = dialogue_predictions
        
        # Run evaluation
        results = self.evaluator.evaluate(formatted_predictions)
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = "results/sgd_tmm_results.json"):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*80)
        print("📊 TMM SYSTEM EVALUATION RESULTS (SGD Metrics)")
        print("="*80)
        
        if "intent_accuracy" in results:
            print(f"\n🎯 INTENT ACCURACY:")
            print(f"  Intent Accuracy: {results['intent_accuracy']['total']:.2f}%")
        
        if "slot_f1" in results:
            print(f"\n📝 SLOT FILLING:")
            print(f"  Slot F1 Score: {results['slot_f1']['total']:.2f}%")
        
        if "success_rate" in results:
            print(f"\n✅ TASK SUCCESS:")
            print(f"  Success Rate: {results['success_rate']['total']:.2f}%")
        
        if "bleu" in results:
            print(f"\n📈 RESPONSE QUALITY:")
            print(f"  BLEU Score: {results['bleu']['bleu']:.2f}%")
        
        print("="*80)
    
    def run_evaluation(self, data_path: str, max_dialogues: int = 250, max_examples: int = None):
        """Run complete SGD evaluation."""
        # Initialize systems
        self.initialize_systems()
        
        # Load data
        dialogues, user_utterances = self.load_sgd_data(data_path, limit=max_dialogues)
        
        # Generate predictions
        predictions = self.generate_tmm_predictions(dialogues, max_dialogues=max_examples)
        
        # Evaluate predictions
        results = self.evaluate_predictions(predictions)
        
        # Save and display results
        self.save_results(results)
        self.print_results(results)
        
        return results


def main():
    """Main evaluation function."""
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Initialize evaluator
    evaluator = SGDEvaluator(api_key)
    
    # Run evaluation on dev set
    data_path = "data/sgd/dialogues_001.json"  # Will be created after cleanup
    results = evaluator.run_evaluation(data_path, max_dialogues=50, max_examples=250)
    
    print("SGD evaluation completed!")


if __name__ == "__main__":
    main()
