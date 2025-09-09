#!/usr/bin/env python3
"""
MultiDoGO Evaluation for TMM System.
Evaluates TMM system performance on the Multi-Domain Goal-Oriented Dialogues dataset.
"""

import json
import logging
import sys
import os
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our MultiDoGO evaluation framework
from evaluation.multidogo_framework.metrics import SimpleMultiDoGOEvaluator
from tmm_pipeline import TMMPipelineFixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiDoGOEvaluator:
    """MultiDoGO evaluation using our TMM system and MultiDoGO-specific metrics."""
    
    def __init__(self, api_key: str):
        """Initialize the evaluator with API key for TMM system."""
        self.api_key = api_key
        self.tmm_pipeline = None
        self.evaluator = None
        
    def initialize_systems(self):
        """Initialize both TMM system and MultiDoGO evaluator."""
        logger.info("Initializing TMM system...")
        self.tmm_pipeline = TMMPipelineFixed(self.api_key)
        
        logger.info("Initializing MultiDoGO evaluator...")
        self.evaluator = SimpleMultiDoGOEvaluator(intent_accuracy=True, slot_f1=True, domain_adaptation=True, response_quality=True)
        
    def load_multidogo_data(self, data_path: str, limit: int = None) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
        """Load MultiDoGO data and extract user-system dialogue pairs."""
        logger.info(f"Loading MultiDoGO data from {data_path}")
        
        # Read TSV file
        df = pd.read_csv(data_path, sep='\t')
        
        # Extract user-system dialogue pairs
        dialogues = {}
        user_utterances = {}
        
        # Group by conversation ID
        for conversation_id, group in df.groupby('conversationId'):
            if limit and len(dialogues) >= limit:
                break
                
            # Extract user utterances (customer turns)
            user_turns = []
            for _, row in group.iterrows():
                if row['authorRole'] == 'customer':
                    user_turns.append(row['utterance'])
            
            if user_turns:  # Only include dialogues with user input
                dialogues[conversation_id] = user_turns
                user_utterances[conversation_id] = user_turns
        
        logger.info(f"Loaded {len(dialogues)} dialogues")
        return dialogues, user_utterances
    
    def generate_tmm_predictions(self, dialogues: Dict[str, List[str]], max_dialogues: int = None) -> Dict[str, List[Dict]]:
        """Generate TMM predictions for MultiDoGO dialogues."""
        logger.info("Generating TMM predictions...")
        
        predictions = {}
        count = 0
        
        for conversation_id, user_turns in dialogues.items():
            if max_dialogues and count >= max_dialogues:
                break
                
            logger.info(f"Processing dialogue {conversation_id}")
            
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
                        "conversation_id": conversation_id,
                        "user_input": user_input
                    }
                }
                
                dialogue_predictions.append(prediction)
            
            predictions[conversation_id] = dialogue_predictions
            count += 1
        
        logger.info(f"Generated predictions for {len(predictions)} dialogues")
        return predictions
    
    def evaluate_predictions(self, predictions: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Evaluate TMM predictions using MultiDoGO metrics."""
        logger.info("Evaluating predictions with MultiDoGO metrics...")
        
        # Format predictions for evaluation
        formatted_predictions = {}
        for conversation_id, dialogue_predictions in predictions.items():
            formatted_predictions[conversation_id] = dialogue_predictions
        
        # Run evaluation
        results = self.evaluator.evaluate(formatted_predictions)
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = "results/multidogo_tmm_results.json"):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*80)
        print("📊 TMM SYSTEM EVALUATION RESULTS (MultiDoGO Metrics)")
        print("="*80)
        
        if "intent_accuracy" in results:
            print(f"\n🎯 INTENT CLASSIFICATION:")
            print(f"  Intent Accuracy: {results['intent_accuracy']['total']:.2f}%")
        
        if "slot_f1" in results:
            print(f"\n📝 SLOT FILLING:")
            print(f"  Slot F1 Score: {results['slot_f1']['total']:.2f}%")
        
        if "domain_adaptation" in results:
            print(f"\n🔄 DOMAIN ADAPTATION:")
            print(f"  Domain Adaptation Score: {results['domain_adaptation']['total']:.2f}%")
        
        if "response_quality" in results:
            print(f"\n📈 RESPONSE QUALITY:")
            print(f"  Response Quality Score: {results['response_quality']['total']:.2f}%")
        
        print("="*80)
    
    def run_evaluation(self, data_path: str, max_dialogues: int = 250, max_examples: int = None):
        """Run complete MultiDoGO evaluation."""
        # Initialize systems
        self.initialize_systems()
        
        # Load data
        dialogues, user_utterances = self.load_multidogo_data(data_path, limit=max_dialogues)
        
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
    evaluator = MultiDoGOEvaluator(api_key)
    
    # Run evaluation on airline domain data
    data_path = "data/multidogo/airline.tsv"
    results = evaluator.run_evaluation(data_path, max_dialogues=50, max_examples=250)
    
    print("MultiDoGO evaluation completed!")


if __name__ == "__main__":
    main()
