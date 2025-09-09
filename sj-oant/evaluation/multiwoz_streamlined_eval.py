#!/usr/bin/env python3
"""
Streamlined MultiWOZ Evaluation for TMM System.
Uses our existing MultiWOZ data and simplified evaluation framework.
"""

import json
import logging
import sys
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our streamlined evaluation framework
from evaluation.multiwoz_framework.metrics import SimpleMultiWOZEvaluator
from tmm_pipeline import TMMPipelineFixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamlinedMultiWOZEvaluator:
    """Streamlined MultiWOZ evaluation using our existing data and simplified framework."""
    
    def __init__(self, api_key: str):
        """Initialize the evaluator with API key for TMM system."""
        self.api_key = api_key
        self.tmm_pipeline = None
        self.evaluator = None
        
    def initialize_systems(self):
        """Initialize both TMM system and simplified evaluator."""
        logger.info("Initializing TMM system...")
        self.tmm_pipeline = TMMPipelineFixed(self.api_key)
        
        logger.info("Initializing streamlined MultiWOZ evaluator...")
        self.evaluator = SimpleMultiWOZEvaluator(bleu=True, success=True, richness=True)
        
    def load_multiwoz_data(self, data_path: str, limit: int = None) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
        """Load MultiWOZ data and extract user-system dialogue pairs."""
        logger.info(f"Loading MultiWOZ data from {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Extract user-system dialogue pairs
        dialogue_pairs = {}
        user_inputs = {}
        count = 0
        
        for dialogue_id, dialogue in data.items():
            if limit and count >= limit:
                break
                
            # Clean dialogue ID (remove .json extension)
            clean_dialogue_id = dialogue_id.replace('.json', '')
            
            # Extract turns from the dialogue
            turns = []
            user_turns = []
            
            for i, turn in enumerate(dialogue['log']):
                if i % 2 == 0:  # User turns (even indices)
                    user_turns.append(turn['text'])
                else:  # System turns (odd indices)
                    # Create a turn in the expected format
                    turn_data = {
                        "response": turn['text'],
                        "state": {},  # We'll extract this from metadata if available
                        "active_domains": []  # We'll extract this from metadata if available
                    }
                    
                    # Extract state from metadata if available
                    if 'metadata' in turn:
                        metadata = turn['metadata']
                        for domain, domain_data in metadata.items():
                            if domain in ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'police', 'hospital']:
                                if 'semi' in domain_data:
                                    turn_data["state"][domain] = domain_data['semi']
                                if 'book' in domain_data and domain_data['book'].get('booked'):
                                    turn_data["active_domains"].append(domain)
                    
                    turns.append(turn_data)
            
            if turns and user_turns:  # Only add dialogues with both user and system turns
                dialogue_pairs[clean_dialogue_id] = turns
                user_inputs[clean_dialogue_id] = user_turns
                count += 1
                
        logger.info(f"Loaded {len(dialogue_pairs)} dialogues")
        return dialogue_pairs, user_inputs
        
    def generate_tmm_predictions(self, dialogue_pairs: Dict[str, List[Dict]], user_inputs: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """Generate predictions using our TMM system with proper user inputs."""
        logger.info("Generating TMM predictions...")
        
        predictions = {}
        
        for dialogue_id, dialogue in dialogue_pairs.items():
            logger.info(f"Processing dialogue {dialogue_id}")
            
            # Reset memory for each dialogue
            if hasattr(self.tmm_pipeline, 'multi_agent_pipeline') and hasattr(self.tmm_pipeline.multi_agent_pipeline, 'memory_store'):
                self.tmm_pipeline.multi_agent_pipeline.memory_store.reset_memory()
            
            tmm_turns = []
            user_turns = user_inputs[dialogue_id]
            
            for i, turn in enumerate(dialogue):
                # Use the corresponding user input
                if i < len(user_turns):
                    user_input = user_turns[i]
                else:
                    user_input = "User request"  # Fallback
                
                try:
                    # Process through TMM pipeline
                    response = self.tmm_pipeline.process(user_input)
                    
                    # Create prediction in the expected format
                    prediction = {
                        "response": response,
                        "state": turn.get("state", {}),
                        "active_domains": turn.get("active_domains", [])
                    }
                    
                    tmm_turns.append(prediction)
                    
                except Exception as e:
                    logger.error(f"Error processing turn in dialogue {dialogue_id}: {e}")
                    # Add a fallback response
                    tmm_turns.append({
                        "response": "I apologize, but I encountered an error processing your request.",
                        "state": turn.get("state", {}),
                        "active_domains": turn.get("active_domains", [])
                    })
            
            predictions[dialogue_id] = tmm_turns
            
        return predictions
        
    def evaluate_predictions(self, predictions: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Evaluate predictions using the streamlined MultiWOZ metrics."""
        logger.info("Evaluating predictions with streamlined MultiWOZ metrics...")
        
        results = self.evaluator.evaluate(predictions)
        
        return results
        
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*80)
        print("📊 TMM SYSTEM EVALUATION RESULTS (Streamlined MultiWOZ Metrics)")
        print("="*80)
        
        # BLEU Results
        if results.get('bleu'):
            print("\n🎯 BLEU SCORES:")
            bleu_data = results['bleu']
            print(f"  BLEU Score: {bleu_data.get('bleu', 0.0):.2f}")
        
        # Success Results
        if results.get('success'):
            print("\n✅ SUCCESS & INFORM RATES:")
            success_data = results['success']
            print(f"  Success Rate: {success_data.get('success', {}).get('total', 0.0):.1f}%")
            print(f"  Inform Rate: {success_data.get('inform', {}).get('total', 0.0):.1f}%")
        
        # Richness Results
        if results.get('richness'):
            print("\n📈 LEXICAL RICHNESS METRICS:")
            richness = results['richness']
            print(f"  Entropy: {richness.get('entropy', 0.0):.3f}")
            print(f"  Average Length: {richness.get('avg_lengths', 0.0):.1f}")
            print(f"  Unique Unigrams: {richness.get('num_unigrams', 0)}")
            print(f"  Unique Bigrams: {richness.get('num_bigrams', 0)}")
            print(f"  Unique Trigrams: {richness.get('num_trigrams', 0)}")
        
        print("\n" + "="*80)
        
    def run_evaluation(self, data_path: str, limit: int = None, output_path: str = None):
        """Run the complete evaluation pipeline."""
        try:
            # Initialize systems
            self.initialize_systems()
            
            # Load data
            dialogue_pairs, user_inputs = self.load_multiwoz_data(data_path, limit)
            
            # Generate predictions
            predictions = self.generate_tmm_predictions(dialogue_pairs, user_inputs)
            
            # Evaluate predictions
            results = self.evaluate_predictions(predictions)
            
            # Print results
            self.print_results(results)
            
            # Save results
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Streamlined MultiWOZ Evaluation for TMM System')
    parser.add_argument('--api-key', type=str, required=True, help='Google API key for TMM system')
    parser.add_argument('--data-path', type=str, 
                       default='data/MULTIWOZ2.4/MULTIWOZ2.4/data.json',
                       help='Path to MultiWOZ data file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of dialogues to evaluate')
    parser.add_argument('--output', type=str, default='results/streamlined_multiwoz_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = StreamlinedMultiWOZEvaluator(args.api_key)
    results = evaluator.run_evaluation(args.data_path, args.limit, args.output)
    
    print(f"\n✅ Streamlined MultiWOZ evaluation completed successfully!")
    print(f"Evaluated {len(results.get('bleu', {}))} dialogues with TMM system")


if __name__ == "__main__":
    main()
