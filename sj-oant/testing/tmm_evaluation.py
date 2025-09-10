#!/usr/bin/env python3
"""
TMM Model Evaluation: Test TMM system against 4 dialogue benchmarks.
"""

import json
import logging
import sys
import os
import random
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our TMM system
from tmm_pipeline import TMMPipelineFixed

# Import evaluation frameworks
from evaluation.multiwoz_framework.metrics import SimpleMultiWOZEvaluator
from evaluation.sgd_framework.metrics import SimpleSGDEvaluator
from evaluation.taskmaster_framework.metrics import SimpleTaskmasterEvaluator
from evaluation.multidogo_framework.metrics import SimpleMultiDoGOEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TMMEvaluator:
    """Evaluator for TMM system against dialogue benchmarks."""
    
    def __init__(self, api_key: str):
        """Initialize the evaluator with API key for TMM system."""
        self.api_key = api_key
        self.tmm_pipeline = None
        self.evaluators = {}
        self.results = {}
        
    def initialize_systems(self):
        """Initialize TMM system and evaluators."""
        logger.info("Initializing TMM system...")
        self.tmm_pipeline = TMMPipelineFixed(self.api_key)
        
        logger.info("Initializing evaluators...")
        self.evaluators = {
            "multiwoz": SimpleMultiWOZEvaluator(),
            "sgd": SimpleSGDEvaluator(),
            "taskmaster": SimpleTaskmasterEvaluator(),
            "multidogo": SimpleMultiDoGOEvaluator()
        }
        
    def load_sample_data(self, benchmark: str, num_examples: int = 25) -> List[Dict]:
        """Load sample data from each benchmark."""
        logger.info(f"Loading {num_examples} examples from {benchmark}...")
        
        if benchmark == "multiwoz":
            return self._load_multiwoz_samples(num_examples)
        elif benchmark == "sgd":
            return self._load_sgd_samples(num_examples)
        elif benchmark == "taskmaster":
            return self._load_taskmaster_samples(num_examples)
        elif benchmark == "multidogo":
            return self._load_multidogo_samples(num_examples)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    def _load_multiwoz_samples(self, num_examples: int) -> List[Dict]:
        """Load MultiWOZ samples."""
        data_path = "data/MULTIWOZ2.4/MULTIWOZ2.4/data.json"
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Get random sample of dialogues
        dialogue_ids = list(data.keys())
        random.shuffle(dialogue_ids)
        selected_ids = dialogue_ids[:num_examples]
        
        samples = []
        for dialogue_id in selected_ids:
            dialogue = data[dialogue_id]
            # Extract user utterances (odd-indexed turns are user turns)
            user_turns = []
            for i, turn in enumerate(dialogue["log"]):
                if i % 2 == 0:  # Even indices (0, 2, 4...) are user turns
                    user_turns.append(turn["text"])
            
            if user_turns:
                samples.append({
                    "dialogue_id": dialogue_id,
                    "user_turns": user_turns,
                    "benchmark": "multiwoz"
                })
        
        return samples[:num_examples]
    
    def _load_sgd_samples(self, num_examples: int) -> List[Dict]:
        """Load SGD samples."""
        data_path = "data/sgd/dialogues_001.json"
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Get random sample of dialogues
        random.shuffle(data)
        selected_dialogues = data[:num_examples]
        
        samples = []
        for dialogue in selected_dialogues:
            # Extract user utterances
            user_turns = []
            for turn in dialogue["turns"]:
                if turn["speaker"] == "USER":
                    user_turns.append(turn["utterance"])
            
            if user_turns:
                samples.append({
                    "dialogue_id": dialogue["dialogue_id"],
                    "user_turns": user_turns,
                    "benchmark": "sgd"
                })
        
        return samples[:num_examples]
    
    def _load_taskmaster_samples(self, num_examples: int) -> List[Dict]:
        """Load Taskmaster samples."""
        data_path = "data/taskmaster/restaurant-search.json"
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Get random sample of dialogues
        random.shuffle(data)
        selected_dialogues = data[:num_examples]
        
        samples = []
        for dialogue in selected_dialogues:
            # Extract user utterances
            user_turns = []
            for utterance in dialogue["utterances"]:
                if utterance["speaker"] == "USER":
                    user_turns.append(utterance["text"])
            
            if user_turns:
                samples.append({
                    "dialogue_id": dialogue["conversation_id"],
                    "user_turns": user_turns,
                    "benchmark": "taskmaster"
                })
        
        return samples[:num_examples]
    
    def _load_multidogo_samples(self, num_examples: int) -> List[Dict]:
        """Load MultiDoGO samples."""
        # Try different MultiDoGO files
        data_files = ["data/multidogo/airline.tsv", "data/multidogo/fastfood.tsv", "data/multidogo/airline_annotated.tsv"]
        
        all_samples = []
        for data_path in data_files:
            try:
                # Use proper TSV parsing with error handling
                df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')
                
                # Get random sample of conversations
                conversation_ids = df['conversationId'].unique()
                random.shuffle(conversation_ids)
                selected_ids = conversation_ids[:min(10, len(conversation_ids))]  # Take up to 10 per file
                
                for conversation_id in selected_ids:
                    # Extract user utterances
                    user_turns = []
                    for _, row in df[df['conversationId'] == conversation_id].iterrows():
                        if row['authorRole'] == 'customer':
                            user_turns.append(row['utterance'])
                    
                    if user_turns:
                        all_samples.append({
                            "dialogue_id": conversation_id,
                            "user_turns": user_turns,
                            "benchmark": "multidogo"
                        })
            except Exception as e:
                logger.warning(f"Could not load {data_path}: {e}")
                continue
        
        # Shuffle and return requested number
        random.shuffle(all_samples)
        return all_samples[:num_examples]
    
    def evaluate_tmm(self, samples: List[Dict], benchmark: str) -> Dict[str, Any]:
        """Evaluate TMM system on samples."""
        logger.info(f"Evaluating TMM system on {benchmark}...")
        
        predictions = {}
        
        for sample in samples:
            dialogue_id = sample["dialogue_id"]
            user_turns = sample["user_turns"]
            
            # Reset memory for each dialogue
            self.tmm_pipeline.multi_agent_pipeline.memory_store.reset_memory()
            
            dialogue_predictions = []
            for user_input in user_turns:
                try:
                    # Process with TMM pipeline
                    response = self.tmm_pipeline.process(user_input)
                    
                    # Collect prediction data
                    prediction = {
                        "response": response,
                        "context": [user_input],
                        "memory_operations": self.tmm_pipeline.multi_agent_pipeline.memory_store.get_memory_summary(),
                        "metadata": {
                            "dialogue_id": dialogue_id,
                            "user_input": user_input,
                            "system": "TMM"
                        }
                    }
                    
                    dialogue_predictions.append(prediction)
                    
                except Exception as e:
                    logger.error(f"Error processing TMM: {str(e)}")
                    prediction = {
                        "response": f"Error: {str(e)}",
                        "context": [user_input],
                        "memory_operations": {},
                        "metadata": {
                            "dialogue_id": dialogue_id,
                            "user_input": user_input,
                            "system": "TMM"
                        }
                    }
                    dialogue_predictions.append(prediction)
            
            predictions[dialogue_id] = dialogue_predictions
        
        # Evaluate predictions
        evaluator = self.evaluators[benchmark]
        results = evaluator.evaluate(predictions)
        
        return results
    
    def run_evaluation(self, num_examples: int = 25):
        """Run TMM evaluation across all benchmarks."""
        logger.info("Starting TMM evaluation...")
        
        # Initialize systems
        self.initialize_systems()
        
        benchmarks = ["multiwoz", "sgd", "taskmaster", "multidogo"]
        all_results = {}
        
        for benchmark in benchmarks:
            logger.info(f"Evaluating {benchmark}...")
            
            # Load sample data
            samples = self.load_sample_data(benchmark, num_examples)
            
            # Evaluate TMM
            try:
                results = self.evaluate_tmm(samples, benchmark)
                all_results[benchmark] = {"TMM": results}
                logger.info(f"Completed TMM on {benchmark}")
            except Exception as e:
                logger.error(f"Error evaluating TMM on {benchmark}: {str(e)}")
                all_results[benchmark] = {"TMM": {"error": str(e)}}
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_path: str = "testing/results/tmm_evaluation_results.json"):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add metadata
        results_with_metadata = {
            "timestamp": datetime.now().isoformat(),
            "system": "TMM",
            "num_examples_per_benchmark": 25,
            "benchmarks": ["multiwoz", "sgd", "taskmaster", "multidogo"],
            "results": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print a comprehensive summary of TMM results."""
        print("\n" + "="*100)
        print("🏆 TMM SYSTEM EVALUATION RESULTS")
        print("="*100)
        
        for benchmark, benchmark_results in self.results.items():
            print(f"\n📊 {benchmark.upper()} BENCHMARK:")
            print("-" * 50)
            
            tmm_results = benchmark_results.get("TMM", {})
            
            if "error" in tmm_results:
                print(f"❌ TMM: {tmm_results['error']}")
                continue
            
            # Extract key metrics based on benchmark
            if benchmark == "multiwoz":
                key_metrics = ["bleu", "success", "inform"]
            elif benchmark == "sgd":
                key_metrics = ["intent_accuracy", "slot_f1", "success_rate"]
            elif benchmark == "taskmaster":
                key_metrics = ["bleu", "rouge", "task_completion"]
            elif benchmark == "multidogo":
                key_metrics = ["intent_accuracy", "slot_f1", "domain_adaptation"]
            
            print(f"✅ TMM System:")
            for metric in key_metrics:
                if metric in tmm_results:
                    if isinstance(tmm_results[metric], dict) and "total" in tmm_results[metric]:
                        score = tmm_results[metric]["total"]
                    elif isinstance(tmm_results[metric], dict):
                        score = list(tmm_results[metric].values())[0]
                    else:
                        score = tmm_results[metric]
                    print(f"   {metric}: {score:.2f}%")
            print()
        
        print("="*100)


def main():
    """Main evaluation function."""
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    # Initialize evaluator
    evaluator = TMMEvaluator(api_key)
    
    print("🚀 Starting TMM System Evaluation")
    print("="*60)
    print("📊 Testing TMM system on 4 dialogue benchmarks")
    print("🎯 25 examples each from MultiWOZ, SGD, Taskmaster, MultiDoGO")
    print("⏱️  This may take 10-15 minutes...")
    print("="*60)
    
    try:
        results = evaluator.run_evaluation(num_examples=25)
        
        # Save results
        evaluator.save_results()
        
        # Print summary
        evaluator.print_summary()
        
        print("\n✅ TMM evaluation completed successfully!")
        print("📁 Results saved to: testing/results/tmm_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ TMM evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
