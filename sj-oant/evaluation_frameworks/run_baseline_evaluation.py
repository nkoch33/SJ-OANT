"""
Run Baseline Evaluation Script

Script to evaluate baseline models (LLMs and RAG systems) on both dialogue performance
and false memory prevention tasks across all benchmarks.
"""

import logging
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

from baseline_evaluator import BaselineEvaluator, EvaluationConfig
from false_memory_evaluation.baseline_models import (
    SimpleRAGBaseline, EmbeddingRAGBaseline,
    SimpleRAGConfig, EmbeddingRAGConfig
)

logger = logging.getLogger(__name__)

def load_benchmark_data(benchmark: str, data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load benchmark data for evaluation.
    
    Args:
        benchmark: Benchmark name (multiwoz, sgd, taskmaster)
        data_dir: Directory containing benchmark data
        
    Returns:
        List of conversations
    """
    # Mock data loading - in practice, this would load actual benchmark data
    mock_conversations = []
    for i in range(100):  # Mock 100 conversations
        conversation = {
            "id": f"{benchmark}_conv_{i}",
            "turns": [
                {
                    "role": "user",
                    "content": f"Mock user input {i}",
                    "reference": f"Mock reference response {i}",
                    "slots": [{"name": "slot1", "value": "value1"}],
                    "intent": "mock_intent"
                },
                {
                    "role": "assistant",
                    "content": f"Mock assistant response {i}"
                }
            ]
        }
        mock_conversations.append(conversation)
    
    return mock_conversations

def load_false_memory_data(benchmark: str, data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load false memory benchmark data.
    
    Args:
        benchmark: Benchmark name
        data_dir: Directory containing benchmark data
        
    Returns:
        List of conversations with false memory injection
    """
    # Mock false memory data - in practice, this would load actual injected data
    mock_conversations = []
    false_facts = [
        "Cambridge is in Scotland",
        "The train leaves at 2:15 PM",
        "The hotel costs $200 per night",
        "The restaurant closes at 8 PM"
    ]
    
    for i in range(100):  # Mock 100 conversations
        conversation = {
            "id": f"{benchmark}_false_conv_{i}",
            "turns": [
                {
                    "role": "user",
                    "content": f"Mock user input {i}",
                    "false_memory_injected": i % 4 == 0,  # Inject every 4th turn
                    "false_fact": false_facts[i % len(false_facts)] if i % 4 == 0 else None
                },
                {
                    "role": "assistant",
                    "content": f"Mock assistant response {i}"
                }
            ]
        }
        mock_conversations.append(conversation)
    
    return mock_conversations

def evaluate_baseline_models():
    """Evaluate all baseline models on all benchmarks."""
    
    # Initialize evaluator
    config = EvaluationConfig(
        num_conversations=100,
        output_dir="baseline_results"
    )
    evaluator = BaselineEvaluator(config)
    
    # Initialize baseline models
    models = {
        "simple_rag": SimpleRAGBaseline(
            model_name="simple_rag",
            config=SimpleRAGConfig(
                retrieval_top_k=5,
                keyword_threshold=0.3
            )
        ),
        "embedding_rag": EmbeddingRAGBaseline(
            model_name="embedding_rag",
            config=EmbeddingRAGConfig(
                model_name="all-MiniLM-L6-v2",
                retrieval_top_k=5,
                similarity_threshold=0.3
            )
        )
    }
    
    # Benchmarks to evaluate
    benchmarks = ["multiwoz", "sgd", "taskmaster"]
    
    # Evaluation results
    all_results = {
        "dialogue_performance": {},
        "false_memory_prevention": {}
    }
    
    # Evaluate dialogue performance
    logger.info("Starting dialogue performance evaluation...")
    for benchmark in benchmarks:
        logger.info(f"Evaluating dialogue performance on {benchmark}")
        
        # Load benchmark data
        conversations = load_benchmark_data(benchmark)
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} on {benchmark}")
            
            # Add some mock knowledge to RAG models
            for i, conv in enumerate(conversations[:10]):  # Add first 10 conversations as knowledge
                for turn in conv["turns"]:
                    if turn["role"] == "user":
                        model.add_knowledge(turn["content"], {"source": f"{benchmark}_conv_{i}"})
            
            # Evaluate dialogue performance
            results = evaluator.evaluate_dialogue_performance(
                model=model,
                benchmark=benchmark,
                conversations=conversations
            )
            
            if benchmark not in all_results["dialogue_performance"]:
                all_results["dialogue_performance"][benchmark] = {}
            all_results["dialogue_performance"][benchmark][model_name] = results
    
    # Evaluate false memory prevention
    logger.info("Starting false memory prevention evaluation...")
    for benchmark in benchmarks:
        logger.info(f"Evaluating false memory prevention on {benchmark}")
        
        # Load false memory data
        false_memory_conversations = load_false_memory_data(benchmark)
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} false memory prevention on {benchmark}")
            
            # Clear and re-add knowledge for false memory evaluation
            model.clear_memory()
            for i, conv in enumerate(false_memory_conversations[:10]):
                for turn in conv["turns"]:
                    if turn["role"] == "user" and not turn.get("false_memory_injected", False):
                        model.add_knowledge(turn["content"], {"source": f"{benchmark}_false_conv_{i}"})
            
            # Evaluate false memory prevention
            results = evaluator.evaluate_false_memory_prevention(
                model=model,
                benchmark=benchmark,
                conversations=false_memory_conversations
            )
            
            if benchmark not in all_results["false_memory_prevention"]:
                all_results["false_memory_prevention"][benchmark] = {}
            all_results["false_memory_prevention"][benchmark][model_name] = results
    
    # Save comprehensive results
    results_file = "baseline_results/comprehensive_baseline_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Comprehensive evaluation results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE EVALUATION SUMMARY")
    print("="*80)
    
    print("\nDIALOGUE PERFORMANCE RESULTS:")
    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()}:")
        for model_name in models.keys():
            if benchmark in all_results["dialogue_performance"] and model_name in all_results["dialogue_performance"][benchmark]:
                metrics = all_results["dialogue_performance"][benchmark][model_name]["metrics"]
                print(f"  {model_name}:")
                print(f"    BLEU: {metrics['bleu']:.3f}")
                print(f"    ROUGE: {metrics['rouge']:.3f}")
                print(f"    Semantic Similarity: {metrics['semantic_similarity']:.3f}")
                print(f"    Slot F1: {metrics['slot_f1']:.3f}")
                print(f"    Intent Accuracy: {metrics['intent_accuracy']:.3f}")
    
    print("\nFALSE MEMORY PREVENTION RESULTS:")
    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()}:")
        for model_name in models.keys():
            if benchmark in all_results["false_memory_prevention"] and model_name in all_results["false_memory_prevention"][benchmark]:
                metrics = all_results["false_memory_prevention"][benchmark][model_name]["false_memory_metrics"]
                print(f"  {model_name}:")
                print(f"    FMR: {metrics['fmr']:.3f}")
                print(f"    MEL: {metrics['mel']:.3f}")
                print(f"    DAR: {metrics['dar']:.3f}")
                print(f"    Contradiction Detection: {metrics['contradiction_detection_rate']:.3f}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--num-conversations", type=int, default=100,
                       help="Number of conversations to evaluate")
    parser.add_argument("--output-dir", type=str, default="baseline_results",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    evaluate_baseline_models()

if __name__ == "__main__":
    main()
