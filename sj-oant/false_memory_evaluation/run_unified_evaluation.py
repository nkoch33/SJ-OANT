#!/usr/bin/env python3
"""
Unified False Memory Evaluation Runner

Runs evaluation of both TMM and baseline models using identical methodology
for perfect research integrity and fair comparison.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from false_memory_evaluation.unified_evaluator import UnifiedFalseMemoryEvaluator, EvaluationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> EvaluationConfig:
    """Load evaluation configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Convert to EvaluationConfig
        config = EvaluationConfig(**config_data)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = EvaluationConfig()
        logger.info("Using default configuration")
    
    return config

def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(description="Unified False Memory Evaluation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per benchmark")
    parser.add_argument("--benchmarks", nargs="+", default=["multiwoz", "sgd", "taskmaster"], 
                       help="Benchmarks to evaluate")
    parser.add_argument("--output", type=str, help="Output path for results")
    parser.add_argument("--no-tmm", action="store_true", help="Skip TMM evaluation")
    parser.add_argument("--no-llama2", action="store_true", help="Skip Llama2 evaluation")
    parser.add_argument("--no-mistral", action="store_true", help="Skip Mistral evaluation")
    parser.add_argument("--no-gpt35", action="store_true", help="Skip GPT-3.5-turbo evaluation")
    parser.add_argument("--tmm-only", action="store_true", help="Evaluate TMM only")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate baseline models only")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.samples:
        config.samples_per_benchmark = args.samples
    if args.output:
        config.output_path = args.output
    if args.no_tmm:
        config.enable_tmm = False
    if args.no_llama2:
        config.enable_llama2 = False
    if args.no_mistral:
        config.enable_mistral = False
    if args.no_gpt35:
        config.enable_gpt35 = False
    if args.tmm_only:
        config.enable_llama2 = False
        config.enable_mistral = False
        config.enable_gpt35 = False
    if args.baseline_only:
        config.enable_tmm = False
    
    # Get TMM API key from environment
    if config.enable_tmm:
        tmm_api_key = os.environ.get('GEMINI_API_KEY')
        if not tmm_api_key:
            logger.error("GEMINI_API_KEY environment variable not set for TMM evaluation")
            logger.info("Please set your Gemini API key: export GEMINI_API_KEY='your_api_key_here'")
            sys.exit(1)
        config.tmm_api_key = tmm_api_key
    
    # Check for baseline model API keys
    if config.enable_llama2 or config.enable_mistral:
        hf_key = os.environ.get('HUGGINGFACE_API_KEY')
        if not hf_key:
            logger.warning("HUGGINGFACE_API_KEY not set - Llama2 and Mistral may not work")
    
    if config.enable_gpt35:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            logger.warning("OPENAI_API_KEY not set - GPT-3.5-turbo may not work")
    
    if config.enable_mistral:
        mistral_key = os.environ.get('MISTRAL_API_KEY')
        if not mistral_key:
            logger.warning("MISTRAL_API_KEY not set - Mistral may not work")
    
    # Print evaluation plan
    print("🚀 UNIFIED FALSE MEMORY EVALUATION")
    print("=" * 60)
    print(f"📊 Benchmarks: {', '.join(args.benchmarks)}")
    print(f"🔢 Samples per benchmark: {config.samples_per_benchmark}")
    print(f"🤖 Models to evaluate:")
    if config.enable_tmm:
        print("   • TMM (Truth-Maintained Memory)")
    if config.enable_llama2:
        print(f"   • Llama2-{config.llama2_size}")
    if config.enable_mistral:
        print(f"   • Mistral-{config.mistral_size}")
    if config.enable_gpt35:
        print("   • GPT-3.5-turbo")
    print(f"📁 Output: {config.output_path}")
    print("=" * 60)
    print()
    
    try:
        # Initialize evaluator
        logger.info("Initializing unified evaluator...")
        evaluator = UnifiedFalseMemoryEvaluator(config)
        
        # Run evaluation
        logger.info("Starting unified evaluation...")
        results = evaluator.evaluate_all_models(args.benchmarks)
        
        # Save results
        logger.info("Saving results...")
        evaluator.save_results(results)
        
        # Print summary
        print("\n🎉 UNIFIED EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Results saved to: {config.output_path}")
        print(f"📄 Summary saved to: {Path(config.output_path).with_suffix('.summary.txt')}")
        print()
        
        # Print key findings
        if 'research_summary' in results:
            research = results['research_summary']
            print("🔬 RESEARCH FINDINGS:")
            for finding in research.get('key_findings', []):
                print(f"   • {finding}")
            print()
        
        # Print model rankings
        if 'cross_model_comparison' in results:
            comparison = results['cross_model_comparison']
            if 'model_rankings' in comparison:
                rankings = comparison['model_rankings']
                print("🏆 MODEL RANKINGS:")
                print(f"   • FMR (False Memory Rate - lower is better): {', '.join(rankings.get('fmr', []))}")
                print(f"   • MEL (Memory Edit Latency - lower is better): {', '.join(rankings.get('mel', []))}")
                print(f"   • DAR (Disturbance Adaptation Rate - higher is better): {', '.join(rankings.get('dar', []))}")
                print()
        
        print("🎯 RESEARCH INTEGRITY: All models evaluated using identical methodology")
        print("   for fair comparison and reproducible results.")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Unified evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
