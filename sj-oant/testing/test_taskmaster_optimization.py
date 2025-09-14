#!/usr/bin/env python3
"""
Test Taskmaster optimization with enhanced task completion indicators
"""

import json
import logging
import os
import sys
import random
from typing import Dict, Any, List

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation_frameworks'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unified_evaluator import UnifiedOfficialEvaluator
from tmm_pipeline import TMMPipelineFixed as TMMPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_taskmaster_optimization():
    """Test Taskmaster optimization with enhanced task completion indicators."""
    
    print("🎯 Testing Taskmaster Optimization - Enhanced Task Completion")
    print("=" * 60)
    
    # Initialize TMM pipeline
    api_key = "AIzaSyB9Sn7qyZ23FQg6kJ3gOJjayzxCXGuTe_4"
    tmm_pipeline = TMMPipeline(api_key)
    
    # Initialize evaluator
    unified_evaluator = UnifiedOfficialEvaluator()
    
    # Load 10 Taskmaster samples
    print("📊 Loading 10 Taskmaster samples...")
    try:
        # Load Taskmaster data directly
        with open("../data/taskmaster/restaurant-search.json", 'r') as f:
            data = json.load(f)
        
        # Get 10 random samples
        selected_samples = random.sample(data, min(10, len(data)))
        
        print(f"✅ Selected {len(selected_samples)} Taskmaster samples")
        
        # Process samples
        tmm_predictions = []
        for i, sample in enumerate(selected_samples):
            print(f"\n🔄 Processing Taskmaster sample {i+1}/10: {sample.get('conversation_id', 'unknown')}")
            
            try:
                user_turns = []
                system_turns = []
                
                for utterance in sample["utterances"]:
                    if utterance["speaker"] == "USER":
                        user_turns.append(utterance["text"])
                    else:
                        system_turns.append(utterance["text"])
                
                print(f"   📝 Found {len(user_turns)} user turns, {len(system_turns)} system turns")
                
                # Reset memory for each dialogue
                tmm_pipeline.reset_memory()
                
                # Generate responses for first 4 user turns (to save time but test completion)
                responses = []
                for k, user_turn in enumerate(user_turns[:4]):
                    print(f"   🔄 Processing turn {k+1}: {user_turn[:50]}...")
                    response = tmm_pipeline.process(user_turn)
                    responses.append(response)
                    print(f"   ✅ Generated: {response[:50]}...")
                    
                    # Check for explicit completion indicators
                    completion_indicators = ["successfully", "completed", "confirmed", "booked", "reserved", "done", "accomplished", "finalized", "processed", "achieved"]
                    if any(indicator in response.lower() for indicator in completion_indicators):
                        print(f"   🎯 COMPLETION DETECTED in response: {response[:100]}...")
                
                tmm_predictions.append({
                    "dialogue_id": sample.get("conversation_id", f"taskmaster_{i}"),
                    "user_turns": user_turns[:4],
                    "responses": responses,
                    "system_turns": system_turns[:4] if len(system_turns) >= 4 else system_turns
                })
                
                print(f"   ✅ Successfully processed Taskmaster sample {i+1}")
                
            except Exception as e:
                print(f"   ❌ Error processing sample {i+1}: {e}")
                continue
        
        print(f"\n✅ Successfully processed {len(tmm_predictions)} Taskmaster samples")
        
        # Run evaluation
        print("🔬 Running Taskmaster evaluation...")
        try:
            results = unified_evaluator.evaluate_benchmark("taskmaster", tmm_predictions)
            print("✅ Taskmaster evaluation completed successfully!")
            
            # Print results
            print("\n📊 TASKMASTER EVALUATION RESULTS:")
            print("-" * 50)
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
            else:
                # BLEU Score
                if "bleu" in results:
                    bleu_data = results["bleu"]
                    bleu_score = bleu_data.get("bleu", 0)
                    print(f"📝 BLEU SCORE: {bleu_score:.2f}%")
                
                # ROUGE Score
                if "rouge" in results:
                    rouge_data = results["rouge"]
                    rouge_score = rouge_data.get("rouge", 0)
                    print(f"📊 ROUGE SCORE: {rouge_score:.2f}%")
                
                # Semantic Similarity
                if "semantic_similarity" in results:
                    semantic_data = results["semantic_similarity"]
                    semantic_score = semantic_data.get("semantic_similarity", 0)
                    print(f"🧠 SEMANTIC SIMILARITY: {semantic_score:.2f}%")
                
                # Task Completion
                if "task_completion" in results:
                    completion_data = results["task_completion"]
                    completion_score = completion_data.get("task_completion", 0)
                    print(f"✅ TASK COMPLETION: {completion_score:.2f}% (CRITICAL METRIC)")
                    
                    if completion_score > 15:
                        print("   🎉 SIGNIFICANT IMPROVEMENT in task completion!")
                    elif completion_score > 10:
                        print("   🔥 Good improvement in task completion!")
                    else:
                        print("   ⚠️ Still needs improvement in task completion")
            
            # Save results
            os.makedirs("results", exist_ok=True)
            results_file = "results/taskmaster_optimization_test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "test_type": "taskmaster_optimization_test",
                    "num_samples": len(tmm_predictions),
                    "results": results
                }, f, indent=4)
            
            print(f"\n💾 Results saved to: {results_file}")
            print("✅ Taskmaster optimization test completed successfully!")
            
        except Exception as e:
            print(f"❌ Taskmaster evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Failed to load Taskmaster data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_taskmaster_optimization()
