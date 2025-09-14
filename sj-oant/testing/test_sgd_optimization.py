#!/usr/bin/env python3
"""
Test SGD optimization with enhanced success indicators
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

def test_sgd_optimization():
    """Test SGD optimization with enhanced success indicators."""
    
    print("🎯 Testing SGD Optimization - Enhanced Success Indicators")
    print("=" * 60)
    
    # Initialize TMM pipeline
    api_key = "AIzaSyB9Sn7qyZ23FQg6kJ3gOJjayzxCXGuTe_4"
    tmm_pipeline = TMMPipeline(api_key)
    
    # Initialize evaluator
    unified_evaluator = UnifiedOfficialEvaluator()
    
    # Load 10 SGD samples
    print("📊 Loading 10 SGD samples...")
    try:
        # Load SGD data directly
        all_dialogues = []
        sgd_data_path = "../data/sgd"  # Corrected path
        
        for root, _, files in os.walk(sgd_data_path):
            for file in files:
                if file.endswith(".json") and "dialogues" in file:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            dialogues = json.load(f)
                        all_dialogues.extend(dialogues)
                    except Exception as e:
                        logger.error(f"Error loading SGD file {file_path}: {e}")
        
        if not all_dialogues:
            print("❌ No SGD dialogues found")
            return
        
        # Get 10 random samples
        selected_samples = random.sample(all_dialogues, min(10, len(all_dialogues)))
        
        print(f"✅ Selected {len(selected_samples)} SGD samples")
        
        # Process samples
        tmm_predictions = []
        for i, sample in enumerate(selected_samples):
            print(f"\n🔄 Processing SGD sample {i+1}/10: {sample.get('dialogue_id', 'unknown')}")
            
            try:
                user_turns = []
                system_turns = []
                
                for turn in sample["turns"]:
                    if turn["speaker"] == "USER":
                        user_turns.append(turn["utterance"])
                    elif turn["speaker"] == "SYSTEM":
                        system_turns.append(turn["utterance"])
                
                print(f"   📝 Found {len(user_turns)} user turns, {len(system_turns)} system turns")
                
                # Reset memory for each dialogue
                tmm_pipeline.reset_memory()
                
                # Generate responses for first 3 user turns only (to save time)
                responses = []
                for k, user_turn in enumerate(user_turns[:3]):
                    print(f"   🔄 Processing turn {k+1}: {user_turn[:50]}...")
                    response = tmm_pipeline.process(user_turn)
                    responses.append(response)
                    print(f"   ✅ Generated: {response[:50]}...")
                
                tmm_predictions.append({
                    "dialogue_id": sample.get("dialogue_id", f"sgd_{i}"),
                    "user_turns": user_turns[:3],
                    "responses": responses,
                    "system_turns": system_turns[:3] if len(system_turns) >= 3 else system_turns
                })
                
                print(f"   ✅ Successfully processed SGD sample {i+1}")
                
            except Exception as e:
                print(f"   ❌ Error processing sample {i+1}: {e}")
                continue
        
        print(f"\n✅ Successfully processed {len(tmm_predictions)} SGD samples")
        
        # Run evaluation
        print("🔬 Running SGD evaluation...")
        try:
            results = unified_evaluator.evaluate_benchmark("sgd", tmm_predictions)
            print("✅ SGD evaluation completed successfully!")
            
            # Print results
            print("\n📊 SGD EVALUATION RESULTS:")
            print("-" * 50)
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
            else:
                # Intent Accuracy
                if "intent_accuracy" in results:
                    intent_data = results["intent_accuracy"]
                    total_accuracy = intent_data.get("total", 0)
                    print(f"🎯 INTENT ACCURACY: {total_accuracy:.2f}%")
                
                # Slot F1 Score
                if "slot_f1" in results:
                    slot_data = results["slot_f1"]
                    total_f1 = slot_data.get("total", 0)
                    print(f"🔧 SLOT F1 SCORE: {total_f1:.2f}%")
                
                # Success Rate
                if "success_rate" in results:
                    success_data = results["success_rate"]
                    total_success = success_data.get("total", 0)
                    print(f"✅ SUCCESS RATE: {total_success:.2f}%")
                
                # BLEU Score
                if "bleu" in results:
                    bleu_data = results["bleu"]
                    bleu_score = bleu_data.get("bleu", 0)
                    print(f"📝 BLEU SCORE: {bleu_score:.2f}%")
            
            # Save results
            os.makedirs("results", exist_ok=True)
            results_file = "results/sgd_optimization_test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "test_type": "sgd_optimization_test",
                    "num_samples": len(tmm_predictions),
                    "results": results
                }, f, indent=4)
            
            print(f"\n💾 Results saved to: {results_file}")
            print("✅ SGD optimization test completed successfully!")
            
        except Exception as e:
            print(f"❌ SGD evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Failed to load SGD data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sgd_optimization()
