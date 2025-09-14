#!/usr/bin/env python3
"""
MultiDoGO Optimization Test - Test TMM performance on MultiDoGO dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmm_pipeline import TMMPipelineFixed
from evaluation_frameworks.multidogo.official_evaluator import OfficialMultiDoGOEvaluator
import json
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multidogo_optimization():
    """Test TMM optimization on MultiDoGO dataset."""
    
    # Initialize TMM pipeline
    api_key = "AIzaSyB9Sn7qyZ23FQg6kJ3gOJjayzxCXGuTe_4"
    tmm_pipeline = TMMPipelineFixed(api_key)
    
    # Load MultiDoGO data (TSV format)
    all_dialogues = []
    multidogo_data_path = "../data/multidogo"
    
    import pandas as pd
    
    for root, _, files in os.walk(multidogo_data_path):
        for file in files:
            if file.endswith(".tsv") and "annotated" in file:
                file_path = os.path.join(root, file)
                try:
                    # Load TSV file
                    df = pd.read_csv(file_path, sep='\t')
                    
                    # Group by conversation ID
                    for conv_id in df['conversationId'].unique():
                        conv_data = df[df['conversationId'] == conv_id]
                        
                        # Extract turns
                        turns = []
                        for _, row in conv_data.iterrows():
                            turns.append({
                                "turn_number": row['turnNumber'],
                                "utterance": row['utterance'],
                                "intent": row['intent'],
                                "slot_labels": row['slot-labels']
                            })
                        
                        all_dialogues.append({
                            "dialogue_id": conv_id,
                            "turns": turns
                        })
                        
                except Exception as e:
                    logger.error(f"Error loading MultiDoGO file {file_path}: {e}")
    
    if not all_dialogues:
        print("❌ No MultiDoGO dialogues found")
        return
    
    # Get 10 random samples for testing
    selected_samples = random.sample(all_dialogues, min(10, len(all_dialogues)))
    
    print("🎯 Testing MultiDoGO Optimization - Enhanced Task Completion")
    print("=" * 60)
    print(f"📊 Loading {len(selected_samples)} MultiDoGO samples...")
    
    predictions = []
    
    for i, sample in enumerate(selected_samples):
        print(f"\n🔄 Processing MultiDoGO sample {i+1}/{len(selected_samples)}: {sample.get('dialogue_id', 'unknown')}")
        
        try:
            # Extract user turns from MultiDoGO format
            user_turns = []
            system_turns = []
            
            # MultiDoGO TSV format - all turns are user turns
            if "turns" in sample:
                for turn in sample["turns"]:
                    utterance = turn.get("utterance", "")
                    if utterance.strip():
                        user_turns.append(utterance)
                        # Create dummy system response for evaluation
                        system_turns.append("I understand your request. Let me help you with that.")
            
            print(f"   📝 Found {len(user_turns)} user turns, {len(system_turns)} system turns")
            
            if not user_turns or not system_turns:
                print(f"   ⚠️ Skipping sample with insufficient turns")
                continue
            
            # Reset memory for each dialogue
            tmm_pipeline.reset_memory()
            
            # Process first 3 turns
            responses = []
            for turn_idx in range(min(3, len(user_turns))):
                user_input = user_turns[turn_idx]
                print(f"   🔄 Processing turn {turn_idx + 1}: {user_input[:50]}...")
                
                # Process through TMM
                response = tmm_pipeline.process(user_input)
                responses.append(response)
                print(f"   ✅ Generated: {response[:50]}...")
            
            # Store prediction
            predictions.append({
                "dialogue_id": sample.get("dialogue_id", f"multidogo_{i}"),
                "user_turns": user_turns[:3],
                "system_turns": system_turns[:3],
                "responses": responses
            })
            
            print(f"   ✅ Successfully processed MultiDoGO sample {i+1}")
            
        except Exception as e:
            print(f"   ❌ Error processing sample: {e}")
            continue
    
    if not predictions:
        print("❌ No valid predictions generated")
        return
    
    print(f"\n✅ Successfully processed {len(predictions)} MultiDoGO samples")
    
    # Run evaluation
    print("🔬 Running MultiDoGO evaluation...")
    evaluator = OfficialMultiDoGOEvaluator()
    results = evaluator.evaluate(predictions)
    
    print("✅ MultiDoGO evaluation completed successfully!")
    
    # Display results
    print("\n📊 MULTIDOGO EVALUATION RESULTS:")
    print("-" * 50)
    print(f"🎯 INTENT CLASSIFICATION ACCURACY: {results.get('intent_accuracy', {}).get('total', 0):.2f}%")
    print(f"🔧 SLOT FILLING F1 SCORE: {results.get('slot_f1', {}).get('total', 0):.2f}%")
    print(f"🌐 DOMAIN ADAPTATION SCORE: {results.get('domain_adaptation', {}).get('total', 0):.2f}%")
    print(f"📝 RESPONSE QUALITY SCORE: {results.get('response_quality', {}).get('total', 0):.2f}%")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_file = "results/multidogo_optimization_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_type": "multidogo_optimization_test",
            "num_samples": len(predictions),
            "results": results,
            "benchmark": "multidogo",
            "evaluation_framework": "official"
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("✅ MultiDoGO optimization test completed successfully!")

if __name__ == "__main__":
    test_multidogo_optimization()
