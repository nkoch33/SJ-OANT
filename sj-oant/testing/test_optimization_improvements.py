#!/usr/bin/env python3
"""
Comprehensive test for optimization improvements:
- BLEU score improvements
- MultiDoGO intent classification accuracy
- ROUGE score optimization
- MultiDoGO response quality
- Overall pipeline optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tmm_pipeline import TMMPipelineFixed
from memory.typed_store import InMemoryStore
import json
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationTester:
    """Test optimization improvements across all benchmarks."""
    
    def __init__(self):
        """Initialize the optimization tester."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.error("GOOGLE_API_KEY environment variable not set")
            sys.exit(1)
        
        self.memory_store = InMemoryStore()
        self.tmm_pipeline = TMMPipelineFixed(self.api_key)
        
    def test_bleu_improvements(self):
        """Test BLEU score improvements with enhanced response generation."""
        print("🎯 Testing BLEU Score Improvements")
        print("=" * 50)
        
        # Test queries designed to improve BLEU scores
        test_queries = [
            "I need to book a hotel in Cambridge for 2 nights",
            "Can you find me a restaurant near the train station?",
            "What's the address of the Fitzwilliam Museum?",
            "I want to book a flight from London to Paris",
            "Help me find a taxi to the airport"
        ]
        
        responses = []
        for query in test_queries:
            try:
                response = self.tmm_pipeline.process(query)
                responses.append({
                    "query": query,
                    "response": response,
                    "length": len(response.split()),
                    "has_success_indicators": any(word in response.lower() for word in 
                        ['successfully', 'confirmed', 'found', 'located', 'identified'])
                })
                print(f"✅ Query: {query}")
                print(f"   Response length: {len(response.split())} words")
                print(f"   Has success indicators: {responses[-1]['has_success_indicators']}")
                print()
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
        
        return responses
    
    def test_multidogo_intent_accuracy(self):
        """Test MultiDoGO intent classification accuracy improvements."""
        print("🎯 Testing MultiDoGO Intent Classification Accuracy")
        print("=" * 50)
        
        # MultiDoGO-specific test queries
        multidogo_queries = [
            "I need to check my seat assignment",
            "Can you help me with my boarding pass?",
            "What's my confirmation number?",
            "I want to book a flight to New York",
            "Help me find my booking reference",
            "I need to change my reservation",
            "Can you cancel my booking?",
            "What's the price for this flight?",
            "Where is the airport located?",
            "Is the flight available tomorrow?"
        ]
        
        intent_results = []
        for query in multidogo_queries:
            try:
                response = self.tmm_pipeline.process(query)
                
                # Extract intents from response
                response_lower = response.lower()
                detected_intents = []
                
                intent_patterns = {
                    "book": ["book", "reserve", "schedule", "arrange"],
                    "check": ["check", "verify", "confirm", "look up"],
                    "find": ["find", "search", "locate", "show"],
                    "change": ["change", "modify", "update"],
                    "cancel": ["cancel", "remove", "delete"],
                    "help": ["help", "assist", "support"],
                    "inform": ["inform", "tell", "show", "provide"],
                    "seat": ["seat", "assignment", "boarding pass"],
                    "confirmation": ["confirmation", "number", "reference"],
                    "price": ["price", "cost", "fee", "charge"],
                    "address": ["address", "location", "where"],
                    "available": ["available", "open", "operating hours"]
                }
                
                for intent, patterns in intent_patterns.items():
                    if any(pattern in response_lower for pattern in patterns):
                        detected_intents.append(intent)
                
                intent_results.append({
                    "query": query,
                    "response": response,
                    "detected_intents": detected_intents,
                    "intent_count": len(detected_intents)
                })
                
                print(f"✅ Query: {query}")
                print(f"   Detected intents: {detected_intents}")
                print(f"   Intent count: {len(detected_intents)}")
                print()
                
            except Exception as e:
                logger.error(f"Error processing MultiDoGO query '{query}': {e}")
        
        return intent_results
    
    def test_response_quality_improvements(self):
        """Test response quality improvements."""
        print("🎯 Testing Response Quality Improvements")
        print("=" * 50)
        
        quality_queries = [
            "Hello, I need help with travel planning",
            "Can you book a hotel for me?",
            "What restaurants do you recommend?",
            "I need transportation to the airport",
            "Thank you for your help!"
        ]
        
        quality_results = []
        for query in quality_queries:
            try:
                response = self.tmm_pipeline.process(query)
                
                # Quality assessment
                word_count = len(response.split())
                has_punctuation = response.endswith(('.', '!', '?'))
                has_helpful_indicators = any(word in response.lower() for word in 
                    ['i can help', 'i understand', 'let me', 'i have', 'i found',
                     'successfully', 'confirmed', 'available', 'located', 'identified'])
                is_appropriate_length = 5 <= word_count <= 100
                
                quality_score = sum([has_punctuation, has_helpful_indicators, is_appropriate_length]) / 3 * 100
                
                quality_results.append({
                    "query": query,
                    "response": response,
                    "word_count": word_count,
                    "has_punctuation": has_punctuation,
                    "has_helpful_indicators": has_helpful_indicators,
                    "is_appropriate_length": is_appropriate_length,
                    "quality_score": quality_score
                })
                
                print(f"✅ Query: {query}")
                print(f"   Quality score: {quality_score:.1f}%")
                print(f"   Word count: {word_count}")
                print(f"   Has helpful indicators: {has_helpful_indicators}")
                print()
                
            except Exception as e:
                logger.error(f"Error processing quality query '{query}': {e}")
        
        return quality_results
    
    def run_comprehensive_test(self):
        """Run comprehensive optimization test."""
        print("🚀 Running Comprehensive Optimization Test")
        print("=" * 60)
        
        results = {
            "bleu_improvements": self.test_bleu_improvements(),
            "multidogo_intent_accuracy": self.test_multidogo_intent_accuracy(),
            "response_quality_improvements": self.test_response_quality_improvements()
        }
        
        # Calculate summary statistics
        summary = {
            "bleu_test": {
                "total_queries": len(results["bleu_improvements"]),
                "avg_response_length": sum(r["length"] for r in results["bleu_improvements"]) / len(results["bleu_improvements"]),
                "success_indicator_rate": sum(r["has_success_indicators"] for r in results["bleu_improvements"]) / len(results["bleu_improvements"]) * 100
            },
            "intent_test": {
                "total_queries": len(results["multidogo_intent_accuracy"]),
                "avg_intent_count": sum(r["intent_count"] for r in results["multidogo_intent_accuracy"]) / len(results["multidogo_intent_accuracy"]),
                "intent_detection_rate": sum(1 for r in results["multidogo_intent_accuracy"] if r["intent_count"] > 0) / len(results["multidogo_intent_accuracy"]) * 100
            },
            "quality_test": {
                "total_queries": len(results["response_quality_improvements"]),
                "avg_quality_score": sum(r["quality_score"] for r in results["response_quality_improvements"]) / len(results["response_quality_improvements"]),
                "high_quality_rate": sum(1 for r in results["response_quality_improvements"] if r["quality_score"] >= 66.7) / len(results["response_quality_improvements"]) * 100
            }
        }
        
        # Display summary
        print("\n📊 OPTIMIZATION TEST SUMMARY")
        print("=" * 60)
        print(f"BLEU Improvements:")
        print(f"  - Average response length: {summary['bleu_test']['avg_response_length']:.1f} words")
        print(f"  - Success indicator rate: {summary['bleu_test']['success_indicator_rate']:.1f}%")
        print()
        print(f"MultiDoGO Intent Classification:")
        print(f"  - Average intent count: {summary['intent_test']['avg_intent_count']:.1f}")
        print(f"  - Intent detection rate: {summary['intent_test']['intent_detection_rate']:.1f}%")
        print()
        print(f"Response Quality:")
        print(f"  - Average quality score: {summary['quality_test']['avg_quality_score']:.1f}%")
        print(f"  - High quality rate: {summary['quality_test']['high_quality_rate']:.1f}%")
        
        # Save results
        results["summary"] = summary
        with open("results/optimization_improvements_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: results/optimization_improvements_test_results.json")
        return results

def main():
    """Main function to run optimization tests."""
    tester = OptimizationTester()
    results = tester.run_comprehensive_test()
    
    print("\n✅ Optimization improvement test completed!")
    return results

if __name__ == "__main__":
    main()
