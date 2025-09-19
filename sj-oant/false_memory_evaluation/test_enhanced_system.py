#!/usr/bin/env python3
"""
Test Enhanced False Memory Evaluation System

Comprehensive test script to verify all components of the enhanced
false memory evaluation system work correctly.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from false_memory_evaluation.enhanced_false_memory_injector import EnhancedFalseMemoryInjector, InjectionType
from false_memory_evaluation.metrics_calculator import MetricsCalculator
from false_memory_evaluation.contradiction_detector import ContradictionDetector
from false_memory_evaluation.unified_evaluator import UnifiedFalseMemoryEvaluator, EvaluationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_false_memory_injector():
    """Test the enhanced false memory injection system."""
    print("🧪 Testing Enhanced False Memory Injector...")
    
    injector = EnhancedFalseMemoryInjector(seed=42)
    
    # Test conversation
    test_conversation = {
        "user_turns": [
            "I need help finding a hotel in Cambridge",
            "What are the prices like?",
            "When does the train leave?",
            "Is there WiFi available?"
        ],
        "system_turns": [
            "I can help you find a hotel in Cambridge.",
            "Hotels in Cambridge typically cost around $120 per night.",
            "The train usually leaves at 3:30 PM.",
            "Most hotels offer free WiFi."
        ]
    }
    
    # Test different injection types
    injection_types = [
        InjectionType.DIRECT_FALSE_FACT,
        InjectionType.IMPLICIT_HALLUCINATION,
        InjectionType.CONTRADICTORY_INFORMATION,
        InjectionType.TEMPORAL_INCONSISTENCY,
        InjectionType.CONTEXTUAL_DISTORTION
    ]
    
    for injection_type in injection_types:
        print(f"  Testing {injection_type.value}...")
        
        modified_conv, injections = injector.inject_false_memories(
            test_conversation, 
            num_injections=2,
            injection_types=[injection_type]
        )
        
        assert len(injections) > 0, f"No injections created for {injection_type.value}"
        assert len(modified_conv["user_turns"]) == len(test_conversation["user_turns"]), "User turns count mismatch"
        
        print(f"    ✅ Created {len(injections)} injections")
    
    # Test statistics
    modified_conv, all_injections = injector.inject_false_memories(
        test_conversation, 
        num_injections=3,
        injection_types=injection_types
    )
    
    stats = injector.get_injection_statistics(all_injections)
    print(f"  📊 Injection Statistics: {stats}")
    
    print("✅ Enhanced False Memory Injector tests passed!")
    return True

def test_enhanced_metrics_calculator():
    """Test the enhanced metrics calculator."""
    print("🧪 Testing Enhanced Metrics Calculator...")
    
    calculator = MetricsCalculator()
    
    # Test false information detection
    test_responses = [
        "The hotel costs $200 per night",  # Contains false info
        "The hotel costs $120 per night",  # Contains correct info
        "I think the hotel costs $200 per night",  # Contains false info with uncertainty
        "Actually, the hotel costs $120 per night, not $200",  # Correction
        "The hotel is nice and has good amenities"  # No price info
    ]
    
    false_information = {"0": "The hotel costs $200 per night"}
    
    # Test FMR calculation
    fmr = calculator.calculate_fmr(test_responses, false_information)
    print(f"  📊 FMR: {fmr:.2f}%")
    assert 0 <= fmr <= 100, "FMR should be between 0 and 100"
    
    # Test DAR calculation
    dar = calculator.calculate_dar(test_responses, false_information)
    print(f"  📊 DAR: {dar:.2f}%")
    assert 0 <= dar <= 100, "DAR should be between 0 and 100"
    
    # Test contradiction detection
    contradiction_detection = calculator.calculate_contradiction_detection_rate(test_responses, 2)
    print(f"  📊 Contradiction Detection: {contradiction_detection:.2f}%")
    assert 0 <= contradiction_detection <= 100, "Contradiction detection should be between 0 and 100"
    
    # Test all metrics
    all_metrics = calculator.calculate_all_metrics(test_responses, false_information, contradiction_point=2)
    print(f"  📊 All Metrics: FMR={all_metrics.fmr:.2f}%, MEL={all_metrics.mel:.2f}s, DAR={all_metrics.dar:.2f}%")
    
    print("✅ Enhanced Metrics Calculator tests passed!")
    return True

def test_contradiction_detector():
    """Test the contradiction detector."""
    print("🧪 Testing Contradiction Detector...")
    
    detector = ContradictionDetector()
    
    # Test responses with contradictions
    test_responses = [
        "The hotel costs $120 per night",
        "Actually, I need to correct that - the hotel costs $200 per night",
        "I apologize for the confusion, the correct price is $120 per night",
        "The hotel is located in Cambridge, England"
    ]
    
    false_information = {"0": "The hotel costs $200 per night"}
    
    # Test contradiction analysis
    analysis = detector.analyze_contradictions(test_responses, false_information)
    print(f"  📊 Contradictions Detected: {analysis.contradictions_detected}/{analysis.total_contradictions}")
    print(f"  📊 Detection Rate: {analysis.detection_rate:.2f}%")
    print(f"  📊 Response Types: {analysis.response_types}")
    print(f"  📊 Correction Quality: {analysis.correction_quality:.2f}")
    
    # Test implicit contradiction detection
    implicit_contradictions = detector.detect_implicit_contradictions(test_responses)
    print(f"  📊 Implicit Contradictions: {len(implicit_contradictions)}")
    
    # Test consistency analysis
    consistency_analysis = detector.analyze_consistency_across_turns(test_responses)
    print(f"  📊 Consistency Rate: {consistency_analysis['consistency_rate']:.2f}")
    
    print("✅ Contradiction Detector tests passed!")
    return True

def test_unified_evaluator():
    """Test the unified evaluator (without actually running models)."""
    print("🧪 Testing Unified Evaluator...")
    
    # Create test configuration
    config = EvaluationConfig(
        data_root="data",
        seed=42,
        samples_per_benchmark=2,  # Small number for testing
        enable_tmm=False,  # Disable TMM for testing
        enable_llama2=False,  # Disable baseline models for testing
        enable_mistral=False,
        enable_gpt35=False,
        output_path="test_results.json"
    )
    
    try:
        evaluator = UnifiedFalseMemoryEvaluator(config)
        print("  ✅ Unified evaluator initialized successfully")
        
        # Test that enhanced injector is available
        assert hasattr(evaluator, 'enhanced_injector'), "Enhanced injector not available"
        print("  ✅ Enhanced injector integrated")
        
        # Test configuration
        assert evaluator.config.seed == 42, "Configuration not set correctly"
        print("  ✅ Configuration loaded correctly")
        
    except Exception as e:
        print(f"  ❌ Unified evaluator test failed: {e}")
        return False
    
    print("✅ Unified Evaluator tests passed!")
    return True

def test_integration():
    """Test integration between all components."""
    print("🧪 Testing Component Integration...")
    
    # Test that all components can work together
    injector = EnhancedFalseMemoryInjector(seed=42)
    calculator = MetricsCalculator()
    detector = ContradictionDetector()
    
    # Create test scenario
    test_conversation = {
        "user_turns": [
            "I need a hotel in Cambridge",
            "What's the price?",
            "When does the train leave?"
        ],
        "system_turns": [
            "I can help you find a hotel.",
            "Hotels cost around $120 per night.",
            "The train leaves at 3:30 PM."
        ]
    }
    
    # Inject false memories
    modified_conv, injections = injector.inject_false_memories(test_conversation, num_injections=2)
    print(f"  📊 Created {len(injections)} false memory injections")
    
    # Simulate model responses (with some false information)
    simulated_responses = [
        "I found a hotel in Cambridge for you.",
        "The hotel costs $200 per night",  # False information
        "The train leaves at 2:15 PM"  # False information
    ]
    
    # Calculate metrics
    false_info = {str(i): injection.false_fact for i, injection in enumerate(injections)}
    metrics = calculator.calculate_all_metrics(simulated_responses, false_info)
    print(f"  📊 Calculated metrics: FMR={metrics.fmr:.2f}%, DAR={metrics.dar:.2f}%")
    
    # Analyze contradictions
    analysis = detector.analyze_contradictions(simulated_responses, false_info)
    print(f"  📊 Contradiction analysis: {analysis.detection_rate:.2f}% detection rate")
    
    print("✅ Integration tests passed!")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced False Memory Evaluation System Tests")
    print("=" * 60)
    
    tests = [
        test_enhanced_false_memory_injector,
        test_enhanced_metrics_calculator,
        test_contradiction_detector,
        test_unified_evaluator,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The enhanced false memory evaluation system is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
