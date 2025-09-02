#!/usr/bin/env python3
"""
Test script to verify the evaluation system is ready.

This script tests that the TMM pipeline and baseline systems can be
instantiated and run basic evaluations.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from evaluation.fictionalqa_eval import FictionalQAEvaluator
        from baselines.simple_systems import create_baseline_systems
        from tmm_pipeline import TMMPipeline
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dataset_loading():
    """Test that FictionalQA dataset can be loaded."""
    print("\nTesting FictionalQA dataset loading...")
    
    try:
        from evaluation.fictionalqa_eval import FictionalQAEvaluator
        evaluator = FictionalQAEvaluator()
        
        # Load a small sample
        examples = evaluator.load_dataset(split="validation")
        print(f"✅ Loaded {len(examples)} FictionalQA examples")
        
        # Show first example
        if examples:
            example = examples[0]
            print(f"   Sample story: {example.story[:100]}...")
            print(f"   Sample question: {example.question}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_baseline_creation():
    """Test that baseline systems can be created (without API key)."""
    print("\nTesting baseline system creation...")
    
    try:
        from baselines.simple_systems import create_baseline_systems
        
        # Mock LLM for testing
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "Mock response"
                return MockResponse()
        
        mock_llm = MockLLM()
        baseline_systems = create_baseline_systems(mock_llm)
        
        print(f"✅ Created {len(baseline_systems)} baseline systems:")
        for name in baseline_systems.keys():
            print(f"   - {name}")
        
        return True
    except Exception as e:
        print(f"❌ Baseline creation failed: {e}")
        return False

def test_tmm_pipeline_creation():
    """Test TMM pipeline creation (without API key)."""
    print("\nTesting TMM pipeline creation...")
    
    try:
        from tmm_pipeline import TMMPipeline
        
        # Mock LLM for testing
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "Mock response"
                return MockResponse()
        
        mock_llm = MockLLM()
        pipeline = TMMPipeline(mock_llm)
        
        print("✅ TMM pipeline created successfully")
        return True
    except Exception as e:
        print(f"❌ TMM pipeline creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SJ-OANT EVALUATION SYSTEM READINESS TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_dataset_loading,
        test_baseline_creation,
        test_tmm_pipeline_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for evaluation.")
        print("\nTo run evaluation with API key:")
        print("python runners/eval_baselines.py --api-key YOUR_GOOGLE_API_KEY --limit 10")
    else:
        print("❌ Some tests failed. Please fix the issues before running evaluation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
