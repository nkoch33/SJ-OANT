#!/usr/bin/env python3
"""
🚀 TMM SYSTEM PRE-FLIGHT CHECK

This script performs a comprehensive readiness check before running evaluations:
1. Environment setup verification
2. API key validation
3. Dataset availability 
4. System component integration
5. Pipeline connectivity test
6. Results directory setup

Run this before any evaluation to ensure everything is ready.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging for pre-flight check."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_environment():
    """Check environment variables and API keys."""
    print("🔑 CHECKING ENVIRONMENT SETUP...")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("📝 Create .env file with your API key:")
        print("   cp env.example .env")
        print("   # Then edit .env and add your GOOGLE_API_KEY")
        return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in .env file")
            return False
        
        if api_key == "your_google_api_key_here":
            print("❌ GOOGLE_API_KEY not set properly (still template value)")
            return False
        
        # Validate API key format (basic check)
        if not api_key.startswith("AIza") or len(api_key) < 35:
            print("⚠️  API key format looks suspicious")
            print(f"   Length: {len(api_key)}, Starts with: {api_key[:6]}...")
        
        print(f"✅ API key loaded: {api_key[:10]}...{api_key[-4:]}")
        return True
        
    except ImportError:
        print("❌ python-dotenv not installed")
        return False
    except Exception as e:
        print(f"❌ Error loading environment: {e}")
        return False

def check_dataset():
    """Check dataset availability and loading."""
    print("\n📊 CHECKING DATASET AVAILABILITY...")
    
    try:
        from datasets import load_dataset
        
        # Test SQuAD loading
        print("   Loading SQuAD dataset (this may take a moment)...")
        ds = load_dataset("squad")
        
        print(f"✅ SQuAD loaded successfully:")
        print(f"   - Train examples: {len(ds['train'])}")
        print(f"   - Validation examples: {len(ds['validation'])}")
        print(f"   - Columns: {list(ds['train'].column_names)}")
        
        # Test sample data
        sample = ds['validation'][0]
        print(f"   - Sample context length: {len(sample['context'].split())} words")
        print(f"   - Sample question: {sample['question'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def check_components():
    """Check all TMM system components."""
    print("\n🧠 CHECKING TMM SYSTEM COMPONENTS...")
    
    try:
        # Test core imports
        from core.types import MemoryRecord, MemoryTier
        from core.ports import BaseMemoryStore, BaseVerifier
        print("✅ Core types and ports imported")
        
        # Test memory system
        from memory.typed_store import InMemoryStore, MemoryState
        memory_store = InMemoryStore()
        print("✅ Memory store created")
        
        # Test agents
        from agents.planner import StrategicPlanner
        from agents.arbiter import ContextArbiter  
        from agents.responder import Responder
        from agents.writer_editor import WriterEditor
        print("✅ All agents imported")
        
        # Test truth verification
        from truth.verifier import create_verifier
        verifier = create_verifier("rule_based")
        print("✅ Truth verifier created")
        
        # Test baselines
        from baselines.simple_systems import create_baseline_systems
        print("✅ Baseline systems available")
        
        # Test evaluation
        from evaluation.squad_eval import SQuADEvaluator
        evaluator = SQuADEvaluator()
        print("✅ Evaluation system ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Component check failed: {e}")
        return False

def check_pipeline():
    """Test end-to-end pipeline creation."""
    print("\n🔄 CHECKING TMM PIPELINE INTEGRATION...")
    
    try:
        from tmm_pipeline import TMMPipeline
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Create mock LLM for testing (no API call)
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "Mock response for testing"
                return MockResponse()
        
        mock_llm = MockLLM()
        pipeline = TMMPipeline(mock_llm)
        
        print("✅ TMM pipeline created successfully")
        print(f"   - Components: planner, filter, verifier, writer, responder")
        print(f"   - Memory store initialized")
        print(f"   - LangGraph compiled")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration failed: {e}")
        return False

def check_results_directory():
    """Check and setup results directory."""
    print("\n📁 CHECKING RESULTS DIRECTORY...")
    
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir()
        print("✅ Created results directory")
    else:
        print("✅ Results directory exists")
    
    # Check write permissions
    test_file = results_dir / "test_write.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Write permissions confirmed")
        return True
    except Exception as e:
        print(f"❌ Cannot write to results directory: {e}")
        return False

def test_api_connection():
    """Test actual API connection with minimal call."""
    print("\n🌐 TESTING API CONNECTION...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )
        
        # Minimal test call
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("Say 'API test successful' and nothing else.")
        
        print("   Making test API call...")
        response = llm.invoke(prompt.format_messages())
        
        print(f"✅ API connection successful")
        print(f"   Response: {response.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("   Check your API key and internet connection")
        return False

def main():
    """Run complete pre-flight check."""
    print("🚀" + "="*60)
    print("🚀 TMM SYSTEM PRE-FLIGHT CHECK")
    print("🚀" + "="*60)
    
    setup_logging()
    
    checks = [
        ("Environment & API Keys", check_environment),
        ("Dataset Loading", check_dataset),
        ("System Components", check_components),
        ("Pipeline Integration", check_pipeline),
        ("Results Directory", check_results_directory),
        ("API Connection", test_api_connection)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{'='*60}")
        try:
            if check_func():
                passed += 1
            else:
                print(f"❌ {check_name} failed")
        except Exception as e:
            print(f"❌ {check_name} crashed: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎯 PRE-FLIGHT RESULTS: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL SYSTEMS GO! Ready for evaluation.")
        print("\n🚀 Next steps:")
        print("   python runners/eval_squad2.py --api-key YOUR_API_KEY --limit 5")
        print("   python runners/eval_multiturn.py --api-key YOUR_API_KEY --scenarios 5")
        return True
    else:
        print("❌ Some checks failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
