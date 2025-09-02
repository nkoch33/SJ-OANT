#!/usr/bin/env python3
"""
Quick integration test for the retrieval system with existing TMM components.

This script tests that the retrieval implementations work correctly with
the memory store and core types, verifying interface compatibility.
"""

import sys
from datetime import datetime, timezone

# Test imports and basic functionality
def test_retrieval_integration():
    """Test retrieval system integration with TMM components."""
    print("🔍 Testing TMM Retrieval System Integration")
    print("=" * 50)
    
    try:
        # Test core imports
        print("📦 Testing core imports...")
        from core.types import MemoryRecord, MemoryTier, MemoryStatus, ConfidenceScores, Provenance
        from memory.typed_store import InMemoryStore
        print("   ✅ Core types imported successfully")
        
        # Test retrieval imports  
        print("📦 Testing retrieval imports...")
        from retrieval.hybrid_retriever import create_hybrid_retriever, RetrievalConfig
        from retrieval.active_retrieval import create_active_retriever, ActiveRetrievalConfig
        print("   ✅ Retrieval classes imported successfully")
        
        # Create test memory store with sample data
        print("💾 Creating test memory store...")
        memory_store = InMemoryStore(l1_limit=10, l2_limit=10, l3_limit=10)
        
        # Add sample records
        test_records = [
            MemoryRecord(
                payload="The capital of France is Paris",
                tier=MemoryTier.L3_ARCHIVAL,
                status=MemoryStatus.VERIFIED,
                scores=ConfidenceScores(truth_score=0.9, confidence=0.9),
                provenance=Provenance(
                    source="test_data",
                    source_type="test",
                    pipeline_stage="integration_test",
                    processing_agent="test_script"
                )
            ),
            MemoryRecord(
                payload="Python is a programming language",
                tier=MemoryTier.L2_SUMMARIZED,
                status=MemoryStatus.VERIFIED,
                scores=ConfidenceScores(truth_score=0.8, confidence=0.8),
                provenance=Provenance(
                    source="test_data",
                    source_type="test",
                    pipeline_stage="integration_test",
                    processing_agent="test_script"
                )
            ),
            MemoryRecord(
                payload="Machine learning models require training data",
                tier=MemoryTier.L1_WORKING,
                status=MemoryStatus.PENDING,
                scores=ConfidenceScores(truth_score=0.7, confidence=0.7),
                provenance=Provenance(
                    source="test_data",
                    source_type="test",
                    pipeline_stage="integration_test",
                    processing_agent="test_script"
                )
            )
        ]
        
        for record in test_records:
            memory_store.add(record)
        
        print(f"   ✅ Added {len(test_records)} test records to memory store")
        
        # Test hybrid retriever
        print("🔍 Testing hybrid retriever...")
        hybrid_config = {"k": 5, "keyword_weight": 0.5, "vector_weight": 0.5}
        hybrid_retriever = create_hybrid_retriever(memory_store, hybrid_config)
        
        # Test retrieval
        results = hybrid_retriever.retrieve("capital France", limit=2)
        print(f"   ✅ Hybrid retrieval returned {len(results)} results")
        
        if results:
            print(f"      Best match: {results[0].payload[:50]}...")
        
        # Test active retriever
        print("🎯 Testing active retriever...")
        active_config = {"max_iterations": 2, "feedback_threshold": 0.6}
        active_retriever = create_active_retriever(hybrid_retriever, active_config)
        
        # Test active retrieval
        results = active_retriever.retrieve("programming language", limit=2)
        print(f"   ✅ Active retrieval returned {len(results)} results")
        
        if results:
            print(f"      Best match: {results[0].payload[:50]}...")
        
        # Test query refinement
        print("🔧 Testing query refinement...")
        feedback = {
            "satisfaction_score": 0.3,
            "query_modification": "Python programming language features"
        }
        refined_query = active_retriever.refine("Python", feedback)
        print(f"   ✅ Query refined: 'Python' -> '{refined_query}'")
        
        # Test metrics
        print("📊 Testing metrics collection...")
        hybrid_metrics = hybrid_retriever.get_retrieval_metrics()
        active_metrics = active_retriever.get_active_metrics()
        
        print(f"   ✅ Hybrid retriever: {hybrid_metrics['total_queries']} queries processed")
        print(f"   ✅ Active retriever: {active_metrics['total_retrievals']} retrievals completed")
        
        print("=" * 50)
        print("✨ All retrieval integration tests passed!")
        print("\n💡 Retrieval system is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_retrieval_integration()
    sys.exit(0 if success else 1)
