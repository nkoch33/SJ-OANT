"""
TMM Retrieval Package

This package contains retrieval implementations for the Truth-Maintained Memory system.

Key Components:
- hybrid_retriever: Combines keyword and vector search strategies
- active_retrieval: Context-aware retrieval with iterative refinement

The retrieval system provides flexible, high-performance memory access
with pluggable strategies and comprehensive performance monitoring.

Example Usage:
    from retrieval.hybrid_retriever import create_hybrid_retriever
    from retrieval.active_retrieval import create_active_retriever
    
    # Create hybrid retriever with custom config
    hybrid = create_hybrid_retriever(
        memory_store, 
        config={"k": 15, "keyword_weight": 0.4}
    )
    
    # Enhance with active retrieval capabilities
    active = create_active_retriever(
        hybrid, 
        config={"max_iterations": 5, "feedback_threshold": 0.7}
    )
    
    # Retrieve relevant records
    results = active.retrieve("user query", limit=10)
"""

from retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalConfig,
    RetrievalResult,
    KeywordSearcher,
    VectorSearcher,
    create_hybrid_retriever
)

from retrieval.active_retrieval import (
    ActiveRetriever,
    ActiveRetrievalConfig,
    QueryRefiner,
    RetrievalFeedback,
    QueryContext,
    QueryComplexity,
    create_active_retriever
)

__version__ = "0.1.0"

__all__ = [
    # Main retriever classes
    "HybridRetriever",
    "ActiveRetriever",
    
    # Configuration classes
    "RetrievalConfig",
    "ActiveRetrievalConfig",
    
    # Component classes
    "KeywordSearcher",
    "VectorSearcher", 
    "QueryRefiner",
    
    # Data structures
    "RetrievalResult",
    "RetrievalFeedback",
    "QueryContext",
    "QueryComplexity",
    
    # Factory functions
    "create_hybrid_retriever",
    "create_active_retriever"
]
