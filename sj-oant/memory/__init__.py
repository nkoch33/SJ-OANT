"""
TMM Memory Package

This package implements the multi-tiered memory management system
for the Truth-Maintained Memory architecture.

Key Components:
- typed_store: Enterprise-grade memory storage with indexing and search

Example Usage:
    from memory import InMemoryStore
    from core.types import ConfidenceScores
    
    # Create memory store
    store = InMemoryStore(l1_limit=100, l2_limit=500, l3_limit=1000)
    
    # Add a record
    scores = ConfidenceScores(truth_score=0.9, confidence=0.8)
    record_id = store.add_to_l1("Important fact", scores)
"""

from memory.typed_store import (
    InMemoryStore,
    TypedMemoryStore,  # Legacy alias
    MemoryState  # For LangGraph compatibility
)

# Policy system removed - using simplified memory management

__version__ = "0.1.0"

__all__ = [
    # Primary implementations
    "InMemoryStore",
    "TypedMemoryStore",
    "MemoryState"
]
