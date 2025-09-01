"""
TMM Memory Package

This package implements the multi-tiered memory management system
for the Truth-Maintained Memory architecture.

Key Components:
- typed_store: Enterprise-grade memory storage with indexing and search
- policies: Selective addition and combined deletion policies
- voting: Multi-agent voting system for memory decisions (TODO)

Example Usage:
    from memory import InMemoryStore, PolicyEngine, PolicyConfig
    
    # Create memory store
    store = InMemoryStore(l1_limit=100, l2_limit=500, l3_limit=1000)
    
    # Create policy engine
    config = PolicyConfig(selective_add_trust_threshold=0.8)
    policies = PolicyEngine(config)
    
    # Add a record
    record = MemoryRecord(payload="Important fact")
    if policies.evaluate_addition(record).decision:
        store.add(record)
"""

from memory.typed_store import (
    InMemoryStore,
    TypedMemoryStore,  # Legacy alias
    MemoryState  # For LangGraph compatibility
)

from memory.policies import (
    PolicyEngine,
    PolicyConfig,
    PolicyDecision,
    SelectiveAdditionPolicy,
    CombinedDeletionPolicy
)

__version__ = "0.1.0"

__all__ = [
    # Primary implementations
    "InMemoryStore",
    "TypedMemoryStore",
    "MemoryState",
    
    # Policy system
    "PolicyEngine",
    "PolicyConfig", 
    "PolicyDecision",
    "SelectiveAdditionPolicy",
    "CombinedDeletionPolicy"
]
