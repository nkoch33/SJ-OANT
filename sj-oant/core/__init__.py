"""
TMM Core Package

This package contains the foundational types, interfaces, and protocols
that form the backbone of the Truth-Maintained Memory system.

Key Components:
- types: Core data structures (MemoryRecord, ConfidenceScores, etc.)
- ports: Protocol interfaces for dependency injection and clean architecture

Example Usage:
    from core.types import MemoryRecord, MemoryTier, ConfidenceScores
    from core.ports import MemoryStorePort, VerifierPort
    
    # Create a memory record
    record = MemoryRecord(
        payload="Some important information",
        tier=MemoryTier.L1_WORKING,
        scores=ConfidenceScores(truth_score=0.8, confidence=0.9)
    )
"""

from core.types import (
    # Core data structures
    MemoryRecord,
    ConfidenceScores,
    Provenance,
    
    # Enumerations
    MemoryTier,
    MemoryStatus,
    ContentType,
    
    # Type aliases
    RecordID,
    RecordCollection,
    SearchQuery,
    FilterCriteria
)

from core.ports import (
    # Primary interfaces
    MemoryStorePort,
    VerifierPort,
    RetrieverPort,
    FilterPort,
    PolicyPort,
    AgentPort,
    
    # Base classes
    BaseMemoryStore,
    BaseVerifier,
    BaseRetriever,
    
    # Exceptions
    TMMemoryError,
    StorageError,
    VerificationError,
    RetrievalError,
    ProcessingError
)

__version__ = "0.1.0"
__author__ = "TMM Development Team"

__all__ = [
    # Core types
    "MemoryRecord",
    "ConfidenceScores", 
    "Provenance",
    "MemoryTier",
    "MemoryStatus",
    "ContentType",
    "RecordID",
    "RecordCollection",
    "SearchQuery",
    "FilterCriteria",
    
    # Interfaces
    "MemoryStorePort",
    "VerifierPort",
    "RetrieverPort",
    "FilterPort",
    "PolicyPort",
    "AgentPort",
    
    # Base classes
    "BaseMemoryStore",
    "BaseVerifier",
    "BaseRetriever",
    
    # Exceptions
    "TMMemoryError",
    "StorageError",
    "VerificationError",
    "RetrievalError",
    "ProcessingError"
]
