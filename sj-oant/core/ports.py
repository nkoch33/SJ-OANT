"""
core.ports - Protocol Interfaces for Dependency Injection

This module defines the Protocol interfaces that enable clean dependency injection
and loose coupling throughout the TMM system. Following hexagonal architecture
principles with clear separation between domain logic and infrastructure.

Design Principles:
- Protocol-based interfaces for maximum flexibility
- Async support for scalable operations
- Comprehensive error handling contracts
- Clear separation of concerns
- Future-compatible with distributed systems
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any, AsyncIterator, Dict, Iterator, List, Optional, Protocol, 
    Set, Tuple, Union
)
from uuid import UUID

from core.types import (
    ConfidenceScores, FilterCriteria, MemoryRecord, MemoryStatus, 
    MemoryTier, RecordCollection, RecordID, SearchQuery
)


class MemoryStorePort(Protocol):
    """
    Protocol interface for memory storage implementations.
    
    Defines the contract for persistent and transient memory storage,
    enabling multiple backend implementations (in-memory, database,
    distributed cache, etc.) while maintaining consistent interface.
    
    All methods are designed to be thread-safe and support both
    synchronous and asynchronous operations.
    """
    
    def add(self, record: MemoryRecord) -> RecordID:
        """
        Add a new memory record to the store.
        
        Args:
            record: MemoryRecord instance to store
            
        Returns:
            The ID of the stored record
            
        Raises:
            ValueError: If record is invalid or already exists
            StorageError: If storage operation fails
        """
        ...
    
    def get(self, record_id: RecordID) -> Optional[MemoryRecord]:
        """
        Retrieve a memory record by ID.
        
        Args:
            record_id: Unique identifier of the record
            
        Returns:
            MemoryRecord if found, None otherwise
            
        Raises:
            StorageError: If retrieval operation fails
        """
        ...
    
    def update(self, record: MemoryRecord) -> bool:
        """
        Update an existing memory record.
        
        Args:
            record: Updated MemoryRecord instance
            
        Returns:
            True if update successful, False if record not found
            
        Raises:
            ValueError: If record is invalid
            StorageError: If update operation fails
        """
        ...
    
    def delete(self, record_id: RecordID) -> bool:
        """
        Delete a memory record from the store.
        
        Args:
            record_id: Unique identifier of record to delete
            
        Returns:
            True if deletion successful, False if record not found
            
        Raises:
            StorageError: If deletion operation fails
        """
        ...
    
    def search(self, query: SearchQuery, 
               filters: Optional[FilterCriteria] = None,
               limit: Optional[int] = None) -> RecordCollection:
        """
        Search for memory records matching query and filters.
        
        Args:
            query: Search query (text or structured)
            filters: Optional filtering criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching MemoryRecord instances
            
        Raises:
            SearchError: If search operation fails
        """
        ...
    
    def get_by_tier(self, tier: MemoryTier, 
                   limit: Optional[int] = None) -> RecordCollection:
        """
        Retrieve all records in a specific memory tier.
        
        Args:
            tier: Target memory tier
            limit: Maximum number of records to return
            
        Returns:
            List of MemoryRecord instances in the tier
        """
        ...
    
    def get_by_status(self, status: MemoryStatus,
                     limit: Optional[int] = None) -> RecordCollection:
        """
        Retrieve all records with a specific status.
        
        Args:
            status: Target memory status
            limit: Maximum number of records to return
            
        Returns:
            List of MemoryRecord instances with the status
        """
        ...
    
    def count(self, filters: Optional[FilterCriteria] = None) -> int:
        """
        Count records matching optional filters.
        
        Args:
            filters: Optional filtering criteria
            
        Returns:
            Number of matching records
        """
        ...
    
    def clear_tier(self, tier: MemoryTier) -> int:
        """
        Remove all records from a specific tier.
        
        Args:
            tier: Memory tier to clear
            
        Returns:
            Number of records removed
        """
        ...


class VerifierPort(Protocol):
    """
    Protocol interface for truth verification implementations.
    
    Enables pluggable verification strategies (rule-based, ML-based,
    ensemble methods) while maintaining consistent interface for
    truth assessment and confidence scoring.
    """
    
    def verify(self, content: str, context: Optional[Dict[str, Any]] = None) -> ConfidenceScores:
        """
        Verify the truthfulness and quality of content.
        
        Args:
            content: Text content to verify
            context: Optional context for verification
            
        Returns:
            ConfidenceScores with truth and quality metrics
            
        Raises:
            VerificationError: If verification process fails
        """
        ...
    
    def detect_contradictions(self, content: str, 
                            existing_records: RecordCollection) -> List[Tuple[RecordID, float]]:
        """
        Detect contradictions with existing memory records.
        
        Args:
            content: New content to check
            existing_records: Records to check against
            
        Returns:
            List of (record_id, contradiction_score) tuples
            
        Raises:
            VerificationError: If contradiction detection fails
        """
        ...
    
    def assess_evidence(self, content: str) -> float:
        """
        Assess the quality of evidence in content.
        
        Args:
            content: Content to assess
            
        Returns:
            Evidence quality score [0.0, 1.0]
        """
        ...


class RetrieverPort(Protocol):
    """
    Protocol interface for memory retrieval implementations.
    
    Supports multiple retrieval strategies (semantic, keyword, hybrid)
    and enables efficient access to relevant memory records based on
    queries and context.
    """
    
    def retrieve(self, query: str, 
                context: Optional[Dict[str, Any]] = None,
                tier_filter: Optional[Set[MemoryTier]] = None,
                limit: int = 10) -> RecordCollection:
        """
        Retrieve relevant memory records for a query.
        
        Args:
            query: Search query string
            context: Optional context for retrieval
            tier_filter: Optional set of tiers to search
            limit: Maximum number of records to return
            
        Returns:
            List of relevant MemoryRecord instances
            
        Raises:
            RetrievalError: If retrieval operation fails
        """
        ...
    
    def retrieve_similar(self, record: MemoryRecord,
                        threshold: float = 0.7,
                        limit: int = 10) -> RecordCollection:
        """
        Retrieve records similar to the given record.
        
        Args:
            record: Reference record for similarity search
            threshold: Minimum similarity score
            limit: Maximum number of records to return
            
        Returns:
            List of similar MemoryRecord instances
        """
        ...
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text content.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...


class FilterPort(Protocol):
    """
    Protocol interface for content filtering implementations.
    
    Enables pluggable filtering strategies for relevance, redundancy,
    and noise removal while maintaining consistent interface.
    """
    
    def filter_relevance(self, content: str, 
                        context: Dict[str, Any]) -> float:
        """
        Score content relevance to given context.
        
        Args:
            content: Content to score
            context: Contextual information
            
        Returns:
            Relevance score [0.0, 1.0]
        """
        ...
    
    def detect_redundancy(self, content: str,
                         existing_content: List[str]) -> Tuple[bool, float]:
        """
        Detect if content is redundant with existing content.
        
        Args:
            content: New content to check
            existing_content: List of existing content to compare
            
        Returns:
            Tuple of (is_redundant, confidence_score)
        """
        ...
    
    def remove_noise(self, content: str) -> str:
        """
        Remove noise and clean content.
        
        Args:
            content: Raw content to clean
            
        Returns:
            Cleaned content string
        """
        ...


class PolicyPort(Protocol):
    """
    Protocol interface for memory management policy implementations.
    
    Enables pluggable policies for memory addition, deletion, and
    tier management while maintaining consistent decision interface.
    """
    
    def should_add(self, record: MemoryRecord,
                  context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if a record should be added to memory.
        
        Args:
            record: MemoryRecord to evaluate
            context: Optional decision context
            
        Returns:
            True if record should be added, False otherwise
        """
        ...
    
    def should_delete(self, record: MemoryRecord,
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if a record should be deleted from memory.
        
        Args:
            record: MemoryRecord to evaluate
            context: Optional decision context
            
        Returns:
            True if record should be deleted, False otherwise
        """
        ...
    
    def should_promote(self, record: MemoryRecord,
                      target_tier: MemoryTier) -> bool:
        """
        Determine if a record should be promoted to target tier.
        
        Args:
            record: MemoryRecord to evaluate
            target_tier: Target memory tier
            
        Returns:
            True if record should be promoted, False otherwise
        """
        ...


class AgentPort(Protocol):
    """
    Protocol interface for TMM agent implementations.
    
    Defines the contract for processing stages in the TMM pipeline,
    enabling modular and swappable agent implementations.
    """
    
    def process(self, input_data: Any, 
               context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process input data and return results.
        
        Args:
            input_data: Input to process
            context: Optional processing context
            
        Returns:
            Processing results
            
        Raises:
            ProcessingError: If processing fails
        """
        ...


# Abstract base classes for common implementations
class BaseMemoryStore(ABC):
    """
    Abstract base class for memory store implementations.
    
    Provides common functionality and enforces interface compliance
    for concrete memory store implementations.
    """
    
    @abstractmethod
    def add(self, record: MemoryRecord) -> RecordID:
        """Add a memory record to the store."""
        pass
    
    @abstractmethod
    def get(self, record_id: RecordID) -> Optional[MemoryRecord]:
        """Retrieve a memory record by ID."""
        pass
    
    @abstractmethod
    def update(self, record: MemoryRecord) -> bool:
        """Update an existing memory record."""
        pass
    
    @abstractmethod
    def delete(self, record_id: RecordID) -> bool:
        """Delete a memory record from the store."""
        pass
    
    @abstractmethod
    def search(self, query: SearchQuery, 
               filters: Optional[FilterCriteria] = None,
               limit: Optional[int] = None) -> RecordCollection:
        """Search for memory records."""
        pass


class BaseVerifier(ABC):
    """
    Abstract base class for verifier implementations.
    
    Provides structure for truth verification and confidence scoring
    implementations.
    """
    
    @abstractmethod
    def verify(self, content: str, context: Optional[Dict[str, Any]] = None) -> ConfidenceScores:
        """Verify content truthfulness."""
        pass
    
    @abstractmethod
    def detect_contradictions(self, content: str, 
                            existing_records: RecordCollection) -> List[Tuple[RecordID, float]]:
        """Detect contradictions with existing records."""
        pass


class BaseRetriever(ABC):
    """
    Abstract base class for retriever implementations.
    
    Provides structure for memory retrieval and similarity search
    implementations.
    """
    
    @abstractmethod
    def retrieve(self, query: str, 
                context: Optional[Dict[str, Any]] = None,
                tier_filter: Optional[Set[MemoryTier]] = None,
                limit: int = 10) -> RecordCollection:
        """Retrieve relevant memory records."""
        pass


# Custom exceptions for error handling
class TMMemoryError(Exception):
    """Base exception for TMM system errors."""
    pass


class StorageError(TMMemoryError):
    """Exception raised for storage operation failures."""
    pass


class VerificationError(TMMemoryError):
    """Exception raised for verification process failures."""
    pass


class RetrievalError(TMMemoryError):
    """Exception raised for retrieval operation failures."""
    pass


class EmbeddingError(TMMemoryError):
    """Exception raised for embedding generation failures."""
    pass


class ProcessingError(TMMemoryError):
    """Exception raised for agent processing failures."""
    pass


class SearchError(TMMemoryError):
    """Exception raised for search operation failures."""
    pass
