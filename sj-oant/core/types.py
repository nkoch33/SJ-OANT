"""
core.types - Core Data Types and Structures

This module defines the foundational data types used throughout the TMM system.
Following enterprise patterns with immutable dataclasses, comprehensive typing,
and clear separation of concerns.

Design Principles:
- Immutable by default for thread safety
- Rich type annotations for IDE support and runtime validation
- Comprehensive field validation and documentation
- Future-compatible with serialization frameworks
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID


class MemoryTier(Enum):
    """
    Enumeration of memory tiers in the TMM system hierarchy.
    
    The tier system implements a hierarchical approach to memory storage
    with different confidence and persistence guarantees.
    """
    L1_WORKING = "L1_WORKING"          # Recent raw inputs, high churn
    L2_SUMMARIZED = "L2_SUMMARIZED"    # Processed summaries, medium persistence  
    L3_ARCHIVAL = "L3_ARCHIVAL"        # Verified facts, high persistence
    FLAGGED = "FLAGGED"                # Quarantined content, review required


class MemoryStatus(Enum):
    """
    Status enumeration for memory records lifecycle management.
    
    Enables tracking of record state throughout the verification,
    curation, and storage pipeline.
    """
    PENDING = "PENDING"                # Awaiting processing
    VERIFIED = "VERIFIED"              # Passed truth verification
    REJECTED = "REJECTED"              # Failed verification criteria
    FLAGGED = "FLAGGED"                # Requires human review
    ARCHIVED = "ARCHIVED"              # Moved to long-term storage
    DELETED = "DELETED"                # Marked for deletion


class ContentType(Enum):
    """
    Classification of memory content types for specialized handling.
    
    Enables type-specific processing and retrieval strategies.
    """
    USER_INPUT = "USER_INPUT"          # Direct user messages
    FACT = "FACT"                      # Verified factual information
    SUMMARY = "SUMMARY"                # Processed summaries
    CONTEXT = "CONTEXT"                # Environmental/metadata information
    CORRECTION = "CORRECTION"          # Updates to existing information


@dataclass(frozen=True)
class ConfidenceScores:
    """
    Comprehensive confidence and quality metrics for memory records.
    
    These scores drive the TMM system's truth maintenance and curation
    decisions. All scores are normalized to [0.0, 1.0] range.
    
    Attributes:
        truth_score: Likelihood that content is factually accurate
        confidence: System confidence in the assessment
        evidentiality: Quality and strength of supporting evidence
        relevance: Relevance to current context/conversation
        utility: Expected value for future retrieval and use
        source_credibility: Trustworthiness of information source
    """
    truth_score: float = 0.5
    confidence: float = 0.5
    evidentiality: float = 0.5
    relevance: float = 0.5
    utility: float = 0.5
    source_credibility: float = 0.5
    
    def __post_init__(self) -> None:
        """Validate all scores are in valid range [0.0, 1.0]."""
        for field_name, score in self.__dict__.items():
            if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{field_name} must be a number between 0.0 and 1.0, got {score}")
    
    @property
    def overall_score(self) -> float:
        """
        Computed overall confidence score using weighted average.
        
        Returns:
            Weighted combination of all confidence metrics
        """
        # Weights based on TMM methodology priorities
        weights = {
            'truth_score': 0.3,
            'confidence': 0.25,
            'evidentiality': 0.2,
            'relevance': 0.1,
            'utility': 0.1,
            'source_credibility': 0.05
        }
        
        return sum(getattr(self, field) * weight for field, weight in weights.items())


@dataclass(frozen=True)
class Provenance:
    """
    Comprehensive provenance tracking for memory records.
    
    Enables full audit trail and accountability for information sources,
    processing steps, and decision points throughout the TMM pipeline.
    
    Attributes:
        source: Primary source of the information
        source_type: Classification of source (user, system, external, etc.)
        pipeline_stage: TMM pipeline stage that created/modified record
        processing_agent: Specific agent or component responsible
        parent_records: IDs of records this was derived from
        session_id: Session/conversation identifier for grouping
        metadata: Additional context-specific information
    """
    source: str
    source_type: str
    pipeline_stage: str
    processing_agent: str
    parent_records: Set[UUID] = field(default_factory=set)
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_parent(self, parent_id: UUID) -> Provenance:
        """
        Create new provenance with additional parent record.
        
        Args:
            parent_id: UUID of parent record to add
            
        Returns:
            New Provenance instance with updated parent set
        """
        new_parents = self.parent_records | {parent_id}
        return self.__class__(
            source=self.source,
            source_type=self.source_type,
            pipeline_stage=self.pipeline_stage,
            processing_agent=self.processing_agent,
            parent_records=new_parents,
            session_id=self.session_id,
            metadata=self.metadata
        )


@dataclass(frozen=True)
class MemoryRecord:
    """
    Core data structure for memory storage in the TMM system.
    
    This is the fundamental unit of information storage, containing content,
    metadata, confidence scores, and provenance information. Designed for
    immutability, type safety, and comprehensive tracking.
    
    Attributes:
        id: Unique identifier for the record
        payload: The actual content/information being stored
        content_type: Classification of content for processing
        tier: Current memory tier assignment
        status: Lifecycle status of the record
        scores: Confidence and quality metrics
        provenance: Source and processing history
        created_at: Record creation timestamp
        updated_at: Last modification timestamp
        tags: Searchable tags for categorization
        embedding: Optional vector embedding for semantic search
    """
    # Core identification and content
    id: UUID = field(default_factory=uuid.uuid4)
    payload: str = ""
    content_type: ContentType = ContentType.USER_INPUT
    
    # Memory management
    tier: MemoryTier = MemoryTier.L1_WORKING
    status: MemoryStatus = MemoryStatus.PENDING
    
    # Quality and provenance
    scores: ConfidenceScores = field(default_factory=ConfidenceScores)
    provenance: Provenance = field(default_factory=lambda: Provenance(
        source="unknown",
        source_type="system",
        pipeline_stage="creation",
        processing_agent="system"
    ))
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional fields for enhanced functionality
    tags: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        """Validate record integrity and constraints."""
        if not self.payload.strip():
            raise ValueError("MemoryRecord payload cannot be empty")
        
        if self.created_at > datetime.now(timezone.utc):
            raise ValueError("MemoryRecord created_at cannot be in the future")
        
        if self.updated_at < self.created_at:
            raise ValueError("MemoryRecord updated_at cannot be before created_at")
    
    def update_scores(self, new_scores: ConfidenceScores) -> MemoryRecord:
        """
        Create updated record with new confidence scores.
        
        Args:
            new_scores: Updated confidence scores
            
        Returns:
            New MemoryRecord instance with updated scores and timestamp
        """
        return self.__class__(
            id=self.id,
            payload=self.payload,
            content_type=self.content_type,
            tier=self.tier,
            status=self.status,
            scores=new_scores,
            provenance=self.provenance,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            tags=self.tags,
            embedding=self.embedding
        )
    
    def update_status(self, new_status: MemoryStatus, 
                     processing_agent: str = "system") -> MemoryRecord:
        """
        Create updated record with new status and provenance.
        
        Args:
            new_status: New status to assign
            processing_agent: Agent making the status change
            
        Returns:
            New MemoryRecord instance with updated status and provenance
        """
        new_provenance = self.__class__(
            source=self.provenance.source,
            source_type=self.provenance.source_type,
            pipeline_stage=f"status_update_{new_status.value.lower()}",
            processing_agent=processing_agent,
            parent_records=self.provenance.parent_records,
            session_id=self.provenance.session_id,
            metadata={**self.provenance.metadata, "previous_status": self.status.value}
        )
        
        return self.__class__(
            id=self.id,
            payload=self.payload,
            content_type=self.content_type,
            tier=self.tier,
            status=new_status,
            scores=self.scores,
            provenance=new_provenance,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            tags=self.tags,
            embedding=self.embedding
        )
    
    def promote_tier(self, new_tier: MemoryTier, 
                    processing_agent: str = "memory_curator") -> MemoryRecord:
        """
        Create updated record promoted to new memory tier.
        
        Args:
            new_tier: Target memory tier
            processing_agent: Agent performing the promotion
            
        Returns:
            New MemoryRecord instance in the new tier
        """
        new_provenance = self.__class__(
            source=self.provenance.source,
            source_type=self.provenance.source_type,
            pipeline_stage=f"tier_promotion_{new_tier.value.lower()}",
            processing_agent=processing_agent,
            parent_records=self.provenance.parent_records,
            session_id=self.provenance.session_id,
            metadata={**self.provenance.metadata, "previous_tier": self.tier.value}
        )
        
        return self.__class__(
            id=self.id,
            payload=self.payload,
            content_type=self.content_type,
            tier=new_tier,
            status=self.status,
            scores=self.scores,
            provenance=new_provenance,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            tags=self.tags,
            embedding=self.embedding
        )
    
    def add_tags(self, new_tags: Set[str]) -> MemoryRecord:
        """
        Create updated record with additional tags.
        
        Args:
            new_tags: Set of tags to add
            
        Returns:
            New MemoryRecord instance with updated tags
        """
        combined_tags = self.tags | new_tags
        
        return self.__class__(
            id=self.id,
            payload=self.payload,
            content_type=self.content_type,
            tier=self.tier,
            status=self.status,
            scores=self.scores,
            provenance=self.provenance,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            tags=combined_tags,
            embedding=self.embedding
        )
    
    @property
    def age_seconds(self) -> float:
        """
        Age of the record in seconds since creation.
        
        Returns:
            Seconds elapsed since record creation
        """
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    @property
    def is_verified(self) -> bool:
        """Check if record has been verified for truth."""
        return self.status == MemoryStatus.VERIFIED
    
    @property
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        Check if record meets high confidence threshold.
        
        Args:
            threshold: Minimum confidence score required
            
        Returns:
            True if overall score meets threshold
        """
        return self.scores.overall_score >= threshold


# Type aliases for better code readability
RecordID = UUID
RecordCollection = List[MemoryRecord]
RecordIndex = Dict[RecordID, MemoryRecord]

# Search and filter types
SearchQuery = Union[str, Dict[str, Any]]
FilterCriteria = Dict[str, Any]
