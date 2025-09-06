"""
memory.typed_store - Multi-Tiered Memory Storage

This module implements a production-ready, thread-safe memory storage system
following clean architecture patterns with comprehensive logging, metrics, and error handling.

The implementation follows hexagonal architecture principles with clear separation
between domain logic and infrastructure concerns, enabling easy testing and
multiple backend implementations.

Design Features:
- Thread-safe operations with proper locking
- Comprehensive error handling and validation
- Rich logging and metrics collection
- Configurable tier limits and policies
- Extensible search and filtering capabilities
- Full audit trail and provenance tracking
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
from core.types import ConfidenceScores
from uuid import UUID

from core.ports import BaseMemoryStore, FilterCriteria, SearchError, StorageError
from core.types import (
    MemoryRecord, MemoryStatus, MemoryTier, RecordCollection, 
    RecordID, SearchQuery
)

logger = logging.getLogger(__name__)


class InMemoryStore(BaseMemoryStore):
    """
    Thread-safe in-memory implementation of the MemoryStore interface.
    
    This implementation provides a production-ready memory store for development
    and testing environments. Features comprehensive indexing, search capabilities,
    and proper concurrency control.
    
    For production deployments, this can be replaced with persistent storage
    implementations (PostgreSQL, Redis, etc.) without changing client code.
    
    Attributes:
        tier_limits: Maximum number of records per tier
        records: Main record storage indexed by ID
        tier_index: Secondary index for efficient tier-based queries
        status_index: Secondary index for efficient status-based queries
        _lock: Thread safety lock for concurrent operations
    """
    
    def __init__(self, 
                 l1_limit: int = 100,
                 l2_limit: int = 500,
                 l3_limit: int = 1000,
                 flagged_limit: int = 200):
        """
        Initialize the in-memory store with configurable limits.
        
        Args:
            l1_limit: Maximum records in L1 working memory
            l2_limit: Maximum records in L2 summarized memory
            l3_limit: Maximum records in L3 archival memory
            flagged_limit: Maximum records in flagged storage
        """
        self.tier_limits = {
            MemoryTier.L1_WORKING: l1_limit,
            MemoryTier.L2_SUMMARIZED: l2_limit,
            MemoryTier.L3_ARCHIVAL: l3_limit,
            MemoryTier.FLAGGED: flagged_limit
        }
        
        # Primary storage and secondary indexes
        self.records: Dict[RecordID, MemoryRecord] = {}
        self.tier_index: Dict[MemoryTier, Set[RecordID]] = defaultdict(set)
        self.status_index: Dict[MemoryStatus, Set[RecordID]] = defaultdict(set)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics tracking
        self._operation_counts = defaultdict(int)
        
        logger.info(
            f"Initialized InMemoryStore with limits: "
            f"L1={l1_limit}, L2={l2_limit}, L3={l3_limit}, Flagged={flagged_limit}"
        )
    
    @property 
    def _records(self):
        """Legacy property access to records for compatibility."""
        return self.records
    
    def get_state(self) -> "MemoryState":
        """Get current memory state for LangGraph compatibility."""
        with self._lock:
            return {
                "L1": [r for r in self.records.values() if r.tier == MemoryTier.L1_WORKING],
                "L2": [r for r in self.records.values() if r.tier == MemoryTier.L2_SUMMARIZED],  
                "L3": [r for r in self.records.values() if r.tier == MemoryTier.L3_ARCHIVAL],
                "flagged": [r for r in self.records.values() if r.tier == MemoryTier.FLAGGED],
                "user_input": "",
                "final_response": ""
            }
    
    def add_to_l1(self, content: str, scores: Optional[ConfidenceScores] = None, **kwargs) -> RecordID:
        """Add content directly to L1 working memory."""
        from core.types import MemoryRecord, ConfidenceScores, Provenance
        from uuid import uuid4
        from datetime import datetime, timezone
        
        if scores is None:
            scores = ConfidenceScores()
        
        record = MemoryRecord(
            id=uuid4(),
            payload=content,
            tier=MemoryTier.L1_WORKING,
            status=MemoryStatus.PENDING,
            scores=scores,
            provenance=Provenance(
                source="direct_add",
                source_type="system",
                pipeline_stage="memory_store",
                processing_agent="InMemoryStore"
            ),
            **kwargs
        )
        return self.add(record)
    
    def add_to_flagged(self, content: str, reason: str, metadata: Dict[str, Any] = None) -> RecordID:
        """Add content to flagged memory tier."""
        from core.types import MemoryRecord, ConfidenceScores, Provenance
        from uuid import uuid4
        from datetime import datetime, timezone
        
        if metadata is None:
            metadata = {}
        
        # Add flagging information to metadata
        metadata.update({"flag_reason": reason, "flagged_at": datetime.now(timezone.utc).isoformat()})
        
        record = MemoryRecord(
            id=uuid4(),
            payload=content,
            tier=MemoryTier.FLAGGED,
            status=MemoryStatus.FLAGGED,
            scores=ConfidenceScores(confidence=0.1, truth_score=0.1),  # Low scores for flagged content
            provenance=Provenance(
                source="flagged_content",
                source_type="system",
                pipeline_stage="memory_curation",
                processing_agent="MemoryCurationAgent",
                metadata=metadata
            )
        )
        return self.add(record)
    
    def add(self, record: MemoryRecord) -> RecordID:
        """
        Add a new memory record to the store.
        
        Validates record integrity, checks tier capacity limits,
        and updates all relevant indexes atomically.
        
        Args:
            record: MemoryRecord instance to store
            
        Returns:
            The ID of the stored record
            
        Raises:
            ValueError: If record is invalid or already exists
            StorageError: If tier capacity exceeded or storage fails
        """
        if not record.payload.strip():
            raise ValueError("Cannot store record with empty payload")
        
        with self._lock:
            # Check if record already exists
            if record.id in self.records:
                raise ValueError(f"Record {record.id} already exists")
            
            # Check tier capacity limits
            tier_count = len(self.tier_index[record.tier])
            tier_limit = self.tier_limits[record.tier]
            
            if tier_count >= tier_limit:
                logger.warning(
                    f"Tier {record.tier.value} at capacity ({tier_count}/{tier_limit})"
                )
                raise StorageError(
                    f"Tier {record.tier.value} capacity exceeded "
                    f"({tier_count}/{tier_limit})"
                )
            
            try:
                # Add to primary storage
                self.records[record.id] = record
                
                # Update indexes
                self.tier_index[record.tier].add(record.id)
                self.status_index[record.status].add(record.id)
                
                # Update metrics
                self._operation_counts['add'] += 1
                
                logger.debug(
                    f"Added record {record.id} to tier {record.tier.value} "
                    f"with status {record.status.value}"
                )
                
                return record.id
                
            except Exception as e:
                # Rollback on error
                self.records.pop(record.id, None)
                self.tier_index[record.tier].discard(record.id)
                self.status_index[record.status].discard(record.id)
                
                logger.error(f"Failed to add record {record.id}: {e}")
                raise StorageError(f"Failed to store record: {e}") from e
    
    def get(self, record_id: RecordID) -> Optional[MemoryRecord]:
        """
        Retrieve a memory record by ID.
        
        Args:
            record_id: Unique identifier of the record
            
        Returns:
            MemoryRecord if found, None otherwise
        """
        with self._lock:
            record = self.records.get(record_id)
            if record:
                self._operation_counts['get_hit'] += 1
                logger.debug(f"Retrieved record {record_id}")
            else:
                self._operation_counts['get_miss'] += 1
                logger.debug(f"Record {record_id} not found")
            
            return record
    
    def update(self, record: MemoryRecord) -> bool:
        """
        Update an existing memory record.
        
        Updates both primary storage and secondary indexes atomically.
        Maintains referential integrity across all data structures.
        
        Args:
            record: Updated MemoryRecord instance
            
        Returns:
            True if update successful, False if record not found
            
        Raises:
            ValueError: If record is invalid
            StorageError: If update operation fails
        """
        if not record.payload.strip():
            raise ValueError("Cannot update to record with empty payload")
        
        with self._lock:
            # Check if record exists
            existing_record = self.records.get(record.id)
            if not existing_record:
                logger.debug(f"Update failed - record {record.id} not found")
                return False
            
            try:
                # Update indexes if tier or status changed
                if existing_record.tier != record.tier:
                    self.tier_index[existing_record.tier].discard(record.id)
                    self.tier_index[record.tier].add(record.id)
                    
                    # Check new tier capacity
                    tier_count = len(self.tier_index[record.tier])
                    tier_limit = self.tier_limits[record.tier]
                    
                    if tier_count > tier_limit:
                        # Rollback tier change
                        self.tier_index[record.tier].discard(record.id)
                        self.tier_index[existing_record.tier].add(record.id)
                        raise StorageError(
                            f"Tier {record.tier.value} capacity would be exceeded"
                        )
                
                if existing_record.status != record.status:
                    self.status_index[existing_record.status].discard(record.id)
                    self.status_index[record.status].add(record.id)
                
                # Update primary storage
                self.records[record.id] = record
                
                # Update metrics
                self._operation_counts['update'] += 1
                
                logger.debug(f"Updated record {record.id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update record {record.id}: {e}")
                raise StorageError(f"Failed to update record: {e}") from e
    
    def delete(self, record_id: RecordID) -> bool:
        """
        Delete a memory record from the store.
        
        Removes record from primary storage and all secondary indexes atomically.
        
        Args:
            record_id: Unique identifier of record to delete
            
        Returns:
            True if deletion successful, False if record not found
        """
        with self._lock:
            record = self.records.get(record_id)
            if not record:
                logger.debug(f"Delete failed - record {record_id} not found")
                return False
            
            try:
                # Remove from primary storage
                del self.records[record_id]
                
                # Remove from indexes
                self.tier_index[record.tier].discard(record_id)
                self.status_index[record.status].discard(record_id)
                
                # Update metrics
                self._operation_counts['delete'] += 1
                
                logger.debug(f"Deleted record {record_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete record {record_id}: {e}")
                raise StorageError(f"Failed to delete record: {e}") from e
    
    def search(self, 
               query: SearchQuery, 
               filters: Optional[FilterCriteria] = None,
               limit: Optional[int] = None) -> RecordCollection:
        """
        Search for memory records matching query and filters.
        
        Supports both text-based and structured queries with comprehensive
        filtering capabilities. Uses indexes for efficient retrieval.
        
        Args:
            query: Search query (text string or structured dict)
            filters: Optional filtering criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching MemoryRecord instances
            
        Raises:
            SearchError: If search operation fails
        """
        try:
            with self._lock:
                # Start with all records if no specific criteria
                candidate_ids = set(self.records.keys())
                
                # Apply filters to narrow candidate set
                if filters:
                    candidate_ids = self._apply_filters(candidate_ids, filters)
                
                # Apply text search if query is string
                if isinstance(query, str) and query.strip():
                    candidate_ids = self._apply_text_search(candidate_ids, query)
                elif isinstance(query, dict):
                    candidate_ids = self._apply_structured_search(candidate_ids, query)
                
                # Convert IDs to records
                results = [
                    self.records[record_id] 
                    for record_id in candidate_ids 
                    if record_id in self.records
                ]
                
                # Sort by relevance/recency (most recent first)
                results.sort(key=lambda r: r.updated_at, reverse=True)
                
                # Apply limit
                if limit is not None:
                    results = results[:limit]
                
                # Update metrics
                self._operation_counts['search'] += 1
                
                logger.debug(
                    f"Search returned {len(results)} results for query: {query}"
                )
                
                return results
                
        except Exception as e:
            logger.error(f"Search failed for query {query}: {e}")
            raise SearchError(f"Search operation failed: {e}") from e
    
    def get_by_tier(self, 
                   tier: MemoryTier, 
                   limit: Optional[int] = None) -> RecordCollection:
        """
        Retrieve all records in a specific memory tier.
        
        Uses tier index for efficient retrieval without scanning all records.
        
        Args:
            tier: Target memory tier
            limit: Maximum number of records to return
            
        Returns:
            List of MemoryRecord instances in the tier
        """
        with self._lock:
            record_ids = list(self.tier_index[tier])
            
            # Sort by creation time (newest first)
            records = [self.records[rid] for rid in record_ids if rid in self.records]
            records.sort(key=lambda r: r.created_at, reverse=True)
            
            if limit is not None:
                records = records[:limit]
            
            logger.debug(f"Retrieved {len(records)} records from tier {tier.value}")
            return records
    
    def get_by_status(self, 
                     status: MemoryStatus,
                     limit: Optional[int] = None) -> RecordCollection:
        """
        Retrieve all records with a specific status.
        
        Uses status index for efficient retrieval without scanning all records.
        
        Args:
            status: Target memory status
            limit: Maximum number of records to return
            
        Returns:
            List of MemoryRecord instances with the status
        """
        with self._lock:
            record_ids = list(self.status_index[status])
            
            # Sort by update time (newest first)
            records = [self.records[rid] for rid in record_ids if rid in self.records]
            records.sort(key=lambda r: r.updated_at, reverse=True)
            
            if limit is not None:
                records = records[:limit]
            
            logger.debug(f"Retrieved {len(records)} records with status {status.value}")
            return records
    
    def count(self, filters: Optional[FilterCriteria] = None) -> int:
        """
        Count records matching optional filters.
        
        Args:
            filters: Optional filtering criteria
            
        Returns:
            Number of matching records
        """
        with self._lock:
            if not filters:
                return len(self.records)
            
            candidate_ids = set(self.records.keys())
            candidate_ids = self._apply_filters(candidate_ids, filters)
            
            return len(candidate_ids)
    
    def clear_tier(self, tier: MemoryTier) -> int:
        """
        Remove all records from a specific tier.
        
        Args:
            tier: Memory tier to clear
            
        Returns:
            Number of records removed
        """
        with self._lock:
            record_ids = list(self.tier_index[tier])
            removed_count = 0
            
            for record_id in record_ids:
                if self.delete(record_id):
                    removed_count += 1
            
            logger.info(f"Cleared {removed_count} records from tier {tier.value}")
            return removed_count
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get operational metrics for monitoring and debugging.
        
        Returns:
            Dictionary of operation counts and store statistics
        """
        with self._lock:
            tier_counts = {
                f"tier_{tier.value.lower()}_count": len(ids)
                for tier, ids in self.tier_index.items()
            }
            
            status_counts = {
                f"status_{status.value.lower()}_count": len(ids)
                for status, ids in self.status_index.items()
            }
            
            return {
                **dict(self._operation_counts),
                "total_records": len(self.records),
                **tier_counts,
                **status_counts
            }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary in expected format for writer_editor compatibility."""
        metrics = self.get_metrics()
        return {
            "total_records": metrics.get("total_records", 0),
            "tier_sizes": {
                "L1": metrics.get("tier_l1_working_count", 0),
                "L2": metrics.get("tier_l2_summarized_count", 0), 
                "L3": metrics.get("tier_l3_archival_count", 0),
                "flagged": metrics.get("tier_flagged_count", 0)
            }
        }
    
    def _apply_filters(self, 
                      candidate_ids: Set[RecordID], 
                      filters: FilterCriteria) -> Set[RecordID]:
        """Apply filtering criteria to candidate record IDs."""
        filtered_ids = candidate_ids.copy()
        
        # Filter by tier
        if 'tier' in filters:
            tier_filter = filters['tier']
            if isinstance(tier_filter, MemoryTier):
                tier_ids = self.tier_index[tier_filter]
                filtered_ids &= tier_ids
            elif isinstance(tier_filter, (list, set)):
                tier_ids = set()
                for tier in tier_filter:
                    tier_ids.update(self.tier_index[tier])
                filtered_ids &= tier_ids
        
        # Filter by status
        if 'status' in filters:
            status_filter = filters['status']
            if isinstance(status_filter, MemoryStatus):
                status_ids = self.status_index[status_filter]
                filtered_ids &= status_ids
            elif isinstance(status_filter, (list, set)):
                status_ids = set()
                for status in status_filter:
                    status_ids.update(self.status_index[status])
                filtered_ids &= status_ids
        
        # Filter by confidence threshold
        if 'min_confidence' in filters:
            min_conf = filters['min_confidence']
            filtered_ids = {
                rid for rid in filtered_ids
                if rid in self.records and 
                self.records[rid].scores.overall_score >= min_conf
            }
        
        # Filter by age (created within last N seconds)
        if 'max_age_seconds' in filters:
            max_age = filters['max_age_seconds']
            current_time = datetime.now(timezone.utc)
            filtered_ids = {
                rid for rid in filtered_ids
                if rid in self.records and
                (current_time - self.records[rid].created_at).total_seconds() <= max_age
            }
        
        # Filter by tags
        if 'tags' in filters:
            required_tags = set(filters['tags'])
            filtered_ids = {
                rid for rid in filtered_ids
                if rid in self.records and
                required_tags.issubset(self.records[rid].tags)
            }
        
        return filtered_ids
    
    def _apply_text_search(self, 
                          candidate_ids: Set[RecordID], 
                          query: str) -> Set[RecordID]:
        """Apply text-based search to candidate record IDs with semantic matching."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Enhanced stop words for better matching (Phase 2.2 optimization)
        stop_words = {
            # Basic articles and determiners
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            # Prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'throughout', 'within', 'without',
            # Conjunctions
            'and', 'or', 'but', 'nor', 'so', 'yet', 'although', 'though', 'because', 'since', 'while',
            # Auxiliary verbs and forms of 'be'
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
            # Question words (often noise in matching)
            'what', 'where', 'when', 'who', 'how', 'which', 'why',
            # Common pronouns
            'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'i', 'me', 'my'
        }
        meaningful_query_words = query_words - stop_words
        
        matching_ids = set()
        
        for rid in candidate_ids:
            if rid in self.records:
                record = self.records[rid]
                payload_lower = record.payload.lower()
                payload_words = set(payload_lower.split())
                
                # Calculate word overlap
                overlap = len(meaningful_query_words & payload_words)
                
                # Include record if:
                # 1. Has meaningful word overlap, OR
                # 2. Is substantial content (likely stored context), OR  
                # 3. Contains exact query substring (original behavior)
                if (overlap >= 1 or  # At least 1 meaningful word overlap
                    len(record.payload) > 100 or  # Substantial content (context)
                    query_lower in payload_lower):  # Exact substring match
                    matching_ids.add(rid)
        
        return matching_ids
    
    def _apply_structured_search(self, 
                               candidate_ids: Set[RecordID], 
                               query: dict) -> Set[RecordID]:
        """Apply structured search criteria to candidate record IDs."""
        # TODO: Implement more sophisticated structured search
        # For now, just treat as additional filters
        return self._apply_filters(candidate_ids, query)
    
    def reset_memory(self) -> None:
        """
        Reset all memory by clearing all tiers and records.
        
        This method is used to simulate fresh conversations in evaluation.
        """
        with self._lock:
            # Clear all records
            self.records.clear()
            
            # Clear all tier indices
            for tier in MemoryTier:
                self.tier_index[tier].clear()
            
            # Clear all status indices
            for status in MemoryStatus:
                self.status_index[status].clear()
            
            # Reset operation counts
            self._operation_counts.clear()
            
            logger.info("Memory store reset - all records and indices cleared")


# Legacy compatibility for existing code
TypedMemoryStore = InMemoryStore

# For LangGraph pipeline compatibility
from typing import TypedDict

class MemoryState(TypedDict):
    """Legacy MemoryState for LangGraph pipeline compatibility."""
    L1: List[str]
    L2: List[str] 
    L3: List[str]
    flagged: List[str]
    user_input: str
    final_response: str
