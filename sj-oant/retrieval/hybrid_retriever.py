"""
retrieval.hybrid_retriever - Hybrid Memory Retrieval System

This module implements a hybrid retrieval system that combines multiple
retrieval strategies for optimal memory access in the TMM system.

Key Features:
- Combines keyword search with vector similarity search
- Multi-tier memory search across L1/L2/L3 tiers
- Configurable scoring and ranking strategies
- Extensible architecture for adding new retrieval methods
- Performance optimization with caching and indexing

The hybrid retriever provides the foundation for memory access,
ensuring relevant information is efficiently retrieved from the
multi-tiered memory store.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from abc import ABC, abstractmethod

from core.ports import BaseRetriever, FilterCriteria, SearchError
from core.types import (
    MemoryRecord, MemoryTier, RecordCollection, SearchQuery
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration parameters for hybrid retrieval."""
    k: int = 10                           # Number of results to return
    keyword_weight: float = 0.3           # Weight for keyword search
    vector_weight: float = 0.7            # Weight for vector search
    tier_weights: Dict[MemoryTier, float] = field(default_factory=lambda: {
        MemoryTier.L1_WORKING: 1.0,      # Prioritize recent content
        MemoryTier.L2_SUMMARIZED: 0.8,   # Medium priority for summaries
        MemoryTier.L3_ARCHIVAL: 0.6,     # Lower priority for archived
        MemoryTier.FLAGGED: 0.1          # Very low priority for flagged
    })
    min_similarity_threshold: float = 0.1 # Minimum similarity to include
    enable_caching: bool = True           # Enable query result caching


@dataclass
class RetrievalResult:
    """Container for retrieval results with scoring metadata."""
    record: MemoryRecord
    relevance_score: float
    keyword_score: float = 0.0
    vector_score: float = 0.0
    tier_bonus: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Calculate total weighted score."""
        return self.relevance_score + self.tier_bonus


class KeywordSearcher:
    """Handles keyword-based search functionality."""
    
    def __init__(self, config: RetrievalConfig):
        """Initialize keyword searcher with configuration."""
        self.config = config
        logger.debug("Initialized KeywordSearcher")
    
    def search(self, query: str, records: RecordCollection) -> List[RetrievalResult]:
        """
        Perform keyword search on memory records.
        
        Args:
            query: Search query string
            records: Collection of memory records to search
            
        Returns:
            List of retrieval results with keyword scores
        """
        if not query.strip():
            return []
        
        query_terms = set(query.lower().split())
        results = []
        
        for record in records:
            score = self._calculate_keyword_score(query_terms, record)
            
            if score > 0:  # Only include records with keyword matches
                result = RetrievalResult(
                    record=record,
                    relevance_score=score * self.config.keyword_weight,
                    keyword_score=score
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.keyword_score, reverse=True)
    
    def _calculate_keyword_score(self, query_terms: Set[str], record: MemoryRecord) -> float:
        """
        Calculate keyword similarity score for a record.
        
        Args:
            query_terms: Set of lowercase query terms
            record: Memory record to score
            
        Returns:
            Keyword similarity score (0.0 to 1.0)
        """
        # Combine all searchable text from the record
        searchable_text = " ".join([
            record.payload,
            record.provenance.source,
            " ".join(record.tags)
        ]).lower()
        
        record_terms = set(searchable_text.split())
        
        if not record_terms:
            return 0.0
        
        # Calculate overlap ratio
        overlap = len(query_terms & record_terms)
        total_query_terms = len(query_terms)
        
        return overlap / total_query_terms if total_query_terms > 0 else 0.0


class VectorSearcher:
    """Handles vector-based similarity search functionality."""
    
    def __init__(self, config: RetrievalConfig):
        """Initialize vector searcher with configuration."""
        self.config = config
        self._embeddings_cache: Dict[str, List[float]] = {}
        logger.debug("Initialized VectorSearcher")
    
    def search(self, query: str, records: RecordCollection) -> List[RetrievalResult]:
        """
        Perform vector similarity search on memory records.
        
        Args:
            query: Search query string
            records: Collection of memory records to search
            
        Returns:
            List of retrieval results with vector scores
        """
        if not query.strip():
            return []
        
        # TODO: Implement actual vector embeddings with FAISS
        # For now, use dummy similarity based on text overlap
        query_embedding = self._get_dummy_embedding(query)
        results = []
        
        for record in records:
            record_embedding = self._get_dummy_embedding(record.payload)
            similarity = self._calculate_cosine_similarity(query_embedding, record_embedding)
            
            if similarity >= self.config.min_similarity_threshold:
                result = RetrievalResult(
                    record=record,
                    relevance_score=similarity * self.config.vector_weight,
                    vector_score=similarity
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.vector_score, reverse=True)
    
    def _get_dummy_embedding(self, text: str) -> List[float]:
        """
        Generate dummy embedding for text.
        
        TODO: Replace with actual embedding model (e.g., sentence-transformers)
        TODO: Integrate with FAISS for efficient similarity search
        
        Args:
            text: Input text to embed
            
        Returns:
            Dummy embedding vector
        """
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]
        
        # Create simple character-based dummy embedding
        embedding = []
        for i in range(0, min(len(text), 100), 10):
            char_sum = sum(ord(c) for c in text[i:i+10])
            embedding.append(char_sum / 1000.0)  # Normalize
        
        # Pad to fixed size
        while len(embedding) < 10:
            embedding.append(0.0)
        
        self._embeddings_cache[text] = embedding[:10]
        return embedding[:10]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Ensure same length
        min_len = min(len(vec1), len(vec2))
        vec1, vec2 = vec1[:min_len], vec2[:min_len]
        
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval system combining keyword and vector search.
    
    This retriever implements a multi-strategy approach to memory retrieval,
    combining the precision of keyword matching with the semantic understanding
    of vector similarity search for optimal relevance ranking.
    
    Key Features:
    - Dual-mode search: keyword + vector similarity
    - Configurable weighting between search strategies
    - Tier-aware scoring with configurable tier weights
    - Result caching for performance optimization
    - Extensible architecture for additional search methods
    """
    
    def __init__(self, 
                 memory_store,  # BaseMemoryStore - avoiding circular import
                 config: Optional[RetrievalConfig] = None):
        """
        Initialize hybrid retriever with dependency injection.
        
        Args:
            memory_store: Memory store instance for data access
            config: Retrieval configuration parameters
        """
        self.memory_store = memory_store
        self.config = config or RetrievalConfig()
        
        # Initialize search components
        self.keyword_searcher = KeywordSearcher(self.config)
        self.vector_searcher = VectorSearcher(self.config)
        
        # Performance tracking
        self._query_count = 0
        self._cache_hits = 0
        self._cache: Dict[str, RecordCollection] = {}
        
        logger.info(f"Initialized HybridRetriever with k={self.config.k}")
    
    def retrieve(self, 
                query: str,
                context: Optional[Dict[str, Any]] = None,
                tier_filter: Optional[Set[MemoryTier]] = None,
                limit: int = 10) -> RecordCollection:
        """
        Retrieve relevant memory records using hybrid search.
        
        This method combines keyword search and vector similarity search
        to find the most relevant records across all memory tiers,
        with configurable weighting and filtering.
        
        Args:
            query: Search query string
            context: Optional context for search refinement
            tier_filter: Optional set of memory tiers to search
            limit: Maximum number of results to return
            
        Returns:
            Ranked list of relevant memory records
            
        Raises:
            SearchError: If retrieval operation fails
        """
        start_time = time.perf_counter()
        self._query_count += 1
        
        try:
            logger.debug(f"Hybrid retrieval for query: {query[:50]}...")
            
            # Check cache if enabled
            cache_key = self._get_cache_key(query, tier_filter, limit)
            if self.config.enable_caching and cache_key in self._cache:
                self._cache_hits += 1
                logger.debug("Cache hit for retrieval query")
                return self._cache[cache_key]
            
            # Get candidate records from memory store
            candidate_records = self._get_candidate_records(tier_filter)
            logger.debug(f"Retrieved {len(candidate_records)} candidate records")
            
            if not candidate_records:
                return []
            
            # Perform keyword search
            keyword_results = self.keyword_searcher.search(query, candidate_records)
            
            # Perform vector search
            vector_results = self.vector_searcher.search(query, candidate_records)
            
            # Combine and rank results
            combined_results = self._combine_results(keyword_results, vector_results)
            
            # Apply tier bonuses
            for result in combined_results:
                tier_weight = self.config.tier_weights.get(result.record.tier, 1.0)
                result.tier_bonus = (tier_weight - 1.0) * 0.1  # Small tier bonus
            
            # Sort by total score and limit results
            final_results = sorted(combined_results, key=lambda x: x.total_score, reverse=True)
            final_records = [r.record for r in final_results[:limit]]
            
            # Cache results
            if self.config.enable_caching:
                self._cache[cache_key] = final_records
            
            retrieval_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Hybrid retrieval completed in {retrieval_time:.1f}ms, "
                        f"returned {len(final_records)} records")
            
            return final_records
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}", exc_info=True)
            raise SearchError(f"Retrieval operation failed: {e}") from e
    
    def _get_candidate_records(self, tier_filter: Optional[Set[MemoryTier]]) -> RecordCollection:
        """
        Get candidate records from memory store with optional tier filtering.
        
        Args:
            tier_filter: Optional set of tiers to search
            
        Returns:
            Collection of candidate memory records
        """
        if tier_filter:
            # Get records from specified tiers
            candidates = []
            for tier in tier_filter:
                tier_records = self.memory_store.get_by_tier(tier)
                candidates.extend(tier_records)
            return candidates
        else:
            # Get records from all tiers
            all_candidates = []
            for tier in MemoryTier:
                tier_records = self.memory_store.get_by_tier(tier)
                all_candidates.extend(tier_records)
            return all_candidates
    
    def _combine_results(self, 
                        keyword_results: List[RetrievalResult],
                        vector_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Combine keyword and vector search results.
        
        Args:
            keyword_results: Results from keyword search
            vector_results: Results from vector search
            
        Returns:
            Combined and deduplicated results
        """
        # Create mapping from record ID to result
        combined_map: Dict[str, RetrievalResult] = {}
        
        # Add keyword results
        for result in keyword_results:
            record_id = str(result.record.id)
            combined_map[record_id] = result
        
        # Merge vector results
        for result in vector_results:
            record_id = str(result.record.id)
            if record_id in combined_map:
                # Combine scores for records found by both methods
                existing = combined_map[record_id]
                existing.vector_score = result.vector_score
                existing.relevance_score = (
                    existing.keyword_score * self.config.keyword_weight +
                    result.vector_score * self.config.vector_weight
                )
            else:
                # Add new vector-only result
                combined_map[record_id] = result
        
        return list(combined_map.values())
    
    def _get_cache_key(self, 
                      query: str,
                      tier_filter: Optional[Set[MemoryTier]],
                      limit: int) -> str:
        """Generate cache key for query parameters."""
        tier_str = ",".join(sorted(t.value for t in tier_filter)) if tier_filter else "all"
        return f"{hash(query)}:{tier_str}:{limit}"
    
    def get_retrieval_metrics(self) -> Dict[str, Any]:
        """
        Get retrieval performance metrics.
        
        Returns:
            Dictionary of retrieval metrics and statistics
        """
        cache_hit_rate = self._cache_hits / max(self._query_count, 1)
        
        return {
            "total_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "config": {
                "k": self.config.k,
                "keyword_weight": self.config.keyword_weight,
                "vector_weight": self.config.vector_weight,
                "caching_enabled": self.config.enable_caching
            }
        }


# Factory function for easy instantiation
def create_hybrid_retriever(memory_store, 
                          config: Optional[Dict[str, Any]] = None) -> HybridRetriever:
    """
    Factory function for creating hybrid retriever instances.
    
    Args:
        memory_store: Memory store instance
        config: Configuration parameters dictionary
        
    Returns:
        Configured hybrid retriever instance
    """
    retrieval_config = RetrievalConfig()
    
    if config:
        # Update config with provided parameters
        for key, value in config.items():
            if hasattr(retrieval_config, key):
                setattr(retrieval_config, key, value)
    
    return HybridRetriever(memory_store, retrieval_config)
