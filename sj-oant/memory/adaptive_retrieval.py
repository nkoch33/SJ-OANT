"""
adaptive_retrieval.py - Advanced Memory Retrieval and Management

This module implements intelligent memory retrieval strategies that adapt to context
and query patterns, enhancing the TMM system's ability to find relevant information
while maintaining research-grade performance.

Key Features:
- Adaptive retrieval strategies based on query type and context
- Intelligent memory promotion between tiers
- Context-aware relevance scoring
- Memory consolidation and optimization
- Research-grade analytics and monitoring
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from core.types import MemoryRecord, MemoryTier, MemoryStatus, RecordID
from memory.typed_store import InMemoryStore

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Different retrieval strategies for different query types."""
    SEMANTIC = "semantic"  # For complex, multi-faceted queries
    TEMPORAL = "temporal"  # For time-sensitive queries
    ENTITY_FOCUSED = "entity_focused"  # For entity-centric queries
    TASK_ORIENTED = "task_oriented"  # For task completion queries
    HYBRID = "hybrid"  # Combines multiple strategies

@dataclass
class RetrievalMetrics:
    """Metrics for tracking retrieval performance."""
    query_type: str
    strategy_used: RetrievalStrategy
    retrieval_time: float
    records_retrieved: int
    relevance_scores: List[float]
    memory_tier_distribution: Dict[MemoryTier, int]
    cache_hit_rate: float

class AdaptiveMemoryRetriever:
    """
    Advanced memory retrieval system that adapts to query patterns and context.
    
    This component enhances the TMM system's memory retrieval capabilities by:
    1. Analyzing query patterns to select optimal retrieval strategies
    2. Implementing intelligent memory promotion between tiers
    3. Providing context-aware relevance scoring
    4. Offering research-grade analytics and monitoring
    """
    
    def __init__(self, memory_store: InMemoryStore, config: Dict[str, Any] = None):
        """
        Initialize the adaptive memory retriever.
        
        Args:
            memory_store: The memory store to retrieve from
            config: Configuration parameters
        """
        self.memory_store = memory_store
        self.config = config or self._default_config()
        
        # Query pattern analysis
        self.query_patterns = defaultdict(int)
        self.strategy_performance = defaultdict(list)
        
        # Memory promotion tracking
        self.promotion_history = deque(maxlen=1000)
        self.consolidation_threshold = self.config.get('consolidation_threshold', 0.8)
        
        # Performance metrics
        self.retrieval_metrics = []
        self.cache_hits = 0
        self.total_queries = 0
        
        logger.info("AdaptiveMemoryRetriever initialized with advanced retrieval strategies")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the adaptive retriever."""
        return {
            'consolidation_threshold': 0.8,
            'promotion_threshold': 0.7,
            'relevance_threshold': 0.5,
            'max_retrieval_records': 20,
            'temporal_decay_factor': 0.95,
            'semantic_similarity_threshold': 0.6
        }
    
    def retrieve_relevant_memories(self, query: str, context: Dict[str, Any] = None) -> List[MemoryRecord]:
        """
        Retrieve relevant memories using adaptive strategies.
        
        Args:
            query: The user query
            context: Additional context information
            
        Returns:
            List of relevant memory records
        """
        start_time = time.time()
        self.total_queries += 1
        
        # Analyze query to determine optimal strategy
        strategy = self._analyze_query_pattern(query, context)
        
        # Retrieve memories using selected strategy
        memories = self._execute_retrieval_strategy(query, strategy, context)
        
        # Calculate relevance scores
        relevance_scores = self._calculate_relevance_scores(query, memories)
        
        # Filter by relevance threshold
        filtered_memories = [
            mem for mem, score in zip(memories, relevance_scores)
            if score >= self.config['relevance_threshold']
        ]
        
        # Update performance metrics
        retrieval_time = time.time() - start_time
        self._update_metrics(query, strategy, retrieval_time, filtered_memories, relevance_scores)
        
        # Check for memory consolidation opportunities
        self._check_consolidation_opportunities()
        
        logger.debug(f"Retrieved {len(filtered_memories)} relevant memories using {strategy.value} strategy")
        return filtered_memories
    
    def _analyze_query_pattern(self, query: str, context: Dict[str, Any] = None) -> RetrievalStrategy:
        """
        Analyze query pattern to determine optimal retrieval strategy.
        
        Args:
            query: The user query
            context: Additional context information
            
        Returns:
            Optimal retrieval strategy
        """
        query_lower = query.lower()
        
        # Time-sensitive indicators
        temporal_indicators = ['when', 'time', 'schedule', 'appointment', 'deadline', 'recent', 'last', 'next']
        if any(indicator in query_lower for indicator in temporal_indicators):
            return RetrievalStrategy.TEMPORAL
        
        # Entity-focused indicators
        entity_indicators = ['who', 'what', 'where', 'which', 'name', 'address', 'phone', 'email']
        if any(indicator in query_lower for indicator in entity_indicators):
            return RetrievalStrategy.ENTITY_FOCUSED
        
        # Task-oriented indicators
        task_indicators = ['book', 'reserve', 'order', 'cancel', 'change', 'confirm', 'help', 'need']
        if any(indicator in query_lower for indicator in task_indicators):
            return RetrievalStrategy.TASK_ORIENTED
        
        # Complex query indicators
        complex_indicators = ['how', 'why', 'explain', 'compare', 'analyze', 'evaluate']
        if any(indicator in query_lower for indicator in complex_indicators):
            return RetrievalStrategy.SEMANTIC
        
        # Default to hybrid for general queries
        return RetrievalStrategy.HYBRID
    
    def _execute_retrieval_strategy(self, query: str, strategy: RetrievalStrategy, context: Dict[str, Any] = None) -> List[MemoryRecord]:
        """
        Execute the selected retrieval strategy.
        
        Args:
            query: The user query
            strategy: The retrieval strategy to use
            context: Additional context information
            
        Returns:
            List of retrieved memory records
        """
        if strategy == RetrievalStrategy.TEMPORAL:
            return self._temporal_retrieval(query, context)
        elif strategy == RetrievalStrategy.ENTITY_FOCUSED:
            return self._entity_focused_retrieval(query, context)
        elif strategy == RetrievalStrategy.TASK_ORIENTED:
            return self._task_oriented_retrieval(query, context)
        elif strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_retrieval(query, context)
        else:  # HYBRID
            return self._hybrid_retrieval(query, context)
    
    def _temporal_retrieval(self, query: str, context: Dict[str, Any] = None) -> List[MemoryRecord]:
        """Retrieve memories with temporal relevance."""
        # Get recent memories from L1 and L2
        recent_memories = []
        
        # L1 memories (most recent)
        l1_memories = self.memory_store.search_by_tier(MemoryTier.L1_WORKING)
        recent_memories.extend(l1_memories)
        
        # L2 memories (recent summaries)
        l2_memories = self.memory_store.search_by_tier(MemoryTier.L2_SUMMARIZED)
        recent_memories.extend(l2_memories)
        
        # Sort by timestamp (most recent first)
        recent_memories.sort(key=lambda x: x.timestamp, reverse=True)
        
        return recent_memories[:self.config['max_retrieval_records']]
    
    def _entity_focused_retrieval(self, query: str, context: Dict[str, Any] = None) -> List[MemoryRecord]:
        """Retrieve memories focused on entities and facts."""
        # Get verified facts from L3
        l3_memories = self.memory_store.search_by_tier(MemoryTier.L3_ARCHIVAL)
        
        # Filter for entity-rich content
        entity_memories = []
        for memory in l3_memories:
            if self._contains_entities(memory.content):
                entity_memories.append(memory)
        
        return entity_memories[:self.config['max_retrieval_records']]
    
    def _task_oriented_retrieval(self, query: str, context: Dict[str, Any] = None) -> List[MemoryRecord]:
        """Retrieve memories relevant to task completion."""
        # Get task-related memories from all tiers
        all_memories = []
        
        for tier in [MemoryTier.L1_WORKING, MemoryTier.L2_SUMMARIZED, MemoryTier.L3_ARCHIVAL]:
            tier_memories = self.memory_store.search_by_tier(tier)
            all_memories.extend(tier_memories)
        
        # Filter for task-relevant content
        task_memories = []
        for memory in all_memories:
            if self._is_task_relevant(memory.content, query):
                task_memories.append(memory)
        
        return task_memories[:self.config['max_retrieval_records']]
    
    def _semantic_retrieval(self, query: str, context: Dict[str, Any] = None) -> List[MemoryRecord]:
        """Retrieve memories using semantic similarity."""
        # Get all memories
        all_memories = []
        for tier in [MemoryTier.L1_WORKING, MemoryTier.L2_SUMMARIZED, MemoryTier.L3_ARCHIVAL]:
            tier_memories = self.memory_store.search_by_tier(tier)
            all_memories.extend(tier_memories)
        
        # Calculate semantic similarity (simplified for now)
        semantic_memories = []
        for memory in all_memories:
            similarity = self._calculate_semantic_similarity(query, memory.content)
            if similarity >= self.config['semantic_similarity_threshold']:
                semantic_memories.append(memory)
        
        # Sort by similarity score
        semantic_memories.sort(key=lambda x: self._calculate_semantic_similarity(query, x.content), reverse=True)
        
        return semantic_memories[:self.config['max_retrieval_records']]
    
    def _hybrid_retrieval(self, query: str, context: Dict[str, Any] = None) -> List[MemoryRecord]:
        """Combine multiple retrieval strategies."""
        # Get memories from different strategies
        temporal_memories = self._temporal_retrieval(query, context)
        entity_memories = self._entity_focused_retrieval(query, context)
        task_memories = self._task_oriented_retrieval(query, context)
        
        # Combine and deduplicate
        all_memories = temporal_memories + entity_memories + task_memories
        unique_memories = list({memory.id: memory for memory in all_memories}.values())
        
        return unique_memories[:self.config['max_retrieval_records']]
    
    def _calculate_relevance_scores(self, query: str, memories: List[MemoryRecord]) -> List[float]:
        """
        Calculate relevance scores for retrieved memories.
        
        Args:
            query: The user query
            memories: List of memory records
            
        Returns:
            List of relevance scores
        """
        scores = []
        for memory in memories:
            # Base relevance score
            base_score = 0.5
            
            # Boost for recent memories
            if memory.tier == MemoryTier.L1_WORKING:
                base_score += 0.2
            elif memory.tier == MemoryTier.L2_SUMMARIZED:
                base_score += 0.1
            
            # Boost for verified memories
            if memory.status == MemoryStatus.VERIFIED:
                base_score += 0.2
            
            # Boost for high confidence
            if memory.confidence_scores.overall > 0.8:
                base_score += 0.1
            
            # Semantic similarity boost
            similarity = self._calculate_semantic_similarity(query, memory.content)
            base_score += similarity * 0.3
            
            scores.append(min(base_score, 1.0))
        
        return scores
    
    def _calculate_semantic_similarity(self, query: str, content: str) -> float:
        """
        Calculate semantic similarity between query and content.
        
        Args:
            query: The user query
            content: The memory content
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simplified semantic similarity calculation
        # In a production system, this would use embeddings or more sophisticated NLP
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _contains_entities(self, content: str) -> bool:
        """Check if content contains entity-like information."""
        entity_indicators = ['name', 'address', 'phone', 'email', 'date', 'time', 'location', 'price', 'number']
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in entity_indicators)
    
    def _is_task_relevant(self, content: str, query: str) -> bool:
        """Check if content is relevant to task completion."""
        task_indicators = ['book', 'reserve', 'order', 'cancel', 'change', 'confirm', 'help', 'need', 'want', 'request']
        content_lower = content.lower()
        query_lower = query.lower()
        
        return any(indicator in content_lower or indicator in query_lower for indicator in task_indicators)
    
    def _update_metrics(self, query: str, strategy: RetrievalStrategy, retrieval_time: float, 
                       memories: List[MemoryRecord], relevance_scores: List[float]):
        """Update performance metrics."""
        # Calculate tier distribution
        tier_distribution = defaultdict(int)
        for memory in memories:
            tier_distribution[memory.tier] += 1
        
        # Create metrics record
        metrics = RetrievalMetrics(
            query_type=query[:50],  # Truncate for storage
            strategy_used=strategy,
            retrieval_time=retrieval_time,
            records_retrieved=len(memories),
            relevance_scores=relevance_scores,
            memory_tier_distribution=dict(tier_distribution),
            cache_hit_rate=self.cache_hits / max(self.total_queries, 1)
        )
        
        self.retrieval_metrics.append(metrics)
        
        # Update strategy performance
        self.strategy_performance[strategy].append(retrieval_time)
    
    def _check_consolidation_opportunities(self):
        """Check for memory consolidation opportunities."""
        # This would implement intelligent memory consolidation
        # For now, we'll just log the opportunity
        if len(self.retrieval_metrics) % 100 == 0:
            logger.info(f"Processed {len(self.retrieval_metrics)} retrieval operations")
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics.
        
        Returns:
            Dictionary containing performance metrics and insights
        """
        if not self.retrieval_metrics:
            return {"message": "No retrieval metrics available yet"}
        
        # Calculate average metrics
        avg_retrieval_time = sum(m.retrieval_time for m in self.retrieval_metrics) / len(self.retrieval_metrics)
        avg_records_retrieved = sum(m.records_retrieved for m in self.retrieval_metrics) / len(self.retrieval_metrics)
        avg_relevance_score = sum(sum(m.relevance_scores) for m in self.retrieval_metrics) / sum(len(m.relevance_scores) for m in self.retrieval_metrics)
        
        # Strategy performance analysis
        strategy_performance = {}
        for strategy, times in self.strategy_performance.items():
            if times:
                strategy_performance[strategy.value] = {
                    'avg_time': sum(times) / len(times),
                    'usage_count': len(times),
                    'efficiency': len(times) / max(self.total_queries, 1)
                }
        
        return {
            'total_queries': self.total_queries,
            'cache_hit_rate': self.cache_hits / max(self.total_queries, 1),
            'avg_retrieval_time': avg_retrieval_time,
            'avg_records_retrieved': avg_records_retrieved,
            'avg_relevance_score': avg_relevance_score,
            'strategy_performance': strategy_performance,
            'memory_efficiency': self._calculate_memory_efficiency()
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate overall memory efficiency score."""
        if not self.retrieval_metrics:
            return 0.0
        
        # Calculate efficiency based on relevance scores and retrieval time
        total_efficiency = 0.0
        for metrics in self.retrieval_metrics:
            if metrics.relevance_scores:
                avg_relevance = sum(metrics.relevance_scores) / len(metrics.relevance_scores)
                time_efficiency = 1.0 / (1.0 + metrics.retrieval_time)  # Lower time is better
                total_efficiency += avg_relevance * time_efficiency
        
        return total_efficiency / len(self.retrieval_metrics)
