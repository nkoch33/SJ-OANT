"""
retrieval.active_retrieval - Active and Contextual Retrieval System

This module implements an active retrieval system that performs intelligent,
context-aware memory retrieval with iterative refinement capabilities.

Key Features:
- Context-aware retrieval based on conversation state
- Query refinement with feedback loops
- Multi-hop reasoning support for complex queries
- Adaptive retrieval strategy selection
- Learning from retrieval performance

The active retriever enhances basic retrieval with intelligence
and adaptability, optimizing retrieval based on context and performance.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum

from core.ports import BaseRetriever, SearchError
from core.types import MemoryRecord, MemoryTier, RecordCollection

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Enumeration of query complexity levels for retrieval strategy selection."""
    SIMPLE = "simple"          # Direct factual queries
    MODERATE = "moderate"      # Queries requiring some reasoning
    COMPLEX = "complex"        # Multi-step reasoning required
    RESEARCH = "research"      # Extensive exploration needed


@dataclass
class ActiveRetrievalConfig:
    """Configuration parameters for active retrieval."""
    max_iterations: int = 3                    # Maximum refinement iterations
    feedback_threshold: float = 0.5           # Minimum satisfaction for early stop
    context_window_size: int = 5              # Number of previous queries to consider
    enable_multi_hop: bool = True             # Enable multi-hop reasoning
    complexity_adaptation: bool = True        # Adapt strategy by complexity
    learning_rate: float = 0.1                # Rate for adaptive learning
    min_confidence_threshold: float = 0.3     # Minimum confidence to return results


@dataclass
class RetrievalFeedback:
    """Container for retrieval feedback and performance metrics."""
    satisfaction_score: float       # User/system satisfaction with results
    relevance_scores: List[float]   # Relevance scores for returned records
    iteration: int                  # Which iteration this feedback is from
    query_modification: str = ""    # Suggested query modification
    strategy_recommendation: str = "" # Recommended strategy adjustment


@dataclass
class QueryContext:
    """Context information for active retrieval."""
    previous_queries: List[str] = field(default_factory=list)
    previous_results: List[RecordCollection] = field(default_factory=list)
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    user_intent: str = ""
    complexity_level: QueryComplexity = QueryComplexity.SIMPLE


class QueryRefiner:
    """Handles query refinement and expansion for better retrieval."""
    
    def __init__(self, config: ActiveRetrievalConfig):
        """Initialize query refiner with configuration."""
        self.config = config
        self._refinement_history: Dict[str, List[str]] = {}
        logger.debug("Initialized QueryRefiner")
    
    def refine(self, query: str, feedback: Optional[RetrievalFeedback] = None,
               context: Optional[QueryContext] = None) -> str:
        """
        Refine query based on feedback and context.
        
        Args:
            query: Original query string
            feedback: Optional feedback from previous retrieval attempts
            context: Optional context information
            
        Returns:
            Refined query string
        """
        refined_query = query.strip()
        
        # Track refinement history
        if query not in self._refinement_history:
            self._refinement_history[query] = []
        
        # Apply feedback-based refinement
        if feedback and feedback.query_modification:
            refined_query = feedback.query_modification
            logger.debug(f"Applied feedback refinement: {refined_query}")
        
        # Apply context-based expansion
        if context:
            refined_query = self._apply_context_expansion(refined_query, context)
        
        # Apply complexity-based refinement
        complexity = context.complexity_level if context else QueryComplexity.SIMPLE
        refined_query = self._apply_complexity_refinement(refined_query, complexity)
        
        # Store refinement history
        if refined_query != query:
            self._refinement_history[query].append(refined_query)
        
        return refined_query
    
    def _apply_context_expansion(self, query: str, context: QueryContext) -> str:
        """
        Apply context-based query expansion.
        
        Args:
            query: Original query
            context: Query context information
            
        Returns:
            Context-expanded query
        """
        expanded_terms = []
        
        # Add terms from previous successful queries
        for prev_query in context.previous_queries[-2:]:  # Last 2 queries
            prev_terms = set(prev_query.lower().split())
            query_terms = set(query.lower().split())
            
            # Find related terms not already in query
            related_terms = prev_terms - query_terms
            if related_terms and len(expanded_terms) < 3:  # Limit expansion
                expanded_terms.extend(list(related_terms)[:3-len(expanded_terms)])
        
        # Add intent-based terms
        if context.user_intent:
            intent_terms = context.user_intent.lower().split()
            for term in intent_terms:
                if term not in query.lower() and len(expanded_terms) < 5:
                    expanded_terms.append(term)
        
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            logger.debug(f"Context expansion: {query} -> {expanded_query}")
            return expanded_query
        
        return query
    
    def _apply_complexity_refinement(self, query: str, complexity: QueryComplexity) -> str:
        """
        Apply complexity-based query refinement.
        
        Args:
            query: Original query
            complexity: Query complexity level
            
        Returns:
            Complexity-refined query
        """
        if complexity == QueryComplexity.SIMPLE:
            # For simple queries, keep as-is
            return query
        
        elif complexity == QueryComplexity.MODERATE:
            # Add reasoning keywords for moderate complexity
            if not any(word in query.lower() for word in ["how", "why", "explain"]):
                return f"explain {query}"
        
        elif complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH]:
            # Add analytical keywords for complex queries
            if not any(word in query.lower() for word in ["analyze", "compare", "relationship"]):
                return f"analyze {query} and related concepts"
        
        return query


class ActiveRetriever(BaseRetriever):
    """
    Active retrieval system with context awareness and iterative refinement.
    
    This retriever implements intelligent, adaptive retrieval strategies that
    learn from feedback and context to improve retrieval quality over time.
    
    Key Features:
    - Iterative query refinement based on retrieval feedback
    - Context-aware retrieval using conversation history
    - Multi-hop reasoning for complex information needs
    - Adaptive strategy selection based on query complexity
    - Performance learning and optimization
    """
    
    def __init__(self,
                 base_retriever: BaseRetriever,
                 config: Optional[ActiveRetrievalConfig] = None):
        """
        Initialize active retriever with dependency injection.
        
        Args:
            base_retriever: Base retriever to enhance with active capabilities
            config: Active retrieval configuration parameters
        """
        self.base_retriever = base_retriever
        self.config = config or ActiveRetrievalConfig()
        
        # Initialize components
        self.query_refiner = QueryRefiner(self.config)
        
        # Performance tracking
        self._retrieval_count = 0
        self._successful_retrievals = 0
        self._average_iterations = 0.0
        self._query_contexts: Dict[str, QueryContext] = {}
        
        logger.info(f"Initialized ActiveRetriever with max_iterations={self.config.max_iterations}")
    
    def retrieve(self,
                query: str,
                context: Optional[Dict[str, Any]] = None,
                tier_filter: Optional[Set[MemoryTier]] = None,
                limit: int = 10) -> RecordCollection:
        """
        Retrieve relevant memory records using active retrieval with refinement.
        
        This method implements iterative retrieval with query refinement,
        context awareness, and adaptive strategy selection for optimal results.
        
        Args:
            query: Search query string
            context: Optional context for retrieval
            tier_filter: Optional set of memory tiers to search
            limit: Maximum number of results to return
            
        Returns:
            Ranked list of relevant memory records
            
        Raises:
            SearchError: If retrieval operation fails
        """
        start_time = time.perf_counter()
        self._retrieval_count += 1
        
        try:
            logger.debug(f"Active retrieval for query: {query[:50]}...")
            
            # Build query context
            query_context = self._build_query_context(query, context)
            
            # Determine query complexity
            complexity = self._assess_query_complexity(query, query_context)
            query_context.complexity_level = complexity
            
            # Perform iterative retrieval with refinement
            best_results = []
            current_query = query
            iteration = 0
            
            for iteration in range(self.config.max_iterations):
                logger.debug(f"Active retrieval iteration {iteration + 1}: {current_query[:50]}...")
                
                # Retrieve with current query
                results = self.base_retriever.retrieve(
                    current_query, context, tier_filter, limit
                )
                
                # Evaluate results quality
                quality_score = self._evaluate_results_quality(results, query, query_context)
                
                # Check if results are satisfactory
                if quality_score >= self.config.feedback_threshold or iteration == 0:
                    best_results = results
                    
                    # Early stopping if quality is good enough
                    if quality_score >= self.config.feedback_threshold:
                        logger.debug(f"Early stopping at iteration {iteration + 1} (quality: {quality_score:.2f})")
                        break
                
                # Generate feedback for next iteration
                if iteration < self.config.max_iterations - 1:
                    feedback = self._generate_feedback(results, quality_score, iteration + 1)
                    
                    # Refine query for next iteration
                    current_query = self.query_refiner.refine(current_query, feedback, query_context)
                    
                    # Avoid infinite loops with same query
                    if current_query == query and iteration > 0:
                        break
            
            # Update query context history
            self._update_query_context(query, best_results, query_context)
            
            # Update performance metrics
            self._update_metrics(iteration + 1, len(best_results) > 0)
            
            retrieval_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Active retrieval completed in {iteration + 1} iterations, "
                        f"{retrieval_time:.1f}ms, returned {len(best_results)} records")
            
            return best_results
            
        except Exception as e:
            logger.error(f"Active retrieval failed: {e}", exc_info=True)
            raise SearchError(f"Active retrieval operation failed: {e}") from e
    
    def refine(self, query: str, feedback: Dict[str, Any]) -> str:
        """
        Refine query based on explicit feedback.
        
        This method allows external systems to provide feedback on retrieval
        quality and get an improved query for better results.
        
        Args:
            query: Original query string
            feedback: Feedback dictionary containing satisfaction scores and suggestions
            
        Returns:
            Refined query string
        """
        # Convert feedback dict to RetrievalFeedback object
        retrieval_feedback = RetrievalFeedback(
            satisfaction_score=feedback.get("satisfaction_score", 0.5),
            relevance_scores=feedback.get("relevance_scores", []),
            iteration=feedback.get("iteration", 1),
            query_modification=feedback.get("query_modification", ""),
            strategy_recommendation=feedback.get("strategy_recommendation", "")
        )
        
        # Get context if available
        context = self._query_contexts.get(query, QueryContext())
        
        # Apply refinement
        refined_query = self.query_refiner.refine(query, retrieval_feedback, context)
        
        logger.debug(f"Manual query refinement: {query} -> {refined_query}")
        return refined_query
    
    def _build_query_context(self, query: str, context: Optional[Dict[str, Any]]) -> QueryContext:
        """Build query context from available information."""
        query_context = QueryContext()
        
        if context:
            query_context.conversation_state = context.get("conversation_state", {})
            query_context.user_intent = context.get("user_intent", "")
        
        # Add previous queries from context history
        for prev_query, prev_context in list(self._query_contexts.items())[-self.config.context_window_size:]:
            query_context.previous_queries.append(prev_query)
        
        return query_context
    
    def _assess_query_complexity(self, query: str, context: QueryContext) -> QueryComplexity:
        """Assess the complexity level of a query."""
        query_lower = query.lower()
        
        # Research-level indicators
        if any(word in query_lower for word in ["research", "comprehensive", "analyze thoroughly"]):
            return QueryComplexity.RESEARCH
        
        # Complex reasoning indicators
        if any(word in query_lower for word in ["analyze", "compare", "relationship", "implications"]):
            return QueryComplexity.COMPLEX
        
        # Moderate complexity indicators
        if any(word in query_lower for word in ["how", "why", "explain", "describe"]):
            return QueryComplexity.MODERATE
        
        # Consider context
        if len(context.previous_queries) > 2:
            return QueryComplexity.MODERATE  # Follow-up questions are often more complex
        
        return QueryComplexity.SIMPLE
    
    def _evaluate_results_quality(self, results: RecordCollection, 
                                 original_query: str, context: QueryContext) -> float:
        """
        Evaluate the quality of retrieval results.
        
        Args:
            results: Retrieved memory records
            original_query: Original user query
            context: Query context information
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        # Basic quality indicators
        result_count_score = min(len(results) / 5.0, 1.0)  # Prefer 5+ results
        
        # Content relevance score (simplified)
        query_terms = set(original_query.lower().split())
        relevance_scores = []
        
        for record in results:
            record_terms = set(record.payload.lower().split())
            overlap = len(query_terms & record_terms)
            relevance = overlap / len(query_terms) if query_terms else 0.0
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Combine scores
        quality_score = (result_count_score * 0.3 + avg_relevance * 0.7)
        
        return min(quality_score, 1.0)
    
    def _generate_feedback(self, results: RecordCollection, 
                          quality_score: float, iteration: int) -> RetrievalFeedback:
        """Generate feedback for query refinement."""
        query_modification = ""
        strategy_recommendation = ""
        
        # Generate suggestions based on quality
        if quality_score < 0.3:
            query_modification = "expand search terms"
            strategy_recommendation = "broaden_search"
        elif quality_score < 0.5:
            query_modification = "refine search terms"
            strategy_recommendation = "adjust_terms"
        
        return RetrievalFeedback(
            satisfaction_score=quality_score,
            relevance_scores=[0.5] * len(results),  # Simplified
            iteration=iteration,
            query_modification=query_modification,
            strategy_recommendation=strategy_recommendation
        )
    
    def _update_query_context(self, query: str, results: RecordCollection, context: QueryContext):
        """Update query context with results."""
        context.previous_results.append(results)
        self._query_contexts[query] = context
        
        # Limit context history size
        if len(self._query_contexts) > 50:
            # Remove oldest entries
            oldest_keys = list(self._query_contexts.keys())[:-40]
            for key in oldest_keys:
                del self._query_contexts[key]
    
    def _update_metrics(self, iterations_used: int, success: bool):
        """Update performance metrics."""
        if success:
            self._successful_retrievals += 1
        
        # Update running average for iterations
        total_iterations = self._average_iterations * (self._retrieval_count - 1)
        self._average_iterations = (total_iterations + iterations_used) / self._retrieval_count
    
    def get_active_metrics(self) -> Dict[str, Any]:
        """
        Get active retrieval performance metrics.
        
        Returns:
            Dictionary of active retrieval metrics and statistics
        """
        success_rate = self._successful_retrievals / max(self._retrieval_count, 1)
        
        return {
            "total_retrievals": self._retrieval_count,
            "successful_retrievals": self._successful_retrievals,
            "success_rate": success_rate,
            "average_iterations": self._average_iterations,
            "query_contexts_stored": len(self._query_contexts),
            "config": {
                "max_iterations": self.config.max_iterations,
                "feedback_threshold": self.config.feedback_threshold,
                "multi_hop_enabled": self.config.enable_multi_hop
            }
        }


# Factory function for easy instantiation
def create_active_retriever(base_retriever: BaseRetriever,
                          config: Optional[Dict[str, Any]] = None) -> ActiveRetriever:
    """
    Factory function for creating active retriever instances.
    
    Args:
        base_retriever: Base retriever to enhance
        config: Configuration parameters dictionary
        
    Returns:
        Configured active retriever instance
    """
    active_config = ActiveRetrievalConfig()
    
    if config:
        # Update config with provided parameters
        for key, value in config.items():
            if hasattr(active_config, key):
                setattr(active_config, key, value)
    
    return ActiveRetriever(base_retriever, active_config)
