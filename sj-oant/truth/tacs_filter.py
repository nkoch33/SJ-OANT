"""
tacs_filter.py - Token-level Adaptive Context Screening Filter

This module contains the TACS (Token-level Adaptive Context Screening) filter
responsible for context filtering in the Truth-Maintained Memory (TMM) system.

The TACS filter:

1. Performs token-level and span-level gating on incoming context
2. Down-weights or removes misleading, irrelevant, or off-task content
3. Identifies and filters out potential distractors and noise
4. Preserves relevant information while removing confounding elements
5. Adapts filtering thresholds based on context and query type

Key responsibilities:
- Memory retrieval from storage based on query relevance
- Token-level relevance scoring and filtering
- Span-level coherence analysis
- Distractor and noise detection
- Adaptive threshold management
- Context cleaning and preparation for verification

The TACS filter is the first line of defense against false memory formation,
preventing irrelevant or misleading information from reaching the verification
stage and potentially corrupting the memory store.
"""

from typing import Dict, Any, List, Optional
import json
from langchain_core.prompts import ChatPromptTemplate
from memory.typed_store import MemoryState, InMemoryStore
from core.types import MemoryTier


class MemoryRetriever:
    """
    Component of TACS filter responsible for retrieving relevant memory from storage.
    
    This is the critical missing piece that was causing 0 retrievals - we need to
    actually search the memory store for relevant information based on the user query.
    """
    
    def __init__(self, memory_store: InMemoryStore, retrieval_limit: int = 10):
        """
        Initialize the memory retriever.
        
        Args:
            memory_store: Reference to the memory store for retrieval
            retrieval_limit: Maximum number of records to retrieve per query
        """
        self.memory_store = memory_store
        self.retrieval_limit = retrieval_limit
    
    def retrieve_relevant_memory(self, user_input: str) -> List[Any]:
        """
        Retrieve relevant memory records from the store based on user input.
        
        Args:
            user_input: User's current input to match against stored memory
            
        Returns:
            List of relevant MemoryRecord objects
        """
        print("🔍 Memory Retriever: Searching for relevant memory...")
        
        try:
            # Search across all tiers for relevant content
            relevant_records = self.memory_store.search(
                query=user_input,
                limit=self.retrieval_limit
            )
            
            print(f"   Retrieved {len(relevant_records)} relevant memory records")
            
            # Log what we found
            for i, record in enumerate(relevant_records[:3]):  # Show first 3
                print(f"   {i+1}. [{record.tier.value}] {record.payload[:50]}...")
            
            if len(relevant_records) > 3:
                print(f"   ... and {len(relevant_records) - 3} more")
            
            return relevant_records
            
        except Exception as e:
            print(f"   Error retrieving memory: {e}")
            return []
    
    def execute(self, state: MemoryState) -> MemoryState:
        """
        Execute memory retrieval and update state with retrieved content.
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with retrieved memory records
        """
        # Retrieve relevant memory from store
        retrieved_records = self.retrieve_relevant_memory(state["user_input"])
        
        # Add retrieved records to state for downstream processing
        # Convert to the format expected by the state
        updated_state = state.copy()
        
        # Add retrieved records to appropriate tiers in state
        for record in retrieved_records:
            if record.tier == MemoryTier.L1_WORKING:
                updated_state.setdefault("L1", []).append(record)
            elif record.tier == MemoryTier.L2_SUMMARIZED:
                updated_state.setdefault("L2", []).append(record)
            elif record.tier == MemoryTier.L3_ARCHIVAL:
                updated_state.setdefault("L3", []).append(record)
            elif record.tier == MemoryTier.FLAGGED:
                updated_state.setdefault("flagged", []).append(record)
        
        return updated_state


class RedundancyFilter:
    """
    Component of TACS filter that detects and handles redundant information.
    
    This filter compares new input against recent memory history to identify
    and discard redundant information, preventing memory bloat and maintaining
    efficiency in the TMM system.
    """
    
    def __init__(self, llm):
        """
        Initialize the redundancy filter.
        
        Args:
            llm: Language model instance for redundancy detection
        """
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("human", """You are the Redundancy Filter Agent. Compare new input against recent history
            to identify and discard redundant information.

            Recent L1 Cache: {l1_cache}

            Return JSON with:
            - "is_redundant": boolean
            - "confidence": float (0-1)
            - "reasoning": string
            - "unique_elements": list of unique information if not redundant

            New input to check: {input}""")
        ])
    
    def check_redundancy(self, new_input: str, l1_cache: List[str]) -> Dict[str, Any]:
        """
        Check if new input is redundant with recent memory.
        
        Args:
            new_input: New information to check
            l1_cache: Recent memory entries for comparison
            
        Returns:
            Dict with redundancy analysis results
        """
        # Format recent cache for comparison
        cache_context = "\n".join(l1_cache) if l1_cache else "Empty"
        
        # TODO: Invoke LLM for sophisticated redundancy detection
        # For now, implement basic string similarity
        
        redundancy_result = {
            "is_redundant": False,
            "confidence": 0.0,
            "reasoning": "No redundancy detected",
            "unique_elements": [new_input]
        }
        
        # Basic redundancy check
        if l1_cache:
            for cached_item in l1_cache:
                # Simple similarity check
                if self._simple_similarity(new_input, cached_item) > 0.8:
                    redundancy_result.update({
                        "is_redundant": True,
                        "confidence": 0.9,
                        "reasoning": f"High similarity with cached item: {cached_item[:50]}...",
                        "unique_elements": []
                    })
                    break
        
        return redundancy_result
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Basic word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def execute(self, state: MemoryState) -> MemoryState:
        """
        Execute redundancy filtering as part of the TMM pipeline.
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with redundancy analysis
        """
        print("🔍 Redundancy Filter: Checking for duplicate information...")
        
        # Check redundancy against L1 cache
        redundancy_result = self.check_redundancy(
            state["user_input"], 
            state["L1"]
        )
        
        print(f"   Redundancy check: {redundancy_result['reasoning']}")
        
        # If redundant, we might skip downstream processing
        # For now, just log and continue
        updated_state = state.copy()
        
        # TODO: Add redundancy metadata to state for downstream decisions
        
        return updated_state


class ContextualRelevanceFilter:
    """
    Component of TACS filter that scores contextual relevance.
    
    This filter analyzes the relevance of information to the current
    conversation context and query intent, filtering out off-task content.
    """
    
    def __init__(self, relevance_threshold: float = 0.5):
        """
        Initialize the contextual relevance filter.
        
        Args:
            relevance_threshold: Minimum relevance score to pass filter
        """
        self.relevance_threshold = relevance_threshold
    
    def score_relevance(self, content: str, context: Dict[str, Any]) -> float:
        """
        Score the relevance of content to the current context.
        
        Args:
            content: Content to score
            context: Current conversation context
            
        Returns:
            Relevance score between 0 and 1
        """
        # TODO: Implement sophisticated relevance scoring
        # For now, basic heuristics
        
        score = 0.5  # Neutral baseline
        
        # Boost score for question-related content if current input is a question
        if "?" in context.get("user_input", ""):
            if any(word in content.lower() for word in ["what", "when", "where", "why", "how"]):
                score += 0.2
        
        # Boost for topic continuity (basic keyword matching)
        context_words = set(context.get("user_input", "").lower().split())
        content_words = set(content.lower().split())
        overlap = len(context_words & content_words) / max(len(context_words), 1)
        score += overlap * 0.3
        
        return min(score, 1.0)
    
    def filter_by_relevance(self, content_list: List[str], context: Dict[str, Any]) -> List[str]:
        """
        Filter a list of content by relevance scores.
        
        Args:
            content_list: List of content to filter
            context: Current conversation context
            
        Returns:
            Filtered list of relevant content
        """
        relevant_content = []
        
        for content in content_list:
            relevance_score = self.score_relevance(content, context)
            if relevance_score >= self.relevance_threshold:
                relevant_content.append(content)
                print(f"   Keeping content (relevance: {relevance_score:.2f}): {content[:50]}...")
            else:
                print(f"   Filtering out (relevance: {relevance_score:.2f}): {content[:50]}...")
        
        return relevant_content


class TACSFilter:
    """
    Token-level Adaptive Context Screening Filter.
    
    Main TACS filter that coordinates memory retrieval, redundancy filtering, 
    relevance scoring, and other context screening operations to prepare 
    clean input for verification.
    
    This is the FIXED version that includes memory retrieval functionality.
    """
    
    def __init__(self, llm, memory_store: InMemoryStore, relevance_threshold: float = 0.5):
        """
        Initialize the TACS filter with all sub-components.
        
        Args:
            llm: Language model instance
            memory_store: Reference to memory store for retrieval
            relevance_threshold: Minimum relevance score for content filtering
        """
        self.memory_retriever = MemoryRetriever(memory_store)
        self.redundancy_filter = RedundancyFilter(llm)
        self.relevance_filter = ContextualRelevanceFilter(relevance_threshold)
    
    def execute(self, state: MemoryState) -> MemoryState:
        """
        Execute the complete TACS filtering pipeline with memory retrieval.
        
        This is the FIXED version that actually retrieves relevant memory
        from the store instead of just filtering existing state.
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with retrieved and filtered context
        """
        print("🎯 TACS Filter: Screening context and filtering noise...")
        
        # STEP 1: Retrieve relevant memory from store (THIS WAS MISSING!)
        print("   Step 1: Retrieving relevant memory from store...")
        state = self.memory_retriever.execute(state)
        
        # STEP 2: Check for redundancy
        print("   Step 2: Checking for redundancy...")
        # Convert MemoryRecord objects to strings for redundancy check
        l1_strings = [record.payload if hasattr(record, 'payload') else str(record) for record in state.get("L1", [])]
        redundancy_state = {
            "user_input": state["user_input"],
            "L1": l1_strings
        }
        redundancy_state = self.redundancy_filter.execute(redundancy_state)
        
        # STEP 3: Filter memory context by relevance
        print("   Step 3: Filtering by relevance...")
        context = {"user_input": state["user_input"]}
        
        # Convert MemoryRecord objects to strings for filtering
        l1_strings = [record.payload if hasattr(record, 'payload') else str(record) for record in state.get("L1", [])]
        l2_strings = [record.payload if hasattr(record, 'payload') else str(record) for record in state.get("L2", [])]
        l3_strings = [record.payload if hasattr(record, 'payload') else str(record) for record in state.get("L3", [])]
        
        # Filter each memory tier by relevance (on string content)
        filtered_l1_strings = self.relevance_filter.filter_by_relevance(l1_strings, context)
        filtered_l2_strings = self.relevance_filter.filter_by_relevance(l2_strings, context)
        filtered_l3_strings = self.relevance_filter.filter_by_relevance(l3_strings, context)
        
        # Keep original MemoryRecord objects that passed filtering
        state["L1"] = [record for record in state.get("L1", []) 
                       if (record.payload if hasattr(record, 'payload') else str(record)) in filtered_l1_strings]
        state["L2"] = [record for record in state.get("L2", []) 
                       if (record.payload if hasattr(record, 'payload') else str(record)) in filtered_l2_strings]
        state["L3"] = [record for record in state.get("L3", []) 
                       if (record.payload if hasattr(record, 'payload') else str(record)) in filtered_l3_strings]
        
        print(f"   Final filtered context - L1: {len(state['L1'])} items, L2: {len(state['L2'])} items, L3: {len(state['L3'])} items")
        
        return state
