"""
writer_editor.py - Memory Writing and Editing Agent

This module contains the Writer/Editor agent responsible for memory write operations
and editing in the Truth-Maintained Memory (TMM) system.

The writer/editor:

1. Receives verified information from the verification pipeline
2. Implements Selective Addition policies for new memory entries
3. Executes Combined Deletion policies for outdated/contradicted information
4. Manages memory updates and overwrites when corrections are needed
5. Coordinates with the Memory Curator for final storage decisions

Key responsibilities:
- Selective Addition policy implementation (high-trust, high-utility facts only)
- Combined Deletion policy execution (periodic pruning of low-value entries)
- Memory update and correction handling
- Coordination with memory storage systems
- Audit trail maintenance for memory operations

The writer/editor is critical for preventing false memory formation by ensuring
that only verified, high-quality information enters long-term memory while
systematically removing or correcting problematic content.
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from memory.typed_store import MemoryState, TypedMemoryStore


class MemoryCurationAgent:
    """
    Memory curation agent responsible for orchestrating memory updates and 
    making final storage decisions based on verification results.
    
    This agent implements the memory curation logic from the original pipeline,
    managing the flow of information between memory tiers and ensuring
    consistency across the memory hierarchy.
    """
    
    def __init__(self, llm, memory_store: TypedMemoryStore):
        """
        Initialize the memory curation agent.
        
        Args:
            llm: Language model instance
            memory_store: TypedMemoryStore instance for memory operations
        """
        self.llm = llm
        self.memory_store = memory_store
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are the Memory Curation Agent (Manager).
            Your role is to orchestrate memory updates, making final storage decisions
            based on proposals and votes from other agents.
            You must ensure memory consistency, prevent contradictions, and maintain
            a clear hierarchy across the caches:

            - L1 (Working Memory): recent raw inputs.
            - L2 (Summarized Memory): concise summaries of recent turns.
            - L3 (Archival Memory): verified facts and anchors.
            - FLAGGED: contradictory or irrelevant content (do not discard permanently, just quarantine).

            Promotion Rules:
            - When L1 exceeds its limit, compress content into L2.
            - When L2 exceeds its limit, compress content into L3.
            - Never add content directly to L3 without summarization or verification.
            - If contradiction is reported, route to FLAGGED instead of caches.
            """)
        ])
    
    def make_storage_decision(self, content: str, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final decision about where and how to store information.
        
        Args:
            content: Content to store
            verification_result: Results from truth verification
            
        Returns:
            Storage decision with target tier and metadata
        """
        decision = {
            "target_tier": "L1",  # Default to working memory
            "should_store": True,
            "reason": "Standard storage to L1",
            "metadata": {}
        }
        
        # If contradiction detected, flag for review
        if verification_result.get("contradiction_detected", False):
            decision.update({
                "target_tier": "flagged",
                "reason": f"Contradiction detected: {verification_result.get('conflicting_facts', [])}",
                "metadata": {"flag_reason": "contradiction"}
            })
        
        # If low confidence, flag for review
        elif verification_result.get("confidence", 0.0) < 0.5:
            decision.update({
                "target_tier": "flagged",
                "reason": f"Low confidence: {verification_result.get('confidence', 0.0):.2f}",
                "metadata": {"flag_reason": "low_confidence"}
            })
        
        # If high confidence and verified, consider for L2/L3
        elif verification_result.get("confidence", 0.0) > 0.8:
            if verification_result.get("truth_score", 0.0) > 0.9:
                decision.update({
                    "target_tier": "L2",  # High-quality content goes to L2 first
                    "reason": "High confidence and truth score",
                    "metadata": {"quality": "high"}
                })
        
        return decision
    
    def execute_storage_decision(self, content: str, decision: Dict[str, Any]) -> None:
        """
        Execute the storage decision by updating the memory store.
        
        Args:
            content: Content to store
            decision: Storage decision from make_storage_decision
        """
        target_tier = decision["target_tier"]
        metadata = decision.get("metadata", {})
        
        if target_tier == "flagged":
            reason = decision.get("reason", "Unknown")
            self.memory_store.add_to_flagged(content, reason)
        elif target_tier == "L1":
            # Create proper ConfidenceScores object
            from core.types import ConfidenceScores
            confidence = metadata.get("confidence", 0.7)
            scores = ConfidenceScores(
                truth_score=confidence,
                confidence=confidence,
                evidentiality=confidence
            )
            # Don't pass metadata as a separate argument since it may conflict with MemoryRecord
            self.memory_store.add_to_l1(content, scores=scores)
        # TODO: Add direct L2/L3 addition methods to TypedMemoryStore
        
        print(f"   Stored to {target_tier}: {content[:50]}...")
        print(f"   Reason: {decision['reason']}")
    
    def manage_memory_limits(self) -> None:
        """
        Check and manage memory tier limits, triggering compressions as needed.
        
        This implements the promotion rules from the original curation logic.
        """
        # Memory store handles this automatically in add_to_l1()
        # But we could add explicit management here
        summary = self.memory_store.get_memory_summary()
        
        from core.types import MemoryTier
        if summary["tier_sizes"]["L1"] > self.memory_store.tier_limits[MemoryTier.L1_WORKING]:
            print("   L1 over limit, compression will occur on next addition")
        
        if summary["tier_sizes"]["L2"] > self.memory_store.tier_limits[MemoryTier.L2_SUMMARIZED]:
            print("   L2 over limit, promotion to L3 will occur")
    
    def execute(self, state: MemoryState, verification_result: Dict[str, Any] = None) -> MemoryState:
        """
        Execute memory curation as part of the TMM pipeline.
        
        Args:
            state: Current memory state
            verification_result: Results from truth verification (if available)
            
        Returns:
            Updated memory state after curation
        """
        print("📚 Memory Curator: Making storage decisions...")
        
        content = state["user_input"]
        
        # Use verification result if provided, otherwise create default
        if verification_result is None:
            verification_result = {
                "confidence": 0.7,
                "truth_score": 0.7,
                "contradiction_detected": False
            }
        
        # Make storage decision
        decision = self.make_storage_decision(content, verification_result)
        
        # Execute the decision
        self.execute_storage_decision(content, decision)
        
        # Manage memory limits
        self.manage_memory_limits()
        
        # Update state with current memory store state
        updated_state = self.memory_store.get_state()
        updated_state["user_input"] = state["user_input"]
        
        # Display memory summary
        summary = self.memory_store.get_memory_summary()
        print(f"   Memory summary: L1={summary['tier_sizes']['L1']}, "
              f"L2={summary['tier_sizes']['L2']}, L3={summary['tier_sizes']['L3']}, "
              f"Flagged={summary['tier_sizes']['flagged']}")
        
        return updated_state


class SelectiveAdditionPolicy:
    """
    Implementation of the Selective Addition policy for memory writes.
    
    This policy ensures that only high-trust, high-utility information
    is added to the memory store, preventing false memory accumulation.
    """
    
    def __init__(self, trust_threshold: float = 0.8, utility_threshold: float = 0.7):
        """
        Initialize the selective addition policy.
        
        Args:
            trust_threshold: Minimum trust score for addition
            utility_threshold: Minimum utility score for addition
        """
        self.trust_threshold = trust_threshold
        self.utility_threshold = utility_threshold
    
    def should_add(self, content: str, verification_result: Dict[str, Any]) -> bool:
        """
        Determine if content should be added based on selective addition criteria.
        
        Args:
            content: Content to evaluate
            verification_result: Verification results with trust/utility scores
            
        Returns:
            True if content meets addition criteria, False otherwise
        """
        trust_score = verification_result.get("confidence", 0.0)
        utility_score = verification_result.get("truth_score", 0.0)  # Using truth_score as proxy
        
        meets_trust = trust_score >= self.trust_threshold
        meets_utility = utility_score >= self.utility_threshold
        no_contradictions = not verification_result.get("contradiction_detected", False)
        
        return meets_trust and meets_utility and no_contradictions


class CombinedDeletionPolicy:
    """
    Implementation of the Combined Deletion policy for memory cleanup.
    
    This policy periodically prunes contradicted and low-utility entries
    from memory to maintain quality and prevent false memory persistence.
    """
    
    def __init__(self, cleanup_interval: int = 50, low_utility_threshold: float = 0.3):
        """
        Initialize the combined deletion policy.
        
        Args:
            cleanup_interval: Number of operations between cleanup cycles
            low_utility_threshold: Utility threshold for deletion
        """
        self.cleanup_interval = cleanup_interval
        self.low_utility_threshold = low_utility_threshold
        self.operation_count = 0
    
    def should_cleanup(self) -> bool:
        """
        Determine if periodic cleanup should be triggered.
        
        Returns:
            True if cleanup should run, False otherwise
        """
        self.operation_count += 1
        return self.operation_count % self.cleanup_interval == 0
    
    def identify_deletion_candidates(self, memory_store: TypedMemoryStore) -> List[str]:
        """
        Identify entries that should be deleted based on policy criteria.
        
        Args:
            memory_store: Memory store to analyze
            
        Returns:
            List of entry identifiers to delete
        """
        deletion_candidates = []
        
        # TODO: Implement sophisticated deletion candidate identification
        # For now, placeholder logic
        
        # Check flagged items for permanent deletion
        flagged_records = [r for r in memory_store.records.values() if r.tier == MemoryTier.FLAGGED]
        for entry in flagged_records:
            if entry.scores.confidence < self.low_utility_threshold:
                deletion_candidates.append(entry.content)
        
        return deletion_candidates


class WriterEditor:
    """
    Main writer/editor agent that coordinates memory writing and editing operations.
    
    This agent combines the memory curation agent with selective addition and
    combined deletion policies to implement the complete memory management system.
    """
    
    def __init__(self, llm, memory_store: TypedMemoryStore):
        """
        Initialize the writer/editor with all sub-components.
        
        Args:
            llm: Language model instance
            memory_store: TypedMemoryStore instance
        """
        self.memory_curator = MemoryCurationAgent(llm, memory_store)
        self.selective_addition = SelectiveAdditionPolicy()
        self.combined_deletion = CombinedDeletionPolicy()
        self.memory_store = memory_store
    
    def execute(self, state: MemoryState, verification_result: Dict[str, Any] = None) -> MemoryState:
        """
        Execute the complete memory writing and editing pipeline.
        
        Args:
            state: Current memory state
            verification_result: Results from truth verification
            
        Returns:
            Updated memory state after writing/editing operations
        """
        print("✍️  Writer/Editor: Managing memory operations...")
        
        # Execute memory curation
        updated_state = self.memory_curator.execute(state, verification_result)
        
        # Check if periodic cleanup should run
        if self.combined_deletion.should_cleanup():
            print("   🧹 Running periodic memory cleanup...")
            deletion_candidates = self.combined_deletion.identify_deletion_candidates(self.memory_store)
            if deletion_candidates:
                print(f"   Found {len(deletion_candidates)} deletion candidates")
                # TODO: Implement actual deletion logic
        
        return updated_state
