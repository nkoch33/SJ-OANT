"""
TMM Pipeline - Fixed Architecture

This is a complete rewrite of the TMM pipeline to address the critical integration issues:
1. Eliminated LangGraph state management overhead
2. Direct component integration without state loss
3. Simplified architecture with optional component activation
4. Proper context preservation throughout the pipeline

Key fixes:
- Direct method calls instead of LangGraph nodes
- Preserved context and state between components
- Optional component activation for ablation testing
- Streamlined memory operations
- Proper error handling and logging
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Import core components
from memory.typed_store import InMemoryStore, MemoryRecord
from truth.verifier import RuleBasedVerifier
from core.types import ConfidenceScores

logger = logging.getLogger(__name__)

@dataclass
class ProcessingContext:
    """Maintains context throughout pipeline processing."""
    original_input: str
    processed_input: str
    context_records: List[str]
    memory_state: Dict[str, Any]
    confidence_scores: ConfidenceScores
    processing_metadata: Dict[str, Any]

class TMMPipelineFixed:
    """
    Simplified TMM Pipeline with direct component integration.
    
    This implementation fixes the critical issues identified in Phase 2:
    1. No LangGraph overhead - direct method calls
    2. Context preservation - no information loss between stages
    3. Optional components - for ablation testing
    4. Streamlined operations - minimal overhead
    """
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """Initialize the fixed TMM pipeline."""
        self.api_key = api_key
        self.config = config or self._default_config()
        
        # Initialize LLM based on provider
        provider = self.config.get("api_provider", "google")
        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,
                    openai_api_key=api_key,
                    max_retries=3
                )
            except ImportError:
                logger.warning("OpenAI not available, falling back to Google Gemini")
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.1,
                    google_api_key=api_key
                )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=api_key
            )
        
        # Initialize core components
        self.memory_store = InMemoryStore()
        self.truth_verifier = RuleBasedVerifier()
        
        # Component activation flags (for ablation testing)
        self.enable_memory = self.config.get("enable_memory", True)
        self.enable_filtering = self.config.get("enable_filtering", True)
        self.enable_verification = self.config.get("enable_verification", True)
        
        logger.info(f"TMM Pipeline initialized with components: "
                   f"Memory={self.enable_memory}, "
                   f"Filtering={self.enable_filtering}, "
                   f"Verification={self.enable_verification}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration with simplified settings."""
        return {
            "enable_memory": True,
            "enable_filtering": True,
            "enable_verification": True,
            "relevance_threshold": 0.4,  # Lowered from 0.5 to reduce over-filtering
            "confidence_threshold": 0.6,  # Lowered from 0.8 to be less strict
            "max_context_length": 1000,  # Prevent context explosion
            "memory_retrieval_limit": 10,  # Increased from 5 for better memory utilization
            "context_relevance_threshold": 0.005,  # Much lower threshold for stored context (Priority 1)
            "recency_bonus_weight": 0.2,  # Weight for recency bonus (Priority 1)
            "max_recency_hours": 24  # Consider content recent within 24 hours
        }
    
    def process(self, user_input: str) -> str:
        """
        Process user input through the TMM pipeline.
        
        This is the main entry point that replaces the complex LangGraph execution
        with direct method calls and proper context preservation.
        """
        try:
            # Initialize processing context
            context = ProcessingContext(
                original_input=user_input,
                processed_input=user_input.lower().strip(),
                context_records=[],
                memory_state={},
                confidence_scores=ConfidenceScores(
                    truth_score=0.8,
                    confidence=0.8,
                    evidentiality=0.8,
                    relevance=1.0,
                    utility=0.8,
                    source_credibility=0.8
                ),
                processing_metadata={"start_time": time.time()}
            )
            
            logger.info(f"Processing input: {user_input[:100]}...")
            
            # Stage 1: Memory Retrieval (if enabled)
            if self.enable_memory:
                context = self._memory_retrieval_stage(context)
            
            # Stage 2: Context Filtering (if enabled)
            if self.enable_filtering:
                context = self._context_filtering_stage(context)
            
            # Stage 3: Truth Verification (if enabled)
            if self.enable_verification:
                context = self._truth_verification_stage(context)
            
            # Stage 4: Response Generation
            response = self._response_generation_stage(context)
            
            # Stage 5: Memory Storage (if enabled and appropriate)
            if self.enable_memory and self._should_store_interaction(user_input, response):
                self._memory_storage_stage(context, response)
            
            context.processing_metadata["end_time"] = time.time()
            context.processing_metadata["total_time"] = (
                context.processing_metadata["end_time"] - 
                context.processing_metadata["start_time"]
            )
            
            logger.info(f"Processing completed in {context.processing_metadata['total_time']:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"TMM pipeline error: {e}")
            # Fallback to direct LLM call
            return self._fallback_response(user_input)
    
    def _memory_retrieval_stage(self, context: ProcessingContext) -> ProcessingContext:
        """Retrieve relevant memories with proper context preservation."""
        try:
            # Get memory retrieval limit from config with fallback
            retrieval_limit = self.config.get("memory_retrieval_limit", 5)
            
            # Simple memory search without complex filtering
            memory_records = self.memory_store.search(
                context.processed_input, 
                limit=retrieval_limit
            )
            
            # Enhanced memory retrieval with context chaining (Priority 1)
            relevant_records = []
            stored_contexts = []
            chained_contexts = []  # For context chaining/multi-hop reasoning

            for record in memory_records:
                if hasattr(record, 'payload'):
                    payload = record.payload

                    # This IS the stored context content - highest priority!
                    if len(payload) > 50 and not payload.startswith("Q:"):  # Context content (not Q&A)
                        stored_contexts.append(payload)

                        # Context chaining: Look for related concepts in other stored records
                        payload_words = set(payload.lower().split())
                        for other_record in memory_records:
                            if (other_record != record and
                                hasattr(other_record, 'payload') and
                                len(other_record.payload) > 50 and
                                not other_record.payload.startswith("Q:")):

                                other_words = set(other_record.payload.lower().split())
                                shared_concepts = payload_words & other_words
                                if len(shared_concepts) >= 3:  # At least 3 shared concepts
                                    if other_record.payload not in stored_contexts:
                                        chained_contexts.append(other_record.payload)

                    # Also include relevant Q&A pairs for additional context
                    elif payload.startswith("Q:") and "?" in payload:
                        question_words = set(context.processed_input.lower().split())
                        record_words = set(payload.lower().split())
                        if len(question_words & record_words) > 1:  # Good overlap
                            relevant_records.append(payload)

            # Prioritize: stored context + chained contexts + relevant Q&A (enhanced for Priority 1)
            max_stored = 2  # Allow more stored contexts
            max_chained = 1  # Add chained contexts for multi-hop reasoning
            max_qa = 1      # Keep one relevant Q&A

            context.context_records = (
                stored_contexts[:max_stored] +
                chained_contexts[:max_chained] +
                relevant_records[:max_qa]
            )
            
            # Update memory state
            metrics = self.memory_store.get_metrics()
            context.memory_state = {
                "retrieved_records": len(context.context_records),
                "total_memory_records": metrics.get("total_records", 0),
                "filtered_from": len(memory_records)
            }
            
            logger.debug(f"Retrieved {len(context.context_records)} relevant records from {len(memory_records)} total")
            
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            context.context_records = []
            context.memory_state = {"error": str(e)}
        
        return context
    
    def _context_filtering_stage(self, context: ProcessingContext) -> ProcessingContext:
        """Filter context with reduced aggressiveness to prevent over-filtering."""
        try:
            if not context.context_records:
                return context
            
            # Enhanced filtering with recency bonus and lower thresholds (Priority 1)
            filtered_records = []
            threshold = self.config.get("context_relevance_threshold", 0.005)  # Much lower threshold
            recency_weight = self.config.get("recency_bonus_weight", 0.2)
            max_recency_hours = self.config.get("max_recency_hours", 24)

            for record in context.context_records:
                # Query-aware filtering with semantic similarity (Priority 2)
                input_words = set(context.processed_input.lower().split())
                record_words = set(record.lower().split())

                if not input_words or not record_words:
                    relevance = 0.0
                else:
                    # Basic word overlap
                    overlap = len(input_words & record_words)
                    basic_relevance = overlap / max(len(input_words), len(record_words))

                    # Query-aware relevance scoring (Priority 2)
                    question_type = self._analyze_question_type(context.original_input)
                    record_type = self._analyze_record_type(record)

                    # Semantic similarity bonus based on question and record types
                    semantic_bonus = self._calculate_semantic_similarity(question_type, record_type)

                    # Content density bonus (records with more unique information)
                    unique_words = len(record_words)
                    density_bonus = min(unique_words / 50, 0.3)  # Cap at 0.3

                    relevance = basic_relevance + semantic_bonus + density_bonus

                    # Apply recency bonus for recently stored content (Priority 1)
                    if len(record) > 100:  # Likely stored context content
                        relevance += recency_weight  # Add recency bonus

                # For substantial stored content, be very permissive (Priority 1 enhancement)
                if len(record) > 100:  # This is likely user-stored context
                    relevance = max(relevance, 0.1)  # Higher minimum relevance for stored content

                if relevance >= threshold or len(record) > 100:  # Include substantial content regardless
                    filtered_records.append(record)
                    logger.debug(f"Kept record (relevance: {relevance:.3f}): {record[:50]}...")
                else:
                    logger.debug(f"Filtered record (relevance: {relevance:.3f}): {record[:50]}...")
            
            original_count = len(context.context_records)
            context.context_records = filtered_records
            
            # Update confidence scores based on retention rate
            retention_rate = len(filtered_records) / max(1, original_count)
            context.confidence_scores = ConfidenceScores(
                relevance=retention_rate,
                truth_score=context.confidence_scores.truth_score,
                confidence=context.confidence_scores.confidence,
                evidentiality=context.confidence_scores.evidentiality,
                utility=context.confidence_scores.utility,
                source_credibility=context.confidence_scores.source_credibility
            )
            
            logger.debug(f"Context filtering: {len(filtered_records)} records retained")
            
        except Exception as e:
            logger.warning(f"Context filtering failed: {e}")
            # Don't filter on error - preserve context
        
        return context
    
    def _truth_verification_stage(self, context: ProcessingContext) -> ProcessingContext:
        """Verify context truth with relaxed thresholds."""
        try:
            if not context.context_records:
                return context
            
            # Simple consistency check without strict verification
            consistency_scores = []
            
            for record in context.context_records:
                # Basic consistency scoring (simplified)
                score = self.truth_verifier.verify(record, {"input": context.processed_input})
                consistency_scores.append(score.truth_score)
            
            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                # Update confidence scores (create new since it's immutable)
                context.confidence_scores = ConfidenceScores(
                    relevance=context.confidence_scores.relevance,
                    confidence=avg_consistency
                )
                
                # Only remove records with very low consistency (relaxed threshold)
                verified_records = []
                low_threshold = 0.3  # Much lower than original 0.8
                
                for i, record in enumerate(context.context_records):
                    if i < len(consistency_scores) and consistency_scores[i] >= low_threshold:
                        verified_records.append(record)
                
                context.context_records = verified_records
                logger.debug(f"Truth verification: {len(verified_records)} records verified")
            
        except Exception as e:
            logger.warning(f"Truth verification failed: {e}")
            # Don't verify on error - preserve context
        
        return context
    
    def _response_generation_stage(self, context: ProcessingContext) -> str:
        """Enhanced response generation with multi-context reasoning and evidence aggregation (Priority 3)."""
        try:
            # Check if this is a question (not a context storage command)
            if not context.original_input.startswith("Please remember this context:"):
                # This is a question - we need to use stored context
                if context.context_records:
                    # Multi-context reasoning and evidence aggregation (Priority 3)
                    if len(context.context_records) > 1:
                        # Multiple contexts available - use advanced synthesis
                        synthesized_context = self._synthesize_multi_context(context.context_records, context.original_input)
                        evidence_score = self._calculate_evidence_strength(context.context_records, context.original_input)

                        # Enhanced prompt with multi-context reasoning
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "Answer the question using the provided context information. "
                                     f"You have access to {len(context.context_records)} relevant pieces of information. "
                                     f"Evidence strength: {evidence_score:.1f}/1.0. "
                                     "Synthesize the most accurate answer from all available context. "
                                     "Be precise and cite supporting evidence when possible. "
                                     "If the context doesn't contain enough information, say 'I don't know'."),
                            ("user", f"Synthesized Context: {synthesized_context}\n\nQuestion: {context.original_input}")
                        ])

                        logger.debug(f"Multi-context synthesis: {len(context.context_records)} records, "
                                   f"evidence score: {evidence_score:.2f}")
                    else:
                        # Single context - use enhanced single-context reasoning
                        primary_context = context.context_records[0]
                        confidence_score = self._calculate_single_context_confidence(primary_context, context.original_input)

                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "Answer the question using ONLY the provided context. "
                                     f"Context confidence: {confidence_score:.1f}/1.0. "
                                     "Be precise and factual. Extract the most direct answer from the context. "
                                     "If the context doesn't contain the answer, say 'I don't know'. "
                                     "Do not make assumptions beyond what's stated."),
                            ("user", f"Context: {primary_context}\n\nQuestion: {context.original_input}")
                        ])

                        logger.debug(f"Single context reasoning: confidence {confidence_score:.2f}, "
                                   f"context: {primary_context[:100]}...")
                else:
                    # No stored context - use enhanced fallback
                    logger.warning(f"No context found for question: {context.original_input}")
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Answer the question with the most direct, concise response possible. "
                                 "Since no relevant context is available, provide a general answer if you know it. "
                                 "If you don't know the answer, say 'I don't know'. Avoid elaboration."),
                        ("user", context.original_input)
                    ])
            else:
                # This is a context storage command - acknowledge with memory confirmation
                return "I have stored the context information and will use it to answer future questions."
            
            # Generate response
            response = self.llm.invoke(prompt.format_messages())
            final_response = response.content.strip()
            
            logger.debug(f"Generated response: {final_response}")
            return final_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._fallback_response(context.original_input)
    
    def _memory_storage_stage(self, context: ProcessingContext, response: str):
        """Store interaction in memory with simplified logic."""
        try:
            # Special handling for context storage commands
            if context.original_input.startswith("Please remember this context:"):
                # Extract and store the actual context content
                context_content = context.original_input.replace("Please remember this context:", "").strip()
                if self._is_memory_worthy(context_content):
                    context_record = MemoryRecord(
                        payload=context_content,
                        scores=context.confidence_scores
                    )
                    self.memory_store.add(context_record)
                    logger.debug("Context content stored in memory")
            else:
                # Store question-answer pairs
                if "?" in context.original_input and self._is_memory_worthy(response):
                    qa_record = MemoryRecord(
                        payload=f"Q: {context.original_input} A: {response}",
                        scores=context.confidence_scores
                    )
                    self.memory_store.add(qa_record)
                    logger.debug("Q&A pair stored in memory")
                
        except Exception as e:
            logger.warning(f"Memory storage failed: {e}")
    
    def _should_store_interaction(self, user_input: str, response: str) -> bool:
        """Determine if interaction should be stored in memory."""
        # Don't store context storage commands
        if user_input.startswith("Please remember this context:"):
            return True  # Store the context content for future use
        
        # Don't store simple greetings or very short interactions
        if len(user_input.strip()) < 10 or len(response.strip()) < 10:
            return False
        
        # Don't store error responses
        if "error" in response.lower() or "sorry" in response.lower():
            return False
        
        # Store question-answer pairs
        if "?" in user_input:
            return True
        
        return False
    
    def _is_memory_worthy(self, text: str) -> bool:
        """Check if text is worth storing in memory."""
        if len(text.strip()) < 5:
            return False
        
        # Don't store very common phrases
        common_phrases = ["hello", "hi", "thanks", "thank you", "ok", "yes", "no"]
        if text.lower().strip() in common_phrases:
            return False
        
        return True

    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of question being asked (Priority 2)."""
        question_lower = question.lower()

        # Factual questions
        if any(word in question_lower for word in ['what', 'who', 'where', 'when']):
            if 'what is' in question_lower or 'what are' in question_lower:
                return 'definitional'
            elif 'where' in question_lower:
                return 'locational'
            elif 'when' in question_lower:
                return 'temporal'
            elif 'who' in question_lower:
                return 'personnel'
            else:
                return 'factual'

        # Quantitative questions
        elif any(word in question_lower for word in ['how many', 'how much', 'what percentage']):
            return 'quantitative'

        # Comparative questions
        elif any(word in question_lower for word in ['compare', 'difference', 'better', 'worse']):
            return 'comparative'

        # Process/explanation questions
        elif any(word in question_lower for word in ['how', 'why', 'explain']):
            return 'process'

        # Default
        else:
            return 'general'

    def _analyze_record_type(self, record: str) -> str:
        """Analyze the type of content in a record (Priority 2)."""
        record_lower = record.lower()

        # Check for Q&A pattern
        if record.startswith('Q:') and 'A:' in record:
            return 'qa_pair'

        # Definitional content
        if any(phrase in record_lower for phrase in [' is ', ' are ', ' refers to ', ' means ']):
            return 'definitional'

        # Locational content
        if any(word in record_lower for word in ['located', 'location', 'in ', 'at ', 'near']):
            return 'locational'

        # Temporal content
        if any(word in record_lower for word in ['time', 'period', 'era', 'century', 'year']):
            return 'temporal'

        # Quantitative content
        if any(char.isdigit() for char in record):
            return 'quantitative'

        # Process/explanation content
        if any(word in record_lower for word in ['process', 'method', 'how', 'because']):
            return 'process'

        # Default
        return 'general'

    def _calculate_semantic_similarity(self, question_type: str, record_type: str) -> float:
        """Calculate semantic similarity bonus between question and record types (Priority 2)."""

        # Perfect match
        if question_type == record_type:
            return 0.3

        # Good matches (related types)
        good_matches = {
            'factual': ['definitional', 'general'],
            'definitional': ['factual', 'general'],
            'locational': ['factual', 'general'],
            'temporal': ['factual', 'general'],
            'quantitative': ['factual', 'general'],
            'comparative': ['factual', 'general'],
            'process': ['factual', 'general']
        }

        if record_type in good_matches.get(question_type, []):
            return 0.2

        # Partial matches
        partial_matches = {
            'general': ['factual', 'definitional']  # General content can help with specific questions
        }

        if record_type in partial_matches.get(question_type, []):
            return 0.1

                # No match
        return 0.0

    def _synthesize_multi_context(self, context_records: List[str], question: str) -> str:
        """Synthesize multiple contexts for better reasoning (Priority 3)."""
        if not context_records:
            return ""

        # Prioritize stored context over Q&A pairs
        stored_contexts = []
        qa_pairs = []

        for record in context_records:
            if len(record) > 50 and not record.startswith("Q:"):
                stored_contexts.append(record)
            elif record.startswith("Q:"):
                qa_pairs.append(record)

        # Combine contexts with clear separation
        synthesized_parts = []

        if stored_contexts:
            synthesized_parts.append("Stored Context Information:")
            for i, ctx in enumerate(stored_contexts[:3], 1):  # Limit to top 3
                synthesized_parts.append(f"{i}. {ctx}")

        if qa_pairs:
            if stored_contexts:
                synthesized_parts.append("")
            synthesized_parts.append("Related Q&A Information:")
            for qa in qa_pairs[:2]:  # Limit to top 2
                synthesized_parts.append(qa)

        return "\n".join(synthesized_parts)

    def _calculate_evidence_strength(self, context_records: List[str], question: str) -> float:
        """Calculate evidence strength across multiple contexts (Priority 3)."""
        if not context_records:
            return 0.0

        question_words = set(question.lower().split())
        total_overlap = 0
        max_possible_overlap = len(question_words) * len(context_records)

        for record in context_records:
            record_words = set(record.lower().split())
            overlap = len(question_words & record_words)
            total_overlap += overlap

        # Normalize to 0-1 scale
        if max_possible_overlap == 0:
            return 0.0

        strength = total_overlap / max_possible_overlap

        # Bonus for multiple agreeing contexts
        if len(context_records) > 1:
            strength *= 1.2

        return min(strength, 1.0)

    def _calculate_single_context_confidence(self, context: str, question: str) -> float:
        """Calculate confidence score for single context (Priority 3)."""
        if not context:
            return 0.0

        question_words = set(question.lower().split())
        context_words = set(context.lower().split())

        # Basic word overlap
        overlap = len(question_words & context_words)
        if len(question_words) == 0:
            return 0.0

        base_confidence = overlap / len(question_words)

        # Length bonus (longer contexts generally more informative)
        length_bonus = min(len(context) / 200, 0.3)  # Cap at 0.3

        # Content quality indicators
        quality_indicators = ['is', 'are', 'was', 'were', 'located', 'in', 'at', 'during']
        quality_matches = sum(1 for indicator in quality_indicators if indicator in context.lower())
        quality_bonus = min(quality_matches * 0.05, 0.2)  # Cap at 0.2

        confidence = base_confidence + length_bonus + quality_bonus
        return min(confidence, 1.0)

    def _fallback_response(self, user_input: str) -> str:
        """Fallback response using direct LLM call."""
        try:
            response = self.llm.invoke(f"Answer this question: {user_input}")
            return response.content.strip()
        except Exception as e:
            logger.error(f"Fallback response failed: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def reset_memory(self):
        """Reset memory store to empty state."""
        try:
            self.memory_store = InMemoryStore()
            logger.info("Memory store reset to empty state")
        except Exception as e:
            logger.error(f"Memory reset failed: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get current memory state summary."""
        try:
            return self.memory_store.get_metrics()
        except Exception as e:
            logger.error(f"Memory summary failed: {e}")
            return {"error": str(e)}

def create_tmm_pipeline(api_key: str, config: Dict[str, Any] = None, provider: str = "google") -> TMMPipelineFixed:
    """
    Factory function to create a fixed TMM pipeline with API provider support.
    
    Args:
        api_key: API key for the LLM provider
        config: Optional configuration dictionary
        provider: "openai" or "google" (default)
    """
    if config is None:
        config = {}
    config["api_provider"] = provider
    return TMMPipelineFixed(api_key, config)

# For ablation testing - create variants with disabled components
def create_tmm_variant(api_key: str, disabled_components: List[str] = None, provider: str = "google") -> TMMPipelineFixed:
    """Create TMM variant with specific components disabled for ablation testing."""
    config = {
        "enable_memory": "memory" not in (disabled_components or []),
        "enable_filtering": "filtering" not in (disabled_components or []),
        "enable_verification": "verification" not in (disabled_components or []),
        "api_provider": provider
    }
    return TMMPipelineFixed(api_key, config)
