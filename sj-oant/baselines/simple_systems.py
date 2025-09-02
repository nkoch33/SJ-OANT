"""
baselines.simple_systems - Baseline System Implementations

This module implements baseline systems for comparison with the TMM system:
1. Direct LLM (no memory, context-only)
2. Simple RAG (basic retrieval-augmented generation)
3. Long Context (concatenate all history)
4. Basic Memory (store everything, no filtering)

These baselines represent the current state-of-the-art approaches that
the TMM system should outperform on truth-maintained memory tasks.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class BaselineSystem(ABC):
    """Base class for baseline systems."""
    
    @abstractmethod
    def process(self, input_text: str) -> str:
        """Process input and return response."""
        pass

@dataclass
class SimpleMemoryRecord:
    """Simple memory record for baseline systems."""
    content: str
    timestamp: float
    id: str

class DirectLLMBaseline(BaselineSystem):
    """
    Direct LLM baseline - no memory, just context-aware responses.
    
    This represents the simplest baseline where the LLM only sees
    the current input without any external memory or retrieval.
    """
    
    def __init__(self, llm: BaseChatModel):
        """Initialize with an LLM."""
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_template(
            "Answer the following question based on the provided information:\n\n{input_text}\n\nAnswer:"
        )
        logger.info("Initialized DirectLLM baseline")
    
    def process(self, input_text: str) -> str:
        """Process input with direct LLM call."""
        try:
            prompt = self.prompt_template.format_messages(input_text=input_text)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"DirectLLM processing failed: {e}")
            return f"Error: {e}"

class LongContextBaseline(BaselineSystem):
    """
    Long context baseline - maintains conversation history.
    
    This baseline concatenates all previous interactions and provides
    them as context to the LLM, representing the "just use long context" approach.
    """
    
    def __init__(self, llm: BaseChatModel, max_context_length: int = 8000):
        """Initialize with LLM and context length limit."""
        self.llm = llm
        self.max_context_length = max_context_length
        self.conversation_history: List[str] = []
        self.prompt_template = ChatPromptTemplate.from_template(
            "Previous conversation:\n{context}\n\nCurrent input: {input_text}\n\nResponse:"
        )
        logger.info(f"Initialized LongContext baseline (max_length={max_context_length})")
    
    def process(self, input_text: str) -> str:
        """Process input with full conversation context."""
        try:
            # Add current input to history
            self.conversation_history.append(f"Input: {input_text}")
            
            # Build context (truncate if too long)
            context = "\n".join(self.conversation_history)
            if len(context) > self.max_context_length:
                # Simple truncation from the beginning
                words = context.split()
                while len(" ".join(words)) > self.max_context_length and words:
                    words.pop(0)
                context = " ".join(words)
            
            # Generate response
            prompt = self.prompt_template.format_messages(
                context=context,
                input_text=input_text
            )
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            # Add response to history
            self.conversation_history.append(f"Response: {response_text}")
            
            return response_text
        except Exception as e:
            logger.error(f"LongContext processing failed: {e}")
            return f"Error: {e}"

class SimpleRAGBaseline(BaselineSystem):
    """
    Simple RAG baseline - basic retrieval-augmented generation.
    
    This baseline stores all inputs in a simple memory and retrieves
    relevant information using basic keyword matching for context.
    """
    
    def __init__(self, llm: BaseChatModel, max_retrievals: int = 3):
        """Initialize with LLM and retrieval settings."""
        self.llm = llm
        self.max_retrievals = max_retrievals
        self.memory: List[SimpleMemoryRecord] = []
        self.prompt_template = ChatPromptTemplate.from_template(
            "Based on the following relevant information:\n{retrieved_context}\n\n"
            "Answer this question: {input_text}\n\nAnswer:"
        )
        logger.info(f"Initialized SimpleRAG baseline (max_retrievals={max_retrievals})")
    
    def _store_information(self, content: str):
        """Store information in simple memory."""
        import time
        import uuid
        
        record = SimpleMemoryRecord(
            content=content,
            timestamp=time.time(),
            id=str(uuid.uuid4())
        )
        self.memory.append(record)
        logger.debug(f"Stored memory record: {content[:100]}...")
    
    def _retrieve_relevant(self, query: str) -> List[SimpleMemoryRecord]:
        """Simple keyword-based retrieval."""
        query_words = set(query.lower().split())
        scored_records = []
        
        for record in self.memory:
            record_words = set(record.content.lower().split())
            overlap = len(query_words.intersection(record_words))
            if overlap > 0:
                score = overlap / len(query_words)  # Simple Jaccard-like similarity
                scored_records.append((record, score))
        
        # Sort by score and return top results
        scored_records.sort(key=lambda x: x[1], reverse=True)
        return [record for record, score in scored_records[:self.max_retrievals]]
    
    def process(self, input_text: str) -> str:
        """Process input with RAG approach."""
        try:
            # If this looks like a statement/story, store it
            if any(keyword in input_text.lower() for keyword in ["story:", "remember", "this is", "let me tell"]):
                self._store_information(input_text)
                return "I've remembered that information."
            
            # If this looks like a question, retrieve and answer
            retrieved_records = self._retrieve_relevant(input_text)
            retrieved_context = "\n".join([record.content for record in retrieved_records])
            
            if not retrieved_context:
                retrieved_context = "No relevant information found in memory."
            
            prompt = self.prompt_template.format_messages(
                retrieved_context=retrieved_context,
                input_text=input_text
            )
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"SimpleRAG processing failed: {e}")
            return f"Error: {e}"

class BasicMemoryBaseline(BaselineSystem):
    """
    Basic memory baseline - store everything, no filtering.
    
    This baseline represents a naive memory approach where everything
    is stored without any truth verification or conflict resolution.
    """
    
    def __init__(self, llm: BaseChatModel):
        """Initialize with LLM."""
        self.llm = llm
        self.memory: List[SimpleMemoryRecord] = []
        self.prompt_template = ChatPromptTemplate.from_template(
            "Based on everything I remember:\n{all_memory}\n\n"
            "Question: {input_text}\n\nAnswer:"
        )
        logger.info("Initialized BasicMemory baseline")
    
    def process(self, input_text: str) -> str:
        """Process input with basic memory approach."""
        try:
            # Store everything as memory
            import time
            import uuid
            
            record = SimpleMemoryRecord(
                content=input_text,
                timestamp=time.time(),
                id=str(uuid.uuid4())
            )
            self.memory.append(record)
            
            # Use all memory as context (truncate if necessary)
            all_memory = "\n".join([record.content for record in self.memory])
            if len(all_memory) > 4000:  # Simple truncation
                words = all_memory.split()
                while len(" ".join(words)) > 4000 and words:
                    words.pop(0)
                all_memory = " ".join(words)
            
            prompt = self.prompt_template.format_messages(
                all_memory=all_memory,
                input_text=input_text
            )
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"BasicMemory processing failed: {e}")
            return f"Error: {e}"

def create_baseline_systems(llm: BaseChatModel) -> Dict[str, BaselineSystem]:
    """
    Factory function to create all baseline systems.
    
    Args:
        llm: Language model to use for all baselines
        
    Returns:
        Dictionary of baseline system name -> instance
    """
    return {
        "DirectLLM": DirectLLMBaseline(llm),
        "LongContext": LongContextBaseline(llm, max_context_length=8000),
        "SimpleRAG": SimpleRAGBaseline(llm, max_retrievals=3),
        "BasicMemory": BasicMemoryBaseline(llm)
    }
