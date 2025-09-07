"""
Multi-Turn Baseline Systems for MultiWOZ Evaluation

This module implements baseline systems specifically designed for multi-turn dialogue
evaluation using the MultiWOZ dataset. These baselines test different approaches to
handling conversational memory and state tracking.

Baselines:
1. DirectLLM - Direct LLM response without memory
2. LongContext - Concatenate all previous turns as context
3. SimpleStateTracker - Basic dialogue state tracking
4. NaiveMemory - Simple memory storage with keyword retrieval
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class DialogueState:
    """Represents the current state of a dialogue."""
    domain: str
    slots: Dict[str, Any]
    history: List[str]
    turn_count: int

class BaseMultiTurnSystem:
    """Base class for multi-turn dialogue systems."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.dialogue_state = None
    
    def process(self, user_input: str) -> str:
        """Process user input and return system response."""
        raise NotImplementedError
    
    def reset_memory(self):
        """Reset system memory/state."""
        self.dialogue_state = None

class DirectLLMBaseline(BaseMultiTurnSystem):
    """
    Direct LLM baseline - responds to each turn independently without memory.
    
    This baseline tests if the LLM can handle dialogue tasks using only its
    pre-trained knowledge without any memory or state tracking.
    """
    
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant for task-oriented dialogue. "
            "Respond to the user's request: {user_input}\n\nResponse:"
        )
        logger.info("Initialized DirectLLM baseline for multi-turn dialogue")
    
    def process(self, user_input: str) -> str:
        """Process user input with direct LLM response."""
        try:
            prompt = self.prompt_template.format_messages(user_input=user_input)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"DirectLLM processing failed: {e}")
            return f"I'm sorry, I encountered an error: {e}"

class LongContextBaseline(BaseMultiTurnSystem):
    """
    Long context baseline - concatenates all previous turns as context.
    
    This baseline tests if simply providing all previous conversation history
    as context is sufficient for handling multi-turn dialogues.
    """
    
    def __init__(self, llm: BaseChatModel, max_context_length: int = 8000):
        super().__init__(llm)
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant for task-oriented dialogue. "
            "Here is the conversation history:\n{conversation_history}\n\n"
            "Current user input: {user_input}\n\nResponse:"
        )
        logger.info(f"Initialized LongContext baseline (max_length={max_context_length})")
    
    def process(self, user_input: str) -> str:
        """Process user input with full conversation history."""
        try:
            # Add user input to history
            self.conversation_history.append(f"User: {user_input}")
            
            # Build conversation context
            conversation_text = "\n".join(self.conversation_history)
            
            # Truncate if too long (character-based to prevent infinite loops)
            if len(conversation_text) > self.max_context_length:
                conversation_text = conversation_text[-self.max_context_length:]
            
            # Limit history size to prevent memory issues
            if len(self.conversation_history) > 100:  # Max 100 turns
                self.conversation_history = self.conversation_history[-50:]  # Keep last 50
            
            # Generate response
            prompt = self.prompt_template.format_messages(
                conversation_history=conversation_text,
                user_input=user_input
            )
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            # Add response to history
            self.conversation_history.append(f"System: {response_text}")
            
            return response_text
        except Exception as e:
            logger.error(f"LongContext processing failed: {e}")
            return f"I'm sorry, I encountered an error: {e}"
    
    def reset_memory(self):
        """Reset conversation history."""
        super().reset_memory()
        self.conversation_history = []

class SimpleStateTrackerBaseline(BaseMultiTurnSystem):
    """
    Simple state tracker baseline - maintains basic dialogue state.
    
    This baseline implements a simplified version of traditional dialogue state tracking,
    maintaining key-value pairs for different domains and slots.
    """
    
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.dialogue_state = DialogueState(
            domain="",
            slots={},
            history=[],
            turn_count=0
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant for task-oriented dialogue. "
            "Current dialogue state: {dialogue_state}\n\n"
            "User input: {user_input}\n\n"
            "Respond appropriately and update the dialogue state if needed.\n\nResponse:"
        )
        logger.info("Initialized SimpleStateTracker baseline")
    
    def process(self, user_input: str) -> str:
        """Process user input with state tracking."""
        try:
            # Update turn count
            self.dialogue_state.turn_count += 1
            self.dialogue_state.history.append(f"Turn {self.dialogue_state.turn_count}: {user_input}")
            
            # Extract domain and slots from user input (simplified)
            self._update_state(user_input)
            
            # Generate response
            state_text = self._format_state()
            prompt = self.prompt_template.format_messages(
                dialogue_state=state_text,
                user_input=user_input
            )
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            # Add response to history
            self.dialogue_state.history.append(f"System: {response_text}")
            
            return response_text
        except Exception as e:
            logger.error(f"SimpleStateTracker processing failed: {e}")
            return f"I'm sorry, I encountered an error: {e}"
    
    def _update_state(self, user_input: str):
        """Update dialogue state based on user input."""
        user_lower = user_input.lower()
        
        # Simple domain detection
        if any(word in user_lower for word in ["restaurant", "food", "eat", "dining"]):
            self.dialogue_state.domain = "restaurant"
        elif any(word in user_lower for word in ["hotel", "accommodation", "stay", "room"]):
            self.dialogue_state.domain = "hotel"
        elif any(word in user_lower for word in ["taxi", "ride", "transport"]):
            self.dialogue_state.domain = "taxi"
        elif any(word in user_lower for word in ["attraction", "visit", "see", "tourist"]):
            self.dialogue_state.domain = "attraction"
        elif any(word in user_lower for word in ["train", "travel", "journey"]):
            self.dialogue_state.domain = "train"
        
        # Simple slot extraction
        if "price" in user_lower or "cost" in user_lower:
            self.dialogue_state.slots["price_range"] = "mentioned"
        if "time" in user_lower or "when" in user_lower:
            self.dialogue_state.slots["time"] = "mentioned"
        if "location" in user_lower or "where" in user_lower:
            self.dialogue_state.slots["location"] = "mentioned"
    
    def _format_state(self) -> str:
        """Format dialogue state as text."""
        state_parts = []
        if self.dialogue_state.domain:
            state_parts.append(f"Domain: {self.dialogue_state.domain}")
        if self.dialogue_state.slots:
            state_parts.append(f"Slots: {json.dumps(self.dialogue_state.slots)}")
        state_parts.append(f"Turn: {self.dialogue_state.turn_count}")
        return "\n".join(state_parts)
    
    def reset_memory(self):
        """Reset dialogue state."""
        super().reset_memory()
        self.dialogue_state = DialogueState(
            domain="",
            slots={},
            history=[],
            turn_count=0
        )

class NaiveMemoryBaseline(BaseMultiTurnSystem):
    """
    Naive memory baseline - stores all user inputs and retrieves by keyword matching.
    
    This baseline tests if simple memory storage and keyword-based retrieval
    can improve dialogue performance over no memory at all.
    """
    
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.memory = []
        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant for task-oriented dialogue. "
            "Relevant information from previous conversation:\n{relevant_memory}\n\n"
            "User input: {user_input}\n\nResponse:"
        )
        logger.info("Initialized NaiveMemory baseline")
    
    def process(self, user_input: str) -> str:
        """Process user input with naive memory retrieval."""
        try:
            # Store user input in memory
            self.memory.append({
                "content": user_input,
                "timestamp": time.time(),
                "turn_id": len(self.memory)
            })
            
            # Limit memory size to prevent memory issues
            if len(self.memory) > 1000:  # Max 1000 memory items
                self.memory = self.memory[-500:]  # Keep last 500
            
            # Retrieve relevant memory
            relevant_memory = self._retrieve_relevant_memory(user_input)
            
            # Generate response
            memory_text = "\n".join([item["content"] for item in relevant_memory])
            prompt = self.prompt_template.format_messages(
                relevant_memory=memory_text if memory_text else "No relevant information found.",
                user_input=user_input
            )
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"NaiveMemory processing failed: {e}")
            return f"I'm sorry, I encountered an error: {e}"
    
    def _retrieve_relevant_memory(self, user_input: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memory based on keyword matching."""
        user_words = set(user_input.lower().split())
        relevant_items = []
        
        for item in self.memory:
            item_words = set(item["content"].lower().split())
            overlap = len(user_words.intersection(item_words))
            if overlap > 0:
                relevant_items.append((item, overlap))
        
        # Sort by relevance and return top items
        relevant_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in relevant_items[:3]]  # Top 3 most relevant
    
    def reset_memory(self):
        """Reset memory storage."""
        super().reset_memory()
        self.memory = []

def create_multiturn_baselines(api_key: str) -> Dict[str, BaseMultiTurnSystem]:
    """
    Factory function to create all multi-turn baseline systems.
    
    Args:
        api_key: API key for the language model
        
    Returns:
        Dictionary mapping baseline names to system instances
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )
    
    baselines = {
        "DirectLLM": DirectLLMBaseline(llm),
        "LongContext": LongContextBaseline(llm),
        "SimpleStateTracker": SimpleStateTrackerBaseline(llm),
        "NaiveMemory": NaiveMemoryBaseline(llm)
    }
    
    logger.info(f"Created {len(baselines)} multi-turn baseline systems")
    return baselines
