"""
Base LLM Wrapper for Baseline Testing

Provides a common interface for all baseline LLM models to ensure
consistent evaluation across different models.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    text: str
    response_time: float
    model_name: str
    metadata: Dict[str, Any]

class BaseLLMWrapper(ABC):
    """
    Abstract base class for LLM wrappers.
    
    Provides a common interface for all baseline models to ensure
    consistent evaluation and comparison with TMM.
    """
    
    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_name: Name of the model
            api_key: API key for the model (if required)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.conversation_history = []
        self.response_count = 0
        self.total_response_time = 0.0
        
        logger.info(f"Initialized {model_name} wrapper")
    
    @abstractmethod
    def _call_model(self, prompt: str, **kwargs) -> str:
        """
        Call the underlying model with a prompt.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model response text
        """
        pass
    
    def process_conversation_turn(self, user_input: str, **kwargs) -> LLMResponse:
        """
        Process a single conversation turn.
        
        Args:
            user_input: User's input for this turn
            **kwargs: Additional model-specific parameters
            
        Returns:
            Standardized LLM response
        """
        start_time = time.time()
        
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Create prompt from conversation history
            prompt = self._create_prompt_from_history()
            
            # Call the model
            response_text = self._call_model(prompt, **kwargs)
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Calculate response time
            response_time = time.time() - start_time
            self.total_response_time += response_time
            self.response_count += 1
            
            # Create standardized response
            response = LLMResponse(
                text=response_text,
                response_time=response_time,
                model_name=self.model_name,
                metadata={
                    "turn_number": len(self.conversation_history) // 2,
                    "conversation_length": len(self.conversation_history),
                    "model_specific": kwargs
                }
            )
            
            logger.debug(f"{self.model_name} processed turn in {response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"{self.model_name} failed to process turn: {e}")
            response_time = time.time() - start_time
            
            return LLMResponse(
                text=f"Error: {str(e)}",
                response_time=response_time,
                model_name=self.model_name,
                metadata={"error": str(e)}
            )
    
    def _create_prompt_from_history(self) -> str:
        """Create a prompt from conversation history."""
        prompt_parts = []
        
        for turn in self.conversation_history:
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    def reset_conversation(self):
        """Reset conversation history for a new conversation."""
        self.conversation_history = []
        logger.debug(f"{self.model_name} conversation reset")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this model."""
        avg_response_time = (
            self.total_response_time / self.response_count 
            if self.response_count > 0 else 0.0
        )
        
        return {
            "model_name": self.model_name,
            "total_responses": self.response_count,
            "total_response_time": self.total_response_time,
            "avg_response_time": avg_response_time,
            "conversation_turns": len(self.conversation_history) // 2
        }
    
    def process_full_conversation(self, user_turns: List[str], **kwargs) -> List[LLMResponse]:
        """
        Process a full conversation with multiple turns.
        
        Args:
            user_turns: List of user inputs for the conversation
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of LLM responses for each turn
        """
        self.reset_conversation()
        responses = []
        
        for turn_idx, user_input in enumerate(user_turns):
            logger.debug(f"{self.model_name} processing turn {turn_idx + 1}/{len(user_turns)}")
            response = self.process_conversation_turn(user_input, **kwargs)
            responses.append(response)
        
        logger.info(f"{self.model_name} completed conversation with {len(responses)} responses")
        return responses
