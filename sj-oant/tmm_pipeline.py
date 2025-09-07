"""
TMM Pipeline - Multi-Agent Implementation

This module provides the main TMM pipeline interface that uses the complete
multi-agent context-filtering chain as described in the research paper.

The pipeline orchestrates all individual agents:
Strategic Planner → TACS Filter → Truth Verifier → Memory Curator → Responder

This is the proper implementation that matches the research claims.
"""

import logging
from typing import Dict, Any, List, Optional

# Import the proper multi-agent pipeline
from multi_agent_pipeline import MultiAgentTMMPipeline, create_multi_agent_pipeline

logger = logging.getLogger(__name__)

class TMMPipelineFixed:
    """
    TMM Pipeline wrapper that uses the proper multi-agent implementation.
    
    This class provides backward compatibility while using the complete
    multi-agent context-filtering chain as described in the research.
    """
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """Initialize the TMM pipeline with multi-agent implementation."""
        self.api_key = api_key
        self.config = config or self._default_config()
        
        # Use the proper multi-agent pipeline
        self.multi_agent_pipeline = create_multi_agent_pipeline(api_key, config)
        
        logger.info("TMM Pipeline initialized with multi-agent implementation")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the multi-agent pipeline."""
        return {
            "relevance_threshold": 0.5,
            "confidence_threshold": 0.6,
            "max_context_length": 1000,
            "memory_retrieval_limit": 10
        }
    
    def process(self, user_input: str) -> str:
        """
        Process user input through the multi-agent TMM pipeline.
        
        This delegates to the proper multi-agent implementation that uses
        the complete context-filtering agent chain.
        """
        return self.multi_agent_pipeline.process(user_input)
    
    def reset_memory(self):
        """Reset the memory store."""
        self.multi_agent_pipeline.memory_store.reset_memory()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        return self.multi_agent_pipeline.memory_store.get_memory_summary()

def create_tmm_pipeline(api_key: str, config: Dict[str, Any] = None, provider: str = "google") -> TMMPipelineFixed:
    """
    Factory function to create a TMM pipeline.
    
    Args:
        api_key: API key for the language model
        config: Optional configuration dictionary
        provider: API provider (currently only "google" supported)
        
    Returns:
        Initialized TMMPipelineFixed instance with multi-agent implementation
    """
    return TMMPipelineFixed(api_key, config)

def create_tmm_variant(api_key: str, disabled_components: List[str] = None, provider: str = "google") -> TMMPipelineFixed:
    """
    Create a TMM pipeline variant with specific components disabled for ablation testing.
    
    Args:
        api_key: API key for the language model
        disabled_components: List of component names to disable
        provider: API provider (currently only "google" supported)
        
    Returns:
        Initialized TMMPipelineFixed instance with specified components disabled
    """
    config = {}
    if disabled_components:
        for component in disabled_components:
            if component == "memory":
                config["enable_memory"] = False
            elif component == "filtering":
                config["enable_filtering"] = False
            elif component == "verification":
                config["enable_verification"] = False
    
    return TMMPipelineFixed(api_key, config)