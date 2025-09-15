"""
TMM Pipeline - Multi-Agent Implementation with Enhanced Features

This module provides the main TMM pipeline interface that uses the complete
multi-agent context-filtering chain as described in the research paper.

The pipeline orchestrates all individual agents:
Strategic Planner → TACS Filter → Truth Verifier → Memory Curator → Responder

Enhanced Features:
- Adaptive memory retrieval and management
- Enhanced multi-agent coordination
- Advanced truth verification
- Research-grade monitoring and analytics
- Comprehensive performance optimization

This is the proper implementation that matches the research claims.
"""

import logging
import time
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import the proper multi-agent pipeline
from multi_agent_pipeline import MultiAgentTMMPipeline, create_multi_agent_pipeline

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPipelineConfig:
    """Configuration for the enhanced TMM pipeline."""
    enable_adaptive_retrieval: bool = False
    enable_enhanced_coordination: bool = False
    enable_advanced_verification: bool = False
    enable_research_analytics: bool = False
    performance_monitoring: bool = False
    adaptive_learning: bool = False

class TMMPipelineFixed:
    """
    TMM Pipeline wrapper that uses the proper multi-agent implementation with enhanced features.
    
    This class provides backward compatibility while using the complete
    multi-agent context-filtering chain as described in the research.
    """
    
    def __init__(self, api_key: str = None, config: Dict[str, Any] = None, enhanced_config: EnhancedPipelineConfig = None):
        """Initialize the TMM pipeline with multi-agent implementation and enhanced features."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.config = config or self._default_config()
        self.enhanced_config = enhanced_config or EnhancedPipelineConfig()
        
        # Use the proper multi-agent pipeline
        self.multi_agent_pipeline = create_multi_agent_pipeline(self.api_key, self.config)
        
        # Initialize enhanced components
        self._initialize_enhanced_components()
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.performance_metrics = {}
        
        logger.info("TMM Pipeline initialized with multi-agent implementation and enhanced features")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the multi-agent pipeline."""
        return {
            "relevance_threshold": 0.5,
            "confidence_threshold": 0.6,
            "max_context_length": 1000,
            "memory_retrieval_limit": 10
        }
    
    def _initialize_enhanced_components(self):
        """Initialize all enhanced components."""
        try:
            # Initialize adaptive memory retriever
            if self.enhanced_config.enable_adaptive_retrieval:
                from memory.adaptive_retrieval import AdaptiveMemoryRetriever
                self.adaptive_retriever = AdaptiveMemoryRetriever(
                    memory_store=self.multi_agent_pipeline.memory_store
                )
                logger.info("Adaptive memory retriever initialized")
        except ImportError:
            logger.warning("Adaptive memory retriever not available")
        
        try:
            # Initialize multi-agent coordinator
            if self.enhanced_config.enable_enhanced_coordination:
                from agents.coordinator import MultiAgentCoordinator, AgentRole
                self.coordinator = MultiAgentCoordinator()
                self._register_agents_with_coordinator()
                logger.info("Multi-agent coordinator initialized")
        except ImportError:
            logger.warning("Multi-agent coordinator not available")
        
        try:
            # Initialize enhanced truth verifier
            if self.enhanced_config.enable_advanced_verification:
                from truth.enhanced_verifier import EnhancedTruthVerifier
                self.enhanced_verifier = EnhancedTruthVerifier()
                logger.info("Enhanced truth verifier initialized")
        except ImportError:
            logger.warning("Enhanced truth verifier not available")
        
        try:
            # Initialize research analytics
            if self.enhanced_config.enable_research_analytics:
                from monitoring.research_analytics import ResearchAnalytics
                self.analytics = ResearchAnalytics()
                logger.info("Research analytics initialized")
        except ImportError:
            logger.warning("Research analytics not available")
    
    def _register_agents_with_coordinator(self):
        """Register all agents with the coordinator."""
        try:
            from agents.coordinator import AgentRole
            
            # Register strategic planner
            if hasattr(self.multi_agent_pipeline, 'strategic_planner'):
                self.coordinator.register_agent(
                    'strategic_planner',
                    AgentRole.STRATEGIC_PLANNER,
                    self.multi_agent_pipeline.strategic_planner
                )
            
            # Register TACS filter
            if hasattr(self.multi_agent_pipeline, 'tacs_filter'):
                self.coordinator.register_agent(
                    'tacs_filter',
                    AgentRole.TACS_FILTER,
                    self.multi_agent_pipeline.tacs_filter
                )
            
            # Register truth verifier
            if hasattr(self.multi_agent_pipeline, 'truth_verifier'):
                self.coordinator.register_agent(
                    'truth_verifier',
                    AgentRole.TRUTH_VERIFIER,
                    self.multi_agent_pipeline.truth_verifier
                )
            
            # Register memory curator
            if hasattr(self.multi_agent_pipeline, 'memory_curator'):
                self.coordinator.register_agent(
                    'memory_curator',
                    AgentRole.MEMORY_CURATOR,
                    self.multi_agent_pipeline.memory_curator
                )
            
            # Register responder
            if hasattr(self.multi_agent_pipeline, 'responder'):
                self.coordinator.register_agent(
                    'responder',
                    AgentRole.RESPONDER,
                    self.multi_agent_pipeline.responder
                )
        except Exception as e:
            logger.warning(f"Failed to register agents with coordinator: {e}")
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """
        Process user input through the multi-agent TMM pipeline with enhanced features.
        
        Args:
            user_input: The user's input string
            context: Additional context information
            
        Returns:
            Generated response string
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Record request start
            if self.enhanced_config.enable_research_analytics and hasattr(self, 'analytics'):
                self.analytics.record_request(True, 0.0, {'component': 'pipeline_start'})
            
            # Enhanced processing
            if self.enhanced_config.enable_enhanced_coordination and hasattr(self, 'coordinator'):
                response = self._process_with_coordination(user_input, context)
            else:
                response = self._process_with_enhancements(user_input, context)
            
            # Record successful request
            processing_time = time.time() - start_time
            self.successful_requests += 1
            
            if self.enhanced_config.enable_research_analytics and hasattr(self, 'analytics'):
                self.analytics.record_request(True, processing_time, {
                    'component': 'pipeline_complete',
                    'response_length': len(response)
                })
            
            logger.debug(f"Enhanced processing completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            # Record failed request
            processing_time = time.time() - start_time
            logger.error(f"Enhanced processing failed: {e}")
            
            if self.enhanced_config.enable_research_analytics and hasattr(self, 'analytics'):
                self.analytics.record_request(False, processing_time, {
                    'component': 'pipeline_error',
                    'error': str(e)
                })
            
            # Fallback to base pipeline
            return self._fallback_processing(user_input, context)
    
    def _process_with_coordination(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process input using enhanced coordination."""
        try:
            # Use coordinator to orchestrate processing
            coordination_result = self.coordinator.coordinate_processing(user_input, context)
            
            # Extract response from coordination result
            if 'agent_results' in coordination_result:
                responder_result = coordination_result['agent_results'].get('responder', {})
                if 'result' in responder_result:
                    return responder_result['result'].get('output', 'Processing completed')
            
            # Fallback if coordination doesn't provide response
            return self._process_with_enhancements(user_input, context)
        except Exception as e:
            logger.warning(f"Coordination processing failed: {e}")
            return self._process_with_enhancements(user_input, context)
    
    def _process_with_enhancements(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process input using base pipeline with enhancements."""
        # Enhanced memory retrieval
        if self.enhanced_config.enable_adaptive_retrieval and hasattr(self, 'adaptive_retriever'):
            try:
                relevant_memories = self.adaptive_retriever.retrieve_relevant_memories(user_input, context)
                # Add retrieved memories to context
                if context is None:
                    context = {}
                context['retrieved_memories'] = relevant_memories
            except Exception as e:
                logger.warning(f"Adaptive retrieval failed: {e}")
        
        # Enhanced truth verification
        if self.enhanced_config.enable_advanced_verification and hasattr(self, 'enhanced_verifier'):
            try:
                # Get existing memories for verification
                existing_memories = []
                if hasattr(self.multi_agent_pipeline, 'memory_store'):
                    # Get recent memories for verification
                    for tier in ['L1_WORKING', 'L2_SUMMARIZED', 'L3_ARCHIVAL']:
                        try:
                            tier_memories = self.multi_agent_pipeline.memory_store.search_by_tier(tier)
                            existing_memories.extend(tier_memories)
                        except:
                            pass
                
                # Perform enhanced verification
                verification_result = self.enhanced_verifier.verify_truth(
                    user_input, context, existing_memories
                )
                
                # Add verification results to context
                if context is None:
                    context = {}
                context['verification_result'] = verification_result
            except Exception as e:
                logger.warning(f"Enhanced verification failed: {e}")
        
        # Process with base pipeline
        return self.multi_agent_pipeline.process(user_input)
    
    def _fallback_processing(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Fallback processing when enhanced features fail."""
        logger.warning("Using fallback processing due to enhanced feature failure")
        return self.multi_agent_pipeline.process(user_input)
    
    def reset_memory(self):
        """Reset the memory store."""
        self.multi_agent_pipeline.memory_store.reset_memory()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        return self.multi_agent_pipeline.memory_store.get_memory_summary()
    
    def get_truth_verification_calls(self) -> int:
        """Get the number of truth verification calls made."""
        return self.multi_agent_pipeline.get_truth_verification_calls()
    
    def get_contradiction_detections(self) -> int:
        """Get the number of contradictions detected."""
        return self.multi_agent_pipeline.get_contradiction_detections()
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics.
        
        Returns:
            Dictionary containing performance metrics and insights
        """
        analytics = {
            'pipeline_metrics': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'enhanced_features_enabled': {
                    'adaptive_retrieval': self.enhanced_config.enable_adaptive_retrieval,
                    'enhanced_coordination': self.enhanced_config.enable_enhanced_coordination,
                    'advanced_verification': self.enhanced_config.enable_advanced_verification,
                    'research_analytics': self.enhanced_config.enable_research_analytics
                }
            }
        }
        
        # Add component-specific analytics
        if self.enhanced_config.enable_adaptive_retrieval and hasattr(self, 'adaptive_retriever'):
            analytics['adaptive_retrieval'] = self.adaptive_retriever.get_performance_analytics()
        
        if self.enhanced_config.enable_enhanced_coordination and hasattr(self, 'coordinator'):
            analytics['coordination'] = self.coordinator.get_coordination_analytics()
        
        if self.enhanced_config.enable_advanced_verification and hasattr(self, 'enhanced_verifier'):
            analytics['verification'] = self.enhanced_verifier.get_verification_analytics()
        
        if self.enhanced_config.enable_research_analytics and hasattr(self, 'analytics'):
            analytics['research_analytics'] = self.analytics.get_comprehensive_analytics()
        
        return analytics
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            System health information
        """
        health = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'component_status': {},
            'performance_indicators': {}
        }
        
        # Check pipeline health
        success_rate = self.successful_requests / max(self.total_requests, 1)
        health['component_status']['pipeline'] = 'healthy' if success_rate > 0.9 else 'degraded'
        health['performance_indicators']['success_rate'] = success_rate
        
        # Check component health
        if self.enhanced_config.enable_research_analytics and hasattr(self, 'analytics'):
            try:
                analytics_health = self.analytics.get_system_health()
                health['component_status']['analytics'] = analytics_health['status']
                health['performance_indicators'].update(analytics_health)
            except:
                health['component_status']['analytics'] = 'unknown'
        
        # Determine overall status
        component_statuses = list(health['component_status'].values())
        if 'critical' in component_statuses:
            health['overall_status'] = 'critical'
        elif 'degraded' in component_statuses:
            health['overall_status'] = 'degraded'
        elif 'warning' in component_statuses:
            health['overall_status'] = 'warning'
        
        return health

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