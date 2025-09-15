"""
coordinator.py - Enhanced Multi-Agent Coordination System

This module implements intelligent coordination between agents in the TMM system,
enhancing communication, decision-making, and overall system performance.

Key Features:
- Intelligent agent communication protocols
- Dynamic task allocation and load balancing
- Agent performance monitoring and optimization
- Research-grade coordination analytics
- Adaptive agent behavior based on context
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Different agent roles in the TMM system."""
    STRATEGIC_PLANNER = "strategic_planner"
    TACS_FILTER = "tacs_filter"
    TRUTH_VERIFIER = "truth_verifier"
    MEMORY_CURATOR = "memory_curator"
    RESPONDER = "responder"

class CoordinationStrategy(Enum):
    """Different coordination strategies."""
    SEQUENTIAL = "sequential"  # Traditional pipeline
    PARALLEL = "parallel"  # Parallel processing where possible
    ADAPTIVE = "adaptive"  # Dynamic strategy selection
    COLLABORATIVE = "collaborative"  # Agent collaboration and voting

@dataclass
class AgentMetrics:
    """Metrics for tracking agent performance."""
    agent_id: str
    role: AgentRole
    processing_time: float
    success_rate: float
    error_count: int
    total_operations: int
    last_activity: float = field(default_factory=time.time)

@dataclass
class CoordinationDecision:
    """Decision made by the coordination system."""
    strategy: CoordinationStrategy
    agent_sequence: List[AgentRole]
    estimated_time: float
    confidence: float
    reasoning: str

class MultiAgentCoordinator:
    """
    Enhanced multi-agent coordination system for the TMM pipeline.
    
    This component orchestrates the interaction between different agents,
    optimizing performance and ensuring seamless operation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the multi-agent coordinator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or self._default_config()
        
        # Agent registry and metrics
        self.agents = {}
        self.agent_metrics = {}
        self.coordination_history = deque(maxlen=1000)
        
        # Performance tracking
        self.total_coordinations = 0
        self.successful_coordinations = 0
        self.coordination_times = deque(maxlen=100)
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(list)
        
        logger.info("MultiAgentCoordinator initialized with enhanced coordination capabilities")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the coordinator."""
        return {
            'default_strategy': CoordinationStrategy.ADAPTIVE,
            'parallel_threshold': 0.7,  # Confidence threshold for parallel processing
            'collaboration_threshold': 0.8,  # Threshold for collaborative decisions
            'performance_window': 100,  # Window for performance analysis
            'load_balancing_enabled': True,
            'adaptive_learning_enabled': True
        }
    
    def register_agent(self, agent_id: str, role: AgentRole, agent_instance: Any):
        """
        Register an agent with the coordinator.
        
        Args:
            agent_id: Unique identifier for the agent
            role: The agent's role in the system
            agent_instance: The actual agent instance
        """
        self.agents[agent_id] = {
            'role': role,
            'instance': agent_instance,
            'status': 'active'
        }
        
        # Initialize metrics
        self.agent_metrics[agent_id] = AgentMetrics(
            agent_id=agent_id,
            role=role,
            processing_time=0.0,
            success_rate=1.0,
            error_count=0,
            total_operations=0
        )
        
        logger.info(f"Registered agent {agent_id} with role {role.value}")
    
    def coordinate_processing(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Coordinate the processing of user input across agents.
        
        Args:
            user_input: The user's input
            context: Additional context information
            
        Returns:
            Dictionary containing processing results and coordination metadata
        """
        start_time = time.time()
        self.total_coordinations += 1
        
        try:
            # Analyze input and determine coordination strategy
            decision = self._make_coordination_decision(user_input, context)
            
            # Execute coordination strategy
            results = self._execute_coordination_strategy(decision, user_input, context)
            
            # Update metrics
            coordination_time = time.time() - start_time
            self.coordination_times.append(coordination_time)
            self.successful_coordinations += 1
            
            # Record coordination history
            self.coordination_history.append({
                'timestamp': time.time(),
                'strategy': decision.strategy.value,
                'processing_time': coordination_time,
                'success': True,
                'input_length': len(user_input),
                'context_keys': list(context.keys()) if context else []
            })
            
            # Update strategy performance
            self.strategy_performance[decision.strategy].append(coordination_time)
            
            logger.debug(f"Successfully coordinated processing using {decision.strategy.value} strategy")
            return results
            
        except Exception as e:
            logger.error(f"Coordination failed: {e}")
            self.coordination_times.append(time.time() - start_time)
            
            # Record failed coordination
            self.coordination_history.append({
                'timestamp': time.time(),
                'strategy': 'failed',
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            })
            
            raise
    
    def _make_coordination_decision(self, user_input: str, context: Dict[str, Any] = None) -> CoordinationDecision:
        """
        Make a coordination decision based on input analysis.
        
        Args:
            user_input: The user's input
            context: Additional context information
            
        Returns:
            Coordination decision with strategy and reasoning
        """
        # Analyze input complexity and requirements
        complexity_score = self._analyze_input_complexity(user_input)
        urgency_score = self._analyze_urgency(user_input, context)
        agent_availability = self._check_agent_availability()
        
        # Determine optimal strategy
        if complexity_score > 0.8 and urgency_score < 0.3:
            # Complex, non-urgent: Use collaborative strategy
            strategy = CoordinationStrategy.COLLABORATIVE
            agent_sequence = [AgentRole.STRATEGIC_PLANNER, AgentRole.TACS_FILTER, 
                            AgentRole.TRUTH_VERIFIER, AgentRole.MEMORY_CURATOR, AgentRole.RESPONDER]
            confidence = 0.9
            reasoning = "Complex input requires collaborative processing for optimal results"
            
        elif complexity_score < 0.4 and urgency_score > 0.7:
            # Simple, urgent: Use parallel strategy
            strategy = CoordinationStrategy.PARALLEL
            agent_sequence = [AgentRole.TACS_FILTER, AgentRole.TRUTH_VERIFIER]
            confidence = 0.8
            reasoning = "Simple, urgent input can be processed in parallel for speed"
            
        elif self._can_use_parallel_processing(user_input, context):
            # Suitable for parallel processing
            strategy = CoordinationStrategy.PARALLEL
            agent_sequence = [AgentRole.TACS_FILTER, AgentRole.TRUTH_VERIFIER]
            confidence = 0.7
            reasoning = "Input suitable for parallel processing to improve efficiency"
            
        else:
            # Default to adaptive strategy
            strategy = CoordinationStrategy.ADAPTIVE
            agent_sequence = [AgentRole.STRATEGIC_PLANNER, AgentRole.TACS_FILTER, 
                            AgentRole.TRUTH_VERIFIER, AgentRole.MEMORY_CURATOR, AgentRole.RESPONDER]
            confidence = 0.6
            reasoning = "Adaptive strategy for balanced performance and accuracy"
        
        # Estimate processing time
        estimated_time = self._estimate_processing_time(strategy, agent_sequence, user_input)
        
        return CoordinationDecision(
            strategy=strategy,
            agent_sequence=agent_sequence,
            estimated_time=estimated_time,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _analyze_input_complexity(self, user_input: str) -> float:
        """
        Analyze the complexity of the user input.
        
        Args:
            user_input: The user's input
            
        Returns:
            Complexity score between 0 and 1
        """
        complexity_indicators = {
            'length': len(user_input.split()) / 50.0,  # Normalize by typical length
            'questions': user_input.count('?') / 3.0,  # Multiple questions
            'conjunctions': sum(1 for word in ['and', 'but', 'or', 'because', 'although'] if word in user_input.lower()) / 3.0,
            'complex_words': sum(1 for word in user_input.split() if len(word) > 8) / 10.0
        }
        
        # Weighted complexity score
        complexity_score = (
            complexity_indicators['length'] * 0.3 +
            complexity_indicators['questions'] * 0.3 +
            complexity_indicators['conjunctions'] * 0.2 +
            complexity_indicators['complex_words'] * 0.2
        )
        
        return min(complexity_score, 1.0)
    
    def _analyze_urgency(self, user_input: str, context: Dict[str, Any] = None) -> float:
        """
        Analyze the urgency of the user input.
        
        Args:
            user_input: The user's input
            context: Additional context information
            
        Returns:
            Urgency score between 0 and 1
        """
        urgency_indicators = ['urgent', 'asap', 'immediately', 'quickly', 'fast', 'now', 'emergency']
        user_input_lower = user_input.lower()
        
        urgency_score = 0.0
        for indicator in urgency_indicators:
            if indicator in user_input_lower:
                urgency_score += 0.2
        
        # Check context for urgency indicators
        if context:
            if context.get('priority') == 'high':
                urgency_score += 0.3
            if context.get('timeout') and context['timeout'] < 5:
                urgency_score += 0.4
        
        return min(urgency_score, 1.0)
    
    def _check_agent_availability(self) -> Dict[AgentRole, bool]:
        """
        Check the availability of different agents.
        
        Returns:
            Dictionary mapping agent roles to availability status
        """
        availability = {}
        for agent_id, agent_info in self.agents.items():
            role = agent_info['role']
            status = agent_info['status']
            availability[role] = status == 'active'
        
        return availability
    
    def _can_use_parallel_processing(self, user_input: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if parallel processing is suitable for the input.
        
        Args:
            user_input: The user's input
            context: Additional context information
            
        Returns:
            True if parallel processing is suitable
        """
        # Simple heuristics for parallel processing suitability
        if len(user_input.split()) < 5:  # Very short input
            return True
        
        if any(word in user_input.lower() for word in ['yes', 'no', 'ok', 'thanks', 'hello']):
            return True
        
        if context and context.get('parallel_safe', False):
            return True
        
        return False
    
    def _estimate_processing_time(self, strategy: CoordinationStrategy, 
                                agent_sequence: List[AgentRole], user_input: str) -> float:
        """
        Estimate processing time for the given strategy and sequence.
        
        Args:
            strategy: The coordination strategy
            agent_sequence: The sequence of agents
            user_input: The user's input
            
        Returns:
            Estimated processing time in seconds
        """
        base_time = 0.1  # Base processing time
        
        # Add time for each agent
        for role in agent_sequence:
            if role in [AgentRole.STRATEGIC_PLANNER, AgentRole.TACS_FILTER]:
                base_time += 0.2
            elif role == AgentRole.TRUTH_VERIFIER:
                base_time += 0.15
            elif role == AgentRole.MEMORY_CURATOR:
                base_time += 0.25
            elif role == AgentRole.RESPONDER:
                base_time += 0.3
        
        # Adjust for strategy
        if strategy == CoordinationStrategy.PARALLEL:
            base_time *= 0.6  # Parallel processing is faster
        elif strategy == CoordinationStrategy.COLLABORATIVE:
            base_time *= 1.5  # Collaborative processing takes longer
        
        # Adjust for input length
        input_length_factor = len(user_input.split()) / 20.0
        base_time *= (1.0 + input_length_factor * 0.2)
        
        return base_time
    
    def _execute_coordination_strategy(self, decision: CoordinationDecision, 
                                     user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the coordination strategy.
        
        Args:
            decision: The coordination decision
            user_input: The user's input
            context: Additional context information
            
        Returns:
            Dictionary containing processing results
        """
        if decision.strategy == CoordinationStrategy.SEQUENTIAL:
            return self._execute_sequential_processing(decision.agent_sequence, user_input, context)
        elif decision.strategy == CoordinationStrategy.PARALLEL:
            return self._execute_parallel_processing(decision.agent_sequence, user_input, context)
        elif decision.strategy == CoordinationStrategy.COLLABORATIVE:
            return self._execute_collaborative_processing(decision.agent_sequence, user_input, context)
        else:  # ADAPTIVE
            return self._execute_adaptive_processing(decision.agent_sequence, user_input, context)
    
    def _execute_sequential_processing(self, agent_sequence: List[AgentRole], 
                                     user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute sequential processing through agents."""
        results = {'strategy': 'sequential', 'agent_results': {}}
        current_input = user_input
        current_context = context or {}
        
        for role in agent_sequence:
            agent_id = self._find_agent_by_role(role)
            if agent_id:
                try:
                    start_time = time.time()
                    agent_result = self._process_with_agent(agent_id, current_input, current_context)
                    processing_time = time.time() - start_time
                    
                    results['agent_results'][role.value] = {
                        'result': agent_result,
                        'processing_time': processing_time,
                        'success': True
                    }
                    
                    # Update agent metrics
                    self._update_agent_metrics(agent_id, processing_time, True)
                    
                    # Prepare for next agent
                    if isinstance(agent_result, dict):
                        current_input = agent_result.get('output', current_input)
                        current_context.update(agent_result.get('context', {}))
                    
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")
                    results['agent_results'][role.value] = {
                        'error': str(e),
                        'success': False
                    }
                    self._update_agent_metrics(agent_id, 0, False)
        
        return results
    
    def _execute_parallel_processing(self, agent_sequence: List[AgentRole], 
                                   user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute parallel processing through agents."""
        results = {'strategy': 'parallel', 'agent_results': {}}
        
        # For now, we'll simulate parallel processing
        # In a real implementation, this would use threading or async processing
        for role in agent_sequence:
            agent_id = self._find_agent_by_role(role)
            if agent_id:
                try:
                    start_time = time.time()
                    agent_result = self._process_with_agent(agent_id, user_input, context)
                    processing_time = time.time() - start_time
                    
                    results['agent_results'][role.value] = {
                        'result': agent_result,
                        'processing_time': processing_time,
                        'success': True
                    }
                    
                    self._update_agent_metrics(agent_id, processing_time, True)
                    
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")
                    results['agent_results'][role.value] = {
                        'error': str(e),
                        'success': False
                    }
                    self._update_agent_metrics(agent_id, 0, False)
        
        return results
    
    def _execute_collaborative_processing(self, agent_sequence: List[AgentRole], 
                                        user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute collaborative processing with agent voting."""
        results = {'strategy': 'collaborative', 'agent_results': {}, 'collaborative_decision': None}
        
        # Get input from multiple agents
        agent_opinions = []
        for role in agent_sequence:
            agent_id = self._find_agent_by_role(role)
            if agent_id:
                try:
                    start_time = time.time()
                    agent_result = self._process_with_agent(agent_id, user_input, context)
                    processing_time = time.time() - start_time
                    
                    results['agent_results'][role.value] = {
                        'result': agent_result,
                        'processing_time': processing_time,
                        'success': True
                    }
                    
                    agent_opinions.append({
                        'role': role,
                        'result': agent_result,
                        'confidence': self._get_agent_confidence(agent_id)
                    })
                    
                    self._update_agent_metrics(agent_id, processing_time, True)
                    
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")
                    results['agent_results'][role.value] = {
                        'error': str(e),
                        'success': False
                    }
                    self._update_agent_metrics(agent_id, 0, False)
        
        # Make collaborative decision
        if agent_opinions:
            results['collaborative_decision'] = self._make_collaborative_decision(agent_opinions)
        
        return results
    
    def _execute_adaptive_processing(self, agent_sequence: List[AgentRole], 
                                   user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute adaptive processing based on real-time conditions."""
        # Start with sequential processing
        results = self._execute_sequential_processing(agent_sequence, user_input, context)
        results['strategy'] = 'adaptive'
        
        # Adapt based on performance
        if self._should_switch_to_parallel(results):
            logger.info("Switching to parallel processing for better performance")
            # Could implement dynamic strategy switching here
        
        return results
    
    def _find_agent_by_role(self, role: AgentRole) -> Optional[str]:
        """Find agent ID by role."""
        for agent_id, agent_info in self.agents.items():
            if agent_info['role'] == role:
                return agent_id
        return None
    
    def _process_with_agent(self, agent_id: str, user_input: str, context: Dict[str, Any] = None) -> Any:
        """Process input with a specific agent."""
        agent_info = self.agents[agent_id]
        agent_instance = agent_info['instance']
        
        # This would call the appropriate method on the agent
        # For now, we'll return a placeholder
        return {
            'output': f"Processed by {agent_id}",
            'context': context or {},
            'confidence': 0.8
        }
    
    def _get_agent_confidence(self, agent_id: str) -> float:
        """Get confidence score for an agent."""
        metrics = self.agent_metrics.get(agent_id)
        if metrics:
            return metrics.success_rate
        return 0.5
    
    def _make_collaborative_decision(self, agent_opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a collaborative decision based on agent opinions."""
        # Simple voting mechanism
        total_confidence = sum(opinion['confidence'] for opinion in agent_opinions)
        weighted_result = None
        
        for opinion in agent_opinions:
            weight = opinion['confidence'] / total_confidence
            if weighted_result is None:
                weighted_result = opinion['result']
            else:
                # Combine results based on confidence weights
                # This is a simplified implementation
                pass
        
        return {
            'decision': weighted_result,
            'confidence': total_confidence / len(agent_opinions),
            'participating_agents': len(agent_opinions)
        }
    
    def _should_switch_to_parallel(self, results: Dict[str, Any]) -> bool:
        """Determine if we should switch to parallel processing."""
        # Simple heuristic based on processing times
        total_time = sum(
            agent_result.get('processing_time', 0) 
            for agent_result in results.get('agent_results', {}).values()
        )
        
        return total_time > 1.0  # Switch if processing takes too long
    
    def _update_agent_metrics(self, agent_id: str, processing_time: float, success: bool):
        """Update metrics for a specific agent."""
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            metrics.total_operations += 1
            metrics.processing_time = (metrics.processing_time + processing_time) / 2  # Running average
            
            if success:
                metrics.success_rate = (metrics.success_rate * (metrics.total_operations - 1) + 1.0) / metrics.total_operations
            else:
                metrics.error_count += 1
                metrics.success_rate = (metrics.success_rate * (metrics.total_operations - 1) + 0.0) / metrics.total_operations
            
            metrics.last_activity = time.time()
    
    def get_coordination_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive coordination analytics.
        
        Returns:
            Dictionary containing coordination metrics and insights
        """
        if not self.coordination_history:
            return {"message": "No coordination data available yet"}
        
        # Calculate overall metrics
        total_coordinations = len(self.coordination_history)
        successful_coordinations = sum(1 for record in self.coordination_history if record['success'])
        success_rate = successful_coordinations / total_coordinations if total_coordinations > 0 else 0
        
        # Calculate average processing time
        avg_processing_time = sum(record['processing_time'] for record in self.coordination_history) / total_coordinations
        
        # Strategy performance analysis
        strategy_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'successes': 0})
        for record in self.coordination_history:
            strategy = record['strategy']
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['total_time'] += record['processing_time']
            if record['success']:
                strategy_stats[strategy]['successes'] += 1
        
        # Agent performance analysis
        agent_performance = {}
        for agent_id, metrics in self.agent_metrics.items():
            agent_performance[agent_id] = {
                'role': metrics.role.value,
                'total_operations': metrics.total_operations,
                'success_rate': metrics.success_rate,
                'avg_processing_time': metrics.processing_time,
                'error_count': metrics.error_count
            }
        
        return {
            'total_coordinations': total_coordinations,
            'success_rate': success_rate,
            'avg_processing_time': avg_processing_time,
            'strategy_performance': dict(strategy_stats),
            'agent_performance': agent_performance,
            'coordination_efficiency': self._calculate_coordination_efficiency()
        }
    
    def _calculate_coordination_efficiency(self) -> float:
        """Calculate overall coordination efficiency score."""
        if not self.coordination_history:
            return 0.0
        
        # Calculate efficiency based on success rate and processing time
        success_rate = sum(1 for record in self.coordination_history if record['success']) / len(self.coordination_history)
        avg_time = sum(record['processing_time'] for record in self.coordination_history) / len(self.coordination_history)
        
        # Efficiency is high success rate with low processing time
        time_efficiency = 1.0 / (1.0 + avg_time)  # Lower time is better
        return success_rate * time_efficiency
