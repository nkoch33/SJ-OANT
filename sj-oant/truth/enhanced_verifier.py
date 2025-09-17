"""
enhanced_verifier.py - Enhanced Truth Verification System

This module implements advanced truth verification capabilities that enhance the TMM system's
ability to assess truthfulness, detect contradictions, and maintain high-quality memory.

Key Features:
- Multi-layered verification strategies
- Contradiction detection and resolution
- Confidence calibration and uncertainty quantification
- Research-grade verification analytics
- Adaptive verification based on content type
"""

import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from core.types import ConfidenceScores, MemoryRecord, MemoryTier
from truth.verifier import RuleBasedVerifier

logger = logging.getLogger(__name__)

class VerificationStrategy(Enum):
    """Different verification strategies."""
    RULE_BASED = "rule_based"
    CONSISTENCY_CHECK = "consistency_check"
    CONTRADICTION_DETECTION = "contradiction_detection"
    EVIDENCE_WEIGHTING = "evidence_weighting"
    TEMPORAL_ANALYSIS = "temporal_analysis"

class ContradictionType(Enum):
    """Types of contradictions that can be detected."""
    FACTUAL = "factual"  # Direct factual contradiction
    TEMPORAL = "temporal"  # Time-based contradiction
    LOGICAL = "logical"  # Logical inconsistency
    SEMANTIC = "semantic"  # Semantic contradiction

@dataclass
class VerificationResult:
    """Enhanced verification result with detailed analysis."""
    confidence_scores: ConfidenceScores
    verification_strategy: VerificationStrategy
    contradiction_detected: bool
    contradiction_type: Optional[ContradictionType]
    evidence_strength: float
    uncertainty_quantification: float
    verification_time: float
    reasoning: str
    recommendations: List[str]

@dataclass
class ContradictionReport:
    """Report of detected contradictions."""
    contradiction_id: str
    contradiction_type: ContradictionType
    conflicting_records: List[str]
    confidence_difference: float
    resolution_strategy: str
    timestamp: float

class EnhancedTruthVerifier:
    """
    Enhanced truth verification system with advanced capabilities.
    
    This component provides sophisticated truth verification that goes beyond
    basic rule-based verification to include contradiction detection, evidence
    weighting, and adaptive verification strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced truth verifier.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or self._default_config()
        
        # Initialize base verifier
        self.base_verifier = RuleBasedVerifier()
        
        # Contradiction detection
        self.contradiction_patterns = self._initialize_contradiction_patterns()
        self.detected_contradictions = []
        
        # Evidence tracking
        self.evidence_history = defaultdict(list)
        self.consistency_scores = defaultdict(list)
        
        # Performance tracking
        self.verification_metrics = {
            'total_verifications': 0,
            'contradictions_detected': 0,
            'avg_verification_time': 0.0,
            'strategy_performance': defaultdict(list)
        }
        
        logger.info("EnhancedTruthVerifier initialized with advanced verification capabilities")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the enhanced verifier."""
        return {
            'contradiction_threshold': 0.3,
            'evidence_weight_threshold': 0.6,
            'consistency_threshold': 0.7,
            'temporal_window_hours': 24,
            'max_verification_time': 5.0,
            'enable_contradiction_detection': True,
            'enable_evidence_weighting': True,
            'enable_temporal_analysis': True
        }
    
    def verify_truth(self, content: str, context: Dict[str, Any] = None, 
                    existing_memories: List[MemoryRecord] = None) -> VerificationResult:
        """
        Perform enhanced truth verification.
        
        Args:
            content: Content to verify
            context: Additional context information
            existing_memories: Existing memories for consistency checking
            
        Returns:
            Enhanced verification result
        """
        start_time = time.time()
        self.verification_metrics['total_verifications'] += 1
        
        try:
            # Determine optimal verification strategy
            strategy = self._select_verification_strategy(content, context, existing_memories)
            
            # Perform base verification
            base_result = self.base_verifier.verify_truth(content, context)
            
            # Apply enhanced verification based on strategy
            enhanced_result = self._apply_enhanced_verification(
                content, context, existing_memories, base_result, strategy
            )
            
            # Update metrics
            verification_time = time.time() - start_time
            self._update_verification_metrics(strategy, verification_time, enhanced_result)
            
            logger.debug(f"Enhanced verification completed using {strategy.value} strategy")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced verification failed: {e}")
            # Return fallback result
            return self._create_fallback_result(content, str(e))
    
    def _select_verification_strategy(self, content: str, context: Dict[str, Any] = None,
                                    existing_memories: List[MemoryRecord] = None) -> VerificationStrategy:
        """
        Select optimal verification strategy based on content and context.
        
        Args:
            content: Content to verify
            context: Additional context information
            existing_memories: Existing memories for analysis
            
        Returns:
            Optimal verification strategy
        """
        # Analyze content characteristics
        content_analysis = self._analyze_content_characteristics(content)
        
        # Check for existing memories that might conflict
        has_potential_conflicts = self._has_potential_conflicts(content, existing_memories)
        
        # Determine strategy based on analysis
        if has_potential_conflicts:
            return VerificationStrategy.CONTRADICTION_DETECTION
        elif content_analysis['has_temporal_elements']:
            return VerificationStrategy.TEMPORAL_ANALYSIS
        elif content_analysis['has_evidence_indicators']:
            return VerificationStrategy.EVIDENCE_WEIGHTING
        elif existing_memories and len(existing_memories) > 1:
            return VerificationStrategy.CONSISTENCY_CHECK
        else:
            return VerificationStrategy.RULE_BASED
    
    def _analyze_content_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze content characteristics to inform verification strategy."""
        content_lower = content.lower()
        
        # Temporal indicators
        temporal_indicators = ['yesterday', 'today', 'tomorrow', 'last week', 'next month', 'at 3pm', 'on monday']
        has_temporal_elements = any(indicator in content_lower for indicator in temporal_indicators)
        
        # Evidence indicators
        evidence_indicators = ['according to', 'research shows', 'studies indicate', 'data suggests', 'evidence shows']
        has_evidence_indicators = any(indicator in content_lower for indicator in evidence_indicators)
        
        # Factual indicators
        factual_indicators = ['is', 'are', 'was', 'were', 'will be', 'has been', 'have been']
        has_factual_indicators = any(indicator in content_lower for indicator in factual_indicators)
        
        # Uncertainty indicators
        uncertainty_indicators = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'likely', 'probably']
        has_uncertainty_indicators = any(indicator in content_lower for indicator in uncertainty_indicators)
        
        return {
            'has_temporal_elements': has_temporal_elements,
            'has_evidence_indicators': has_evidence_indicators,
            'has_factual_indicators': has_factual_indicators,
            'has_uncertainty_indicators': has_uncertainty_indicators,
            'content_length': len(content.split()),
            'complexity_score': self._calculate_complexity_score(content)
        }
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate content complexity score."""
        words = content.split()
        if not words:
            return 0.0
        
        # Factors that increase complexity
        long_words = sum(1 for word in words if len(word) > 8)
        complex_structures = content.count(',') + content.count(';') + content.count(':')
        questions = content.count('?')
        
        complexity = (
            (long_words / len(words)) * 0.4 +
            (complex_structures / len(words)) * 0.3 +
            (questions / len(words)) * 0.3
        )
        
        return min(complexity, 1.0)
    
    def _has_potential_conflicts(self, content: str, existing_memories: List[MemoryRecord] = None) -> bool:
        """Check if content has potential conflicts with existing memories."""
        if not existing_memories:
            return False
        
        # Simple conflict detection based on keyword overlap
        content_words = set(content.lower().split())
        
        for memory in existing_memories:
            memory_words = set(memory.content.lower().split())
            
            # Check for significant overlap
            overlap = len(content_words.intersection(memory_words))
            if overlap > 3:  # Threshold for potential conflict
                return True
        
        return False
    
    def _apply_enhanced_verification(self, content: str, context: Dict[str, Any] = None,
                                   existing_memories: List[MemoryRecord] = None,
                                   base_result: Any = None, strategy: VerificationStrategy = None) -> VerificationResult:
        """
        Apply enhanced verification based on selected strategy.
        
        Args:
            content: Content to verify
            context: Additional context information
            existing_memories: Existing memories for analysis
            base_result: Base verification result
            strategy: Verification strategy to use
            
        Returns:
            Enhanced verification result
        """
        if strategy == VerificationStrategy.CONTRADICTION_DETECTION:
            return self._contradiction_detection_verification(content, context, existing_memories, base_result)
        elif strategy == VerificationStrategy.CONSISTENCY_CHECK:
            return self._consistency_check_verification(content, context, existing_memories, base_result)
        elif strategy == VerificationStrategy.EVIDENCE_WEIGHTING:
            return self._evidence_weighting_verification(content, context, existing_memories, base_result)
        elif strategy == VerificationStrategy.TEMPORAL_ANALYSIS:
            return self._temporal_analysis_verification(content, context, existing_memories, base_result)
        else:
            return self._rule_based_verification(content, context, existing_memories, base_result)
    
    def _contradiction_detection_verification(self, content: str, context: Dict[str, Any] = None,
                                            existing_memories: List[MemoryRecord] = None,
                                            base_result: Any = None) -> VerificationResult:
        """Perform contradiction detection verification."""
        contradiction_detected = False
        contradiction_type = None
        evidence_strength = 0.5
        uncertainty_quantification = 0.3
        reasoning = "Contradiction detection analysis performed"
        recommendations = []
        
        if existing_memories:
            # Check for contradictions with existing memories
            contradictions = self._detect_contradictions(content, existing_memories)
            
            if contradictions:
                contradiction_detected = True
                contradiction_type = contradictions[0].contradiction_type
                evidence_strength = 0.8  # High evidence of contradiction
                uncertainty_quantification = 0.1  # Low uncertainty when contradiction detected
                reasoning = f"Detected {len(contradictions)} contradictions with existing memories"
                recommendations.append("Review conflicting information before storage")
                recommendations.append("Consider memory consolidation or flagging")
                
                # Record contradictions
                self.detected_contradictions.extend(contradictions)
                self.verification_metrics['contradictions_detected'] += len(contradictions)
        
        # Adjust confidence scores based on contradiction detection
        if base_result and hasattr(base_result, 'confidence_scores'):
            confidence_scores = base_result.confidence_scores
        else:
            confidence_scores = ConfidenceScores(
                overall=0.5,
                factual_accuracy=0.5,
                source_reliability=0.5,
                temporal_consistency=0.5,
                logical_coherence=0.5
            )
        
        # Reduce confidence if contradiction detected
        if contradiction_detected:
            confidence_scores.overall *= 0.3
            confidence_scores.factual_accuracy *= 0.2
            confidence_scores.logical_coherence *= 0.1
        
        return VerificationResult(
            confidence_scores=confidence_scores,
            verification_strategy=VerificationStrategy.CONTRADICTION_DETECTION,
            contradiction_detected=contradiction_detected,
            contradiction_type=contradiction_type,
            evidence_strength=evidence_strength,
            uncertainty_quantification=uncertainty_quantification,
            verification_time=0.1,  # Placeholder
            reasoning=reasoning,
            recommendations=recommendations
        )
    
    def _consistency_check_verification(self, content: str, context: Dict[str, Any] = None,
                                      existing_memories: List[MemoryRecord] = None,
                                      base_result: Any = None) -> VerificationResult:
        """Perform consistency check verification."""
        consistency_score = 0.8  # Default high consistency
        reasoning = "Consistency check performed"
        recommendations = []
        
        if existing_memories:
            # Calculate consistency with existing memories
            consistency_scores = []
            for memory in existing_memories:
                score = self._calculate_consistency_score(content, memory.content)
                consistency_scores.append(score)
            
            if consistency_scores:
                consistency_score = sum(consistency_scores) / len(consistency_scores)
                
                if consistency_score < self.config['consistency_threshold']:
                    reasoning = f"Low consistency detected (score: {consistency_score:.2f})"
                    recommendations.append("Review content for consistency issues")
                else:
                    reasoning = f"High consistency maintained (score: {consistency_score:.2f})"
        
        # Adjust confidence based on consistency
        if base_result and hasattr(base_result, 'confidence_scores'):
            confidence_scores = base_result.confidence_scores
        else:
            confidence_scores = ConfidenceScores(
                overall=0.5,
                factual_accuracy=0.5,
                source_reliability=0.5,
                temporal_consistency=0.5,
                logical_coherence=0.5
            )
        
        # Boost confidence for high consistency
        confidence_scores.overall *= (0.5 + consistency_score * 0.5)
        confidence_scores.logical_coherence *= consistency_score
        
        return VerificationResult(
            confidence_scores=confidence_scores,
            verification_strategy=VerificationStrategy.CONSISTENCY_CHECK,
            contradiction_detected=False,
            contradiction_type=None,
            evidence_strength=consistency_score,
            uncertainty_quantification=1.0 - consistency_score,
            verification_time=0.1,  # Placeholder
            reasoning=reasoning,
            recommendations=recommendations
        )
    
    def _evidence_weighting_verification(self, content: str, context: Dict[str, Any] = None,
                                       existing_memories: List[MemoryRecord] = None,
                                       base_result: Any = None) -> VerificationResult:
        """Perform evidence weighting verification."""
        evidence_strength = self._calculate_evidence_strength(content)
        reasoning = f"Evidence strength: {evidence_strength:.2f}"
        recommendations = []
        
        if evidence_strength < self.config['evidence_weight_threshold']:
            recommendations.append("Content lacks strong evidence indicators")
            recommendations.append("Consider additional verification")
        
        # Adjust confidence based on evidence strength
        if base_result and hasattr(base_result, 'confidence_scores'):
            confidence_scores = base_result.confidence_scores
        else:
            confidence_scores = ConfidenceScores(
                overall=0.5,
                factual_accuracy=0.5,
                source_reliability=0.5,
                temporal_consistency=0.5,
                logical_coherence=0.5
            )
        
        # Boost confidence for strong evidence
        confidence_scores.overall *= (0.3 + evidence_strength * 0.7)
        confidence_scores.source_reliability *= evidence_strength
        
        return VerificationResult(
            confidence_scores=confidence_scores,
            verification_strategy=VerificationStrategy.EVIDENCE_WEIGHTING,
            contradiction_detected=False,
            contradiction_type=None,
            evidence_strength=evidence_strength,
            uncertainty_quantification=1.0 - evidence_strength,
            verification_time=0.1,  # Placeholder
            reasoning=reasoning,
            recommendations=recommendations
        )
    
    def _temporal_analysis_verification(self, content: str, context: Dict[str, Any] = None,
                                      existing_memories: List[MemoryRecord] = None,
                                      base_result: Any = None) -> VerificationResult:
        """Perform temporal analysis verification."""
        temporal_consistency = self._analyze_temporal_consistency(content, existing_memories)
        reasoning = f"Temporal consistency: {temporal_consistency:.2f}"
        recommendations = []
        
        if temporal_consistency < 0.7:
            recommendations.append("Review temporal elements for consistency")
        
        # Adjust confidence based on temporal consistency
        if base_result and hasattr(base_result, 'confidence_scores'):
            confidence_scores = base_result.confidence_scores
        else:
            confidence_scores = ConfidenceScores(
                overall=0.5,
                factual_accuracy=0.5,
                source_reliability=0.5,
                temporal_consistency=0.5,
                logical_coherence=0.5
            )
        
        confidence_scores.temporal_consistency = temporal_consistency
        confidence_scores.overall *= (0.5 + temporal_consistency * 0.5)
        
        return VerificationResult(
            confidence_scores=confidence_scores,
            verification_strategy=VerificationStrategy.TEMPORAL_ANALYSIS,
            contradiction_detected=False,
            contradiction_type=None,
            evidence_strength=temporal_consistency,
            uncertainty_quantification=1.0 - temporal_consistency,
            verification_time=0.1,  # Placeholder
            reasoning=reasoning,
            recommendations=recommendations
        )
    
    def _rule_based_verification(self, content: str, context: Dict[str, Any] = None,
                               existing_memories: List[MemoryRecord] = None,
                               base_result: Any = None) -> VerificationResult:
        """Perform rule-based verification."""
        if base_result and hasattr(base_result, 'confidence_scores'):
            confidence_scores = base_result.confidence_scores
        else:
            confidence_scores = ConfidenceScores(
                overall=0.5,
                factual_accuracy=0.5,
                source_reliability=0.5,
                temporal_consistency=0.5,
                logical_coherence=0.5
            )
        
        return VerificationResult(
            confidence_scores=confidence_scores,
            verification_strategy=VerificationStrategy.RULE_BASED,
            contradiction_detected=False,
            contradiction_type=None,
            evidence_strength=0.5,
            uncertainty_quantification=0.3,
            verification_time=0.1,  # Placeholder
            reasoning="Rule-based verification performed",
            recommendations=[]
        )
    
    def _detect_contradictions(self, content: str, existing_memories: List[MemoryRecord]) -> List[ContradictionReport]:
        """Detect contradictions between content and existing memories."""
        contradictions = []
        
        for memory in existing_memories:
            # Check for direct contradictions
            if self._is_direct_contradiction(content, memory.content):
                contradiction = ContradictionReport(
                    contradiction_id=f"contradiction_{len(contradictions)}",
                    contradiction_type=ContradictionType.FACTUAL,
                    conflicting_records=[memory.id],
                    confidence_difference=0.5,
                    resolution_strategy="flag_for_review",
                    timestamp=time.time()
                )
                contradictions.append(contradiction)
        
        return contradictions
    
    def _is_direct_contradiction(self, content1: str, content2: str) -> bool:
        """Check if two pieces of content directly contradict each other."""
        # Enhanced contradiction detection
        negation_patterns = [
            (r'\b(?:is|are|was|were)\s+(?:not|n\'t)\b', r'\b(?:is|are|was|were)\b'),
            (r'\b(?:does|do|did)\s+(?:not|n\'t)\b', r'\b(?:does|do|did)\b'),
            (r'\b(?:has|have|had)\s+(?:not|n\'t)\b', r'\b(?:has|have|had)\b')
        ]
        
        # Check negation patterns
        for neg_pattern, pos_pattern in negation_patterns:
            if re.search(neg_pattern, content1.lower()) and re.search(pos_pattern, content2.lower()):
                return True
            if re.search(neg_pattern, content2.lower()) and re.search(pos_pattern, content1.lower()):
                return True
        
        # Check for factual contradictions (enhanced)
        factual_contradictions = [
            (r'\b(\d+)\s+(?:hours?|hrs?)\b', r'\b(\d+)\s+(?:hours?|hrs?)\b'),  # Different durations
            (r'\b(\$\d+)\b', r'\b(\$\d+)\b'),  # Different prices
            (r'\bin\s+(\w+)\b', r'\bin\s+(\w+)\b'),  # Different locations
            (r'\bat\s+(\d+:\d+)\b', r'\bat\s+(\d+:\d+)\b')  # Different times
        ]
        
        for pattern in factual_contradictions:
            matches1 = re.findall(pattern, content1.lower())
            matches2 = re.findall(pattern, content2.lower())
            if matches1 and matches2 and matches1 != matches2:
                return True
        
        return False
    
    def _calculate_consistency_score(self, content1: str, content2: str) -> float:
        """Calculate consistency score between two pieces of content."""
        # Simple consistency calculation based on semantic similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_evidence_strength(self, content: str) -> float:
        """Calculate evidence strength of content."""
        evidence_indicators = {
            'strong': ['research shows', 'studies prove', 'data confirms', 'evidence demonstrates'],
            'moderate': ['suggests', 'indicates', 'appears to', 'likely'],
            'weak': ['might', 'could', 'possibly', 'perhaps']
        }
        
        content_lower = content.lower()
        strength_score = 0.0
        
        for strength, indicators in evidence_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    if strength == 'strong':
                        strength_score += 0.3
                    elif strength == 'moderate':
                        strength_score += 0.2
                    else:
                        strength_score += 0.1
        
        return min(strength_score, 1.0)
    
    def _analyze_temporal_consistency(self, content: str, existing_memories: List[MemoryRecord] = None) -> float:
        """Analyze temporal consistency of content."""
        # Extract temporal elements from content
        temporal_elements = self._extract_temporal_elements(content)
        
        if not temporal_elements:
            return 0.8  # Default high consistency for non-temporal content
        
        # Check consistency with existing memories
        if existing_memories:
            for memory in existing_memories:
                memory_temporal = self._extract_temporal_elements(memory.content)
                if memory_temporal and not self._are_temporal_consistent(temporal_elements, memory_temporal):
                    return 0.3  # Low consistency if temporal conflict
        
        return 0.8  # High consistency if no conflicts
    
    def _extract_temporal_elements(self, content: str) -> List[str]:
        """Extract temporal elements from content."""
        temporal_patterns = [
            r'\b(?:yesterday|today|tomorrow)\b',
            r'\b(?:last|next)\s+(?:week|month|year)\b',
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b\d{1,2}:\d{2}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b'
        ]
        
        temporal_elements = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, content.lower())
            temporal_elements.extend(matches)
        
        return temporal_elements
    
    def _are_temporal_consistent(self, temporal1: List[str], temporal2: List[str]) -> bool:
        """Check if two sets of temporal elements are consistent."""
        # Simple consistency check - in a real system, this would be more sophisticated
        return len(set(temporal1).intersection(set(temporal2))) > 0
    
    def _initialize_contradiction_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for contradiction detection."""
        return {
            'negation_patterns': [
                r'\b(?:is|are|was|were)\s+(?:not|n\'t)\b',
                r'\b(?:does|do|did)\s+(?:not|n\'t)\b',
                r'\b(?:has|have|had)\s+(?:not|n\'t)\b'
            ],
            'opposite_patterns': [
                (r'\b(?:hot|warm)\b', r'\b(?:cold|cool)\b'),
                (r'\b(?:big|large)\b', r'\b(?:small|tiny)\b'),
                (r'\b(?:fast|quick)\b', r'\b(?:slow|slowly)\b')
            ]
        }
    
    def _update_verification_metrics(self, strategy: VerificationStrategy, verification_time: float, result: VerificationResult):
        """Update verification metrics."""
        self.verification_metrics['strategy_performance'][strategy.value].append(verification_time)
        
        # Update average verification time
        total_verifications = self.verification_metrics['total_verifications']
        current_avg = self.verification_metrics['avg_verification_time']
        self.verification_metrics['avg_verification_time'] = (
            (current_avg * (total_verifications - 1) + verification_time) / total_verifications
        )
    
    def _create_fallback_result(self, content: str, error_message: str) -> VerificationResult:
        """Create a fallback verification result when verification fails."""
        return VerificationResult(
            confidence_scores=ConfidenceScores(
                overall=0.3,
                factual_accuracy=0.3,
                source_reliability=0.3,
                temporal_consistency=0.3,
                logical_coherence=0.3
            ),
            verification_strategy=VerificationStrategy.RULE_BASED,
            contradiction_detected=False,
            contradiction_type=None,
            evidence_strength=0.1,
            uncertainty_quantification=0.9,
            verification_time=0.0,
            reasoning=f"Fallback verification due to error: {error_message}",
            recommendations=["Manual verification recommended"]
        )
    
    def get_verification_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive verification analytics.
        
        Returns:
            Dictionary containing verification metrics and insights
        """
        return {
            'total_verifications': self.verification_metrics['total_verifications'],
            'contradictions_detected': self.verification_metrics['contradictions_detected'],
            'avg_verification_time': self.verification_metrics['avg_verification_time'],
            'strategy_performance': dict(self.verification_metrics['strategy_performance']),
            'recent_contradictions': [
                {
                    'id': c.contradiction_id,
                    'type': c.contradiction_type.value,
                    'timestamp': c.timestamp,
                    'resolution_strategy': c.resolution_strategy
                }
                for c in self.detected_contradictions[-10:]  # Last 10 contradictions
            ],
            'verification_quality': self._assess_verification_quality()
        }
    
    def _assess_verification_quality(self) -> Dict[str, Any]:
        """Assess overall verification quality."""
        total_verifications = self.verification_metrics['total_verifications']
        contradictions_detected = self.verification_metrics['contradictions_detected']
        
        if total_verifications == 0:
            return {'quality_score': 0.0, 'assessment': 'No verification data available'}
        
        # Calculate quality metrics
        contradiction_rate = contradictions_detected / total_verifications
        avg_time = self.verification_metrics['avg_verification_time']
        
        # Quality score based on contradiction detection and efficiency
        quality_score = 0.8  # Base score
        if contradiction_rate > 0.1:  # High contradiction rate
            quality_score -= 0.2
        if avg_time > 2.0:  # Slow verification
            quality_score -= 0.1
        
        if quality_score >= 0.8:
            assessment = "High quality verification"
        elif quality_score >= 0.6:
            assessment = "Good quality verification"
        elif quality_score >= 0.4:
            assessment = "Moderate quality verification"
        else:
            assessment = "Low quality verification - needs improvement"
        
        return {
            'quality_score': quality_score,
            'assessment': assessment,
            'contradiction_rate': contradiction_rate,
            'avg_verification_time': avg_time
        }
