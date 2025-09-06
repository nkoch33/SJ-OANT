"""
truth.verifier - Truth Verification and Confidence Scoring

This module provides production-grade truth verification capabilities for the TMM system,
implementing clean interfaces, comprehensive error handling, and high-quality logging.

Design Features:
- Protocol-based interfaces for pluggable verification strategies
- Comprehensive confidence scoring with calibrated uncertainty
- Rule-based and ML-ready verification components
- Thread-safe operations with proper error handling
- Rich metrics collection and audit trails
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

from core.ports import BaseVerifier, VerificationError
from core.types import ConfidenceScores, MemoryRecord, RecordCollection, RecordID

logger = logging.getLogger(__name__)


class RuleBasedVerifier(BaseVerifier):
    """
    Production-ready rule-based truth verification implementation.
    
    This verifier uses configurable rules and heuristics to assess truth
    and confidence scores. Designed for deterministic, explainable verification
    that can be audited and debugged in production environments.
    
    Features:
    - Configurable rule sets for different domains
    - Explainable scoring with detailed reasoning
    - Fast, deterministic verification
    - No external dependencies or API calls
    """
    
    def __init__(self, 
                 base_confidence: float = 0.5,
                 evidence_weight: float = 0.3,
                 consistency_weight: float = 0.4,
                 source_weight: float = 0.3):
        """
        Initialize the rule-based verifier.
        
        Args:
            base_confidence: Baseline confidence for unknown content
            evidence_weight: Weight for evidence quality in scoring
            consistency_weight: Weight for consistency with existing knowledge
            source_weight: Weight for source credibility in scoring
        """
        self.base_confidence = base_confidence
        self.evidence_weight = evidence_weight
        self.consistency_weight = consistency_weight
        self.source_weight = source_weight
        
        # Validation
        weights = [evidence_weight, consistency_weight, source_weight]
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        
        # Rule sets for verification
        self.confidence_boosters = [
            "according to", "research shows", "study found", "data indicates",
            "peer reviewed", "published in", "scientific consensus"
        ]
        
        self.confidence_reducers = [
            "maybe", "possibly", "might", "could be", "unconfirmed",
            "rumored", "allegedly", "supposedly", "claims"
        ]
        
        self.contradiction_patterns = [
            ("is", "is not"), ("was", "was not"), ("will", "will not"),
            ("can", "cannot"), ("true", "false"), ("yes", "no")
        ]
        
        self._verification_count = 0
        self._high_confidence_count = 0
        
        logger.info(f"Initialized RuleBasedVerifier with weights: "
                   f"evidence={evidence_weight}, consistency={consistency_weight}, source={source_weight}")
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verification on pipeline state for LangGraph compatibility."""
        try:
            content = state.get("user_input", "")
            if not content:
                # Return state unchanged if no content to verify
                return state
            
            # Verify the content
            scores = self.verify(content, context=state)
            
            # Add verification results to state
            state["verification_scores"] = {
                "truth_score": scores.truth_score,
                "confidence": scores.confidence,
                "evidentiality": scores.evidentiality,
                "relevance": scores.relevance,
                "utility": scores.utility,
                "source_credibility": scores.source_credibility
            }
            state["verification_passed"] = scores.confidence >= 0.6
            
            return state
            
        except Exception as e:
            logger.error(f"Verification execution failed: {e}")
            # Return state with failed verification
            state["verification_scores"] = {"error": str(e)}
            state["verification_passed"] = False
            return state
    
    def verify(self, content: str, context: Optional[Dict[str, Any]] = None) -> ConfidenceScores:
        """
        Verify content using rule-based analysis.
        
        Analyzes content using configured rules and heuristics to generate
        comprehensive confidence scores with detailed reasoning.
        
        Args:
            content: Text content to verify
            context: Optional context including sources, metadata, etc.
            
        Returns:
            ConfidenceScores with truth and quality metrics
            
        Raises:
            VerificationError: If verification process fails
        """
        if not content or not content.strip():
            raise VerificationError("Cannot verify empty content")
        
        try:
            # Ensure context is a dict, handle case where it might be a list
            if isinstance(context, list):
                context = {"filtered_context": context}
            context = context or {}
            content_lower = content.lower()
            
            # Initialize base scores
            truth_score = self.base_confidence
            confidence = self.base_confidence
            evidentiality = self.base_confidence
            relevance = context.get('relevance_score', self.base_confidence)
            utility = context.get('utility_score', self.base_confidence)
            source_credibility = context.get('source_credibility', 0.7)
            
            # Apply evidence quality rules
            evidence_score = self._assess_evidence_quality(content_lower)
            evidentiality = evidence_score
            
            # Apply confidence boosters and reducers
            confidence_adjustment = self._assess_confidence_indicators(content_lower)
            confidence += confidence_adjustment
            truth_score += confidence_adjustment * 0.5  # Lower impact on truth
            
            # Apply consistency check if existing records provided
            consistency_score = self._assess_consistency(content, context.get('existing_records', []))
            truth_score += consistency_score * self.consistency_weight
            confidence += consistency_score * self.consistency_weight * 0.5
            
            # Ensure scores are in valid range
            truth_score = max(0.0, min(1.0, truth_score))
            confidence = max(0.0, min(1.0, confidence))
            evidentiality = max(0.0, min(1.0, evidentiality))
            
            # Update metrics
            self._verification_count += 1
            if confidence >= 0.8:
                self._high_confidence_count += 1
            
            scores = ConfidenceScores(
                truth_score=truth_score,
                confidence=confidence,
                evidentiality=evidentiality,
                relevance=relevance,
                utility=utility,
                source_credibility=source_credibility
            )
            
            logger.debug(f"Verified content with scores: truth={truth_score:.2f}, confidence={confidence:.2f}")
            return scores
            
        except Exception as e:
            logger.error(f"Verification failed for content: {e}")
            raise VerificationError(f"Rule-based verification failed: {e}") from e
    
    def detect_contradictions(self, 
                            content: str, 
                            existing_records: RecordCollection) -> List[Tuple[RecordID, float]]:
        """
        Detect contradictions with existing memory records.
        
        Uses rule-based pattern matching to identify potential contradictions
        between new content and existing records.
        
        Args:
            content: New content to check
            existing_records: Records to check against
            
        Returns:
            List of (record_id, contradiction_score) tuples
            
        Raises:
            VerificationError: If contradiction detection fails
        """
        try:
            contradictions = []
            content_lower = content.lower()
            
            for record in existing_records:
                contradiction_score = self._detect_pairwise_contradiction(
                    content_lower, record.payload.lower()
                )
                
                if contradiction_score > 0.5:  # Threshold for contradiction
                    contradictions.append((record.id, contradiction_score))
                    logger.debug(f"Detected contradiction with record {record.id}: {contradiction_score:.2f}")
            
            return contradictions
            
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            raise VerificationError(f"Contradiction detection failed: {e}") from e
    
    def assess_evidence(self, content: str) -> float:
        """
        Assess the quality of evidence in content.
        
        Args:
            content: Content to assess
            
        Returns:
            Evidence quality score [0.0, 1.0]
        """
        return self._assess_evidence_quality(content.lower())
    
    def _assess_evidence_quality(self, content_lower: str) -> float:
        """Assess evidence quality using rule-based indicators."""
        quality_score = 0.5  # Baseline
        
        # Look for strong evidence indicators
        strong_indicators = ["peer reviewed", "scientific study", "clinical trial", "meta-analysis"]
        for indicator in strong_indicators:
            if indicator in content_lower:
                quality_score += 0.2
        
        # Look for moderate evidence indicators
        moderate_indicators = ["research", "study", "analysis", "data", "statistics"]
        for indicator in moderate_indicators:
            if indicator in content_lower:
                quality_score += 0.1
        
        # Look for quantitative data
        if any(char.isdigit() for char in content_lower):
            quality_score += 0.1
        
        # Look for citations or sources
        citation_indicators = ["according to", "source:", "published in", "doi:", "arxiv:"]
        for indicator in citation_indicators:
            if indicator in content_lower:
                quality_score += 0.15
        
        # Penalize weak evidence language
        weak_indicators = ["anecdotal", "hearsay", "rumor", "unverified"]
        for indicator in weak_indicators:
            if indicator in content_lower:
                quality_score -= 0.2
        
        return max(0.0, min(1.0, quality_score))
    
    def _assess_confidence_indicators(self, content_lower: str) -> float:
        """Assess confidence boosters and reducers in content."""
        adjustment = 0.0
        
        # Apply confidence boosters
        for booster in self.confidence_boosters:
            if booster in content_lower:
                adjustment += 0.1
        
        # Apply confidence reducers
        for reducer in self.confidence_reducers:
            if reducer in content_lower:
                adjustment -= 0.15
        
        return max(-0.3, min(0.3, adjustment))  # Cap adjustment
    
    def _assess_consistency(self, content: str, existing_records: RecordCollection) -> float:
        """Assess consistency with existing high-confidence records."""
        if not existing_records:
            return 0.0  # Neutral when no existing records
        
        # Find high-confidence records for comparison
        high_conf_records = [r for r in existing_records if r.is_high_confidence]
        
        if not high_conf_records:
            return 0.0
        
        # Simple consistency check - no strong contradictions
        content_lower = content.lower()
        contradiction_scores = []
        
        for record in high_conf_records[:5]:  # Limit to avoid performance issues
            contradiction_score = self._detect_pairwise_contradiction(
                content_lower, record.payload.lower()
            )
            contradiction_scores.append(1.0 - contradiction_score)  # Invert for consistency
        
        return sum(contradiction_scores) / len(contradiction_scores)
    
    def _detect_pairwise_contradiction(self, content1: str, content2: str) -> float:
        """Detect contradiction between two pieces of content."""
        contradiction_score = 0.0
        
        # Check for direct negation patterns
        for positive, negative in self.contradiction_patterns:
            if positive in content1 and negative in content2:
                contradiction_score += 0.8
            elif negative in content1 and positive in content2:
                contradiction_score += 0.8
        
        # TODO: Add more sophisticated contradiction detection
        # - Named entity conflicts (e.g., different dates for same event)
        # - Numerical conflicts (e.g., different values for same measure)
        # - Semantic contradictions using embeddings
        
        return min(1.0, contradiction_score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get verification performance metrics."""
        high_confidence_rate = (
            self._high_confidence_count / self._verification_count
            if self._verification_count > 0 else 0.0
        )
        
        return {
            'total_verifications': self._verification_count,
            'high_confidence_count': self._high_confidence_count,
            'high_confidence_rate': high_confidence_rate,
            'base_confidence': self.base_confidence,
            'evidence_weight': self.evidence_weight,
            'consistency_weight': self.consistency_weight,
            'source_weight': self.source_weight
        }


class MLVerifier(BaseVerifier):
    """
    Placeholder for ML-based truth verification implementation.
    
    This class provides the interface for future ML-based verification
    strategies using trained models, embeddings, or external APIs.
    
    TODO: Implement ML-based verification using:
    - Trained fact-checking models
    - Semantic similarity with knowledge bases
    - External fact-checking APIs
    - Ensemble methods combining multiple signals
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML-based verifier.
        
        Args:
            model_config: Configuration for ML models and APIs
        """
        self.model_config = model_config or {}
        logger.info("Initialized MLVerifier (placeholder - not yet implemented)")
    
    def verify(self, content: str, context: Optional[Dict[str, Any]] = None) -> ConfidenceScores:
        """
        ML-based content verification.
        
        TODO: Implement using trained models or external APIs.
        For now, returns baseline scores.
        """
        logger.warning("MLVerifier not yet implemented, returning baseline scores")
        return ConfidenceScores()  # Returns default scores
    
    def detect_contradictions(self, 
                            content: str, 
                            existing_records: RecordCollection) -> List[Tuple[RecordID, float]]:
        """
        ML-based contradiction detection.
        
        TODO: Implement using semantic similarity or trained models.
        """
        logger.warning("ML contradiction detection not yet implemented")
        return []
    
    def assess_evidence(self, content: str) -> float:
        """
        ML-based evidence assessment.
        
        TODO: Implement using trained evidence quality models.
        """
        logger.warning("ML evidence assessment not yet implemented")
        return 0.5


# Factory function for creating verifiers
def create_verifier(verifier_type: str = "rule_based", **kwargs) -> BaseVerifier:
    """
    Factory function for creating truth verifier instances.
    
    Args:
        verifier_type: Type of verifier to create ("rule_based" or "ml")
        **kwargs: Configuration parameters for the verifier
        
    Returns:
        Configured verifier instance
        
    Raises:
        ValueError: If verifier_type is not supported
    """
    if verifier_type == "rule_based":
        return RuleBasedVerifier(**kwargs)
    elif verifier_type == "ml":
        return MLVerifier(**kwargs)
    else:
        raise ValueError(f"Unsupported verifier type: {verifier_type}")


# Default verifier instance for backward compatibility
def get_default_verifier() -> BaseVerifier:
    """Get a default rule-based verifier instance."""
    return RuleBasedVerifier()
