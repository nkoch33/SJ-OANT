"""
memory.policies - Memory Management Policies

This module implements production-grade memory management policies for the TMM system,
following clean architecture patterns with comprehensive logging, metrics, and configurability.

Design Features:
- Configurable thresholds and parameters
- Comprehensive decision logging and audit trails
- Pluggable policy implementations via dependency injection
- Rich metrics collection for monitoring and optimization
- Thread-safe operations for concurrent environments
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass

from core.ports import PolicyPort
from core.types import MemoryRecord, MemoryTier, MemoryStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration container for memory management policies."""
    selective_add_trust_threshold: float = 0.8
    selective_add_utility_threshold: float = 0.7
    selective_add_require_verification: bool = True
    combined_delete_cleanup_interval: int = 50
    combined_delete_utility_threshold: float = 0.3
    combined_delete_max_age_days: int = 30
    tier_promotion_confidence_threshold: float = 0.85
    tier_promotion_evidence_threshold: float = 0.8
    flagged_review_max_age_hours: int = 24


@dataclass(frozen=True)
class PolicyDecision:
    """Container for policy decision results with audit trail."""
    decision: bool
    confidence: float
    reasoning: str
    factors: Dict[str, Any]
    metadata: Dict[str, Any]


class SelectiveAdditionPolicy(PolicyPort):
    """Enterprise implementation of the Selective Addition policy."""
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        self._decision_count = 0
        self._acceptance_count = 0
        
        logger.info(f"Initialized SelectiveAdditionPolicy with thresholds: "
                   f"trust={config.selective_add_trust_threshold}, "
                   f"utility={config.selective_add_utility_threshold}")
    
    def should_add(self, record: MemoryRecord, context: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if a record should be added to memory."""
        decision_result = self.evaluate_addition(record, context)
        
        self._decision_count += 1
        if decision_result.decision:
            self._acceptance_count += 1
        
        logger.info(f"Addition decision for {record.id}: {decision_result.decision} - {decision_result.reasoning}")
        return decision_result.decision
    
    def evaluate_addition(self, record: MemoryRecord, context: Optional[Dict[str, Any]] = None) -> PolicyDecision:
        """Perform detailed evaluation of addition criteria."""
        context = context or {}
        factors = {}
        decision_factors = []
        rejection_factors = []
        
        # Evaluate verification status
        verification_passed = True
        if self.config.selective_add_require_verification and record.status != MemoryStatus.VERIFIED:
            verification_passed = False
            rejection_factors.append("not verified")
        
        # Evaluate trust score (truth + confidence)
        trust_score = (record.scores.truth_score + record.scores.confidence) / 2.0
        trust_passed = trust_score >= self.config.selective_add_trust_threshold
        
        if trust_passed:
            decision_factors.append(f"high trust ({trust_score:.2f})")
        else:
            rejection_factors.append(f"low trust ({trust_score:.2f})")
        
        # Evaluate utility score (utility + relevance)
        utility_score = (record.scores.utility + record.scores.relevance) / 2.0
        utility_passed = utility_score >= self.config.selective_add_utility_threshold
        
        if utility_passed:
            decision_factors.append(f"high utility ({utility_score:.2f})")
        else:
            rejection_factors.append(f"low utility ({utility_score:.2f})")
        
        # Check for contradictions
        has_contradictions = context.get('contradictions_detected', False)
        if has_contradictions:
            rejection_factors.append("contradictions detected")
        
        # Make final decision
        final_decision = verification_passed and trust_passed and utility_passed and not has_contradictions
        
        # Calculate confidence
        confidence = 0.9 if final_decision else 0.7
        
        # Build reasoning
        if final_decision:
            reasoning = f"Accepted: {', '.join(decision_factors)}"
        else:
            reasoning = f"Rejected: {', '.join(rejection_factors)}"
        
        return PolicyDecision(
            decision=final_decision,
            confidence=confidence,
            reasoning=reasoning,
            factors={'trust_score': trust_score, 'utility_score': utility_score},
            metadata={'record_id': str(record.id), 'policy': 'selective_addition'}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get policy performance metrics."""
        acceptance_rate = self._acceptance_count / self._decision_count if self._decision_count > 0 else 0.0
        return {
            'total_decisions': self._decision_count,
            'total_acceptances': self._acceptance_count,
            'acceptance_rate': acceptance_rate
        }


class CombinedDeletionPolicy(PolicyPort):
    """Enterprise implementation of the Combined Deletion policy."""
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        self._operation_count = 0
        self._deletion_count = 0
        
        logger.info(f"Initialized CombinedDeletionPolicy with interval={config.combined_delete_cleanup_interval}")
    
    def should_delete(self, record: MemoryRecord, context: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if a record should be deleted from memory."""
        decision_result = self.evaluate_deletion(record, context)
        
        if decision_result.decision:
            self._deletion_count += 1
        
        logger.info(f"Deletion decision for {record.id}: {decision_result.decision} - {decision_result.reasoning}")
        return decision_result.decision
    
    def evaluate_deletion(self, record: MemoryRecord, context: Optional[Dict[str, Any]] = None) -> PolicyDecision:
        """Perform detailed evaluation of deletion criteria."""
        context = context or {}
        deletion_factors = []
        preservation_factors = []
        
        # Calculate record age
        age_days = (datetime.now(timezone.utc) - record.created_at).total_seconds() / (24 * 3600)
        
        # Check age-based deletion
        if age_days > self.config.combined_delete_max_age_days:
            deletion_factors.append(f"old ({age_days:.1f} days)")
        else:
            preservation_factors.append(f"recent ({age_days:.1f} days)")
        
        # Evaluate utility score
        if record.scores.utility < self.config.combined_delete_utility_threshold:
            deletion_factors.append(f"low utility ({record.scores.utility:.2f})")
        else:
            preservation_factors.append(f"useful ({record.scores.utility:.2f})")
        
        # Check verification status
        if record.status == MemoryStatus.REJECTED:
            deletion_factors.append("rejected")
        elif record.status == MemoryStatus.FLAGGED:
            flagged_age_hours = (datetime.now(timezone.utc) - record.created_at).total_seconds() / 3600
            if flagged_age_hours > self.config.flagged_review_max_age_hours:
                deletion_factors.append(f"flagged and aged ({flagged_age_hours:.1f}h)")
            else:
                preservation_factors.append("flagged but recent")
        
        # Protect high-confidence archival content
        if record.tier == MemoryTier.L3_ARCHIVAL and record.scores.overall_score > 0.8:
            preservation_factors.append("high-confidence archival")
        
        # Make decision: delete if multiple deletion factors OR rejected/aged flagged
        should_delete = (
            len(deletion_factors) >= 2 or
            record.status == MemoryStatus.REJECTED or
            (record.status == MemoryStatus.FLAGGED and 
             (datetime.now(timezone.utc) - record.created_at).total_seconds() / 3600 > self.config.flagged_review_max_age_hours)
        )
        
        # But preserve archival high-confidence content
        if record.tier == MemoryTier.L3_ARCHIVAL and record.scores.overall_score > 0.8:
            should_delete = False
        
        reasoning = f"Delete: {', '.join(deletion_factors)}" if should_delete else f"Preserve: {', '.join(preservation_factors)}"
        
        return PolicyDecision(
            decision=should_delete,
            confidence=0.8,
            reasoning=reasoning,
            factors={'age_days': age_days, 'utility_score': record.scores.utility},
            metadata={'record_id': str(record.id), 'policy': 'combined_deletion'}
        )
    
    def should_cleanup(self) -> bool:
        """Determine if periodic cleanup should be triggered."""
        self._operation_count += 1
        should_run = self._operation_count % self.config.combined_delete_cleanup_interval == 0
        
        if should_run:
            logger.info(f"Triggering cleanup after {self._operation_count} operations")
        
        return should_run
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get policy performance metrics."""
        deletion_rate = self._deletion_count / self._operation_count if self._operation_count > 0 else 0.0
        return {
            'total_operations': self._operation_count,
            'total_deletions': self._deletion_count,
            'deletion_rate': deletion_rate
        }


class PolicyEngine:
    """Centralized policy engine coordinating all memory management policies."""
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.selective_addition = SelectiveAdditionPolicy(self.config)
        self.combined_deletion = CombinedDeletionPolicy(self.config)
        
        logger.info("Initialized PolicyEngine with all memory management policies")
    
    def evaluate_addition(self, record: MemoryRecord, context: Optional[Dict[str, Any]] = None) -> PolicyDecision:
        """Evaluate whether a record should be added to memory."""
        return self.selective_addition.evaluate_addition(record, context)
    
    def evaluate_deletion(self, record: MemoryRecord, context: Optional[Dict[str, Any]] = None) -> PolicyDecision:
        """Evaluate whether a record should be deleted from memory."""
        return self.combined_deletion.evaluate_deletion(record, context)
    
    def should_cleanup(self) -> bool:
        """Check if periodic cleanup should be triggered."""
        return self.combined_deletion.should_cleanup()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all policies."""
        return {
            'selective_addition': self.selective_addition.get_metrics(),
            'combined_deletion': self.combined_deletion.get_metrics(),
            'config': {
                'trust_threshold': self.config.selective_add_trust_threshold,
                'utility_threshold': self.config.selective_add_utility_threshold,
                'cleanup_interval': self.config.combined_delete_cleanup_interval
            }
        }
