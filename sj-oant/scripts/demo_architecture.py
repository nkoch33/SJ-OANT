#!/usr/bin/env python3
"""
TMM Architecture Demonstration

This script demonstrates the clean, modular architecture of the TMM system
with proper dependency injection, comprehensive error handling, and
production-quality logging.

Usage:
    python demo_architecture.py
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Demonstrate the TMM architecture with clean interfaces
    and dependency injection patterns.
    """
    logger.info("🚀 Starting TMM Architecture Demonstration")
    
    try:
        # Import core components
        from core.types import (
            MemoryRecord, MemoryTier, MemoryStatus, ContentType,
            ConfidenceScores, Provenance
        )
        from memory import InMemoryStore, PolicyEngine, PolicyConfig
        from truth import create_verifier
        
        logger.info("✅ Successfully imported all TMM components")
        
        # 1. Initialize the memory store with dependency injection
        logger.info("📦 Initializing memory store...")
        memory_store = InMemoryStore(
            l1_limit=50,
            l2_limit=200, 
            l3_limit=500,
            flagged_limit=100
        )
        
        # 2. Initialize the policy engine with configuration
        logger.info("⚖️ Initializing policy engine...")
        policy_config = PolicyConfig(
            selective_add_trust_threshold=0.7,
            selective_add_utility_threshold=0.6,
            combined_delete_utility_threshold=0.3,
            combined_delete_max_age_days=30
        )
        policy_engine = PolicyEngine(policy_config)
        
        # 3. Initialize the truth verifier
        logger.info("🔍 Initializing truth verifier...")
        verifier = create_verifier(
            "rule_based",
            base_confidence=0.6,
            evidence_weight=0.3,
            consistency_weight=0.4,
            source_weight=0.3
        )
        
        # 4. Create some sample memory records
        logger.info("📝 Creating sample memory records...")
        
        # High-quality factual record
        high_quality_record = MemoryRecord(
            payload="According to peer-reviewed research published in Nature, photosynthesis converts CO2 to oxygen.",
            content_type=ContentType.FACT,
            tier=MemoryTier.L1_WORKING,
            status=MemoryStatus.PENDING,
            scores=ConfidenceScores(
                truth_score=0.9,
                confidence=0.95,
                evidentiality=0.9,
                relevance=0.8,
                utility=0.85,
                source_credibility=0.95
            ),
            provenance=Provenance(
                source="scientific_paper",
                source_type="peer_reviewed",
                pipeline_stage="user_input",
                processing_agent="demo_script",
                session_id="demo_session_001"
            ),
            tags={"science", "biology", "photosynthesis"}
        )
        
        # Low-quality speculative record
        low_quality_record = MemoryRecord(
            payload="Someone mentioned that maybe aliens might have visited Earth, but it's unconfirmed.",
            content_type=ContentType.USER_INPUT,
            tier=MemoryTier.L1_WORKING,
            status=MemoryStatus.PENDING,
            scores=ConfidenceScores(
                truth_score=0.2,
                confidence=0.3,
                evidentiality=0.1,
                relevance=0.4,
                utility=0.2,
                source_credibility=0.3
            ),
            provenance=Provenance(
                source="casual_conversation",
                source_type="hearsay",
                pipeline_stage="user_input",
                processing_agent="demo_script",
                session_id="demo_session_001"
            ),
            tags={"speculation", "aliens"}
        )
        
        # 5. Demonstrate policy-based decision making
        logger.info("🧠 Demonstrating policy-based decision making...")
        
        # Evaluate addition decisions
        high_quality_decision = policy_engine.evaluate_addition(high_quality_record)
        low_quality_decision = policy_engine.evaluate_addition(low_quality_record)
        
        logger.info(f"High-quality record decision: {high_quality_decision.decision} - {high_quality_decision.reasoning}")
        logger.info(f"Low-quality record decision: {low_quality_decision.decision} - {low_quality_decision.reasoning}")
        
        # 6. Add approved records to memory store
        logger.info("💾 Adding approved records to memory store...")
        
        if high_quality_decision.decision:
            record_id = memory_store.add(high_quality_record)
            logger.info(f"✅ Added high-quality record: {record_id}")
        
        if low_quality_decision.decision:
            record_id = memory_store.add(low_quality_record)
            logger.info(f"✅ Added low-quality record: {record_id}")
        else:
            logger.info("❌ Rejected low-quality record based on policy")
        
        # 7. Demonstrate truth verification
        logger.info("🔬 Demonstrating truth verification...")
        
        test_content = "Research from MIT shows that artificial intelligence can improve memory systems."
        verification_scores = verifier.verify(test_content)
        
        logger.info(f"Verification results for test content:")
        logger.info(f"  Truth score: {verification_scores.truth_score:.2f}")
        logger.info(f"  Confidence: {verification_scores.confidence:.2f}")
        logger.info(f"  Evidence quality: {verification_scores.evidentiality:.2f}")
        
        # 8. Demonstrate memory search and retrieval
        logger.info("🔍 Demonstrating memory search...")
        
        # Search by content
        search_results = memory_store.search("photosynthesis", limit=5)
        logger.info(f"Found {len(search_results)} records matching 'photosynthesis'")
        
        # Search by tier
        l1_records = memory_store.get_by_tier(MemoryTier.L1_WORKING, limit=10)
        logger.info(f"Found {len(l1_records)} records in L1 tier")
        
        # Search with filters
        filtered_results = memory_store.search(
            "", 
            filters={
                'min_confidence': 0.8,
                'tags': ['science']
            },
            limit=5
        )
        logger.info(f"Found {len(filtered_results)} high-confidence science records")
        
        # 9. Demonstrate metrics collection
        logger.info("📊 Collecting system metrics...")
        
        memory_metrics = memory_store.get_metrics()
        policy_metrics = policy_engine.get_all_metrics()
        verifier_metrics = verifier.get_metrics()
        
        logger.info("Memory Store Metrics:")
        for key, value in memory_metrics.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Policy Engine Metrics:")
        for policy_name, metrics in policy_metrics.items():
            if isinstance(metrics, dict):
                logger.info(f"  {policy_name}:")
                for key, value in metrics.items():
                    logger.info(f"    {key}: {value}")
        
        logger.info("Verifier Metrics:")
        for key, value in verifier_metrics.items():
            logger.info(f"  {key}: {value}")
        
        # 10. Demonstrate error handling
        logger.info("⚠️ Demonstrating error handling...")
        
        try:
            # Try to add empty record (should fail)
            invalid_record = MemoryRecord(payload="")
            memory_store.add(invalid_record)
        except ValueError as e:
            logger.info(f"✅ Correctly caught validation error: {e}")
        
        try:
            # Try to verify empty content (should fail)
            verifier.verify("")
        except Exception as e:
            logger.info(f"✅ Correctly caught verification error: {e}")
        
        logger.info("🎉 TMM Architecture demonstration completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("🏆 TMM ARCHITECTURE DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"📦 Memory Records Stored: {memory_metrics.get('total_records', 0)}")
        print(f"⚖️ Policy Decisions Made: {policy_metrics['selective_addition']['total_decisions']}")
        print(f"🔬 Verifications Performed: {verifier_metrics['total_verifications']}")
        print(f"✅ System Status: OPERATIONAL")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
