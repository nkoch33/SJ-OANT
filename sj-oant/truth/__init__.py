"""
TMM Truth Package

This package implements truth verification and content filtering
for the Truth-Maintained Memory system.

Key Components:
- verifier: Rule-based and ML truth verification
- tacs_filter: Token-level adaptive context screening (TODO)

Example Usage:
    from truth import RuleBasedVerifier, create_verifier
    
    # Create a verifier
    verifier = create_verifier("rule_based", base_confidence=0.6)
    
    # Verify content
    scores = verifier.verify("The sky is blue.")
    print(f"Truth score: {scores.truth_score:.2f}")
"""

from truth.verifier import (
    RuleBasedVerifier,
    MLVerifier,
    create_verifier,
    get_default_verifier
)

# TODO: Add tacs_filter exports when implemented
# from truth.tacs_filter import TACSFilter

__version__ = "0.1.0"

__all__ = [
    # Verifier implementations
    "RuleBasedVerifier",
    "MLVerifier",
    
    # Factory functions
    "create_verifier",
    "get_default_verifier"
]
