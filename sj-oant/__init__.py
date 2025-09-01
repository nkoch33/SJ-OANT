"""
SJ-OANT: Truth-Maintained Memory for LLM Agents

A research implementation of proactive, multi-agent memory systems that prevent
false memory formation in large language model agents through truth-maintained
filtering, verification, and curation.

Key Components:
- Multi-agent pipeline with truth filtering and verification
- Multi-tiered memory store (L1/L2/L3/Flagged)
- FABLE benchmark for false-memory-aware evaluation
- Comprehensive evaluation suite with novel metrics (FMR, MEL, DAR)

This package implements the complete TMM architecture as described in the
research methodology, providing both the system implementation and evaluation
framework for advancing trustworthy long-term LLM agent memory.
"""

__version__ = "0.1.0"
__author__ = "SJ-OANT Research Team"

# TODO: Import main components when implemented
# from .agents import Planner, Arbiter, Responder, WriterEditor
# from .truth import TACSFilter, Verifier  
# from .memory import TypedStore, MemoryPolicies, VotingSystem
# from .retrieval import HybridRetriever, ActiveRetriever
# from .benchmarks.fable import FABLEBenchmark
# from .runners import eval_fable, eval_baselines, run_ablations
