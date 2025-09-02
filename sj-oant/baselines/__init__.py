"""
baselines package - Baseline System Implementations

This package provides baseline systems for comparison with the TMM system,
including DirectLLM, LongContext, SimpleRAG, and BasicMemory approaches.
"""

from .simple_systems import (
    BaselineSystem, 
    DirectLLMBaseline, 
    LongContextBaseline, 
    SimpleRAGBaseline, 
    BasicMemoryBaseline,
    create_baseline_systems
)

__all__ = [
    "BaselineSystem",
    "DirectLLMBaseline", 
    "LongContextBaseline", 
    "SimpleRAGBaseline", 
    "BasicMemoryBaseline",
    "create_baseline_systems"
]
