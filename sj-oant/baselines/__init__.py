"""
baselines package - Multi-Turn Baseline System Implementations

This package provides baseline systems for comparison with the TMM system
on multi-turn dialogue tasks using the MultiWOZ dataset.
"""

from .multiturn_systems import (
    BaseMultiTurnSystem,
    DirectLLMBaseline,
    LongContextBaseline,
    SimpleStateTrackerBaseline,
    NaiveMemoryBaseline,
    create_multiturn_baselines
)

__all__ = [
    "BaseMultiTurnSystem",
    "DirectLLMBaseline",
    "LongContextBaseline",
    "SimpleStateTrackerBaseline",
    "NaiveMemoryBaseline",
    "create_multiturn_baselines"
]
