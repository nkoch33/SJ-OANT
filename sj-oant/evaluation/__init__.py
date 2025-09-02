"""
evaluation package - SQuAD Evaluation Pipeline

This package provides evaluation capabilities for the TMM system using
the SQuAD dataset and baseline comparison systems.
"""

from .squad_eval import SQuADEvaluator, EvaluationResult, SQuADExample

__all__ = ["SQuADEvaluator", "EvaluationResult", "SQuADExample"]