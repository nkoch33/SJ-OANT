"""
evaluation package - FictionalQA Evaluation Pipeline

This package provides evaluation capabilities for the TMM system using
the FictionalQA dataset and baseline comparison systems.
"""

from .fictionalqa_eval import FictionalQAEvaluator, EvaluationResult, FictionalQAExample

__all__ = ["FictionalQAEvaluator", "EvaluationResult", "FictionalQAExample"]
