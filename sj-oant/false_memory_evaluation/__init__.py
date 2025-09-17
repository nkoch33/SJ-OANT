"""
False Memory Formation Evaluation Framework

This module implements the core research vision metrics for evaluating
Truth-Maintained Memory Agent (TMMA) performance on false memory formation:

- FMR (False Memory Rate): How often the system repeats seeded falsehoods
- MEL (Memory Edit Latency): How quickly it corrects after contradictions  
- DAR (Disturbance Adaptation Rate): How well it handles mixed true/false context
- Contradiction Detection: How well it identifies conflicting information

This augments the existing dialogue evaluation frameworks without replacing them.
"""

from .false_memory_evaluator import FalseMemoryEvaluator
from .test_scenarios import FalseMemoryTestScenarios
from .metrics_calculator import MetricsCalculator
from .contradiction_detector import ContradictionDetector

__all__ = [
    'FalseMemoryEvaluator',
    'FalseMemoryTestScenarios', 
    'MetricsCalculator',
    'ContradictionDetector'
]
