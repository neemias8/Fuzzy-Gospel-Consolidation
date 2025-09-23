"""
Evaluation Module

This module implements comprehensive evaluation metrics
for the consolidated Gospel summaries.
"""

from .evaluation_suite import EvaluationSuite
from .metrics import AutomaticMetrics
from .human_evaluation import HumanEvaluationProtocol

__all__ = [
    'EvaluationSuite',
    'AutomaticMetrics',
    'HumanEvaluationProtocol'
]
