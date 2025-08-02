"""
Evaluation package for RAG system.
Provides both quantitative (metrics) and qualitative (error analysis) tools.
"""

from .metrics import (
    compute_accuracy,
    compute_precision_recall_f1,
    compute_bleu,
    compute_rouge,
    compute_meteor,
)

from .qualitative import analyze_errors, sample_failure_cases
from .evaluator import RAGEvaluator
from .visualizer import EvaluationVisualizer

__all__ = [
    "compute_accuracy",
    "compute_precision_recall_f1",
    "compute_bleu",
    "compute_rouge",
    "compute_meteor",
    "analyze_errors",
    "sample_failure_cases",
    "RAGEvaluator",
    "EvaluationVisualizer",
]
