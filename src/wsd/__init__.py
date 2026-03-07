"""
Evaluation framework for German Word Sense Disambiguation.

This package provides tools for evaluating WSD systems against gold standard
annotations, including metrics calculation, error analysis, and embedding-based
similarity evaluation.
"""

from .embedder import Embedder, get_available_models, RECOMMENDED_MODELS
from .similarity_wsd import SimilarityWSD, WSDResult, WSDPrediction
from .wsd_evaluator import WSDEvaluator, WSDMetrics, ErrorAnalyzer

__all__ = [
    "Embedder",
    "get_available_models",
    "RECOMMENDED_MODELS",
    "SimilarityWSD",
    "WSDResult",
    "WSDPrediction",
    "WSDEvaluator",
    "WSDMetrics",
    "ErrorAnalyzer",
]