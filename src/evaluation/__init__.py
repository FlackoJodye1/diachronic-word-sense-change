# -*- coding: utf-8 -*-
"""
Evaluation framework for German Word Sense Disambiguation.

This package provides tools for evaluating WSD systems against gold standard
annotations, including metrics calculation, error analysis, and embedding-based
similarity evaluation.
"""

from .wsd_evaluator import WSDEvaluator
from .metrics import WSDMetrics
from .error_analysis import ErrorAnalyzer
from .embedder import Embedder, get_available_models, RECOMMENDED_MODELS
from .similarity_wsd import SimilarityWSD, WSDResult, WSDPrediction

__all__ = [
    "WSDEvaluator",
    "WSDMetrics",
    "ErrorAnalyzer",
    "Embedder",
    "get_available_models",
    "RECOMMENDED_MODELS",
    "SimilarityWSD",
    "WSDResult",
    "WSDPrediction",
]