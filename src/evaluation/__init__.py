# -*- coding: utf-8 -*-
"""
Evaluation framework for German Word Sense Disambiguation.

This package provides tools for evaluating WSD systems against gold standard
annotations, including metrics calculation and error analysis.
"""

from .wsd_evaluator import WSDEvaluator
from .metrics import WSDMetrics
from .error_analysis import ErrorAnalyzer

__all__ = [
    'WSDEvaluator',
    'WSDMetrics', 
    'ErrorAnalyzer'
]