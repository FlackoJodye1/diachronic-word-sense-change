"""Diachronic analysis modules for temporal word sense disambiguation."""

from .io import (
    load_sense_counts,
    load_predictions,
    load_dwds_contexts,
    load_senses,
)
from .metrics import (
    compute_entropy_per_period,
    compute_period_divergence,
    detect_sense_changes,
)
from .aggregation import (
    aggregate_by_period,
    compute_period_stats,
)
from .predictor import (
    is_e5_model,
    add_prefix,
    predict_senses,
    predict_senses_for_lemma,
)
from .visualization import (
    plot_sense_trajectory,
    plot_entropy_over_time,
    plot_sense_heatmap,
    plot_all_lemmas_summary,
)

__all__ = [
    # I/O
    "load_sense_counts",
    "load_predictions",
    "load_dwds_contexts",
    "load_senses",
    # Metrics
    "compute_entropy_per_period",
    "compute_period_divergence",
    "detect_sense_changes",
    # Aggregation
    "aggregate_by_period",
    "compute_period_stats",
    # Predictor
    "is_e5_model",
    "add_prefix",
    "predict_senses",
    "predict_senses_for_lemma",
    # Visualization
    "plot_sense_trajectory",
    "plot_entropy_over_time",
    "plot_sense_heatmap",
    "plot_all_lemmas_summary",
]