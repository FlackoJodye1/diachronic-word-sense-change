"""Temporal sense change metrics for diachronic analysis."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def compute_entropy_per_period(
    df: pd.DataFrame,
    period_column: str = "decade",
) -> pd.DataFrame:
    """Compute entropy of sense distribution per lemma per period.

    Higher entropy = more uniform distribution across senses (more ambiguous)
    Lower entropy = dominated by one sense (less ambiguous)
    """
    logger.info(f"\nComputing entropy per {period_column}...")

    results = []
    for (lemma, period), group in df.groupby(["lemma", period_column]):
        proportions = group["proportion"].values
        # Entropy in bits (log base 2)
        h = entropy(proportions, base=2)
        n_senses = len(proportions)
        max_entropy = np.log2(n_senses) if n_senses > 1 else 0
        normalized_entropy = h / max_entropy if max_entropy > 0 else 0

        results.append({
            "lemma": lemma,
            period_column: period,
            "entropy": h,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized_entropy,
            "n_senses": n_senses,
        })

    result_df = pd.DataFrame(results)
    logger.info(f"  Computed entropy for {len(result_df)} lemma-period pairs")
    return result_df


def compute_period_divergence(
    df: pd.DataFrame,
    period_column: str = "decade",
) -> pd.DataFrame:
    """Compute Jensen-Shannon divergence between adjacent periods.

    JSD measures how different two probability distributions are.
    JSD = 0 means identical distributions
    JSD = 1 (max for log base 2) means completely different
    """
    logger.info(f"\nComputing divergence between adjacent {period_column}s...")

    results = []
    for lemma, group in df.groupby("lemma"):
        periods = sorted(group[period_column].unique())

        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i + 1]

            # Get distributions for each period
            dist1 = group[group[period_column] == p1].set_index("predicted_sense_id")["proportion"]
            dist2 = group[group[period_column] == p2].set_index("predicted_sense_id")["proportion"]

            # Align distributions (fill missing senses with 0)
            all_senses = set(dist1.index) | set(dist2.index)
            dist1 = dist1.reindex(all_senses, fill_value=0)
            dist2 = dist2.reindex(all_senses, fill_value=0)

            # Compute JSD
            jsd = jensenshannon(dist1.values, dist2.values, base=2)

            results.append({
                "lemma": lemma,
                f"{period_column}_from": p1,
                f"{period_column}_to": p2,
                "jsd": jsd,
                "n_senses_from": (dist1 > 0).sum(),
                "n_senses_to": (dist2 > 0).sum(),
            })

    result_df = pd.DataFrame(results)
    logger.info(f"  Computed {len(result_df)} period transitions")
    return result_df


def detect_sense_changes(
    df: pd.DataFrame,
    period_column: str = "decade",
    emergence_threshold: float = 0.1,
    decline_threshold: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect sense emergence and decline events.

    A sense emerges when it appears with proportion >= threshold in period N
    but was absent or below threshold in period N-1.

    A sense declines when it drops below threshold in period N
    but was above threshold in period N-1.
    """
    logger.info(f"\nDetecting sense changes (threshold={emergence_threshold})...")

    emergences = []
    declines = []

    for lemma, group in df.groupby("lemma"):
        periods = sorted(group[period_column].unique())

        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i + 1]

            # Get proportions for each period
            props1 = group[group[period_column] == p1].set_index("predicted_sense_id")["proportion"]
            props2 = group[group[period_column] == p2].set_index("predicted_sense_id")["proportion"]

            all_senses = set(props1.index) | set(props2.index)

            for sense in all_senses:
                prop1 = props1.get(sense, 0)
                prop2 = props2.get(sense, 0)

                # Emergence: was below threshold, now above
                if prop1 < emergence_threshold <= prop2:
                    emergences.append({
                        "lemma": lemma,
                        "sense_id": sense,
                        f"{period_column}_from": p1,
                        f"{period_column}_to": p2,
                        "proportion_from": prop1,
                        "proportion_to": prop2,
                        "change": prop2 - prop1,
                    })

                # Decline: was above threshold, now below
                if prop1 >= decline_threshold > prop2:
                    declines.append({
                        "lemma": lemma,
                        "sense_id": sense,
                        f"{period_column}_from": p1,
                        f"{period_column}_to": p2,
                        "proportion_from": prop1,
                        "proportion_to": prop2,
                        "change": prop2 - prop1,
                    })

    emergence_df = pd.DataFrame(emergences)
    decline_df = pd.DataFrame(declines)

    logger.info(f"  Detected {len(emergence_df)} emergence events")
    logger.info(f"  Detected {len(decline_df)} decline events")

    return emergence_df, decline_df
