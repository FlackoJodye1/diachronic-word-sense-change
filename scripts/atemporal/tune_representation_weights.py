#!/usr/bin/env python3
"""
Tune the weights for combining definition and example embeddings.

This script optimizes the α parameter in:
    weighted_emb = α × definition_emb + (1-α) × avg(examples_emb)

Currently α=0.5 achieves 54.8% accuracy. This script explores whether
other values of α perform better.

Usage:
    python scripts/atemporal/tune_representation_weights.py
    python scripts/atemporal/tune_representation_weights.py --alphas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    python scripts/atemporal/tune_representation_weights.py --fine-grid  # 0.3 to 0.7 in steps of 0.05
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.wsd import Embedder, SimilarityWSD, WSDResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
ATEMPORAL_DIR = DATA_DIR / "evaluation" / "atemporal"
SENSE_FILE = ATEMPORAL_DIR / "input" / "sense_text_representations.jsonl"
GOLD_FILE = ATEMPORAL_DIR / "input" / "gold_annotations.csv"
OUTPUT_FILE = ATEMPORAL_DIR / "results" / "weight_tuning.csv"

# Best model from previous experiments
MODEL_NAME = "intfloat/multilingual-e5-large"

# Columns for weighted combination
DEF_COL = "llm_definition"
EXAMPLE_COLS = [f"llm_example_{i}" for i in range(5)]


@dataclass
class WeightResult:
    """Results for a single alpha value."""
    alpha: float
    accuracy: float
    top3: float
    top5: float
    mrr: float
    evaluated: int
    skipped: int


def is_e5_model(model_name: str) -> bool:
    """Check if model is an E5 model requiring query/passage prefixes."""
    return "e5" in model_name.lower()


def add_e5_prefix(texts: list[str], prefix: str) -> list[str]:
    """Add E5 prefix to texts."""
    return [f"{prefix}{t}" for t in texts]


def load_sense_representations(path: Path) -> pd.DataFrame:
    """Load sense representations from JSONL."""
    df = pd.read_json(path, lines=True)

    # Expand llm_examples list into separate columns if needed
    if "llm_examples" in df.columns:
        examples_expanded = df["llm_examples"].apply(pd.Series)
        examples_expanded.columns = [f"llm_example_{i}" for i in range(len(examples_expanded.columns))]
        df = pd.concat([df.drop(columns=["llm_examples"]), examples_expanded], axis=1)

    return df


def compute_weighted_embeddings(
    def_embeddings: np.ndarray,
    example_embeddings_list: list[np.ndarray],
    alpha: float,
    sense_df: pd.DataFrame
) -> np.ndarray:
    """Compute weighted combination: α×def + (1-α)×avg(examples).

    Args:
        def_embeddings: Definition embeddings (n_senses, dim)
        example_embeddings_list: List of 5 example embedding arrays
        alpha: Weight for definition (0.0 to 1.0)
        sense_df: Sense dataframe (for checking which examples exist)

    Returns:
        Weighted embeddings (n_senses, dim)
    """
    # Stack examples: (5, n_senses, dim)
    examples_stacked = np.stack(example_embeddings_list, axis=0)

    # Create mask for non-empty examples
    masks = []
    for ex_col in EXAMPLE_COLS:
        raw_texts = sense_df[ex_col].fillna("").tolist()
        is_nonempty = [bool(t.strip()) for t in raw_texts]
        masks.append(is_nonempty)

    masks_array = np.array(masks, dtype=np.float32)  # (5, n_senses)
    masks_expanded = masks_array[:, :, np.newaxis]  # (5, n_senses, 1)

    # Apply mask and compute mean over available examples
    masked = examples_stacked * masks_expanded
    sum_examples = np.sum(masked, axis=0)  # (n_senses, dim)
    counts = np.sum(masks_array, axis=0, keepdims=True).T  # (n_senses, 1)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    mean_examples = sum_examples / counts

    # Weighted combination
    weighted = alpha * def_embeddings + (1 - alpha) * mean_examples

    # Re-normalize
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    weighted = weighted / np.where(norms == 0, 1, norms)

    return weighted


def tune_weights(
    alphas: list[float],
    sense_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    model_name: str = MODEL_NAME
) -> list[WeightResult]:
    """Run weight tuning experiment.

    Args:
        alphas: List of alpha values to try
        sense_df: Sense representations
        gold_df: Gold annotations
        model_name: Embedding model to use

    Returns:
        List of WeightResult for each alpha
    """
    logger.info(f"Loading model: {model_name}")
    embedder = Embedder(model_name=model_name, normalize=True)

    use_e5_prefix = is_e5_model(model_name)
    if use_e5_prefix:
        logger.info("  (E5 model: applying query/passage prefixes)")

    logger.info("Encoding context sentences...")
    raw_contexts = gold_df["sentence"].fillna("").tolist()
    contexts = add_e5_prefix(raw_contexts, "query: ") if use_e5_prefix else raw_contexts
    context_embeddings = embedder.encode(contexts, show_progress=True)

    context_lemmas = gold_df["lemma"].tolist()
    gold_sense_ids = gold_df["gold_sense_id"].tolist()

    logger.info("Encoding definitions...")
    raw_defs = sense_df[DEF_COL].fillna("").tolist()
    defs = add_e5_prefix(raw_defs, "passage: ") if use_e5_prefix else raw_defs
    def_embeddings = embedder.encode(defs, show_progress=True)

    logger.info("Encoding examples...")
    example_embeddings_list = []
    for i, ex_col in enumerate(EXAMPLE_COLS):
        logger.info(f"  Example {i+1}/5...")
        raw_exs = sense_df[ex_col].fillna("").tolist()
        exs = add_e5_prefix(raw_exs, "passage: ") if use_e5_prefix else raw_exs
        ex_embs = embedder.encode(exs, show_progress=False)
        example_embeddings_list.append(ex_embs)

    results = []
    logger.info(f"Trying {len(alphas)} alpha values...")
    logger.info("=" * 60)

    for alpha in alphas:
        label = ""
        if alpha == 1.0:
            label = "(pure definition)"
        elif alpha == 0.0:
            label = "(pure examples)"
        elif alpha == 0.5:
            label = "(equal weight - current baseline)"
        logger.info(f"a = {alpha:.2f}  {label}")

        # Compute weighted embeddings
        weighted_embs = compute_weighted_embeddings(
            def_embeddings,
            example_embeddings_list,
            alpha,
            sense_df
        )

        # Evaluate
        wsd = SimilarityWSD.from_dataframe(
            sense_df=sense_df,
            embeddings=weighted_embs,
            sense_id_col="synset_id",
            lemma_col="lemma"
        )

        wsd_result: WSDResult = wsd.evaluate(
            context_embeddings=context_embeddings,
            lemmas=context_lemmas,
            gold_sense_ids=gold_sense_ids
        )

        results.append(WeightResult(
            alpha=alpha,
            accuracy=wsd_result.accuracy,
            top3=wsd_result.top3_accuracy,
            top5=wsd_result.top5_accuracy,
            mrr=wsd_result.mrr,
            evaluated=wsd_result.evaluated,
            skipped=wsd_result.skipped
        ))

        acc_str = f"{wsd_result.accuracy:.1%}"
        improvement = "  <- current" if alpha == 0.5 else ""
        logger.info(f"  Accuracy: {acc_str:>6s}  |  Top-3: {wsd_result.top3_accuracy:.1%}  |  MRR: {wsd_result.mrr:.3f}{improvement}")

    return results


def log_summary(results_df: pd.DataFrame) -> None:
    """Log summary of results."""
    logger.info("=" * 60)
    logger.info("WEIGHT TUNING RESULTS")
    logger.info("=" * 60)
    logger.info(results_df.to_string(index=False))

    best_idx = results_df["accuracy"].idxmax()
    best = results_df.loc[best_idx]

    logger.info("=" * 60)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  a = {best['alpha']:.2f}")
    logger.info(f"  Accuracy: {best['accuracy']:.1%}")
    logger.info(f"  Top-3: {best['top3']:.1%}")
    logger.info(f"  MRR: {best['mrr']:.3f}")

    baseline_row = results_df[results_df["alpha"] == 0.5]
    if not baseline_row.empty:
        baseline_acc = baseline_row.iloc[0]["accuracy"]
        improvement = best["accuracy"] - baseline_acc
        logger.info(f"  Improvement over a=0.5: {improvement:+.1%}")

        if improvement > 0.01:
            logger.info("  Found better weight!")
        elif improvement > 0:
            logger.info("  Marginal improvement")
        else:
            logger.info("  No improvement over current baseline")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune weights for combining definition and example embeddings"
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help="Specific alpha values to try (default: 0.0 to 1.0 in steps of 0.1)"
    )
    parser.add_argument(
        "--fine-grid",
        action="store_true",
        help="Use fine grid around 0.5 (0.3 to 0.7 in steps of 0.05)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model to use (default: {MODEL_NAME})"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.alphas:
        alphas = sorted(args.alphas)
    elif args.fine_grid:
        alphas = [round(x, 2) for x in np.arange(0.3, 0.71, 0.05)]
    else:
        alphas = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

    logger.info("=" * 60)
    logger.info("WEIGHT TUNING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Alpha values: {alphas}")
    logger.info("Formula: weighted_emb = a*definition + (1-a)*avg(examples)")

    logger.info("Loading data...")
    sense_df = load_sense_representations(SENSE_FILE)
    gold_df = pd.read_csv(GOLD_FILE)
    logger.info(f"  {len(sense_df)} senses, {len(gold_df)} test instances")

    results = tune_weights(alphas, sense_df, gold_df, args.model)

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to: {OUTPUT_FILE}")

    log_summary(results_df)


if __name__ == "__main__":
    main()