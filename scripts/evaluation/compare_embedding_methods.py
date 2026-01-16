#!/usr/bin/env python3
"""
Compare embedding models on the quick_eval gold annotations.

This script evaluates different embedding models and text variants for WSD
by comparing sense embeddings against context sentence embeddings using
cosine similarity.

Reports accuracy, top-k accuracy, and MRR for each model/variant combination.

Usage:
    python compare_embedding_methods.py                  # Normal run
    python compare_embedding_methods.py --clear-cache   # Clear model cache after each model
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation import Embedder, SimilarityWSD, WSDResult

# Configuration
DATA_DIR = Path("data")
QUICK_EVAL_DIR = DATA_DIR / "evaluation" / "quick_eval"
SENSE_REPRESENTATIONS_PATH = QUICK_EVAL_DIR / "sense_text_representations.jsonl"
GOLD_ANNOTATIONS_PATH = QUICK_EVAL_DIR / "gold_annotations.csv"

# Models to compare (only sentence-transformer compatible models for fair comparison)
MODELS = [
    # German-specific models
    # "T-Systems-onsite/german-roberta-sentence-transformer-v2",
    # Higher quality multilingual
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    # E5 models (strong retrieval/similarity performance)
    # Note: E5 models require "query: " / "passage: " prefixes, handled automatically
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    # BGE-M3: strong multilingual model, competitive for German
    "BAAI/bge-m3",
]

# Text variants to evaluate (None = auto-detect)
# Available variants after loading: baseline, llm_definition, llm_example_0..4
TEXT_VARIANTS: Optional[List[str]] = None

# Special weighted variant: 0.5 * def_embedding + 0.5 * mean(example_embeddings)
# Computes true mean over NON-EMPTY example embeddings only (per sense)
# Prefix __ indicates this is a synthetic variant computed on-the-fly
WEIGHTED_VARIANT_NAME = "__weighted_def_plus_examples"
WEIGHTED_DEF_COL = "llm_definition"
WEIGHTED_EX_COLS = [f"llm_example_{i}" for i in range(5)]
WEIGHTED_DEF_WEIGHT = 0.5
WEIGHTED_EX_WEIGHT = 0.5

OUTPUT_PATH = QUICK_EVAL_DIR / "embedding_comparison.csv"


@dataclass
class ComparisonResult:
    """Results for a single model + text variant combination."""

    model_name: str
    text_variant: str
    accuracy: float
    top3: float
    top5: float
    mrr: float
    evaluated: int
    skipped: int


def is_e5_model(model_name: str) -> bool:
    """Check if model is an E5 model requiring query/passage prefixes."""
    return "e5" in model_name.lower()


def add_e5_prefix(texts: List[str], prefix: str) -> List[str]:
    """Add E5 prefix to texts."""
    return [f"{prefix}{t}" for t in texts]


def clear_model_cache(model_name: str) -> None:
    """Clear a specific model from HuggingFace cache to free disk space."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HuggingFace stores models with -- instead of /
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = cache_dir / model_cache_name

    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
        print(f"  Cleared disk cache for {model_name}")


def run_comparison(
    models: Sequence[str],
    text_variants: Sequence[str],
    sense_variants: pd.DataFrame,
    gold: pd.DataFrame,
    clear_cache: bool = False,
) -> List[ComparisonResult]:
    """Run WSD comparison across models and text variants.

    Args:
        models: List of model names to evaluate
        text_variants: List of text variant column names
        sense_variants: DataFrame with sense variant texts
        gold: DataFrame with gold annotations
        clear_cache: Whether to clear disk cache after each model

    Returns:
        List of ComparisonResult for each model/variant combination
    """
    results: List[ComparisonResult] = []
    raw_context_texts = gold["sentence"].fillna("").tolist()
    context_lemmas = gold["lemma"].tolist()
    gold_sense_ids = gold["gold_sense_id"].tolist()

    for model_name in models:
        print(f"\nLoading model: {model_name}")
        embedder = Embedder(model_name=model_name, normalize=True)

        # Check if E5 model (requires prefixes)
        use_e5_prefix = is_e5_model(model_name)
        if use_e5_prefix:
            print("  (E5 model: applying query/passage prefixes)")

        # Prepare context texts (with prefix if E5)
        context_texts = (
            add_e5_prefix(raw_context_texts, "query: ")
            if use_e5_prefix
            else raw_context_texts
        )

        # Encode contexts once per model
        print("  Encoding context sentences...")
        context_embeddings = embedder.encode(context_texts, show_progress=True)

        for text_variant in text_variants:
            print(f"  Evaluating variant: {text_variant}")

            # Handle weighted variant specially
            if text_variant == WEIGHTED_VARIANT_NAME:
                # Validate required columns exist
                if WEIGHTED_DEF_COL not in sense_variants.columns:
                    raise KeyError(f"Weighted variant requires column '{WEIGHTED_DEF_COL}'")
                missing_ex_cols = [c for c in WEIGHTED_EX_COLS if c not in sense_variants.columns]
                if missing_ex_cols:
                    raise KeyError(f"Weighted variant requires columns: {missing_ex_cols}")

                # Encode definition
                raw_def_texts = sense_variants[WEIGHTED_DEF_COL].fillna("").tolist()
                def_texts = (
                    add_e5_prefix(raw_def_texts, "passage: ")
                    if use_e5_prefix
                    else raw_def_texts
                )
                def_embeddings = embedder.encode(def_texts, show_progress=False)

                # Encode each example column and compute mean over NON-EMPTY examples
                example_embeddings_list = []
                example_masks = []  # Track which examples are non-empty per sense

                for ex_col in WEIGHTED_EX_COLS:
                    raw_ex_texts = sense_variants[ex_col].fillna("").tolist()
                    # Create mask: True if example text is non-empty
                    is_nonempty = [bool(t.strip()) for t in raw_ex_texts]
                    example_masks.append(is_nonempty)

                    ex_texts = (
                        add_e5_prefix(raw_ex_texts, "passage: ")
                        if use_e5_prefix
                        else raw_ex_texts
                    )
                    ex_emb = embedder.encode(ex_texts, show_progress=False)
                    example_embeddings_list.append(ex_emb)

                # Stack: (5, n_senses, dim)
                examples_stacked = np.stack(example_embeddings_list, axis=0)
                # Mask: (5, n_senses) -> (5, n_senses, 1) for broadcasting
                masks_stacked = np.array(example_masks, dtype=np.float32)  # (5, n_senses)
                masks_expanded = masks_stacked[:, :, np.newaxis]  # (5, n_senses, 1)

                # Apply mask and sum
                masked_embeddings = examples_stacked * masks_expanded
                sum_embeddings = np.sum(masked_embeddings, axis=0)  # (n_senses, dim)

                # Count non-empty examples per sense
                counts = np.sum(masks_stacked, axis=0, keepdims=True).T  # (n_senses, 1)
                counts = np.maximum(counts, 1)  # Avoid division by zero

                # Compute mean over available examples only
                mean_example_embeddings = sum_embeddings / counts

                # Weighted average: v = 0.5 * def_emb + 0.5 * mean(example_embs)
                sense_embeddings = (
                    WEIGHTED_DEF_WEIGHT * def_embeddings
                    + WEIGHTED_EX_WEIGHT * mean_example_embeddings
                )
                # Re-normalize after averaging
                norms = np.linalg.norm(sense_embeddings, axis=1, keepdims=True)
                sense_embeddings = sense_embeddings / np.where(norms == 0, 1, norms)

                del (
                    def_embeddings,
                    example_embeddings_list,
                    example_masks,
                    examples_stacked,
                    masks_stacked,
                    masks_expanded,
                    masked_embeddings,
                    sum_embeddings,
                    counts,
                    mean_example_embeddings,
                )
            else:
                # Standard: encode single text column
                raw_sense_texts = sense_variants[text_variant].fillna("").tolist()
                sense_texts = (
                    add_e5_prefix(raw_sense_texts, "passage: ")
                    if use_e5_prefix
                    else raw_sense_texts
                )

                # Encode sense texts for this variant
                sense_embeddings = embedder.encode(sense_texts, show_progress=False)

            # Create WSD predictor and evaluate
            wsd = SimilarityWSD.from_dataframe(
                sense_df=sense_variants,
                embeddings=sense_embeddings,
                sense_id_col="synset_id",
                lemma_col="lemma",
            )

            wsd_result: WSDResult = wsd.evaluate(
                context_embeddings=context_embeddings,
                lemmas=context_lemmas,
                gold_sense_ids=gold_sense_ids,
            )

            results.append(
                ComparisonResult(
                    model_name=model_name,
                    text_variant=text_variant,
                    accuracy=wsd_result.accuracy,
                    top3=wsd_result.top3_accuracy,
                    top5=wsd_result.top5_accuracy,
                    mrr=wsd_result.mrr,
                    evaluated=wsd_result.evaluated,
                    skipped=wsd_result.skipped,
                )
            )

            print(
                f"    Accuracy: {wsd_result.accuracy:.3f} | "
                f"Top-3: {wsd_result.top3_accuracy:.3f} | "
                f"MRR: {wsd_result.mrr:.3f}"
            )

            # Free RAM after each variant
            del sense_embeddings, wsd, wsd_result
            gc.collect()

        # Free RAM after each model
        del context_embeddings, embedder
        gc.collect()

        # Optionally clear disk cache
        if clear_cache:
            clear_model_cache(model_name)

    return results


def print_summary(results_df: pd.DataFrame, output_path: Path) -> None:
    """Print comparison summary and best configuration."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")

    # Find best configuration
    best_idx = results_df["accuracy"].idxmax()
    best = results_df.loc[best_idx]
    print(f"\nBest configuration:")
    print(f"  Model: {best['model_name']}")
    print(f"  Variant: {best['text_variant']}")
    print(f"  Accuracy: {best['accuracy']:.3f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare embedding models for WSD evaluation"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear each model from HuggingFace cache after evaluation (saves disk space)",
    )
    return parser.parse_args()


def load_sense_representations(path: Path) -> pd.DataFrame:
    """Load sense representations from JSONL and expand examples into columns.

    The JSONL has fields: lemma, pos, synset_id, class, baseline, llm_definition, llm_examples
    This expands llm_examples (a list) into llm_example_0, llm_example_1, etc.
    """
    df = pd.read_json(path, lines=True)

    # Expand llm_examples list into separate columns
    if "llm_examples" in df.columns:
        examples_expanded = df["llm_examples"].apply(pd.Series)
        examples_expanded.columns = [f"llm_example_{i}" for i in range(len(examples_expanded.columns))]
        df = pd.concat([df.drop(columns=["llm_examples"]), examples_expanded], axis=1)

    return df


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.clear_cache:
        print("Running with --clear-cache: models will be removed after evaluation")

    print("Loading data...")
    sense_df = load_sense_representations(SENSE_REPRESENTATIONS_PATH)
    gold = pd.read_csv(GOLD_ANNOTATIONS_PATH)

    # Auto-detect text variants if not specified
    # Available: baseline, llm_definition, llm_example_0..N
    text_variants = TEXT_VARIANTS
    if not text_variants:
        text_cols = ["baseline", "llm_definition"]
        example_cols = [c for c in sense_df.columns if c.startswith("llm_example_")]
        text_variants = text_cols + sorted(example_cols)

        # Add weighted variant if required columns exist
        has_def_col = WEIGHTED_DEF_COL in sense_df.columns
        has_ex_cols = all(c in sense_df.columns for c in WEIGHTED_EX_COLS)
        if has_def_col and has_ex_cols:
            text_variants.append(WEIGHTED_VARIANT_NAME)

    if not text_variants:
        raise SystemExit("No text variants found in sense representations.")

    print(f"Found {len(text_variants)} text variants: {text_variants}")
    print(f"Comparing {len(MODELS)} models")

    # Run comparison
    results = run_comparison(
        models=MODELS,
        text_variants=text_variants,
        sense_variants=sense_df,
        gold=gold,
        clear_cache=args.clear_cache,
    )

    # Save results
    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df = results_df[
        ["model_name", "text_variant", "accuracy", "top3", "top5", "mrr", "evaluated", "skipped"]
    ]
    results_df.to_csv(OUTPUT_PATH, index=False)

    print_summary(results_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()