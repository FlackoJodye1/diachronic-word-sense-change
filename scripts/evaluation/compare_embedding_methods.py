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

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation import Embedder, SimilarityWSD, WSDResult

# Configuration
DATA_DIR = Path("data")
QUICK_EVAL_DIR = DATA_DIR / "evaluation" / "quick_eval"
SENSE_VARIANTS_PATH = QUICK_EVAL_DIR / "sense_variants.csv"
GOLD_ANNOTATIONS_PATH = QUICK_EVAL_DIR / "gold_annotations.csv"

# Models to compare (only sentence-transformer compatible models for fair comparison)
MODELS = [
    # Fast multilingual models
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    # Higher quality multilingual
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    # E5 models (strong retrieval/similarity performance)
    # Note: E5 models require "query: " / "passage: " prefixes, handled automatically
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
]

# Text variants to evaluate (None = auto-detect from columns starting with "text_")
TEXT_VARIANTS: Optional[List[str]] = None

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

            # Prepare sense texts (with prefix if E5)
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


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.clear_cache:
        print("Running with --clear-cache: models will be removed after evaluation")

    print("Loading data...")
    sense_variants = pd.read_csv(SENSE_VARIANTS_PATH)
    gold = pd.read_csv(GOLD_ANNOTATIONS_PATH)

    # Auto-detect text variants if not specified
    text_variants = TEXT_VARIANTS
    if not text_variants:
        text_variants = [col for col in sense_variants.columns if col.startswith("text_")]

    if not text_variants:
        raise SystemExit("No text variants found. Expected columns starting with 'text_'.")

    print(f"Found {len(text_variants)} text variants: {text_variants}")
    print(f"Comparing {len(MODELS)} models")

    # Run comparison
    results = run_comparison(
        models=MODELS,
        text_variants=text_variants,
        sense_variants=sense_variants,
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