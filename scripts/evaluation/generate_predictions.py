#!/usr/bin/env python3
"""
Generate per-instance predictions for error analysis.

Runs the best configuration (E5-large + weighted def + examples) and saves
detailed predictions including gold sense, predicted sense, rank, and top-k
candidates for each instance.

Usage:
    python scripts/evaluation/generate_predictions.py
    python scripts/evaluation/generate_predictions.py --output path/to/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation import Embedder, SimilarityWSD

# Configuration - best performing setup
MODEL_NAME = "intfloat/multilingual-e5-large"
DATA_DIR = Path("data")
QUICK_EVAL_DIR = DATA_DIR / "evaluation" / "quick_eval"
SENSE_REPRESENTATIONS_PATH = QUICK_EVAL_DIR / "sense_text_representations.jsonl"
GOLD_ANNOTATIONS_PATH = QUICK_EVAL_DIR / "gold_annotations.csv"
DEFAULT_OUTPUT_PATH = QUICK_EVAL_DIR / "predictions.csv"

# Weighted variant config
WEIGHTED_DEF_COL = "llm_definition"
WEIGHTED_EX_COLS = [f"llm_example_{i}" for i in range(5)]
WEIGHTED_DEF_WEIGHT = 0.5
WEIGHTED_EX_WEIGHT = 0.5


def load_sense_representations(path: Path) -> pd.DataFrame:
    """Load sense representations from JSONL and expand examples into columns."""
    df = pd.read_json(path, lines=True)
    if "llm_examples" in df.columns:
        examples_expanded = df["llm_examples"].apply(pd.Series)
        examples_expanded.columns = [f"llm_example_{i}" for i in range(len(examples_expanded.columns))]
        df = pd.concat([df.drop(columns=["llm_examples"]), examples_expanded], axis=1)
    return df


def add_e5_prefix(texts: List[str], prefix: str) -> List[str]:
    """Add E5 prefix to texts."""
    return [f"{prefix}{t}" for t in texts]


def compute_weighted_embeddings(
    sense_df: pd.DataFrame,
    embedder: Embedder,
    use_e5_prefix: bool = True,
) -> np.ndarray:
    """Compute weighted sense embeddings (0.5 * definition + 0.5 * mean(examples))."""
    # Encode definition
    raw_def_texts = sense_df[WEIGHTED_DEF_COL].fillna("").tolist()
    def_texts = add_e5_prefix(raw_def_texts, "passage: ") if use_e5_prefix else raw_def_texts
    def_embeddings = embedder.encode(def_texts, show_progress=False)

    # Encode examples and compute mean
    example_embeddings_list = []
    example_masks = []

    for ex_col in WEIGHTED_EX_COLS:
        raw_ex_texts = sense_df[ex_col].fillna("").tolist()
        is_nonempty = [bool(t.strip()) for t in raw_ex_texts]
        example_masks.append(is_nonempty)

        ex_texts = add_e5_prefix(raw_ex_texts, "passage: ") if use_e5_prefix else raw_ex_texts
        ex_emb = embedder.encode(ex_texts, show_progress=False)
        example_embeddings_list.append(ex_emb)

    # Stack and compute masked mean
    examples_stacked = np.stack(example_embeddings_list, axis=0)
    masks_stacked = np.array(example_masks, dtype=np.float32)
    masks_expanded = masks_stacked[:, :, np.newaxis]

    masked_embeddings = examples_stacked * masks_expanded
    sum_embeddings = np.sum(masked_embeddings, axis=0)
    counts = np.sum(masks_stacked, axis=0, keepdims=True).T
    counts = np.maximum(counts, 1)
    mean_example_embeddings = sum_embeddings / counts

    # Weighted average
    sense_embeddings = (
        WEIGHTED_DEF_WEIGHT * def_embeddings
        + WEIGHTED_EX_WEIGHT * mean_example_embeddings
    )

    # Re-normalize
    norms = np.linalg.norm(sense_embeddings, axis=1, keepdims=True)
    sense_embeddings = sense_embeddings / np.where(norms == 0, 1, norms)

    return sense_embeddings


@dataclass
class InstancePrediction:
    """Prediction details for a single instance."""
    lemma: str
    sentence: str
    gold_sense: str
    pred_sense: str
    correct: bool
    gold_rank: int  # 1-indexed rank of gold sense (1 = correct prediction)
    top5_senses: List[str]
    top5_scores: List[float]


def generate_predictions(
    sense_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    embedder: Embedder,
) -> List[InstancePrediction]:
    """Generate per-instance predictions."""
    print("Computing weighted sense embeddings...")
    sense_embeddings = compute_weighted_embeddings(sense_df, embedder, use_e5_prefix=True)

    # Build sense lookup
    sense_ids = sense_df["synset_id"].tolist()
    sense_lemmas = sense_df["lemma"].tolist()
    sense_id_to_idx = {sid: i for i, sid in enumerate(sense_ids)}

    # Encode contexts
    print("Encoding context sentences...")
    raw_contexts = gold_df["sentence"].fillna("").tolist()
    context_texts = add_e5_prefix(raw_contexts, "query: ")
    context_embeddings = embedder.encode(context_texts, show_progress=True)

    predictions = []

    for i, row in gold_df.iterrows():
        lemma = row["lemma"]
        sentence = row["sentence"]
        gold_sense_id = row["gold_sense_id"]

        # Get candidate senses for this lemma
        candidate_mask = [l == lemma for l in sense_lemmas]
        candidate_indices = [j for j, m in enumerate(candidate_mask) if m]

        if not candidate_indices:
            continue

        # Compute similarities
        context_emb = context_embeddings[i]
        candidate_embs = sense_embeddings[candidate_indices]
        similarities = np.dot(candidate_embs, context_emb)

        # Rank candidates
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_sense_ids = [sense_ids[candidate_indices[j]] for j in ranked_indices]
        ranked_scores = [float(similarities[j]) for j in ranked_indices]

        # Find gold rank
        if gold_sense_id in ranked_sense_ids:
            gold_rank = ranked_sense_ids.index(gold_sense_id) + 1
        else:
            gold_rank = -1  # Gold sense not in candidates (shouldn't happen)

        pred_sense_id = ranked_sense_ids[0]

        predictions.append(InstancePrediction(
            lemma=lemma,
            sentence=sentence,
            gold_sense=gold_sense_id,
            pred_sense=pred_sense_id,
            correct=(pred_sense_id == gold_sense_id),
            gold_rank=gold_rank,
            top5_senses=ranked_sense_ids[:5],
            top5_scores=ranked_scores[:5],
        ))

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for error analysis")
    parser.add_argument("--output", type=str, default=None, help="Output path for predictions CSV")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_PATH

    print(f"Loading data...")
    sense_df = load_sense_representations(SENSE_REPRESENTATIONS_PATH)
    gold_df = pd.read_csv(GOLD_ANNOTATIONS_PATH)
    print(f"  {len(sense_df)} senses, {len(gold_df)} test instances")

    print(f"\nLoading model: {MODEL_NAME}")
    embedder = Embedder(model_name=MODEL_NAME, normalize=True)

    print(f"\nGenerating predictions...")
    predictions = generate_predictions(sense_df, gold_df, embedder)

    # Convert to DataFrame
    results = []
    for p in predictions:
        results.append({
            "lemma": p.lemma,
            "sentence": p.sentence,
            "gold_sense": p.gold_sense,
            "pred_sense": p.pred_sense,
            "correct": p.correct,
            "gold_rank": p.gold_rank,
            "top5_senses": str(p.top5_senses),
            "top5_scores": str([round(s, 4) for s in p.top5_scores]),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # Summary
    accuracy = results_df["correct"].mean()
    print(f"Results saved to: {output_path}")
    print(f"Instances: {len(results_df)}")
    print(f"Accuracy:  {accuracy:.1%}")
    print(f"Errors:    {(1 - accuracy):.1%} ({(~results_df['correct']).sum()} instances)")

    # Quick breakdown
    print(f"\nGold rank distribution:")
    rank_counts = results_df["gold_rank"].value_counts().sort_index()
    for rank, count in rank_counts.items():
        pct = count / len(results_df) * 100
        label = "correct" if rank == 1 else f"rank {rank}"
        print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()