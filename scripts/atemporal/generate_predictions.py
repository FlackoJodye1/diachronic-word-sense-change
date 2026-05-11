#!/usr/bin/env python3
"""
Generate per-instance predictions for error analysis.

Supports three variants:
  - weighted (default): E5-large + 0.5*definition + 0.5*mean(examples)
  - text-only: E5-large + GermaNet hypernym text only (no LLM enrichment)
  - random: analytical random-chance per instance (1/n_senses fine,
    1/n_clusters coarse). No embedder needed.

Optionally applies a coarse cluster mapping to also report coarse accuracy.

Usage:
    python scripts/atemporal/generate_predictions.py
    python scripts/atemporal/generate_predictions.py --variant text-only --cluster-mapping data/processed/germanet/sense_clusters_reviewed.json
    python scripts/atemporal/generate_predictions.py --variant random --cluster-mapping data/processed/germanet/sense_clusters_reviewed.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.wsd import Embedder, SimilarityWSD

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration - best performing setup
MODEL_NAME = "intfloat/multilingual-e5-large"
DATA_DIR = Path("data")
ATEMPORAL_DIR = DATA_DIR / "evaluation" / "atemporal"
SENSE_REPRESENTATIONS_PATH = ATEMPORAL_DIR / "input" / "sense_text_representations.jsonl"
GOLD_ANNOTATIONS_PATH = ATEMPORAL_DIR / "input" / "gold_annotations.csv"
DEFAULT_OUTPUT_PATH = ATEMPORAL_DIR / "results" / "predictions.csv"

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


def add_e5_prefix(texts: list[str], prefix: str) -> list[str]:
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


def compute_text_only_embeddings(
    sense_df: pd.DataFrame,
    embedder: Embedder,
    use_e5_prefix: bool = True,
) -> np.ndarray:
    """Encode the baseline column (GermaNet hypernym text, no LLM enrichment)."""
    raw_texts = sense_df["baseline"].fillna("").tolist()
    texts = add_e5_prefix(raw_texts, "passage: ") if use_e5_prefix else raw_texts
    return embedder.encode(texts, show_progress=False)


@dataclass
class InstancePrediction:
    """Prediction details for a single instance."""
    lemma: str
    sentence: str
    gold_sense: str
    pred_sense: str
    correct: bool
    gold_rank: int  # 1-indexed rank of gold sense (1 = correct prediction)
    top5_senses: list[str]
    top5_scores: list[float]


def build_flat_coarse_mapping(cluster_mapping_path: Path) -> dict[str, str]:
    """Build a flat sense_id -> coarse_cluster_id lookup from the reviewed cluster JSON."""
    with open(cluster_mapping_path) as f:
        cm = json.load(f)
    flat: dict[str, str] = {}
    for lemma_data in cm.values():
        flat.update(lemma_data["mapping"])
    return flat


def generate_predictions(
    sense_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    embedder: Embedder,
    variant: str = "weighted",
) -> list[InstancePrediction]:
    """Generate per-instance predictions for an embedding-based variant."""
    if variant == "weighted":
        logger.info("Computing weighted sense embeddings...")
        sense_embeddings = compute_weighted_embeddings(sense_df, embedder, use_e5_prefix=True)
    elif variant == "text-only":
        logger.info("Computing text-only sense embeddings (GermaNet hypernym text)...")
        sense_embeddings = compute_text_only_embeddings(sense_df, embedder, use_e5_prefix=True)
    else:
        raise ValueError(f"Unknown embedding variant: {variant!r}")

    # Build sense lookup
    sense_ids = sense_df["synset_id"].tolist()
    sense_lemmas = sense_df["lemma"].tolist()
    sense_id_to_idx = {sid: i for i, sid in enumerate(sense_ids)}

    logger.info("Encoding context sentences...")
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


def run_random_baseline(
    sense_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    output_path: Path,
    cluster_mapping_path: Path | None,
) -> None:
    """Compute analytical random-chance accuracy per instance and write CSV."""
    # Fine: n_senses per lemma
    n_fine_per_lemma = sense_df.groupby("lemma")["synset_id"].nunique().to_dict()

    flat_mapping: dict[str, str] | None = None
    n_coarse_per_lemma: dict[str, int] = {}
    if cluster_mapping_path is not None:
        flat_mapping = build_flat_coarse_mapping(cluster_mapping_path)
        for lemma, group in sense_df.groupby("lemma"):
            clusters = {flat_mapping[sid] for sid in group["synset_id"] if sid in flat_mapping}
            n_coarse_per_lemma[lemma] = len(clusters)

    rows = []
    for _, row in gold_df.iterrows():
        lemma = row["lemma"]
        n_fine = n_fine_per_lemma.get(lemma, 0)
        p_fine = 1.0 / n_fine if n_fine else 0.0
        entry = {
            "lemma": lemma,
            "sentence": row["sentence"],
            "gold_sense": row["gold_sense_id"],
            "n_senses_fine": n_fine,
            "p_correct_fine": p_fine,
        }
        if flat_mapping is not None:
            gold_coarse = flat_mapping.get(row["gold_sense_id"])
            n_coarse = n_coarse_per_lemma.get(lemma, 0)
            entry["gold_coarse"] = gold_coarse
            entry["n_clusters_coarse"] = n_coarse
            entry["p_correct_coarse"] = (1.0 / n_coarse) if (gold_coarse and n_coarse) else 0.0
        rows.append(entry)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, index=False)

    fine_acc = results_df["p_correct_fine"].mean()
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Instances: {len(results_df)}")
    logger.info(f"Expected fine accuracy (random): {fine_acc:.1%}")

    if flat_mapping is not None:
        unmapped = results_df["gold_coarse"].isna().sum()
        if unmapped:
            logger.warning(f"  {unmapped} gold senses have no coarse mapping — excluded from coarse eval")
        coarse_df = results_df.dropna(subset=["gold_coarse"])
        coarse_acc = coarse_df["p_correct_coarse"].mean()
        logger.info(f"Expected coarse accuracy (random): {coarse_acc:.1%} (n={len(coarse_df)})")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for error analysis")
    parser.add_argument("--output", type=str, default=None, help="Output path for predictions CSV")
    parser.add_argument(
        "--variant",
        choices=["weighted", "text-only", "random"],
        default="weighted",
        help="Sense representation variant (default: weighted). 'text-only' = E5 + GermaNet hypernym text. 'random' = analytical random chance.",
    )
    parser.add_argument(
        "--cluster-mapping",
        type=str,
        default=None,
        help="Path to sense_clusters_reviewed.json for coarse accuracy evaluation",
    )
    args = parser.parse_args()

    variant_output_names = {
        "weighted": "predictions.csv",
        "text-only": "predictions_text_only.csv",
        "random": "predictions_random.csv",
    }
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ATEMPORAL_DIR / "results" / variant_output_names[args.variant]

    logger.info("Loading data...")
    sense_df = load_sense_representations(SENSE_REPRESENTATIONS_PATH)
    gold_df = pd.read_csv(GOLD_ANNOTATIONS_PATH)
    logger.info(f"  {len(sense_df)} senses, {len(gold_df)} test instances")

    cluster_mapping_path = Path(args.cluster_mapping) if args.cluster_mapping else None

    if args.variant == "random":
        logger.info("Computing analytical random-chance baseline...")
        run_random_baseline(sense_df, gold_df, output_path, cluster_mapping_path)
        return

    logger.info(f"Loading model: {MODEL_NAME}")
    embedder = Embedder(model_name=MODEL_NAME, normalize=True)

    logger.info(f"Generating predictions (variant: {args.variant})...")
    predictions = generate_predictions(sense_df, gold_df, embedder, variant=args.variant)

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

    accuracy = results_df["correct"].mean()
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Instances: {len(results_df)}")
    logger.info(f"Accuracy:  {accuracy:.1%}")
    logger.info(f"Errors:    {(1 - accuracy):.1%} ({(~results_df['correct']).sum()} instances)")

    logger.info("Gold rank distribution:")
    rank_counts = results_df["gold_rank"].value_counts().sort_index()
    for rank, count in rank_counts.items():
        pct = count / len(results_df) * 100
        label = "correct" if rank == 1 else f"rank {rank}"
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    if cluster_mapping_path is not None:
        logger.info("")
        logger.info("Applying coarse cluster mapping...")
        flat_mapping = build_flat_coarse_mapping(cluster_mapping_path)

        results_df["gold_coarse"] = results_df["gold_sense"].map(flat_mapping)
        results_df["pred_coarse"] = results_df["pred_sense"].map(flat_mapping)

        unmapped = results_df["gold_coarse"].isna().sum()
        if unmapped:
            logger.warning(f"  {unmapped} gold senses have no coarse mapping — excluded from coarse eval")

        coarse_df = results_df.dropna(subset=["gold_coarse", "pred_coarse"])
        coarse_correct = (coarse_df["gold_coarse"] == coarse_df["pred_coarse"]).sum()
        coarse_accuracy = coarse_correct / len(coarse_df)

        logger.info(f"Coarse accuracy: {coarse_accuracy:.1%} ({coarse_correct}/{len(coarse_df)})")


if __name__ == "__main__":
    main()