#!/usr/bin/env python3
"""
Generate annotation tasks from evaluation contexts and GermaNet senses.

Produces annotation_tasks.pkl (for the Flask app) and a JSON backup.

Usage:
    # Quick eval set (default)
    python scripts/annotation/generate_annotation_tasks.py

    # Custom paths
    python scripts/annotation/generate_annotation_tasks.py \
        --contexts-path data/evaluation/eval_contexts.csv \
        --words-path data/processed/target_words/eval_set_selection.tsv \
        --lemma-col word
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.annotation.task import AnnotationInstance, AnnotationTask

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_PATH / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotation tasks")
    parser.add_argument(
        "--contexts-path",
        default=str(DATA_DIR / "evaluation" / "quick_eval" / "contexts.csv"),
        help="Path to contexts CSV (must have 'Hit' and lemma column)",
    )
    parser.add_argument(
        "--words-path",
        default=str(DATA_DIR / "evaluation" / "quick_eval" / "words.csv"),
        help="Path to words CSV",
    )
    parser.add_argument(
        "--senses-path",
        default=str(DATA_DIR / "processed" / "germanet" / "wsd_senses.csv"),
        help="Path to WSD senses CSV",
    )
    parser.add_argument(
        "--lemma-col",
        default="lemma",
        help="Column name for the lemma in the words file (default: 'lemma')",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR / "evaluation"),
        help="Output directory for annotation_tasks.pkl and .json",
    )
    return parser.parse_args()


def create_tasks(
    contexts_df: pd.DataFrame,
    words_df: pd.DataFrame,
    senses_df: pd.DataFrame,
    lemma_col: str,
) -> Dict[str, AnnotationTask]:
    """Build annotation tasks from contexts, words and senses DataFrames."""
    tasks = {}

    for _, word_row in words_df.iterrows():
        lemma = word_row[lemma_col]
        pos = word_row.get("pos", "UNKNOWN")

        word_contexts = contexts_df[contexts_df["lemma"] == lemma]
        word_senses = senses_df[senses_df["lemma"] == lemma]

        if len(word_contexts) == 0 or len(word_senses) == 0:
            logger.warning(f"Skipping '{lemma}': no contexts or senses found")
            continue

        candidate_senses = [
            {
                "synset_id": row["synset_id"],
                "text_for_embedding": row["text_for_embedding"],
                "class": row["class"],
                "paraphrase": row.get("paraphrase", ""),
            }
            for _, row in word_senses.iterrows()
        ]

        instances = [
            AnnotationInstance(
                instance_id=f"{lemma}_{idx}",
                target_word=lemma,
                pos=pos,
                sentence=ctx_row["Hit"],
                target_position=0,
                candidate_senses=candidate_senses,
            )
            for idx, (_, ctx_row) in enumerate(word_contexts.iterrows())
        ]

        tasks[lemma] = AnnotationTask(lemma, instances)
        logger.info(f"  {lemma}: {len(instances)} instances, {len(candidate_senses)} senses")

    return tasks


def save_tasks(tasks: Dict[str, AnnotationTask], output_dir: Path) -> None:
    """Save tasks as pickle and JSON backup."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "annotation_tasks.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(tasks, f)
    logger.info(f"Saved: {pkl_path}")

    json_data = {
        word: {
            "target_word": task.target_word,
            "num_instances": len(task.instances),
            "num_senses": len(task.instances[0].candidate_senses) if task.instances else 0,
            "instances": [
                {
                    "instance_id": inst.instance_id,
                    "sentence": inst.sentence,
                    "candidate_senses": inst.candidate_senses,
                }
                for inst in task.instances
            ],
        }
        for word, task in tasks.items()
    }

    json_path = output_dir / "annotation_tasks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {json_path}")


def main() -> int:
    args = parse_args()

    logger.info("Loading data...")
    contexts_df = pd.read_csv(args.contexts_path)
    words_df = pd.read_csv(args.words_path)
    senses_df = pd.read_csv(args.senses_path)
    logger.info(f"  {len(contexts_df)} contexts, {len(words_df)} words, {len(senses_df)} senses")

    logger.info("Creating annotation tasks...")
    tasks = create_tasks(contexts_df, words_df, senses_df, lemma_col=args.lemma_col)

    if not tasks:
        logger.error("No tasks created — check your input files")
        return 1

    save_tasks(tasks, Path(args.output_dir))

    total = sum(len(t.instances) for t in tasks.values())
    logger.info(f"\nDone: {len(tasks)} words, {total} instances total")
    logger.info("To annotate, run: python scripts/annotation/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())