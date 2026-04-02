"""Data loading utilities for diachronic analysis."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_sense_counts(filepath: Path) -> pd.DataFrame:
    """Load aggregated sense counts."""
    logger.info(f"Loading sense counts from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"  Loaded {len(df)} records for {df['lemma'].nunique()} words")
    return df


def load_predictions(filepath: Path) -> pd.DataFrame:
    """Load predictions from CSV."""
    logger.info(f"Loading predictions from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"  Loaded {len(df)} predictions for {df['lemma'].nunique()} words")
    return df


def load_dwds_contexts(filepath: Path) -> pd.DataFrame:
    """Load DWDS contexts from JSONL file."""
    logger.info(f"Loading DWDS contexts from {filepath}...")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)
    df["decade"] = (df["year"] // 10) * 10
    logger.info(f"  Loaded {len(df)} contexts for {df['lemma'].nunique()} words")
    logger.info(f"  Year range: {df['year'].min()}-{df['year'].max()}")
    return df


_SENSE_FILES = [
    "data/processed/germanet/senses_pilot.jsonl",
    "data/processed/germanet/senses_expanded.jsonl",
    "data/evaluation/atemporal/input/sense_text_representations.jsonl",
]


def load_all_senses(
    base_dir: Optional[Path] = None,
    target_lemmas: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load senses from all known sense files and merge into one DataFrame.

    Covers both pilot words (senses_pilot.jsonl) and eval words
    (sense_text_representations.jsonl). Use this instead of load_senses()
    when you need a complete sense lookup across all 96 lemmas.

    Args:
        base_dir: Project root. Defaults to three levels above this file.
        target_lemmas: Optional list of lemmas to filter to.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    frames = []
    for rel_path in _SENSE_FILES:
        path = base_dir / rel_path
        if path.exists():
            frames.append(load_senses(path))
        else:
            logger.warning(f"Sense file not found, skipping: {path}")

    if not frames:
        raise FileNotFoundError("No sense files found.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["synset_id"])

    if target_lemmas:
        df = df[df["lemma"].isin(target_lemmas)]

    logger.info(f"load_all_senses: {len(df)} senses for {df['lemma'].nunique()} words total")
    return df


def load_senses(
    filepath: Path,
    target_lemmas: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load senses from CSV or JSONL, optionally filtered to target lemmas.

    Supports:
    - CSV files (senses_base.csv)
    - JSONL files (senses_pilot.jsonl, senses_full.jsonl) — expands the
      llm_examples list column into individual llm_example_1..5 columns
      as expected by embed_with_representation()

    Args:
        filepath: Path to senses file (.csv or .jsonl)
        target_lemmas: Optional list of lemmas to filter to
    """
    logger.info(f"Loading senses from {filepath}...")

    if Path(filepath).suffix == ".jsonl":
        df = pd.read_json(filepath, lines=True)
        if "llm_examples" in df.columns:
            examples_expanded = df["llm_examples"].apply(pd.Series)
            examples_expanded.columns = [
                f"llm_example_{i + 1}" for i in range(len(examples_expanded.columns))
            ]
            df = pd.concat([df.drop(columns=["llm_examples"]), examples_expanded], axis=1)
    else:
        df = pd.read_csv(filepath)

    if target_lemmas:
        df = df[df["lemma"].isin(target_lemmas)]

    logger.info(f"  Loaded {len(df)} senses for {df['lemma'].nunique()} words")
    return df
