"""Data loading utilities for diachronic analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_sense_counts(filepath: Path) -> pd.DataFrame:
    """Load aggregated sense counts."""
    print(f"Loading sense counts from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} records for {df['lemma'].nunique()} words")
    return df


def load_predictions(filepath: Path) -> pd.DataFrame:
    """Load predictions from CSV."""
    print(f"Loading predictions from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} predictions for {df['lemma'].nunique()} words")
    return df


def load_dwds_contexts(filepath: Path) -> pd.DataFrame:
    """Load DWDS contexts from JSONL file."""
    print(f"Loading DWDS contexts from {filepath}...")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)
    df["decade"] = (df["year"] // 10) * 10
    print(f"  Loaded {len(df)} contexts for {df['lemma'].nunique()} words")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
    return df


def load_senses(
    filepath: Path,
    text_column: str,
    target_lemmas: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load senses from CSV, optionally filtered to target lemmas.

    Args:
        filepath: Path to senses CSV (wsd_senses.csv or sense_variants.csv)
        text_column: Column name containing text for embedding
        target_lemmas: Optional list of lemmas to filter to
    """
    print(f"Loading senses from {filepath}...")
    df = pd.read_csv(filepath)

    # Validate text column exists
    if text_column not in df.columns:
        available = [c for c in df.columns if "text" in c.lower()]
        raise ValueError(
            f"Text column '{text_column}' not found. Available text columns: {available}"
        )

    if target_lemmas:
        df = df[df["lemma"].isin(target_lemmas)]

    print(f"  Loaded {len(df)} senses for {df['lemma'].nunique()} words")
    print(f"  Using text column: {text_column}")
    return df