from __future__ import annotations

import hashlib
import re
import string
from typing import Optional

import pandas as pd


def clean_context(text: str) -> str:
    """Clean artifacts from context text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\.\.\.\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[sic\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def target_matches_token(text: str, target_word: str) -> bool:
    """Check if target word occurs as a standalone token (case-insensitive)."""
    pattern = rf"(?<!\w){re.escape(target_word)}(?!\w)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches) == 1


def normalize_for_comparison(text: str) -> str:
    """Normalize text for near-duplicate comparison."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity on token sets."""
    tokens1 = set(normalize_for_comparison(text1).split())
    tokens2 = set(normalize_for_comparison(text2).split())

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union


def is_near_duplicate(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """Check if two texts are near duplicates using Jaccard similarity."""
    return jaccard_similarity(text1, text2) > threshold


def get_doc_key(row: pd.Series) -> str:
    """Get document key for deduplication."""
    if "doc_id" in row and pd.notna(row["doc_id"]):
        return str(row["doc_id"])

    bibl = str(row.get("Bibl", row.get("bibl", "")))
    if ":" in bibl:
        author_title = bibl.split(":", 1)[0] + ":" + bibl.split(":", 1)[1].split(",")[0]
        return author_title.strip()
    return bibl


def derive_seed(global_seed: int, lemma: str) -> int:
    """Derive per-lemma seed from global seed and lemma."""
    combined = f"{global_seed}_{lemma}"
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


def parse_year(value: Optional[object]) -> Optional[int]:
    """Extract a 4-digit year from a value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value)
    match = re.search(r"\b(\d{4})\b", s)
    if match:
        return int(match.group(1))
    return None


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add year and decade columns derived from Date/year fields."""
    df = df.copy()
    if "year" in df.columns:
        df["year"] = df["year"].apply(parse_year)
    else:
        df["year"] = df.get("Date", pd.Series([None] * len(df))).apply(parse_year)
    df["decade"] = df["year"].apply(lambda y: (y // 10) * 10 if pd.notna(y) else None)
    return df
