from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import pandas as pd

from .client import DWDSClient
from .util.context_utils import (
    add_temporal_columns,
    clean_context,
    count_words,
    derive_seed,
    get_doc_key,
    is_near_duplicate,
    target_matches_token,
)


@dataclass
class FetchOptions:
    corpus: str = "kern"
    limit: int = 200
    sort: Optional[str] = "random"
    min_tokens: int = 8
    max_tokens: int = 35
    max_per_doc: int = 1
    near_dup_threshold: float = 0.7
    target_n: int = 100
    auto_select: bool = False
    per_decade_n: int = 0
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    seed: int = 42


def apply_filters(
    contexts_df: pd.DataFrame,
    target_word: str,
    options: FetchOptions,
) -> pd.DataFrame:
    """Apply filtering criteria to select best contexts."""
    if contexts_df.empty:
        return contexts_df

    df = contexts_df.copy()
    df["Hit_clean"] = df["Hit"].apply(clean_context)
    df["token_count"] = df["Hit_clean"].apply(count_words)

    df = df[
        (df["token_count"] >= options.min_tokens)
        & (df["token_count"] <= options.max_tokens)
    ].copy()

    df["target_matches"] = df["Hit_clean"].apply(
        lambda x: target_matches_token(x, target_word)
    )
    df = df[df["target_matches"]].copy()

    if df.empty:
        return df

    df["chosen_doc_key"] = df.apply(get_doc_key, axis=1)
    if options.max_per_doc == 1:
        df = df.drop_duplicates(subset=["chosen_doc_key"]).copy()
    else:
        df = df.groupby("chosen_doc_key").head(options.max_per_doc).copy()

    unique_contexts = []
    for _, row in df.iterrows():
        current_text = row["Hit_clean"]
        if not any(
            is_near_duplicate(current_text, prev["Hit_clean"], options.near_dup_threshold)
            for prev in unique_contexts
        ):
            unique_contexts.append(row)

    if not unique_contexts:
        return df.iloc[0:0]

    df = pd.DataFrame(unique_contexts)
    df = add_temporal_columns(df)

    if options.min_year is not None:
        df = df[df["year"] >= options.min_year]
    if options.max_year is not None:
        df = df[df["year"] <= options.max_year]

    if options.per_decade_n > 0:
        per_decade_rows = []
        lemma_seed = derive_seed(options.seed, target_word)
        for decade, group in df.groupby("decade"):
            if pd.isna(decade):
                continue
            indices = list(group.index)
            random.Random(lemma_seed + int(decade)).shuffle(indices)
            indices = indices[: options.per_decade_n]
            per_decade_rows.append(group.loc[indices])
        if per_decade_rows:
            df = pd.concat(per_decade_rows, ignore_index=True)

    if options.auto_select and 0 < options.target_n < len(df):
        lemma_seed = derive_seed(options.seed, target_word)
        indices = list(df.index)
        random.Random(lemma_seed).shuffle(indices)
        df = df.loc[indices[: options.target_n]].copy()

    return df.reset_index(drop=True)


def fetch_contexts_for_word(
    word: str,
    client: DWDSClient,
    options: FetchOptions,
    limit_override: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch and filter DWDS contexts for a single word."""
    limit = limit_override or options.limit
    contexts_df = client.corpus_csv(
        word,
        corpus=options.corpus,
        limit=limit,
        sort=options.sort,
    )
    if contexts_df is None or len(contexts_df) == 0:
        return pd.DataFrame()

    filtered_df = apply_filters(contexts_df, word, options)
    if len(filtered_df) == 0:
        return filtered_df

    filtered_df["lemma"] = word
    return filtered_df
