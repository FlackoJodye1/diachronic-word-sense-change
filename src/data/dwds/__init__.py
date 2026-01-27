"""DWDS corpus client and utilities."""

from .client import DWDSClient, DWDSError
from .fetcher import FetchOptions, fetch_contexts_for_word, apply_filters
from .context_utils import (
    clean_context,
    count_words,
    target_matches_token,
    is_near_duplicate,
    parse_year,
    add_temporal_columns,
)

__all__ = [
    "DWDSClient",
    "DWDSError",
    "FetchOptions",
    "fetch_contexts_for_word",
    "apply_filters",
    "clean_context",
    "count_words",
    "target_matches_token",
    "is_near_duplicate",
    "parse_year",
    "add_temporal_columns",
]