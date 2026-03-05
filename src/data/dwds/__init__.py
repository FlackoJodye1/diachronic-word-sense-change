"""DWDS corpus client and utilities."""

from .client import DWDSClient
from .fetcher import FetchOptions, fetch_contexts_for_word, apply_filters

__all__ = [
    "DWDSClient",
    "FetchOptions",
    "fetch_contexts_for_word",
    "apply_filters",
]