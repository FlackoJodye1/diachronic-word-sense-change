#!/usr/bin/env python3
"""
Fetch DWDS contexts for a list of target words and write JSONL output.

This is a general-purpose fetcher for building diachronic inputs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dwds.client import DWDSClient
from src.data.dwds.fetcher import FetchOptions, fetch_contexts_for_word
from src.data.dwds.util.context_utils import parse_year

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_words(words_path: Path) -> list[str]:
    """Load target words from a .txt list or a table with a word/lemma column."""
    if words_path.suffix.lower() == ".txt":
        with words_path.open("r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        return words

    df = pd.read_csv(words_path, sep=None, engine="python")
    for col in ["word", "lemma", "target_word"]:
        if col in df.columns:
            return [str(w).strip() for w in df[col].dropna().tolist() if str(w).strip()]

    raise ValueError(
        f"Could not find a word column in {words_path}. "
        "Expected one of: word, lemma, target_word"
    )


def record_from_row(row: pd.Series, lemma: str, corpus: str) -> dict:
    """Build JSONL record from DWDS row."""
    date = row.get("Date", row.get("date", None))
    year = row.get("year")
    decade = row.get("decade")
    if pd.isna(year):
        year = parse_year(date)
    if pd.isna(decade) and year is not None:
        decade = (int(year) // 10) * 10

    pos = row.get("Wortart", row.get("pos", "UNKNOWN"))

    return {
        "lemma": lemma,
        "pos": pos if pd.notna(pos) else "UNKNOWN",
        "sentence": row.get("Hit_clean", row.get("Hit", "")),
        "date": date,
        "year": int(year) if year is not None and not pd.isna(year) else None,
        "decade": int(decade) if decade is not None and not pd.isna(decade) else None,
        "genre": row.get("Genre", row.get("genre", None)),
        "bibl": row.get("Bibl", row.get("bibl", None)),
        "corpus": corpus,
        "source": "dwds_corpus_csv",
    }


def iter_existing_lemmas(output_path: Path) -> set[str]:
    """Yield lemmas already present in a JSONL output file."""
    if not output_path.exists():
        return set()
    lemmas = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "lemma" in data:
                    lemmas.add(data["lemma"])
            except json.JSONDecodeError:
                continue
    return lemmas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch DWDS contexts for a list of target words"
    )
    parser.add_argument(
        "--words-file",
        type=Path,
        default=Path("data/raw/target_words/word_list_750.txt"),
        help="Path to target word list (.txt or .csv/.tsv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dwds/contexts.jsonl"),
        help="JSONL output file path",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="kern",
        help="DWDS corpus name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of contexts to fetch per word before filtering",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=100,
        help="Target number of contexts per word after filtering (0 = keep all)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=8,
        help="Minimum number of tokens in context",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=35,
        help="Maximum number of tokens in context",
    )
    parser.add_argument(
        "--max-per-doc",
        type=int,
        default=1,
        help="Maximum contexts per document",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.7,
        help="Jaccard similarity threshold for near-duplicate removal",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        default=False,
        help="Auto-select target_n contexts (default: keep all filtered)",
    )
    parser.add_argument(
        "--per-decade-n",
        type=int,
        default=0,
        help="Limit contexts per decade (0 = disabled)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Minimum year to keep",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Maximum year to keep",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="random",
        help="DWDS sort order (e.g., random)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between DWDS requests (seconds)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip lemmas already present in the output JSONL",
    )
    parser.add_argument(
        "--session-cookie",
        type=str,
        default=None,
        help="DWDS session cookie for accessing licensed corpora (ZEIT, FAZ, Spiegel). "
             "Log in at dwds.de, open DevTools → Application → Cookies, copy the session cookie value.",
    )
    parser.add_argument(
        "--paginate",
        action="store_true",
        default=False,
        help="Fetch all pages of results (use with deterministic sort, not --sort random).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    words = load_words(args.words_file)
    logger.info(f"Loaded {len(words)} target words from {args.words_file}")

    done_lemmas = set()
    if args.resume:
        done_lemmas = set(iter_existing_lemmas(args.output))
        if done_lemmas:
            logger.info(f"Resuming: {len(done_lemmas)} lemmas already present in output")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_mode = "a" if args.resume else "w"
    client = DWDSClient(delay=args.delay, session_cookie=args.session_cookie)
    options = FetchOptions(
        corpus=args.corpus,
        paginate=args.paginate,
        limit=args.limit,
        sort=args.sort,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_per_doc=args.max_per_doc,
        near_dup_threshold=args.near_dup_threshold,
        target_n=args.target_n,
        auto_select=args.auto_select,
        per_decade_n=args.per_decade_n,
        min_year=args.min_year,
        max_year=args.max_year,
        seed=args.seed,
    )

    summary = []
    with output_path.open(write_mode, encoding="utf-8") as out_f:
        for lemma in words:
            if lemma in done_lemmas:
                logger.info(f"Skipping {lemma} (already in output)")
                continue

            logger.info(f"Fetching '{lemma}'...")
            try:
                filtered_df = fetch_contexts_for_word(lemma, client, options)
            except Exception as e:
                logger.error(f"Error fetching '{lemma}': {e}")
                summary.append((lemma, 0, "fetch_error"))
                continue

            if len(filtered_df) == 0:
                logger.warning(f"No contexts kept after filtering for '{lemma}'")
                summary.append((lemma, 0, "filtered_out"))
                continue

            for _, row in filtered_df.iterrows():
                record = record_from_row(row, lemma, args.corpus)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary.append((lemma, len(filtered_df), "ok"))
            logger.info(f"  Saved {len(filtered_df)} contexts for '{lemma}'")

    if summary:
        ok_count = sum(1 for _, _, status in summary if status == "ok")
        total_contexts = sum(count for _, count, _ in summary)
        logger.info(
            f"Done. {ok_count}/{len(summary)} lemmas produced contexts "
            f"({total_contexts} total). Output: {output_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
