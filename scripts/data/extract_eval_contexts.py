#!/usr/bin/env python3
"""
Extract DWDS contexts with filtering for WSD embedding evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dwds.client import DWDSClient
from src.data.dwds.fetcher import FetchOptions, fetch_contexts_for_word

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

def extract_contexts(args: argparse.Namespace) -> pd.DataFrame | None:
    """Extract and filter contexts for each evaluation word."""
    
    # Load selected words
    words_path = BASE_DIR / 'data' / 'processed' / 'target_words' / 'eval_set_selection.tsv'
    
    if not words_path.exists():
        logger.error(f"Word selection file not found: {words_path}")
        return None
        
    try:
        words_df = pd.read_csv(words_path, sep='\t')
        words = words_df['word'].tolist()
    except Exception as e:
        logger.error(f"Error loading word selection: {e}")
        return None
    
    logger.info(f"Extracting contexts for {len(words)} words: {words}")
    logger.info(f"Settings: extract_n={args.extract_n}, target_n={args.target_n}, seed={args.seed}")
    
    # Initialize DWDS client
    client = DWDSClient(delay=1.0)
    options = FetchOptions(
        limit=args.extract_n,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_per_doc=args.max_per_doc,
        target_n=args.target_n,
        auto_select=args.auto_select,
        seed=args.seed,
    )
    
    # Extract and filter contexts
    all_contexts = []
    
    for word in words:
        logger.info(f"Processing '{word}'...")
        
        # Determine extract count (special case for "sehen")
        extract_count = 400 if word == "sehen" else args.extract_n
        
        try:
            filtered_df = fetch_contexts_for_word(
                word,
                client,
                options,
                limit_override=extract_count,
            )
            if len(filtered_df) > 0:
                filtered_df["target_word"] = word
                output_cols = [
                    "target_word",
                    "No.",
                    "Date",
                    "Genre",
                    "Bibl",
                    "Hit",
                    "token_count",
                    "chosen_doc_key",
                ]
                available_cols = [col for col in output_cols if col in filtered_df.columns]
                all_contexts.append(filtered_df[available_cols].reset_index(drop=True))
                logger.info(f"Got {len(filtered_df)} contexts for '{word}'")
            else:
                logger.warning(f"No contexts passed filters for '{word}'")
                
        except Exception as e:
            logger.error(f"Error processing '{word}': {e}")
    
    if all_contexts:
        # Combine all contexts
        final_df = pd.concat(all_contexts, ignore_index=True)
        
        # Sort by target_word for better organization
        final_df = final_df.sort_values(['target_word', 'No.']).reset_index(drop=True)
        
        # Save results
        output_path = BASE_DIR / 'data' / 'evaluation' / 'eval_contexts.csv'
        output_path.parent.mkdir(exist_ok=True)
        
        try:
            final_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(final_df)} filtered contexts to {output_path}")
            
            # Show summary
            summary = final_df.groupby('target_word').size()
            logger.info(f"Final summary:")
            for word, count in summary.items():
                logger.info(f"  {word}: {count} contexts")
            
            return final_df
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    else:
        logger.error("No contexts extracted for any words")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Extract DWDS contexts for WSD embedding evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--extract-n', type=int, default=200,
                      help='Number of contexts to extract per word before filtering')
    parser.add_argument('--target-n', type=int, default=10,
                      help='Target number of contexts per word after filtering')
    parser.add_argument('--min-tokens', type=int, default=8,
                      help='Minimum number of tokens in context')
    parser.add_argument('--max-tokens', type=int, default=35,
                      help='Maximum number of tokens in context')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for deterministic selection')
    parser.add_argument('--max-per-doc', type=int, default=1,
                      help='Maximum contexts per document')
    parser.add_argument('--auto-select', action='store_true', default=False,
                      help='Automatically select target_n contexts (default: manual selection)')
    
    args = parser.parse_args()
    
    # Print usage info
    logger.info("DWDS Context Extraction for WSD Evaluation")
    logger.info(f"Extract {args.extract_n} contexts per word, filter to {args.target_n}")
    logger.info(f"Token length: {args.min_tokens}-{args.max_tokens}")
    logger.info(f"Max per document: {args.max_per_doc}")
    logger.info(f"Random seed: {args.seed}")
    
    result = extract_contexts(args)
    
    if result is not None:
        logger.info("Context extraction completed successfully")
        return 0
    else:
        logger.error("Context extraction failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
