#!/usr/bin/env python3
"""
Extract DWDS contexts with filtering for WSD embedding evaluation.
"""

import sys
import pandas as pd
import logging
import re
import argparse
import hashlib
import random
import string
from pathlib import Path
from typing import Set, List, Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.dwds.client import DWDSClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def clean_context(text: str) -> str:
    """Clean artifacts from context text."""
    if not isinstance(text, str):
        return ""
    
    # Remove only [...] and [sic] (case-insensitive)
    text = re.sub(r'\[\.\.\.\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[sic\]', ' ', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def target_matches_token(text: str, target_word: str) -> bool:
    """Check if target word occurs as standalone token(s) using regex."""
    # Case-insensitive, word boundary matching
    pattern = rf'(?<!\w){re.escape(target_word)}(?!\w)'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches) == 1

def normalize_for_comparison(text: str) -> str:
    """Normalize text for near-duplicate comparison."""
    # Lowercase, strip punctuation, collapse whitespace
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
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
    """Get document key for deduplication, extract author + title only."""
    if 'doc_id' in row and pd.notna(row['doc_id']):
        return str(row['doc_id'])
    
    bibl = str(row.get('Bibl', ''))
    
    # Extract author + title from Bibl format: "Author: Title, Publisher info"
    # Example: "Dückers, Tanja: Spielzone, Berlin: Aufbau-Taschenbuch-Verl. 2002 [1999], S. 22"
    if ':' in bibl:
        # Split at first colon to separate "Author: Title" from rest
        author_title = bibl.split(':', 1)[0] + ':' + bibl.split(':', 1)[1].split(',')[0]
        return author_title.strip()
    
    return bibl

def derive_seed(global_seed: int, lemma: str) -> int:
    """Derive per-lemma seed from global seed and lemma."""
    combined = f"{global_seed}_{lemma}"
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)

def filter_contexts(contexts_df: pd.DataFrame, target_word: str, args: argparse.Namespace) -> pd.DataFrame:
    """Apply filtering criteria to select best contexts."""
    
    if contexts_df.empty:
        return contexts_df
    
    logger.info(f"  Filtering {len(contexts_df)} contexts for '{target_word}'...")
    
    # 1. Clean contexts
    contexts_df = contexts_df.copy()
    contexts_df['Hit_clean'] = contexts_df['Hit'].apply(clean_context)
    
    # 2. Length filter
    contexts_df['token_count'] = contexts_df['Hit_clean'].apply(count_words)
    length_filtered = contexts_df[
        (contexts_df['token_count'] >= args.min_tokens) & 
        (contexts_df['token_count'] <= args.max_tokens)
    ].copy()
    
    logger.info(f"    After length filter ({args.min_tokens}-{args.max_tokens} tokens): {len(length_filtered)}")
    
    # 3. Target token matching
    length_filtered['target_matches'] = length_filtered['Hit_clean'].apply(
        lambda x: target_matches_token(x, target_word)
    )
    target_filtered = length_filtered[length_filtered['target_matches']].copy()
    
    logger.info(f"    After target token filter: {len(target_filtered)}")
    
    # 4. Document deduplication with max per doc
    target_filtered['chosen_doc_key'] = target_filtered.apply(get_doc_key, axis=1)
    
    if args.max_per_doc == 1:
        doc_filtered = target_filtered.drop_duplicates(subset=['chosen_doc_key']).copy()
    else:
        # Keep up to max_per_doc per document
        doc_filtered = target_filtered.groupby('chosen_doc_key').head(args.max_per_doc).copy()
    
    logger.info(f"    After doc filter (max {args.max_per_doc} per doc): {len(doc_filtered)}")
    
    # 5. Remove near duplicates
    if len(doc_filtered) > 1:
        unique_contexts = []
        texts_seen = []
        
        for _, row in doc_filtered.iterrows():
            text = row['Hit_clean']
            
            # Check for near duplicates
            is_dup = any(is_near_duplicate(text, seen) for seen in texts_seen)
            
            if not is_dup:
                unique_contexts.append(row)
                texts_seen.append(text)
        
        if unique_contexts:
            final_filtered = pd.DataFrame(unique_contexts)
        else:
            final_filtered = doc_filtered.head(1)
    else:
        final_filtered = doc_filtered
    
    logger.info(f"    After near-duplicate removal: {len(final_filtered)}")
    
    # 6. Select TARGET_N (optional automatic selection)
    if args.auto_select and len(final_filtered) > args.target_n:
        # Use per-lemma seed for deterministic shuffle
        lemma_seed = derive_seed(args.seed, target_word)
        random.seed(lemma_seed)
        
        # Convert to list, shuffle, take first N
        indices = list(final_filtered.index)
        random.shuffle(indices)
        selected_indices = indices[:args.target_n]
        final_filtered = final_filtered.loc[selected_indices].copy()
        
        logger.info(f"    Auto-selected {args.target_n} contexts via deterministic shuffle")
    elif not args.auto_select:
        logger.info(f"    Manual selection mode: keeping all {len(final_filtered)} filtered contexts")
    
    logger.info(f"  → {len(final_filtered)} contexts after filtering")
    
    # Return with desired columns (target_word first for better visibility)
    output_cols = ['target_word', 'No.', 'Date', 'Genre', 'Bibl', 'Hit', 'token_count', 'chosen_doc_key']
    available_cols = [col for col in output_cols if col in final_filtered.columns]
    
    return final_filtered[available_cols].reset_index(drop=True)

def extract_contexts(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    """Extract and filter contexts for each evaluation word."""
    
    # Load selected words
    words_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'target_words' / 'eval_set_selection.tsv'
    
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
    
    # Extract and filter contexts
    all_contexts = []
    
    for word in words:
        logger.info(f"Processing '{word}'...")
        
        # Determine extract count (special case for "sehen")
        extract_count = 400 if word == "sehen" else args.extract_n
        
        try:
            # Get larger set for filtering
            contexts_df = client.corpus_csv(word, limit=extract_count)
            
            if contexts_df is not None and len(contexts_df) > 0:
                # Add word information
                contexts_df['target_word'] = word
                
                # Apply filters
                filtered_df = filter_contexts(contexts_df, word, args)
                
                if len(filtered_df) > 0:
                    all_contexts.append(filtered_df)
                    logger.info(f"✅ Got {len(filtered_df)} contexts for '{word}'")
                else:
                    logger.warning(f"⚠️ No contexts passed filters for '{word}'")
            else:
                logger.warning(f"❌ No contexts found for '{word}'")
                
        except Exception as e:
            logger.error(f"❌ Error processing '{word}': {e}")
    
    if all_contexts:
        # Combine all contexts
        final_df = pd.concat(all_contexts, ignore_index=True)
        
        # Sort by target_word for better organization
        final_df = final_df.sort_values(['target_word', 'No.']).reset_index(drop=True)
        
        # Save results
        output_path = Path(__file__).parent.parent.parent / 'data' / 'evaluation' / 'eval_contexts.csv'
        output_path.parent.mkdir(exist_ok=True)
        
        try:
            final_df.to_csv(output_path, index=False)
            logger.info(f"✅ Saved {len(final_df)} filtered contexts to {output_path}")
            
            # Show summary
            summary = final_df.groupby('target_word').size()
            logger.info(f"Final summary:")
            for word, count in summary.items():
                logger.info(f"  {word}: {count} contexts")
            
            return final_df
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
            return None
    else:
        logger.error("❌ No contexts extracted for any words")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Extract DWDS contexts for WSD embedding evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--extract_n', type=int, default=200,
                      help='Number of contexts to extract per word before filtering')
    parser.add_argument('--target_n', type=int, default=10,
                      help='Target number of contexts per word after filtering')
    parser.add_argument('--min_tokens', type=int, default=8,
                      help='Minimum number of tokens in context')
    parser.add_argument('--max_tokens', type=int, default=35,
                      help='Maximum number of tokens in context')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for deterministic selection')
    parser.add_argument('--max_per_doc', type=int, default=1,
                      help='Maximum contexts per document')
    parser.add_argument('--auto_select', action='store_true', default=False,
                      help='Automatically select target_n contexts (default: manual selection)')
    
    args = parser.parse_args()
    
    # Print usage info
    logger.info("DWDS Context Extraction for WSD Evaluation")
    logger.info("=" * 45)
    logger.info(f"Extract {args.extract_n} contexts per word, filter to {args.target_n}")
    logger.info(f"Token length: {args.min_tokens}-{args.max_tokens}")
    logger.info(f"Max per document: {args.max_per_doc}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("")
    
    result = extract_contexts(args)
    
    if result is not None:
        logger.info("✅ Context extraction completed successfully")
        return 0
    else:
        logger.error("❌ Context extraction failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())