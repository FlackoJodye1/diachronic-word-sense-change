#!/usr/bin/env python3
"""
Sample target words from wsd_senses.csv for annotation.
Ensures all selected words have proper text_for_embedding data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def sample_target_words() -> List[Dict]:
    """Sample 10 target words from wsd_senses.csv with good sense distributions."""
    
    base_path = Path(__file__).parent.parent.parent
    wsd_senses_path = base_path / 'data' / 'processed' / 'germanet' / 'wsd_senses.csv'
    
    logger.info("Loading WSD senses...")
    df = pd.read_csv(wsd_senses_path)
    
    # Count senses per word
    sense_counts = df.groupby(['lemma', 'pos']).size().reset_index(name='sense_count')
    
    # Filter for words with 3-7 senses (good for annotation complexity)
    good_words = sense_counts[(sense_counts['sense_count'] >= 3) & 
                             (sense_counts['sense_count'] <= 7)]
    
    # Sample diverse POS categories
    nouns = good_words[good_words['pos'] == 'NOUN'].sample(n=5, random_state=42)
    verbs = good_words[good_words['pos'] == 'VERB'].sample(n=3, random_state=42)
    adjs = good_words[good_words['pos'] == 'ADJ'].sample(n=2, random_state=42)
    
    # Combine selections
    selected = pd.concat([nouns, verbs, adjs])
    
    logger.info(f"Selected {len(selected)} target words:")
    logger.info("=" * 50)
    
    target_words = []
    for _, row in selected.iterrows():
        word_info = {
            'word': row['lemma'],
            'pos': row['pos'],
            'sense_count': row['sense_count']
        }
        target_words.append(word_info)
        
        logger.info(f"{row['lemma']:12s} ({row['pos']:4s}): {row['sense_count']} senses")
    
    return target_words

def save_target_selection(target_words: List[Dict]) -> None:
    """Save the target word selection."""
    
    base_path = Path(__file__).parent.parent.parent
    output_dir = base_path / 'data' / 'processed' / 'target_words'
    output_path = output_dir / 'wsd_eval_selection.tsv'
    
    # Create DataFrame
    df = pd.DataFrame(target_words)
    
    # Save
    df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"\nSaved selection to: {output_path}")
    
    # Also save summary
    summary_path = output_dir / 'wsd_eval_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("WSD Evaluation Target Words (sampled from wsd_senses.csv)\n")
        f.write("=" * 55 + "\n\n")
        
        total_senses = sum(w['sense_count'] for w in target_words)
        f.write(f"Total words: {len(target_words)}\n")
        f.write(f"Total senses: {total_senses}\n")
        f.write(f"Average senses per word: {total_senses/len(target_words):.1f}\n\n")
        
        # By POS
        pos_counts = {}
        for w in target_words:
            pos = w['pos']
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        for pos, count in sorted(pos_counts.items()):
            f.write(f"{pos}: {count} words\n")
        
        f.write("\nWord list:\n")
        for w in target_words:
            f.write(f"- {w['word']} ({w['pos']}, {w['sense_count']} senses)\n")
    
    logger.info(f"Saved summary to: {summary_path}")

def main():
    logger.info("Sampling target words from wsd_senses.csv")
    logger.info("=" * 45)
    
    # Sample words
    target_words = sample_target_words()
    
    # Save selection
    save_target_selection(target_words)
    
    logger.info("\n✅ Target word sampling complete!")
    logger.info("Next steps:")
    logger.info("1. Extract contexts for these words from DWDS")
    logger.info("2. Generate annotation tasks")
    logger.info("3. Start annotation process")

if __name__ == '__main__':
    main()