#!/usr/bin/env python3
"""
Simple evaluation word selection from 750-word dataset.
"""

import pandas as pd
import random
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def select_evaluation_words():
    """Select 10 words with 3-4-3 distribution."""
    
    # Load data  
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'target_words' / 'target_words_with_germanet_sense_counts.tsv'
    df = pd.read_csv(data_path, sep='\t')
    logger.info(f"Loaded {len(df)} words")
    
    # Categorize by difficulty
    easy = df[df['num_senses'] <= 2]
    medium = df[(df['num_senses'] >= 3) & (df['num_senses'] <= 4)]
    hard = df[df['num_senses'] >= 5]
    
    logger.info(f"Available: {len(easy)} easy, {len(medium)} medium, {len(hard)} hard")
    
    # Random selection with seed for reproducibility
    random.seed(1050)
    
    # Select 3-4-3
    selected = {
        'easy': easy.sample(3, random_state=42)['word'].tolist(),
        'medium': medium.sample(4, random_state=43)['word'].tolist(), 
        'hard': hard.sample(3, random_state=44)['word'].tolist()
    }
    
    # Results
    all_words = []
    for difficulty, words in selected.items():
        logger.info(f"{difficulty.capitalize()}: {words}")
        all_words.extend(words)
    
    # Save
    final_df = df[df['word'].isin(all_words)].copy()
    output_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'target_words' / 'eval_selection.tsv'
    final_df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Saved to {output_path.name}")
    
    return selected

if __name__ == '__main__':
    select_evaluation_words()