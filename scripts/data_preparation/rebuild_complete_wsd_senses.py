#!/usr/bin/env python3
"""
Rebuild complete wsd_senses.csv from full GermaNet data.
Creates text_for_embedding for all lemma-sense pairs.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List
import ast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_germanet_data():
    """Load full GermaNet data."""
    
    base_path = Path(__file__).parent.parent.parent
    germanet_base = base_path / 'data' / 'processed' / 'germanet_base'
    
    # Load lemma-sense mappings
    lemma_senses_path = germanet_base / 'lexical_layer' / 'lemma_senses.csv'
    synsets_path = germanet_base / 'synsets' / 'synsets.csv'
    
    logger.info("Loading GermaNet data...")
    lemma_senses_df = pd.read_csv(lemma_senses_path)
    synsets_df = pd.read_csv(synsets_path)
    
    logger.info(f"Loaded {len(lemma_senses_df)} lemma-sense pairs")
    logger.info(f"Loaded {len(synsets_df)} synsets")
    
    return lemma_senses_df, synsets_df

def create_text_for_embedding(row, synset_info):
    """Create text_for_embedding from lemma and synset information."""
    
    lemma = row.lemma
    synset_class = getattr(row, 'class', '')
    paraphrase = row.paraphrase
    
    # Start with lemma
    text_parts = [f"{lemma}"]
    
    # Add paraphrase if available
    if paraphrase and str(paraphrase) != 'nan' and paraphrase.strip():
        text_parts.append(f": {paraphrase}")
    
    # Add semantic class information
    if synset_class and str(synset_class) != 'nan':
        text_parts.append(f". {lemma} ist ein Begriff aus {synset_class}")
    
    return "".join(text_parts)

def create_complete_wsd_senses():
    """Create complete wsd_senses.csv with all GermaNet data."""
    
    # Load data
    lemma_senses_df, synsets_df = load_germanet_data()
    
    logger.info("Creating complete WSD senses...")
    
    # Merge lemma_senses with synset information
    merged_df = lemma_senses_df.merge(
        synsets_df[['synset_id', 'class']],
        on='synset_id',
        how='left'
    )
    
    logger.info(f"Merged data: {len(merged_df)} rows")
    
    # Create text_for_embedding for each row
    wsd_senses = []
    
    # Group by lemma to assign sense frequencies
    lemma_groups = merged_df.groupby('lemma')
    
    for lemma, group in lemma_groups:
        if len(wsd_senses) % 10000 == 0:
            logger.info(f"Processed {len(wsd_senses)} senses...")
            
        for sense_freq, (_, row) in enumerate(group.itertuples(), 1):
            # Create text_for_embedding
            text_for_embedding = create_text_for_embedding(row, None)
            
            wsd_sense = {
                'lemma': row.lemma,
                'pos': row.pos,
                'synset_id': row.synset_id,
                'class': getattr(row, 'class', ''),
                'paraphrase': row.paraphrase if pd.notna(row.paraphrase) else '',
                'text_for_embedding': text_for_embedding,
                'text_source': 'germanet_full',
                'sense_frequency': sense_freq,
                'embedding_word_count': len(text_for_embedding.split()),
                'has_examples': bool(row.examples and str(row.examples) != '[]')
            }
            
            wsd_senses.append(wsd_sense)
    
    logger.info(f"Created {len(wsd_senses)} complete WSD senses")
    
    return pd.DataFrame(wsd_senses)

def save_complete_wsd_senses(wsd_df):
    """Save the complete wsd_senses.csv file."""
    
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'data' / 'processed' / 'germanet' / 'wsd_senses_complete.csv'
    
    # Save complete version
    wsd_df.to_csv(output_path, index=False)
    logger.info(f"Saved complete WSD senses to: {output_path}")
    
    # Also backup original and replace
    original_path = base_path / 'data' / 'processed' / 'germanet' / 'wsd_senses.csv'
    backup_path = base_path / 'data' / 'processed' / 'germanet' / 'wsd_senses_original.csv'
    
    if original_path.exists():
        original_path.rename(backup_path)
        logger.info(f"Backed up original to: {backup_path}")
    
    # Copy complete version to main filename
    wsd_df.to_csv(original_path, index=False)
    logger.info(f"Replaced main file: {original_path}")
    
    # Print statistics
    logger.info("\n" + "="*50)
    logger.info("COMPLETE WSD SENSES STATISTICS")
    logger.info("="*50)
    logger.info(f"Total senses: {len(wsd_df):,}")
    logger.info(f"Unique lemmas: {wsd_df['lemma'].nunique():,}")
    logger.info(f"Average senses per lemma: {len(wsd_df)/wsd_df['lemma'].nunique():.1f}")
    
    pos_stats = wsd_df['pos'].value_counts()
    logger.info(f"\nBy POS:")
    for pos, count in pos_stats.items():
        logger.info(f"  {pos}: {count:,}")
    
    # Check our missing words
    logger.info(f"\nChecking previously missing words:")
    for word in ['rufen', 'sehen']:
        count = len(wsd_df[wsd_df['lemma'] == word])
        logger.info(f"  {word}: {count} senses")

def main():
    logger.info("Rebuilding complete wsd_senses.csv from full GermaNet")
    logger.info("=" * 55)
    
    try:
        # Create complete WSD senses
        wsd_df = create_complete_wsd_senses()
        
        # Save results
        save_complete_wsd_senses(wsd_df)
        
        logger.info("\n✅ Complete WSD senses rebuilt successfully!")
        logger.info("Now all GermaNet lemmas have proper text_for_embedding data")
        
    except Exception as e:
        logger.error(f"Error rebuilding WSD senses: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())