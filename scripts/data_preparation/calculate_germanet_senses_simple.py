#!/usr/bin/env python3
"""
Calculate GermaNet sense counts using the existing lemma_senses.csv data.
"""

import pandas as pd
from pathlib import Path
from collections import Counter

def calculate_germanet_senses():
    """Calculate sense counts from existing GermaNet data."""
    
    # Load the 750 words
    words_path = Path(__file__).parent.parent / 'data' / 'raw' / 'target_words' / 'word_list_750.txt'
    
    with open(words_path, 'r', encoding='utf-8') as f:
        target_words = set(word.strip() for word in f if word.strip())
    
    print(f"🔍 Calculating GermaNet senses for {len(target_words)} words...")
    
    # Load GermaNet lemma senses
    lemma_senses_path = Path(__file__).parent.parent / 'data' / 'processed' / 'germanet_base' / 'lexical_layer' / 'lemma_senses.csv'
    
    if not lemma_senses_path.exists():
        print(f"❌ GermaNet data not found: {lemma_senses_path}")
        return
        
    lemma_senses = pd.read_csv(lemma_senses_path)
    print(f"📚 Loaded {len(lemma_senses)} lemma-sense mappings from GermaNet")
    
    # Count synsets per lemma for our target words
    word_sense_counts = []
    
    for word in sorted(target_words):
        # Get all synsets this word appears in
        word_synsets = lemma_senses[lemma_senses['lemma'] == word]['synset_id'].unique()
        sense_count = len(word_synsets)
        
        word_sense_counts.append({
            'word': word,
            'num_senses': sense_count,
            'has_senses': sense_count > 0
        })
        
        if sense_count == 0:
            print(f"⚠️  '{word}' not found in GermaNet")
    
    # Convert to DataFrame
    df = pd.DataFrame(word_sense_counts)
    
    # Statistics
    words_with_senses = df['has_senses'].sum()
    words_without_senses = len(df) - words_with_senses
    
    print(f"\\n✅ Processing complete:")
    print(f"Words with senses: {words_with_senses}")
    print(f"Words without senses: {words_without_senses}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'target_words'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save words with senses only
    words_with_senses_df = df[df['has_senses']].copy()
    output_path = output_dir / 'target_words_750_with_germanet_senses.tsv'
    words_with_senses_df[['word', 'num_senses']].to_csv(output_path, sep='\\t', index=False)
    
    # Save full analysis
    full_output_path = output_dir / 'all_750_words_germanet_analysis.tsv'
    df.to_csv(full_output_path, sep='\\t', index=False)
    
    print(f"\\n📁 Results saved:")
    print(f"Words with senses ({len(words_with_senses_df)}): {output_path}")
    print(f"Full analysis ({len(df)}): {full_output_path}")
    
    if len(words_with_senses_df) > 0:
        # Show distribution
        sense_counts = words_with_senses_df['num_senses'].value_counts().sort_index()
        print(f"\\n📊 Sense count distribution:")
        for senses, count in sense_counts.head(15).items():
            print(f"  {senses} senses: {count} words")
        
        # Categorize by difficulty for selection
        easy_words = words_with_senses_df[words_with_senses_df['num_senses'] <= 2]
        medium_words = words_with_senses_df[(words_with_senses_df['num_senses'] >= 3) & (words_with_senses_df['num_senses'] <= 4)]
        hard_words = words_with_senses_df[words_with_senses_df['num_senses'] >= 5]
        
        print(f"\\n🎯 Words available for selection:")
        print(f"Easy (1-2 senses): {len(easy_words)} words")
        print(f"Medium (3-4 senses): {len(medium_words)} words")
        print(f"Hard (5+ senses): {len(hard_words)} words")
        
        # Show some examples
        if len(easy_words) > 0:
            print(f"\\nEasy examples: {list(easy_words['word'].head(5))}")
        if len(medium_words) > 0:
            print(f"Medium examples: {list(medium_words['word'].head(5))}")
        if len(hard_words) > 0:
            print(f"Hard examples: {list(hard_words['word'].head(5))}")
        
        # Check our previous selection against new sense counts
        previous_selection = ['klein', 'Autobahn', 'Vampir', 'dauern', 'Musik', 'merken', 'Leute', 'Geheimnis', 'Zeit', 'Einsicht']
        print(f"\\n🔍 Checking previous selection against new GermaNet counts:")
        
        for word in previous_selection:
            if word in words_with_senses_df['word'].values:
                senses = words_with_senses_df[words_with_senses_df['word'] == word]['num_senses'].iloc[0]
                if senses <= 2:
                    difficulty = "Easy"
                elif senses <= 4:
                    difficulty = "Medium"
                else:
                    difficulty = "Hard"
                print(f"  {word}: {senses} senses → {difficulty}")
            else:
                print(f"  {word}: ❌ Not found in GermaNet")
    
    return df

if __name__ == '__main__':
    calculate_germanet_senses()