#!/usr/bin/env python3
"""
Evaluate the quality of GermaNet embedding texts for German WSD.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: sentence-transformers not available. Install with: pip install sentence-transformers")

# Paths
DATA_DIR = Path("data")
GERMANET_DIR = DATA_DIR / "processed" / "germanet"

def load_wsd_data():
    """Load the enhanced WSD data"""
    print("Loading WSD data...")
    wsd_data = pd.read_csv(GERMANET_DIR / "wsd_senses_enhanced.csv")
    print(f"Loaded {len(wsd_data)} senses for {wsd_data['lemma'].nunique()} words")
    return wsd_data

def text_quality_analysis(wsd_data):
    """Analyze text quality metrics"""
    print("\n=== TEXT QUALITY ANALYSIS ===")
    
    # Basic statistics
    print(f"Total senses: {len(wsd_data)}")
    print(f"Unique words: {wsd_data['lemma'].nunique()}")
    
    # Text source breakdown
    source_counts = wsd_data['text_source'].value_counts()
    print(f"\nText sources:")
    for source, count in source_counts.items():
        pct = count/len(wsd_data)*100
        print(f"  {source}: {count} ({pct:.1f}%)")
    
    # Word count analysis
    print(f"\nWord count statistics:")
    by_source = wsd_data.groupby('text_source')['embedding_word_count']
    for source, group in by_source:
        print(f"  {source}: {group.mean():.1f} words avg (min: {group.min()}, max: {group.max()})")
    
    # Polysemy analysis
    word_senses = wsd_data.groupby('lemma').size()
    print(f"\nPolysemy statistics:")
    print(f"  Average senses per word: {word_senses.mean():.1f}")
    print(f"  Most polysemous words:")
    for word, count in word_senses.nlargest(5).items():
        print(f"    {word}: {count} senses")

def semantic_diversity_analysis(wsd_data):
    """Analyze semantic diversity in our dataset"""
    print("\n=== SEMANTIC DIVERSITY ANALYSIS ===")
    
    # Class distribution
    class_counts = wsd_data['class'].value_counts()
    print(f"Semantic classes represented: {len(class_counts)}")
    print(f"Top 10 classes:")
    for cls, count in class_counts.head(10).items():
        pct = count/len(wsd_data)*100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    # POS distribution
    pos_counts = wsd_data['pos'].value_counts()
    print(f"\nPart-of-speech distribution:")
    for pos, count in pos_counts.items():
        pct = count/len(wsd_data)*100
        print(f"  {pos}: {count} ({pct:.1f}%)")

def embedding_similarity_analysis(wsd_data, sample_size=100):
    """Analyze embedding similarities if transformers available"""
    if not TRANSFORMERS_AVAILABLE:
        print("\n=== EMBEDDING ANALYSIS SKIPPED ===")
        print("Install sentence-transformers to run embedding analysis")
        return
    
    print(f"\n=== EMBEDDING SIMILARITY ANALYSIS ===")
    print("Loading multilingual BERT model...")
    
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    
    # Sample data for analysis
    if len(wsd_data) > sample_size:
        sample_data = wsd_data.sample(n=sample_size, random_state=42)
        print(f"Analyzing {sample_size} random samples...")
    else:
        sample_data = wsd_data
        print(f"Analyzing all {len(sample_data)} senses...")
    
    # Get embeddings
    texts = sample_data['text_for_embedding'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Calculate similarities
    similarities = cosine_similarity(embeddings)
    
    # Overall similarity distribution
    upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
    print(f"\nSimilarity statistics:")
    print(f"  Mean similarity: {upper_tri.mean():.3f}")
    print(f"  Std similarity: {upper_tri.std():.3f}")
    print(f"  Min similarity: {upper_tri.min():.3f}")
    print(f"  Max similarity: {upper_tri.max():.3f}")
    
    # Same-word vs different-word similarities
    same_word_sims = []
    diff_word_sims = []
    
    for i in range(len(sample_data)):
        for j in range(i+1, len(sample_data)):
            sim = similarities[i, j]
            if sample_data.iloc[i]['lemma'] == sample_data.iloc[j]['lemma']:
                same_word_sims.append(sim)
            else:
                diff_word_sims.append(sim)
    
    if same_word_sims:
        print(f"\nSame word senses similarity: {np.mean(same_word_sims):.3f} ± {np.std(same_word_sims):.3f}")
    if diff_word_sims:
        print(f"Different word senses similarity: {np.mean(diff_word_sims):.3f} ± {np.std(diff_word_sims):.3f}")
    
    # Clustering analysis
    if len(sample_data) >= 10:
        print(f"\nClustering analysis...")
        n_clusters = min(10, len(sample_data) // 5)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"  Silhouette score with {n_clusters} clusters: {silhouette_avg:.3f}")
        
        # Show cluster composition
        cluster_words = {}
        for idx, cluster in enumerate(cluster_labels):
            word = sample_data.iloc[idx]['lemma']
            if cluster not in cluster_words:
                cluster_words[cluster] = []
            cluster_words[cluster].append(word)
        
        print(f"  Sample cluster compositions:")
        for cluster_id, words in list(cluster_words.items())[:3]:
            word_counts = Counter(words)
            top_words = [f"{w}({c})" for w, c in word_counts.most_common(3)]
            print(f"    Cluster {cluster_id}: {', '.join(top_words)}")

def text_source_comparison(wsd_data):
    """Compare example texts vs relation-based texts"""
    print("\n=== TEXT SOURCE COMPARISON ===")
    
    examples = wsd_data[wsd_data['text_source'] == 'example']
    relations = wsd_data[wsd_data['text_source'] == 'direct_relations']
    
    if len(examples) > 0 and len(relations) > 0:
        print(f"Example texts ({len(examples)}):")
        for i, (_, row) in enumerate(examples.head(3).iterrows(), 1):
            print(f"  {i}. {row['lemma']}: {row['text_for_embedding'][:100]}...")
        
        print(f"\nRelation texts ({len(relations)}):")
        for i, (_, row) in enumerate(relations.head(3).iterrows(), 1):
            print(f"  {i}. {row['lemma']}: {row['text_for_embedding'][:100]}...")
        
        print(f"\nQuality comparison:")
        print(f"  Examples avg words: {examples['embedding_word_count'].mean():.1f}")
        print(f"  Relations avg words: {relations['embedding_word_count'].mean():.1f}")

def sense_frequency_analysis(wsd_data):
    """Analyze sense frequency distribution"""
    print("\n=== SENSE FREQUENCY ANALYSIS ===")
    
    freq_counts = wsd_data['sense_frequency'].value_counts().sort_index()
    print(f"Sense frequency distribution:")
    for freq, count in freq_counts.head(10).items():
        pct = count/len(wsd_data)*100
        print(f"  Frequency {freq}: {count} senses ({pct:.1f}%)")
    
    # Most frequent senses
    print(f"\nMost frequent senses (frequency=1):")
    freq_1 = wsd_data[wsd_data['sense_frequency'] == 1]
    for i, (_, row) in enumerate(freq_1.head(10).iterrows(), 1):
        print(f"  {i}. {row['lemma']} ({row['class']})")

def main():
    """Run comprehensive embedding evaluation"""
    print("🔍 EVALUATING GERMANET WSD EMBEDDING QUALITY")
    print("=" * 50)
    
    # Load data
    wsd_data = load_wsd_data()
    
    # Run analyses
    text_quality_analysis(wsd_data)
    semantic_diversity_analysis(wsd_data)
    text_source_comparison(wsd_data)
    sense_frequency_analysis(wsd_data)
    embedding_similarity_analysis(wsd_data)
    
    print("\n" + "=" * 50)
    print("✅ EVALUATION COMPLETE")
    
    if TRANSFORMERS_AVAILABLE:
        print("\n📊 For deeper analysis, consider:")
        print("  - Intrinsic evaluation: word similarity benchmarks")
        print("  - Extrinsic evaluation: downstream WSD task performance")
        print("  - Human evaluation: semantic similarity judgments")
    else:
        print("\n💡 Install sentence-transformers for embedding similarity analysis:")
        print("  pip install sentence-transformers")

if __name__ == "__main__":
    main()