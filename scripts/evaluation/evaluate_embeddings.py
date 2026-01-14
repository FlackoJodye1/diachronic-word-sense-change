#!/usr/bin/env python3
"""
Evaluate the quality of GermaNet embedding texts for German WSD.

This script performs exploratory analysis on sense embedding quality:
- Text quality metrics (word counts, sources)
- Semantic diversity analysis
- Embedding similarity analysis (same-word vs different-word senses)
- Clustering analysis

Unlike compare_embedding_methods.py, this does NOT evaluate WSD accuracy.
It provides insights into the quality and characteristics of the sense texts.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation import Embedder

# Configuration
DATA_DIR = Path("data")
GERMANET_DIR = DATA_DIR / "processed" / "germanet"
WSD_DATA_PATH = GERMANET_DIR / "wsd_senses_enhanced.csv"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_wsd_data() -> pd.DataFrame:
    """Load the enhanced WSD data."""
    print("Loading WSD data...")
    df = pd.read_csv(WSD_DATA_PATH)
    print(f"  Loaded {len(df)} senses for {df['lemma'].nunique()} words")
    return df


def text_quality_analysis(wsd_data: pd.DataFrame) -> None:
    """Analyze text quality metrics."""
    print("\n" + "=" * 50)
    print("TEXT QUALITY ANALYSIS")
    print("=" * 50)

    print(f"\nBasic Statistics:")
    print(f"  Total senses: {len(wsd_data)}")
    print(f"  Unique words: {wsd_data['lemma'].nunique()}")

    # Text source breakdown
    if "text_source" in wsd_data.columns:
        source_counts = wsd_data["text_source"].value_counts()
        print(f"\nText sources:")
        for source, count in source_counts.items():
            pct = count / len(wsd_data) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")

    # Word count analysis
    if "embedding_word_count" in wsd_data.columns:
        print(f"\nWord count statistics:")
        if "text_source" in wsd_data.columns:
            by_source = wsd_data.groupby("text_source")["embedding_word_count"]
            for source, group in by_source:
                print(f"  {source}: {group.mean():.1f} words avg (min: {group.min()}, max: {group.max()})")
        else:
            wc = wsd_data["embedding_word_count"]
            print(f"  Overall: {wc.mean():.1f} words avg (min: {wc.min()}, max: {wc.max()})")

    # Polysemy analysis
    word_senses = wsd_data.groupby("lemma").size()
    print(f"\nPolysemy statistics:")
    print(f"  Average senses per word: {word_senses.mean():.1f}")
    print(f"  Most polysemous words:")
    for word, count in word_senses.nlargest(5).items():
        print(f"    {word}: {count} senses")


def semantic_diversity_analysis(wsd_data: pd.DataFrame) -> None:
    """Analyze semantic diversity in the dataset."""
    print("\n" + "=" * 50)
    print("SEMANTIC DIVERSITY ANALYSIS")
    print("=" * 50)

    # Class distribution
    if "class" in wsd_data.columns:
        class_counts = wsd_data["class"].value_counts()
        print(f"\nSemantic classes represented: {len(class_counts)}")
        print(f"Top 10 classes:")
        for cls, count in class_counts.head(10).items():
            pct = count / len(wsd_data) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")

    # POS distribution
    if "pos" in wsd_data.columns:
        pos_counts = wsd_data["pos"].value_counts()
        print(f"\nPart-of-speech distribution:")
        for pos, count in pos_counts.items():
            pct = count / len(wsd_data) * 100
            print(f"  {pos}: {count} ({pct:.1f}%)")


def embedding_similarity_analysis(
    wsd_data: pd.DataFrame,
    sample_size: int = 100,
    model_name: str = DEFAULT_MODEL,
) -> None:
    """Analyze embedding similarities."""
    print("\n" + "=" * 50)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("=" * 50)

    # Sample data if needed
    if len(wsd_data) > sample_size:
        sample_data = wsd_data.sample(n=sample_size, random_state=42)
        print(f"\nAnalyzing {sample_size} random samples...")
    else:
        sample_data = wsd_data
        print(f"\nAnalyzing all {len(sample_data)} senses...")

    # Get embeddings using shared Embedder
    print(f"Loading model: {model_name}")
    embedder = Embedder(model_name=model_name, normalize=True)

    texts = sample_data["text_for_embedding"].fillna("").tolist()
    embeddings = embedder.encode(texts, show_progress=True)

    # Calculate similarities
    similarities = cosine_similarity(embeddings)

    # Overall similarity distribution
    upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
    print(f"\nSimilarity statistics (all pairs):")
    print(f"  Mean: {upper_tri.mean():.3f}")
    print(f"  Std:  {upper_tri.std():.3f}")
    print(f"  Min:  {upper_tri.min():.3f}")
    print(f"  Max:  {upper_tri.max():.3f}")

    # Same-word vs different-word similarities
    same_word_sims = []
    diff_word_sims = []

    lemmas = sample_data["lemma"].tolist()
    for i in range(len(sample_data)):
        for j in range(i + 1, len(sample_data)):
            sim = similarities[i, j]
            if lemmas[i] == lemmas[j]:
                same_word_sims.append(sim)
            else:
                diff_word_sims.append(sim)

    if same_word_sims:
        print(f"\nSame-word senses similarity:")
        print(f"  Mean: {np.mean(same_word_sims):.3f} +/- {np.std(same_word_sims):.3f}")
        print(f"  Count: {len(same_word_sims)} pairs")

    if diff_word_sims:
        print(f"\nDifferent-word senses similarity:")
        print(f"  Mean: {np.mean(diff_word_sims):.3f} +/- {np.std(diff_word_sims):.3f}")
        print(f"  Count: {len(diff_word_sims)} pairs")

    if same_word_sims and diff_word_sims:
        diff = np.mean(same_word_sims) - np.mean(diff_word_sims)
        print(f"\nSeparation (same - different): {diff:.3f}")
        if diff > 0.1:
            print("  Good: Same-word senses are more similar than different-word senses")
        elif diff > 0:
            print("  Moderate: Slight tendency for same-word senses to be more similar")
        else:
            print("  Warning: Same-word senses not more similar than different-word senses")

    # Clustering analysis
    if len(sample_data) >= 10:
        print(f"\nClustering analysis...")
        n_clusters = min(10, len(sample_data) // 5)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        silhouette = silhouette_score(embeddings, cluster_labels)
        print(f"  Silhouette score ({n_clusters} clusters): {silhouette:.3f}")

        # Show cluster composition
        cluster_words: dict = {}
        for idx, cluster in enumerate(cluster_labels):
            word = lemmas[idx]
            cluster_words.setdefault(cluster, []).append(word)

        print(f"  Sample cluster compositions:")
        for cluster_id, words in list(cluster_words.items())[:3]:
            word_counts = Counter(words)
            top_words = [f"{w}({c})" for w, c in word_counts.most_common(3)]
            print(f"    Cluster {cluster_id}: {', '.join(top_words)}")


def text_source_comparison(wsd_data: pd.DataFrame) -> None:
    """Compare example texts vs relation-based texts."""
    if "text_source" not in wsd_data.columns:
        return

    print("\n" + "=" * 50)
    print("TEXT SOURCE COMPARISON")
    print("=" * 50)

    examples = wsd_data[wsd_data["text_source"] == "example"]
    relations = wsd_data[wsd_data["text_source"] == "direct_relations"]

    if len(examples) > 0:
        print(f"\nExample texts ({len(examples)}):")
        for i, (_, row) in enumerate(examples.head(3).iterrows(), 1):
            text = row["text_for_embedding"][:100] if pd.notna(row["text_for_embedding"]) else "(empty)"
            print(f"  {i}. {row['lemma']}: {text}...")

    if len(relations) > 0:
        print(f"\nRelation texts ({len(relations)}):")
        for i, (_, row) in enumerate(relations.head(3).iterrows(), 1):
            text = row["text_for_embedding"][:100] if pd.notna(row["text_for_embedding"]) else "(empty)"
            print(f"  {i}. {row['lemma']}: {text}...")

    if len(examples) > 0 and len(relations) > 0 and "embedding_word_count" in wsd_data.columns:
        print(f"\nQuality comparison:")
        print(f"  Examples avg words: {examples['embedding_word_count'].mean():.1f}")
        print(f"  Relations avg words: {relations['embedding_word_count'].mean():.1f}")


def sense_frequency_analysis(wsd_data: pd.DataFrame) -> None:
    """Analyze sense frequency distribution."""
    if "sense_frequency" not in wsd_data.columns:
        return

    print("\n" + "=" * 50)
    print("SENSE FREQUENCY ANALYSIS")
    print("=" * 50)

    freq_counts = wsd_data["sense_frequency"].value_counts().sort_index()
    print(f"\nSense frequency distribution:")
    for freq, count in freq_counts.head(10).items():
        pct = count / len(wsd_data) * 100
        print(f"  Frequency {freq}: {count} senses ({pct:.1f}%)")

    # Most frequent senses
    print(f"\nMost frequent senses (frequency=1):")
    freq_1 = wsd_data[wsd_data["sense_frequency"] == 1]
    for i, (_, row) in enumerate(freq_1.head(10).iterrows(), 1):
        cls = row.get("class", "?")
        print(f"  {i}. {row['lemma']} ({cls})")


def main() -> None:
    """Run comprehensive embedding evaluation."""
    print("EVALUATING GERMANET WSD EMBEDDING QUALITY")
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
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print("\nFor WSD accuracy evaluation, use: compare_embedding_methods.py")


if __name__ == "__main__":
    main()