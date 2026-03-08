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

import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.wsd import Embedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
GERMANET_DIR = DATA_DIR / "processed" / "germanet"
WSD_DATA_PATH = GERMANET_DIR / "wsd_senses_enhanced.csv"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_wsd_data() -> pd.DataFrame:
    """Load the enhanced WSD data."""
    logger.info("Loading WSD data...")
    df = pd.read_csv(WSD_DATA_PATH)
    logger.info(f"  Loaded {len(df)} senses for {df['lemma'].nunique()} words")
    return df


def text_quality_analysis(wsd_data: pd.DataFrame) -> None:
    """Analyze text quality metrics."""
    logger.info("=" * 50)
    logger.info("TEXT QUALITY ANALYSIS")
    logger.info("=" * 50)
    logger.info("Basic Statistics:")
    logger.info(f"  Total senses: {len(wsd_data)}")
    logger.info(f"  Unique words: {wsd_data['lemma'].nunique()}")

    if "text_source" in wsd_data.columns:
        source_counts = wsd_data["text_source"].value_counts()
        logger.info("Text sources:")
        for source, count in source_counts.items():
            pct = count / len(wsd_data) * 100
            logger.info(f"  {source}: {count} ({pct:.1f}%)")

    if "embedding_word_count" in wsd_data.columns:
        logger.info("Word count statistics:")
        if "text_source" in wsd_data.columns:
            by_source = wsd_data.groupby("text_source")["embedding_word_count"]
            for source, group in by_source:
                logger.info(f"  {source}: {group.mean():.1f} words avg (min: {group.min()}, max: {group.max()})")
        else:
            wc = wsd_data["embedding_word_count"]
            logger.info(f"  Overall: {wc.mean():.1f} words avg (min: {wc.min()}, max: {wc.max()})")

    word_senses = wsd_data.groupby("lemma").size()
    logger.info("Polysemy statistics:")
    logger.info(f"  Average senses per word: {word_senses.mean():.1f}")
    logger.info("  Most polysemous words:")
    for word, count in word_senses.nlargest(5).items():
        logger.info(f"    {word}: {count} senses")


def semantic_diversity_analysis(wsd_data: pd.DataFrame) -> None:
    """Analyze semantic diversity in the dataset."""
    logger.info("=" * 50)
    logger.info("SEMANTIC DIVERSITY ANALYSIS")
    logger.info("=" * 50)

    if "class" in wsd_data.columns:
        class_counts = wsd_data["class"].value_counts()
        logger.info(f"Semantic classes represented: {len(class_counts)}")
        logger.info("Top 10 classes:")
        for cls, count in class_counts.head(10).items():
            pct = count / len(wsd_data) * 100
            logger.info(f"  {cls}: {count} ({pct:.1f}%)")

    if "pos" in wsd_data.columns:
        pos_counts = wsd_data["pos"].value_counts()
        logger.info("Part-of-speech distribution:")
        for pos, count in pos_counts.items():
            pct = count / len(wsd_data) * 100
            logger.info(f"  {pos}: {count} ({pct:.1f}%)")


def embedding_similarity_analysis(
    wsd_data: pd.DataFrame,
    sample_size: int = 100,
    model_name: str = DEFAULT_MODEL,
) -> None:
    """Analyze embedding similarities."""
    logger.info("=" * 50)
    logger.info("EMBEDDING SIMILARITY ANALYSIS")
    logger.info("=" * 50)

    if len(wsd_data) > sample_size:
        sample_data = wsd_data.sample(n=sample_size, random_state=42)
        logger.info(f"Analyzing {sample_size} random samples...")
    else:
        sample_data = wsd_data
        logger.info(f"Analyzing all {len(sample_data)} senses...")

    logger.info(f"Loading model: {model_name}")
    embedder = Embedder(model_name=model_name, normalize=True)

    texts = sample_data["text_for_embedding"].fillna("").tolist()
    embeddings = embedder.encode(texts, show_progress=True)

    similarities = cosine_similarity(embeddings)

    upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
    logger.info("Similarity statistics (all pairs):")
    logger.info(f"  Mean: {upper_tri.mean():.3f}")
    logger.info(f"  Std:  {upper_tri.std():.3f}")
    logger.info(f"  Min:  {upper_tri.min():.3f}")
    logger.info(f"  Max:  {upper_tri.max():.3f}")

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
        logger.info("Same-word senses similarity:")
        logger.info(f"  Mean: {np.mean(same_word_sims):.3f} +/- {np.std(same_word_sims):.3f}")
        logger.info(f"  Count: {len(same_word_sims)} pairs")

    if diff_word_sims:
        logger.info("Different-word senses similarity:")
        logger.info(f"  Mean: {np.mean(diff_word_sims):.3f} +/- {np.std(diff_word_sims):.3f}")
        logger.info(f"  Count: {len(diff_word_sims)} pairs")

    if same_word_sims and diff_word_sims:
        diff = np.mean(same_word_sims) - np.mean(diff_word_sims)
        logger.info(f"Separation (same - different): {diff:.3f}")
        if diff > 0.1:
            logger.info("  Good: Same-word senses are more similar than different-word senses")
        elif diff > 0:
            logger.info("  Moderate: Slight tendency for same-word senses to be more similar")
        else:
            logger.warning("  Same-word senses not more similar than different-word senses")

    if len(sample_data) >= 10:
        logger.info("Clustering analysis...")
        n_clusters = min(10, len(sample_data) // 5)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        silhouette = silhouette_score(embeddings, cluster_labels)
        logger.info(f"  Silhouette score ({n_clusters} clusters): {silhouette:.3f}")

        cluster_words: dict = {}
        for idx, cluster in enumerate(cluster_labels):
            word = lemmas[idx]
            cluster_words.setdefault(cluster, []).append(word)

        logger.info("  Sample cluster compositions:")
        for cluster_id, words in list(cluster_words.items())[:3]:
            word_counts = Counter(words)
            top_words = [f"{w}({c})" for w, c in word_counts.most_common(3)]
            logger.info(f"    Cluster {cluster_id}: {', '.join(top_words)}")


def text_source_comparison(wsd_data: pd.DataFrame) -> None:
    """Compare example texts vs relation-based texts."""
    if "text_source" not in wsd_data.columns:
        return

    logger.info("=" * 50)
    logger.info("TEXT SOURCE COMPARISON")
    logger.info("=" * 50)

    examples = wsd_data[wsd_data["text_source"] == "example"]
    relations = wsd_data[wsd_data["text_source"] == "direct_relations"]

    if len(examples) > 0:
        logger.info(f"Example texts ({len(examples)}):")
        for i, (_, row) in enumerate(examples.head(3).iterrows(), 1):
            text = row["text_for_embedding"][:100] if pd.notna(row["text_for_embedding"]) else "(empty)"
            logger.info(f"  {i}. {row['lemma']}: {text}...")

    if len(relations) > 0:
        logger.info(f"Relation texts ({len(relations)}):")
        for i, (_, row) in enumerate(relations.head(3).iterrows(), 1):
            text = row["text_for_embedding"][:100] if pd.notna(row["text_for_embedding"]) else "(empty)"
            logger.info(f"  {i}. {row['lemma']}: {text}...")

    if len(examples) > 0 and len(relations) > 0 and "embedding_word_count" in wsd_data.columns:
        logger.info("Quality comparison:")
        logger.info(f"  Examples avg words: {examples['embedding_word_count'].mean():.1f}")
        logger.info(f"  Relations avg words: {relations['embedding_word_count'].mean():.1f}")


def sense_frequency_analysis(wsd_data: pd.DataFrame) -> None:
    """Analyze sense frequency distribution."""
    if "sense_frequency" not in wsd_data.columns:
        return

    logger.info("=" * 50)
    logger.info("SENSE FREQUENCY ANALYSIS")
    logger.info("=" * 50)

    freq_counts = wsd_data["sense_frequency"].value_counts().sort_index()
    logger.info("Sense frequency distribution:")
    for freq, count in freq_counts.head(10).items():
        pct = count / len(wsd_data) * 100
        logger.info(f"  Frequency {freq}: {count} senses ({pct:.1f}%)")

    logger.info("Most frequent senses (frequency=1):")
    freq_1 = wsd_data[wsd_data["sense_frequency"] == 1]
    for i, (_, row) in enumerate(freq_1.head(10).iterrows(), 1):
        cls = row.get("class", "?")
        logger.info(f"  {i}. {row['lemma']} ({cls})")


def main() -> None:
    """Run comprehensive embedding evaluation."""
    logger.info("EVALUATING GERMANET WSD EMBEDDING QUALITY")
    logger.info("=" * 50)

    wsd_data = load_wsd_data()

    text_quality_analysis(wsd_data)
    semantic_diversity_analysis(wsd_data)
    text_source_comparison(wsd_data)
    sense_frequency_analysis(wsd_data)
    embedding_similarity_analysis(wsd_data)

    logger.info("=" * 50)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 50)
    logger.info("For WSD accuracy evaluation, use: compare_embedding_methods.py")


if __name__ == "__main__":
    main()