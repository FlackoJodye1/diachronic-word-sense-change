"""Coarse sense clustering for diachronic WSD.

Clusters fine-grained GermaNet senses per lemma using agglomerative
clustering (average linkage, cosine distance) on weighted E5-large embeddings.

Typical usage:
    senses_df = load_senses('data/processed/germanet/senses_pilot.jsonl')
    embedder = Embedder(model_name='intfloat/multilingual-e5-large', normalize=True)
    embeddings = embed_with_representation(senses_df, embedder, 'weighted', passage_prefix='passage: ')
    mapping = cluster_all_lemmas(senses_df, embeddings, threshold=0.18)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix.

    Embeddings must be L2-normalized (as produced by Embedder with normalize=True).
    Cosine distance = 1 - dot_product for L2-normalized vectors.
    """
    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def generate_coarse_cluster_id(lemma: str, cluster_index: int) -> str:
    """Generate a globally unique, human-readable coarse cluster ID.

    Example: generate_coarse_cluster_id('schlagen', 0) -> 'schlagen_c0'
    """
    return f"{lemma}_c{cluster_index}"


def cluster_senses_for_lemma(
    embeddings: np.ndarray,
    sense_ids: List[str],
    threshold: float,
    min_clusters: int = 2,
) -> Dict[str, int]:
    """Cluster senses for a single lemma.

    Args:
        embeddings: L2-normalized sense embeddings, shape (n_senses, dim)
        sense_ids: List of synset IDs, one per embedding row
        threshold: Cosine distance threshold for agglomerative clustering
        min_clusters: Minimum number of clusters (default 2). Enforced only
            when the threshold would collapse everything to 1 cluster — ensures
            every word gets at least one meaningful split. No upper bound is
            applied: the threshold alone determines how fine the clustering is,
            and sparse word-decade cells are filtered at analysis time.

    Returns:
        Dict mapping fine_sense_id -> cluster_index (0-based integer)
    """
    n = len(sense_ids)

    if n == 1:
        return {sense_ids[0]: 0}

    dist_matrix = cosine_distance_matrix(embeddings)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='precomputed',
        linkage='average',
    )
    labels = model.fit_predict(dist_matrix)
    n_clusters = len(set(labels))

    # Enforce minimum only — no upper bound
    if n_clusters < min_clusters <= n:
        model = AgglomerativeClustering(
            n_clusters=min_clusters,
            metric='precomputed',
            linkage='average',
        )
        labels = model.fit_predict(dist_matrix)

    return {sid: int(lbl) for sid, lbl in zip(sense_ids, labels)}


def cluster_all_lemmas(
    senses_df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    min_clusters: int = 2,
) -> Dict:
    """Cluster senses for all lemmas and return the full mapping structure.

    Args:
        senses_df: DataFrame with columns ['lemma', 'synset_id', ...]
        embeddings: L2-normalized embeddings aligned row-for-row with senses_df
        threshold: Cosine distance threshold
        min_clusters: Minimum clusters per lemma

    Returns:
        Nested dict with structure:
        {
            lemma: {
                "mapping": {fine_sense_id: coarse_cluster_id, ...},
                "clusters": {
                    coarse_cluster_id: {"label": None, "members": [sense_ids]},
                    ...
                }
            }
        }
    """
    result = {}

    for lemma, group in senses_df.groupby('lemma'):
        idx = group.index.tolist()
        local_embs = embeddings[idx]
        sense_ids = group['synset_id'].tolist()

        raw_mapping = cluster_senses_for_lemma(
            local_embs, sense_ids, threshold,
            min_clusters=min_clusters,
        )

        # Build flat mapping (fine_id -> coarse_cluster_id)
        mapping = {
            sid: generate_coarse_cluster_id(lemma, cid)
            for sid, cid in raw_mapping.items()
        }

        # Build cluster registry (coarse_cluster_id -> {label, members})
        clusters = {}
        for sid, coarse_id in mapping.items():
            if coarse_id not in clusters:
                clusters[coarse_id] = {"label": None, "members": []}
            clusters[coarse_id]["members"].append(sid)

        result[lemma] = {"mapping": mapping, "clusters": clusters}

    return result


def compute_cluster_summary(
    mapping_dict: Dict,
    senses_df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Compute per-lemma clustering statistics.

    Args:
        mapping_dict: Output of cluster_all_lemmas()
        senses_df: Senses DataFrame aligned with embeddings
        embeddings: L2-normalized embeddings aligned with senses_df
        threshold: Threshold used (stored in output for reference)

    Returns:
        DataFrame with columns:
            lemma, threshold, n_fine_senses, n_organic_clusters, n_coarse_clusters,
            constrained, n_singleton_clusters, reduction_ratio,
            mean_intra_cluster_similarity, min_intra_cluster_similarity,
            max_cluster_size

        constrained=True means the min_clusters constraint overrode the organic
        threshold result (threshold collapsed everything to 1 cluster).
        n_organic_clusters is what the algorithm produced before any constraint.
    """
    rows = []

    for lemma, data in mapping_dict.items():
        mapping = data['mapping']
        clusters = data['clusters']

        group_idx = senses_df[senses_df['lemma'] == lemma].index.tolist()
        group_sids = senses_df.loc[group_idx, 'synset_id'].tolist()
        local_embs = embeddings[group_idx]

        n_fine = len(mapping)
        n_coarse = len(clusters)
        n_singleton = sum(1 for c in clusters.values() if len(c['members']) == 1)
        max_size = max(len(c['members']) for c in clusters.values())
        reduction = round(1.0 - n_coarse / n_fine, 3) if n_fine > 0 else 0.0

        # Compute organic (unconstrained) cluster count to detect when min/max kicked in
        n_organic = n_fine  # default for n==1 edge case
        if n_fine > 1:
            dist_matrix = cosine_distance_matrix(local_embs)
            raw_model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='precomputed',
                linkage='average',
            )
            n_organic = len(set(raw_model.fit_predict(dist_matrix)))

        mean_sim, min_sim = _intra_cluster_similarities(local_embs, group_sids, mapping)

        rows.append({
            'lemma': lemma,
            'threshold': threshold,
            'n_fine_senses': n_fine,
            'n_organic_clusters': n_organic,
            'n_coarse_clusters': n_coarse,
            'constrained': n_organic != n_coarse,
            'n_singleton_clusters': n_singleton,
            'reduction_ratio': reduction,
            'mean_intra_cluster_similarity': round(mean_sim, 3) if not np.isnan(mean_sim) else None,
            'min_intra_cluster_similarity': round(min_sim, 3) if not np.isnan(min_sim) else None,
            'max_cluster_size': max_size,
        })

    return pd.DataFrame(rows).sort_values('lemma').reset_index(drop=True)


def get_cluster_centroids(
    embeddings: np.ndarray,
    mapping: Dict[str, str],
    sense_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Compute cluster centroid embeddings (for optional direct coarse prediction).

    Args:
        embeddings: L2-normalized sense embeddings aligned with sense_ids
        mapping: flat fine_sense_id -> coarse_cluster_id dict
        sense_ids: List of sense IDs corresponding to embedding rows

    Returns:
        Tuple of (centroid_embeddings, coarse_cluster_ids) both aligned by cluster
    """
    clusters: Dict[str, List[int]] = {}
    for i, sid in enumerate(sense_ids):
        coarse_id = mapping[sid]
        clusters.setdefault(coarse_id, []).append(i)

    coarse_ids = sorted(clusters.keys())
    centroids = []
    for coarse_id in coarse_ids:
        member_embs = embeddings[clusters[coarse_id]]
        centroid = member_embs.mean(axis=0)
        # Re-normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    return np.array(centroids), coarse_ids


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _intra_cluster_similarities(
    embeddings: np.ndarray,
    sense_ids: List[str],
    mapping: Dict[str, str],
) -> Tuple[float, float]:
    """Compute mean and min pairwise cosine similarity within clusters."""
    sim = embeddings @ embeddings.T
    clusters: Dict[str, List[int]] = {}
    for i, sid in enumerate(sense_ids):
        coarse_id = mapping[sid]
        clusters.setdefault(coarse_id, []).append(i)

    sims = []
    for indices in clusters.values():
        if len(indices) < 2:
            continue
        for ii, i in enumerate(indices):
            for j in indices[ii + 1:]:
                sims.append(float(sim[i, j]))

    if not sims:
        return float('nan'), float('nan')
    return float(np.mean(sims)), float(np.min(sims))
