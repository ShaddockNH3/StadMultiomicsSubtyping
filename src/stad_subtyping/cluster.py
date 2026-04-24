from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import CONSENSUS_FRACTION, CONSENSUS_REPS, K_RANGE, MIN_CLUSTER_SIZE
from .snf import spectral_cluster, silhouette_from_affinity


def pac_score(consensus: np.ndarray, lower: float = 0.1, upper: float = 0.9) -> float:
    upper_prop = np.mean((consensus > lower) & (consensus < upper))
    return float(upper_prop)


def consensus_clustering(
    fused_affinity: pd.DataFrame,
    k_values=K_RANGE,
    reps: int = CONSENSUS_REPS,
    sample_fraction: float = CONSENSUS_FRACTION,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    random_state: int = 0,
) -> tuple[pd.DataFrame, int, pd.Series]:
    rng = np.random.default_rng(random_state)
    samples = fused_affinity.index.to_numpy()
    n = len(samples)
    records: list[dict] = []
    final_labels: dict[int, pd.Series] = {}
    for k in k_values:
        consensus = np.zeros((n, n), dtype=np.float32)
        co_sampled = np.zeros((n, n), dtype=np.float32)
        draw_size = max(k + 1, int(round(n * sample_fraction)))
        for rep in range(reps):
            subset_idx = np.sort(rng.choice(n, size=draw_size, replace=False))
            subset_samples = samples[subset_idx]
            sub_affinity = fused_affinity.loc[subset_samples, subset_samples]
            labels = spectral_cluster(sub_affinity, n_clusters=k, random_state=random_state + rep)
            label_values = labels.to_numpy()
            co_sampled[np.ix_(subset_idx, subset_idx)] += 1
            for cluster_value in np.unique(label_values):
                members = subset_idx[label_values == cluster_value]
                consensus[np.ix_(members, members)] += 1
        consensus = np.divide(consensus, co_sampled, out=np.zeros_like(consensus), where=co_sampled > 0)
        pac = pac_score(consensus)
        labels = spectral_cluster(fused_affinity, n_clusters=k, random_state=random_state)
        cluster_sizes = labels.value_counts().sort_index()
        min_size = int(cluster_sizes.min())
        silhouette = silhouette_from_affinity(fused_affinity, labels)
        valid = min_size >= min_cluster_size
        records.append(
            {
                "k": k,
                "pac_score": pac,
                "silhouette": silhouette,
                "min_cluster_size": min_size,
                "is_valid": valid,
            }
        )
        final_labels[k] = labels

    summary = pd.DataFrame(records).sort_values(["is_valid", "pac_score", "silhouette"], ascending=[False, True, False])
    valid_summary = summary[summary["is_valid"]]
    if valid_summary.empty:
        best_row = summary.sort_values(["pac_score", "silhouette"], ascending=[True, False]).iloc[0]
    else:
        best_row = valid_summary.sort_values(["pac_score", "silhouette"], ascending=[True, False]).iloc[0]
    best_k = int(best_row["k"])
    labels = final_labels[best_k].map(lambda x: f"C{x + 1}")
    labels.name = "subtype"
    summary = summary.sort_values("k").reset_index(drop=True)
    summary["selected"] = summary["k"] == best_k
    return summary, best_k, labels


def ari_between(label_a: pd.Series, label_b: pd.Series) -> float:
    common = sorted(set(label_a.index) & set(label_b.index))
    if not common:
        return float("nan")
    return float(adjusted_rand_score(label_a.loc[common], label_b.loc[common]))


def _co_clustering_matrix(labels: pd.Series) -> np.ndarray:
    values = labels.to_numpy()
    return values[:, None] == values[None, :]


def co_clustering_jaccard(label_a: pd.Series, label_b: pd.Series) -> float:
    common = sorted(set(label_a.index) & set(label_b.index))
    if len(common) < 2:
        return float("nan")
    a = _co_clustering_matrix(label_a.loc[common])
    b = _co_clustering_matrix(label_b.loc[common])
    upper = np.triu_indices(len(common), k=1)
    a_pairs = a[upper]
    b_pairs = b[upper]
    union = np.logical_or(a_pairs, b_pairs).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(a_pairs, b_pairs).sum()
    return float(intersection / union)


def label_comparison_metrics(label_a: pd.Series, label_b: pd.Series) -> dict[str, float]:
    common = sorted(set(label_a.index) & set(label_b.index))
    if not common:
        return {
            "n_overlap": 0,
            "ari": float("nan"),
            "nmi": float("nan"),
            "co_clustering_jaccard": float("nan"),
        }
    a = label_a.loc[common]
    b = label_b.loc[common]
    return {
        "n_overlap": int(len(common)),
        "ari": float(adjusted_rand_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
        "co_clustering_jaccard": co_clustering_jaccard(a, b),
    }
