from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score

from .config import CONSENSUS_FRACTION, CONSENSUS_REPS, K_RANGE, MIN_CLUSTER_SIZE

EARLY_INTEGRATION_COMPONENT_CAP = 30
NMF_COMPONENT_CAP = 20


def _concat_modality_embeddings(modality_embeddings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not modality_embeddings:
        raise ValueError("modality_embeddings must not be empty")
    merged = pd.concat([frame.add_prefix(f"{name}__") for name, frame in modality_embeddings.items()], axis=1)
    return merged.sort_index().astype(np.float32)


def _zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    scaled = (frame - frame.mean(axis=0)) / frame.std(axis=0).replace(0, 1.0).fillna(1.0)
    return scaled.fillna(0.0).astype(np.float32)


def _minmax_columns(frame: pd.DataFrame) -> pd.DataFrame:
    minima = frame.min(axis=0)
    maxima = frame.max(axis=0)
    denom = (maxima - minima).replace(0, 1.0).fillna(1.0)
    scaled = (frame - minima) / denom
    return scaled.fillna(0.0).astype(np.float32)


def build_early_integration_embedding(
    modality_embeddings: dict[str, pd.DataFrame],
    component_cap: int = EARLY_INTEGRATION_COMPONENT_CAP,
) -> pd.DataFrame:
    merged = _zscore_columns(_concat_modality_embeddings(modality_embeddings))
    n_components = max(1, min(component_cap, merged.shape[0] - 1, merged.shape[1]))
    model = PCA(n_components=n_components, random_state=0)
    embedding = model.fit_transform(merged.to_numpy(dtype=np.float32))
    cols = [f"integrated_component_{i + 1}" for i in range(embedding.shape[1])]
    return pd.DataFrame(embedding, index=merged.index, columns=cols, dtype=np.float32)


def build_nmf_integration_embedding(
    modality_embeddings: dict[str, pd.DataFrame],
    component_cap: int = NMF_COMPONENT_CAP,
) -> pd.DataFrame:
    merged = _minmax_columns(_concat_modality_embeddings(modality_embeddings))
    n_components = max(1, min(component_cap, merged.shape[0] - 1, merged.shape[1]))
    model = NMF(n_components=n_components, init="nndsvda", max_iter=2000, random_state=0)
    embedding = model.fit_transform(merged.to_numpy(dtype=np.float32))
    cols = [f"nmf_component_{i + 1}" for i in range(embedding.shape[1])]
    return pd.DataFrame(embedding, index=merged.index, columns=cols, dtype=np.float32)


def silhouette_from_embedding(embedding: pd.DataFrame, labels: pd.Series) -> float:
    if labels.nunique() < 2:
        return float("-inf")
    return float(silhouette_score(embedding.to_numpy(dtype=np.float32), labels.to_numpy()))


def embedding_consensus_clustering(
    embedding: pd.DataFrame,
    k_values=K_RANGE,
    reps: int = CONSENSUS_REPS,
    sample_fraction: float = CONSENSUS_FRACTION,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    random_state: int = 0,
) -> tuple[pd.DataFrame, int, pd.Series]:
    rng = np.random.default_rng(random_state)
    samples = embedding.index.to_numpy()
    x = embedding.to_numpy(dtype=np.float32)
    n = len(samples)
    records: list[dict[str, object]] = []
    final_labels: dict[int, pd.Series] = {}

    for k in k_values:
        consensus = np.zeros((n, n), dtype=np.float32)
        co_sampled = np.zeros((n, n), dtype=np.float32)
        draw_size = max(k + 1, int(round(n * sample_fraction)))

        for rep in range(reps):
            subset_idx = np.sort(rng.choice(n, size=draw_size, replace=False))
            sub_x = x[subset_idx]
            model = KMeans(n_clusters=k, random_state=random_state + rep, n_init=20)
            subset_labels = model.fit_predict(sub_x)
            co_sampled[np.ix_(subset_idx, subset_idx)] += 1
            for cluster_value in np.unique(subset_labels):
                members = subset_idx[subset_labels == cluster_value]
                consensus[np.ix_(members, members)] += 1

        consensus = np.divide(consensus, co_sampled, out=np.zeros_like(consensus), where=co_sampled > 0)
        pac = float(np.mean((consensus > 0.1) & (consensus < 0.9)))

        final_model = KMeans(n_clusters=k, random_state=random_state, n_init=50)
        labels_array = final_model.fit_predict(x)
        labels = pd.Series(labels_array, index=embedding.index, name="cluster")
        cluster_sizes = labels.value_counts().sort_index()
        min_size = int(cluster_sizes.min())
        silhouette = silhouette_from_embedding(embedding, labels)
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


def early_integration_consensus_clustering(
    integrated_embedding: pd.DataFrame,
    k_values=K_RANGE,
    reps: int = CONSENSUS_REPS,
    sample_fraction: float = CONSENSUS_FRACTION,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    random_state: int = 0,
) -> tuple[pd.DataFrame, int, pd.Series]:
    return embedding_consensus_clustering(
        integrated_embedding,
        k_values=k_values,
        reps=reps,
        sample_fraction=sample_fraction,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
    )
