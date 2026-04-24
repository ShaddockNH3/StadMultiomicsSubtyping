from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import rbf_kernel

from .config import SNF_K, SNF_T


def build_affinity_matrix(embedding: pd.DataFrame, k: int = SNF_K) -> pd.DataFrame:
    x = embedding.to_numpy(dtype=np.float32)
    samples = embedding.index.tolist()
    distances = pairwise_distances(x, metric="euclidean")
    order = np.argsort(distances, axis=1)
    kth_index = min(k, distances.shape[1] - 1)
    sigma = distances[np.arange(distances.shape[0]), order[:, kth_index]]
    sigma = np.where(sigma <= 0, np.nanmedian(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0, sigma)
    sigma = np.where(np.isnan(sigma), 1.0, sigma)

    affinity = np.zeros_like(distances, dtype=np.float32)
    n = distances.shape[0]
    for i in range(n):
        neighbors = order[i, 1 : kth_index + 1]
        for j in neighbors:
            local_sigma = sigma[i] * sigma[j]
            if local_sigma <= 0:
                local_sigma = 1.0
            value = np.exp(-(distances[i, j] ** 2) / (local_sigma + 1e-8))
            affinity[i, j] = value
            affinity[j, i] = max(affinity[j, i], value)
    affinity = (affinity + affinity.T) / 2.0
    np.fill_diagonal(affinity, 0.0)
    row_sums = affinity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    affinity = affinity / row_sums
    affinity = (affinity + affinity.T) / 2.0
    np.fill_diagonal(affinity, 0.0)
    return pd.DataFrame(affinity, index=samples, columns=samples)


def normalize_affinity(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = matrix / row_sums
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def dominant_set(matrix: np.ndarray, k: int = SNF_K) -> np.ndarray:
    order = np.argsort(matrix, axis=1)[:, ::-1]
    out = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        keep = order[i, :k]
        out[i, keep] = matrix[i, keep]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    out = out / row_sums
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def snf(affinities: list[pd.DataFrame], k: int = SNF_K, t: int = SNF_T) -> pd.DataFrame:
    if len(affinities) == 1:
        return affinities[0]
    matrices = [normalize_affinity(aff.to_numpy(dtype=np.float32)) for aff in affinities]
    sparse_mats = [dominant_set(mat, k=k) for mat in matrices]
    for _ in range(t):
        updated = []
        for i, mat in enumerate(matrices):
            others = [matrices[j] for j in range(len(matrices)) if j != i]
            mean_other = sum(others) / max(1, len(others))
            fused = sparse_mats[i] @ mean_other @ sparse_mats[i].T
            fused = normalize_affinity(fused)
            updated.append(fused)
        matrices = updated
    fused = sum(matrices) / len(matrices)
    fused = (fused + fused.T) / 2.0
    np.fill_diagonal(fused, 0.0)
    idx = affinities[0].index
    return pd.DataFrame(fused, index=idx, columns=idx)


def fused_embedding(fused_affinity: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    n_components = max(1, min(n_components, fused_affinity.shape[0] - 1))
    kernel = fused_affinity.to_numpy(dtype=np.float32)
    model = SpectralClustering(
        n_clusters=n_components,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    )
    labels = model.fit_predict(kernel)
    features = rbf_kernel(kernel, gamma=1.0 / max(kernel.shape[1], 1))
    cols = [f"fused_{i + 1}" for i in range(features.shape[1])]
    out = pd.DataFrame(features, index=fused_affinity.index, columns=cols, dtype=np.float32)
    out["spectral_seed_label"] = labels
    return out


def spectral_cluster(fused_affinity: pd.DataFrame, n_clusters: int, random_state: int = 0) -> pd.Series:
    matrix = fused_affinity.to_numpy(dtype=np.float32)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    embedding = spectral_embedding(
        matrix,
        n_components=max(1, n_clusters),
        random_state=random_state,
        drop_first=False,
    )
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding = embedding / norms
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(embedding)
    return pd.Series(labels, index=fused_affinity.index, name="cluster")


def silhouette_from_affinity(fused_affinity: pd.DataFrame, labels: pd.Series) -> float:
    if labels.nunique() < 2:
        return float("-inf")
    distance = 1.0 - fused_affinity.to_numpy(dtype=np.float32)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return float(silhouette_score(distance, labels.to_numpy(), metric="precomputed"))
