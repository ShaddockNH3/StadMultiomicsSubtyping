from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA, TruncatedSVD

from .config import PCA_COMPONENT_CAP


def row_mad(frame: pd.DataFrame) -> pd.Series:
    values = frame.to_numpy(dtype=np.float32)
    mad = median_abs_deviation(values, axis=1, scale=1.0, nan_policy="omit")
    return pd.Series(np.asarray(mad, dtype=np.float32), index=frame.index)


def preprocess_modality(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    qc = {"input_features": int(frame.shape[0]), "input_samples": int(frame.shape[1])}
    data = frame.copy()

    if name == "mutation":
        freq = (data.fillna(0) != 0).mean(axis=1)
        data = data.loc[freq >= 0.03].fillna(0)
    elif name == "cnv":
        freq = (data.fillna(0).abs() > 0).mean(axis=1)
        data = data.loc[freq >= 0.05].fillna(0)
    elif name == "expression":
        data = fill_missing_by_row_median(data)
        mad = row_mad(data)
        keep = mad.sort_values(ascending=False).head(min(2000, len(mad))).index
        data = data.loc[keep]
        data = zscore_rows(data)
    elif name == "mirna":
        missing = data.isna().mean(axis=1)
        data = data.loc[missing <= 0.2]
        data = fill_missing_by_row_median(data)
        mad = row_mad(data)
        keep = mad.sort_values(ascending=False).head(min(500, len(mad))).index
        data = data.loc[keep]
        data = zscore_rows(data)
    elif name == "methylation":
        missing = data.isna().mean(axis=1)
        data = data.loc[missing <= 0.2]
        data = fill_missing_by_row_median(data)
        mad = row_mad(data)
        keep = mad.sort_values(ascending=False).head(min(5000, len(mad))).index
        data = data.loc[keep]
        data = zscore_rows(data)
    elif name == "rppa":
        data = fill_missing_by_row_median(data)
        data = zscore_rows(data)
    else:
        raise ValueError(f"Unknown modality: {name}")

    qc["selected_features"] = int(data.shape[0])
    qc["selected_samples"] = int(data.shape[1])
    return data.astype(np.float32), qc


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    means = frame.mean(axis=1)
    stds = frame.std(axis=1).replace(0, 1.0).fillna(1.0)
    scaled = frame.sub(means, axis=0).div(stds, axis=0)
    return scaled.astype(np.float32)


def fill_missing_by_row_median(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(axis=0, how="all").copy()
    medians = frame.median(axis=1).fillna(0.0)
    frame = frame.T.fillna(medians).T
    return frame.astype(np.float32)


def fit_modality_reducer(name: str, frame: pd.DataFrame) -> tuple[PCA | TruncatedSVD, np.ndarray]:
    n_features, n_samples = frame.shape
    n_components = max(1, min(PCA_COMPONENT_CAP, n_samples - 1, n_features))
    matrix = frame.T.to_numpy(dtype=np.float32)
    if name in {"mutation", "cnv"}:
        model = TruncatedSVD(n_components=n_components, random_state=0)
    else:
        model = PCA(n_components=n_components, random_state=0)
    embedding = model.fit_transform(matrix)
    return model, embedding


def reduce_modality(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    model, embedding = fit_modality_reducer(name, frame)
    cols = [f"component_{i + 1}" for i in range(embedding.shape[1])]
    return pd.DataFrame(embedding, index=frame.columns, columns=cols, dtype=np.float32)
