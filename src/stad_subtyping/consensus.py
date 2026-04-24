from __future__ import annotations

import numpy as np
import pandas as pd

from .cluster import label_comparison_metrics


MODEL_WEIGHT_SCHEMES: tuple[dict[str, float | str | None], ...] = (
    {
        "scheme_name": "stability_focused",
        "stability_weight": 0.50,
        "structure_weight": 0.30,
        "clinical_weight": 0.20,
        "resolution_weight": 0.00,
        "shrinkage": 0.00,
        "max_model_weight": None,
    },
    {
        "scheme_name": "balanced",
        "stability_weight": 0.30,
        "structure_weight": 0.25,
        "clinical_weight": 0.15,
        "resolution_weight": 0.30,
        "shrinkage": 0.15,
        "max_model_weight": 0.48,
    },
    {
        "scheme_name": "resolution_guarded",
        "stability_weight": 0.20,
        "structure_weight": 0.25,
        "clinical_weight": 0.10,
        "resolution_weight": 0.45,
        "shrinkage": 0.25,
        "max_model_weight": 0.44,
    },
    {
        "scheme_name": "anti_collapse",
        "stability_weight": 0.15,
        "structure_weight": 0.20,
        "clinical_weight": 0.10,
        "resolution_weight": 0.55,
        "shrinkage": 0.35,
        "max_model_weight": 0.40,
    },
)


def _minmax_scale(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = series.astype(float)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(values), dtype=float), index=values.index)
    scaled = (values - values.min()) / (values.max() - values.min())
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.fillna(0.0)


def _normalized_entropy(weights: pd.Series) -> float:
    values = weights.astype(float).to_numpy()
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    entropy = -np.sum(values * np.log(values))
    return float(entropy / np.log(len(weights)))


def _apply_shrinkage(weights: pd.Series, shrinkage: float) -> pd.Series:
    if shrinkage <= 0:
        return weights / weights.sum()
    uniform = 1.0 / len(weights)
    shrunk = (1.0 - shrinkage) * weights + shrinkage * uniform
    return shrunk / shrunk.sum()


def _apply_weight_cap(weights: pd.Series, max_model_weight: float | None) -> pd.Series:
    normalized = weights.astype(float) / weights.sum()
    if max_model_weight is None or len(normalized) <= 1:
        return normalized
    if max_model_weight <= 1.0 / len(normalized):
        return pd.Series(np.repeat(1.0 / len(normalized), len(normalized)), index=normalized.index)

    capped = normalized.copy()
    for _ in range(10):
        overflow_mask = capped > (max_model_weight + 1e-12)
        if not overflow_mask.any():
            break
        overflow = float((capped[overflow_mask] - max_model_weight).sum())
        capped.loc[overflow_mask] = max_model_weight
        under_mask = ~overflow_mask
        if not under_mask.any():
            capped[:] = 1.0 / len(capped)
            break
        redistribute = capped.loc[under_mask].copy()
        if redistribute.sum() <= 0:
            redistribute[:] = 1.0
        capped.loc[under_mask] += overflow * (redistribute / redistribute.sum())
        capped = capped / capped.sum()
    return capped / capped.sum()


def _build_quality_table(
    model_evaluation_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
) -> pd.DataFrame:
    evaluation = model_evaluation_summary.copy()
    stability = model_stability_summary.set_index("model_name") if not model_stability_summary.empty else pd.DataFrame()
    evaluation["stability_nmi"] = evaluation["model_name"].map(stability["overlap_nmi"]) if not stability.empty else np.nan
    evaluation["stability_jaccard"] = (
        evaluation["model_name"].map(stability["overlap_co_clustering_jaccard"]) if not stability.empty else np.nan
    )
    evaluation["msi_signal"] = -np.log10(evaluation["msi_status_p"].clip(lower=1e-300))

    rows: list[pd.DataFrame] = []
    for _, group in evaluation.groupby("queue_key", sort=False):
        queue_group = group.copy()
        if "selected_min_cluster_size" not in queue_group.columns:
            queue_group["selected_min_cluster_size"] = np.nan
        queue_group["pac_quality"] = _minmax_scale(queue_group["selected_pac_score"], higher_is_better=False)
        queue_group["silhouette_quality"] = _minmax_scale(queue_group["selected_silhouette"], higher_is_better=True)
        queue_group["clinical_quality"] = _minmax_scale(queue_group["msi_signal"], higher_is_better=True)
        queue_group["stability_nmi_quality"] = _minmax_scale(queue_group["stability_nmi"], higher_is_better=True)
        queue_group["stability_jaccard_quality"] = _minmax_scale(queue_group["stability_jaccard"], higher_is_better=True)
        queue_group["best_k_quality"] = _minmax_scale(queue_group["best_k"], higher_is_better=True)
        queue_group["min_cluster_resolution_quality"] = _minmax_scale(
            queue_group["selected_min_cluster_size"],
            higher_is_better=False,
        )
        queue_group["stability_score"] = queue_group[["stability_nmi_quality", "stability_jaccard_quality"]].mean(axis=1)
        queue_group["structure_score"] = queue_group[["pac_quality", "silhouette_quality"]].mean(axis=1)
        queue_group["clinical_score"] = queue_group["clinical_quality"]
        queue_group["resolution_score"] = queue_group[["best_k_quality", "min_cluster_resolution_quality"]].mean(axis=1)
        queue_group["base_scheme_name"] = "base"
        queue_group["rescue_applied"] = False
        queue_group["rescued_model_name"] = ""
        queue_group["rescue_cap"] = np.nan
        rows.append(queue_group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_consensus_weight_tables(
    model_evaluation_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
    schemes: tuple[dict[str, float | str | None], ...] = MODEL_WEIGHT_SCHEMES,
) -> pd.DataFrame:
    quality_table = _build_quality_table(model_evaluation_summary, model_stability_summary)
    if quality_table.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for queue_key, group in quality_table.groupby("queue_key", sort=False):
        for scheme in schemes:
            queue_group = group.copy()
            scheme_name = str(scheme["scheme_name"])
            queue_group["scheme_name"] = scheme_name
            queue_group["scheme_stability_weight"] = float(scheme["stability_weight"])
            queue_group["scheme_structure_weight"] = float(scheme["structure_weight"])
            queue_group["scheme_clinical_weight"] = float(scheme["clinical_weight"])
            queue_group["scheme_resolution_weight"] = float(scheme["resolution_weight"])
            queue_group["scheme_shrinkage"] = float(scheme["shrinkage"])
            queue_group["scheme_max_model_weight"] = (
                float(scheme["max_model_weight"]) if scheme["max_model_weight"] is not None else np.nan
            )
            queue_group["raw_weight"] = (
                1.0
                + queue_group["scheme_stability_weight"] * queue_group["stability_score"]
                + queue_group["scheme_structure_weight"] * queue_group["structure_score"]
                + queue_group["scheme_clinical_weight"] * queue_group["clinical_score"]
                + queue_group["scheme_resolution_weight"] * queue_group["resolution_score"]
            )
            queue_group["pre_shrink_weight"] = queue_group["raw_weight"] / queue_group["raw_weight"].sum()
            shrunk = _apply_shrinkage(queue_group["pre_shrink_weight"], float(scheme["shrinkage"]))
            capped = _apply_weight_cap(shrunk, scheme["max_model_weight"])
            queue_group["normalized_weight"] = capped.to_numpy()
            queue_group["weight_rank"] = queue_group["normalized_weight"].rank(ascending=False, method="dense").astype(int)
            queue_group["queue_weight_entropy"] = _normalized_entropy(queue_group["normalized_weight"])
            queue_group["queue_max_model_weight"] = float(queue_group["normalized_weight"].max())
            rows.append(queue_group)

    weight_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    keep_columns = [
        "queue_key",
        "analysis_name",
        "scheme_name",
        "base_scheme_name",
        "model_name",
        "best_k",
        "n_samples",
        "selected_pac_score",
        "selected_silhouette",
        "selected_min_cluster_size",
        "msi_status_p",
        "stability_nmi",
        "stability_jaccard",
        "stability_score",
        "structure_score",
        "clinical_score",
        "resolution_score",
        "scheme_stability_weight",
        "scheme_structure_weight",
        "scheme_clinical_weight",
        "scheme_resolution_weight",
        "scheme_shrinkage",
        "scheme_max_model_weight",
        "rescue_applied",
        "rescued_model_name",
        "rescue_cap",
        "raw_weight",
        "pre_shrink_weight",
        "normalized_weight",
        "queue_weight_entropy",
        "queue_max_model_weight",
        "weight_rank",
    ]
    return weight_table[keep_columns] if not weight_table.empty else pd.DataFrame(columns=keep_columns)


def build_consensus_weight_table(
    model_evaluation_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
    scheme_name: str = "stability_focused",
) -> pd.DataFrame:
    weight_tables = build_consensus_weight_tables(model_evaluation_summary, model_stability_summary)
    if weight_tables.empty:
        return weight_tables
    return weight_tables[weight_tables["scheme_name"] == scheme_name].reset_index(drop=True)


def co_clustering_frame(labels: pd.Series) -> pd.DataFrame:
    ordered = labels.sort_index()
    values = ordered.to_numpy()
    matrix = (values[:, None] == values[None, :]).astype(np.float32)
    return pd.DataFrame(matrix, index=ordered.index, columns=ordered.index)


def weighted_consensus_matrix(queue_results: list[dict], weight_table: pd.DataFrame) -> pd.DataFrame:
    if not queue_results:
        raise ValueError("queue_results must not be empty")
    sample_order = queue_results[0]["labels"].sort_index().index
    weighted_sum = np.zeros((len(sample_order), len(sample_order)), dtype=np.float32)
    total_weight = 0.0
    for result in queue_results:
        model_name = result["model_name"]
        weight_row = weight_table.loc[weight_table["model_name"] == model_name]
        if weight_row.empty:
            continue
        weight = float(weight_row.iloc[0]["normalized_weight"])
        labels = result["labels"].sort_index().loc[sample_order]
        matrix = co_clustering_frame(labels).to_numpy(dtype=np.float32)
        weighted_sum += weight * matrix
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("No positive weights found for weighted consensus matrix")
    consensus = weighted_sum / total_weight
    consensus = (consensus + consensus.T) / 2.0
    np.fill_diagonal(consensus, 0.0)
    return pd.DataFrame(consensus, index=sample_order, columns=sample_order)


def consensus_candidate_summary_frame(
    candidate_results: list[dict],
    comparison_results: list[tuple[str, dict]],
    weight_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparison_lookup: dict[tuple[str, str], dict] = {
        (queue_key, result["model_name"]): result for queue_key, result in comparison_results
    }
    for candidate in candidate_results:
        queue_key = str(candidate["queue_key"])
        scheme_name = str(candidate["scheme_name"])
        queue_weight_table = weight_table[
            (weight_table["queue_key"] == queue_key) & (weight_table["scheme_name"] == scheme_name)
        ].copy()
        dominant_weight_row = queue_weight_table.sort_values("normalized_weight", ascending=False).iloc[0]
        selected_row = candidate["k_summary"].loc[candidate["k_summary"]["selected"]].iloc[0]
        clinical_lookup = candidate["clinical_table"].set_index("variable") if not candidate["clinical_table"].empty else pd.DataFrame()
        survival_lookup = candidate["survival_table"].set_index("endpoint") if not candidate["survival_table"].empty else pd.DataFrame()
        msi_status_p = float(clinical_lookup.loc["msi_status", "p_value"]) if "msi_status" in clinical_lookup.index else np.nan
        row: dict[str, object] = {
            "queue_key": queue_key,
            "analysis_name": str(candidate["analysis_name"]),
            "scheme_name": scheme_name,
            "best_k": int(candidate["best_k"]),
            "n_samples": int(len(candidate["labels"])),
            "selected_pac_score": float(selected_row["pac_score"]),
            "selected_silhouette": float(selected_row["silhouette"]),
            "selected_min_cluster_size": int(selected_row["min_cluster_size"]),
            "msi_status_p": msi_status_p,
            "os_logrank_p": float(survival_lookup.loc["OS", "logrank_p"]) if "OS" in survival_lookup.index else np.nan,
            "dominant_model_name": str(dominant_weight_row["model_name"]),
            "dominant_model_weight": float(dominant_weight_row["normalized_weight"]),
            "weight_entropy": float(queue_weight_table["queue_weight_entropy"].iloc[0]),
            "weight_max_model_weight": float(queue_weight_table["queue_max_model_weight"].iloc[0]),
            "base_scheme_name": (
                str(queue_weight_table["base_scheme_name"].iloc[0]) if "base_scheme_name" in queue_weight_table.columns else scheme_name
            ),
            "rescue_applied": bool(queue_weight_table["rescue_applied"].any()) if "rescue_applied" in queue_weight_table.columns else False,
            "rescued_model_name": (
                str(queue_weight_table.loc[queue_weight_table["rescue_applied"], "rescued_model_name"].iloc[0])
                if "rescue_applied" in queue_weight_table.columns and queue_weight_table["rescue_applied"].any()
                else ""
            ),
        }
        ari_values: list[float] = []
        nmi_values: list[float] = []
        jaccard_values: list[float] = []
        for model_name in ("snf", "early_integration", "nmf_integration"):
            base_result = comparison_lookup.get((queue_key, model_name))
            if base_result is None:
                continue
            metrics = label_comparison_metrics(candidate["labels"], base_result["labels"])
            row[f"ari_to_{model_name}"] = float(metrics["ari"])
            row[f"nmi_to_{model_name}"] = float(metrics["nmi"])
            row[f"jaccard_to_{model_name}"] = float(metrics["co_clustering_jaccard"])
            ari_values.append(float(metrics["ari"]))
            nmi_values.append(float(metrics["nmi"]))
            jaccard_values.append(float(metrics["co_clustering_jaccard"]))
        row["mean_ari_to_models"] = float(np.nanmean(ari_values)) if ari_values else np.nan
        row["mean_nmi_to_models"] = float(np.nanmean(nmi_values)) if nmi_values else np.nan
        row["mean_jaccard_to_models"] = float(np.nanmean(jaccard_values)) if jaccard_values else np.nan
        row["max_ari_to_single_model"] = float(np.nanmax(ari_values)) if ari_values else np.nan
        row["is_exact_copy"] = bool(row["max_ari_to_single_model"] >= 0.999999)
        if row["is_exact_copy"]:
            ari_lookup = {
                "snf": row.get("ari_to_snf", float("nan")),
                "early_integration": row.get("ari_to_early_integration", float("nan")),
                "nmf_integration": row.get("ari_to_nmf_integration", float("nan")),
            }
            row["exact_copy_model_name"] = max(ari_lookup, key=lambda key: float(ari_lookup[key]))
        else:
            row["exact_copy_model_name"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_copy_rescued_weight_tables(
    weight_table: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    rescue_cap: float = 0.10,
) -> pd.DataFrame:
    if weight_table.empty or candidate_summary.empty:
        return pd.DataFrame(columns=weight_table.columns)

    rescued_tables: list[pd.DataFrame] = []
    for scheme_name, scheme_weights in weight_table.groupby("scheme_name", sort=False):
        scheme_summary = candidate_summary[candidate_summary["scheme_name"] == scheme_name]
        if scheme_summary.empty or not bool(scheme_summary["is_exact_copy"].any()):
            continue

        rescued_scheme = scheme_weights.copy()
        rescued_scheme["scheme_name"] = f"{scheme_name}_copy_rescued"
        rescued_scheme["base_scheme_name"] = scheme_name
        rescued_any = False

        for _, summary_row in scheme_summary.iterrows():
            if not bool(summary_row["is_exact_copy"]):
                continue

            queue_key = str(summary_row["queue_key"])
            copied_model_name = str(summary_row.get("exact_copy_model_name", ""))
            queue_mask = rescued_scheme["queue_key"] == queue_key
            queue_weights = rescued_scheme.loc[queue_mask].copy()
            if queue_weights.empty or copied_model_name not in set(queue_weights["model_name"]):
                continue

            copied_best_k = int(queue_weights.loc[queue_weights["model_name"] == copied_model_name, "best_k"].iloc[0])
            queue_max_k = int(queue_weights["best_k"].max())
            if copied_best_k >= queue_max_k:
                continue

            normalized = queue_weights.set_index("model_name")["normalized_weight"].astype(float)
            original_weight = float(normalized.loc[copied_model_name])
            if original_weight <= rescue_cap:
                continue
            normalized.loc[copied_model_name] = rescue_cap
            excess = original_weight - rescue_cap

            other_models = [model for model in normalized.index if model != copied_model_name]
            redistribute_basis = (
                queue_weights.set_index("model_name").loc[other_models, "normalized_weight"].astype(float)
                * (1.0 + queue_weights.set_index("model_name").loc[other_models, "resolution_score"].astype(float).fillna(0.0))
            )
            if redistribute_basis.sum() <= 0:
                redistribute_basis = pd.Series(1.0, index=other_models, dtype=float)
            normalized.loc[other_models] += excess * (redistribute_basis / redistribute_basis.sum())
            normalized = normalized / normalized.sum()

            queue_indices = rescued_scheme.index[queue_mask]
            rescued_scheme.loc[queue_indices, "normalized_weight"] = queue_weights["model_name"].map(normalized).to_numpy()
            rescued_scheme.loc[queue_indices, "queue_weight_entropy"] = _normalized_entropy(normalized)
            rescued_scheme.loc[queue_indices, "queue_max_model_weight"] = float(normalized.max())
            rescued_scheme.loc[queue_indices, "weight_rank"] = (
                pd.Series(normalized)
                .rank(ascending=False, method="dense")
                .reindex(queue_weights["model_name"])
                .astype(int)
                .to_numpy()
            )
            rescued_scheme.loc[queue_indices, "rescue_applied"] = True
            rescued_scheme.loc[queue_indices, "rescued_model_name"] = copied_model_name
            rescued_scheme.loc[queue_indices, "rescue_cap"] = rescue_cap
            rescued_any = True

        if rescued_any:
            rescued_tables.append(rescued_scheme)

    if not rescued_tables:
        return pd.DataFrame(columns=weight_table.columns)
    return pd.concat(rescued_tables, ignore_index=True)


def select_consensus_scheme(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    if candidate_summary.empty:
        return pd.DataFrame(
            columns=[
                "scheme_name",
                "mean_queue_score",
                "n_queues",
                "n_exact_copy_queues",
                "mean_best_k",
                "mean_weight_entropy",
                "mean_max_ari_to_single_model",
                "selected",
            ]
        )

    scored_rows: list[pd.DataFrame] = []
    for _, group in candidate_summary.groupby("queue_key", sort=False):
        queue_group = group.copy()
        queue_group["anti_collapse_quality"] = _minmax_scale(
            queue_group["max_ari_to_single_model"],
            higher_is_better=False,
        )
        queue_group["diversity_quality"] = (
            _minmax_scale(queue_group["weight_entropy"], higher_is_better=True)
            + _minmax_scale(queue_group["dominant_model_weight"], higher_is_better=False)
        ) / 2.0
        queue_group["resolution_quality"] = _minmax_scale(queue_group["best_k"], higher_is_better=True)
        queue_group["structure_quality"] = (
            _minmax_scale(queue_group["selected_pac_score"], higher_is_better=False)
            + _minmax_scale(queue_group["selected_silhouette"], higher_is_better=True)
        ) / 2.0
        queue_group["clinical_quality"] = _minmax_scale(
            -np.log10(queue_group["msi_status_p"].clip(lower=1e-300)),
            higher_is_better=True,
        )
        queue_group["copy_penalty"] = queue_group["is_exact_copy"].astype(float)
        queue_group["queue_score"] = (
            0.35 * queue_group["anti_collapse_quality"]
            + 0.25 * queue_group["diversity_quality"]
            + 0.20 * queue_group["resolution_quality"]
            + 0.15 * queue_group["structure_quality"]
            + 0.05 * queue_group["clinical_quality"]
            - 0.35 * queue_group["copy_penalty"]
        )
        scored_rows.append(queue_group)

    scored = pd.concat(scored_rows, ignore_index=True)
    selection = (
        scored.groupby("scheme_name", sort=False)
        .agg(
            mean_queue_score=("queue_score", "mean"),
            n_queues=("queue_key", "nunique"),
            n_exact_copy_queues=("is_exact_copy", "sum"),
            mean_best_k=("best_k", "mean"),
            mean_weight_entropy=("weight_entropy", "mean"),
            mean_max_ari_to_single_model=("max_ari_to_single_model", "mean"),
            mean_selected_silhouette=("selected_silhouette", "mean"),
        )
        .reset_index()
    )
    selected_idx = selection.sort_values(
        ["n_exact_copy_queues", "mean_queue_score", "mean_best_k", "mean_weight_entropy", "mean_selected_silhouette"],
        ascending=[True, False, False, False, False],
    ).index[0]
    selection["selected"] = False
    selection.loc[selected_idx, "selected"] = True
    return selection


def consensus_summary_frame(
    queue_configs: list[tuple[str, str]],
    consensus_results: list[dict],
    comparison_results: list[tuple[str, dict]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparison_lookup: dict[tuple[str, str], dict] = {
        (queue_key, result["model_name"]): result for queue_key, result in comparison_results
    }
    for queue_key, analysis_name in queue_configs:
        consensus_result = next(result for result in consensus_results if result["queue_key"] == queue_key)
        row: dict[str, object] = {
            "queue_key": queue_key,
            "analysis_name": analysis_name,
            "scheme_name": str(consensus_result.get("scheme_name", "stability_focused")),
            "best_k": int(consensus_result["best_k"]),
            "n_samples": int(len(consensus_result["labels"])),
            "output_dir": str(consensus_result["out_dir"]),
        }
        for model_name in ("snf", "early_integration", "nmf_integration"):
            base_result = comparison_lookup.get((queue_key, model_name))
            if base_result is None:
                continue
            metrics = label_comparison_metrics(consensus_result["labels"], base_result["labels"])
            row[f"ari_to_{model_name}"] = float(metrics["ari"])
            row[f"nmi_to_{model_name}"] = float(metrics["nmi"])
            row[f"jaccard_to_{model_name}"] = float(metrics["co_clustering_jaccard"])
        rows.append(row)
    return pd.DataFrame(rows)
