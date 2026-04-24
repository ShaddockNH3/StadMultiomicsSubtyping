from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import chi2
from scipy.stats import chi2_contingency, fisher_exact, kruskal


def build_subtype_table(
    labels: pd.Series,
    clinical: pd.DataFrame,
    survival: pd.DataFrame,
    analysis_name: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=labels.index)
    out["analysis"] = analysis_name
    out["subtype"] = labels
    out = out.join(clinical)
    out = out.join(survival[["OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time"]])
    out.index.name = "patient_id"
    return out.reset_index()


def survival_stats(labels: pd.Series, survival: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, time_col in [("OS", "OS.time"), ("DSS", "DSS.time"), ("DFI", "DFI.time"), ("PFI", "PFI.time")]:
        frame = survival[[endpoint, time_col]].dropna()
        frame = frame.join(labels, how="inner")
        if frame.empty or frame["subtype"].nunique() < 2:
            rows.append({"endpoint": endpoint, "logrank_p": np.nan, "n_samples": len(frame)})
            continue
        result = multivariate_logrank_test(frame[time_col], frame["subtype"], frame[event_col(endpoint)])
        row = {
            "endpoint": endpoint,
            "logrank_p": float(result.p_value),
            "n_samples": int(len(frame)),
        }
        for subtype, group in frame.groupby("subtype"):
            row[f"{subtype}_median_time"] = float(group[time_col].median())
            row[f"{subtype}_n"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def event_col(endpoint: str) -> str:
    return endpoint


def _clean_category(series: pd.Series) -> pd.Series:
    cleaned = series.replace({"[Discrepancy]": np.nan, "[Not Available]": np.nan, "[Not Applicable]": np.nan, "": np.nan})
    return cleaned.dropna()


def _mode_or_na(series: pd.Series) -> str:
    cleaned = _clean_category(series)
    if cleaned.empty:
        return "NA"
    mode = cleaned.mode()
    if mode.empty:
        return "NA"
    return str(mode.iloc[0])


def clinical_associations(labels: pd.Series, clinical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    joined = clinical.join(labels, how="inner")
    categorical_cols = ["msi_status", "pathologic_stage", "grade", "gender", "histological_type"]
    for col in categorical_cols:
        data = joined[[col, "subtype"]].copy()
        data[col] = _clean_category(data[col])
        data = data.dropna()
        if data.empty:
            rows.append({"variable": col, "test": "chi2", "p_value": np.nan, "n": 0})
            continue
        contingency = pd.crosstab(data[col], data["subtype"])
        if contingency.shape == (2, 2):
            _, p_value = fisher_exact(contingency.to_numpy())
            test_name = "fisher"
        else:
            _, p_value, _, _ = chi2_contingency(contingency)
            test_name = "chi2"
        rows.append({"variable": col, "test": test_name, "p_value": float(p_value), "n": int(len(data))})

    age_frame = joined[["age", "subtype"]].copy()
    age_frame["age"] = pd.to_numeric(age_frame["age"], errors="coerce")
    age_frame = age_frame.dropna()
    age_groups = [group["age"].to_numpy() for _, group in age_frame.groupby("subtype")]
    if len(age_groups) >= 2 and all(len(group) > 0 for group in age_groups):
        _, age_p = kruskal(*age_groups)
    else:
        age_p = np.nan
    rows.append({"variable": "age", "test": "kruskal", "p_value": float(age_p) if not pd.isna(age_p) else np.nan, "n": int(len(age_frame))})
    return pd.DataFrame(rows)


def _stage_group(series: pd.Series) -> pd.Series:
    cleaned = _clean_category(series.astype(str))
    grouped = cleaned.str.extract(r"Stage\s+(IV|III|II|I)", expand=False)
    return grouped


def multivariable_cox(
    subtype_table: pd.DataFrame,
    penalizer: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    endpoints = [("OS", "OS.time"), ("DSS", "DSS.time"), ("DFI", "DFI.time"), ("PFI", "PFI.time")]

    for endpoint, time_col in endpoints:
        frame = subtype_table[["subtype", "age", "gender", "pathologic_stage", endpoint, time_col]].copy()
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
        frame["gender"] = _clean_category(frame["gender"].astype("object"))
        frame["stage_group"] = _stage_group(frame["pathologic_stage"].astype("object"))
        frame = frame.drop(columns=["pathologic_stage"])
        frame[endpoint] = pd.to_numeric(frame[endpoint], errors="coerce")
        frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[endpoint, time_col, "subtype", "age", "gender", "stage_group"])
        frame = frame[frame[time_col] > 0].copy()

        if frame.empty or frame["subtype"].nunique() < 2:
            global_rows.append(
                {
                    "endpoint": endpoint,
                    "n_samples": int(len(frame)),
                    "n_events": int(frame[endpoint].sum()) if not frame.empty else 0,
                    "subtype_lrt_p": np.nan,
                    "full_concordance": np.nan,
                    "status": "insufficient_data",
                }
            )
            continue

        base_covariates = pd.get_dummies(frame[["age", "gender", "stage_group"]], columns=["gender", "stage_group"], drop_first=True)
        subtype_covariates = pd.get_dummies(frame[["subtype"]], columns=["subtype"], drop_first=True)
        full_covariates = pd.concat([base_covariates, subtype_covariates], axis=1)

        base_model_df = pd.concat([frame[[time_col, endpoint]], base_covariates], axis=1)
        full_model_df = pd.concat([frame[[time_col, endpoint]], full_covariates], axis=1)

        try:
            base_model = CoxPHFitter(penalizer=penalizer)
            base_model.fit(base_model_df, duration_col=time_col, event_col=endpoint)
            full_model = CoxPHFitter(penalizer=penalizer)
            full_model.fit(full_model_df, duration_col=time_col, event_col=endpoint)
        except Exception as exc:  # pragma: no cover - defensive path for real-world instability
            global_rows.append(
                {
                    "endpoint": endpoint,
                    "n_samples": int(len(frame)),
                    "n_events": int(frame[endpoint].sum()),
                    "subtype_lrt_p": np.nan,
                    "full_concordance": np.nan,
                    "status": f"fit_failed: {type(exc).__name__}",
                }
            )
            continue

        subtype_cols = [col for col in full_covariates.columns if col.startswith("subtype_")]
        lrt_stat = 2.0 * float(full_model.log_likelihood_ - base_model.log_likelihood_)
        lrt_df = max(1, len(subtype_cols))
        subtype_lrt_p = float(chi2.sf(max(lrt_stat, 0.0), df=lrt_df))

        global_rows.append(
            {
                "endpoint": endpoint,
                "n_samples": int(len(frame)),
                "n_events": int(frame[endpoint].sum()),
                "subtype_lrt_p": subtype_lrt_p,
                "subtype_df": int(lrt_df),
                "full_concordance": float(full_model.concordance_index_),
                "status": "ok",
            }
        )

        summary_table = full_model.summary.reset_index().rename(columns={"covariate": "term", "index": "term"})
        for _, row in summary_table.iterrows():
            detail_rows.append(
                {
                    "endpoint": endpoint,
                    "term": str(row["term"]),
                    "coef": float(row["coef"]),
                    "hazard_ratio": float(row["exp(coef)"]),
                    "p_value": float(row["p"]),
                    "coef_lower_95": float(row["coef lower 95%"]),
                    "coef_upper_95": float(row["coef upper 95%"]),
                    "hazard_ratio_lower_95": float(row["exp(coef) lower 95%"]),
                    "hazard_ratio_upper_95": float(row["exp(coef) upper 95%"]),
                    "n_samples": int(len(frame)),
                    "n_events": int(frame[endpoint].sum()),
                }
            )

    return pd.DataFrame(detail_rows), pd.DataFrame(global_rows)


def cluster_summary(
    labels: pd.Series,
    mechanism_features: pd.DataFrame,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    joined_features = mechanism_features.join(labels.rename("subtype"), how="inner")
    histology = clinical["histological_type"].fillna("")
    msi = clinical["msi_status"].fillna("")
    for subtype, feature_group in joined_features.groupby("subtype", sort=True):
        idx = feature_group.index
        rows.append(
            {
                "subtype": subtype,
                "n_samples": int(len(idx)),
                "tmb_mean": float(feature_group["tmb"].mean()),
                "mutation_burden_mean": float(feature_group["tmb"].mean()),
                "mutation_event_fraction_mean": float(feature_group["mutation_event_fraction"].mean()),
                "cnv_burden_mean": float(feature_group["cnv_burden"].mean()),
                "cnv_gain_burden_mean": float(feature_group["cnv_gain_burden"].mean()),
                "cnv_loss_burden_mean": float(feature_group["cnv_loss_burden"].mean()),
                "cnv_altered_fraction_mean": float(feature_group["cnv_altered_fraction"].mean()),
                "global_methylation_mean": float(feature_group["global_methylation"].mean()),
                "msi_h_rate": float((msi.loc[idx] == "MSI-H").mean()),
                "diffuse_histology_rate": float(histology.loc[idx].str.contains("Diffuse|Signet Ring", case=False, regex=True).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("subtype").reset_index(drop=True)


def tcga_like_mapping(summary: pd.DataFrame) -> pd.DataFrame:
    remaining = summary.copy()
    label_map: dict[str, str] = {}
    if not remaining.empty:
        msi_idx = remaining.sort_values(["msi_h_rate", "mutation_burden_mean"], ascending=[False, False]).iloc[0]["subtype"]
        label_map[msi_idx] = "MSI-like"
        remaining = remaining[remaining["subtype"] != msi_idx]
    if not remaining.empty:
        cin_idx = remaining.sort_values("cnv_burden_mean", ascending=False).iloc[0]["subtype"]
        label_map[cin_idx] = "CIN-like"
        remaining = remaining[remaining["subtype"] != cin_idx]
    if not remaining.empty:
        gs_idx = remaining.sort_values(["cnv_burden_mean", "diffuse_histology_rate"], ascending=[True, False]).iloc[0]["subtype"]
        label_map[gs_idx] = "GS-like"
        remaining = remaining[remaining["subtype"] != gs_idx]
    if not remaining.empty:
        ebv_idx = remaining.sort_values("global_methylation_mean", ascending=False).iloc[0]["subtype"]
        label_map[ebv_idx] = "EBV-like candidate"
        remaining = remaining[remaining["subtype"] != ebv_idx]
    summary = summary.copy()
    summary["tcga_like_label"] = summary["subtype"].map(label_map).fillna("Unassigned")
    return summary


def top_features_per_cluster(labels: pd.Series, omics: dict[str, pd.DataFrame], probe_map: pd.DataFrame) -> pd.DataFrame:
    records = []
    probe_map = probe_map.set_index("feature_id") if "feature_id" in probe_map.columns else probe_map.set_index("#id")
    for modality, frame in omics.items():
        for subtype in sorted(labels.unique()):
            in_cluster = frame.loc[:, labels == subtype]
            out_cluster = frame.loc[:, labels != subtype]
            score = in_cluster.mean(axis=1) - out_cluster.mean(axis=1)
            top = score.abs().sort_values(ascending=False).head(10)
            for feature_id, abs_score in top.items():
                row = {
                    "modality": modality,
                    "subtype": subtype,
                    "feature_id": feature_id,
                    "effect_size": float(score.loc[feature_id]),
                    "abs_effect_size": float(abs_score),
                }
                if modality == "methylation" and feature_id in probe_map.index:
                    gene = probe_map.loc[feature_id].get("gene", ".")
                    row["annotation"] = gene
                else:
                    row["annotation"] = feature_id
                records.append(row)
    return pd.DataFrame(records).sort_values(["subtype", "modality", "abs_effect_size"], ascending=[True, True, False])


def subtype_interpretation_table(
    subtype_table: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    survival_table: pd.DataFrame,
    top_features: pd.DataFrame,
) -> pd.DataFrame:
    subtype_table = subtype_table.copy()
    interpretation_rows = []
    survival_lookup = {
        row["endpoint"]: row for _, row in survival_table.iterrows()
    }
    merged = subtype_table.merge(cluster_summary_df, on="subtype", how="left")

    for subtype, group in merged.groupby("subtype", sort=True):
        row = {
            "subtype": subtype,
            "tcga_like_label": group["tcga_like_label"].iloc[0],
            "n_samples": int(group["n_samples"].iloc[0]),
            "dominant_msi_status": _mode_or_na(group["msi_status"]),
            "dominant_pathologic_stage": _mode_or_na(group["pathologic_stage"]),
            "dominant_grade": _mode_or_na(group["grade"]),
            "dominant_histological_type": _mode_or_na(group["histological_type"]),
            "median_age": float(pd.to_numeric(group["age"], errors="coerce").median()),
            "msi_h_rate": float(group["msi_h_rate"].iloc[0]),
            "diffuse_histology_rate": float(group["diffuse_histology_rate"].iloc[0]),
            "tmb_mean": float(group["tmb_mean"].iloc[0]),
            "mutation_burden_mean": float(group["mutation_burden_mean"].iloc[0]),
            "mutation_event_fraction_mean": float(group["mutation_event_fraction_mean"].iloc[0]),
            "cnv_burden_mean": float(group["cnv_burden_mean"].iloc[0]),
            "cnv_gain_burden_mean": float(group["cnv_gain_burden_mean"].iloc[0]),
            "cnv_loss_burden_mean": float(group["cnv_loss_burden_mean"].iloc[0]),
            "cnv_altered_fraction_mean": float(group["cnv_altered_fraction_mean"].iloc[0]),
            "global_methylation_mean": float(group["global_methylation_mean"].iloc[0]),
        }
        for endpoint in ["OS", "DSS", "DFI", "PFI"]:
            endpoint_row = survival_lookup.get(endpoint)
            row[f"{endpoint.lower()}_median_time"] = (
                float(endpoint_row.get(f"{subtype}_median_time", np.nan)) if endpoint_row is not None else np.nan
            )

        subtype_features = top_features[top_features["subtype"] == subtype].copy()
        for modality in sorted(subtype_features["modality"].unique()):
            subset = subtype_features[subtype_features["modality"] == modality].copy()
            subset["display_name"] = subset["annotation"].fillna(subset["feature_id"]).astype(str)
            subset["display_name"] = subset["display_name"].replace({".": pd.NA}).fillna(subset["feature_id"].astype(str))
            subset["marker"] = subset.apply(
                lambda x: f"{x['display_name']} ({'up' if x['effect_size'] >= 0 else 'down'})",
                axis=1,
            )
            row[f"{modality}_markers"] = "; ".join(subset["marker"].head(5))
        interpretation_rows.append(row)

    out = pd.DataFrame(interpretation_rows).sort_values("subtype").reset_index(drop=True)
    return out


def subtype_interpretation_markdown(
    interpretation_table: pd.DataFrame,
    clinical_table: pd.DataFrame,
    survival_table: pd.DataFrame,
    analysis_name: str,
) -> str:
    lines = [
        f"# {analysis_name} subtype interpretation",
        "",
        "## Global association summary",
    ]
    for _, row in clinical_table.iterrows():
        p_value = row["p_value"]
        p_text = "NA" if pd.isna(p_value) else f"{p_value:.4g}"
        lines.append(f"- {row['variable']}: {row['test']} p={p_text} (n={int(row['n'])})")

    lines.append("")
    lines.append("## Survival summary")
    for _, row in survival_table.iterrows():
        p_value = row["logrank_p"]
        p_text = "NA" if pd.isna(p_value) else f"{p_value:.4g}"
        lines.append(f"- {row['endpoint']}: log-rank p={p_text} (n={int(row['n_samples'])})")

    lines.append("")
    lines.append("## Subtype details")
    for _, row in interpretation_table.iterrows():
        lines.append(f"### {row['subtype']} - {row['tcga_like_label']}")
        lines.append(
            f"- Samples: {int(row['n_samples'])}; dominant MSI: {row['dominant_msi_status']}; "
            f"dominant stage: {row['dominant_pathologic_stage']}; dominant grade: {row['dominant_grade']}"
        )
        lines.append(
            f"- Clinical profile: MSI-H rate {row['msi_h_rate']:.1%}, diffuse histology rate {row['diffuse_histology_rate']:.1%}, "
            f"median age {row['median_age']:.1f}"
        )
        lines.append(
            f"- Molecular profile: TMB {row['tmb_mean']:.1f}, "
            f"CNV burden {row['cnv_burden_mean']:.1f} "
            f"(gain {row['cnv_gain_burden_mean']:.1f}, loss {row['cnv_loss_burden_mean']:.1f}), "
            f"global methylation {row['global_methylation_mean']:.3f}"
        )
        lines.append(
            f"- Survival: median OS {row['os_median_time']:.1f}, median DSS {row['dss_median_time']:.1f}, "
            f"median DFI {row['dfi_median_time']:.1f}, median PFI {row['pfi_median_time']:.1f}"
        )
        for modality in ["mutation", "cnv", "methylation", "expression", "mirna", "rppa"]:
            col = f"{modality}_markers"
            if col in row and isinstance(row[col], str) and row[col]:
                lines.append(f"- {modality} markers: {row[col]}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_km_plot(labels: pd.Series, survival: pd.DataFrame, output_path: Path, endpoint: str, title: str) -> None:
    time_col = f"{endpoint}.time"
    frame = survival[[endpoint, time_col]].join(labels, how="inner").dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    for subtype, group in frame.groupby("subtype"):
        kmf.fit(group[time_col], event_observed=group[endpoint], label=subtype)
        kmf.plot_survival_function(ax=ax, ci_show=False)
    if frame["subtype"].nunique() >= 2:
        result = multivariate_logrank_test(frame[time_col], frame["subtype"], frame[endpoint])
        p_text = f"log-rank p={result.p_value:.3g}"
    else:
        p_text = "log-rank p=NA"
    ax.set_title(f"{title}\n{p_text}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_clinical_stacked_plot(labels: pd.Series, clinical: pd.DataFrame, output_path: Path) -> None:
    frame = clinical.join(labels, how="inner")
    counts = pd.crosstab(frame["subtype"], frame["msi_status"].fillna("NA"))
    counts = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Proportion")
    ax.set_title("MSI status distribution by subtype")
    ax.legend(title="MSI")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_heatmap(
    labels: pd.Series,
    omics: dict[str, pd.DataFrame],
    top_features: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_blocks = []
    ordered_samples = labels.sort_values().index.tolist()
    for modality, frame in omics.items():
        features = top_features[top_features["modality"] == modality]["feature_id"].drop_duplicates().head(10)
        if features.empty:
            continue
        block = frame.loc[frame.index.intersection(features), ordered_samples]
        selected_blocks.append(block)
    if not selected_blocks:
        return
    heatmap_df = pd.concat(selected_blocks, axis=0)
    fig_h = max(6, heatmap_df.shape[0] * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.heatmap(heatmap_df, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Scaled value"})
    ax.set_title("Integrated subtype heatmap")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_sensitivity_plot(
    left_labels: pd.Series,
    right_labels: pd.Series,
    left_best_k: int,
    right_best_k: int,
    ari: float,
    output_path: Path,
    left_label: str = "Primary",
    right_label: str = "Comparison",
    title: str = "Queue comparison",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    left_labels.value_counts().sort_index().plot(kind="bar", ax=axes[0], title=f"{left_label} (K={left_best_k})")
    right_labels.value_counts().sort_index().plot(kind="bar", ax=axes[1], title=f"{right_label} (K={right_best_k})")
    for ax in axes:
        ax.set_ylabel("Samples")
    fig.suptitle(f"{title}, ARI={ari:.3f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
