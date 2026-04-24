from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import ari_between, consensus_clustering
from .cluster import label_comparison_metrics
from .consensus import (
    build_copy_rescued_weight_tables,
    build_consensus_weight_table,
    build_consensus_weight_tables,
    consensus_candidate_summary_frame,
    consensus_summary_frame,
    select_consensus_scheme,
    weighted_consensus_matrix,
)
from .config import (
    ANALYSIS_QUEUES,
    CONSENSUS_FRACTION,
    SNF_K,
    SENSITIVITY_SEED_OFFSETS,
    SENSITIVITY_SNF_K_VALUES,
    AnalysisQueueConfig,
)
from .evaluate import (
    build_subtype_table,
    clinical_associations,
    cluster_summary,
    multivariable_cox,
    save_clinical_stacked_plot,
    save_heatmap,
    save_km_plot,
    save_sensitivity_plot,
    survival_stats,
    tcga_like_mapping,
    subtype_interpretation_markdown,
    subtype_interpretation_table,
    top_features_per_cluster,
)
from .features import compute_mechanism_summary_features, feature_layer_manifest, mechanism_summary_feature_metadata
from .io import LoadedTables, load_tables_for_analysis, write_qc_json
from .models import (
    build_early_integration_embedding,
    build_nmf_integration_embedding,
    early_integration_consensus_clustering,
    embedding_consensus_clustering,
)
from .preprocess import preprocess_modality, reduce_modality
from .snf import build_affinity_matrix, snf
from .signature import (
    build_final_panel_candidates,
    build_frozen_panel_reduction_candidates,
    build_final_panel_selection,
    build_reduced_panel_selection,
    build_refined_panel_freeze_candidates,
    build_refined_raw_marker_panels,
    build_raw_proxy_candidate_scores,
    build_signature_feature_evidence,
    extract_selected_final_panel_features,
    extract_frozen_refined_panel_features,
    final_signature_panel_summary_frame,
    final_signature_shortlist_markdown,
    frozen_refined_signature_markdown,
    build_latent_component_marker_table,
    load_mirbase_mature_aliases,
    build_signature_panels,
    build_raw_marker_panels,
    build_raw_panel_feature_matrix,
    raw_signature_panel_summary_frame,
    reduced_signature_shortlist_markdown,
    reduction_signature_panel_summary_frame,
    build_final_signature_interpretation,
    collect_final_signature_classifier_evidence,
    final_signature_interpretation_markdown,
    raw_signature_shortlist_markdown,
    signature_panel_summary_frame,
    signature_shortlist_markdown,
    write_signature_readme,
)
from .supervised import (
    CLASSIFIER_NAMES,
    CLASSIFIER_EXPORTERS,
    CLASSIFIER_RUNNERS,
    SIGNATURE_SOURCE_CLASSIFIER_NAMES,
    build_classifier_feature_matrix,
    export_classifier_results,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class PreparedAnalysis:
    loaded: LoadedTables
    preprocessed: dict[str, pd.DataFrame]
    embeddings: dict[str, pd.DataFrame]
    qc: dict


def prepare_analysis_inputs(
    data_dir: Path,
    modality_names: list[str] | tuple[str, ...],
    analysis_name: str,
    fixed_patients: list[str] | None = None,
) -> PreparedAnalysis:
    print(f"[{analysis_name}] loading tables", flush=True)
    loaded = load_tables_for_analysis(data_dir, modality_names, required_patients=fixed_patients)
    preprocessed = {}
    embeddings = {}
    qc = loaded.qc | {"analysis_name": analysis_name, "modalities": modality_names}

    for name, frame in loaded.omics.items():
        print(f"[{analysis_name}] preprocessing {name}", flush=True)
        processed, process_qc = preprocess_modality(name, frame)
        preprocessed[name] = processed
        embeddings[name] = reduce_modality(name, processed)
        qc.setdefault("preprocessing", {})[name] = process_qc
        qc.setdefault("embedding_dims", {})[name] = int(embeddings[name].shape[1])
    return PreparedAnalysis(loaded=loaded, preprocessed=preprocessed, embeddings=embeddings, qc=qc)


def evaluate_analysis_tables(
    labels: pd.Series,
    loaded: LoadedTables,
    analysis_name: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subtype_table = build_subtype_table(labels, loaded.clinical, loaded.survival, analysis_name)
    subtype_table.insert(1, "model_name", model_name)
    survival_table = survival_stats(labels, loaded.survival)
    clinical_table = clinical_associations(labels, loaded.clinical)
    cox_detail_table, cox_global_table = multivariable_cox(subtype_table)
    return subtype_table, survival_table, clinical_table, cox_detail_table, cox_global_table


def fit_snf_analysis(
    prepared: PreparedAnalysis,
    seed: int,
    snf_k: int = SNF_K,
    consensus_fraction: float = CONSENSUS_FRACTION,
) -> tuple[pd.DataFrame, int, pd.Series]:
    affinities = [build_affinity_matrix(embedding, k=snf_k) for embedding in prepared.embeddings.values()]
    fused = snf(affinities, k=snf_k)
    return consensus_clustering(fused, sample_fraction=consensus_fraction, random_state=seed)


def make_sensitivity_variant_record(
    config: AnalysisQueueConfig,
    baseline_result: dict,
    variant_group: str,
    variant_name: str,
    modalities: tuple[str, ...],
    seed: int,
    snf_k: int,
    fixed_sample_cohort: bool,
    labels: pd.Series,
    k_summary: pd.DataFrame,
    best_k: int,
    clinical_table: pd.DataFrame,
    survival_table: pd.DataFrame,
    cox_global_table: pd.DataFrame,
) -> dict[str, object]:
    selected = k_summary.loc[k_summary["selected"]].iloc[0]
    clinical_lookup = clinical_table.set_index("variable") if not clinical_table.empty else pd.DataFrame()
    survival_lookup = survival_table.set_index("endpoint") if not survival_table.empty else pd.DataFrame()
    cox_lookup = cox_global_table.set_index("endpoint") if not cox_global_table.empty else pd.DataFrame()
    baseline_labels = baseline_result["labels"]
    comparison = label_comparison_metrics(baseline_labels, labels)
    return {
        "queue_key": config.key,
        "analysis_name": config.analysis_name,
        "variant_group": variant_group,
        "variant_name": variant_name,
        "modalities": ",".join(modalities),
        "n_modalities": int(len(modalities)),
        "random_seed": int(seed),
        "snf_k": int(snf_k),
        "fixed_sample_cohort": bool(fixed_sample_cohort),
        "baseline_samples": int(len(baseline_labels)),
        "n_samples": int(len(labels)),
        "sample_delta_vs_baseline": int(len(labels) - len(baseline_labels)),
        "best_k": int(best_k),
        "selected_pac_score": float(selected["pac_score"]),
        "selected_silhouette": float(selected["silhouette"]),
        "selected_min_cluster_size": int(selected["min_cluster_size"]),
        "msi_status_p": float(clinical_lookup.loc["msi_status", "p_value"]) if "msi_status" in clinical_lookup.index else np.nan,
        "age_p": float(clinical_lookup.loc["age", "p_value"]) if "age" in clinical_lookup.index else np.nan,
        "os_logrank_p": float(survival_lookup.loc["OS", "logrank_p"]) if "OS" in survival_lookup.index else np.nan,
        "dss_logrank_p": float(survival_lookup.loc["DSS", "logrank_p"]) if "DSS" in survival_lookup.index else np.nan,
        "os_subtype_cox_p": float(cox_lookup.loc["OS", "subtype_lrt_p"]) if "OS" in cox_lookup.index else np.nan,
        "dss_subtype_cox_p": float(cox_lookup.loc["DSS", "subtype_lrt_p"]) if "DSS" in cox_lookup.index else np.nan,
        "ari_to_baseline": float(comparison["ari"]),
        "nmi_to_baseline": float(comparison["nmi"]),
        "jaccard_to_baseline": float(comparison["co_clustering_jaccard"]),
    }


def sensitivity_overview_frame(variant_summary: pd.DataFrame) -> pd.DataFrame:
    if variant_summary.empty:
        return pd.DataFrame(
            columns=[
                "queue_key",
                "analysis_name",
                "variant_group",
                "n_variants",
                "mean_ari_to_baseline",
                "min_ari_to_baseline",
                "mean_nmi_to_baseline",
                "mean_jaccard_to_baseline",
                "same_best_k_rate",
                "max_abs_sample_delta",
                "most_stable_variant",
                "least_stable_variant",
            ]
        )

    baseline_lookup = (
        variant_summary[variant_summary["variant_group"] == "baseline"]
        .set_index(["queue_key", "analysis_name"])["best_k"]
        .to_dict()
    )
    records: list[dict[str, object]] = []
    grouped = variant_summary[variant_summary["variant_group"] != "baseline"].groupby(["queue_key", "analysis_name", "variant_group"], sort=True)
    for (queue_key, analysis_name, variant_group), group in grouped:
        baseline_best_k = baseline_lookup.get((queue_key, analysis_name))
        worst_idx = group["ari_to_baseline"].astype(float).idxmin()
        best_idx = group["ari_to_baseline"].astype(float).idxmax()
        records.append(
            {
                "queue_key": queue_key,
                "analysis_name": analysis_name,
                "variant_group": variant_group,
                "n_variants": int(len(group)),
                "mean_ari_to_baseline": float(group["ari_to_baseline"].mean()),
                "min_ari_to_baseline": float(group["ari_to_baseline"].min()),
                "mean_nmi_to_baseline": float(group["nmi_to_baseline"].mean()),
                "mean_jaccard_to_baseline": float(group["jaccard_to_baseline"].mean()),
                "same_best_k_rate": float((group["best_k"] == baseline_best_k).mean()) if baseline_best_k is not None else np.nan,
                "max_abs_sample_delta": int(group["sample_delta_vs_baseline"].abs().max()),
                "most_stable_variant": str(group.loc[best_idx, "variant_name"]),
                "least_stable_variant": str(group.loc[worst_idx, "variant_name"]),
            }
        )
    return pd.DataFrame(records)


def export_analysis_outputs(
    prepared: PreparedAnalysis,
    out_dir: Path,
    analysis_name: str,
    model_name: str,
    labels: pd.Series,
    k_summary: pd.DataFrame,
    best_k: int,
    extra_latent_artifacts: dict[str, pd.DataFrame] | None = None,
) -> dict:
    ensure_dir(out_dir)
    figures_dir = out_dir / "figures"
    ensure_dir(figures_dir)
    feature_cache_dir = out_dir / "feature_cache"
    ensure_dir(feature_cache_dir)

    loaded = prepared.loaded
    mechanism_features = compute_mechanism_summary_features(loaded.omics)
    mechanism_feature_metadata = mechanism_summary_feature_metadata(loaded.omics)
    mechanism_feature_metadata = mechanism_feature_metadata[
        mechanism_feature_metadata["feature_name"].isin(mechanism_features.columns)
    ].reset_index(drop=True)
    layer_manifest = feature_layer_manifest(
        prepared.preprocessed,
        prepared.embeddings,
        mechanism_feature_metadata,
        extra_latent_artifacts=extra_latent_artifacts,
    )
    subtype_table, survival_table, clinical_table, cox_detail_table, cox_global_table = evaluate_analysis_tables(
        labels,
        loaded,
        analysis_name,
        model_name,
    )
    summary = cluster_summary(labels, mechanism_features, loaded.clinical)
    summary = tcga_like_mapping(summary)
    top_features = top_features_per_cluster(labels, prepared.preprocessed, loaded.probe_map)
    interpretation_table = subtype_interpretation_table(subtype_table, summary, survival_table, top_features)
    interpretation_markdown = subtype_interpretation_markdown(
        interpretation_table, clinical_table, survival_table, f"{analysis_name} [{model_name}]"
    )

    subtype_table.to_csv(out_dir / "subtypes.tsv", sep="\t", index=False)
    k_summary.to_csv(out_dir / "k_selection.tsv", sep="\t", index=False)
    survival_table.to_csv(out_dir / "survival_stats.tsv", sep="\t", index=False)
    clinical_table.to_csv(out_dir / "clinical_assoc.tsv", sep="\t", index=False)
    cox_detail_table.to_csv(out_dir / "cox_multivariable.tsv", sep="\t", index=False)
    cox_global_table.to_csv(out_dir / "cox_multivariable_global.tsv", sep="\t", index=False)
    mechanism_features.to_csv(out_dir / "mechanism_summary_features.tsv", sep="\t", index=True)
    mechanism_feature_metadata.to_csv(out_dir / "mechanism_summary_feature_metadata.tsv", sep="\t", index=False)
    layer_manifest.to_csv(out_dir / "feature_layer_manifest.tsv", sep="\t", index=False)
    top_features.to_csv(out_dir / "top_features.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "tcga_like_mapping.tsv", sep="\t", index=False)
    interpretation_table.to_csv(out_dir / "subtype_interpretation.tsv", sep="\t", index=False)
    (out_dir / "subtype_interpretation.md").write_text(interpretation_markdown, encoding="utf-8")
    for modality, embedding in prepared.embeddings.items():
        embedding.to_csv(feature_cache_dir / f"{modality}_embedding.tsv", sep="\t", index=True)
    if extra_latent_artifacts:
        for artifact_name, artifact in extra_latent_artifacts.items():
            artifact.to_csv(feature_cache_dir / f"{artifact_name}.tsv", sep="\t", index=True)

    plot_prefix = f"{analysis_name} [{model_name}]"
    save_km_plot(labels, loaded.survival, figures_dir / "km_os.png", endpoint="OS", title=f"{plot_prefix} OS")
    save_km_plot(labels, loaded.survival, figures_dir / "km_dss.png", endpoint="DSS", title=f"{plot_prefix} DSS")
    save_km_plot(labels, loaded.survival, figures_dir / "km_dfi.png", endpoint="DFI", title=f"{plot_prefix} DFI")
    save_km_plot(labels, loaded.survival, figures_dir / "km_pfi.png", endpoint="PFI", title=f"{plot_prefix} PFI")
    save_clinical_stacked_plot(labels, loaded.clinical, figures_dir / "clinical_msi_stacked.png")
    save_heatmap(labels, prepared.preprocessed, top_features, figures_dir / "integrated_heatmap.png")

    qc = prepared.qc.copy()
    qc["model_name"] = model_name
    qc["best_k"] = best_k
    qc["cluster_sizes"] = labels.value_counts().sort_index().to_dict()
    qc["analysis_samples"] = int(labels.shape[0])
    qc["mechanism_summary_features"] = int(mechanism_features.shape[1])
    qc["latent_embedding_exports"] = {modality: int(embedding.shape[1]) for modality, embedding in prepared.embeddings.items()}
    if extra_latent_artifacts:
        qc["extra_latent_artifacts"] = {name: int(frame.shape[1]) for name, frame in extra_latent_artifacts.items()}
    selected_row = k_summary.loc[k_summary["selected"]].iloc[0]
    qc["selection_fallback_used"] = not bool(selected_row["is_valid"])
    write_qc_json(out_dir / "qc_summary.json", qc)

    return {
        "labels": labels,
        "best_k": best_k,
        "k_summary": k_summary,
        "model_name": model_name,
        "clinical_table": clinical_table,
        "survival_table": survival_table,
        "cox_global_table": cox_global_table,
        "qc": qc,
        "out_dir": out_dir,
    }


def export_consensus_candidate_artifacts(
    out_dir: Path,
    labels: pd.Series,
    k_summary: pd.DataFrame,
    consensus_matrix: pd.DataFrame,
    weight_table: pd.DataFrame,
    scheme_name: str,
    analysis_name: str,
) -> None:
    ensure_dir(out_dir)
    labels.rename("subtype").to_csv(out_dir / "subtypes.tsv", sep="\t", index=True, index_label="patient_id")
    k_summary.to_csv(out_dir / "k_selection.tsv", sep="\t", index=False)
    consensus_matrix.to_csv(out_dir / "consensus_matrix.tsv", sep="\t", index=True)
    weight_table.to_csv(out_dir / "weights.tsv", sep="\t", index=False)
    selected_row = k_summary.loc[k_summary["selected"]].iloc[0]
    qc = {
        "analysis_name": analysis_name,
        "scheme_name": scheme_name,
        "best_k": int(selected_row["k"]),
        "cluster_sizes": labels.value_counts().sort_index().to_dict(),
        "n_samples": int(len(labels)),
        "dominant_model": (
            weight_table.sort_values("normalized_weight", ascending=False).iloc[0]["model_name"]
            if not weight_table.empty
            else None
        ),
        "dominant_model_weight": (
            float(weight_table["normalized_weight"].max())
            if not weight_table.empty
            else float("nan")
        ),
    }
    write_qc_json(out_dir / "qc_summary.json", qc)


def run_consensus_candidate_batch(
    out_dir: Path,
    queue_prepared: dict[str, PreparedAnalysis],
    queue_model_results: list[tuple[str, dict]],
    consensus_weight_table: pd.DataFrame,
    seed: int,
) -> list[dict]:
    candidate_results: list[dict] = []
    for scheme_name in consensus_weight_table["scheme_name"].drop_duplicates().tolist():
        for config in ANALYSIS_QUEUES:
            queue_weight_table = consensus_weight_table[
                (consensus_weight_table["queue_key"] == config.key)
                & (consensus_weight_table["scheme_name"] == scheme_name)
            ].copy()
            if queue_weight_table.empty:
                continue
            queue_results = [result for queue_key, result in queue_model_results if queue_key == config.key]
            consensus_matrix = weighted_consensus_matrix(queue_results, queue_weight_table)
            consensus_candidate_dir = out_dir / scheme_name / config.key
            consensus_k_summary, consensus_best_k, consensus_labels = consensus_clustering(consensus_matrix, random_state=seed)
            export_consensus_candidate_artifacts(
                consensus_candidate_dir,
                consensus_labels,
                consensus_k_summary,
                consensus_matrix,
                queue_weight_table,
                scheme_name,
                config.analysis_name,
            )
            _, candidate_survival, candidate_clinical, _, _ = evaluate_analysis_tables(
                consensus_labels,
                queue_prepared[config.key].loaded,
                config.analysis_name,
                f"consensus_{scheme_name}",
            )
            candidate_results.append(
                {
                    "queue_key": config.key,
                    "analysis_name": config.analysis_name,
                    "scheme_name": scheme_name,
                    "labels": consensus_labels,
                    "best_k": consensus_best_k,
                    "k_summary": consensus_k_summary,
                    "clinical_table": candidate_clinical,
                    "survival_table": candidate_survival,
                    "out_dir": consensus_candidate_dir,
                    "consensus_matrix": consensus_matrix,
                }
            )
    return candidate_results


def run_single_analysis(
    data_dir: Path,
    out_dir: Path,
    modality_names: list[str] | tuple[str, ...],
    analysis_name: str,
    seed: int,
) -> dict:
    prepared = prepare_analysis_inputs(data_dir, modality_names, analysis_name)
    print(f"[{analysis_name}] building fused network", flush=True)
    k_summary, best_k, labels = fit_snf_analysis(prepared, seed=seed)
    print(f"[{analysis_name}] evaluating outputs", flush=True)
    result = export_analysis_outputs(prepared, out_dir, analysis_name, "snf", labels, k_summary, best_k)
    print(f"[{analysis_name}] finished", flush=True)
    return result


def run_early_integration_analysis(
    prepared: PreparedAnalysis,
    out_dir: Path,
    analysis_name: str,
    seed: int,
) -> dict:
    print(f"[{analysis_name}] early integration model", flush=True)
    integrated_embedding = build_early_integration_embedding(prepared.embeddings)
    k_summary, best_k, labels = early_integration_consensus_clustering(integrated_embedding, random_state=seed)
    result = export_analysis_outputs(
        prepared,
        out_dir,
        analysis_name,
        "early_integration",
        labels,
        k_summary,
        best_k,
        extra_latent_artifacts={"integrated_embedding": integrated_embedding},
    )
    print(f"[{analysis_name}] early integration finished", flush=True)
    return result


def run_nmf_integration_analysis(
    prepared: PreparedAnalysis,
    out_dir: Path,
    analysis_name: str,
    seed: int,
) -> dict:
    print(f"[{analysis_name}] nmf integration model", flush=True)
    nmf_embedding = build_nmf_integration_embedding(prepared.embeddings)
    k_summary, best_k, labels = embedding_consensus_clustering(nmf_embedding, random_state=seed)
    result = export_analysis_outputs(
        prepared,
        out_dir,
        analysis_name,
        "nmf_integration",
        labels,
        k_summary,
        best_k,
        extra_latent_artifacts={"nmf_embedding": nmf_embedding},
    )
    print(f"[{analysis_name}] nmf integration finished", flush=True)
    return result


def queue_summary_frame(results: list[tuple[AnalysisQueueConfig, dict]]) -> pd.DataFrame:
    rows = []
    reference_labels = results[0][1]["labels"] if results else pd.Series(dtype=object)
    for config, result in results:
        rows.append(
            {
                "queue_key": config.key,
                "analysis_name": config.analysis_name,
                "out_subdir": config.out_subdir or ".",
                "n_modalities": len(config.modalities),
                "modalities": ",".join(config.modalities),
                "description": config.description,
                "best_k": result["best_k"],
                "n_samples": int(len(result["labels"])),
                "ari_to_primary_discovery": (
                    float(ari_between(reference_labels, result["labels"])) if not reference_labels.empty else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def model_comparison_frame(
    results: list[tuple[AnalysisQueueConfig, dict]],
    reference_labels: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for config, result in results:
        selected = result["k_summary"].loc[result["k_summary"]["selected"]].iloc[0]
        reference = reference_labels.get(config.key)
        rows.append(
            {
                "queue_key": config.key,
                "analysis_name": config.analysis_name,
                "model_name": result["model_name"],
                "best_k": result["best_k"],
                "n_samples": int(len(result["labels"])),
                "selected_pac_score": float(selected["pac_score"]),
                "selected_silhouette": float(selected["silhouette"]),
                "selected_min_cluster_size": int(selected["min_cluster_size"]),
                "ari_to_snf_same_queue": (
                    label_comparison_metrics(reference, result["labels"])["ari"]
                    if reference is not None and result["model_name"] != "snf"
                    else 1.0
                ),
                "nmi_to_snf_same_queue": (
                    label_comparison_metrics(reference, result["labels"])["nmi"]
                    if reference is not None and result["model_name"] != "snf"
                    else 1.0
                ),
                "jaccard_to_snf_same_queue": (
                    label_comparison_metrics(reference, result["labels"])["co_clustering_jaccard"]
                    if reference is not None and result["model_name"] != "snf"
                    else 1.0
                ),
                "output_dir": str(result["out_dir"]),
            }
        )
    return pd.DataFrame(rows)


def model_stability_summary_frame(results: list[tuple[AnalysisQueueConfig, dict]]) -> pd.DataFrame:
    by_model: dict[str, dict[str, dict]] = {}
    for config, result in results:
        by_model.setdefault(result["model_name"], {})[config.key] = result

    rows = []
    for model_name, model_results in sorted(by_model.items()):
        primary = model_results.get("primary_discovery")
        validation = model_results.get("validation_with_rppa")
        if primary is None or validation is None:
            continue
        metrics = label_comparison_metrics(primary["labels"], validation["labels"])
        rows.append(
            {
                "model_name": model_name,
                "primary_best_k": int(primary["best_k"]),
                "validation_best_k": int(validation["best_k"]),
                "n_overlap": int(metrics["n_overlap"]),
                "overlap_ari": float(metrics["ari"]),
                "overlap_nmi": float(metrics["nmi"]),
                "overlap_co_clustering_jaccard": float(metrics["co_clustering_jaccard"]),
            }
        )
    return pd.DataFrame(rows)


def model_evaluation_summary_frame(results: list[tuple[AnalysisQueueConfig, dict]]) -> pd.DataFrame:
    rows = []
    for config, result in results:
        comparison_selected = result["k_summary"].loc[result["k_summary"]["selected"]].iloc[0]
        clinical_lookup = result["clinical_table"].set_index("variable") if not result["clinical_table"].empty else pd.DataFrame()
        survival_lookup = result["survival_table"].set_index("endpoint") if not result["survival_table"].empty else pd.DataFrame()
        cox_lookup = result["cox_global_table"].set_index("endpoint") if not result["cox_global_table"].empty else pd.DataFrame()
        rows.append(
            {
                "queue_key": config.key,
                "analysis_name": config.analysis_name,
                "model_name": result["model_name"],
                "best_k": int(result["best_k"]),
                "n_samples": int(len(result["labels"])),
                "selected_pac_score": float(comparison_selected["pac_score"]),
                "selected_silhouette": float(comparison_selected["silhouette"]),
                "selected_min_cluster_size": int(comparison_selected["min_cluster_size"]),
                "msi_status_p": float(clinical_lookup.loc["msi_status", "p_value"]) if "msi_status" in clinical_lookup.index else np.nan,
                "age_p": float(clinical_lookup.loc["age", "p_value"]) if "age" in clinical_lookup.index else np.nan,
                "os_logrank_p": float(survival_lookup.loc["OS", "logrank_p"]) if "OS" in survival_lookup.index else np.nan,
                "dss_logrank_p": float(survival_lookup.loc["DSS", "logrank_p"]) if "DSS" in survival_lookup.index else np.nan,
                "os_subtype_cox_p": float(cox_lookup.loc["OS", "subtype_lrt_p"]) if "OS" in cox_lookup.index else np.nan,
                "dss_subtype_cox_p": float(cox_lookup.loc["DSS", "subtype_lrt_p"]) if "DSS" in cox_lookup.index else np.nan,
                "full_cox_concordance_os": float(cox_lookup.loc["OS", "full_concordance"]) if "OS" in cox_lookup.index else np.nan,
                "output_dir": str(result["out_dir"]),
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity_analyses(
    data_dir: Path,
    queue_prepared: dict[str, PreparedAnalysis],
    baseline_results: list[tuple[AnalysisQueueConfig, dict]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for config, baseline_result in baseline_results:
        baseline_prepared = queue_prepared[config.key]
        baseline_labels = baseline_result["labels"]
        _, baseline_survival, baseline_clinical, _, baseline_cox = evaluate_analysis_tables(
            baseline_labels,
            baseline_prepared.loaded,
            config.analysis_name,
            "snf",
        )
        records.append(
            make_sensitivity_variant_record(
                config=config,
                baseline_result=baseline_result,
                variant_group="baseline",
                variant_name="baseline",
                modalities=config.modalities,
                seed=seed,
                snf_k=SNF_K,
                fixed_sample_cohort=True,
                labels=baseline_labels,
                k_summary=baseline_result["k_summary"],
                best_k=baseline_result["best_k"],
                clinical_table=baseline_clinical,
                survival_table=baseline_survival,
                cox_global_table=baseline_cox,
            )
        )

        for seed_offset in SENSITIVITY_SEED_OFFSETS:
            variant_seed = seed + seed_offset
            variant_name = f"seed_{variant_seed}"
            k_summary, best_k, labels = fit_snf_analysis(baseline_prepared, seed=variant_seed)
            _, survival_table, clinical_table, _, cox_global_table = evaluate_analysis_tables(
                labels,
                baseline_prepared.loaded,
                config.analysis_name,
                "snf",
            )
            records.append(
                make_sensitivity_variant_record(
                    config=config,
                    baseline_result=baseline_result,
                    variant_group="seed",
                    variant_name=variant_name,
                    modalities=config.modalities,
                    seed=variant_seed,
                    snf_k=SNF_K,
                    fixed_sample_cohort=True,
                    labels=labels,
                    k_summary=k_summary,
                    best_k=best_k,
                    clinical_table=clinical_table,
                    survival_table=survival_table,
                    cox_global_table=cox_global_table,
                )
            )

        for variant_snf_k in SENSITIVITY_SNF_K_VALUES:
            variant_name = f"snf_k_{variant_snf_k}"
            k_summary, best_k, labels = fit_snf_analysis(baseline_prepared, seed=seed, snf_k=variant_snf_k)
            _, survival_table, clinical_table, _, cox_global_table = evaluate_analysis_tables(
                labels,
                baseline_prepared.loaded,
                config.analysis_name,
                "snf",
            )
            records.append(
                make_sensitivity_variant_record(
                    config=config,
                    baseline_result=baseline_result,
                    variant_group="snf_k",
                    variant_name=variant_name,
                    modalities=config.modalities,
                    seed=seed,
                    snf_k=variant_snf_k,
                    fixed_sample_cohort=True,
                    labels=labels,
                    k_summary=k_summary,
                    best_k=best_k,
                    clinical_table=clinical_table,
                    survival_table=survival_table,
                    cox_global_table=cox_global_table,
                )
            )

        fixed_patients = baseline_labels.index.tolist()
        for dropped_modality in config.modalities:
            ablated_modalities = tuple(modality for modality in config.modalities if modality != dropped_modality)
            variant_name = f"drop_{dropped_modality}"
            variant_analysis_name = f"{config.analysis_name}__{variant_name}"
            print(f"[{config.analysis_name}] sensitivity {variant_name}", flush=True)
            ablated_prepared = prepare_analysis_inputs(
                data_dir,
                ablated_modalities,
                variant_analysis_name,
                fixed_patients=fixed_patients,
            )
            k_summary, best_k, labels = fit_snf_analysis(ablated_prepared, seed=seed)
            _, survival_table, clinical_table, _, cox_global_table = evaluate_analysis_tables(
                labels,
                ablated_prepared.loaded,
                config.analysis_name,
                "snf",
            )
            records.append(
                make_sensitivity_variant_record(
                    config=config,
                    baseline_result=baseline_result,
                    variant_group="modality_ablation",
                    variant_name=variant_name,
                    modalities=ablated_modalities,
                    seed=seed,
                    snf_k=SNF_K,
                    fixed_sample_cohort=True,
                    labels=labels,
                    k_summary=k_summary,
                    best_k=best_k,
                    clinical_table=clinical_table,
                    survival_table=survival_table,
                    cox_global_table=cox_global_table,
                )
            )

    variant_summary = pd.DataFrame(records)
    overview = sensitivity_overview_frame(variant_summary)
    return variant_summary, overview


def run_full_pipeline(data_dir: str | Path, out_dir: str | Path, seed: int = 2026) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "consensus")
    ensure_dir(out_dir / "consensus_candidates")
    ensure_dir(out_dir / "model_comparison")
    ensure_dir(out_dir / "classifiers")
    ensure_dir(out_dir / "signatures")

    analysis_results: list[tuple[AnalysisQueueConfig, dict]] = []
    comparison_results: list[tuple[AnalysisQueueConfig, dict]] = []
    queue_prepared: dict[str, PreparedAnalysis] = {}
    snf_reference_labels: dict[str, pd.Series] = {}
    for config in ANALYSIS_QUEUES:
        queue_out_dir = out_dir if not config.out_subdir else out_dir / config.out_subdir
        prepared = prepare_analysis_inputs(data_dir, config.modalities, config.analysis_name)
        queue_prepared[config.key] = prepared

        print(f"[{config.analysis_name}] building fused network", flush=True)
        snf_k_summary, snf_best_k, snf_labels = fit_snf_analysis(prepared, seed=seed)
        print(f"[{config.analysis_name}] evaluating outputs", flush=True)
        snf_result = export_analysis_outputs(prepared, queue_out_dir, config.analysis_name, "snf", snf_labels, snf_k_summary, snf_best_k)
        print(f"[{config.analysis_name}] finished", flush=True)
        analysis_results.append((config, snf_result))
        comparison_results.append((config, snf_result))
        snf_reference_labels[config.key] = snf_result["labels"]

        early_out_dir = out_dir / "model_comparison" / config.key / "early_integration"
        early_result = run_early_integration_analysis(prepared, early_out_dir, config.analysis_name, seed)
        comparison_results.append((config, early_result))

        nmf_out_dir = out_dir / "model_comparison" / config.key / "nmf_integration"
        nmf_result = run_nmf_integration_analysis(prepared, nmf_out_dir, config.analysis_name, seed)
        comparison_results.append((config, nmf_result))

    primary_cfg, primary = analysis_results[0]
    extension_cfg, extension = analysis_results[1]
    overlap_ari = ari_between(primary["labels"], extension["labels"])
    save_sensitivity_plot(
        primary["labels"],
        extension["labels"],
        primary["best_k"],
        extension["best_k"],
        overlap_ari,
        out_dir / "figures" / "queue_comparison.png",
        left_label="5-omics discovery",
        right_label="6-omics validation",
        title="Queue comparison",
    )
    save_sensitivity_plot(
        primary["labels"],
        extension["labels"],
        primary["best_k"],
        extension["best_k"],
        overlap_ari,
        out_dir / "figures" / "sensitivity_summary.png",
        left_label="5-omics discovery",
        right_label="6-omics validation",
        title="Queue comparison",
    )
    queue_summary = queue_summary_frame(analysis_results)
    queue_summary.to_csv(out_dir / "queue_summary.tsv", sep="\t", index=False)
    model_comparison = model_comparison_frame(comparison_results, snf_reference_labels)
    model_comparison.to_csv(out_dir / "model_comparison.tsv", sep="\t", index=False)
    model_stability = model_stability_summary_frame(comparison_results)
    model_stability.to_csv(out_dir / "model_stability_summary.tsv", sep="\t", index=False)
    model_evaluation = model_evaluation_summary_frame(comparison_results)
    model_evaluation.to_csv(out_dir / "model_evaluation_summary.tsv", sep="\t", index=False)
    sensitivity_variant_summary, sensitivity_overview = run_sensitivity_analyses(data_dir, queue_prepared, analysis_results, seed)
    sensitivity_variant_summary.to_csv(out_dir / "sensitivity_variant_summary.tsv", sep="\t", index=False)
    sensitivity_overview.to_csv(out_dir / "sensitivity_overview.tsv", sep="\t", index=False)

    consensus_weight_candidates = build_consensus_weight_tables(model_evaluation, model_stability)
    queue_model_results = [(config.key, result) for config, result in comparison_results]
    consensus_candidate_results = run_consensus_candidate_batch(
        out_dir / "consensus_candidates",
        queue_prepared,
        queue_model_results,
        consensus_weight_candidates,
        seed,
    )
    consensus_tuning_summary = consensus_candidate_summary_frame(
        consensus_candidate_results,
        queue_model_results,
        consensus_weight_candidates,
    )
    rescued_weight_candidates = build_copy_rescued_weight_tables(consensus_weight_candidates, consensus_tuning_summary)
    if not rescued_weight_candidates.empty:
        consensus_weight_candidates = pd.concat([consensus_weight_candidates, rescued_weight_candidates], ignore_index=True)
        rescued_candidate_results = run_consensus_candidate_batch(
            out_dir / "consensus_candidates",
            queue_prepared,
            queue_model_results,
            rescued_weight_candidates,
            seed,
        )
        consensus_candidate_results.extend(rescued_candidate_results)
        consensus_tuning_summary = consensus_candidate_summary_frame(
            consensus_candidate_results,
            queue_model_results,
            consensus_weight_candidates,
        )
    consensus_weight_candidates.to_csv(out_dir / "consensus_scheme_weights.tsv", sep="\t", index=False)
    consensus_tuning_summary.to_csv(out_dir / "consensus_tuning_summary.tsv", sep="\t", index=False)
    consensus_scheme_selection = select_consensus_scheme(consensus_tuning_summary)
    consensus_scheme_selection.to_csv(out_dir / "consensus_scheme_selection.tsv", sep="\t", index=False)
    selected_scheme = str(consensus_scheme_selection.loc[consensus_scheme_selection["selected"], "scheme_name"].iloc[0])
    consensus_weight_table = consensus_weight_candidates[
        consensus_weight_candidates["scheme_name"] == selected_scheme
    ].reset_index(drop=True)
    consensus_weight_table.to_csv(out_dir / "consensus_model_weights.tsv", sep="\t", index=False)

    selected_candidate_lookup = {
        (result["queue_key"], result["scheme_name"]): result for result in consensus_candidate_results
    }
    consensus_results: list[dict] = []
    for config in ANALYSIS_QUEUES:
        selected_candidate = selected_candidate_lookup[(config.key, selected_scheme)]
        consensus_out_dir = out_dir / "consensus" / config.key
        ensure_dir(consensus_out_dir)
        selected_candidate["consensus_matrix"].to_csv(consensus_out_dir / "consensus_matrix.tsv", sep="\t", index=True)
        prepared = queue_prepared[config.key]
        consensus_result = export_analysis_outputs(
            prepared,
            consensus_out_dir,
            config.analysis_name,
            "consensus",
            selected_candidate["labels"],
            selected_candidate["k_summary"],
            selected_candidate["best_k"],
            extra_latent_artifacts={"weighted_consensus_matrix": selected_candidate["consensus_matrix"]},
        )
        consensus_result["queue_key"] = config.key
        consensus_result["scheme_name"] = selected_scheme
        consensus_results.append(consensus_result)

    consensus_summary = consensus_summary_frame(
        [(config.key, config.analysis_name) for config in ANALYSIS_QUEUES],
        consensus_results,
        queue_model_results,
    )
    selected_candidate_summary = consensus_tuning_summary[
        consensus_tuning_summary["scheme_name"] == selected_scheme
    ][
        [
            "queue_key",
            "analysis_name",
            "scheme_name",
            "dominant_model_name",
            "dominant_model_weight",
            "weight_entropy",
            "max_ari_to_single_model",
            "is_exact_copy",
        ]
    ]
    consensus_summary = consensus_summary.merge(
        selected_candidate_summary,
        on=["queue_key", "analysis_name", "scheme_name"],
        how="left",
    )
    consensus_summary.to_csv(out_dir / "consensus_summary.tsv", sep="\t", index=False)

    classifier_summary_rows: list[pd.DataFrame] = []
    classifier_runs: list[dict[str, object]] = []
    signature_classifier_runs: list[dict[str, object]] = []
    for config in ANALYSIS_QUEUES:
        prepared = queue_prepared[config.key]
        consensus_result = next(result for result in consensus_results if result["queue_key"] == config.key)
        feature_frame = build_classifier_feature_matrix(prepared.loaded.omics, prepared.embeddings)
        for classifier_name in CLASSIFIER_NAMES:
            classifier_out_dir = out_dir / "classifiers" / classifier_name / config.key
            classifier_cv = CLASSIFIER_RUNNERS[classifier_name](
                feature_frame,
                consensus_result["labels"],
                random_state=seed,
            )
            classifier_summary_rows.append(
                CLASSIFIER_EXPORTERS[classifier_name](
                    classifier_out_dir,
                    feature_frame,
                    consensus_result["labels"],
                    classifier_cv,
                    config.key,
                    config.analysis_name,
                    selected_scheme,
                )
            )
            classifier_runs.append(
                {
                    "queue_key": config.key,
                    "analysis_name": config.analysis_name,
                    "classifier_name": classifier_name,
                    "feature_frame": feature_frame,
                    "labels": consensus_result["labels"],
                    "cv_result": classifier_cv,
                }
            )
            if classifier_name in SIGNATURE_SOURCE_CLASSIFIER_NAMES:
                signature_classifier_runs.append(classifier_runs[-1])
    classifier_summary = pd.concat(classifier_summary_rows, ignore_index=True)
    classifier_summary.to_csv(out_dir / "classifier_summary.tsv", sep="\t", index=False)

    signature_dir = out_dir / "signatures"
    signature_evidence = build_signature_feature_evidence(signature_classifier_runs)
    signature_evidence.to_csv(out_dir / "signature_feature_evidence.tsv", sep="\t", index=False)
    signature_evidence.to_csv(signature_dir / "feature_evidence.tsv", sep="\t", index=False)
    signature_panels = build_signature_panels(signature_evidence)
    signature_panels.to_csv(signature_dir / "panel_features.tsv", sep="\t", index=False)

    signature_summary_rows: list[pd.DataFrame] = []
    for panel_name, panel_df in signature_panels.groupby("panel_name", sort=False):
        panel_feature_names = panel_df["feature_name"].tolist()
        for run in signature_classifier_runs:
            available_features = [feature_name for feature_name in panel_feature_names if feature_name in run["feature_frame"].columns]
            if len(available_features) < 2:
                continue
            reduced_feature_frame = run["feature_frame"][available_features].copy()
            reduced_cv = CLASSIFIER_RUNNERS[str(run["classifier_name"])](
                reduced_feature_frame,
                run["labels"],
                random_state=seed,
            )
            panel_out_dir = signature_dir / panel_name / str(run["classifier_name"]) / str(run["queue_key"])
            panel_summary = export_classifier_results(
                panel_out_dir,
                reduced_feature_frame,
                run["labels"],
                reduced_cv,
                str(run["queue_key"]),
                str(run["analysis_name"]),
                selected_scheme,
            )
            panel_summary.insert(4, "panel_name", panel_name)
            panel_summary.insert(5, "panel_feature_count", int(len(available_features)))
            signature_summary_rows.append(panel_summary)

    signature_panel_summary = signature_panel_summary_frame(signature_summary_rows, classifier_summary)
    signature_panel_summary.to_csv(out_dir / "signature_panel_summary.tsv", sep="\t", index=False)
    signature_panel_summary.to_csv(signature_dir / "panel_summary.tsv", sep="\t", index=False)
    shortlist_markdown = signature_shortlist_markdown(signature_panels, signature_panel_summary)
    (signature_dir / "signature_shortlist.md").write_text(shortlist_markdown, encoding="utf-8")

    mirna_aliases = load_mirbase_mature_aliases(signature_dir / "_cache" / "mirbase_mature.fa")
    mirna_aliases.to_csv(out_dir / "signature_mirna_aliases.tsv", sep="\t", index=False)
    mirna_aliases.to_csv(signature_dir / "mirna_aliases.tsv", sep="\t", index=False)

    combined_probe_map = pd.concat(
        [prepared.loaded.probe_map for prepared in queue_prepared.values()],
        axis=0,
        ignore_index=True,
    ).drop_duplicates()
    latent_component_markers = build_latent_component_marker_table(
        signature_panels,
        queue_prepared["primary_discovery"].preprocessed,
        combined_probe_map,
        mirna_aliases=mirna_aliases,
    )
    latent_component_markers.to_csv(out_dir / "signature_raw_marker_mapping.tsv", sep="\t", index=False)
    latent_component_markers.to_csv(signature_dir / "raw_marker_mapping.tsv", sep="\t", index=False)
    proxy_candidate_scores = build_raw_proxy_candidate_scores(latent_component_markers, queue_prepared)
    proxy_candidate_scores.to_csv(out_dir / "signature_raw_proxy_scores.tsv", sep="\t", index=False)
    proxy_candidate_scores.to_csv(signature_dir / "proxy_candidate_scores.tsv", sep="\t", index=False)
    raw_marker_panels = build_raw_marker_panels(signature_panels, latent_component_markers)
    raw_marker_panels.to_csv(signature_dir / "raw_panel_features.tsv", sep="\t", index=False)
    refined_raw_panels = build_refined_raw_marker_panels(signature_panels, latent_component_markers, proxy_candidate_scores)
    refined_raw_panels.to_csv(signature_dir / "refined_panel_features.tsv", sep="\t", index=False)

    raw_panel_summary_rows: list[pd.DataFrame] = []
    for raw_panel_name, raw_panel_df in raw_marker_panels.groupby("raw_panel_name", sort=False):
        source_panel_name = str(raw_panel_df["source_panel_name"].iloc[0])
        for run in signature_classifier_runs:
            reduced_feature_frame = build_raw_panel_feature_matrix(
                raw_panel_df,
                queue_prepared[str(run["queue_key"])].preprocessed,
                queue_prepared[str(run["queue_key"])].loaded.omics,
            )
            if reduced_feature_frame.empty or reduced_feature_frame.shape[1] < 2:
                continue
            reduced_cv = CLASSIFIER_RUNNERS[str(run["classifier_name"])](
                reduced_feature_frame,
                run["labels"],
                random_state=seed,
            )
            panel_out_dir = signature_dir / raw_panel_name / str(run["classifier_name"]) / str(run["queue_key"])
            panel_summary = export_classifier_results(
                panel_out_dir,
                reduced_feature_frame,
                run["labels"],
                reduced_cv,
                str(run["queue_key"]),
                str(run["analysis_name"]),
                selected_scheme,
            )
            panel_summary.insert(4, "raw_panel_name", raw_panel_name)
            panel_summary.insert(5, "source_panel_name", source_panel_name)
            panel_summary.insert(6, "raw_panel_feature_count", int(reduced_feature_frame.shape[1]))
            raw_panel_summary_rows.append(panel_summary)

    raw_signature_panel_summary = raw_signature_panel_summary_frame(
        raw_panel_summary_rows,
        classifier_summary,
        signature_panel_summary,
    )
    raw_signature_panel_summary.to_csv(out_dir / "signature_raw_panel_summary.tsv", sep="\t", index=False)
    raw_signature_panel_summary.to_csv(signature_dir / "raw_panel_summary.tsv", sep="\t", index=False)
    raw_shortlist_markdown = raw_signature_shortlist_markdown(raw_marker_panels, raw_signature_panel_summary)
    (signature_dir / "raw_signature_shortlist.md").write_text(raw_shortlist_markdown, encoding="utf-8")

    refined_panel_summary_rows: list[pd.DataFrame] = []
    for raw_panel_name, raw_panel_df in refined_raw_panels.groupby("raw_panel_name", sort=False):
        source_panel_name = str(raw_panel_df["source_panel_name"].iloc[0])
        for run in signature_classifier_runs:
            reduced_feature_frame = build_raw_panel_feature_matrix(
                raw_panel_df,
                queue_prepared[str(run["queue_key"])].preprocessed,
                queue_prepared[str(run["queue_key"])].loaded.omics,
            )
            if reduced_feature_frame.empty or reduced_feature_frame.shape[1] < 2:
                continue
            reduced_cv = CLASSIFIER_RUNNERS[str(run["classifier_name"])](
                reduced_feature_frame,
                run["labels"],
                random_state=seed,
            )
            panel_out_dir = signature_dir / raw_panel_name / str(run["classifier_name"]) / str(run["queue_key"])
            panel_summary = export_classifier_results(
                panel_out_dir,
                reduced_feature_frame,
                run["labels"],
                reduced_cv,
                str(run["queue_key"]),
                str(run["analysis_name"]),
                selected_scheme,
            )
            panel_summary.insert(4, "raw_panel_name", raw_panel_name)
            panel_summary.insert(5, "source_panel_name", source_panel_name)
            panel_summary.insert(6, "raw_panel_feature_count", int(reduced_feature_frame.shape[1]))
            refined_panel_summary_rows.append(panel_summary)

    refined_signature_panel_summary = raw_signature_panel_summary_frame(
        refined_panel_summary_rows,
        classifier_summary,
        signature_panel_summary,
    )
    refined_signature_panel_summary.to_csv(out_dir / "signature_refined_panel_summary.tsv", sep="\t", index=False)
    refined_signature_panel_summary.to_csv(signature_dir / "refined_panel_summary.tsv", sep="\t", index=False)
    refined_shortlist_markdown = raw_signature_shortlist_markdown(
        refined_raw_panels,
        refined_signature_panel_summary,
        title="Refined Raw Signature Shortlist",
    )
    (signature_dir / "refined_signature_shortlist.md").write_text(refined_shortlist_markdown, encoding="utf-8")
    freeze_candidates = build_refined_panel_freeze_candidates(
        refined_raw_panels,
        refined_signature_panel_summary,
    )
    freeze_candidates.to_csv(out_dir / "signature_freeze_candidates.tsv", sep="\t", index=False)
    freeze_candidates.to_csv(signature_dir / "freeze_candidates.tsv", sep="\t", index=False)
    freeze_selection = freeze_candidates.copy()
    freeze_selection.to_csv(out_dir / "signature_freeze_selection.tsv", sep="\t", index=False)
    freeze_selection.to_csv(signature_dir / "freeze_selection.tsv", sep="\t", index=False)
    frozen_panel_features = extract_frozen_refined_panel_features(refined_raw_panels, freeze_selection)
    frozen_panel_features.to_csv(signature_dir / "frozen_panel_features.tsv", sep="\t", index=False)
    frozen_shortlist_markdown = frozen_refined_signature_markdown(freeze_selection, frozen_panel_features)
    (signature_dir / "frozen_signature_shortlist.md").write_text(frozen_shortlist_markdown, encoding="utf-8")
    reduced_panel_features = build_frozen_panel_reduction_candidates(frozen_panel_features)
    reduced_panel_features.to_csv(out_dir / "signature_reduced_panel_features.tsv", sep="\t", index=False)
    reduced_panel_features.to_csv(signature_dir / "reduced_panel_features.tsv", sep="\t", index=False)

    reduced_panel_summary_rows: list[pd.DataFrame] = []
    for reduced_panel_name, reduced_panel_df in reduced_panel_features.groupby("reduced_panel_name", sort=False):
        source_panel_name = str(reduced_panel_df["source_panel_name"].iloc[0])
        parent_frozen_panel_name = str(reduced_panel_df["parent_frozen_panel_name"].iloc[0])
        for run in signature_classifier_runs:
            reduced_feature_frame = build_raw_panel_feature_matrix(
                reduced_panel_df,
                queue_prepared[str(run["queue_key"])].preprocessed,
                queue_prepared[str(run["queue_key"])].loaded.omics,
            )
            if reduced_feature_frame.empty or reduced_feature_frame.shape[1] < 2:
                continue
            reduced_cv = CLASSIFIER_RUNNERS[str(run["classifier_name"])](
                reduced_feature_frame,
                run["labels"],
                random_state=seed,
            )
            panel_out_dir = signature_dir / reduced_panel_name / str(run["classifier_name"]) / str(run["queue_key"])
            panel_summary = export_classifier_results(
                panel_out_dir,
                reduced_feature_frame,
                run["labels"],
                reduced_cv,
                str(run["queue_key"]),
                str(run["analysis_name"]),
                selected_scheme,
            )
            panel_summary.insert(4, "reduced_panel_name", reduced_panel_name)
            panel_summary.insert(5, "parent_frozen_panel_name", parent_frozen_panel_name)
            panel_summary.insert(6, "source_panel_name", source_panel_name)
            panel_summary.insert(7, "raw_panel_feature_count", int(reduced_feature_frame.shape[1]))
            reduced_panel_summary_rows.append(panel_summary)

    reduced_panel_summary = reduction_signature_panel_summary_frame(
        reduced_panel_summary_rows,
        classifier_summary,
        signature_panel_summary,
        refined_signature_panel_summary,
    )
    reduced_panel_summary.to_csv(out_dir / "signature_reduced_panel_summary.tsv", sep="\t", index=False)
    reduced_panel_summary.to_csv(signature_dir / "reduced_panel_summary.tsv", sep="\t", index=False)
    reduced_panel_selection = build_reduced_panel_selection(reduced_panel_features, reduced_panel_summary)
    reduced_panel_selection.to_csv(out_dir / "signature_reduced_panel_selection.tsv", sep="\t", index=False)
    reduced_panel_selection.to_csv(signature_dir / "reduced_panel_selection.tsv", sep="\t", index=False)
    reduced_shortlist_markdown = reduced_signature_shortlist_markdown(
        reduced_panel_features,
        reduced_panel_summary,
        reduced_panel_selection,
    )
    (signature_dir / "reduced_signature_shortlist.md").write_text(reduced_shortlist_markdown, encoding="utf-8")
    selected_reduced_rows = reduced_panel_selection.loc[reduced_panel_selection["selected"]]
    selected_reduced_name = (
        str(selected_reduced_rows.iloc[0]["reduced_panel_name"])
        if not selected_reduced_rows.empty
        else ""
    )
    selected_reduced_panel_features = reduced_panel_features.loc[
        reduced_panel_features["reduced_panel_name"] == selected_reduced_name
    ].copy()
    final_panel_features = build_final_panel_candidates(selected_reduced_panel_features)
    final_panel_features.to_csv(out_dir / "signature_final_panel_features.tsv", sep="\t", index=False)
    final_panel_features.to_csv(signature_dir / "final_panel_features.tsv", sep="\t", index=False)
    final_panel_summary_rows: list[pd.DataFrame] = []
    for final_panel_name, final_panel_df in final_panel_features.groupby("final_panel_name", sort=False):
        source_panel_name = str(final_panel_df["source_panel_name"].iloc[0])
        parent_reduced_panel_name = str(final_panel_df["parent_reduced_panel_name"].iloc[0])
        for run in signature_classifier_runs:
            final_feature_frame = build_raw_panel_feature_matrix(
                final_panel_df,
                queue_prepared[str(run["queue_key"])].preprocessed,
                queue_prepared[str(run["queue_key"])].loaded.omics,
            )
            if final_feature_frame.empty or final_feature_frame.shape[1] < 2:
                continue
            final_cv = CLASSIFIER_RUNNERS[str(run["classifier_name"])](
                final_feature_frame,
                run["labels"],
                random_state=seed,
            )
            panel_out_dir = signature_dir / final_panel_name / str(run["classifier_name"]) / str(run["queue_key"])
            final_summary = export_classifier_results(
                panel_out_dir,
                final_feature_frame,
                run["labels"],
                final_cv,
                str(run["queue_key"]),
                str(run["analysis_name"]),
                selected_scheme,
            )
            final_summary.insert(4, "final_panel_name", final_panel_name)
            final_summary.insert(5, "parent_reduced_panel_name", parent_reduced_panel_name)
            final_summary.insert(6, "source_panel_name", source_panel_name)
            final_summary.insert(7, "raw_panel_feature_count", int(final_feature_frame.shape[1]))
            final_panel_summary_rows.append(final_summary)
    final_panel_summary = final_signature_panel_summary_frame(
        final_panel_summary_rows,
        classifier_summary,
        signature_panel_summary,
        reduced_panel_summary,
    )
    final_panel_summary.to_csv(out_dir / "signature_final_panel_summary.tsv", sep="\t", index=False)
    final_panel_summary.to_csv(signature_dir / "final_panel_summary.tsv", sep="\t", index=False)
    final_panel_selection = build_final_panel_selection(
        final_panel_features,
        final_panel_summary,
        reduced_panel_features,
        reduced_panel_summary,
        reduced_panel_selection,
    )
    final_panel_selection.to_csv(out_dir / "signature_final_panel_selection.tsv", sep="\t", index=False)
    final_panel_selection.to_csv(signature_dir / "final_panel_selection.tsv", sep="\t", index=False)
    selected_final_panel_features = extract_selected_final_panel_features(
        final_panel_features,
        final_panel_selection,
        reduced_panel_features,
    )
    selected_final_panel_features.to_csv(out_dir / "signature_final_recommended_panel_features.tsv", sep="\t", index=False)
    selected_final_panel_features.to_csv(signature_dir / "final_recommended_panel_features.tsv", sep="\t", index=False)
    final_shortlist_markdown = final_signature_shortlist_markdown(
        final_panel_features,
        final_panel_summary,
        final_panel_selection,
        selected_final_panel_features,
    )
    (signature_dir / "final_signature_shortlist.md").write_text(final_shortlist_markdown, encoding="utf-8")
    selected_final_name = (
        str(selected_final_panel_features["selected_final_panel_name"].dropna().iloc[0])
        if not selected_final_panel_features.empty and "selected_final_panel_name" in selected_final_panel_features.columns
        else ""
    )
    final_classifier_evidence = collect_final_signature_classifier_evidence(
        signature_dir,
        selected_final_name,
        selected_final_panel_features["mapped_feature_name"].dropna().astype(str).tolist()
        if not selected_final_panel_features.empty and "mapped_feature_name" in selected_final_panel_features.columns
        else None,
    )
    final_classifier_evidence.to_csv(out_dir / "signature_final_classifier_evidence.tsv", sep="\t", index=False)
    final_classifier_evidence.to_csv(signature_dir / "final_classifier_evidence.tsv", sep="\t", index=False)
    final_interpretation = build_final_signature_interpretation(
        selected_final_panel_features,
        final_classifier_evidence,
    )
    final_interpretation.to_csv(out_dir / "signature_final_interpretation.tsv", sep="\t", index=False)
    final_interpretation.to_csv(signature_dir / "final_signature_interpretation.tsv", sep="\t", index=False)
    final_interpretation_markdown = final_signature_interpretation_markdown(
        final_interpretation,
        final_panel_selection,
    )
    (signature_dir / "final_signature_interpretation.md").write_text(final_interpretation_markdown, encoding="utf-8")
    write_signature_readme(
        signature_dir,
        signature_panels,
        signature_panel_summary,
        raw_panel_features=raw_marker_panels,
        raw_panel_summary=raw_signature_panel_summary,
        refined_panel_features=refined_raw_panels,
        refined_panel_summary=refined_signature_panel_summary,
        freeze_candidates=freeze_candidates,
        freeze_selection=freeze_selection,
        frozen_panel_features=frozen_panel_features,
        reduced_panel_features=reduced_panel_features,
        reduced_panel_summary=reduced_panel_summary,
        reduced_panel_selection=reduced_panel_selection,
        final_panel_features=final_panel_features,
        final_panel_summary=final_panel_summary,
        final_panel_selection=final_panel_selection,
        selected_final_panel_features=selected_final_panel_features,
    )

    summary = {
        "primary_queue": asdict(primary_cfg),
        "validation_queue": asdict(extension_cfg),
        "primary_best_k": primary["best_k"],
        "validation_best_k": extension["best_k"],
        "primary_samples": int(len(primary["labels"])),
        "validation_samples": int(len(extension["labels"])),
        "overlap_ari": overlap_ari,
        "comparison_models": sorted({result["model_name"] for _, result in comparison_results}),
        "sensitivity_variant_rows": int(len(sensitivity_variant_summary)),
        "consensus_models": ["snf", "early_integration", "nmf_integration"],
        "consensus_scheme_candidates": sorted(consensus_weight_candidates["scheme_name"].drop_duplicates().tolist()),
        "selected_consensus_scheme": selected_scheme,
        "classifier_models": list(CLASSIFIER_NAMES),
        "signature_source_classifier_models": list(SIGNATURE_SOURCE_CLASSIFIER_NAMES),
        "signature_panels": signature_panels["panel_name"].drop_duplicates().tolist(),
        "raw_signature_panels": raw_marker_panels["raw_panel_name"].drop_duplicates().tolist(),
        "refined_signature_panels": refined_raw_panels["raw_panel_name"].drop_duplicates().tolist(),
        "frozen_refined_signature_panel": (
            str(freeze_selection.loc[freeze_selection["selected"], "representative_panel_name"].iloc[0])
            if not freeze_selection.empty and bool(freeze_selection["selected"].any())
            else None
        ),
        "reduced_signature_panels": reduced_panel_features["reduced_panel_name"].drop_duplicates().tolist(),
        "selected_reduced_signature_panel": (
            str(reduced_panel_selection.loc[reduced_panel_selection["selected"], "reduced_panel_name"].iloc[0])
            if not reduced_panel_selection.empty and bool(reduced_panel_selection["selected"].any())
            else None
        ),
        "final_signature_panels": final_panel_features["final_panel_name"].drop_duplicates().tolist(),
        "selected_final_signature_panel": (
            str(final_panel_selection.loc[final_panel_selection["selected"], "final_panel_name"].iloc[0])
            if not final_panel_selection.empty and bool(final_panel_selection["selected"].any())
            else None
        ),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[pipeline] all analyses finished", flush=True)
    return summary
