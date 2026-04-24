from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stad_subtyping.cluster import consensus_clustering
from stad_subtyping.cluster import label_comparison_metrics
from stad_subtyping.consensus import (
    build_copy_rescued_weight_tables,
    build_consensus_weight_table,
    build_consensus_weight_tables,
    consensus_candidate_summary_frame,
    consensus_summary_frame,
    select_consensus_scheme,
    weighted_consensus_matrix,
)
from stad_subtyping.config import ANALYSIS_QUEUES, data_path
from stad_subtyping.evaluate import subtype_interpretation_markdown, subtype_interpretation_table
from stad_subtyping.evaluate import multivariable_cox
from stad_subtyping.features import compute_mechanism_summary_features, feature_layer_manifest, mechanism_summary_feature_metadata
from stad_subtyping.io import is_primary_tumor, tcga_patient_id
from stad_subtyping.models import (
    build_early_integration_embedding,
    build_nmf_integration_embedding,
    early_integration_consensus_clustering,
    embedding_consensus_clustering,
)
from stad_subtyping.pipeline import model_comparison_frame, model_evaluation_summary_frame, model_stability_summary_frame, queue_summary_frame
from stad_subtyping.pipeline import make_sensitivity_variant_record, sensitivity_overview_frame
from stad_subtyping.preprocess import preprocess_modality, reduce_modality
from stad_subtyping.signature import build_signature_feature_evidence, build_signature_panels, signature_panel_summary_frame
from stad_subtyping.signature import (
    build_final_panel_candidates,
    build_final_signature_interpretation,
    build_final_panel_selection,
    build_frozen_panel_reduction_candidates,
    build_reduced_panel_selection,
    build_refined_panel_freeze_candidates,
    build_refined_raw_marker_panels,
    collect_final_signature_classifier_evidence,
    build_raw_proxy_candidate_scores,
    build_latent_component_marker_table,
    extract_frozen_refined_panel_features,
    final_signature_interpretation_markdown,
    frozen_refined_signature_markdown,
    load_mirbase_mature_aliases,
    parse_mirbase_mature_fasta,
    build_raw_marker_panels,
    build_raw_panel_feature_matrix,
    extract_selected_final_panel_features,
    final_signature_panel_summary_frame,
    final_signature_shortlist_markdown,
    raw_signature_panel_summary_frame,
    reduction_signature_panel_summary_frame,
)
from stad_subtyping.snf import build_affinity_matrix, snf
from stad_subtyping.supervised import (
    CLASSIFIER_RUNNERS,
    SIGNATURE_SOURCE_CLASSIFIER_NAMES,
    build_classifier_feature_matrix,
    run_elastic_net_cv,
    run_lstm_cv,
    run_mlp_cv,
    run_svm_cv,
    run_transformer_cv,
    run_xgboost_cv,
)


def test_tcga_id_helpers():
    assert tcga_patient_id("TCGA-AB-1234-01") == "TCGA-AB-1234"
    assert is_primary_tumor("TCGA-AB-1234-01")
    assert not is_primary_tumor("TCGA-AB-1234-11")


def test_preprocess_and_reduce_expression():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        rng.normal(size=(3000, 12)).astype(np.float32),
        index=[f"G{i}" for i in range(3000)],
        columns=[f"S{i}" for i in range(12)],
    )
    processed, qc = preprocess_modality("expression", frame)
    assert processed.shape[0] == 2000
    embedding = reduce_modality("expression", processed)
    assert embedding.shape[0] == 12
    assert embedding.shape[1] <= 11
    assert qc["selected_features"] == 2000


def test_data_path_falls_back_across_aliases(tmp_path: Path):
    target = tmp_path / "HumanMethylation450"
    target.write_text("demo", encoding="utf-8")
    resolved = data_path(
        tmp_path,
        (
            "TCGA.STAD.sampleMap_HumanMethylation450.gz",
            "TCGA.STAD.sampleMap_HumanMethylation450",
            "HumanMethylation450",
        ),
    )
    assert resolved == target


def test_analysis_queues_match_new_project_direction():
    assert len(ANALYSIS_QUEUES) == 2
    assert ANALYSIS_QUEUES[0].key == "primary_discovery"
    assert ANALYSIS_QUEUES[0].modalities == ("mutation", "cnv", "methylation", "expression", "mirna")
    assert ANALYSIS_QUEUES[1].key == "validation_with_rppa"
    assert ANALYSIS_QUEUES[1].modalities[-1] == "rppa"


def test_consensus_clustering_reproducible():
    rng = np.random.default_rng(42)
    x = np.vstack(
        [
            rng.normal(loc=-2, scale=0.2, size=(20, 5)),
            rng.normal(loc=2, scale=0.2, size=(20, 5)),
        ]
    )
    embedding = pd.DataFrame(x, index=[f"S{i}" for i in range(40)])
    affinity = build_affinity_matrix(embedding, k=10)
    fused = snf([affinity])
    summary_a, best_k_a, labels_a = consensus_clustering(
        fused, k_values=range(2, 4), reps=20, sample_fraction=0.8, min_cluster_size=5, random_state=123
    )
    summary_b, best_k_b, labels_b = consensus_clustering(
        fused, k_values=range(2, 4), reps=20, sample_fraction=0.8, min_cluster_size=5, random_state=123
    )
    assert best_k_a == best_k_b
    assert labels_a.equals(labels_b)
    assert list(summary_a["k"]) == [2, 3]


def test_subtype_interpretation_outputs():
    subtype_table = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "subtype": ["C1", "C1", "C2", "C2"],
            "msi_status": ["MSI-H", "MSI-H", "MSS", "MSS"],
            "pathologic_stage": ["Stage II", "Stage II", "Stage III", "Stage III"],
            "grade": ["G3", "G3", "G2", "G2"],
            "histological_type": ["Diffuse", "Diffuse", "Tubular", "Tubular"],
            "age": [60, 62, 55, 57],
        }
    )
    summary = pd.DataFrame(
        {
            "subtype": ["C1", "C2"],
            "n_samples": [2, 2],
            "tmb_mean": [10.0, 5.0],
            "mutation_burden_mean": [10.0, 5.0],
            "mutation_event_fraction_mean": [0.4, 0.2],
            "cnv_burden_mean": [7.0, 12.0],
            "cnv_gain_burden_mean": [5.0, 9.0],
            "cnv_loss_burden_mean": [2.0, 3.0],
            "cnv_altered_fraction_mean": [0.3, 0.7],
            "global_methylation_mean": [0.2, -0.1],
            "msi_h_rate": [1.0, 0.0],
            "diffuse_histology_rate": [1.0, 0.0],
            "tcga_like_label": ["MSI-like", "CIN-like"],
        }
    )
    survival = pd.DataFrame(
        {
            "endpoint": ["OS", "DSS", "DFI", "PFI"],
            "C1_median_time": [500.0, 450.0, 250.0, 300.0],
            "C2_median_time": [400.0, 350.0, 150.0, 200.0],
            "logrank_p": [0.01, 0.02, 0.03, 0.04],
            "n_samples": [4, 4, 4, 4],
        }
    )
    top_features = pd.DataFrame(
        {
            "modality": ["expression", "mutation", "expression", "mutation"],
            "subtype": ["C1", "C1", "C2", "C2"],
            "feature_id": ["GENE1", "TP53", "GENE2", "ARID1A"],
            "annotation": ["GENE1", "TP53", "GENE2", "ARID1A"],
            "effect_size": [1.0, 0.5, -1.0, 0.7],
            "abs_effect_size": [1.0, 0.5, 1.0, 0.7],
        }
    )
    interpretation = subtype_interpretation_table(subtype_table, summary, survival, top_features)
    assert set(interpretation["subtype"]) == {"C1", "C2"}
    assert "expression_markers" in interpretation.columns
    assert "dss_median_time" in interpretation.columns
    assert "dfi_median_time" in interpretation.columns
    markdown = subtype_interpretation_markdown(interpretation, pd.DataFrame({"variable": ["msi_status"], "test": ["chi2"], "p_value": [0.01], "n": [4]}), survival, "demo")
    assert "C1 - MSI-like" in markdown
    assert "expression markers" in markdown
    assert "median DSS" in markdown
    assert "TMB 10.0" in markdown


def test_multivariable_cox_outputs():
    subtype_table = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(1, 13)],
            "subtype": ["C1"] * 6 + ["C2"] * 6,
            "age": [50, 52, 55, 57, 60, 62, 48, 49, 51, 53, 54, 56],
            "gender": ["MALE", "FEMALE"] * 6,
            "pathologic_stage": ["Stage II", "Stage III", "Stage II", "Stage III", "Stage II", "Stage III"] * 2,
            "OS": [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            "OS.time": [10, 12, 15, 18, 20, 25, 11, 14, 16, 19, 22, 24],
            "DSS": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            "DSS.time": [10, 12, 15, 18, 20, 25, 11, 14, 16, 19, 22, 24],
            "DFI": [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            "DFI.time": [8, 12, 14, 16, 18, 22, 9, 13, 15, 17, 19, 21],
            "PFI": [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            "PFI.time": [9, 11, 14, 17, 19, 24, 10, 13, 15, 18, 21, 23],
        }
    )
    detail, global_table = multivariable_cox(subtype_table)
    assert not detail.empty
    assert not global_table.empty
    assert set(global_table["endpoint"]) == {"OS", "DSS", "DFI", "PFI"}
    assert "subtype_lrt_p" in global_table.columns


def test_mechanism_summary_features_from_raw_omics():
    mutation = pd.DataFrame(
        [[1, 0], [0, 1], [1, 1]],
        index=["M1", "M2", "M3"],
        columns=["P1", "P2"],
    )
    cnv = pd.DataFrame(
        [[2, -1], [0, -2], [1, 0]],
        index=["C1", "C2", "C3"],
        columns=["P1", "P2"],
    )
    methylation = pd.DataFrame(
        [[0.2, 0.4], [0.6, np.nan]],
        index=["cg1", "cg2"],
        columns=["P1", "P2"],
    )
    expression = pd.DataFrame(
        [[1.0, 0.0], [0.5, 2.0], [1.0, 3.0], [2.0, 5.0], [1.0, 1.0]],
        index=["EPCAM", "VIM", "MKI67", "CXCL9", "STAT1"],
        columns=["P1", "P2"],
    )
    mirna = pd.DataFrame(
        [[1.0, 2.0], [3.0, 1.0]],
        index=["M1", "M2"],
        columns=["P1", "P2"],
    )
    rppa = pd.DataFrame(
        [[1.0, 2.0], [2.0, 4.0], [1.0, 0.0], [3.0, 6.0]],
        index=["Akt-R-V", "mTOR-R-V", "EGFR-R-V", "E-Cadherin-R-V"],
        columns=["P1", "P2"],
    )
    features = compute_mechanism_summary_features(
        {"mutation": mutation, "cnv": cnv, "methylation": methylation, "expression": expression, "mirna": mirna, "rppa": rppa}
    )
    assert list(features.index) == ["P1", "P2"]
    assert float(features.loc["P1", "tmb"]) == 2.0
    assert float(features.loc["P2", "tmb"]) == 2.0
    assert float(features.loc["P1", "cnv_burden"]) == 3.0
    assert float(features.loc["P2", "cnv_gain_burden"]) == 0.0
    assert float(features.loc["P2", "cnv_loss_burden"]) == 3.0
    assert np.isclose(float(features.loc["P2", "global_methylation"]), 0.5)
    assert "expression_epithelial_score" in features.columns
    assert "mirna_global_abundance" in features.columns
    assert "rppa_pi3k_akt_mtor_score" in features.columns
    metadata = mechanism_summary_feature_metadata({"expression": expression, "rppa": rppa})
    assert "global_methylation" in set(metadata["feature_name"])
    assert int(metadata.loc[metadata["feature_name"] == "expression_epithelial_score", "matched_feature_count"].iloc[0]) >= 1
    manifest = feature_layer_manifest(
        {"expression": expression},
        {"expression": pd.DataFrame([[1.0], [2.0]], index=["P1", "P2"], columns=["component_1"])},
        metadata,
    )
    assert "raw_selected" in set(manifest["feature_layer"])
    assert "latent_embedding" in set(manifest["feature_layer"])
    assert "mechanism_summary" in set(manifest["feature_layer"])


def test_queue_summary_frame_reports_queue_metadata():
    labels_a = pd.Series(["C1", "C1", "C2"], index=["P1", "P2", "P3"], name="subtype")
    labels_b = pd.Series(["C1", "C2"], index=["P1", "P3"], name="subtype")
    results = [
        (ANALYSIS_QUEUES[0], {"labels": labels_a, "best_k": 2}),
        (ANALYSIS_QUEUES[1], {"labels": labels_b, "best_k": 2}),
    ]
    summary = queue_summary_frame(results)
    assert list(summary["queue_key"]) == ["primary_discovery", "validation_with_rppa"]
    assert int(summary.loc[0, "n_modalities"]) == 5
    assert int(summary.loc[1, "n_modalities"]) == 6


def test_early_integration_embedding_and_consensus():
    emb_a = pd.DataFrame(
        [[-2.0, -1.8], [-1.9, -2.1], [2.0, 1.8], [2.2, 2.1]],
        index=["S1", "S2", "S3", "S4"],
        columns=["a1", "a2"],
    )
    emb_b = pd.DataFrame(
        [[-1.0, -1.1], [-1.2, -0.9], [1.1, 1.0], [0.9, 1.2]],
        index=["S1", "S2", "S3", "S4"],
        columns=["b1", "b2"],
    )
    integrated = build_early_integration_embedding({"m1": emb_a, "m2": emb_b}, component_cap=3)
    assert integrated.shape[0] == 4
    assert integrated.shape[1] <= 3
    summary, best_k, labels = early_integration_consensus_clustering(
        integrated, k_values=range(2, 4), reps=20, sample_fraction=0.75, min_cluster_size=2, random_state=7
    )
    assert best_k in {2, 3}
    assert labels.name == "subtype"
    assert set(summary.columns) >= {"k", "pac_score", "silhouette", "selected"}


def test_nmf_integration_embedding_and_consensus():
    emb_a = pd.DataFrame(
        [[0.0, 0.2], [0.1, 0.0], [2.0, 2.1], [2.2, 1.9]],
        index=["S1", "S2", "S3", "S4"],
        columns=["a1", "a2"],
    )
    emb_b = pd.DataFrame(
        [[0.0, 0.1], [0.2, 0.0], [1.0, 0.9], [1.1, 1.2]],
        index=["S1", "S2", "S3", "S4"],
        columns=["b1", "b2"],
    )
    nmf_embedding = build_nmf_integration_embedding({"m1": emb_a, "m2": emb_b}, component_cap=3)
    assert nmf_embedding.shape[0] == 4
    assert nmf_embedding.shape[1] <= 3
    assert (nmf_embedding.to_numpy() >= 0).all()
    summary, best_k, labels = embedding_consensus_clustering(
        nmf_embedding, k_values=range(2, 4), reps=20, sample_fraction=0.75, min_cluster_size=2, random_state=9
    )
    assert best_k in {2, 3}
    assert labels.name == "subtype"
    assert set(summary.columns) >= {"k", "pac_score", "silhouette", "selected"}


def test_model_comparison_frame_reports_ari_to_snf():
    snf_labels = pd.Series(["C1", "C1", "C2"], index=["P1", "P2", "P3"], name="subtype")
    early_labels = pd.Series(["C1", "C2", "C2"], index=["P1", "P2", "P3"], name="subtype")
    nmf_labels = pd.Series(["C1", "C1", "C2"], index=["P1", "P2", "P3"], name="subtype")
    results = [
        (
            ANALYSIS_QUEUES[0],
            {
                "labels": snf_labels,
                "best_k": 2,
                "model_name": "snf",
                "out_dir": Path("results"),
                "k_summary": pd.DataFrame(
                    {"k": [2], "pac_score": [0.3], "silhouette": [0.5], "min_cluster_size": [1], "selected": [True]}
                ),
            },
        ),
        (
            ANALYSIS_QUEUES[0],
            {
                "labels": early_labels,
                "best_k": 2,
                "model_name": "early_integration",
                "out_dir": Path("results/model_comparison"),
                "k_summary": pd.DataFrame(
                    {"k": [2], "pac_score": [0.4], "silhouette": [0.2], "min_cluster_size": [1], "selected": [True]}
                ),
            },
        ),
        (
            ANALYSIS_QUEUES[0],
            {
                "labels": nmf_labels,
                "best_k": 2,
                "model_name": "nmf_integration",
                "out_dir": Path("results/model_comparison/nmf"),
                "k_summary": pd.DataFrame(
                    {"k": [2], "pac_score": [0.2], "silhouette": [0.3], "min_cluster_size": [1], "selected": [True]}
                ),
            },
        ),
    ]
    comparison = model_comparison_frame(results, {"primary_discovery": snf_labels})
    assert list(comparison["model_name"]) == ["snf", "early_integration", "nmf_integration"]
    assert float(comparison.loc[0, "ari_to_snf_same_queue"]) == 1.0
    assert float(comparison.loc[1, "ari_to_snf_same_queue"]) <= 1.0
    assert float(comparison.loc[2, "ari_to_snf_same_queue"]) == 1.0
    assert "nmi_to_snf_same_queue" in comparison.columns
    assert "jaccard_to_snf_same_queue" in comparison.columns


def test_label_comparison_and_summary_frames():
    primary_snf = pd.Series(["C1", "C1", "C2", "C2"], index=["P1", "P2", "P3", "P4"], name="subtype")
    validation_snf = pd.Series(["C1", "C2", "C2"], index=["P1", "P3", "P4"], name="subtype")
    metrics = label_comparison_metrics(primary_snf, validation_snf)
    assert metrics["n_overlap"] == 3
    assert "nmi" in metrics
    base_result = {
        "labels": primary_snf,
        "best_k": 2,
        "model_name": "snf",
        "out_dir": Path("results"),
        "k_summary": pd.DataFrame({"k": [2], "pac_score": [0.3], "silhouette": [0.1], "min_cluster_size": [2], "selected": [True]}),
        "clinical_table": pd.DataFrame({"variable": ["msi_status", "age"], "p_value": [0.01, 0.02]}),
        "survival_table": pd.DataFrame({"endpoint": ["OS", "DSS"], "logrank_p": [0.03, 0.04]}),
        "cox_global_table": pd.DataFrame({"endpoint": ["OS", "DSS"], "subtype_lrt_p": [0.05, 0.06], "full_concordance": [0.7, 0.65]}),
    }
    validation_result = {
        **base_result,
        "labels": validation_snf,
        "out_dir": Path("results/validation"),
    }
    stability = model_stability_summary_frame(
        [
            (ANALYSIS_QUEUES[0], base_result),
            (ANALYSIS_QUEUES[1], validation_result),
        ]
    )
    assert "overlap_nmi" in stability.columns
    evaluation = model_evaluation_summary_frame(
        [
            (ANALYSIS_QUEUES[0], base_result),
        ]
    )
    assert "os_subtype_cox_p" in evaluation.columns


def test_sensitivity_variant_record_and_overview():
    baseline_labels = pd.Series(["C1", "C1", "C2", "C2"], index=["P1", "P2", "P3", "P4"], name="subtype")
    variant_labels = pd.Series(["C1", "C2", "C2", "C2"], index=["P1", "P2", "P3", "P4"], name="subtype")
    k_summary = pd.DataFrame(
        {
            "k": [2, 3],
            "pac_score": [0.2, 0.5],
            "silhouette": [0.1, 0.05],
            "min_cluster_size": [2, 1],
            "is_valid": [True, False],
            "selected": [True, False],
        }
    )
    clinical_table = pd.DataFrame({"variable": ["msi_status", "age"], "p_value": [0.01, 0.2]})
    survival_table = pd.DataFrame({"endpoint": ["OS", "DSS"], "logrank_p": [0.3, 0.4]})
    cox_global = pd.DataFrame({"endpoint": ["OS", "DSS"], "subtype_lrt_p": [0.5, 0.6], "full_concordance": [0.61, 0.62]})
    baseline_result = {"labels": baseline_labels, "best_k": 2}
    record = make_sensitivity_variant_record(
        config=ANALYSIS_QUEUES[0],
        baseline_result=baseline_result,
        variant_group="seed",
        variant_name="seed_2024",
        modalities=ANALYSIS_QUEUES[0].modalities,
        seed=2024,
        snf_k=10,
        fixed_sample_cohort=True,
        labels=variant_labels,
        k_summary=k_summary,
        best_k=2,
        clinical_table=clinical_table,
        survival_table=survival_table,
        cox_global_table=cox_global,
    )
    assert record["variant_group"] == "seed"
    assert record["ari_to_baseline"] < 1.0
    assert record["nmi_to_baseline"] >= 0.0
    variant_summary = pd.DataFrame(
        [
            {
                "queue_key": "primary_discovery",
                "analysis_name": "primary_5_omics",
                "variant_group": "baseline",
                "variant_name": "baseline",
                "best_k": 2,
                "sample_delta_vs_baseline": 0,
                "ari_to_baseline": 1.0,
                "nmi_to_baseline": 1.0,
                "jaccard_to_baseline": 1.0,
            },
            record,
            {
                **record,
                "variant_name": "seed_2025",
                "ari_to_baseline": 0.8,
                "nmi_to_baseline": 0.85,
                "jaccard_to_baseline": 0.75,
                "best_k": 3,
            },
        ]
    )
    overview = sensitivity_overview_frame(variant_summary)
    assert len(overview) == 1
    assert overview.loc[0, "variant_group"] == "seed"
    assert np.isclose(float(overview.loc[0, "mean_ari_to_baseline"]), (record["ari_to_baseline"] + 0.8) / 2)
    assert overview.loc[0, "least_stable_variant"] in {"seed_2024", "seed_2025"}


def test_consensus_weighting_and_summary():
    model_evaluation = pd.DataFrame(
        {
            "queue_key": ["primary_discovery"] * 3,
            "analysis_name": ["primary_5_omics"] * 3,
            "model_name": ["snf", "early_integration", "nmf_integration"],
            "best_k": [2, 2, 3],
            "n_samples": [4, 4, 4],
            "selected_pac_score": [0.4, 0.6, 0.2],
            "selected_silhouette": [0.01, 0.04, 0.15],
            "msi_status_p": [1e-5, 1e-8, 1e-2],
        }
    )
    model_stability = pd.DataFrame(
        {
            "model_name": ["snf", "early_integration", "nmf_integration"],
            "overlap_nmi": [0.02, 0.6, 0.1],
            "overlap_co_clustering_jaccard": [0.2, 0.75, 0.3],
        }
    )
    weights = build_consensus_weight_table(model_evaluation, model_stability)
    assert np.isclose(float(weights["normalized_weight"].sum()), 1.0)
    assert "weight_rank" in weights.columns
    all_weights = build_consensus_weight_tables(model_evaluation, model_stability)
    assert set(all_weights["scheme_name"]) >= {"stability_focused", "balanced", "resolution_guarded", "anti_collapse"}
    grouped_sums = all_weights.groupby(["queue_key", "scheme_name"])["normalized_weight"].sum()
    assert all(np.isclose(grouped_sums.to_numpy(), 1.0))
    anti_collapse_max = all_weights.loc[all_weights["scheme_name"] == "anti_collapse", "normalized_weight"].max()
    assert anti_collapse_max <= 0.400001

    snf_labels = pd.Series(["C1", "C1", "C2", "C2"], index=["P1", "P2", "P3", "P4"], name="subtype")
    early_labels = pd.Series(["C1", "C1", "C2", "C2"], index=["P1", "P2", "P3", "P4"], name="subtype")
    nmf_labels = pd.Series(["C1", "C2", "C3", "C3"], index=["P1", "P2", "P3", "P4"], name="subtype")
    consensus_matrix = weighted_consensus_matrix(
        [
            {"model_name": "snf", "labels": snf_labels},
            {"model_name": "early_integration", "labels": early_labels},
            {"model_name": "nmf_integration", "labels": nmf_labels},
        ],
        weights,
    )
    assert consensus_matrix.shape == (4, 4)
    assert np.isclose(float(consensus_matrix.loc["P1", "P2"]), float(consensus_matrix.loc["P2", "P1"]))
    assert float(consensus_matrix.loc["P1", "P2"]) > float(consensus_matrix.loc["P1", "P3"])

    summary = consensus_summary_frame(
        [("primary_discovery", "primary_5_omics")],
        [
            {
                "queue_key": "primary_discovery",
                "best_k": 2,
                "labels": snf_labels,
                "out_dir": Path("results/consensus/primary_discovery"),
            }
        ],
        [
            ("primary_discovery", {"model_name": "snf", "labels": snf_labels}),
            ("primary_discovery", {"model_name": "early_integration", "labels": early_labels}),
            ("primary_discovery", {"model_name": "nmf_integration", "labels": nmf_labels}),
        ],
    )
    assert "ari_to_snf" in summary.columns
    assert float(summary.loc[0, "ari_to_snf"]) == 1.0


def test_consensus_candidate_selection_prefers_non_collapsed_scheme():
    candidate_summary = pd.DataFrame(
        {
            "queue_key": [
                "primary_discovery",
                "primary_discovery",
                "validation_with_rppa",
                "validation_with_rppa",
            ],
            "analysis_name": [
                "primary_5_omics",
                "primary_5_omics",
                "validation_6_omics",
                "validation_6_omics",
            ],
            "scheme_name": [
                "stability_focused",
                "anti_collapse",
                "stability_focused",
                "anti_collapse",
            ],
            "best_k": [2, 3, 2, 4],
            "selected_pac_score": [0.50, 0.18, 0.62, 0.20],
            "selected_silhouette": [0.03, 0.11, 0.04, 0.12],
            "msi_status_p": [1e-12, 1e-8, 1e-10, 1e-7],
            "dominant_model_weight": [0.55, 0.39, 0.58, 0.37],
            "weight_entropy": [0.72, 0.96, 0.70, 0.94],
            "max_ari_to_single_model": [1.0, 0.62, 1.0, 0.74],
            "is_exact_copy": [True, False, True, False],
        }
    )
    selection = select_consensus_scheme(candidate_summary)
    assert bool(selection.loc[selection["scheme_name"] == "anti_collapse", "selected"].iloc[0])


def test_consensus_candidate_summary_reports_collapse_metrics():
    weight_table = pd.DataFrame(
        {
            "queue_key": ["primary_discovery"] * 6,
            "analysis_name": ["primary_5_omics"] * 6,
            "scheme_name": ["stability_focused"] * 3 + ["anti_collapse"] * 3,
            "model_name": ["snf", "early_integration", "nmf_integration"] * 2,
            "normalized_weight": [0.25, 0.50, 0.25, 0.35, 0.30, 0.35],
            "queue_weight_entropy": [0.94] * 6,
            "queue_max_model_weight": [0.50] * 3 + [0.35] * 3,
        }
    )
    snf_labels = pd.Series(["C1", "C1", "C2", "C2"], index=["P1", "P2", "P3", "P4"], name="subtype")
    early_labels = pd.Series(["C1", "C1", "C2", "C2"], index=["P1", "P2", "P3", "P4"], name="subtype")
    nmf_labels = pd.Series(["C1", "C2", "C3", "C3"], index=["P1", "P2", "P3", "P4"], name="subtype")
    candidates = [
        {
            "queue_key": "primary_discovery",
            "analysis_name": "primary_5_omics",
            "scheme_name": "stability_focused",
            "labels": early_labels,
            "best_k": 2,
            "k_summary": pd.DataFrame(
                {"k": [2], "pac_score": [0.4], "silhouette": [0.03], "min_cluster_size": [2], "selected": [True]}
            ),
            "clinical_table": pd.DataFrame({"variable": ["msi_status"], "p_value": [0.01]}),
            "survival_table": pd.DataFrame({"endpoint": ["OS"], "logrank_p": [0.5]}),
        },
        {
            "queue_key": "primary_discovery",
            "analysis_name": "primary_5_omics",
            "scheme_name": "anti_collapse",
            "labels": nmf_labels,
            "best_k": 3,
            "k_summary": pd.DataFrame(
                {"k": [3], "pac_score": [0.2], "silhouette": [0.09], "min_cluster_size": [1], "selected": [True]}
            ),
            "clinical_table": pd.DataFrame({"variable": ["msi_status"], "p_value": [0.02]}),
            "survival_table": pd.DataFrame({"endpoint": ["OS"], "logrank_p": [0.4]}),
        },
    ]
    comparison_results = [
        ("primary_discovery", {"model_name": "snf", "labels": snf_labels}),
        ("primary_discovery", {"model_name": "early_integration", "labels": early_labels}),
        ("primary_discovery", {"model_name": "nmf_integration", "labels": nmf_labels}),
    ]
    summary = consensus_candidate_summary_frame(candidates, comparison_results, weight_table)
    assert "max_ari_to_single_model" in summary.columns
    stability_row = summary.loc[summary["scheme_name"] == "stability_focused"].iloc[0]
    anti_row = summary.loc[summary["scheme_name"] == "anti_collapse"].iloc[0]
    assert bool(stability_row["is_exact_copy"])
    assert int(anti_row["best_k"]) == 3


def test_copy_rescued_weight_tables_trim_exact_copy_model():
    weight_table = pd.DataFrame(
        {
            "queue_key": ["primary_discovery"] * 3 + ["validation_with_rppa"] * 3,
            "analysis_name": ["primary_5_omics"] * 3 + ["validation_6_omics"] * 3,
            "scheme_name": ["balanced"] * 6,
            "base_scheme_name": ["base"] * 6,
            "model_name": ["snf", "early_integration", "nmf_integration"] * 2,
            "best_k": [2, 2, 3, 4, 2, 4],
            "n_samples": [4] * 6,
            "selected_pac_score": [0.2] * 6,
            "selected_silhouette": [0.1] * 6,
            "selected_min_cluster_size": [2] * 6,
            "msi_status_p": [0.01] * 6,
            "stability_nmi": [0.1] * 6,
            "stability_jaccard": [0.2] * 6,
            "stability_score": [0.1] * 6,
            "structure_score": [0.2] * 6,
            "clinical_score": [0.3] * 6,
            "resolution_score": [0.1, 0.0, 0.8, 1.0, 0.0, 0.9],
            "scheme_stability_weight": [0.3] * 6,
            "scheme_structure_weight": [0.25] * 6,
            "scheme_clinical_weight": [0.15] * 6,
            "scheme_resolution_weight": [0.3] * 6,
            "scheme_shrinkage": [0.15] * 6,
            "scheme_max_model_weight": [0.48] * 6,
            "rescue_applied": [False] * 6,
            "rescued_model_name": [""] * 6,
            "rescue_cap": [np.nan] * 6,
            "raw_weight": [1.0] * 6,
            "pre_shrink_weight": [0.3, 0.3, 0.4, 0.304537, 0.336773, 0.358690],
            "normalized_weight": [0.3, 0.3, 0.4, 0.304537, 0.336773, 0.358690],
            "queue_weight_entropy": [0.98] * 6,
            "queue_max_model_weight": [0.4] * 6,
            "weight_rank": [2, 2, 1, 3, 2, 1],
        }
    )
    candidate_summary = pd.DataFrame(
        {
            "queue_key": ["primary_discovery", "validation_with_rppa"],
            "analysis_name": ["primary_5_omics", "validation_6_omics"],
            "scheme_name": ["balanced", "balanced"],
            "is_exact_copy": [False, True],
            "exact_copy_model_name": ["", "early_integration"],
        }
    )
    rescued = build_copy_rescued_weight_tables(weight_table, candidate_summary, rescue_cap=0.10)
    assert set(rescued["scheme_name"]) == {"balanced_copy_rescued"}
    validation = rescued[rescued["queue_key"] == "validation_with_rppa"].set_index("model_name")
    assert np.isclose(float(validation["normalized_weight"].sum()), 1.0)
    assert np.isclose(float(validation.loc["early_integration", "normalized_weight"]), 0.10)
    assert bool(validation["rescue_applied"].all())
    primary = rescued[rescued["queue_key"] == "primary_discovery"].set_index("model_name")
    assert np.isclose(float(primary.loc["snf", "normalized_weight"]), 0.3)


def test_classifier_feature_matrix_and_supervised_models():
    samples = [f"P{i}" for i in range(1, 19)]
    mutation = pd.DataFrame(
        np.random.default_rng(0).integers(0, 2, size=(6, len(samples))),
        index=[f"M{i}" for i in range(6)],
        columns=samples,
    )
    cnv = pd.DataFrame(
        np.random.default_rng(1).normal(size=(8, len(samples))),
        index=[f"C{i}" for i in range(8)],
        columns=samples,
    )
    methylation = pd.DataFrame(
        np.random.default_rng(2).normal(size=(10, len(samples))),
        index=[f"cg{i}" for i in range(10)],
        columns=samples,
    )
    expression = pd.DataFrame(
        np.random.default_rng(3).normal(size=(12, len(samples))),
        index=["EPCAM", "KRT8", "KRT18", "VIM", "CDH2", "MKI67", "PCNA", "CXCL9", "CXCL10", "STAT1", "IRF1", "FN1"],
        columns=samples,
    )
    mirna = pd.DataFrame(
        np.random.default_rng(4).normal(size=(5, len(samples))),
        index=[f"miR{i}" for i in range(5)],
        columns=samples,
    )
    embeddings = {
        "mutation": pd.DataFrame(np.random.default_rng(5).normal(size=(len(samples), 4)), index=samples, columns=[f"m{i}" for i in range(4)]),
        "expression": pd.DataFrame(np.random.default_rng(6).normal(size=(len(samples), 5)), index=samples, columns=[f"e{i}" for i in range(5)]),
    }
    feature_matrix = build_classifier_feature_matrix(
        {"mutation": mutation, "cnv": cnv, "methylation": methylation, "expression": expression, "mirna": mirna},
        embeddings,
    )
    assert set(feature_matrix.index) == set(samples)
    assert list(feature_matrix.index) == sorted(samples)
    assert any(col.startswith("summary__") for col in feature_matrix.columns)
    assert any(col.startswith("latent__mutation__") for col in feature_matrix.columns)

    labels = pd.Series(
        ["C1"] * 6 + ["C2"] * 6 + ["C3"] * 6,
        index=samples,
        name="subtype",
    )
    result = run_elastic_net_cv(feature_matrix, labels, random_state=7)
    assert not result["summary"].empty
    assert not result["predictions"].empty
    assert set(result["predictions"]["true_label"]) == {"C1", "C2", "C3"}
    assert "macro_f1" in result["summary"].columns
    assert "selected_in_final_model" in result["selected_features"].columns
    assert result["summary"]["classifier_name"].iloc[0] == "elastic_net"

    svm_result = run_svm_cv(feature_matrix, labels, random_state=7)
    assert not svm_result["summary"].empty
    assert not svm_result["predictions"].empty
    assert set(svm_result["predictions"]["true_label"]) == {"C1", "C2", "C3"}
    assert "macro_f1" in svm_result["summary"].columns
    assert "selected_in_final_model" in svm_result["selected_features"].columns
    assert svm_result["summary"]["classifier_name"].iloc[0] == "svm"
    assert not svm_result["coefficients"].empty

    xgboost_result = run_xgboost_cv(feature_matrix, labels, random_state=7)
    assert not xgboost_result["summary"].empty
    assert not xgboost_result["predictions"].empty
    assert set(xgboost_result["predictions"]["true_label"]) == {"C1", "C2", "C3"}
    assert "macro_f1" in xgboost_result["summary"].columns
    assert "selected_in_final_model" in xgboost_result["selected_features"].columns
    assert xgboost_result["summary"]["classifier_name"].iloc[0] == "xgboost"
    assert not xgboost_result["coefficients"].empty
    assert set(xgboost_result["coefficients"]["class_label"]) == {"global_importance"}

    mlp_result = run_mlp_cv(feature_matrix, labels, random_state=7)
    assert not mlp_result["summary"].empty
    assert not mlp_result["predictions"].empty
    assert set(mlp_result["predictions"]["true_label"]) == {"C1", "C2", "C3"}
    assert mlp_result["summary"]["classifier_name"].iloc[0] == "mlp"
    assert "selected_in_final_model" in mlp_result["selected_features"].columns

    lstm_result = run_lstm_cv(feature_matrix, labels, random_state=7)
    assert not lstm_result["summary"].empty
    assert not lstm_result["predictions"].empty
    assert set(lstm_result["predictions"]["true_label"]) == {"C1", "C2", "C3"}
    assert lstm_result["summary"]["classifier_name"].iloc[0] == "lstm"
    assert "sequence__n_steps" in lstm_result["final_params"]["best_params"].iloc[0]

    transformer_result = run_transformer_cv(feature_matrix, labels, random_state=7)
    assert not transformer_result["summary"].empty
    assert not transformer_result["predictions"].empty
    assert set(transformer_result["predictions"]["true_label"]) == {"C1", "C2", "C3"}
    assert transformer_result["summary"]["classifier_name"].iloc[0] == "transformer"
    assert "sequence__n_steps" in transformer_result["final_params"]["best_params"].iloc[0]

    assert set(CLASSIFIER_RUNNERS) >= {"elastic_net", "svm", "xgboost", "mlp", "lstm", "transformer"}
    assert set(SIGNATURE_SOURCE_CLASSIFIER_NAMES) == {"elastic_net", "svm", "xgboost"}


def test_signature_evidence_and_panels():
    classifier_runs = [
        {
            "queue_key": "primary_discovery",
            "analysis_name": "primary_5_omics",
            "classifier_name": "elastic_net",
            "cv_result": {
                "selected_features": pd.DataFrame(
                    {
                        "feature_name": [
                            "summary__cnv_burden",
                            "summary__tmb",
                            "latent__expression__component_1",
                            "latent__mirna__component_2",
                        ],
                        "selected_fold_count": [5, 4, 5, 3],
                        "selected_fold_fraction": [1.0, 0.8, 1.0, 0.6],
                        "selected_in_final_model": [True, True, True, False],
                    }
                ),
                "coefficients": pd.DataFrame(
                    {
                        "class_label": ["C1", "C1", "C2"],
                        "feature_name": [
                            "summary__cnv_burden",
                            "latent__expression__component_1",
                            "summary__tmb",
                        ],
                        "coefficient": [0.6, -0.4, 0.2],
                        "abs_coefficient": [0.6, 0.4, 0.2],
                    }
                ),
            },
        },
        {
            "queue_key": "validation_with_rppa",
            "analysis_name": "validation_6_omics",
            "classifier_name": "svm",
            "cv_result": {
                "selected_features": pd.DataFrame(
                    {
                        "feature_name": [
                            "summary__cnv_burden",
                            "summary__tmb",
                            "latent__expression__component_1",
                            "summary__mirna_global_abundance",
                        ],
                        "selected_fold_count": [5, 5, 4, 5],
                        "selected_fold_fraction": [1.0, 1.0, 0.8, 1.0],
                        "selected_in_final_model": [True, True, True, True],
                    }
                ),
                "coefficients": pd.DataFrame(
                    {
                        "class_label": ["C1__vs__C2", "C1__vs__C2", "C2"],
                        "feature_name": [
                            "summary__cnv_burden",
                            "summary__mirna_global_abundance",
                            "latent__expression__component_1",
                        ],
                        "coefficient": [0.5, 0.7, -0.9],
                        "abs_coefficient": [0.5, 0.7, 0.9],
                    }
                ),
            },
        },
    ]
    evidence = build_signature_feature_evidence(classifier_runs)
    assert not evidence.empty
    assert "is_cross_queue_stable" in evidence.columns
    cnv_row = evidence.set_index("feature_name").loc["summary__cnv_burden"]
    assert bool(cnv_row["is_selected_in_all_runs"])
    assert bool(cnv_row["is_cross_classifier_stable"])
    panels = build_signature_panels(evidence)
    assert not panels.empty
    assert set(panels["panel_name"]) >= {"stable_core_8", "summary_core_10"}
    summary_only = panels.loc[panels["panel_name"] == "summary_core_10", "feature_layer"]
    assert set(summary_only) == {"mechanism_summary"}


def test_signature_panel_summary_frame():
    panel_summary_rows = [
        pd.DataFrame(
            [
                {
                    "queue_key": "primary_discovery",
                    "analysis_name": "primary_5_omics",
                    "label_scheme_name": "scheme_x",
                    "output_dir": "results/signatures/stable_core_8/elastic_net/primary_discovery",
                    "panel_name": "stable_core_8",
                    "panel_feature_count": 8,
                    "classifier_name": "elastic_net",
                    "n_samples": 100,
                    "n_classes": 4,
                    "n_features": 8,
                    "outer_folds": 5,
                    "macro_f1": 0.70,
                    "balanced_accuracy": 0.71,
                    "accuracy": 0.69,
                    "macro_ovr_auroc": 0.90,
                    "n_selected_features_final": 8,
                    "best_params": "{}",
                }
            ]
        )
    ]
    full_summary = pd.DataFrame(
        [
            {
                "queue_key": "primary_discovery",
                "analysis_name": "primary_5_omics",
                "label_scheme_name": "scheme_x",
                "output_dir": "results/classifiers/elastic_net/primary_discovery",
                "classifier_name": "elastic_net",
                "n_samples": 100,
                "n_classes": 4,
                "n_features": 80,
                "outer_folds": 5,
                "macro_f1": 0.75,
                "balanced_accuracy": 0.74,
                "accuracy": 0.72,
                "macro_ovr_auroc": 0.92,
            }
        ]
    )
    summary = signature_panel_summary_frame(panel_summary_rows, full_summary)
    assert "delta_macro_f1" in summary.columns
    assert np.isclose(float(summary.loc[0, "delta_macro_f1"]), -0.05)
    assert int(summary.loc[0, "baseline_n_features"]) == 80


def test_raw_marker_mapping_and_summary():
    panel_features = pd.DataFrame(
        [
            {
                "panel_name": "stable_core_8",
                "panel_description": "demo",
                "panel_target_size": 2,
                "panel_feature_count": 2,
                "feature_rank_within_panel": 1,
                "feature_name": "latent__expression__component_1",
                "feature_layer": "latent_embedding",
                "modality_name": "expression",
                "feature_display_name": "expression component 1",
                "stability_score": 10.0,
                "is_selected_in_all_runs": True,
            },
            {
                "panel_name": "stable_core_8",
                "panel_description": "demo",
                "panel_target_size": 2,
                "panel_feature_count": 2,
                "feature_rank_within_panel": 2,
                "feature_name": "summary__cnv_burden",
                "feature_layer": "mechanism_summary",
                "modality_name": "cnv",
                "feature_display_name": "cnv burden",
                "stability_score": 9.0,
                "is_selected_in_all_runs": True,
            },
        ]
    )
    expression = pd.DataFrame(
        [[3.0, 3.1, -2.9, -3.2], [2.0, 2.1, -2.0, -2.2], [0.1, -0.1, 0.2, -0.2]],
        index=["GENE1", "GENE2", "GENE3"],
        columns=["P1", "P2", "P3", "P4"],
    )
    preprocessed = {"expression": expression.astype(np.float32)}
    probe_map = pd.DataFrame({"feature_id": ["cg1"], "gene": ["GENE_A"]})
    marker_table = build_latent_component_marker_table(panel_features, preprocessed, probe_map, top_n=2)
    assert not marker_table.empty
    assert set(marker_table["source_feature_name"]) == {"latent__expression__component_1"}
    raw_panels = build_raw_marker_panels(panel_features, marker_table)
    assert not raw_panels.empty
    assert "stable_raw_8" in set(raw_panels["raw_panel_name"])

    omics = {
        "expression": expression.astype(np.float32),
        "cnv": pd.DataFrame(
            [[1.0, 1.1, -1.0, -1.2], [0.5, 0.4, -0.4, -0.5]],
            index=["C1", "C2"],
            columns=["P1", "P2", "P3", "P4"],
        ),
    }
    raw_feature_frame = build_raw_panel_feature_matrix(raw_panels, preprocessed, omics)
    assert not raw_feature_frame.empty
    assert any(col.startswith("raw__expression__") for col in raw_feature_frame.columns)
    assert "summary__cnv_burden" in raw_feature_frame.columns

    raw_summary_rows = [
        pd.DataFrame(
            [
                {
                    "queue_key": "primary_discovery",
                    "analysis_name": "primary_5_omics",
                    "label_scheme_name": "scheme_x",
                    "output_dir": "results/signatures/stable_raw_8/elastic_net/primary_discovery",
                    "raw_panel_name": "stable_raw_8",
                    "source_panel_name": "stable_core_8",
                    "raw_panel_feature_count": 2,
                    "classifier_name": "elastic_net",
                    "n_samples": 100,
                    "n_classes": 4,
                    "n_features": 2,
                    "outer_folds": 5,
                    "macro_f1": 0.60,
                    "balanced_accuracy": 0.61,
                    "accuracy": 0.59,
                    "macro_ovr_auroc": 0.82,
                    "n_selected_features_final": 2,
                    "best_params": "{}",
                }
            ]
        )
    ]
    full_summary = pd.DataFrame(
        [
            {
                "queue_key": "primary_discovery",
                "analysis_name": "primary_5_omics",
                "label_scheme_name": "scheme_x",
                "output_dir": "results/classifiers/elastic_net/primary_discovery",
                "classifier_name": "elastic_net",
                "n_samples": 100,
                "n_classes": 4,
                "n_features": 80,
                "outer_folds": 5,
                "macro_f1": 0.75,
                "balanced_accuracy": 0.74,
                "accuracy": 0.72,
                "macro_ovr_auroc": 0.92,
            }
        ]
    )
    latent_summary = pd.DataFrame(
        [
            {
                "panel_name": "stable_core_8",
                "queue_key": "primary_discovery",
                "classifier_name": "elastic_net",
                "macro_f1": 0.70,
                "balanced_accuracy": 0.71,
                "accuracy": 0.69,
                "macro_ovr_auroc": 0.90,
            }
        ]
    )
    raw_summary = raw_signature_panel_summary_frame(raw_summary_rows, full_summary, latent_summary)
    assert "delta_macro_f1_vs_source_panel" in raw_summary.columns
    assert np.isclose(float(raw_summary.loc[0, "delta_macro_f1_vs_source_panel"]), -0.10)


def test_parse_mirbase_mature_fasta_and_cache_loader(tmp_path: Path):
    fasta_text = "\n".join(
        [
            ">hsa-miR-139-3p MIMAT0004552 Homo sapiens miR-139-3p",
            "UAAUACUGUCUGGUAAAGAUGG",
            ">mmu-miR-1a-3p MIMAT0000123 Mus musculus miR-1a-3p",
            "UGGAAUGUAAAGAAGUAUGUAU",
            ">hsa-miR-374a-3p MIMAT0004688 Homo sapiens miR-374a-3p",
            "CUUAUAUUCUUGCUGAUAAGUG",
        ]
    )
    parsed = parse_mirbase_mature_fasta(fasta_text)
    assert list(parsed["accession"]) == ["MIMAT0004552", "MIMAT0004688"]
    cache_path = tmp_path / "mature.fa"
    cache_path.write_text(fasta_text, encoding="utf-8")
    loaded = load_mirbase_mature_aliases(cache_path)
    assert loaded.set_index("accession").loc["MIMAT0004552", "mirna_name"] == "hsa-miR-139-3p"


def test_raw_proxy_scores_and_refined_panel_selection():
    panel_features = pd.DataFrame(
        [
            {
                "panel_name": "stable_core_8",
                "panel_description": "demo",
                "panel_target_size": 2,
                "panel_feature_count": 2,
                "feature_rank_within_panel": 1,
                "feature_name": "latent__mirna__component_1",
                "feature_layer": "latent_embedding",
                "modality_name": "mirna",
                "feature_display_name": "mirna component 1",
                "stability_score": 10.0,
                "is_selected_in_all_runs": True,
            },
            {
                "panel_name": "stable_core_8",
                "panel_description": "demo",
                "panel_target_size": 2,
                "panel_feature_count": 2,
                "feature_rank_within_panel": 2,
                "feature_name": "latent__mutation__component_1",
                "feature_layer": "latent_embedding",
                "modality_name": "mutation",
                "feature_display_name": "mutation component 1",
                "stability_score": 9.0,
                "is_selected_in_all_runs": True,
            },
        ]
    )
    component_marker_table = pd.DataFrame(
        [
            {
                "panel_name": "stable_core_8",
                "source_feature_name": "latent__mirna__component_1",
                "modality_name": "mirna",
                "component_index": 1,
                "loading_rank": 1,
                "raw_feature_id": "MIMAT_A",
                "raw_feature_annotation": "hsa-miR-A (MIMAT_A)",
                "loading": 0.10,
                "abs_loading": 0.10,
            },
            {
                "panel_name": "stable_core_8",
                "source_feature_name": "latent__mirna__component_1",
                "modality_name": "mirna",
                "component_index": 1,
                "loading_rank": 2,
                "raw_feature_id": "MIMAT_B",
                "raw_feature_annotation": "hsa-miR-B (MIMAT_B)",
                "loading": 0.09,
                "abs_loading": 0.09,
            },
            {
                "panel_name": "stable_core_8",
                "source_feature_name": "latent__mutation__component_1",
                "modality_name": "mutation",
                "component_index": 1,
                "loading_rank": 1,
                "raw_feature_id": "TTN",
                "raw_feature_annotation": "TTN",
                "loading": 0.15,
                "abs_loading": 0.15,
            },
            {
                "panel_name": "stable_core_8",
                "source_feature_name": "latent__mutation__component_1",
                "modality_name": "mutation",
                "component_index": 1,
                "loading_rank": 2,
                "raw_feature_id": "ARID1A",
                "raw_feature_annotation": "ARID1A",
                "loading": 0.14,
                "abs_loading": 0.14,
            },
        ]
    )
    prepared_primary = SimpleNamespace(
        preprocessed={
            "mirna": pd.DataFrame(
                [[-1.0, 0.1, 0.3, 1.0], [-1.0, -0.8, 0.9, 1.1]],
                index=["MIMAT_A", "MIMAT_B"],
                columns=["P1", "P2", "P3", "P4"],
            ),
            "mutation": pd.DataFrame(
                [[1.0, 0.8, 0.2, 0.0], [0.0, 0.0, 1.0, 1.0]],
                index=["TTN", "ARID1A"],
                columns=["P1", "P2", "P3", "P4"],
            ),
        },
        embeddings={
            "mirna": pd.DataFrame({"component_1": [-1.0, -0.8, 0.9, 1.1]}, index=["P1", "P2", "P3", "P4"]),
            "mutation": pd.DataFrame({"component_1": [-1.0, -0.9, 0.8, 1.0]}, index=["P1", "P2", "P3", "P4"]),
        },
    )
    prepared_validation = SimpleNamespace(
        preprocessed={
            "mirna": pd.DataFrame(
                [[-1.2, -0.1, 0.4, 0.9], [-0.8, -1.0, 0.7, 1.0]],
                index=["MIMAT_A", "MIMAT_B"],
                columns=["V1", "V2", "V3", "V4"],
            ),
            "mutation": pd.DataFrame(
                [[1.0, 0.7, 0.3, 0.1], [0.0, 0.0, 1.0, 1.0]],
                index=["TTN", "ARID1A"],
                columns=["V1", "V2", "V3", "V4"],
            ),
        },
        embeddings={
            "mirna": pd.DataFrame({"component_1": [-0.9, -1.1, 0.8, 1.2]}, index=["V1", "V2", "V3", "V4"]),
            "mutation": pd.DataFrame({"component_1": [-0.9, -1.0, 0.9, 1.1]}, index=["V1", "V2", "V3", "V4"]),
        },
    )
    proxy_scores = build_raw_proxy_candidate_scores(
        component_marker_table,
        {
            "primary_discovery": prepared_primary,
            "validation_with_rppa": prepared_validation,
        },
    )
    best_mirna = proxy_scores.loc[proxy_scores["source_feature_name"] == "latent__mirna__component_1"].iloc[0]
    best_mutation = proxy_scores.loc[proxy_scores["source_feature_name"] == "latent__mutation__component_1"].iloc[0]
    assert best_mirna["raw_feature_id"] == "MIMAT_B"
    assert best_mutation["raw_feature_id"] == "ARID1A"

    refined = build_refined_raw_marker_panels(panel_features, component_marker_table, proxy_scores)
    assert "stable_refined_8" in set(refined["raw_panel_name"])
    assert "raw__mirna__MIMAT_B" in set(refined["mapped_feature_name"])
    assert "raw__mutation__ARID1A" in set(refined["mapped_feature_name"])


def test_refined_panel_freeze_candidates_collapse_duplicate_feature_sets():
    refined_panel_features = pd.DataFrame(
        [
            {
                "raw_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "raw_panel_description": "demo",
                "source_feature_name": "latent__expression__component_1",
                "mapped_feature_name": "raw__expression__PROC",
                "mapped_feature_layer": "raw_selected",
                "modality_name": "expression",
                "mapped_feature_annotation": "PROC",
                "mapping_source": "PROC",
                "raw_panel_feature_count": 2,
                "feature_rank_within_panel": 1,
            },
            {
                "raw_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "raw_panel_description": "demo",
                "source_feature_name": "summary__tmb",
                "mapped_feature_name": "summary__tmb",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "tmb",
                "mapped_feature_annotation": "tmb",
                "mapping_source": "summary_preserved",
                "raw_panel_feature_count": 2,
                "feature_rank_within_panel": 2,
            },
            {
                "raw_panel_name": "hybrid_refined_12",
                "source_panel_name": "hybrid_core_12",
                "raw_panel_description": "demo",
                "source_feature_name": "summary__tmb",
                "mapped_feature_name": "summary__tmb",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "tmb",
                "mapped_feature_annotation": "tmb",
                "mapping_source": "summary_preserved",
                "raw_panel_feature_count": 2,
                "feature_rank_within_panel": 1,
            },
            {
                "raw_panel_name": "hybrid_refined_12",
                "source_panel_name": "hybrid_core_12",
                "raw_panel_description": "demo",
                "source_feature_name": "latent__expression__component_1",
                "mapped_feature_name": "raw__expression__PROC",
                "mapped_feature_layer": "raw_selected",
                "modality_name": "expression",
                "mapped_feature_annotation": "PROC",
                "mapping_source": "PROC",
                "raw_panel_feature_count": 2,
                "feature_rank_within_panel": 2,
            },
            {
                "raw_panel_name": "stable_refined_8",
                "source_panel_name": "stable_core_8",
                "raw_panel_description": "demo",
                "source_feature_name": "latent__expression__component_1",
                "mapped_feature_name": "raw__expression__PROC",
                "mapped_feature_layer": "raw_selected",
                "modality_name": "expression",
                "mapped_feature_annotation": "PROC",
                "mapping_source": "PROC",
                "raw_panel_feature_count": 1,
                "feature_rank_within_panel": 1,
            },
        ]
    )
    refined_panel_summary = pd.DataFrame(
        [
            {
                "raw_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "queue_key": "validation_with_rppa",
                "classifier_name": "svm",
                "macro_f1": 0.56,
                "delta_macro_f1_vs_source_panel": -0.001,
            },
            {
                "raw_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "queue_key": "validation_with_rppa",
                "classifier_name": "elastic_net",
                "macro_f1": 0.55,
                "delta_macro_f1_vs_source_panel": -0.02,
            },
            {
                "raw_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "queue_key": "primary_discovery",
                "classifier_name": "svm",
                "macro_f1": 0.48,
                "delta_macro_f1_vs_source_panel": -0.01,
            },
            {
                "raw_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "queue_key": "primary_discovery",
                "classifier_name": "elastic_net",
                "macro_f1": 0.45,
                "delta_macro_f1_vs_source_panel": -0.03,
            },
            {
                "raw_panel_name": "hybrid_refined_12",
                "source_panel_name": "hybrid_core_12",
                "queue_key": "validation_with_rppa",
                "classifier_name": "svm",
                "macro_f1": 0.56,
                "delta_macro_f1_vs_source_panel": -0.001,
            },
            {
                "raw_panel_name": "hybrid_refined_12",
                "source_panel_name": "hybrid_core_12",
                "queue_key": "validation_with_rppa",
                "classifier_name": "elastic_net",
                "macro_f1": 0.55,
                "delta_macro_f1_vs_source_panel": -0.02,
            },
            {
                "raw_panel_name": "hybrid_refined_12",
                "source_panel_name": "hybrid_core_12",
                "queue_key": "primary_discovery",
                "classifier_name": "svm",
                "macro_f1": 0.48,
                "delta_macro_f1_vs_source_panel": -0.01,
            },
            {
                "raw_panel_name": "hybrid_refined_12",
                "source_panel_name": "hybrid_core_12",
                "queue_key": "primary_discovery",
                "classifier_name": "elastic_net",
                "macro_f1": 0.45,
                "delta_macro_f1_vs_source_panel": -0.03,
            },
            {
                "raw_panel_name": "stable_refined_8",
                "source_panel_name": "stable_core_8",
                "queue_key": "validation_with_rppa",
                "classifier_name": "svm",
                "macro_f1": 0.54,
                "delta_macro_f1_vs_source_panel": -0.03,
            },
            {
                "raw_panel_name": "stable_refined_8",
                "source_panel_name": "stable_core_8",
                "queue_key": "validation_with_rppa",
                "classifier_name": "elastic_net",
                "macro_f1": 0.56,
                "delta_macro_f1_vs_source_panel": -0.01,
            },
            {
                "raw_panel_name": "stable_refined_8",
                "source_panel_name": "stable_core_8",
                "queue_key": "primary_discovery",
                "classifier_name": "svm",
                "macro_f1": 0.47,
                "delta_macro_f1_vs_source_panel": -0.02,
            },
            {
                "raw_panel_name": "stable_refined_8",
                "source_panel_name": "stable_core_8",
                "queue_key": "primary_discovery",
                "classifier_name": "elastic_net",
                "macro_f1": 0.45,
                "delta_macro_f1_vs_source_panel": -0.04,
            },
        ]
    )
    candidates = build_refined_panel_freeze_candidates(refined_panel_features, refined_panel_summary)
    assert len(candidates) == 2
    selected = candidates.loc[candidates["selected"]].iloc[0]
    assert selected["representative_panel_name"] == "stable_refined_12"
    assert int(selected["n_equivalent_panels"]) == 2
    frozen_features = extract_frozen_refined_panel_features(refined_panel_features, candidates)
    assert set(frozen_features["mapped_feature_name"]) == {"raw__expression__PROC", "summary__tmb"}
    markdown = frozen_refined_signature_markdown(candidates, frozen_features)
    assert "stable_refined_12" in markdown
    assert "Equivalent panels collapsed" in markdown


def test_frozen_reduction_candidates_and_selection():
    frozen_panel_features = pd.DataFrame(
        [
            {
                "frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "source_feature_name": "latent__cnv__component_1",
                "mapped_feature_name": "raw__cnv__TMEM189",
                "mapped_feature_layer": "raw_selected",
                "modality_name": "cnv",
                "mapped_feature_annotation": "TMEM189",
                "mapping_source": "TMEM189",
                "raw_panel_feature_count": 12,
                "feature_rank_within_panel": 1,
            },
            {
                "frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "source_feature_name": "summary__cnv_loss_burden",
                "mapped_feature_name": "summary__cnv_loss_burden",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "cnv",
                "mapped_feature_annotation": "cnv loss burden",
                "mapping_source": "summary_preserved",
                "raw_panel_feature_count": 12,
                "feature_rank_within_panel": 2,
            },
            {
                "frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "source_feature_name": "summary__cnv_altered_fraction",
                "mapped_feature_name": "summary__cnv_altered_fraction",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "cnv",
                "mapped_feature_annotation": "cnv altered fraction",
                "mapping_source": "summary_preserved",
                "raw_panel_feature_count": 12,
                "feature_rank_within_panel": 3,
            },
            {
                "frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "source_feature_name": "summary__cnv_burden",
                "mapped_feature_name": "summary__cnv_burden",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "cnv",
                "mapped_feature_annotation": "cnv burden",
                "mapping_source": "summary_preserved",
                "raw_panel_feature_count": 12,
                "feature_rank_within_panel": 4,
            },
            {
                "frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "source_feature_name": "summary__tmb",
                "mapped_feature_name": "summary__tmb",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "mutation",
                "mapped_feature_annotation": "tmb",
                "mapping_source": "summary_preserved",
                "raw_panel_feature_count": 12,
                "feature_rank_within_panel": 5,
            },
            {
                "frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "source_feature_name": "summary__mutation_event_fraction",
                "mapped_feature_name": "summary__mutation_event_fraction",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "mutation",
                "mapped_feature_annotation": "mutation event fraction",
                "mapping_source": "summary_preserved",
                "raw_panel_feature_count": 12,
                "feature_rank_within_panel": 6,
            },
        ]
    )
    reduced = build_frozen_panel_reduction_candidates(frozen_panel_features)
    assert not reduced.empty
    assert "stable_reduced_9_cnv_loss_tmb" in set(reduced["reduced_panel_name"])
    assert "stable_reduced_10_cnv_loss_only" in set(reduced["reduced_panel_name"])

    full_classifier_summary = pd.DataFrame(
        [
            {"queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.64, "balanced_accuracy": 0.64, "accuracy": 0.64, "macro_ovr_auroc": 0.86, "n_features": 100},
            {"queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.63, "balanced_accuracy": 0.63, "accuracy": 0.63, "macro_ovr_auroc": 0.85, "n_features": 100},
            {"queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.56, "balanced_accuracy": 0.57, "accuracy": 0.56, "macro_ovr_auroc": 0.77, "n_features": 100},
            {"queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.55, "balanced_accuracy": 0.56, "accuracy": 0.55, "macro_ovr_auroc": 0.74, "n_features": 100},
        ]
    )
    latent_panel_summary = pd.DataFrame(
        [
            {"panel_name": "stable_core_12", "queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.50, "balanced_accuracy": 0.50, "accuracy": 0.50, "macro_ovr_auroc": 0.76},
            {"panel_name": "stable_core_12", "queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.48, "balanced_accuracy": 0.49, "accuracy": 0.48, "macro_ovr_auroc": 0.75},
            {"panel_name": "stable_core_12", "queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.56, "balanced_accuracy": 0.56, "accuracy": 0.56, "macro_ovr_auroc": 0.78},
            {"panel_name": "stable_core_12", "queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.57, "balanced_accuracy": 0.57, "accuracy": 0.57, "macro_ovr_auroc": 0.76},
        ]
    )
    frozen_panel_summary = pd.DataFrame(
        [
            {"raw_panel_name": "stable_refined_12", "queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.48, "balanced_accuracy": 0.52, "accuracy": 0.50, "macro_ovr_auroc": 0.74},
            {"raw_panel_name": "stable_refined_12", "queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.50, "balanced_accuracy": 0.51, "accuracy": 0.50, "macro_ovr_auroc": 0.76},
            {"raw_panel_name": "stable_refined_12", "queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.558, "balanced_accuracy": 0.57, "accuracy": 0.56, "macro_ovr_auroc": 0.758},
            {"raw_panel_name": "stable_refined_12", "queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.495, "balanced_accuracy": 0.50, "accuracy": 0.49, "macro_ovr_auroc": 0.712},
        ]
    )
    summary = reduction_signature_panel_summary_frame(
        [
            pd.DataFrame(
                [
                    {"queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.49, "balanced_accuracy": 0.52, "accuracy": 0.50, "macro_ovr_auroc": 0.745, "n_features": 10, "reduced_panel_name": "stable_reduced_10_cnv_loss_only", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.56, "balanced_accuracy": 0.57, "accuracy": 0.56, "macro_ovr_auroc": 0.759, "n_features": 10, "reduced_panel_name": "stable_reduced_10_cnv_loss_only", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.515, "balanced_accuracy": 0.52, "accuracy": 0.51, "macro_ovr_auroc": 0.765, "n_features": 10, "reduced_panel_name": "stable_reduced_10_cnv_loss_only", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.52, "balanced_accuracy": 0.52, "accuracy": 0.51, "macro_ovr_auroc": 0.72, "n_features": 10, "reduced_panel_name": "stable_reduced_10_cnv_loss_only", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.47, "balanced_accuracy": 0.51, "accuracy": 0.49, "macro_ovr_auroc": 0.735, "n_features": 9, "reduced_panel_name": "stable_reduced_9_cnv_loss_tmb", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.552, "balanced_accuracy": 0.565, "accuracy": 0.552, "macro_ovr_auroc": 0.756, "n_features": 9, "reduced_panel_name": "stable_reduced_9_cnv_loss_tmb", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.505, "balanced_accuracy": 0.51, "accuracy": 0.50, "macro_ovr_auroc": 0.758, "n_features": 9, "reduced_panel_name": "stable_reduced_9_cnv_loss_tmb", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.505, "balanced_accuracy": 0.508, "accuracy": 0.50, "macro_ovr_auroc": 0.716, "n_features": 9, "reduced_panel_name": "stable_reduced_9_cnv_loss_tmb", "parent_frozen_panel_name": "stable_refined_12", "source_panel_name": "stable_core_12"},
                ]
            )
        ],
        full_classifier_summary,
        latent_panel_summary,
        frozen_panel_summary,
    )
    selection = build_reduced_panel_selection(reduced, summary)
    assert not selection.empty
    chosen = selection.loc[selection["selected"]].iloc[0]
    assert chosen["reduced_panel_name"] == "stable_reduced_10_cnv_loss_only"
    assert chosen["validation_xgboost_delta_vs_parent_frozen"] > 0


def test_final_panel_candidates_and_selection():
    reduced_panel_features = pd.DataFrame(
        [
            {
                "reduced_panel_name": "stable_reduced_10_cnv_burden_only",
                "parent_frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "reduction_strategy": "cnv_group_collapse",
                "reduction_description": "keep cnv burden only",
                "dropped_feature_names": "summary__cnv_altered_fraction,summary__cnv_loss_burden",
                "raw_panel_feature_count": 10,
                "feature_rank_within_panel": 1,
                "source_feature_name": "latent__cnv__component_1",
                "mapped_feature_name": "raw__cnv__TMEM189",
                "mapped_feature_layer": "raw_selected",
                "modality_name": "cnv",
                "mapped_feature_annotation": "TMEM189",
                "mapping_source": "TMEM189",
            },
            {
                "reduced_panel_name": "stable_reduced_10_cnv_burden_only",
                "parent_frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "reduction_strategy": "cnv_group_collapse",
                "reduction_description": "keep cnv burden only",
                "dropped_feature_names": "summary__cnv_altered_fraction,summary__cnv_loss_burden",
                "raw_panel_feature_count": 10,
                "feature_rank_within_panel": 2,
                "source_feature_name": "summary__cnv_burden",
                "mapped_feature_name": "summary__cnv_burden",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "cnv",
                "mapped_feature_annotation": "cnv burden",
                "mapping_source": "summary_preserved",
            },
            {
                "reduced_panel_name": "stable_reduced_10_cnv_burden_only",
                "parent_frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "reduction_strategy": "cnv_group_collapse",
                "reduction_description": "keep cnv burden only",
                "dropped_feature_names": "summary__cnv_altered_fraction,summary__cnv_loss_burden",
                "raw_panel_feature_count": 10,
                "feature_rank_within_panel": 3,
                "source_feature_name": "summary__tmb",
                "mapped_feature_name": "summary__tmb",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "mutation",
                "mapped_feature_annotation": "tmb",
                "mapping_source": "summary_preserved",
            },
            {
                "reduced_panel_name": "stable_reduced_10_cnv_burden_only",
                "parent_frozen_panel_name": "stable_refined_12",
                "source_panel_name": "stable_core_12",
                "reduction_strategy": "cnv_group_collapse",
                "reduction_description": "keep cnv burden only",
                "dropped_feature_names": "summary__cnv_altered_fraction,summary__cnv_loss_burden",
                "raw_panel_feature_count": 10,
                "feature_rank_within_panel": 4,
                "source_feature_name": "summary__mutation_event_fraction",
                "mapped_feature_name": "summary__mutation_event_fraction",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "mutation",
                "mapped_feature_annotation": "mutation event fraction",
                "mapping_source": "summary_preserved",
            },
        ]
    )
    candidates = build_final_panel_candidates(reduced_panel_features)
    assert set(candidates["final_panel_name"]) == {
        "stable_final_9_no_mutation_event_fraction",
        "stable_final_9_no_tmb",
        "stable_final_8_cnv_burden_only",
    }
    full_classifier_summary = pd.DataFrame(
        [
            {"queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.64, "balanced_accuracy": 0.64, "accuracy": 0.64, "macro_ovr_auroc": 0.86, "n_features": 100},
            {"queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.63, "balanced_accuracy": 0.63, "accuracy": 0.63, "macro_ovr_auroc": 0.85, "n_features": 100},
            {"queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.56, "balanced_accuracy": 0.57, "accuracy": 0.56, "macro_ovr_auroc": 0.77, "n_features": 100},
            {"queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.55, "balanced_accuracy": 0.56, "accuracy": 0.55, "macro_ovr_auroc": 0.74, "n_features": 100},
        ]
    )
    latent_panel_summary = pd.DataFrame(
        [
            {"panel_name": "stable_core_12", "queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.50, "balanced_accuracy": 0.50, "accuracy": 0.50, "macro_ovr_auroc": 0.76},
            {"panel_name": "stable_core_12", "queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.48, "balanced_accuracy": 0.49, "accuracy": 0.48, "macro_ovr_auroc": 0.75},
            {"panel_name": "stable_core_12", "queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.56, "balanced_accuracy": 0.56, "accuracy": 0.56, "macro_ovr_auroc": 0.78},
            {"panel_name": "stable_core_12", "queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.57, "balanced_accuracy": 0.57, "accuracy": 0.57, "macro_ovr_auroc": 0.76},
        ]
    )
    reduced_panel_summary = pd.DataFrame(
        [
            {"reduced_panel_name": "stable_reduced_10_cnv_burden_only", "queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.472, "balanced_accuracy": 0.52, "accuracy": 0.50, "macro_ovr_auroc": 0.745},
            {"reduced_panel_name": "stable_reduced_10_cnv_burden_only", "queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.505, "balanced_accuracy": 0.52, "accuracy": 0.51, "macro_ovr_auroc": 0.755},
            {"reduced_panel_name": "stable_reduced_10_cnv_burden_only", "queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.580, "balanced_accuracy": 0.58, "accuracy": 0.57, "macro_ovr_auroc": 0.758},
            {"reduced_panel_name": "stable_reduced_10_cnv_burden_only", "queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.534, "balanced_accuracy": 0.53, "accuracy": 0.52, "macro_ovr_auroc": 0.702},
        ]
    )
    final_summary = final_signature_panel_summary_frame(
        [
            pd.DataFrame(
                [
                    {"queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.474, "balanced_accuracy": 0.52, "accuracy": 0.50, "macro_ovr_auroc": 0.745, "n_features": 9, "final_panel_name": "stable_final_9_no_mutation_event_fraction", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.582, "balanced_accuracy": 0.58, "accuracy": 0.57, "macro_ovr_auroc": 0.759, "n_features": 9, "final_panel_name": "stable_final_9_no_mutation_event_fraction", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.498, "balanced_accuracy": 0.52, "accuracy": 0.50, "macro_ovr_auroc": 0.751, "n_features": 9, "final_panel_name": "stable_final_9_no_mutation_event_fraction", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.528, "balanced_accuracy": 0.53, "accuracy": 0.52, "macro_ovr_auroc": 0.704, "n_features": 9, "final_panel_name": "stable_final_9_no_mutation_event_fraction", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.465, "balanced_accuracy": 0.51, "accuracy": 0.49, "macro_ovr_auroc": 0.739, "n_features": 9, "final_panel_name": "stable_final_9_no_tmb", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.572, "balanced_accuracy": 0.57, "accuracy": 0.56, "macro_ovr_auroc": 0.756, "n_features": 9, "final_panel_name": "stable_final_9_no_tmb", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.487, "balanced_accuracy": 0.51, "accuracy": 0.50, "macro_ovr_auroc": 0.748, "n_features": 9, "final_panel_name": "stable_final_9_no_tmb", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.515, "balanced_accuracy": 0.52, "accuracy": 0.51, "macro_ovr_auroc": 0.698, "n_features": 9, "final_panel_name": "stable_final_9_no_tmb", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "svm", "macro_f1": 0.458, "balanced_accuracy": 0.50, "accuracy": 0.49, "macro_ovr_auroc": 0.732, "n_features": 8, "final_panel_name": "stable_final_8_cnv_burden_only", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "svm", "macro_f1": 0.568, "balanced_accuracy": 0.57, "accuracy": 0.56, "macro_ovr_auroc": 0.752, "n_features": 8, "final_panel_name": "stable_final_8_cnv_burden_only", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "primary_discovery", "classifier_name": "xgboost", "macro_f1": 0.480, "balanced_accuracy": 0.50, "accuracy": 0.49, "macro_ovr_auroc": 0.742, "n_features": 8, "final_panel_name": "stable_final_8_cnv_burden_only", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                    {"queue_key": "validation_with_rppa", "classifier_name": "xgboost", "macro_f1": 0.500, "balanced_accuracy": 0.51, "accuracy": 0.50, "macro_ovr_auroc": 0.690, "n_features": 8, "final_panel_name": "stable_final_8_cnv_burden_only", "parent_reduced_panel_name": "stable_reduced_10_cnv_burden_only", "source_panel_name": "stable_core_12"},
                ]
            )
        ],
        full_classifier_summary,
        latent_panel_summary,
        reduced_panel_summary,
    )
    reduced_panel_selection = pd.DataFrame(
        [
            {
                "reduced_panel_name": "stable_reduced_10_cnv_burden_only",
                "source_panel_name": "stable_core_12",
                "selected": True,
            }
        ]
    )
    selection = build_final_panel_selection(
        candidates,
        final_summary,
        reduced_panel_features,
        reduced_panel_summary,
        reduced_panel_selection,
    )
    assert not selection.empty
    chosen = selection.loc[selection["selected"]].iloc[0]
    assert chosen["final_panel_name"] in {"stable_reduced_10_cnv_burden_only", "stable_final_9_no_mutation_event_fraction"}
    selected_features = extract_selected_final_panel_features(candidates, selection, reduced_panel_features)
    assert not selected_features.empty
    markdown = final_signature_shortlist_markdown(candidates, final_summary, selection, selected_features)
    assert "Final Signature Decision" in markdown


def test_final_signature_interpretation_and_classifier_evidence(tmp_path: Path):
    panel_features = pd.DataFrame(
        [
            {
                "selected_final_panel_name": "stable_reduced_10_cnv_burden_only",
                "feature_rank_within_panel": 1,
                "source_feature_name": "latent__cnv__component_1",
                "mapped_feature_name": "raw__cnv__TMEM189",
                "mapped_feature_layer": "raw_selected",
                "modality_name": "cnv",
                "mapped_feature_annotation": "TMEM189",
            },
            {
                "selected_final_panel_name": "stable_reduced_10_cnv_burden_only",
                "feature_rank_within_panel": 2,
                "source_feature_name": "summary__cnv_burden",
                "mapped_feature_name": "summary__cnv_burden",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "cnv",
                "mapped_feature_annotation": "cnv burden",
            },
            {
                "selected_final_panel_name": "stable_reduced_10_cnv_burden_only",
                "feature_rank_within_panel": 3,
                "source_feature_name": "summary__tmb",
                "mapped_feature_name": "summary__tmb",
                "mapped_feature_layer": "mechanism_summary",
                "modality_name": "mutation",
                "mapped_feature_annotation": "tmb",
            },
            {
                "selected_final_panel_name": "stable_reduced_10_cnv_burden_only",
                "feature_rank_within_panel": 4,
                "source_feature_name": "latent__expression__component_4",
                "mapped_feature_name": "raw__expression__CDX1",
                "mapped_feature_layer": "raw_selected",
                "modality_name": "expression",
                "mapped_feature_annotation": "CDX1",
            },
        ]
    )
    coefficient_dir = tmp_path / "stable_reduced_10_cnv_burden_only" / "elastic_net" / "primary_discovery"
    coefficient_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"class_label": "C1", "feature_name": "summary__cnv_burden", "coefficient": 0.7, "abs_coefficient": 0.7},
            {"class_label": "C1", "feature_name": "raw__expression__CDX1", "coefficient": -0.2, "abs_coefficient": 0.2},
            {"class_label": "C2", "feature_name": "summary__cnv_burden", "coefficient": -0.9, "abs_coefficient": 0.9},
        ]
    ).to_csv(coefficient_dir / "coefficients.tsv", sep="\t", index=False)

    evidence = collect_final_signature_classifier_evidence(
        tmp_path,
        "stable_reduced_10_cnv_burden_only",
        panel_features["mapped_feature_name"].tolist(),
    )
    assert float(evidence.loc[evidence["feature_name"] == "summary__cnv_burden", "max_abs_coefficient"].iloc[0]) == 0.9

    interpretation = build_final_signature_interpretation(panel_features, evidence)
    assert list(interpretation["feature_name"]) == [
        "raw__cnv__TMEM189",
        "summary__cnv_burden",
        "summary__tmb",
        "raw__expression__CDX1",
    ]
    assert "copy-number instability" in set(interpretation["biological_axis"])
    assert "intestinal differentiation program" in set(interpretation["biological_axis"])
    assert int(interpretation.loc[interpretation["feature_name"] == "summary__cnv_burden", "n_classifier_queue_runs"].iloc[0]) == 1

    selection = pd.DataFrame(
        [
            {
                "selected": True,
                "validation_mean_macro_f1": 0.552,
                "validation_mean_delta_vs_parent_reduced": 0.0,
                "is_safe_shrink": False,
            }
        ]
    )
    markdown = final_signature_interpretation_markdown(interpretation, selection)
    assert "Final Signature Biological Interpretation" in markdown
    assert "copy-number instability" in markdown
    assert "CDX1" in markdown
