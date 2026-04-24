from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess import fill_missing_by_row_median, zscore_rows

EXPRESSION_SIGNATURES: dict[str, list[str]] = {
    "expression_epithelial_score": ["EPCAM", "KRT8", "KRT18", "KRT19", "CLDN3", "CLDN4", "CLDN7", "CDH1"],
    "expression_emt_score": ["VIM", "CDH2", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "FN1"],
    "expression_proliferation_score": ["MKI67", "PCNA", "TOP2A", "CCNB1", "UBE2C", "AURKA", "CDK1"],
    "expression_ifng_immune_score": ["CXCL9", "CXCL10", "STAT1", "IRF1", "HLA-DRA", "IDO1", "IFIT1"],
}

RPPA_SIGNATURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rppa_pi3k_akt_mtor_score": ("akt", "mtor", "4e-bp1", "ampk", "gsk3", "acc", "eef2", "irs1", "ndrg1"),
    "rppa_mapk_rtk_score": ("egfr", "her2", "her3", "c-met", "mapk", "erk", "mek1", "raf", "jnk", "kit"),
    "rppa_cell_cycle_dna_damage_score": ("cyclin", "cdk1", "chk1", "chk2", "rb", "foxm1", "53bp1", "atm", "brca2", "mre11", "msh2", "msh6"),
    "rppa_emt_adhesion_score": ("e-cadherin", "n-cadherin", "catenin", "claudin-7", "caveolin-1", "fibronectin", "collagen_vi", "myosin"),
}


def _sample_index(omics: dict[str, pd.DataFrame]) -> pd.Index:
    if not omics:
        return pd.Index([], name="patient_id")
    sample_ids = next(iter(omics.values())).columns
    return pd.Index(sample_ids, name="patient_id")


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return zscore_rows(fill_missing_by_row_median(frame))


def _mean_signature_scores(frame: pd.DataFrame, signatures: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    normalized = _normalized_frame(frame)
    out = pd.DataFrame(index=normalized.columns)
    matched_counts: dict[str, int] = {}
    for feature_name, genes in signatures.items():
        matched = [gene for gene in genes if gene in normalized.index]
        matched_counts[feature_name] = len(matched)
        if not matched:
            continue
        out[feature_name] = normalized.loc[matched].mean(axis=0).astype(np.float32)
    return out, matched_counts


def _keyword_signature_scores(frame: pd.DataFrame, signatures: dict[str, tuple[str, ...]]) -> tuple[pd.DataFrame, dict[str, int]]:
    normalized = _normalized_frame(frame)
    out = pd.DataFrame(index=normalized.columns)
    feature_names = normalized.index.astype(str)
    lowered = pd.Index(feature_names.str.lower())
    matched_counts: dict[str, int] = {}
    for summary_name, keywords in signatures.items():
        mask = lowered.map(lambda value: any(keyword in value for keyword in keywords))
        matched = feature_names[mask]
        matched_counts[summary_name] = int(len(matched))
        if len(matched) == 0:
            continue
        out[summary_name] = normalized.loc[list(matched)].mean(axis=0).astype(np.float32)
    return out, matched_counts


def compute_mechanism_summary_features(omics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    samples = _sample_index(omics)
    features = pd.DataFrame(index=samples)

    mutation = omics.get("mutation")
    if mutation is not None:
        mutation = mutation.fillna(0)
        features["tmb"] = (mutation != 0).sum(axis=0).astype(np.float32)
        features["mutation_event_fraction"] = (mutation != 0).mean(axis=0).astype(np.float32)

    cnv = omics.get("cnv")
    if cnv is not None:
        cnv = cnv.fillna(0)
        features["cnv_burden"] = cnv.abs().sum(axis=0).astype(np.float32)
        features["cnv_gain_burden"] = cnv.clip(lower=0).sum(axis=0).astype(np.float32)
        features["cnv_loss_burden"] = (-cnv.clip(upper=0)).sum(axis=0).astype(np.float32)
        features["cnv_altered_fraction"] = (cnv != 0).mean(axis=0).astype(np.float32)

    methylation = omics.get("methylation")
    if methylation is not None:
        methylation = fill_missing_by_row_median(methylation)
        features["global_methylation"] = methylation.mean(axis=0).astype(np.float32)

    expression = omics.get("expression")
    if expression is not None:
        expression_scores, _ = _mean_signature_scores(expression, EXPRESSION_SIGNATURES)
        features = features.join(expression_scores, how="left")

    mirna = omics.get("mirna")
    if mirna is not None:
        mirna_imputed = fill_missing_by_row_median(mirna)
        mirna_medians = mirna_imputed.median(axis=1)
        features["mirna_global_abundance"] = mirna_imputed.mean(axis=0).astype(np.float32)
        features["mirna_high_abundance_fraction"] = mirna_imputed.gt(mirna_medians, axis=0).mean(axis=0).astype(np.float32)

    rppa = omics.get("rppa")
    if rppa is not None:
        rppa_scores, _ = _keyword_signature_scores(rppa, RPPA_SIGNATURE_KEYWORDS)
        features = features.join(rppa_scores, how="left")

    return features.sort_index()


def mechanism_summary_feature_metadata(omics: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    expression_matches = (
        _mean_signature_scores(omics["expression"], EXPRESSION_SIGNATURES)[1]
        if omics is not None and "expression" in omics
        else {name: 0 for name in EXPRESSION_SIGNATURES}
    )
    rppa_matches = (
        _keyword_signature_scores(omics["rppa"], RPPA_SIGNATURE_KEYWORDS)[1]
        if omics is not None and "rppa" in omics
        else {name: 0 for name in RPPA_SIGNATURE_KEYWORDS}
    )
    records = [
        {
            "feature_name": "tmb",
            "feature_layer": "mechanism_summary",
            "source_modality": "mutation",
            "description": "Count of non-zero mutation features per patient in the raw mutation matrix.",
            "unit": "mutated_features",
            "formula": "(mutation != 0).sum(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "mutation_event_fraction",
            "feature_layer": "mechanism_summary",
            "source_modality": "mutation",
            "description": "Fraction of mutation features carrying at least one non-zero event per patient.",
            "unit": "fraction",
            "formula": "(mutation != 0).mean(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "cnv_burden",
            "feature_layer": "mechanism_summary",
            "source_modality": "cnv",
            "description": "Absolute copy-number burden aggregated across all CNV features.",
            "unit": "absolute_copy_number_sum",
            "formula": "abs(cnv).sum(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "cnv_gain_burden",
            "feature_layer": "mechanism_summary",
            "source_modality": "cnv",
            "description": "Positive copy-number burden aggregated across all CNV features.",
            "unit": "copy_number_gain_sum",
            "formula": "clip(cnv, lower=0).sum(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "cnv_loss_burden",
            "feature_layer": "mechanism_summary",
            "source_modality": "cnv",
            "description": "Absolute negative copy-number burden aggregated across all CNV features.",
            "unit": "copy_number_loss_sum",
            "formula": "(-clip(cnv, upper=0)).sum(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "cnv_altered_fraction",
            "feature_layer": "mechanism_summary",
            "source_modality": "cnv",
            "description": "Fraction of CNV features with a non-zero alteration per patient.",
            "unit": "fraction",
            "formula": "(cnv != 0).mean(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "global_methylation",
            "feature_layer": "mechanism_summary",
            "source_modality": "methylation",
            "description": "Mean methylation value across all available probes after row-median imputation.",
            "unit": "mean_beta_like_value",
            "formula": "fill_missing_by_row_median(methylation).mean(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "expression_epithelial_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "expression",
            "description": "Mean z-scored expression across a compact epithelial marker set.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(expression)[epithelial_genes])",
            "matched_feature_count": expression_matches["expression_epithelial_score"],
        },
        {
            "feature_name": "expression_emt_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "expression",
            "description": "Mean z-scored expression across a compact EMT marker set.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(expression)[emt_genes])",
            "matched_feature_count": expression_matches["expression_emt_score"],
        },
        {
            "feature_name": "expression_proliferation_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "expression",
            "description": "Mean z-scored expression across a compact proliferation marker set.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(expression)[proliferation_genes])",
            "matched_feature_count": expression_matches["expression_proliferation_score"],
        },
        {
            "feature_name": "expression_ifng_immune_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "expression",
            "description": "Mean z-scored expression across a compact interferon-gamma / immune marker set.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(expression)[ifng_immune_genes])",
            "matched_feature_count": expression_matches["expression_ifng_immune_score"],
        },
        {
            "feature_name": "mirna_global_abundance",
            "feature_layer": "mechanism_summary",
            "source_modality": "mirna",
            "description": "Mean miRNA abundance after row-median imputation; temporary summary interface before target-level mapping is added.",
            "unit": "mean_expression",
            "formula": "fill_missing_by_row_median(mirna).mean(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "mirna_high_abundance_fraction",
            "feature_layer": "mechanism_summary",
            "source_modality": "mirna",
            "description": "Fraction of miRNAs above their cohort median in each sample; temporary summary interface before target-level mapping is added.",
            "unit": "fraction",
            "formula": "(fill_missing_by_row_median(mirna) > row_median).mean(axis=0)",
            "matched_feature_count": np.nan,
        },
        {
            "feature_name": "rppa_pi3k_akt_mtor_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "rppa",
            "description": "Mean z-scored RPPA signal across PI3K/AKT/mTOR-related antibodies matched by keyword.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(rppa)[pi3k_akt_mtor_keywords])",
            "matched_feature_count": rppa_matches["rppa_pi3k_akt_mtor_score"],
        },
        {
            "feature_name": "rppa_mapk_rtk_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "rppa",
            "description": "Mean z-scored RPPA signal across RTK/MAPK-related antibodies matched by keyword.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(rppa)[mapk_rtk_keywords])",
            "matched_feature_count": rppa_matches["rppa_mapk_rtk_score"],
        },
        {
            "feature_name": "rppa_cell_cycle_dna_damage_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "rppa",
            "description": "Mean z-scored RPPA signal across cell-cycle and DNA-damage-related antibodies matched by keyword.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(rppa)[cell_cycle_dna_damage_keywords])",
            "matched_feature_count": rppa_matches["rppa_cell_cycle_dna_damage_score"],
        },
        {
            "feature_name": "rppa_emt_adhesion_score",
            "feature_layer": "mechanism_summary",
            "source_modality": "rppa",
            "description": "Mean z-scored RPPA signal across EMT/adhesion-related antibodies matched by keyword.",
            "unit": "mean_z_score",
            "formula": "mean(zscore_rows(rppa)[emt_adhesion_keywords])",
            "matched_feature_count": rppa_matches["rppa_emt_adhesion_score"],
        },
    ]
    return pd.DataFrame.from_records(records)


def feature_layer_manifest(
    preprocessed: dict[str, pd.DataFrame],
    embeddings: dict[str, pd.DataFrame],
    mechanism_metadata: pd.DataFrame,
    extra_latent_artifacts: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for modality, frame in preprocessed.items():
        rows.append(
            {
                "feature_layer": "raw_selected",
                "source_modality": modality,
                "feature_name": f"{modality}__selected_matrix",
                "n_features": int(frame.shape[0]),
                "n_samples": int(frame.shape[1]),
                "artifact_file": "",
                "description": "Preprocessed feature matrix retained for modeling in the current analysis.",
            }
        )
    for modality, embedding in embeddings.items():
        rows.append(
            {
                "feature_layer": "latent_embedding",
                "source_modality": modality,
                "feature_name": f"{modality}__embedding",
                "n_features": int(embedding.shape[1]),
                "n_samples": int(embedding.shape[0]),
                "artifact_file": f"feature_cache/{modality}_embedding.tsv",
                "description": "Low-dimensional sample embedding exported for downstream modeling.",
            }
        )
    if extra_latent_artifacts:
        for artifact_name, embedding in extra_latent_artifacts.items():
            rows.append(
                {
                    "feature_layer": "latent_embedding",
                    "source_modality": "integrated",
                    "feature_name": artifact_name,
                    "n_features": int(embedding.shape[1]),
                    "n_samples": int(embedding.shape[0]),
                    "artifact_file": f"feature_cache/{artifact_name}.tsv",
                    "description": "Integrated low-dimensional embedding exported for downstream comparison models.",
                }
            )
    for _, row in mechanism_metadata.iterrows():
        rows.append(
            {
                "feature_layer": row["feature_layer"],
                "source_modality": row["source_modality"],
                "feature_name": row["feature_name"],
                "n_features": 1,
                "n_samples": pd.NA,
                "artifact_file": "mechanism_summary_features.tsv",
                "description": row["description"],
            }
        )
    return pd.DataFrame(rows)
