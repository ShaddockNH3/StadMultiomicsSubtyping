from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from .features import compute_mechanism_summary_features
from .preprocess import fit_modality_reducer


@dataclass(frozen=True)
class SignaturePanelSpec:
    name: str
    size: int
    description: str
    summary_only: bool = False
    min_summary_features: int = 0


SIGNATURE_PANEL_SPECS: tuple[SignaturePanelSpec, ...] = (
    SignaturePanelSpec(
        name="stable_core_8",
        size=8,
        description="Top globally stable features across queues and classifiers.",
    ),
    SignaturePanelSpec(
        name="stable_core_12",
        size=12,
        description="A larger globally stable core panel with modest modality diversity control.",
    ),
    SignaturePanelSpec(
        name="hybrid_core_12",
        size=12,
        description="Balanced panel that forces some mechanism-summary features into the stable latent core.",
        min_summary_features=4,
    ),
    SignaturePanelSpec(
        name="summary_core_10",
        size=10,
        description="Interpretable summary-only panel built from stable mechanism features.",
        summary_only=True,
        min_summary_features=10,
    ),
)


MIRBASE_MATURE_FA_URL = "https://mirbase.org/download/mature.fa"

FROZEN_CNV_SUMMARY_FEATURES: tuple[str, ...] = (
    "summary__cnv_loss_burden",
    "summary__cnv_altered_fraction",
    "summary__cnv_burden",
)
FROZEN_MUTATION_SUMMARY_FEATURES: tuple[str, ...] = (
    "summary__tmb",
    "summary__mutation_event_fraction",
)


FINAL_SIGNATURE_FEATURE_INTERPRETATIONS: dict[str, dict[str, str]] = {
    "raw__cnv__TMEM189": {
        "feature_type": "focal_cnv_marker",
        "biological_axis": "copy-number instability",
        "report_role": "CNV marker supporting the chromosomal-instability axis.",
        "interpretation": (
            "TMEM189 is retained as a focal CNV proxy from the CNV latent component. In this project it is best treated "
            "as a compact marker of regional copy-number structure rather than as a standalone driver claim."
        ),
        "caveat": "CNV-selected marker; biological causality should not be inferred from copy-number association alone.",
    },
    "raw__cnv__GOLM1": {
        "feature_type": "focal_cnv_marker",
        "biological_axis": "copy-number instability",
        "report_role": "CNV marker complementing the global CNV burden feature.",
        "interpretation": (
            "GOLM1 provides a second raw CNV proxy for the CNV component. Together with CNV burden it helps distinguish "
            "subtypes driven by broad chromosomal alteration patterns."
        ),
        "caveat": "Use as a subtype-discrimination marker, not as direct evidence that GOLM1 is the causal alteration.",
    },
    "summary__cnv_burden": {
        "feature_type": "mechanism_summary",
        "biological_axis": "copy-number instability",
        "report_role": "Global CNV burden summary and the kept representative of the collapsed CNV-summary group.",
        "interpretation": (
            "CNV burden summarizes genome-wide copy-number alteration load. The final panel intentionally keeps this "
            "single summary feature after collapsing redundant CNV altered/loss burden summaries."
        ),
        "caveat": "A high burden indicates chromosomal instability at the cohort level, not a specific amplified/deleted locus.",
    },
    "raw__expression__PROC": {
        "feature_type": "expression_marker",
        "biological_axis": "coagulation and vascular/stromal context",
        "report_role": "Expression marker contributing a non-CNV transcriptional axis.",
        "interpretation": (
            "PROC enters as an expression proxy from the stable expression component. In the report it can be framed as "
            "a vascular/coagulation-context signal that adds transcriptomic information beyond CNV and mutation burden."
        ),
        "caveat": "Interpret as a data-driven expression marker; pathway-level claims need enrichment or external validation.",
    },
    "raw__expression__CDX1": {
        "feature_type": "expression_marker",
        "biological_axis": "intestinal differentiation program",
        "report_role": "Expression marker connected to gastric/intestinal epithelial differentiation.",
        "interpretation": (
            "CDX1 is a compact marker for an intestinal differentiation-like transcriptional program. It is useful for "
            "explaining why the final signature is not only measuring genomic instability."
        ),
        "caveat": "Subtype labels should remain TCGA-like/heuristic unless confirmed by external annotation.",
    },
    "raw__mirna__MIMAT0004552": {
        "feature_type": "miRNA_marker",
        "biological_axis": "post-transcriptional regulation",
        "report_role": "miRNA marker adding a regulatory layer to the final panel.",
        "interpretation": (
            "hsa-miR-139-3p is retained as a miRNA-layer proxy. It helps preserve multi-omics character in the compact "
            "signature by representing post-transcriptional regulation."
        ),
        "caveat": "The direction and target mechanism are not inferred here; this is a subtype-discrimination marker.",
    },
    "raw__mirna__MIMAT0015378": {
        "feature_type": "miRNA_marker",
        "biological_axis": "post-transcriptional regulation",
        "report_role": "Second miRNA marker supporting robustness of the regulatory layer.",
        "interpretation": (
            "hsa-miR-3065-3p is the second retained miRNA proxy, indicating that the classifier repeatedly benefits from "
            "a regulatory layer rather than relying purely on DNA-level summaries."
        ),
        "caveat": "Use conservatively as a selected miRNA feature; detailed target validation is outside this code baseline.",
    },
    "raw__mutation__ARID1A": {
        "feature_type": "mutation_marker",
        "biological_axis": "chromatin remodeling and mutation load",
        "report_role": "Raw mutation marker anchoring the mutation/chromatin-remodeling axis.",
        "interpretation": (
            "ARID1A mutation is retained as a raw mutation marker. In the final story it links the compact signature to "
            "chromatin-remodeling disruption and mutation-enriched subtype structure."
        ),
        "caveat": "Do not equate ARID1A alone with MSI status; it should be interpreted together with TMB and subtype enrichment.",
    },
    "summary__tmb": {
        "feature_type": "mechanism_summary",
        "biological_axis": "mutation load",
        "report_role": "Global tumor mutation burden summary.",
        "interpretation": (
            "TMB captures the overall mutation-load axis and is central for explaining MSI-like or hypermutated subtype "
            "patterns in the final panel."
        ),
        "caveat": "This is computed from the available mutation matrix and is not a clinically validated TMB assay.",
    },
    "summary__mutation_event_fraction": {
        "feature_type": "mechanism_summary",
        "biological_axis": "mutation load",
        "report_role": "Mutation event fraction complementing TMB.",
        "interpretation": (
            "Mutation event fraction measures the fraction of retained mutation features altered in each sample. It "
            "stabilizes the mutation-load axis alongside TMB."
        ),
        "caveat": "Partly correlated with TMB; kept because the final selection did not safely remove it.",
    },
}


FINAL_SIGNATURE_INTERPRETATION_COLUMNS: tuple[str, ...] = (
    "selected_final_panel_name",
    "feature_rank_within_panel",
    "source_feature_name",
    "feature_name",
    "feature_annotation",
    "modality_name",
    "feature_layer",
    "feature_type",
    "biological_axis",
    "report_role",
    "interpretation",
    "caveat",
    "n_classifier_queue_runs",
    "n_classifiers",
    "n_queues",
    "mean_abs_coefficient",
    "max_abs_coefficient",
)


def parse_mirbase_mature_fasta(text: str, species_prefix: str = "hsa-") -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith(">"):
            continue
        parts = line[1:].split()
        if len(parts) < 2:
            continue
        mirna_name = parts[0].strip()
        accession = parts[1].strip()
        if species_prefix and not mirna_name.startswith(species_prefix):
            continue
        records.append(
            {
                "accession": accession,
                "mirna_name": mirna_name,
            }
        )
    if not records:
        return pd.DataFrame(columns=["accession", "mirna_name"])
    aliases = pd.DataFrame(records).drop_duplicates(subset=["accession"]).reset_index(drop=True)
    return aliases


def load_mirbase_mature_aliases(cache_path: Path | None = None, species_prefix: str = "hsa-") -> pd.DataFrame:
    text: str | None = None
    if cache_path is not None and cache_path.exists():
        text = cache_path.read_text(encoding="utf-8", errors="ignore")
    if text is None:
        try:
            with urllib.request.urlopen(MIRBASE_MATURE_FA_URL, timeout=30) as response:
                text = response.read().decode("utf-8", errors="ignore")
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
        except Exception:
            if cache_path is not None and cache_path.exists():
                text = cache_path.read_text(encoding="utf-8", errors="ignore")
            else:
                return pd.DataFrame(columns=["accession", "mirna_name"])
    return parse_mirbase_mature_fasta(text, species_prefix=species_prefix)


def parse_signature_feature_name(feature_name: str) -> dict[str, str]:
    if feature_name.startswith("summary__"):
        payload = feature_name.removeprefix("summary__")
        modality_name = payload.split("_", 1)[0]
        return {
            "feature_layer": "mechanism_summary",
            "modality_name": modality_name,
            "feature_token": payload,
            "feature_display_name": payload.replace("_", " "),
        }
    if feature_name.startswith("latent__"):
        parts = feature_name.split("__")
        modality_name = parts[1] if len(parts) > 1 else "unknown"
        component_name = parts[2] if len(parts) > 2 else "component"
        return {
            "feature_layer": "latent_embedding",
            "modality_name": modality_name,
            "feature_token": component_name,
            "feature_display_name": f"{modality_name} {component_name}".replace("_", " "),
        }
    return {
        "feature_layer": "other",
        "modality_name": "other",
        "feature_token": feature_name,
        "feature_display_name": feature_name,
    }


def build_signature_feature_evidence(classifier_runs: list[dict[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    all_run_keys = {(str(run["queue_key"]), str(run["classifier_name"])) for run in classifier_runs}
    n_total_runs = len(all_run_keys)
    n_total_queues = len({str(run["queue_key"]) for run in classifier_runs})
    n_total_classifiers = len({str(run["classifier_name"]) for run in classifier_runs})

    for run in classifier_runs:
        selected = pd.DataFrame(run["cv_result"]["selected_features"]).copy()
        selected = selected.loc[selected["selected_in_final_model"]].copy()
        coefficients = pd.DataFrame(run["cv_result"]["coefficients"]).copy()
        coefficient_lookup = (
            coefficients.groupby("feature_name")["abs_coefficient"].max().to_dict()
            if not coefficients.empty
            else {}
        )
        for row in selected.itertuples(index=False):
            annotation = parse_signature_feature_name(str(row.feature_name))
            records.append(
                {
                    "feature_name": str(row.feature_name),
                    "queue_key": str(run["queue_key"]),
                    "analysis_name": str(run["analysis_name"]),
                    "classifier_name": str(run["classifier_name"]),
                    "selected_fold_count": int(row.selected_fold_count),
                    "selected_fold_fraction": float(row.selected_fold_fraction),
                    "selected_in_final_model": True,
                    "max_abs_coefficient_within_run": float(coefficient_lookup.get(str(row.feature_name), 0.0)),
                    **annotation,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "feature_layer",
                "modality_name",
                "feature_token",
                "feature_display_name",
                "n_runs_selected",
                "n_queues_selected",
                "n_classifiers_selected",
                "mean_selected_fold_fraction",
                "min_selected_fold_fraction",
                "mean_abs_coefficient",
                "max_abs_coefficient",
                "is_cross_queue_stable",
                "is_cross_classifier_stable",
                "is_selected_in_all_runs",
                "stability_score",
            ]
        )

    evidence = pd.DataFrame(records)
    aggregated = (
        evidence.groupby(
            ["feature_name", "feature_layer", "modality_name", "feature_token", "feature_display_name"],
            dropna=False,
        )
        .agg(
            n_runs_selected=("classifier_name", "size"),
            n_queues_selected=("queue_key", "nunique"),
            n_classifiers_selected=("classifier_name", "nunique"),
            mean_selected_fold_fraction=("selected_fold_fraction", "mean"),
            min_selected_fold_fraction=("selected_fold_fraction", "min"),
            mean_abs_coefficient=("max_abs_coefficient_within_run", "mean"),
            max_abs_coefficient=("max_abs_coefficient_within_run", "max"),
        )
        .reset_index()
    )
    aggregated["is_cross_queue_stable"] = aggregated["n_queues_selected"] == n_total_queues
    aggregated["is_cross_classifier_stable"] = aggregated["n_classifiers_selected"] == n_total_classifiers
    aggregated["is_selected_in_all_runs"] = aggregated["n_runs_selected"] == n_total_runs
    aggregated["stability_score"] = (
        aggregated["n_runs_selected"] * 10.0
        + aggregated["mean_selected_fold_fraction"] * 5.0
        + aggregated["n_queues_selected"] * 2.0
        + aggregated["n_classifiers_selected"] * 2.0
        + np.log1p(aggregated["mean_abs_coefficient"].clip(lower=0.0))
    )
    aggregated = aggregated.sort_values(
        [
            "is_selected_in_all_runs",
            "is_cross_queue_stable",
            "is_cross_classifier_stable",
            "stability_score",
            "mean_abs_coefficient",
            "feature_name",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    aggregated["global_rank"] = np.arange(1, len(aggregated) + 1)
    return aggregated


def _panel_modality_cap(size: int) -> int:
    return max(2, math.ceil(size * 0.4))


def _select_ranked_features(
    ordered: pd.DataFrame,
    size: int,
    modality_cap: int | None = None,
    exclude_features: set[str] | None = None,
) -> list[str]:
    exclude_features = exclude_features or set()
    selected: list[str] = []
    modality_counts: Counter[str] = Counter()
    for row in ordered.itertuples(index=False):
        feature_name = str(row.feature_name)
        if feature_name in exclude_features:
            continue
        modality_name = str(row.modality_name)
        if modality_cap is not None and modality_counts[modality_name] >= modality_cap:
            continue
        selected.append(feature_name)
        modality_counts[modality_name] += 1
        if len(selected) >= size:
            break
    if len(selected) >= size:
        return selected[:size]
    for row in ordered.itertuples(index=False):
        feature_name = str(row.feature_name)
        if feature_name in exclude_features or feature_name in selected:
            continue
        selected.append(feature_name)
        if len(selected) >= size:
            break
    return selected[:size]


def build_signature_panels(
    feature_evidence: pd.DataFrame,
    panel_specs: tuple[SignaturePanelSpec, ...] = SIGNATURE_PANEL_SPECS,
) -> pd.DataFrame:
    if feature_evidence.empty:
        return pd.DataFrame(
            columns=[
                "panel_name",
                "panel_description",
                "panel_target_size",
                "panel_feature_count",
                "feature_rank_within_panel",
                "feature_name",
                "feature_layer",
                "modality_name",
                "feature_display_name",
                "stability_score",
                "is_selected_in_all_runs",
            ]
        )

    panels: list[dict[str, object]] = []
    ordered = feature_evidence.copy()
    summary_pool = ordered[ordered["feature_layer"] == "mechanism_summary"].reset_index(drop=True)

    for spec in panel_specs:
        pool = ordered.copy()
        if spec.summary_only:
            pool = summary_pool.copy()
        selected_features: list[str] = []
        if spec.min_summary_features > 0 and not summary_pool.empty:
            selected_features.extend(
                _select_ranked_features(
                    summary_pool,
                    size=min(spec.min_summary_features, spec.size),
                    modality_cap=_panel_modality_cap(spec.min_summary_features),
                )
            )
        remaining = max(0, spec.size - len(selected_features))
        if remaining > 0:
            selected_features.extend(
                _select_ranked_features(
                    pool,
                    size=remaining,
                    modality_cap=None if spec.summary_only else _panel_modality_cap(spec.size),
                    exclude_features=set(selected_features),
                )
            )
        panel_rows = ordered.set_index("feature_name").loc[selected_features].reset_index()
        for rank_idx, row in enumerate(panel_rows.itertuples(index=False), start=1):
            panels.append(
                {
                    "panel_name": spec.name,
                    "panel_description": spec.description,
                    "panel_target_size": spec.size,
                    "panel_feature_count": int(len(panel_rows)),
                    "feature_rank_within_panel": rank_idx,
                    "feature_name": str(row.feature_name),
                    "feature_layer": str(row.feature_layer),
                    "modality_name": str(row.modality_name),
                    "feature_display_name": str(row.feature_display_name),
                    "stability_score": float(row.stability_score),
                    "is_selected_in_all_runs": bool(row.is_selected_in_all_runs),
                }
            )
    return pd.DataFrame(panels)


def signature_panel_summary_frame(
    panel_summary_rows: list[pd.DataFrame],
    full_classifier_summary: pd.DataFrame,
) -> pd.DataFrame:
    if not panel_summary_rows:
        return pd.DataFrame()
    panel_summary = pd.concat(panel_summary_rows, ignore_index=True)
    baseline = full_classifier_summary[
        ["queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc", "n_features"]
    ].rename(
        columns={
            "macro_f1": "baseline_macro_f1",
            "balanced_accuracy": "baseline_balanced_accuracy",
            "accuracy": "baseline_accuracy",
            "macro_ovr_auroc": "baseline_macro_ovr_auroc",
            "n_features": "baseline_n_features",
        }
    )
    summary = panel_summary.merge(baseline, on=["queue_key", "classifier_name"], how="left")
    summary["delta_macro_f1"] = summary["macro_f1"] - summary["baseline_macro_f1"]
    summary["delta_balanced_accuracy"] = summary["balanced_accuracy"] - summary["baseline_balanced_accuracy"]
    summary["delta_accuracy"] = summary["accuracy"] - summary["baseline_accuracy"]
    summary["delta_macro_ovr_auroc"] = summary["macro_ovr_auroc"] - summary["baseline_macro_ovr_auroc"]
    return summary.sort_values(["panel_name", "queue_key", "classifier_name"]).reset_index(drop=True)


def signature_shortlist_markdown(
    panel_features: pd.DataFrame,
    panel_summary: pd.DataFrame,
    title: str = "Signature Shortlist",
) -> str:
    lines = [f"# {title}", "", "## Panels"]
    for panel_name, panel_df in panel_features.groupby("panel_name", sort=False):
        lines.append(f"### {panel_name}")
        lines.append(str(panel_df["panel_description"].iloc[0]))
        lines.append("")
        feature_bits = [
            f"{row.feature_name} ({row.modality_name}, {row.feature_layer})"
            for row in panel_df.itertuples(index=False)
        ]
        lines.append("- " + "\n- ".join(feature_bits))
        lines.append("")
    lines.append("## Performance")
    for panel_name, panel_df in panel_summary.groupby("panel_name", sort=False):
        lines.append(f"### {panel_name}")
        for row in panel_df.itertuples(index=False):
            lines.append(
                f"- {row.queue_key} / {row.classifier_name}: "
                f"macro-F1 {row.macro_f1:.3f} "
                f"(delta {row.delta_macro_f1:+.3f}), "
                f"balanced accuracy {row.balanced_accuracy:.3f} "
                f"(delta {row.delta_balanced_accuracy:+.3f}), "
                f"AUROC {row.macro_ovr_auroc:.3f} "
                f"(delta {row.delta_macro_ovr_auroc:+.3f})"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_signature_readme(
    out_dir: Path,
    panel_features: pd.DataFrame,
    panel_summary: pd.DataFrame,
    raw_panel_features: pd.DataFrame | None = None,
    raw_panel_summary: pd.DataFrame | None = None,
    refined_panel_features: pd.DataFrame | None = None,
    refined_panel_summary: pd.DataFrame | None = None,
    freeze_candidates: pd.DataFrame | None = None,
    freeze_selection: pd.DataFrame | None = None,
    frozen_panel_features: pd.DataFrame | None = None,
    reduced_panel_features: pd.DataFrame | None = None,
    reduced_panel_summary: pd.DataFrame | None = None,
    reduced_panel_selection: pd.DataFrame | None = None,
    final_panel_features: pd.DataFrame | None = None,
    final_panel_summary: pd.DataFrame | None = None,
    final_panel_selection: pd.DataFrame | None = None,
    selected_final_panel_features: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Signature Outputs",
        "",
        "本目录保存基于 `Elastic Net + SVM` 稳定特征证据收缩得到的候选 signature，以及由 latent component 反向映射得到的 raw-marker 版本。",
        "",
        "关键文件：",
        "",
        "- `feature_evidence.tsv`: 跨队列、跨分类器的特征稳定性证据表",
        "- `panel_features.tsv`: 每个候选面板包含的特征列表",
        "- `signature_shortlist.md`: 候选面板与性能摘要",
        "- `raw_marker_mapping.tsv`: latent component 到候选 raw marker 的映射表",
        "- `mirna_aliases.tsv`: miRNA accession 到成熟 miRNA 名称的注释表",
        "- `proxy_candidate_scores.tsv`: 候选 raw marker 的跨队列 proxy 打分表",
        "- `raw_panel_features.tsv`: raw-marker 面板列表",
        "- `raw_panel_summary.tsv`: raw-marker 面板性能汇总",
        "- `raw_signature_shortlist.md`: raw-marker 面板与性能摘要",
        "- `refined_panel_features.tsv`: refined raw-marker 面板列表",
        "- `refined_panel_summary.tsv`: refined raw-marker 面板性能汇总",
        "- `refined_signature_shortlist.md`: refined raw-marker 面板与性能摘要",
        "- `freeze_candidates.tsv`: refined 面板去重后的冻结候选总表",
        "- `freeze_selection.tsv`: 当前冻结规则选中的 refined 面板",
        "- `frozen_panel_features.tsv`: 当前冻结 refined 面板的特征列表",
        "- `frozen_signature_shortlist.md`: 当前冻结 refined 面板的简要说明",
        "- `reduced_panel_features.tsv`: 基于冻结 refined 面板继续做轻量去冗余得到的候选面板",
        "- `reduced_panel_summary.tsv`: reduced 面板性能汇总",
        "- `reduced_panel_selection.tsv`: reduced 面板自动选型结果",
        "- `reduced_signature_shortlist.md`: reduced 面板与性能摘要",
        "- `final_panel_features.tsv`: 围绕当前 reduced 主候选继续做最终压缩得到的 final-candidate 面板",
        "- `final_panel_summary.tsv`: final-candidate 面板性能汇总",
        "- `final_panel_selection.tsv`: 最终“继续压缩还是停在 reduced panel”决策表",
        "- `final_recommended_panel_features.tsv`: 当前正式推荐的最终面板特征列表",
        "- `final_signature_shortlist.md`: 最终面板决策摘要",
        "- `final_classifier_evidence.tsv`: 最终面板特征在分类器输出中的系数/重要性证据汇总",
        "- `final_signature_interpretation.tsv`: 最终面板逐特征生物学解释表",
        "- `final_signature_interpretation.md`: 最终面板可直接写入报告的解释摘要",
        "",
        "当前面板：",
        "",
    ]
    for panel_name, panel_df in panel_features.groupby("panel_name", sort=False):
        lines.append(
            f"- `{panel_name}`: {panel_df['panel_description'].iloc[0]} "
            f"(实际特征数 {int(panel_df['panel_feature_count'].iloc[0])})"
        )
    if not panel_summary.empty:
        lines.extend(["", "性能概况：", ""])
        best_rows = (
            panel_summary.sort_values(["panel_name", "delta_macro_f1"], ascending=[True, False])
            .groupby("panel_name", sort=False)
            .head(1)
        )
        for row in best_rows.itertuples(index=False):
            lines.append(
                f"- `{row.panel_name}` 最佳结果出现在 `{row.queue_key} / {row.classifier_name}`："
                f"`macro-F1 = {row.macro_f1:.3f}`，"
                f"`delta_macro_f1 = {row.delta_macro_f1:+.3f}`"
            )
    if raw_panel_features is not None and not raw_panel_features.empty:
        lines.extend(["", "当前 raw-marker 面板：", ""])
        for panel_name, panel_df in raw_panel_features.groupby("raw_panel_name", sort=False):
            source_panel_name = str(panel_df["source_panel_name"].iloc[0])
            lines.append(
                f"- `{panel_name}`: 来自 `{source_panel_name}` 的 raw-marker 翻译版 "
                f"(实际特征数 {int(panel_df['raw_panel_feature_count'].iloc[0])})"
            )
    if raw_panel_summary is not None and not raw_panel_summary.empty:
        lines.extend(["", "raw-marker 面板性能概况：", ""])
        best_rows = (
            raw_panel_summary.sort_values(
                ["raw_panel_name", "delta_macro_f1_vs_source_panel", "delta_macro_f1"],
                ascending=[True, False, False],
            )
            .groupby("raw_panel_name", sort=False)
            .head(1)
        )
        for row in best_rows.itertuples(index=False):
            lines.append(
                f"- `{row.raw_panel_name}` 最佳结果出现在 `{row.queue_key} / {row.classifier_name}`："
                f"`macro-F1 = {row.macro_f1:.3f}`，"
                f"相对全特征 `delta_macro_f1 = {row.delta_macro_f1:+.3f}`，"
                f"相对源 latent 面板 `delta_macro_f1_vs_source_panel = {row.delta_macro_f1_vs_source_panel:+.3f}`"
            )
    if refined_panel_features is not None and not refined_panel_features.empty:
        lines.extend(["", "当前 refined raw-marker 面板：", ""])
        for panel_name, panel_df in refined_panel_features.groupby("raw_panel_name", sort=False):
            source_panel_name = str(panel_df["source_panel_name"].iloc[0])
            lines.append(
                f"- `{panel_name}`: 基于 proxy 打分优化、来自 `{source_panel_name}` 的 refined raw-marker 版本 "
                f"(实际特征数 {int(panel_df['raw_panel_feature_count'].iloc[0])})"
            )
    if refined_panel_summary is not None and not refined_panel_summary.empty:
        lines.extend(["", "refined raw-marker 面板性能概况：", ""])
        best_rows = (
            refined_panel_summary.sort_values(
                ["raw_panel_name", "delta_macro_f1_vs_source_panel", "delta_macro_f1"],
                ascending=[True, False, False],
            )
            .groupby("raw_panel_name", sort=False)
            .head(1)
        )
        for row in best_rows.itertuples(index=False):
            lines.append(
                f"- `{row.raw_panel_name}` 最佳结果出现在 `{row.queue_key} / {row.classifier_name}`："
                f"`macro-F1 = {row.macro_f1:.3f}`，"
                f"相对全特征 `delta_macro_f1 = {row.delta_macro_f1:+.3f}`，"
                f"相对源 latent 面板 `delta_macro_f1_vs_source_panel = {row.delta_macro_f1_vs_source_panel:+.3f}`"
            )
    if freeze_candidates is not None and not freeze_candidates.empty:
        lines.extend(["", "refined 面板冻结候选：", ""])
        for row in freeze_candidates.itertuples(index=False):
            duplicate_note = (
                f"，折叠了 {int(row.n_equivalent_panels)} 个同内容面板"
                if int(row.n_equivalent_panels) > 1
                else ""
            )
            lines.append(
                f"- `{row.representative_panel_name}`: validation `SVM` "
                f"`macro-F1 = {row.validation_svm_macro_f1:.3f}`，"
                f"`delta_vs_source = {row.validation_svm_delta_macro_f1_vs_source_panel:+.3f}`，"
                f"primary mean macro-F1 `{row.primary_mean_macro_f1:.3f}`{duplicate_note}"
            )
    if freeze_selection is not None and not freeze_selection.empty:
        selected_rows = freeze_selection.loc[freeze_selection["selected"]]
        if not selected_rows.empty:
            row = selected_rows.iloc[0]
            lines.extend(
                [
                    "",
                    "当前冻结 refined 面板：",
                    "",
                    f"- 选中 `{row.representative_panel_name}`，来自 `{row.source_panel_names}`",
                    f"- 原因：验证队列 `SVM` 上最接近源 latent 面板 "
                    f"(`delta_vs_source = {row.validation_svm_delta_macro_f1_vs_source_panel:+.3f}`)，"
                    f"同时保持验证队列 `macro-F1 = {row.validation_svm_macro_f1:.3f}`",
                ]
            )
    if frozen_panel_features is not None and not frozen_panel_features.empty:
        lines.extend(["", "冻结面板特征：", ""])
        for row in frozen_panel_features.sort_values("feature_rank_within_panel").itertuples(index=False):
            lines.append(
                f"- `{row.mapped_feature_name}` <- `{row.source_feature_name}` ({row.mapped_feature_annotation})"
            )
    if reduced_panel_features is not None and not reduced_panel_features.empty:
        lines.extend(["", "当前 reduced 面板：", ""])
        for panel_name, panel_df in reduced_panel_features.groupby("reduced_panel_name", sort=False):
            lines.append(
                f"- `{panel_name}`: {panel_df['reduction_description'].iloc[0]} "
                f"(实际特征数 {int(panel_df['raw_panel_feature_count'].iloc[0])}，"
                f"去掉 `{panel_df['dropped_feature_names'].iloc[0]}`)"
            )
    if reduced_panel_summary is not None and not reduced_panel_summary.empty:
        lines.extend(["", "reduced 面板性能概况：", ""])
        best_rows = (
            reduced_panel_summary.sort_values(
                ["reduced_panel_name", "delta_macro_f1_vs_parent_frozen", "delta_macro_f1_vs_source_panel"],
                ascending=[True, False, False],
            )
            .groupby("reduced_panel_name", sort=False)
            .head(1)
        )
        for row in best_rows.itertuples(index=False):
            lines.append(
                f"- `{row.reduced_panel_name}` 最佳结果出现在 `{row.queue_key} / {row.classifier_name}`："
                f"`macro-F1 = {row.macro_f1:.3f}`，"
                f"相对冻结面板 `delta_macro_f1_vs_parent_frozen = {row.delta_macro_f1_vs_parent_frozen:+.3f}`，"
                f"相对源 latent 面板 `delta_macro_f1_vs_source_panel = {row.delta_macro_f1_vs_source_panel:+.3f}`"
            )
    if reduced_panel_selection is not None and not reduced_panel_selection.empty:
        chosen = reduced_panel_selection.loc[reduced_panel_selection["selected"]]
        if not chosen.empty:
            row = chosen.iloc[0]
            lines.extend(
                [
                    "",
                    "当前推荐 reduced 面板：",
                    "",
                    f"- 选中 `{row.reduced_panel_name}`",
                    f"- 验证队列平均 `macro-F1 = {row.validation_mean_macro_f1:.3f}`",
                    f"- 相对冻结面板验证均值提升 `delta = {row.validation_mean_delta_vs_parent_frozen:+.3f}`",
                    f"- 验证队列 `XGBoost` 相对冻结面板提升 `delta = {row.validation_xgboost_delta_vs_parent_frozen:+.3f}`",
                    f"- 主发现队列平均 `macro-F1 = {row.primary_mean_macro_f1:.3f}`",
                ]
            )
    if final_panel_features is not None and not final_panel_features.empty:
        lines.extend(["", "当前 final-candidate 面板：", ""])
        for panel_name, panel_df in final_panel_features.groupby("final_panel_name", sort=False):
            lines.append(
                f"- `{panel_name}`: {panel_df['compression_description'].iloc[0]} "
                f"(实际特征数 {int(panel_df['raw_panel_feature_count'].iloc[0])}，"
                f"去掉 `{panel_df['dropped_feature_names'].iloc[0]}`)"
            )
    if final_panel_summary is not None and not final_panel_summary.empty:
        lines.extend(["", "final-candidate 面板性能概况：", ""])
        best_rows = (
            final_panel_summary.sort_values(
                ["final_panel_name", "delta_macro_f1_vs_parent_reduced", "delta_macro_f1_vs_source_panel"],
                ascending=[True, False, False],
            )
            .groupby("final_panel_name", sort=False)
            .head(1)
        )
        for row in best_rows.itertuples(index=False):
            lines.append(
                f"- `{row.final_panel_name}` 最佳结果出现在 `{row.queue_key} / {row.classifier_name}`："
                f"`macro-F1 = {row.macro_f1:.3f}`，"
                f"相对 reduced 父面板 `delta_macro_f1_vs_parent_reduced = {row.delta_macro_f1_vs_parent_reduced:+.3f}`，"
                f"相对源 latent 面板 `delta_macro_f1_vs_source_panel = {row.delta_macro_f1_vs_source_panel:+.3f}`"
            )
    if final_panel_selection is not None and not final_panel_selection.empty:
        chosen = final_panel_selection.loc[final_panel_selection["selected"]]
        if not chosen.empty:
            row = chosen.iloc[0]
            lines.extend(
                [
                    "",
                    "当前正式推荐最终面板：",
                    "",
                    f"- 选中 `{row.final_panel_name}`",
                    f"- 策略：`{row.compression_strategy}`",
                    f"- 是否安全继续压缩：`{bool(row.is_safe_shrink)}`",
                    f"- 验证队列平均 `macro-F1 = {row.validation_mean_macro_f1:.3f}`",
                    f"- 相对 reduced 父面板验证均值 `delta = {row.validation_mean_delta_vs_parent_reduced:+.3f}`",
                    f"- 相对 reduced 父面板验证队列 `XGBoost delta = {row.validation_xgboost_delta_vs_parent_reduced:+.3f}`",
                    f"- 主发现队列平均 `macro-F1 = {row.primary_mean_macro_f1:.3f}`",
                ]
            )
    if selected_final_panel_features is not None and not selected_final_panel_features.empty:
        lines.extend(["", "正式推荐最终面板特征：", ""])
        for row in selected_final_panel_features.sort_values("feature_rank_within_panel").itertuples(index=False):
            lines.append(
                f"- `{row.mapped_feature_name}` <- `{row.source_feature_name}` ({row.mapped_feature_annotation})"
            )
        lines.extend(
            [
                "",
                "最终面板解释口径：",
                "",
                "- 当前 `stable_reduced_10_cnv_burden_only` 应作为正式 compact signature，而不是继续机械删特征的中间候选",
                "- 报告中建议按 `CNV instability / mutation load / intestinal differentiation / vascular-stromal expression / miRNA regulation` 五类轴解释",
                "- 逐特征解释见 `final_signature_interpretation.tsv` 与 `final_signature_interpretation.md`",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def collect_final_signature_classifier_evidence(
    signature_dir: Path,
    panel_name: str,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    columns = [
        "feature_name",
        "n_classifier_queue_runs",
        "n_classifiers",
        "n_queues",
        "mean_abs_coefficient",
        "max_abs_coefficient",
    ]
    if not panel_name:
        return pd.DataFrame(columns=columns)

    panel_dir = signature_dir / panel_name
    if not panel_dir.exists():
        return pd.DataFrame(columns=columns)

    feature_filter = set(feature_names or [])
    records: list[dict[str, object]] = []
    for coefficient_path in sorted(panel_dir.glob("*/*/coefficients.tsv")):
        classifier_name = coefficient_path.parent.parent.name
        queue_key = coefficient_path.parent.name
        try:
            coefficients = pd.read_csv(coefficient_path, sep="\t")
        except Exception:
            continue
        if "feature_name" not in coefficients.columns or "abs_coefficient" not in coefficients.columns:
            continue
        if feature_filter:
            coefficients = coefficients.loc[coefficients["feature_name"].isin(feature_filter)].copy()
        if coefficients.empty:
            continue
        within_run = (
            coefficients.groupby("feature_name", dropna=False)["abs_coefficient"]
            .max()
            .reset_index()
        )
        for row in within_run.itertuples(index=False):
            records.append(
                {
                    "feature_name": str(row.feature_name),
                    "classifier_name": classifier_name,
                    "queue_key": queue_key,
                    "abs_coefficient": float(row.abs_coefficient),
                }
            )

    if not records:
        return pd.DataFrame(columns=columns)

    evidence = pd.DataFrame(records)
    aggregated = (
        evidence.groupby("feature_name", dropna=False)
        .agg(
            n_classifier_queue_runs=("abs_coefficient", "size"),
            n_classifiers=("classifier_name", "nunique"),
            n_queues=("queue_key", "nunique"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
        .reset_index()
    )
    return aggregated.sort_values(["max_abs_coefficient", "feature_name"], ascending=[False, True]).reset_index(drop=True)


def _final_signature_row_value(row: pd.Series, candidates: tuple[str, ...], default: str = "") -> str:
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                return value
    return default


def build_final_signature_interpretation(
    selected_final_panel_features: pd.DataFrame,
    classifier_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if selected_final_panel_features.empty:
        return pd.DataFrame(columns=list(FINAL_SIGNATURE_INTERPRETATION_COLUMNS))

    panel = selected_final_panel_features.copy()
    rank_column = "feature_rank_within_panel" if "feature_rank_within_panel" in panel.columns else None
    if rank_column:
        panel = panel.sort_values(rank_column)

    evidence_lookup = (
        classifier_evidence.set_index("feature_name").to_dict(orient="index")
        if classifier_evidence is not None and not classifier_evidence.empty and "feature_name" in classifier_evidence.columns
        else {}
    )

    records: list[dict[str, object]] = []
    for idx, row in panel.reset_index(drop=True).iterrows():
        feature_name = _final_signature_row_value(row, ("mapped_feature_name", "feature_name"))
        annotation = _final_signature_row_value(
            row,
            ("mapped_feature_annotation", "feature_display_name", "feature_annotation"),
            default=feature_name,
        )
        modality_name = _final_signature_row_value(row, ("modality_name",), default=parse_signature_feature_name(feature_name)["modality_name"])
        feature_layer = _final_signature_row_value(row, ("mapped_feature_layer", "feature_layer"), default=parse_signature_feature_name(feature_name)["feature_layer"])
        interpretation = FINAL_SIGNATURE_FEATURE_INTERPRETATIONS.get(
            feature_name,
            {
                "feature_type": feature_layer,
                "biological_axis": modality_name,
                "report_role": "Data-driven marker retained in the final compact signature.",
                "interpretation": (
                    "This feature was retained by the final selection rule and should be interpreted as a subtype-discrimination "
                    "marker from its source modality."
                ),
                "caveat": "No feature-specific external biological interpretation has been hard-coded for this marker.",
            },
        )
        evidence = evidence_lookup.get(feature_name, {})
        records.append(
            {
                "selected_final_panel_name": _final_signature_row_value(
                    row,
                    ("selected_final_panel_name", "final_panel_name", "reduced_panel_name"),
                    default="",
                ),
                "feature_rank_within_panel": int(row[rank_column]) if rank_column else int(idx + 1),
                "source_feature_name": _final_signature_row_value(row, ("source_feature_name",), default=feature_name),
                "feature_name": feature_name,
                "feature_annotation": annotation,
                "modality_name": modality_name,
                "feature_layer": feature_layer,
                "feature_type": interpretation["feature_type"],
                "biological_axis": interpretation["biological_axis"],
                "report_role": interpretation["report_role"],
                "interpretation": interpretation["interpretation"],
                "caveat": interpretation["caveat"],
                "n_classifier_queue_runs": int(evidence.get("n_classifier_queue_runs", 0)),
                "n_classifiers": int(evidence.get("n_classifiers", 0)),
                "n_queues": int(evidence.get("n_queues", 0)),
                "mean_abs_coefficient": float(evidence.get("mean_abs_coefficient", np.nan)),
                "max_abs_coefficient": float(evidence.get("max_abs_coefficient", np.nan)),
            }
        )

    return pd.DataFrame(records, columns=list(FINAL_SIGNATURE_INTERPRETATION_COLUMNS))


def final_signature_interpretation_markdown(
    interpretation_table: pd.DataFrame,
    final_panel_selection: pd.DataFrame | None = None,
) -> str:
    if interpretation_table.empty:
        return "# Final Signature Interpretation\n\nNo final signature interpretation is available.\n"

    panel_name = str(interpretation_table["selected_final_panel_name"].dropna().iloc[0])
    if not panel_name:
        panel_name = "final signature"
    lines = [
        "# Final Signature Biological Interpretation",
        "",
        f"推荐最终面板：`{panel_name}`。",
        "",
        "该 10 特征面板不是继续压缩的临时结果，而是当前代码版实验的正式 compact signature：它保留 CNV、突变负荷、表达和 miRNA 四类可解释轴，同时避免把三个高度相关的 CNV summary 同时放入最终面板。",
        "",
    ]

    if final_panel_selection is not None and not final_panel_selection.empty and "selected" in final_panel_selection.columns:
        selected = final_panel_selection.loc[final_panel_selection["selected"]]
        if not selected.empty:
            row = selected.iloc[0]
            lines.extend(
                [
                    "## Selection Evidence",
                    "",
                    f"- 验证队列平均 macro-F1：`{float(row.get('validation_mean_macro_f1', np.nan)):.3f}`",
                    f"- 相对 reduced 父面板验证均值变化：`{float(row.get('validation_mean_delta_vs_parent_reduced', np.nan)):+.3f}`",
                    f"- 是否安全继续压缩：`{bool(row.get('is_safe_shrink', False))}`",
                    "",
                ]
            )

    axis_summary = (
        interpretation_table.groupby("biological_axis", sort=False)
        .agg(n_features=("feature_name", "size"), features=("feature_annotation", lambda x: ", ".join(map(str, x))))
        .reset_index()
    )
    lines.extend(["## Biological Axes", ""])
    for row in axis_summary.itertuples(index=False):
        lines.append(f"- `{row.biological_axis}` ({int(row.n_features)}): {row.features}")

    lines.extend(["", "## Feature-Level Notes", ""])
    for row in interpretation_table.sort_values("feature_rank_within_panel").itertuples(index=False):
        evidence_note = ""
        if int(row.n_classifier_queue_runs) > 0 and not pd.isna(row.max_abs_coefficient):
            evidence_note = (
                f" Classifier evidence: max abs coefficient/importances `{float(row.max_abs_coefficient):.3f}` "
                f"across {int(row.n_classifier_queue_runs)} classifier-queue runs."
            )
        lines.append(
            f"- `{row.feature_name}` ({row.feature_annotation}; {row.biological_axis}): "
            f"{row.report_role} {row.interpretation}{evidence_note}"
        )

    lines.extend(
        [
            "",
            "## Reporting Caution",
            "",
            "这些解释用于支撑实验报告中的机制叙述。除 ARID1A、TMB、CNV burden 等相对直观的特征外，raw marker 仍应被写作“分型判别标志物”或“proxy marker”，不要写成已经验证的致病驱动或临床检测方案。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _component_index_from_feature(feature_name: str) -> int:
    token = parse_signature_feature_name(feature_name)["feature_token"]
    if token.startswith("component_"):
        return max(0, int(token.split("_")[-1]) - 1)
    raise ValueError(f"Cannot parse component index from {feature_name}")


def _raw_feature_annotation(
    modality_name: str,
    feature_id: str,
    probe_map: pd.DataFrame | None,
    mirna_alias_lookup: dict[str, str] | None = None,
) -> str:
    if modality_name == "mirna" and mirna_alias_lookup is not None:
        alias = mirna_alias_lookup.get(feature_id)
        if alias:
            return f"{alias} ({feature_id})"
    if modality_name == "methylation" and probe_map is not None and not probe_map.empty:
        lookup = probe_map.set_index("feature_id") if "feature_id" in probe_map.columns else probe_map.set_index("#id")
        if feature_id in lookup.index:
            gene = str(lookup.loc[feature_id].get("gene", "")).strip()
            if gene and gene != ".":
                return gene
    return feature_id


def build_latent_component_marker_table(
    panel_features: pd.DataFrame,
    preprocessed: dict[str, pd.DataFrame],
    probe_map: pd.DataFrame,
    mirna_aliases: pd.DataFrame | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    latent_panel_features = panel_features.loc[panel_features["feature_layer"] == "latent_embedding", ["panel_name", "feature_name"]].drop_duplicates()
    if latent_panel_features.empty:
        return pd.DataFrame(
            columns=[
                "panel_name",
                "source_feature_name",
                "modality_name",
                "component_index",
                "loading_rank",
                "raw_feature_id",
                "raw_feature_annotation",
                "loading",
                "abs_loading",
            ]
        )

    records: list[dict[str, object]] = []
    mirna_alias_lookup = (
        mirna_aliases.set_index("accession")["mirna_name"].to_dict()
        if mirna_aliases is not None and not mirna_aliases.empty
        else {}
    )
    for modality_name, modality_group in latent_panel_features.groupby(latent_panel_features["feature_name"].map(lambda x: parse_signature_feature_name(x)["modality_name"])):
        if modality_name not in preprocessed:
            continue
        frame = preprocessed[modality_name]
        model, _ = fit_modality_reducer(modality_name, frame)
        components = np.asarray(model.components_, dtype=float)
        for row in modality_group.itertuples(index=False):
            feature_name = str(row.feature_name)
            component_index = _component_index_from_feature(feature_name)
            if component_index >= components.shape[0]:
                continue
            loadings = components[component_index]
            ranked_indices = np.argsort(np.abs(loadings))[::-1][:top_n]
            for rank_idx, feature_idx in enumerate(ranked_indices, start=1):
                raw_feature_id = str(frame.index[feature_idx])
                loading = float(loadings[feature_idx])
                records.append(
                    {
                        "panel_name": str(row.panel_name),
                        "source_feature_name": feature_name,
                        "modality_name": modality_name,
                        "component_index": int(component_index + 1),
                        "loading_rank": rank_idx,
                        "raw_feature_id": raw_feature_id,
                        "raw_feature_annotation": _raw_feature_annotation(modality_name, raw_feature_id, probe_map, mirna_alias_lookup),
                        "loading": loading,
                        "abs_loading": float(abs(loading)),
                    }
                )
    return pd.DataFrame(records).sort_values(
        ["panel_name", "source_feature_name", "loading_rank", "raw_feature_id"]
    ).reset_index(drop=True)


def build_raw_marker_panels(
    panel_features: pd.DataFrame,
    component_marker_table: pd.DataFrame,
) -> pd.DataFrame:
    if panel_features.empty:
        return pd.DataFrame()
    latent_panels = [panel_name for panel_name, group in panel_features.groupby("panel_name", sort=False) if (group["feature_layer"] == "latent_embedding").any()]
    rows: list[dict[str, object]] = []
    component_lookup = component_marker_table.groupby(["panel_name", "source_feature_name"], sort=False)

    for source_panel_name in latent_panels:
        source_panel = panel_features.loc[panel_features["panel_name"] == source_panel_name].copy()
        raw_panel_name = source_panel_name.replace("core", "raw")
        used_raw_features: set[tuple[str, str]] = set()
        for row in source_panel.itertuples(index=False):
            if str(row.feature_layer) == "mechanism_summary":
                rows.append(
                    {
                        "raw_panel_name": raw_panel_name,
                        "source_panel_name": source_panel_name,
                        "raw_panel_description": f"Raw-marker translation of {source_panel_name}.",
                        "source_feature_name": str(row.feature_name),
                        "mapped_feature_name": str(row.feature_name),
                        "mapped_feature_layer": "mechanism_summary",
                        "modality_name": str(row.modality_name),
                        "mapped_feature_annotation": str(row.feature_display_name),
                        "mapping_source": "summary_preserved",
                    }
                )
                continue
            candidates = component_lookup.get_group((source_panel_name, str(row.feature_name))).copy()
            chosen = None
            for candidate in candidates.itertuples(index=False):
                raw_key = (str(candidate.modality_name), str(candidate.raw_feature_id))
                if raw_key not in used_raw_features:
                    chosen = candidate
                    used_raw_features.add(raw_key)
                    break
            if chosen is None:
                chosen = candidates.iloc[0]
                used_raw_features.add((str(chosen["modality_name"]), str(chosen["raw_feature_id"])))
                chosen = type("Chosen", (), chosen.to_dict())
            rows.append(
                {
                    "raw_panel_name": raw_panel_name,
                    "source_panel_name": source_panel_name,
                    "raw_panel_description": f"Raw-marker translation of {source_panel_name}.",
                    "source_feature_name": str(row.feature_name),
                    "mapped_feature_name": f"raw__{chosen.modality_name}__{chosen.raw_feature_id}",
                    "mapped_feature_layer": "raw_selected",
                    "modality_name": str(chosen.modality_name),
                    "mapped_feature_annotation": str(chosen.raw_feature_annotation),
                    "mapping_source": str(chosen.raw_feature_id),
                }
            )
    raw_panels = pd.DataFrame(rows)
    if raw_panels.empty:
        return raw_panels
    raw_panels["raw_panel_feature_count"] = raw_panels.groupby("raw_panel_name")["mapped_feature_name"].transform("nunique")
    raw_panels["feature_rank_within_panel"] = raw_panels.groupby("raw_panel_name").cumcount() + 1
    return raw_panels


def build_raw_proxy_candidate_scores(
    component_marker_table: pd.DataFrame,
    queue_prepared: dict[str, object],
    primary_queue_key: str = "primary_discovery",
) -> pd.DataFrame:
    if component_marker_table.empty:
        return pd.DataFrame(
            columns=[
                "panel_name",
                "source_feature_name",
                "modality_name",
                "component_index",
                "raw_feature_id",
                "raw_feature_annotation",
                "loading_rank",
                "abs_loading",
                "n_queues_available",
                "primary_abs_component_correlation",
                "mean_abs_component_correlation",
                "min_abs_component_correlation",
                "proxy_selection_score",
            ]
        )

    rows: list[dict[str, object]] = []
    for candidate in component_marker_table.itertuples(index=False):
        component_name = f"component_{int(candidate.component_index)}"
        queue_corrs: dict[str, float] = {}
        for queue_key, prepared in queue_prepared.items():
            preprocessed = getattr(prepared, "preprocessed", {})
            embeddings = getattr(prepared, "embeddings", {})
            if candidate.modality_name not in preprocessed or candidate.modality_name not in embeddings:
                continue
            modality_frame = preprocessed[candidate.modality_name]
            modality_embedding = embeddings[candidate.modality_name]
            if candidate.raw_feature_id not in modality_frame.index or component_name not in modality_embedding.columns:
                continue
            raw_series = modality_frame.loc[candidate.raw_feature_id].astype(float)
            component_series = modality_embedding[component_name].astype(float)
            corr = raw_series.corr(component_series)
            if pd.notna(corr):
                queue_corrs[str(queue_key)] = abs(float(corr))

        corr_values = list(queue_corrs.values())
        primary_corr = queue_corrs.get(primary_queue_key, np.nan)
        mean_corr = float(np.mean(corr_values)) if corr_values else np.nan
        min_corr = float(np.min(corr_values)) if corr_values else np.nan
        proxy_score = (
            (0.0 if np.isnan(primary_corr) else primary_corr * 8.0)
            + (0.0 if np.isnan(mean_corr) else mean_corr * 6.0)
            + (0.0 if np.isnan(min_corr) else min_corr * 4.0)
            + float(candidate.abs_loading) * 2.0
            + len(corr_values)
        )
        rows.append(
            {
                "panel_name": str(candidate.panel_name),
                "source_feature_name": str(candidate.source_feature_name),
                "modality_name": str(candidate.modality_name),
                "component_index": int(candidate.component_index),
                "raw_feature_id": str(candidate.raw_feature_id),
                "raw_feature_annotation": str(candidate.raw_feature_annotation),
                "loading_rank": int(candidate.loading_rank),
                "abs_loading": float(candidate.abs_loading),
                "n_queues_available": int(len(corr_values)),
                "primary_abs_component_correlation": primary_corr,
                "mean_abs_component_correlation": mean_corr,
                "min_abs_component_correlation": min_corr,
                "proxy_selection_score": float(proxy_score),
            }
        )

    proxy_table = pd.DataFrame(rows)
    if proxy_table.empty:
        return proxy_table
    return proxy_table.sort_values(
        [
            "panel_name",
            "source_feature_name",
            "proxy_selection_score",
            "mean_abs_component_correlation",
            "abs_loading",
            "loading_rank",
        ],
        ascending=[True, True, False, False, False, True],
    ).reset_index(drop=True)


def build_refined_raw_marker_panels(
    panel_features: pd.DataFrame,
    component_marker_table: pd.DataFrame,
    proxy_candidate_scores: pd.DataFrame,
) -> pd.DataFrame:
    if panel_features.empty or proxy_candidate_scores.empty:
        return pd.DataFrame()

    latent_panels = [panel_name for panel_name, group in panel_features.groupby("panel_name", sort=False) if (group["feature_layer"] == "latent_embedding").any()]
    rows: list[dict[str, object]] = []
    proxy_lookup = proxy_candidate_scores.groupby(["panel_name", "source_feature_name"], sort=False)

    for source_panel_name in latent_panels:
        source_panel = panel_features.loc[panel_features["panel_name"] == source_panel_name].copy()
        raw_panel_name = source_panel_name.replace("core", "refined")
        used_raw_features: set[tuple[str, str]] = set()
        for row in source_panel.itertuples(index=False):
            if str(row.feature_layer) == "mechanism_summary":
                rows.append(
                    {
                        "raw_panel_name": raw_panel_name,
                        "source_panel_name": source_panel_name,
                        "raw_panel_description": f"Proxy-optimized raw-marker translation of {source_panel_name}.",
                        "source_feature_name": str(row.feature_name),
                        "mapped_feature_name": str(row.feature_name),
                        "mapped_feature_layer": "mechanism_summary",
                        "modality_name": str(row.modality_name),
                        "mapped_feature_annotation": str(row.feature_display_name),
                        "mapping_source": "summary_preserved",
                    }
                )
                continue
            candidates = proxy_lookup.get_group((source_panel_name, str(row.feature_name))).copy()
            chosen = None
            for candidate in candidates.itertuples(index=False):
                raw_key = (str(candidate.modality_name), str(candidate.raw_feature_id))
                if raw_key not in used_raw_features:
                    chosen = candidate
                    used_raw_features.add(raw_key)
                    break
            if chosen is None:
                chosen = candidates.iloc[0]
                used_raw_features.add((str(chosen["modality_name"]), str(chosen["raw_feature_id"])))
                chosen = type("Chosen", (), chosen.to_dict())
            rows.append(
                {
                    "raw_panel_name": raw_panel_name,
                    "source_panel_name": source_panel_name,
                    "raw_panel_description": f"Proxy-optimized raw-marker translation of {source_panel_name}.",
                    "source_feature_name": str(row.feature_name),
                    "mapped_feature_name": f"raw__{chosen.modality_name}__{chosen.raw_feature_id}",
                    "mapped_feature_layer": "raw_selected",
                    "modality_name": str(chosen.modality_name),
                    "mapped_feature_annotation": str(chosen.raw_feature_annotation),
                    "mapping_source": str(chosen.raw_feature_id),
                }
            )
    refined_panels = pd.DataFrame(rows)
    if refined_panels.empty:
        return refined_panels
    refined_panels["raw_panel_feature_count"] = refined_panels.groupby("raw_panel_name")["mapped_feature_name"].transform("nunique")
    refined_panels["feature_rank_within_panel"] = refined_panels.groupby("raw_panel_name").cumcount() + 1
    return refined_panels


def build_raw_panel_feature_matrix(
    raw_panel_features: pd.DataFrame,
    preprocessed: dict[str, pd.DataFrame],
    omics: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if raw_panel_features.empty:
        return pd.DataFrame()
    mechanism_features = compute_mechanism_summary_features(omics).sort_index()
    panel_frames: list[pd.DataFrame] = []
    for row in raw_panel_features.itertuples(index=False):
        if str(row.mapped_feature_layer) == "mechanism_summary":
            summary_name = str(row.mapped_feature_name).removeprefix("summary__")
            if summary_name not in mechanism_features.columns:
                continue
            panel_frames.append(mechanism_features[[summary_name]].rename(columns={summary_name: str(row.mapped_feature_name)}))
        else:
            modality_name = str(row.modality_name)
            raw_feature_id = str(row.mapping_source)
            if modality_name not in preprocessed or raw_feature_id not in preprocessed[modality_name].index:
                continue
            series = preprocessed[modality_name].loc[raw_feature_id].astype(np.float32)
            panel_frames.append(series.to_frame(name=str(row.mapped_feature_name)))
    if not panel_frames:
        return pd.DataFrame(index=mechanism_features.index)
    feature_frame = pd.concat(panel_frames, axis=1)
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()].sort_index()
    return feature_frame.astype(np.float32)


def raw_signature_panel_summary_frame(
    panel_summary_rows: list[pd.DataFrame],
    full_classifier_summary: pd.DataFrame,
    latent_panel_summary: pd.DataFrame,
) -> pd.DataFrame:
    if not panel_summary_rows:
        return pd.DataFrame()
    summary = pd.concat(panel_summary_rows, ignore_index=True)
    baseline = signature_panel_summary_frame([], full_classifier_summary)
    del baseline
    full_baseline = full_classifier_summary[
        ["queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc", "n_features"]
    ].rename(
        columns={
            "macro_f1": "baseline_macro_f1",
            "balanced_accuracy": "baseline_balanced_accuracy",
            "accuracy": "baseline_accuracy",
            "macro_ovr_auroc": "baseline_macro_ovr_auroc",
            "n_features": "baseline_n_features",
        }
    )
    summary = summary.merge(full_baseline, on=["queue_key", "classifier_name"], how="left")
    summary["delta_macro_f1"] = summary["macro_f1"] - summary["baseline_macro_f1"]
    summary["delta_balanced_accuracy"] = summary["balanced_accuracy"] - summary["baseline_balanced_accuracy"]
    summary["delta_accuracy"] = summary["accuracy"] - summary["baseline_accuracy"]
    summary["delta_macro_ovr_auroc"] = summary["macro_ovr_auroc"] - summary["baseline_macro_ovr_auroc"]
    latent_lookup = latent_panel_summary[
        ["panel_name", "queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc"]
    ].rename(
        columns={
            "panel_name": "source_panel_name",
            "macro_f1": "source_panel_macro_f1",
            "balanced_accuracy": "source_panel_balanced_accuracy",
            "accuracy": "source_panel_accuracy",
            "macro_ovr_auroc": "source_panel_macro_ovr_auroc",
        }
    )
    summary = summary.merge(latent_lookup, on=["source_panel_name", "queue_key", "classifier_name"], how="left")
    summary["delta_macro_f1_vs_source_panel"] = summary["macro_f1"] - summary["source_panel_macro_f1"]
    summary["delta_balanced_accuracy_vs_source_panel"] = summary["balanced_accuracy"] - summary["source_panel_balanced_accuracy"]
    summary["delta_accuracy_vs_source_panel"] = summary["accuracy"] - summary["source_panel_accuracy"]
    summary["delta_macro_ovr_auroc_vs_source_panel"] = summary["macro_ovr_auroc"] - summary["source_panel_macro_ovr_auroc"]
    return summary.sort_values(["raw_panel_name", "queue_key", "classifier_name"]).reset_index(drop=True)


def raw_signature_shortlist_markdown(
    raw_panel_features: pd.DataFrame,
    raw_panel_summary: pd.DataFrame,
    title: str = "Raw Signature Shortlist",
) -> str:
    lines = [f"# {title}", "", "## Raw Panels"]
    for panel_name, panel_df in raw_panel_features.groupby("raw_panel_name", sort=False):
        lines.append(f"### {panel_name}")
        lines.append(str(panel_df["raw_panel_description"].iloc[0]))
        lines.append("")
        feature_bits = [
            f"{row.mapped_feature_name} <- {row.source_feature_name} ({row.mapped_feature_annotation})"
            for row in panel_df.itertuples(index=False)
        ]
        lines.append("- " + "\n- ".join(feature_bits))
        lines.append("")
    lines.append("## Performance")
    for panel_name, panel_df in raw_panel_summary.groupby("raw_panel_name", sort=False):
        lines.append(f"### {panel_name}")
        for row in panel_df.itertuples(index=False):
            lines.append(
                f"- {row.queue_key} / {row.classifier_name}: "
                f"macro-F1 {row.macro_f1:.3f} "
                f"(vs full {row.delta_macro_f1:+.3f}, "
                f"vs {row.source_panel_name} {row.delta_macro_f1_vs_source_panel:+.3f}), "
                f"AUROC {row.macro_ovr_auroc:.3f}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _panel_feature_signature(panel_df: pd.DataFrame) -> str:
    feature_names = sorted({str(value) for value in panel_df["mapped_feature_name"].tolist()})
    return " | ".join(feature_names)


def _panel_name_priority(panel_name: str) -> tuple[int, str]:
    lowered = panel_name.lower()
    if lowered.startswith("stable_"):
        return (0, panel_name)
    if lowered.startswith("hybrid_"):
        return (1, panel_name)
    if lowered.startswith("summary_"):
        return (2, panel_name)
    return (3, panel_name)


def _frozen_reduction_candidate_specs(feature_names: set[str]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for feature_name in (
        "summary__cnv_burden",
        "summary__cnv_altered_fraction",
        "summary__mutation_event_fraction",
        "summary__tmb",
    ):
        if feature_name in feature_names:
            candidates.append(
                {
                    "reduced_panel_name": f"stable_reduced_{11 if feature_name.startswith('summary__') else 11}_no_{feature_name.removeprefix('summary__')}",
                    "reduction_strategy": "single_feature_drop",
                    "reduction_description": f"Drop `{feature_name}` from the frozen refined panel to test whether this summary axis is redundant.",
                    "drop_features": (feature_name,),
                }
            )
    cnv_present = [feature_name for feature_name in FROZEN_CNV_SUMMARY_FEATURES if feature_name in feature_names]
    mutation_present = [feature_name for feature_name in FROZEN_MUTATION_SUMMARY_FEATURES if feature_name in feature_names]
    if len(cnv_present) >= 2:
        keep_to_suffix = {
            "summary__cnv_loss_burden": "cnv_loss_only",
            "summary__cnv_altered_fraction": "cnv_altered_only",
            "summary__cnv_burden": "cnv_burden_only",
        }
        for keep_feature in cnv_present:
            drop_features = tuple(feature_name for feature_name in cnv_present if feature_name != keep_feature)
            if not drop_features:
                continue
            candidates.append(
                {
                    "reduced_panel_name": f"stable_reduced_10_{keep_to_suffix[keep_feature]}",
                    "reduction_strategy": "cnv_group_collapse",
                    "reduction_description": f"Collapse the three CNV summary features to `{keep_feature}` only.",
                    "drop_features": drop_features,
                }
            )
    if {"summary__cnv_loss_burden", "summary__tmb"}.issubset(feature_names):
        drop_features = tuple(
            feature_name
            for feature_name in (
                "summary__cnv_altered_fraction",
                "summary__cnv_burden",
                "summary__mutation_event_fraction",
            )
            if feature_name in feature_names
        )
        candidates.append(
            {
                "reduced_panel_name": "stable_reduced_9_cnv_loss_tmb",
                "reduction_strategy": "group_collapse",
                "reduction_description": "Keep one directional CNV summary (`cnv_loss_burden`) and one mutation summary (`tmb`) while preserving all raw markers.",
                "drop_features": drop_features,
            }
        )
    if {"summary__cnv_altered_fraction", "summary__tmb"}.issubset(feature_names):
        drop_features = tuple(
            feature_name
            for feature_name in (
                "summary__cnv_loss_burden",
                "summary__cnv_burden",
                "summary__mutation_event_fraction",
            )
            if feature_name in feature_names
        )
        candidates.append(
            {
                "reduced_panel_name": "stable_reduced_9_cnv_altered_tmb",
                "reduction_strategy": "group_collapse",
                "reduction_description": "Keep the CNV altered-fraction summary plus `tmb` to test a more global CNV summary anchor.",
                "drop_features": drop_features,
            }
        )
    deduped: list[dict[str, object]] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in candidates:
        key = (
            str(candidate["reduced_panel_name"]),
            tuple(sorted(str(feature_name) for feature_name in candidate["drop_features"])),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(candidate)
    return deduped


def _queue_classifier_metric(
    panel_summary: pd.DataFrame,
    queue_key: str,
    classifier_name: str,
    column: str,
) -> float:
    rows = panel_summary[
        (panel_summary["queue_key"] == queue_key)
        & (panel_summary["classifier_name"] == classifier_name)
    ]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


def build_frozen_panel_reduction_candidates(
    frozen_panel_features: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "reduced_panel_name",
        "parent_frozen_panel_name",
        "source_panel_name",
        "reduction_strategy",
        "reduction_description",
        "dropped_feature_names",
        "raw_panel_feature_count",
        "feature_rank_within_panel",
        "source_feature_name",
        "mapped_feature_name",
        "mapped_feature_layer",
        "modality_name",
        "mapped_feature_annotation",
        "mapping_source",
    ]
    if frozen_panel_features.empty:
        return pd.DataFrame(columns=columns)
    feature_names = {str(value) for value in frozen_panel_features["mapped_feature_name"].tolist()}
    candidates = _frozen_reduction_candidate_specs(feature_names)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        drop_features = {str(feature_name) for feature_name in candidate["drop_features"]}
        reduced = frozen_panel_features.loc[
            ~frozen_panel_features["mapped_feature_name"].astype(str).isin(drop_features)
        ].copy()
        if reduced.empty or reduced["mapped_feature_name"].nunique() < 2:
            continue
        reduced = reduced.drop(
            columns=[
                "raw_panel_feature_count",
                "parent_frozen_panel_name",
                "reduction_strategy",
                "reduction_description",
                "dropped_feature_names",
            ],
            errors="ignore",
        )
        reduced = reduced.sort_values("feature_rank_within_panel").reset_index(drop=True)
        reduced["feature_rank_within_panel"] = np.arange(1, len(reduced) + 1)
        reduced.insert(0, "raw_panel_feature_count", int(reduced["mapped_feature_name"].nunique()))
        reduced.insert(0, "dropped_feature_names", ",".join(sorted(drop_features)))
        reduced.insert(0, "reduction_description", str(candidate["reduction_description"]))
        reduced.insert(0, "reduction_strategy", str(candidate["reduction_strategy"]))
        reduced.insert(0, "parent_frozen_panel_name", str(reduced["frozen_panel_name"].iloc[0]))
        reduced.insert(0, "reduced_panel_name", str(candidate["reduced_panel_name"]))
        rows.extend(reduced.loc[:, columns].to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["raw_panel_feature_count", "reduced_panel_name", "feature_rank_within_panel"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def reduction_signature_panel_summary_frame(
    panel_summary_rows: list[pd.DataFrame],
    full_classifier_summary: pd.DataFrame,
    latent_panel_summary: pd.DataFrame,
    frozen_panel_summary: pd.DataFrame,
) -> pd.DataFrame:
    if not panel_summary_rows:
        return pd.DataFrame()
    summary = pd.concat(panel_summary_rows, ignore_index=True)
    full_baseline = full_classifier_summary[
        ["queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc", "n_features"]
    ].rename(
        columns={
            "macro_f1": "baseline_macro_f1",
            "balanced_accuracy": "baseline_balanced_accuracy",
            "accuracy": "baseline_accuracy",
            "macro_ovr_auroc": "baseline_macro_ovr_auroc",
            "n_features": "baseline_n_features",
        }
    )
    summary = summary.merge(full_baseline, on=["queue_key", "classifier_name"], how="left")
    summary["delta_macro_f1"] = summary["macro_f1"] - summary["baseline_macro_f1"]
    summary["delta_balanced_accuracy"] = summary["balanced_accuracy"] - summary["baseline_balanced_accuracy"]
    summary["delta_accuracy"] = summary["accuracy"] - summary["baseline_accuracy"]
    summary["delta_macro_ovr_auroc"] = summary["macro_ovr_auroc"] - summary["baseline_macro_ovr_auroc"]

    latent_lookup = latent_panel_summary[
        ["panel_name", "queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc"]
    ].rename(
        columns={
            "panel_name": "source_panel_name",
            "macro_f1": "source_panel_macro_f1",
            "balanced_accuracy": "source_panel_balanced_accuracy",
            "accuracy": "source_panel_accuracy",
            "macro_ovr_auroc": "source_panel_macro_ovr_auroc",
        }
    )
    summary = summary.merge(latent_lookup, on=["source_panel_name", "queue_key", "classifier_name"], how="left")
    summary["delta_macro_f1_vs_source_panel"] = summary["macro_f1"] - summary["source_panel_macro_f1"]
    summary["delta_balanced_accuracy_vs_source_panel"] = summary["balanced_accuracy"] - summary["source_panel_balanced_accuracy"]
    summary["delta_accuracy_vs_source_panel"] = summary["accuracy"] - summary["source_panel_accuracy"]
    summary["delta_macro_ovr_auroc_vs_source_panel"] = summary["macro_ovr_auroc"] - summary["source_panel_macro_ovr_auroc"]

    frozen_lookup = frozen_panel_summary[
        ["raw_panel_name", "queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc"]
    ].rename(
        columns={
            "raw_panel_name": "parent_frozen_panel_name",
            "macro_f1": "parent_frozen_macro_f1",
            "balanced_accuracy": "parent_frozen_balanced_accuracy",
            "accuracy": "parent_frozen_accuracy",
            "macro_ovr_auroc": "parent_frozen_macro_ovr_auroc",
        }
    )
    summary = summary.merge(
        frozen_lookup,
        on=["parent_frozen_panel_name", "queue_key", "classifier_name"],
        how="left",
    )
    summary["delta_macro_f1_vs_parent_frozen"] = summary["macro_f1"] - summary["parent_frozen_macro_f1"]
    summary["delta_balanced_accuracy_vs_parent_frozen"] = summary["balanced_accuracy"] - summary["parent_frozen_balanced_accuracy"]
    summary["delta_accuracy_vs_parent_frozen"] = summary["accuracy"] - summary["parent_frozen_accuracy"]
    summary["delta_macro_ovr_auroc_vs_parent_frozen"] = summary["macro_ovr_auroc"] - summary["parent_frozen_macro_ovr_auroc"]
    return summary.sort_values(["reduced_panel_name", "queue_key", "classifier_name"]).reset_index(drop=True)


def build_reduced_panel_selection(
    reduced_panel_features: pd.DataFrame,
    reduced_panel_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "reduced_panel_name",
        "parent_frozen_panel_name",
        "source_panel_name",
        "reduction_strategy",
        "dropped_feature_names",
        "raw_panel_feature_count",
        "validation_mean_macro_f1",
        "validation_mean_delta_vs_parent_frozen",
        "validation_svm_macro_f1",
        "validation_svm_delta_vs_source_panel",
        "validation_xgboost_macro_f1",
        "validation_xgboost_delta_vs_parent_frozen",
        "primary_mean_macro_f1",
        "primary_mean_delta_vs_parent_frozen",
        "overall_mean_macro_f1",
        "selection_score",
        "selection_rank",
        "selected",
    ]
    if reduced_panel_features.empty or reduced_panel_summary.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for panel_name, panel_df in reduced_panel_features.groupby("reduced_panel_name", sort=False):
        summary_rows = reduced_panel_summary.loc[reduced_panel_summary["reduced_panel_name"] == panel_name].copy()
        if summary_rows.empty:
            continue
        validation_rows = summary_rows.loc[summary_rows["queue_key"] == "validation_with_rppa"].copy()
        primary_rows = summary_rows.loc[summary_rows["queue_key"] == "primary_discovery"].copy()
        validation_svm = validation_rows.loc[validation_rows["classifier_name"] == "svm"]
        validation_xgb = validation_rows.loc[validation_rows["classifier_name"] == "xgboost"]
        selection_score = (
            float(validation_rows["delta_macro_f1_vs_parent_frozen"].mean()) * 140.0
            + float(validation_rows["macro_f1"].mean()) * 10.0
            + float(validation_xgb["delta_macro_f1_vs_parent_frozen"].mean()) * 80.0
            + float(primary_rows["delta_macro_f1_vs_parent_frozen"].mean()) * 70.0
            + float(validation_svm["delta_macro_f1_vs_source_panel"].mean()) * 20.0
            - int(panel_df["raw_panel_feature_count"].iloc[0]) * 0.03
        )
        records.append(
            {
                "reduced_panel_name": str(panel_name),
                "parent_frozen_panel_name": str(panel_df["parent_frozen_panel_name"].iloc[0]),
                "source_panel_name": str(panel_df["source_panel_name"].iloc[0]),
                "reduction_strategy": str(panel_df["reduction_strategy"].iloc[0]),
                "dropped_feature_names": str(panel_df["dropped_feature_names"].iloc[0]),
                "raw_panel_feature_count": int(panel_df["raw_panel_feature_count"].iloc[0]),
                "validation_mean_macro_f1": float(validation_rows["macro_f1"].mean()) if not validation_rows.empty else np.nan,
                "validation_mean_delta_vs_parent_frozen": (
                    float(validation_rows["delta_macro_f1_vs_parent_frozen"].mean()) if not validation_rows.empty else np.nan
                ),
                "validation_svm_macro_f1": (
                    float(validation_svm["macro_f1"].iloc[0]) if not validation_svm.empty else np.nan
                ),
                "validation_svm_delta_vs_source_panel": (
                    float(validation_svm["delta_macro_f1_vs_source_panel"].iloc[0]) if not validation_svm.empty else np.nan
                ),
                "validation_xgboost_macro_f1": (
                    float(validation_xgb["macro_f1"].iloc[0]) if not validation_xgb.empty else np.nan
                ),
                "validation_xgboost_delta_vs_parent_frozen": (
                    float(validation_xgb["delta_macro_f1_vs_parent_frozen"].iloc[0]) if not validation_xgb.empty else np.nan
                ),
                "primary_mean_macro_f1": float(primary_rows["macro_f1"].mean()) if not primary_rows.empty else np.nan,
                "primary_mean_delta_vs_parent_frozen": (
                    float(primary_rows["delta_macro_f1_vs_parent_frozen"].mean()) if not primary_rows.empty else np.nan
                ),
                "overall_mean_macro_f1": float(summary_rows["macro_f1"].mean()),
                "selection_score": float(selection_score),
            }
        )
    selection = pd.DataFrame(records)
    if selection.empty:
        return pd.DataFrame(columns=columns)
    selection = selection.sort_values(
        [
            "selection_score",
            "validation_mean_delta_vs_parent_frozen",
            "validation_xgboost_delta_vs_parent_frozen",
            "validation_mean_macro_f1",
            "primary_mean_delta_vs_parent_frozen",
            "raw_panel_feature_count",
            "reduced_panel_name",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)
    selection["selection_rank"] = np.arange(1, len(selection) + 1)
    selection["selected"] = False
    selection.loc[selection.index[0], "selected"] = True
    return selection.loc[:, columns]


def reduced_signature_shortlist_markdown(
    reduced_panel_features: pd.DataFrame,
    reduced_panel_summary: pd.DataFrame,
    reduced_panel_selection: pd.DataFrame,
    title: str = "Reduced Frozen Signature Shortlist",
) -> str:
    lines = [f"# {title}", "", "## Reduced Panels"]
    for panel_name, panel_df in reduced_panel_features.groupby("reduced_panel_name", sort=False):
        lines.append(f"### {panel_name}")
        lines.append(str(panel_df["reduction_description"].iloc[0]))
        lines.append("")
        lines.append(f"- Parent frozen panel: `{panel_df['parent_frozen_panel_name'].iloc[0]}`")
        lines.append(f"- Dropped features: `{panel_df['dropped_feature_names'].iloc[0]}`")
        lines.append("")
        feature_bits = [
            f"{row.mapped_feature_name} <- {row.source_feature_name} ({row.mapped_feature_annotation})"
            for row in panel_df.itertuples(index=False)
        ]
        lines.append("- " + "\n- ".join(feature_bits))
        lines.append("")
    lines.append("## Performance")
    for panel_name, panel_df in reduced_panel_summary.groupby("reduced_panel_name", sort=False):
        lines.append(f"### {panel_name}")
        for row in panel_df.itertuples(index=False):
            lines.append(
                f"- {row.queue_key} / {row.classifier_name}: "
                f"macro-F1 {row.macro_f1:.3f} "
                f"(vs full {row.delta_macro_f1:+.3f}, "
                f"vs source {row.delta_macro_f1_vs_source_panel:+.3f}, "
                f"vs frozen {row.delta_macro_f1_vs_parent_frozen:+.3f}), "
                f"AUROC {row.macro_ovr_auroc:.3f}"
            )
        lines.append("")
    if not reduced_panel_selection.empty:
        chosen = reduced_panel_selection.loc[reduced_panel_selection["selected"]].iloc[0]
        lines.extend(
            [
                "## Selection",
                "",
                f"- Selected reduced panel: `{chosen.reduced_panel_name}`",
                f"- Validation mean macro-F1: `{chosen.validation_mean_macro_f1:.3f}`",
                f"- Validation mean delta vs parent frozen: `{chosen.validation_mean_delta_vs_parent_frozen:+.3f}`",
                f"- Validation XGBoost delta vs parent frozen: `{chosen.validation_xgboost_delta_vs_parent_frozen:+.3f}`",
                f"- Primary mean delta vs parent frozen: `{chosen.primary_mean_delta_vs_parent_frozen:+.3f}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_final_panel_candidates(
    selected_reduced_panel_features: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "final_panel_name",
        "parent_reduced_panel_name",
        "parent_frozen_panel_name",
        "source_panel_name",
        "compression_strategy",
        "compression_description",
        "dropped_feature_names",
        "raw_panel_feature_count",
        "feature_rank_within_panel",
        "source_feature_name",
        "mapped_feature_name",
        "mapped_feature_layer",
        "modality_name",
        "mapped_feature_annotation",
        "mapping_source",
    ]
    if selected_reduced_panel_features.empty:
        return pd.DataFrame(columns=columns)
    feature_names = {str(value) for value in selected_reduced_panel_features["mapped_feature_name"].tolist()}
    candidate_specs: list[dict[str, object]] = []
    if "summary__mutation_event_fraction" in feature_names:
        candidate_specs.append(
            {
                "final_panel_name": "stable_final_9_no_mutation_event_fraction",
                "compression_strategy": "single_feature_drop",
                "compression_description": "Drop `mutation event fraction` from the reduced panel to test whether the remaining `tmb` summary is sufficient.",
                "drop_features": ("summary__mutation_event_fraction",),
            }
        )
    if "summary__tmb" in feature_names:
        candidate_specs.append(
            {
                "final_panel_name": "stable_final_9_no_tmb",
                "compression_strategy": "single_feature_drop",
                "compression_description": "Drop `tmb` from the reduced panel to test whether `mutation event fraction` carries the mutation-summary signal alone.",
                "drop_features": ("summary__tmb",),
            }
        )
    if {"summary__tmb", "summary__mutation_event_fraction"}.issubset(feature_names):
        candidate_specs.append(
            {
                "final_panel_name": "stable_final_8_cnv_burden_only",
                "compression_strategy": "mutation_summary_collapse",
                "compression_description": "Drop both mutation-summary features and keep only the raw markers plus `cnv_burden`.",
                "drop_features": ("summary__tmb", "summary__mutation_event_fraction"),
            }
        )
    rows: list[dict[str, object]] = []
    for candidate in candidate_specs:
        drop_features = {str(feature_name) for feature_name in candidate["drop_features"]}
        reduced = selected_reduced_panel_features.loc[
            ~selected_reduced_panel_features["mapped_feature_name"].astype(str).isin(drop_features)
        ].copy()
        if reduced.empty or reduced["mapped_feature_name"].nunique() < 2:
            continue
        reduced = reduced.drop(
            columns=[
                "raw_panel_feature_count",
                "compression_strategy",
                "compression_description",
                "dropped_feature_names",
                "final_panel_name",
            ],
            errors="ignore",
        )
        reduced = reduced.sort_values("feature_rank_within_panel").reset_index(drop=True)
        reduced["feature_rank_within_panel"] = np.arange(1, len(reduced) + 1)
        reduced.insert(0, "raw_panel_feature_count", int(reduced["mapped_feature_name"].nunique()))
        reduced.insert(0, "dropped_feature_names", ",".join(sorted(drop_features)))
        reduced.insert(0, "compression_description", str(candidate["compression_description"]))
        reduced.insert(0, "compression_strategy", str(candidate["compression_strategy"]))
        reduced.insert(0, "parent_reduced_panel_name", str(reduced["reduced_panel_name"].iloc[0]))
        reduced.insert(0, "final_panel_name", str(candidate["final_panel_name"]))
        rows.extend(reduced.loc[:, columns].to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["raw_panel_feature_count", "final_panel_name", "feature_rank_within_panel"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def final_signature_panel_summary_frame(
    panel_summary_rows: list[pd.DataFrame],
    full_classifier_summary: pd.DataFrame,
    latent_panel_summary: pd.DataFrame,
    reduced_panel_summary: pd.DataFrame,
) -> pd.DataFrame:
    if not panel_summary_rows:
        return pd.DataFrame()
    summary = pd.concat(panel_summary_rows, ignore_index=True)
    full_baseline = full_classifier_summary[
        ["queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc", "n_features"]
    ].rename(
        columns={
            "macro_f1": "baseline_macro_f1",
            "balanced_accuracy": "baseline_balanced_accuracy",
            "accuracy": "baseline_accuracy",
            "macro_ovr_auroc": "baseline_macro_ovr_auroc",
            "n_features": "baseline_n_features",
        }
    )
    summary = summary.merge(full_baseline, on=["queue_key", "classifier_name"], how="left")
    summary["delta_macro_f1"] = summary["macro_f1"] - summary["baseline_macro_f1"]
    summary["delta_balanced_accuracy"] = summary["balanced_accuracy"] - summary["baseline_balanced_accuracy"]
    summary["delta_accuracy"] = summary["accuracy"] - summary["baseline_accuracy"]
    summary["delta_macro_ovr_auroc"] = summary["macro_ovr_auroc"] - summary["baseline_macro_ovr_auroc"]

    latent_lookup = latent_panel_summary[
        ["panel_name", "queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc"]
    ].rename(
        columns={
            "panel_name": "source_panel_name",
            "macro_f1": "source_panel_macro_f1",
            "balanced_accuracy": "source_panel_balanced_accuracy",
            "accuracy": "source_panel_accuracy",
            "macro_ovr_auroc": "source_panel_macro_ovr_auroc",
        }
    )
    summary = summary.merge(latent_lookup, on=["source_panel_name", "queue_key", "classifier_name"], how="left")
    summary["delta_macro_f1_vs_source_panel"] = summary["macro_f1"] - summary["source_panel_macro_f1"]
    summary["delta_balanced_accuracy_vs_source_panel"] = summary["balanced_accuracy"] - summary["source_panel_balanced_accuracy"]
    summary["delta_accuracy_vs_source_panel"] = summary["accuracy"] - summary["source_panel_accuracy"]
    summary["delta_macro_ovr_auroc_vs_source_panel"] = summary["macro_ovr_auroc"] - summary["source_panel_macro_ovr_auroc"]

    reduced_lookup = reduced_panel_summary[
        ["reduced_panel_name", "queue_key", "classifier_name", "macro_f1", "balanced_accuracy", "accuracy", "macro_ovr_auroc"]
    ].rename(
        columns={
            "reduced_panel_name": "parent_reduced_panel_name",
            "macro_f1": "parent_reduced_macro_f1",
            "balanced_accuracy": "parent_reduced_balanced_accuracy",
            "accuracy": "parent_reduced_accuracy",
            "macro_ovr_auroc": "parent_reduced_macro_ovr_auroc",
        }
    )
    summary = summary.merge(
        reduced_lookup,
        on=["parent_reduced_panel_name", "queue_key", "classifier_name"],
        how="left",
    )
    summary["delta_macro_f1_vs_parent_reduced"] = summary["macro_f1"] - summary["parent_reduced_macro_f1"]
    summary["delta_balanced_accuracy_vs_parent_reduced"] = summary["balanced_accuracy"] - summary["parent_reduced_balanced_accuracy"]
    summary["delta_accuracy_vs_parent_reduced"] = summary["accuracy"] - summary["parent_reduced_accuracy"]
    summary["delta_macro_ovr_auroc_vs_parent_reduced"] = summary["macro_ovr_auroc"] - summary["parent_reduced_macro_ovr_auroc"]
    return summary.sort_values(["final_panel_name", "queue_key", "classifier_name"]).reset_index(drop=True)


def build_final_panel_selection(
    final_panel_features: pd.DataFrame,
    final_panel_summary: pd.DataFrame,
    reduced_panel_features: pd.DataFrame,
    reduced_panel_summary: pd.DataFrame,
    reduced_panel_selection: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "final_panel_name",
        "parent_reduced_panel_name",
        "source_panel_name",
        "compression_strategy",
        "dropped_feature_names",
        "raw_panel_feature_count",
        "is_safe_shrink",
        "validation_mean_macro_f1",
        "validation_mean_delta_vs_parent_reduced",
        "validation_svm_macro_f1",
        "validation_svm_delta_vs_parent_reduced",
        "validation_xgboost_macro_f1",
        "validation_xgboost_delta_vs_parent_reduced",
        "primary_mean_macro_f1",
        "primary_mean_delta_vs_parent_reduced",
        "overall_mean_macro_f1",
        "selection_score",
        "selection_rank",
        "selected",
    ]
    if reduced_panel_selection.empty:
        return pd.DataFrame(columns=columns)
    chosen_parent = reduced_panel_selection.loc[reduced_panel_selection["selected"]]
    if chosen_parent.empty:
        return pd.DataFrame(columns=columns)
    parent_row = chosen_parent.iloc[0]
    parent_name = str(parent_row["reduced_panel_name"])
    source_panel_name = str(parent_row["source_panel_name"])
    parent_features = reduced_panel_features.loc[reduced_panel_features["reduced_panel_name"] == parent_name].copy()
    parent_summary = reduced_panel_summary.loc[reduced_panel_summary["reduced_panel_name"] == parent_name].copy()

    records: list[dict[str, object]] = [
        {
            "final_panel_name": parent_name,
            "parent_reduced_panel_name": parent_name,
            "source_panel_name": source_panel_name,
            "compression_strategy": "keep_parent_reduced",
            "dropped_feature_names": "",
            "raw_panel_feature_count": int(parent_features["raw_panel_feature_count"].iloc[0]) if not parent_features.empty else np.nan,
            "is_safe_shrink": True,
            "validation_mean_macro_f1": float(parent_summary.loc[parent_summary["queue_key"] == "validation_with_rppa", "macro_f1"].mean()),
            "validation_mean_delta_vs_parent_reduced": 0.0,
            "validation_svm_macro_f1": _queue_classifier_metric(parent_summary, "validation_with_rppa", "svm", "macro_f1"),
            "validation_svm_delta_vs_parent_reduced": 0.0,
            "validation_xgboost_macro_f1": _queue_classifier_metric(parent_summary, "validation_with_rppa", "xgboost", "macro_f1"),
            "validation_xgboost_delta_vs_parent_reduced": 0.0,
            "primary_mean_macro_f1": float(parent_summary.loc[parent_summary["queue_key"] == "primary_discovery", "macro_f1"].mean()),
            "primary_mean_delta_vs_parent_reduced": 0.0,
            "overall_mean_macro_f1": float(parent_summary["macro_f1"].mean()),
            "selection_score": 0.0,
        }
    ]
    if not final_panel_features.empty and not final_panel_summary.empty:
        for panel_name, panel_df in final_panel_features.groupby("final_panel_name", sort=False):
            summary_rows = final_panel_summary.loc[final_panel_summary["final_panel_name"] == panel_name].copy()
            if summary_rows.empty:
                continue
            validation_rows = summary_rows.loc[summary_rows["queue_key"] == "validation_with_rppa"].copy()
            primary_rows = summary_rows.loc[summary_rows["queue_key"] == "primary_discovery"].copy()
            validation_svm = validation_rows.loc[validation_rows["classifier_name"] == "svm"]
            validation_xgb = validation_rows.loc[validation_rows["classifier_name"] == "xgboost"]
            validation_mean_delta = float(validation_rows["delta_macro_f1_vs_parent_reduced"].mean()) if not validation_rows.empty else np.nan
            primary_mean_delta = float(primary_rows["delta_macro_f1_vs_parent_reduced"].mean()) if not primary_rows.empty else np.nan
            validation_xgb_delta = float(validation_xgb["delta_macro_f1_vs_parent_reduced"].iloc[0]) if not validation_xgb.empty else np.nan
            is_safe_shrink = bool(
                (not np.isnan(validation_mean_delta) and validation_mean_delta >= -0.002)
                and (not np.isnan(validation_xgb_delta) and validation_xgb_delta >= -0.01)
                and (not np.isnan(primary_mean_delta) and primary_mean_delta >= -0.01)
            )
            selection_score = (
                float(validation_mean_delta) * 160.0
                + float(validation_xgb_delta) * 90.0
                + float(_queue_classifier_metric(validation_rows, "validation_with_rppa", "svm", "delta_macro_f1_vs_parent_reduced")) * 50.0
                + float(primary_mean_delta) * 80.0
                - int(panel_df["raw_panel_feature_count"].iloc[0]) * 0.08
            )
            records.append(
                {
                    "final_panel_name": str(panel_name),
                    "parent_reduced_panel_name": str(panel_df["parent_reduced_panel_name"].iloc[0]),
                    "source_panel_name": str(panel_df["source_panel_name"].iloc[0]),
                    "compression_strategy": str(panel_df["compression_strategy"].iloc[0]),
                    "dropped_feature_names": str(panel_df["dropped_feature_names"].iloc[0]),
                    "raw_panel_feature_count": int(panel_df["raw_panel_feature_count"].iloc[0]),
                    "is_safe_shrink": is_safe_shrink,
                    "validation_mean_macro_f1": float(validation_rows["macro_f1"].mean()) if not validation_rows.empty else np.nan,
                    "validation_mean_delta_vs_parent_reduced": validation_mean_delta,
                    "validation_svm_macro_f1": _queue_classifier_metric(validation_rows, "validation_with_rppa", "svm", "macro_f1"),
                    "validation_svm_delta_vs_parent_reduced": _queue_classifier_metric(
                        validation_rows,
                        "validation_with_rppa",
                        "svm",
                        "delta_macro_f1_vs_parent_reduced",
                    ),
                    "validation_xgboost_macro_f1": _queue_classifier_metric(validation_rows, "validation_with_rppa", "xgboost", "macro_f1"),
                    "validation_xgboost_delta_vs_parent_reduced": validation_xgb_delta,
                    "primary_mean_macro_f1": float(primary_rows["macro_f1"].mean()) if not primary_rows.empty else np.nan,
                    "primary_mean_delta_vs_parent_reduced": primary_mean_delta,
                    "overall_mean_macro_f1": float(summary_rows["macro_f1"].mean()),
                    "selection_score": float(selection_score),
                }
            )
    selection = pd.DataFrame(records)
    selection = selection.sort_values(
        [
            "is_safe_shrink",
            "selection_score",
            "validation_mean_delta_vs_parent_reduced",
            "validation_xgboost_delta_vs_parent_reduced",
            "primary_mean_delta_vs_parent_reduced",
            "raw_panel_feature_count",
            "final_panel_name",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)
    selection["selection_rank"] = np.arange(1, len(selection) + 1)
    selection["selected"] = False
    selection.loc[selection.index[0], "selected"] = True
    return selection.loc[:, columns]


def extract_selected_final_panel_features(
    final_panel_features: pd.DataFrame,
    final_panel_selection: pd.DataFrame,
    reduced_panel_features: pd.DataFrame,
) -> pd.DataFrame:
    if final_panel_selection.empty:
        return pd.DataFrame()
    chosen = final_panel_selection.loc[final_panel_selection["selected"]]
    if chosen.empty:
        return pd.DataFrame()
    selected_name = str(chosen.iloc[0]["final_panel_name"])
    if selected_name in set(final_panel_features["final_panel_name"].astype(str)):
        selected = final_panel_features.loc[final_panel_features["final_panel_name"] == selected_name].copy()
        selected.insert(0, "selected_final_panel_name", selected_name)
        return selected.reset_index(drop=True)
    selected = reduced_panel_features.loc[reduced_panel_features["reduced_panel_name"] == selected_name].copy()
    if selected.empty:
        return selected
    selected = selected.rename(columns={"reduced_panel_name": "selected_final_panel_name"})
    return selected.reset_index(drop=True)


def final_signature_shortlist_markdown(
    final_panel_features: pd.DataFrame,
    final_panel_summary: pd.DataFrame,
    final_panel_selection: pd.DataFrame,
    selected_final_panel_features: pd.DataFrame,
    title: str = "Final Signature Decision",
) -> str:
    lines = [f"# {title}", "", "## Final Candidates"]
    if final_panel_features.empty:
        lines.extend(["- No smaller final-candidate panel passed generation. The current recommendation keeps the selected reduced panel.", ""])
    else:
        for panel_name, panel_df in final_panel_features.groupby("final_panel_name", sort=False):
            lines.append(f"### {panel_name}")
            lines.append(str(panel_df["compression_description"].iloc[0]))
            lines.append("")
            lines.append(f"- Parent reduced panel: `{panel_df['parent_reduced_panel_name'].iloc[0]}`")
            lines.append(f"- Dropped features: `{panel_df['dropped_feature_names'].iloc[0]}`")
            lines.append("")
    lines.append("## Performance")
    if final_panel_summary.empty:
        lines.append("- No final-candidate performance table was generated.")
    else:
        for panel_name, panel_df in final_panel_summary.groupby("final_panel_name", sort=False):
            lines.append(f"### {panel_name}")
            for row in panel_df.itertuples(index=False):
                lines.append(
                    f"- {row.queue_key} / {row.classifier_name}: "
                    f"macro-F1 {row.macro_f1:.3f} "
                    f"(vs full {row.delta_macro_f1:+.3f}, "
                    f"vs source {row.delta_macro_f1_vs_source_panel:+.3f}, "
                    f"vs reduced {row.delta_macro_f1_vs_parent_reduced:+.3f}), "
                    f"AUROC {row.macro_ovr_auroc:.3f}"
                )
            lines.append("")
    if not final_panel_selection.empty:
        chosen = final_panel_selection.loc[final_panel_selection["selected"]].iloc[0]
        lines.extend(
            [
                "## Decision",
                "",
                f"- Selected final recommendation: `{chosen.final_panel_name}`",
                f"- Strategy: `{chosen.compression_strategy}`",
                f"- Safe shrink: `{bool(chosen.is_safe_shrink)}`",
                f"- Validation mean delta vs reduced parent: `{chosen.validation_mean_delta_vs_parent_reduced:+.3f}`",
                f"- Validation XGBoost delta vs reduced parent: `{chosen.validation_xgboost_delta_vs_parent_reduced:+.3f}`",
                f"- Primary mean delta vs reduced parent: `{chosen.primary_mean_delta_vs_parent_reduced:+.3f}`",
                "",
                "## Selected Features",
                "",
            ]
        )
        for row in selected_final_panel_features.sort_values("feature_rank_within_panel").itertuples(index=False):
            feature_name = getattr(row, "mapped_feature_name")
            source_feature = getattr(row, "source_feature_name")
            annotation = getattr(row, "mapped_feature_annotation")
            lines.append(f"- `{feature_name}` <- `{source_feature}` ({annotation})")
    return "\n".join(lines).strip() + "\n"


def build_refined_panel_freeze_candidates(
    refined_panel_features: pd.DataFrame,
    refined_panel_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "candidate_family_id",
        "representative_panel_name",
        "member_panel_names",
        "n_equivalent_panels",
        "source_panel_names",
        "raw_panel_feature_count",
        "feature_signature",
        "mapped_feature_annotations",
        "validation_best_classifier",
        "validation_best_macro_f1",
        "validation_mean_macro_f1",
        "validation_best_delta_macro_f1_vs_source_panel",
        "validation_mean_delta_macro_f1_vs_source_panel",
        "validation_svm_macro_f1",
        "validation_svm_delta_macro_f1_vs_source_panel",
        "validation_elastic_net_macro_f1",
        "validation_elastic_net_delta_macro_f1_vs_source_panel",
        "primary_best_classifier",
        "primary_best_macro_f1",
        "primary_mean_macro_f1",
        "primary_svm_macro_f1",
        "primary_elastic_net_macro_f1",
        "overall_mean_macro_f1",
        "freeze_priority_score",
        "selection_rank",
        "selected",
    ]
    if refined_panel_features.empty or refined_panel_summary.empty:
        return pd.DataFrame(columns=columns)

    family_records: list[dict[str, object]] = []
    families: dict[str, list[str]] = {}
    panel_frames = {
        str(panel_name): panel_df.copy()
        for panel_name, panel_df in refined_panel_features.groupby("raw_panel_name", sort=False)
    }
    for panel_name, panel_df in panel_frames.items():
        families.setdefault(_panel_feature_signature(panel_df), []).append(panel_name)

    for family_idx, feature_signature in enumerate(sorted(families), start=1):
        member_names = sorted(families[feature_signature], key=_panel_name_priority)
        representative = member_names[0]
        representative_features = panel_frames[representative].copy()
        family_summary = refined_panel_summary[
            refined_panel_summary["raw_panel_name"].isin(member_names)
        ].copy()
        if family_summary.empty:
            continue
        validation_rows = family_summary.loc[family_summary["queue_key"] == "validation_with_rppa"].copy()
        primary_rows = family_summary.loc[family_summary["queue_key"] == "primary_discovery"].copy()
        validation_best = (
            validation_rows.sort_values(
                ["delta_macro_f1_vs_source_panel", "macro_f1", "classifier_name"],
                ascending=[False, False, True],
            ).iloc[0]
            if not validation_rows.empty
            else None
        )
        primary_best = (
            primary_rows.sort_values(
                ["macro_f1", "delta_macro_f1_vs_source_panel", "classifier_name"],
                ascending=[False, False, True],
            ).iloc[0]
            if not primary_rows.empty
            else None
        )
        validation_svm_macro_f1 = _queue_classifier_metric(validation_rows, "validation_with_rppa", "svm", "macro_f1")
        validation_svm_delta = _queue_classifier_metric(
            validation_rows,
            "validation_with_rppa",
            "svm",
            "delta_macro_f1_vs_source_panel",
        )
        validation_enet_macro_f1 = _queue_classifier_metric(
            validation_rows,
            "validation_with_rppa",
            "elastic_net",
            "macro_f1",
        )
        validation_enet_delta = _queue_classifier_metric(
            validation_rows,
            "validation_with_rppa",
            "elastic_net",
            "delta_macro_f1_vs_source_panel",
        )
        primary_svm_macro_f1 = _queue_classifier_metric(primary_rows, "primary_discovery", "svm", "macro_f1")
        primary_enet_macro_f1 = _queue_classifier_metric(
            primary_rows,
            "primary_discovery",
            "elastic_net",
            "macro_f1",
        )
        freeze_priority_score = (
            (0.0 if np.isnan(validation_svm_delta) else validation_svm_delta * 100.0)
            + (0.0 if np.isnan(validation_svm_macro_f1) else validation_svm_macro_f1 * 10.0)
            + (0.0 if validation_rows.empty else float(validation_rows["macro_f1"].mean()) * 4.0)
            + (0.0 if primary_rows.empty else float(primary_rows["macro_f1"].mean()) * 3.0)
            - int(representative_features["raw_panel_feature_count"].iloc[0]) * 0.02
        )
        family_records.append(
            {
                "candidate_family_id": f"refined_family_{family_idx}",
                "representative_panel_name": representative,
                "member_panel_names": ",".join(member_names),
                "n_equivalent_panels": int(len(member_names)),
                "source_panel_names": ",".join(sorted(set(family_summary["source_panel_name"].astype(str)))),
                "raw_panel_feature_count": int(representative_features["raw_panel_feature_count"].iloc[0]),
                "feature_signature": feature_signature,
                "mapped_feature_annotations": " | ".join(
                    representative_features.sort_values("feature_rank_within_panel")["mapped_feature_annotation"].astype(str).tolist()
                ),
                "validation_best_classifier": (
                    str(validation_best["classifier_name"]) if validation_best is not None else ""
                ),
                "validation_best_macro_f1": (
                    float(validation_best["macro_f1"]) if validation_best is not None else np.nan
                ),
                "validation_mean_macro_f1": (
                    float(validation_rows["macro_f1"].mean()) if not validation_rows.empty else np.nan
                ),
                "validation_best_delta_macro_f1_vs_source_panel": (
                    float(validation_best["delta_macro_f1_vs_source_panel"]) if validation_best is not None else np.nan
                ),
                "validation_mean_delta_macro_f1_vs_source_panel": (
                    float(validation_rows["delta_macro_f1_vs_source_panel"].mean())
                    if not validation_rows.empty
                    else np.nan
                ),
                "validation_svm_macro_f1": validation_svm_macro_f1,
                "validation_svm_delta_macro_f1_vs_source_panel": validation_svm_delta,
                "validation_elastic_net_macro_f1": validation_enet_macro_f1,
                "validation_elastic_net_delta_macro_f1_vs_source_panel": validation_enet_delta,
                "primary_best_classifier": str(primary_best["classifier_name"]) if primary_best is not None else "",
                "primary_best_macro_f1": float(primary_best["macro_f1"]) if primary_best is not None else np.nan,
                "primary_mean_macro_f1": float(primary_rows["macro_f1"].mean()) if not primary_rows.empty else np.nan,
                "primary_svm_macro_f1": primary_svm_macro_f1,
                "primary_elastic_net_macro_f1": primary_enet_macro_f1,
                "overall_mean_macro_f1": float(family_summary["macro_f1"].mean()),
                "freeze_priority_score": float(freeze_priority_score),
            }
        )

    candidates = pd.DataFrame(family_records)
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates = candidates.sort_values(
        [
            "validation_svm_delta_macro_f1_vs_source_panel",
            "validation_svm_macro_f1",
            "validation_mean_macro_f1",
            "primary_mean_macro_f1",
            "raw_panel_feature_count",
            "representative_panel_name",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    candidates["selection_rank"] = np.arange(1, len(candidates) + 1)
    candidates["selected"] = False
    candidates.loc[candidates.index[0], "selected"] = True
    return candidates.loc[:, columns]


def extract_frozen_refined_panel_features(
    refined_panel_features: pd.DataFrame,
    freeze_selection: pd.DataFrame,
) -> pd.DataFrame:
    if refined_panel_features.empty or freeze_selection.empty:
        return pd.DataFrame()
    chosen = freeze_selection.loc[freeze_selection["selected"]]
    if chosen.empty:
        return pd.DataFrame()
    representative_panel_name = str(chosen.iloc[0]["representative_panel_name"])
    frozen = refined_panel_features.loc[
        refined_panel_features["raw_panel_name"] == representative_panel_name
    ].copy()
    if frozen.empty:
        return frozen
    frozen.insert(0, "frozen_panel_name", representative_panel_name)
    frozen.insert(1, "freeze_family_id", str(chosen.iloc[0]["candidate_family_id"]))
    frozen.insert(2, "freeze_member_panel_names", str(chosen.iloc[0]["member_panel_names"]))
    return frozen.reset_index(drop=True)


def frozen_refined_signature_markdown(
    freeze_selection: pd.DataFrame,
    frozen_panel_features: pd.DataFrame,
    title: str = "Frozen Refined Signature",
) -> str:
    lines = [f"# {title}"]
    if freeze_selection.empty or frozen_panel_features.empty:
        lines.extend(["", "No frozen refined panel is available.", ""])
        return "\n".join(lines).strip() + "\n"

    chosen = freeze_selection.loc[freeze_selection["selected"]].iloc[0]
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Selected family: `{chosen.candidate_family_id}`",
            f"- Representative panel: `{chosen.representative_panel_name}`",
            f"- Equivalent panels collapsed: `{chosen.member_panel_names}`",
            f"- Selection rule: prioritize validation-queue `SVM` fidelity to the source latent panel, then validation macro-F1, then primary-queue mean macro-F1, and finally shorter panel size",
            f"- Validation `SVM` macro-F1: `{chosen.validation_svm_macro_f1:.3f}`",
            f"- Validation `SVM` delta vs source panel: `{chosen.validation_svm_delta_macro_f1_vs_source_panel:+.3f}`",
            f"- Primary mean macro-F1 across classifiers: `{chosen.primary_mean_macro_f1:.3f}`",
            "",
            "## Features",
            "",
        ]
    )
    for row in frozen_panel_features.sort_values("feature_rank_within_panel").itertuples(index=False):
        lines.append(
            f"- `{row.mapped_feature_name}` <- `{row.source_feature_name}` ({row.mapped_feature_annotation})"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This frozen panel is the current recommended refined raw-marker candidate for downstream discussion and future tree-model checks.",
            "- Equivalent duplicate panels are collapsed before selection so the recommendation reflects feature content rather than naming differences.",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"
