from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModalityConfig:
    name: str
    path: str | tuple[str, ...]
    feature_col: str
    min_frequency: float | None = None
    max_missing_rate: float | None = None
    top_mad_features: int | None = None
    is_discrete: bool = False


@dataclass(frozen=True)
class AnalysisQueueConfig:
    key: str
    analysis_name: str
    out_subdir: str
    modalities: tuple[str, ...]
    description: str


MODALITIES: dict[str, ModalityConfig] = {
    "mutation": ModalityConfig(
        name="mutation",
        path=(
            "mc3_gene_level_STAD_mc3_gene_level.txt.gz",
            "mc3_gene_level_STAD_mc3_gene_level.txt",
            "STAD_mc3_gene_level.txt.gz",
            "STAD_mc3_gene_level.txt",
        ),
        feature_col="sample",
        min_frequency=0.03,
        is_discrete=True,
    ),
    "cnv": ModalityConfig(
        name="cnv",
        path=(
            "TCGA.STAD.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
            "TCGA.STAD.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
            "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        ),
        feature_col="Gene Symbol",
        min_frequency=0.05,
        is_discrete=True,
    ),
    "methylation": ModalityConfig(
        name="methylation",
        path=(
            "TCGA.STAD.sampleMap_HumanMethylation450.gz",
            "TCGA.STAD.sampleMap_HumanMethylation450",
            "HumanMethylation450",
        ),
        feature_col="sample",
        max_missing_rate=0.2,
        top_mad_features=5000,
    ),
    "expression": ModalityConfig(
        name="expression",
        path=(
            "TCGA.STAD.sampleMap_HiSeqV2.gz",
            "TCGA.STAD.sampleMap_HiSeqV2",
            "HiSeqV2",
        ),
        feature_col="sample",
        top_mad_features=2000,
    ),
    "mirna": ModalityConfig(
        name="mirna",
        path=(
            "TCGA.STAD.sampleMap_miRNA_HiSeq_gene.gz",
            "TCGA.STAD.sampleMap_miRNA_HiSeq_gene",
            "miRNA_HiSeq_gene",
        ),
        feature_col="sample",
        max_missing_rate=0.2,
        top_mad_features=500,
    ),
    "rppa": ModalityConfig(
        name="rppa",
        path=(
            "TCGA.STAD.sampleMap_RPPA.gz",
            "TCGA.STAD.sampleMap_RPPA",
            "RPPA",
        ),
        feature_col="sample",
        max_missing_rate=1.0,
    ),
}


CLINICAL_PATH = ("TCGA.STAD.sampleMap_STAD_clinicalMatrix",)
SURVIVAL_PATH = ("survival_STAD_survival.txt",)
METHYLATION_PROBE_MAP = ("probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy",)

PRIMARY_DISCOVERY_QUEUE = AnalysisQueueConfig(
    key="primary_discovery",
    analysis_name="primary_5_omics",
    out_subdir="",
    modalities=("mutation", "cnv", "methylation", "expression", "mirna"),
    description="Main discovery cohort without RPPA to preserve sample size.",
)

EXTENSION_VALIDATION_QUEUE = AnalysisQueueConfig(
    key="validation_with_rppa",
    analysis_name="validation_6_omics",
    out_subdir="validation_with_rppa",
    modalities=("mutation", "cnv", "methylation", "expression", "mirna", "rppa"),
    description="Extension validation cohort that adds RPPA on the strict six-omics overlap.",
)

ANALYSIS_QUEUES: tuple[AnalysisQueueConfig, ...] = (
    PRIMARY_DISCOVERY_QUEUE,
    EXTENSION_VALIDATION_QUEUE,
)

K_RANGE = range(2, 5)
SNF_K = 10
SNF_T = 20
CONSENSUS_REPS = 100
CONSENSUS_FRACTION = 0.8
MIN_CLUSTER_SIZE = 25
PCA_COMPONENT_CAP = 50
SENSITIVITY_SEED_OFFSETS = (-2, -1, 1)
SENSITIVITY_SNF_K_VALUES = (8, 12)


def data_path(data_dir: Path, relative: str | tuple[str, ...]) -> Path:
    candidates = (relative,) if isinstance(relative, str) else relative
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    joined = ", ".join(str(data_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find any expected data file. Tried: {joined}")
