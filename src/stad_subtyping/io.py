from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CLINICAL_PATH,
    METHYLATION_PROBE_MAP,
    MODALITIES,
    SURVIVAL_PATH,
    ModalityConfig,
    data_path,
)


@dataclass
class LoadedTables:
    omics: dict[str, pd.DataFrame]
    clinical: pd.DataFrame
    survival: pd.DataFrame
    probe_map: pd.DataFrame
    qc: dict


def tcga_patient_id(sample: str) -> str | None:
    if not isinstance(sample, str) or not sample.startswith("TCGA-"):
        return None
    return sample[:12]


def is_primary_tumor(sample: str) -> bool:
    return isinstance(sample, str) and sample.startswith("TCGA-") and len(sample) >= 15 and sample[13:15] == "01"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def collect_primary_tumor_patients(data_dir: Path, modality_names: list[str]) -> dict[str, set[str]]:
    sample_sets: dict[str, set[str]] = {}
    for name in modality_names:
        cfg = MODALITIES[name]
        path = data_path(data_dir, cfg.path)
        with _open_text(path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        sample_sets[name] = {
            tcga_patient_id(sample)
            for sample in header[1:]
            if is_primary_tumor(sample) and tcga_patient_id(sample) is not None
        }

    clinical_path = data_path(data_dir, CLINICAL_PATH)
    clinical_df = pd.read_csv(clinical_path, sep="\t", dtype=str)
    sample_sets["clinical"] = {
        tcga_patient_id(sample)
        for sample in clinical_df["sampleID"].astype(str)
        if is_primary_tumor(sample) and tcga_patient_id(sample) is not None
    }

    survival_path = data_path(data_dir, SURVIVAL_PATH)
    survival_df = pd.read_csv(survival_path, sep="\t", dtype=str)
    sample_sets["survival"] = {
        tcga_patient_id(sample)
        for sample in survival_df["sample"].astype(str)
        if is_primary_tumor(sample) and tcga_patient_id(sample) is not None
    }
    return sample_sets


def load_probe_map(data_dir: Path) -> pd.DataFrame:
    path = data_path(data_dir, METHYLATION_PROBE_MAP)
    probe_map = pd.read_csv(path, sep="\t", dtype=str)
    return probe_map.rename(columns={"#id": "feature_id"})


def _build_usecols(path: Path, selected_patients: list[str]) -> tuple[list[str], dict[str, str]]:
    with _open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
    keep_columns = [header[0]]
    rename_map: dict[str, str] = {}
    seen: set[str] = set()
    for sample in header[1:]:
        if not is_primary_tumor(sample):
            continue
        patient = tcga_patient_id(sample)
        if patient not in selected_patients or patient in seen or patient is None:
            continue
        keep_columns.append(sample)
        rename_map[sample] = patient
        seen.add(patient)
    return keep_columns, rename_map


def _mad(values: np.ndarray) -> float:
    med = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - med)))


def _fill_nan_with_median(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values)
    if np.isnan(med):
        med = 0.0
    return np.where(np.isnan(values), med, values)


def load_regular_modality(
    data_dir: Path,
    cfg: ModalityConfig,
    selected_patients: list[str],
) -> tuple[pd.DataFrame, dict]:
    path = data_path(data_dir, cfg.path)
    usecols, rename_map = _build_usecols(path, selected_patients)
    df = pd.read_csv(path, sep="\t", compression="infer", usecols=usecols)
    df = df.rename(columns=rename_map).rename(columns={cfg.feature_col: "feature_id"})
    df["feature_id"] = df["feature_id"].astype(str)
    df = df.drop_duplicates(subset="feature_id").set_index("feature_id")
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df = df.reindex(columns=selected_patients)

    qc = {
        "raw_features": int(df.shape[0]),
        "samples": int(df.shape[1]),
    }
    return df, qc


def load_methylation_modality(
    data_dir: Path,
    cfg: ModalityConfig,
    selected_patients: list[str],
) -> tuple[pd.DataFrame, dict]:
    path = data_path(data_dir, cfg.path)
    usecols, rename_map = _build_usecols(path, selected_patients)
    chunk_iter = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=5000,
    )

    top_n = cfg.top_mad_features or 5000
    selected_rows: list[tuple[float, str, np.ndarray]] = []
    total_rows = 0
    kept_rows = 0
    for chunk in chunk_iter:
        total_rows += len(chunk)
        chunk = chunk.rename(columns=rename_map).rename(columns={cfg.feature_col: "feature_id"})
        feature_ids = chunk["feature_id"].astype(str).to_numpy()
        values = chunk.drop(columns=["feature_id"]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        missing_rate = np.isnan(values).mean(axis=1)
        valid_mask = missing_rate <= (cfg.max_missing_rate or 0.2)
        feature_ids = feature_ids[valid_mask]
        values = values[valid_mask]
        kept_rows += int(valid_mask.sum())
        for feature_id, row in zip(feature_ids, values, strict=False):
            row = _fill_nan_with_median(row.astype(np.float32, copy=False))
            mad = _mad(row)
            selected_rows.append((mad, feature_id, row))

    selected_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_rows = selected_rows[:top_n]
    feature_ids = [item[1] for item in top_rows]
    matrix = np.vstack([item[2] for item in top_rows]).astype(np.float32)
    df = pd.DataFrame(matrix, index=feature_ids, columns=selected_patients)
    qc = {
        "raw_features": int(total_rows),
        "missing_filtered_features": int(kept_rows),
        "selected_features": int(df.shape[0]),
        "samples": int(df.shape[1]),
    }
    return df, qc


def load_clinical(data_dir: Path, selected_patients: list[str]) -> tuple[pd.DataFrame, dict]:
    path = data_path(data_dir, CLINICAL_PATH)
    clinical = pd.read_csv(path, sep="\t", dtype=str)
    clinical = clinical[clinical["sampleID"].map(is_primary_tumor)].copy()
    clinical["patient_id"] = clinical["sampleID"].map(tcga_patient_id)
    duplicate_counts = clinical["patient_id"].value_counts()
    duplicate_patients = duplicate_counts[duplicate_counts > 1].index.tolist()
    clinical = clinical.drop_duplicates(subset="patient_id", keep="first").set_index("patient_id")
    clinical = clinical.reindex(selected_patients)

    rename_map = {
        "CDE_ID_3226963": "msi_status",
        "age_at_initial_pathologic_diagnosis": "age",
        "gender": "gender",
        "histological_type": "histological_type",
        "neoplasm_histologic_grade": "grade",
        "pathologic_stage": "pathologic_stage",
        "vital_status": "vital_status",
        "person_neoplasm_cancer_status": "cancer_status",
    }
    clinical = clinical.rename(columns=rename_map)
    keep = ["sampleID", *rename_map.values()]
    clinical = clinical[keep]
    qc = {
        "samples": int(clinical.shape[0]),
        "duplicate_patients_removed": duplicate_patients,
    }
    return clinical, qc


def load_survival(data_dir: Path, selected_patients: list[str]) -> tuple[pd.DataFrame, dict]:
    path = data_path(data_dir, SURVIVAL_PATH)
    survival = pd.read_csv(path, sep="\t", dtype=str)
    survival = survival[survival["sample"].map(is_primary_tumor)].copy()
    survival["patient_id"] = survival["sample"].map(tcga_patient_id)
    duplicate_counts = survival["patient_id"].value_counts()
    duplicate_patients = duplicate_counts[duplicate_counts > 1].index.tolist()
    survival = survival.drop_duplicates(subset="patient_id", keep="first").set_index("patient_id")
    survival = survival.reindex(selected_patients)

    numeric_cols = ["OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time"]
    for col in numeric_cols:
        survival[col] = pd.to_numeric(survival[col], errors="coerce")
    qc = {
        "samples": int(survival.shape[0]),
        "duplicate_patients_removed": duplicate_patients,
    }
    return survival, qc


def load_tables_for_analysis(
    data_dir: Path,
    modality_names: list[str],
    required_patients: list[str] | None = None,
) -> LoadedTables:
    patient_sets = collect_primary_tumor_patients(data_dir, modality_names)
    selected_patients = sorted(
        set.intersection(*(patient_sets[name] for name in modality_names), patient_sets["clinical"], patient_sets["survival"])
    )
    if required_patients is not None:
        available_patients = set(selected_patients)
        selected_patients = [patient for patient in required_patients if patient in available_patients]

    omics: dict[str, pd.DataFrame] = {}
    qc: dict = {"sample_sets": {name: len(patient_sets[name]) for name in patient_sets}, "analysis_patients": len(selected_patients)}

    for name in modality_names:
        cfg = MODALITIES[name]
        if name == "methylation":
            df, modality_qc = load_methylation_modality(data_dir, cfg, selected_patients)
        else:
            df, modality_qc = load_regular_modality(data_dir, cfg, selected_patients)
        omics[name] = df
        qc[name] = modality_qc

    clinical, clinical_qc = load_clinical(data_dir, selected_patients)
    survival, survival_qc = load_survival(data_dir, selected_patients)
    qc["clinical"] = clinical_qc
    qc["survival"] = survival_qc
    probe_map = load_probe_map(data_dir)
    return LoadedTables(omics=omics, clinical=clinical, survival=survival, probe_map=probe_map, qc=qc)


def write_qc_json(path: Path, qc: dict) -> None:
    path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
