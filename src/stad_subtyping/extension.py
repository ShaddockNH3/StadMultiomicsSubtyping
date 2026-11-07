"""在已有分类器缓存上补跑额外分类器（MLP / LSTM / Transformer）。

原来这段逻辑写在 `scripts/run_classifier_extension.py` 里，命令行和实现混在一个文件，
脚本一跑就得先 import pandas。搬到包里之后，脚本只剩参数解析，CLI 也能直接调。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .supervised import CLASSIFIER_EXPORTERS, CLASSIFIER_NAMES, CLASSIFIER_RUNNERS

DEFAULT_EXTENSION_CLASSIFIERS = ("mlp", "lstm", "transformer")
DEFAULT_QUEUE_KEYS = ("primary_discovery", "validation_with_rppa")


def base_classifier_dir(results_dir: Path, queue_key: str) -> Path:
    """扩展分类器复用 elastic_net 的划分与特征矩阵，缺了就直接报清楚。"""
    base = results_dir / "classifiers" / "elastic_net" / queue_key
    if not base.exists():
        raise FileNotFoundError(f"缺少基础分类器缓存: {base}")
    return base


def queue_metadata(summary: pd.DataFrame, queue_key: str) -> tuple[str, str]:
    rows = summary.loc[summary["queue_key"] == queue_key]
    if rows.empty:
        return queue_key, "unknown_scheme"
    row = rows.iloc[0]
    return str(row["analysis_name"]), str(row["label_scheme_name"])


def run_classifier_extension(
    results_dir: Path,
    classifier_names: tuple[str, ...] = DEFAULT_EXTENSION_CLASSIFIERS,
    queue_keys: tuple[str, ...] = DEFAULT_QUEUE_KEYS,
    seed: int = 2026,
) -> pd.DataFrame:
    """跑完把结果写回 classifier_summary.tsv，并按队列/分类器顺序重排。"""
    summary_path = Path(results_dir) / "classifier_summary.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少分类器汇总表: {summary_path}")

    unknown = [name for name in classifier_names if name not in CLASSIFIER_RUNNERS]
    if unknown:
        raise KeyError(f"未知分类器: {', '.join(unknown)}")

    existing_summary = pd.read_csv(summary_path, sep="\t")
    summary_rows: list[pd.DataFrame] = []

    for queue_key in queue_keys:
        base_dir = base_classifier_dir(Path(results_dir), queue_key)
        feature_frame = pd.read_csv(base_dir / "feature_matrix.tsv", sep="\t", index_col=0)
        labels = pd.read_csv(base_dir / "labels.tsv", sep="\t", index_col=0)["subtype"]
        analysis_name, scheme_name = queue_metadata(existing_summary, queue_key)

        for classifier_name in classifier_names:
            out_dir = Path(results_dir) / "classifiers" / classifier_name / queue_key
            cv_result = CLASSIFIER_RUNNERS[classifier_name](
                feature_frame,
                labels,
                random_state=seed,
            )
            summary_rows.append(
                CLASSIFIER_EXPORTERS[classifier_name](
                    out_dir,
                    feature_frame,
                    labels,
                    cv_result,
                    queue_key,
                    analysis_name,
                    scheme_name,
                )
            )

    new_summary = pd.concat(summary_rows, ignore_index=True)
    replaced = set(classifier_names)
    combined = pd.concat(
        [
            existing_summary.loc[~existing_summary["classifier_name"].isin(replaced)],
            new_summary,
        ],
        ignore_index=True,
    )

    combined = _sorted_summary(combined, queue_keys)
    combined.to_csv(summary_path, sep="\t", index=False)
    return combined


def _sorted_summary(combined: pd.DataFrame, queue_keys: tuple[str, ...]) -> pd.DataFrame:
    """按“队列顺序 → 分类器固定顺序”重排，保证汇总表每次结构一致。"""
    queue_order = {key: index for index, key in enumerate(queue_keys)}
    classifier_order = {name: index for index, name in enumerate(CLASSIFIER_NAMES)}

    combined = combined.copy()
    combined["_queue_order"] = combined["queue_key"].map(queue_order).fillna(999)
    combined["_classifier_order"] = combined["classifier_name"].map(classifier_order).fillna(999)
    return (
        combined.sort_values(["_queue_order", "_classifier_order", "classifier_name"])
        .drop(columns=["_queue_order", "_classifier_order"])
        .reset_index(drop=True)
    )
