from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stad_subtyping.supervised import CLASSIFIER_EXPORTERS, CLASSIFIER_NAMES, CLASSIFIER_RUNNERS


DEFAULT_EXTENSION_CLASSIFIERS = ("mlp", "lstm", "transformer")
DEFAULT_QUEUE_KEYS = ("primary_discovery", "validation_with_rppa")


def _base_classifier_dir(results_dir: Path, queue_key: str) -> Path:
    base = results_dir / "classifiers" / "elastic_net" / queue_key
    if not base.exists():
        raise FileNotFoundError(f"Missing base classifier cache: {base}")
    return base


def _queue_metadata(summary: pd.DataFrame, queue_key: str) -> tuple[str, str]:
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
    summary_path = results_dir / "classifier_summary.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing classifier summary: {summary_path}")

    existing_summary = pd.read_csv(summary_path, sep="\t")
    summary_rows: list[pd.DataFrame] = []
    for queue_key in queue_keys:
        base_dir = _base_classifier_dir(results_dir, queue_key)
        feature_frame = pd.read_csv(base_dir / "feature_matrix.tsv", sep="\t", index_col=0)
        labels = pd.read_csv(base_dir / "labels.tsv", sep="\t", index_col=0)["subtype"]
        analysis_name, scheme_name = _queue_metadata(existing_summary, queue_key)

        for classifier_name in classifier_names:
            if classifier_name not in CLASSIFIER_RUNNERS:
                raise KeyError(f"Unknown classifier: {classifier_name}")
            out_dir = results_dir / "classifiers" / classifier_name / queue_key
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
    replace_names = set(classifier_names)
    combined = pd.concat(
        [
            existing_summary.loc[~existing_summary["classifier_name"].isin(replace_names)],
            new_summary,
        ],
        ignore_index=True,
    )
    queue_order = {queue_key: idx for idx, queue_key in enumerate(queue_keys)}
    classifier_order = {classifier_name: idx for idx, classifier_name in enumerate(CLASSIFIER_NAMES)}
    combined["_queue_order"] = combined["queue_key"].map(queue_order).fillna(999)
    combined["_classifier_order"] = combined["classifier_name"].map(classifier_order).fillna(999)
    combined = (
        combined.sort_values(["_queue_order", "_classifier_order", "classifier_name"])
        .drop(columns=["_queue_order", "_classifier_order"])
        .reset_index(drop=True)
    )
    combined.to_csv(summary_path, sep="\t", index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run additional supervised classifiers from existing classifier caches.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--classifiers", nargs="+", default=list(DEFAULT_EXTENSION_CLASSIFIERS))
    parser.add_argument("--queues", nargs="+", default=list(DEFAULT_QUEUE_KEYS))
    args = parser.parse_args()
    summary = run_classifier_extension(
        args.results_dir,
        classifier_names=tuple(args.classifiers),
        queue_keys=tuple(args.queues),
        seed=args.seed,
    )
    print(summary.loc[summary["classifier_name"].isin(args.classifiers)].to_string(index=False))


if __name__ == "__main__":
    main()
