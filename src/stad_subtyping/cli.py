"""命令行入口。重依赖都在各个子命令里 import，`--help` 不用装 pandas 也能跑。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stad_subtyping",
        description="TCGA-STAD 多组学分型：跑全流程，或在已有缓存上补跑分类器",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pipeline = sub.add_parser("pipeline", help="从原始多组学数据跑到完整结果")
    pipeline.add_argument("--data-dir", type=Path, required=True, help="数据目录")
    pipeline.add_argument("--out-dir", type=Path, required=True, help="结果目录")
    pipeline.add_argument("--seed", type=int, default=2026)

    classifier = sub.add_parser("classifier", help="复用已有缓存，补跑额外分类器")
    classifier.add_argument("--results-dir", type=Path, default=ROOT / "results")
    classifier.add_argument("--seed", type=int, default=2026)
    classifier.add_argument("--classifiers", nargs="+", default=None)
    classifier.add_argument("--queues", nargs="+", default=None)

    return parser


def _run_pipeline(args: argparse.Namespace) -> int:
    from .pipeline import run_full_pipeline  # pylint: disable=import-outside-toplevel

    run_full_pipeline(args.data_dir, args.out_dir, seed=args.seed)
    return 0


def _run_classifier(args: argparse.Namespace) -> int:
    from .extension import (  # pylint: disable=import-outside-toplevel
        DEFAULT_EXTENSION_CLASSIFIERS,
        DEFAULT_QUEUE_KEYS,
        run_classifier_extension,
    )

    classifiers = tuple(args.classifiers or DEFAULT_EXTENSION_CLASSIFIERS)
    queues = tuple(args.queues or DEFAULT_QUEUE_KEYS)

    summary = run_classifier_extension(
        args.results_dir,
        classifier_names=classifiers,
        queue_keys=queues,
        seed=args.seed,
    )
    print(summary.loc[summary["classifier_name"].isin(classifiers)].to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"pipeline": _run_pipeline, "classifier": _run_classifier}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
