from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stad_subtyping.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STAD multi-omics subtyping pipeline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_full_pipeline(args.data_dir, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
