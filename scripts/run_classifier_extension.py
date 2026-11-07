"""补跑分类器入口。实现搬到 stad_subtyping.extension，这里只转发参数。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stad_subtyping.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["classifier", *sys.argv[1:]]))
