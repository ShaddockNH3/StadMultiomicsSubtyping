"""全流程入口。参数与实现在 stad_subtyping.cli，这里只做转发。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stad_subtyping.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["pipeline", *sys.argv[1:]]))
