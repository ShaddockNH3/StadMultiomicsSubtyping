"""STAD multi-omics subtyping pipeline.

`from stad_subtyping import run_full_pipeline` 仍然可用，但不在这里 import 重依赖：
pandas / numpy 只在真正调用时加载，所以 `--help` 和不碰数据的测试不需要装整套
科学计算栈。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = ["run_full_pipeline"]

# 公开名字 → 所在子模块，第一次访问时才 import
_LAZY_EXPORTS = {
    "run_full_pipeline": "pipeline",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
