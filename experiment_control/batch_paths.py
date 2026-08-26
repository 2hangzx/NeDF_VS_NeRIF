"""正式实验包的批次标识与集中保存路径。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PACKAGE_ROOT / "experiments"
BATCH_ENV = "NIR_BOS_BATCH_ID"
CHECKPOINT_ENV = "NIR_BOS_CHECKPOINT"
EXTENSION_ENV = "NIR_BOS_EXTENSION"
VALID_ROUTES = {"baseline", "gradient_output"}
_BATCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_batch_id(batch_id: str) -> str:
    if not _BATCH_PATTERN.fullmatch(batch_id):
        raise ValueError(
            "批次号必须以字母或数字开头，只能包含字母、数字、点、下划线、"
            "连字符，且长度不超过 64")
    return batch_id


def batch_root(batch_id: str) -> Path:
    return EXPERIMENTS_ROOT / validate_batch_id(batch_id)


def batch_manifest_path(batch_id: str) -> Path:
    return batch_root(batch_id) / "batch_manifest.json"


def load_batch_manifest(batch_id: str) -> dict[str, Any]:
    path = batch_manifest_path(batch_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"批次不存在或尚未创建: {batch_id}; 请先运行 experiment.py create")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("batch_id") != batch_id:
        raise ValueError(f"批次清单 ID 与目录不一致: {path}")
    return data


def route_workspace(batch_id: str, route: str) -> Path:
    if route not in VALID_ROUTES:
        raise ValueError(f"未知路线: {route}")
    return batch_root(batch_id) / route


def comparison_directory(batch_id: str, comparison: str) -> Path:
    if comparison not in {"flow", "gradient"}:
        raise ValueError(f"未知比较类型: {comparison}")
    return batch_root(batch_id) / "comparisons" / comparison


def require_batch_workspace(route: str) -> Path:
    """由环境变量选择已创建批次；训练入口只调用本函数获取 workspace。"""
    batch_id = os.environ.get(BATCH_ENV, "").strip()
    if not batch_id:
        raise RuntimeError(
            f"缺少环境变量 {BATCH_ENV}。请通过 experiment_control/experiment.py "
            "启动正式训练，不要直接运行未指定批次的 main_BOS.py。")
    manifest = load_batch_manifest(validate_batch_id(batch_id))
    declared = manifest.get("paths", {}).get(route)
    expected = route_workspace(batch_id, route).relative_to(PACKAGE_ROOT).as_posix()
    if declared != expected:
        raise ValueError(
            f"批次清单中的 {route} 路径不合法: declared={declared}, expected={expected}")
    return route_workspace(batch_id, route)
