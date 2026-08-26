"""验证基线版、梯度版与 MATLAB 真值是否使用同一公共评价网格。

本工具只读取两个 ``main_BOS.py`` 中人工硬编码的 ``sys.argv``，不会修改训练
参数，也不会自动统一网格。它按照两版 renderer 现有的 ``torch.linspace``
定义重建三个坐标轴，并检查：

1. 两版的 ROIsize、ROInum、ROIvoxelsize 完全一致；
2. ``2 * ROIsize == ROInum * ROIvoxelsize``；
3. 实际相邻坐标间距等于 ROIvoxelsize；
4. 网格元数据与 MATLAB 导出的精确 ground truth 一致。

用法：

    cd claudedo/gradient_output
    python validate_common_grid.py
    python validate_common_grid.py --output result/grid_validation.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
AXIS_NAMES = ("x", "y", "z")
ABS_TOL = 5e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-main",
        type=Path,
        default=PROJECT_ROOT / "PYTHON/NIR-BOS/main_BOS.py",
        help="基线版 main_BOS.py",
    )
    parser.add_argument(
        "--gradient-main",
        type=Path,
        default=SCRIPT_DIR / "main_BOS.py",
        help="梯度版 main_BOS.py",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=(PROJECT_ROOT / "MATLAB/Test_data/Phantom 1/140x294x140/"
                 "flow_ground_truth.mat"),
        help="包含 spacing、roi_size、roi_num 的精确 MATLAB ground truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可选 JSON 报告路径",
    )
    return parser.parse_args()


def extract_hardcoded_argv(path: Path) -> list[str]:
    """从源码 AST 中读取 ``sys.argv = [...]`` 的字面量。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到主程序: {path}")

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        is_sys_argv = any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
            and target.attr == "argv"
            for target in node.targets
        )
        if not is_sys_argv:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value):
            raise ValueError(f"{path} 中的 sys.argv 必须是字符串列表字面量")
        assignments.append(value)

    if len(assignments) != 1:
        raise ValueError(
            f"{path} 应有且仅有一个硬编码 sys.argv，实际找到 {len(assignments)} 个")
    return assignments[0]


def option_values(argv: list[str], flag: str) -> list[str]:
    if flag not in argv:
        raise ValueError(f"硬编码 sys.argv 缺少 {flag}")
    start = argv.index(flag) + 1
    values: list[str] = []
    for token in argv[start:]:
        if token.startswith("--"):
            break
        values.append(token)
    if not values:
        raise ValueError(f"硬编码参数 {flag} 没有取值")
    return values


def load_main_grid(path: Path) -> dict[str, Any]:
    argv = extract_hardcoded_argv(path)
    roi_size = np.asarray(
        [float(value) for value in option_values(argv, "--ROIsize")],
        dtype=np.float64,
    )
    roi_num = np.asarray(
        [int(value) for value in option_values(argv, "--ROInum")],
        dtype=np.int64,
    )
    voxel_values = option_values(argv, "--ROIvoxelsize")

    if roi_size.shape != (3,) or roi_num.shape != (3,):
        raise ValueError(f"{path} 的 ROIsize 和 ROInum 都必须恰好包含三个值")
    if len(voxel_values) != 1:
        raise ValueError(f"{path} 的 ROIvoxelsize 必须恰好包含一个值")
    voxel_size = float(voxel_values[0])
    if np.any(roi_size <= 0) or np.any(roi_num < 2) or voxel_size <= 0:
        raise ValueError(f"{path} 包含非正网格参数")

    coordinates = [
        np.linspace(
            -roi_size[index] + voxel_size / 2,
            roi_size[index] - voxel_size / 2,
            int(roi_num[index]),
            dtype=np.float64,
        )
        for index in range(3)
    ]
    actual_spacing = np.asarray(
        [np.diff(axis).mean() for axis in coordinates], dtype=np.float64)

    return {
        "file": str(path.resolve()),
        "roi_size": roi_size,
        "roi_num": roi_num,
        "voxel_size": voxel_size,
        "coordinates": coordinates,
        "actual_spacing": actual_spacing,
    }


def load_ground_truth_grid(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到精确 ground truth: {path}")
    values = sio.loadmat(path)
    missing = [name for name in ("spacing", "roi_size", "roi_num")
               if name not in values]
    if missing:
        raise KeyError(f"{path} 缺少网格元数据: {missing}")
    return {
        "spacing": np.asarray(values["spacing"], dtype=np.float64).reshape(-1),
        "roi_size": np.asarray(values["roi_size"], dtype=np.float64).reshape(-1),
        "roi_num": np.asarray(values["roi_num"], dtype=np.int64).reshape(-1),
    }


def serializable_grid(grid: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": grid["file"],
        "roi_size": grid["roi_size"].tolist(),
        "roi_num": grid["roi_num"].tolist(),
        "voxel_size": grid["voxel_size"],
        "actual_spacing": grid["actual_spacing"].tolist(),
        "axis_ranges": {
            name: [float(values[0]), float(values[-1])]
            for name, values in zip(AXIS_NAMES, grid["coordinates"])
        },
    }


def main() -> None:
    args = parse_args()
    baseline = load_main_grid(args.baseline_main)
    gradient = load_main_grid(args.gradient_main)
    truth = load_ground_truth_grid(args.ground_truth)

    expected_half_size = baseline["roi_num"] * baseline["voxel_size"] / 2
    expected_spacing = np.full(3, baseline["voxel_size"], dtype=np.float64)

    checks = {
        "baseline_and_gradient_roi_size_match": np.allclose(
            baseline["roi_size"], gradient["roi_size"], rtol=0, atol=ABS_TOL),
        "baseline_and_gradient_roi_num_match": np.array_equal(
            baseline["roi_num"], gradient["roi_num"]),
        "baseline_and_gradient_voxel_size_match": np.isclose(
            baseline["voxel_size"], gradient["voxel_size"], rtol=0, atol=ABS_TOL),
        "roi_half_size_matches_num_times_spacing": np.allclose(
            baseline["roi_size"], expected_half_size, rtol=0, atol=ABS_TOL),
        "baseline_actual_spacing_matches_declared": np.allclose(
            baseline["actual_spacing"], expected_spacing, rtol=0, atol=ABS_TOL),
        "gradient_actual_spacing_matches_declared": np.allclose(
            gradient["actual_spacing"], expected_spacing, rtol=0, atol=ABS_TOL),
        "baseline_and_gradient_coordinates_match": all(
            np.array_equal(first, second)
            for first, second in zip(
                baseline["coordinates"], gradient["coordinates"])),
        "ground_truth_roi_size_matches": np.allclose(
            baseline["roi_size"], truth["roi_size"], rtol=0, atol=ABS_TOL),
        "ground_truth_roi_num_matches": np.array_equal(
            baseline["roi_num"], truth["roi_num"]),
        "ground_truth_spacing_matches": np.allclose(
            expected_spacing, truth["spacing"], rtol=0, atol=ABS_TOL),
    }
    passed = all(bool(value) for value in checks.values())

    report = {
        "passed": passed,
        "policy": "validate only; never rewrite training or grid parameters",
        "axis_order": list(AXIS_NAMES),
        "grid_definition": (
            "voxel centers from -ROIsize + spacing/2 to "
            "+ROIsize - spacing/2 using linspace"
        ),
        "baseline": serializable_grid(baseline),
        "gradient": serializable_grid(gradient),
        "ground_truth": {
            "file": str(args.ground_truth.resolve()),
            "roi_size": truth["roi_size"].tolist(),
            "roi_num": truth["roi_num"].tolist(),
            "spacing": truth["spacing"].tolist(),
        },
        "checks": {name: bool(value) for name, value in checks.items()},
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"公共评价网格验证失败: {failed}")


if __name__ == "__main__":
    main()
