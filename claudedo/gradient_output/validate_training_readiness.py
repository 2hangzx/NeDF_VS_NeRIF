"""只读验证批次声明与两版训练母版，不修改任何硬编码参数。

本工具检查两版 ``main_BOS.py`` 中人工维护的配置是否满足：共同数据、共同网络
主干、共同训练预算和采样参数、从 scratch 开始、使用互不覆盖的新 workspace。
它还检查两版是否保留受控 checkpoint 恢复和跨批次扩展入口，并报告基线训练
有限差分步长与公共评价网格差分步长，防止混淆两种尺度。

用法：

    cd claudedo/gradient_output
    python validate_training_readiness.py --batch-id strict_control_run_001
    python validate_training_readiness.py --batch-id strict_control_run_001 --output training_readiness.json

workspace 已经开始写入后，如需复查其余配置，可显式加入
``--allow-existing-workspaces``。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import sys
from typing import Any

from validate_common_grid import extract_hardcoded_argv, option_values


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiment_control.batch_paths import (
    batch_root,
    load_batch_manifest,
    route_workspace,
    validate_batch_id,
)

BASELINE_MAIN = PROJECT_ROOT / "PYTHON/NIR-BOS/main_BOS.py"
GRADIENT_MAIN = SCRIPT_DIR / "main_BOS.py"
FORMAL_WORKSPACE_LITERAL = "__FORMAL_BATCH_WORKSPACE__"
FOURIER_INPUT_DIM = 39  # input_dim=3, multires=6: 3 * (2 * 6 + 1)


def declared_common_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """把批次声明转换成与入口源码抽取结果相同的公共字段。"""
    training = profile["training"]
    grid = profile["evaluation_grid"]
    return {
        "checkpoint": training["checkpoint"],
        "seed": training["seed"],
        "iterations": training["iterations"],
        "learning_rate": training["learning_rate"],
        "num_rays": training["num_rays"],
        "max_steps": training["max_steps"],
        "bound": training["bound"],
        "scale": training["scale"],
        "dt_gamma": training["dt_gamma"],
        "roi_size": grid["roi_size"],
        "roi_num": grid["roi_num"],
        "roi_voxel_size": grid["roi_voxel_size"],
        "valbound": grid["valbound"],
        "flags": {
            "fp16": training["fp16"],
            "cuda_ray": training["cuda_ray"],
            "maskflag": training["maskflag"],
            "test": False,
        },
        "model": {
            "encoding": training["encoding"],
            "num_layers": training["num_layers"],
            "hidden_dim": training["hidden_dim"],
            "density_scale": training["density_scale"],
        },
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读验证基线版与梯度版的批次声明及训练母版。")
    parser.add_argument("--baseline-main", type=Path, default=BASELINE_MAIN)
    parser.add_argument("--gradient-main", type=Path, default=GRADIENT_MAIN)
    parser.add_argument("--batch-id", required=True, help="已由控制程序创建的批次号")
    parser.add_argument("--output", type=Path, help="可选 JSON 报告路径")
    parser.add_argument(
        "--allow-existing-workspaces",
        action="store_true",
        help="训练开始后复查配置时，允许 workspace 已存在",
    )
    return parser.parse_args()


def scalar_option(argv: list[str], flag: str, cast: Any) -> Any:
    values = option_values(argv, flag)
    if len(values) != 1:
        raise ValueError(f"{flag} 必须恰好有一个值，实际为 {values}")
    return cast(values[0])


def literal_model_options(path: Path) -> dict[str, Any]:
    """读取唯一 NeRFNetwork(...) 调用中的显式字面量参数。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "NeRFNetwork"
    ]
    if len(calls) != 1:
        raise ValueError(f"{path} 应有且仅有一个 NeRFNetwork 调用")

    result: dict[str, Any] = {}
    for keyword in calls[0].keywords:
        if keyword.arg in {"encoding", "num_layers", "hidden_dim", "density_scale"}:
            try:
                result[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"{path} 的 {keyword.arg} 必须显式写成字面量") from error

    required = {"encoding", "num_layers", "hidden_dim", "density_scale"}
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"{path} 的 NeRFNetwork 缺少显式参数: {missing}")
    return result


def computation_fingerprints(path: Path) -> dict[str, str]:
    """比较两版入口中优化器和学习率调度器的 AST。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, str] = {}
    for wanted in ("optimizer", "scheduler"):
        values = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == wanted
                   for target in node.targets):
                values.append(ast.dump(node.value, include_attributes=False))
        if len(values) != 1:
            raise ValueError(f"{path} 应有且仅有一个生效的 {wanted} 赋值")
        result[wanted] = values[0]
    return result


def load_profile(path: Path, route: str) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    argv = extract_hardcoded_argv(path)
    model = literal_model_options(path)
    main_dir = path.resolve().parent

    roi_size = [float(value) for value in option_values(argv, "--ROIsize")]
    roi_num = [int(value) for value in option_values(argv, "--ROInum")]
    workspace_text = scalar_option(argv, "--workspace", str)
    dataset_text = argv[1]
    layers = int(model["num_layers"])
    width = int(model["hidden_dim"])
    output_dim = 1 if route == "baseline" else 3
    parameter_count = (
        FOURIER_INPUT_DIM * width
        + max(layers - 2, 0) * width * width
        + width * output_dim
        + output_dim
        + (3 if route == "gradient_output" else 0)
    )

    return {
        "route": route,
        "main_file": str(path.resolve()),
        "dataset_argument": dataset_text,
        "dataset_resolved": str((main_dir / dataset_text).resolve()),
        "dataset_exists": (main_dir / dataset_text).is_dir(),
        "workspace_argument": workspace_text,
        "workspace_resolved": str((main_dir / workspace_text).resolve()),
        "workspace_exists": (main_dir / workspace_text).exists(),
        "checkpoint": scalar_option(argv, "--ckpt", str),
        "runtime_checkpoint_override_present": all(token in source for token in (
            "CHECKPOINT_ENV",
            "checkpoint_override",
            "opt.ckpt = checkpoint_override",
        )),
        "runtime_extension_override_present": all(token in source for token in (
            "EXTENSION_ENV",
            "extension_mode",
            "restart_cosine_scheduler_for_extension",
            "record_extension",
        )),
        "seed": scalar_option(argv, "--seed", int),
        "iterations": scalar_option(argv, "--iters", int),
        "learning_rate": scalar_option(argv, "--lr", float),
        "num_rays": scalar_option(argv, "--num_rays", int),
        "max_steps": scalar_option(argv, "--max_steps", int),
        "bound": scalar_option(argv, "--bound", float),
        "scale": scalar_option(argv, "--scale", float),
        "dt_gamma": scalar_option(argv, "--dt_gamma", float),
        "roi_size": roi_size,
        "roi_num": roi_num,
        "roi_voxel_size": scalar_option(argv, "--ROIvoxelsize", float),
        "valbound": [float(value) for value in option_values(argv, "--valbound")],
        "flags": {
            "fp16": "--fp16" in argv,
            "cuda_ray": "--cuda_ray" in argv,
            "maskflag": "--maskflag" in argv,
            "test": "--test" in argv,
        },
        "model": model,
        "trainable_parameter_count_calculated": parameter_count,
        "computation_fingerprints": computation_fingerprints(path),
    }


def same(first: dict[str, Any], second: dict[str, Any], key: str) -> bool:
    left, right = first[key], second[key]
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    return left == right


def matches_declared_profile(
        profile: dict[str, Any], declared: dict[str, Any],
) -> bool:
    return (
        all(profile[key] == value for key, value in declared.items())
        and profile["workspace_argument"] == FORMAL_WORKSPACE_LITERAL
    )


def main() -> None:
    args = parse_args()
    batch_id = validate_batch_id(args.batch_id)
    batch_manifest = load_batch_manifest(batch_id)
    declared_profile_path = batch_root(batch_id) / "declared_profile.json"
    declared_profile = json.loads(
        declared_profile_path.read_text(encoding="utf-8"))
    expected_common = declared_common_profile(declared_profile)
    baseline = load_profile(args.baseline_main, "baseline")
    gradient = load_profile(args.gradient_main, "gradient_output")
    expected_workspaces = {
        route: str(route_workspace(batch_id, route).resolve())
        for route in ("baseline", "gradient_output")
    }
    for profile in (baseline, gradient):
        runtime_workspace = route_workspace(batch_id, profile["route"])
        profile["workspace_resolved"] = str(runtime_workspace.resolve())
        profile["workspace_exists"] = runtime_workspace.exists()
        profile["batch_id"] = batch_id

    equality_fields = (
        "seed", "iterations", "learning_rate", "num_rays", "max_steps",
        "bound", "scale", "dt_gamma", "roi_size", "roi_num",
        "roi_voxel_size", "valbound", "flags", "model",
        "computation_fingerprints",
    )
    checks = {
        "same_physical_dataset": (
            baseline["dataset_resolved"] == gradient["dataset_resolved"]),
        "baseline_dataset_exists": baseline["dataset_exists"],
        "gradient_dataset_exists": gradient["dataset_exists"],
        **{
            f"same_{field}": same(baseline, gradient, field)
            for field in equality_fields
        },
        "both_train_from_scratch": (
            baseline["checkpoint"] == gradient["checkpoint"] == "scratch"),
        "both_expose_controlled_checkpoint_resume": (
            baseline["runtime_checkpoint_override_present"]
            and gradient["runtime_checkpoint_override_present"]),
        "both_expose_controlled_budget_extension": (
            baseline["runtime_extension_override_present"]
            and gradient["runtime_extension_override_present"]),
        "both_are_training_profiles": (
            not baseline["flags"]["test"] and not gradient["flags"]["test"]),
        "baseline_matches_declared_profile": (
            matches_declared_profile(baseline, expected_common)),
        "gradient_matches_declared_profile": (
            matches_declared_profile(gradient, expected_common)),
        "workspaces_are_distinct": (
            baseline["workspace_resolved"] != gradient["workspace_resolved"]),
        "batch_profile_matches_declared_profile": (
            batch_manifest.get("profile_id")
            == declared_profile.get("profile_id")),
        "declares_controlled_extension_policy": (
            declared_profile.get("extension_policy", {}) == {
                "mode": "cosine_restart_over_remaining_iterations",
                "child_batch_required": True,
                "restart_learning_rate_source": "training.learning_rate",
                "eta_min": 1e-6,
                "parent_batch_is_immutable": True,
            }),
        "declared_nominal_budget_matches": (
            declared_profile.get("nominal_budget", {}).get("total_rays")
            == baseline["iterations"] * baseline["num_rays"]
            and declared_profile.get("nominal_budget", {}).get(
                "max_ray_samples")
            == baseline["iterations"] * baseline["num_rays"]
            * baseline["max_steps"]),
        "baseline_workspace_identifies_route": (
            Path(baseline["workspace_resolved"]).name == "baseline"),
        "gradient_workspace_identifies_route": (
            Path(gradient["workspace_resolved"]).name == "gradient_output"),
        "workspaces_are_new": (
            args.allow_existing_workspaces
            or (not baseline["workspace_exists"]
                and not gradient["workspace_exists"])),
    }

    iterations = baseline["iterations"]
    rays = baseline["num_rays"]
    max_steps = baseline["max_steps"]
    training_fd_step = baseline["bound"] / max_steps
    evaluation_fd_step = baseline["roi_voxel_size"]
    report = {
        "passed": all(bool(value) for value in checks.values()),
        "policy": (
            "validate only; parameters remain manually hardcoded in both main_BOS.py files"),
        "experiment_group": declared_profile.get("profile_id"),
        "batch_id": batch_id,
        "declared_common_profile": expected_common,
        "declared_workspaces": expected_workspaces,
        "baseline": baseline,
        "gradient_output": gradient,
        "budget": {
            "iterations": iterations,
            "rays_per_iteration": rays,
            "max_steps_per_ray": max_steps,
            "nominal_total_rays": iterations * rays,
            "nominal_max_ray_samples": iterations * rays * max_steps,
            "note": "upper-bound proxy; occupancy and ray termination change actual samples",
        },
        "finite_difference_steps": {
            "baseline_training_loss_fd": training_fd_step,
            "public_grid_comparison_fd": evaluation_fd_step,
            "decoupled_by_existing_roles": not math.isclose(
                training_fd_step, evaluation_fd_step, rel_tol=0, abs_tol=1e-12),
            "note": (
                "training loss keeps delta_r = bound / max_steps; comparison uses "
                "ROIvoxelsize on the exported public grid"),
        },
        "parameter_count_note": (
            "calculated for reporting only; profiles do not declare or validate "
            "parameter counts. The 1-output versus 3-output heads and gradient "
            "scale explain the route difference"),
        "checks": {name: bool(value) for name, value in checks.items()},
    }

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if not report["passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"批次声明与训练母版验证失败: {failed}")


if __name__ == "__main__":
    main()
