"""正式实验批次的创建、检查、训练、比较和状态查询入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from batch_paths import (
    BATCH_ENV,
    CHECKPOINT_ENV,
    EXTENSION_ENV,
    PACKAGE_ROOT,
    batch_manifest_path,
    batch_root,
    comparison_directory,
    load_batch_manifest,
    route_workspace,
    validate_batch_id,
)


CONTROL_DIR = Path(__file__).resolve().parent
PROFILES_DIR = CONTROL_DIR / "profiles"
BASELINE_DIR = PACKAGE_ROOT / "PYTHON/NIR-BOS"
GRADIENT_DIR = PACKAGE_ROOT / "claudedo/gradient_output"
MATLAB_DATA = PACKAGE_ROOT / "MATLAB/Test_data/Phantom 1"
PYTHON_DATA = PACKAGE_ROOT / "PYTHON/NIR-BOS/data/Phantom 1"
FULL_CHECKPOINT_PATTERN = re.compile(r"^ngp_ep(?P<epoch>\d+)\.pth$")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_batch(batch_id: str, updater: Any) -> dict[str, Any]:
    path = batch_manifest_path(batch_id)
    data = load_batch_manifest(batch_id)
    updater(data)
    data["updated_at"] = now()
    write_json_atomic(path, data)
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_env(
        batch_id: str | None = None,
        checkpoint: Path | None = None,
        extension: bool = False,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # 控制器必须显式决定 scratch 或 resume，不能继承用户终端中的残留变量。
    environment.pop(BATCH_ENV, None)
    environment.pop(CHECKPOINT_ENV, None)
    environment.pop(EXTENSION_ENV, None)
    if batch_id:
        environment[BATCH_ENV] = batch_id
    if checkpoint:
        environment[CHECKPOINT_ENV] = str(checkpoint)
    if extension:
        environment[EXTENSION_ENV] = "1"
    return environment


def run(
        command: list[str],
        cwd: Path,
        batch_id: str | None = None,
        checkpoint: Path | None = None,
        extension: bool = False,
) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n[run] cwd={cwd}\n[run] {printable}", flush=True)
    subprocess.run(
        command,
        cwd=cwd,
        env=command_env(batch_id, checkpoint, extension),
        check=True,
    )


def resolve_resume_checkpoint(workspace: Path, request: str) -> Path:
    checkpoint_dir = workspace / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"路线不存在 checkpoint 目录: {checkpoint_dir}")

    if request == "latest":
        candidates = []
        for path in checkpoint_dir.glob("ngp_ep*.pth"):
            match = FULL_CHECKPOINT_PATTERN.fullmatch(path.name)
            if match and path.is_file():
                candidates.append((int(match.group("epoch")), path))
        if not candidates:
            raise FileNotFoundError(
                f"没有可恢复的完整 epoch checkpoint: {checkpoint_dir}")
        checkpoint = max(candidates, key=lambda item: item[0])[1]
    else:
        if Path(request).name != request or not FULL_CHECKPOINT_PATTERN.fullmatch(
                request):
            raise ValueError(
                "--checkpoint 只能是 latest 或同一路线 checkpoints/ 下的 "
                "ngp_epXXXX.pth 文件名")
        checkpoint = checkpoint_dir / request

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"指定 checkpoint 不存在: {checkpoint}")
    if checkpoint.parent != checkpoint_dir.resolve():
        raise ValueError(f"checkpoint 必须位于当前路线目录内: {checkpoint}")
    return checkpoint


def load_declared_profile(batch_id: str) -> dict[str, Any]:
    path = batch_root(batch_id) / "declared_profile.json"
    if not path.is_file():
        raise FileNotFoundError(f"批次缺少 declared_profile.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_completed_route_manifest(batch_id: str, route: str) -> dict[str, Any]:
    batch = load_batch_manifest(batch_id)
    if batch.get("routes", {}).get(route, {}).get("status") != "completed":
        raise RuntimeError(f"父批次 {batch_id} 的 {route} 尚未完整结束")

    path = route_workspace(
        batch_id, route) / "results/experiment_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"父批次缺少已完成实验清单: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("route") != route:
        raise ValueError(f"父批次实验清单状态或路线不符: {path}")
    return manifest


def source_hash_with_suffix(
        manifest: dict[str, Any], suffix: str,
) -> str | None:
    normalized_suffix = suffix.replace("\\", "/")
    matches = [
        digest for path, digest in manifest.get("source_sha256", {}).items()
        if str(path).replace("\\", "/").endswith(normalized_suffix)
    ]
    if len(matches) > 1:
        raise ValueError(f"父批次源码清单中 {suffix} 出现多次")
    return matches[0] if matches else None


def verify_extension_source(
        parent_batch_id: str,
        child_batch_id: str,
        route: str,
        checkpoint_request: str,
) -> tuple[Path, dict[str, Any]]:
    """核验已完成父批次，并生成可审计的跨批次扩展记录。"""
    if parent_batch_id == child_batch_id:
        raise ValueError("扩展训练必须写入新子批次，父批次和子批次不能相同")

    parent_manifest = load_completed_route_manifest(parent_batch_id, route)
    parent_workspace = route_workspace(parent_batch_id, route)
    checkpoint = resolve_resume_checkpoint(parent_workspace, checkpoint_request)

    artifact_matches = [
        record for record in parent_manifest.get(
            "artifacts", {}).get("checkpoints", [])
        if record.get("name") == checkpoint.name
    ]
    if len(artifact_matches) != 1:
        raise ValueError(
            f"父批次完成清单未唯一登记 checkpoint: {checkpoint.name}")
    artifact = artifact_matches[0]
    actual_hash = file_sha256(checkpoint)
    if not artifact.get("sha256") or artifact["sha256"] != actual_hash:
        raise ValueError(f"父批次 checkpoint 哈希不匹配: {checkpoint}")

    # 延迟导入 torch 相关模块，使 create/status 等轻量命令无需先加载 torch。
    from training_extension import inspect_full_checkpoint

    checkpoint_state = inspect_full_checkpoint(checkpoint)
    child_profile = load_declared_profile(child_batch_id)
    training = child_profile.get("training", {})
    target_iterations = int(training.get("iterations", 0))
    restart_learning_rate = float(training.get("learning_rate", 0))
    if target_iterations <= checkpoint_state["global_step"]:
        raise ValueError(
            "子批次目标总迭代数必须大于父 checkpoint 的 global_step: "
            f"target={target_iterations}, checkpoint="
            f"{checkpoint_state['global_step']}")
    if restart_learning_rate <= 0:
        raise ValueError("子批次声明的 learning_rate 必须大于 0")

    parent_options = parent_manifest.get("options", {})
    grid = child_profile.get("evaluation_grid", {})
    immutable_option_pairs = {
        "seed": training.get("seed"),
        "bound": training.get("bound"),
        "scale": training.get("scale"),
        "dt_gamma": training.get("dt_gamma"),
        "num_rays": training.get("num_rays"),
        "max_steps": training.get("max_steps"),
        "fp16": training.get("fp16"),
        "cuda_ray": training.get("cuda_ray"),
        "maskflag": training.get("maskflag"),
        "ROIsize": grid.get("roi_size"),
        "ROInum": grid.get("roi_num"),
        "ROIvoxelsize": grid.get("roi_voxel_size"),
        "valbound": grid.get("valbound"),
    }
    option_mismatches = [
        f"{key}: parent={parent_options.get(key)!r}, "
        f"child={expected!r}"
        for key, expected in immutable_option_pairs.items()
        if parent_options.get(key) != expected
    ]
    if option_mismatches:
        raise ValueError(
            "预算延长只允许改变 iterations 和第二阶段 learning_rate；"
            "以下父/子参数不一致: " + "; ".join(option_mismatches))

    expected_model = child_profile.get("expected_trainable_parameters", {})
    source_model = parent_manifest.get("model", {})
    expected_count = expected_model.get(route)
    model_mismatches = []
    for key in ("num_layers", "hidden_dim", "encoding", "density_scale", "bound"):
        expected = training.get(key)
        if expected is not None and source_model.get(key) != expected:
            model_mismatches.append(
                f"{key}: parent={source_model.get(key)!r}, child={expected!r}")
    if (expected_count is not None
            and source_model.get("trainable_parameter_count") != expected_count):
        model_mismatches.append(
            "trainable_parameter_count: "
            f"parent={source_model.get('trainable_parameter_count')!r}, "
            f"child={expected_count!r}")
    if model_mismatches:
        raise ValueError(
            "父 checkpoint 与子批次网络结构不兼容: "
            + "; ".join(model_mismatches))

    main_dir = BASELINE_DIR if route == "baseline" else GRADIENT_DIR
    source_mismatches = []
    for relative in ("nerf/network.py", "nerf/renderer.py", "nerf/utils.py"):
        declared_hash = source_hash_with_suffix(parent_manifest, relative)
        current_path = main_dir / relative
        if declared_hash is None:
            source_mismatches.append(f"父清单缺少 {relative}")
        elif not current_path.is_file() or file_sha256(current_path) != declared_hash:
            source_mismatches.append(f"当前 {relative} 与父批次不一致")
    if source_mismatches:
        raise ValueError(
            "扩展训练只允许改变训练预算/学习率声明，计算源码必须兼容: "
            + "; ".join(source_mismatches))

    source_iterations = int(parent_options.get("iters", 0))
    if source_iterations <= 0:
        raise ValueError("父批次实验清单缺少有效的原始 iterations")
    if target_iterations <= source_iterations:
        raise ValueError(
            "子批次声明的目标 iterations 必须大于父批次原始声明: "
            f"parent={source_iterations}, child={target_iterations}")
    policy = child_profile.get("extension_policy", {})
    if (
        policy.get("mode") != "cosine_restart_over_remaining_iterations"
        or policy.get("child_batch_required") is not True
    ):
        raise ValueError("子批次 declared_profile.json 缺少受控 extension_policy")
    eta_min = float(policy.get("eta_min", 1e-6))
    if eta_min != 1e-6:
        raise ValueError("当前受控扩展实现要求 extension_policy.eta_min = 1e-6")
    if eta_min < 0 or eta_min >= restart_learning_rate:
        raise ValueError(
            "extension_policy.eta_min 必须非负且小于 learning_rate")

    record = {
        "requested_at": now(),
        "parent_batch_id": parent_batch_id,
        "child_batch_id": child_batch_id,
        "route": route,
        "parent_experiment_run_id": parent_manifest.get("run_id"),
        "checkpoint": checkpoint.relative_to(PACKAGE_ROOT).as_posix(),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": actual_hash,
        "checkpoint_epoch": checkpoint_state["epoch"],
        "checkpoint_global_step": checkpoint_state["global_step"],
        "source_declared_iterations": source_iterations,
        "target_declared_iterations": target_iterations,
        "remaining_iterations": (
            target_iterations - checkpoint_state["global_step"]),
        "scheduler_policy": "cosine_restart_over_remaining_iterations",
        "restart_learning_rate": restart_learning_rate,
        "eta_min": eta_min,
        "parent_scheduler": {
            "t_max": checkpoint_state["saved_scheduler_t_max"],
            "last_epoch": checkpoint_state["saved_scheduler_last_epoch"],
            "optimizer_lrs": checkpoint_state["saved_optimizer_lrs"],
        },
    }
    return checkpoint, record


def create_batch(args: argparse.Namespace) -> None:
    batch_id = validate_batch_id(args.batch_id)
    root = batch_root(batch_id)
    if root.exists():
        raise FileExistsError(
            f"批次目录已经存在，拒绝覆盖: {root}; 请使用新的批次号")
    profile_source = PROFILES_DIR / f"{args.profile}.json"
    if not profile_source.is_file():
        raise FileNotFoundError(f"找不到实验配置声明: {profile_source}")
    profile = json.loads(profile_source.read_text(encoding="utf-8"))

    root.mkdir(parents=True)
    (root / "comparisons").mkdir()
    (root / "metadata").mkdir()
    shutil.copy2(profile_source, root / "declared_profile.json")
    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "profile_id": profile["profile_id"],
        "created_at": now(),
        "updated_at": now(),
        "note": args.note,
        "status": "created",
        "paths": {
            "baseline": route_workspace(batch_id, "baseline").relative_to(
                PACKAGE_ROOT).as_posix(),
            "gradient_output": route_workspace(
                batch_id, "gradient_output").relative_to(PACKAGE_ROOT).as_posix(),
            "flow_comparison": comparison_directory(
                batch_id, "flow").relative_to(PACKAGE_ROOT).as_posix(),
            "gradient_comparison": comparison_directory(
                batch_id, "gradient").relative_to(PACKAGE_ROOT).as_posix(),
        },
        "preflight": {"status": "pending"},
        "routes": {
            "baseline": {"status": "pending"},
            "gradient_output": {"status": "pending"},
        },
        "comparisons": {"status": "pending"},
    }
    write_json_atomic(root / "batch_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def sync_data(_: argparse.Namespace) -> None:
    if not MATLAB_DATA.is_dir():
        raise FileNotFoundError(f"MATLAB 数据目录不存在: {MATLAB_DATA}")
    copied = 0
    for source in MATLAB_DATA.rglob("*"):
        if not source.is_file():
            continue
        target = PYTHON_DATA / source.relative_to(MATLAB_DATA)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    print(f"已从 MATLAB 同步 {copied} 个文件到 Python 数据目录: {PYTHON_DATA}")


def preflight(args: argparse.Namespace) -> None:
    batch_id = validate_batch_id(args.batch_id)
    load_batch_manifest(batch_id)
    report_dir = batch_root(batch_id) / "metadata/preflight"
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        run([
            sys.executable,
            str(CONTROL_DIR / "validate_package.py"),
            "--batch-id", batch_id,
            "--output", str(report_dir / "package_report.json"),
        ], PACKAGE_ROOT)
        run([
            sys.executable, "validate_common_grid.py",
            "--output", str(report_dir / "common_grid_report.json"),
        ], GRADIENT_DIR)
        run([
            sys.executable, "validate_training_readiness.py",
            "--batch-id", batch_id,
            "--output", str(report_dir / "training_readiness_report.json"),
        ], GRADIENT_DIR, batch_id)
        run([sys.executable, "validate_experiment_logging.py"], GRADIENT_DIR)
        run([
            sys.executable,
            str(CONTROL_DIR / "validate_training_extension.py"),
        ], PACKAGE_ROOT)
        run([sys.executable, "compare_gradients.py", "--self-test"], GRADIENT_DIR)
    except Exception:
        update_batch(batch_id, lambda data: data.update({
            "status": "preflight_failed",
            "preflight": {"status": "failed", "completed_at": now()},
        }))
        raise

    update_batch(batch_id, lambda data: data.update({
        "status": "ready",
        "preflight": {
            "status": "passed",
            "completed_at": now(),
            "reports": "metadata/preflight",
        },
    }))
    print(f"批次 {batch_id} 训练前检查通过。")


def verify_route_result(batch_id: str, route: str) -> dict[str, Any]:
    workspace = route_workspace(batch_id, route)
    manifest_path = workspace / "results/experiment_manifest.json"
    result_path = workspace / "results/sigmas0.mat"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"{route} 缺少完整结果或实验清单: {workspace}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("route") != route:
        raise ValueError(f"{route} 实验清单未完成或路线不符: {manifest_path}")
    return {
        "workspace": str(workspace),
        "result": str(result_path),
        "experiment_run_id": manifest.get("run_id"),
        "global_step": manifest.get("trainer_state", {}).get("final", {}).get(
            "global_step"),
    }


def train_route(
        args: argparse.Namespace,
        route: str,
        resume: bool = False,
) -> None:
    batch_id = validate_batch_id(args.batch_id)
    manifest = load_batch_manifest(batch_id)
    if manifest.get("preflight", {}).get("status") != "passed":
        raise RuntimeError("训练前检查尚未通过，请先运行 preflight")
    workspace = route_workspace(batch_id, route)
    checkpoint = None
    resume_record = None
    route_state = manifest.get("routes", {}).get(route, {})

    if resume:
        prior_status = route_state.get("status")
        if prior_status == "completed":
            raise RuntimeError(
                f"{route} 已经完整结束，拒绝在同一批次追加训练；"
                "如需提高训练预算，请建立新 profile 和新批次，并使用对应的 "
                "extend 命令")
        if prior_status == "running" and not args.allow_stale_running:
            raise RuntimeError(
                f"{route} 仍标记为 running。确认原训练进程已经终止后，"
                "加 --allow-stale-running 恢复断电或进程被强制结束的批次")
        if prior_status not in {"failed", "interrupted", "running"}:
            raise RuntimeError(
                f"{route} 当前状态为 {prior_status!r}，没有可恢复的中断训练")
        if not workspace.is_dir():
            raise FileNotFoundError(f"待恢复路线 workspace 不存在: {workspace}")
        checkpoint = resolve_resume_checkpoint(workspace, args.checkpoint)
        resume_record = {
            "requested_at": now(),
            "prior_status": prior_status,
            "checkpoint": checkpoint.relative_to(PACKAGE_ROOT).as_posix(),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
        }
    elif workspace.exists():
        raise FileExistsError(
            f"路线 workspace 已存在，拒绝覆盖: {workspace}; "
            "中断续训请使用对应的 resume 命令")

    main_dir = BASELINE_DIR if route == "baseline" else GRADIENT_DIR

    launch_time = now()

    def mark_running(data: dict[str, Any]) -> None:
        data["status"] = f"resuming_{route}" if resume else f"training_{route}"
        state = data["routes"].setdefault(route, {})
        for key in ("failed_at", "interrupted_at", "completed_at", "error"):
            state.pop(key, None)
        state["status"] = "running"
        state["launch_mode"] = "resume" if resume else "scratch"
        state.setdefault("started_at", launch_time)
        if resume_record is not None:
            state["last_resumed_at"] = launch_time
            state.setdefault("resume_history", []).append(resume_record)

    def mark_stopped(
            data: dict[str, Any], status_name: str, error: BaseException,
    ) -> None:
        data["status"] = f"{route}_{status_name}"
        state = data["routes"].setdefault(route, {})
        state["status"] = status_name
        state[f"{status_name}_at"] = now()
        state["error"] = repr(error)

    update_batch(batch_id, mark_running)
    try:
        run(
            [sys.executable, "main_BOS.py"],
            main_dir,
            batch_id,
            checkpoint,
        )
        result = verify_route_result(batch_id, route)
    except KeyboardInterrupt as error:
        update_batch(
            batch_id,
            lambda data: mark_stopped(data, "interrupted", error),
        )
        raise
    except subprocess.CalledProcessError as error:
        # 不同平台可能把 Ctrl+C 表示为 SIGINT、shell 130 或 Windows 0xC000013A。
        interrupted = error.returncode in {-2, 130, -1073741510, 3221225786}
        update_batch(
            batch_id,
            lambda data: mark_stopped(
                data, "interrupted" if interrupted else "failed", error),
        )
        raise
    except Exception as error:
        update_batch(
            batch_id,
            lambda data: mark_stopped(data, "failed", error),
        )
        raise

    def mark_completed(data: dict[str, Any]) -> None:
        data["status"] = f"{route}_completed"
        state = data["routes"].setdefault(route, {})
        for key in ("failed_at", "interrupted_at", "error"):
            state.pop(key, None)
        state.update({
            "status": "completed",
            "completed_at": now(),
            **result,
        })

    update_batch(batch_id, mark_completed)
    print(f"批次 {batch_id} 的 {route} 已完整结束。")


def extend_route(args: argparse.Namespace, route: str) -> None:
    """从已完成父批次延长到新子批次，不改写父批次。"""
    child_batch_id = validate_batch_id(args.batch_id)
    parent_batch_id = validate_batch_id(args.from_batch)
    child_manifest = load_batch_manifest(child_batch_id)
    if child_manifest.get("preflight", {}).get("status") != "passed":
        raise RuntimeError("子批次训练前检查尚未通过，请先运行 preflight")

    child_workspace = route_workspace(child_batch_id, route)
    child_state = child_manifest.get("routes", {}).get(route, {})
    if child_state.get("status") != "pending":
        raise RuntimeError(
            f"子批次 {route} 状态必须为 pending，实际为 "
            f"{child_state.get('status')!r}")
    if child_workspace.exists():
        raise FileExistsError(f"子批次路线 workspace 已存在，拒绝覆盖: {child_workspace}")

    checkpoint, extension_record = verify_extension_source(
        parent_batch_id,
        child_batch_id,
        route,
        args.checkpoint,
    )
    main_dir = BASELINE_DIR if route == "baseline" else GRADIENT_DIR
    launch_time = now()

    def mark_running(data: dict[str, Any]) -> None:
        data["status"] = f"extending_{route}"
        state = data["routes"].setdefault(route, {})
        state.update({
            "status": "running",
            "launch_mode": "extension",
            "started_at": launch_time,
            "extension_source": extension_record,
        })

    def mark_stopped(
            data: dict[str, Any], status_name: str, error: BaseException,
    ) -> None:
        data["status"] = f"{route}_{status_name}"
        state = data["routes"].setdefault(route, {})
        state["status"] = status_name
        state[f"{status_name}_at"] = now()
        state["error"] = repr(error)

    update_batch(child_batch_id, mark_running)
    try:
        run(
            [sys.executable, "main_BOS.py"],
            main_dir,
            child_batch_id,
            checkpoint,
            extension=True,
        )
        result = verify_route_result(child_batch_id, route)
        result_manifest_path = (
            child_workspace / "results/experiment_manifest.json")
        result_manifest = json.loads(
            result_manifest_path.read_text(encoding="utf-8"))
        extension_details = result_manifest.get("extension", {})
        expected_remaining = extension_record["remaining_iterations"]
        if (
            extension_details.get("policy")
            != "cosine_restart_over_remaining_iterations"
            or extension_details.get("source_global_step")
            != extension_record["checkpoint_global_step"]
            or extension_details.get("remaining_iterations")
            != expected_remaining
        ):
            raise ValueError(
                f"子批次实验清单中的扩展策略记录不完整: {result_manifest_path}")
    except KeyboardInterrupt as error:
        update_batch(
            child_batch_id,
            lambda data: mark_stopped(data, "interrupted", error),
        )
        raise
    except subprocess.CalledProcessError as error:
        interrupted = error.returncode in {-2, 130, -1073741510, 3221225786}
        update_batch(
            child_batch_id,
            lambda data: mark_stopped(
                data, "interrupted" if interrupted else "failed", error),
        )
        raise
    except Exception as error:
        update_batch(
            child_batch_id,
            lambda data: mark_stopped(data, "failed", error),
        )
        raise

    def mark_completed(data: dict[str, Any]) -> None:
        data["status"] = f"{route}_completed"
        state = data["routes"].setdefault(route, {})
        state.update({
            "status": "completed",
            "completed_at": now(),
            **result,
        })

    update_batch(child_batch_id, mark_completed)
    print(
        f"父批次 {parent_batch_id} 的 {route} 已延长到子批次 "
        f"{child_batch_id}。")


def compare(args: argparse.Namespace) -> None:
    batch_id = validate_batch_id(args.batch_id)
    baseline = verify_route_result(batch_id, "baseline")
    gradient = verify_route_result(batch_id, "gradient_output")
    flow_output = comparison_directory(batch_id, "flow")
    gradient_output = comparison_directory(batch_id, "gradient")

    update_batch(batch_id, lambda data: (
        data.update({"status": "comparing"}),
        data.update({"comparisons": {"status": "running", "started_at": now()}}),
    ))
    common = [
        "--baseline", baseline["result"],
        "--gradient", gradient["result"],
    ]
    try:
        flow_command = [
            sys.executable, "compare_reconstructions.py", *common,
            "--output-dir", str(flow_output),
        ]
        gradient_command = [
            sys.executable, "compare_gradients.py", *common,
            "--output-dir", str(gradient_output),
        ]
        if not args.show:
            flow_command.append("--no-show")
            gradient_command.append("--no-show")
        if args.skip_field_save:
            gradient_command.append("--skip-field-save")
        run(flow_command, GRADIENT_DIR)
        run(gradient_command, GRADIENT_DIR)
    except Exception as error:
        update_batch(batch_id, lambda data: (
            data.update({"status": "comparison_failed"}),
            data.update({"comparisons": {
                "status": "failed", "failed_at": now(), "error": repr(error),
            }}),
        ))
        raise

    update_batch(batch_id, lambda data: (
        data.update({"status": "completed"}),
        data.update({"comparisons": {
            "status": "completed",
            "completed_at": now(),
            "flow_report": str(flow_output / "comparison_report.json"),
            "gradient_report": str(
                gradient_output / "gradient_comparison_report.json"),
        }}),
    ))
    print(f"批次 {batch_id} 的两类比较已完成。")


def status(args: argparse.Namespace) -> None:
    batch_id = validate_batch_id(args.batch_id)
    manifest = load_batch_manifest(batch_id)
    summary = {
        "batch_id": batch_id,
        "profile_id": manifest.get("profile_id"),
        "status": manifest.get("status"),
        "preflight": manifest.get("preflight"),
        "routes": manifest.get("routes"),
        "comparisons": manifest.get("comparisons"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="创建全新实验批次和保存路径")
    create.add_argument("--batch-id", required=True)
    create.add_argument("--profile", default="strict_control_v1")
    create.add_argument("--note", default="")
    create.set_defaults(function=create_batch)

    sync = subparsers.add_parser("sync-data", help="把 MATLAB 数据同步到 Python")
    sync.set_defaults(function=sync_data)

    check = subparsers.add_parser("preflight", help="执行正式训练前全部检查")
    check.add_argument("--batch-id", required=True)
    check.set_defaults(function=preflight)

    baseline = subparsers.add_parser("train-baseline", help="训练基线版")
    baseline.add_argument("--batch-id", required=True)
    baseline.set_defaults(function=lambda args: train_route(args, "baseline"))

    gradient = subparsers.add_parser("train-gradient", help="训练梯度版")
    gradient.add_argument("--batch-id", required=True)
    gradient.set_defaults(function=lambda args: train_route(args, "gradient_output"))

    def add_resume_parser(name: str, route: str, help_text: str) -> None:
        resume = subparsers.add_parser(name, help=help_text)
        resume.add_argument("--batch-id", required=True)
        resume.add_argument(
            "--checkpoint",
            default="latest",
            help="latest（默认）或同一路线 checkpoints/ 下的 ngp_epXXXX.pth 文件名",
        )
        resume.add_argument(
            "--allow-stale-running",
            action="store_true",
            help="仅在确认旧训练进程已终止后，恢复仍标记为 running 的断电批次",
        )
        resume.set_defaults(
            function=lambda args, selected_route=route: train_route(
                args, selected_route, resume=True))

    add_resume_parser(
        "resume-baseline", "baseline", "从同批次完整 checkpoint 恢复基线版")
    add_resume_parser(
        "resume-gradient", "gradient_output", "从同批次完整 checkpoint 恢复梯度版")

    def add_extend_parser(name: str, route: str, help_text: str) -> None:
        extension = subparsers.add_parser(name, help=help_text)
        extension.add_argument(
            "--from-batch", required=True, help="已完成路线所在的父批次号")
        extension.add_argument(
            "--batch-id", required=True, help="通过 preflight 的全新子批次号")
        extension.add_argument(
            "--checkpoint",
            default="latest",
            help=(
                "latest（默认）或父批次同路线 checkpoints/ 下的 "
                "ngp_epXXXX.pth 文件名"),
        )
        extension.set_defaults(
            function=lambda args, selected_route=route: extend_route(
                args, selected_route))

    add_extend_parser(
        "extend-baseline",
        "baseline",
        "从已完成父批次 checkpoint 延长基线版到新子批次",
    )
    add_extend_parser(
        "extend-gradient",
        "gradient_output",
        "从已完成父批次 checkpoint 延长梯度版到新子批次",
    )

    comparison = subparsers.add_parser("compare", help="运行标量和梯度两类比较")
    comparison.add_argument("--batch-id", required=True)
    comparison.add_argument("--show", action="store_true", help="显示交互窗口")
    comparison.add_argument("--skip-field-save", action="store_true")
    comparison.set_defaults(function=compare)

    show_status = subparsers.add_parser("status", help="查看批次状态")
    show_status.add_argument("--batch-id", required=True)
    show_status.set_defaults(function=status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
