"""验证正式实验包可移植内容、数据副本和可选批次结构。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from batch_paths import (
    PACKAGE_ROOT,
    batch_root,
    load_batch_manifest,
    route_workspace,
    validate_batch_id,
)


REQUIRED_PATHS = (
    "MATLAB/step1_InitBOSLAB.m",
    "MATLAB/step2_Compile.m",
    "MATLAB/Utilities/Demos/step3_generate_phantom1_synthetic_data.m",
    "MATLAB/Utilities/Demos/export_phantom1_flow_ground_truth.m",
    "MATLAB/Test_data/Phantom 1/n_GroundTruth.mat",
    "MATLAB/Test_data/Phantom 1/140x294x140/flow_ground_truth.mat",
    "Common/CUDA/BOSLAB_common.cu",
    "Common/CUDA/ray_interpolated_projection.cu",
    "PYTHON/NIR-BOS/environment.yml",
    "PYTHON/NIR-BOS/main_BOS.py",
    "PYTHON/NIR-BOS/experiment_logging.py",
    "PYTHON/NIR-BOS/nerf/network.py",
    "PYTHON/NIR-BOS/nerf/renderer.py",
    "PYTHON/NIR-BOS/raymarching/src/raymarching.cu",
    "PYTHON/NIR-BOS/data/Phantom 1/140x294x140/transforms_train.json",
    "PYTHON/NIR-BOS/data/Phantom 1/140x294x140/3Dmask.mat",
    "claudedo/gradient_output/main_BOS.py",
    "claudedo/gradient_output/experiment_logging.py",
    "claudedo/gradient_output/compare_reconstructions.py",
    "claudedo/gradient_output/compare_gradients.py",
    "claudedo/gradient_output/validate_common_grid.py",
    "claudedo/gradient_output/validate_training_readiness.py",
    "claudedo/gradient_output/nerf/poisson_solver.py",
    "claudedo/gradient_output/raymarching/src/raymarching.cu",
    "experiment_control/experiment.py",
    "experiment_control/batch_paths.py",
    "experiment_control/training_extension.py",
    "experiment_control/validate_training_extension.py",
    "experiment_control/validate_package.py",
    "experiment_control/verify_transfer.py",
    "experiment_control/generate_checksums.py",
    "experiment_control/profiles/strict_control_v1.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_copy_check() -> tuple[bool, dict[str, Any]]:
    matlab = PACKAGE_ROOT / "MATLAB/Test_data/Phantom 1"
    python = PACKAGE_ROOT / "PYTHON/NIR-BOS/data/Phantom 1"
    missing: list[str] = []
    mismatched: list[str] = []
    compared = 0
    for source in sorted(matlab.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(matlab)
        if relative.as_posix() == "140x294x140/flow_ground_truth.mat":
            continue
        target = python / relative
        if not target.is_file():
            missing.append(relative.as_posix())
            continue
        compared += 1
        if source.stat().st_size != target.stat().st_size or sha256(source) != sha256(target):
            mismatched.append(relative.as_posix())
    details = {
        "compared_files": compared,
        "missing_in_python_data": missing,
        "hash_mismatches": mismatched,
        "matlab_only_exact_ground_truth": (
            matlab / "140x294x140/flow_ground_truth.mat").is_file(),
    }
    return not missing and not mismatched, details


def checksum_check() -> tuple[bool, dict[str, Any]]:
    checksum_path = PACKAGE_ROOT / "PACKAGE_CHECKSUMS.sha256"
    if not checksum_path.is_file():
        return False, {"status": "missing", "file": str(checksum_path)}
    mismatches: list[str] = []
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = PACKAGE_ROOT / relative
        checked += 1
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(relative)
    return not mismatches, {
        "status": "verified" if not mismatches else "mismatch",
        "checked_files": checked,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-package-checksums",
        action="store_true",
        help="仅限封装过程中、校验和文件生成之前使用",
    )
    parser.add_argument(
        "--require-portable-clean",
        action="store_true",
        help=(
            "要求 Python 源码树不含任何目标设备生成的 build、pyd 等文件；"
            "仅用于交付前封装验收"
        ),
    )
    args = parser.parse_args()

    required = {
        relative: (PACKAGE_ROOT / relative).is_file()
        for relative in REQUIRED_PATHS
    }
    data_ok, data_details = data_copy_check()
    forbidden = []
    for base in (
        PACKAGE_ROOT / "PYTHON/NIR-BOS",
        PACKAGE_ROOT / "claudedo/gradient_output",
    ):
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            if path.suffix.lower() in {".pyc", ".pyd", ".pdb", ".so"}:
                forbidden.append(relative)
            elif any(part in {"result", "__pycache__", "build", "build_temp", "dist"}
                     for part in path.parts):
                forbidden.append(relative)

    checksum_ok, checksum_details = (
        (True, {"status": "skipped_during_packaging"})
        if args.skip_package_checksums else checksum_check()
    )
    checks: dict[str, bool] = {
        "all_required_paths_exist": all(required.values()),
        "matlab_and_python_dataset_copies_match": data_ok,
        "portable_source_tree_clean_when_required": (
            not forbidden if args.require_portable_clean else True),
        "standalone_cpp_project_intentionally_excluded": (
            not (PACKAGE_ROOT / "C++").exists()),
        "package_checksums_valid": checksum_ok,
    }

    batch_details = None
    if args.batch_id:
        batch_id = validate_batch_id(args.batch_id)
        manifest = load_batch_manifest(batch_id)
        expected_paths = {
            route: route_workspace(batch_id, route).relative_to(PACKAGE_ROOT).as_posix()
            for route in ("baseline", "gradient_output")
        }
        batch_details = {
            "root": str(batch_root(batch_id)),
            "manifest": manifest,
            "expected_route_paths": expected_paths,
        }
        checks.update({
            "batch_manifest_id_matches": manifest.get("batch_id") == batch_id,
            "batch_profile_is_declared": bool(manifest.get("profile_id")),
            "batch_route_paths_match_policy": all(
                manifest.get("paths", {}).get(route) == expected
                for route, expected in expected_paths.items()),
            "batch_route_paths_are_distinct": (
                expected_paths["baseline"] != expected_paths["gradient_output"]),
        })

    report = {
        "passed": all(checks.values()),
        "package_root": str(PACKAGE_ROOT),
        "required_paths": required,
        "data_copy": data_details,
        "machine_build_artifact_policy": (
            "must_be_absent" if args.require_portable_clean else "reported_only"),
        "portable_source_tree_clean": not forbidden,
        "forbidden_build_artifacts": forbidden,
        "package_checksums": checksum_details,
        "batch": batch_details,
        "checks": checks,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if not report["passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"正式实验包验证失败: {failed}")


if __name__ == "__main__":
    main()
