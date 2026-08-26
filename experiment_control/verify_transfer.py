"""新设备收到实验包后，一次性核验完整搬运内容。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = PACKAGE_ROOT / "TRANSFER_CHECKSUMS.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not CHECKSUM_FILE.is_file():
        raise SystemExit(f"缺少搬运校验文件: {CHECKSUM_FILE}")
    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = PACKAGE_ROOT / relative
        checked += 1
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != expected:
            mismatched.append(relative)
    report = {
        "passed": not missing and not mismatched,
        "package_root": str(PACKAGE_ROOT),
        "checked_files": checked,
        "missing": missing,
        "mismatched": mismatched,
        "note": (
            "run once immediately after transfer, before MATLAB regeneration, "
            "MEX compilation, or data synchronization"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("正式实验包搬运校验失败")


if __name__ == "__main__":
    main()
