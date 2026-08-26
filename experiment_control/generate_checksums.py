"""封装完成时生成不可变源码校验和与一次性完整搬运校验和。"""

from __future__ import annotations

import hashlib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_OUTPUT = PACKAGE_ROOT / "PACKAGE_CHECKSUMS.sha256"
TRANSFER_OUTPUT = PACKAGE_ROOT / "TRANSFER_CHECKSUMS.sha256"
GENERATED_OUTPUTS = {PACKAGE_OUTPUT.name, TRANSFER_OUTPUT.name}
MUTABLE_PREFIXES = (
    "MATLAB/Test_data/",
    "MATLAB/Mex_files/",
    "PYTHON/NIR-BOS/data/",
    "experiments/",
)
MUTABLE_EXACT = {"MATLAB/mex_CUDA_win64.xml"}
FORBIDDEN_PARTS = {
    ".git", "__pycache__", "build", "build_temp", "dist", "result",
}
# MATLAB 随包携带的 MEX 调试符号也属于首次搬运内容，因此不排除 .pdb。
# Python 源码树中的旧设备 .pdb 已在组包时移除。
FORBIDDEN_SUFFIXES = {".pyc", ".pyd", ".so", ".obj"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if path.name in GENERATED_OUTPUTS:
            continue
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        if relative.startswith("experiments/") and relative != "experiments/README.md":
            continue
        files.append((relative, path))
    return sorted(files)


def write_checksums(path: Path, files: list[tuple[str, Path]]) -> None:
    lines = [f"{sha256(file)}  {relative}" for relative, file in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    transfer_files = portable_files()
    immutable_files = [
        (relative, path) for relative, path in transfer_files
        if not relative.startswith(MUTABLE_PREFIXES)
        and relative not in MUTABLE_EXACT
    ]
    write_checksums(PACKAGE_OUTPUT, immutable_files)

    # 完整搬运清单还要覆盖刚生成的不可变清单本身。
    transfer_files.append((PACKAGE_OUTPUT.name, PACKAGE_OUTPUT))
    transfer_files.sort()
    write_checksums(TRANSFER_OUTPUT, transfer_files)
    print(
        f"immutable files: {len(immutable_files)}; "
        f"transfer files: {len(transfer_files)}")


if __name__ == "__main__":
    main()
