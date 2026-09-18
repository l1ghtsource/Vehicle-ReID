from pathlib import Path

CONTEST_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
    ".engine",
    ".plan",
    ".safetensors",
    ".ckpt",
    ".trt",
    ".pb",
    ".tflite",
    ".npz",
)
EXTRA_SUFFIXES = (".cbm",)
SKIP_NAMES = frozenset({"SHA256SUMS", ".gitkeep"})
LIMIT_BYTES: int = 2 * 1024**3


def _as_path(root) -> Path:
    return Path(root)


def _suffix(path: Path) -> str:
    return path.suffix.lower()


def is_contest_weight(path: Path) -> bool:
    return path.is_file() and path.name not in SKIP_NAMES and _suffix(path) in CONTEST_SUFFIXES


def is_extra_weight(path: Path) -> bool:
    return path.is_file() and path.name not in SKIP_NAMES and _suffix(path) in EXTRA_SUFFIXES


def iter_files(roots) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        path = _as_path(root)
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for item in candidates:
            resolved = item.resolve()
            if resolved in seen or not item.is_file():
                continue
            seen.add(resolved)
            files.append(resolved)
    return files


def describe(path: Path) -> dict:
    return {
        "path": str(path),
        "name": path.name,
        "suffix": _suffix(path),
        "bytes": path.stat().st_size,
    }


def inventory(roots, limit_bytes: int = LIMIT_BYTES) -> dict:
    files = iter_files(roots)
    contest = [describe(path) for path in files if is_contest_weight(path)]
    extra = [describe(path) for path in files if is_extra_weight(path)]
    total = int(sum(item["bytes"] for item in contest))
    extra_bytes = int(sum(item["bytes"] for item in extra))
    return {
        "files": contest,
        "total_bytes": total,
        "limit_bytes": int(limit_bytes),
        "within_limit": total <= limit_bytes,
        "headroom_bytes": int(limit_bytes) - total,
        "extra_files": extra,
        "extra_bytes": extra_bytes,
        "scanned_roots": [str(_as_path(root)) for root in roots],
    }
