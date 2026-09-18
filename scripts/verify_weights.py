import argparse
import hashlib
import re
from pathlib import Path

SKIP_NAMES = frozenset({"SHA256SUMS", ".gitkeep"})
SUMS_NAME = "SHA256SUMS"
HASH_LINE = re.compile(r"^([0-9a-f]{64})(?:  | \*)(.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and path.name not in SKIP_NAMES]
    return sorted(files)


def parse_sha256sums(text: str) -> dict[str, str]:
    listed = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = HASH_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line: {raw}")
        name = match.group(2)
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"invalid SHA256SUMS path: {name}")
        listed[name] = match.group(1)
    return listed


def verify_digests(root: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(name)
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"{name}: expected {digest}, got {actual}")


def write_sha256sums(root: Path) -> Path:
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in payload_files(root)]
    target = root / SUMS_NAME
    target.write_text("".join(lines))
    return target


def verify_tree(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    sums_path = root / SUMS_NAME
    listed = parse_sha256sums(sums_path.read_text() if sums_path.is_file() else "")
    present = {path.relative_to(root).as_posix() for path in payload_files(root)}
    extra = sorted(present - set(listed))
    if extra and not listed:
        raise ValueError(f"payload files are missing from {SUMS_NAME}: {extra}")
    if extra:
        raise ValueError(f"unlisted payload files: {extra}")
    verify_digests(root, listed)
    return listed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("weights/finetuned"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root
    if args.write:
        write_sha256sums(root)
    listed = verify_tree(root)
    print(f"{root}: {len(listed)} files")


if __name__ == "__main__":
    main()
