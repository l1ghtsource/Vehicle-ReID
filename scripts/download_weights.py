import argparse
from pathlib import Path
from typing import Literal, TypedDict

from huggingface_hub import hf_hub_download, snapshot_download

from scripts.verify_weights import verify_digests


class SnapshotSpec(TypedDict):
    kind: Literal["snapshot"]
    repo_id: str
    allow_patterns: list[str]
    sha256: dict[str, str]


class FileSpec(TypedDict):
    kind: Literal["file"]
    repo_id: str
    filename: str
    revision: str
    sha256: dict[str, str]


MODELS: dict[str, SnapshotSpec | FileSpec] = {
    "dinov3_base": {
        "kind": "snapshot",
        "repo_id": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        "allow_patterns": ["*.json", "*.safetensors", "README.md"],
        "sha256": {
            "model.safetensors": "ec90bd798b5fc5b8e30443796a6c24a7a73e28ad85c6c0ceda78b1d249a694cc",
            "config.json": "a449c3c6beb7ea71afc822360eada0db6838f757cf1b0115c6609a18e2bc8da0",
            "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
        },
    },
    "dinov3_large": {
        "kind": "snapshot",
        "repo_id": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        "allow_patterns": ["*.json", "*.safetensors", "README.md"],
        "sha256": {
            "model.safetensors": "8bab82eaa133954d0557bebf70416e0b667fa417a19ea47b697683b34190633d",
            "config.json": "cb33dd99f77a2a73806fa218df80782a15e81422eb6d6fb2816534a3cb620360",
            "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
        },
    },
    "radio": {
        "kind": "file",
        "repo_id": "nvidia/C-RADIOv4-SO400M",
        "filename": "model.safetensors",
        "revision": "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
        "sha256": {
            "model.safetensors": "23e0c117de49d4ce909150fe6658d470829e6639647c7a5b035ce82e0d5b763c",
        },
    },
    "llm2clip": {
        "kind": "file",
        "repo_id": "microsoft/LLM2CLIP-EVA02-L-14-336",
        "filename": "LLM2CLIP-EVA02-L-14-336.pt",
        "revision": "e6a348e3b1448ff6e2b0f6b09c1b8e9343461e4b",
        "sha256": {
            "LLM2CLIP-EVA02-L-14-336.pt": "408f4c9dfd708155972af4a4d224a7bf42bb2e71420e9b9a82377dee737dedbb",
        },
    },
    "efficientloftr": {
        "kind": "snapshot",
        "repo_id": "zju-community/efficientloftr",
        "allow_patterns": ["*.json", "*.safetensors", "README.md"],
        "sha256": {
            "model.safetensors": "00a5edc343fa222eba763643553548ad37a05aa3d00d266553c9f6cf67bb0e64",
            "config.json": "eaa141ed38f64ec65efe7b663338e2b44d260ff2b26f5829624e5241ff29f1d8",
            "preprocessor_config.json": "22151d20a71d5d25bee2b5f5c3a5999befa3285597218a9d1ea63998f84dc606",
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--directory", default="weights")
    args = parser.parse_args()
    spec = MODELS[args.model]
    root = Path(args.directory) / args.model
    if spec["kind"] == "snapshot":
        print(
            snapshot_download(
                spec["repo_id"],
                local_dir=root,
                allow_patterns=spec["allow_patterns"],
            )
        )
    else:
        print(
            hf_hub_download(
                spec["repo_id"],
                spec["filename"],
                revision=spec["revision"],
                local_dir=root,
            )
        )
    verify_digests(root, spec["sha256"])


if __name__ == "__main__":
    main()
