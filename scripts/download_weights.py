import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["dinov3_base", "dinov3_large", "radio", "llm2clip"])
    p.add_argument("--directory", default="weights")
    args = p.parse_args()
    root = Path(args.directory) / args.model
    if args.model.startswith("dinov3"):
        name = "base" if args.model.endswith("base") else "large"
        print(
            snapshot_download(
                f"facebook/dinov3-convnext-{name}-pretrain-lvd1689m",
                local_dir=root,
                allow_patterns=["*.json", "*.safetensors", "README.md"],
            )
        )
    else:
        repo, filename, revision = {
            "radio": (
                "nvidia/C-RADIOv4-SO400M",
                "model.safetensors",
                "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
            ),
            "llm2clip": (
                "microsoft/LLM2CLIP-EVA02-L-14-336",
                "LLM2CLIP-EVA02-L-14-336.pt",
                "e6a348e3b1448ff6e2b0f6b09c1b8e9343461e4b",
            ),
        }[args.model]
        print(hf_hub_download(repo, filename, revision=revision, local_dir=root))


if __name__ == "__main__":
    main()
