from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

REQUIRED = ("config.json", "preprocessor_config.json", "model.safetensors")


def default_weights():
    return Path(__file__).resolve().parents[1] / "weights" / "efficientloftr"


def load_matcher(path=None, device="cpu"):
    root = Path(default_weights() if path is None else path)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"EfficientLoFTR weights missing under {root}: {missing}")
    processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
    model = AutoModelForKeypointMatching.from_pretrained(root, local_files_only=True)
    model.to(torch.device(device)).eval()
    return processor, model
