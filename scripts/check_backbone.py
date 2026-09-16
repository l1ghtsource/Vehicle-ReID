import argparse
import gc
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from models import ReIDModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "models",
        nargs="*",
        default=[
            "convnext_tiny",
            "vit",
            "dinov3_convnext_base",
            "dinov3_convnext_large",
            "dinov3_convnext_multilevel",
            "radio",
            "llm2clip",
        ],
    )
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    results = []
    for name in args.models:
        with initialize_config_dir(
            version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
        ):
            cfg = compose(config_name="config", overrides=[f"model={name}", "model.pretrained=false"])
        if name == "llm2clip":
            cfg.data.image_size = [336, 336]
        model = ReIDModel(cfg).cuda().train(args.backward)
        x = torch.randn(2, 3, *cfg.data.image_size, device="cuda")
        with torch.set_grad_enabled(args.backward), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = out["raw"].square().mean()
        if args.backward:
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(name)
        assert torch.isfinite(out["embedding"]).all()
        result = {
            "model": name,
            "shape": list(out["embedding"].shape),
            "backward": args.backward,
            "pretrained": False,
        }
        print(json.dumps(result), flush=True)
        results.append(result)
        del model, x, out, loss
        gc.collect()
        torch.cuda.empty_cache()
    Path("artifacts/backbone_checks.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
