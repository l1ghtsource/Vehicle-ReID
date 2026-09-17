from omegaconf import OmegaConf


def spatial_multiple(model_cfg) -> int:
    backend = str(getattr(model_cfg, "backend", ""))
    if backend == "custom":
        return 1
    value = OmegaConf.select(model_cfg, "spatial_multiple", default=1)
    multiple = int(1 if value is None else value)
    if multiple < 1:
        raise ValueError("model.spatial_multiple must be >= 1")
    return multiple


def scaled_hw(height: int, width: int, scale: float, multiple: int) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("TTA scales/rotations must be nonempty and scales positive")
    if scale == 1:
        return height, width

    def snap(dim: int) -> int:
        raw = max(1, round(dim * float(scale)))
        aligned = int(round(raw / multiple) * multiple)
        return max(multiple, aligned)

    return snap(height), snap(width)


def validate_image_geometry(cfg) -> None:
    sizes = [int(value) for value in cfg.data.image_size]
    if len(sizes) != 2 or min(sizes) < 1:
        raise ValueError("data.image_size must be a positive [height, width]")
    height, width = sizes
    backend = str(cfg.model.backend)
    if backend == "llm2clip" and [height, width] != [336, 336]:
        raise ValueError("LLM2CLIP preset requires 336x336; position/RoPE interpolation is not implicit")
    multiple = spatial_multiple(cfg.model)
    if height % multiple or width % multiple:
        raise ValueError(
            f"data.image_size {[height, width]} must be divisible by {multiple} for {cfg.model.name}"
        )
    tta = cfg.eval.tta
    if not tta.enabled:
        return
    for scale in tta.scales:
        scaled_hw(height, width, float(scale), multiple)
