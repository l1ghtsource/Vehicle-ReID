import json
from pathlib import Path
from typing import Any

import optuna
from omegaconf import DictConfig, ListConfig, OmegaConf

LOSS_PRESETS = [
    "combined",
    "arcface",
    "cosface",
    "sphereface2",
    "triplet",
    "triplet_semihard",
    "triplet_distanceweighted",
    "adasp",
    "arcface_adasp",
    "ce_triplet",
    "circle",
    "multisimilarity",
    "supcon",
    "proxyanchor",
    "contrastive",
    "ntxent",
    "proxynca",
    "softtriple",
    "fastap",
    "lifted",
]

CLASSIFIER_LOSSES = {
    "combined",
    "arcface",
    "cosface",
    "sphereface2",
    "arcface_adasp",
    "ce_triplet",
}

ADASP_LOSSES = {"adasp", "arcface_adasp"}

LOSS_TERM_COUNTS = {
    "combined": 2,
    "arcface": 1,
    "cosface": 1,
    "sphereface2": 1,
    "triplet": 1,
    "triplet_semihard": 1,
    "triplet_distanceweighted": 1,
    "adasp": 1,
    "arcface_adasp": 2,
    "ce_triplet": 2,
    "circle": 1,
    "multisimilarity": 1,
    "supcon": 1,
    "proxyanchor": 1,
    "contrastive": 1,
    "ntxent": 1,
    "proxynca": 1,
    "softtriple": 1,
    "fastap": 1,
    "lifted": 1,
}

LOSS_PARAM_SPECS: dict[str, dict[str, tuple[Any, ...]]] = {
    "combined": {
        "loss.terms.0.params.margin": ("float", 0.1, 0.6),
        "loss.terms.0.params.scale": ("float", 16.0, 96.0, True),
        "loss.terms.0.params.label_smoothing": ("float", 0.0, 0.2),
        "loss.terms.1.params.margin": ("float", 0.05, 0.8),
        "loss.terms.1.params.soft_margin": ("bool",),
        "loss.terms.1.params.normalize": ("bool",),
        "loss.terms.1.params.mining": ("categorical", ["hard", "semihard", "weighted", "all"]),
    },
    "arcface": {
        "loss.terms.0.params.margin": ("float", 0.1, 0.6),
        "loss.terms.0.params.scale": ("float", 16.0, 96.0, True),
        "loss.terms.0.params.label_smoothing": ("float", 0.0, 0.2),
    },
    "cosface": {
        "loss.terms.0.params.margin": ("float", 0.1, 0.6),
        "loss.terms.0.params.scale": ("float", 16.0, 96.0, True),
    },
    "sphereface2": {
        "loss.terms.0.params.alpha": ("float", 0.3, 0.9),
        "loss.terms.0.params.r": ("float", 16.0, 80.0, True),
        "loss.terms.0.params.m": ("float", 0.1, 0.7),
        "loss.terms.0.params.t": ("float", 1.0, 5.0),
        "loss.terms.0.params.lw": ("float", 10.0, 100.0, True),
    },
    "triplet": {
        "loss.terms.0.params.margin": ("float", 0.05, 0.8),
        "loss.terms.0.params.soft_margin": ("bool",),
        "loss.terms.0.params.normalize": ("bool",),
        "loss.terms.0.params.mining": ("categorical", ["hard", "semihard", "weighted", "all"]),
    },
    "triplet_semihard": {
        "loss.terms.0.params.margin": ("float", 0.05, 0.8),
        "loss.terms.0.miner.margin": ("float", 0.05, 0.8),
        "loss.terms.0.miner.type_of_triplets": (
            "categorical",
            ["semihard", "hard", "easy", "all"],
        ),
    },
    "triplet_distanceweighted": {
        "loss.terms.0.params.margin": ("float", 0.05, 0.8),
        "loss.terms.0.miner.cutoff": ("float", 0.2, 0.8),
        "loss.terms.0.miner.nonzero_loss_cutoff": ("float", 0.8, 2.0),
    },
    "adasp": {
        "loss.terms.0.params.temp": ("float", 0.01, 0.2, True),
        "loss.terms.0.params.loss_type": ("categorical", ["adasp", "sp-h", "sp-lh"]),
    },
    "arcface_adasp": {
        "loss.terms.0.params.margin": ("float", 0.1, 0.6),
        "loss.terms.0.params.scale": ("float", 16.0, 96.0, True),
        "loss.terms.1.params.temp": ("float", 0.01, 0.2, True),
        "loss.terms.1.params.loss_type": ("categorical", ["adasp", "sp-h", "sp-lh"]),
    },
    "ce_triplet": {
        "loss.terms.0.params.label_smoothing": ("float", 0.0, 0.3),
        "loss.terms.1.params.margin": ("float", 0.05, 0.8),
        "loss.terms.1.params.normalize": ("bool",),
        "loss.terms.1.params.mining": ("categorical", ["hard", "semihard", "weighted", "all"]),
    },
    "circle": {
        "loss.terms.0.params.m": ("float", 0.05, 0.5),
        "loss.terms.0.params.gamma": ("float", 16.0, 160.0, True),
    },
    "multisimilarity": {
        "loss.terms.0.params.alpha": ("float", 1.0, 4.0),
        "loss.terms.0.params.beta": ("float", 20.0, 100.0, True),
        "loss.terms.0.params.base": ("float", 0.2, 0.8),
        "loss.terms.0.miner.epsilon": ("float", 0.05, 0.3),
    },
    "supcon": {"loss.terms.0.params.temperature": ("float", 0.02, 0.3, True)},
    "proxyanchor": {
        "loss.terms.0.params.margin": ("float", 0.02, 0.4),
        "loss.terms.0.params.alpha": ("float", 8.0, 64.0, True),
    },
    "contrastive": {
        "loss.terms.0.params.pos_margin": ("float", 0.0, 0.5),
        "loss.terms.0.params.neg_margin": ("float", 0.5, 2.0),
        "loss.terms.0.miner.pos_margin": ("float", 0.0, 0.5),
        "loss.terms.0.miner.neg_margin": ("float", 0.5, 2.0),
    },
    "ntxent": {"loss.terms.0.params.temperature": ("float", 0.02, 0.3, True)},
    "proxynca": {"loss.terms.0.params.softmax_scale": ("float", 1.0, 10.0, True)},
    "softtriple": {
        "loss.terms.0.params.centers_per_class": ("int", 1, 10),
        "loss.terms.0.params.la": ("float", 5.0, 50.0, True),
        "loss.terms.0.params.gamma": ("float", 0.02, 0.5, True),
        "loss.terms.0.params.margin": ("float", 0.0, 0.2),
    },
    "fastap": {"loss.terms.0.params.num_bins": ("int", 5, 50)},
    "lifted": {
        "loss.terms.0.params.neg_margin": ("float", 0.5, 2.0),
        "loss.terms.0.params.pos_margin": ("float", 0.0, 0.5),
    },
}

TRANSFORM_PARAM_SPECS: list[dict[str, tuple[Any, ...]]] = [
    {},
    {
        "scale.0": ("float", 0.5, 0.95),
        "scale.1": ("float", 0.96, 1.0),
        "ratio.0": ("float", 0.6, 0.95),
        "ratio.1": ("float", 1.05, 1.5),
    },
    {
        "scale.0": ("float", 0.75, 1.0),
        "scale.1": ("float", 1.0, 1.25),
        "translate_percent.0": ("float", -0.15, -0.01),
        "translate_percent.1": ("float", 0.01, 0.15),
        "rotate.0": ("int", -20, -2),
        "rotate.1": ("int", 2, 20),
        "shear.0": ("int", -10, -1),
        "shear.1": ("int", 1, 10),
        "border_mode": ("categorical", [0, 1, 2, 4]),
    },
    {"scale.0": ("float", 0.01, 0.05), "scale.1": ("float", 0.06, 0.15)},
    {
        "brightness": ("float", 0.05, 0.4),
        "contrast": ("float", 0.05, 0.4),
        "saturation": ("float", 0.05, 0.4),
        "hue": ("float", 0.0, 0.1),
    },
    {
        "brightness_limit": ("float", 0.05, 0.4),
        "contrast_limit": ("float", 0.05, 0.4),
    },
    {"gamma_limit.0": ("int", 50, 95), "gamma_limit.1": ("int", 105, 180)},
    {
        "hue_shift_limit": ("int", 1, 20),
        "sat_shift_limit": ("int", 5, 40),
        "val_shift_limit": ("int", 5, 40),
    },
    {
        "r_shift_limit": ("int", 2, 30),
        "g_shift_limit": ("int", 2, 30),
        "b_shift_limit": ("int", 2, 30),
    },
    {"clip_limit.0": ("float", 1.0, 2.0), "clip_limit.1": ("float", 2.0, 6.0)},
    {"num_output_channels": ("categorical", [3])},
    {},
    {"quality_range.0": ("int", 10, 60), "quality_range.1": ("int", 70, 100)},
    {"scale_range.0": ("float", 0.2, 0.6), "scale_range.1": ("float", 0.65, 0.95)},
    {"blur_limit.0": ("categorical", [3]), "blur_limit.1": ("categorical", [3, 5, 7, 9])},
    {"blur_limit.0": ("categorical", [3]), "blur_limit.1": ("categorical", [5, 7, 9, 11, 15])},
    {
        "radius.0": ("int", 1, 3),
        "radius.1": ("int", 4, 8),
        "alias_blur.0": ("float", 0.05, 0.2),
        "alias_blur.1": ("float", 0.3, 0.8),
    },
    {"std_range.0": ("float", 0.001, 0.03), "std_range.1": ("float", 0.04, 0.15)},
    {"intensity.0": ("float", 0.05, 0.2), "intensity.1": ("float", 0.2, 0.6)},
    {"multiplier.0": ("float", 0.7, 0.98), "multiplier.1": ("float", 1.02, 1.3)},
    {
        "fog_coef_range.0": ("float", 0.01, 0.15),
        "fog_coef_range.1": ("float", 0.16, 0.5),
    },
    {
        "rain_type": ("categorical", ["drizzle", "heavy", "torrential"]),
        "blur_value": ("categorical", [3, 5, 7]),
    },
    {
        "snow_point_range.0": ("float", 0.01, 0.1),
        "snow_point_range.1": ("float", 0.11, 0.4),
        "brightness_coeff": ("float", 0.8, 2.0),
    },
    {
        "num_shadows_limit.0": ("int", 1, 2),
        "num_shadows_limit.1": ("int", 2, 5),
        "shadow_dimension": ("int", 3, 8),
    },
    {"src_radius": ("int", 20, 150)},
    {"alpha.0": ("float", 0.0, 0.3), "alpha.1": ("float", 0.3, 0.8)},
    {
        "num_holes_range.0": ("int", 1, 3),
        "num_holes_range.1": ("int", 4, 12),
        "hole_height_range.0": ("float", 0.02, 0.1),
        "hole_height_range.1": ("float", 0.11, 0.35),
        "hole_width_range.0": ("float", 0.02, 0.1),
        "hole_width_range.1": ("float", 0.11, 0.35),
        "fill": ("categorical", ["random", "random_uniform", 0]),
    },
    {
        "scale.0": ("float", 0.01, 0.08),
        "scale.1": ("float", 0.1, 0.4),
        "ratio.0": ("float", 0.1, 0.5),
        "ratio.1": ("float", 2.0, 5.0),
        "value": ("categorical", ["random", 0]),
    },
    {"ratio": ("float", 0.1, 0.6)},
    {"grid": ("int", 2, 6)},
    {"severity": ("int", 1, 10)},
    {"num_ops": ("int", 1, 5), "magnitude": ("int", 1, 15)},
]

BASE_SPECS: dict[str, tuple[Any, ...]] = {
    "data.context_pct": ("float", 0.0, 30.0),
    "data.context_jitter_pct": ("float", 0.0, 20.0),
    "data.resize_mode": ("categorical", ["pad", "stretch"]),
    "data.sampler.identities": ("categorical", [8, 12, 16, 20]),
    "data.sampler.instances": ("categorical", [2, 3, 4]),
    "data.sampler.camera_diverse": ("bool",),
    "train.epochs": ("int", 10, 50),
    "train.gradient_clip_val": ("float", 0.5, 20.0, True),
    "train.freeze_backbone_epochs": ("int", 0, 5),
    "train.backbone_lr_multiplier": ("float", 0.01, 1.0, True),
    "train.layer_decay": ("float", 0.6, 1.0),
    "train.no_weight_decay_bias_norm": ("bool",),
    "model.drop_path_rate": ("float", 0.0, 0.4),
    "model.pooling.p": ("float", 1.0, 6.0),
    "model.pooling.trainable": ("bool",),
    "model.pooling.attention_hidden": ("categorical", [64, 128, 256, 512]),
    "model.head.embedding_dim": ("categorical", [256, 384, 512, 768, 1024]),
    "model.head.bnneck": ("bool",),
    "model.head.dropout": ("float", 0.0, 0.6),
    "model.head.retrieval_feature": ("categorical", ["raw", "neck"]),
}


def _value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _sample(trial: optuna.Trial, name: str, spec: tuple[Any, ...]) -> Any:
    kind = spec[0]
    if kind == "bool":
        return trial.suggest_categorical(name, [False, True])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    if kind == "int":
        return trial.suggest_int(name, int(spec[1]), int(spec[2]))
    if kind == "float":
        log = bool(spec[3]) if len(spec) > 3 else False
        return trial.suggest_float(name, float(spec[1]), float(spec[2]), log=log)
    raise ValueError(f"Unknown search-space kind {kind}")


def _append(overrides: list[str], path: str, value: Any) -> None:
    overrides.append(f"{path}={_value(value)}")


def _loss_param(loss_name: str, path: str) -> str:
    return f"{loss_name}/{path}"


def suggest_overrides(trial: optuna.Trial, model: str) -> list[str]:
    overrides = []
    for path, spec in BASE_SPECS.items():
        _append(overrides, path, _sample(trial, path, spec))

    image_sizes = [336] if model == "llm2clip" else [224, 256, 288, 320, 384]
    image_size = trial.suggest_categorical("data.image_size", image_sizes)
    _append(overrides, "data.image_size", [image_size, image_size])

    pool_choices = ["gap", "gem", "signed_gem", "max", "avgmax", "attn"]
    if model == "vit":
        pool_choices.append("cls")
    _append(
        overrides,
        "model.pooling.kind",
        trial.suggest_categorical("model.pooling.kind", pool_choices),
    )
    local_parts = 0
    if model not in {"vit", "radio", "llm2clip"}:
        local_parts = trial.suggest_categorical("model.head.local_parts", [0, 2, 3, 4, 6])
    _append(overrides, "model.head.local_parts", local_parts)

    loss_name = trial.suggest_categorical("loss", LOSS_PRESETS)
    overrides.append(f"loss={loss_name}")
    for index in range(LOSS_TERM_COUNTS[loss_name]):
        weight_name = _loss_param(loss_name, f"loss.terms.{index}.weight")
        feature_name = _loss_param(loss_name, f"loss.terms.{index}.feature")
        _append(
            overrides,
            f"loss.terms.{index}.weight",
            trial.suggest_float(weight_name, 0.1, 3.0, log=True),
        )
        _append(
            overrides,
            f"loss.terms.{index}.feature",
            trial.suggest_categorical(feature_name, ["raw", "neck"]),
        )
    for path, spec in LOSS_PARAM_SPECS[loss_name].items():
        _append(overrides, path, _sample(trial, _loss_param(loss_name, path), spec))

    sampler_kind = "pk"
    if loss_name not in ADASP_LOSSES:
        sampler_kind = trial.suggest_categorical("data.sampler.kind", ["pk", "random"])
    _append(overrides, "data.sampler.kind", sampler_kind)

    optimizer = trial.suggest_categorical("optimizer", ["adamw", "sgd", "lamb", "lion"])
    overrides.append(f"optimizer={optimizer}")
    _append(overrides, "optimizer.lr", trial.suggest_float("optimizer.lr", 1e-6, 3e-2, log=True))
    _append(
        overrides,
        "optimizer.weight_decay",
        trial.suggest_float("optimizer.weight_decay", 1e-7, 1e-1, log=True),
    )
    if optimizer == "sgd":
        _append(overrides, "optimizer.momentum", trial.suggest_float("optimizer.momentum", 0.5, 0.99))
        _append(
            overrides,
            "optimizer.nesterov",
            trial.suggest_categorical("optimizer.nesterov", [False, True]),
        )
    else:
        beta1 = trial.suggest_float("optimizer.beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("optimizer.beta2", 0.95, 0.9999)
        _append(overrides, "optimizer.betas", [beta1, beta2])
        if optimizer != "lion":
            _append(
                overrides,
                "optimizer.eps",
                trial.suggest_float("optimizer.eps", 1e-10, 1e-5, log=True),
            )

    _append(
        overrides,
        "scheduler.kind",
        trial.suggest_categorical("scheduler.kind", ["cosine", "linear", "multistep", "constant"]),
    )
    _append(overrides, "scheduler.warmup_epochs", trial.suggest_int("scheduler.warmup_epochs", 0, 8))
    _append(
        overrides,
        "scheduler.warmup_start_factor",
        trial.suggest_float("scheduler.warmup_start_factor", 1e-4, 0.5, log=True),
    )
    _append(
        overrides,
        "scheduler.min_lr_ratio",
        trial.suggest_float("scheduler.min_lr_ratio", 1e-4, 0.5, log=True),
    )
    milestone_a = trial.suggest_int("scheduler.milestone_1", 5, 55)
    milestone_b = trial.suggest_int("scheduler.milestone_2", 5, 55)
    first_milestone, second_milestone = sorted([milestone_a, milestone_b])
    if first_milestone == second_milestone:
        second_milestone += 1
    _append(overrides, "scheduler.milestones", [first_milestone, second_milestone])
    _append(overrides, "scheduler.gamma", trial.suggest_float("scheduler.gamma", 0.05, 0.8))

    rdrop_enabled = False
    if loss_name in CLASSIFIER_LOSSES:
        rdrop_enabled = trial.suggest_categorical("train.rdrop.enabled", [False, True])
    _append(overrides, "train.rdrop.enabled", rdrop_enabled)
    _append(
        overrides,
        "train.rdrop.weight",
        trial.suggest_float("train.rdrop.weight", 0.01, 2.0, log=True),
    )
    _append(
        overrides,
        "train.rdrop.temperature",
        trial.suggest_float("train.rdrop.temperature", 0.3, 3.0, log=True),
    )

    awp_enabled = trial.suggest_categorical("train.awp.enabled", [False, True])
    _append(overrides, "train.awp.enabled", awp_enabled)
    accumulate = 1
    if not awp_enabled:
        accumulate = trial.suggest_categorical("train.accumulate_grad_batches", [1, 2, 4])
    _append(overrides, "train.accumulate_grad_batches", accumulate)
    _append(overrides, "train.awp.start_epoch", trial.suggest_int("train.awp.start_epoch", 0, 15))
    _append(overrides, "train.awp.lr", trial.suggest_float("train.awp.lr", 1e-4, 0.1, log=True))
    _append(overrides, "train.awp.eps", trial.suggest_float("train.awp.eps", 1e-5, 0.02, log=True))
    _append(
        overrides,
        "train.awp.weight",
        trial.suggest_float("train.awp.weight", 0.01, 2.0, log=True),
    )
    _append(overrides, "train.awp.parameter_pattern", "weight")

    ema_enabled = trial.suggest_categorical("train.ema.enabled", [False, True])
    _append(overrides, "train.ema.enabled", ema_enabled)
    _append(overrides, "train.ema.decay", trial.suggest_float("train.ema.decay", 0.9, 0.99999))
    _append(
        overrides,
        "train.ema.validate",
        trial.suggest_categorical("train.ema.validate", [False, True]),
    )

    for index, params in enumerate(TRANSFORM_PARAM_SPECS):
        _append(
            overrides,
            f"augmentation.transforms.{index}.enabled",
            trial.suggest_categorical(f"augmentation.transforms.{index}.enabled", [False, True]),
        )
        _append(
            overrides,
            f"augmentation.transforms.{index}.p",
            trial.suggest_float(f"augmentation.transforms.{index}.p", 0.0, 0.8),
        )
        for suffix, spec in params.items():
            path = f"augmentation.transforms.{index}.params.{suffix}"
            _append(overrides, path, _sample(trial, path, spec))

    tta_enabled = trial.suggest_categorical("eval.tta.enabled", [False, True])
    _append(overrides, "eval.tta.enabled", tta_enabled)
    _append(
        overrides,
        "eval.tta.hflip",
        trial.suggest_categorical("eval.tta.hflip", [False, True]),
    )
    scale_delta = trial.suggest_float("eval.tta.scale_delta", 0.0, 0.2)
    if model == "llm2clip":
        _append(overrides, "eval.tta.scales", [1.0])
    else:
        _append(overrides, "eval.tta.scales", [1.0 - scale_delta, 1.0, 1.0 + scale_delta])
    rotation = trial.suggest_int("eval.tta.rotation", 0, 12)
    _append(overrides, "eval.tta.rotations", [-rotation, 0, rotation])
    tta_context = trial.suggest_float("eval.tta.context_pct", 0.0, 30.0)
    _append(overrides, "eval.tta.context_pcts", [0.0, tta_context])
    eval_weights = "raw"
    if ema_enabled:
        eval_weights = trial.suggest_categorical("eval.weights", ["raw", "ema"])
    _append(overrides, "eval.weights", eval_weights)
    postproc_enabled = trial.suggest_categorical("postproc.enabled", [False, True])
    _append(overrides, "postproc.enabled", postproc_enabled)
    _append(
        overrides,
        "postproc.aqe.enabled",
        trial.suggest_categorical("postproc.aqe.enabled", [False, True]),
    )
    _append(overrides, "postproc.aqe.k", trial.suggest_int("postproc.aqe.k", 1, 20))
    _append(
        overrides,
        "postproc.aqe.alpha",
        trial.suggest_float("postproc.aqe.alpha", 0.0, 5.0),
    )
    _append(
        overrides,
        "postproc.aqe.iterations",
        trial.suggest_int("postproc.aqe.iterations", 1, 3),
    )
    _append(
        overrides,
        "postproc.aqe.gallery_only",
        trial.suggest_categorical("postproc.aqe.gallery_only", [False, True]),
    )
    _append(
        overrides,
        "postproc.gallery_aggregation.enabled",
        trial.suggest_categorical("postproc.gallery_aggregation.enabled", [False, True]),
    )
    _append(
        overrides,
        "postproc.gallery_aggregation.k",
        trial.suggest_int("postproc.gallery_aggregation.k", 1, 20),
    )
    _append(
        overrides,
        "postproc.gallery_aggregation.alpha",
        trial.suggest_float("postproc.gallery_aggregation.alpha", 0.0, 1.0),
    )
    _append(
        overrides,
        "postproc.gallery_aggregation.min_similarity",
        trial.suggest_float("postproc.gallery_aggregation.min_similarity", 0.5, 0.99),
    )
    _append(
        overrides,
        "postproc.rerank.kind",
        trial.suggest_categorical("postproc.rerank.kind", ["none", "k_reciprocal", "gnn"]),
    )
    _append(overrides, "postproc.rerank.k1", trial.suggest_int("postproc.rerank.k1", 5, 50))
    _append(overrides, "postproc.rerank.k2", trial.suggest_int("postproc.rerank.k2", 1, 20))
    _append(
        overrides,
        "postproc.rerank.lambda_value",
        trial.suggest_float("postproc.rerank.lambda_value", 0.0, 1.0),
    )
    _append(overrides, "postproc.rerank.device", "cpu")
    return overrides


def _config_value(cfg: DictConfig | ListConfig, path: str) -> Any:
    value: Any = cfg
    for component in path.split("."):
        value = value[int(component)] if component.isdigit() else value[component]
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _loss_name(cfg: DictConfig) -> str:
    terms = cfg.loss.terms
    names = [str(term.name) for term in terms]
    if names == ["arcface", "triplet"]:
        return "combined"
    if names == ["arcface", "adasp"]:
        return "arcface_adasp"
    if names == ["ce", "triplet"]:
        return "ce_triplet"
    if names != ["pml"]:
        return names[0]
    target = str(terms[0].target).rsplit(".", maxsplit=1)[-1]
    miner = str(terms[0].get("miner", {}).get("_target_", "")).rsplit(".", maxsplit=1)[-1]
    pml_names = {
        ("TripletMarginLoss", "TripletMarginMiner"): "triplet_semihard",
        ("TripletMarginLoss", "DistanceWeightedMiner"): "triplet_distanceweighted",
        ("CircleLoss", ""): "circle",
        ("MultiSimilarityLoss", "MultiSimilarityMiner"): "multisimilarity",
        ("SupConLoss", ""): "supcon",
        ("ProxyAnchorLoss", ""): "proxyanchor",
        ("ContrastiveLoss", "PairMarginMiner"): "contrastive",
        ("NTXentLoss", ""): "ntxent",
        ("ProxyNCALoss", ""): "proxynca",
        ("SoftTripleLoss", ""): "softtriple",
        ("FastAPLoss", ""): "fastap",
        ("GeneralizedLiftedStructureLoss", ""): "lifted",
    }
    return pml_names[(target, miner)]


def _optimizer_name(cfg: DictConfig) -> str:
    target = str(cfg.optimizer._target_).rsplit(".", maxsplit=1)[-1].lower()
    return {"adamw": "adamw", "sgd": "sgd", "lamb": "lamb", "lion": "lion"}[target]


def _seed_eval_weights(cfg: DictConfig, path: Path) -> str:
    choice = str(cfg.eval.weights)
    if choice != "auto":
        return choice
    metrics_path = path.parent / "val" / "metrics.json"
    if metrics_path.is_file():
        recorded = json.loads(metrics_path.read_text()).get("weights")
        if recorded in {"raw", "ema"}:
            return str(recorded)
    return "ema" if bool(cfg.train.ema.validate) else "raw"


def params_from_config(path: Path, model: str) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError("Seed configuration must be a mapping")
    cfg = loaded
    params = {name: _config_value(cfg, name) for name in BASE_SPECS}
    params["data.image_size"] = int(cfg.data.image_size[0])
    params["model.pooling.kind"] = str(cfg.model.pooling.kind)
    if model not in {"vit", "radio", "llm2clip"}:
        params["model.head.local_parts"] = int(cfg.model.head.local_parts)

    loss_name = _loss_name(cfg)
    params["loss"] = loss_name
    for index in range(LOSS_TERM_COUNTS[loss_name]):
        params[_loss_param(loss_name, f"loss.terms.{index}.weight")] = float(cfg.loss.terms[index].weight)
        params[_loss_param(loss_name, f"loss.terms.{index}.feature")] = str(cfg.loss.terms[index].feature)
    for name in LOSS_PARAM_SPECS[loss_name]:
        params[_loss_param(loss_name, name)] = _config_value(cfg, name)
    if loss_name not in ADASP_LOSSES:
        params["data.sampler.kind"] = str(cfg.data.sampler.kind)

    optimizer = _optimizer_name(cfg)
    params["optimizer"] = optimizer
    params["optimizer.lr"] = float(cfg.optimizer.lr)
    params["optimizer.weight_decay"] = float(cfg.optimizer.weight_decay)
    if optimizer == "sgd":
        params["optimizer.momentum"] = float(cfg.optimizer.momentum)
        params["optimizer.nesterov"] = bool(cfg.optimizer.nesterov)
    else:
        params["optimizer.beta1"] = float(cfg.optimizer.betas[0])
        params["optimizer.beta2"] = float(cfg.optimizer.betas[1])
        if optimizer != "lion":
            params["optimizer.eps"] = float(cfg.optimizer.eps)

    params["scheduler.kind"] = str(cfg.scheduler.kind)
    params["scheduler.warmup_epochs"] = int(cfg.scheduler.warmup_epochs)
    params["scheduler.warmup_start_factor"] = float(cfg.scheduler.warmup_start_factor)
    params["scheduler.min_lr_ratio"] = float(cfg.scheduler.min_lr_ratio)
    params["scheduler.milestone_1"] = int(cfg.scheduler.milestones[0])
    params["scheduler.milestone_2"] = int(cfg.scheduler.milestones[1])
    params["scheduler.gamma"] = float(cfg.scheduler.gamma)

    if loss_name in CLASSIFIER_LOSSES:
        params["train.rdrop.enabled"] = bool(cfg.train.rdrop.enabled)
    params["train.rdrop.weight"] = float(cfg.train.rdrop.weight)
    params["train.rdrop.temperature"] = float(cfg.train.rdrop.temperature)
    params["train.awp.enabled"] = bool(cfg.train.awp.enabled)
    if not cfg.train.awp.enabled:
        params["train.accumulate_grad_batches"] = int(cfg.train.accumulate_grad_batches)
    params["train.awp.start_epoch"] = int(cfg.train.awp.start_epoch)
    params["train.awp.lr"] = float(cfg.train.awp.lr)
    params["train.awp.eps"] = float(cfg.train.awp.eps)
    params["train.awp.weight"] = float(cfg.train.awp.weight)
    params["train.ema.enabled"] = bool(cfg.train.ema.enabled)
    params["train.ema.decay"] = float(cfg.train.ema.decay)
    params["train.ema.validate"] = bool(cfg.train.ema.validate)

    for index, specs in enumerate(TRANSFORM_PARAM_SPECS):
        prefix = f"augmentation.transforms.{index}"
        params[f"{prefix}.enabled"] = bool(cfg.augmentation.transforms[index].enabled)
        params[f"{prefix}.p"] = float(cfg.augmentation.transforms[index].p)
        for suffix in specs:
            name = f"{prefix}.params.{suffix}"
            params[name] = _config_value(cfg, name)

    params["eval.tta.enabled"] = bool(cfg.eval.tta.enabled)
    params["eval.tta.hflip"] = bool(cfg.eval.tta.hflip)
    scales = list(cfg.eval.tta.scales)
    params["eval.tta.scale_delta"] = max(abs(float(scale) - 1.0) for scale in scales)
    rotations = list(cfg.eval.tta.rotations)
    params["eval.tta.rotation"] = max(abs(int(rotation)) for rotation in rotations)
    contexts = cfg.eval.tta.context_pcts
    params["eval.tta.context_pct"] = 0.0 if contexts is None else max(float(value) for value in contexts)
    if cfg.train.ema.enabled:
        params["eval.weights"] = _seed_eval_weights(cfg, path)

    for name in (
        "postproc.enabled",
        "postproc.aqe.enabled",
        "postproc.aqe.k",
        "postproc.aqe.alpha",
        "postproc.aqe.iterations",
        "postproc.aqe.gallery_only",
        "postproc.gallery_aggregation.enabled",
        "postproc.gallery_aggregation.k",
        "postproc.gallery_aggregation.alpha",
        "postproc.gallery_aggregation.min_similarity",
        "postproc.rerank.kind",
        "postproc.rerank.k1",
        "postproc.rerank.k2",
        "postproc.rerank.lambda_value",
    ):
        params[name] = _config_value(cfg, name)
    return params
