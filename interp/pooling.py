from .common import FeatureHook, Unfrozen, as_nchw, prefix_tokens, upsample


def pooling_attention(model, images, reference=None, level=-1):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected NCHW RGB images")
    pools = getattr(model, "pools", None)
    if not pools:
        raise ValueError("Model has no pooling modules")
    with Unfrozen(model), FeatureHook(model) as hook:
        model(images)
        levels = hook.levels
        if not levels:
            raise RuntimeError("Backbone produced no hooked features")
    try:
        feat = levels[level]
        pool = pools[level]
    except IndexError as error:
        raise ValueError(f"Pooling level {level} is out of range for {len(pools)} modules") from error
    spatial = as_nchw(feat, prefix_tokens(model) if feat.ndim != 4 else 0)
    if pool.kind == "attn":
        tokens = spatial.flatten(2).transpose(1, 2)
        weights = pool.attention(tokens).softmax(1).squeeze(-1)
        cam = weights.reshape(len(weights), spatial.shape[2], spatial.shape[3])
    elif pool.kind == "max":
        cam = spatial.amax(1)
    else:
        cam = spatial.mean(1).abs()
    return upsample(cam.float().clamp_min(0), images.shape[-2:]).detach().cpu().numpy()
