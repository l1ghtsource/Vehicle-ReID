import torch

from .common import FeatureHook, Unfrozen, as_nchw, embedding_scores, prefix_tokens, upsample


def occlusion(model, images, reference=None, block=1):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected NCHW RGB images")
    step = int(block)
    if step < 1:
        raise ValueError("occlusion block must be >= 1")
    images = images.detach()
    with Unfrozen(model), FeatureHook(model) as hook:
        out = model(images)
        feat = hook.feature
        if feat is None:
            raise RuntimeError("Backbone produced no hooked features")
        spatial = as_nchw(feat, prefix_tokens(model) if feat.ndim != 4 else 0)
        grid_h, grid_w = int(spatial.shape[2]), int(spatial.shape[3])
        base = embedding_scores(out, reference).detach()
    height, width = int(images.shape[-2]), int(images.shape[-1])
    rows = (grid_h + step - 1) // step
    cols = (grid_w + step - 1) // step
    drops = torch.zeros(len(images), rows, cols, dtype=torch.float32)
    with Unfrozen(model), torch.inference_mode():
        for row in range(rows):
            r0 = row * step
            r1 = min(grid_h, r0 + step)
            y0 = r0 * height // grid_h
            y1 = max(y0 + 1, r1 * height // grid_h)
            for col in range(cols):
                c0 = col * step
                c1 = min(grid_w, c0 + step)
                x0 = c0 * width // grid_w
                x1 = max(x0 + 1, c1 * width // grid_w)
                masked = images.clone()
                masked[:, :, y0:y1, x0:x1] = 0
                scores = embedding_scores(model(masked), reference)
                drops[:, row, col] = (base - scores).float().cpu()
    return upsample(drops, images.shape[-2:]).numpy()
