import torch

from .common import FeatureHook, Unfrozen, as_nchw, embedding_score, prefix_tokens, upsample


def _maps(cam, size):
    return upsample(cam.float().clamp_min(0), size).detach().cpu().numpy()


def _activations(model, images, reference, need_grad):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected NCHW RGB images")
    with Unfrozen(model), FeatureHook(model) as hook:
        images = images.detach().requires_grad_(bool(need_grad))
        out = model(images)
        feat = hook.feature
        if feat is None:
            raise RuntimeError("Backbone produced no hooked features")
        spatial = as_nchw(feat, prefix_tokens(model))
        if need_grad:
            score = embedding_score(out, reference)
            grad = torch.autograd.grad(score, feat, retain_graph=False, allow_unused=True)[0]
            if grad is None:
                raise RuntimeError("No gradient reached backbone features")
            return spatial, as_nchw(grad, prefix_tokens(model)), images.shape[-2:]
        return spatial, None, images.shape[-2:]


def grad_cam(model, images, reference=None):
    act, grad, size = _activations(model, images, reference, True)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    return _maps((weights * act).sum(1), size)


def hires_cam(model, images, reference=None):
    act, grad, size = _activations(model, images, reference, True)
    return _maps((grad * act).sum(1), size)


def layer_cam(model, images, reference=None):
    act, grad, size = _activations(model, images, reference, True)
    return _maps((grad.clamp_min(0) * act).sum(1), size)


def grad_sim(model, images, reference=None):
    act, grad, size = _activations(model, images, reference, True)
    return _maps((grad * act).abs().sum(1), size)


def eigen_cam(model, images, reference=None):
    act, _, size = _activations(model, images, reference, False)
    batch, _, height, width = act.shape
    flat = act.reshape(batch, act.shape[1], height * width)
    components = []
    for item in flat:
        _, _, vh = torch.linalg.svd(item.float(), full_matrices=False)
        components.append(vh[0].abs().reshape(height, width))
    return _maps(torch.stack(components), size)
