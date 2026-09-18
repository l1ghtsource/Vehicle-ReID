import torch

from .common import AttentionCapture, Unfrozen, embedding_score, prefix_tokens, upsample


def _cls_maps(joint, prefix, size):
    if joint.ndim != 3:
        raise ValueError("Expected BNN attention rollout")
    spatial = joint[:, 0, int(prefix) :]
    if spatial.numel() == 0:
        raise ValueError("No spatial tokens after stripping prefix tokens")
    tokens = spatial.shape[1]
    side = int(tokens**0.5)
    height, width = (side, side) if side * side == tokens else (1, tokens)
    maps = spatial.reshape(len(spatial), height, width)
    return upsample(maps.float().clamp_min(0), size).detach().cpu().numpy()


def _forward_attentions(model, images, reference, with_score):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected NCHW RGB images")
    with Unfrozen(model), AttentionCapture() as capture:
        images = images.detach().requires_grad_(bool(with_score))
        out = model(images)
        if not capture.maps:
            raise ValueError("No 4D token-attention softmax found; this backbone is not a ViT")
        score = embedding_score(out, reference) if with_score else None
        return capture.maps, score, images.shape[-2:]


def last_attention(model, images, reference=None):
    maps, _, size = _forward_attentions(model, images, reference, False)
    attn = maps[-1].mean(1)
    return _cls_maps(attn, prefix_tokens(model), size)


def attention_rollout(model, images, reference=None):
    maps, _, size = _forward_attentions(model, images, reference, False)
    tokens = maps[0].shape[-1]
    joint = None
    eye = None
    for attn in maps:
        mix = attn.mean(1)
        if eye is None:
            eye = torch.eye(tokens, device=mix.device, dtype=mix.dtype).expand(len(mix), tokens, tokens)
            joint = eye
        residual = 0.5 * mix + 0.5 * eye
        joint = residual @ joint
    return _cls_maps(joint, prefix_tokens(model), size)


def chefer_attribution(model, images, reference=None):
    maps, score, size = _forward_attentions(model, images, reference, True)
    tokens = maps[0].shape[-1]
    eye = torch.eye(tokens, device=maps[0].device, dtype=maps[0].dtype)
    eye = eye.expand(maps[0].shape[0], tokens, tokens)
    relevancy = eye
    for attn in maps:
        grad = torch.autograd.grad(score, attn, retain_graph=True)[0]
        cam = (grad * attn).clamp_min(0).mean(1)
        cam = cam / cam.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        relevancy = relevancy + cam @ relevancy
    return _cls_maps(relevancy, prefix_tokens(model), size)
