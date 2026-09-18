from torch.nn import functional as F

from .attention import attention_rollout, chefer_attribution, grad_rollout, last_attention
from .cam import eigen_cam, grad_cam, grad_sim, hires_cam, layer_cam
from .common import Unfrozen
from .occlusion import occlusion
from .pooling import pooling_attention

METHODS = {
    "pooling": pooling_attention,
    "last_attn": last_attention,
    "rollout": attention_rollout,
    "grad_rollout": grad_rollout,
    "chefer": chefer_attribution,
    "gradsim": grad_sim,
    "gradcam": grad_cam,
    "hirescam": hires_cam,
    "layercam": layer_cam,
    "eigencam": eigen_cam,
    "occlusion": occlusion,
}

METHOD_KWARGS = {
    "occlusion": {"block"},
    "pooling": {"level"},
}


def _call(method, model, images, reference=None, **kwargs):
    extra = {key: value for key, value in kwargs.items() if key in METHOD_KWARGS.get(method, set())}
    return METHODS[method](model, images, reference=reference, **extra)


def interpret(model, images, method="gradcam", reference=None, **kwargs):
    if method not in METHODS:
        raise ValueError(f"Unknown interpretation method {method}")
    return _call(method, model, images, reference=reference, **kwargs)


def embed_images(model, images):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected NCHW RGB images")
    with Unfrozen(model):
        out = model(images)
    emb = out["embedding"] if isinstance(out, dict) else out
    return F.normalize(emb.float(), dim=1)


def interpret_pair(model, query, gallery, method="gradsim", **kwargs):
    if method not in METHODS:
        raise ValueError(f"Unknown interpretation method {method}")
    query_emb = embed_images(model, query).detach()
    gallery_emb = embed_images(model, gallery).detach()
    if len(query_emb) != len(gallery_emb):
        raise ValueError("query/gallery batch sizes must match")
    cosine = (query_emb * gallery_emb).sum(-1)
    query_maps = _call(method, model, query, reference=gallery_emb, **kwargs)
    gallery_maps = _call(method, model, gallery, reference=query_emb, **kwargs)
    return query_maps, gallery_maps, cosine.detach().cpu().numpy()
