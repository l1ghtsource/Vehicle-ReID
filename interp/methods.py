from .attention import attention_rollout, chefer_attribution, last_attention
from .cam import eigen_cam, grad_cam, hires_cam, layer_cam
from .pooling import pooling_attention

METHODS = {
    "pooling": pooling_attention,
    "last_attn": last_attention,
    "rollout": attention_rollout,
    "chefer": chefer_attribution,
    "gradcam": grad_cam,
    "hirescam": hires_cam,
    "layercam": layer_cam,
    "eigencam": eigen_cam,
}


def interpret(model, images, method="gradcam", reference=None):
    if method not in METHODS:
        raise ValueError(f"Unknown interpretation method {method}")
    return METHODS[method](model, images, reference=reference)
