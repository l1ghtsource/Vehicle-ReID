import numpy as np

from .boosting import load_boosting, predict_boosting
from .features import batch_examples
from .threshold import max_cosine


def _queries(query):
    q = np.asarray(query, dtype=np.float32)
    if q.ndim == 1:
        q = q[None]
    if q.ndim != 2 or q.shape[0] < 1:
        raise ValueError("Expected a nonempty query embedding matrix")
    return q


def _require_model(model, model_path):
    if model is not None:
        return model
    if model_path is None or str(model_path).strip() in {"", "null", "None"}:
        raise ValueError("refusal.model_path is required for model/ensemble refusal")
    return load_boosting(model_path)


def refusal_accept(
    kind,
    query,
    gallery,
    *,
    cosine_threshold=None,
    model=None,
    model_path=None,
    model_threshold=None,
    rank_threshold=None,
    k=10,
    with_embeddings=True,
):
    q = _queries(query)
    kind = "none" if kind is None else str(kind).strip().lower()
    if kind in {"none", "off"}:
        return np.ones(len(q), dtype=bool)
    if kind == "threshold":
        if cosine_threshold is None:
            raise ValueError("refusal.cosine_threshold is required for threshold refusal")
        return max_cosine(q, gallery) >= float(cosine_threshold)
    if kind == "model":
        if model_threshold is None:
            raise ValueError("refusal.model_threshold is required for model refusal")
        fitted = _require_model(model, model_path)
        features = batch_examples(q, gallery, k=int(k), with_embeddings=bool(with_embeddings))
        return predict_boosting(fitted, features) >= float(model_threshold)
    if kind == "ensemble":
        if cosine_threshold is None:
            raise ValueError("refusal.cosine_threshold is required for ensemble refusal")
        if model_threshold is None:
            raise ValueError("refusal.model_threshold is required for ensemble refusal")
        fitted = _require_model(model, model_path)
        features = batch_examples(q, gallery, k=int(k), with_embeddings=bool(with_embeddings))
        cosine_ok = max_cosine(q, gallery) >= float(cosine_threshold)
        model_ok = predict_boosting(fitted, features) >= float(model_threshold)
        return cosine_ok & model_ok
    raise ValueError("refusal.kind must be none, threshold, model, or ensemble")
