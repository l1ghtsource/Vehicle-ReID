import numpy as np
import pytest
from omegaconf import OmegaConf

import postproc.retrieval as retrieval
from postproc.retrieval import aggregate_gallery, aqe, gnn_rerank, k_reciprocal, postprocess


def vectors():
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
        dtype=np.float32,
    )
    return query, gallery


def config():
    return OmegaConf.create(
        {
            "enabled": True,
            "max_dense_gb": 1,
            "gallery_aggregation": {
                "enabled": False,
                "k": 2,
                "alpha": 0.5,
                "min_similarity": 0.9,
            },
            "aqe": {
                "enabled": False,
                "k": 2,
                "alpha": 1,
                "iterations": 1,
                "gallery_only": True,
            },
            "rerank": {
                "kind": "none",
                "k1": 2,
                "k2": 1,
                "lambda_value": 0.3,
                "device": "cpu",
            },
        }
    )


def test_aqe_and_aggregation_validation():
    query, gallery = vectors()
    for kwargs in ({"k": 0}, {"iterations": 0}, {"alpha": -1}):
        with pytest.raises(ValueError, match="AQE"):
            aqe(query, gallery, **kwargs)
    expanded, _ = aqe(query, gallery, gallery_only=False, k=5)
    assert expanded.shape == query.shape
    with pytest.raises(ValueError, match="aggregation"):
        aggregate_gallery(gallery, alpha=2)
    aggregated = aggregate_gallery(gallery, k=2, min_similarity=0.8)
    assert not np.array_equal(aggregated[0], gallery[0])


def test_reranker_validation_and_nonfinite(monkeypatch):
    query, gallery = vectors()
    with pytest.raises(ValueError, match="k-reciprocal"):
        k_reciprocal(query, gallery, k1=0)
    monkeypatch.setattr(
        retrieval,
        "re_ranking",
        lambda *args, **kwargs: np.full((len(query), len(gallery)), np.nan),
    )
    with pytest.raises(FloatingPointError):
        k_reciprocal(query, gallery, k1=2, k2=1)
    with pytest.raises(ValueError, match="GNN"):
        gnn_rerank(query, gallery, k1=0)
    assert gnn_rerank(query, gallery, k1=2, k2=1).shape == (2, 3)


def test_postprocess_all_orchestration(monkeypatch):
    query, gallery = vectors()
    cfg = config()
    cfg.enabled = False
    distance, expanded_q, expanded_g = postprocess(query, gallery, cfg)
    assert distance.shape == (2, 3)
    assert expanded_q.shape == query.shape
    assert expanded_g.shape == gallery.shape

    cfg.enabled = True
    cfg.gallery_aggregation.enabled = True
    cfg.aqe.enabled = True
    for kind in ("none", "k_reciprocal", "gnn"):
        cfg.rerank.kind = kind
        if kind == "k_reciprocal":
            monkeypatch.setattr(
                retrieval,
                "re_ranking",
                lambda *args, **kwargs: np.zeros((len(query), len(gallery))),
            )
        assert postprocess(query, gallery, cfg)[0].shape == (2, 3)
    cfg.rerank.kind = "bad"
    with pytest.raises(ValueError, match="Unknown reranker"):
        postprocess(query, gallery, cfg)
