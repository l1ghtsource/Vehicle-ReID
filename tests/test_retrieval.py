import numpy as np
import pytest
import torch

from modules.metrics import bootstrap_mean_ci, query_retrieval_rows, retrieval_metrics
from postproc.retrieval import aggregate_gallery, aqe, dense_guard, gnn_rerank, k_reciprocal, normalize


def test_metrics_known_answer_and_camera_exclusion():

    d = np.array([[0.0, 0.2, 0.1], [0.4, 0.1, 0.2]])
    m = retrieval_metrics(d, [1, 2], [1, 1, 2], [0, 2], [0, 1, 1])
    assert m["mAP"] == 0.5
    assert m["mAP@10"] == 0.5
    assert m["Rank-1"] == 0.0
    assert m["Rank-5"] == 1.0


def test_same_camera_other_identity_stays_in_gallery():
    distance = np.array([[0.0, 0.1]])
    inflated = retrieval_metrics(
        distance,
        [1],
        [2, 1],
        [0],
        [0, 1],
        exclude_all_same_camera=True,
    )
    official = retrieval_metrics(distance, [1], [2, 1], [0], [0, 1])
    assert inflated["mAP@10"] == 1.0
    assert inflated["Rank-1"] == 1.0
    assert official["mAP@10"] == 0.5
    assert official["Rank-1"] == 0.0


def test_no_positive_is_reported_not_scored_as_perfect():
    m = retrieval_metrics(np.array([[0.1], [0.2]]), [1, 99], [1], cross_camera=False)
    assert m["evaluated_queries"] == 1 and m["queries_without_positive"] == 1
    with pytest.raises(ValueError):
        retrieval_metrics(np.array([[0.1]]), [99], [1], cross_camera=False)


def test_map_at_10_truncates_and_uses_min_npos():
    dist = np.arange(20, dtype=np.float32)[None]
    gids = np.zeros(20, dtype=int)
    gids[0] = 1
    gids[19] = 1
    late = retrieval_metrics(dist, [1], gids, cross_camera=False)
    assert late["mAP"] == pytest.approx(0.55)
    assert late["mAP@10"] == pytest.approx(0.5)

    gids = np.zeros(20, dtype=int)
    gids[:3] = 1
    gids[10:19] = 1
    truncated = retrieval_metrics(dist, [1], gids, cross_camera=False)
    assert truncated["mAP@10"] == pytest.approx(0.3)

    gids = np.zeros(15, dtype=int)
    gids[10:] = 1
    missed = retrieval_metrics(np.arange(15, dtype=np.float32)[None], [1], gids, cross_camera=False)
    assert missed["mAP@10"] == 0.0
    assert missed["mAP"] > 0

    gids = np.ones(12, dtype=int)
    full = retrieval_metrics(np.arange(12, dtype=np.float32)[None], [1], gids, cross_camera=False)
    assert full["mAP"] == 1.0
    assert full["mAP@10"] == 1.0
    rows = query_retrieval_rows(np.array([[0.1], [0.2]]), [1, 99], [1], cross_camera=False)
    assert rows[0]["evaluated"] and not rows[1]["evaluated"]
    assert np.isnan(rows[1]["ap"])
    ci = bootstrap_mean_ci([0.2, 0.4, 0.6], n_boot=50, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_ci([np.nan])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_mean_ci([1.0], n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_mean_ci([1.0], alpha=1)


def test_reranking_and_memory_guard():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(4, 8)).astype("float32")
    g = np.concatenate([q + 0.01, rng.normal(size=(4, 8)).astype("float32")])
    for f in [k_reciprocal, gnn_rerank]:
        d = f(q, g, k1=4, k2=2)
        assert d.shape == (4, 8) and np.isfinite(d).all()
    assert np.isfinite(k_reciprocal(np.ones((2, 8)), np.ones((3, 8)))).all()
    with pytest.raises(MemoryError):
        dense_guard(100000, 0.1)
    a, b = aqe(q, g)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1, atol=1e-6)
    assert aggregate_gallery(g).shape == g.shape


def test_gnn_matches_explicit_author_kernel():
    rng = np.random.default_rng(7)
    q, g = normalize(rng.normal(size=(2, 5))), normalize(rng.normal(size=(6, 5)))
    x = torch.from_numpy(np.concatenate([q, g]))
    s, idx = (x @ x.T).topk(4, dim=1)
    a = torch.zeros(8, 8).scatter_(1, idx, 1.0)
    for _ in range(2):
        sym = a + a.T
        rows = []
        for i in range(8):
            row = torch.zeros_like(sym[0])
            for j in range(2):
                row += sym[idx[i, j]] * s[i, j] ** 2
            rows.append(row)
        a = torch.stack(rows)
        a = torch.nn.functional.normalize(a, dim=1)
    expected = 1 - (0.7 * (a[:2] @ a[2:].T) + 0.3 * torch.from_numpy(q @ g.T))
    np.testing.assert_allclose(gnn_rerank(q, g, k1=4, k2=2), expected.numpy(), atol=1e-6)
