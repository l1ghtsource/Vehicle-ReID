import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from posthoc import (
    CORRUPTIONS,
    apply,
    apply_k,
    compare_query,
    embed,
    embedding_stability,
    l2_normalize,
    overlap_curve,
    pixel_shift,
    ranked_indices,
    retrieval_stability,
    score_queries,
    summarize,
    topk,
)
from posthoc.protocol import _mean_fields
from posthoc.retrieval import average_precision, kendall_top


class TinyEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        raw = self.conv(x).mean((2, 3))
        return {"embedding": F.normalize(raw, dim=1)}


class TensorEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return F.normalize(x.mean((2, 3)) + self.bias, dim=1)


class BareEmbed(nn.Module):
    def forward(self, x):
        return {"embedding": F.normalize(x.mean((2, 3)), dim=1)}


class InfEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return {"embedding": torch.full((len(x), 4), torch.inf)}


def rgb(h=16, w=20, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_corruptions_identity_repro_and_guards():
    image = rgb()
    copied = apply(image, "identity", severity=2, seed=0)
    np.testing.assert_array_equal(copied, image)
    assert copied is not image
    for name in CORRUPTIONS:
        for severity in (1, 5):
            a = apply(image, name, severity, seed=0)
            b = apply(image, name, severity, seed=0)
            assert a.shape == image.shape and a.dtype == np.uint8
            np.testing.assert_array_equal(a, b)
        stacked = apply_k(image, name, severity=3, k=3, seed=4)
        assert stacked.shape == (3, *image.shape)
    noisy = apply_k(image, "gaussian_noise", severity=5, k=2, seed=1)
    assert not np.array_equal(noisy[0], noisy[1])
    assert not np.array_equal(apply(image, "jpeg", 5, seed=0), image)
    tiny = rgb(2, 2)
    for name in CORRUPTIONS:
        assert apply(tiny, name, 5, seed=3).shape == tiny.shape
    for seed in range(12):
        apply(image, "motion_blur", 3, seed=seed)
        apply(image, "brightness", 4, seed=seed)
        apply(image, "contrast", 2, seed=seed)
        apply(image, "saturate", 1, seed=seed)
        apply(image, "occlude", 5, seed=seed)
        apply(image, "rotate", 5, seed=seed)
        apply(image, "crop", 5, seed=seed)
        apply(image, "fog", 5, seed=seed)
    with pytest.raises(ValueError, match="Unknown"):
        apply(image, "banana")
    with pytest.raises(ValueError, match="severity"):
        apply(image, "jpeg", 0)
    with pytest.raises(ValueError, match="severity"):
        apply(image, "jpeg", 6)
    with pytest.raises(ValueError, match="HWC"):
        apply(np.zeros((16, 20), dtype=np.uint8), "jpeg")
    with pytest.raises(ValueError, match="uint8"):
        apply(np.zeros((16, 20, 3), dtype=np.float32), "jpeg")
    with pytest.raises(ValueError, match="k must"):
        apply_k(image, "jpeg", k=0)


def test_embedding_and_pixel_metrics():
    orig = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    same = embedding_stability(orig, orig)
    assert same["n"] == 1 and same["cosine_mean"] == pytest.approx(1)
    assert same["pairwise_cosine"] == 1.0
    corr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    stats = embedding_stability(orig * 4, corr * 3)
    assert stats["n"] == 2
    assert stats["cosine_min"] == pytest.approx(0, abs=1e-5)
    assert 0 < stats["cosine_mean"] < 1
    assert stats["angular_mean_deg"] > 0
    shift = pixel_shift(np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 255, np.uint8))
    assert shift["l1"] == pytest.approx(1)
    assert shift["psnr"] < 10
    with pytest.raises(ValueError, match="empty"):
        l2_normalize(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="matching"):
        embedding_stability(orig, np.ones((2, 2)))
    with pytest.raises(ValueError, match="matching"):
        embedding_stability(orig, np.zeros((0, 3)))
    with pytest.raises(ValueError, match="matching"):
        embedding_stability(orig, np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="mismatch"):
        pixel_shift(np.zeros((2, 2, 3)), np.zeros((3, 2, 3)))


def test_retrieval_stability_and_curves():
    orig = np.array([1.0, 0.8, 0.2, 0.0])
    same = retrieval_stability(orig, orig, k=2, qid=1, gids=[1, 1, 2, 3])
    assert same["overlap"] == 1 and same["rank1_image_same"]
    assert same["orig_r1_rank"] == 1 and same["delta_ap"] == 0
    assert same["has_positive"] and same["rank1_id_same"]
    flipped = retrieval_stability(orig, orig[::-1], k=1, keep=np.array([True, True, True, True]))
    assert flipped["overlap"] == 0 and flipped["orig_r1_rank"] == 4
    none = retrieval_stability(orig, orig, k=3, qid=99, gids=[1, 2, 3, 4])
    assert none["has_positive"] is False and np.isnan(none["delta_mean_pos_rank"])
    assert none["ap_orig"] == 0
    unlabeled = retrieval_stability(orig, orig, k=10)
    assert "ap_orig" not in unlabeled and unlabeled["k"] == 4
    order = ranked_indices(orig)
    curve = overlap_curve(order, order, ks=(1, 2, 8))
    assert curve[1] == 1 and curve[8] == 1
    assert average_precision(np.array([False, False])) == 0
    assert kendall_top(np.array([0]), np.array([0]), 1) == 1
    assert kendall_top(np.array([0, 0]), np.array([0, 0]), 2) == 1
    assert topk(orig, 2).tolist() == [0, 1]
    with pytest.raises(ValueError, match="keep mask must match"):
        ranked_indices(orig, keep=np.array([True]))
    with pytest.raises(ValueError, match="empty"):
        ranked_indices(orig, keep=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="k must"):
        retrieval_stability(orig, orig, k=0)
    with pytest.raises(ValueError, match="k must"):
        topk(orig, 0)
    with pytest.raises(ValueError, match="empty"):
        overlap_curve(np.array([]), np.array([]), ks=(1,))
    with pytest.raises(ValueError, match="k must"):
        overlap_curve(order, order, ks=(0,))


def test_kendall_nonfinite(monkeypatch):
    monkeypatch.setattr("posthoc.retrieval.kendalltau", lambda *args, **kwargs: (np.nan, np.nan))
    assert kendall_top(np.arange(4), np.arange(4)[::-1], 4) == 1.0


def test_protocol_compare_score_summarize_embed():
    gallery = l2_normalize(np.eye(4, dtype=np.float32))
    orig = gallery[0]
    corr = np.stack([gallery[0], gallery[1]])
    image = rgb(8, 8)
    crops = apply_k(image, "gaussian_noise", severity=3, k=2, seed=0)
    row = compare_query(
        orig,
        corr,
        gallery,
        qid=1,
        gids=[1, 2, 3, 4],
        k_neighbors=2,
        orig_image=image,
        corr_images=crops,
    )
    assert row["n"] == 2 and "samples" in row and "l1" in row
    compact = compare_query(orig, gallery[1], gallery, include_samples=False)
    assert "samples" not in compact
    assert compact["n"] == 1
    scored = score_queries(
        np.stack([orig, gallery[1]]),
        np.stack([corr, corr]),
        gallery,
        qids=[1, 2],
        gids=[1, 2, 3, 4],
        keep=np.ones((2, 4), dtype=bool),
        k_neighbors=2,
    )
    assert len(scored) == 2 and scored[0]["query_index"] == 0
    scored_none = score_queries(orig[None], corr[None], gallery)
    assert len(scored_none) == 1
    scored_1d = score_queries(orig[None], corr[None], gallery, keep=np.ones(4, dtype=bool))
    assert len(scored_1d) == 1
    table = summarize(
        [
            {
                "corruption": "jpeg",
                "severity": 3,
                "overlap": 1.0,
                "rank1_image_same": True,
                "note": "x",
                "samples": [],
            },
            {"corruption": "jpeg", "severity": 3, "overlap": 0.5, "rank1_image_same": False, "note": "y"},
            {"corruption": "fog", "severity": 5, "overlap": 0.0, "rank1_image_same": False, "note": "z"},
        ]
    )
    jpeg = next(item for item in table if item["corruption"] == "jpeg")
    assert jpeg["n"] == 2 and jpeg["overlap"] == pytest.approx(0.75)
    model = TinyEmbed()
    x = torch.rand(5, 3, 8, 8)
    vec = embed(model, x, device="cpu", batch_size=2)
    assert vec.shape == (5, 4)
    assert embed(TensorEmbed(), x[:2]).shape == (2, 3)
    assert embed(BareEmbed(), x[:1]).shape == (1, 3)
    with pytest.raises(ValueError, match="together"):
        compare_query(orig, corr, gallery, orig_image=image)
    with pytest.raises(ValueError, match="corr_images"):
        compare_query(orig, corr, gallery, orig_image=image, corr_images=crops[:1])
    with pytest.raises(ValueError, match="orig_emb"):
        score_queries(orig, corr[None], gallery)
    with pytest.raises(ValueError, match="corr_emb"):
        score_queries(orig[None], orig[None], gallery)
    with pytest.raises(ValueError, match="gallery_emb"):
        score_queries(orig[None], corr[None], gallery[:, :2])
    with pytest.raises(ValueError, match="qids"):
        score_queries(orig[None], corr[None], gallery, qids=[1, 2])
    with pytest.raises(ValueError, match="gids"):
        score_queries(orig[None], corr[None], gallery, gids=[1])
    with pytest.raises(ValueError, match="keep must match gallery"):
        score_queries(orig[None], corr[None], gallery, keep=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="keep must be"):
        score_queries(orig[None], corr[None], gallery, keep=np.ones((1, 3), dtype=bool))
    with pytest.raises(ValueError, match="keep must be"):
        score_queries(orig[None], corr[None], gallery, keep=np.ones((1, 4, 1), dtype=bool))
    with pytest.raises(ValueError, match="no rows"):
        summarize([])
    with pytest.raises(ValueError, match="no rows"):
        _mean_fields([])
    with pytest.raises(ValueError, match="batch_size"):
        embed(model, x, batch_size=0)
    with pytest.raises(ValueError, match="NCHW"):
        embed(model, torch.rand(3, 8, 8))
    with pytest.raises(ValueError, match="Empty"):
        embed(model, torch.rand(0, 3, 8, 8))
    with pytest.raises(FloatingPointError):
        embed(InfEmbed(), torch.rand(2, 3, 4, 4))
