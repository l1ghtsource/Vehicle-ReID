from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from matching import (
    STAT_KINDS,
    as_hwc,
    default_weights,
    draw_matches,
    evidence_vector,
    geometry,
    hybrid_head,
    load_matcher,
    match_pair,
    match_pairs,
    match_stats,
    model,
    pairs,
    ranking_row,
    ransac_inliers,
    reorder_head,
    summarize,
)
from matching.retrieve import minmax


def rgb(width=16, height=12, color=(40, 80, 120)):
    return Image.new("RGB", (width, height), color=color)


class FakeProcessor:
    def __init__(self, items):
        self.items = items
        self.offset = 0
        self.calls = []

    def __call__(self, images, return_tensors="pt"):
        self.calls.append((images, return_tensors))
        return {"pixel_values": torch.zeros(len(images), 2, 1, 8, 8), "flag": 0}

    def post_process_keypoint_matching(self, outputs, target_sizes, threshold=0.0):
        del outputs, threshold
        n = len(target_sizes)
        chunk = self.items[self.offset : self.offset + n]
        self.offset += n
        return chunk


class ShortProcessor(FakeProcessor):
    def post_process_keypoint_matching(self, outputs, target_sizes, threshold=0.0):
        del outputs, target_sizes, threshold
        return []


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        del kwargs
        return SimpleNamespace()


class ParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flag = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values=None, **kwargs):
        del pixel_values, kwargs
        return SimpleNamespace()


def test_default_weights_and_load_guards(tmp_path, monkeypatch):
    assert default_weights().name == "efficientloftr"
    with pytest.raises(FileNotFoundError, match="missing"):
        load_matcher(tmp_path / "absent")
    root = tmp_path / "weights"
    root.mkdir()
    (root / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="missing"):
        load_matcher(root)
    (root / "preprocessor_config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"stub")
    seen = {}

    def load_proc(path, local_files_only=False):
        seen["proc"] = (str(path), local_files_only)
        return FakeProcessor([])

    def load_net(path, local_files_only=False):
        seen["model"] = (str(path), local_files_only)
        return ParamModel()

    monkeypatch.setattr(model.AutoImageProcessor, "from_pretrained", load_proc)
    monkeypatch.setattr(model.AutoModelForKeypointMatching, "from_pretrained", load_net)
    processor, net = load_matcher(root, device="cpu")
    assert seen["proc"][1] is True and seen["model"][1] is True
    assert processor.__class__ is FakeProcessor
    assert net.training is False
    assert next(net.parameters()).device.type == "cpu"


def test_as_hwc_and_pair_guards():
    pil = rgb()
    assert as_hwc(pil).shape == (12, 16, 3)
    chw = np.zeros((3, 8, 10), dtype=np.float32)
    assert as_hwc(chw).shape == (8, 10, 3)
    uint = np.zeros((6, 7, 3), dtype=np.uint8)
    assert as_hwc(uint).dtype == np.uint8
    empty = np.zeros((0, 4, 3), dtype=np.float32)
    assert as_hwc(empty).shape[2] == 3
    with pytest.raises(ValueError, match="HWC"):
        as_hwc(np.zeros((8, 8)))
    with pytest.raises(ValueError, match="pair"):
        pairs._as_pairs([])
    with pytest.raises(ValueError, match="pair"):
        pairs._as_pairs([pil, pil, pil])
    with pytest.raises(ValueError, match="two images"):
        pairs._as_pairs([[pil, pil, pil]])
    assert len(pairs._as_pairs((pil, pil))) == 1
    assert len(pairs._as_pairs([[pil, rgb(color=(1, 2, 3))]])) == 1


def test_match_pair_and_batches():
    kp0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kp1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    scores = torch.tensor([0.9, 0.4])
    item = {"keypoints0": kp0, "keypoints1": kp1, "matching_scores": scores}
    numpy_item = {
        "keypoints0": np.array([[1.0, 2.0]]),
        "keypoints1": np.array([[3.0, 4.0]]),
        "matching_scores": np.array([0.5]),
    }
    left, right = rgb(), rgb(color=(9, 8, 7))
    processor = FakeProcessor([item, numpy_item, item])
    got = match_pair(processor, FakeModel(), left, right, threshold=0.2)
    assert got["keypoints0"].shape == (2, 2)
    assert got["scores"].tolist() == pytest.approx([0.9, 0.4])
    batched = match_pairs(processor, ParamModel(), [[left, right], [right, left]], batch_size=1)
    assert len(batched) == 2
    assert batched[0]["scores"].shape == (1,)
    assert batched[1]["scores"].shape == (2,)
    with pytest.raises(ValueError, match="threshold"):
        match_pairs(processor, FakeModel(), [left, right], threshold=1.5)
    with pytest.raises(ValueError, match="batch_size"):
        match_pairs(processor, FakeModel(), [left, right], batch_size=0)
    with pytest.raises(ValueError, match="different number"):
        match_pairs(ShortProcessor([item]), FakeModel(), [left, right])
    bad = {"keypoints0": np.zeros((2, 2)), "keypoints1": np.zeros((1, 2)), "matching_scores": np.zeros(2)}
    with pytest.raises(ValueError, match="align"):
        match_pairs(FakeProcessor([bad]), FakeModel(), [left, right])


def test_geometry_and_stats(monkeypatch):
    pts = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [40, 1]], dtype=np.float32)
    other = pts.copy()
    other[-1] = [1, 40]
    mask = ransac_inliers(pts, other, reproj=3.0, seed=0)
    assert mask.shape == (5,)
    assert int(mask[:4].sum()) == 4
    stats = match_stats(pts, other, np.array([0.9, 0.8, 0.7, 0.6, 0.1]))
    assert stats["n_matches"] == 5
    assert stats["n_inliers"] >= 4
    empty = match_stats(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    assert empty["n_matches"] == 0 and empty["score_max"] == 0.0
    three = ransac_inliers(pts[:3], other[:3])
    assert three.tolist() == [False, False, False]
    with pytest.raises(ValueError, match="shape"):
        ransac_inliers(pts, pts[:4])
    with pytest.raises(ValueError, match="reproj"):
        ransac_inliers(pts, other, reproj=0)
    with pytest.raises(ValueError, match="align"):
        match_stats(pts, other, np.zeros(2))
    monkeypatch.setattr(geometry.cv2, "findHomography", lambda *args, **kwargs: (None, None))
    assert ransac_inliers(pts, other).tolist() == [False] * 5


def test_draw_and_retrieve():
    left = np.zeros((8, 10, 3), dtype=np.uint8)
    right = np.ones((6, 7, 3), dtype=np.float32)
    blank = draw_matches(left, right, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    assert blank.shape == (8, 17, 3)
    canvas = draw_matches(left, rgb(7, 6), [[1, 2]], [[3, 4]], [0.8])
    assert canvas.shape[1] == 17
    with pytest.raises(ValueError, match="align"):
        draw_matches(left, right, [[1, 2]], [[3, 4]], [0.1, 0.2])
    cosine = np.array([0.1, 0.9, 0.4, 0.2])
    match = np.array([8.0, 1.0, 7.0, 0.0])
    order = np.array([1, 2, 0, 3])
    reranked = reorder_head(order, match, k=2)
    assert reranked[:2].tolist() == [2, 1]
    mixed = hybrid_head(order, cosine, match, k=2, weight=1.0)
    assert mixed.tolist() == pytest.approx(np.array([0.0, 1.0]))
    assert hybrid_head(np.zeros(0, dtype=int), cosine, match, k=3).shape == (0,)
    assert minmax([3, 3, 3]).tolist() == [0, 0, 0]
    big = np.full((4, 5, 3), 200.0)
    assert as_hwc(big).max() == 200
    gids = np.array([9, 1, 1, 7])
    row = ranking_row(reranked, 1, gids, ranks=(1, 2))
    assert row["rank1"] is True and row["r2"] is True
    summary = summarize([row, {"ap": 0.5, "rank1": False, "pos_rank": 4, "r1": False, "r2": True}])
    assert summary["rank1"] == 0.5
    stats = [{"n_matches": 3, "n_inliers": 2, "score_sum": 1.5, "score_mean": 0.5}]
    assert evidence_vector(stats, "n_inliers").tolist() == [2]
    assert set(STAT_KINDS) == {"n_matches", "n_inliers", "score_sum", "score_mean"}
    with pytest.raises(ValueError, match="k must"):
        reorder_head(order, match, k=0)
    with pytest.raises(ValueError, match="cover"):
        reorder_head(np.array([8]), np.zeros(2), k=1)
    with pytest.raises(ValueError, match="cover"):
        reorder_head(np.array([-1, 0]), np.zeros(2), k=1)
    with pytest.raises(ValueError, match="align"):
        hybrid_head(order, cosine, match[:2], k=2)
    with pytest.raises(ValueError, match="k must"):
        hybrid_head(order, cosine, match, k=0)
    with pytest.raises(ValueError, match="weight"):
        hybrid_head(order, cosine, match, k=2, weight=2)
    with pytest.raises(ValueError, match="nonempty"):
        minmax([])
    with pytest.raises(ValueError, match="no gallery"):
        ranking_row(order, 99, gids)
    with pytest.raises(ValueError, match="ranks"):
        ranking_row(order, 1, gids, ranks=(0,))
    with pytest.raises(ValueError, match="no ranking"):
        summarize([])
    with pytest.raises(ValueError, match="unknown"):
        evidence_vector(stats, "n_outliers")
