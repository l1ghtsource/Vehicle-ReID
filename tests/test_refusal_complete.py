import numpy as np
import pytest
import torch

from refusal import (
    CONTEST_F1_WEIGHT,
    CONTEST_TNR_WEIGHT,
    OPEN_SET_FRACTION,
    STAT_NAMES,
    RefusalTabM,
    balanced_pack,
    balanced_pairs,
    batch_examples,
    candidate_frame,
    candidate_metrics,
    contest_score,
    decision_metrics,
    example_vector,
    feature_names,
    fit_boosting,
    fit_rank_weights,
    fit_tabm,
    gap12,
    has_match,
    load_boosting,
    mask_gallery,
    max_cosine,
    open_set_pack,
    open_set_split,
    predict_boosting,
    predict_tabm,
    rank_average,
    ranking_metrics,
    refusal_accept,
    save_boosting,
    scored_pack,
    select_threshold,
    similarities,
    stat_vector,
    sweep_thresholds,
    top_hit,
    vote_fraction,
    write_candidates,
)
from refusal.ensemble import rank_scores
from refusal.features import _clustering, _components, l2_normalize
from refusal.metrics import decide
from refusal.tabm import EnsembleLinear


def toy_embeddings():
    rng = np.random.default_rng(0)
    gallery = rng.normal(size=(20, 8)).astype(np.float32)
    gallery[0:4] += 3
    gallery[4:8] += np.array([3, -3, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    query = np.stack([gallery[0] + 0.05, gallery[5] + 0.05, rng.normal(size=8)]).astype(np.float32)
    qids = np.array([1, 2, 3])
    gids = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 9, 9, 9, 10, 10, 11, 12, 13, 14, 15])
    return query, gallery, qids, gids


def test_features_and_similarity_guards():
    query, gallery, _, _ = toy_embeddings()
    stats = stat_vector(query[0], gallery, k=5)
    assert stats.shape == (len(STAT_NAMES),) and np.isfinite(stats).all()
    small = stat_vector(query[0], gallery[:1], k=10, edge=0.0)
    assert small.shape == (len(STAT_NAMES),)
    vec = example_vector(query[0], gallery, k=3, with_embeddings=True)
    assert vec.shape[0] == len(STAT_NAMES) + 2 * gallery.shape[1]
    only = example_vector(query[0], gallery, k=3, with_embeddings=False)
    assert only.shape == (len(STAT_NAMES),)
    batch = batch_examples(query, gallery, k=4)
    assert batch.shape == (3, len(STAT_NAMES) + 16)
    assert batch_examples(query[0], gallery, k=2, with_embeddings=False).shape == (1, len(STAT_NAMES))
    sim = similarities(query[0], gallery)
    assert sim.shape == (1, 20)
    assert similarities(query, gallery).shape == (3, 20)
    assert feature_names() == list(STAT_NAMES)
    named = feature_names(2)
    assert named[: len(STAT_NAMES)] == list(STAT_NAMES) and named[-1] == "g_1"
    assert _components(np.zeros((0, 0), dtype=bool)) == 0
    assert _clustering(np.zeros((0, 0), dtype=bool)) == 0.0
    adj = np.zeros((3, 3), dtype=bool)
    adj[0, 1] = adj[1, 0] = True
    assert _components(adj) == 2
    assert _clustering(np.ones((3, 3), dtype=bool)) > 0
    with pytest.raises(ValueError, match="empty"):
        l2_normalize(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="k must"):
        stat_vector(query[0], gallery, k=0)
    with pytest.raises(ValueError, match="edge"):
        stat_vector(query[0], gallery, edge=2)
    with pytest.raises(ValueError, match="gallery is empty"):
        similarities(query[0], np.zeros((0, 8)))
    with pytest.raises(ValueError, match="matching"):
        similarities(query[0], np.zeros((4, 3)))
    with pytest.raises(ValueError, match="embedding_dim"):
        feature_names(-1)


def test_protocol_open_set_and_masks():
    query, gallery, qids, gids = toy_embeddings()
    open_q, keep, held = open_set_split(qids, gids, fraction=0.4, seed=1)
    assert open_q.any() and (~open_q).any()
    assert keep.sum() < len(gids)
    assert set(held).issubset(set(qids))
    y = has_match(qids, gids, keep)
    assert y.shape == qids.shape
    assert has_match(qids, gids).all()
    masked = mask_gallery(gallery, keep)
    assert len(masked) == int(keep.sum())
    pos_neg, labels = balanced_pairs(query, gallery, qids, gids, k=4, with_embeddings=False)
    assert set(labels.tolist()) == {0, 1} and len(pos_neg) == 6
    packed_x, packed_y, packed_cos, packed_hit = balanced_pack(
        query, gallery, qids, gids, k=4, with_embeddings=False
    )
    assert packed_y.tolist() == labels.tolist() and packed_cos.shape == packed_y.shape
    assert packed_cos[0] > packed_cos[1]
    assert packed_hit.dtype == bool and not packed_hit[1]
    assert packed_hit[0] == top_hit(query[:1], gallery, qids[:1], gids)[0]
    eval_x, eval_y, eval_cos, eval_hit, held_ids = open_set_pack(
        query, gallery, qids, gids, k=4, with_embeddings=False, fraction=0.2, seed=0
    )
    assert len(eval_y) == len(qids)
    assert 0 < float(eval_y.mean()) < 1
    assert set(held_ids.tolist()).issubset(set(qids.tolist()))
    assert eval_cos.shape == eval_y.shape == eval_hit.shape
    scored_x, scored_y, scored_cos, scored_hit = scored_pack(
        query, gallery, qids, gids, k=4, with_embeddings=False
    )
    assert scored_y.tolist() == [1, 1, 1] and scored_x.shape[0] == 3
    assert scored_cos.shape == scored_hit.shape == scored_y.shape
    hit = top_hit(query, gallery, qids, gids)
    assert hit.shape == (3,) and hit.dtype == bool
    with pytest.raises(ValueError, match="query identities"):
        top_hit(query, gallery, qids[:1], gids)
    with pytest.raises(ValueError, match="gallery identities"):
        top_hit(query, gallery, qids, gids[:3])
    one, one_y = balanced_pairs(query[0], gallery, qids[:1], gids, k=3)
    assert one.shape[0] == 2 and one_y.tolist() == [1, 0]
    missing, miss_y = balanced_pairs(query[2:3], gallery, np.array([99]), gids, k=3, with_embeddings=False)
    assert miss_y.tolist() == [0] and len(missing) == 1
    with pytest.raises(ValueError, match="fraction"):
        open_set_split(qids, gids, fraction=0)
    with pytest.raises(ValueError, match="two query"):
        open_set_split([1, 1], gids)
    with pytest.raises(ValueError, match="gallery_keep"):
        has_match(qids, gids, np.array([True]))
    with pytest.raises(ValueError, match="keep must match"):
        mask_gallery(gallery, np.array([True]))
    with pytest.raises(ValueError, match="empty"):
        mask_gallery(gallery, np.zeros(len(gallery), dtype=bool))
    with pytest.raises(ValueError, match="entire gallery"):
        open_set_split(np.array([1, 2]), np.array([1, 1, 1]), fraction=0.5, seed=1)
    with pytest.raises(ValueError, match="query rows"):
        balanced_pairs(query, gallery, qids[:1], gids)
    with pytest.raises(ValueError, match="gallery rows"):
        balanced_pairs(query, gallery, qids, gids[:3])


def test_metrics_threshold_rules_and_baselines():
    y = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.05, 0.02])
    top = np.array([1, 1, 1, 0, 0, 0])
    hit = candidate_metrics(y, scores, 0.5, top)
    assert hit["tp"] == 2 and hit["fn"] == 1 and hit["tn"] == 2 and hit["fp"] == 1
    assert hit["fp_open"] == 1 and hit["fp_closed"] == 0
    assert 0 < hit["f1"] < 1 and 0 < hit["tnr"] < 1
    assert hit["contest"] == pytest.approx(contest_score(hit["f1"], hit["tnr"]))
    wrong = np.array([1, 0, 1, 0, 0, 0])
    mixed = candidate_metrics(y, scores, 0.5, wrong)
    assert mixed["tp"] == 1 and mixed["fp"] == 2 and mixed["fp_closed"] == 1
    assert mixed["tnr"] == pytest.approx(2 / 3)
    micro = candidate_metrics(np.array([1, 1]), np.array([0.9, 0.8]), 0.0, np.array([True, False]))
    assert micro["tp"] == 1 and micro["fp"] == 1 and micro["fn"] == 0
    assert micro["f1"] == pytest.approx(2 / 3)
    rank = ranking_metrics(y, scores)
    assert 0.5 < rank["pr_auc"] <= 1 and 0.5 < rank["roc_auc"] <= 1
    always = candidate_metrics(y, scores, -1, top)
    assert always["recall"] == 1 and always["tnr"] == 0
    refuse = candidate_metrics(y, scores, 2, top)
    assert refuse["f1"] == 0 and refuse["tnr"] == 1
    assert decide(scores, 0.5).sum() == 3
    f1 = select_threshold(y, scores, top, kind="max_f1")
    youden = select_threshold(y, scores, top, kind="youden")
    mixed_rule = select_threshold(y, scores, top, kind="f1_tnr")
    contest = select_threshold(y, scores, top)
    constrained = select_threshold(y, scores, top, kind="max_f1", min_tnr=0.9)
    infeasible = select_threshold(y, scores, top, kind="max_f1", min_tnr=1.1)
    assert f1["f1"] >= youden["f1"] or youden["tnr"] >= f1["tnr"]
    assert mixed_rule["threshold"] >= 0
    assert contest["contest"] >= f1["contest"] or contest["tnr"] >= f1["tnr"]
    assert contest["contest"] == pytest.approx(contest_score(contest["f1"], contest["tnr"]))
    assert constrained["tnr"] >= 0.5
    assert infeasible["f1"] >= 0
    custom = sweep_thresholds(y, scores, top, thresholds=[0.0, 0.5, 1.0])
    assert len(custom) == 3
    with pytest.raises(ValueError, match="Unknown"):
        select_threshold(y, scores, top, kind="banana")
    with pytest.raises(ValueError, match="aligned"):
        candidate_metrics(y, scores[:2], 0.1, top)
    with pytest.raises(ValueError, match="aligned"):
        candidate_metrics([], [], 0.1, [])
    with pytest.raises(ValueError, match="aligned"):
        candidate_metrics(y, scores, 0.1, top[:2])
    with pytest.raises(ValueError, match="aligned"):
        sweep_thresholds([], [], [])
    with pytest.raises(ValueError, match="both"):
        ranking_metrics(np.ones(4), np.linspace(0, 1, 4))
    with pytest.raises(ValueError, match="aligned"):
        ranking_metrics(y, scores[:1])
    with pytest.raises(ValueError, match="empty"):
        ranking_metrics([], [])
    with pytest.raises(ValueError, match="aligned"):
        decision_metrics(y, np.ones(2, dtype=bool), top)
    with pytest.raises(ValueError, match="aligned"):
        decision_metrics([], [], [])


def test_contest_objective_beats_max_f1_on_open_set_tradeoff():
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    top = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    scores = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.5, 0.95, 0.5])
    low = candidate_metrics(y, scores, 0.4, top)
    high = candidate_metrics(y, scores, 0.9, top)
    assert low["tnr"] == 0
    assert low["f1"] == pytest.approx(8 / 9)
    assert low["contest"] == pytest.approx(0.7 * 8 / 9)
    assert high["f1"] == pytest.approx(0.875)
    assert high["tnr"] == pytest.approx(0.5)
    assert high["contest"] == pytest.approx(0.7625)
    assert contest_score(1.0, 0.0) == pytest.approx(CONTEST_F1_WEIGHT)
    assert contest_score(0.0, 1.0) == pytest.approx(CONTEST_TNR_WEIGHT)
    assert high["contest"] > low["contest"]
    f1 = select_threshold(y, scores, top, kind="max_f1")
    contest = select_threshold(y, scores, top, kind="contest")
    assert f1["tnr"] == 0
    assert contest["tnr"] == pytest.approx(0.5)
    assert contest["contest"] == pytest.approx(0.7625)
    assert contest["threshold"] == pytest.approx(0.95)


def test_contest_threshold_depends_on_prevalence():
    closed = np.array([0.95, 0.93, 0.91, 0.89, 0.60])
    open_scores = np.array([0.88, 0.86, 0.84, 0.50, 0.40])
    y_balanced = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    scores_balanced = np.concatenate([closed, open_scores])
    top_balanced = y_balanced.astype(bool)
    y_open = np.array([1, 1, 1, 1, 0])
    scores_open = np.array([0.95, 0.93, 0.91, 0.60, 0.88])
    top_open = y_open.astype(bool)
    balanced = select_threshold(y_balanced, scores_balanced, top_balanced, kind="contest")
    realistic = select_threshold(y_open, scores_open, top_open, kind="contest")
    assert balanced["threshold"] != realistic["threshold"]
    assert OPEN_SET_FRACTION == 0.2


def test_pooled_oof_uses_per_fold_inner_thresholds():
    y = np.array([1, 0, 1, 0])
    top = np.array([1, 0, 1, 0], dtype=bool)
    scores = np.array([0.9, 0.1, 0.4, 0.3])
    accept = np.concatenate(
        [
            decide(scores[:2], 0.5),
            decide(scores[2:], 0.35),
        ]
    )
    honest = decision_metrics(y, accept, top)
    leaked = candidate_metrics(y, scores, 0.425, top)
    assert honest["fn"] == 0 and honest["fp"] == 0
    assert leaked["fn"] == 1
    assert honest["contest"] > leaked["contest"]


def test_simple_scores_boosting_and_tabm(tmp_path):
    query, gallery, qids, gids = toy_embeddings()
    mx = max_cosine(query, gallery)
    gp = gap12(query, gallery)
    assert mx.shape == (3,) and gp.shape == (3,)
    assert mx[0] > mx[2]
    one = gap12(query[:1], gallery[:1])
    assert one.shape == (1,) and one[0] == 0
    y = has_match(qids, gids).astype(int)
    y[2] = 0
    X = batch_examples(query, gallery, k=5, with_embeddings=True)
    X = np.concatenate([X, X + 0.01, X - 0.01], 0)
    y_rep = np.concatenate([y, y, y])
    boost = fit_boosting(X, y_rep, iterations=8, depth=2, seed=0)
    p_boost = predict_boosting(boost, X)
    assert p_boost.shape == (len(X),) and np.isfinite(p_boost).all()
    dumped = save_boosting(boost, tmp_path / "models" / "refuse.cbm")
    loaded = load_boosting(dumped)
    np.testing.assert_allclose(predict_boosting(loaded, X), p_boost)
    with pytest.raises(ValueError, match="not found"):
        load_boosting(tmp_path / "missing.cbm")
    tabm = fit_tabm(
        X, y_rep, epochs=3, batch_size=4, val_frac=0.3, patience=2, hidden=(8, 4), dropout=0.1, k=3
    )
    p_tabm = predict_tabm(tabm, X)
    assert p_tabm.shape == (len(X),) and np.isfinite(p_tabm).all()
    tiny = fit_tabm(X, y_rep, epochs=2, batch_size=8, val_frac=0.0, patience=1, hidden=(8,), dropout=0.0, k=2)
    assert predict_tabm(tiny, X[:2]).shape == (2,)
    cramped = fit_tabm(X[:4], y_rep[:4], epochs=6, batch_size=2, val_frac=0.8, patience=1, hidden=(4,), k=2)
    assert isinstance(cramped, RefusalTabM)
    stuck = fit_tabm(
        X, y_rep, epochs=8, batch_size=4, val_frac=0.3, patience=1, hidden=(8,), dropout=0.0, k=2, lr=1e-12
    )
    assert isinstance(stuck, RefusalTabM)
    layer = EnsembleLinear(3, 2, k=2, scaling_init="random-signs")
    out = layer(torch.zeros(4, 2, 3))
    assert out.shape == (4, 2, 2)
    ones = EnsembleLinear(3, 2, k=2, scaling_init="ones")
    assert torch.allclose(ones.r, torch.ones_like(ones.r))
    model = RefusalTabM(4, hidden=(3, 3), k=2, dropout=0.0)
    model.set_norm(np.zeros(4), np.ones(4))
    assert torch.isfinite(model(torch.zeros(2, 4))).all()
    with pytest.raises(ValueError, match="2D feature"):
        fit_boosting(X[0], y_rep)
    with pytest.raises(ValueError, match="2D feature"):
        predict_boosting(boost, X[0])
    with pytest.raises(ValueError, match="2D feature"):
        fit_tabm(X[0], y_rep)
    with pytest.raises(ValueError, match="2D feature"):
        predict_tabm(tabm, X[0])
    with pytest.raises(ValueError, match="Invalid TabM"):
        RefusalTabM(0, hidden=(4,))
    with pytest.raises(ValueError, match="Invalid TabM"):
        RefusalTabM(4, hidden=(), dropout=0.0)
    with pytest.raises(ValueError, match="Invalid TabM"):
        RefusalTabM(4, hidden=(4,), dropout=1)
    with pytest.raises(ValueError, match="hyperparameters"):
        fit_tabm(X, y_rep, epochs=0)
    with pytest.raises(ValueError, match="Invalid ensemble"):
        EnsembleLinear(0, 2, k=2)
    with pytest.raises(ValueError, match="Unknown scaling"):
        EnsembleLinear(3, 2, k=2, scaling_init="uniform")
    with pytest.raises(ValueError, match="ensemble input"):
        layer(torch.zeros(2, 3, 3))
    with pytest.raises(ValueError, match="2D feature"):
        model(torch.zeros(4))
    with pytest.raises(ValueError, match="Normalization stats"):
        model.set_norm(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="2D feature"):
        fit_boosting(X[:3], y_rep[:3])
    with pytest.raises(ValueError, match="TabM needs"):
        fit_tabm(np.ones((5, 3)), np.zeros(5))
    with pytest.raises(ValueError, match="hyperparameters"):
        fit_tabm(X, y_rep, lr=0)
    with pytest.raises(ValueError, match="hyperparameters"):
        fit_tabm(X, y_rep, k=0)


def test_rank_ensemble_and_votes():
    y = np.array([1, 1, 1, 0, 0, 0])
    a = np.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.1])
    b = np.array([0.95, 0.1, 0.85, 0.2, 0.05, 0.4])
    c = np.array([0.7, 0.75, 0.8, 0.3, 0.25, 0.2])
    blended = rank_average([a, b, c])
    assert blended.shape == (6,) and np.isfinite(blended).all()
    weighted = rank_average([a, b], weights=[1.0, 0.0])
    np.testing.assert_allclose(weighted, rank_average([a, a], weights=[1.0, 0.0]))
    votes = vote_fraction([a, b, c], [0.5, 0.5, 0.5])
    assert votes[0] == 1.0 and 0 <= votes.min() <= votes.max() <= 1
    w, picked = fit_rank_weights(y, [a, b, c], y.astype(bool), grid=3)
    assert w.shape == (3,) and abs(w.sum() - 1) < 1e-12 and picked["f1"] > 0
    w_f1, picked_f1 = fit_rank_weights(y, [a, b, c], y.astype(bool), grid=3, kind="max_f1")
    assert w_f1.shape == (3,) and picked_f1["f1"] > 0
    assert rank_scores(np.array([0.3]))[0] == 0.5
    with pytest.raises(ValueError, match="two score"):
        rank_average([a])
    with pytest.raises(ValueError, match="aligned"):
        rank_average([a, b[:3]])
    with pytest.raises(ValueError, match="nonempty"):
        rank_average([np.array([]), np.array([])])
    with pytest.raises(ValueError, match="nonempty"):
        rank_scores(np.array([]))
    with pytest.raises(ValueError, match="weights"):
        rank_average([a, b], weights=[-1.0, 1.0])
    with pytest.raises(ValueError, match="thresholds"):
        vote_fraction([a, b], [0.5])
    with pytest.raises(ValueError, match="align"):
        fit_rank_weights(y[:3], [a, b], y[:3].astype(bool))
    with pytest.raises(ValueError, match="grid"):
        fit_rank_weights(y, [a, b], y.astype(bool), grid=1)


def test_candidates_csv_omits_refused_queries(tmp_path):
    queries = np.array(["q1", "q2", "q3"])
    gallery = np.array([["g1", ""], ["-", "g2"], ["g3", "nan"]], dtype=object)
    conf = np.array([[0.9, 0.1], [0.8, 0.7], [0.4, np.nan]])
    frame = candidate_frame(queries, gallery, conf, accept=np.array([True, False, True]))
    assert list(frame.columns) == ["query_id", "gallery_id", "confidence"]
    assert frame.query_id.tolist() == ["q1", "q3"]
    assert frame.gallery_id.tolist() == ["g1", "g3"]
    assert frame.confidence.tolist() == [0.9, 0.4]
    path = tmp_path / "out" / "candidates.csv"
    written = write_candidates(path, queries, gallery, conf, accept=np.zeros(3, dtype=bool))
    text = path.read_text()
    assert text.splitlines()[0] == "query_id,gallery_id,confidence"
    assert len(written) == 0 and "q1" not in text
    numbered = candidate_frame(["q"], np.array([[np.int64(7)]]), np.array([[0.5]]))
    assert numbered.gallery_id.tolist() == ["7"]
    with pytest.raises(ValueError, match="matching"):
        candidate_frame(["q"], np.array(["g"]), np.array([0.1]))
    with pytest.raises(ValueError, match="query_ids"):
        candidate_frame(["q1", "q2"], gallery[:1], conf[:1])
    with pytest.raises(ValueError, match="accept"):
        candidate_frame(queries, gallery, conf, accept=np.array([True]))
    none = candidate_frame(["q"], np.array([[None, float("nan")]], dtype=object), np.array([[0.2, 0.3]]))
    assert len(none) == 0


def test_refusal_accept_kinds(tmp_path):
    query, gallery, qids, gids = toy_embeddings()
    y = has_match(qids, gids).astype(int)
    y[2] = 0
    X = batch_examples(query, gallery, k=5, with_embeddings=True)
    X = np.concatenate([X, X + 0.01, X - 0.01], 0)
    y_rep = np.concatenate([y, y, y])
    boost = fit_boosting(X, y_rep, iterations=8, depth=2, seed=0)
    path = save_boosting(boost, tmp_path / "refuse.cbm")
    none = refusal_accept("none", query, gallery)
    assert none.all() and none.shape == (3,)
    assert refusal_accept(None, query[0], gallery).tolist() == [True]
    assert refusal_accept("off", query, gallery).all()
    high = refusal_accept("threshold", query, gallery, cosine_threshold=2.0)
    low = refusal_accept("  Threshold  ", query, gallery, cosine_threshold=0.0)
    assert not high.any() and low.all()
    model = refusal_accept("model", query, gallery, model=boost, model_threshold=0.0, k=5)
    assert model.all()
    from_disk = refusal_accept(
        "model", query, gallery, model_path=path, model_threshold=2.0, k=5, with_embeddings=True
    )
    assert not from_disk.any()
    mixed = refusal_accept(
        "ensemble",
        query,
        gallery,
        model=boost,
        cosine_threshold=0.0,
        model_threshold=0.0,
        k=5,
        with_embeddings=True,
    )
    assert mixed.all()
    from_ens = refusal_accept(
        "ensemble",
        query,
        gallery,
        model_path=path,
        cosine_threshold=2.0,
        model_threshold=2.0,
        k=5,
    )
    assert not from_ens.any()
    with pytest.raises(ValueError, match="nonempty"):
        refusal_accept("none", np.zeros((0, 8)), gallery)
    with pytest.raises(ValueError, match="nonempty"):
        refusal_accept("none", np.zeros((1, 1, 8)), gallery)
    with pytest.raises(ValueError, match="cosine_threshold"):
        refusal_accept("threshold", query, gallery)
    with pytest.raises(ValueError, match="model_threshold"):
        refusal_accept("model", query, gallery, model=boost)
    with pytest.raises(ValueError, match="cosine_threshold"):
        refusal_accept("ensemble", query, gallery, model=boost)
    with pytest.raises(ValueError, match="cosine_threshold"):
        refusal_accept("ensemble", query, gallery, model=boost, model_threshold=0.5)
    with pytest.raises(ValueError, match="model_threshold"):
        refusal_accept("ensemble", query, gallery, model=boost, cosine_threshold=0.5)
    with pytest.raises(ValueError, match="model_path"):
        refusal_accept("model", query, gallery, model_threshold=0.5, model_path="")
    with pytest.raises(ValueError, match="model_path"):
        refusal_accept(
            "ensemble", query, gallery, cosine_threshold=0.1, model_threshold=0.1, model_path="null"
        )
    with pytest.raises(ValueError, match="model_path"):
        refusal_accept("model", query, gallery, model_threshold=0.5, model_path="None")
    with pytest.raises(ValueError, match="none, threshold, model, or ensemble"):
        refusal_accept("tabm", query, gallery)
