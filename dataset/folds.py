import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def fingerprint(df):
    cols = [c for c in ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"] if c in df]
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()


def split_fingerprint(df):
    cols = [c for c in ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id", "fold"] if c in df]
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()


def read_annotations(path, labeled=False):
    df = pd.read_csv(path, dtype={"image_id": str})
    needed = {"image_id", "x", "y", "w", "h"} | ({"vehicle_id"} if labeled else set())
    if needed - set(df):
        raise ValueError(f"{path}: missing columns {needed - set(df)}")
    if df[list(needed)].isna().any().any():
        raise ValueError(f"{path}: missing values")
    if not np.isfinite(df[["x", "y", "w", "h"]].values).all() or (df[["w", "h"]] <= 0).any().any():
        raise ValueError(f"{path}: invalid bbox")
    if df.image_id.duplicated().any():
        raise ValueError("Repeated image_id needs object-level keys; do not silently merge annotated objects")
    if "camera_id" not in df:
        df["camera_id"] = -1
    df["row_id"] = np.arange(len(df))
    return df


def make_folds(df, n_folds=5, seed=42, group_column="vehicle_id"):
    if group_column != "vehicle_id":
        raise ValueError("Open-set validation must group by vehicle_id")
    if df[group_column].nunique() < n_folds:
        raise ValueError("Fewer identities than folds")
    result = df.copy()
    result["fold"] = -1
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(splitter.split(df, groups=df[group_column])):
        assert set(df.iloc[tr].vehicle_id).isdisjoint(df.iloc[va].vehicle_id)
        result.loc[result.index[va], "fold"] = fold
    return result


def ensure_folds(cfg):
    df = read_annotations(cfg.data.train_csv, labeled=True)
    path = Path(cfg.data.folds_file)
    meta = {
        "fingerprint": fingerprint(df),
        "seed": int(cfg.seed),
        "n_folds": int(cfg.data.n_folds),
        "group_column": cfg.data.group_column,
        "protocol": 1,
    }
    if path.exists():
        meta_path = path.with_suffix(".json")
        if not meta_path.exists() or json.loads(meta_path.read_text()) != meta:
            raise ValueError(f"Stale/mismatched fold manifest {path}; use a new data.folds_file")
        saved = pd.read_csv(path, dtype={"image_id": str})
        if fingerprint(saved) != fingerprint(df):
            raise ValueError("Fold rows do not match train.csv")
        if saved.groupby("vehicle_id").fold.nunique().max() != 1:
            raise ValueError("Identity leakage in fold manifest")
        if set(saved.fold) != set(range(int(cfg.data.n_folds))):
            raise ValueError("Invalid fold indices")
        return saved
    result = make_folds(df, cfg.data.n_folds, cfg.seed, cfg.data.group_column)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    result.to_csv(tmp, index=False)
    os.replace(tmp, path)
    mtmp = path.with_suffix(f".{os.getpid()}.json.tmp")
    mtmp.write_text(json.dumps(meta, indent=2))
    os.replace(mtmp, path.with_suffix(".json"))
    return result


def query_gallery_split(df, seed=42, query_per_identity=1, cross_camera=True):
    rng = np.random.default_rng(seed)
    queries, galleries = [], []
    for _, rows in df.groupby("vehicle_id", sort=True):
        if len(rows) < 2:
            galleries.extend(rows.index)
            continue
        cameras = sorted(rows.camera_id.unique())
        if cross_camera and (len(cameras) < 2 or -1 in cameras):
            galleries.extend(rows.index)
            continue
        if cross_camera:
            cam = rng.choice(cameras)
            candidates = rows[rows.camera_id == cam]
            q = rng.choice(candidates.index, min(query_per_identity, len(candidates)), replace=False)

            g = rows[rows.camera_id != cam].index
        else:
            q = rng.choice(rows.index, min(query_per_identity, len(rows) - 1), replace=False)
            g = rows.index.difference(q)
        queries.extend(q)
        galleries.extend(g)
    if not queries or not galleries:
        raise ValueError("No evaluable query/gallery split")
    q, g = df.loc[sorted(queries)].copy(), df.loc[sorted(galleries)].copy()
    assert set(q.image_id).isdisjoint(g.image_id)
    return q.reset_index(drop=True), g.reset_index(drop=True)
