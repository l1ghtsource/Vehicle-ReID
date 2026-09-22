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


def optional_csv_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return Path(text)


def holdout_identities(cfg, train: pd.DataFrame) -> set[int] | None:
    path = optional_csv_path(cfg.data.get("val_source_csv"))
    if path is None:
        return None
    orig = read_annotations(path, labeled=True)
    present = set(orig.vehicle_id.astype(int)) & set(train.vehicle_id.astype(int))
    if present == set(train.vehicle_id.astype(int)):
        return None
    if not present:
        raise ValueError("val_source_csv shares no identities with train.csv")
    return present


def check_fold_indices(values, n_folds: int) -> None:
    folds = {int(value) for value in values}
    allowed = set(range(n_folds)) | {-1}
    if folds - allowed or not set(range(n_folds)).issubset(folds):
        raise ValueError("Invalid fold indices")


def make_folds(df, n_folds=5, seed=42, group_column="vehicle_id", val_identities=None):
    if group_column != "vehicle_id":
        raise ValueError("Open-set validation must group by vehicle_id")
    result = df.copy()
    result["fold"] = -1
    if val_identities is None:
        holdout = df
    else:
        ids = {int(value) for value in val_identities}
        holdout = df[df[group_column].astype(int).isin(ids)]
        if holdout.empty:
            raise ValueError("val_source_csv shares no identities with train.csv")
    if holdout[group_column].nunique() < n_folds:
        raise ValueError("Fewer identities than folds")
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(splitter.split(holdout, groups=holdout[group_column])):
        assert set(holdout.iloc[tr].vehicle_id).isdisjoint(holdout.iloc[va].vehicle_id)
        result.loc[holdout.index[va], "fold"] = fold
    return result


def ensure_folds(cfg):
    df = read_annotations(cfg.data.train_csv, labeled=True)
    path = Path(cfg.data.folds_file)
    val_ids = holdout_identities(cfg, df)
    meta = {
        "fingerprint": fingerprint(df),
        "seed": int(cfg.seed),
        "n_folds": int(cfg.data.n_folds),
        "group_column": cfg.data.group_column,
        "protocol": 1,
    }
    source = optional_csv_path(cfg.data.get("val_source_csv"))
    if val_ids is not None and source is not None:
        meta["val_source_csv"] = str(source)
        meta["val_source_fingerprint"] = fingerprint(read_annotations(source, labeled=True))
    if path.exists():
        meta_path = path.with_suffix(".json")
        if not meta_path.exists() or json.loads(meta_path.read_text()) != meta:
            raise ValueError(f"Stale/mismatched fold manifest {path}; use a new data.folds_file")
        saved = pd.read_csv(path, dtype={"image_id": str})
        if fingerprint(saved) != fingerprint(df):
            raise ValueError("Fold rows do not match train.csv")
        if saved.groupby("vehicle_id").fold.nunique().max() != 1:
            raise ValueError("Identity leakage in fold manifest")
        check_fold_indices(saved.fold, int(cfg.data.n_folds))
        if val_ids is not None:
            extra = ~saved.vehicle_id.astype(int).isin(val_ids)
            if (saved.loc[~extra, "fold"] < 0).any():
                raise ValueError("Holdout identities have fold=-1")
            if (saved.loc[extra, "fold"] != -1).any():
                raise ValueError("Pseudo-label identities must use fold=-1")
        return saved
    result = make_folds(df, cfg.data.n_folds, cfg.seed, cfg.data.group_column, val_ids)
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
