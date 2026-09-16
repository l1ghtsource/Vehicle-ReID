import hydra

from dataset.folds import ensure_folds, query_gallery_split


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg):
    df = ensure_folds(cfg)
    for i in range(cfg.data.n_folds):
        tr, va = df[df.fold != i], df[df.fold == i]
        q, g = query_gallery_split(
            va, cfg.seed, cfg.data.validation.query_per_identity, cfg.data.validation.cross_camera
        )
        print(
            f"fold={i} train={len(tr)}/{tr.vehicle_id.nunique()}IDs "
            f"val={len(va)}/{va.vehicle_id.nunique()}IDs query={len(q)} gallery={len(g)}"
        )


if __name__ == "__main__":
    main()
