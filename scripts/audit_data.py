import argparse
import json
from pathlib import Path

from PIL import Image

from dataset.folds import fingerprint, read_annotations
from dataset.images import crop_bbox, image_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("artifacts/data_audit.json"))
    p.add_argument("--check-images", action="store_true")
    args = p.parse_args()
    frames = {
        name: read_annotations(args.root / f"{name}.csv", labeled=name == "train")
        for name in ["train", "test_query", "test_gallery"]
    }
    report = {}
    for name, df in frames.items():
        report[name] = {"rows": len(df), "fingerprint": fingerprint(df), "missing_files": 0}
        if "vehicle_id" in df:
            report[name].update(
                identities=int(df.vehicle_id.nunique()),
                cameras=int(df.camera_id.nunique()),
                images_per_identity=df.groupby("vehicle_id").size().describe().to_dict(),
                single_camera_identities=int((df.groupby("vehicle_id").camera_id.nunique() == 1).sum()),
            )
        for row in df.itertuples():
            path = image_path(args.root / "images", row.image_id)
            if args.check_images:
                with Image.open(path) as im:
                    crop_bbox(im, [row.x, row.y, row.w, row.h])
        for other, other_df in frames.items():
            if name != other:
                overlap = set(df.image_id) & set(other_df.image_id)
                report[name][f"overlap_{other}"] = len(overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
