from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from augmentations import build_transforms

from .datamodule import ReIDDataModule
from .folds import query_gallery_split, read_annotations
from .images import VehicleDataset


def _xml_without_declaration(text: str) -> str:
    end = text.find("?>")
    return text[end + 2 :] if end >= 0 else text


def read_veri(root: Path) -> pd.DataFrame:
    image_dir = root / "image_train"
    xml_path = root / "train_label.xml"
    if not image_dir.is_dir() or not xml_path.is_file():
        raise FileNotFoundError(f"Incomplete VeRi dataset at {root}")
    tree = ElementTree.fromstring(_xml_without_declaration(xml_path.read_bytes().decode("gb18030")))
    rows = []
    for item in tree.iter("Item"):
        image_name = item.attrib["imageName"]
        rows.append(
            {
                "image_id": f"veri:{image_name}",
                "image_path": str((image_dir / image_name).resolve()),
                "identity_key": f"veri:{item.attrib['vehicleID']}",
                "camera_key": f"veri:{item.attrib['cameraID']}",
                "full_image": True,
                "source": "veri",
            }
        )
    if not rows:
        raise ValueError(f"VeRi annotations are empty: {xml_path}")
    return pd.DataFrame(rows)


def read_vric(root: Path) -> pd.DataFrame:
    image_dir = root / "train_images"
    annotation_path = root / "vric_train.txt"
    if not image_dir.is_dir() or not annotation_path.is_file():
        raise FileNotFoundError(f"Incomplete VRIC dataset at {root}")
    rows = []
    for line_number, line in enumerate(annotation_path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{annotation_path}:{line_number}: expected image, identity, camera")
        image_name, identity, camera = fields
        rows.append(
            {
                "image_id": f"vric:{image_name}",
                "image_path": str((image_dir / image_name).resolve()),
                "identity_key": f"vric:{identity}",
                "camera_key": f"vric:{camera}",
                "full_image": True,
                "source": "vric",
            }
        )
    if not rows:
        raise ValueError(f"VRIC annotations are empty: {annotation_path}")
    return pd.DataFrame(rows)


READERS = {"veri": read_veri, "vric": read_vric}


def load_external_data(cfg) -> pd.DataFrame:
    selected = [str(name).lower() for name in cfg.pretrain.datasets]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("pretrain.datasets must contain unique dataset names")
    unknown = set(selected) - set(READERS)
    if unknown:
        raise ValueError(f"Unknown pretraining datasets: {sorted(unknown)}")
    frames = [READERS[name](Path(getattr(cfg.pretrain, name).root)) for name in selected]
    frame = pd.concat(frames, ignore_index=True)
    if frame.image_id.duplicated().any():
        raise ValueError("External image IDs must be unique")
    identities = {value: index for index, value in enumerate(sorted(frame.identity_key.unique()))}
    cameras = {value: index for index, value in enumerate(sorted(frame.camera_key.unique()))}
    frame["vehicle_id"] = frame.identity_key.map(identities).astype(int)
    frame["camera_id"] = frame.camera_key.map(cameras).astype(int)
    frame["row_id"] = range(len(frame))
    return frame


class PretrainDataModule(ReIDDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.external_frame: pd.DataFrame | None = None
        self.validation_frame: pd.DataFrame | None = None

    def prepare_data(self):
        load_external_data(self.cfg)
        read_annotations(self.cfg.pretrain.validation_csv, labeled=True)

    def setup(self, stage=None):
        train = load_external_data(self.cfg)
        validation = read_annotations(self.cfg.pretrain.validation_csv, labeled=True)
        self.external_frame = train
        self.validation_frame = validation
        self.folds = train
        self.train_frame = train.reset_index(drop=True)
        self.label_map = {
            int(identity): index for index, identity in enumerate(sorted(train.vehicle_id.unique()))
        }
        self.num_classes = len(self.label_map)
        query, gallery = query_gallery_split(
            validation,
            self.cfg.seed,
            query_per_identity=self.cfg.data.validation.query_per_identity,
            cross_camera=self.cfg.data.validation.cross_camera,
        )
        self.query_frame = query
        self.gallery_frame = gallery
        self.train_set = VehicleDataset(
            self.train_frame,
            self.cfg,
            build_transforms(self.cfg, True),
            self.label_map,
            True,
        )
        self.val_set = VehicleDataset(
            pd.concat([query, gallery], ignore_index=True),
            self.cfg,
            build_transforms(self.cfg),
        )
        self.num_query = len(query)
