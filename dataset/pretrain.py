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
SSL_CROP = "test_train_ssl_crop"
SSL_FULL = "test_train_ssl_full"
SSL_SOURCE = "test_train_ssl"
SSL_SOURCES = {SSL_SOURCE, SSL_CROP, SSL_FULL}


def ssl_image_mode(name) -> str | None:
    token = str(name).lower()
    if token in {SSL_SOURCE, SSL_CROP}:
        return "crop"
    if token == SSL_FULL:
        return "full"
    return None


def read_test_train_ssl(cfg, mode="crop") -> pd.DataFrame:
    if mode not in {"crop", "full"}:
        raise ValueError("SSL image mode must be crop or full")
    parts = [
        ("train", read_annotations(cfg.data.train_csv, labeled=True)),
        ("query", read_annotations(cfg.data.query_csv, labeled=False)),
        ("gallery", read_annotations(cfg.data.gallery_csv, labeled=False)),
    ]
    frames = []
    for _, frame in parts:
        part = frame.copy()
        part["identity_key"] = "ssl:" + part.image_id.astype(str)
        part["camera_key"] = "ssl:" + part.camera_id.astype(str)
        part["full_image"] = mode == "full"
        part["source"] = SSL_FULL if mode == "full" else SSL_CROP
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise ValueError("test_train_ssl annotations are empty")
    if mode == "full":
        frame = frame.drop_duplicates("image_id", keep="first").reset_index(drop=True)
    elif frame.image_id.duplicated().any():
        raise ValueError("Competition train/test image IDs must be unique for SSL pretrain")
    return frame[["image_id", "x", "y", "w", "h", "identity_key", "camera_key", "full_image", "source"]]


def load_external_data(cfg) -> pd.DataFrame:
    selected = [str(name).lower() for name in cfg.pretrain.datasets]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("pretrain.datasets must contain unique dataset names")
    allowed = set(READERS) | SSL_SOURCES
    unknown = set(selected) - allowed
    if unknown:
        raise ValueError(f"Unknown pretraining datasets: {sorted(unknown)}")
    ssl_modes = [ssl_image_mode(name) for name in selected]
    if any(mode is not None for mode in ssl_modes) and (
        any(mode is None for mode in ssl_modes) or len(selected) != 1
    ):
        raise ValueError("test_train_ssl cannot mix with labeled extra datasets")
    frames = []
    for name in selected:
        mode = ssl_image_mode(name)
        if mode is not None:
            frames.append(read_test_train_ssl(cfg, mode))
        else:
            frames.append(READERS[name](Path(getattr(cfg.pretrain, name).root)))
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
        ssl = any(ssl_image_mode(name) is not None for name in self.cfg.pretrain.datasets)
        self.train_set = VehicleDataset(
            self.train_frame,
            self.cfg,
            build_transforms(self.cfg, True),
            self.label_map,
            True,
            views=2 if ssl else 1,
        )
        self.val_set = VehicleDataset(
            pd.concat([query, gallery], ignore_index=True),
            self.cfg,
            build_transforms(self.cfg),
        )
        self.num_query = len(query)
