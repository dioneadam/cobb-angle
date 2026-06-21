"""
Módulo de dados: transforms de treino/validação e SpinalDataset.
"""
import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import get_config


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_train_transform() -> A.Compose:
    """
    Augmentações para treino: simula variações de posicionamento,
    exposição e ruído de aparelhos de imagem médica.
    """
    return A.Compose(
        [
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                shear=0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussNoise(std_range=(10.0 / 255, 50.0 / 255), p=0.3),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_val_transform() -> A.Compose:
    """Transform para validação/inferência (só normalização)."""
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpinalDataset(Dataset):
    """
    Dataset COCO-format para detecção de vértebras em raio-X da coluna.

    Retorna por item:
        image     : Tensor [3, H, W] normalizado
        mask      : Tensor [H, W] binário
        keypoints : Tensor [136] com coordenadas normalizadas [0, 1]
        filename  : str com nome do arquivo original
    """

    def __init__(
        self,
        json_path: str,
        img_root: str,
        subset_name: str | None = None,
        transform: A.Compose | None = None,
    ) -> None:
        cfg = get_config()["data"]
        self.img_root = img_root
        self.subset_name = subset_name or cfg["subset_name"]
        self.transform = transform
        self.new_w: int = cfg["image_width"]
        self.new_h: int = cfg["image_height"]
        self.start_x: int = cfg["crop_start_x"]
        self.end_x: int = cfg["crop_end_x"]
        self.min_vertebrae: int = cfg["min_vertebrae"]

        self.coco = COCO(json_path)
        self.img_ids = self._filter_valid_ids()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_valid_ids(self) -> list[int]:
        valid = []
        for img_id in self.coco.getImgIds():
            info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(
                self.img_root, self.subset_name, info["file_name"]
            )
            ann_count = len(self.coco.getAnnIds(imgIds=img_id))
            if os.path.exists(path) and ann_count >= self.min_vertebrae:
                valid.append(img_id)
        return valid

    def _apply_clahe(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]

        # 1. Leitura e pré-processamento da imagem
        img_path = os.path.join(
            self.img_root, self.subset_name, info["file_name"]
        )
        image = cv2.imread(img_path)
        image = self._apply_clahe(image)

        # 2. Crop lateral + resize
        image_res = cv2.resize(
            image[:, self.start_x : self.end_x],
            (self.new_w, self.new_h),
        )

        # 3. Anotações
        anns = sorted(
            self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)),
            key=lambda a: a["bbox"][1],
        )
        mask = np.zeros((self.new_h, self.new_w), dtype=np.uint8)
        kpts: list[list[float]] = []

        scale_x = self.new_w / (self.end_x - self.start_x)
        scale_y = self.new_h / info["height"]

        for ann in anns:
            if "segmentation" not in ann or len(ann["segmentation"][0]) != 8:
                continue
            pts = np.array(ann["segmentation"][0]).reshape(-1, 2)
            pts[:, 0] = (pts[:, 0] - self.start_x) * scale_x
            pts[:, 1] = pts[:, 1] * scale_y
            kpts.extend(pts.tolist())

            m = cv2.resize(
                self.coco.annToMask(ann)[:, self.start_x : self.end_x],
                (self.new_w, self.new_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = np.maximum(mask, m)

        # 4. Augmentações
        kpts_np = np.array(kpts).reshape(-1, 2) if kpts else np.zeros((0, 2))
        if self.transform:
            aug = self.transform(
                image=image_res, mask=mask, keypoints=kpts_np
            )
            image_res = aug["image"]
            mask = aug["mask"]
            kpts_np = np.array(aug["keypoints"])

        # 5. Normalização dos keypoints → [0, 1]
        kpts_final = np.zeros(136, dtype=np.float32)
        if len(kpts_np) > 0:
            kpts_np = kpts_np.copy().astype(np.float32)
            kpts_np[:, 0] /= self.new_w
            kpts_np[:, 1] /= self.new_h
            flat = kpts_np.flatten()[:136]
            kpts_final[: len(flat)] = flat

        return {
            "image": image_res,
            "mask": mask,
            "keypoints": torch.from_numpy(kpts_final),
            "filename": info["file_name"],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dataset(
    json_path: str,
    img_root: str,
    is_train: bool = True,
    subset_name: str | None = None,
) -> SpinalDataset:
    transform = build_train_transform() if is_train else build_val_transform()
    return SpinalDataset(json_path, img_root, subset_name, transform)
