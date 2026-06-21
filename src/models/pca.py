"""
Treinamento e carregamento do modelo PCA para regularização anatômica.
O PCA aprende a 'coluna média' do dataset e penaliza formas impossíveis.
"""
from pathlib import Path

import joblib
import numpy as np
import torch
from pycocotools.coco import COCO
from sklearn.decomposition import PCA

from src.config import get_config


def train_spine_pca(json_path: str) -> PCA | None:
    """
    Extrai os vetores de forma de todas as colunas com exatamente
    17 vértebras anotadas e treina o PCA.
    """
    cfg = get_config()
    variance = cfg["pca"]["variance_threshold"]
    pca_path = Path(cfg["pca"]["model_path"])

    print("Iniciando extração de coordenadas para o PCA...")
    coco = COCO(json_path)
    shapes = []

    for img_id in coco.getImgIds():
        anns = sorted(
            coco.loadAnns(coco.getAnnIds(imgIds=img_id)),
            key=lambda a: a["bbox"][1],
        )
        pts_list: list[float] = []
        for ann in anns:
            if "segmentation" not in ann:
                continue
            pts = np.array(ann["segmentation"][0], dtype=np.float32).reshape(-1, 2)
            if len(pts) == 4:
                pts[:, 0] /= 512.0
                pts[:, 1] /= 512.0
                pts_list.extend(pts.flatten().tolist())
        if len(pts_list) == 136:
            shapes.append(pts_list)

    if not shapes:
        print("Nenhuma coluna com 17 vértebras encontrada.")
        return None

    shapes_arr = np.array(shapes)
    print(f"Total de amostras para PCA: {len(shapes_arr)}")

    pca = PCA(n_components=variance).fit(shapes_arr)
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, pca_path)
    print(f"PCA treinado com {pca.n_components_} componentes. Salvo em {pca_path}")
    return pca


def load_pca_tensors(
    json_path: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Carrega o PCA (treinando se necessário) e retorna
    (mean_spine, components) como tensores no device correto.
    """
    pca_path = Path(get_config()["pca"]["model_path"])

    if not pca_path.exists():
        train_spine_pca(json_path)

    pca: PCA = joblib.load(pca_path)
    mean_spine = torch.tensor(pca.mean_, dtype=torch.float32).to(device)
    components = torch.tensor(pca.components_, dtype=torch.float32).to(device)
    return mean_spine, components
