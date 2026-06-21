"""
Análise exploratória do dataset e verificação de sincronia
entre imagens, máscaras e keypoints.
"""
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset


def analyze_json_consistency(json_path: str) -> None:
    """Histograma da quantidade de vértebras anotadas por imagem."""
    with open(json_path, "r") as f:
        data = json.load(f)

    img_id_to_count = Counter(
        [ann["image_id"] for ann in data["annotations"]]
    )
    counts = list(img_id_to_count.values())

    print(
        f"Total de imagens: {len(data['images'])} | "
        f"Média de vértebras: {np.mean(counts):.2f}"
    )
    plt.figure(figsize=(8, 4))
    plt.hist(
        counts,
        bins=range(min(counts), max(counts) + 2),
        align="left",
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Vértebras por Imagem")
    plt.xlabel("Qtd. vértebras")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()


def visualize_dataset_samples(
    dataloader: DataLoader, num_samples: int = 3
) -> None:
    """
    Exibe amostras do DataLoader com overlay de máscara e keypoints
    para verificar sincronia após pré-processamento e augmentação.
    """
    _MEAN = np.array([0.485, 0.456, 0.406])
    _STD = np.array([0.229, 0.224, 0.225])

    batch = next(iter(dataloader))
    images = batch["image"]
    masks = batch["mask"]
    keypoints = batch["keypoints"]

    plt.figure(figsize=(15, 6 * num_samples))

    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * _STD + _MEAN).clip(0, 1)
        mask = masks[i].cpu().numpy()
        kpts = keypoints[i].cpu().numpy().reshape(-1, 2)
        kpts[:, 0] *= 512
        kpts[:, 1] *= 1024

        # Imagem + máscara
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.3, cmap="Greens")
        plt.title(f"Amostra {i+1} — Augmentação + Máscara")
        plt.axis("off")

        # Keypoints
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(img)
        valid = kpts[np.any(kpts > 0, axis=1)]
        plt.scatter(valid[:, 0], valid[:, 1], c="red", s=20, edgecolors="white")
        for v in range(len(valid) // 4):
            pts = valid[v * 4 : (v + 1) * 4]
            if len(pts) == 4:
                poly = np.array([pts[0], pts[1], pts[3], pts[2], pts[0]])
                plt.plot(poly[:, 0], poly[:, 1], "y-", alpha=0.6, linewidth=1.5)
        plt.title(f"Amostra {i+1} — Sincronia de Keypoints (GT)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
