"""
Arquitetura da rede multi-tarefa: segmentação + regressão de keypoints.
Backbone: HRNet-W32.
"""
import timm
import torch
import torch.nn as nn

from src.config import get_config


class SpinalMultiTaskNet(nn.Module):
    """
    Rede multi-tarefa para análise de coluna vertebral em raio-X.

    Saídas:
        mask_pred  : Tensor [B, 1, H, W] — máscara de segmentação (sigmoid)
        kpts_pred  : Tensor [B, 136]     — coordenadas normalizadas (sigmoid)
    """

    # Canais de cada nível de feature da HRNet-W32
    _CHANNELS_SEG = 64    # features[0] — alta resolução para bordas nítidas
    _CHANNELS_REG = 1024  # features[-1] — semântica profunda para coordenadas

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config()
        img_h = cfg["data"]["image_height"]
        img_w = cfg["data"]["image_width"]
        num_kpts = cfg["model"]["num_keypoints"]
        backbone_name = cfg["model"]["backbone"]
        pretrained = cfg["model"]["pretrained"]

        # Backbone — retorna lista de feature maps em múltiplas resoluções
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )

        # Cabeça de Segmentação (usa feature de alta resolução)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self._CHANNELS_SEG, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Upsample(size=(img_h, img_w), mode="bilinear"),
            nn.Sigmoid(),
        )

        # Cabeça Geométrica (usa feature de semântica profunda)
        self.geometric_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self._CHANNELS_REG, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_kpts),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mask_pred = self.segmentation_head(features[0])
        kpts_pred = self.geometric_head(features[-1])
        return mask_pred, kpts_pred
