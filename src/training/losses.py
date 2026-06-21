"""
Funções de perda do treinamento:
  - DiceLoss         : segmentação de máscara
  - pca_loss         : regularização anatômica via PCA
  - compute_hybrid_loss : loss final ponderada
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_config


class DiceLoss(nn.Module):
    """Dice Loss para segmentação binária."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, predict: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        predict = predict.view(-1)
        target = target.view(-1)
        intersection = (predict * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predict.sum() + target.sum() + self.smooth
        )
        return 1 - dice


class HybridLoss(nn.Module):
    """
    Loss híbrida com 3 componentes:
      1. Dice Loss        — segmentação da massa óssea
      2. Smooth L1        — regressão das coordenadas dos keypoints
      3. PCA Loss         — regularização anatômica

    Os componentes são ponderados pelos pesos definidos no config.yaml.
    """

    def __init__(
        self,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
    ) -> None:
        super().__init__()
        cfg = get_config()["loss"]
        self.w_seg = cfg["w_seg"]
        self.w_reg = cfg["w_reg"]
        self.w_pca = cfg["w_pca"]
        self.beta = get_config()["training"]["beta_smooth"]

        self.dice = DiceLoss()
        self.register_buffer("mean_spine", pca_mean)
        self.register_buffer("components", pca_components)

    # ------------------------------------------------------------------
    # Componentes individuais
    # ------------------------------------------------------------------

    def _pca_loss(self, pred_kpts: torch.Tensor) -> torch.Tensor:
        """
        Penaliza predições que representam formas anatomicamente impossíveis.
        Projeta os keypoints no espaço PCA e mede o erro de reconstrução.
        """
        centered = pred_kpts - self.mean_spine
        proj = torch.matmul(centered, self.components.t())
        recon = torch.matmul(proj, self.components) + self.mean_spine
        return F.mse_loss(pred_kpts, recon)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        mask_pred: torch.Tensor,
        mask_gt: torch.Tensor,
        kpts_pred: torch.Tensor,
        kpts_gt: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Segmentação
        l_seg = self.dice(mask_pred, mask_gt)

        # 2. Regressão de keypoints (ignora pontos sem anotação)
        mask_k = (kpts_gt > 0).float()
        l_reg_raw = F.smooth_l1_loss(
            kpts_pred, kpts_gt, reduction="none", beta=self.beta
        )
        l_reg = (l_reg_raw * mask_k).sum() / (mask_k.sum() + 1e-6)

        # 3. Regularização anatômica
        l_pca = self._pca_loss(kpts_pred)

        return self.w_seg * l_seg + self.w_reg * l_reg + self.w_pca * l_pca
