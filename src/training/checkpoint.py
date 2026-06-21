"""
Gerenciamento de checkpoints locais.
Opcional: sync com Google Drive (ativado no config.yaml).
"""
from pathlib import Path

import torch

from src.config import get_config


class CheckpointManager:
    """
    Salva e carrega checkpoints de modelo localmente.
    Se drive.enabled=true no config, espelha também no Google Drive.
    """

    def __init__(self, drive_service=None) -> None:
        cfg = get_config()
        self.ckpt_dir = Path(cfg["checkpoints"]["dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_global_path = self.ckpt_dir / cfg["checkpoints"]["best_global"]
        self.best_global_mae = float("inf")

        # Drive
        self._drive = None
        if drive_service and get_config()["drive"]["enabled"]:
            from src.utils.drive import DriveSync
            folder_id = get_config()["drive"]["checkpoint_folder_id"]
            self._drive = DriveSync(drive_service, folder_id)

    # ------------------------------------------------------------------
    # Salvar
    # ------------------------------------------------------------------

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        fold: int,
        mae: float,
        is_best_in_fold: bool,
        converged: bool = False,
    ) -> None:
        state = {
            "epoch": epoch,
            "fold": fold,
            "mae": mae,
            "converged": converged,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        latest_path = self.ckpt_dir / f"latest_fold_{fold}.pth"
        best_path = self.ckpt_dir / f"best_model_fold_{fold}.pth"

        # Sempre salva e sincroniza o latest para acompanhamento
        torch.save(state, latest_path)
        self._sync(latest_path)

        # Salva o best apenas se for o melhor do fold
        if is_best_in_fold:
            torch.save(state, best_path)
            self._sync(best_path)

            if mae < self.best_global_mae:
                self.best_global_mae = mae
                torch.save(model.state_dict(), self.best_global_path)
                self._sync(self.best_global_path)
                print(f"NOVO RECORDE GLOBAL — Fold {fold}, MAE: {mae:.2f} px")

    # ------------------------------------------------------------------
    # Carregar
    # ------------------------------------------------------------------

    def load_best_fold(
        self, fold: int, device: torch.device
    ) -> dict | None:
        """Carrega o melhor checkpoint de um fold (se existir)."""
        path = self.ckpt_dir / f"best_model_fold_{fold}.pth"
        if not path.exists():
            # Tenta baixar do Drive se disponível
            if self._drive:
                ok = self._drive.download(f"best_model_fold_{fold}.pth", path)
                if not ok:
                    return None
            else:
                return None
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Erro ao carregar checkpoint fold {fold}: {e}")
            return None

    def load_champion(
        self, model: torch.nn.Module, device: torch.device
    ) -> torch.nn.Module:
        """Injeta os pesos do melhor modelo global."""
        if not self.best_global_path.exists() and self._drive:
            self._drive.download(
                self.best_global_path.name, self.best_global_path
            )

        if self.best_global_path.exists():
            ckpt = torch.load(
                self.best_global_path, map_location=device, weights_only=False
            )
            weights = (
                ckpt["model_state_dict"]
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                else ckpt
            )
            model.load_state_dict(weights)
            mae_info = ckpt.get("mae", "N/A") if isinstance(ckpt, dict) else "N/A"
            print(f"Campeão Global carregado! (MAE: {mae_info})")
        else:
            print("Arquivo de campeão não encontrado. Usando pesos aleatórios.")

        return model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync(self, local_path: Path) -> None:
        if self._drive:
            try:
                self._drive.upload(local_path, local_path.name)
            except Exception as e:
                print(f"Drive sync falhou para {local_path.name}: {e}")
