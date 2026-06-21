"""
Motor de treinamento: epoch loop, early stopping e K-Fold cross-validation.
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import get_config
from src.data.dataset import get_dataset
from src.models.network import SpinalMultiTaskNet
from src.training.losses import HybridLoss
from src.training.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Salva fisicamente os pesos campeões em disco sempre que um recorde é batido,
    garantindo que o loop de K-Fold recupere os pesos corretos.
    """
    def __init__(self, patience: int = 15, delta: float = 0.0, path: str = 'checkpoint_es.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_mae_min = float('inf')

    def __call__(self, val_mae: float, model: torch.nn.Module) -> None:
        score = -val_mae

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_mae, model) # Salva o primeiro recorde
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(val_mae, model) # Salva sempre que houver um novo recorde
            self.counter = 0

    def save_checkpoint(self, val_mae: float, model: torch.nn.Module) -> None:
        """Salva o modelo localmente quando o MAE de validação diminui."""
        torch.save(model.state_dict(), self.path)
        self.val_mae_min = val_mae


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: SpinalMultiTaskNet,
    loader: DataLoader,
    criterion: HybridLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    is_train: bool = True,
) -> dict:
    """Executa uma época completa de treino ou validação."""
    model.train() if is_train else model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    total_loss = 0.0
    mae_list: list[float] = []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False, desc="train" if is_train else "val"):
            images = batch["image"].to(device)
            masks_gt = batch["mask"].unsqueeze(1).float().to(device)
            kpts_gt = batch["keypoints"].to(device)

            with torch.amp.autocast("cuda"):
                masks_p, kpts_p = model(images)
                loss = criterion(masks_p, masks_gt, kpts_p, kpts_gt)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

            # MAE em pixels reais (resolução de treino)
            cfg = get_config()["data"]
            p = kpts_p.view(-1, 68, 2)
            g = kpts_gt.view(-1, 68, 2)
            err = torch.abs(p - g)
            ex = err[..., 0] * cfg["image_width"]
            ey = err[..., 1] * cfg["image_height"]
            dist = torch.sqrt(ex**2 + ey**2)
            mae_list.append(dist.mean().item())

    if device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "loss": total_loss / len(loader),
        "mae_kpts": float(np.mean(mae_list)),
        "duration_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# K-Fold Training
# ---------------------------------------------------------------------------

def run_kfold_training(
    json_path: str,
    img_root: str,
    device: torch.device,
    pca_mean: torch.Tensor,
    pca_components: torch.Tensor,
    checkpoint_manager: CheckpointManager,
) -> None:
    cfg = get_config()
    t_cfg = cfg["training"]
    epochs = t_cfg["epochs"]
    batch_size = t_cfg["batch_size"]
    lr = t_cfg["learning_rate"]
    num_workers = t_cfg["num_workers"]
    k_folds = t_cfg["k_folds"]
    patience = t_cfg["early_stopping_patience"]
    results_dir = Path(cfg["results"]["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = get_dataset(json_path, img_root, is_train=True)
    all_ids = list(range(len(full_dataset)))

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=t_cfg["random_seed"])

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
        fold_n = fold + 1
        print(f"\n{'='*45}")
        print(f"  FOLD {fold_n}/{k_folds}")
        print(f"{'='*45}")

        # Define o nome do arquivo csv
        csv_filename = f"metrics_fold_{fold_n}.csv" 
        
        # Monta o caminho local
        csv_path = results_dir / csv_filename

        # --- Resume ---
        start_epoch = 1
        best_fold_mae = float("inf")
        ckpt = checkpoint_manager.load_best_fold(fold_n, device)

        if ckpt:
            if ckpt.get("converged", False):
                print(
                    f"[SKIP] Fold {fold_n} já convergiu na época "
                    f"{ckpt.get('epoch')}. Pulando."
                )
                continue
            start_epoch = ckpt.get("epoch", 0) + 1
            best_fold_mae = ckpt.get("mae", float("inf"))
            print(f"Retomada detectada! Continuando da época {start_epoch}.")

        # --- DataLoaders ---
        train_loader = DataLoader(
            Subset(get_dataset(json_path, img_root, is_train=True), train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            Subset(get_dataset(json_path, img_root, is_train=False), val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # --- Modelo e Otimizador ---
        model = SpinalMultiTaskNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        if start_epoch > 1 and ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        criterion = HybridLoss(pca_mean, pca_components).to(device)
        scaler = torch.amp.GradScaler("cuda")
        best_es_path = str(Path(cfg["checkpoints"]["dir"]) / f"checkpoint_es_fold_{fold_n}.pth")
        early_stopping = EarlyStopping(patience=patience, path=best_es_path)
        fold_logs: list[dict] = _resume_csv(csv_path, start_epoch, csv_filename, checkpoint_manager)

        # --- Loop de Épocas ---
        for epoch in range(start_epoch, epochs + 1):
            train_res = run_epoch(
                model, train_loader, criterion, optimizer, scaler, device, True
            )
            val_res = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device, False
            )

            val_mae = val_res["mae_kpts"]
            is_best = val_mae < best_fold_mae
            if is_best:
                best_fold_mae = val_mae

            fold_logs = _append_csv(
                fold_logs, fold_n, epoch,
                optimizer.param_groups[0]["lr"],
                train_res, val_res, csv_path,
                checkpoint_manager
            )

            # Persistência padrão de época (estado atual, não necessariamente o melhor)
            checkpoint_manager.save(
                model, optimizer, epoch, fold_n, val_mae, is_best, converged=False
            )

            # Early stopping salva os pesos campeões localmente quando batem recorde
            early_stopping(val_mae, model)

            if epoch % 5 == 0 or epoch == start_epoch:
                print(
                    f"Ep [{epoch}/{epochs}] | "
                    f"Loss: {train_res['loss']:.4f} | "
                    f"MAE Val: {val_mae:.2f} px"
                )

            if early_stopping.early_stop or epoch == epochs:
                reason = "Early Stopping" if early_stopping.early_stop else "Fim das épocas"
                print(f"{reason} no Fold {fold_n}, Época {epoch}.")

                # Recupera os pesos campeões reais antes do salvamento definitivo
                try:
                    model.load_state_dict(torch.load(best_es_path, map_location=device))
                    print("Pesos campeões recuperados para o salvamento definitivo.")
                except Exception as e:
                    print(f"Erro ao recuperar pesos do EarlyStopping: {e}. Salvando estado atual.")

                # Carimba converged=True com os pesos corretos e o melhor MAE registrado
                checkpoint_manager.save(
                    model, optimizer, epoch, fold_n, best_fold_mae,
                    is_best_in_fold=True, converged=True,
                )
                break

        del model, optimizer, train_loader, val_loader, criterion
        torch.cuda.empty_cache()
        gc.collect()

    print("\nTreinamento K-Fold finalizado com sucesso.")


# ---------------------------------------------------------------------------
# Helpers de CSV
# ---------------------------------------------------------------------------

def _resume_csv(
    csv_path: Path, 
    start_epoch: int, 
    drive_csv: str, 
    checkpoint_manager: CheckpointManager
) -> list[dict]:
    if start_epoch > 1:
        if checkpoint_manager._drive and checkpoint_manager._drive.download(drive_csv, csv_path):
            try:
                logs = pd.read_csv(csv_path).to_dict(orient="records")
                print(f"Histórico do CSV sincronizado com sucesso até a época {start_epoch-1}")
                return [r for r in logs if r["epoch"] < start_epoch]
            except Exception as e:
                print(f"Falha ao ler histórico do CSV baixado ({e}). Iniciando log vazio.")
    return []


def _append_csv(
    logs: list[dict],
    fold: int,
    epoch: int,
    lr: float,
    train_res: dict,
    val_res: dict,
    csv_path: Path,
    checkpoint_manager: CheckpointManager,
) -> list[dict]:
    logs.append(
        {
            "fold": fold,
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_res["loss"],
            "train_mae_kpts": train_res["mae_kpts"],
            "train_time_sec": train_res["duration_sec"],
            "val_loss": val_res["loss"],
            "val_mae_kpts": val_res["mae_kpts"],
            "val_time_sec": val_res["duration_sec"],
            "total_epoch_time_sec": train_res["duration_sec"] + val_res["duration_sec"],
        }
    )
    
    # Salva fisicamente o arquivo local
    pd.DataFrame(logs).to_csv(csv_path, index=False)
    
    checkpoint_manager._sync(csv_path)
        
    return logs
