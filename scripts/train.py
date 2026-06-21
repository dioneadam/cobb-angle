"""
Script principal de treinamento.

Uso:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

# Garante que o pacote src seja encontrado ao rodar da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import load_config, get_config
from src.models.pca import load_pca_tensors
from src.training.checkpoint import CheckpointManager
from src.training.trainer import run_kfold_training

from google.colab import auth
from googleapiclient.discovery import build

def get_drive_service():
    auth.authenticate_user()
    return build('drive', 'v3')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo SpinalCobbAI")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho para um config.yaml alternativo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Carrega config (usa o padrão se não especificado)
    if args.config:
        import src.config as _cfg_module
        _cfg_module._cfg = load_config(args.config)

    cfg = get_config()

    drive_service = None
    if cfg["drive"]["enabled"]:
        drive_service = get_drive_service()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    json_path = cfg["data"]["json_path"]
    img_root = cfg["data"]["images_root"]

    # PCA
    pca_mean, pca_components = load_pca_tensors(json_path, device)

    # Checkpoint manager (sem Google Drive por padrão)
    ckpt_manager = CheckpointManager(drive_service=drive_service)

    # Treino
    run_kfold_training(
        json_path=json_path,
        img_root=img_root,
        device=device,
        pca_mean=pca_mean,
        pca_components=pca_components,
        checkpoint_manager=ckpt_manager,
    )


if __name__ == "__main__":
    main()
