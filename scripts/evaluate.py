"""
Script de avaliação e geração de laudos visuais.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --samples 20 --save-plots
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import get_config
from src.data.dataset import get_dataset
from src.evaluation.visualizer import (
    run_evaluation_gallery,
    run_final_validation,
    evaluate_dataset_triple_cobb,
)
from src.models.network import SpinalMultiTaskNet
from src.training.checkpoint import CheckpointManager
from src.utils.cobb import load_cobb_gt, load_cobb_gt_triple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o modelo SpinalCobbAI")
    parser.add_argument("--samples", type=int, default=15,
                        help="Número de amostras para galeria visual")
    parser.add_argument("--val-samples", type=int, default=50,
                        help="Número de amostras para validação estatística")
    parser.add_argument("--save-plots", action="store_true",
                        help="Salva os painéis em vez de exibir na tela")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Dataset de avaliação (sem augmentações)
    dataset = get_dataset(
        cfg["data"]["json_path"],
        cfg["data"]["images_root"],
        is_train=False,
    )

    # Carrega o melhor modelo
    model = SpinalMultiTaskNet().to(device)
    ckpt_manager = CheckpointManager()
    model = ckpt_manager.load_champion(model, device)
    model.eval()

    # Ground truth clínico
    gt_map = load_cobb_gt(cfg["data"]["cobb_gt_path"])
    gt_map_triple = load_cobb_gt_triple(cfg["data"]["cobb_gt_path"])

    # Galeria visual
    output_dir = Path(cfg["results"]["dir"]) / "plots" if args.save_plots else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    run_evaluation_gallery(
        model=model,
        dataset=dataset,
        device=device,
        gt_map=gt_map,
        num_samples=args.samples,
        output_dir=output_dir,
    )

    # Validação estatística
    metrics = run_final_validation(
        model=model,
        dataset=dataset,
        device=device,
        gt_map=gt_map,
        num_samples=args.val_samples,
    )

    # Avaliação por região (3 ângulos)
    df = evaluate_dataset_triple_cobb(
        model=model,
        dataset=dataset,
        device=device,
        gt_map=gt_map_triple,
    )
    results_path = Path(cfg["results"]["dir"]) / "triple_cobb_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResultados salvos em {results_path}")


if __name__ == "__main__":
    main()
