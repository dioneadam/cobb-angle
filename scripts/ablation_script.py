"""
Script para comparar múltiplos checkpoints .pth.

Uso:
    python scripts/ablation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.evaluation.ablation import run_ablation_study


def main() -> None:
    cfg = get_config()
    test_cfg = cfg["test"]

    # ── Edite os caminhos dos seus 4 modelos ──────────────────────────────
    MODEL_PATHS = {
        "Baseline (L1 Pura)":                "checkpoints/best_L1.pth",
        "Restrição Global (L1 + PCA)":       "checkpoints/best_L1_PCA.pth",
        "Consistência Local (L1 + HCL)":     "checkpoints/best_L1_HCL.pth",
        "Híbrido Completo (L1 + HCL + PCA)": "checkpoints/best_L1_HCL_PCA.pth",
    }
    # ─────────────────────────────────────────────────────────────────────

    run_ablation_study(
        model_paths=MODEL_PATHS,
        test_json=test_cfg["json_path"],
        test_images=test_cfg["images_root"],
        test_subset=test_cfg["subset_name"],
        test_cobb_gt=test_cfg["cobb_gt_path"],
        output_csv=Path(cfg["results"]["dir"]) / "ablation_study.csv",
    )


if __name__ == "__main__":
    main()
