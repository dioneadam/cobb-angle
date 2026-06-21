"""
Avaliação clínica comparativa entre configurações de modelo (ablation study).
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_config
from src.data.dataset import get_dataset
from src.models.network import SpinalMultiTaskNet
from src.utils.cobb import get_regional_cobb_angles, load_cobb_gt_triple


# ---------------------------------------------------------------------------
# Helper estatístico
# ---------------------------------------------------------------------------

def _compute_stats(errors: list, thresholds: list) -> dict:
    """
    Dado uma lista de erros, retorna MAE, desvio padrão, percentis
    e % de casos dentro de cada threshold.
    """
    arr = np.array(errors)
    result = {
        "MAE":  round(float(np.mean(arr)), 2),
        "Std":  round(float(np.std(arr)),  2),
        "P50":  round(float(np.percentile(arr, 50)), 2),
        "P75":  round(float(np.percentile(arr, 75)), 2),
        "P90":  round(float(np.percentile(arr, 90)), 2),
        "P95":  round(float(np.percentile(arr, 95)), 2),
    }
    for t in thresholds:
        key = f"≤{int(t)}" if t == int(t) else f"≤{t}"
        result[key] = round(float((arr <= t).mean() * 100), 1)
    return result


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

def _run_sanity_check(test_loader, cobb_gt):
    errors = {"pt": [], "mt": [], "tl": [], "max": []}

    for batch in tqdm(test_loader, desc="Sanity Check (Gabarito JSON)", leave=False):
        img_name = batch["filename"][0]
        if img_name not in cobb_gt:
            continue

        gt_pt  = cobb_gt[img_name]["Alta"]
        gt_mt  = cobb_gt[img_name]["Toracica"]
        gt_tl  = cobb_gt[img_name]["Lombar"]
        gt_max = max(gt_pt, gt_mt, gt_tl)

        kpts_perfect = batch["keypoints"].view(-1).cpu().numpy()
        math_pt, math_mt, math_tl = get_regional_cobb_angles(kpts_perfect)
        math_max = max(math_pt, math_mt, math_tl)

        errors["pt"].append(abs(math_pt - gt_pt))
        errors["mt"].append(abs(math_mt - gt_mt))
        errors["tl"].append(abs(math_tl - gt_tl))
        errors["max"].append(abs(math_max - gt_max))

    all_errors = errors["pt"] + errors["mt"] + errors["tl"]
    s = _compute_stats(errors["max"], thresholds=[5, 10])

    return {
        "Configuração da Rede": "Gabarito Matemático (Keypoints Reais)",
        "PT (°)":       round(float(np.mean(errors["pt"])), 2),
        "MT (°)":       round(float(np.mean(errors["mt"])), 2),
        "TL/L (°)":     round(float(np.mean(errors["tl"])), 2),
        "Max Cobb (°)": s["MAE"],
        "±Std (°)":     s["Std"],
        "P90 (°)":      s["P90"],
        "≤5° (%)":      s["≤5"],
        "≤10° (%)":     s["≤10"],
        "Global (°)":   round(float(np.mean(all_errors)), 2),
        "p-value":      "ref.",
    }, errors


# ---------------------------------------------------------------------------
# Avaliação de um modelo — Cobb angle
# ---------------------------------------------------------------------------

def _evaluate_model_cobb(model_name, model_path, test_loader, cobb_gt, device):
    print(f"\nCarregando: {model_name}")

    model = SpinalMultiTaskNet().to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    weights = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(weights)
    model.eval()

    errors = {"pt": [], "mt": [], "tl": [], "max": []}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{model_name}", leave=False):
            images   = batch["image"].to(device)
            img_name = batch["filename"][0]
            if img_name not in cobb_gt:
                continue

            gt_pt  = cobb_gt[img_name]["Alta"]
            gt_mt  = cobb_gt[img_name]["Toracica"]
            gt_tl  = cobb_gt[img_name]["Lombar"]
            gt_max = max(gt_pt, gt_mt, gt_tl)

            with torch.amp.autocast("cuda"):
                _, kpts_p = model(images)

            pred_pt, pred_mt, pred_tl = get_regional_cobb_angles(
                kpts_p.view(-1).cpu().numpy()
            )
            pred_max = max(pred_pt, pred_mt, pred_tl)

            errors["pt"].append(abs(pred_pt - gt_pt))
            errors["mt"].append(abs(pred_mt - gt_mt))
            errors["tl"].append(abs(pred_tl - gt_tl))
            errors["max"].append(abs(pred_max - gt_max))

    del model
    torch.cuda.empty_cache()
    gc.collect()

    all_errors = errors["pt"] + errors["mt"] + errors["tl"]
    s = _compute_stats(errors["max"], thresholds=[5, 10])

    return {
        "Configuração da Rede": model_name,
        "PT (°)":       round(float(np.mean(errors["pt"])), 2),
        "MT (°)":       round(float(np.mean(errors["mt"])), 2),
        "TL/L (°)":     round(float(np.mean(errors["tl"])), 2),
        "Max Cobb (°)": s["MAE"],
        "±Std (°)":     s["Std"],
        "P90 (°)":      s["P90"],
        "≤5° (%)":      s["≤5"],
        "≤10° (%)":     s["≤10"],
        "Global (°)":   round(float(np.mean(all_errors)), 2),
        "p-value":      None,
    }, errors


# ---------------------------------------------------------------------------
# run_ablation_study
# ---------------------------------------------------------------------------

def run_ablation_study(
    model_paths: dict,
    test_json,
    test_images,
    test_cobb_gt,
    test_subset=None,
    output_csv=None,
) -> pd.DataFrame:
    """
    Ablation study de ângulo de Cobb com:
      - MAE por região (PT, MT, TL) e Max Cobb
      - Desvio padrão e percentil P90
      - % dentro de 5° e 10°
    """
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = test_subset or cfg["test"]["subset_name"]

    print(f"\nDispositivo: {device}")
    cobb_gt = load_cobb_gt_triple(test_cobb_gt)

    test_dataset = get_dataset(
        json_path=str(test_json),
        img_root=str(test_images),
        is_train=False,
        subset_name=subset,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    print(f"Dataset de teste: {len(test_dataset)} imagens (subset: {subset})\n")

    results = []

    # Sanity Check
    print("[Fase 1] Sanity Check (keypoints perfeitos do JSON)...")
    sanity_row, _ = _run_sanity_check(test_loader, cobb_gt)
    results.append(sanity_row)

    # Modelos
    print("\n[Fase 2] Avaliando modelos...")
    for name, path in model_paths.items():
        if not Path(path).exists():
            print(f"  AVISO: não encontrado, pulando — {path}")
            continue
        row, _ = _evaluate_model_cobb(name, path, test_loader, cobb_gt, device)
        results.append(row)

    df = pd.DataFrame(results)
    df["_sort"] = df["Configuração da Rede"].apply(lambda x: 0 if "Gabarito" in x else 1)
    df = df.sort_values(["_sort", "Max Cobb (°)"]).drop(columns="_sort").reset_index(drop=True)

    print("\n\n" + "=" * 70)
    print("  TABELA DE ABLAÇÃO — ÂNGULO DE COBB (HOLD-OUT)")
    print("=" * 70)
    print(df.to_markdown(index=False))
    print("=" * 70)
    print("* Gabarito: erro inerente da fórmula com keypoints perfeitos.")
    print("* ≤5°/≤10°: % de casos dentro do threshold.")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nTabela salva em: {output_csv}")

    return df
