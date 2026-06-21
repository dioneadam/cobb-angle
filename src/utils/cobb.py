"""
Utilitários para cálculo do ângulo de Cobb e carregamento do gabarito clínico.
"""
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Cálculo do ângulo de Cobb
# ---------------------------------------------------------------------------

def get_detailed_cobb(
    kpts_pred,
    inicio_busca: int = 2,
    fim_busca: int = 16,
) -> dict:
    """
    Calcula o ângulo de Cobb com dupla ordenação espacial e filtro de segurança.

    Args:
        kpts_pred   : array flat (136,) ou (17, 4, 2) com coordenadas normalizadas
        inicio_busca: índice de vértebra onde começa a busca (base-0)
        fim_busca   : índice de vértebra onde termina a busca (exclusivo)

    Returns:
        dict com chaves: angle, v_superior, v_inferior, all_slopes, coords
    """
    kpts = np.array(kpts_pred).reshape(17, 4, 2)
    scale = np.array([512.0, 512.0])
    vertebrae_real = kpts * scale

    slopes: list[float | None] = []
    valid_indices: list[int] = []

    for i in range(17):
        v_pts = vertebrae_real[i]

        if np.all(v_pts == 0):
            slopes.append(None)
            continue

        # Identifica face superior (menor Y) e ordena esq → dir
        v_pts_sorted_y = v_pts[v_pts[:, 1].argsort()]
        top_pair = v_pts_sorted_y[:2]
        p1, p2 = top_pair[top_pair[:, 0].argsort()]

        angle_rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        slopes.append(np.degrees(angle_rad))
        valid_indices.append(i)

    slopes_no_segmento = [
        i for i in range(inicio_busca, fim_busca) if i in valid_indices
    ]

    if not slopes_no_segmento:
        return {
            "angle": 0.0,
            "v_superior": 0,
            "v_inferior": 0,
            "all_slopes": slopes,
            "coords": kpts,
        }

    slopes_busca = [slopes[i] for i in slopes_no_segmento]
    idx_sup = slopes_no_segmento[int(np.argmin(slopes_busca))]
    idx_inf = slopes_no_segmento[int(np.argmax(slopes_busca))]

    diff = abs(slopes[idx_sup] - slopes[idx_inf])
    cobb_angle = abs(180 - diff) if diff > 90 else diff

    return {
        "angle": float(cobb_angle),
        "v_superior": idx_sup,
        "v_inferior": idx_inf,
        "all_slopes": slopes,
        "coords": kpts,
    }


def calculate_three_cobb_angles(kpts_flat) -> dict:
    """
    Calcula os 3 ângulos de Cobb clínicos (PT, MT, TL)
    usando janelas anatômicas fixas.
    """
    windows = {
        "PT_alto":      (0, 5),
        "MT_toracico":  (5, 14),
        "TL_lombar":    (14, 17),
    }
    result = {}
    for name, (ini, fim) in windows.items():
        res = get_detailed_cobb(kpts_flat, inicio_busca=ini, fim_busca=fim)
        result[f"{name}_angle"] = res["angle"]
        result[f"{name}_limites"] = (res["v_superior"] + 1, res["v_inferior"] + 1)
    return result


def get_regional_cobb_angles(kpts_flat) -> tuple[float, float, float]:
    """
    Retorna (cobb_pt, cobb_mt, cobb_tl) usando slopes diretos por região.
    Alternativa estável ao método de busca de extremos.
    """

    kpts = np.array(kpts_flat).reshape(17, 4, 2)
    scale = np.array([512.0, 1024.0])
    vertebrae_real = kpts * scale

    slopes = []
    for i in range(17):
        v_pts = vertebrae_real[i]

        # Proteção contra vértebras ausentes
        if np.all(v_pts == 0):
            slopes.append(0.0)
            continue

        v_pts_sorted_y = v_pts[v_pts[:, 1].argsort()]
        top_pair = v_pts_sorted_y[:2]
        p1, p2 = top_pair[top_pair[:, 0].argsort()]
        slopes.append(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))

    def _cobb(a, b):
        diff = abs(slopes[a] - slopes[b])
        return abs(180 - diff) if diff > 90 else diff

    return _cobb(0, 5), _cobb(5, 12), _cobb(12, 16)


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------

def load_cobb_gt(file_path: str | Path) -> dict:
    """
    Lê o arquivo .txt do SpinalAI-2024 e retorna um dicionário
    {filename: max(PT, MT, TL)} com a curva principal de cada imagem.
    """
    gt_map: dict = {}
    path = Path(file_path)

    if not path.exists():
        print(f"Arquivo GT não encontrado: {file_path}")
        return gt_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                filename = parts[0]
                angles = [float(parts[1]), float(parts[2]), float(parts[3])]
                gt_map[filename] = max(angles)
            except ValueError:
                continue

    print(f"{len(gt_map)} gabaritos carregados de {path.name}")
    return gt_map


def load_cobb_gt_triple(file_path: str | Path) -> dict:
    """
    Versão completa: retorna {filename: {'Alta': pt, 'Toracica': mt, 'Lombar': tl, 'Lista': [...]}}
    """
    gt_map: dict = {}
    path = Path(file_path)

    if not path.exists():
        print(f"Arquivo GT não encontrado: {file_path}")
        return gt_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                filename = parts[0]
                pt, mt, tl = float(parts[1]), float(parts[2]), float(parts[3])
                gt_map[filename] = {
                    "Alta": pt,
                    "Toracica": mt,
                    "Lombar": tl,
                    "Lista": [pt, mt, tl],
                }
            except ValueError:
                continue

    print(f"{len(gt_map)} gabaritos triplos carregados.")
    return gt_map
