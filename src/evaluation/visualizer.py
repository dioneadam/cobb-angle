"""
Avaliação e visualização de resultados
"""
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.network import SpinalMultiTaskNet
from src.utils.cobb import get_detailed_cobb

from torch.utils.data import DataLoader
from src.config import get_config
from src.data.dataset import get_dataset

# ---------------------------------------------------------------------------
# Helpers visuais
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(tensor_img: torch.Tensor) -> np.ndarray:
    """Reverte a normalização ImageNet para exibição."""
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    return (img * _IMAGENET_STD + _IMAGENET_MEAN).clip(0, 1)


def denormalize_kpts(
    flat_kpts: torch.Tensor | np.ndarray,
    w: int = 512,
    h: int = 1024,
) -> np.ndarray:
    """Converte keypoints de [0,1] para pixels."""
    kpts = np.array(flat_kpts).reshape(17, 4, 2).copy()
    kpts[:, :, 0] *= w
    kpts[:, :, 1] *= h
    return kpts


def draw_skeleton(
    ax,
    kpts_flat,
    color_poly: str = "lime",
    color_pts: str = "red",
    w: int = 512,
    h: int = 1024,
) -> None:
    """Desenha polígonos e pontos das vértebras sobre um eixo Matplotlib."""
    kpts = np.array(kpts_flat).reshape(17, 4, 2)
    scale = np.array([w, h])
    vertebrae_real = kpts * scale

    for i in range(17):
        pts = vertebrae_real[i]
        if np.all(pts == 0):
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=15, c=color_pts, edgecolors="white", zorder=3)
        poly = patches.Polygon(
            pts[[0, 1, 3, 2]],
            linewidth=1.5,
            edgecolor=color_poly,
            facecolor="none",
            alpha=0.8,
            zorder=2,
        )
        ax.add_patch(poly)


# ---------------------------------------------------------------------------
# Painel comparativo principal
# ---------------------------------------------------------------------------

def visualize_comparative_results(
    sample_idx: int,
    model: SpinalMultiTaskNet,
    dataset: Dataset,
    device: torch.device,
    gt_map: dict | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Gera painel de 3 colunas: Original | Ground Truth | Predição IA.
    Inclui ângulo de Cobb estimado e métricas de erro.
    """
    model.eval()
    sample = dataset[sample_idx]
    image_tensor = sample["image"].to(device).unsqueeze(0)
    filename = sample["filename"]
    gt_kpts_raw = sample["keypoints"].numpy().flatten()

    with torch.no_grad():
        mask_pred, kpts_pred = model(image_tensor)
    pred_kpts_raw = kpts_pred.cpu().numpy().flatten()

    img_viz = denormalize_image(sample["image"])

    res_ia = get_detailed_cobb(pred_kpts_raw, inicio_busca=0, fim_busca=17)
    res_gt = get_detailed_cobb(gt_kpts_raw, inicio_busca=0, fim_busca=17)
    cobb_ia = res_ia["angle"]
    cobb_json = res_gt["angle"]
    cobb_doc = gt_map.get(filename, cobb_json) if gt_map else cobb_json
    v_sup = res_ia["v_superior"]
    v_inf = res_ia["v_inferior"]

    fig, ax = plt.subplots(1, 3, figsize=(24, 11))

    # Coluna 1 — Original
    ax[0].imshow(img_viz)
    ax[0].set_title(f"1. Imagem Original\nID: {filename}", fontsize=12, fontweight="bold")
    ax[0].axis("off")

    # Coluna 2 — Ground Truth
    ax[1].imshow(img_viz)
    draw_skeleton(ax[1], gt_kpts_raw, color_poly="cyan", color_pts="blue")
    ax[1].set_title(
        f"2. Gabarito do Especialista\nLaudo .txt: {cobb_doc:.1f}°  |  JSON: {cobb_json:.1f}°",
        fontsize=12, fontweight="bold", color="blue", pad=10,
    )
    ax[1].axis("off")

    # Coluna 3 — IA
    ax[2].imshow(img_viz)
    m_p = mask_pred.squeeze().cpu().numpy()
    ax[2].imshow(m_p, alpha=0.3, cmap="viridis")
    draw_skeleton(ax[2], pred_kpts_raw, color_poly="lime", color_pts="red")

    # Linhas de Cobb
    scale_visual = np.array([512.0, 1024.0])
    kpts_pixels = pred_kpts_raw.reshape(17, 4, 2) * scale_visual
    v_topo = min(v_sup, v_inf)
    v_base = max(v_sup, v_inf)

    for idx, cor, label in [(v_topo, "magenta", "Topo"), (v_base, "yellow", "Base")]:
        if np.all(kpts_pixels[idx] == 0):
            continue
        p1, p2 = kpts_pixels[idx, 0], kpts_pixels[idx, 1]
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6)
        x_vals = np.array([p1[0] - 160, p2[0] + 160])
        y_vals = np.array([p1[1] - slope * 160, p2[1] + slope * 140])
        ax[2].plot(x_vals, y_vals, color=cor, linewidth=3.0, linestyle="--")
        cx, cy = np.mean(kpts_pixels[idx, :, 0]), np.mean(kpts_pixels[idx, :, 1])
        ax[2].text(
            cx + 25, cy, f"V{idx+1} ({label})", color="white", fontsize=9,
            fontweight="bold", bbox=dict(facecolor=cor, alpha=0.85, edgecolor="none"),
        )

    diff_doc = abs(cobb_ia - cobb_doc)
    diff_json = abs(cobb_ia - cobb_json)
    ax[2].set_title(
        f"3. Laudo Automatizado (IA)\n"
        f"Cobb: {cobb_ia:.1f}° (V{v_topo+1}–V{v_base+1})\n"
        f"Δ vs Texto: {diff_doc:.1f}°  |  Δ vs JSON: {diff_json:.1f}°",
        fontsize=12, fontweight="bold", color="darkgreen", pad=10,
    )
    ax[2].axis("off")

    # Legenda
    handles = [
        mlines.Line2D([], [], color="red", marker="o", linestyle="None",
                      markeredgecolor="white", markersize=7, label="Keypoints IA"),
        mlines.Line2D([], [], color="magenta", linestyle="--", linewidth=2,
                      label=f"Limite Superior (V{v_topo+1})"),
        mlines.Line2D([], [], color="yellow", linestyle="--", linewidth=2,
                      label=f"Limite Inferior (V{v_base+1})"),
        patches.Patch(color="#1f9e89", alpha=0.3, label="Massa Óssea Segmentada"),
    ]
    ax[2].legend(handles=handles, loc="upper right", fontsize=9,
                 framealpha=0.95, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Painel salvo em {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Galeria em lote
# ---------------------------------------------------------------------------

def run_evaluation_gallery(
    model: SpinalMultiTaskNet,
    dataset: Dataset,
    device: torch.device,
    gt_map: dict | None = None,
    num_samples: int = 15,
    output_dir: Path | None = None,
) -> None:
    """Gera painéis comparativos para N amostras."""
    total = min(num_samples, len(dataset))
    print(f"Gerando {total} painéis diagnósticos...")
    for i in range(total):
        save_path = output_dir / f"painel_{i:03d}.png" if output_dir else None
        visualize_comparative_results(i, model, dataset, device, gt_map, save_path)
    print("Renderização concluída.")


# ---------------------------------------------------------------------------
# Validação estatística cruzada
# ---------------------------------------------------------------------------

def run_final_validation(
    model: SpinalMultiTaskNet,
    dataset: Dataset,
    device: torch.device,
    gt_map: dict | None = None,
    num_samples: int = 50,
) -> dict:
    """
    Valida o modelo contra o laudo em texto (.txt) e contra a geometria
    dos keypoints anotados (JSON). Retorna MAE e desvio padrão para ambos.
    """
    model.eval()
    errors_doc, errors_json = [], []

    print(f"Iniciando Validação Cruzada em {num_samples} amostras...\n")
    header = f"{'Amostra':<12} | {'Cobb IA':>7} | {'Cobb Doc':>8} | {'Cobb JSON':>9} | {'Err Doc':>7} | {'Err JSON':>8}"
    print(header)
    print("-" * len(header))

    total = min(num_samples, len(dataset))
    for i in range(total):
        sample = dataset[i]
        filename = sample["filename"]
        gt_kpts_raw = sample["keypoints"].numpy().flatten()

        with torch.no_grad():
            _, kpts_pred = model(sample["image"].unsqueeze(0).to(device))
        pred_raw = kpts_pred.cpu().numpy().flatten()

        angle_ia = get_detailed_cobb(pred_raw, 0, 17)["angle"]
        angle_json = get_detailed_cobb(gt_kpts_raw, 0, 17)["angle"]
        angle_doc = gt_map.get(filename, angle_json) if gt_map else angle_json

        err_doc = abs(angle_ia - angle_doc)
        err_json = abs(angle_ia - angle_json)
        errors_doc.append(err_doc)
        errors_json.append(err_json)

        if i < 10:
            print(
                f"{filename:<12} | {angle_ia:>6.1f}° | {angle_doc:>7.1f}° | "
                f"{angle_json:>8.1f}° | {err_doc:>6.1f}° | {err_json:>7.1f}°"
            )

    mae_doc = float(np.mean(errors_doc))
    std_doc = float(np.std(errors_doc))
    mae_json = float(np.mean(errors_json))
    std_json = float(np.std(errors_json))

    print(f"\n{'='*60}")
    print("  RELATÓRIO ESTATÍSTICO")
    print(f"{'='*60}")
    print(f"  Métrica Clínica (IA vs .txt) : MAE = {mae_doc:.2f}° ± {std_doc:.2f}°")
    print(f"  Métrica Geométrica (IA vs JSON): MAE = {mae_json:.2f}° ± {std_json:.2f}°")
    print(f"{'='*60}")
    if mae_json <= 5.0:
        print("  META ATINGIDA: MAE geométrico < 5° ✓")

    return {
        "mae_doc": mae_doc, "std_doc": std_doc,
        "mae_json": mae_json, "std_json": std_json,
    }


# ---------------------------------------------------------------------------
# Avaliação em lote — 3 ângulos
# ---------------------------------------------------------------------------

def evaluate_dataset_triple_cobb(
    model: SpinalMultiTaskNet,
    dataset: Dataset,
    device: torch.device,
    gt_map: dict | None = None,
) -> pd.DataFrame:
    """
    Roda a inferência em todo o dataset e calcula os 3 ângulos clínicos,
    comparando IA vs JSON e IA vs TXT para cada região anatômica.
    """
    model.eval()
    windows = {"Alta": (0, 5), "Toracica": (5, 12), "Lombar": (12, 17)}
    records = []

    print(f"Avaliando {len(dataset)} amostras (3 ângulos)...")

    for idx in range(len(dataset)):
        sample = dataset[idx]
        filename = sample["filename"]
        gt_kpts_raw = sample["keypoints"].numpy().flatten()

        with torch.no_grad():
            _, kpts_pred = model(sample["image"].unsqueeze(0).to(device))
        pred_raw = kpts_pred.cpu().numpy().flatten()

        row: dict = {"filename": filename}
        txt_ok = gt_map is not None and filename in gt_map

        for name, (ini, fim) in windows.items():
            res_ia = get_detailed_cobb(pred_raw, ini, fim)
            res_gt = get_detailed_cobb(gt_kpts_raw, ini, fim)
            row[f"IA_{name}"] = res_ia["angle"]
            row[f"JSON_{name}"] = res_gt["angle"]

            if txt_ok and isinstance(gt_map[filename], dict):
                row[f"TXT_{name}"] = gt_map[filename].get(name, res_gt["angle"])
            else:
                row[f"TXT_{name}"] = res_gt["angle"]

        records.append(row)

    df = pd.DataFrame(records)

    print(f"\n{'='*50}")
    print("  RELATÓRIO DE MAE — 3 REGIÕES ANATÔMICAS")
    print(f"{'='*50}")
    for name in windows:
        mae_j = np.mean(np.abs(df[f"IA_{name}"] - df[f"JSON_{name}"]))
        mae_t = np.mean(np.abs(df[f"IA_{name}"] - df[f"TXT_{name}"]))
        print(f"  {name.upper():<10} | vs JSON: {mae_j:.2f}°  | vs TXT: {mae_t:.2f}°")
    print(f"{'='*50}")

    return df


# ---------------------------------------------------------------------------
# Plot Samples
# ---------------------------------------------------------------------------

def plot_ablation_samples(
    model_paths: dict[str, str | Path],
    test_json: str | Path,
    test_images: str | Path,
    num_samples: int = 3,
    test_subset: str | None = None
):
    """
    Gera subplots lado a lado (1x5) comparando visualmente o Ground Truth 
    com as predições dos 4 modelos.
    """
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = test_subset or cfg["test"]["subset_name"]
    
    test_dataset = get_dataset(
        json_path=str(test_json), img_root=str(test_images), is_train=False, subset_name=subset
    )

    # shuffle=True para pegar imagens diferentes toda vez que rodar a célula
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    img_w = cfg["data"].get("image_width", 512.0)
    img_h = cfg["data"].get("image_height", 1024.0)

    # 1. Carrega todos os modelos para a VRAM de uma vez
    loaded_models = {}
    print("Carregando pesos dos modelos para visualização...")
    for name, path in model_paths.items():
        if Path(path).exists():
            model = SpinalMultiTaskNet().to(device)
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            loaded_models[name] = model

    print(f"Gerando {num_samples} amostras visuais...")

    # 2. Loop de Inferência e Plotagem
    count = 0
    for batch in test_loader:
        if count >= num_samples:
            break

        image_tensor = batch["image"].to(device)
        kpts_gt = batch["keypoints"][0].view(-1, 2).cpu().numpy()
        img_name = batch["img_name"][0] if "img_name" in batch else f"Sample {count+1}"

        img_display = denormalize_image(image_tensor[0])

        # Cria a figura: 1 linha, 5 colunas (GT + 4 modelos)
        fig, axes = plt.subplots(1, 5, figsize=(25, 8))
        fig.suptitle(f"Análise Visual - Imagem: {img_name}", fontsize=16, fontweight='bold', y=0.98)

        # Escala os pontos para as dimensões originais
        gt_x = kpts_gt[:, 0] * img_w
        gt_y = kpts_gt[:, 1] * img_h

        # ---------------------------------------------------------
        # Plot 1: Ground Truth (Verde)
        # ---------------------------------------------------------
        axes[0].imshow(img_display, cmap='gray')
        axes[0].scatter(gt_x, gt_y, c='lime', s=15, label="Ground Truth")
        axes[0].set_title("1. Ground Truth", fontsize=12)
        axes[0].axis('off')
        axes[0].legend(loc="lower right")

        # ---------------------------------------------------------
        # Plots 2 a 5: Predições dos Modelos (Vermelho sobre Verde)
        # ---------------------------------------------------------
        for idx, (name, model) in enumerate(loaded_models.items()):
            ax = axes[idx + 1]
            
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _, kpts_p = model(image_tensor)
            
            pred = kpts_p.view(-1, 2).cpu().numpy()
            pred_x = pred[:, 0] * img_w
            pred_y = pred[:, 1] * img_h

            ax.imshow(img_display, cmap='gray')
            # Desenha o GT no fundo
            ax.scatter(gt_x, gt_y, c='lime', s=15, alpha=0.4) 
            # Desenha os keypoints previstos por cima
            ax.scatter(pred_x, pred_y, c='red', s=15, marker='x', label=f"Modelo: {name}")
            
            ax.set_title(f"{idx + 2}. {name}", fontsize=12)
            ax.axis('off')
            ax.legend(loc="lower right")

        plt.tight_layout()
        plt.show()
        count += 1

    del loaded_models
    torch.cuda.empty_cache()

def plot_ablation_cobb_angles(
    model_paths: dict[str, str | Path],
    test_json: str | Path,
    test_images: str | Path,
    test_cobb_gt: str | Path = None,
    num_samples: int = 3,
    test_subset: str | None = None
):
    """
    Gera subplots lado a lado (1x5) focados puramente na métrica clínica.
    Desenha os keypoints e calcula o ângulo de Cobb
    """
    from src.utils.cobb import load_cobb_gt, get_detailed_cobb
    import matplotlib.lines as mlines
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = test_subset or cfg["test"]["subset_name"]
    
    # Carrega o GT oficial
    gt_map = load_cobb_gt(test_cobb_gt) if test_cobb_gt else {}
    
    test_dataset = get_dataset(
        json_path=str(test_json), img_root=str(test_images), is_train=False, subset_name=subset
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    loaded_models = {}
    print("Carregando pesos dos modelos para renderização do Cobb...")
    for name, path in model_paths.items():
        if Path(path).exists():
            model = SpinalMultiTaskNet().to(device)
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            loaded_models[name] = model

    def _draw_cobb_lines(ax, kpts_flat, res_dict, color_top="magenta", color_base="yellow"):
        """Função auxiliar para desenhar as retas de Cobb com a inclinação correta."""

        kpts_vis = kpts_flat.reshape(17, 4, 2) * np.array([512.0, 1024.0])
        v_sup = min(res_dict["v_superior"], res_dict["v_inferior"])
        v_inf = max(res_dict["v_superior"], res_dict["v_inferior"])
        
        for idx, color, label in [(v_sup, color_top, "Topo"), (v_inf, color_base, "Base")]:
            v_pts = kpts_vis[idx]
            if np.all(v_pts == 0): continue
            
            # Ordena os pontos visualmente para pegar sempre a face superior da vértebra
            v_pts_sorted_y = v_pts[v_pts[:, 1].argsort()]
            p1, p2 = v_pts_sorted_y[:2]
            if p1[0] > p2[0]: p1, p2 = p2, p1 # Garante p1 na esquerda
            
            # Calcula a reta
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6)
            x_vals = np.array([p1[0] - 130, p2[0] + 130])
            y_vals = np.array([p1[1] - slope * 130, p2[1] + slope * 130])
            
            ax.plot(x_vals, y_vals, color=color, linewidth=2.5, linestyle="--")
            cx, cy = np.mean(v_pts[:, 0]), np.mean(v_pts[:, 1])
            ax.text(cx + 25, cy, f"V{idx+1}", color="white", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor="none"))

    count = 0
    for batch in test_loader:
        if count >= num_samples:
            break

        image_tensor = batch["image"].to(device)
        kpts_gt_norm = batch["keypoints"][0].view(-1).cpu().numpy()
        img_name = batch["filename"][0] if "filename" in batch else f"Sample {count+1}"

        # Ajuste de cor
        img_display = denormalize_image(image_tensor[0])
        if img_display.shape[-1] == 3:
            img_display = img_display.mean(axis=-1)

        fig, axes = plt.subplots(1, 5, figsize=(26, 9))
        fig.suptitle(f"Análise Clínica de Ângulos de Cobb - Paciente: {img_name}", fontsize=18, fontweight='bold', y=0.98)

        escala_matematica = np.array([512.0, 512.0])
        
        # 1. Ground Truth
        gt_fisico = (kpts_gt_norm.reshape(-1, 2) * escala_matematica).flatten()
        res_gt = get_detailed_cobb(gt_fisico, 0, 17)
        cobb_json = res_gt["angle"]
        cobb_txt = gt_map.get(img_name, cobb_json) # Puxa do .txt se existir

        axes[0].imshow(img_display, cmap='gray')

        # Desenha a coluna e as retas do GT
        axes[0].scatter(kpts_gt_norm[0::2]*512, kpts_gt_norm[1::2]*1024, c='lime', s=10)
        _draw_cobb_lines(axes[0], kpts_gt_norm, res_gt, "cyan", "cyan")
        
        axes[0].set_title(f"1. Gabarito do Especialista\nLaudo .txt: {cobb_txt:.1f}° | JSON: {cobb_json:.1f}°", 
                          fontsize=12, fontweight="bold", color="darkgreen")
        axes[0].axis('off')

        # ---------------------------------------------------------
        # 2 a 5. Predições
        # ---------------------------------------------------------
        for idx, (name, model) in enumerate(loaded_models.items()):
            ax = axes[idx + 1]
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _, kpts_p = model(image_tensor)
            
            pred_norm = kpts_p.view(-1).cpu().numpy()
            pred_fisico = (pred_norm.reshape(-1, 2) * escala_matematica).flatten()
            res_ia = get_detailed_cobb(pred_fisico, 0, 17)
            
            cobb_ia = res_ia["angle"]
            erro_txt = abs(cobb_ia - cobb_txt)

            ax.imshow(img_display, cmap='gray')
            ax.scatter(pred_norm[0::2]*512, pred_norm[1::2]*1024, c='red', s=10)
            _draw_cobb_lines(ax, pred_norm, res_ia, "magenta", "yellow")
            
            # Título dinâmico mostrando o Ângulo e o Erro
            ax.set_title(f"{idx + 2}. {name}\nCobb: {cobb_ia:.1f}° | Erro (vs Txt): {erro_txt:.1f}°", 
                         fontsize=12, fontweight="bold", color="darkred" if erro_txt > 5.0 else "navy")
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.88) # Dá um espaço para o título principal não encostar
        plt.show()
        count += 1

    del loaded_models
    torch.cuda.empty_cache()

def plot_ablation_specific_sample_by_filename(
    model_paths: dict[str, str | Path],
    test_json: str | Path,
    test_images: str | Path,
    target_filename: str,
    test_cobb_gt: str | Path = None,
    test_subset: str | None = None
):
    """
    Gera um painel de 5 colunas procurando a imagem diretamente pelo nome do arquivo.
    - target_filename: Nome exato do arquivo (ex: '017797.jpg').
    """
    from src.utils.cobb import load_cobb_gt
    import matplotlib.lines as mlines
    import numpy as np
    import torch
    from pathlib import Path
    import matplotlib.pyplot as plt
    from src.config import get_config
    from src.data.dataset import get_dataset
    from src.models.network import SpinalMultiTaskNet
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = test_subset or cfg["test"]["subset_name"]
    
    # 1. Carrega o GT
    gt_map = load_cobb_gt(test_cobb_gt) if test_cobb_gt else {}
    
    # 2. Carrega o dataset
    test_dataset = get_dataset(
        json_path=str(test_json), img_root=str(test_images), is_train=False, subset_name=subset
    )
    
    # 3. Busca rápida do índice pelo nome do arquivo (sem abrir as imagens)
    sample_idx = None
    if hasattr(test_dataset, 'img_ids') and hasattr(test_dataset, 'coco'):
        for i, img_id in enumerate(test_dataset.img_ids):
            info = test_dataset.coco.loadImgs(img_id)[0]
            if info["file_name"] == target_filename:
                sample_idx = i
                break
    else:
        # Fallback caso a estrutura do dataset mude
        for i in range(len(test_dataset)):
            if test_dataset[i]["filename"] == target_filename:
                sample_idx = i
                break
                
    if sample_idx is None:
        raise ValueError(f"A imagem '{target_filename}' não foi encontrada no dataset de teste.")

    # 4. Carrega todos os modelos
    loaded_models = {}
    print(f"A processar a radiografia '{target_filename}' para o Painel Visual Minimalista...")
    for name, path in model_paths.items():
        if Path(path).exists():
            model = SpinalMultiTaskNet().to(device)
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            loaded_models[name] = model

    sample = test_dataset[sample_idx]
    
    # Prepara o tensor simulando o batch size 1 (unsqueeze)
    image_tensor = sample["image"].unsqueeze(0).to(device)
    kpts_gt_norm = sample["keypoints"].numpy().flatten()

    img_display = denormalize_image(sample["image"])
    
    # Garante que a imagem está em tons de cinza para o Matplotlib
    if img_display.shape[-1] == 3:
        img_display = img_display.mean(axis=-1)

    # Configura as 5 colunas
    fig, axes = plt.subplots(1, 5, figsize=(26, 9))
    
    axes[0].imshow(img_display, cmap='gray')
    
    # Desenha apenas os keypoints e contornos do Ground Truth
    draw_skeleton(axes[0], kpts_gt_norm, color_poly="cyan", color_pts="blue")
    
    # Título
    axes[0].set_title(
        "1. Ground Truth",
        fontsize=13, fontweight="bold", color="navy"
    )
    axes[0].axis("off")

    # ==========================================
    # COLUNAS 2 A 5: Modelos de Ablação
    # ==========================================
    for idx, (name, model) in enumerate(loaded_models.items()):
        ax = axes[idx + 1]
        
        with torch.no_grad(), torch.amp.autocast("cuda"):
            mask_p, kpts_p = model(image_tensor)
        
        # Pega as predições e normaliza [0,1]
        pred_norm = kpts_p.view(-1).cpu().numpy()

        ax.imshow(img_display, cmap='gray')
        
        # --- REGRA DA MÁSCARA: Exibe apenas se for HCL ---
        if "HCL" in name:
            m_p = mask_p.squeeze().cpu().numpy()
            ax.imshow(m_p, alpha=0.3, cmap="viridis")
        
        # Desenha o esqueleto
        draw_skeleton(ax, pred_norm, color_poly="lime", color_pts="red")
        
        # Título Minimalista
        ax.set_title(
            f"{idx + 2}. {name}",
            fontsize=13, fontweight="bold", color="darkred"
        )
        ax.axis('off')

        if idx == len(loaded_models) - 1:
            handles = [
                mlines.Line2D([], [], color="red", marker="o", linestyle="None",
                              markeredgecolor="white", markersize=7, label="Keypoints")
            ]
            
            ax.legend(handles=handles, loc="lower right", fontsize=10,
                      framealpha=0.95, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    plt.show()

    del loaded_models
    torch.cuda.empty_cache()