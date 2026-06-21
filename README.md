# Spinal Cobb AI

Detecção automática de vértebras em radiografias da coluna e estimativa do ângulo de Cobb via Deep Learning.

Trabalho de Conclusão de Curso (TCC).

---

## Visão geral

O modelo recebe uma radiografia da coluna vertebral e produz:

- **Máscara de segmentação** — delimita a massa óssea de cada vértebra
- **Keypoints** — 4 pontos de canto para cada uma das 17 vértebras
- **Ângulo de Cobb** — calculado geometricamente a partir dos keypoints (regiões PT, MT e TL/L)

**Arquitetura:** HRNet-W32 (backbone) com duas cabeças independentes (segmentação + regressão).  
**Treinamento:** K-Fold cross-validation (5 folds) com Early Stopping.  
**Loss:** Híbrida — Dice Loss + Smooth L1 + Regularização anatômica via PCA.

---

## Estrutura do projeto

```
spinal_cobb_ai/
├── configs/
│   └── config.yaml          # Todos os hiperparâmetros e caminhos
├── data/                    # Dataset (não versionado — ver .gitignore)
│   ├── Spinal-AI2024_train_annotation.json
│   ├── Spinal-AI2024_Cobb_train_gt.txt
│   └── Spinal-AI2024-dataset/
├── scripts/
│   ├── train.py             # Ponto de entrada do treinamento
│   └── evaluate.py          # Geração de laudos e métricas
└── src/
    ├── config.py            # Carregador do config.yaml
    ├── data/
    │   └── dataset.py       # SpinalDataset + transforms
    ├── models/
    │   ├── network.py       # SpinalMultiTaskNet (HRNet-W32)
    │   └── pca.py           # Treinamento/carregamento do PCA anatômico
    ├── training/
    │   ├── losses.py        # DiceLoss + HybridLoss
    │   ├── trainer.py       # run_epoch + run_kfold_training + EarlyStopping
    │   └── checkpoint.py    # CheckpointManager (local + Drive opcional)
    ├── evaluation/
    │   ├── visualizer.py    # Painéis comparativos GT vs modelos treinados
    │   └── analysis.py      # Análise do dataset e sync check
    └── utils/
        ├── cobb.py          # Cálculo do ângulo de Cobb
        └── drive.py         # Sync com Google Drive (opcional)
```

---

## Instalação

```bash
git clone https://github.com/dioneadam/cobb-angle.git
cd cobb-angle
pip install -r requirements.txt
```

---

## Dados

Coloque os arquivos do dataset em `data/`:

```
data/
├── Spinal-AI2024_train_annotation.json
├── Spinal-AI2024_Cobb_train_gt.txt
└── Spinal-AI2024-dataset/
    └── Spinal-AI2024-subset1/
        ├── 000001.jpg
        └── ...
```

Atualize os caminhos em `configs/config.yaml` se necessário.

---

## Uso

### Treinamento

```bash
python scripts/train.py
```


### Avaliação

```bash
# Gera 15 painéis visuais + relatório estatístico
python scripts/evaluate.py

# Mais amostras, salvando os plots em disco
python scripts/evaluate.py --samples 30 --val-samples 100 --save-plots
```

---

## Configuração

Todos os hiperparâmetros ficam em `configs/config.yaml`:

```yaml
training:
  epochs: 150
  batch_size: 8
  learning_rate: 0.0001
  k_folds: 5
  early_stopping_patience: 15

loss:
  w_reg: 1.0     # Peso do Smooth L1 (keypoints)
  w_seg: 0.65    # Peso da Dice Loss (segmentação)
  w_pca: 0.02    # Peso da regularização anatômica PCA
```

Para ativar o sync com Google Drive, edite a seção `drive:` no config.

---

## Ângulo de Cobb

O cálculo segue o método clínico padrão: encontra as vértebras de maior e menor inclinação dentro do segmento e computa a diferença angular entre suas faces superiores. Três regiões são avaliadas:

| Região
|--------|
| PT (Proximal Torácica) 
| MT (Torácica Principal)
| TL/L (Toracolombar/Lombar)

---

## Dataset

[SpinalAI 2024](https://github.com/ernestchenchen/spinal-ai2024)
