# Guia de Rastreamento — MLflow Local no Google Drive (Colab)

Este projeto usa **MLflow em modo *file store* (`file://`)** gravado diretamente no Google Drive durante os treinos no Google Colab, com fallback automático para `./mlruns` em desenvolvimento local.

## Estrutura de Pastas no Google Drive

```text
/content/drive/MyDrive/painel-confinamento/
├── dados/          # FBSI + derivado compactado do CVB (5 fps, 640 px)
├── checkpoints/    # Pesos .pt/.pth (fora do MLflow para economizar cota e tempo)
└── mlflow/
    └── mlruns/     # File store do MLflow (métricas, parâmetros, tags e figuras leves)
```

> **Por que *file store* (`file://`) em vez de SQLite?**
> O sistema de arquivos FUSE do Google Drive no Colab pode travar ou corromper bancos SQLite (`database is locked`). O *file store* grava arquivos simples por métrica/parâmetro e persiste os resultados incrementalmente caso a sessão do Colab caia.

## Regras de Uso no Projeto

1. **Logar métricas por época (não por batch)** para evitar lentidão de I/O no Drive.
2. **Não salvar pesos `.pt`/`.pth` como artefato do MLflow** — salve os pesos em `checkpoints/` no Drive e registre apenas o caminho ou nome do arquivo nos parâmetros/tags.
3. **Respeitar `RNF02`:** usar `split="val"` durante todo o tuning e reservar `split="test"` para a avaliação final única.

## Uso no Google Colab

### 1. Montar o Drive e clonar/atualizar o repositório

```python
from google.colab import drive

drive.mount("/content/drive")
```

### 2. Treinos PyTorch / `timm` (Módulo A e modelos temporais do Módulo B)

```python
import mlflow

from train.seed import set_seed
from train.tracking import EXPERIMENT_MODULO_A, start_experiment_run

set_seed(42)

with start_experiment_run(
    EXPERIMENT_MODULO_A,
    run_name="resnet50-ordinal",
    seed=42,
    split="val",
    params={"backbone": "resnet50", "lr": 1e-4, "epochs": 30},
):
    for epoch in range(1, 31):
        # ... loop de treino e validação ...
        mlflow.log_metrics(
            {"val_f1_macro": val_f1, "val_mae_ordinal": val_mae},
            step=epoch,
        )
```

### 3. Fine-tuning de YOLO com `ultralytics` (Módulo B)

```python
from ultralytics import YOLO

from train.tracking import EXPERIMENT_MODULO_B, configure_ultralytics_mlflow

configure_ultralytics_mlflow(experiment_name=EXPERIMENT_MODULO_B)

model = YOLO("yolo11n.pt")
model.train(data="cvb.yaml", epochs=50, imgsz=640)
```

## Gerar Tabelas Comparativas para o Relatório

Não é necessário subir a interface web do MLflow para analisar os resultados. Você pode consultar as runs diretamente como `DataFrame`:

```python
from train.tracking import EXPERIMENT_MODULO_A, load_runs_dataframe

df = load_runs_dataframe(EXPERIMENT_MODULO_A)
print(df[["tags.mlflow.runName", "tags.split", "metrics.val_f1_macro", "metrics.val_mae_ordinal"]])
```

Se quiser abrir a UI visual do MLflow localmente na sua máquina, baixe a pasta `mlruns/` do Google Drive e execute:

```bash
mlflow ui --backend-store-uri file:./mlruns
```
