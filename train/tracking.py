"""Configuração de rastreamento de experimentos com MLflow local no Google Drive (Colab)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

DEFAULT_DRIVE_MOUNT = Path("/content/drive/MyDrive")
DEFAULT_PROJECT_DIRNAME = "painel-confinamento"
MLFLOW_RELATIVE_DIR = Path("mlflow") / "mlruns"
DEFAULT_LOCAL_MLRUNS = Path("mlruns")

EXPERIMENT_MODULO_A = "modulo-a-cocho"
EXPERIMENT_MODULO_B = "modulo-b-curral"
VALID_SPLITS = frozenset({"val", "test"})


def _get_mlflow() -> Any:
    """Importa mlflow sob demanda para não acoplar ambientes sem dependências de treino."""
    import mlflow

    return mlflow


def _ensure_local_file_uri_dir(tracking_uri: str) -> None:
    """Garante que o diretório exista quando a tracking URI usa o esquema file://."""
    parsed = urlparse(tracking_uri)
    if parsed.scheme != "file":
        return

    raw_path = unquote(parsed.path)
    # No Windows, urlparse("file:///C:/foo").path retorna "/C:/foo".
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]

    Path(raw_path).mkdir(parents=True, exist_ok=True)


def is_colab_drive_mounted(drive_mount: Path = DEFAULT_DRIVE_MOUNT) -> bool:
    """Retorna True se o diretório raiz do Google Drive estiver montado."""
    return drive_mount.exists()


def resolve_tracking_uri(
    drive_mount: Path = DEFAULT_DRIVE_MOUNT,
    project_dirname: str = DEFAULT_PROJECT_DIRNAME,
    local_dir: Path = DEFAULT_LOCAL_MLRUNS,
) -> str:
    """Resolve a tracking URI do MLflow com prioridade: env var > Google Drive > local.

    Usa file store (`file://`) em vez de SQLite para evitar locks e corrupção no
    sistema de arquivos FUSE do Google Drive no Colab.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    if is_colab_drive_mounted(drive_mount):
        target_dir = drive_mount / project_dirname / MLFLOW_RELATIVE_DIR
    else:
        target_dir = local_dir

    return target_dir.resolve().as_uri()


def get_git_commit() -> str:
    """Retorna o hash curto do commit atual ou 'unknown' fora de um repositório Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        return commit or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def configure_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
    *,
    drive_mount: Path = DEFAULT_DRIVE_MOUNT,
    project_dirname: str = DEFAULT_PROJECT_DIRNAME,
    local_dir: Path = DEFAULT_LOCAL_MLRUNS,
) -> str:
    """Configura o MLflow para gravar no Google Drive (ou fallback local) e define o experimento."""
    uri = tracking_uri or resolve_tracking_uri(
        drive_mount=drive_mount,
        project_dirname=project_dirname,
        local_dir=local_dir,
    )
    _ensure_local_file_uri_dir(uri)

    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def configure_ultralytics_mlflow(
    experiment_name: str = EXPERIMENT_MODULO_B,
    tracking_uri: str | None = None,
    *,
    drive_mount: Path = DEFAULT_DRIVE_MOUNT,
    project_dirname: str = DEFAULT_PROJECT_DIRNAME,
    local_dir: Path = DEFAULT_LOCAL_MLRUNS,
) -> str:
    """Configura variáveis de ambiente para o callback nativo de MLflow do Ultralytics."""
    uri = tracking_uri or resolve_tracking_uri(
        drive_mount=drive_mount,
        project_dirname=project_dirname,
        local_dir=local_dir,
    )
    _ensure_local_file_uri_dir(uri)

    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    return uri


@contextmanager
def start_experiment_run(
    experiment_name: str,
    run_name: str | None = None,
    *,
    seed: int = 42,
    split: str = "val",
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
    drive_mount: Path = DEFAULT_DRIVE_MOUNT,
    project_dirname: str = DEFAULT_PROJECT_DIRNAME,
    local_dir: Path = DEFAULT_LOCAL_MLRUNS,
) -> Iterator[Any]:
    """Abre uma run do MLflow com tags de reprodutibilidade (RNF01/RNF02) e parâmetros.

    Deve-se logar métricas por época (não por batch) para evitar latência de I/O no Drive
    e não salvar pesos `.pt`/`.pth` como artefato do MLflow (manter em `checkpoints/`).
    """
    if split not in VALID_SPLITS:
        valid = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"split inválido: {split!r}. Valores aceitos: {valid}.")

    configure_mlflow(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        drive_mount=drive_mount,
        project_dirname=project_dirname,
        local_dir=local_dir,
    )

    mlflow = _get_mlflow()
    default_tags = {
        "seed": str(seed),
        "split": split,
        "git_commit": get_git_commit(),
        "environment": "colab" if is_colab_drive_mounted(drive_mount) else "local",
    }
    if tags:
        default_tags.update(tags)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(default_tags)
        if params:
            mlflow.log_params(params)
        yield run


def load_runs_dataframe(
    experiment_name: str,
    tracking_uri: str | None = None,
    *,
    drive_mount: Path = DEFAULT_DRIVE_MOUNT,
    project_dirname: str = DEFAULT_PROJECT_DIRNAME,
    local_dir: Path = DEFAULT_LOCAL_MLRUNS,
) -> Any:
    """Carrega todas as runs de um experimento como DataFrame pandas para tabelas comparativas."""
    configure_mlflow(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        drive_mount=drive_mount,
        project_dirname=project_dirname,
        local_dir=local_dir,
    )
    mlflow = _get_mlflow()
    return mlflow.search_runs(experiment_names=[experiment_name])
