"""Testes unitários do módulo de rastreamento MLflow (train/tracking.py)."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from train import tracking


class TestTrackingUriResolution(unittest.TestCase):
    """Testa a prioridade de resolução da URI de tracking."""

    def test_env_var_has_highest_priority(self) -> None:
        custom_uri = "file:///tmp/custom_mlruns"
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": custom_uri}, clear=False):
            uri = tracking.resolve_tracking_uri()
        self.assertEqual(uri, custom_uri)

    def test_uses_google_drive_when_mounted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_drive = Path(tmp) / "MyDrive"
            fake_drive.mkdir()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MLFLOW_TRACKING_URI", None)
                uri = tracking.resolve_tracking_uri(
                    drive_mount=fake_drive,
                    project_dirname="painel-confinamento",
                )

            expected = (fake_drive / "painel-confinamento" / "mlflow" / "mlruns").resolve().as_uri()
            self.assertEqual(uri, expected)

    def test_falls_back_to_local_dir_when_drive_not_mounted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_drive = Path(tmp) / "nao_existe"
            local_dir = Path(tmp) / "mlruns_local"
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MLFLOW_TRACKING_URI", None)
                uri = tracking.resolve_tracking_uri(
                    drive_mount=missing_drive,
                    local_dir=local_dir,
                )

            self.assertEqual(uri, local_dir.resolve().as_uri())


class TestMlflowConfiguration(unittest.TestCase):
    """Testa a configuração de MLflow, Ultralytics e o context manager de runs."""

    def test_configure_ultralytics_sets_env_and_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_drive = Path(tmp) / "MyDrive"
            fake_drive.mkdir()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MLFLOW_TRACKING_URI", None)
                os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
                uri = tracking.configure_ultralytics_mlflow(
                    experiment_name=tracking.EXPERIMENT_MODULO_B,
                    drive_mount=fake_drive,
                )

                self.assertEqual(os.environ["MLFLOW_TRACKING_URI"], uri)
                self.assertEqual(
                    os.environ["MLFLOW_EXPERIMENT_NAME"],
                    tracking.EXPERIMENT_MODULO_B,
                )
                expected_dir = fake_drive / "painel-confinamento" / "mlflow" / "mlruns"
                self.assertTrue(expected_dir.is_dir())

    def test_start_experiment_run_rejects_invalid_split(self) -> None:
        with self.assertRaises(ValueError):
            with tracking.start_experiment_run(
                tracking.EXPERIMENT_MODULO_A,
                split="train",
            ):
                pass

    def test_start_experiment_run_logs_default_tags_and_params(self) -> None:
        fake_mlflow = MagicMock()
        fake_run = MagicMock(name="run")

        @contextmanager
        def _fake_start_run(run_name: str | None = None):
            yield fake_run

        fake_mlflow.start_run.side_effect = _fake_start_run

        with tempfile.TemporaryDirectory() as tmp:
            fake_drive = Path(tmp) / "MyDrive"
            fake_drive.mkdir()

            with (
                patch.dict(os.environ, {}, clear=False),
                patch.object(tracking, "_get_mlflow", return_value=fake_mlflow),
                patch.object(tracking, "get_git_commit", return_value="abc1234"),
            ):
                os.environ.pop("MLFLOW_TRACKING_URI", None)
                with tracking.start_experiment_run(
                    tracking.EXPERIMENT_MODULO_A,
                    run_name="resnet50-ordinal",
                    seed=42,
                    split="val",
                    params={"model": "resnet50", "lr": 1e-4},
                    tags={"loss": "ordinal"},
                    drive_mount=fake_drive,
                ) as run:
                    self.assertIs(run, fake_run)

            fake_mlflow.set_tracking_uri.assert_called_once()
            fake_mlflow.set_experiment.assert_called_once_with(tracking.EXPERIMENT_MODULO_A)
            fake_mlflow.start_run.assert_called_once_with(run_name="resnet50-ordinal")
            fake_mlflow.set_tags.assert_called_once_with(
                {
                    "seed": "42",
                    "split": "val",
                    "git_commit": "abc1234",
                    "environment": "colab",
                    "loss": "ordinal",
                }
            )
            fake_mlflow.log_params.assert_called_once_with({"model": "resnet50", "lr": 1e-4})


if __name__ == "__main__":
    unittest.main()
