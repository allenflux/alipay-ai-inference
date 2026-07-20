from pathlib import Path

import pytest

from receipt_inference import cli, models


def test_default_model_uses_project_relative_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(models, "PROJECT_ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoints" / "receipt_lrcnn_v1" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert models.resolve_checkpoint("receipt_lrcnn_v1") == checkpoint.resolve()


def test_missing_checkpoint_explains_exact_copy_destination(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(models, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match=r"checkpoints.receipt_lrcnn_v1.best.pt"):
        models.resolve_checkpoint("receipt_lrcnn_v1")


def test_cli_runs_only_receipt_model(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    captured = {}

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return [{"status": "written"}]

    monkeypatch.setattr(cli, "run_inference", fake_run_inference)
    cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(tmp_path / "receipt.png"),
            "--output",
            str(tmp_path / "results"),
            "--ocr",
            "none",
            "--require-complete",
        ]
    )

    assert captured["checkpoint"] == checkpoint.resolve()
    assert captured["status_style_checkpoint"] is None
    assert captured["use_ocr"] is False
    assert captured["require_complete"] is True

