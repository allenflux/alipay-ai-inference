import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from receipt_inference import cli, models
from transfer_receipt_ai.geometry import RectificationOptions, RectificationResult
from transfer_receipt_ai.infer import _write_ocr_stage_artifacts
from transfer_receipt_ai.model import Detection
from transfer_receipt_ai.ocr import OCRResult
from transfer_receipt_ai.ocr_enrich import run_ocr_enrichment
from transfer_receipt_ai.pipeline import ExtractedDetection, ReceiptPipeline, ReceiptResult, write_receipt_result


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

    import transfer_receipt_ai.infer as inference

    monkeypatch.setattr(inference, "run_inference", fake_run_inference)
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


def test_paddle_cli_uses_two_sequential_workers_in_the_same_venv(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    commands: list[list[str]] = []

    def fake_run_child(command: list[str]) -> None:
        commands.append(command)

    monkeypatch.setattr(cli, "_run_child", fake_run_child)
    monkeypatch.setattr(cli, "_read_final_manifest", lambda output: [{"status": "written"}])
    monkeypatch.setattr(cli, "_needs_windows_gpu_split", lambda args: True)
    cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(tmp_path / "receipt.png"),
            "--output",
            str(tmp_path / "results"),
            "--device",
            "cuda",
            "--ocr",
            "paddle",
            "--require-complete",
            "--skip-existing",
        ]
    )

    assert len(commands) == 2
    detector, ocr = commands
    assert detector[:3] == [cli.sys.executable, "-m", "transfer_receipt_ai.infer"]
    assert detector[detector.index("--ocr") + 1] == "none"
    assert "--_write-ocr-stage-artifacts" in detector
    assert detector[detector.index("--device") + 1] == "cuda"
    assert "--skip-existing" in detector
    assert detector[detector.index("--_skip-existing-output") + 1] == str(tmp_path / "results")
    assert ocr[:3] == [cli.sys.executable, "-m", "transfer_receipt_ai.ocr_enrich"]
    assert ocr[ocr.index("--device") + 1] == "cuda"
    assert "--skip-existing" in ocr


def test_ocr_enrichment_writes_the_existing_result_bundle_shape(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    source_rgb = np.full((40, 60, 3), 255, dtype=np.uint8)
    Image.fromarray(source_rgb).save(source_path)
    stage_output = tmp_path / "stage"
    stage_output.mkdir()
    stage_stem = stage_output / "source"
    detection = Detection("amount", 0.9, (5.25, 5.25, 35.75, 18.75))
    rectification = RectificationResult(
        source_rgb=source_rgb,
        rectified_rgb=source_rgb.copy(),
        original_to_rectified=np.eye(3),
        rectified_to_original=np.eye(3),
        screen_quad_original=np.array([[0, 0], [59, 0], [59, 39], [0, 39]], dtype=np.float32),
        rotation_degrees=0,
        screen_detected=False,
    )
    result = ReceiptResult(
        source_path=source_path.resolve().as_posix(),
        rectification=rectification,
        detections=[
            ExtractedDetection(
                detection,
                None,
                np.array([[5.25, 5.25], [35.75, 5.25], [35.75, 18.75], [5.25, 18.75]], dtype=np.float32),
            )
        ],
        fields={},
    )
    _write_ocr_stage_artifacts(result, stage_stem)
    stage_stem.with_suffix(".json").write_text("{}", encoding="utf-8")
    (stage_output / "inference_manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": source_path.resolve().as_posix(),
                    "result": stage_stem.with_suffix(".json").resolve().as_posix(),
                    "status": "written",
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeReader:
        def recognize(self, image_rgb):
            assert image_rgb.size
            return OCRResult("¥12.30", 0.99)

    output_dir = tmp_path / "output"
    records = run_ocr_enrichment(stage_output_dir=stage_output, output_dir=output_dir, reader=FakeReader())

    assert records[0]["status"] == "written"
    payload = json.loads((output_dir / "source.json").read_text(encoding="utf-8"))
    assert payload["fields"]["amount"]["normalized"] == "¥12.30"
    assert payload["detections"][0]["ocr"] == {"text": "¥12.30", "confidence": 0.99}
    assert (output_dir / "source_rectified_annotated.jpg").is_file()
    assert (output_dir / "source_original_annotated.jpg").is_file()


def test_ocr_enrichment_carries_detector_skip_records_without_stage_artifacts(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(source_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_stem = output_dir / "source"
    record = {
        "source": source_path.resolve().as_posix(),
        "result": output_stem.with_suffix(".json").resolve().as_posix(),
        "annotated_rectified": output_stem.with_name("source_rectified_annotated.jpg").resolve().as_posix(),
        "annotated_original": output_stem.with_name("source_original_annotated.jpg").resolve().as_posix(),
        "status": "skipped_existing",
    }
    stage_output = tmp_path / "stage"
    stage_output.mkdir()
    (stage_output / "inference_manifest.json").write_text(json.dumps([record]), encoding="utf-8")

    class FailingReader:
        def recognize(self, image_rgb):
            raise AssertionError("OCR must not run for a detector skip record")

    records = run_ocr_enrichment(stage_output_dir=stage_output, output_dir=output_dir, reader=FailingReader())

    assert records == [record]
    assert json.loads((output_dir / "inference_manifest.json").read_text(encoding="utf-8")) == [record]


def test_staged_ocr_matches_the_original_result_json_and_annotations(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    source_rgb = np.full((40, 60, 3), 245, dtype=np.uint8)
    Image.fromarray(source_rgb).save(source_path)

    class FakePredictor:
        def predict(self, image_rgb):
            return [Detection("amount", 0.9, (5.25, 5.25, 35.75, 18.75))]

    class FakeReader:
        def recognize(self, image_rgb):
            return OCRResult("¥12.30", 0.99)

    options = RectificationOptions(orientation_degrees=0, auto_screen=False, max_side=0)
    direct_result = ReceiptPipeline(FakePredictor(), ocr=FakeReader(), rectification_options=options).run(source_path)
    direct_stem = tmp_path / "direct" / "source"
    write_receipt_result(direct_result, direct_stem)

    staged_detector_result = ReceiptPipeline(FakePredictor(), ocr=None, rectification_options=options).run(source_path)
    stage_output = tmp_path / "stage"
    stage_output.mkdir()
    stage_stem = stage_output / "source"
    _write_ocr_stage_artifacts(staged_detector_result, stage_stem)
    (stage_output / "inference_manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": source_path.resolve().as_posix(),
                    "result": stage_stem.with_suffix(".json").resolve().as_posix(),
                    "status": "written",
                }
            ]
        ),
        encoding="utf-8",
    )
    staged_output = tmp_path / "staged"
    run_ocr_enrichment(stage_output_dir=stage_output, output_dir=staged_output, reader=FakeReader())

    staged_stem = staged_output / "source"
    assert direct_stem.with_suffix(".json").read_text(encoding="utf-8") == staged_stem.with_suffix(
        ".json"
    ).read_text(encoding="utf-8")
    assert direct_stem.with_name("source_rectified_annotated.jpg").read_bytes() == staged_stem.with_name(
        "source_rectified_annotated.jpg"
    ).read_bytes()
    assert direct_stem.with_name("source_original_annotated.jpg").read_bytes() == staged_stem.with_name(
        "source_original_annotated.jpg"
    ).read_bytes()
