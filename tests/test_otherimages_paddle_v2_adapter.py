from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from transfer_receipt_ai import otherimages_paddle_v2_adapter as adapter_module
from transfer_receipt_ai.otherimages_paddle_teacher import (
    PADDLE_EFFECTIVE_ARG_KEYS,
    PADDLE_INPUT_COLOR_ORDER,
    _validate_adapter_evidence,
    canonical_paddle_color_contract,
)
from transfer_receipt_ai.otherimages_paddle_v2_adapter import PinnedPaddleOcrV2Adapter, _capture_with_engine


def _box(y0: float, y1: float) -> np.ndarray:
    return np.asarray([[10.0, y0], [80.0, y0], [80.0, y1], [10.0, y1]], dtype=np.float32)


class _FakeRawPaddleEngine:
    def __init__(self, recognitions: list[tuple[str, float]]) -> None:
        self.recognitions = recognitions
        self.calls: list[str] = []
        self.detector_pixel: tuple[int, int, int] | None = None
        self.args = types.SimpleNamespace(cls_thresh=0.9)

    def text_detector(self, image: np.ndarray) -> tuple[list[np.ndarray], float]:
        assert image.shape == (100, 100, 3)
        self.detector_pixel = tuple(int(value) for value in image[0, 0])
        self.calls.append("db")
        return [_box(70.0, 80.0), _box(10.0, 20.0)], 0.01

    def text_classifier(self, crops: list[np.ndarray]) -> tuple[list[np.ndarray], list[object], float]:
        self.calls.append("cls")
        return crops, [("0", 0.99)] * len(crops), 0.01

    def text_recognizer(self, crops: list[np.ndarray]) -> tuple[list[tuple[str, float]], float]:
        assert len(crops) == 2
        self.calls.append("rec")
        return self.recognitions, 0.01


def _cropper(_source: np.ndarray, box: np.ndarray) -> np.ndarray:
    return np.full((8, 16, 3), int(round(float(np.mean(box[:, 1])))), dtype=np.uint8)


def test_raw_adapter_calls_db_cls_rec_and_preserves_low_score_lines_before_filtering() -> None:
    engine = _FakeRawPaddleEngine([("顶部", 0.99), ("低置信底部", 0.20)])
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[0, 0] = [11, 22, 33]

    batch = _capture_with_engine(engine, pixels, cropper=_cropper, use_angle_cls=True)

    assert engine.calls == ["db", "cls", "rec"]
    assert engine.detector_pixel == (11, 22, 33)
    assert batch.raw_detected_line_count == 2
    assert batch.recognition_attempted_line_count == 2
    assert batch.recognition_rejected_line_count == 0
    assert [line.text for line in batch.lines] == ["顶部", "低置信底部"]
    assert [line.confidence for line in batch.lines] == [0.99, 0.20]
    assert [line.orientation_degrees for line in batch.lines] == [0, 0]
    assert batch.lines[0].quad_normalized[0] == (10.0 / 99.0, 10.0 / 99.0)
    assert canonical_paddle_color_contract()["input_color_order"] == PADDLE_INPUT_COLOR_ORDER


def test_raw_adapter_keeps_db_geometry_and_explicitly_counts_missing_recognition() -> None:
    engine = _FakeRawPaddleEngine([("顶部", 0.99)])
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)

    batch = _capture_with_engine(engine, pixels, cropper=_cropper, use_angle_cls=True)

    assert batch.raw_detected_line_count == 2
    assert batch.recognition_attempted_line_count == 2
    assert batch.recognition_rejected_line_count == 1
    assert len(batch.lines) == 2
    assert batch.lines[1].text == ""
    assert batch.lines[1].confidence == 0.0


def test_raw_adapter_records_only_the_cls_rotation_paddle_actually_applies() -> None:
    engine = _FakeRawPaddleEngine([("顶部", 0.99), ("底部", 0.98)])

    def classify(crops: list[np.ndarray]) -> tuple[list[np.ndarray], list[object], float]:
        engine.calls.append("cls")
        # Paddle rotates only when score is strictly greater than cls_thresh.
        return crops, [("180", 0.91), ("180", 0.90)], 0.01

    engine.text_classifier = classify  # type: ignore[method-assign]
    batch = _capture_with_engine(
        engine,
        np.zeros((100, 100, 3), dtype=np.uint8),
        cropper=_cropper,
        use_angle_cls=True,
    )

    assert [line.orientation_degrees for line in batch.lines] == [180, 0]


def test_raw_adapter_fails_closed_on_unbound_cls_results() -> None:
    engine = _FakeRawPaddleEngine([("顶部", 0.99), ("底部", 0.98)])

    def classify(crops: list[np.ndarray]) -> tuple[list[np.ndarray], list[object], float]:
        return crops, [("90", 0.99), ("0", 0.99)], 0.01

    engine.text_classifier = classify  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="label 0/180"):
        _capture_with_engine(
            engine,
            np.zeros((100, 100, 3), dtype=np.uint8),
            cropper=_cropper,
            use_angle_cls=True,
        )


def test_raw_adapter_rejects_recognition_results_without_a_db_box() -> None:
    engine = _FakeRawPaddleEngine(
        [("顶部", 0.99), ("底部", 0.98), ("无对应检测框", 0.97)]
    )
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="more results than raw DB boxes"):
        _capture_with_engine(engine, pixels, cropper=_cropper, use_angle_cls=True)


def test_importing_repository_adapter_does_not_import_paddle_runtime() -> None:
    code = (
        "import sys; "
        "import transfer_receipt_ai.otherimages_paddle_v2_adapter; "
        "print(int('paddle' in sys.modules or 'paddleocr' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


def test_pinned_factory_binds_actual_engine_args_and_model_files_with_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "models"
    paths: dict[str, str] = {}
    for role in ("det", "cls", "rec"):
        role_root = model_root / role
        role_root.mkdir(parents=True)
        (role_root / "inference.pdiparams").write_bytes(f"model-{role}".encode("ascii"))
        paths[f"{role}_model_dir"] = str(role_root)
    dictionary = model_root / "keys.txt"
    dictionary.write_text("甲\n乙\n", encoding="utf-8", newline="\n")
    paths["rec_char_dict_path"] = str(dictionary)
    effective_args: dict[str, object] = {
        "ocr_version": "PP-OCRv4",
        "det_algorithm": "DB",
        "det_limit_side_len": 960,
        "det_limit_type": "max",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "det_db_unclip_ratio": 1.5,
        "det_db_score_mode": "fast",
        "det_box_type": "quad",
        "rec_algorithm": "SVTR_LCNet",
        "rec_image_shape": "3, 48, 320",
        "rec_batch_num": 6,
        "max_text_length": 25,
        "use_space_char": True,
        "cls_image_shape": "3, 48, 192",
        "cls_batch_num": 6,
        "cls_thresh": 0.9,
        "use_angle_cls": True,
        "drop_score": 0.5,
        "use_onnx": False,
        "precision": "fp32",
        "use_tensorrt": False,
        "enable_mkldnn": False,
        "cpu_threads": 10,
        "use_gpu": False,
        "gpu_id": 0,
    }
    assert set(effective_args) == set(PADDLE_EFFECTIVE_ARG_KEYS)
    engines: list[object] = []

    class FakeReader:
        def __init__(self, **options: object) -> None:
            assert options == {
                "language": "ch",
                "use_angle_cls": True,
                "device": "auto",
                "require_v2": True,
            }
            loaded_det_sha256 = adapter_module._sha256(Path(paths["det_model_dir"]) / "inference.pdiparams")
            engine = types.SimpleNamespace(
                args=types.SimpleNamespace(**paths, **effective_args),
                loaded_det_sha256=loaded_det_sha256,
            )
            engines.append(engine)
            # Reproduce the old A-in-memory/B-on-disk window after the
            # bootstrap engine loaded.  The adapter must discard this engine,
            # bind B, and use a fresh engine that loaded B.
            if len(engines) == 1:
                (Path(paths["det_model_dir"]) / "inference.pdiparams").write_bytes(b"model-det-after-bootstrap")
            self._engine = engine

    fake_paddle = types.ModuleType("paddle")
    fake_paddle.__version__ = "3.0.0-fixture"
    fake_paddle.get_device = lambda: "cpu"
    versions = {
        "paddleocr": "2.10.0",
        "albumentations": "1.4.10",
        "albucore": "0.0.13",
        "Pillow": "10.4.0",
    }
    monkeypatch.setattr(adapter_module, "PaddleOCRReader", FakeReader)
    monkeypatch.setattr(adapter_module.metadata, "version", versions.__getitem__)
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)

    adapter = PinnedPaddleOcrV2Adapter(device="auto")
    evidence = _validate_adapter_evidence(adapter.evidence(), location="fixture")

    assert evidence["effective_paddle_args"] == effective_args
    assert evidence["execution_device"] == "cpu"
    assert evidence["model_assets"]["det"]["files"][0]["path"] == "inference.pdiparams"
    assert evidence["model_assets"]["dictionary"]["files"][0]["path"] == "keys.txt"
    assert len(engines) == 2
    assert adapter._engine is engines[1]
    assert engines[1].loaded_det_sha256 == evidence["model_assets"]["det"]["files"][0]["sha256"]
