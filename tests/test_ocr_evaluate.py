from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

torch = pytest.importorskip("torch")

from transfer_receipt_ai.ocr_evaluate import (
    _acceptance_failures,
    _create_session,
    evaluate_onnx,
    levenshtein_distance,
    semantic_value,
)
from transfer_receipt_ai.ocr_train import RecognizerConfig, export_onnx, train_recognizer
from transfer_receipt_ai.ocr_train import GENERIC_TEXT_LINE_FIELD


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        ("收款方 张三", "收款方 李三", 1),
        ("¥12.30", "¥12.30", 0),
        ("", "转账成功", 4),
    ],
)
def test_levenshtein_distance_counts_unicode_characters(reference: str, candidate: str, expected: int) -> None:
    assert levenshtein_distance(reference, candidate) == expected


def test_semantic_value_reuses_current_recipient_extraction() -> None:
    assert semantic_value("recipient_field", "收款方 张三(**)") == "张三(**)"
    assert semantic_value("amount", "¥12.30") == "¥12.30"
    assert semantic_value(GENERIC_TEXT_LINE_FIELD, "任意整行文本") is None


def test_generic_line_acceptance_rejects_semantic_gate_as_not_applicable() -> None:
    metrics = {
        GENERIC_TEXT_LINE_FIELD: {
            "raw_exact_match": 1.0,
            "semantic_exact_match": None,
            "micro_cer": 0.0,
            "oov_reference_rate": 0.0,
        }
    }
    assert _acceptance_failures(
        metrics,
        min_raw_exact_match=0.99,
        min_semantic_exact_match=0.99,
        max_micro_cer=0.01,
        max_oov_reference_rate=0.0,
    ) == [f"{GENERIC_TEXT_LINE_FIELD}: semantic_exact_match is not applicable"]


def test_cuda_session_preloads_dlls_before_constructing_onnx_session(tmp_path: Path) -> None:
    class FakeSession:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class FakeOrt:
        preloaded = False

        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @classmethod
        def preload_dlls(cls):
            cls.preloaded = True

        @classmethod
        def InferenceSession(cls, _path, *, providers):
            assert cls.preloaded
            assert providers[0][0] == "CUDAExecutionProvider"
            return FakeSession()

    session, active = _create_session(FakeOrt, tmp_path / "model.onnx", device="cuda:0")
    assert isinstance(session, FakeSession)
    assert active[0] == "CUDAExecutionProvider"


def test_acceptance_can_require_zero_oov_reference_characters() -> None:
    metrics = {
        "recipient_field": {
            "raw_exact_match": 1.0,
            "semantic_exact_match": 1.0,
            "micro_cer": 0.0,
            "oov_reference_rate": 0.25,
        }
    }
    assert _acceptance_failures(
        metrics,
        min_raw_exact_match=0.99,
        min_semantic_exact_match=0.99,
        max_micro_cer=0.005,
        max_oov_reference_rate=0.0,
    ) == ["recipient_field: oov_reference_rate=0.2500 > 0.0000"]


def test_validation_split_is_never_a_formal_acceptance_gate() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "transfer_receipt_ai" / "ocr_evaluate.py").read_text(
        encoding="utf-8"
    )

    assert 'diagnostic_only = split != "test"' in source
    assert '"passed": not failures and not diagnostic_only' in source
    assert "formal acceptance requires the frozen test split" in source


def test_onnx_evaluation_runs_on_a_group_held_out_split_when_dependencies_are_installed(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        image_name = f"images/{index}.png"
        image = np.full((20, 50, 3), 255, dtype=np.uint8)
        image[:, 8 + index : 16 + index] = 20
        Image.fromarray(image).save(dataset / image_name)
        rows.append(
            {
                "schema_version": 1,
                "id": image_name,
                "image": image_name,
                "field": "amount",
                "text": "1",
                "split": split,
                "group_id": f"receipt-{split}",
                "crop_sha256": f"crop-{split}",
                "source": f"source-{split}.png",
            }
        )
    records = dataset / "pseudo_labels.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = RecognizerConfig(image_height=32, image_width=64, hidden_size=16, lstm_layers=1)
    checkpoint = train_recognizer(
        records_path=records,
        output_dir=tmp_path / "run",
        fields=("amount",),
        config=config,
        device="cpu",
        epochs=1,
        batch_size=1,
    )
    onnx_path, _, _ = export_onnx(checkpoint_path=checkpoint, output_path=tmp_path / "receipt_ocr_ctc.onnx")

    summary, failures = evaluate_onnx(
        model_path=onnx_path,
        records_path=records,
        output_dir=tmp_path / "evaluation",
        split="test",
        fields=("amount",),
        device="cpu",
    )

    assert failures == []
    assert summary["overall"]["records"] == 1
    assert (tmp_path / "evaluation" / "comparisons.jsonl").is_file()
