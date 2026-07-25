from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from transfer_receipt_ai.ocr_lite_classifier import (
    ClassifierConfig,
    build_classifier,
    export_onnx,
    load_records,
    train_classifier,
)


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((24, 72, 3), 255, dtype=np.uint8)
    pixels[:, 6:20] = shade
    Image.fromarray(pixels).save(path)


def _record(index: int, class_name: str, split: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"sample-{index}",
        "image": f"images/{index}.png",
        "field": "transfer_status",
        "class_name": class_name,
        "split": split,
        "group_id": f"receipt-{index}",
    }


def test_tiny_classifier_training_writes_a_checkpoint(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    records = (
        _record(1, "success", "train"),
        _record(2, "failed", "train"),
        _record(3, "success", "val"),
        _record(4, "failed", "val"),
    )
    for index, record in enumerate(records):
        _write_image(dataset / str(record["image"]), 20 + index * 50)
    records_path = dataset / "status.jsonl"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    checkpoint = train_classifier(
        records_path=records_path,
        output_dir=tmp_path / "run",
        config=ClassifierConfig(image_height=32, image_width=64, base_channels=8, pooled_width=2),
        device="cpu",
        epochs=1,
        batch_size=2,
    )

    assert checkpoint.is_file()
    assert (checkpoint.parent / "last.pt").is_file()
    assert (checkpoint.parent / "labels.json").is_file()


def test_classifier_onnx_export_matches_torch_when_onnx_dependencies_are_installed(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    dataset = tmp_path / "dataset"
    records = (
        _record(1, "success", "train"),
        _record(2, "failed", "train"),
        _record(3, "success", "val"),
        _record(4, "failed", "val"),
    )
    for index, record in enumerate(records):
        _write_image(dataset / str(record["image"]), 20 + index * 50)
    records_path = dataset / "status.jsonl"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    config = ClassifierConfig(image_height=32, image_width=64, base_channels=8, pooled_width=2)
    checkpoint = train_classifier(
        records_path=records_path,
        output_dir=tmp_path / "run",
        config=config,
        device="cpu",
        epochs=1,
        batch_size=2,
    )
    model_path, labels_path, contract_path = export_onnx(checkpoint_path=checkpoint, output_path=tmp_path / "status.onnx")

    onnx.checker.check_model(onnx.load_model(model_path))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_classifier(class_count=len(payload["classes"]), config=config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    sample = torch.rand((1, 1, config.image_height, config.image_width), dtype=torch.float32)
    expected = model(sample).detach().cpu().numpy()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"image": sample.numpy()})[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert labels_path.is_file()
    assert contract_path.is_file()


def test_classifier_loader_rejects_multiple_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    first = _record(1, "success", "train")
    second = {**_record(2, "bank_card", "val"), "field": "payment_method_field"}
    for record in (first, second):
        _write_image(dataset / str(record["image"]), 20)
    records_path = dataset / "bad.jsonl"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in (first, second)), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one field"):
        load_records(records_path)
