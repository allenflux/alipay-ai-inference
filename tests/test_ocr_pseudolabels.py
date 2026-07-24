from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.ocr_pseudolabels import (
    UnsafeOcrDatasetOutputError,
    build_pseudo_label_dataset,
)
from transfer_receipt_ai.ocr_train import load_records


def _write_result(
    *,
    results_dir: Path,
    source: Path,
    detections: list[dict[str, object]],
    name: str = "receipt.json",
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    source_rgb = np.asarray(Image.open(source).convert("RGB"))
    height, width = source_rgb.shape[:2]
    payload = {
        "source": source.resolve().as_posix(),
        "geometry": {
            "source_size": {"width": width, "height": height},
            "rectified_size": {"width": width, "height": height},
            "H_original_to_rectified": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "detections": detections,
        "fields": {},
    }
    path = results_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _detection(
    label: str,
    text: str,
    *,
    confidence: float = 0.99,
    score: float = 0.95,
    x_offset: float = 0.0,
) -> dict[str, object]:
    return {
        "label": label,
        "score": score,
        "bbox_rectified": [4.25 + x_offset, 5.25, 24.75 + x_offset, 24.75],
        "ocr": {"text": text, "confidence": confidence},
    }


def test_build_pseudo_labels_exports_clean_crops_records_and_group_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "receipt_20260724093000.png"
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(60, dtype=np.uint8)
    rgb[:, :, 1] = 120
    Image.fromarray(rgb).save(source)
    results = tmp_path / "results"
    _write_result(
        results_dir=results,
        source=source,
        detections=[
            _detection("amount", "¥12.30", x_offset=0),
            _detection("time", "09:30", x_offset=5),
            _detection("transfer_status", "转账成功", x_offset=10),
            _detection("recipient_field", "收款方 张三", x_offset=15),
            _detection("payment_method_field", "付款方式 余额", x_offset=20),
        ],
    )
    # Result-root metadata must not accidentally become a training sample.
    (results / "inference_manifest.json").write_text("[]", encoding="utf-8")

    output = tmp_path / "pseudo"
    records = build_pseudo_label_dataset(
        results_dir=results,
        output_dir=output,
        validation_ratio=0.5,
        review_ratio=1.0,
    )

    assert len(records) == 5
    assert {record["field"] for record in records} == {
        "amount",
        "time",
        "transfer_status",
        "recipient_field",
        "payment_method_field",
    }
    assert len({record["group_id"] for record in records}) == 1
    assert len({record["split"] for record in records}) == 1
    assert all((output / str(record["image"])).is_file() for record in records)
    assert len((output / "review_candidates.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    assert (output / "charset.txt").read_text(encoding="utf-8").strip()
    coverage = json.loads((output / "character_coverage.json").read_text(encoding="utf-8"))
    assert coverage["recipient_field"][records[0]["split"]]["records"] == 1
    assert coverage["recipient_field"][records[0]["split"]]["characters"]["张"] == 1
    loaded = load_records(output / "pseudo_labels.jsonl")
    assert [record["text"] for record in loaded] == [record["text"] for record in records]


def test_low_confidence_and_invalid_semantics_are_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "receipt.png"
    image = np.full((40, 60, 3), 200, dtype=np.uint8)
    image[8:24, 8:45] = [20, 40, 60]
    Image.fromarray(image).save(source)
    results = tmp_path / "results"
    _write_result(
        results_dir=results,
        source=source,
        detections=[
            _detection("amount", "¥12.30", confidence=0.70),
            _detection("time", "99:99"),
        ],
    )

    output = tmp_path / "pseudo"
    records = build_pseudo_label_dataset(results_dir=results, output_dir=output, review_ratio=0.0)

    assert records == []
    rejected = [json.loads(line) for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {record["reason"] for record in rejected} == {"low_ocr_confidence", "field_semantic_validation_failed"}


def test_refuses_dataset_output_inside_results_tree(tmp_path: Path) -> None:
    source = tmp_path / "raw.png"
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(source)
    results = tmp_path / "results"
    _write_result(results_dir=results, source=source, detections=[_detection("amount", "¥1.00")])

    with pytest.raises(UnsafeOcrDatasetOutputError, match="must not overlap"):
        build_pseudo_label_dataset(results_dir=results, output_dir=results / "dataset")


def test_rejects_source_image_modified_after_paddle_result(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "receipt.png"
    Image.fromarray(np.full((40, 60, 3), 180, dtype=np.uint8)).save(source)
    results = tmp_path / "results"
    result = _write_result(results_dir=results, source=source, detections=[_detection("amount", "¥1.00")])
    result_mtime = result.stat().st_mtime_ns
    os.utime(source, ns=(result_mtime + 1_000_000, result_mtime + 1_000_000))

    with pytest.raises(ValueError, match="source image is newer"):
        build_pseudo_label_dataset(results_dir=results, output_dir=tmp_path / "pseudo")
