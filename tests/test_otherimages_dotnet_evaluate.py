from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transfer_receipt_ai.otherimages_dotnet_evaluate import (
    WhiteEvaluationError,
    _canonical_sha256,
    _canonical_view_contract,
    _parser,
    score_white_results,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def _fixture(
    tmp_path: Path, *, predicted_text: str = "ＡＢＣ  123", split: str = "test"
) -> dict[str, Path]:
    source = tmp_path / "images" / "白图.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"white-image-fixture")
    teacher = tmp_path / "teacher" / "teacher_manifest.jsonl"
    raw_sha256 = _sha256(source)
    record_id = hashlib.sha256(
        b"otherimages-image-record-v1\0" + "白图.png".encode("utf-8") + b"\0" + raw_sha256.encode("ascii")
    ).hexdigest()
    teacher_row: dict[str, object] = {
        "schema_version": 1,
        "kind": "otherimages_paddle_teacher_record_v1",
        "record_id": record_id,
        "group_id": "group-1",
        "split": split,
        "split_use": f"heldout_{split}",
        "source_root": str(source.parent.resolve()),
        "source_relative_path": source.name,
        "source_absolute_path": str(source.resolve()),
        "raw_sha256": raw_sha256,
        "decoded_pixel_sha256": "d" * 64,
        "text": "ABC 123",
        "text_sha256": hashlib.sha256(b"ABC 123").hexdigest(),
        "text_normalization": "NFKC_then_collapse_line_whitespace_v1",
        "lines": [
            {
                "index": 0,
                "text": "ABC 123",
                "confidence": 0.99,
                "orientation_degrees": 0,
                "transformed_quad_pixels": [[0.0, 0.0], [100.0, 0.0], [100.0, 5.0], [0.0, 5.0]],
                "quad_normalized": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.1]],
            }
        ],
        "label_source": "paddle_db_cls_rec_three_view_consensus",
        "consensus": {
            "dominant_text_votes": 3,
            "dominant_view_ids": ["original_rgb", "grayscale_clahe", "upscale_sharpen"],
            "geometry_support_votes": 3,
            "geometry_support_view_ids": ["original_rgb", "grayscale_clahe", "upscale_sharpen"],
            "agreement": "3_of_3",
            "chosen_geometry_view_id": "original_rgb",
            "minimum_pairwise_line_quad_iou": 1.0,
            "support_confidences": [
                {
                    "view_id": view_id,
                    "minimum_line_confidence": 0.99,
                    "mean_line_confidence": 0.99,
                }
                for view_id in ("original_rgb", "grayscale_clahe", "upscale_sharpen")
            ],
        },
        "chosen_view": {
            "view_id": "original_rgb",
            "view_contract_sha256": _canonical_sha256(_canonical_view_contract("original_rgb")),
            "transformed_pixel_sha256": "f" * 64,
            "source_width": 101,
            "source_height": 51,
            "transformed_width": 101,
            "transformed_height": 51,
            "coordinate_mapping": "full_frame_scale_source_normalized_identity_v1",
        },
        "training_eligible": False,
        "evaluation_only": True,
        "held_out": True,
        "automatic_teacher_validation": True,
        "manual_review_required": False,
    }
    _write_jsonl(teacher, [teacher_row])
    reject = teacher.with_name("reject_manifest.jsonl")
    _write_jsonl(reject, [])
    contract = teacher.with_name("teacher.contract.json")
    contract_payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "otherimages_paddle_teacher_contract_v1",
        "sealed": True,
        "output_directory": str(teacher.parent.resolve()),
        "split_use": {
            "train": "training_eligible",
            "val": "heldout_evaluation_only",
            "test": "heldout_evaluation_only",
            "group_split_source": "inventory_suggested_split",
            "groups_may_cross_splits": False,
        },
        "counts": {
            "accepted_teacher_records": 1,
            "quarantined_records": 0,
            "accepted_by_split": {split: 1},
            "training_eligible_records": 0,
            "evaluation_only_records": 1,
        },
        "inputs": {"model_assets": {"adapter_contract_sha256": "b" * 64}},
        "configuration": {
            "text_normalization": "NFKC_then_collapse_line_whitespace_v1",
            "minimum_line_confidence": 0.90,
        },
        "artifacts": [_binding(teacher), _binding(reject)],
        "training_authorization": False,
    }
    closure = {
        name: contract_payload[name]
        for name in ("schema_version", "inputs", "configuration", "counts", "split_use", "artifacts")
    }
    contract_payload["closure_sha256"] = hashlib.sha256(
        json.dumps(
            closure, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    _write_json(contract, contract_payload)
    receipt = teacher.with_name("teacher.receipt.json")
    _write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "otherimages_paddle_teacher_receipt_v1",
            "sealed": True,
            "contract": _binding(contract),
            "contract_closure_sha256": contract_payload["closure_sha256"],
        },
    )

    results = tmp_path / "results"
    result_path = results / f"{record_id}.json"
    result: dict[str, object] = {
        "result_schema_version": 1,
        "document_type": "white",
        "source": str(source.resolve()),
        "inference_engine": "dotnet_onnxruntime_cpu",
        "route": {"review_required": True},
        "ocr": {
            "provider": "cpu",
            "delivery_policy": "review_only",
            "aggregate_text": "ABC 123",
            "student_model_status": "integrated_review_only",
            "student_provider": "cpu",
            "student_comparison_line_count": 2,
            "student_normalized_exact_match_line_count": 2 if predicted_text == "ＡＢＣ  123" else 1,
            "student_crop_source": "same_paddle_db_cls_oriented_crop",
        },
        "lines": [
            {
                "index": 0,
                "text": "ABC 123",
                "confidence": 0.98,
                "passes_drop_score": True,
                "quad": [{"x": 0.0, "y": 0.0}] * 4,
                "student": {
                    "text": predicted_text,
                    "confidence": 0.97,
                    "normalized_exact_match": predicted_text == "ＡＢＣ  123",
                    "provider": "cpu",
                    "delivery_policy": "review_only",
                    "crop_source": "same_paddle_db_cls_oriented_crop",
                },
            },
            {
                "index": 1,
                "text": "diagnostic-only",
                "confidence": 0.10,
                "passes_drop_score": False,
                "quad": [{"x": 0.0, "y": 0.0}] * 4,
                "student": {
                    "text": "diagnostic-only",
                    "confidence": 0.70,
                    "normalized_exact_match": True,
                    "provider": "cpu",
                    "delivery_policy": "review_only",
                    "crop_source": "same_paddle_db_cls_oriented_crop",
                },
            },
        ],
        "model_contracts": {
            "device_sha256": "1" * 64,
            "device_contract_sha256": "2" * 64,
            "ocr_bundle_contract_sha256": "3" * 64,
            "ocr_detector_sha256": "4" * 64,
            "ocr_classifier_sha256": "5" * 64,
            "ocr_recognizer_sha256": "6" * 64,
            "ocr_dictionary_sha256": "7" * 64,
            "ocr_source_audit_contract_sha256": "8" * 64,
            "ocr_dictionary_snapshot_sha256": "9" * 64,
            "white_student_model": "white.onnx",
            "white_student_model_sha256": "a" * 64,
            "white_student_model_snapshot_size_bytes": 101,
            "white_student_charset": "white.charset.json",
            "white_student_charset_sha256": "b" * 64,
            "white_student_charset_snapshot_size_bytes": 102,
            "white_student_contract": "white.contract.json",
            "white_student_contract_sha256": "c" * 64,
            "white_student_contract_snapshot_size_bytes": 103,
            "white_student_runtime_source": "immutable_verified_bytes",
            "white_student_reopened_paths_after_verification": False,
            "runtime_source": "immutable_verified_bytes",
            "reopened_paths_after_verification": False,
        },
    }
    _write_json(result_path, result)
    _write_json(
        results / "inference_manifest.json",
        [
            {
                "source": str(source.resolve()),
                "result": str(result_path.resolve()),
                "status": "written",
                "inference_ms": 12.3,
            }
        ],
    )
    _write_json(
        results / "inference_summary.json",
        {
            "document_type": "white",
            "requested_device": "cpu",
            "paddle_ocr_provider": "cpu",
            "white_student_provider": "cpu",
            "input": 1,
            "written": 1,
            "skipped": 0,
            "errors": 0,
            "total_seconds": 0.1,
        },
    )
    (results / "inference_errors.jsonl").write_text("", encoding="utf-8")
    return {
        "teacher": teacher,
        "contract": contract,
        "receipt": receipt,
        "reject": reject,
        "results": results,
        "result": result_path,
        "source": source,
    }


def _reseal_teacher(fixture: dict[str, Path]) -> None:
    contract = json.loads(fixture["contract"].read_text(encoding="utf-8"))
    contract["artifacts"] = [_binding(fixture["teacher"]), _binding(fixture["reject"])]
    closure = {
        name: contract[name]
        for name in ("schema_version", "inputs", "configuration", "counts", "split_use", "artifacts")
    }
    contract["closure_sha256"] = hashlib.sha256(
        json.dumps(
            closure, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    _write_json(fixture["contract"], contract)
    _write_json(
        fixture["receipt"],
        {
            "schema_version": 1,
            "kind": "otherimages_paddle_teacher_receipt_v1",
            "sealed": True,
            "contract": _binding(fixture["contract"]),
            "contract_closure_sha256": contract["closure_sha256"],
        },
    )


def test_perfect_nfkc_cpu_teacher_parity_passes_and_writes_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "evaluation"

    summary = score_white_results(
        teacher_manifest=fixture["teacher"],
        teacher_contract=fixture["contract"],
        results_root=fixture["results"],
        output_dir=output,
        split="test",
    )

    assert summary["accepted"] is True
    assert summary["coverage"]["result_coverage"] == 1.0
    assert summary["teacher_agreement"]["overall"]["character_error_rate"] == 0.0
    assert summary["teacher_agreement"]["overall"]["document_exact_match"] == 1.0
    assert summary["teacher_agreement"]["overall"]["line_exact_precision"] == 1.0
    assert summary["teacher_agreement"]["metric_subject"] == "white_line_student"
    assert summary["teacher_agreement"]["by_consensus"]["3_of_3"]["records"] == 1
    assert summary["paddle_runtime_self_consistency_diagnostic"]["document_exact_match"] == 1.0
    assert "pseudo-label" in summary["warning"]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["accepted"] is True
    comparison = json.loads((output / "comparisons.jsonl").read_text(encoding="utf-8"))
    assert comparison["predicted_text"] == "ABC 123"


@pytest.mark.parametrize("with_existing_evidence", [False, True])
def test_existing_output_directory_is_never_reused_or_overwritten(
    tmp_path: Path, with_existing_evidence: bool
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "evaluation"
    output.mkdir()
    if with_existing_evidence:
        (output / "summary.json").write_bytes(b"prior-summary-evidence\n")
        (output / "comparisons.jsonl").write_bytes(b"prior-comparison-evidence\n")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(WhiteEvaluationError, match="already exists and cannot be reused"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            teacher_contract=fixture["contract"],
            results_root=fixture["results"],
            output_dir=output,
            split="test",
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_text_disagreement_is_reported_and_fails_floor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, predicted_text="XYZ")

    summary = score_white_results(
        teacher_manifest=fixture["teacher"],
        results_root=fixture["results"],
        output_dir=tmp_path / "evaluation",
        split="test",
    )

    assert summary["accepted"] is False
    assert summary["teacher_agreement"]["overall"]["character_error_rate"] > 0.05
    assert summary["paddle_runtime_self_consistency_diagnostic"]["character_error_rate"] == 0.0
    assert any("CER" in failure for failure in summary["failures"])
    assert any("document exact" in failure for failure in summary["failures"])


def test_non_cpu_result_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result_path = fixture["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["ocr"]["provider"] = "cuda"
    _write_json(result_path, result)

    with pytest.raises(WhiteEvaluationError, match="CPU/review/closure"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_result_outside_runtime_root_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    external = tmp_path / "external-result.json"
    external.write_bytes(fixture["result"].read_bytes())
    manifest_path = fixture["results"] / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[0]["result"] = str(external.resolve())
    _write_json(manifest_path, manifest)

    with pytest.raises(WhiteEvaluationError, match="outside the supplied results root"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_missing_student_line_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result_path = fixture["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    del result["lines"][1]["student"]
    _write_json(result_path, result)

    with pytest.raises(WhiteEvaluationError, match="has no student evidence"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_student_provider_tamper_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result_path = fixture["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["lines"][0]["student"]["provider"] = "cuda"
    _write_json(result_path, result)

    with pytest.raises(WhiteEvaluationError, match="student provider/delivery/crop"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_student_bundle_closure_tamper_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result_path = fixture["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    del result["model_contracts"]["white_student_contract_sha256"]
    _write_json(result_path, result)

    with pytest.raises(WhiteEvaluationError, match="model/student SHA closure"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_source_mutation_after_teacher_seal_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["source"].write_bytes(b"changed")

    with pytest.raises(WhiteEvaluationError, match="source is missing or differs"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_missing_teacher_orientation_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    row = json.loads(fixture["teacher"].read_text(encoding="utf-8"))
    del row["lines"][0]["orientation_degrees"]
    _write_jsonl(fixture["teacher"], [row])
    _reseal_teacher(fixture)

    with pytest.raises(WhiteEvaluationError, match="orientation_degrees"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_cli_defaults_to_frozen_test_split() -> None:
    args = _parser().parse_args(
        [
            "--teacher-manifest",
            "teacher_manifest.jsonl",
            "--results",
            "results",
            "--output",
            "evaluation",
        ]
    )

    assert args.split == "test"


def test_validation_split_is_explicitly_diagnostic_and_never_accepted(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, split="val")

    summary = score_white_results(
        teacher_manifest=fixture["teacher"],
        results_root=fixture["results"],
        output_dir=tmp_path / "evaluation",
        split="val",
    )

    assert summary["teacher_agreement"]["overall"]["character_error_rate"] == 0.0
    assert summary["accepted"] is False
    assert any("diagnostic-only" in failure for failure in summary["failures"])


def test_teacher_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt["contract_closure_sha256"] = "0" * 64
    _write_json(fixture["receipt"], receipt)

    with pytest.raises(WhiteEvaluationError, match="receipt does not bind"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_invalid_teacher_record_id_fails_after_full_reseal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    row = json.loads(fixture["teacher"].read_text(encoding="utf-8"))
    row["record_id"] = "0" * 64
    _write_jsonl(fixture["teacher"], [row])
    _reseal_teacher(fixture)

    with pytest.raises(WhiteEvaluationError, match="record_id does not bind"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )


def test_incomplete_teacher_row_provenance_fails_after_full_reseal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    row = json.loads(fixture["teacher"].read_text(encoding="utf-8"))
    del row["source_relative_path"]
    _write_jsonl(fixture["teacher"], [row])
    _reseal_teacher(fixture)

    with pytest.raises(WhiteEvaluationError, match="source_relative_path"):
        score_white_results(
            teacher_manifest=fixture["teacher"],
            results_root=fixture["results"],
            output_dir=tmp_path / "evaluation",
            split="test",
        )
