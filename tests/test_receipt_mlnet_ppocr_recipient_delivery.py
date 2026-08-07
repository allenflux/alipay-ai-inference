from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "scripts" / "receipt-ocr-ppocrv4-freeze-4090.ps1"
PACKAGE = ROOT / "scripts" / "receipt-ocr-ppocrv4-recipient-onnx-package.ps1"
CPU_AB = ROOT / "scripts" / "receipt-mlnet-hybrid-recipient-cpu-ab.ps1"
SAMPLE = ROOT / "scripts" / "receipt-ppocr-val-parity-sample.py"
COMPARATOR = ROOT / "scripts" / "receipt-mlnet-hybrid-recipient-ab.py"
DOTNET_PARITY_PROGRAM = ROOT / "dotnet" / "ReceiptMlNet.Cli.PaddleParity" / "Program.cs"
DOTNET_PROGRAM = ROOT / "dotnet" / "ReceiptMlNet.Cli" / "Program.cs"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conversion_launchers_are_val_only_dynamic_and_paddle_free_at_delivery() -> None:
    freeze = FREEZE.read_text(encoding="utf-8")
    package = PACKAGE.read_text(encoding="utf-8")

    assert "paddle_ocr_bundle snapshot" in freeze
    assert "--device cuda" in freeze
    assert "allow-model-download" not in freeze
    assert 'evaluation_split -ne "val"' in package
    assert 'inference_mode.name -ne "full_det_cls_rec"' in package
    assert "anchored_value_exact_match -lt 0.90" in package
    assert "--opset-version 11" in freeze
    assert "--min-text-exact-match 1.0" in package
    assert "--max-confidence-delta 0.01" in package
    assert "package-delivery" in package
    assert "verify-delivery" in package
    assert "OnnxRuntimeFlavor=cpu" in package
    assert "TrustedManifestSha256" in package
    assert "$expectedFullValRecords = 6789" in package
    assert '$null -ne $valSummary.limit' in package
    assert "native_asset_identity_sha256" in package
    assert "dotnetParityProject" in package
    assert "dotnetParityComparator" in package
    assert "Move-Item -LiteralPath $deliveryStage -Destination $DeliveryDirectory" in package
    assert package.index("dotnetParityComparator") < package.index(
        "Move-Item -LiteralPath $deliveryStage -Destination $DeliveryDirectory"
    )
    assert "--split test" not in package


def test_dotnet_parity_emits_single_line_jsonl_records() -> None:
    source = DOTNET_PARITY_PROGRAM.read_text(encoding="utf-8")
    assert "WriteIndented = false" in source
    assert "writer.WriteLine(JsonSerializer.Serialize(record, JsonOptions))" in source


def test_cpu_ab_runs_same_val_inputs_through_v13_and_hybrid_on_cpu() -> None:
    source = CPU_AB.read_text(encoding="utf-8")

    assert "prepare --records $Records --output $inputList --split val" in source
    assert '"--device", "cpu"' in source
    assert '"--device-model", $DeviceModel' in source
    assert '"--ocr", "unified"' in source
    assert '"--ocr", "hybrid-recipient"' in source
    assert '"--ocr-bundle", $PaddleDeliveryBundle' in source
    assert '"--delivery", $PaddleDeliveryBundle' in source
    assert '"--rectification", "max-side-1600"' in source
    assert "receipt-mlnet-hybrid-recipient-ab.py" in source
    assert '"--amount-floor", "0.7885"' in source
    assert '"--time-floor", "0.9840"' in source
    assert '"--payment-floor", "0.9325"' in source
    assert '"--recipient-floor", "0.90"' in source
    assert '"--status-floor", "0.90"' in source
    assert '[ValidateSet("pilot", "formal")]' in source
    assert '[double]$MaxP95OverheadMs = 250.0' in source
    assert '$modeName -eq "formal" -and $Limit -ne 0' in source
    assert 'formal_delivery_gate=true' in source
    assert '$passLabel = "PILOT PASS"' in source
    assert '$passLabel = "FORMAL PASS"' in source
    assert '"--require-complete"' not in source
    assert '"--continue-on-error"' not in source
    assert "--split test" not in source


def test_dotnet_complete_detection_contract_excludes_status_bar_clock() -> None:
    source = DOTNET_PROGRAM.read_text(encoding="utf-8")
    block = source.split("private static readonly string[] RequiredLabels =", 1)[1].split("};", 1)[0]

    assert '"amount"' in block
    assert '"transfer_status"' in block
    assert '"recipient_field"' in block
    assert '"payment_method_field"' in block
    assert '"time"' not in block


def test_val_parity_sample_rejects_non_val_or_non_full_pipeline(tmp_path: Path) -> None:
    module = _load_module(SAMPLE, "receipt_ppocr_val_sample")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    image = tmp_path / "crop.png"
    image.write_bytes(b"fake-image")
    summary = {
        "evaluation_split": "test",
        "records": 1,
        "anchored_value_exact_match": 1.0,
        "inference_mode": {
            "name": "full_det_cls_rec",
            "detection_enabled": True,
            "angle_classifier_enabled": True,
            "recognizer_enabled": True,
        },
        "acceptance": {"passed": True},
    }
    (evidence / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (evidence / "comparisons.jsonl").write_text(
        json.dumps({"id": "x", "split": "test", "inference_mode": "full_det_cls_rec", "image": str(image)})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation-split"):
        module.create_sample(evidence=evidence, output=tmp_path / "out", limit=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crop_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        shape = (rgb.height, rgb.width, 3)
        pixels = rgb.tobytes()
    digest = hashlib.sha256()
    digest.update(str(shape).encode("ascii"))
    digest.update(pixels)
    return digest.hexdigest()


def _write_bound_full_val_evidence(tmp_path: Path) -> tuple[Path, Path, str, Path]:
    audit_bundle = tmp_path / "audit"
    audit_bundle.mkdir()
    native_identity = {
        "sha256": "1" * 64,
        "components": {role: character * 64 for role, character in zip(("det", "rec", "cls", "dictionary"), "2345")},
    }
    audit_contract = {
        "schema_version": 1,
        "kind": "paddle_ocr_v2_bundle",
        "onnx": {role: {} for role in ("det", "rec", "cls")},
        "native_asset_identity": native_identity,
    }
    audit_contract_path = audit_bundle / "paddle_ocr_bundle.contract.json"
    audit_contract_path.write_text(json.dumps(audit_contract), encoding="utf-8")

    manifest = tmp_path / "unified_fields.jsonl"
    manifest.write_text('{"fixture":"trusted full manifest"}\n', encoding="utf-8")
    manifest_sha256 = _sha256(manifest)
    image = tmp_path / "crop.png"
    Image.new("RGB", (8, 5), color=(17, 23, 42)).save(image)
    comparison = {
        "schema_version": 1,
        "kind": "receipt_paddle_recipient_teacher_parity_v1",
        "id": "val-one",
        "split": "val",
        "inference_mode": "full_det_cls_rec",
        "image": str(image.resolve()),
        "crop_sha256": _crop_sha256(image),
        "crop_file_sha256": _sha256(image),
        "anchored_value_exact": True,
    }
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comparisons = evidence / "comparisons.jsonl"
    comparisons.write_text(json.dumps(comparison) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "kind": "receipt_paddle_recipient_teacher_parity_v1",
        "evaluation_split": "val",
        "records": 1,
        "limit": None,
        "requested_device": "cuda",
        "runtime": {"active_paddle_device": "gpu:0"},
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "comparisons_sha256": _sha256(comparisons),
        "inference_mode": {
            "name": "full_det_cls_rec",
            "experimental": False,
            "detection_enabled": True,
            "angle_classifier_enabled": True,
            "recognizer_enabled": True,
        },
        "anchored_value_exact_matches": 1,
        "anchored_value_exact_match": 1.0,
        "acceptance": {"passed": True, "target_anchored_value_exact_match": 0.90},
        "frozen_bundle": {
            "path": str(audit_bundle.resolve()),
            "contract_kind": "paddle_ocr_v2_bundle",
            "contract_sha256": _sha256(audit_contract_path),
            "native_asset_identity_sha256": native_identity["sha256"],
            "native_component_sha256": native_identity["components"],
            "live_source_bytes_verified": True,
            "verified_before_and_after": True,
            "verified": True,
        },
    }
    (evidence / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return evidence, audit_bundle, manifest_sha256, image


def test_val_parity_sample_accepts_only_bound_unbounded_full_evidence(tmp_path: Path) -> None:
    module = _load_module(SAMPLE, "receipt_ppocr_val_sample_bound")
    evidence, audit_bundle, manifest_sha256, _ = _write_bound_full_val_evidence(tmp_path)

    output = module.create_sample(
        evidence=evidence,
        output=tmp_path / "sample",
        limit=1,
        audit_bundle=audit_bundle,
        trusted_manifest_sha256=manifest_sha256,
        expected_records=1,
    )

    sample = json.loads((output / "sample_manifest.json").read_text(encoding="utf-8"))
    assert sample["source_manifest_sha256"] == manifest_sha256
    assert sample["source_audit_contract_sha256"] == _sha256(
        audit_bundle / "paddle_ocr_bundle.contract.json"
    )


@pytest.mark.parametrize("tamper", ["partial", "bundle", "crop"])
def test_val_parity_sample_rejects_partial_unbound_or_tampered_evidence(tmp_path: Path, tamper: str) -> None:
    module = _load_module(SAMPLE, f"receipt_ppocr_val_sample_bad_{tamper}")
    evidence, audit_bundle, manifest_sha256, image = _write_bound_full_val_evidence(tmp_path)
    if tamper == "partial":
        summary_path = evidence / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["limit"] = 1
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    elif tamper == "bundle":
        summary_path = evidence / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["frozen_bundle"]["contract_sha256"] = "9" * 64
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        image.write_bytes(b"crop-was-changed")

    with pytest.raises(ValueError, match="(unbounded|bundle identity|crop (?:file )?hash mismatch)"):
        module.create_sample(
            evidence=evidence,
            output=tmp_path / "sample",
            limit=1,
            audit_bundle=audit_bundle,
            trusted_manifest_sha256=manifest_sha256,
            expected_records=1,
        )


def _write_delivery(directory: Path) -> tuple[Path, str]:
    directory.mkdir()
    models: dict[str, dict[str, object]] = {}
    for role in ("det", "rec", "cls"):
        path = directory / "onnx" / f"paddle_ocr_{role}.onnx"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"fixture-{role}".encode())
        models[role] = {
            "path": path.relative_to(directory).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    dictionary_path = directory / "charset" / "ppocr_keys_v1.txt"
    dictionary_path.parent.mkdir()
    dictionary_path.write_text("收\n款\n方\n", encoding="utf-8")
    contract = {
        "schema_version": 1,
        "kind": "paddle_ocr_v2_delivery",
        "models": models,
        "dictionary": {
            "path": dictionary_path.relative_to(directory).as_posix(),
            "sha256": _sha256(dictionary_path),
            "size_bytes": dictionary_path.stat().st_size,
        },
    }
    contract_path = directory / "paddle_ocr_delivery.contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return directory, _sha256(contract_path)


def _write_ab_run(
    directory: Path,
    *,
    hybrid: bool,
    paddle_contract_sha256: str,
    amount_candidate: str = "10.00",
) -> None:
    directory.mkdir()
    summary = {
        "requested_device": "cpu",
        "paddle_ocr_provider": "cpu" if hybrid else None,
        "unified_provider": "cpu",
        "input": 1,
        "written": 1,
        "skipped": 0,
        "errors": 0,
        "inference_latency_ms": {"p50": 100.0 if not hybrid else 120.0, "p95": 110.0 if not hybrid else 135.0},
        "stage_latency_ms": {"paddle_ocr": {"p50": None if not hybrid else 20.0, "p95": None if not hybrid else 25.0}},
    }
    (directory / "inference_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    source = str((directory.parent / "receipt.jpg").resolve())
    result_path = directory / "result.json"
    contract = {
        "detector": "det.contract.json",
        "detector_sha256": "a" * 64,
        "detector_contract_sha256": "b" * 64,
        "device": "device.contract.json",
        "device_sha256": "c" * 64,
        "device_contract_sha256": "d" * 64,
        "unified_ocr_model": "v13.onnx",
        "unified_ocr_contract": "v13.contract.json",
        "unified_ocr_model_sha256": "e" * 64,
        "unified_ocr_labels_sha256": "f" * 64,
        "unified_ocr_contract_sha256": "0" * 64,
    }
    if hybrid:
        contract.update(
            {
                "ocr_bundle": "paddle_ocr_delivery.contract.json",
                "ocr_bundle_contract_sha256": paddle_contract_sha256,
            }
        )
    common_field = {"state": "review", "candidate": amount_candidate, "delivery_value": "review"}
    recipient = {
        "state": "review",
        "candidate": "新收款方" if hybrid else "旧收款方",
        "ctc_candidate": "新收款方" if hybrid else "旧收款方",
        "detector_score": 0.99,
        "delivery_policy": "review_only_pending_independent_human_truth_calibration",
        "delivery_value": "review",
        "value": "review",
    }
    detections = [
        {"label": "amount", "score": 0.99, "bbox_image": [0, 0, 10, 10], "ocr": {"text": amount_candidate}},
        {
            "label": "recipient_field",
            "score": 0.99,
            "bbox_image": [0, 10, 10, 20],
            "ocr": {"text": recipient["candidate"]},
        },
    ]
    result = {
        "result_schema_version": 1,
        "result_semantics_version": "fixture",
        "source": source,
        "inference_engine": "mlnet",
        "geometry": {"rectification": "max-side-1600"},
        "device": {"platform": "android"},
        "model_contracts": contract,
        "fields": {
            "time": {"state": "review", "candidate": "12:30", "delivery_value": "review"},
            "amount": common_field,
            "transfer_status": {"state": "review", "candidate": "转账成功", "normalized": "success"},
            "recipient": recipient,
            "payment_method": {"state": "review", "candidate": "余额", "delivery_value": "review"},
        },
        "detections": detections,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (directory / "inference_manifest.json").write_text(
        json.dumps([{"source": source, "result": str(result_path), "status": "written"}]),
        encoding="utf-8",
    )


def test_cpu_ab_comparator_allows_only_recipient_candidate_change(tmp_path: Path) -> None:
    module = _load_module(COMPARATOR, "receipt_hybrid_ab")
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_ab_run(baseline, hybrid=False, paddle_contract_sha256=delivery_sha256)
    _write_ab_run(hybrid, hybrid=True, paddle_contract_sha256=delivery_sha256)

    summary, accepted = module.compare(
        baseline_dir=baseline,
        hybrid_dir=hybrid,
        paddle_delivery=delivery,
        output=tmp_path / "report",
        max_p95_overhead_ms=30.0,
    )

    assert accepted is True
    assert summary["invariant_records"] == 1
    assert summary["recipient_candidate_coverage"] == 1.0
    assert summary["cpu"]["p95_overhead_ms"] == 25.0
    assert summary["paddle_delivery"]["contract_sha256"] == delivery_sha256
    assert summary["cli_summary_counts_verified"] is True


def test_cpu_ab_comparator_normalizes_read_and_unreadable_detector_score_shapes() -> None:
    module = _load_module(COMPARATOR, "receipt_hybrid_ab_detector_score_shape")

    assert module._canonical_detector_score({"detector_score": 0.99}, "read") == pytest.approx(0.99)
    assert module._canonical_detector_score({"score": 0.99}, "unreadable") == pytest.approx(0.99)
    assert module._canonical_detector_score({}, "absent") is None

    with pytest.raises(module.ComparisonError, match="contains both"):
        module._canonical_detector_score(
            {"detector_score": 0.99, "score": 0.99},
            "conflicting schema",
        )
    with pytest.raises(module.ComparisonError, match="finite number"):
        module._canonical_detector_score({"score": "0.99"}, "wrong type")


def test_cpu_ab_comparator_rejects_protected_field_change(tmp_path: Path) -> None:
    module = _load_module(COMPARATOR, "receipt_hybrid_ab_changed")
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_ab_run(baseline, hybrid=False, paddle_contract_sha256=delivery_sha256)
    _write_ab_run(
        hybrid,
        hybrid=True,
        paddle_contract_sha256=delivery_sha256,
        amount_candidate="11.00",
    )

    summary, accepted = module.compare(
        baseline_dir=baseline,
        hybrid_dir=hybrid,
        paddle_delivery=delivery,
        output=tmp_path / "report",
        max_p95_overhead_ms=250.0,
    )

    assert accepted is False
    assert any("fields.amount changed" in failure for failure in summary["failures"])


def test_cpu_ab_comparator_rejects_tampered_delivery_asset(tmp_path: Path) -> None:
    module = _load_module(COMPARATOR, "receipt_hybrid_ab_tampered_delivery")
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_ab_run(baseline, hybrid=False, paddle_contract_sha256=delivery_sha256)
    _write_ab_run(hybrid, hybrid=True, paddle_contract_sha256=delivery_sha256)
    (delivery / "onnx" / "paddle_ocr_rec.onnx").write_bytes(b"tampered-recognizer")

    with pytest.raises(module.ComparisonError, match="rec model (size|SHA-256) mismatch"):
        module.compare(
            baseline_dir=baseline,
            hybrid_dir=hybrid,
            paddle_delivery=delivery,
            output=tmp_path / "report",
        )


def test_cpu_ab_comparator_rejects_unbound_or_non_lowercase_contract_hash(tmp_path: Path) -> None:
    module = _load_module(COMPARATOR, "receipt_hybrid_ab_bad_contract_hash")
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_ab_run(baseline, hybrid=False, paddle_contract_sha256=delivery_sha256)
    _write_ab_run(hybrid, hybrid=True, paddle_contract_sha256=delivery_sha256.upper())

    with pytest.raises(module.ComparisonError, match="64 lowercase hexadecimal"):
        module.compare(
            baseline_dir=baseline,
            hybrid_dir=hybrid,
            paddle_delivery=delivery,
            output=tmp_path / "report",
        )


def test_cpu_ab_comparator_rejects_summary_manifest_count_mismatch(tmp_path: Path) -> None:
    module = _load_module(COMPARATOR, "receipt_hybrid_ab_bad_count")
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_ab_run(baseline, hybrid=False, paddle_contract_sha256=delivery_sha256)
    _write_ab_run(hybrid, hybrid=True, paddle_contract_sha256=delivery_sha256)
    for run in (baseline, hybrid):
        summary_path = run / "inference_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["input"] = summary["written"] = 2
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(module.ComparisonError, match="manifest/result count 1 differs"):
        module.compare(
            baseline_dir=baseline,
            hybrid_dir=hybrid,
            paddle_delivery=delivery,
            output=tmp_path / "report",
        )


def test_cpu_ab_comparator_cannot_relax_fixed_p95_ceiling(tmp_path: Path) -> None:
    module = _load_module(COMPARATOR, "receipt_hybrid_ab_p95_ceiling")
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_ab_run(baseline, hybrid=False, paddle_contract_sha256=delivery_sha256)
    _write_ab_run(hybrid, hybrid=True, paddle_contract_sha256=delivery_sha256)

    with pytest.raises(module.ComparisonError, match="fixed release ceiling"):
        module.compare(
            baseline_dir=baseline,
            hybrid_dir=hybrid,
            paddle_delivery=delivery,
            output=tmp_path / "report",
            max_p95_overhead_ms=250.01,
        )
