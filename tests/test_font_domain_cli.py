from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

import transfer_receipt_ai.font_domain_cli as cli_module
from transfer_receipt_ai.font_domain import RESULT_KIND
from transfer_receipt_ai.font_domain_cli import RUN_KIND, main
from transfer_receipt_ai.font_domain_dataset import DOCUMENT_KIND


def _source_hash(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _write_pattern(path: Path, domain: str, seed: int) -> None:
    """Write a deterministic, information-gate-passing synthetic text line."""

    image = Image.new("RGB", (144, 48), "white")
    draw = ImageDraw.Draw(image)
    for index in range(6):
        x = 8 + index * 21 + ((seed + index) % 2)
        y = 9 + ((seed + index * 2) % 3)
        if domain == "ios_alipay":
            width = 5 + ((seed + index) % 3)
            height = 24 + ((seed * 2 + index) % 4)
        else:
            width = 14 + ((seed + index) % 4)
            height = 9 + ((seed * 3 + index) % 3)
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill="black")
    # Bind every fixture to a distinct decoded-pixel identity without adding
    # another connected component or materially changing its domain geometry.
    draw.point((10, 15), fill=(20 + seed % 180,) * 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _region(image: str, region_id: str, role: str) -> dict[str, object]:
    return {
        "id": region_id,
        "role": role,
        "image": image,
        "include_in_consistency": True,
    }


def _document(
    document_id: str,
    *,
    split: str,
    domain: str | None,
    regions: list[dict[str, object]],
    strict_metadata: bool = True,
    content_group_id: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": DOCUMENT_KIND,
        "id": document_id,
        "source_group_id": f"source-{document_id}",
        "split": split,
        "font_domain": domain,
        "regions": regions,
    }
    if domain is not None:
        record["label_source"] = "synthetic_test_fixture"
    if strict_metadata:
        record["content_group_id"] = content_group_id or f"content-{document_id}"
        record["source_image_sha256"] = _source_hash(f"source-image:{document_id}")
    return record


def _write_manifest(root: Path, name: str, records: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _training_manifest(
    root: Path,
    *,
    include_calibration: bool = True,
    include_test: bool = False,
) -> Path:
    records: list[dict[str, object]] = []
    roles = ("amount", "recipient", "time")
    seed = 10
    for split in (("train", "calibration") if include_calibration else ("train",)):
        for domain in ("android_alipay", "ios_alipay"):
            if split == "calibration":
                for index in range(20):
                    role = roles[index % len(roles)]
                    relative = f"images/{split}-{domain}-{index}.png"
                    _write_pattern(root / relative, domain, seed)
                    seed += 1
                    document_id = f"{split}-{domain}-{index}"
                    records.append(
                        _document(
                            document_id,
                            split=split,
                            domain=domain,
                            regions=[_region(relative, role, role)],
                        )
                    )
                continue
            regions: list[dict[str, object]] = []
            # Three train rows satisfy the default fit gate.  Calibration uses
            # twenty independent source groups per domain above so a 0.05
            # conformal tail can actually reject an outlier.
            count = 3
            for index in range(count):
                role = roles[index]
                relative = f"images/{split}-{domain}-{index}.png"
                _write_pattern(root / relative, domain, seed)
                seed += 1
                regions.append(_region(relative, f"{role}-{index}", role))
            document_id = f"{split}-{domain}"
            records.append(
                _document(
                    document_id,
                    split=split,
                    domain=domain,
                    regions=regions,
                )
            )
    if include_test:
        for domain in ("android_alipay", "ios_alipay"):
            for document_index in range(2):
                regions = []
                document_id = f"test-{domain}-{document_index}"
                for region_index, role in enumerate(roles):
                    relative = (
                        f"images/{document_id}-{region_index}.png"
                    )
                    _write_pattern(root / relative, domain, seed)
                    seed += 1
                    regions.append(_region(relative, f"{role}-{region_index}", role))
                records.append(
                    _document(
                        document_id,
                        split="test",
                        domain=domain,
                        regions=regions,
                    )
                )
    return _write_manifest(root, "training.jsonl", records)


def _inference_manifest(root: Path) -> Path:
    regions: list[dict[str, object]] = []
    for index, role in enumerate(("amount", "recipient", "time")):
        relative = f"images/inference-{index}.png"
        _write_pattern(root / relative, "ios_alipay", 100 + index)
        regions.append(_region(relative, role, role))
    record = _document(
        "receipt-inference",
        split="inference",
        domain=None,
        regions=regions,
        strict_metadata=False,
    )
    # A contrary device prior must not affect font-only analysis.
    record["device_prior_domain"] = "android_alipay"
    return _write_manifest(root, "inference.jsonl", [record])


def _invoke(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    code = main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_validate_fit_and_analyze_publish_bound_sidecar_without_clobber(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    training = _training_manifest(tmp_path / "training")

    code, stdout, stderr = _invoke(
        capsys,
        "validate",
        "--records",
        str(training),
        "--skip-near-duplicate-audit",
    )
    assert code == 0
    assert stderr == ""
    validation = json.loads(stdout)
    assert validation["validation"] == "passed"
    assert validation["mode"] == "training"
    assert validation["dataset"]["documents"] == 42
    assert validation["dataset"]["splits"] == {"calibration": 40, "train": 2}

    model = tmp_path / "published" / "font-domain-model.json"
    fit_arguments = (
        "fit",
        "--records",
        str(training),
        "--output",
        str(model),
        "--skip-near-duplicate-audit",
        "--confidence-threshold",
        "0.5",
        "--margin-threshold",
        "0",
        "--fit-p-threshold",
        "0",
    )
    code, stdout, stderr = _invoke(capsys, *fit_arguments)
    assert code == 0
    assert stderr == ""
    fitted = json.loads(stdout)
    assert fitted["calibration_prerequisites_met"] is True
    assert fitted["calibration_source"] == {
        "android_alipay": "calibration",
        "ios_alipay": "calibration",
    }
    assert fitted["authenticity"] == "not_assessed"
    assert fitted["publication_prerequisites_recorded"] is False
    assert fitted["model"]["path"] == str(model.resolve())
    held_out = fitted["held_out_evaluation"]
    assert held_out["status"] == "not_available"
    assert held_out["authenticity"] == "not_assessed"
    assert held_out["documents"]["total"] == 0
    assert held_out["regions"]["total"] == 0
    assert held_out["device_prior_used"] is False
    assert held_out["decision_counts"] == {"PASS": 0, "REVIEW": 0, "UNKNOWN": 0}
    model_before = model.read_bytes()

    code, stdout, stderr = _invoke(capsys, *fit_arguments)
    assert code == 2
    assert stdout == ""
    assert "refusing to overwrite model artifact" in stderr
    assert model.read_bytes() == model_before

    inference = _inference_manifest(tmp_path / "inference")
    output = tmp_path / "analysis" / "receipt-001"
    analyze_arguments = (
        "analyze",
        "--model",
        str(model),
        "--records",
        str(inference),
        "--output",
        str(output),
        "--allow-experimental-model",
    )
    code, stdout, stderr = _invoke(capsys, *analyze_arguments)
    assert code == 0
    assert stderr == ""
    published = json.loads(stdout)
    assert published["output"] == str(output.resolve())

    sidecar_path = output / "font_domain.sidecar.jsonl"
    errors_path = output / "errors.jsonl"
    run_path = output / "run.json"
    assert sidecar_path.is_file()
    assert errors_path.is_file()
    assert run_path.is_file()
    sidecar_bytes = sidecar_path.read_bytes()
    errors_bytes = errors_path.read_bytes()
    run_bytes = run_path.read_bytes()
    assert errors_bytes == b""

    sidecar_rows = [json.loads(line) for line in sidecar_bytes.decode("utf-8").splitlines()]
    assert len(sidecar_rows) == 1
    assert sidecar_rows[0]["kind"] == RESULT_KIND
    assert sidecar_rows[0]["document_id"] == "receipt-inference"
    assert sidecar_rows[0]["authenticity"] == "not_assessed"
    assert len(sidecar_rows[0]["lines"]) == 3
    assert sidecar_rows[0]["decision"] == "PASS"
    assert sidecar_rows[0]["requires_manual_review"] is True
    assert sidecar_rows[0]["model_evidence"]["evaluation_status"] == "not_assessed"
    assert sidecar_rows[0]["model_evidence"]["device_prior_used"] is False
    assert sidecar_rows[0]["model_evidence"]["prediction_inputs"] == [
        "crop_pixels",
        "region_role",
    ]

    run = json.loads(run_bytes)
    assert run["kind"] == RUN_KIND
    assert run["documents"] == 1
    assert sum(run["decisions"].values()) == 1
    assert run["failures"] == 0
    assert run["authenticity"] == "not_assessed"
    assert run["aggregation"]["device_prior_used"] is False
    assert run["outputs"]["font_domain.sidecar.jsonl"] == {
        "records": 1,
        "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "size_bytes": len(sidecar_bytes),
    }
    assert run["outputs"]["errors.jsonl"] == {
        "records": 0,
        "sha256": hashlib.sha256(errors_bytes).hexdigest(),
        "size_bytes": 0,
    }

    code, stdout, stderr = _invoke(capsys, *analyze_arguments)
    assert code == 2
    assert stdout == ""
    assert "refusing to overwrite analysis output" in stderr
    assert sidecar_path.read_bytes() == sidecar_bytes
    assert errors_path.read_bytes() == errors_bytes
    assert run_path.read_bytes() == run_bytes


def test_platform_proxy_font_labels_force_experimental_model_safety(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _training_manifest(tmp_path / "weak-training", include_test=True)
    rows = [
        json.loads(line)
        for line in training.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["label_source"] = "device_platform_proxy_font_rendering_weak_v1"
        if row["split"] == "test":
            # Simulate a legacy manifest carrying the label-derived prior.
            # The evaluator must remove it before asking for predictions.
            row["device_prior_domain"] = row["font_domain"]
    training.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    def fake_predict_document(
        model: object,
        document: object,
        **aggregation_options: object,
    ) -> SimpleNamespace:
        del model
        assert getattr(document, "device_prior_domain") is None
        assert aggregation_options == {"minimum_regions": 1, "minimum_roles": 1}
        reference = str(getattr(document, "font_domain"))
        other = "ios_alipay" if reference == "android_alipay" else "android_alipay"
        correct_document = str(getattr(document, "document_id")).endswith("-0")
        if correct_document:
            lines = (
                SimpleNamespace(include_in_consistency=True, accepted=True, label=reference),
                SimpleNamespace(include_in_consistency=True, accepted=True, label=reference),
                SimpleNamespace(include_in_consistency=True, accepted=False, label="unknown"),
            )
            return SimpleNamespace(
                dominant_domain=reference,
                decision="PASS",
                lines=lines,
            )
        lines = tuple(
            SimpleNamespace(include_in_consistency=True, accepted=True, label=other)
            for _ in range(3)
        )
        return SimpleNamespace(dominant_domain=other, decision="REVIEW", lines=lines)

    monkeypatch.setattr(cli_module, "predict_document", fake_predict_document)

    model = tmp_path / "weak-model.json"
    code, stdout, stderr = _invoke(
        capsys,
        "fit",
        "--records",
        str(training),
        "--output",
        str(model),
        "--skip-near-duplicate-audit",
        "--confidence-threshold",
        "0.5",
        "--margin-threshold",
        "0",
        "--fit-p-threshold",
        "0",
    )

    assert code == 0
    assert stderr == ""
    fitted = json.loads(stdout)
    assert fitted["near_duplicate_audit"] is None
    assert fitted["label_sources"] == [
        "device_platform_proxy_font_rendering_weak_v1"
    ]
    assert fitted["training_label_sources"] == [
        "device_platform_proxy_font_rendering_weak_v1"
    ]
    assert fitted["test_label_sources"] == [
        "device_platform_proxy_font_rendering_weak_v1"
    ]
    assert fitted["label_evidence_status"] == "font_rendering_platform_proxy_weak"
    assert fitted["publication_safety"]["leakage_metadata"] == (
        "font_rendering_platform_proxy_weak"
    )
    assert fitted["publication_prerequisites_recorded"] is False
    held_out = fitted["held_out_evaluation"]
    assert held_out["status"] == "platform_proxy_font_rendering_test_split"
    assert held_out["authenticity"] == "not_assessed"
    assert held_out["device_prior_used"] is False
    assert held_out["reference_label_used_for_prediction"] is False
    assert held_out["reference_label_sources"] == [
        "device_platform_proxy_font_rendering_weak_v1"
    ]
    assert held_out["decision_counts"] == {"PASS": 2, "REVIEW": 2, "UNKNOWN": 0}
    assert held_out["decision_rates"] == {"PASS": 0.5, "REVIEW": 0.5, "UNKNOWN": 0.0}
    assert held_out["documents"] == {
        "total": 4,
        "covered": 4,
        "correct": 2,
        "classification_coverage": 1.0,
        "selective_accuracy": 0.5,
        "overall_accuracy": 0.5,
        "prediction_counts": {"android_alipay": 2, "ios_alipay": 2},
        "per_domain": {
            domain: {
                "total": 2,
                "covered": 2,
                "correct": 1,
                "classification_coverage": 1.0,
                "selective_accuracy": 0.5,
                "overall_accuracy": 0.5,
                "prediction_counts": {"android_alipay": 1, "ios_alipay": 1},
            }
            for domain in ("android_alipay", "ios_alipay")
        },
    }
    assert held_out["regions"]["total"] == 12
    assert held_out["regions"]["covered"] == 10
    assert held_out["regions"]["correct"] == 4
    assert held_out["regions"]["classification_coverage"] == 0.833333
    assert held_out["regions"]["selective_accuracy"] == 0.4
    assert held_out["regions"]["overall_accuracy"] == 0.333333
    for level in ("documents", "regions"):
        metrics = held_out[level]
        assert metrics["classification_coverage"] == round(
            metrics["covered"] / metrics["total"], 6
        )
        assert metrics["overall_accuracy"] == round(
            metrics["correct"] / metrics["total"], 6
        )
        if metrics["covered"]:
            assert metrics["selective_accuracy"] == round(
                metrics["correct"] / metrics["covered"], 6
            )
        else:
            assert metrics["selective_accuracy"] is None
        assert set(metrics["per_domain"]) == {
            "android_alipay",
            "ios_alipay",
        }
    model_payload = json.loads(model.read_text(encoding="utf-8"))
    assert model_payload["publication_safety"]["leakage_metadata"] == (
        "font_rendering_platform_proxy_weak"
    )


def test_test_only_weak_labels_do_not_relabel_training_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    training = _training_manifest(tmp_path / "test-only-weak", include_test=True)
    rows = [json.loads(line) for line in training.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["split"] == "test":
            row["label_source"] = "device_platform_proxy_font_rendering_weak_v1"
            row["device_prior_domain"] = row["font_domain"]
    training.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    code, stdout, stderr = _invoke(
        capsys,
        "fit",
        "--records",
        str(training),
        "--output",
        str(tmp_path / "test-only-weak.model.json"),
        "--skip-near-duplicate-audit",
    )

    assert code == 0
    assert stderr == ""
    fitted = json.loads(stdout)
    assert fitted["training_label_sources"] == ["synthetic_test_fixture"]
    assert fitted["test_label_sources"] == [
        "device_platform_proxy_font_rendering_weak_v1"
    ]
    assert fitted["label_evidence_status"] == "not_independently_assessed"
    assert fitted["publication_safety"]["leakage_metadata"] == "required_and_present"
    assert fitted["held_out_evaluation"]["status"] == (
        "platform_proxy_font_rendering_test_split"
    )
    assert fitted["held_out_evaluation"]["device_prior_used"] is False


def test_mixed_test_labels_apply_single_region_mode_only_to_proxy_documents(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _training_manifest(tmp_path / "mixed-test-labels", include_test=True)
    rows = [json.loads(line) for line in training.read_text(encoding="utf-8").splitlines()]
    weak_document_id = next(
        str(row["id"])
        for row in rows
        if row["split"] == "test"
    )
    for row in rows:
        if row["id"] == weak_document_id:
            row["label_source"] = "device_platform_weak_pseudo_v1"
            row["device_prior_domain"] = row["font_domain"]
    training.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    options_by_document: dict[str, dict[str, object]] = {}

    def fake_predict_document(
        model: object,
        document: object,
        **aggregation_options: object,
    ) -> SimpleNamespace:
        del model
        document_id = str(getattr(document, "document_id"))
        options_by_document[document_id] = dict(aggregation_options)
        reference = str(getattr(document, "font_domain"))
        return SimpleNamespace(
            dominant_domain=reference,
            decision="PASS",
            lines=tuple(
                SimpleNamespace(
                    include_in_consistency=True,
                    accepted=True,
                    label=reference,
                )
                for _ in range(3)
            ),
        )

    monkeypatch.setattr(cli_module, "predict_document", fake_predict_document)
    code, stdout, stderr = _invoke(
        capsys,
        "fit",
        "--records",
        str(training),
        "--output",
        str(tmp_path / "mixed-test-labels.model.json"),
        "--skip-near-duplicate-audit",
    )

    assert code == 0
    assert stderr == ""
    assert options_by_document[weak_document_id] == {
        "minimum_regions": 1,
        "minimum_roles": 1,
    }
    assert all(
        options == {}
        for document_id, options in options_by_document.items()
        if document_id != weak_document_id
    )
    held_out = json.loads(stdout)["held_out_evaluation"]
    assert held_out["status"] == (
        "mixed_including_platform_proxy_font_rendering_test_split"
    )
    assert held_out["aggregation_mode"] == "per_document_label_source"
    assert held_out["evidence_groups"]["platform_proxy_font_rendering"][
        "documents"
    ]["total"] == 1
    assert held_out["evidence_groups"]["other_labels"]["documents"]["total"] == 3


def test_legacy_device_proxy_labels_remain_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    training = _training_manifest(tmp_path / "legacy-proxy")
    rows = [json.loads(line) for line in training.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["label_source"] = "device_platform_weak_pseudo_v1"
    training.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    code, stdout, stderr = _invoke(
        capsys,
        "fit",
        "--records",
        str(training),
        "--output",
        str(tmp_path / "legacy-proxy.model.json"),
        "--skip-near-duplicate-audit",
    )

    assert code == 0
    assert stderr == ""
    fitted = json.loads(stdout)
    assert fitted["label_evidence_status"] == "font_rendering_platform_proxy_weak"
    assert fitted["publication_safety"]["leakage_metadata"] == (
        "font_rendering_platform_proxy_weak"
    )
    assert fitted["publication_prerequisites_recorded"] is False


def test_training_cli_enforces_leakage_metadata_and_content_group_split(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_root = tmp_path / "missing-metadata"
    relative = "images/line.png"
    _write_pattern(missing_root / relative, "ios_alipay", 1)
    missing = _write_manifest(
        missing_root,
        "missing.jsonl",
        [
            _document(
                "missing",
                split="train",
                domain="ios_alipay",
                regions=[_region(relative, "amount", "amount")],
                strict_metadata=False,
            )
        ],
    )

    code, stdout, stderr = _invoke(
        capsys,
        "validate",
        "--records",
        str(missing),
        "--skip-near-duplicate-audit",
    )
    assert code == 2
    assert stdout == ""
    assert "supervised publication requires content_group_id and source_image_sha256" in stderr

    code, stdout, stderr = _invoke(
        capsys,
        "validate",
        "--records",
        str(missing),
        "--skip-near-duplicate-audit",
        "--allow-incomplete-leakage-metadata",
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["validation"] == "passed"

    code, stdout, stderr = _invoke(
        capsys,
        "validate",
        "--records",
        str(missing),
        "--mode",
        "mixed",
        "--skip-near-duplicate-audit",
    )
    assert code == 2
    assert stdout == ""
    assert "supervised publication requires" in stderr

    leaking_root = tmp_path / "cross-split"
    records: list[dict[str, object]] = []
    for index, split in enumerate(("train", "calibration")):
        image = f"images/{split}.png"
        _write_pattern(leaking_root / image, "ios_alipay", 20 + index)
        records.append(
            _document(
                f"leak-{split}",
                split=split,
                domain="ios_alipay",
                content_group_id="same-content",
                regions=[_region(image, "amount", "amount")],
            )
        )
    leaking = _write_manifest(leaking_root, "leaking.jsonl", records)
    code, stdout, stderr = _invoke(
        capsys,
        "validate",
        "--records",
        str(leaking),
        "--skip-near-duplicate-audit",
    )
    assert code == 2
    assert stdout == ""
    assert "content_group_id 'same-content' crosses train/calibration splits" in stderr


def test_fit_requires_independent_calibration_and_fails_before_publication(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    training_only = _training_manifest(tmp_path / "training-only", include_calibration=False)
    model = tmp_path / "must-not-exist.json"

    code, stdout, stderr = _invoke(
        capsys,
        "fit",
        "--records",
        str(training_only),
        "--output",
        str(model),
        "--skip-near-duplicate-audit",
    )

    assert code == 2
    assert stdout == ""
    assert "independent calibration source groups are insufficient for domains" in stderr
    assert "android_alipay, ios_alipay" in stderr
    assert not model.exists()
