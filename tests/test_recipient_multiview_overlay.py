from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import ctypes
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.ocr_pseudolabels import _crop_digest
from transfer_receipt_ai.ocr_unified import UnifiedReaderConfig, load_records
from transfer_receipt_ai.ocr_unified_dataset import (
    KIND_V12,
    RECIPIENT_QUALITY_POLICY_VERSION,
    STATUS_CLASSES,
    V9_SLOT_ORDER,
)
from transfer_receipt_ai.pipeline import crop_field_with_margin
from transfer_receipt_ai.recipient_blind_manifest import build_blind_manifest
import transfer_receipt_ai.recipient_multiview_overlay as overlay_module
from transfer_receipt_ai.recipient_multiview_overlay import (
    FIXED2_ANALYSIS_CONTRACT_KIND,
    FIXED2_ANALYSIS_MARKER_NAME,
    FIXED2_ANALYSIS_PUBLICATION_AUTHORITY,
    FIXED2_CANONICAL_CONTRACT_NAME,
    FIXED2_SELECTOR_DOMAIN,
    FIXED2_CONTRACT_KIND,
    FIXED2_PUBLICATION_AUTHORITY,
    FIXED2_SELECTOR_MODE,
    FIXED2_VIEWS,
    materialize_fixed2_overlay as _formal_materialize_fixed2_overlay,
    verify_fixed2_overlay_contract as _formal_verify_fixed2_overlay_contract,
)
from transfer_receipt_ai.recipient_multiview_teacher_export import (
    export_recipient_multiview_teacher,
)


# Fixture materialization deliberately exercises the private analysis-only
# POSIX boundary.  Public/formal behavior is covered separately and must fail
# closed off Windows.
materialize_fixed2_overlay = (
    overlay_module._materialize_fixed2_overlay_analysis_test_only
)
verify_fixed2_overlay_contract = (
    overlay_module._verify_fixed2_overlay_analysis_test_only
)


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def _train_row(
    root: Path,
    dataset_root: Path,
    *,
    index: int,
    target: str,
) -> dict[str, object]:
    y, x = np.mgrid[:60, :100]
    pixels = np.stack(
        (
            (x * (index + 3) + y) % 256,
            (x + y * (index + 5)) % 256,
            (x * 7 + y * 11 + index * 17) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    source = root / "raw" / f"train-{index}.png"
    _write_png(source, pixels)
    result = root / "results" / f"train-{index}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "source": source.as_posix(),
                "geometry": {
                    "source_size": {"width": 100, "height": 60},
                    "rectified_size": {"width": 100, "height": 60},
                    "H_original_to_rectified": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    bbox = (20.0 + index, 20.0, 80.0, 40.0)
    crop_pixels = np.ascontiguousarray(crop_field_with_margin(pixels, bbox))
    crop_sha = _crop_digest(crop_pixels)
    crop = dataset_root / "images" / "recipient_field" / f"{crop_sha}.png"
    _write_png(crop, crop_pixels)
    return {
        "schema_version": 1,
        "id": f"receipt-train-{index}",
        "group_id": f"receipt:train:{index}",
        "split": "train",
        "source": source.as_posix(),
        "result_json": result.as_posix(),
        "label_source": "paddle_pseudo",
        "slots": {
            "recipient_field": {
                "image": crop.relative_to(dataset_root).as_posix(),
                "text": target,
                "recipient_visible_text": f"收款方 {target}",
                "recipient_label": "收款方",
                "recipient_value": target,
                "semantic_value": target,
                "recipient_quality_policy": RECIPIENT_QUALITY_POLICY_VERSION,
                "bbox_rectified": list(bbox),
                "crop_sha256": crop_sha,
            }
        },
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    dataset_root = tmp_path / "source-data" / "paddle-dataset"
    rows = [
        _train_row(tmp_path, dataset_root, index=0, target="商户甲"),
        _train_row(tmp_path, dataset_root, index=1, target="商户乙"),
    ]
    val_crops: dict[str, Path] = {}
    val_slots: dict[str, dict[str, object]] = {}
    val_fields = (
        ("amount", {"text": "12.34"}),
        ("time", {"text": "12:34"}),
        ("transfer_status", {"class_name": "success"}),
        ("payment_method_field", {"text": "银行卡(1234)"}),
        (
            "recipient_field",
            {
                "text": "验证商户",
                "recipient_visible_text": "收款方 验证商户",
                "recipient_label": "收款方",
                "recipient_value": "验证商户",
                "semantic_value": "验证商户",
                "recipient_quality_policy": RECIPIENT_QUALITY_POLICY_VERSION,
                "bbox_rectified": [20.0, 20.0, 80.0, 40.0],
            },
        ),
    )
    for index, (field, labels) in enumerate(val_fields):
        height = 18 + index
        width = 43 + index * 3
        val_pixels = np.full(
            (height, width, 3),
            247 - index * 17,
            dtype=np.uint8,
        )
        val_pixels[3 : height - 3, 7 : width - 7, :] = 20 + index * 11
        val_sha = _crop_digest(val_pixels)
        val_crop = dataset_root / "images" / field / f"{val_sha}.png"
        _write_png(val_crop, val_pixels)
        val_crops[field] = val_crop
        val_slots[field] = {
            "image": val_crop.relative_to(dataset_root).as_posix(),
            "crop_sha256": val_sha,
            **labels,
        }
    rows.append(
        {
            "schema_version": 1,
            "id": "receipt-val",
            "group_id": "receipt:val",
            "split": "val",
            "source": (tmp_path / "heldout-val-source.png").as_posix(),
            "result_json": (tmp_path / "heldout-val-result.json").as_posix(),
            "label_source": "paddle_pseudo",
            "slots": val_slots,
        }
    )
    rows.append(
        {
            "schema_version": 1,
            "id": "receipt-test",
            "group_id": "receipt:test",
            "split": "test",
            "source": (tmp_path / "never-open-test.png").as_posix(),
            "slots": {"recipient_field": {"text": "测试秘密", "crop_sha256": "a" * 64}},
        }
    )
    input_root = tmp_path / "source-data" / "manifest"
    input_root.mkdir(parents=True)
    full_records = input_root / "unified_fields.jsonl"
    full_records.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    source_contract = input_root / "dataset.contract.json"
    source_contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": KIND_V12,
                "dataset_root": dataset_root.as_posix(),
                "slot_order": list(V9_SLOT_ORDER),
                "status_classes": list(STATUS_CLASSES),
                "architecture": "v12",
                "recipient_target": "anchored_recipient_value_with_dedicated_high_resolution_value_view",
                "recipient_charset": sorted(set("商户甲乙验证")),
                "recipient_charset_sha256": "unused-by-loader",
                "recipient_charset_source": "train_only_anchored_recipient_value",
                "recipient_quality_policy": {
                    "version": RECIPIENT_QUALITY_POLICY_VERSION,
                    "requires_leading_recipient_label": True,
                    "target": "anchored_recipient_value",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    blind_root = tmp_path / "blind-evidence"
    blind_records = blind_root / "unified_fields.train-val.jsonl"
    blind_contract = blind_root / "blind.contract.json"
    build_blind_manifest(
        source=full_records,
        output=blind_records,
        contract=blind_contract,
    )
    multiview_root = tmp_path / "overlay-data" / "multiview"
    export_recipient_multiview_teacher(
        manifest=blind_records,
        dataset_contract=source_contract,
        dataset_root=dataset_root,
        output_dir=multiview_root,
    )
    return {
        "dataset_root": dataset_root,
        "full_records": full_records,
        "source_contract": source_contract,
        "blind_records": blind_records,
        "blind_contract": blind_contract,
        "multiview_root": multiview_root,
        "val_crop": val_crops["recipient_field"],
        "val_crops": val_crops,
        "test_source": tmp_path / "never-open-test.png",
    }


def _materialize(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    fixture = _fixture(tmp_path)
    contract = materialize_fixed2_overlay(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=tmp_path / "fixed2-output",
    )
    return fixture, contract


def _materialize_formal_windows_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str = "fixed2-formal-mock-output",
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    fixture = _fixture(tmp_path)
    created_names: list[str] = []

    def simulated_windows_atomic_create(
        parent: object,
        *,
        name: str,
    ) -> object:
        created_names.append(name)
        return overlay_module._create_stage_lease(
            parent,
            stage=parent.path / name,
        )

    monkeypatch.setattr(
        overlay_module,
        "_formal_windows_publication_available",
        lambda: True,
    )
    monkeypatch.setattr(
        overlay_module,
        "create_anchored_stage_directory",
        simulated_windows_atomic_create,
    )
    contract = _formal_materialize_fixed2_overlay(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=tmp_path / output_name,
    )
    return fixture, contract, created_names


def _rebuild_payload(
    contract: dict[str, object],
    *,
    selector: dict[str, object] | None = None,
    selected: list[dict[str, object]] | None = None,
    validation: list[dict[str, object]] | None = None,
    artifacts: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    return overlay_module._fixed2_contract_payload(
        contract_kind=str(contract["kind"]),
        publication_authority=str(contract["publication_authority"]),
        consumer_optimizer_input_ready=bool(
            contract["consumer_optimizer_input_ready"]
        ),
        publication_identity=(
            json.loads(json.dumps(contract["publication_identity"]))
            if contract["publication_identity"] is not None
            else None
        ),
        multiview_root=Path(str(contract["multiview_root"])),
        original_dataset_root=Path(str(contract["original_dataset_root"])),
        composite_records_path=Path(str(contract["composite_records_path"])),
        composite_dataset_contract_path=Path(
            str(contract["composite_dataset_contract"])
        ),
        composite_dataset_root=Path(str(contract["composite_dataset_root"])),
        selector_evidence=(
            selector
            if selector is not None
            else json.loads(json.dumps(contract["selector"]))
        ),
        selected_composite_bindings=(
            selected
            if selected is not None
            else json.loads(json.dumps(contract["selected_composite_bindings"]))
        ),
        validation_pixel_bindings=(
            validation
            if validation is not None
            else json.loads(json.dumps(contract["validation_pixel_bindings"]))
        ),
        artifacts=(
            artifacts
            if artifacts is not None
            else json.loads(json.dumps(contract["artifacts"]))
        ),
    )


def _mutate_png_pixels(path: Path) -> None:
    with Image.open(path) as opened:
        pixels = np.asarray(opened.convert("RGB")).copy()
    pixels[0, 0, 0] = np.uint8(int(pixels[0, 0, 0]) ^ 0xFF)
    _write_png(path, pixels)


def test_fixed2_materializer_is_schema_compatible_balanced_and_val_unchanged(
    tmp_path: Path,
) -> None:
    fixture, contract = _materialize(tmp_path)

    assert contract["kind"] == FIXED2_ANALYSIS_CONTRACT_KIND
    assert (
        contract["publication_authority"]
        == FIXED2_ANALYSIS_PUBLICATION_AUTHORITY
    )
    assert contract["analysis_only"] is True
    assert contract["production_route_authorized"] is False
    assert contract["test_opened"] is False
    assert contract["selected_views"] == list(FIXED2_VIEWS)
    assert contract["selector_mode"] == FIXED2_SELECTOR_MODE
    assert contract["train_multiplier"] == 1
    assert contract["val_unchanged"] is True
    assert contract["consumer_optimizer_input_ready"] is False
    assert contract["publication_identity"] is None
    assert contract["producer_optimizer_input_ready"] is False
    assert len(str(contract["overlay_subject_id"])) == 64
    assert len(str(contract["code_closure_sha256"])) == 64
    assert contract["validation_slot_count"] == 5
    assert contract["test_physical_files_opened"] == 0
    output_root = Path(str(contract["composite_records_path"])).parent
    assert {path.name for path in output_root.iterdir()} == {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
        FIXED2_ANALYSIS_MARKER_NAME,
    }
    assert not (output_root / FIXED2_CANONICAL_CONTRACT_NAME).exists()
    dataset_contract = json.loads(
        (output_root / "dataset.contract.json").read_text(encoding="utf-8")
    )
    assert dataset_contract["fixed2_overlay"] == {
        "kind": FIXED2_ANALYSIS_CONTRACT_KIND,
        "publication_authority": FIXED2_ANALYSIS_PUBLICATION_AUTHORITY,
        "analysis_only": True,
        "production_route_authorized": False,
        "consumer_optimizer_input_ready": False,
        "selected_views": list(FIXED2_VIEWS),
        "selector_mode": FIXED2_SELECTOR_MODE,
        "train_multiplier": 1,
        "val_unchanged": True,
        "source_dataset_contract": str(fixture["source_contract"]),
        "source_dataset_contract_sha256": overlay_module._sha256(
            fixture["source_contract"]
        ),
    }
    assert all(
        set(binding)
        == {
            "record_id",
            "group_id",
            "split",
            "target",
            "view",
            "pixel_sha256",
            "file_sha256",
        }
        for binding in contract["selected_composite_bindings"]
    )
    assert all(
        set(binding)
        == {
            "record_id",
            "field",
            "pixel_sha256",
            "file_sha256",
            "size_bytes",
            "width",
            "height",
        }
        for binding in contract["validation_pixel_bindings"]
    )
    assert {
        (binding["record_id"], binding["field"])
        for binding in contract["validation_pixel_bindings"]
    } == {
        ("receipt-val", field)
        for field in (
            "amount",
            "time",
            "transfer_status",
            "payment_method_field",
            "recipient_field",
        )
    }
    assert set(contract["semantic_artifact_names"]) == {
        "full_records",
        "blind_records",
        "blind_contract",
        "multiview_export_contract",
        "multiview_export_manifest",
        "source_dataset_contract",
    }
    assert contract["selector"]["selected_view_counts"] == {
        "standard": 1,
        "fixed_value": 1,
    }
    assert contract["selector"]["selector_domain"] == FIXED2_SELECTOR_DOMAIN
    assert contract["selector"]["bound_blind_manifest_sha256"] == hashlib.sha256(
        fixture["blind_records"].read_bytes()
    ).hexdigest()

    composite = Path(str(contract["composite_records_path"]))
    common_root = Path(str(contract["composite_dataset_root"]))
    config = UnifiedReaderConfig(
        architecture_version=12,
        recipient_input_height=128,
        recipient_input_width=1024,
    )
    loaded = load_records(composite, dataset_root=common_root, config=config)
    assert [row["id"] for row in loaded] == [
        "receipt-train-0",
        "receipt-train-1",
        "receipt-val",
    ]
    assert len(loaded) == 3
    val = next(row for row in loaded if row["split"] == "val")
    assert os.path.samefile(
        val["slots"]["recipient_field"]["image_path"],
        fixture["val_crop"],
    )
    selected_paths = {
        Path(str(row["slots"]["recipient_field"]["image_path"]))
        for row in loaded
        if row["split"] == "train"
    }
    assert len(selected_paths) == 2
    assert all(fixture["multiview_root"] in path.parents for path in selected_paths)


def test_fixed2_semantic_subject_is_independent_of_output_root(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = materialize_fixed2_overlay(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=tmp_path / "fixed2-output-a",
    )
    second = materialize_fixed2_overlay(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=tmp_path / "fixed2-output-b",
    )
    assert first["overlay_subject_id"] == second["overlay_subject_id"]
    assert first["artifacts"]["composite_dataset_contract"]["sha256"] != (
        second["artifacts"]["composite_dataset_contract"]["sha256"]
    )


def test_fixed2_semantic_subject_survives_export_tree_relocation_and_both_verify(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    relocated_multiview = tmp_path / "relocated-export" / "multiview"
    shutil.copytree(fixture["multiview_root"], relocated_multiview)
    first = materialize_fixed2_overlay(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=tmp_path / "fixed2-original-export",
    )
    second = materialize_fixed2_overlay(
        multiview_root=relocated_multiview,
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=tmp_path / "fixed2-relocated-export",
    )
    assert first["overlay_subject_id"] == second["overlay_subject_id"]
    assert first["artifacts"]["composite_records"]["sha256"] != (
        second["artifacts"]["composite_records"]["sha256"]
    )
    assert first["artifacts"]["multiview_export_manifest"]["sha256"] == (
        second["artifacts"]["multiview_export_manifest"]["sha256"]
    )
    for contract, multiview in (
        (first, fixture["multiview_root"]),
        (second, relocated_multiview),
    ):
        contract_path = Path(str(contract["composite_records_path"])).parent / (
            FIXED2_ANALYSIS_MARKER_NAME
        )
        assert verify_fixed2_overlay_contract(
            contract_path,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            multiview_root=multiview,
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        ) == contract


def test_fixed2_semantic_subject_changes_for_selected_pixels_target_group_or_selector(
    tmp_path: Path,
) -> None:
    _, contract = _materialize(tmp_path)
    original_subject = contract["overlay_subject_id"]
    for field, replacement in (
        ("pixel_sha256", "0" * 64),
        ("target", "不同商户"),
        ("group_id", "receipt:changed-group"),
    ):
        selected = json.loads(json.dumps(contract["selected_composite_bindings"]))
        if selected[0][field] == replacement:
            replacement = "1" * 64
        selected[0][field] = replacement
        assert _rebuild_payload(contract, selected=selected)["overlay_subject_id"] != (
            original_subject
        )
    selector = json.loads(json.dumps(contract["selector"]))
    selector["selector_domain"] = "receipt-recipient-fixed2-rank-v2"
    assert _rebuild_payload(contract, selector=selector)["overlay_subject_id"] != (
        original_subject
    )
    validation = json.loads(json.dumps(contract["validation_pixel_bindings"]))
    validation[0]["pixel_sha256"] = "0" * 64
    assert _rebuild_payload(contract, validation=validation)["overlay_subject_id"] != (
        original_subject
    )
    validation = json.loads(json.dumps(contract["validation_pixel_bindings"]))
    validation[0]["file_sha256"] = "0" * 64
    file_only = _rebuild_payload(contract, validation=validation)
    assert file_only["overlay_subject_id"] == original_subject
    assert file_only["validation_file_integrity_sha256"] != (
        contract["validation_file_integrity_sha256"]
    )


def test_fixed2_validation_file_encoding_is_integrity_only_not_semantic(
    tmp_path: Path,
) -> None:
    fixture, first = _materialize(tmp_path)
    val_crop = fixture["val_crops"]["amount"]
    with Image.open(val_crop) as opened:
        pixels = np.asarray(opened.convert("RGB")).copy()
    original_pixel_sha = _crop_digest(pixels)
    original_bytes = val_crop.read_bytes()
    Image.fromarray(pixels, mode="RGB").save(val_crop, format="PNG", compress_level=0)
    assert val_crop.read_bytes() != original_bytes
    with Image.open(val_crop) as opened:
        assert _crop_digest(np.asarray(opened.convert("RGB"))) == original_pixel_sha

    first_contract_path = Path(str(first["composite_records_path"])).parent / (
        FIXED2_ANALYSIS_MARKER_NAME
    )
    with pytest.raises(ValueError, match="recomputed source evidence"):
        verify_fixed2_overlay_contract(
            first_contract_path,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )
    second = materialize_fixed2_overlay(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=tmp_path / "fixed2-reencoded-val",
    )
    assert second["overlay_subject_id"] == first["overlay_subject_id"]
    assert second["validation_pixel_semantic_sha256"] == (
        first["validation_pixel_semantic_sha256"]
    )
    assert second["validation_file_integrity_sha256"] != (
        first["validation_file_integrity_sha256"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "recipient_field",
        "amount",
        "time",
        "payment_method_field",
        "transfer_status",
    ],
)
def test_fixed2_materialize_rejects_changed_validation_slot_pixels(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _fixture(tmp_path)
    _mutate_png_pixels(fixture["val_crops"][field])
    with pytest.raises(
        ValueError,
        match=rf"validation slot image receipt-val/{field} pixel hash changed",
    ):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=tmp_path / "fixed2-output",
        )


def test_fixed2_verify_rejects_validation_pixel_change_after_materialize(
    tmp_path: Path,
) -> None:
    fixture, contract = _materialize(tmp_path)
    _mutate_png_pixels(fixture["val_crops"]["payment_method_field"])
    contract_path = Path(str(contract["composite_records_path"])).parent / (
        FIXED2_ANALYSIS_MARKER_NAME
    )
    with pytest.raises(ValueError, match="validation slot image.*pixel hash changed"):
        verify_fixed2_overlay_contract(
            contract_path,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )


def test_fixed2_never_opens_missing_physical_test_artifacts(tmp_path: Path) -> None:
    fixture, contract = _materialize(tmp_path)
    assert not fixture["test_source"].exists()
    assert contract["test_opened"] is False
    assert contract["test_physical_files_opened"] == 0


def test_fixed2_cli_has_only_materialize_and_verify_surface() -> None:
    parser = overlay_module.build_parser()
    help_text = parser.format_help()
    assert "materialize-fixed2" in help_text
    assert "verify-fixed2" in help_text

    option_strings: set[str] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        for action in current._actions:
            option_strings.update(action.option_strings)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                pending.extend(choices.values())
    assert not any(
        option.startswith(("--train", "--test", "--onnx"))
        for option in option_strings
    )
    assert {"--multiview-root", "--blind-records", "--blind-contract"} <= (
        option_strings
    )


@pytest.mark.skipif(os.name == "nt", reason="non-Windows fail-closed boundary")
def test_fixed2_public_materializer_fails_closed_off_windows(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="requires Windows atomic") as caught:
        _formal_materialize_fixed2_overlay(
            multiview_root=tmp_path / "unused-multiview",
            full_records=tmp_path / "unused-full.jsonl",
            blind_records=tmp_path / "unused-blind.jsonl",
            blind_contract=tmp_path / "unused-blind.contract.json",
            original_dataset_root=tmp_path / "unused-dataset",
            output_root=tmp_path / "unused-output",
        )
    assert caught.value.errno == overlay_module.errno.ENOTSUP
    assert not (tmp_path / "unused-output").exists()


def test_fixed2_analysis_marker_and_naive_canonical_copy_both_fail_formal_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, contract = _materialize(tmp_path)
    output = Path(str(contract["composite_records_path"])).parent
    analysis_marker = output / FIXED2_ANALYSIS_MARKER_NAME
    monkeypatch.setattr(
        overlay_module,
        "_formal_windows_publication_available",
        lambda: True,
    )
    with pytest.raises(ValueError, match="canonical commit-marker filename"):
        _formal_verify_fixed2_overlay_contract(
            analysis_marker,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )

    canonical_copy = output / FIXED2_CANONICAL_CONTRACT_NAME
    shutil.copyfile(analysis_marker, canonical_copy)
    with pytest.raises(ValueError, match="fixed2 kind mismatch"):
        _formal_verify_fixed2_overlay_contract(
            canonical_copy,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )


@pytest.mark.skipif(os.name == "nt", reason="non-Windows fail-closed boundary")
def test_fixed2_public_verifier_fails_closed_before_any_posix_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_read(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("POSIX formal verifier read an artifact before fail-close")

    monkeypatch.setattr(overlay_module, "_existing", forbid_read)
    monkeypatch.setattr(overlay_module, "_strict_json", forbid_read)
    with pytest.raises(OSError, match="requires Windows publication authority") as caught:
        _formal_verify_fixed2_overlay_contract(
            tmp_path / FIXED2_CANONICAL_CONTRACT_NAME,
            blind_records=tmp_path / "unused-blind",
            blind_contract=tmp_path / "unused-contract",
        )
    assert caught.value.errno == overlay_module.errno.ENOTSUP


def test_fixed2_windows_public_materializer_mock_dispatch_produces_formal_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, contract, created_names = _materialize_formal_windows_mock(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "fixed2-formal-mock-output"
    assert len(created_names) == 1
    assert contract["kind"] == FIXED2_CONTRACT_KIND
    assert contract["publication_authority"] == FIXED2_PUBLICATION_AUTHORITY
    assert contract["consumer_optimizer_input_ready"] is True
    publication_identity = contract["publication_identity"]
    assert publication_identity["scheme"] == (
        "windows_native_directory_and_file_identity_v1"
    )
    assert set(publication_identity["pre_marker_files"]) == {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
    }
    assert {path.name for path in output.iterdir()} == {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
        FIXED2_CANONICAL_CONTRACT_NAME,
    }
    assert not (output / FIXED2_ANALYSIS_MARKER_NAME).exists()
    verified = _formal_verify_fixed2_overlay_contract(
        output / FIXED2_CANONICAL_CONTRACT_NAME,
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        expected_full_records=fixture["full_records"],
        original_dataset_root=fixture["dataset_root"],
    )
    assert verified == contract


@pytest.mark.skipif(os.name == "nt", reason="unpatched POSIX mint boundary")
def test_fixed2_posix_analysis_cannot_mint_or_verify_a_resealed_formal_profile(
    tmp_path: Path,
) -> None:
    fixture, analysis = _materialize(tmp_path)
    forged = json.loads(json.dumps(analysis))
    forged["kind"] = FIXED2_CONTRACT_KIND
    forged["publication_authority"] = FIXED2_PUBLICATION_AUTHORITY
    forged["consumer_optimizer_input_ready"] = True
    with pytest.raises(OSError, match="require Windows atomic") as mint_error:
        _rebuild_payload(forged)
    assert mint_error.value.errno == overlay_module.errno.ENOTSUP
    with pytest.raises(OSError, match="require Windows atomic"):
        overlay_module._composite_dataset_contract(
            contract_kind=FIXED2_CONTRACT_KIND,
            publication_authority=FIXED2_PUBLICATION_AUTHORITY,
            consumer_optimizer_input_ready=True,
            source_contract={},
            source_contract_path=tmp_path / "unused-source-contract",
            composite_records_path=tmp_path / "unused-records",
            composite_dataset_root=tmp_path,
            rows=(),
        )

    output = Path(str(analysis["composite_records_path"])).parent
    records = output / "unified_fields.train-val.fixed2.jsonl"
    dataset_contract = output / "dataset.contract.json"
    forged["publication_identity"] = overlay_module._fixed2_publication_identity(
        directory_identity=overlay_module._directory_identity(output),
        pre_marker_file_identities={
            records.name: overlay_module._file_identity(records),
            dataset_contract.name: overlay_module._file_identity(dataset_contract),
        },
    )
    unsigned = {
        key: value for key, value in forged.items() if key != "integrity_sha256"
    }
    forged["integrity_sha256"] = overlay_module._canonical_sha256(unsigned)
    canonical = output / FIXED2_CANONICAL_CONTRACT_NAME
    canonical.write_text(
        json.dumps(forged, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(OSError, match="requires Windows publication authority"):
        _formal_verify_fixed2_overlay_contract(
            canonical,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["directory_claim", "records_mtime", "dataset_contract_mtime"],
)
def test_fixed2_formal_verifier_reopens_bound_publication_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture, contract, _ = _materialize_formal_windows_mock(
        tmp_path,
        monkeypatch,
        output_name=f"formal-identity-{mutation}",
    )
    output = Path(str(contract["composite_records_path"])).parent
    marker = output / FIXED2_CANONICAL_CONTRACT_NAME
    if mutation == "directory_claim":
        changed = json.loads(marker.read_text(encoding="utf-8"))
        changed["publication_identity"]["directory"]["file_index"] += 1
        unsigned = {
            key: value
            for key, value in changed.items()
            if key != "integrity_sha256"
        }
        changed["integrity_sha256"] = overlay_module._canonical_sha256(unsigned)
        marker.write_text(
            json.dumps(changed, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
    else:
        target = output / (
            "unified_fields.train-val.fixed2.jsonl"
            if mutation == "records_mtime"
            else "dataset.contract.json"
        )
        information = target.stat()
        os.utime(
            target,
            ns=(information.st_atime_ns, information.st_mtime_ns + 1_000_000),
        )
    with pytest.raises(ValueError, match="recomputed source evidence"):
        _formal_verify_fixed2_overlay_contract(
            marker,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )


def test_fixed2_formal_contract_copy_to_another_directory_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, contract, _ = _materialize_formal_windows_mock(
        tmp_path,
        monkeypatch,
    )
    output = Path(str(contract["composite_records_path"])).parent
    copied = tmp_path / "copied-formal-output"
    shutil.copytree(output, copied)
    with pytest.raises(ValueError, match="not contained by the contract directory"):
        _formal_verify_fixed2_overlay_contract(
            copied / FIXED2_CANONICAL_CONTRACT_NAME,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )


def test_fixed2_cli_materialize_and_verify_emit_one_json_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "fixed2-cli-output"
    monkeypatch.setattr(
        overlay_module,
        "materialize_fixed2_overlay",
        overlay_module._materialize_fixed2_overlay_analysis_test_only,
    )
    monkeypatch.setattr(
        overlay_module,
        "verify_fixed2_overlay_contract",
        overlay_module._verify_fixed2_overlay_analysis_test_only,
    )
    overlay_module.main(
        [
            "materialize-fixed2",
            "--multiview-root",
            str(fixture["multiview_root"]),
            "--full-records",
            str(fixture["full_records"]),
            "--blind-records",
            str(fixture["blind_records"]),
            "--blind-contract",
            str(fixture["blind_contract"]),
            "--original-dataset-root",
            str(fixture["dataset_root"]),
            "--output-root",
            str(output),
        ]
    )
    materialize_output = capsys.readouterr()
    assert materialize_output.err == ""
    assert len(materialize_output.out.splitlines()) == 1
    materialized = json.loads(materialize_output.out)
    assert materialized["kind"] == FIXED2_ANALYSIS_CONTRACT_KIND

    overlay_module.main(
        [
            "verify-fixed2",
            "--contract-path",
            str(output / FIXED2_ANALYSIS_MARKER_NAME),
            "--full-records",
            str(fixture["full_records"]),
            "--blind-records",
            str(fixture["blind_records"]),
            "--blind-contract",
            str(fixture["blind_contract"]),
            "--original-dataset-root",
            str(fixture["dataset_root"]),
        ]
    )
    verify_output = capsys.readouterr()
    assert verify_output.err == ""
    assert len(verify_output.out.splitlines()) == 1
    assert json.loads(verify_output.out) == materialized


def test_fixed2_cli_failure_is_nonzero_and_emits_no_stdout(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not prior_pythonpath
        else str(source_root) + os.pathsep + prior_pythonpath
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "transfer_receipt_ai.recipient_multiview_overlay",
            "verify-fixed2",
            "--contract-path",
            str(missing / "fixed2_overlay.contract.json"),
            "--blind-records",
            str(missing / "blind.jsonl"),
            "--blind-contract",
            str(missing / "blind.contract.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""


def test_fixed2_verify_rebuilds_every_binding_and_supports_full_records_alias(
    tmp_path: Path,
) -> None:
    fixture, contract = _materialize(tmp_path)
    verified = verify_fixed2_overlay_contract(
        Path(str(contract["artifacts"]["composite_records"]["path"])).parent
        / FIXED2_ANALYSIS_MARKER_NAME,
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        full_records=fixture["full_records"],
        original_dataset_root=fixture["dataset_root"],
    )
    assert verified == contract
    assert set(verified["artifacts"]) == {
        "full_records",
        "blind_records",
        "blind_contract",
        "multiview_export_contract",
        "multiview_export_manifest",
        "source_dataset_contract",
        "composite_records",
        "composite_dataset_contract",
        "consumer_code",
        "producer_code",
        "geometry_helper_code",
        "ocr_helper_code",
        "pseudolabel_helper_code",
        "unified_dataset_helper_code",
        "pipeline_crop_helper_code",
        "status_crop_helper_code",
    }


def test_fixed2_verify_rejects_changed_composite_and_never_reuses_output(tmp_path: Path) -> None:
    fixture, contract = _materialize(tmp_path)
    composite = Path(str(contract["composite_records_path"]))
    composite.write_text(composite.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="composite records|SHA-256 changed"):
        verify_fixed2_overlay_contract(
            composite.parent / FIXED2_ANALYSIS_MARKER_NAME,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            multiview_root=fixture["multiview_root"],
            expected_full_records=fixture["full_records"],
        )
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=composite.parent,
        )


def test_fixed2_code_drift_does_not_change_semantic_subject_but_old_binding_fails(
    tmp_path: Path,
) -> None:
    fixture, contract = _materialize(tmp_path)
    altered_artifacts = json.loads(json.dumps(contract["artifacts"]))
    original_code_sha = altered_artifacts["consumer_code"]["sha256"]
    altered_artifacts["consumer_code"]["sha256"] = (
        "0" * 64 if original_code_sha != "0" * 64 else "1" * 64
    )
    rebuilt = _rebuild_payload(contract, artifacts=altered_artifacts)
    assert rebuilt["overlay_subject_id"] == contract["overlay_subject_id"]
    assert rebuilt["code_closure_sha256"] != contract["code_closure_sha256"]

    resealed = {
        **rebuilt,
        "integrity_sha256": overlay_module._canonical_sha256(rebuilt),
    }
    contract_path = Path(str(contract["composite_records_path"])).parent / (
        FIXED2_ANALYSIS_MARKER_NAME
    )
    contract_path.write_text(
        json.dumps(resealed, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="consumer_code SHA-256 changed"):
        verify_fixed2_overlay_contract(
            contract_path,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            expected_full_records=fixture["full_records"],
            original_dataset_root=fixture["dataset_root"],
        )


def test_fixed2_verify_rejects_reparse_composite_artifact(tmp_path: Path) -> None:
    fixture, contract = _materialize(tmp_path)
    composite = Path(str(contract["composite_records_path"]))
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(composite.read_bytes())
    composite.unlink()
    try:
        composite.symlink_to(replacement)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symlink|reparse"):
        verify_fixed2_overlay_contract(
            composite.parent / FIXED2_ANALYSIS_MARKER_NAME,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            multiview_root=fixture["multiview_root"],
            expected_full_records=fixture["full_records"],
        )


def _require_directory_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "symlink-target"
    link = tmp_path / "symlink-probe"
    target.mkdir()
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    link.unlink()


def test_fixed2_cleanup_never_follows_a_replaced_stage_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_directory_symlinks(tmp_path)
    fixture = _fixture(tmp_path)
    foreign = tmp_path / "foreign-stage-target"
    foreign.mkdir()
    sentinel_names = (
        "fixed2_overlay.contract.json",
        "dataset.contract.json",
        "unified_fields.train-val.fixed2.jsonl",
    )
    expected = {name: f"foreign:{name}".encode() for name in sentinel_names}
    for name, payload in expected.items():
        (foreign / name).write_bytes(payload)

    def replace_stage_then_fail(
        parent_lease: object,
        source_lease: object,
        *,
        source: Path,
        destination: Path,
    ) -> None:
        del parent_lease, source_lease, destination
        try:
            source.rename(source.with_name(f"{source.name}.owned"))
        except OSError as error:
            if os.name != "nt":
                raise
            raise RuntimeError("Windows stage lease blocked injected replacement") from error
        source.symlink_to(foreign, target_is_directory=True)
        raise OSError("injected publication failure after stage replacement")

    monkeypatch.setattr(
        overlay_module,
        "_rename_directory_no_replace_anchored",
        replace_stage_then_fail,
    )
    output = tmp_path / "fixed2-output"
    with pytest.raises(
        (OSError, RuntimeError),
        match="injected publication failure|stage lease blocked injected replacement",
    ):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert not os.path.lexists(output)
    assert {name: (foreign / name).read_bytes() for name in sentinel_names} == expected
    retained = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".fixed2-output")
        and path.is_dir()
        and not path.is_symlink()
    ]
    assert len(retained) == 1
    assert not (retained[0] / FIXED2_ANALYSIS_MARKER_NAME).exists()


def test_fixed2_failure_never_calls_name_based_unlink_or_rmdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "fixed2-output"
    deletion_calls: list[tuple[str, object]] = []

    def forbid_unlink(path: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        deletion_calls.append(("unlink", path))
        raise AssertionError("failure handling must never unlink by name")

    def forbid_rmdir(path: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        deletion_calls.append(("rmdir", path))
        raise AssertionError("failure handling must never rmdir by name")

    def fail_after_stage(
        checkpoint: str,
        *,
        parent: Path,
        stage: Path,
        output_root: Path,
    ) -> None:
        del parent, stage, output_root
        if checkpoint == "after_stage_creation":
            raise RuntimeError("injected failure before any publication file")

    monkeypatch.setattr(overlay_module.os, "unlink", forbid_unlink)
    monkeypatch.setattr(overlay_module.os, "rmdir", forbid_rmdir)
    monkeypatch.setattr(overlay_module, "_fixed2_publication_hook", fail_after_stage)
    with pytest.raises(RuntimeError, match="injected failure"):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert deletion_calls == []
    assert not os.path.lexists(output)
    retained = [path for path in tmp_path.iterdir() if path.name.startswith(".fixed2-output")]
    assert len(retained) == 1
    assert list(retained[0].iterdir()) == []


def test_fixed2_full_verifier_failure_is_quarantined_before_directory_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "fixed2-output"
    original_internal_verify = overlay_module._verify_fixed2_overlay_payload

    def fail_internal_verify(*args: object, **kwargs: object) -> dict[str, object]:
        if kwargs.get("actual_composite_records") is not None:
            raise ValueError("injected full verifier failure")
        return original_internal_verify(*args, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "_verify_fixed2_overlay_payload",
        fail_internal_verify,
    )
    with pytest.raises(ValueError, match="injected full verifier failure") as caught:
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert not os.path.lexists(output)
    retained = [path for path in tmp_path.iterdir() if path.name.startswith(".fixed2-output")]
    assert len(retained) == 1
    stage = retained[0]
    assert not (stage / FIXED2_ANALYSIS_MARKER_NAME).exists()
    assert {path.name for path in stage.iterdir()} == {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
    }
    assert "no files or directories were deleted" in str(
        caught.value.fixed2_quarantine
    )
    with pytest.raises(FileNotFoundError, match="analysis fixture marker"):
        verify_fixed2_overlay_contract(
            stage / FIXED2_ANALYSIS_MARKER_NAME,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            multiview_root=fixture["multiview_root"],
            expected_full_records=fixture["full_records"],
        )
    monkeypatch.setattr(
        overlay_module,
        "_verify_fixed2_overlay_payload",
        original_internal_verify,
    )
    with pytest.raises(FileExistsError, match="retained staging/failure evidence"):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )


def test_fixed2_postrename_failure_leaves_output_uncommitted_and_verify_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "fixed2-output"

    def fail_after_rename(
        checkpoint: str,
        *,
        parent: Path,
        stage: Path,
        output_root: Path,
    ) -> None:
        del parent, stage, output_root
        if checkpoint == "immediately_after_rename":
            raise RuntimeError("injected post-rename failure")

    monkeypatch.setattr(
        overlay_module,
        "_fixed2_publication_hook",
        fail_after_rename,
    )
    with pytest.raises(RuntimeError, match="injected post-rename failure") as caught:
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
    )
    assert output.is_dir()
    assert not (output / FIXED2_ANALYSIS_MARKER_NAME).exists()
    assert {path.name for path in output.iterdir()} == {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
    }
    assert "no files or directories were deleted" in str(
        caught.value.fixed2_quarantine
    )
    with pytest.raises(FileNotFoundError, match="analysis fixture marker"):
        verify_fixed2_overlay_contract(
            output / FIXED2_ANALYSIS_MARKER_NAME,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            multiview_root=fixture["multiview_root"],
            expected_full_records=fixture["full_records"],
        )


def test_public_fixed2_verifier_exposes_no_staged_publication_override() -> None:
    parameters = inspect.signature(_formal_verify_fixed2_overlay_contract).parameters
    assert set(parameters) == {
        "contract_path",
        "blind_records",
        "blind_contract",
        "multiview_root",
        "expected_full_records",
        "full_records",
        "original_dataset_root",
    }
    assert not any(name.startswith("_publication") for name in parameters)


def test_fixed2_partial_analysis_commit_is_retained_and_analysis_verify_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "fixed2-output"
    original_write = overlay_module._write_anchored_stage_file

    def fail_canonical_commit(
        stage: object,
        *,
        name: str,
        payload: bytes,
    ) -> object:
        if name == FIXED2_ANALYSIS_MARKER_NAME:
            original_write(stage, name=name, payload=b'{"partial":')
            raise OSError("injected canonical commit failure")
        return original_write(stage, name=name, payload=payload)

    monkeypatch.setattr(
        overlay_module,
        "_write_anchored_stage_file",
        fail_canonical_commit,
    )
    with pytest.raises(OSError, match="injected canonical commit failure") as caught:
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert output.is_dir()
    assert (output / FIXED2_ANALYSIS_MARKER_NAME).read_bytes() == b'{"partial":'
    assert "no files or directories were deleted" in str(
        caught.value.fixed2_quarantine
    )
    with pytest.raises(ValueError, match="Unable to read strict JSON"):
        verify_fixed2_overlay_contract(
            output / FIXED2_ANALYSIS_MARKER_NAME,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            multiview_root=fixture["multiview_root"],
            expected_full_records=fixture["full_records"],
        )


def test_fixed2_postrename_clone_never_receives_analysis_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "fixed2-output"
    owned_output = tmp_path / "fixed2-output-owned"
    replacement_blocked = False

    def replace_output_with_clone(
        checkpoint: str,
        *,
        parent: Path,
        stage: Path,
        output_root: Path,
    ) -> None:
        nonlocal replacement_blocked
        del parent, stage
        if checkpoint != "before_fixed2_contract_commit":
            return
        try:
            output_root.rename(owned_output)
        except OSError as error:
            if os.name != "nt":
                raise
            replacement_blocked = True
            raise RuntimeError("Windows stage lease blocked output clone") from error
        output_root.mkdir()
        for name in (
            "unified_fields.train-val.fixed2.jsonl",
            "dataset.contract.json",
        ):
            (output_root / name).write_bytes((owned_output / name).read_bytes())

    monkeypatch.setattr(
        overlay_module,
        "_fixed2_publication_use_hook",
        replace_output_with_clone,
    )
    with pytest.raises(
        (ValueError, RuntimeError),
        match="committed output entry identity changed|stage lease blocked output clone",
    ):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert output.is_dir()
    assert not (output / FIXED2_ANALYSIS_MARKER_NAME).exists()
    assert {path.name for path in output.iterdir()} == {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
    }
    with pytest.raises(FileNotFoundError, match="analysis fixture marker"):
        verify_fixed2_overlay_contract(
            output / FIXED2_ANALYSIS_MARKER_NAME,
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            multiview_root=fixture["multiview_root"],
            expected_full_records=fixture["full_records"],
        )
    if not replacement_blocked:
        assert (owned_output / FIXED2_ANALYSIS_MARKER_NAME).is_file()
        with pytest.raises(ValueError, match="not contained by the contract directory"):
            verify_fixed2_overlay_contract(
                owned_output / FIXED2_ANALYSIS_MARKER_NAME,
                blind_records=fixture["blind_records"],
                blind_contract=fixture["blind_contract"],
                multiview_root=fixture["multiview_root"],
                expected_full_records=fixture["full_records"],
            )


def test_windows_directory_lease_access_and_share_constants_are_fail_closed() -> None:
    assert overlay_module._WINDOWS_FILE_CREATE == 2
    assert overlay_module._WINDOWS_FILE_OPEN == 1
    assert overlay_module._WINDOWS_FILE_DIRECTORY_FILE == 0x00000001
    assert overlay_module._WINDOWS_FILE_OPEN_REPARSE_POINT == 0x00200000
    assert overlay_module._WINDOWS_DIRECTORY_LEASE_ACCESS & (
        overlay_module._WINDOWS_FILE_TRAVERSE
    )
    assert overlay_module._WINDOWS_DIRECTORY_LEASE_ACCESS & (
        overlay_module._WINDOWS_GENERIC_EXECUTE
    )
    assert overlay_module._WINDOWS_DIRECTORY_LEASE_SHARE == (
        overlay_module._WINDOWS_FILE_SHARE_READ
        | overlay_module._WINDOWS_FILE_SHARE_WRITE
    )
    assert not (
        overlay_module._WINDOWS_DIRECTORY_LEASE_SHARE
        & overlay_module._WINDOWS_FILE_SHARE_DELETE
    )
    assert not (
        overlay_module._WINDOWS_DIRECTORY_LEASE_ACCESS
        & overlay_module._WINDOWS_DELETE
    )
    assert (
        overlay_module._WINDOWS_DIRECTORY_LEASE_ACCESS
        | overlay_module._WINDOWS_DELETE
    ) & overlay_module._WINDOWS_DELETE


def test_windows_ntcreatefile_abi_uses_full_width_parent_handle_and_relative_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_handle = 0x1234_5678_9ABC
    returned_handle = 0x2345_6789_ABCD
    captured: dict[str, object] = {}

    class FakeNtCreateFile:
        argtypes: object = None
        restype: object = None

        def __call__(
            self,
            handle_pointer: object,
            desired_access: int,
            object_attributes_pointer: object,
            io_status_pointer: object,
            allocation_size: object,
            file_attributes: int,
            share_access: int,
            create_disposition: int,
            create_options: int,
            extended_attributes: object,
            extended_attributes_length: int,
        ) -> int:
            del io_status_pointer
            handle_pointer._obj.value = returned_handle  # type: ignore[attr-defined]
            attributes = object_attributes_pointer._obj  # type: ignore[attr-defined]
            unicode_name = attributes.object_name.contents
            captured.update(
                {
                    "desired_access": desired_access,
                    "root_directory": attributes.root_directory,
                    "object_attributes_length": attributes.length,
                    "object_attributes_size": ctypes.sizeof(type(attributes)),
                    "object_attributes_flags": attributes.attributes,
                    "name_length": unicode_name.length,
                    "name_maximum_length": unicode_name.maximum_length,
                    "name": ctypes.string_at(
                        unicode_name.buffer,
                        unicode_name.length,
                    ).decode("utf-16-le"),
                    "allocation_size": allocation_size,
                    "file_attributes": file_attributes,
                    "share_access": share_access,
                    "create_disposition": create_disposition,
                    "create_options": create_options,
                    "extended_attributes": extended_attributes,
                    "extended_attributes_length": extended_attributes_length,
                }
            )
            return 0

    fake_ntcreate = FakeNtCreateFile()

    class FakeNtdll:
        NtCreateFile = fake_ntcreate

    monkeypatch.setattr(
        overlay_module.ctypes,
        "WinDLL",
        lambda *args, **kwargs: FakeNtdll(),
        raising=False,
    )
    desired_access = (
        overlay_module._WINDOWS_DIRECTORY_LEASE_ACCESS
        | overlay_module._WINDOWS_DELETE
        | overlay_module._WINDOWS_SYNCHRONIZE
    )
    handle = overlay_module._windows_nt_directory_handle(
        parent_handle,
        name=".anchored-child",
        create_disposition=overlay_module._WINDOWS_FILE_CREATE,
        desired_access=desired_access,
        share_access=overlay_module._WINDOWS_DIRECTORY_LEASE_SHARE,
    )

    assert handle == returned_handle
    assert captured == {
        "desired_access": desired_access,
        "root_directory": parent_handle,
        "object_attributes_length": captured["object_attributes_size"],
        "object_attributes_size": captured["object_attributes_size"],
        "object_attributes_flags": 0x00000040,
        "name_length": len(".anchored-child".encode("utf-16-le")),
        "name_maximum_length": len(".anchored-child".encode("utf-16-le")) + 2,
        "name": ".anchored-child",
        "allocation_size": None,
        "file_attributes": 0,
        "share_access": overlay_module._WINDOWS_DIRECTORY_LEASE_SHARE,
        "create_disposition": overlay_module._WINDOWS_FILE_CREATE,
        "create_options": (
            overlay_module._WINDOWS_FILE_DIRECTORY_FILE
            | overlay_module._WINDOWS_FILE_OPEN_REPARSE_POINT
        ),
        "extended_attributes": None,
        "extended_attributes_length": 0,
    }
    assert fake_ntcreate.argtypes[0] == ctypes.POINTER(ctypes.c_void_p)
    assert fake_ntcreate.argtypes[1] == ctypes.c_uint32
    assert fake_ntcreate.argtypes[4] == ctypes.c_void_p
    assert fake_ntcreate.restype == ctypes.c_int32


def test_windows_open_directory_lease_rejects_swap_restored_path_by_handle_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "parent"
    path.mkdir()
    expected = (7, 11, 0x10)
    swapped = (7, 12, 0x10)
    closed: list[int] = []
    checkpoints: list[str] = []

    monkeypatch.setattr(overlay_module.os, "name", "nt")
    monkeypatch.setattr(
        overlay_module,
        "_windows_open_path_directory_handle",
        lambda *args, **kwargs: 501,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_directory_handle_identity",
        lambda handle: swapped if handle == 501 else expected,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_close_handle",
        lambda handle: closed.append(handle),
    )

    def restored_path_must_not_be_restat(
        candidate: Path,
    ) -> tuple[int, int, int]:
        del candidate
        raise AssertionError("Windows lease identity must come from the opened handle")

    def record_open(
        checkpoint: str,
        *,
        path: Path,
        handle: int | None,
    ) -> None:
        del path, handle
        checkpoints.append(checkpoint)

    monkeypatch.setattr(
        overlay_module,
        "_directory_identity",
        restored_path_must_not_be_restat,
    )
    monkeypatch.setattr(overlay_module, "_directory_lease_open_hook", record_open)

    with pytest.raises(
        ValueError,
        match="directory handle identity does not match expected identity",
    ):
        overlay_module._open_directory_lease(path, expected=expected)
    assert checkpoints == ["before_windows_open", "after_windows_open_before_identity"]
    assert closed == [501]


def test_windows_atomic_stage_creation_binds_created_handle_to_parent_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_identity = (13, 21, 0x10)
    child_identity = (13, 22, 0x10)
    parent = overlay_module._DirectoryLease(
        path=tmp_path,
        identity=parent_identity,
        windows_handle=601,
        windows_identity=parent_identity,
    )
    calls: list[dict[str, int | str]] = []
    closed: list[int] = []
    checkpoints: list[tuple[str, int | None]] = []

    def fake_nt_open(
        parent_handle: int,
        *,
        name: str,
        create_disposition: int,
        desired_access: int,
        share_access: int,
    ) -> int:
        calls.append(
            {
                "parent_handle": parent_handle,
                "name": name,
                "create_disposition": create_disposition,
                "desired_access": desired_access,
                "share_access": share_access,
            }
        )
        if create_disposition == overlay_module._WINDOWS_FILE_CREATE:
            return 602
        assert create_disposition == overlay_module._WINDOWS_FILE_OPEN
        return 603

    def fake_identity(handle: int) -> tuple[int, int, int]:
        if handle == 601:
            return parent_identity
        assert handle in {602, 603}
        return child_identity

    def record_creation(
        checkpoint: str,
        *,
        parent: object,
        name: str,
        handle: int | None,
    ) -> None:
        del parent
        assert name == ".stage-atomic"
        checkpoints.append((checkpoint, handle))

    monkeypatch.setattr(overlay_module, "_windows_nt_directory_handle", fake_nt_open)
    monkeypatch.setattr(
        overlay_module,
        "_windows_directory_handle_identity",
        fake_identity,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_close_handle",
        lambda handle: closed.append(handle),
    )
    monkeypatch.setattr(
        overlay_module,
        "_stage_directory_creation_hook",
        record_creation,
    )

    lease = overlay_module.create_anchored_stage_directory(
        parent,
        name=".stage-atomic",
    )
    assert lease.path == tmp_path / ".stage-atomic"
    assert lease.identity == child_identity
    assert lease.windows_identity == child_identity
    assert lease.windows_handle == 602
    assert lease.windows_rename_capable is True
    assert checkpoints == [
        ("after_windows_atomic_create_before_parent_relative_reopen", 602)
    ]
    assert [call["parent_handle"] for call in calls] == [601, 601]
    assert [call["name"] for call in calls] == [".stage-atomic", ".stage-atomic"]
    assert calls[0]["create_disposition"] == overlay_module._WINDOWS_FILE_CREATE
    assert calls[0]["desired_access"] & overlay_module._WINDOWS_DELETE
    assert calls[0]["desired_access"] & overlay_module._WINDOWS_SYNCHRONIZE
    assert calls[0]["share_access"] == overlay_module._WINDOWS_DIRECTORY_LEASE_SHARE
    assert not (
        calls[0]["share_access"] & overlay_module._WINDOWS_FILE_SHARE_DELETE
    )
    assert calls[1]["create_disposition"] == overlay_module._WINDOWS_FILE_OPEN
    assert calls[1]["desired_access"] == overlay_module._WINDOWS_FILE_READ_ATTRIBUTES
    assert calls[1]["share_access"] & overlay_module._WINDOWS_FILE_SHARE_DELETE
    assert closed == [603]
    lease.close()
    assert closed == [603, 602]


def test_windows_anchored_rename_reports_complete_file_rename_info_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_identity = (13, 41, 0x10)
    stage_identity = (13, 42, 0x10)
    parent = overlay_module._DirectoryLease(
        path=tmp_path,
        identity=parent_identity,
        windows_handle=611,
        windows_identity=parent_identity,
    )
    source = tmp_path / ".stage-atomic"
    destination = tmp_path / "published-output"
    stage = overlay_module._DirectoryLease(
        path=source,
        identity=stage_identity,
        windows_handle=612,
        windows_rename_capable=True,
        windows_identity=stage_identity,
    )
    calls: list[tuple[ctypes.c_void_p, int, bytes, int]] = []

    class _FakeSetFileInformationByHandle:
        argtypes: object = None
        restype: object = None

        def __call__(
            self,
            handle: ctypes.c_void_p,
            information_class: int,
            information: object,
            size: int,
        ) -> int:
            calls.append(
                (handle, information_class, ctypes.string_at(information, size), size)
            )
            return 1

    class _FakeKernel32:
        SetFileInformationByHandle = _FakeSetFileInformationByHandle()

    monkeypatch.setattr(
        overlay_module,
        "_windows_directory_handle_identity",
        lambda handle: {611: parent_identity, 612: stage_identity}[handle],
    )
    monkeypatch.setattr(
        overlay_module.ctypes,
        "WinDLL",
        lambda *args, **kwargs: _FakeKernel32(),
        raising=False,
    )

    overlay_module._rename_directory_no_replace_anchored(
        parent,
        stage,
        source=source,
        destination=destination,
    )

    assert len(calls) == 1
    handle, information_class, raw, reported_size = calls[0]
    assert handle.value == 612
    assert information_class == 3  # FileRenameInfo

    class _FileRenameInfoPrefix(ctypes.Structure):
        _fields_ = (
            ("flags", ctypes.c_uint32),
            ("root_directory", ctypes.c_void_p),
            ("file_name_length", ctypes.c_uint32),
            ("file_name", ctypes.c_uint16 * 1),
        )

    encoded_name = destination.name.encode("utf-16-le")
    assert reported_size == ctypes.sizeof(_FileRenameInfoPrefix) + len(encoded_name)
    information = _FileRenameInfoPrefix.from_buffer_copy(raw)
    assert information.flags == 0  # ReplaceIfExists == FALSE: no clobber.
    assert information.root_directory == 611
    assert information.file_name_length == len(encoded_name)
    name_offset = _FileRenameInfoPrefix.file_name.offset
    assert raw[name_offset : name_offset + len(encoded_name)] == encoded_name


def test_windows_atomic_stage_creation_rejects_postcreate_entry_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_identity = (17, 31, 0x10)
    created_identity = (17, 32, 0x10)
    substituted_identity = (17, 33, 0x10)
    parent = overlay_module._DirectoryLease(
        path=tmp_path,
        identity=parent_identity,
        windows_handle=701,
        windows_identity=parent_identity,
    )
    closed: list[int] = []
    dispositions: list[int] = []
    replacement_injected = False

    def fake_nt_open(
        parent_handle: int,
        *,
        name: str,
        create_disposition: int,
        desired_access: int,
        share_access: int,
    ) -> int:
        nonlocal replacement_injected
        del parent_handle, name, desired_access, share_access
        dispositions.append(create_disposition)
        if create_disposition == overlay_module._WINDOWS_FILE_CREATE:
            return 702
        assert replacement_injected is True
        return 703

    def fake_identity(handle: int) -> tuple[int, int, int]:
        return {
            701: parent_identity,
            702: created_identity,
            703: substituted_identity,
        }[handle]

    def inject_replacement_after_create(
        checkpoint: str,
        *,
        parent: object,
        name: str,
        handle: int | None,
    ) -> None:
        nonlocal replacement_injected
        del parent
        assert checkpoint == "after_windows_atomic_create_before_parent_relative_reopen"
        assert name == ".stage-swapped"
        assert handle == 702
        replacement_injected = True

    monkeypatch.setattr(overlay_module, "_windows_nt_directory_handle", fake_nt_open)
    monkeypatch.setattr(
        overlay_module,
        "_windows_directory_handle_identity",
        fake_identity,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_close_handle",
        lambda handle: closed.append(handle),
    )
    monkeypatch.setattr(
        overlay_module,
        "_stage_directory_creation_hook",
        inject_replacement_after_create,
    )

    with pytest.raises(
        ValueError,
        match="created stage handle is not the child entry bound",
    ):
        overlay_module.create_anchored_stage_directory(parent, name=".stage-swapped")
    assert dispositions == [
        overlay_module._WINDOWS_FILE_CREATE,
        overlay_module._WINDOWS_FILE_OPEN,
    ]
    assert replacement_injected is True
    assert closed == [703, 702]


@pytest.mark.skipif(os.name == "nt", reason="POSIX compatibility boundary")
def test_posix_stage_helper_rejects_postcreate_preopen_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = tmp_path / "publication-parent"
    parent_path.mkdir()
    parent_identity = overlay_module._directory_identity(parent_path)
    parent = overlay_module._open_directory_lease(
        parent_path,
        expected=parent_identity,
    )
    stage = parent_path / ".stage-race"
    owned = parent_path / ".stage-race-owned"
    checkpoints: list[str] = []

    def replace_after_mkdir(
        checkpoint: str,
        *,
        parent: object,
        name: str,
        handle: int | None,
    ) -> None:
        del parent, handle
        checkpoints.append(checkpoint)
        assert name == stage.name
        os.rename(
            stage.name,
            owned.name,
            src_dir_fd=parent_path_fd,
            dst_dir_fd=parent_path_fd,
        )
        os.mkdir(stage.name, mode=0o700, dir_fd=parent_path_fd)

    assert parent.posix_fd is not None
    parent_path_fd = parent.posix_fd
    monkeypatch.setattr(
        overlay_module,
        "_stage_directory_creation_hook",
        replace_after_mkdir,
    )
    try:
        with pytest.raises(
            ValueError,
            match="changed between creation and lease acquisition",
        ):
            overlay_module._create_stage_lease(parent, stage=stage)
    finally:
        parent.close()
    assert checkpoints == ["after_posix_mkdir_before_open"]
    assert owned.is_dir()
    assert stage.is_dir()


def test_fixed2_publication_is_atomic_no_clobber_against_a_late_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "fixed2-output"
    original_rename = overlay_module._rename_directory_no_replace_anchored

    def install_competing_output(
        parent_lease: object,
        source_lease: object,
        *,
        source: Path,
        destination: Path,
    ) -> None:
        destination.mkdir()
        (destination / "foreign.txt").write_bytes(b"must-survive")
        original_rename(
            parent_lease,
            source_lease,
            source=source,
            destination=destination,
        )

    monkeypatch.setattr(
        overlay_module,
        "_rename_directory_no_replace_anchored",
        install_competing_output,
    )
    with pytest.raises(FileExistsError):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert (output / "foreign.txt").read_bytes() == b"must-survive"
    assert not (output / "fixed2_overlay.contract.json").exists()


def test_fixed2_publication_rejects_a_replaced_stage_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_directory_symlinks(tmp_path)
    fixture = _fixture(tmp_path)
    foreign = tmp_path / "foreign-published-target"
    foreign.mkdir()
    (foreign / "foreign.txt").write_bytes(b"must-survive")
    output = tmp_path / "fixed2-output"
    original_rename = overlay_module._rename_directory_no_replace_anchored

    def publish_replaced_stage(
        parent_lease: object,
        source_lease: object,
        *,
        source: Path,
        destination: Path,
    ) -> None:
        try:
            source.rename(source.with_name(f"{source.name}.owned"))
        except OSError as error:
            if os.name != "nt":
                raise
            raise RuntimeError("Windows stage lease blocked injected replacement") from error
        source.symlink_to(foreign, target_is_directory=True)
        original_rename(
            parent_lease,
            source_lease,
            source=source,
            destination=destination,
        )

    monkeypatch.setattr(
        overlay_module,
        "_rename_directory_no_replace_anchored",
        publish_replaced_stage,
    )
    with pytest.raises(
        (ValueError, RuntimeError),
        match="published output entry identity changed|stage lease blocked injected replacement",
    ):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    if os.name == "nt":
        assert not os.path.lexists(output)
    else:
        assert output.is_symlink()
    assert (foreign / "foreign.txt").read_bytes() == b"must-survive"


@pytest.mark.parametrize("replacement_kind", ["regular", "symlink"])
def test_fixed2_publication_rejects_output_parent_replacement_after_stage_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    if replacement_kind == "symlink":
        _require_directory_symlinks(tmp_path)
    fixture = _fixture(tmp_path)
    publication_parent = tmp_path / "publication-parent"
    publication_parent.mkdir()
    owned_parent = tmp_path / "publication-parent-owned"
    foreign = tmp_path / "foreign-parent-target"
    if replacement_kind == "symlink":
        foreign.mkdir()
    sentinel_parent = foreign if replacement_kind == "symlink" else publication_parent
    triggered = False
    replacement_blocked = False

    def replace_parent(
        checkpoint: str,
        *,
        parent: Path,
        stage: Path,
        output_root: Path,
    ) -> None:
        nonlocal triggered, replacement_blocked
        del stage, output_root
        if triggered or checkpoint != "after_stage_creation":
            return
        triggered = True
        try:
            parent.rename(owned_parent)
        except OSError as error:
            if os.name != "nt":
                raise
            replacement_blocked = True
            raise RuntimeError("Windows parent lease blocked injected replacement") from error
        if replacement_kind == "regular":
            parent.mkdir()
        else:
            parent.symlink_to(foreign, target_is_directory=True)
        (sentinel_parent / "foreign.txt").write_bytes(b"must-survive")

    monkeypatch.setattr(overlay_module, "_fixed2_publication_hook", replace_parent)
    output = publication_parent / "fixed2-output"
    with pytest.raises(
        (ValueError, RuntimeError),
        match=(
            "output parent identity changed at after_stage_creation"
            "|parent lease blocked injected replacement"
        ),
    ):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert triggered is True
    if replacement_blocked:
        assert os.name == "nt"
        assert not owned_parent.exists()
        assert publication_parent.is_dir()
        retained = list(publication_parent.iterdir())
        assert len(retained) == 1
        assert retained[0].name.startswith(".fixed2-output")
        assert not (retained[0] / "fixed2_overlay.contract.json").exists()
        return
    assert (sentinel_parent / "foreign.txt").read_bytes() == b"must-survive"
    assert not os.path.lexists(sentinel_parent / "fixed2-output")
    assert not any(path.name.startswith(".fixed2-output") for path in sentinel_parent.iterdir())
    retained = list(owned_parent.iterdir())
    assert len(retained) == 1
    assert retained[0].name.startswith(".fixed2-output")
    assert not (retained[0] / "fixed2_overlay.contract.json").exists()


@pytest.mark.parametrize(
    ("replacement_kind", "use_checkpoint"),
    [
        ("regular", "before_composite_records_write"),
        ("symlink", "immediately_before_rename"),
    ],
)
def test_fixed2_anchored_publication_never_mutates_a_parent_replaced_after_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
    use_checkpoint: str,
) -> None:
    if replacement_kind == "symlink":
        _require_directory_symlinks(tmp_path)
    fixture = _fixture(tmp_path)
    publication_parent = tmp_path / "publication-parent"
    publication_parent.mkdir()
    owned_parent = tmp_path / "publication-parent-owned"
    foreign = tmp_path / "foreign-parent-target"
    foreign.mkdir()
    (foreign / "foreign.txt").write_bytes(b"must-survive")
    triggered = False
    replacement_blocked = False

    def replace_parent_after_check(
        checkpoint: str,
        *,
        parent: Path,
        stage: Path,
        output_root: Path,
    ) -> None:
        nonlocal triggered, replacement_blocked
        del stage, output_root
        if triggered or checkpoint != use_checkpoint:
            return
        triggered = True
        try:
            parent.rename(owned_parent)
        except OSError as error:
            if os.name != "nt":
                raise
            replacement_blocked = True
            raise RuntimeError("Windows parent lease blocked injected replacement") from error
        if replacement_kind == "regular":
            parent.mkdir()
            (parent / "foreign.txt").write_bytes(b"must-survive")
        else:
            parent.symlink_to(foreign, target_is_directory=True)

    monkeypatch.setattr(
        overlay_module,
        "_fixed2_publication_use_hook",
        replace_parent_after_check,
    )
    output = publication_parent / "fixed2-output"
    with pytest.raises(
        (ValueError, RuntimeError),
        match="output parent identity changed|parent lease blocked injected replacement",
    ):
        materialize_fixed2_overlay(
            multiview_root=fixture["multiview_root"],
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    assert triggered is True
    assert list(foreign.iterdir()) == [foreign / "foreign.txt"]
    assert (foreign / "foreign.txt").read_bytes() == b"must-survive"
    if replacement_blocked:
        assert os.name == "nt"
        assert not owned_parent.exists()
        assert publication_parent.is_dir()
        retained = list(publication_parent.iterdir())
        assert len(retained) == 1
        assert retained[0].name.startswith(".fixed2-output")
        assert not (retained[0] / "fixed2_overlay.contract.json").exists()
        return
    assert owned_parent.is_dir()
    retained = list(owned_parent.iterdir())
    assert len(retained) == 1
    if use_checkpoint == "immediately_before_rename":
        assert retained[0].name == "fixed2-output"
    else:
        assert retained[0].name.startswith(".fixed2-output")
    assert not (retained[0] / "fixed2_overlay.contract.json").exists()
    if replacement_kind == "regular":
        assert publication_parent.is_dir()
        assert list(publication_parent.iterdir()) == [publication_parent / "foreign.txt"]
        assert (publication_parent / "foreign.txt").read_bytes() == b"must-survive"
    else:
        assert publication_parent.is_symlink()
        assert list(publication_parent.iterdir()) == [publication_parent / "foreign.txt"]
    assert not os.path.lexists(output)
    assert not any(
        path.name.startswith(".fixed2-output") for path in publication_parent.iterdir()
    )


def test_fixed2_publication_checks_parent_identity_at_every_mutation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    publication_parent = tmp_path / "publication-parent"
    publication_parent.mkdir()
    observed: list[str] = []

    def record_checkpoint(
        checkpoint: str,
        *,
        parent: Path,
        stage: Path,
        output_root: Path,
    ) -> None:
        del parent, stage, output_root
        observed.append(checkpoint)

    monkeypatch.setattr(
        overlay_module,
        "_fixed2_publication_hook",
        record_checkpoint,
    )
    materialize_fixed2_overlay(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=publication_parent / "fixed2-output",
    )
    assert observed == [
        "before_stage_creation",
        "after_stage_creation",
        "before_composite_records_write",
        "before_composite_records_snapshot",
        "after_composite_records_snapshot",
        "before_composite_dataset_contract_write",
        "before_composite_dataset_contract_snapshot",
        "after_composite_dataset_contract_snapshot",
        "before_fixed2_contract_seal",
        "after_fixed2_contract_seal",
        "before_stage_snapshot",
        "after_stage_snapshot",
        "before_final_verify",
        "after_final_verify",
        "immediately_before_rename",
        "immediately_after_rename",
        "after_published_output_snapshot",
        "before_fixed2_contract_commit",
        "after_fixed2_contract_commit",
    ]
