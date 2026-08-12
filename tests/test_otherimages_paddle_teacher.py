from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.otherimages_paddle_capture import (
    PaddleViewContract,
    _canonical_transform,
    _load_bound_upright_rgb,
    _pixel_sha256,
)
from transfer_receipt_ai import otherimages_paddle_teacher as teacher_module
from transfer_receipt_ai.otherimages_inventory import build_otherimages_inventory
from transfer_receipt_ai.otherimages_paddle_teacher import (
    ADAPTER_EVIDENCE_KIND,
    CAPTURE_KIND,
    PINNED_ADAPTER_IMPLEMENTATION,
    SCHEMA_VERSION,
    TEACHER_CONTRACT_KIND,
    TEACHER_RECEIPT_KIND,
    TeacherContractError,
    build_paddle_teacher_consensus,
    canonical_view_contract,
    canonical_paddle_color_contract,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _image_pixels(index: int = 0) -> np.ndarray:
    pixels = np.full((320, 200, 3), 255, dtype=np.uint8)
    pixels[10:20, 10 + index : 60 + index] = 0
    pixels[80:85, 20 : 140 + index] = 20
    pixels[150:155, 30 : 170 - index] = 40
    return pixels


def _build_inventory(tmp_path: Path, *, exact_duplicate: bool = False) -> tuple[Path, list[dict[str, object]]]:
    source = tmp_path / "OtherImages"
    source.mkdir()
    Image.fromarray(_image_pixels()).save(source / "receipt.png")
    if exact_duplicate:
        shutil.copyfile(source / "receipt.png", source / "receipt-copy.png")
    output = tmp_path / "inventory"
    build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=4)
    rows = _read_jsonl(output / "paddle_teacher_pending.jsonl")
    return output, rows


def _view_contract(view_id: str) -> dict[str, object]:
    return canonical_view_contract(view_id)


def _asset_binding(path: Path) -> dict[str, object]:
    if path.is_file():
        root = path.parent
        files = [path]
    else:
        root = path
        files = sorted(item for item in path.rglob("*") if item.is_file())
    bindings = [
        {
            "path": item.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            "size_bytes": item.stat().st_size,
        }
        for item in files
    ]
    return {
        "path": str(path.resolve()),
        "files": bindings,
        "closure_sha256": teacher_module._canonical_sha256(bindings),
        "size_bytes": sum(int(item["size_bytes"]) for item in bindings),
    }


def _adapter(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "paddle-model-fixture"
    root.mkdir(exist_ok=True)
    assets: dict[str, dict[str, object]] = {}
    for role in ("det", "cls", "rec"):
        role_root = root / role
        role_root.mkdir(exist_ok=True)
        model_file = role_root / "inference.pdiparams"
        if not model_file.exists():
            model_file.write_bytes(f"fixture-{role}".encode("ascii"))
        assets[role] = _asset_binding(role_root)
    dictionary = root / "chars.txt"
    if not dictionary.exists():
        dictionary.write_text("甲\n乙\n", encoding="utf-8", newline="\n")
    assets["dictionary"] = _asset_binding(dictionary)
    runtime_versions = {
        "paddleocr": "2.10.0",
        "paddlepaddle": "3.0.0-fixture",
        "albumentations": "1.4.10",
        "albucore": "0.0.13",
        "opencv": "4.10.0-fixture",
        "numpy": "1.26.4-fixture",
        "pillow": "10.3.0-fixture",
    }
    effective_args = {
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
    drop_score = 0.5
    model_payload = {
        "adapter_implementation": PINNED_ADAPTER_IMPLEMENTATION,
        "paddleocr_version": "2.10.0",
        "runtime_versions": runtime_versions,
        "effective_paddle_args": effective_args,
        "device": "cpu",
        "drop_score": drop_score,
        "assets": assets,
        "paddle_color_contract": canonical_paddle_color_contract(),
    }
    return {
        "kind": ADAPTER_EVIDENCE_KIND,
        "adapter_implementation": PINNED_ADAPTER_IMPLEMENTATION,
        "paddle_version": "2.10.0",
        "model_contract_sha256": teacher_module._canonical_sha256(model_payload),
        "drop_score": drop_score,
        "stages": {"db": True, "cls": True, "rec": True},
        "execution_device": "cpu",
        "runtime_versions": runtime_versions,
        "effective_paddle_args": effective_args,
        "model_assets": assets,
        "raw_db_lines_preserved_before_drop_filter": True,
        "paddle_color_contract": canonical_paddle_color_contract(),
    }


def _line(
    text: str,
    *,
    confidence: float = 0.99,
    quad: list[list[float]] | None = None,
    index: int = 0,
    transformed_quad: list[list[float]] | None = None,
) -> dict[str, object]:
    return {
        "index": index,
        "text": text,
        "confidence": confidence,
        "passes_drop_score": confidence >= 0.5,
        "orientation_degrees": 0,
        "transformed_quad_pixels": transformed_quad or [[20.0, 32.0], [159.0, 32.0], [159.0, 64.0], [20.0, 64.0]],
        "quad_normalized": quad or [[0.1, 0.1], [0.8, 0.1], [0.8, 0.2], [0.1, 0.2]],
    }


def _capture_row(
    inventory: dict[str, object],
    *,
    view_id: str,
    inventory_manifest_sha256: str,
    inventory_contract_sha256: str,
    adapter: dict[str, object],
    lines: list[dict[str, object]] | None = None,
    capture_state: str = "ok",
) -> dict[str, object]:
    view_contract = _view_contract(view_id)
    source_rgb, _observation = _load_bound_upright_rgb(
        Path(str(inventory["source_absolute_path"])),
        expected_raw_sha256=str(inventory["raw_sha256"]),
    )
    transformed_sha256 = _pixel_sha256(
        _canonical_transform(
            source_rgb,
            PaddleViewContract(
                view_id=view_id,
                operations=tuple(str(item) for item in view_contract["operations"]),
            ),
        )
    )
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": CAPTURE_KIND,
        "view_id": view_id,
        "view_contract": view_contract,
        "view_contract_sha256": teacher_module._canonical_sha256(view_contract),
        "adapter": adapter,
        "inventory_manifest_sha256": inventory_manifest_sha256,
        "inventory_contract_sha256": inventory_contract_sha256,
        "record_id": inventory["record_id"],
        "group_id": inventory["group_id"],
        "raw_sha256": inventory["raw_sha256"],
        "decoded_pixel_sha256": inventory["decoded_pixel_sha256"],
        "transform_receipt": {
            "view_id": view_id,
            "view_contract_sha256": teacher_module._canonical_sha256(view_contract),
            "source_decoded_pixel_sha256": inventory["decoded_pixel_sha256"],
            "transformed_pixel_sha256": transformed_sha256,
            "source_width": inventory["upright_width"],
            "source_height": inventory["upright_height"],
            "transformed_width": (
                inventory["upright_width"] * 2 if view_id == "upscale_sharpen" else inventory["upright_width"]
            ),
            "transformed_height": (
                inventory["upright_height"] * 2 if view_id == "upscale_sharpen" else inventory["upright_height"]
            ),
            "coordinate_mapping": "full_frame_scale_source_normalized_identity_v1",
        },
        "capture_state": capture_state,
        "lines": lines if capture_state == "ok" else None,
        "raw_detected_line_count": len(lines or []) if capture_state == "ok" else None,
        "recognition_attempted_line_count": len(lines or []) if capture_state == "ok" else None,
        "recognition_rejected_line_count": 0 if capture_state == "ok" else None,
    }
    if capture_state == "error":
        row["error"] = "fixture adapter failure"
    return row


def _write_three_views(
    tmp_path: Path,
    pending: list[dict[str, object]],
    *,
    texts: tuple[str, str, str] = ("ＡＢＣ  １２３", "ABC 123", "ＡBC　123"),
    confidences: tuple[float, float, float] = (0.99, 0.98, 0.97),
    quads: tuple[list[list[float]], list[list[float]], list[list[float]]] | None = None,
    states: tuple[str, str, str] = ("ok", "ok", "ok"),
    orientations: tuple[int, int, int] = (0, 0, 0),
) -> list[Path]:
    capture_directory = tmp_path / "captured-views"
    capture_directory.mkdir(exist_ok=True)
    default_quads = (
        [[0.10, 0.10], [0.80, 0.10], [0.80, 0.20], [0.10, 0.20]],
        [[0.11, 0.10], [0.81, 0.10], [0.81, 0.20], [0.11, 0.20]],
        [[0.09, 0.10], [0.79, 0.10], [0.79, 0.20], [0.09, 0.20]],
    )
    quads = quads or default_quads
    paths: list[Path] = []
    pending_only = [row for row in pending if row["teacher_state"] == "pending"]
    inventory_manifest_sha256 = hashlib.sha256(
        (tmp_path / "inventory" / "paddle_teacher_pending.jsonl").read_bytes()
    ).hexdigest()
    inventory_contract_sha256 = hashlib.sha256(
        (tmp_path / "inventory" / "inventory.contract.json").read_bytes()
    ).hexdigest()
    adapter = _adapter(tmp_path)
    for view_index, view_id in enumerate(("original_rgb", "grayscale_clahe", "upscale_sharpen")):
        rows = [
            _capture_row(
                inventory,
                view_id=view_id,
                inventory_manifest_sha256=inventory_manifest_sha256,
                inventory_contract_sha256=inventory_contract_sha256,
                adapter=adapter,
                capture_state=states[view_index],
                lines=[
                    _line(
                        texts[view_index],
                        confidence=confidences[view_index],
                        quad=quads[view_index],
                        transformed_quad=(
                            [
                                [point[0] * 399.0, point[1] * 639.0]
                                for point in quads[view_index]
                            ]
                            if view_index == 2
                            else [
                                [point[0] * 199.0, point[1] * 319.0]
                                for point in quads[view_index]
                            ]
                        ),
                    )
                    | {"orientation_degrees": orientations[view_index]}
                ],
            )
            for inventory in pending_only
        ]
        path = capture_directory / f"{view_id}.jsonl"
        _write_jsonl(path, rows)
        paths.append(path)
    return paths


def _build_teacher(
    tmp_path: Path,
    inventory: Path,
    views: list[Path],
    *,
    output_name: str = "teacher",
    **options: object,
) -> tuple[Path, dict[str, object]]:
    output = tmp_path / output_name
    contract = build_paddle_teacher_consensus(
        inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
        view_results=views,
        output_dir=output,
        **options,
    )
    return output, contract


def test_three_of_three_nfkc_consensus_publishes_sealed_teacher_and_hash_closure(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    source = Path(str(pending[0]["source_absolute_path"]))
    before = source.read_bytes()
    views = _write_three_views(tmp_path, pending)

    output, contract = _build_teacher(tmp_path, inventory, views)

    assert source.read_bytes() == before
    assert contract["kind"] == TEACHER_CONTRACT_KIND
    assert contract["sealed"] is True
    assert contract["training_authorization"] is False
    assert contract["ocr_execution_performed_by_this_module"] is False
    assert contract["counts"]["accepted_teacher_records"] == 1
    assert contract["counts"]["quarantined_records"] == 0
    assert {path.name for path in output.iterdir()} == {
        "teacher_manifest.jsonl",
        "reject_manifest.jsonl",
        "teacher.contract.json",
        "teacher.receipt.json",
    }
    teacher = _read_jsonl(output / "teacher_manifest.jsonl")[0]
    assert teacher["text"] == "ABC 123"
    assert teacher["text_sha256"] == hashlib.sha256(b"ABC 123").hexdigest()
    assert teacher["consensus"]["agreement"] == "3_of_3"
    assert teacher["consensus"]["dominant_text_votes"] == 3
    assert teacher["training_eligible"] is True
    assert teacher["lines"][0]["orientation_degrees"] == 0
    assert teacher["manual_review_required"] is False
    assert "independent human ground truth" in teacher["limitations"][0]
    assert _read_jsonl(output / "reject_manifest.jsonl") == []

    disk_contract = json.loads((output / "teacher.contract.json").read_text(encoding="utf-8"))
    assert disk_contract == contract
    receipt = json.loads((output / "teacher.receipt.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == TEACHER_RECEIPT_KIND
    contract_bytes = (output / "teacher.contract.json").read_bytes()
    assert receipt["contract"]["sha256"] == hashlib.sha256(contract_bytes).hexdigest()
    for binding in contract["artifacts"]:
        data = (output / binding["path"]).read_bytes()
        assert binding["sha256"] == hashlib.sha256(data).hexdigest()
        assert binding["size_bytes"] == len(data)
    closure_payload = {
        "schema_version": SCHEMA_VERSION,
        "inputs": contract["inputs"],
        "configuration": contract["configuration"],
        "counts": contract["counts"],
        "split_use": contract["split_use"],
        "artifacts": contract["artifacts"],
    }
    assert contract["closure_sha256"] == teacher_module._canonical_sha256(closure_payload)


def test_teacher_binds_cls_orientation_from_the_chosen_geometry_view(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending, orientations=(180, 0, 0))

    output, _contract = _build_teacher(tmp_path, inventory, views)

    teacher = _read_jsonl(output / "teacher_manifest.jsonl")[0]
    # original_rgb has the highest confidence and is the chosen geometry view.
    assert teacher["consensus"]["chosen_geometry_view_id"] == "original_rgb"
    assert teacher["lines"][0]["orientation_degrees"] == 180


def test_unique_two_of_three_dominant_text_is_accepted_without_using_conflicting_view(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(
        tmp_path,
        pending,
        texts=("商户甲", "商户甲", "商户乙"),
    )

    output, contract = _build_teacher(tmp_path, inventory, views)
    teacher = _read_jsonl(output / "teacher_manifest.jsonl")[0]

    assert contract["counts"]["accepted_teacher_records"] == 1
    assert teacher["text"] == "商户甲"
    assert teacher["consensus"]["agreement"] == "2_of_3"
    assert teacher["consensus"]["dominant_view_ids"] == ["grayscale_clahe", "original_rgb"]
    assert "upscale_sharpen" not in teacher["consensus"]["geometry_support_view_ids"]


@pytest.mark.parametrize(
    ("texts", "confidences", "states", "expected_reason"),
    [
        (("甲", "乙", "丙"), (0.99, 0.99, 0.99), ("ok", "ok", "ok"), "text_conflict"),
        (("甲", "甲", "乙"), (0.70, 0.75, 0.99), ("ok", "ok", "ok"), "low_confidence"),
        (("甲", "甲", "甲"), (0.99, 0.99, 0.99), ("error", "error", "ok"), "capture_error"),
    ],
)
def test_conflict_low_confidence_and_capture_failure_are_quarantined_without_guessed_label(
    tmp_path: Path,
    texts: tuple[str, str, str],
    confidences: tuple[float, float, float],
    states: tuple[str, str, str],
    expected_reason: str,
) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending, texts=texts, confidences=confidences, states=states)

    output, contract = _build_teacher(tmp_path, inventory, views)
    rejection = _read_jsonl(output / "reject_manifest.jsonl")[0]

    assert contract["counts"]["accepted_teacher_records"] == 0
    assert _read_jsonl(output / "teacher_manifest.jsonl") == []
    assert rejection["quarantine_reason"] == expected_reason
    assert rejection["training_eligible"] is False
    assert rejection["manual_review_required"] is False
    assert rejection["guessed_label_present"] is False
    assert "text" not in rejection


def test_dominant_text_with_incompatible_geometry_is_quarantined(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    quads = (
        [[0.05, 0.05], [0.40, 0.05], [0.40, 0.15], [0.05, 0.15]],
        [[0.60, 0.70], [0.95, 0.70], [0.95, 0.80], [0.60, 0.80]],
        [[0.10, 0.40], [0.50, 0.40], [0.50, 0.50], [0.10, 0.50]],
    )
    views = _write_three_views(tmp_path, pending, texts=("甲", "甲", "乙"), quads=quads)

    output, _contract = _build_teacher(tmp_path, inventory, views)
    rejection = _read_jsonl(output / "reject_manifest.jsonl")[0]

    assert rejection["quarantine_reason"] == "geometry_disagreement"
    assert rejection["dominant_text_votes"] == 2
    assert rejection["training_eligible"] is False


def test_polygon_iou_rejects_opposite_diagonal_quads_with_identical_bounding_boxes(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    rising = [[0.10, 0.15], [0.15, 0.10], [0.90, 0.85], [0.85, 0.90]]
    falling = [[0.85, 0.10], [0.90, 0.15], [0.15, 0.90], [0.10, 0.85]]
    assert teacher_module._quad_iou(rising, falling) < 0.10
    views = _write_three_views(
        tmp_path,
        pending,
        texts=("相同正文", "相同正文", "不同正文"),
        quads=(rising, falling, [[0.1, 0.4], [0.8, 0.4], [0.8, 0.5], [0.1, 0.5]]),
    )

    output, _contract = _build_teacher(tmp_path, inventory, views)
    rejection = _read_jsonl(output / "reject_manifest.jsonl")[0]

    assert rejection["quarantine_reason"] == "geometry_disagreement"
    assert rejection["training_eligible"] is False


def test_db_line_below_drop_score_cannot_be_silently_omitted_from_teacher_text(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending, texts=("甲", "甲", "甲"))
    for view in views:
        rows = _read_jsonl(view)
        rows[0]["lines"].append(
            _line(
                "可能遗漏的正文",
                confidence=0.40,
                index=1,
                quad=[[0.1, 0.3], [0.8, 0.3], [0.8, 0.4], [0.1, 0.4]],
            )
        )
        rows[0]["raw_detected_line_count"] = 2
        rows[0]["recognition_attempted_line_count"] = 2
        _write_jsonl(view, rows)

    output, contract = _build_teacher(tmp_path, inventory, views)
    rejection = _read_jsonl(output / "reject_manifest.jsonl")[0]

    assert contract["counts"]["accepted_teacher_records"] == 0
    assert rejection["quarantine_reason"] == "low_confidence"
    assert rejection["training_eligible"] is False


@pytest.mark.parametrize(
    ("text", "quad", "expected_reason"),
    [
        ("\u200b甲", [[0.1, 0.1], [0.8, 0.1], [0.8, 0.2], [0.1, 0.2]], "semantic_invalid"),
        ("\t甲", [[0.1, 0.1], [0.8, 0.1], [0.8, 0.2], [0.1, 0.2]], "semantic_invalid"),
        ("甲", [[0.1, 0.1], [0.8, 0.2], [0.1, 0.2], [0.8, 0.1]], "geometry_invalid"),
    ],
)
def test_semantic_or_quad_failure_is_record_local_quarantine(
    tmp_path: Path,
    text: str,
    quad: list[list[float]],
    expected_reason: str,
) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(
        tmp_path,
        pending,
        texts=(text, text, "不同"),
        quads=(quad, quad, [[0.1, 0.1], [0.8, 0.1], [0.8, 0.2], [0.1, 0.2]]),
    )

    output, _contract = _build_teacher(tmp_path, inventory, views)
    rejection = _read_jsonl(output / "reject_manifest.jsonl")[0]

    assert rejection["quarantine_reason"] == expected_reason
    assert rejection["training_eligible"] is False


def test_rejected_recognition_count_quarantines_instead_of_accepting_partial_document(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending, texts=("残缺正文", "残缺正文", "残缺正文"))
    for path in views:
        rows = _read_jsonl(path)
        rows[0]["recognition_rejected_line_count"] = 1
        _write_jsonl(path, rows)

    output, contract = _build_teacher(tmp_path, inventory, views)
    rejection = _read_jsonl(output / "reject_manifest.jsonl")[0]

    assert contract["counts"]["accepted_teacher_records"] == 0
    assert rejection["quarantine_reason"] == "incomplete_recognition_coverage"
    assert rejection["training_eligible"] is False


def test_duplicate_transformed_view_pixels_are_not_treated_as_independent_votes(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending, texts=("相同", "相同", "相同"))
    grayscale = _read_jsonl(views[1])
    upscale = _read_jsonl(views[2])
    upscale[0]["transform_receipt"]["transformed_pixel_sha256"] = grayscale[0]["transform_receipt"][
        "transformed_pixel_sha256"
    ]
    _write_jsonl(views[2], upscale)

    output = tmp_path / "must-not-publish"

    with pytest.raises(TeacherContractError, match="canonical recipe"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()


def test_canonical_view_recipe_cannot_be_spoofed_even_with_recomputed_sha(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    rows = _read_jsonl(views[1])
    rows[0]["view_contract"]["operations"][-1] = "unapproved_transform"
    spoofed_sha = teacher_module._canonical_sha256(rows[0]["view_contract"])
    rows[0]["view_contract_sha256"] = spoofed_sha
    rows[0]["transform_receipt"]["view_contract_sha256"] = spoofed_sha
    _write_jsonl(views[1], rows)
    output = tmp_path / "must-not-publish"

    with pytest.raises(TeacherContractError, match="exact canonical"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("option", "unsafe_value"),
    [
        ("minimum_line_confidence", 0.0),
        ("minimum_view_confidence", 0.0),
        ("minimum_geometry_iou", 0.0),
        ("minimum_quad_area", 1e-8),
    ],
)
def test_teacher_safety_thresholds_cannot_be_lowered(
    tmp_path: Path,
    option: str,
    unsafe_value: float,
) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    output = tmp_path / "must-not-publish"

    with pytest.raises(TeacherContractError, match="safety floor"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
            **{option: unsafe_value},
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("option", "unsafe_value"),
    [
        ("maximum_lines", teacher_module.DEFAULT_MAX_LINES + 1),
        ("maximum_line_characters", teacher_module.DEFAULT_MAX_LINE_CHARACTERS + 1),
        ("maximum_document_characters", teacher_module.DEFAULT_MAX_DOCUMENT_CHARACTERS + 1),
    ],
)
def test_teacher_semantic_safety_ceilings_cannot_be_raised(
    tmp_path: Path,
    option: str,
    unsafe_value: int,
) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    output = tmp_path / "must-not-publish"

    with pytest.raises(TeacherContractError, match="safety ceiling"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
            **{option: unsafe_value},
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("split", "training_eligible", "split_use"),
    [
        ("train", True, "training"),
        ("val", False, "heldout_val"),
        ("test", False, "heldout_test"),
    ],
)
def test_split_use_never_marks_val_or_test_pseudo_labels_training_eligible(
    tmp_path: Path,
    split: str,
    training_eligible: bool,
    split_use: str,
) -> None:
    _inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    capture_rows = []
    for path in views:
        row = _read_jsonl(path)[0]
        capture_rows.append((row, row["adapter"]))
    inventory_row = dict(pending[0])
    inventory_row["suggested_split"] = split

    accepted, rejected = teacher_module._consensus_record(
        inventory_row,
        capture_rows,
        minimum_line_confidence=teacher_module.DEFAULT_MIN_LINE_CONFIDENCE,
        minimum_view_confidence=teacher_module.DEFAULT_MIN_VIEW_CONFIDENCE,
        minimum_geometry_iou=teacher_module.DEFAULT_MIN_GEOMETRY_IOU,
        minimum_quad_area=teacher_module.DEFAULT_MIN_NORMALIZED_QUAD_AREA,
        maximum_lines=teacher_module.DEFAULT_MAX_LINES,
        maximum_line_characters=teacher_module.DEFAULT_MAX_LINE_CHARACTERS,
        maximum_document_characters=teacher_module.DEFAULT_MAX_DOCUMENT_CHARACTERS,
    )

    assert rejected is None
    assert accepted is not None
    assert accepted["training_eligible"] is training_eligible
    assert accepted["evaluation_only"] is (not training_eligible)
    assert accepted["held_out"] is (not training_eligible)
    assert accepted["split_use"] == split_use


def test_inventory_exact_duplicate_remains_quarantined_and_is_not_required_in_capture_views(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path, exact_duplicate=True)
    assert sorted(row["teacher_state"] for row in pending) == ["pending", "quarantine"]
    views = _write_three_views(tmp_path, pending, texts=("甲", "甲", "甲"))

    output, contract = _build_teacher(tmp_path, inventory, views)
    quarantine = _read_jsonl(output / "reject_manifest.jsonl")

    assert contract["counts"]["inventory_records"] == 2
    assert contract["counts"]["pending_records"] == 1
    assert contract["counts"]["accepted_teacher_records"] == 1
    assert contract["counts"]["quarantined_records"] == 1
    assert quarantine[0]["quarantine_reason"] == "inventory_quarantine"
    assert quarantine[0]["inventory_quarantine_reason"] == "decoded_pixel_duplicate"


@pytest.mark.parametrize("damage", ["missing_record", "raw_hash", "same_view_contract"])
def test_capture_set_or_binding_damage_is_fatal_and_publishes_nothing(tmp_path: Path, damage: str) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    if damage == "missing_record":
        _write_jsonl(views[0], [])
    elif damage == "raw_hash":
        rows = _read_jsonl(views[0])
        rows[0]["raw_sha256"] = "f" * 64
        _write_jsonl(views[0], rows)
    else:
        first = _read_jsonl(views[0])[0]
        rows = _read_jsonl(views[1])
        rows[0]["view_contract"] = first["view_contract"]
        rows[0]["view_contract_sha256"] = first["view_contract_sha256"]
        rows[0]["view_id"] = first["view_id"]
        _write_jsonl(views[1], rows)
    output = tmp_path / "must-not-publish"

    with pytest.raises(TeacherContractError):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()


def test_changed_source_image_is_fatal_before_output_creation(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    source = Path(str(pending[0]["source_absolute_path"]))
    source.write_bytes(source.read_bytes() + b"changed")
    output = tmp_path / "must-not-publish"

    with pytest.raises(TeacherContractError, match="raw SHA-256 differs"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()


def test_teacher_output_cannot_be_nested_in_inventory_source_root(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    output = Path(str(pending[0]["source_root"])) / "teacher-output"

    with pytest.raises(TeacherContractError, match="source_root"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()


def test_model_asset_tamper_is_fatal_before_teacher_publication(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    asset = tmp_path / "paddle-model-fixture" / "det" / "inference.pdiparams"
    asset.write_bytes(b"tampered-model")
    output = tmp_path / "must-not-publish"

    with pytest.raises(TeacherContractError, match="differs from adapter evidence"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()


def test_teacher_output_cannot_modify_complete_capture_publication_directory(tmp_path: Path) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    output = views[0].parent / "teacher"

    with pytest.raises(TeacherContractError, match="capture publication directories"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()


def test_teacher_rejects_output_parent_replacement_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    output_parent = tmp_path / "teacher-publication-parent"
    output_parent.mkdir()
    moved_parent = tmp_path / "teacher-publication-parent-before-swap"
    output = output_parent / "teacher"
    original_consensus = teacher_module._consensus_record
    replaced = False

    def replace_parent(*args: object, **kwargs: object) -> object:
        nonlocal replaced
        result = original_consensus(*args, **kwargs)
        if not replaced:
            output_parent.rename(moved_parent)
            output_parent.mkdir()
            replaced = True
        return result

    monkeypatch.setattr(teacher_module, "_consensus_record", replace_parent)
    with pytest.raises(TeacherContractError, match="output parent identity changed"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert not output.exists()
    assert not list(output_parent.glob(".teacher.teacher-building-*"))


def test_output_must_be_fresh_and_atomic_no_replace_race_does_not_clobber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    output = tmp_path / "raced-output"
    original_publish = teacher_module._rename_directory_no_replace

    def race(
        stage: Path,
        destination: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_stage_identity: tuple[int, int] | None = None,
    ) -> None:
        destination.mkdir()
        original_publish(
            stage,
            destination,
            expected_parent_identity=expected_parent_identity,
            expected_stage_identity=expected_stage_identity,
        )

    monkeypatch.setattr(teacher_module, "_rename_directory_no_replace", race)
    with pytest.raises(FileExistsError):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert output.is_dir()
    assert list(output.iterdir()) == []
    failed_stages = list(tmp_path.glob(".raced-output.teacher-building-*"))
    assert len(failed_stages) == 1
    assert (failed_stages[0] / "teacher.contract.json").is_file()


def test_teacher_publish_parent_swap_never_recursively_deletes_unrelated_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, pending = _build_inventory(tmp_path)
    views = _write_three_views(tmp_path, pending)
    output_parent = tmp_path / "teacher-publish-parent"
    output_parent.mkdir()
    moved_parent = tmp_path / "teacher-publish-parent-before-swap"
    output = output_parent / "teacher"
    original_publish = teacher_module._rename_directory_no_replace
    stage_name: str | None = None

    def swap_parent_then_publish(
        stage: Path,
        destination: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_stage_identity: tuple[int, int] | None = None,
    ) -> None:
        nonlocal stage_name
        stage_name = stage.name
        output_parent.rename(moved_parent)
        output_parent.mkdir()
        imposter = output_parent / stage.name
        imposter.mkdir()
        (imposter / "unrelated.txt").write_text("preserve me", encoding="utf-8")
        original_publish(
            stage,
            destination,
            expected_parent_identity=expected_parent_identity,
            expected_stage_identity=expected_stage_identity,
        )

    monkeypatch.setattr(teacher_module, "_rename_directory_no_replace", swap_parent_then_publish)
    with pytest.raises(RuntimeError, match="output parent identity changed"):
        build_paddle_teacher_consensus(
            inventory_manifest=inventory / "paddle_teacher_pending.jsonl",
            view_results=views,
            output_dir=output,
        )

    assert stage_name is not None
    assert (moved_parent / stage_name / "teacher.contract.json").is_file()
    assert (output_parent / stage_name / "unrelated.txt").read_text(encoding="utf-8") == "preserve me"
    assert not output.exists()


def test_checkout_wrapper_exposes_exactly_three_repeatable_view_inputs() -> None:
    wrapper = Path(__file__).parents[1] / "scripts" / "otherimages-paddle-teacher.py"
    result = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--inventory INVENTORY" in result.stdout
    assert "--view-result VIEW_RESULT" in result.stdout
    assert "exactly three" in result.stdout
