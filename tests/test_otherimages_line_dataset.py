from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.ocr_train import (
    GENERIC_TEXT_LINE_FIELD,
    GENERIC_TEXT_LINE_PREPROCESS,
    GENERIC_TEXT_LINE_PADDLE_COLOR_CONTRACT,
    LEGACY_RECEIPT_PREPROCESS,
    RecognizerConfig,
    _ReceiptOcrDataset,
    _collate_batch_worker,
    _preprocess_contract,
    _validation_due,
    build_train_parser,
    load_records,
    preprocess_image,
)
from transfer_receipt_ai.otherimages_line_dataset import (
    CONTRACT_NAME,
    LINE_DATASET_CONTRACT_KIND,
    LINE_DATASET_RECEIPT_KIND,
    MANIFEST_NAME,
    RECEIPT_NAME,
    LineDatasetContractError,
    _crop_line,
    materialize_otherimages_line_dataset,
)
from transfer_receipt_ai.otherimages_paddle_capture import (
    PaddleViewContract,
    _canonical_transform,
    _load_bound_upright_rgb,
    _pixel_sha256,
)
from transfer_receipt_ai.otherimages_paddle_teacher import (
    SCHEMA_VERSION,
    TEACHER_CONTRACT_KIND,
    TEACHER_RECEIPT_KIND,
    TEACHER_RECORD_KIND,
    _canonical_sha256,
    canonical_paddle_color_contract,
    canonical_view_contract,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _binding(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def _teacher_record(
    *,
    source_root: Path,
    source: Path,
    split: str,
    group_id: str,
    text: str,
    orientation_degrees: int,
    view_id: str = "original_rgb",
) -> dict[str, object]:
    source_rgb, observation = _load_bound_upright_rgb(
        source,
        expected_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    record_id = hashlib.sha256(f"{group_id}:{source.name}".encode("utf-8")).hexdigest()
    training = split == "train"
    view_contract = canonical_view_contract(view_id)
    transformed_rgb = _canonical_transform(
        source_rgb,
        PaddleViewContract(
            view_id=view_id,
            operations=tuple(str(value) for value in view_contract["operations"]),
        ),
    )
    transformed_height, transformed_width = transformed_rgb.shape[:2]
    transformed_quad = [
        [0.1 * (transformed_width - 1), 0.2 * (transformed_height - 1)],
        [0.9 * (transformed_width - 1), 0.2 * (transformed_height - 1)],
        [0.9 * (transformed_width - 1), 0.6 * (transformed_height - 1)],
        [0.1 * (transformed_width - 1), 0.6 * (transformed_height - 1)],
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": TEACHER_RECORD_KIND,
        "record_id": record_id,
        "group_id": group_id,
        "split": split,
        "split_use": "training" if training else f"heldout_{split}",
        "source_root": str(source_root.resolve()),
        "source_relative_path": source.name,
        "source_absolute_path": str(source.resolve()),
        "raw_sha256": observation["sha256"],
        "decoded_pixel_sha256": _pixel_sha256(source_rgb),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_normalization": "NFKC_then_collapse_line_whitespace_v1",
        "lines": [
            {
                "index": 0,
                "text": text,
                "confidence": 0.99,
                "orientation_degrees": orientation_degrees,
                "transformed_quad_pixels": transformed_quad,
                "quad_normalized": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.6], [0.1, 0.6]],
            }
        ],
        "label_source": "paddle_db_cls_rec_three_view_consensus",
        "paddle_color_contract": canonical_paddle_color_contract(),
        "consensus": {"agreement": "3_of_3", "chosen_geometry_view_id": view_id},
        "chosen_view": {
            "view_id": view_id,
            "view_contract_sha256": _canonical_sha256(view_contract),
            "transformed_pixel_sha256": _pixel_sha256(transformed_rgb),
            "source_width": int(source_rgb.shape[1]),
            "source_height": int(source_rgb.shape[0]),
            "transformed_width": transformed_width,
            "transformed_height": transformed_height,
            "coordinate_mapping": "full_frame_scale_source_normalized_identity_v1",
        },
        "training_eligible": training,
        "evaluation_only": not training,
        "held_out": not training,
        "automatic_teacher_validation": True,
        "manual_review_required": False,
    }


def _sealed_teacher(
    tmp_path: Path, *, view_id: str = "original_rgb", include_test: bool = True
) -> Path:
    source_root = tmp_path / "OtherImages"
    source_root.mkdir()
    names = ("train.png", "val.png", "test.png") if include_test else ("train.png", "val.png")
    for index, name in enumerate(names):
        pixels = np.full((80, 120, 3), 255, dtype=np.uint8)
        pixels[18:25, 18 + index * 4 : 70 + index * 4] = 20
        pixels[37:45, 25:35] = (30, 80, 160)
        Image.fromarray(pixels).save(source_root / name)

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    rows = [
        _teacher_record(
            source_root=source_root,
            source=source_root / "train.png",
            split="train",
            group_id="train-group",
            text="A1",
            orientation_degrees=0,
            view_id=view_id,
        ),
        _teacher_record(
            source_root=source_root,
            source=source_root / "val.png",
            split="val",
            group_id="val-group",
            text="A1",
            orientation_degrees=180,
            view_id=view_id,
        ),
    ]
    if include_test:
        rows.append(
            _teacher_record(
                source_root=source_root,
                source=source_root / "test.png",
                split="test",
                group_id="test-group",
                text="A1",
                orientation_degrees=0,
                view_id=view_id,
            )
        )
    _write_jsonl(teacher / "teacher_manifest.jsonl", rows)
    _write_jsonl(teacher / "reject_manifest.jsonl", [])
    artifacts = [_binding(teacher / name) for name in ("teacher_manifest.jsonl", "reject_manifest.jsonl")]
    configuration = {
        "minimum_line_confidence": 0.90,
        "paddle_color_contract": canonical_paddle_color_contract(),
    }
    counts = {
        "accepted_teacher_records": len(rows),
        "quarantined_records": 0,
    }
    inputs: dict[str, object] = {}
    split_use: dict[str, object] = {}
    closure_payload = {
        "schema_version": SCHEMA_VERSION,
        "inputs": inputs,
        "configuration": configuration,
        "counts": counts,
        "split_use": split_use,
        "artifacts": artifacts,
    }
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": TEACHER_CONTRACT_KIND,
        "sealed": True,
        "output_directory": str(teacher.resolve()),
        "inputs": inputs,
        "configuration": configuration,
        "counts": counts,
        "split_use": split_use,
        "artifacts": artifacts,
        "closure_sha256": _canonical_sha256(closure_payload),
        "training_authorization": False,
    }
    (teacher / "teacher.contract.json").write_bytes(_pretty_json_bytes(contract))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": TEACHER_RECEIPT_KIND,
        "sealed": True,
        "contract": _binding(teacher / "teacher.contract.json"),
        "contract_closure_sha256": contract["closure_sha256"],
    }
    (teacher / "teacher.receipt.json").write_bytes(_pretty_json_bytes(receipt))
    return teacher


def test_materializer_publishes_closed_line_dataset_and_ocr_loader_accepts_it(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path)
    output = tmp_path / "line-dataset"

    contract = materialize_otherimages_line_dataset(
        teacher_dir=teacher,
        output_dir=output,
        authorize_training=True,
    )

    assert contract["kind"] == LINE_DATASET_CONTRACT_KIND
    assert contract["sealed"] is True
    assert contract["training_authorization"] is True
    assert contract["crop_recipe"]["paddle_color_contract"] == GENERIC_TEXT_LINE_PADDLE_COLOR_CONTRACT
    assert contract["counts"]["by_split"] == {"train": 1, "val": 1, "test": 1}
    rows = [json.loads(line) for line in (output / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()]
    assert [row["field"] for row in rows] == [GENERIC_TEXT_LINE_FIELD] * 3
    assert {row["split"]: row["teacher_line_orientation_degrees"] for row in rows} == {
        "train": 0,
        "val": 180,
        "test": 0,
    }
    assert all((output / str(row["image"])).is_file() for row in rows)
    assert json.loads((output / "dataset.receipt.json").read_text(encoding="utf-8"))["kind"] == (
        LINE_DATASET_RECEIPT_KIND
    )

    loaded = load_records(
        output / MANIFEST_NAME,
        fields=(GENERIC_TEXT_LINE_FIELD,),
        dataset_root=output,
    )
    assert len(loaded) == 3


def test_materializer_requires_nonempty_train_val_and_test_splits(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path, include_test=False)

    with pytest.raises(LineDatasetContractError, match="train, val, and test.*missing=.*test"):
        materialize_otherimages_line_dataset(
            teacher_dir=teacher,
            output_dir=tmp_path / "line-dataset",
            authorize_training=True,
        )


def test_materializer_requires_explicit_training_authorization(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path)

    with pytest.raises(LineDatasetContractError, match="explicit authorize_training"):
        materialize_otherimages_line_dataset(
            teacher_dir=teacher,
            output_dir=tmp_path / "line-dataset",
        )


def test_materializer_rejects_sealed_legacy_bgr_teacher(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path)
    contract_path = teacher / "teacher.contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    configuration = dict(contract["configuration"])
    configuration.pop("paddle_color_contract")
    configuration["adapter_input_color_bridge"] = "opencv_rgb8_to_bgr8_v1"
    contract["configuration"] = configuration
    contract["closure_sha256"] = _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": contract["inputs"],
            "configuration": configuration,
            "counts": contract["counts"],
            "split_use": contract["split_use"],
            "artifacts": contract["artifacts"],
        }
    )
    contract_path.write_bytes(_pretty_json_bytes(contract))
    receipt_path = teacher / "teacher.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["contract"] = _binding(contract_path)
    receipt["contract_closure_sha256"] = contract["closure_sha256"]
    receipt_path.write_bytes(_pretty_json_bytes(receipt))

    with pytest.raises(LineDatasetContractError, match="canonical RGB byte-order contract"):
        materialize_otherimages_line_dataset(
            teacher_dir=teacher,
            output_dir=tmp_path / "legacy-bgr-output",
            authorize_training=True,
        )


def test_ocr_loader_rejects_resealed_legacy_bgr_line_dataset(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path)
    output = tmp_path / "line-dataset"
    materialize_otherimages_line_dataset(
        teacher_dir=teacher,
        output_dir=output,
        authorize_training=True,
    )
    contract_path = output / CONTRACT_NAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    crop_recipe = dict(contract["crop_recipe"])
    crop_recipe.pop("paddle_color_contract")
    crop_recipe["adapter_input_color_bridge"] = "opencv_rgb8_to_bgr8_v1"
    contract["crop_recipe"] = crop_recipe
    contract["closure_sha256"] = _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": contract["inputs"],
            "crop_recipe": crop_recipe,
            "counts": contract["counts"],
            "split_use": contract["split_use"],
            "artifacts": contract["artifacts"],
        }
    )
    contract_path.write_bytes(_pretty_json_bytes(contract))
    receipt_path = output / RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["contract"] = _binding(contract_path)
    receipt["contract_closure_sha256"] = contract["closure_sha256"]
    receipt_path.write_bytes(_pretty_json_bytes(receipt))

    with pytest.raises(ValueError, match="canonical RGB byte-order contract"):
        load_records(
            output / MANIFEST_NAME,
            fields=(GENERIC_TEXT_LINE_FIELD,),
            dataset_root=output,
        )


def test_generic_line_preprocess_is_fixed_cross_runtime_fixture(tmp_path: Path) -> None:
    rgb = np.asarray(
        [
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            [[12, 34, 56], [78, 90, 123], [200, 10, 30], [4, 250, 128], [33, 66, 99]],
            [[250, 128, 4], [17, 222, 19], [90, 45, 180], [1, 2, 3], [127, 127, 127]],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "generic-preprocess-fixture.png"
    Image.fromarray(rgb, mode="RGB").save(image_path)
    result = preprocess_image(
        image_path,
        config=RecognizerConfig(image_height=7, image_width=11),
        field=GENERIC_TEXT_LINE_FIELD,
    )
    expected_canvas = np.asarray(
        [
            [0, 14, 48, 83, 116, 150, 95, 40, 111, 214, 255],
            [4, 18, 51, 83, 111, 138, 98, 56, 113, 194, 227],
            [17, 29, 60, 86, 95, 104, 105, 105, 119, 137, 144],
            [30, 41, 68, 88, 79, 69, 112, 155, 126, 79, 60],
            [82, 87, 100, 107, 89, 71, 81, 92, 92, 90, 89],
            [133, 132, 132, 126, 99, 73, 52, 30, 59, 100, 117],
            [150, 148, 142, 132, 103, 74, 41, 8, 47, 104, 127],
        ],
        dtype=np.uint8,
    )

    assert result.shape == (1, 1, 7, 11)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(np.rint(result[0, 0] * 255).astype(np.uint8), expected_canvas)
    assert hashlib.sha256(expected_canvas.tobytes()).hexdigest() == (
        "1fb19240fc1f573408341333604958ebb16c5579161eaf157e04a96cba2a05a5"
    )
    assert hashlib.sha256(result.tobytes()).hexdigest() == (
        "ee1b0457871cc38344a994509057a99aebc96cc3820c678838240ecddf185c7f"
    )


def test_preprocess_contract_isolated_generic_without_changing_receipt_abi() -> None:
    assert _preprocess_contract((GENERIC_TEXT_LINE_FIELD,)) == GENERIC_TEXT_LINE_PREPROCESS
    assert _preprocess_contract(("amount",)) == LEGACY_RECEIPT_PREPROCESS


def test_training_dataset_and_collate_are_pickle_safe_for_windows_spawn() -> None:
    dataset = _ReceiptOcrDataset(
        [
            {
                "image_path": Path("fixture.png"),
                "text": "A",
                "field": GENERIC_TEXT_LINE_FIELD,
            }
        ],
        character_to_id={"A": 1},
        config=RecognizerConfig(),
    )

    assert pickle.loads(pickle.dumps(dataset))._records == dataset._records
    assert pickle.loads(pickle.dumps(_collate_batch_worker)) is _collate_batch_worker


def test_validation_cadence_preserves_default_and_forces_final_epoch() -> None:
    assert [_validation_due(epoch=epoch, epochs=5, validation_every=1) for epoch in range(1, 6)] == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert [_validation_due(epoch=epoch, epochs=5, validation_every=2) for epoch in range(1, 6)] == [
        False,
        True,
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="validation_every must be positive"):
        _validation_due(epoch=1, epochs=1, validation_every=0)


def test_training_parser_exposes_bounded_validation_and_progress_cadence() -> None:
    defaults = build_train_parser().parse_args(["--records", "records.jsonl", "--output", "run"])
    assert defaults.validation_every == 1
    assert defaults.train_progress_every == 0
    accelerated = build_train_parser().parse_args(
        [
            "--records",
            "generic_text_lines.jsonl",
            "--output",
            "run",
            "--fields",
            GENERIC_TEXT_LINE_FIELD,
            "--validation-every",
            "2",
            "--train-progress-every",
            "25",
        ]
    )
    assert accelerated.validation_every == 2
    assert accelerated.train_progress_every == 25


def test_line_crop_applies_teacher_bound_cls_180_to_pixels() -> None:
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[:20, :, 0] = 255
    pixels[20:, :, 2] = 255
    quad = [[0.0, 0.0], [79.0, 0.0], [79.0, 39.0], [0.0, 39.0]]

    upright, upright_transform = _crop_line(pixels, quad, orientation_degrees=0)
    rotated, rotated_transform = _crop_line(pixels, quad, orientation_degrees=180)

    np.testing.assert_array_equal(rotated, np.rot90(upright, k=2))
    assert upright_transform["paddle_cls_orientation_degrees"] == 0
    assert rotated_transform["paddle_cls_orientation_degrees"] == 180


def _paddle_210_reference_crop(
    transformed_rgb: np.ndarray,
    transformed_quad: list[list[float]],
    *,
    orientation_degrees: int,
) -> np.ndarray:
    points = np.asarray(transformed_quad, dtype=np.float32)
    crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    destination = np.float32([[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]])
    matrix = cv2.getPerspectiveTransform(points, destination)
    crop_rgb = cv2.warpPerspective(
        transformed_rgb,
        matrix,
        (crop_width, crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop_rgb.shape[0] / crop_rgb.shape[1] >= 1.5:
        crop_rgb = np.rot90(crop_rgb)
    if orientation_degrees == 180:
        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_180)
    return np.ascontiguousarray(crop_rgb)


@pytest.mark.parametrize("view_id", ("original_rgb", "grayscale_clahe", "upscale_sharpen"))
def test_line_crop_matches_pinned_paddle_210_rec_input_for_each_canonical_view(view_id: str) -> None:
    y, x = np.indices((37, 53), dtype=np.uint16)
    source = np.stack(
        ((x * 13 + y * 3) % 256, (x * 5 + y * 17) % 256, (x * 19 + y * 7) % 256),
        axis=2,
    ).astype(np.uint8)
    view_contract = canonical_view_contract(view_id)
    transformed = _canonical_transform(
        source,
        PaddleViewContract(view_id=view_id, operations=tuple(str(v) for v in view_contract["operations"])),
    )
    height, width = transformed.shape[:2]
    quad = [
        [0.10 * (width - 1), 0.20 * (height - 1)],
        [0.86 * (width - 1), 0.16 * (height - 1)],
        [0.82 * (width - 1), 0.70 * (height - 1)],
        [0.14 * (width - 1), 0.74 * (height - 1)],
    ]

    actual, _receipt = _crop_line(transformed, quad, orientation_degrees=0)
    expected = _paddle_210_reference_crop(transformed, quad, orientation_degrees=0)

    np.testing.assert_array_equal(actual, expected)
    if view_id != "original_rgb":
        original = _canonical_transform(
            source,
            PaddleViewContract(
                view_id="original_rgb",
                operations=tuple(str(v) for v in canonical_view_contract("original_rgb")["operations"]),
            ),
        )
        original_quad = [
            [0.10 * (original.shape[1] - 1), 0.20 * (original.shape[0] - 1)],
            [0.86 * (original.shape[1] - 1), 0.16 * (original.shape[0] - 1)],
            [0.82 * (original.shape[1] - 1), 0.70 * (original.shape[0] - 1)],
            [0.14 * (original.shape[1] - 1), 0.74 * (original.shape[0] - 1)],
        ]
        original_crop = _paddle_210_reference_crop(original, original_quad, orientation_degrees=0)
        assert hashlib.sha256(actual.tobytes()).hexdigest() != hashlib.sha256(original_crop.tobytes()).hexdigest()


def test_line_crop_matches_pinned_paddle_tall_rotation_then_cls_180() -> None:
    transformed = np.arange(96 * 48 * 3, dtype=np.uint16).reshape(96, 48, 3).astype(np.uint8)
    quad = [[12.0, 5.0], [28.0, 7.0], [30.0, 89.0], [10.0, 87.0]]

    actual, receipt = _crop_line(transformed, quad, orientation_degrees=180)
    expected = _paddle_210_reference_crop(transformed, quad, orientation_degrees=180)

    np.testing.assert_array_equal(actual, expected)
    assert receipt["rotation_applied_degrees_ccw"] == 90
    assert receipt["paddle_cls_orientation_degrees"] == 180


def test_materializer_fails_closed_without_cls_orientation_evidence(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path)
    manifest = teacher / "teacher_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    del rows[0]["lines"][0]["orientation_degrees"]
    _write_jsonl(manifest, rows)
    # Re-seal the modified source so the failure reaches the line-level gate.
    contract_path = teacher / "teacher.contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["artifacts"][0] = _binding(manifest)
    closure_payload = {
        "schema_version": SCHEMA_VERSION,
        "inputs": contract["inputs"],
        "configuration": contract["configuration"],
        "counts": contract["counts"],
        "split_use": contract["split_use"],
        "artifacts": contract["artifacts"],
    }
    contract["closure_sha256"] = _canonical_sha256(closure_payload)
    contract_path.write_bytes(_pretty_json_bytes(contract))
    receipt_path = teacher / "teacher.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["contract"] = _binding(contract_path)
    receipt["contract_closure_sha256"] = contract["closure_sha256"]
    receipt_path.write_bytes(_pretty_json_bytes(receipt))

    with pytest.raises(LineDatasetContractError, match="does not bind.*CLS orientation"):
        materialize_otherimages_line_dataset(
            teacher_dir=teacher,
            output_dir=tmp_path / "line-dataset",
            authorize_training=True,
        )


def test_generic_ocr_loader_rejects_crop_tampering_after_publication(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path)
    output = tmp_path / "line-dataset"
    materialize_otherimages_line_dataset(
        teacher_dir=teacher,
        output_dir=output,
        authorize_training=True,
    )
    row = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()[0])
    (output / row["image"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="crop bytes differ"):
        load_records(output / MANIFEST_NAME, fields=(GENERIC_TEXT_LINE_FIELD,), dataset_root=output)


def test_generic_ocr_loader_rejects_unsealed_or_renamed_manifest(tmp_path: Path) -> None:
    teacher = _sealed_teacher(tmp_path)
    output = tmp_path / "line-dataset"
    materialize_otherimages_line_dataset(
        teacher_dir=teacher,
        output_dir=output,
        authorize_training=True,
    )
    renamed = output / "arbitrary.jsonl"
    renamed.write_bytes((output / MANIFEST_NAME).read_bytes())

    with pytest.raises(ValueError, match="requires records basename"):
        load_records(renamed, fields=(GENERIC_TEXT_LINE_FIELD,), dataset_root=output)
