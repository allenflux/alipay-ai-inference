"""Materialize sealed OtherImages Paddle teacher lines as CTC crops.

The three-view teacher intentionally stores geometry in the EXIF-upright
source coordinate system and does not write derivative images.  This module is
the narrow, auditable bridge from that sealed evidence to an independent
generic-line recognizer dataset.  It never changes the teacher publication,
never assigns receipt-field semantics, and preserves the inventory group split
on every emitted line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .ocr_train import GENERIC_TEXT_LINE_FIELD
from .otherimages_inventory import _bind_stage_identity, _rename_directory_no_replace
from .otherimages_paddle_capture import (
    PaddleViewContract,
    _canonical_transform,
    _load_bound_upright_rgb,
    _pixel_sha256,
)
from .otherimages_paddle_teacher import (
    SCHEMA_VERSION,
    TEACHER_CONTRACT_KIND,
    TEACHER_RECEIPT_KIND,
    TEACHER_RECORD_KIND,
    TeacherContractError,
    _bind_output_parent,
    _canonical_sha256,
    canonical_paddle_color_contract,
    canonical_view_contract,
    _load_json_object_bytes,
    _load_jsonl_bytes,
    _normalise_text,
    _parse_quad,
    _paths_overlap,
    _read_bound_file,
    _require_no_reparse_ancestors,
    _verify_observation,
    _verify_output_parent,
)


LINE_RECORD_KIND = "otherimages_generic_text_line_record_v1"
LINE_DATASET_CONTRACT_KIND = "otherimages_generic_text_line_dataset_contract_v1"
LINE_DATASET_RECEIPT_KIND = "otherimages_generic_text_line_dataset_receipt_v1"
MANIFEST_NAME = "generic_text_lines.jsonl"
CONTRACT_NAME = "dataset.contract.json"
RECEIPT_NAME = "dataset.receipt.json"
_TEACHER_FILES = {
    "teacher_manifest.jsonl",
    "reject_manifest.jsonl",
    "teacher.contract.json",
    "teacher.receipt.json",
}
_SHA256_LENGTH = 64


class LineDatasetContractError(TeacherContractError):
    """Raised when teacher closure or line-dataset publication is invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LineDatasetContractError(f"{description} must be a lowercase SHA-256")
    return value


def _require_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise LineDatasetContractError(f"{description} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LineDatasetContractError(f"{description} must be a non-negative integer")
    return value


def _binding(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to is not None else path.name,
        "sha256": _sha256_bytes(data),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def _binding_matches(observation: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    return (
        expected.get("sha256") == observation.get("sha256")
        and expected.get("size_bytes") == observation.get("size_bytes")
        and expected.get("line_count") == observation.get("line_count")
    )


def _artifact(contract: Mapping[str, object], name: str) -> Mapping[str, object]:
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, list):
        raise LineDatasetContractError("teacher contract artifacts must be an array")
    matches = [value for value in artifacts if isinstance(value, Mapping) and value.get("path") == name]
    if len(matches) != 1:
        raise LineDatasetContractError(f"teacher contract must bind exactly one {name!r} artifact")
    return matches[0]


def _load_teacher_publication(
    teacher_dir: Path,
) -> tuple[
    Path,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    _require_no_reparse_ancestors(teacher_dir)
    root = teacher_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    members = {path.name for path in root.iterdir()}
    if members != _TEACHER_FILES or any(not path.is_file() for path in root.iterdir()):
        raise LineDatasetContractError(
            f"sealed teacher directory membership differs: expected={sorted(_TEACHER_FILES)}, "
            f"observed={sorted(members)}"
        )

    observations: list[dict[str, object]] = []
    snapshots: dict[str, bytes] = {}
    by_name: dict[str, dict[str, object]] = {}
    for name in sorted(_TEACHER_FILES):
        data, observation = _read_bound_file(root / name, description=f"sealed teacher {name}")
        snapshots[name] = data
        observations.append(observation)
        by_name[name] = observation

    contract = _load_json_object_bytes(snapshots["teacher.contract.json"], source=str(root / "teacher.contract.json"))
    receipt = _load_json_object_bytes(snapshots["teacher.receipt.json"], source=str(root / "teacher.receipt.json"))
    accepted = _load_jsonl_bytes(snapshots["teacher_manifest.jsonl"], source=str(root / "teacher_manifest.jsonl"))
    rejected = _load_jsonl_bytes(snapshots["reject_manifest.jsonl"], source=str(root / "reject_manifest.jsonl"))

    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != TEACHER_CONTRACT_KIND
        or contract.get("sealed") is not True
    ):
        raise LineDatasetContractError("unsupported or unsealed teacher contract")
    contracted_output = Path(_require_string(contract.get("output_directory"), description="teacher output_directory"))
    if contracted_output.expanduser().resolve(strict=True) != root:
        raise LineDatasetContractError("teacher contract output_directory does not bind the supplied directory")
    if contract.get("training_authorization") is not False:
        raise LineDatasetContractError("source teacher contract must retain training_authorization=false")
    configuration = contract.get("configuration")
    if (
        not isinstance(configuration, Mapping)
        or configuration.get("paddle_color_contract") != canonical_paddle_color_contract()
    ):
        raise LineDatasetContractError(
            "source teacher does not bind the canonical RGB byte-order contract"
        )

    for name in ("teacher_manifest.jsonl", "reject_manifest.jsonl"):
        if not _binding_matches(by_name[name], _artifact(contract, name)):
            raise LineDatasetContractError(f"teacher contract artifact binding changed: {name}")
    if len(contract.get("artifacts", [])) != 2:
        raise LineDatasetContractError("teacher contract must bind exactly the accepted and rejected manifests")

    closure_payload = {
        "schema_version": SCHEMA_VERSION,
        "inputs": contract.get("inputs"),
        "configuration": contract.get("configuration"),
        "counts": contract.get("counts"),
        "split_use": contract.get("split_use"),
        "artifacts": contract.get("artifacts"),
    }
    if contract.get("closure_sha256") != _canonical_sha256(closure_payload):
        raise LineDatasetContractError("teacher contract closure SHA-256 is invalid")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("kind") != TEACHER_RECEIPT_KIND
        or receipt.get("sealed") is not True
        or receipt.get("contract_closure_sha256") != contract.get("closure_sha256")
    ):
        raise LineDatasetContractError("teacher receipt does not bind the sealed contract closure")
    receipt_contract = receipt.get("contract")
    if not isinstance(receipt_contract, Mapping) or not _binding_matches(
        by_name["teacher.contract.json"], receipt_contract
    ) or receipt_contract.get("path") != "teacher.contract.json":
        raise LineDatasetContractError("teacher receipt contract binding changed")

    counts = contract.get("counts")
    if not isinstance(counts, Mapping):
        raise LineDatasetContractError("teacher contract counts must be an object")
    if _require_nonnegative_int(
        counts.get("accepted_teacher_records"), description="accepted_teacher_records"
    ) != len(accepted):
        raise LineDatasetContractError("teacher accepted record count differs from manifest")
    if _require_nonnegative_int(counts.get("quarantined_records"), description="quarantined_records") != len(rejected):
        raise LineDatasetContractError("teacher rejected record count differs from manifest")
    return root, accepted, rejected, contract, receipt, observations


def _validate_source_membership(row: Mapping[str, object], *, location: str) -> tuple[Path, Path]:
    source_root = Path(_require_string(row.get("source_root"), description=f"source_root at {location}"))
    source_root = source_root.expanduser().resolve(strict=True)
    _require_no_reparse_ancestors(source_root)
    relative = _require_string(row.get("source_relative_path"), description=f"source_relative_path at {location}")
    parts = relative.split("/")
    if (
        "\\" in relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise LineDatasetContractError(f"unsafe source_relative_path at {location}: {relative!r}")
    source = Path(_require_string(row.get("source_absolute_path"), description=f"source path at {location}"))
    source = source.expanduser().resolve(strict=True)
    expected = (source_root / Path(*parts)).resolve(strict=True)
    if source != expected:
        raise LineDatasetContractError(f"source path differs from source_root/relative_path at {location}")
    try:
        source.relative_to(source_root)
    except ValueError:
        raise LineDatasetContractError(f"source escapes source_root at {location}") from None
    if not source.is_file():
        raise FileNotFoundError(source)
    return source_root, source


def _validate_teacher_record(
    row: Mapping[str, object],
    *,
    location: str,
    group_splits: dict[str, str],
    minimum_line_confidence: float,
    minimum_quad_area: float,
) -> tuple[str, str, str, Path, Path, dict[str, object], list[dict[str, object]]]:
    if row.get("schema_version") != SCHEMA_VERSION or row.get("kind") != TEACHER_RECORD_KIND:
        raise LineDatasetContractError(f"unsupported teacher row at {location}")
    record_id = _require_sha256(row.get("record_id"), description=f"record_id at {location}")
    group_id = _require_string(row.get("group_id"), description=f"group_id at {location}")
    split = row.get("split")
    if split not in {"train", "val", "test"}:
        raise LineDatasetContractError(f"invalid split at {location}")
    split = str(split)
    prior = group_splits.setdefault(group_id, split)
    if prior != split:
        raise LineDatasetContractError(f"teacher group {group_id!r} crosses splits")
    training_eligible = split == "train"
    if (
        row.get("split_use") != ("training" if training_eligible else f"heldout_{split}")
        or row.get("training_eligible") is not training_eligible
        or row.get("evaluation_only") is not (not training_eligible)
        or row.get("held_out") is not (not training_eligible)
    ):
        raise LineDatasetContractError(f"teacher split-use flags differ at {location}")
    if (
        row.get("label_source") != "paddle_db_cls_rec_three_view_consensus"
        or row.get("automatic_teacher_validation") is not True
        or row.get("manual_review_required") is not False
        or row.get("paddle_color_contract") != canonical_paddle_color_contract()
    ):
        raise LineDatasetContractError(f"teacher provenance flags differ at {location}")
    consensus = row.get("consensus")
    if not isinstance(consensus, Mapping) or consensus.get("agreement") not in {"2_of_3", "3_of_3"}:
        raise LineDatasetContractError(f"teacher consensus evidence is invalid at {location}")
    chosen_view_value = row.get("chosen_view")
    if not isinstance(chosen_view_value, Mapping):
        raise LineDatasetContractError(f"teacher chosen_view evidence is missing at {location}")
    view_id = chosen_view_value.get("view_id")
    if view_id not in {"original_rgb", "grayscale_clahe", "upscale_sharpen"}:
        raise LineDatasetContractError(f"teacher chosen_view id is invalid at {location}")
    if consensus.get("chosen_geometry_view_id") != view_id:
        raise LineDatasetContractError(f"teacher chosen_view differs from consensus geometry at {location}")
    expected_view_contract = canonical_view_contract(str(view_id))
    if chosen_view_value.get("view_contract_sha256") != _canonical_sha256(expected_view_contract):
        raise LineDatasetContractError(f"teacher chosen_view contract SHA-256 is invalid at {location}")
    chosen_view = {
        "view_id": str(view_id),
        "view_contract_sha256": str(chosen_view_value["view_contract_sha256"]),
        "transformed_pixel_sha256": _require_sha256(
            chosen_view_value.get("transformed_pixel_sha256"),
            description=f"chosen transformed pixel SHA-256 at {location}",
        ),
        "source_width": _require_nonnegative_int(
            chosen_view_value.get("source_width"), description=f"chosen source_width at {location}"
        ),
        "source_height": _require_nonnegative_int(
            chosen_view_value.get("source_height"), description=f"chosen source_height at {location}"
        ),
        "transformed_width": _require_nonnegative_int(
            chosen_view_value.get("transformed_width"), description=f"chosen transformed_width at {location}"
        ),
        "transformed_height": _require_nonnegative_int(
            chosen_view_value.get("transformed_height"), description=f"chosen transformed_height at {location}"
        ),
        "coordinate_mapping": chosen_view_value.get("coordinate_mapping"),
    }
    if any(int(chosen_view[name]) <= 1 for name in ("source_width", "source_height", "transformed_width", "transformed_height")):
        raise LineDatasetContractError(f"teacher chosen_view dimensions must be >1 at {location}")
    if chosen_view["coordinate_mapping"] != "full_frame_scale_source_normalized_identity_v1":
        raise LineDatasetContractError(f"teacher chosen_view coordinate mapping is invalid at {location}")
    source_root, source = _validate_source_membership(row, location=location)
    _require_sha256(row.get("raw_sha256"), description=f"raw_sha256 at {location}")
    _require_sha256(row.get("decoded_pixel_sha256"), description=f"decoded_pixel_sha256 at {location}")

    raw_lines = row.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise LineDatasetContractError(f"teacher lines must be a non-empty array at {location}")
    lines: list[dict[str, object]] = []
    for position, value in enumerate(raw_lines):
        if not isinstance(value, Mapping) or value.get("index") != position:
            raise LineDatasetContractError(f"teacher line index differs at {location}:{position}")
        text = _require_string(value.get("text"), description=f"line text at {location}:{position}")
        if text != _normalise_text(text):
            raise LineDatasetContractError(f"teacher line text is not canonically normalized at {location}:{position}")
        confidence_value = value.get("confidence")
        if isinstance(confidence_value, bool):
            raise LineDatasetContractError(f"teacher line confidence is invalid at {location}:{position}")
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            raise LineDatasetContractError(f"teacher line confidence is invalid at {location}:{position}") from None
        if not math.isfinite(confidence) or not minimum_line_confidence <= confidence <= 1.0:
            raise LineDatasetContractError(f"teacher line confidence is below the sealed floor at {location}:{position}")
        orientation_degrees = value.get("orientation_degrees")
        if (
            isinstance(orientation_degrees, bool)
            or not isinstance(orientation_degrees, int)
            or orientation_degrees not in {0, 180}
        ):
            raise LineDatasetContractError(
                f"teacher line does not bind the applied Paddle CLS orientation at {location}:{position}"
            )
        transformed_quad_value = value.get("transformed_quad_pixels")
        if not isinstance(transformed_quad_value, list) or len(transformed_quad_value) != 4:
            raise LineDatasetContractError(
                f"teacher line does not bind transformed_quad_pixels at {location}:{position}"
            )
        transformed_quad_pixels: list[list[float]] = []
        for point_index, raw_point in enumerate(transformed_quad_value):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise LineDatasetContractError(
                    f"teacher source point {point_index} is invalid at {location}:{position}"
                )
            try:
                transformed_x, transformed_y = float(raw_point[0]), float(raw_point[1])
            except (TypeError, ValueError):
                raise LineDatasetContractError(
                    f"teacher source point {point_index} is invalid at {location}:{position}"
                ) from None
            if (
                not math.isfinite(transformed_x)
                or not math.isfinite(transformed_y)
                or transformed_x < 0.0
                or transformed_y < 0.0
                or transformed_x > int(chosen_view["transformed_width"]) - 1
                or transformed_y > int(chosen_view["transformed_height"]) - 1
            ):
                raise LineDatasetContractError(
                    f"teacher source point {point_index} is invalid at {location}:{position}"
                )
            transformed_quad_pixels.append([transformed_x, transformed_y])
        try:
            quad, _bbox, _area = _parse_quad(
                value.get("quad_normalized"),
                description=f"line quad at {location}:{position}",
                minimum_area=minimum_quad_area,
            )
        except ValueError as error:
            raise LineDatasetContractError(str(error)) from error
        if value.get("quad_normalized") != quad:
            raise LineDatasetContractError(f"teacher line quad is not canonical at {location}:{position}")
        expected_points = sorted(
            (
                transformed_point[0] / max(1, int(chosen_view["transformed_width"]) - 1),
                transformed_point[1] / max(1, int(chosen_view["transformed_height"]) - 1),
            )
            for transformed_point in transformed_quad_pixels
        )
        normalized_points = sorted((float(point[0]), float(point[1])) for point in quad)
        for point_index, ((expected_x, expected_y), normalized_point) in enumerate(
            zip(expected_points, normalized_points)
        ):
            if not (
                math.isclose(float(normalized_point[0]), expected_x, rel_tol=0.0, abs_tol=1e-7)
                and math.isclose(float(normalized_point[1]), expected_y, rel_tol=0.0, abs_tol=1e-7)
            ):
                raise LineDatasetContractError(
                    f"teacher line normalized point set differs from transformed pixels at point {point_index} "
                    f"at {location}:{position}"
                )
        lines.append(
            {
                "index": position,
                "text": text,
                "confidence": confidence,
                "orientation_degrees": orientation_degrees,
                "transformed_quad_pixels": transformed_quad_pixels,
                "quad_normalized": quad,
            }
        )

    document_text = "\n".join(str(line["text"]) for line in lines)
    if row.get("text") != document_text or row.get("text_sha256") != _sha256_bytes(document_text.encode("utf-8")):
        raise LineDatasetContractError(f"teacher document text binding differs at {location}")
    return record_id, group_id, split, source_root, source, chosen_view, lines


def _crop_line(
    transformed_rgb: np.ndarray,
    transformed_quad_pixels: Sequence[Sequence[float]],
    *,
    orientation_degrees: int,
) -> tuple[np.ndarray, dict[str, object]]:
    quad = np.asarray(transformed_quad_pixels, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise LineDatasetContractError("transformed_quad_pixels must be a finite 4x2 array")
    height, width = transformed_rgb.shape[:2]
    if (
        np.any(quad[:, 0] < 0.0)
        or np.any(quad[:, 1] < 0.0)
        or np.any(quad[:, 0] > width - 1)
        or np.any(quad[:, 1] > height - 1)
    ):
        raise LineDatasetContractError("transformed_quad_pixels are outside chosen view pixels")
    crop_width = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    crop_height = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    if crop_width <= 0 or crop_height <= 0:
        raise LineDatasetContractError("transformed_quad_pixels produce an empty Paddle crop")
    destination = np.float32(
        [[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]]
    )
    homography = cv2.getPerspectiveTransform(quad, destination)
    # This intentionally reproduces PaddleOCR 2.10.0's
    # tools.infer.utility.get_rotate_crop_image byte-for-byte in geometry.
    crop = cv2.warpPerspective(
        np.ascontiguousarray(transformed_rgb),
        homography,
        (crop_width, crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    rotation_ccw = 0
    # PaddleOCR's DB crop helper rotates very tall text boxes before CLS/REC.
    # Reproduce that deterministic geometry step, followed below by the
    # teacher-bound binary 0/180 classifier decision.
    if crop.shape[0] / max(1, crop.shape[1]) >= 1.5:
        crop = np.ascontiguousarray(np.rot90(crop), dtype=np.uint8)
        rotation_ccw = 90
    if orientation_degrees == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    elif orientation_degrees != 0:
        raise LineDatasetContractError("teacher line orientation_degrees must be 0 or 180")
    return np.ascontiguousarray(crop, dtype=np.uint8), {
        "coordinate_space": "chosen_canonical_view_pixels",
        "perspective_interpolation": "opencv_inter_cubic",
        "perspective_border_mode": "opencv_border_replicate",
        "tall_crop_rotation": "ccw_90_when_height_over_width_gte_1.5",
        "rotation_applied_degrees_ccw": rotation_ccw,
        "paddle_cls_orientation_degrees": orientation_degrees,
        "paddle_color_contract": canonical_paddle_color_contract(),
        "chosen_view_to_crop_homography": np.round(homography, 10).tolist(),
    }


def _save_png(path: Path, pixels: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", compress_level=9, optimize=False)
    _, observation = _read_bound_file(path, description="materialized generic text line crop")
    with Image.open(path) as opened:
        decoded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if _pixel_sha256(decoded) != _pixel_sha256(pixels):
        raise LineDatasetContractError(f"PNG pixel readback differs after materialization: {path}")
    return {
        **observation,
        "pixel_sha256": _pixel_sha256(pixels),
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def materialize_otherimages_line_dataset(
    *,
    teacher_dir: Path,
    output_dir: Path,
    authorize_training: bool = False,
) -> dict[str, object]:
    """Publish a sealed generic-line crop dataset from one sealed teacher."""
    if authorize_training is not True:
        raise LineDatasetContractError(
            "materialization requires explicit authorize_training=True / --authorize-training; "
            "the source teacher intentionally records training_authorization=false"
        )
    teacher_root, accepted, _rejected, teacher_contract, _teacher_receipt, teacher_observations = (
        _load_teacher_publication(teacher_dir)
    )
    if not accepted:
        raise LineDatasetContractError("sealed teacher has no accepted records to materialize")

    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_dir))))
    _require_no_reparse_ancestors(output, include_leaf=False)
    if output.exists():
        raise FileExistsError(f"line dataset output must be brand-new: {output}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"line dataset output parent must already exist: {output.parent}")
    if _paths_overlap(output, teacher_root):
        raise LineDatasetContractError("line dataset output must be disjoint from the sealed teacher directory")
    output_parent_identity = _bind_output_parent(output)

    configuration = teacher_contract.get("configuration")
    if not isinstance(configuration, Mapping):
        raise LineDatasetContractError("teacher configuration must be an object")
    minimum_line_confidence_value = configuration.get("minimum_line_confidence")
    if isinstance(minimum_line_confidence_value, bool):
        raise LineDatasetContractError("teacher minimum_line_confidence is invalid")
    try:
        minimum_line_confidence = float(minimum_line_confidence_value)
    except (TypeError, ValueError):
        raise LineDatasetContractError("teacher minimum_line_confidence is invalid") from None
    if not math.isfinite(minimum_line_confidence) or not 0.0 <= minimum_line_confidence <= 1.0:
        raise LineDatasetContractError("teacher minimum_line_confidence is invalid")
    minimum_quad_area_value = configuration.get("minimum_normalized_quad_area", 1e-7)
    if isinstance(minimum_quad_area_value, bool):
        raise LineDatasetContractError("teacher minimum_normalized_quad_area is invalid")
    try:
        minimum_quad_area = float(minimum_quad_area_value)
    except (TypeError, ValueError):
        raise LineDatasetContractError("teacher minimum_normalized_quad_area is invalid") from None
    if not math.isfinite(minimum_quad_area) or not 0.0 < minimum_quad_area <= 1.0:
        raise LineDatasetContractError("teacher minimum_normalized_quad_area is invalid")

    group_splits: dict[str, str] = {}
    seen_record_ids: set[str] = set()
    validated: list[
        tuple[Mapping[str, object], str, str, str, Path, Path, dict[str, object], list[dict[str, object]]]
    ] = []
    for index, row in enumerate(accepted, start=1):
        location = f"{teacher_root / 'teacher_manifest.jsonl'}:{index}"
        record_id, group_id, split, source_root, source, chosen_view, lines = _validate_teacher_record(
            row,
            location=location,
            group_splits=group_splits,
            minimum_line_confidence=minimum_line_confidence,
            minimum_quad_area=minimum_quad_area,
        )
        if record_id in seen_record_ids:
            raise LineDatasetContractError(f"duplicate teacher record_id {record_id}")
        seen_record_ids.add(record_id)
        if _paths_overlap(output, source_root):
            raise LineDatasetContractError("line dataset output must be disjoint from every source_root")
        validated.append((row, record_id, group_id, split, source_root, source, chosen_view, lines))

    _verify_output_parent(output, output_parent_identity, location="line dataset staging")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.line-dataset-building-", dir=output.parent))
    stage_identity = _bind_stage_identity(stage, directory=True)
    published = False
    source_observations: list[dict[str, object]] = []
    crop_observations: list[dict[str, object]] = []
    try:
        manifest_rows: list[dict[str, object]] = []
        for teacher_row, record_id, group_id, split, _source_root, source, chosen_view, lines in validated:
            source_rgb, source_observation = _load_bound_upright_rgb(
                source,
                expected_raw_sha256=str(teacher_row["raw_sha256"]),
            )
            source_observations.append(source_observation)
            if _pixel_sha256(source_rgb) != teacher_row["decoded_pixel_sha256"]:
                raise LineDatasetContractError(f"source decoded pixels differ from teacher: {source}")
            if (
                int(source_rgb.shape[1]) != chosen_view["source_width"]
                or int(source_rgb.shape[0]) != chosen_view["source_height"]
            ):
                raise LineDatasetContractError(f"source dimensions differ from chosen teacher view: {source}")
            view_contract = canonical_view_contract(str(chosen_view["view_id"]))
            transformed_rgb = _canonical_transform(
                source_rgb,
                PaddleViewContract(
                    view_id=str(chosen_view["view_id"]),
                    operations=tuple(str(value) for value in view_contract["operations"]),
                ),
            )
            if (
                int(transformed_rgb.shape[1]) != chosen_view["transformed_width"]
                or int(transformed_rgb.shape[0]) != chosen_view["transformed_height"]
                or _pixel_sha256(transformed_rgb) != chosen_view["transformed_pixel_sha256"]
            ):
                raise LineDatasetContractError(f"chosen canonical view pixels differ from sealed teacher: {source}")
            for line in lines:
                line_index = int(line["index"])
                relative_crop = Path("images") / split / f"{record_id}-{line_index:03d}.png"
                crop, transform = _crop_line(
                    transformed_rgb,
                    line["transformed_quad_pixels"],
                    orientation_degrees=int(line["orientation_degrees"]),
                )
                crop_observation = _save_png(stage / relative_crop, crop)
                crop_observations.append(crop_observation)
                training_eligible = split == "train"
                manifest_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": LINE_RECORD_KIND,
                        "id": f"{record_id}:{line_index}",
                        "image": relative_crop.as_posix(),
                        "field": GENERIC_TEXT_LINE_FIELD,
                        "text": line["text"],
                        "paddle_text": line["text"],
                        "split": split,
                        "group_id": group_id,
                        "training_eligible": training_eligible,
                        "evaluation_only": not training_eligible,
                        "held_out": not training_eligible,
                        "label_source": teacher_row["label_source"],
                        "teacher_record_id": record_id,
                        "teacher_line_index": line_index,
                        "teacher_line_confidence": round(float(line["confidence"]), 8),
                        "teacher_line_orientation_degrees": int(line["orientation_degrees"]),
                        "teacher_consensus_agreement": dict(teacher_row["consensus"])["agreement"],
                        "quad_normalized": line["quad_normalized"],
                        "transformed_quad_pixels": line["transformed_quad_pixels"],
                        "chosen_view": chosen_view,
                        "source": str(source),
                        "source_relative_path": teacher_row["source_relative_path"],
                        "source_raw_sha256": teacher_row["raw_sha256"],
                        "source_decoded_pixel_sha256": teacher_row["decoded_pixel_sha256"],
                        "crop_sha256": crop_observation["sha256"],
                        "crop_pixel_sha256": crop_observation["pixel_sha256"],
                        "crop_size_bytes": crop_observation["size_bytes"],
                        "crop_width": crop_observation["width"],
                        "crop_height": crop_observation["height"],
                        "crop_transform": transform,
                        "truth_semantics": "paddle_teacher_parity_not_independent_truth",
                    }
                )
            _verify_observation(source_observation, description="source image after line crop materialization")

        manifest_rows.sort(key=lambda row: (str(row["teacher_record_id"]), int(row["teacher_line_index"])))
        split_counts = Counter(str(row["split"]) for row in manifest_rows)
        missing_splits = [name for name in ("train", "val", "test") if split_counts[name] <= 0]
        if missing_splits:
            raise LineDatasetContractError(
                "generic line dataset requires at least one train, val, and test line after teacher filtering; "
                f"missing={missing_splits}"
            )
        _write_jsonl(stage / MANIFEST_NAME, manifest_rows)
        manifest_binding = _binding(stage / MANIFEST_NAME)
        crop_bindings = sorted(
            (
                {
                    "path": Path(str(observation["path"])).relative_to(stage).as_posix(),
                    "sha256": observation["sha256"],
                    "size_bytes": observation["size_bytes"],
                    "pixel_sha256": observation["pixel_sha256"],
                    "width": observation["width"],
                    "height": observation["height"],
                }
                for observation in crop_observations
            ),
            key=lambda value: str(value["path"]),
        )
        counts = {
            "teacher_records": len(validated),
            "line_records": len(manifest_rows),
            "groups": len(group_splits),
            "by_split": {name: int(split_counts[name]) for name in ("train", "val", "test")},
            "training_eligible_lines": int(split_counts["train"]),
            "evaluation_only_lines": int(split_counts["val"] + split_counts["test"]),
        }
        crop_recipe = {
            "source_pixels": "pillow_exif_transpose_then_rgb8_bound_by_teacher_decoded_pixel_sha256",
            "quad_coordinate_space": "chosen_canonical_view_pixels_exact_db_box_order",
            "perspective_warp": "paddleocr_2.10.0_get_rotate_crop_image_exact_geometry_INTER_CUBIC_BORDER_REPLICATE",
            "tall_crop_rotation": "numpy_rot90_ccw_when_height_over_width_gte_1.5",
            "encoding": "pillow_png_rgb8_compress_level_9_optimize_false",
            "paddle_color_contract": canonical_paddle_color_contract(),
        }
        inputs = {
            "teacher_directory": str(teacher_root),
            "teacher_manifest": _binding(teacher_root / "teacher_manifest.jsonl"),
            "teacher_contract": _binding(teacher_root / "teacher.contract.json"),
            "teacher_receipt": _binding(teacher_root / "teacher.receipt.json"),
            "teacher_contract_closure_sha256": teacher_contract["closure_sha256"],
        }
        artifacts = {
            "manifest": manifest_binding,
            "crops": {
                "count": len(crop_bindings),
                "size_bytes": sum(int(value["size_bytes"]) for value in crop_bindings),
                "closure_sha256": _canonical_sha256(crop_bindings),
            },
        }
        split_use = {
            "train": "training_eligible_teacher_distillation",
            "val": "heldout_teacher_parity_validation_only",
            "test": "heldout_teacher_parity_final_evaluation_only",
            "group_split_preserved": True,
            "groups_may_cross_splits": False,
        }
        closure_payload = {
            "schema_version": SCHEMA_VERSION,
            "inputs": inputs,
            "crop_recipe": crop_recipe,
            "counts": counts,
            "split_use": split_use,
            "artifacts": artifacts,
        }
        contract: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": LINE_DATASET_CONTRACT_KIND,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sealed": True,
            "output_directory": str(output),
            "records": MANIFEST_NAME,
            "field": GENERIC_TEXT_LINE_FIELD,
            "inputs": inputs,
            "crop_recipe": crop_recipe,
            "counts": counts,
            "split_use": split_use,
            "artifacts": artifacts,
            "closure_sha256": _canonical_sha256(closure_payload),
            "training_authorization": True,
            "training_authorization_source": "explicit_materializer_flag",
            "truth_semantics": "teacher_parity_only_not_independent_business_truth",
            "limitations": [
                "Paddle three-view consensus is a pseudo-label, not independent human ground truth",
                "Held-out exact match and CER measure teacher parity, not business OCR accuracy",
                "Crops reproduce the chosen teacher view's DB perspective, tall-box rotation, and applied binary 0/180 CLS decision",
            ],
        }
        _write_json(stage / CONTRACT_NAME, contract)
        contract_binding = _binding(stage / CONTRACT_NAME)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": LINE_DATASET_RECEIPT_KIND,
            "sealed": True,
            "contract": contract_binding,
            "contract_closure_sha256": contract["closure_sha256"],
        }
        _write_json(stage / RECEIPT_NAME, receipt)

        for observation in teacher_observations + source_observations + crop_observations:
            _verify_observation(observation, description="line dataset closure input/artifact")
        if _binding(stage / MANIFEST_NAME) != manifest_binding:
            raise LineDatasetContractError("generic line manifest changed before publication")
        if _binding(stage / CONTRACT_NAME) != contract_binding:
            raise LineDatasetContractError("generic line dataset contract changed before publication")
        expected_files = {MANIFEST_NAME, CONTRACT_NAME, RECEIPT_NAME} | {
            str(value["path"]) for value in crop_bindings
        }
        observed_files = {
            path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()
        }
        if observed_files != expected_files or any(path.is_symlink() for path in stage.rglob("*")):
            raise LineDatasetContractError("generic line dataset stage membership differs before publication")

        if output.exists():
            raise FileExistsError(f"line dataset output appeared during build: {output}")
        _verify_output_parent(output, output_parent_identity, location="line dataset publication")
        _rename_directory_no_replace(
            stage,
            output,
            expected_parent_identity=output_parent_identity,
            expected_stage_identity=stage_identity,
        )
        if _bind_stage_identity(output, directory=True) != stage_identity:
            raise LineDatasetContractError("published line dataset identity differs from bound stage")
        published_files = {
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        }
        if published_files != expected_files or any(path.is_symlink() for path in output.rglob("*")):
            raise LineDatasetContractError("published line dataset membership failed readback")
        if _binding(output / MANIFEST_NAME) != manifest_binding:
            raise LineDatasetContractError("published generic line manifest failed readback")
        for expected in crop_bindings:
            crop_path = output / str(expected["path"])
            data = crop_path.read_bytes()
            with Image.open(crop_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            observed = {
                "path": expected["path"],
                "sha256": _sha256_bytes(data),
                "size_bytes": len(data),
                "pixel_sha256": _pixel_sha256(rgb),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            }
            if observed != expected:
                raise LineDatasetContractError(f"published crop failed readback: {expected['path']}")
        if (output / CONTRACT_NAME).read_bytes() != (json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"):
            raise LineDatasetContractError("published line dataset contract failed exact readback")
        published_receipt = _load_json_object_bytes(
            (output / RECEIPT_NAME).read_bytes(), source=str(output / RECEIPT_NAME)
        )
        if published_receipt != receipt:
            raise LineDatasetContractError("published line dataset receipt failed readback")
        published = True
        return contract
    finally:
        if not published:
            # Preserve the uniquely named failed stage for audit/recovery.
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a sealed OtherImages Paddle teacher into generic CTC line crops"
    )
    parser.add_argument("--teacher", type=Path, required=True, help="sealed teacher publication directory")
    parser.add_argument("--output", type=Path, required=True, help="brand-new sealed line dataset directory")
    parser.add_argument(
        "--authorize-training",
        action="store_true",
        help="explicitly authorize pseudo-label distillation; recorded in the output contract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        contract = materialize_otherimages_line_dataset(
            teacher_dir=args.teacher,
            output_dir=args.output,
            authorize_training=args.authorize_training,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OtherImages line dataset materialization failed:\n{error}") from None
    counts = dict(contract["counts"])
    print(
        f"Sealed {counts['line_records']} generic text line(s) from {counts['teacher_records']} teacher record(s) "
        f"at {Path(args.output).absolute()}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
