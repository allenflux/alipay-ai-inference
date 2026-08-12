"""Seal automatic Paddle DB+CLS+REC labels from three captured OCR views.

The module is an offline aggregator: it does not import Paddle, open a model,
perform OCR, or train anything.  It consumes the read-only OtherImages
inventory plus exactly three complete, already-captured layout-result JSONL
files and independently recomputes the canonical transform pixel hashes from
source images.  Only a unique two-of-three or three-of-three normalized text
result whose supporting views pass confidence, geometry, and semantic gates
can enter ``teacher_manifest.jsonl``.  Every other inventory item is
quarantined without inventing a label.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .otherimages_inventory import (
    _bind_stage_identity,
    _rename_directory_no_replace,
)


SCHEMA_VERSION = 1
INVENTORY_CONTRACT_KIND = "otherimages_read_only_inventory_v1"
INVENTORY_PENDING_KIND = "otherimages_paddle_teacher_pending_v1"
CAPTURE_KIND = "otherimages_paddle_layout_capture_v1"
VIEW_CONTRACT_KIND = "otherimages_paddle_view_contract_v1"
ADAPTER_EVIDENCE_KIND = "paddle_db_cls_rec_adapter_evidence_v1"
PINNED_ADAPTER_IMPLEMENTATION = "pinned_paddleocr_2.10.0_raw_db_cls_rec_v1"
PINNED_PADDLEOCR_VERSION = "2.10.0"
PINNED_ALBUMENTATIONS_VERSION = "1.4.10"
PINNED_ALBUCORE_VERSION = "0.0.13"
MODEL_ASSET_ROLES = ("det", "cls", "rec", "dictionary")
PADDLE_EFFECTIVE_ARG_KEYS = (
    "ocr_version",
    "det_algorithm",
    "det_limit_side_len",
    "det_limit_type",
    "det_db_thresh",
    "det_db_box_thresh",
    "det_db_unclip_ratio",
    "det_db_score_mode",
    "det_box_type",
    "rec_algorithm",
    "rec_image_shape",
    "rec_batch_num",
    "max_text_length",
    "use_space_char",
    "cls_image_shape",
    "cls_batch_num",
    "cls_thresh",
    "use_angle_cls",
    "drop_score",
    "use_onnx",
    "precision",
    "use_tensorrt",
    "enable_mkldnn",
    "cpu_threads",
    "use_gpu",
    "gpu_id",
)
TEACHER_RECORD_KIND = "otherimages_paddle_teacher_record_v1"
QUARANTINE_RECORD_KIND = "otherimages_paddle_teacher_quarantine_v1"
TEACHER_CONTRACT_KIND = "otherimages_paddle_teacher_contract_v1"
TEACHER_RECEIPT_KIND = "otherimages_paddle_teacher_receipt_v1"

DEFAULT_MIN_LINE_CONFIDENCE = 0.90
DEFAULT_MIN_VIEW_CONFIDENCE = 0.93
DEFAULT_MIN_GEOMETRY_IOU = 0.50
DEFAULT_MIN_NORMALIZED_QUAD_AREA = 1e-7
DEFAULT_MAX_LINES = 128
DEFAULT_MAX_LINE_CHARACTERS = 256
DEFAULT_MAX_DOCUMENT_CHARACTERS = 8192

CANONICAL_VIEW_OPERATIONS: dict[str, tuple[str, ...]] = {
    "original_rgb": (
        "pillow_exif_transpose",
        "pillow_convert_rgb8",
        "identity",
    ),
    "grayscale_clahe": (
        "pillow_exif_transpose",
        "pillow_convert_rgb8",
        "opencv_rgb_to_gray",
        "opencv_clahe_clip_limit_2.0_tile_grid_8x8",
        "gray_replicate_to_rgb8",
    ),
    "upscale_sharpen": (
        "pillow_exif_transpose",
        "pillow_convert_rgb8",
        "opencv_resize_exact_2x_inter_cubic",
        "opencv_gaussian_blur_kernel_5x5_sigma_x_1.0_sigma_y_1.0",
        "opencv_add_weighted_source_1.5_blur_-0.5_gamma_0_rgb8",
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VIEW_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_OUTPUT_FILES = ("teacher_manifest.jsonl", "reject_manifest.jsonl")


class TeacherContractError(ValueError):
    """Raised for a fatal input or publication contract violation."""


class CandidateGateError(ValueError):
    """A record-local Paddle candidate failure that must be quarantined."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _reject_json_constant(value: str) -> None:
    raise TeacherContractError(f"non-standard JSON constant {value!r} is forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise TeacherContractError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _loads_json(text: str, *, location: str) -> object:
    if text.startswith("\ufeff"):
        raise TeacherContractError(f"UTF-8 BOM is forbidden at {location}")
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, TeacherContractError) as error:
        raise TeacherContractError(f"invalid JSON at {location}: {error}") from error


def _json_line(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_view_contract(view_id: str) -> dict[str, object]:
    operations = CANONICAL_VIEW_OPERATIONS.get(view_id)
    if operations is None:
        raise TeacherContractError(
            f"view_id must be one of the three canonical recipes: {','.join(sorted(CANONICAL_VIEW_OPERATIONS))}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": VIEW_CONTRACT_KIND,
        "view_id": view_id,
        "operations": list(operations),
        "quad_coordinate_space": "exif_upright_source_normalized",
        "line_order": "top_to_bottom_left_to_right_v1",
        "transform_implementation": "otherimages_paddle_capture_core_v1",
    }


def _is_reparse(path: Path) -> bool:
    status = path.lstat()
    attributes = int(getattr(status, "st_file_attributes", 0))
    reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_attribute)


def _require_no_reparse_ancestors(path: Path, *, include_leaf: bool = True) -> None:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = candidate if include_leaf else candidate.parent
    while True:
        if current.exists() and _is_reparse(current):
            raise TeacherContractError(f"path traverses a symlink/junction/reparse point: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _bind_output_parent(path: Path) -> tuple[int, int]:
    """Bind the already-reviewed publication parent across long OCR work."""
    _require_no_reparse_ancestors(path, include_leaf=False)
    parent = path.parent
    if not parent.is_dir():
        raise NotADirectoryError(f"output parent must be a directory: {parent}")
    return _bind_stage_identity(parent, directory=True)


def _verify_output_parent(path: Path, identity: tuple[int, int], *, location: str) -> None:
    """Fail closed if a parent was replaced by a junction/symlink/directory."""
    _require_no_reparse_ancestors(path, include_leaf=False)
    observed = _bind_stage_identity(path.parent, directory=True)
    if observed != identity:
        raise TeacherContractError(f"output parent identity changed before {location}: {path.parent}")


def _stat_signature(status: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(status.st_size),
        int(status.st_mtime_ns),
        int(getattr(status, "st_dev", 0)),
        int(getattr(status, "st_ino", 0)),
    )


def _read_bound_file(path: Path, *, description: str) -> tuple[bytes, dict[str, object]]:
    _require_no_reparse_ancestors(path)
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {description}: {resolved}")
    with resolved.open("rb") as stream:
        before = _stat_signature(os.fstat(stream.fileno()))
        data = stream.read()
        descriptor_after = _stat_signature(os.fstat(stream.fileno()))
    path_after = _stat_signature(resolved.stat())
    if descriptor_after != before or path_after != before:
        raise TeacherContractError(f"{description} changed while it was read: {resolved}")
    return data, {
        "path": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
        "signature": before,
    }


def _observe_source_file(path: Path, *, expected_sha256: str) -> dict[str, object]:
    _require_no_reparse_ancestors(path)
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"missing inventory source image: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        before = _stat_signature(os.fstat(stream.fileno()))
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        descriptor_after = _stat_signature(os.fstat(stream.fileno()))
    path_after = _stat_signature(resolved.stat())
    observed_sha256 = digest.hexdigest()
    if descriptor_after != before or path_after != before:
        raise TeacherContractError(f"source image changed while it was hashed: {resolved}")
    if observed_sha256 != expected_sha256:
        raise TeacherContractError(
            f"source image raw SHA-256 differs from inventory: {resolved}; "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    return {
        "path": str(resolved),
        "sha256": observed_sha256,
        "size_bytes": before[0],
        "signature": before,
    }


def _verify_observation(observation: Mapping[str, object], *, description: str) -> None:
    path = Path(str(observation["path"]))
    expected_signature = tuple(observation["signature"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = _stat_signature(os.fstat(stream.fileno()))
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = _stat_signature(os.fstat(stream.fileno()))
    if before != expected_signature or after != before or _stat_signature(path.stat()) != before:
        raise TeacherContractError(f"{description} identity changed before publication: {path}")
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != observation["sha256"]:
        raise TeacherContractError(f"{description} SHA-256 changed before publication: {path}")


def _public_binding(observation: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": str(observation["path"]),
        "sha256": str(observation["sha256"]),
        "size_bytes": int(observation["size_bytes"]),
    }


def _load_jsonl_bytes(data: bytes, *, source: str) -> list[dict[str, object]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TeacherContractError(f"{source} is not strict UTF-8: {error}") from error
    if text.startswith("\ufeff"):
        raise TeacherContractError(f"UTF-8 BOM is forbidden in {source}")
    if not text:
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise TeacherContractError(f"blank JSONL line is forbidden at {source}:{line_number}")
        value = _loads_json(line, location=f"{source}:{line_number}")
        if not isinstance(value, dict):
            raise TeacherContractError(f"JSONL row must be an object at {source}:{line_number}")
        rows.append(value)
    return rows


def _load_json_object_bytes(data: bytes, *, source: str) -> dict[str, object]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TeacherContractError(f"{source} is not strict UTF-8: {error}") from error
    value = _loads_json(text, location=source)
    if not isinstance(value, dict):
        raise TeacherContractError(f"JSON document must be an object: {source}")
    return value


def _require_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TeacherContractError(f"{description} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherContractError(f"{description} must be a non-empty string")
    return value


def _finite_unit_score(value: object, *, description: str) -> float:
    if isinstance(value, bool):
        raise CandidateGateError("invalid_confidence", f"{description} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise CandidateGateError("invalid_confidence", f"{description} must be numeric") from None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise CandidateGateError("invalid_confidence", f"{description} must be finite and in [0, 1]")
    return score


def _paths_overlap(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(os.path.abspath(os.fspath(left)))
    right_text = os.path.normcase(os.path.abspath(os.fspath(right)))
    try:
        common = os.path.commonpath((left_text, right_text))
    except ValueError:
        return False
    return common == left_text or common == right_text


def _normalise_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _semantic_text_error(text: str, *, max_characters: int, description: str) -> str | None:
    if not text:
        return f"{description} is empty after NFKC/whitespace normalization"
    if len(text) > max_characters:
        return f"{description} has {len(text)} characters; maximum is {max_characters}"
    if "\ufffd" in text:
        return f"{description} contains Unicode replacement characters"
    for character in text:
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            return f"{description} contains forbidden Unicode category {category}"
        if not character.isprintable():
            return f"{description} contains a non-printable character"
    if not any(unicodedata.category(character)[0] in {"L", "N"} for character in text):
        return f"{description} contains no letter or number"
    return None


def _raw_text_error(text: str, *, description: str) -> str | None:
    if "\ufffd" in text:
        return f"{description} contains Unicode replacement characters"
    for character in text:
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            return f"{description} contains forbidden raw Unicode category {category}"
    return None


def _inventory_artifact_binding(contract: Mapping[str, object], manifest_name: str) -> Mapping[str, object]:
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, list):
        raise TeacherContractError("inventory contract artifacts must be an array")
    matches = [item for item in artifacts if isinstance(item, Mapping) and item.get("path") == manifest_name]
    if len(matches) != 1:
        raise TeacherContractError(
            f"inventory contract must bind exactly one {manifest_name!r} artifact, found {len(matches)}"
        )
    return matches[0]


def load_inventory_for_teacher(
    manifest_path: Path,
    *,
    contract_path: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Load and bind the inventory manifest for the aggregator or capture adapter."""
    manifest_data, manifest_observation = _read_bound_file(manifest_path, description="inventory teacher manifest")
    manifest_resolved = Path(str(manifest_observation["path"]))
    if contract_path is None:
        contract_path = manifest_resolved.parent / "inventory.contract.json"
    contract_data, contract_observation = _read_bound_file(contract_path, description="inventory contract")
    contract = _load_json_object_bytes(contract_data, source=str(contract_observation["path"]))
    if contract.get("kind") != INVENTORY_CONTRACT_KIND or contract.get("schema_version") != SCHEMA_VERSION:
        raise TeacherContractError("unsupported inventory contract kind/schema")
    contract_resolved = Path(str(contract_observation["path"]))
    output_contract = contract.get("output")
    if not isinstance(output_contract, Mapping):
        raise TeacherContractError("inventory contract output must be an object")
    contracted_inventory_directory = Path(
        _require_nonempty_string(
            output_contract.get("output_directory"),
            description="inventory contract output_directory",
        )
    ).expanduser().resolve(strict=True)
    if (
        manifest_resolved.parent != contracted_inventory_directory
        or contract_resolved.parent != contracted_inventory_directory
    ):
        raise TeacherContractError("inventory manifest/contract are outside the bound inventory publication directory")
    source_contract = contract.get("source")
    if not isinstance(source_contract, Mapping):
        raise TeacherContractError("inventory contract source must be an object")
    contracted_source_root = Path(
        _require_nonempty_string(
            source_contract.get("input_directory"),
            description="inventory contract input_directory",
        )
    ).expanduser().resolve(strict=True)
    _require_no_reparse_ancestors(contracted_source_root)
    if not contracted_source_root.is_dir():
        raise TeacherContractError("inventory contract input_directory is not a directory")
    binding = _inventory_artifact_binding(contract, manifest_resolved.name)
    if (
        binding.get("sha256") != manifest_observation["sha256"]
        or binding.get("size_bytes") != manifest_observation["size_bytes"]
        or binding.get("line_count") != manifest_observation["line_count"]
    ):
        raise TeacherContractError("inventory contract binding for paddle_teacher_pending.jsonl differs from file")
    teacher_contract = contract.get("paddle_teacher_contract")
    if not isinstance(teacher_contract, Mapping):
        raise TeacherContractError("inventory contract has no Paddle teacher policy")
    if teacher_contract.get("inventory_contains_labels") is not False:
        raise TeacherContractError("inventory unexpectedly claims to contain labels")
    if teacher_contract.get("inventory_performed_ocr") is not False:
        raise TeacherContractError("inventory unexpectedly claims OCR was performed")

    rows = _load_jsonl_bytes(manifest_data, source=str(manifest_observation["path"]))
    if not rows:
        raise TeacherContractError("inventory teacher manifest has no rows")
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=1):
        location = f"{manifest_observation['path']}:{line_number}"
        if row.get("schema_version") != SCHEMA_VERSION or row.get("kind") != INVENTORY_PENDING_KIND:
            raise TeacherContractError(f"unsupported inventory teacher row at {location}")
        record_id = _require_sha256(row.get("record_id"), description=f"record_id at {location}")
        if record_id in seen_ids:
            raise TeacherContractError(f"duplicate inventory record_id {record_id} at {location}")
        seen_ids.add(record_id)
        group_id = _require_nonempty_string(row.get("group_id"), description=f"group_id at {location}")
        split = row.get("suggested_split")
        if split not in {"train", "val", "test"}:
            raise TeacherContractError(f"invalid suggested_split at {location}")
        prior_split = group_splits.setdefault(group_id, str(split))
        if prior_split != split:
            raise TeacherContractError(f"inventory group {group_id!r} crosses suggested splits")
        _require_sha256(row.get("raw_sha256"), description=f"raw_sha256 at {location}")
        _require_sha256(
            row.get("decoded_pixel_sha256"),
            description=f"decoded_pixel_sha256 at {location}",
        )
        source_root = Path(
            _require_nonempty_string(row.get("source_root"), description=f"source_root at {location}")
        ).expanduser().resolve(strict=True)
        if source_root != contracted_source_root:
            raise TeacherContractError(f"inventory source_root differs from contract at {location}")
        relative_path = _require_nonempty_string(
            row.get("source_relative_path"),
            description=f"source_relative_path at {location}",
        )
        relative_parts = relative_path.split("/")
        if (
            "\\" in relative_path
            or relative_path.startswith("/")
            or any(part in {"", ".", ".."} for part in relative_parts)
            or ":" in relative_parts[0]
        ):
            raise TeacherContractError(f"unsafe source_relative_path at {location}: {relative_path!r}")
        source_absolute_path = Path(
            _require_nonempty_string(
                row.get("source_absolute_path"),
                description=f"source_absolute_path at {location}",
            )
        ).expanduser().resolve(strict=True)
        expected_source_path = (contracted_source_root / Path(*relative_parts)).resolve(strict=True)
        if source_absolute_path != expected_source_path or not _paths_overlap(contracted_source_root, source_absolute_path):
            raise TeacherContractError(f"inventory source path membership differs from contract at {location}")
        if row.get("teacher_state") not in {"pending", "quarantine"}:
            raise TeacherContractError(f"invalid teacher_state at {location}")
        if row.get("labels_present") is not False or row.get("ocr_performed") is not False:
            raise TeacherContractError(f"inventory row unexpectedly contains labels/OCR at {location}")
        if row.get("training_eligible") is not False:
            raise TeacherContractError(f"inventory row is prematurely training-eligible at {location}")
    return rows, contract, [manifest_observation, contract_observation]


def _validate_adapter_evidence(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TeacherContractError(f"adapter evidence must be an object at {location}")
    if value.get("kind") != ADAPTER_EVIDENCE_KIND:
        raise TeacherContractError(f"unsupported adapter evidence at {location}")
    implementation = _require_nonempty_string(
        value.get("adapter_implementation"),
        description=f"adapter implementation at {location}",
    )
    if implementation != PINNED_ADAPTER_IMPLEMENTATION:
        raise TeacherContractError(f"unsupported adapter implementation at {location}: {implementation}")
    version = _require_nonempty_string(value.get("paddle_version"), description=f"Paddle version at {location}")
    if version != PINNED_PADDLEOCR_VERSION:
        raise TeacherContractError(
            f"PaddleOCR must be pinned to {PINNED_PADDLEOCR_VERSION} at {location}; observed {version}"
        )
    model_sha = _require_sha256(
        value.get("model_contract_sha256"),
        description=f"model_contract_sha256 at {location}",
    )
    stages = value.get("stages")
    if not isinstance(stages, Mapping) or any(stages.get(name) is not True for name in ("db", "cls", "rec")):
        raise TeacherContractError(f"adapter must attest DB+CLS+REC stages at {location}")
    try:
        drop_score = float(value.get("drop_score"))
    except (TypeError, ValueError):
        raise TeacherContractError(f"adapter drop_score must be numeric at {location}") from None
    if not math.isfinite(drop_score) or not 0.0 <= drop_score <= 1.0:
        raise TeacherContractError(f"adapter drop_score must be finite and in [0, 1] at {location}")
    if value.get("raw_db_lines_preserved_before_drop_filter") is not True:
        raise TeacherContractError(f"adapter must preserve every raw DB line before drop filtering at {location}")
    if value.get("adapter_input_color_bridge") != "opencv_rgb8_to_bgr8_v1":
        raise TeacherContractError(f"adapter must bind the canonical RGB-to-BGR Paddle bridge at {location}")
    execution_device = _require_nonempty_string(
        value.get("execution_device"),
        description=f"execution_device at {location}",
    )
    if execution_device != "cpu" and re.fullmatch(r"gpu:[0-9]+", execution_device) is None:
        raise TeacherContractError(f"unsupported execution_device at {location}: {execution_device}")
    runtime_versions_value = value.get("runtime_versions")
    if not isinstance(runtime_versions_value, Mapping):
        raise TeacherContractError(f"adapter runtime_versions must be an object at {location}")
    runtime_versions = {
        name: _require_nonempty_string(
            runtime_versions_value.get(name),
            description=f"runtime_versions.{name} at {location}",
        )
        for name in ("paddleocr", "paddlepaddle", "albumentations", "albucore", "opencv", "numpy", "pillow")
    }
    if runtime_versions["paddleocr"] != PINNED_PADDLEOCR_VERSION:
        raise TeacherContractError(f"runtime PaddleOCR version differs from the pinned adapter at {location}")
    if runtime_versions["albumentations"] != PINNED_ALBUMENTATIONS_VERSION:
        raise TeacherContractError(f"runtime Albumentations version differs from the pinned adapter at {location}")
    if runtime_versions["albucore"] != PINNED_ALBUCORE_VERSION:
        raise TeacherContractError(f"runtime Albucore version differs from the pinned adapter at {location}")

    effective_args_value = value.get("effective_paddle_args")
    if not isinstance(effective_args_value, Mapping) or set(effective_args_value) != set(PADDLE_EFFECTIVE_ARG_KEYS):
        raise TeacherContractError(
            f"adapter effective_paddle_args must contain exactly the pinned argument closure at {location}"
        )
    effective_args: dict[str, object] = {}
    for name in PADDLE_EFFECTIVE_ARG_KEYS:
        argument = effective_args_value[name]
        if isinstance(argument, float) and not math.isfinite(argument):
            raise TeacherContractError(f"effective_paddle_args.{name} must be finite at {location}")
        if not isinstance(argument, (str, bool, int, float)):
            raise TeacherContractError(f"effective_paddle_args.{name} must be a JSON scalar at {location}")
        effective_args[name] = argument
    if (
        effective_args["det_algorithm"] != "DB"
        or effective_args["det_box_type"] != "quad"
        or effective_args["use_angle_cls"] is not True
        or effective_args["use_onnx"] is not False
        or float(effective_args["drop_score"]) != drop_score
    ):
        raise TeacherContractError(f"effective Paddle DB+CLS+REC arguments violate the pinned contract at {location}")

    assets_value = value.get("model_assets")
    if not isinstance(assets_value, Mapping) or set(assets_value) != set(MODEL_ASSET_ROLES):
        raise TeacherContractError(
            f"adapter model_assets must contain exactly {list(MODEL_ASSET_ROLES)} at {location}"
        )
    assets: dict[str, dict[str, object]] = {}
    for role in MODEL_ASSET_ROLES:
        raw_asset = assets_value.get(role)
        if not isinstance(raw_asset, Mapping):
            raise TeacherContractError(f"model_assets.{role} must be an object at {location}")
        asset_path = _require_nonempty_string(
            raw_asset.get("path"),
            description=f"model_assets.{role}.path at {location}",
        )
        if not os.path.isabs(os.path.expanduser(asset_path)):
            raise TeacherContractError(f"model_assets.{role}.path must be absolute at {location}")
        raw_files = raw_asset.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise TeacherContractError(f"model_assets.{role}.files must be a non-empty array at {location}")
        files: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        seen_casefold_paths: set[str] = set()
        for file_index, raw_file in enumerate(raw_files):
            if not isinstance(raw_file, Mapping):
                raise TeacherContractError(
                    f"model_assets.{role}.files[{file_index}] must be an object at {location}"
                )
            relative_path = _require_nonempty_string(
                raw_file.get("path"),
                description=f"model_assets.{role}.files[{file_index}].path at {location}",
            )
            path_parts = relative_path.split("/")
            if (
                "\\" in relative_path
                or relative_path.startswith("/")
                or any(part in {"", ".", ".."} for part in path_parts)
                or ":" in path_parts[0]
            ):
                raise TeacherContractError(
                    f"model_assets.{role} contains an unsafe relative file path at {location}: {relative_path!r}"
                )
            if relative_path in seen_paths or relative_path.casefold() in seen_casefold_paths:
                raise TeacherContractError(
                    f"model_assets.{role} contains duplicate/case-colliding paths at {location}: {relative_path!r}"
                )
            seen_paths.add(relative_path)
            seen_casefold_paths.add(relative_path.casefold())
            sha256 = _require_sha256(
                raw_file.get("sha256"),
                description=f"model_assets.{role}.files[{file_index}].sha256 at {location}",
            )
            size_bytes = raw_file.get("size_bytes")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
                raise TeacherContractError(
                    f"model_assets.{role}.files[{file_index}].size_bytes must be non-negative at {location}"
                )
            files.append({"path": relative_path, "sha256": sha256, "size_bytes": size_bytes})
        if files != sorted(files, key=lambda item: str(item["path"]).encode("utf-8")):
            raise TeacherContractError(f"model_assets.{role}.files must be canonically sorted at {location}")
        closure_sha256 = _require_sha256(
            raw_asset.get("closure_sha256"),
            description=f"model_assets.{role}.closure_sha256 at {location}",
        )
        if closure_sha256 != _canonical_sha256(files):
            raise TeacherContractError(f"model_assets.{role} closure does not bind its files at {location}")
        size_bytes = raw_asset.get("size_bytes")
        expected_size = sum(int(item["size_bytes"]) for item in files)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes != expected_size:
            raise TeacherContractError(f"model_assets.{role} size closure differs at {location}")
        assets[role] = {
            "path": asset_path,
            "files": files,
            "closure_sha256": closure_sha256,
            "size_bytes": size_bytes,
        }
    normalized: dict[str, object] = {
        "kind": ADAPTER_EVIDENCE_KIND,
        "adapter_implementation": implementation,
        "paddle_version": version,
        "model_contract_sha256": model_sha,
        "drop_score": drop_score,
        "stages": {"db": True, "cls": True, "rec": True},
        "execution_device": execution_device,
        "runtime_versions": runtime_versions,
        "effective_paddle_args": effective_args,
        "model_assets": assets,
        "raw_db_lines_preserved_before_drop_filter": True,
        "adapter_input_color_bridge": "opencv_rgb8_to_bgr8_v1",
    }
    model_identity_payload = {
        "adapter_implementation": implementation,
        "paddleocr_version": version,
        "runtime_versions": runtime_versions,
        "effective_paddle_args": effective_args,
        "device": execution_device,
        "drop_score": drop_score,
        "assets": assets,
        "adapter_input_color_bridge": "opencv_rgb8_to_bgr8_v1",
    }
    if _canonical_sha256(model_identity_payload) != model_sha:
        raise TeacherContractError(f"adapter model_contract_sha256 does not bind runtime/assets at {location}")
    return normalized


def _asset_file_map(
    adapter: Mapping[str, object],
    *,
    location: str,
) -> dict[tuple[str, str], Path]:
    assets = adapter.get("model_assets")
    assert isinstance(assets, Mapping)
    output: dict[tuple[str, str], Path] = {}
    for role in MODEL_ASSET_ROLES:
        asset = assets.get(role)
        assert isinstance(asset, Mapping)
        raw_path = Path(str(asset["path"]))
        _require_no_reparse_ancestors(raw_path)
        resolved = raw_path.expanduser().resolve(strict=True)
        if resolved.is_file():
            actual_files = [(resolved.name, resolved)]
        elif resolved.is_dir():
            actual_files = []
            stack = [resolved]
            while stack:
                directory = stack.pop()
                _require_no_reparse_ancestors(directory)
                with os.scandir(directory) as entries:
                    for entry in entries:
                        entry_path = Path(entry.path)
                        if entry.is_symlink() or _is_reparse(entry_path):
                            raise TeacherContractError(
                                f"model_assets.{role} traverses a symlink/junction/reparse point at {location}: "
                                f"{entry_path}"
                            )
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry_path)
                        elif entry.is_file(follow_symlinks=False):
                            actual_files.append((entry_path.relative_to(resolved).as_posix(), entry_path))
                        else:
                            raise TeacherContractError(
                                f"model_assets.{role} contains a non-regular filesystem entry at {location}: "
                                f"{entry_path}"
                            )
        else:
            raise TeacherContractError(f"model_assets.{role} path is not a regular file/directory at {location}")
        actual_files.sort(key=lambda item: item[0].encode("utf-8"))
        expected_files = asset.get("files")
        assert isinstance(expected_files, list)
        expected_names = [str(item["path"]) for item in expected_files if isinstance(item, Mapping)]
        actual_names = [item[0] for item in actual_files]
        if actual_names != expected_names:
            raise TeacherContractError(
                f"model_assets.{role} filesystem membership differs from evidence at {location}: "
                f"expected={expected_names[:5]}, observed={actual_names[:5]}"
            )
        for relative_path, file_path in actual_files:
            key = role, relative_path
            output[key] = file_path
    return output


def _observe_adapter_assets(
    adapter: Mapping[str, object],
    *,
    location: str,
) -> list[dict[str, object]]:
    paths = _asset_file_map(adapter, location=location)
    assets = adapter.get("model_assets")
    assert isinstance(assets, Mapping)
    observations: list[dict[str, object]] = []
    for (role, relative_path), path in sorted(paths.items()):
        asset = assets[role]
        assert isinstance(asset, Mapping)
        files = asset["files"]
        assert isinstance(files, list)
        expected = next(item for item in files if isinstance(item, Mapping) and item["path"] == relative_path)
        _data, observation = _read_bound_file(path, description=f"Paddle {role} model asset")
        if observation["sha256"] != expected["sha256"] or observation["size_bytes"] != expected["size_bytes"]:
            raise TeacherContractError(
                f"model_assets.{role}/{relative_path} differs from adapter evidence at {location}"
            )
        observation["asset_role"] = role
        observation["asset_relative_path"] = relative_path
        observations.append(observation)
    return observations


def _verify_adapter_assets(
    adapter: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
    *,
    location: str,
) -> None:
    actual = _asset_file_map(adapter, location=location)
    expected_keys = {
        (str(observation["asset_role"]), str(observation["asset_relative_path"]))
        for observation in observations
    }
    if set(actual) != expected_keys:
        raise TeacherContractError(f"Paddle model asset membership changed before {location}")
    for observation in observations:
        _verify_observation(observation, description=f"Paddle model asset before {location}")


def _load_capture_file(
    path: Path,
    *,
    expected_rows: Mapping[str, Mapping[str, object]],
    expected_inventory_manifest_sha256: str,
    expected_inventory_contract_sha256: str,
) -> tuple[
    str,
    str,
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    data, observation = _read_bound_file(path, description="Paddle view capture")
    rows = _load_jsonl_bytes(data, source=str(observation["path"]))
    if not rows:
        raise TeacherContractError(f"Paddle view capture is empty: {observation['path']}")
    by_record_id: dict[str, dict[str, object]] = {}
    view_id: str | None = None
    view_contract_sha256: str | None = None
    view_contract_identity: dict[str, object] | None = None
    adapter_identity: dict[str, object] | None = None
    for line_number, row in enumerate(rows, start=1):
        location = f"{observation['path']}:{line_number}"
        if row.get("schema_version") != SCHEMA_VERSION or row.get("kind") != CAPTURE_KIND:
            raise TeacherContractError(f"unsupported capture row kind/schema at {location}")
        if row.get("inventory_manifest_sha256") != expected_inventory_manifest_sha256:
            raise TeacherContractError(f"capture inventory_manifest_sha256 differs at {location}")
        if row.get("inventory_contract_sha256") != expected_inventory_contract_sha256:
            raise TeacherContractError(f"capture inventory_contract_sha256 differs at {location}")
        row_view_id = row.get("view_id")
        if not isinstance(row_view_id, str) or _VIEW_ID_RE.fullmatch(row_view_id) is None:
            raise TeacherContractError(f"invalid view_id at {location}")
        if view_id is None:
            view_id = row_view_id
        elif view_id != row_view_id:
            raise TeacherContractError(f"capture file mixes view_id values at {location}")
        row_view_contract = _require_sha256(
            row.get("view_contract_sha256"),
            description=f"view_contract_sha256 at {location}",
        )
        if view_contract_sha256 is None:
            view_contract_sha256 = row_view_contract
        elif view_contract_sha256 != row_view_contract:
            raise TeacherContractError(f"capture file mixes view contracts at {location}")
        raw_view_contract = row.get("view_contract")
        if not isinstance(raw_view_contract, Mapping):
            raise TeacherContractError(f"view_contract must be an object at {location}")
        view_contract = dict(raw_view_contract)
        if (
            view_contract.get("kind") != VIEW_CONTRACT_KIND
            or view_contract.get("schema_version") != SCHEMA_VERSION
            or view_contract.get("view_id") != row_view_id
            or view_contract.get("quad_coordinate_space") != "exif_upright_source_normalized"
            or view_contract.get("line_order") != "top_to_bottom_left_to_right_v1"
        ):
            raise TeacherContractError(f"unsupported or inconsistent view_contract at {location}")
        expected_view_contract = canonical_view_contract(row_view_id)
        if view_contract != expected_view_contract:
            raise TeacherContractError(f"view_contract is not the exact canonical {row_view_id} recipe at {location}")
        if _canonical_sha256(view_contract) != row_view_contract:
            raise TeacherContractError(f"view_contract_sha256 does not bind view_contract at {location}")
        if view_contract_identity is None:
            view_contract_identity = view_contract
        elif view_contract_identity != view_contract:
            raise TeacherContractError(f"capture file mixes view_contract payloads at {location}")
        adapter = _validate_adapter_evidence(row.get("adapter"), location=location)
        if adapter_identity is None:
            adapter_identity = adapter
        elif adapter_identity != adapter:
            raise TeacherContractError(f"capture file mixes adapter identities at {location}")
        record_id = _require_sha256(row.get("record_id"), description=f"record_id at {location}")
        if record_id in by_record_id:
            raise TeacherContractError(f"duplicate capture record_id {record_id} at {location}")
        inventory = expected_rows.get(record_id)
        if inventory is None:
            raise TeacherContractError(f"capture contains record not pending in inventory: {record_id}")
        for key in ("group_id", "raw_sha256", "decoded_pixel_sha256"):
            if row.get(key) != inventory.get(key):
                raise TeacherContractError(f"capture {key} differs from inventory for {record_id} at {location}")
        transform_receipt = row.get("transform_receipt")
        if not isinstance(transform_receipt, Mapping):
            raise TeacherContractError(f"capture transform_receipt must be an object at {location}")
        if (
            transform_receipt.get("view_id") != row_view_id
            or transform_receipt.get("view_contract_sha256") != row_view_contract
            or transform_receipt.get("source_decoded_pixel_sha256") != inventory.get("decoded_pixel_sha256")
            or transform_receipt.get("coordinate_mapping") != "full_frame_scale_source_normalized_identity_v1"
        ):
            raise TeacherContractError(f"capture transform_receipt differs from inventory/view contract at {location}")
        transformed_sha = _require_sha256(
            transform_receipt.get("transformed_pixel_sha256"),
            description=f"transformed_pixel_sha256 at {location}",
        )
        for dimension in ("source_width", "source_height", "transformed_width", "transformed_height"):
            value = transform_receipt.get(dimension)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
                raise TeacherContractError(f"capture transform_receipt {dimension} must be an integer >1 at {location}")
        if row_view_id in {"original_rgb", "grayscale_clahe"} and (
            transform_receipt["transformed_width"] != transform_receipt["source_width"]
            or transform_receipt["transformed_height"] != transform_receipt["source_height"]
        ):
            raise TeacherContractError(f"capture transform dimensions violate canonical {row_view_id} recipe at {location}")
        if row_view_id == "upscale_sharpen" and (
            transform_receipt["transformed_width"] != 2 * transform_receipt["source_width"]
            or transform_receipt["transformed_height"] != 2 * transform_receipt["source_height"]
        ):
            raise TeacherContractError(f"capture transform dimensions violate canonical upscale recipe at {location}")
        if row_view_id == "original_rgb" and transformed_sha != inventory.get("decoded_pixel_sha256"):
            raise TeacherContractError(f"original_rgb transformed pixels differ from inventory decoded pixels at {location}")
        state = row.get("capture_state")
        if state not in {"ok", "error"}:
            raise TeacherContractError(f"invalid capture_state at {location}")
        if state == "ok" and not isinstance(row.get("lines"), list):
            raise TeacherContractError(f"successful capture lines must be an array at {location}")
        if state == "error" and row.get("lines") not in (None, []):
            raise TeacherContractError(f"failed capture must not contain OCR lines at {location}")
        if state == "ok":
            lines_value = row.get("lines")
            assert isinstance(lines_value, list)
            for count_name in (
                "raw_detected_line_count",
                "recognition_attempted_line_count",
                "recognition_rejected_line_count",
            ):
                count_value = row.get(count_name)
                if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value < 0:
                    raise TeacherContractError(f"capture {count_name} must be a non-negative integer at {location}")
            if row["raw_detected_line_count"] != len(lines_value):
                raise TeacherContractError(f"capture raw_detected_line_count differs from lines at {location}")
        by_record_id[record_id] = row
    expected_ids = set(expected_rows)
    observed_ids = set(by_record_id)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise TeacherContractError(
            f"capture coverage differs from pending inventory: missing={missing[:5]}, extra={extra[:5]}"
        )
    assert (
        view_id is not None
        and view_contract_sha256 is not None
        and view_contract_identity is not None
        and adapter_identity is not None
    )
    return view_id, view_contract_sha256, view_contract_identity, by_record_id, adapter_identity, observation


def _parse_quad(value: object, *, description: str, minimum_area: float) -> tuple[list[list[float]], tuple[float, float, float, float], float]:
    if not isinstance(value, list) or len(value) != 4:
        raise CandidateGateError("invalid_geometry", f"{description} must contain four points")
    points: list[list[float]] = []
    for point_index, raw_point in enumerate(value):
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise CandidateGateError("invalid_geometry", f"{description} point {point_index} must be [x,y]")
        try:
            x, y = float(raw_point[0]), float(raw_point[1])
        except (TypeError, ValueError):
            raise CandidateGateError("invalid_geometry", f"{description} point {point_index} must be numeric") from None
        if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise CandidateGateError("invalid_geometry", f"{description} point {point_index} is outside [0,1]")
        points.append([x, y])
    area_twice = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )
    area = abs(area_twice) / 2.0
    if area < minimum_area:
        raise CandidateGateError("invalid_geometry", f"{description} normalized area {area} is too small")
    crosses: list[float] = []
    for index in range(4):
        a = points[index]
        b = points[(index + 1) % 4]
        c = points[(index + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    epsilon = 1e-12
    if not (all(value > epsilon for value in crosses) or all(value < -epsilon for value in crosses)):
        raise CandidateGateError("invalid_geometry", f"{description} must be a strictly convex cyclic quad")
    if area_twice < 0.0:
        points.reverse()
    start_index = min(range(4), key=lambda index: (points[index][1], points[index][0]))
    points = points[start_index:] + points[:start_index]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = min(xs), min(ys), max(xs), max(ys)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise CandidateGateError("invalid_geometry", f"{description} has an empty bounding box")
    return points, bbox, area


def _evaluate_view(
    row: Mapping[str, object],
    *,
    adapter: Mapping[str, object],
    minimum_line_confidence: float,
    minimum_view_confidence: float,
    minimum_quad_area: float,
    maximum_lines: int,
    maximum_line_characters: int,
    maximum_document_characters: int,
) -> dict[str, object]:
    view_id = str(row["view_id"])
    diagnostic: dict[str, object] = {
        "view_id": view_id,
        "capture_state": row["capture_state"],
        "eligible": False,
        "reasons": [],
        "candidate_text": None,
        "candidate_sha256": None,
        "accepted_line_count": 0,
        "raw_line_count": 0,
        "minimum_line_confidence": None,
        "mean_line_confidence": None,
        "dropped_line_count": 0,
    }
    if row["capture_state"] == "error":
        error_value = row.get("error")
        diagnostic["reasons"] = [
            {
                "code": "capture_error",
                "detail": str(error_value)[:1024] if error_value is not None else "adapter reported capture failure",
            }
        ]
        return diagnostic
    raw_lines = row.get("lines")
    assert isinstance(raw_lines, list)
    diagnostic["raw_line_count"] = len(raw_lines)
    raw_detected_line_count = row.get("raw_detected_line_count")
    recognition_attempted_line_count = row.get("recognition_attempted_line_count")
    recognition_rejected_line_count = row.get("recognition_rejected_line_count")
    if (
        isinstance(raw_detected_line_count, bool)
        or not isinstance(raw_detected_line_count, int)
        or raw_detected_line_count != len(raw_lines)
        or isinstance(recognition_attempted_line_count, bool)
        or not isinstance(recognition_attempted_line_count, int)
        or recognition_attempted_line_count != raw_detected_line_count
        or isinstance(recognition_rejected_line_count, bool)
        or not isinstance(recognition_rejected_line_count, int)
        or recognition_rejected_line_count != 0
    ):
        diagnostic["reasons"] = [
            {
                "code": "incomplete_recognition_coverage",
                "detail": (
                    "capture must expose every raw DB line, attempt REC for each line, and report zero "
                    "recognition-rejected lines"
                ),
            }
        ]
        return diagnostic
    if len(raw_lines) > maximum_lines:
        diagnostic["reasons"] = [
            {"code": "semantic_invalid", "detail": f"capture has {len(raw_lines)} lines; maximum is {maximum_lines}"}
        ]
        return diagnostic

    accepted: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    reasons: list[dict[str, str]] = []
    drop_score = float(adapter["drop_score"])
    for position, raw_line in enumerate(raw_lines):
        try:
            if not isinstance(raw_line, Mapping):
                raise CandidateGateError("invalid_line", f"line {position} must be an object")
            index = raw_line.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise CandidateGateError("invalid_line", f"line {position} index must be a non-negative integer")
            if index in seen_indices:
                raise CandidateGateError("invalid_line", f"duplicate line index {index}")
            seen_indices.add(index)
            raw_text = raw_line.get("text")
            if not isinstance(raw_text, str):
                raise CandidateGateError("semantic_invalid", f"line {index} text must be a string")
            raw_error = _raw_text_error(raw_text, description=f"line {index}")
            if raw_error is not None:
                raise CandidateGateError("semantic_invalid", raw_error)
            confidence = _finite_unit_score(raw_line.get("confidence"), description=f"line {index} confidence")
            passes = raw_line.get("passes_drop_score")
            if not isinstance(passes, bool):
                raise CandidateGateError("invalid_confidence", f"line {index} passes_drop_score must be boolean")
            if passes != (confidence >= drop_score):
                raise CandidateGateError(
                    "invalid_confidence",
                    f"line {index} passes_drop_score disagrees with adapter drop_score {drop_score}",
                )
            if not passes:
                diagnostic["dropped_line_count"] = int(diagnostic["dropped_line_count"]) + 1
                reasons.append(
                    {
                        "code": "low_confidence",
                        "detail": f"line {index} confidence {confidence} is below adapter drop_score {drop_score}",
                    }
                )
                continue
            normalized_text = _normalise_text(raw_text)
            semantic_error = _semantic_text_error(
                normalized_text,
                max_characters=maximum_line_characters,
                description=f"line {index}",
            )
            if semantic_error is not None:
                raise CandidateGateError("semantic_invalid", semantic_error)
            quad_value = raw_line.get("quad_normalized")
            quad, bbox, area = _parse_quad(
                quad_value,
                description=f"line {index} quad_normalized",
                minimum_area=minimum_quad_area,
            )
            accepted.append(
                {
                    "index": index,
                    "raw_text": raw_text,
                    "text": normalized_text,
                    "confidence": confidence,
                    "quad_normalized": quad,
                    "bbox_normalized": list(bbox),
                    "quad_area_normalized": area,
                }
            )
        except CandidateGateError as error:
            reasons.append({"code": error.code, "detail": str(error)})
    accepted.sort(
        key=lambda line: (
            sum(float(point[1]) for point in line["quad_normalized"]) / 4.0,
            sum(float(point[0]) for point in line["quad_normalized"]) / 4.0,
            int(line["index"]),
        )
    )
    for canonical_index, line in enumerate(accepted):
        line["capture_index"] = line["index"]
        line["index"] = canonical_index
    if not accepted:
        reasons.append({"code": "empty_document", "detail": "no DB+CLS+REC line passed the adapter drop score"})
    document_text = "\n".join(str(line["text"]) for line in accepted)
    if document_text:
        diagnostic["candidate_text"] = document_text
        diagnostic["candidate_sha256"] = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
    document_semantic_error = _semantic_text_error(
        document_text.replace("\n", " "),
        max_characters=maximum_document_characters,
        description="document",
    )
    if document_semantic_error is not None and accepted:
        reasons.append({"code": "semantic_invalid", "detail": document_semantic_error})
    confidences = [float(line["confidence"]) for line in accepted]
    if confidences:
        minimum_confidence = min(confidences)
        mean_confidence = sum(confidences) / len(confidences)
        diagnostic["minimum_line_confidence"] = round(minimum_confidence, 8)
        diagnostic["mean_line_confidence"] = round(mean_confidence, 8)
        if minimum_confidence < minimum_line_confidence:
            reasons.append(
                {
                    "code": "low_confidence",
                    "detail": (
                        f"minimum accepted line confidence {minimum_confidence} is below "
                        f"{minimum_line_confidence}"
                    ),
                }
            )
        if mean_confidence < minimum_view_confidence:
            reasons.append(
                {
                    "code": "low_confidence",
                    "detail": f"mean accepted line confidence {mean_confidence} is below {minimum_view_confidence}",
                }
            )
    diagnostic["accepted_line_count"] = len(accepted)
    if seen_indices and seen_indices != set(range(len(raw_lines))):
        reasons.append(
            {
                "code": "invalid_line",
                "detail": "capture line indices must be a complete zero-based sequence",
            }
        )
    diagnostic["reasons"] = reasons
    diagnostic["eligible"] = not reasons
    diagnostic["lines"] = accepted
    return diagnostic


def _signed_polygon_area(points: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _line_intersection(
    start: Sequence[float],
    end: Sequence[float],
    clip_start: Sequence[float],
    clip_end: Sequence[float],
) -> list[float]:
    dx_subject = end[0] - start[0]
    dy_subject = end[1] - start[1]
    dx_clip = clip_end[0] - clip_start[0]
    dy_clip = clip_end[1] - clip_start[1]
    denominator = dx_subject * dy_clip - dy_subject * dx_clip
    if abs(denominator) <= 1e-15:
        return [float(end[0]), float(end[1])]
    offset_x = clip_start[0] - start[0]
    offset_y = clip_start[1] - start[1]
    parameter = (offset_x * dy_clip - offset_y * dx_clip) / denominator
    return [start[0] + parameter * dx_subject, start[1] + parameter * dy_subject]


def _convex_polygon_intersection(
    subject: Sequence[Sequence[float]],
    clip: Sequence[Sequence[float]],
) -> list[list[float]]:
    output = [[float(point[0]), float(point[1])] for point in subject]
    clip_sign = 1.0 if _signed_polygon_area(clip) > 0.0 else -1.0

    def inside(point: Sequence[float], edge_start: Sequence[float], edge_end: Sequence[float]) -> bool:
        cross = (
            (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
        )
        return clip_sign * cross >= -1e-12

    for edge_index, clip_start in enumerate(clip):
        clip_end = clip[(edge_index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        start = input_points[-1]
        for end in input_points:
            end_inside = inside(end, clip_start, clip_end)
            start_inside = inside(start, clip_start, clip_end)
            if end_inside:
                if not start_inside:
                    output.append(_line_intersection(start, end, clip_start, clip_end))
                output.append([float(end[0]), float(end[1])])
            elif start_inside:
                output.append(_line_intersection(start, end, clip_start, clip_end))
            start = end
    return output


def _quad_iou(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    left_area = abs(_signed_polygon_area(left))
    right_area = abs(_signed_polygon_area(right))
    intersection_points = _convex_polygon_intersection(left, right)
    intersection_area = abs(_signed_polygon_area(intersection_points)) if len(intersection_points) >= 3 else 0.0
    union = left_area + right_area - intersection_area
    return 0.0 if union <= 0.0 else intersection_area / union


def _view_pair_geometry(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    minimum_iou: float,
) -> tuple[bool, float]:
    left_lines = left.get("lines")
    right_lines = right.get("lines")
    if not isinstance(left_lines, list) or not isinstance(right_lines, list) or len(left_lines) != len(right_lines):
        return False, 0.0
    ious: list[float] = []
    for left_line, right_line in zip(left_lines, right_lines):
        if not isinstance(left_line, Mapping) or not isinstance(right_line, Mapping):
            return False, 0.0
        if left_line.get("text") != right_line.get("text"):
            return False, 0.0
        left_quad = left_line.get("quad_normalized")
        right_quad = right_line.get("quad_normalized")
        if not isinstance(left_quad, list) or not isinstance(right_quad, list):
            return False, 0.0
        ious.append(_quad_iou(left_quad, right_quad))
    minimum_observed = min(ious) if ious else 0.0
    return bool(ious) and minimum_observed >= minimum_iou, minimum_observed


def _geometry_support(
    voters: Sequence[dict[str, object]],
    *,
    minimum_iou: float,
) -> tuple[list[dict[str, object]], float] | None:
    compatible: dict[tuple[str, str], tuple[bool, float]] = {}
    for left, right in itertools.combinations(voters, 2):
        key = tuple(sorted((str(left["view_id"]), str(right["view_id"]))))
        compatible[key] = _view_pair_geometry(left, right, minimum_iou=minimum_iou)
    possible: list[tuple[list[dict[str, object]], float, float]] = []
    for size in range(len(voters), 1, -1):
        for subset in itertools.combinations(voters, size):
            pair_results = [
                compatible[tuple(sorted((str(left["view_id"]), str(right["view_id"]))))]
                for left, right in itertools.combinations(subset, 2)
            ]
            if pair_results and all(result[0] for result in pair_results):
                minimum_observed = min(result[1] for result in pair_results)
                confidence_sum = sum(float(item["mean_line_confidence"]) for item in subset)
                possible.append((list(subset), minimum_observed, confidence_sum))
        if possible:
            break
    if not possible:
        return None
    possible.sort(
        key=lambda item: (
            -len(item[0]),
            -item[2],
            tuple(sorted(str(view["view_id"]) for view in item[0])),
        )
    )
    support, minimum_observed, _confidence_sum = possible[0]
    return support, minimum_observed


def _quarantine_reason(diagnostics: Sequence[Mapping[str, object]]) -> str:
    eligible_keys = [item.get("candidate_sha256") for item in diagnostics if item.get("eligible") is True]
    eligible_counts = Counter(key for key in eligible_keys if isinstance(key, str))
    all_candidate_counts = Counter(
        item.get("candidate_sha256")
        for item in diagnostics
        if isinstance(item.get("candidate_sha256"), str)
    )
    all_reason_codes = {
        str(reason.get("code"))
        for item in diagnostics
        for reason in item.get("reasons", [])
        if isinstance(reason, Mapping)
    }
    if any(count >= 2 for count in all_candidate_counts.values()) and "low_confidence" in all_reason_codes:
        return "low_confidence"
    if eligible_counts and max(eligible_counts.values()) < 2 and len(eligible_counts) >= 2:
        return "text_conflict"
    for code, output in (
        ("incomplete_recognition_coverage", "incomplete_recognition_coverage"),
        ("low_confidence", "low_confidence"),
        ("invalid_geometry", "geometry_invalid"),
        ("semantic_invalid", "semantic_invalid"),
        ("invalid_confidence", "invalid_confidence"),
        ("capture_error", "capture_error"),
        ("empty_document", "empty_document"),
        ("invalid_line", "invalid_capture_line"),
    ):
        if code in all_reason_codes:
            return output
    if len(all_candidate_counts) >= 2:
        return "text_conflict"
    return "insufficient_valid_views"


def _public_view_diagnostic(diagnostic: Mapping[str, object]) -> dict[str, object]:
    return {
        key: diagnostic.get(key)
        for key in (
            "view_id",
            "capture_state",
            "eligible",
            "reasons",
            "candidate_text",
            "candidate_sha256",
            "accepted_line_count",
            "raw_line_count",
            "minimum_line_confidence",
            "mean_line_confidence",
            "dropped_line_count",
        )
    }


def _consensus_record(
    inventory: Mapping[str, object],
    capture_rows: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    minimum_line_confidence: float,
    minimum_view_confidence: float,
    minimum_geometry_iou: float,
    minimum_quad_area: float,
    maximum_lines: int,
    maximum_line_characters: int,
    maximum_document_characters: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    diagnostics = [
        _evaluate_view(
            row,
            adapter=adapter,
            minimum_line_confidence=minimum_line_confidence,
            minimum_view_confidence=minimum_view_confidence,
            minimum_quad_area=minimum_quad_area,
            maximum_lines=maximum_lines,
            maximum_line_characters=maximum_line_characters,
            maximum_document_characters=maximum_document_characters,
        )
        for row, adapter in capture_rows
    ]
    transformed_hashes = [
        str(dict(row["transform_receipt"])["transformed_pixel_sha256"])
        for row, _adapter in capture_rows
    ]
    if len(set(transformed_hashes)) != 3:
        return None, {
            "schema_version": SCHEMA_VERSION,
            "kind": QUARANTINE_RECORD_KIND,
            "record_id": inventory["record_id"],
            "group_id": inventory["group_id"],
            "suggested_split": inventory["suggested_split"],
            "source_absolute_path": inventory["source_absolute_path"],
            "raw_sha256": inventory["raw_sha256"],
            "decoded_pixel_sha256": inventory["decoded_pixel_sha256"],
            "quarantine_reason": "non_independent_view_pixels",
            "transformed_pixel_sha256_by_view": {
                str(row["view_id"]): dict(row["transform_receipt"])["transformed_pixel_sha256"]
                for row, _adapter in capture_rows
            },
            "view_diagnostics": [_public_view_diagnostic(item) for item in diagnostics],
            "training_eligible": False,
            "evaluation_only": False,
            "manual_review_required": False,
            "guessed_label_present": False,
        }
    eligible = [diagnostic for diagnostic in diagnostics if diagnostic["eligible"] is True]
    by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    for diagnostic in eligible:
        key = diagnostic.get("candidate_sha256")
        if isinstance(key, str):
            by_key[key].append(diagnostic)
    ordered = sorted(
        by_key.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    if not ordered or len(ordered[0][1]) < 2 or (len(ordered) > 1 and len(ordered[0][1]) == len(ordered[1][1])):
        return None, {
            "schema_version": SCHEMA_VERSION,
            "kind": QUARANTINE_RECORD_KIND,
            "record_id": inventory["record_id"],
            "group_id": inventory["group_id"],
            "suggested_split": inventory["suggested_split"],
            "source_absolute_path": inventory["source_absolute_path"],
            "raw_sha256": inventory["raw_sha256"],
            "decoded_pixel_sha256": inventory["decoded_pixel_sha256"],
            "quarantine_reason": _quarantine_reason(diagnostics),
            "view_diagnostics": [_public_view_diagnostic(item) for item in diagnostics],
            "training_eligible": False,
            "manual_review_required": False,
            "guessed_label_present": False,
        }
    dominant_key, voters = ordered[0]
    geometry = _geometry_support(voters, minimum_iou=minimum_geometry_iou)
    if geometry is None:
        return None, {
            "schema_version": SCHEMA_VERSION,
            "kind": QUARANTINE_RECORD_KIND,
            "record_id": inventory["record_id"],
            "group_id": inventory["group_id"],
            "suggested_split": inventory["suggested_split"],
            "source_absolute_path": inventory["source_absolute_path"],
            "raw_sha256": inventory["raw_sha256"],
            "decoded_pixel_sha256": inventory["decoded_pixel_sha256"],
            "quarantine_reason": "geometry_disagreement",
            "dominant_text_sha256": dominant_key,
            "dominant_text_votes": len(voters),
            "view_diagnostics": [_public_view_diagnostic(item) for item in diagnostics],
            "training_eligible": False,
            "manual_review_required": False,
            "guessed_label_present": False,
        }
    support, minimum_observed_iou = geometry
    support.sort(
        key=lambda item: (
            -float(item["mean_line_confidence"]),
            -float(item["minimum_line_confidence"]),
            str(item["view_id"]),
        )
    )
    chosen = support[0]
    chosen_lines = chosen.get("lines")
    assert isinstance(chosen_lines, list)
    supporting_view_ids = sorted(str(item["view_id"]) for item in support)
    dominant_view_ids = sorted(str(item["view_id"]) for item in voters)
    support_confidences = [
        {
            "view_id": item["view_id"],
            "minimum_line_confidence": item["minimum_line_confidence"],
            "mean_line_confidence": item["mean_line_confidence"],
        }
        for item in sorted(support, key=lambda item: str(item["view_id"]))
    ]
    output_lines = [
        {
            "index": index,
            "text": line["text"],
            "confidence": round(float(line["confidence"]), 8),
            "quad_normalized": [
                [round(float(point[0]), 8), round(float(point[1]), 8)]
                for point in line["quad_normalized"]
            ],
        }
        for index, line in enumerate(chosen_lines)
    ]
    split = str(inventory["suggested_split"])
    training_eligible = split == "train"
    accepted = {
        "schema_version": SCHEMA_VERSION,
        "kind": TEACHER_RECORD_KIND,
        "record_id": inventory["record_id"],
        "group_id": inventory["group_id"],
        "split": split,
        "split_use": "training" if training_eligible else f"heldout_{split}",
        "source_root": inventory["source_root"],
        "source_relative_path": inventory["source_relative_path"],
        "source_absolute_path": inventory["source_absolute_path"],
        "raw_sha256": inventory["raw_sha256"],
        "decoded_pixel_sha256": inventory["decoded_pixel_sha256"],
        "text": chosen["candidate_text"],
        "text_sha256": dominant_key,
        "text_normalization": "NFKC_then_collapse_line_whitespace_v1",
        "lines": output_lines,
        "label_source": "paddle_db_cls_rec_three_view_consensus",
        "consensus": {
            "dominant_text_votes": len(voters),
            "dominant_view_ids": dominant_view_ids,
            "geometry_support_votes": len(support),
            "geometry_support_view_ids": supporting_view_ids,
            "agreement": "3_of_3" if len(support) == 3 else "2_of_3",
            "chosen_geometry_view_id": chosen["view_id"],
            "minimum_pairwise_line_quad_iou": round(minimum_observed_iou, 8),
            "support_confidences": support_confidences,
        },
        "training_eligible": training_eligible,
        "evaluation_only": not training_eligible,
        "held_out": not training_eligible,
        "automatic_teacher_validation": True,
        "manual_review_required": False,
        "limitations": [
            "Paddle three-view consensus is a pseudo-label, not independent human ground truth",
            "accuracy acceptance still requires an independently frozen held-out evaluation",
        ],
    }
    return accepted, None


def _validate_options(
    *,
    minimum_line_confidence: float,
    minimum_view_confidence: float,
    minimum_geometry_iou: float,
    minimum_quad_area: float,
    maximum_lines: int,
    maximum_line_characters: int,
    maximum_document_characters: int,
) -> None:
    for name, value in (
        ("minimum_line_confidence", minimum_line_confidence),
        ("minimum_view_confidence", minimum_view_confidence),
        ("minimum_geometry_iou", minimum_geometry_iou),
        ("minimum_quad_area", minimum_quad_area),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise TeacherContractError(f"{name} must be finite and in [0,1]")
    if minimum_quad_area <= 0.0:
        raise TeacherContractError("minimum_quad_area must be positive")
    if minimum_line_confidence < DEFAULT_MIN_LINE_CONFIDENCE:
        raise TeacherContractError(
            f"minimum_line_confidence cannot be lower than fixed safety floor {DEFAULT_MIN_LINE_CONFIDENCE}"
        )
    if minimum_view_confidence < DEFAULT_MIN_VIEW_CONFIDENCE:
        raise TeacherContractError(
            f"minimum_view_confidence cannot be lower than fixed safety floor {DEFAULT_MIN_VIEW_CONFIDENCE}"
        )
    if minimum_geometry_iou < DEFAULT_MIN_GEOMETRY_IOU:
        raise TeacherContractError(
            f"minimum_geometry_iou cannot be lower than fixed safety floor {DEFAULT_MIN_GEOMETRY_IOU}"
        )
    if minimum_quad_area < DEFAULT_MIN_NORMALIZED_QUAD_AREA:
        raise TeacherContractError(
            f"minimum_quad_area cannot be lower than fixed safety floor {DEFAULT_MIN_NORMALIZED_QUAD_AREA}"
        )
    for name, value in (
        ("maximum_lines", maximum_lines),
        ("maximum_line_characters", maximum_line_characters),
        ("maximum_document_characters", maximum_document_characters),
    ):
        if value <= 0:
            raise TeacherContractError(f"{name} must be positive")
    for name, value, safety_ceiling in (
        ("maximum_lines", maximum_lines, DEFAULT_MAX_LINES),
        ("maximum_line_characters", maximum_line_characters, DEFAULT_MAX_LINE_CHARACTERS),
        ("maximum_document_characters", maximum_document_characters, DEFAULT_MAX_DOCUMENT_CHARACTERS),
    ):
        if value > safety_ceiling:
            raise TeacherContractError(f"{name} cannot exceed fixed safety ceiling {safety_ceiling}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_json_line(row))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact_binding(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def build_paddle_teacher_consensus(
    *,
    inventory_manifest: Path,
    view_results: Sequence[Path],
    output_dir: Path,
    inventory_contract: Path | None = None,
    minimum_line_confidence: float = DEFAULT_MIN_LINE_CONFIDENCE,
    minimum_view_confidence: float = DEFAULT_MIN_VIEW_CONFIDENCE,
    minimum_geometry_iou: float = DEFAULT_MIN_GEOMETRY_IOU,
    minimum_quad_area: float = DEFAULT_MIN_NORMALIZED_QUAD_AREA,
    maximum_lines: int = DEFAULT_MAX_LINES,
    maximum_line_characters: int = DEFAULT_MAX_LINE_CHARACTERS,
    maximum_document_characters: int = DEFAULT_MAX_DOCUMENT_CHARACTERS,
) -> dict[str, object]:
    """Build and seal three-view automatic Paddle teacher evidence."""
    _validate_options(
        minimum_line_confidence=minimum_line_confidence,
        minimum_view_confidence=minimum_view_confidence,
        minimum_geometry_iou=minimum_geometry_iou,
        minimum_quad_area=minimum_quad_area,
        maximum_lines=maximum_lines,
        maximum_line_characters=maximum_line_characters,
        maximum_document_characters=maximum_document_characters,
    )
    if len(view_results) != 3:
        raise TeacherContractError("exactly three Paddle view-result JSONL files are required")
    resolved_view_paths = [Path(path).expanduser().resolve(strict=True) for path in view_results]
    if len({os.path.normcase(str(path)) for path in resolved_view_paths}) != 3:
        raise TeacherContractError("the three Paddle view-result paths must be distinct")
    inventory_rows, inventory_payload, inventory_observations = load_inventory_for_teacher(
        inventory_manifest,
        contract_path=inventory_contract,
    )
    pending_rows = {
        str(row["record_id"]): row
        for row in inventory_rows
        if row["teacher_state"] == "pending"
    }
    inventory_quarantine = [row for row in inventory_rows if row["teacher_state"] == "quarantine"]
    if not pending_rows:
        raise TeacherContractError("inventory contains no pending Paddle teacher rows")

    captures: list[dict[str, object]] = []
    capture_observations: list[dict[str, object]] = []
    for path in resolved_view_paths:
        view_id, view_contract_sha, view_contract, rows, adapter, observation = _load_capture_file(
            path,
            expected_rows=pending_rows,
            expected_inventory_manifest_sha256=str(inventory_observations[0]["sha256"]),
            expected_inventory_contract_sha256=str(inventory_observations[1]["sha256"]),
        )
        captures.append(
            {
                "view_id": view_id,
                "view_contract_sha256": view_contract_sha,
                "view_contract": view_contract,
                "rows": rows,
                "adapter": adapter,
                "observation": observation,
            }
        )
        capture_observations.append(observation)
    view_ids = [str(item["view_id"]) for item in captures]
    if set(view_ids) != set(CANONICAL_VIEW_OPERATIONS):
        raise TeacherContractError(
            "capture set must contain exactly canonical views "
            f"{sorted(CANONICAL_VIEW_OPERATIONS)}, observed={sorted(view_ids)}"
        )
    view_contracts = [str(item["view_contract_sha256"]) for item in captures]
    if len(set(view_contracts)) != 3:
        raise TeacherContractError("three-view capture must bind three distinct view transform contracts")
    model_identities = {_canonical_sha256(dict(item["adapter"])) for item in captures}
    if len(model_identities) != 1:
        raise TeacherContractError("three views must use one identical Paddle model/drop-score contract")
    adapter_identity = dict(captures[0]["adapter"])
    model_asset_observations = _observe_adapter_assets(
        adapter_identity,
        location="teacher aggregation preflight",
    )

    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_dir))))
    _require_no_reparse_ancestors(output, include_leaf=False)
    if output.exists():
        raise FileExistsError(f"teacher output directory must be brand-new: {output}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"teacher output parent must already exist: {output.parent}")
    output_parent_identity = _bind_output_parent(output)
    input_files = [Path(str(item["path"])) for item in inventory_observations + capture_observations]
    if any(_paths_overlap(output, path) for path in input_files):
        raise TeacherContractError("teacher output must be disjoint from inventory and capture inputs")
    inventory_directory = Path(str(inventory_observations[0]["path"])).parent
    if _paths_overlap(output, inventory_directory):
        raise TeacherContractError("teacher output must be disjoint from the complete inventory publication directory")
    capture_directories = {Path(str(item["path"])).parent for item in capture_observations}
    if any(_paths_overlap(output, directory) for directory in capture_directories):
        raise TeacherContractError("teacher output must be disjoint from complete capture publication directories")
    model_assets = adapter_identity["model_assets"]
    assert isinstance(model_assets, Mapping)
    for role, asset_value in model_assets.items():
        assert isinstance(asset_value, Mapping)
        asset_path = Path(str(asset_value["path"])).expanduser().resolve(strict=True)
        protected_path = asset_path if asset_path.is_dir() else asset_path.parent
        if _paths_overlap(output, protected_path):
            raise TeacherContractError(f"teacher output must be disjoint from Paddle {role} model assets")

    source_observations: list[dict[str, object]] = []
    source_roots = {
        Path(str(row["source_root"])).expanduser().resolve(strict=True)
        for row in inventory_rows
    }
    if any(_paths_overlap(output, source_root) for source_root in source_roots):
        raise TeacherContractError("teacher output must be disjoint from every inventory source_root")
    for row in sorted(inventory_rows, key=lambda item: str(item["record_id"])):
        source_path = Path(str(row["source_absolute_path"]))
        if _paths_overlap(output, source_path.parent):
            raise TeacherContractError("teacher output must be disjoint from source image directories")
        source_observations.append(
            _observe_source_file(source_path, expected_sha256=str(row["raw_sha256"]))
        )

    recomputed_transform_hashes: dict[tuple[str, str], str] = {}
    from .otherimages_paddle_capture import PaddleViewContract, _canonical_transform, _load_bound_upright_rgb, _pixel_sha256

    for record_id, inventory in sorted(pending_rows.items()):
        source_rgb, _source_observation = _load_bound_upright_rgb(
            Path(str(inventory["source_absolute_path"])),
            expected_raw_sha256=str(inventory["raw_sha256"]),
        )
        if _pixel_sha256(source_rgb) != inventory["decoded_pixel_sha256"]:
            raise TeacherContractError(
                f"source decoded pixels differ from inventory during teacher transform closure: {record_id}"
            )
        for capture in captures:
            view_payload = dict(capture["view_contract"])
            view = PaddleViewContract(
                view_id=str(capture["view_id"]),
                operations=tuple(str(item) for item in view_payload["operations"]),
            )
            recomputed_transform_hashes[(record_id, view.view_id)] = _pixel_sha256(
                _canonical_transform(source_rgb, view)
            )
    for capture in captures:
        view_id = str(capture["view_id"])
        capture_rows = dict(capture["rows"])
        for record_id, row in capture_rows.items():
            observed = str(dict(row["transform_receipt"])["transformed_pixel_sha256"])
            expected = recomputed_transform_hashes[(record_id, view_id)]
            if observed != expected:
                raise TeacherContractError(
                    f"capture transformed pixels do not reproduce canonical recipe for {record_id}/{view_id}"
                )

    accepted: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    for row in inventory_quarantine:
        quarantined.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": QUARANTINE_RECORD_KIND,
                "record_id": row["record_id"],
                "group_id": row["group_id"],
                "suggested_split": row["suggested_split"],
                "source_absolute_path": row["source_absolute_path"],
                "raw_sha256": row["raw_sha256"],
                "decoded_pixel_sha256": row["decoded_pixel_sha256"],
                "quarantine_reason": "inventory_quarantine",
                "inventory_quarantine_reason": row.get("quarantine_reason"),
                "training_eligible": False,
                "manual_review_required": False,
                "guessed_label_present": False,
            }
        )
    captures.sort(key=lambda item: str(item["view_id"]))
    for record_id, inventory in sorted(pending_rows.items()):
        capture_rows = [
            (
                dict(item["rows"])[record_id],
                dict(item["adapter"]),
            )
            for item in captures
        ]
        teacher, rejection = _consensus_record(
            inventory,
            capture_rows,
            minimum_line_confidence=minimum_line_confidence,
            minimum_view_confidence=minimum_view_confidence,
            minimum_geometry_iou=minimum_geometry_iou,
            minimum_quad_area=minimum_quad_area,
            maximum_lines=maximum_lines,
            maximum_line_characters=maximum_line_characters,
            maximum_document_characters=maximum_document_characters,
        )
        if teacher is not None:
            accepted.append(teacher)
        else:
            assert rejection is not None
            quarantined.append(rejection)
    accepted.sort(key=lambda row: str(row["record_id"]))
    quarantined.sort(key=lambda row: str(row["record_id"]))

    all_observations = (
        inventory_observations
        + capture_observations
        + source_observations
        + model_asset_observations
    )
    for observation in all_observations:
        _verify_observation(observation, description="teacher input")

    _verify_output_parent(output, output_parent_identity, location="teacher staging")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.teacher-building-", dir=output.parent))
    stage_identity = _bind_stage_identity(stage, directory=True)
    published = False
    try:
        _write_jsonl(stage / "teacher_manifest.jsonl", accepted)
        _write_jsonl(stage / "reject_manifest.jsonl", quarantined)
        artifacts = [_artifact_binding(stage / name) for name in _OUTPUT_FILES]
        reason_counts = Counter(str(row["quarantine_reason"]) for row in quarantined)
        source_closure_rows = sorted(
            (
                {
                    "path": str(item["path"]),
                    "sha256": str(item["sha256"]),
                    "size_bytes": int(item["size_bytes"]),
                }
                for item in source_observations
            ),
            key=lambda item: (item["sha256"], item["path"]),
        )
        configuration = {
            "minimum_line_confidence": minimum_line_confidence,
            "minimum_view_confidence": minimum_view_confidence,
            "minimum_geometry_iou": minimum_geometry_iou,
            "minimum_normalized_quad_area": minimum_quad_area,
            "maximum_lines": maximum_lines,
            "maximum_line_characters": maximum_line_characters,
            "maximum_document_characters": maximum_document_characters,
            "text_normalization": "NFKC_then_collapse_line_whitespace_v1",
            "consensus": "unique_dominant_exact_normalized_text_with_two_or_three_geometry_compatible_views_v1",
        }
        inputs = {
            "inventory_manifest": _public_binding(inventory_observations[0]),
            "inventory_contract": _public_binding(inventory_observations[1]),
            "inventory_contract_kind": inventory_payload["kind"],
            "views": [
                {
                    "view_id": item["view_id"],
                    "view_contract_sha256": item["view_contract_sha256"],
                    "view_contract": item["view_contract"],
                    "result": _public_binding(dict(item["observation"])),
                    "adapter": item["adapter"],
                }
                for item in captures
            ],
            "source_images": {
                "records": len(source_closure_rows),
                "closure_sha256": _canonical_sha256(source_closure_rows),
                "raw_sha256_rechecked_before_publication": True,
                "canonical_transform_pixel_sha256_recomputed": True,
            },
            "model_assets": {
                "adapter_contract_sha256": adapter_identity["model_contract_sha256"],
                "asset_file_records": len(model_asset_observations),
                "filesystem_membership_rechecked_before_publication": True,
                "sha256_rechecked_before_publication": True,
            },
        }
        counts = {
            "inventory_records": len(inventory_rows),
            "pending_records": len(pending_rows),
            "accepted_teacher_records": len(accepted),
            "quarantined_records": len(quarantined),
            "quarantine_reasons": dict(sorted(reason_counts.items())),
            "accepted_by_split": dict(sorted(Counter(str(row["split"]) for row in accepted).items())),
            "training_eligible_records": sum(bool(row["training_eligible"]) for row in accepted),
            "evaluation_only_records": sum(bool(row["evaluation_only"]) for row in accepted),
        }
        split_use = {
            "train": "training_eligible",
            "val": "heldout_evaluation_only",
            "test": "heldout_evaluation_only",
            "group_split_source": "inventory_suggested_split",
            "groups_may_cross_splits": False,
        }
        closure_payload = {
            "schema_version": SCHEMA_VERSION,
            "inputs": inputs,
            "configuration": configuration,
            "counts": counts,
            "split_use": split_use,
            "artifacts": artifacts,
        }
        contract: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": TEACHER_CONTRACT_KIND,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sealed": True,
            "output_directory": str(output),
            "inputs": inputs,
            "configuration": configuration,
            "counts": counts,
            "split_use": split_use,
            "artifacts": artifacts,
            "closure_sha256": _canonical_sha256(closure_payload),
            "training_authorization": False,
            "ocr_execution_performed_by_this_module": False,
            "manual_review_required": False,
            "low_confidence_or_conflict_policy": "quarantine_never_guess",
            "limitations": [
                "Teacher consensus is not independent accuracy truth",
                "A later training run requires separate explicit authorization and a frozen held-out accuracy gate",
            ],
        }
        _write_json(stage / "teacher.contract.json", contract)
        contract_binding = _artifact_binding(stage / "teacher.contract.json")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": TEACHER_RECEIPT_KIND,
            "sealed": True,
            "contract": contract_binding,
            "contract_closure_sha256": contract["closure_sha256"],
        }
        _write_json(stage / "teacher.receipt.json", receipt)

        for expected in artifacts:
            if _artifact_binding(stage / str(expected["path"])) != expected:
                raise TeacherContractError(f"teacher artifact changed after contract creation: {expected['path']}")
        if _artifact_binding(stage / "teacher.contract.json") != contract_binding:
            raise TeacherContractError("teacher contract changed after receipt creation")
        _verify_adapter_assets(
            adapter_identity,
            model_asset_observations,
            location="teacher publication",
        )
        for observation in all_observations:
            _verify_observation(observation, description="teacher input at final closure")
        if output.exists():
            raise FileExistsError(f"teacher output directory appeared during build: {output}")
        _verify_output_parent(output, output_parent_identity, location="teacher publication")
        _rename_directory_no_replace(
            stage,
            output,
            expected_parent_identity=output_parent_identity,
            expected_stage_identity=stage_identity,
        )
        if _bind_stage_identity(output, directory=True) != stage_identity:
            raise TeacherContractError("published teacher directory differs from bound stage")
        if {path.name for path in output.iterdir()} != {
            "teacher_manifest.jsonl",
            "reject_manifest.jsonl",
            "teacher.contract.json",
            "teacher.receipt.json",
        }:
            raise TeacherContractError("published teacher directory membership differs after publication")
        for expected in artifacts:
            if _artifact_binding(output / str(expected["path"])) != expected:
                raise TeacherContractError(f"published teacher artifact failed readback: {expected['path']}")
        if _artifact_binding(output / "teacher.contract.json") != contract_binding:
            raise TeacherContractError("published teacher contract failed readback")
        published_receipt = _load_json_object_bytes(
            (output / "teacher.receipt.json").read_bytes(),
            source=str(output / "teacher.receipt.json"),
        )
        if published_receipt != receipt:
            raise TeacherContractError("published teacher receipt failed readback")
        _verify_output_parent(output, output_parent_identity, location="teacher post-publication readback")
        if _bind_stage_identity(output, directory=True) != stage_identity:
            raise TeacherContractError("published teacher identity changed during readback")
        published = True
        return contract
    finally:
        if not published:
            # Preserve failed teacher evidence.  Do not recursively delete a
            # path that may have been replaced concurrently.
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline three-view Paddle DB+CLS+REC consensus and sealed teacher-manifest producer"
    )
    parser.add_argument("--inventory", type=Path, required=True, help="inventory paddle_teacher_pending.jsonl")
    parser.add_argument("--inventory-contract", type=Path, help="sibling inventory.contract.json by default")
    parser.add_argument(
        "--view-result",
        type=Path,
        action="append",
        required=True,
        help="one complete captured Paddle view JSONL; repeat exactly three times",
    )
    parser.add_argument("--output", type=Path, required=True, help="brand-new sealed teacher evidence directory")
    parser.add_argument("--min-line-confidence", type=float, default=DEFAULT_MIN_LINE_CONFIDENCE)
    parser.add_argument("--min-view-confidence", type=float, default=DEFAULT_MIN_VIEW_CONFIDENCE)
    parser.add_argument("--min-geometry-iou", type=float, default=DEFAULT_MIN_GEOMETRY_IOU)
    parser.add_argument("--min-quad-area", type=float, default=DEFAULT_MIN_NORMALIZED_QUAD_AREA)
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--max-line-characters", type=int, default=DEFAULT_MAX_LINE_CHARACTERS)
    parser.add_argument("--max-document-characters", type=int, default=DEFAULT_MAX_DOCUMENT_CHARACTERS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    try:
        contract = build_paddle_teacher_consensus(
            inventory_manifest=arguments.inventory,
            inventory_contract=arguments.inventory_contract,
            view_results=arguments.view_result,
            output_dir=arguments.output,
            minimum_line_confidence=arguments.min_line_confidence,
            minimum_view_confidence=arguments.min_view_confidence,
            minimum_geometry_iou=arguments.min_geometry_iou,
            minimum_quad_area=arguments.min_quad_area,
            maximum_lines=arguments.max_lines,
            maximum_line_characters=arguments.max_line_characters,
            maximum_document_characters=arguments.max_document_characters,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OtherImages Paddle teacher consensus failed:\n{error}") from None
    counts = dict(contract["counts"])
    print(
        f"Sealed {counts['accepted_teacher_records']} Paddle teacher record(s); "
        f"quarantined {counts['quarantined_records']} record(s) at {Path(arguments.output).absolute()}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
