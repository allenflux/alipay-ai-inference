"""Score the .NET white-document CPU route against sealed Paddle teacher consensus.

The teacher manifest is a pseudo-label source, not independent human truth.  This
tool therefore reports *teacher agreement*: held-out coverage, normalized text
exact match, character error rate (CER), and exact-line precision/recall.  A
formal pass is fail-closed on the sealed teacher contract, source hashes, the
exact .NET result set, CPU/provider evidence, and configurable agreement floors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import ntpath
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TEACHER_RECORD_KIND = "otherimages_paddle_teacher_record_v1"
TEACHER_CONTRACT_KIND = "otherimages_paddle_teacher_contract_v1"
TEACHER_RECEIPT_KIND = "otherimages_paddle_teacher_receipt_v1"
SUMMARY_KIND = "otherimages_dotnet_white_teacher_agreement_v1"
DEFAULT_MAX_CER = 0.05
DEFAULT_MIN_DOCUMENT_EXACT = 0.90
DEFAULT_MIN_LINE_PRECISION = 0.90
DEFAULT_MIN_LINE_RECALL = 0.90
DEFAULT_MAX_THREE_OF_THREE_CER = 0.03
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
REQUIRED_MODEL_HASH_FIELDS = frozenset(
    {
        "device_sha256",
        "device_contract_sha256",
        "ocr_bundle_contract_sha256",
        "ocr_source_audit_contract_sha256",
        "ocr_detector_sha256",
        "ocr_classifier_sha256",
        "ocr_recognizer_sha256",
        "ocr_dictionary_sha256",
        "ocr_dictionary_snapshot_sha256",
        "white_student_model_sha256",
        "white_student_charset_sha256",
        "white_student_contract_sha256",
    }
)
REQUIRED_STUDENT_SIZE_FIELDS = frozenset(
    {
        "white_student_model_snapshot_size_bytes",
        "white_student_charset_snapshot_size_bytes",
        "white_student_contract_snapshot_size_bytes",
    }
)
REQUIRED_STUDENT_NAME_FIELDS = frozenset(
    {"white_student_model", "white_student_charset", "white_student_contract"}
)
STUDENT_CROP_SOURCE = "same_paddle_db_cls_oriented_crop"
TEACHER_PUBLICATION_FILES = frozenset(
    {
        "teacher_manifest.jsonl",
        "reject_manifest.jsonl",
        "teacher.contract.json",
        "teacher.receipt.json",
    }
)
TEACHER_VIEW_IDS = frozenset({"original_rgb", "grayscale_clahe", "upscale_sharpen"})
TEXT_NORMALIZATION = "NFKC_then_collapse_line_whitespace_v1"
CANONICAL_VIEW_OPERATIONS: dict[str, tuple[str, ...]] = {
    "original_rgb": ("pillow_exif_transpose", "pillow_convert_rgb8", "identity"),
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


class WhiteEvaluationError(ValueError):
    """Raised when teacher or .NET evidence is malformed or ambiguous."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def _binding_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(expected.get(name) == observed.get(name) for name in ("path", "sha256", "size_bytes", "line_count"))


def _require_nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise WhiteEvaluationError(f"{description} must be a non-empty string")
    return value


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise WhiteEvaluationError(f"{description} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WhiteEvaluationError(f"{description} must be a non-negative integer")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_view_contract(view_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "otherimages_paddle_view_contract_v1",
        "view_id": view_id,
        "operations": list(CANONICAL_VIEW_OPERATIONS[view_id]),
        "quad_coordinate_space": "exif_upright_source_normalized",
        "line_order": "top_to_bottom_left_to_right_v1",
        "transform_implementation": "otherimages_paddle_capture_core_v1",
        "paddle_color_contract": {
            "kind": "otherimages_paddle_rgb_byte_order_contract_v1",
            "input_color_order": "RGB_passthrough_to_paddle_v2",
            "line_crop_color_order": "RGB_passthrough_through_db_crop_cls_rec_v1",
            "pixel_layout": "HxWx3_uint8_byte0_R_byte1_G_byte2_B",
            "channel_conversion": "none",
        },
    }


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exception:
        raise WhiteEvaluationError(f"cannot read {description} {path}: {exception}") from exception


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exception:
        raise WhiteEvaluationError(f"cannot read {description} {path}: {exception}") from exception
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise WhiteEvaluationError(f"blank line in {description} at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exception:
            raise WhiteEvaluationError(
                f"invalid JSON in {description} at {path}:{line_number}: {exception.msg}"
            ) from exception
        if not isinstance(value, dict):
            raise WhiteEvaluationError(f"non-object row in {description} at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_atomic(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n" for row in rows),
    )


def _require_fresh_output_target(path: Path) -> None:
    if os.path.lexists(path):
        raise WhiteEvaluationError(
            f"evaluation output directory already exists and cannot be reused: {path}"
        )


def _reserve_fresh_output_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exception:
        raise WhiteEvaluationError(
            f"evaluation output directory already exists and cannot be reused: {path}"
        ) from exception
    except OSError as exception:
        raise WhiteEvaluationError(
            f"cannot create fresh evaluation output directory {path}: {exception}"
        ) from exception


def _path_key(value: str) -> str:
    if WINDOWS_ABSOLUTE_PATH.match(value):
        return ntpath.normcase(ntpath.normpath(value.replace("/", "\\")))
    return os.path.normcase(os.path.abspath(os.path.expanduser(value)))


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _normalised_lines(
    value: Any,
    description: str,
    *,
    accepted_only: bool,
    require_teacher_orientation: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise WhiteEvaluationError(f"{description} lines must be an array")
    output: list[str] = []
    for index, line in enumerate(value):
        if not isinstance(line, Mapping):
            raise WhiteEvaluationError(f"{description} line {index} must be an object")
        if accepted_only and line.get("passes_drop_score") is not True:
            continue
        if require_teacher_orientation:
            orientation = line.get("orientation_degrees")
            if isinstance(orientation, bool) or orientation not in {0, 180}:
                raise WhiteEvaluationError(
                    f"{description} line {index} orientation_degrees must be 0 or 180"
                )
        text = line.get("text")
        if not isinstance(text, str):
            raise WhiteEvaluationError(f"{description} line {index} text must be a string")
        normalized = _normalise_text(text)
        if normalized:
            output.append(normalized)
    return output


def _student_lines(value: Any, description: str) -> list[str]:
    """Return accepted DB-line student text; require student evidence on every DB line."""
    if not isinstance(value, list):
        raise WhiteEvaluationError(f"{description} lines must be an array")
    output: list[str] = []
    for index, line in enumerate(value):
        if not isinstance(line, Mapping):
            raise WhiteEvaluationError(f"{description} line {index} must be an object")
        student = line.get("student")
        if not isinstance(student, Mapping):
            raise WhiteEvaluationError(f"{description} line {index} has no student evidence")
        text = student.get("text")
        confidence = student.get("confidence")
        if not isinstance(text, str):
            raise WhiteEvaluationError(f"{description} line {index} student.text must be a string")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise WhiteEvaluationError(
                f"{description} line {index} student.confidence must be finite and in [0,1]"
            )
        if (
            student.get("provider") != "cpu"
            or student.get("delivery_policy") != "review_only"
            or student.get("crop_source") != STUDENT_CROP_SOURCE
        ):
            raise WhiteEvaluationError(
                f"{description} line {index} student provider/delivery/crop contract is invalid"
            )
        parent_text = line.get("text")
        expected_exact = (
            isinstance(parent_text, str)
            and _normalise_text(parent_text) == _normalise_text(text)
        )
        if student.get("normalized_exact_match") is not expected_exact:
            raise WhiteEvaluationError(
                f"{description} line {index} student normalized_exact_match is inconsistent"
            )
        if line.get("passes_drop_score") is True:
            normalized = _normalise_text(text)
            if normalized:
                output.append(normalized)
    return output


def _validate_student_bundle_contracts(
    contracts: Mapping[str, Any], description: str
) -> dict[str, Any]:
    if any(
        not isinstance(contracts.get(name), str) or not contracts.get(name)
        for name in REQUIRED_STUDENT_NAME_FIELDS
    ):
        raise WhiteEvaluationError(f"{description} student bundle file names are incomplete")
    hashes = {name: contracts.get(name) for name in REQUIRED_MODEL_HASH_FIELDS}
    if any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in hashes.values()):
        raise WhiteEvaluationError(f"{description} model/student SHA closure is incomplete")
    for name in REQUIRED_STUDENT_SIZE_FIELDS:
        value = contracts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WhiteEvaluationError(f"{description} {name} must be a positive integer")
    if (
        contracts.get("white_student_runtime_source") != "immutable_verified_bytes"
        or contracts.get("white_student_reopened_paths_after_verification") is not False
    ):
        raise WhiteEvaluationError(f"{description} student immutable snapshot closure is invalid")
    return {
        **{name: str(contracts[name]) for name in REQUIRED_STUDENT_NAME_FIELDS},
        **{name: str(value) for name, value in hashes.items()},
        **{name: int(contracts[name]) for name in REQUIRED_STUDENT_SIZE_FIELDS},
        "white_student_runtime_source": "immutable_verified_bytes",
        "white_student_reopened_paths_after_verification": False,
    }


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    teacher_characters = sum(int(row["teacher_characters"]) for row in rows)
    edit_distance = sum(int(row["edit_distance"]) for row in rows)
    expected_lines = sum(int(row["teacher_lines"]) for row in rows)
    predicted_lines = sum(int(row["predicted_lines"]) for row in rows)
    matching_lines = sum(int(row["matching_lines"]) for row in rows)
    cer = _fraction(edit_distance, teacher_characters)
    return {
        "records": len(rows),
        "document_exact_matches": sum(bool(row["document_exact_match"]) for row in rows),
        "document_exact_match": _fraction(
            sum(bool(row["document_exact_match"]) for row in rows), len(rows)
        ),
        "teacher_characters": teacher_characters,
        "edit_distance": edit_distance,
        "character_error_rate": cer,
        "character_agreement": None if cer is None else max(0.0, 1.0 - cer),
        "teacher_lines": expected_lines,
        "predicted_lines": predicted_lines,
        "matching_lines": matching_lines,
        "line_exact_precision": _fraction(matching_lines, predicted_lines),
        "line_exact_recall": _fraction(matching_lines, expected_lines),
    }


def _verify_teacher_publication(
    contract_path: Path, teacher_manifest: Path, teacher_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    publication = teacher_manifest.parent
    expected_contract = publication / "teacher.contract.json"
    if contract_path != expected_contract:
        raise WhiteEvaluationError(
            "teacher contract must be teacher.contract.json beside the supplied teacher manifest"
        )
    try:
        members = {path.name for path in publication.iterdir()}
    except OSError as exception:
        raise WhiteEvaluationError(f"cannot inspect teacher publication {publication}: {exception}") from exception
    if members != TEACHER_PUBLICATION_FILES:
        raise WhiteEvaluationError(
            "sealed teacher publication membership differs: "
            f"expected={sorted(TEACHER_PUBLICATION_FILES)}, observed={sorted(members)}"
        )
    for name in TEACHER_PUBLICATION_FILES:
        member = publication / name
        if not member.is_file() or member.is_symlink():
            raise WhiteEvaluationError(f"sealed teacher publication member is not a regular file: {member}")

    contract = _load_json(contract_path, "teacher contract")
    if not isinstance(contract, Mapping):
        raise WhiteEvaluationError("teacher contract must be one JSON object")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != TEACHER_CONTRACT_KIND
        or contract.get("sealed") is not True
    ):
        raise WhiteEvaluationError("teacher contract is not a sealed OtherImages Paddle teacher contract")
    try:
        contracted_output = Path(
            _require_nonempty_string(contract.get("output_directory"), "teacher output_directory")
        ).expanduser().resolve(strict=True)
    except OSError as exception:
        raise WhiteEvaluationError(f"teacher output_directory cannot be resolved: {exception}") from exception
    if contracted_output != publication:
        raise WhiteEvaluationError("teacher contract output_directory does not bind the supplied publication")
    if contract.get("training_authorization") is not False:
        raise WhiteEvaluationError("teacher contract must retain training_authorization=false")
    split_use = contract.get("split_use")
    if (
        not isinstance(split_use, Mapping)
        or split_use.get("train") != "training_eligible"
        or split_use.get("val") != "heldout_evaluation_only"
        or split_use.get("test") != "heldout_evaluation_only"
        or split_use.get("groups_may_cross_splits") is not False
    ):
        raise WhiteEvaluationError("teacher contract does not forbid groups crossing splits")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, list):
        raise WhiteEvaluationError("teacher contract artifacts must be an array")
    artifact_by_name: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise WhiteEvaluationError("teacher contract artifact must be an object")
        name = artifact.get("path")
        if name not in {"teacher_manifest.jsonl", "reject_manifest.jsonl"} or name in artifact_by_name:
            raise WhiteEvaluationError("teacher contract artifacts must bind each teacher manifest exactly once")
        artifact_by_name[str(name)] = artifact
    if set(artifact_by_name) != {"teacher_manifest.jsonl", "reject_manifest.jsonl"}:
        raise WhiteEvaluationError("teacher contract must bind accepted and rejected manifests exactly")
    observed_manifest = _artifact_binding(teacher_manifest)
    reject_manifest = publication / "reject_manifest.jsonl"
    observed_reject = _artifact_binding(reject_manifest)
    if not _binding_matches(observed_manifest, artifact_by_name["teacher_manifest.jsonl"]):
        raise WhiteEvaluationError("teacher manifest identity differs from its sealed contract")
    if not _binding_matches(observed_reject, artifact_by_name["reject_manifest.jsonl"]):
        raise WhiteEvaluationError("teacher reject manifest identity differs from its sealed contract")
    rejected_rows = _load_jsonl(reject_manifest, "teacher reject manifest")
    counts = contract.get("counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("accepted_teacher_records") != len(teacher_rows)
        or counts.get("quarantined_records") != len(rejected_rows)
    ):
        raise WhiteEvaluationError("teacher contract accepted count differs from teacher manifest")
    closure_payload = {
        "schema_version": contract.get("schema_version"),
        "inputs": contract.get("inputs"),
        "configuration": contract.get("configuration"),
        "counts": contract.get("counts"),
        "split_use": contract.get("split_use"),
        "artifacts": contract.get("artifacts"),
    }
    observed_closure = _canonical_sha256(closure_payload)
    if contract.get("closure_sha256") != observed_closure:
        raise WhiteEvaluationError("teacher contract closure SHA-256 is invalid")

    receipt_path = publication / "teacher.receipt.json"
    receipt = _load_json(receipt_path, "teacher receipt")
    if not isinstance(receipt, Mapping):
        raise WhiteEvaluationError("teacher receipt must be one JSON object")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("kind") != TEACHER_RECEIPT_KIND
        or receipt.get("sealed") is not True
        or receipt.get("contract_closure_sha256") != observed_closure
    ):
        raise WhiteEvaluationError("teacher receipt does not bind the sealed teacher contract")
    receipt_contract = receipt.get("contract")
    if not isinstance(receipt_contract, Mapping) or not _binding_matches(
        _artifact_binding(contract_path), receipt_contract
    ):
        raise WhiteEvaluationError("teacher receipt contract binding differs")

    inputs = contract.get("inputs")
    model_contract_sha256 = None
    if isinstance(inputs, Mapping):
        model_assets = inputs.get("model_assets")
        if isinstance(model_assets, Mapping):
            model_contract_sha256 = model_assets.get("adapter_contract_sha256")
    _require_sha256(model_contract_sha256, "teacher adapter contract SHA-256")
    return {
        **_identity(teacher_manifest),
        "contract": _identity(contract_path),
        "receipt": _identity(receipt_path),
        "reject_manifest": _identity(reject_manifest),
        "contract_closure_sha256": observed_closure,
        "teacher_model_contract_sha256": model_contract_sha256,
        "sealed": True,
    }, contract


def _validate_teacher_record(
    row: Mapping[str, Any],
    *,
    index: int,
    minimum_line_confidence: float,
    group_splits: dict[str, str],
    seen_record_ids: set[str],
) -> tuple[str, str, str]:
    location = f"teacher row {index}"
    if row.get("schema_version") != SCHEMA_VERSION or row.get("kind") != TEACHER_RECORD_KIND:
        raise WhiteEvaluationError(f"{location} has unexpected kind/schema")
    record_id = _require_sha256(row.get("record_id"), f"{location} record_id")
    if record_id in seen_record_ids:
        raise WhiteEvaluationError(f"duplicate teacher record_id: {record_id}")
    seen_record_ids.add(record_id)
    group = _require_nonempty_string(row.get("group_id"), f"{location} group_id")
    split = row.get("split")
    if split not in {"train", "val", "test"}:
        raise WhiteEvaluationError(f"{location} split must be train, val, or test")
    split = str(split)
    prior = group_splits.setdefault(group, split)
    if prior != split:
        raise WhiteEvaluationError(f"teacher group crosses splits: {group}")
    training_eligible = split == "train"
    if (
        row.get("split_use") != ("training" if training_eligible else f"heldout_{split}")
        or row.get("training_eligible") is not training_eligible
        or row.get("evaluation_only") is not (not training_eligible)
        or row.get("held_out") is not (not training_eligible)
    ):
        raise WhiteEvaluationError(f"{location} split-use flags are invalid")
    if (
        row.get("label_source") != "paddle_db_cls_rec_three_view_consensus"
        or row.get("automatic_teacher_validation") is not True
        or row.get("manual_review_required") is not False
    ):
        raise WhiteEvaluationError(f"{location} teacher provenance flags are invalid")

    source_root_raw = _require_nonempty_string(row.get("source_root"), f"{location} source_root")
    source_relative = _require_nonempty_string(
        row.get("source_relative_path"), f"{location} source_relative_path"
    )
    relative_parts = source_relative.split("/")
    if (
        "\\" in source_relative
        or source_relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative_parts)
        or ":" in relative_parts[0]
    ):
        raise WhiteEvaluationError(f"{location} has unsafe source_relative_path")
    try:
        source_root = Path(source_root_raw).expanduser().resolve(strict=True)
        source_path = Path(
            _require_nonempty_string(row.get("source_absolute_path"), f"{location} source_absolute_path")
        ).expanduser().resolve(strict=True)
        expected_source = (source_root / Path(*relative_parts)).resolve(strict=True)
    except OSError as exception:
        raise WhiteEvaluationError(f"{location} source provenance cannot be resolved: {exception}") from exception
    if not source_root.is_dir() or source_path != expected_source or not source_path.is_file():
        raise WhiteEvaluationError(f"{location} source path does not match root/relative provenance")
    raw_sha256 = _require_sha256(row.get("raw_sha256"), f"{location} raw_sha256")
    _require_sha256(row.get("decoded_pixel_sha256"), f"{location} decoded_pixel_sha256")
    if _sha256(source_path) != raw_sha256:
        raise WhiteEvaluationError(f"selected source is missing or differs from teacher SHA-256: {source_path}")
    expected_record_id = hashlib.sha256(
        b"otherimages-image-record-v1\0"
        + source_relative.encode("utf-8")
        + b"\0"
        + raw_sha256.encode("ascii")
    ).hexdigest()
    if record_id != expected_record_id:
        raise WhiteEvaluationError(f"{location} record_id does not bind source relative path and raw SHA-256")

    text = _require_nonempty_string(row.get("text"), f"{location} text")
    if text != _normalise_text(text):
        raise WhiteEvaluationError(f"{location} text is not canonically normalized")
    if row.get("text_normalization") != TEXT_NORMALIZATION:
        raise WhiteEvaluationError(f"{location} text_normalization is invalid")
    if row.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise WhiteEvaluationError(f"{location} text SHA-256 is invalid")

    consensus = row.get("consensus")
    chosen_view = row.get("chosen_view")
    if not isinstance(consensus, Mapping) or consensus.get("agreement") not in {"2_of_3", "3_of_3"}:
        raise WhiteEvaluationError(f"{location} consensus evidence is invalid")
    if not isinstance(chosen_view, Mapping):
        raise WhiteEvaluationError(f"{location} chosen_view evidence is missing")
    view_id = chosen_view.get("view_id")
    if view_id not in TEACHER_VIEW_IDS or consensus.get("chosen_geometry_view_id") != view_id:
        raise WhiteEvaluationError(f"{location} chosen geometry view evidence is invalid")
    view_contract_sha256 = _require_sha256(
        chosen_view.get("view_contract_sha256"), f"{location} view contract SHA-256"
    )
    if view_contract_sha256 != _canonical_sha256(_canonical_view_contract(str(view_id))):
        raise WhiteEvaluationError(f"{location} chosen view contract SHA-256 is invalid")
    _require_sha256(chosen_view.get("transformed_pixel_sha256"), f"{location} transformed pixel SHA-256")
    dimensions = {
        name: _require_nonnegative_int(chosen_view.get(name), f"{location} {name}")
        for name in ("source_width", "source_height", "transformed_width", "transformed_height")
    }
    if any(value <= 1 for value in dimensions.values()) or (
        chosen_view.get("coordinate_mapping") != "full_frame_scale_source_normalized_identity_v1"
    ):
        raise WhiteEvaluationError(f"{location} chosen_view dimensions/mapping are invalid")

    agreement = str(consensus["agreement"])
    support_votes = 3 if agreement == "3_of_3" else 2
    geometry_view_ids = consensus.get("geometry_support_view_ids")
    dominant_view_ids = consensus.get("dominant_view_ids")
    dominant_votes = consensus.get("dominant_text_votes")
    if (
        consensus.get("geometry_support_votes") != support_votes
        or not isinstance(geometry_view_ids, list)
        or len(geometry_view_ids) != support_votes
        or len(set(geometry_view_ids)) != support_votes
        or any(item not in TEACHER_VIEW_IDS for item in geometry_view_ids)
        or view_id not in geometry_view_ids
        or isinstance(dominant_votes, bool)
        or dominant_votes not in {2, 3}
        or not isinstance(dominant_view_ids, list)
        or len(dominant_view_ids) != dominant_votes
        or len(set(dominant_view_ids)) != dominant_votes
        or any(item not in TEACHER_VIEW_IDS for item in dominant_view_ids)
    ):
        raise WhiteEvaluationError(f"{location} consensus vote provenance is invalid")
    minimum_iou = consensus.get("minimum_pairwise_line_quad_iou")
    if (
        isinstance(minimum_iou, bool)
        or not isinstance(minimum_iou, (int, float))
        or not math.isfinite(float(minimum_iou))
        or not 0.0 <= float(minimum_iou) <= 1.0
    ):
        raise WhiteEvaluationError(f"{location} consensus geometry IoU is invalid")
    support_confidences = consensus.get("support_confidences")
    if not isinstance(support_confidences, list) or len(support_confidences) != support_votes:
        raise WhiteEvaluationError(f"{location} consensus support confidence provenance is incomplete")
    confidence_views: set[str] = set()
    for support_index, support in enumerate(support_confidences):
        if not isinstance(support, Mapping):
            raise WhiteEvaluationError(f"{location} support confidence {support_index} is invalid")
        support_view = support.get("view_id")
        minimum_confidence = support.get("minimum_line_confidence")
        mean_confidence = support.get("mean_line_confidence")
        if (
            support_view not in geometry_view_ids
            or support_view in confidence_views
            or isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, (int, float))
            or isinstance(mean_confidence, bool)
            or not isinstance(mean_confidence, (int, float))
            or not math.isfinite(float(minimum_confidence))
            or not math.isfinite(float(mean_confidence))
            or not minimum_line_confidence <= float(minimum_confidence) <= float(mean_confidence) <= 1.0
        ):
            raise WhiteEvaluationError(f"{location} support confidence {support_index} is invalid")
        confidence_views.add(str(support_view))
    if confidence_views != set(str(value) for value in geometry_view_ids):
        raise WhiteEvaluationError(f"{location} consensus support confidence views differ")

    lines = row.get("lines")
    if not isinstance(lines, list) or not lines:
        raise WhiteEvaluationError(f"{location} lines must be a non-empty array")
    canonical_lines: list[str] = []
    for line_index, line in enumerate(lines):
        if not isinstance(line, Mapping) or line.get("index") != line_index:
            raise WhiteEvaluationError(f"{location} line {line_index} index is invalid")
        line_text = _require_nonempty_string(line.get("text"), f"{location} line {line_index} text")
        if line_text != _normalise_text(line_text):
            raise WhiteEvaluationError(f"{location} line {line_index} text is not canonically normalized")
        confidence = line.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not minimum_line_confidence <= float(confidence) <= 1.0
        ):
            raise WhiteEvaluationError(f"{location} line {line_index} confidence is invalid")
        orientation = line.get("orientation_degrees")
        if isinstance(orientation, bool) or orientation not in {0, 180}:
            raise WhiteEvaluationError(f"{location} line {line_index} orientation_degrees must be 0 or 180")
        transformed_quad = line.get("transformed_quad_pixels")
        normalized_quad = line.get("quad_normalized")
        if (
            not isinstance(transformed_quad, list)
            or len(transformed_quad) != 4
            or not isinstance(normalized_quad, list)
            or len(normalized_quad) != 4
        ):
            raise WhiteEvaluationError(f"{location} line {line_index} geometry is incomplete")
        for point_index, (pixel_point, normalized_point) in enumerate(zip(transformed_quad, normalized_quad)):
            if (
                not isinstance(pixel_point, list)
                or len(pixel_point) != 2
                or not isinstance(normalized_point, list)
                or len(normalized_point) != 2
            ):
                raise WhiteEvaluationError(f"{location} line {line_index} point {point_index} is invalid")
            try:
                px, py = float(pixel_point[0]), float(pixel_point[1])
                nx, ny = float(normalized_point[0]), float(normalized_point[1])
            except (TypeError, ValueError):
                raise WhiteEvaluationError(
                    f"{location} line {line_index} point {point_index} is invalid"
                ) from None
            if (
                not all(math.isfinite(value) for value in (px, py, nx, ny))
                or not 0.0 <= px <= dimensions["transformed_width"] - 1
                or not 0.0 <= py <= dimensions["transformed_height"] - 1
                or not 0.0 <= nx <= 1.0
                or not 0.0 <= ny <= 1.0
                or not math.isclose(nx, px / (dimensions["transformed_width"] - 1), abs_tol=1e-7)
                or not math.isclose(ny, py / (dimensions["transformed_height"] - 1), abs_tol=1e-7)
            ):
                raise WhiteEvaluationError(
                    f"{location} line {line_index} point {point_index} geometry binding is invalid"
                )
        canonical_lines.append(line_text)
    if text != "\n".join(canonical_lines):
        raise WhiteEvaluationError(f"{location} text differs from its line projection")
    return group, split, str(source_path)


def _resolve_result_path(results_root: Path, raw: str) -> Path:
    path = Path(raw)
    resolved = (path if path.is_absolute() else results_root / path).resolve()
    try:
        resolved.relative_to(results_root)
    except ValueError:
        raise WhiteEvaluationError(
            f"inference result is outside the supplied results root: {resolved}"
        ) from None
    return resolved


def score_white_results(
    *,
    teacher_manifest: Path,
    results_root: Path,
    output_dir: Path,
    split: str,
    teacher_contract: Path | None = None,
    max_cer: float = DEFAULT_MAX_CER,
    min_document_exact: float = DEFAULT_MIN_DOCUMENT_EXACT,
    min_line_precision: float = DEFAULT_MIN_LINE_PRECISION,
    min_line_recall: float = DEFAULT_MIN_LINE_RECALL,
    max_three_of_three_cer: float = DEFAULT_MAX_THREE_OF_THREE_CER,
    allow_extra_results: bool = False,
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise WhiteEvaluationError("evaluation split must be val or test")
    teacher_manifest = teacher_manifest.resolve()
    results_root = results_root.resolve()
    output_dir = Path(os.path.abspath(os.path.expanduser(os.fspath(output_dir))))
    _require_fresh_output_target(output_dir)
    teacher_rows = _load_jsonl(teacher_manifest, "teacher manifest")
    if teacher_contract is None:
        candidate = teacher_manifest.with_name("teacher.contract.json")
        teacher_contract = candidate if candidate.is_file() else None
    if teacher_contract is None:
        raise WhiteEvaluationError("sealed teacher.contract.json is required for a formal held-out score")
    teacher_binding, teacher_contract_payload = _verify_teacher_publication(
        teacher_contract.resolve(), teacher_manifest, teacher_rows
    )

    configuration = teacher_contract_payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise WhiteEvaluationError("teacher contract configuration must be an object")
    minimum_line_confidence_value = configuration.get("minimum_line_confidence")
    if (
        isinstance(minimum_line_confidence_value, bool)
        or not isinstance(minimum_line_confidence_value, (int, float))
        or not math.isfinite(float(minimum_line_confidence_value))
        or not 0.0 <= float(minimum_line_confidence_value) <= 1.0
    ):
        raise WhiteEvaluationError("teacher minimum_line_confidence is invalid")
    minimum_line_confidence = float(minimum_line_confidence_value)

    group_splits: dict[str, str] = {}
    seen_record_ids: set[str] = set()
    selected: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(teacher_rows):
        group, row_split, source = _validate_teacher_record(
            row,
            index=index,
            minimum_line_confidence=minimum_line_confidence,
            group_splits=group_splits,
            seen_record_ids=seen_record_ids,
        )
        if row_split != split:
            continue
        key = _path_key(source)
        if key in selected:
            raise WhiteEvaluationError(f"duplicate selected teacher source: {source}")
        selected[key] = row
    if not selected:
        raise WhiteEvaluationError(f"teacher manifest has no held-out {split} records")

    counts = teacher_contract_payload.get("counts")
    if not isinstance(counts, Mapping):
        raise WhiteEvaluationError("teacher contract counts must be an object")
    actual_by_split = dict(sorted(Counter(str(row["split"]) for row in teacher_rows).items()))
    if counts.get("accepted_by_split") != actual_by_split:
        raise WhiteEvaluationError("teacher contract accepted_by_split differs from teacher records")
    if counts.get("training_eligible_records") != actual_by_split.get("train", 0):
        raise WhiteEvaluationError("teacher contract training_eligible_records differs")
    if counts.get("evaluation_only_records") != (
        actual_by_split.get("val", 0) + actual_by_split.get("test", 0)
    ):
        raise WhiteEvaluationError("teacher contract evaluation_only_records differs")

    manifest_path = results_root / "inference_manifest.json"
    summary_path = results_root / "inference_summary.json"
    manifest = _load_json(manifest_path, "inference manifest")
    runtime_summary = _load_json(summary_path, "inference summary")
    if not isinstance(manifest, list) or not isinstance(runtime_summary, Mapping):
        raise WhiteEvaluationError("white inference manifest/summary has an invalid JSON shape")
    if (
        runtime_summary.get("document_type") != "white"
        or runtime_summary.get("requested_device") != "cpu"
        or runtime_summary.get("paddle_ocr_provider") != "cpu"
        or runtime_summary.get("white_student_provider") != "cpu"
        or runtime_summary.get("errors") != 0
    ):
        raise WhiteEvaluationError(
            "white inference summary does not prove a zero-error CPU PP-OCR + student run"
        )

    manifest_by_source: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(manifest):
        if not isinstance(row, Mapping):
            raise WhiteEvaluationError(f"inference manifest row {index} must be an object")
        source = row.get("source")
        result = row.get("result")
        if not isinstance(source, str) or not source or not isinstance(result, str) or not result:
            raise WhiteEvaluationError(f"inference manifest row {index} has invalid source/result")
        key = _path_key(source)
        if key in manifest_by_source:
            raise WhiteEvaluationError(f"duplicate inference source: {source}")
        manifest_by_source[key] = row
    if (
        runtime_summary.get("input") != len(manifest)
        or runtime_summary.get("written") != len(manifest)
        or runtime_summary.get("skipped") != 0
    ):
        raise WhiteEvaluationError("white inference summary counts differ from its fresh manifest")
    errors_path = results_root / "inference_errors.jsonl"
    try:
        errors_bytes = errors_path.read_bytes()
    except OSError as exception:
        raise WhiteEvaluationError(f"cannot read white inference errors {errors_path}: {exception}") from exception
    if errors_bytes.strip():
        raise WhiteEvaluationError("white inference errors file is not empty")
    missing = sorted(str(row["source_absolute_path"]) for key, row in selected.items() if key not in manifest_by_source)
    extra = sorted(str(row["source"]) for key, row in manifest_by_source.items() if key not in selected)
    failures: list[str] = []
    if split != "test":
        failures.append("validation split is diagnostic-only; formal acceptance requires the frozen test split")
    if allow_extra_results:
        failures.append("allow-extra-results is diagnostic-only and cannot pass the formal held-out gate")
    if missing:
        failures.append(f"result coverage is incomplete: {len(missing)} held-out source(s) missing")
    if extra and not allow_extra_results:
        failures.append(f"inference manifest contains {len(extra)} source(s) outside the selected held-out split")

    comparisons: list[dict[str, Any]] = []
    model_closure: dict[str, Any] | None = None
    for key, teacher in selected.items():
        manifest_row = manifest_by_source.get(key)
        if manifest_row is None:
            continue
        source = str(teacher["source_absolute_path"])
        if manifest_row.get("status") != "written":
            failures.append(f"held-out result was not freshly written: {source}")
            continue
        result_path = _resolve_result_path(results_root, str(manifest_row["result"])).resolve()
        result = _load_json(result_path, f"white result for {source}")
        if not isinstance(result, Mapping):
            raise WhiteEvaluationError(f"white result must be an object: {result_path}")
        route = result.get("route")
        ocr = result.get("ocr")
        contracts = result.get("model_contracts")
        if (
            result.get("document_type") != "white"
            or result.get("inference_engine") != "dotnet_onnxruntime_cpu"
            or _path_key(str(result.get("source", ""))) != key
            or not isinstance(route, Mapping)
            or route.get("review_required") is not True
            or not isinstance(ocr, Mapping)
            or ocr.get("provider") != "cpu"
            or ocr.get("delivery_policy") != "review_only"
            or ocr.get("student_model_status") != "integrated_review_only"
            or ocr.get("student_provider") != "cpu"
            or ocr.get("student_crop_source") != STUDENT_CROP_SOURCE
            or not isinstance(contracts, Mapping)
            or contracts.get("runtime_source") != "immutable_verified_bytes"
            or contracts.get("reopened_paths_after_verification") is not False
        ):
            raise WhiteEvaluationError(f"white result does not satisfy CPU/review/closure contract: {result_path}")
        result_lines = result.get("lines")
        if not isinstance(result_lines, list):
            raise WhiteEvaluationError(f"white result lines must be an array: {result_path}")
        predicted_lines = _student_lines(result_lines, f"result {source}")
        if ocr.get("student_comparison_line_count") != len(result_lines):
            raise WhiteEvaluationError(f"white result student comparison count is incomplete: {result_path}")
        observed_student_exact = sum(
            isinstance(line, Mapping)
            and isinstance(line.get("student"), Mapping)
            and line["student"].get("normalized_exact_match") is True
            for line in result_lines
        )
        if ocr.get("student_normalized_exact_match_line_count") != observed_student_exact:
            raise WhiteEvaluationError(f"white result student exact-match count is inconsistent: {result_path}")
        current_hashes = _validate_student_bundle_contracts(
            contracts, f"white result {result_path}"
        )
        if model_closure is None:
            model_closure = current_hashes
        elif current_hashes != model_closure:
            raise WhiteEvaluationError("white result model/student closure changed within the held-out run")

        teacher_lines = _normalised_lines(
            teacher["lines"],
            f"teacher {source}",
            accepted_only=False,
            require_teacher_orientation=True,
        )
        paddle_lines = _normalised_lines(result_lines, f"result {source}", accepted_only=True)
        teacher_text = _normalise_text(str(teacher["text"]))
        predicted_text = _normalise_text("\n".join(predicted_lines))
        paddle_text = _normalise_text("\n".join(paddle_lines))
        ocr_aggregate_text = ocr.get("aggregate_text")
        if not isinstance(ocr_aggregate_text, str) or _normalise_text(ocr_aggregate_text) != paddle_text:
            raise WhiteEvaluationError(f"white aggregate_text differs from accepted line projection: {result_path}")
        distance = _edit_distance(teacher_text, predicted_text)
        paddle_distance = _edit_distance(teacher_text, paddle_text)
        matching_lines = sum((Counter(teacher_lines) & Counter(predicted_lines)).values())
        paddle_matching_lines = sum((Counter(teacher_lines) & Counter(paddle_lines)).values())
        comparisons.append(
            {
                "record_id": teacher.get("record_id"),
                "group_id": teacher.get("group_id"),
                "source": source,
                "source_sha256": teacher.get("raw_sha256"),
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
                "consensus_agreement": dict(teacher["consensus"])["agreement"],
                "teacher_text": teacher_text,
                "predicted_text": predicted_text,
                "paddle_runtime_text": paddle_text,
                "document_exact_match": teacher_text == predicted_text,
                "teacher_characters": len(teacher_text),
                "predicted_characters": len(predicted_text),
                "edit_distance": distance,
                "character_error_rate": distance / len(teacher_text),
                "teacher_lines": len(teacher_lines),
                "predicted_lines": len(predicted_lines),
                "matching_lines": matching_lines,
                "paddle_runtime_document_exact_match": teacher_text == paddle_text,
                "paddle_runtime_edit_distance": paddle_distance,
                "paddle_runtime_predicted_lines": len(paddle_lines),
                "paddle_runtime_matching_lines": paddle_matching_lines,
            }
        )

    overall = _summarize(comparisons)
    paddle_diagnostic = _summarize(
        [
            {
                "teacher_characters": row["teacher_characters"],
                "edit_distance": row["paddle_runtime_edit_distance"],
                "teacher_lines": row["teacher_lines"],
                "predicted_lines": row["paddle_runtime_predicted_lines"],
                "matching_lines": row["paddle_runtime_matching_lines"],
                "document_exact_match": row["paddle_runtime_document_exact_match"],
            }
            for row in comparisons
        ]
    )
    by_consensus = {
        agreement: _summarize([row for row in comparisons if row["consensus_agreement"] == agreement])
        for agreement in ("2_of_3", "3_of_3")
    }
    result_coverage = len(comparisons) / len(selected)
    if result_coverage < 1.0:
        failures.append(f"scored result coverage {result_coverage:.6f} is below 1.0")
    if overall["character_error_rate"] is None or overall["character_error_rate"] > max_cer:
        failures.append(f"overall CER exceeds {max_cer:.6f}")
    if overall["document_exact_match"] is None or overall["document_exact_match"] < min_document_exact:
        failures.append(f"document exact agreement is below {min_document_exact:.6f}")
    if overall["line_exact_precision"] is None or overall["line_exact_precision"] < min_line_precision:
        failures.append(f"exact-line precision is below {min_line_precision:.6f}")
    if overall["line_exact_recall"] is None or overall["line_exact_recall"] < min_line_recall:
        failures.append(f"exact-line recall is below {min_line_recall:.6f}")
    three_of_three = by_consensus["3_of_3"]
    if (
        three_of_three["records"] > 0
        and (
            three_of_three["character_error_rate"] is None
            or three_of_three["character_error_rate"] > max_three_of_three_cer
        )
    ):
        failures.append(f"3-of-3 consensus CER exceeds {max_three_of_three_cer:.6f}")

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "accepted": not failures,
        "evaluation_split": split,
        "teacher": teacher_binding,
        "runtime_evidence": {
            "manifest": _identity(manifest_path),
            "summary": _identity(summary_path),
            "errors": _identity(errors_path),
            "document_type": "white",
            "requested_device": "cpu",
            "paddle_ocr_provider": "cpu",
            "white_student_provider": "cpu",
            "model_hashes": {
                key: value
                for key, value in (model_closure or {}).items()
                if key.endswith("_sha256")
            },
            "model_contracts": model_closure or {},
        },
        "coverage": {
            "expected_heldout_records": len(selected),
            "scored_records": len(comparisons),
            "result_coverage": result_coverage,
            "missing_sources": missing,
            "extra_sources": extra,
            "extra_results_allowed": allow_extra_results,
        },
        "teacher_agreement": {
            "metric_subject": "white_line_student",
            "overall": overall,
            "by_consensus": by_consensus,
        },
        "paddle_runtime_self_consistency_diagnostic": paddle_diagnostic,
        "gate": {
            "max_character_error_rate": max_cer,
            "min_document_exact_match": min_document_exact,
            "min_line_exact_precision": min_line_precision,
            "min_line_exact_recall": min_line_recall,
            "max_three_of_three_character_error_rate": max_three_of_three_cer,
        },
        "failures": failures,
        "warning": (
            "Student text is scored against Paddle three-view pseudo-label consensus. This is student-to-teacher "
            "agreement, not independently human-verified OCR accuracy. Paddle runtime self-consistency is "
            "diagnostic only and is never used as the student acceptance metric."
        ),
    }
    _reserve_fresh_output_directory(output_dir)
    _write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _unit_interval(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, help="defaults to teacher.contract.json beside manifest")
    parser.add_argument("--results", type=Path, required=True, help="white .NET inference output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="formal acceptance requires test; val is diagnostic-only and always accepted=false",
    )
    parser.add_argument("--max-cer", type=_unit_interval, default=DEFAULT_MAX_CER)
    parser.add_argument("--min-document-exact", type=_unit_interval, default=DEFAULT_MIN_DOCUMENT_EXACT)
    parser.add_argument("--min-line-precision", type=_unit_interval, default=DEFAULT_MIN_LINE_PRECISION)
    parser.add_argument("--min-line-recall", type=_unit_interval, default=DEFAULT_MIN_LINE_RECALL)
    parser.add_argument(
        "--max-three-of-three-cer", type=_unit_interval, default=DEFAULT_MAX_THREE_OF_THREE_CER
    )
    parser.add_argument(
        "--allow-extra-results",
        action="store_true",
        help="diagnostic only: ignore results outside the selected held-out split",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = score_white_results(
            teacher_manifest=args.teacher_manifest,
            teacher_contract=args.teacher_contract,
            results_root=args.results,
            output_dir=args.output,
            split=args.split,
            max_cer=args.max_cer,
            min_document_exact=args.min_document_exact,
            min_line_precision=args.min_line_precision,
            min_line_recall=args.min_line_recall,
            max_three_of_three_cer=args.max_three_of_three_cer,
            allow_extra_results=args.allow_extra_results,
        )
    except (OSError, WhiteEvaluationError) as exception:
        parser.error(str(exception))
    overall = summary["teacher_agreement"]["overall"]
    document_exact = overall["document_exact_match"]
    character_error_rate = overall["character_error_rate"]
    document_exact_text = "n/a" if document_exact is None else f"{document_exact:.2%}"
    character_error_rate_text = (
        "n/a" if character_error_rate is None else f"{character_error_rate:.2%}"
    )
    print(
        f"White held-out teacher agreement: coverage={summary['coverage']['result_coverage']:.2%}, "
        f"document_exact={document_exact_text}, "
        f"CER={character_error_rate_text}, accepted={summary['accepted']}"
    )
    if summary["failures"]:
        print("White held-out gate failed:\n- " + "\n- ".join(summary["failures"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
