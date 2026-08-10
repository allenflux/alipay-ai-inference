"""Export train-only recipient crops that mirror the production OCR views.

This module is deliberately a data boundary, not a trainer.  It reads the
recipient target already frozen in a Paddle-derived unified manifest and emits
four pixel views for *train rows only*:

``fixed_value``
    The v12/v13 fixed recipient input: the standard detector crop with the
    left 30 percent removed using the deployed banker-rounding rule.

``standard``
    The ordinary detector crop with the production eight-percent margin.

``left_context``
    The hybrid route's full-left row crop.

``right_value``
    The hybrid diagnostic route beginning no earlier than 45 percent of the
    rectified source width.

No OCR is run here and no label is inferred from pixels, filenames, a parser,
or a held-out row.  ``slots.recipient_field.text`` on an existing ``train``
record is the only supervised target.  Validation, test and formal target
values are neither accessed by the exporter logic nor validated/emitted; only
their containing records' split/group/source/crop-hash closure is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from .geometry import load_upright_rgb
from .ocr import parse_anchored_recipient_row
from .ocr_pseudolabels import _bbox, _crop_digest
from .ocr_unified_dataset import (
    KIND_V11 as UNIFIED_KIND_V11,
    KIND_V12 as UNIFIED_KIND_V12,
    KIND_V13 as UNIFIED_KIND_V13,
    RECIPIENT_QUALITY_POLICY_VERSION,
)
from .pipeline import crop_field_with_margin
from .status_crops import (
    _load_json_document,
    _result_payload,
    _source_path,
    reconstruct_rectified,
)


SCHEMA_VERSION = 1
KIND = "receipt_recipient_multiview_teacher_train_export_v1"
RECORD_KIND = "receipt_recipient_multiview_teacher_train_record_v1"
VIEWS = ("fixed_value", "standard", "left_context", "right_value")
TRAIN_SPLIT = "train"
HELD_OUT_SPLITS = frozenset(("val", "test", "formal"))
ALLOWED_SPLITS = frozenset((TRAIN_SPLIT, *HELD_OUT_SPLITS))
STANDARD_MARGIN_RATIO = 0.08
FIXED_VALUE_LEFT_TRIM = 0.30
RIGHT_VALUE_SOURCE_FRACTION = 0.45
SUPPORTED_UNIFIED_KINDS = frozenset(
    (UNIFIED_KIND_V11, UNIFIED_KIND_V12, UNIFIED_KIND_V13)
)
_HEX_DIGITS = frozenset("0123456789abcdef")


FileIdentity = tuple[int, int, int, str]
DirectoryIdentity = tuple[int, int]


@dataclass(frozen=True)
class _GeneratedViewOwner:
    line_number: int
    record_id: str
    view: str
    group_id: str
    target_sha256: str
    shape: tuple[int, ...]


def _register_generated_view_owner(
    owners: dict[str, _GeneratedViewOwner],
    *,
    pixel_sha256: str,
    owner: _GeneratedViewOwner,
) -> None:
    """Reject one generated pixel identity spanning a group or target boundary."""

    prior = owners.setdefault(pixel_sha256, owner)
    group_conflict = prior.group_id != owner.group_id
    target_conflict = prior.target_sha256 != owner.target_sha256
    if not group_conflict and not target_conflict:
        return

    def describe(value: _GeneratedViewOwner) -> str:
        return (
            f"line={value.line_number} record_id={value.record_id!r} "
            f"view={value.view!r} group_id={value.group_id!r} "
            f"target_sha256={value.target_sha256} shape={value.shape!r}"
        )

    raise ValueError(
        f"generated view hash {pixel_sha256} conflict: "
        f"group_conflict={str(group_conflict).lower()} "
        f"target_conflict={str(target_conflict).lower()}; "
        f"prior({describe(prior)}); current({describe(owner)})"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> FileIdentity:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"expected a regular file: {path}")
    return info.st_dev, info.st_ino, info.st_size, _sha256(path)


def _directory_identity(path: Path) -> DirectoryIdentity:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"expected a directory: {path}")
    return info.st_dev, info.st_ino


def _same_file_identity(path: Path, expected: FileIdentity) -> bool:
    try:
        return _file_identity(path) == expected
    except (OSError, ValueError):
        return False


def _same_directory_identity(path: Path, expected: DirectoryIdentity) -> bool:
    try:
        return _directory_identity(path) == expected
    except (OSError, ValueError):
        return False


def _is_reparse_point(path: Path) -> bool:
    info = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_attribute)


def _assert_no_reparse_components(path: Path, *, description: str) -> None:
    """Reject symlink/junction ancestors before resolving a new output path."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    components = (Path(absolute.anchor), *absolute.parts[1:])
    current = components[0]
    for component in components[1:]:
        current = current / component
        if not os.path.lexists(current):
            continue
        if _is_reparse_point(current):
            raise ValueError(f"{description} contains a symlink or reparse component: {current}")


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _assert_output_separate(output: Path, protected: Path, *, description: str) -> None:
    if _paths_overlap(output, protected):
        raise ValueError(f"recipient multiview output overlaps {description}: {protected}")


def _safe_unlink(path: Path, expected: FileIdentity) -> None:
    if _same_file_identity(path, expected):
        path.unlink()


def _safe_rmdir(path: Path, expected: DirectoryIdentity) -> None:
    if not _same_directory_identity(path, expected):
        return
    try:
        path.rmdir()
    except OSError:
        # A foreign file or directory is never removed merely because it
        # appeared under a path originally reserved by this run.
        pass


def _cleanup_owned_tree(
    *,
    files: Sequence[tuple[Path, FileIdentity]],
    directories: Sequence[tuple[Path, DirectoryIdentity]],
) -> None:
    for path, identity in reversed(files):
        try:
            _safe_unlink(path, identity)
        except OSError:
            pass
    for path, identity in reversed(directories):
        try:
            _safe_rmdir(path, identity)
        except OSError:
            pass


def _require_sha256(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _relative_existing_file(root: Path, value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{description} must be relative to the dataset root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"{description} escapes the dataset root") from None
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _absolute_existing_file(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{description} must be absolute")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validated_contract(path: Path) -> dict[str, object]:
    raw = _load_json_document(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: unified dataset contract must be an object")
    contract = dict(raw)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported unified dataset contract schema")
    if contract.get("kind") not in SUPPORTED_UNIFIED_KINDS:
        raise ValueError(f"{path}: recipient multiview export requires a v11-v13 manifest")
    if contract.get("recipient_charset_source") != "train_only_anchored_recipient_value":
        raise ValueError(f"{path}: recipient charset is not train-only anchored text")
    quality = contract.get("recipient_quality_policy")
    if (
        not isinstance(quality, Mapping)
        or quality.get("version") != RECIPIENT_QUALITY_POLICY_VERSION
        or quality.get("requires_leading_recipient_label") is not True
        or quality.get("target") != "anchored_recipient_value"
    ):
        raise ValueError(f"{path}: recipient quality policy is not the frozen anchored policy")
    return contract


def _fixed_value_view(standard: np.ndarray) -> np.ndarray:
    if standard.ndim != 3 or standard.shape[2] != 3:
        raise ValueError("standard recipient crop must be an HxWx3 RGB image")
    height, width, _channels = standard.shape
    if height <= 0 or width <= 0:
        raise ValueError("standard recipient crop must be non-empty")
    left = min(
        width - 1,
        max(0, int(round(width * FIXED_VALUE_LEFT_TRIM))),
    )
    return np.ascontiguousarray(standard[:, left:, :])


def _production_standard_view(
    rectified: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Mirror C# ``CropFieldWithMargin`` using its ``float`` arithmetic."""

    x1, y1, x2, y2 = (np.float32(value) for value in bbox)
    height, width = rectified.shape[:2]
    ratio = np.float32(STANDARD_MARGIN_RATIO)
    margin_x = max(np.float32(2.0), np.float32((x2 - x1) * ratio))
    margin_y = max(np.float32(2.0), np.float32((y2 - y1) * ratio))
    left = max(0, min(width, math.floor(float(np.float32(x1 - margin_x)))))
    top = max(0, min(height, math.floor(float(np.float32(y1 - margin_y)))))
    right = max(0, min(width, math.ceil(float(np.float32(x2 + margin_x)))))
    bottom = max(0, min(height, math.ceil(float(np.float32(y2 + margin_y)))))
    if right <= left or bottom <= top:
        raise ValueError("recipient standard view is empty")
    return np.ascontiguousarray(rectified[top:bottom, left:right, :])


def _production_left_context_view(
    rectified: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Mirror ``CropRecipientRowLeftContext`` on rectified RGB pixels."""

    x1, y1, x2, y2 = (np.float32(value) for value in bbox)
    height, width = rectified.shape[:2]
    ratio = np.float32(STANDARD_MARGIN_RATIO)
    margin_x = max(np.float32(2.0), np.float32((x2 - x1) * ratio))
    margin_y = max(np.float32(2.0), np.float32((y2 - y1) * ratio))
    top = max(0, min(height, math.floor(float(np.float32(y1 - margin_y)))))
    right = max(0, min(width, math.ceil(float(np.float32(x2 + margin_x)))))
    bottom = max(0, min(height, math.ceil(float(np.float32(y2 + margin_y)))))
    if right <= 0 or bottom <= top:
        raise ValueError("recipient left-context view is empty")
    return np.ascontiguousarray(rectified[top:bottom, 0:right, :])


def _production_right_value_view(
    rectified: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Mirror ``CropRecipientRowRightValue`` on rectified RGB pixels."""

    x1, y1, x2, y2 = (np.float32(value) for value in bbox)
    height, width = rectified.shape[:2]
    margin_ratio = np.float32(STANDARD_MARGIN_RATIO)
    margin_x = max(np.float32(2.0), np.float32((x2 - x1) * margin_ratio))
    margin_y = max(np.float32(2.0), np.float32((y2 - y1) * margin_ratio))
    box_value_left = math.floor(float(np.float32(x1 + margin_x)))
    source_value_left = math.floor(
        float(np.float32(np.float32(width) * np.float32(RIGHT_VALUE_SOURCE_FRACTION)))
    )
    left = max(0, min(width, max(box_value_left, source_value_left)))
    top = max(0, min(height, math.floor(float(np.float32(y1 - margin_y)))))
    right = max(0, min(width, math.ceil(float(np.float32(x2 + margin_x)))))
    bottom = max(0, min(height, math.ceil(float(np.float32(y2 + margin_y)))))
    if right <= left or bottom <= top:
        raise ValueError("recipient right-value view is empty")
    return np.ascontiguousarray(rectified[top:bottom, left:right, :])


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


def _write_rgb_png(path: Path, pixels: np.ndarray) -> None:
    """Write one PNG through an exclusive file descriptor."""

    with path.open("xb") as stream:
        Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")


def _raw_records(path: Path) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append((line_number, dict(raw)))
    if not records:
        raise ValueError("recipient multiview source manifest is empty")
    return records


def _target_from_train_record(
    record: Mapping[str, object],
    *,
    source: Path,
    line_number: int,
) -> tuple[str, dict[str, object]]:
    """Return only the already-frozen Paddle target for one train row."""

    if record.get("label_source") != "paddle_pseudo":
        raise ValueError(
            f"{source}:{line_number}: train recipient label_source must be paddle_pseudo"
        )
    slots = record.get("slots")
    if not isinstance(slots, Mapping):
        raise ValueError(f"{source}:{line_number}: train record slots must be an object")
    raw_slot = slots.get("recipient_field")
    if not isinstance(raw_slot, Mapping):
        raise ValueError(f"{source}:{line_number}: train record has no recipient_field slot")
    slot = dict(raw_slot)
    target = slot.get("text")
    if (
        not isinstance(target, str)
        or not target
        or target != target.strip()
        or target != unicodedata.normalize("NFC", target)
        or any(not character.isprintable() for character in target)
    ):
        raise ValueError(
            f"{source}:{line_number}: recipient target must be non-empty, NFC, trimmed printable text"
        )
    if slot.get("recipient_quality_policy") != RECIPIENT_QUALITY_POLICY_VERSION:
        raise ValueError(f"{source}:{line_number}: recipient slot has no frozen quality policy")
    for evidence_name in ("recipient_value", "semantic_value"):
        evidence = slot.get(evidence_name)
        if evidence is not None and evidence != target:
            raise ValueError(
                f"{source}:{line_number}: recipient {evidence_name} conflicts with manifest target"
            )
    visible = slot.get("recipient_visible_text")
    if not isinstance(visible, str):
        raise ValueError(f"{source}:{line_number}: recipient anchored visible text is missing")
    parsed = parse_anchored_recipient_row(visible)
    if parsed is None or parsed[1] != target:
        raise ValueError(
            f"{source}:{line_number}: anchored visible text conflicts with manifest target"
        )
    return target, slot


def _binding(path: Path, bindings: dict[Path, str]) -> str:
    digest = bindings.get(path)
    if digest is None:
        digest = _sha256(path)
        bindings[path] = digest
    return digest


def _path_identity(value: str) -> str:
    """Return one platform-aware lexical identity without opening held-out data."""

    return os.path.normcase(os.path.abspath(os.path.normpath(value)))


def _assert_bindings_unchanged(bindings: Mapping[Path, str]) -> None:
    for path, expected in bindings.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"multiview source changed during export: {path}")


def _assert_files_unchanged(
    files: Sequence[tuple[Path, FileIdentity]],
    *,
    description: str,
) -> None:
    for path, expected in files:
        if not _same_file_identity(path, expected):
            raise ValueError(f"{description} changed during export: {path}")


def export_recipient_multiview_teacher(
    *,
    manifest: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
    dataset_contract: Path | None = None,
) -> dict[str, object]:
    """Export four train-only production-homologous recipient views.

    The function fails the complete export on a malformed row, changed file,
    cross-split group/hash, conflicting target, or crop-geometry mismatch.  It
    never writes a partial optimizer manifest.
    """

    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    contract_path = (
        dataset_contract.resolve()
        if dataset_contract is not None
        else (manifest.parent / "dataset.contract.json").resolve()
    )
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = _validated_contract(contract_path)
    raw_dataset_root = dataset_root if dataset_root is not None else contract.get("dataset_root")
    if isinstance(raw_dataset_root, Path):
        root = raw_dataset_root.resolve()
    elif isinstance(raw_dataset_root, str) and raw_dataset_root:
        root = Path(raw_dataset_root).resolve()
    else:
        raise ValueError("dataset_root is required when the unified contract has no dataset_root")
    if not root.is_dir():
        raise NotADirectoryError(root)

    raw_output = Path(output_dir)
    _assert_no_reparse_components(raw_output, description="recipient multiview output")
    absolute_output = Path(os.path.abspath(os.fspath(raw_output)))
    if os.path.lexists(absolute_output):
        raise FileExistsError(
            f"refusing to overwrite recipient multiview export: {absolute_output}"
        )
    output = absolute_output.resolve()
    _assert_output_separate(output, manifest.parent, description="manifest directory")
    _assert_output_separate(output, contract_path.parent, description="contract directory")
    _assert_output_separate(output, root, description="Paddle dataset root")
    absolute_output.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(
        absolute_output.parent,
        description="recipient multiview output parent",
    )
    output = absolute_output.resolve()
    _assert_output_separate(output, manifest.parent, description="manifest directory")
    _assert_output_separate(output, contract_path.parent, description="contract directory")
    _assert_output_separate(output, root, description="Paddle dataset root")
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    if os.path.lexists(stage):
        raise FileExistsError(stage)
    stage.mkdir()
    stage_directories: list[tuple[Path, DirectoryIdentity]] = [
        (stage, _directory_identity(stage))
    ]
    stage_files: list[tuple[Path, FileIdentity]] = []
    published_directories: list[tuple[Path, DirectoryIdentity]] = []
    published_files: list[tuple[Path, FileIdentity]] = []
    publication_complete = False

    bindings: dict[Path, str] = {}
    manifest_sha256 = _binding(manifest, bindings)
    contract_sha256 = _binding(contract_path, bindings)
    split_counts: Counter[str] = Counter()
    recipient_split_counts: Counter[str] = Counter()
    train_missing_recipient = 0
    group_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    crop_splits: dict[str, str] = {}
    crop_targets: dict[str, str] = {}
    ids: set[str] = set()
    train_rows: list[tuple[int, dict[str, object], str, dict[str, object]]] = []
    records: list[dict[str, object]] = []
    generated_hash_owners: dict[str, _GeneratedViewOwner] = {}

    try:
        # This pass intentionally touches held-out rows only for identifiers,
        # split/group/source and the already-declared crop hash.  It never
        # reads a held-out recipient text target.
        for line_number, record in _raw_records(manifest):
            record_id = record.get("id")
            group_id = record.get("group_id")
            split = record.get("split")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{manifest}:{line_number}: id must be a non-empty string")
            if record_id in ids:
                raise ValueError(f"{manifest}:{line_number}: duplicate id {record_id!r}")
            ids.add(record_id)
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{manifest}:{line_number}: group_id must be a non-empty string")
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"{manifest}:{line_number}: unsupported split {split!r}")
            split = str(split)
            split_counts[split] += 1
            prior_group_split = group_splits.setdefault(group_id, split)
            if prior_group_split != split:
                raise ValueError(
                    f"{manifest}:{line_number}: group {group_id!r} crosses "
                    f"{prior_group_split}/{split} split boundary"
                )
            raw_source = record.get("source")
            if not isinstance(raw_source, str) or not raw_source:
                raise ValueError(
                    f"{manifest}:{line_number}: source must be a non-empty path"
                )
            source_identity = _path_identity(raw_source)
            prior_source_split = source_splits.setdefault(source_identity, split)
            if prior_source_split != split:
                raise ValueError(
                    f"{manifest}:{line_number}: source {raw_source!r} crosses "
                    f"{prior_source_split}/{split} split boundary"
                )
            slots = record.get("slots")
            recipient = slots.get("recipient_field") if isinstance(slots, Mapping) else None
            if recipient is None:
                if split == TRAIN_SPLIT:
                    train_missing_recipient += 1
                continue
            if not isinstance(recipient, Mapping):
                raise ValueError(
                    f"{manifest}:{line_number}: recipient_field slot must be an object"
                )
            recipient_split_counts[split] += 1

            crop_hash = _require_sha256(
                recipient.get("crop_sha256"),
                description=f"{manifest}:{line_number}: recipient crop_sha256",
            )
            prior_crop_split = crop_splits.setdefault(crop_hash, split)
            if prior_crop_split != split:
                raise ValueError(
                    f"{manifest}:{line_number}: recipient crop {crop_hash} crosses "
                    f"{prior_crop_split}/{split} split boundary"
                )

            if split != TRAIN_SPLIT:
                continue
            target, train_slot = _target_from_train_record(
                record,
                source=manifest,
                line_number=line_number,
            )
            crop_hash = _require_sha256(
                train_slot.get("crop_sha256"),
                description=f"{manifest}:{line_number}: train recipient crop_sha256",
            )
            prior_target = crop_targets.setdefault(crop_hash, target)
            if prior_target != target:
                raise ValueError(
                    f"{manifest}:{line_number}: one recipient crop has conflicting train targets"
                )
            train_rows.append((line_number, record, target, train_slot))

        if not train_rows:
            raise ValueError("source manifest has no train recipient targets")

        images_dir = stage / "images"
        images_dir.mkdir()
        stage_directories.append((images_dir, _directory_identity(images_dir)))
        for line_number, record, target, slot in train_rows:
            record_id = str(record["id"])
            group_id = str(record["group_id"])
            source_path = _absolute_existing_file(
                record.get("source"),
                description=f"{manifest}:{line_number}: source",
            )
            result_path = _absolute_existing_file(
                record.get("result_json"),
                description=f"{manifest}:{line_number}: result_json",
            )
            crop_path = _relative_existing_file(
                root,
                slot.get("image"),
                description=f"{manifest}:{line_number}: recipient image",
            )
            if source_path.stat().st_mtime_ns > result_path.stat().st_mtime_ns:
                raise ValueError(
                    f"{manifest}:{line_number}: live source is newer than its Paddle result; "
                    "refusing to apply the frozen target to changed context pixels"
                )
            _assert_output_separate(
                output,
                source_path.parent,
                description=f"live source directory for {record_id}",
            )
            _assert_output_separate(
                output,
                result_path.parent,
                description=f"live result directory for {record_id}",
            )
            _assert_output_separate(
                output,
                crop_path.parent,
                description=f"Paddle crop directory for {record_id}",
            )
            source_sha256 = _binding(source_path, bindings)
            result_sha256 = _binding(result_path, bindings)
            crop_file_sha256 = _binding(crop_path, bindings)
            result_document = _load_json_document(result_path)
            result_payload = _result_payload(result_document)
            if result_payload is None:
                raise ValueError(f"{result_path}: not a receipt result bundle")
            payload_source = _source_path(result_payload, result_path)
            if payload_source != source_path:
                raise ValueError(
                    f"{manifest}:{line_number}: manifest and result bundle source paths disagree"
                )
            source_rgb = load_upright_rgb(source_path)
            rectified = reconstruct_rectified(result_payload, source_rgb)
            bbox = _bbox(slot.get("bbox_rectified"))
            paddle_standard = np.ascontiguousarray(crop_field_with_margin(rectified, bbox))
            if (
                paddle_standard.ndim != 3
                or paddle_standard.shape[2] != 3
                or min(paddle_standard.shape[:2]) <= 0
            ):
                raise ValueError(f"{manifest}:{line_number}: standard recipient crop is empty")
            declared_crop_sha256 = _require_sha256(
                slot.get("crop_sha256"),
                description=f"{manifest}:{line_number}: recipient crop_sha256",
            )
            if _crop_digest(paddle_standard) != declared_crop_sha256:
                raise ValueError(
                    f"{manifest}:{line_number}: reconstructed standard crop does not match "
                    "the Paddle manifest crop hash"
                )
            stored_crop = load_upright_rgb(crop_path)
            if not np.array_equal(stored_crop, paddle_standard):
                raise ValueError(
                    f"{manifest}:{line_number}: stored Paddle crop pixels differ from reconstructed standard view"
                )

            production_standard = _production_standard_view(rectified, bbox)
            views = {
                "fixed_value": _fixed_value_view(production_standard),
                "standard": production_standard,
                "left_context": _production_left_context_view(rectified, bbox),
                "right_value": _production_right_value_view(rectified, bbox),
            }
            target_sha256 = hashlib.sha256(target.encode("utf-8")).hexdigest()
            record_key = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24]
            view_payloads: list[dict[str, object]] = []
            for view_name in VIEWS:
                view = views[view_name]
                view_pixel_sha256 = _crop_digest(view)
                declared_split = crop_splits.get(view_pixel_sha256)
                if declared_split is not None and declared_split != TRAIN_SPLIT:
                    raise ValueError(
                        f"generated train view hash {view_pixel_sha256} crosses "
                        f"the declared {declared_split} recipient crop boundary"
                    )
                _register_generated_view_owner(
                    generated_hash_owners,
                    pixel_sha256=view_pixel_sha256,
                    owner=_GeneratedViewOwner(
                        line_number=line_number,
                        record_id=record_id,
                        view=view_name,
                        group_id=group_id,
                        target_sha256=target_sha256,
                        shape=tuple(int(size) for size in view.shape),
                    ),
                )
                relative_image = Path("images") / f"{record_key}-{view_name.replace('_', '-')}.png"
                image_path = stage / relative_image
                _write_rgb_png(image_path, view)
                image_identity = _file_identity(image_path)
                stage_files.append((image_path, image_identity))
                decoded = load_upright_rgb(image_path)
                if not np.array_equal(decoded, view):
                    raise ValueError(f"saved multiview image changed decoded pixels: {image_path}")
                view_payloads.append(
                    {
                        "view": view_name,
                        "image": relative_image.as_posix(),
                        "width": int(view.shape[1]),
                        "height": int(view.shape[0]),
                        "pixel_sha256": view_pixel_sha256,
                        "file_sha256": _sha256(image_path),
                    }
                )

            group_closure_payload = {
                "source_record_id": record_id,
                "source_group_id": group_id,
                "source_manifest_sha256": manifest_sha256,
                "target_sha256": target_sha256,
                "source_sha256": source_sha256,
                "result_json_sha256": result_sha256,
                "paddle_crop_pixel_sha256": declared_crop_sha256,
                "views": [
                    {
                        "view": payload["view"],
                        "pixel_sha256": payload["pixel_sha256"],
                        "file_sha256": payload["file_sha256"],
                    }
                    for payload in view_payloads
                ],
            }
            group_closure_sha256 = _canonical_sha256(group_closure_payload)
            for payload in view_payloads:
                view_name = str(payload["view"])
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": RECORD_KIND,
                        "id": f"recipient-{record_key}-{view_name.replace('_', '-')}",
                        "group_id": group_id,
                        "source_record_id": record_id,
                        "split": TRAIN_SPLIT,
                        "field": "recipient_field",
                        "view": view_name,
                        "image": payload["image"],
                        "text": target,
                        "target_sha256": target_sha256,
                        "target_source": "slots.recipient_field.text",
                        "target_source_manifest_sha256": manifest_sha256,
                        "optimizer_supervision_split_eligible": True,
                        "optimizer_consumable": False,
                        "group_closure_sha256": group_closure_sha256,
                        "group_view_count": len(VIEWS),
                        "source": source_path.as_posix(),
                        "source_sha256": source_sha256,
                        "result_json": result_path.as_posix(),
                        "result_json_sha256": result_sha256,
                        "bbox_rectified": [round(float(value), 4) for value in bbox],
                        "paddle_crop": crop_path.as_posix(),
                        "paddle_crop_pixel_sha256": declared_crop_sha256,
                        "paddle_crop_file_sha256": crop_file_sha256,
                        "view_width": payload["width"],
                        "view_height": payload["height"],
                        "view_pixel_sha256": payload["pixel_sha256"],
                        "view_file_sha256": payload["file_sha256"],
                    }
                )

        _assert_bindings_unchanged(bindings)
        if _sha256(manifest) != manifest_sha256 or _sha256(contract_path) != contract_sha256:
            raise ValueError("recipient multiview manifest or contract changed during export")

        records.sort(key=lambda value: (str(value["source_record_id"]), VIEWS.index(str(value["view"]))))
        train_path = stage / "multiview_train.jsonl"
        _write_jsonl(train_path, records)
        stage_files.append((train_path, _file_identity(train_path)))
        view_counts = Counter(str(record["view"]) for record in records)
        summary: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "source_manifest": manifest.as_posix(),
            "source_manifest_sha256": manifest_sha256,
            "source_dataset_contract": contract_path.as_posix(),
            "source_dataset_contract_sha256": contract_sha256,
            "source_dataset_kind": contract["kind"],
            "source_dataset_root": root.as_posix(),
            "target_source": "slots.recipient_field.text",
            "target_label_authority": "existing_paddle_train_manifest_only",
            "target_recomputed": False,
            "optimizer_supervision_splits": [TRAIN_SPLIT],
            "optimizer_input_ready": False,
            "records_role": "recipient_multiview_overlay_source_only",
            "optimizer_adapter_required": "strict_recipient_multiview_overlay_loader_not_implemented",
            "held_out_splits_excluded": sorted(HELD_OUT_SPLITS),
            "held_out_target_values_used": False,
            "held_out_target_values_validated": False,
            "held_out_target_values_emitted": False,
            "source_manifest_split_counts": {
                split: int(split_counts[split])
                for split in (TRAIN_SPLIT, "val", "test", "formal")
            },
            "source_split_counts": {
                split: int(recipient_split_counts[split])
                for split in (TRAIN_SPLIT, "val", "test", "formal")
            },
            "output_split_counts": {TRAIN_SPLIT: len(records)},
            "source_train_recipient_records": len(train_rows),
            "source_train_records_without_recipient_target": train_missing_recipient,
            "output_records": len(records),
            "view_order": list(VIEWS),
            "view_counts": {view: int(view_counts[view]) for view in VIEWS},
            "view_geometry": {
                "fixed_value": {
                    "base": "production_standard",
                    "left_trim_fraction": FIXED_VALUE_LEFT_TRIM,
                    "rounding": "bankers_round_midpoint_to_even",
                },
                "standard": {
                    "margin_ratio": STANDARD_MARGIN_RATIO,
                    "arithmetic": "csharp_ieee754_float32",
                },
                "left_context": {
                    "margin_ratio": STANDARD_MARGIN_RATIO,
                    "left": 0,
                },
                "right_value": {
                    "margin_ratio": STANDARD_MARGIN_RATIO,
                    "minimum_source_left_fraction": RIGHT_VALUE_SOURCE_FRACTION,
                },
            },
            "group_hash_closure": {
                "views_per_train_record": len(VIEWS),
                "cross_split_group_conflicts": 0,
                "cross_split_source_conflicts": 0,
                "cross_split_recipient_crop_conflicts": 0,
                "generated_view_target_or_group_conflicts": 0,
            },
            "train_manifest": "multiview_train.jsonl",
            "train_manifest_sha256": _sha256(train_path),
            "publication": "no_clobber_directory_reservation_hardlinks_contract_last",
            "commit_marker": "dataset.contract.json",
            "publication_complete": True,
            "production_route_authorized": False,
            "warning": (
                "This is a train-only Paddle pseudo-label overlay source, not a directly consumable "
                "optimizer manifest. A strict group-aware overlay loader must be implemented and tested "
                "before training. It does not authorize a production route or establish held-out business accuracy."
            ),
        }
        contract_output = stage / "dataset.contract.json"
        _write_json(contract_output, summary)
        stage_files.append((contract_output, _file_identity(contract_output)))

        # Reserve the final directory atomically.  Every published item is a
        # no-clobber hard link from the same-filesystem stage.  The contract is
        # linked last and is the commit marker consumers must require.
        _assert_bindings_unchanged(bindings)
        _assert_files_unchanged(stage_files, description="recipient multiview stage")
        output.mkdir()
        published_directories.append((output, _directory_identity(output)))
        published_images = output / "images"
        published_images.mkdir()
        published_directories.append(
            (published_images, _directory_identity(published_images))
        )
        for source_file, source_identity in sorted(
            (item for item in stage_files if item[0].parent == images_dir),
            key=lambda item: item[0].name,
        ):
            if not _same_file_identity(source_file, source_identity):
                raise ValueError(f"recipient multiview staged image changed: {source_file}")
            destination = published_images / source_file.name
            os.link(source_file, destination)
            published_files.append((destination, _file_identity(destination)))
        published_train = output / train_path.name
        if not _same_file_identity(train_path, dict(stage_files)[train_path]):
            raise ValueError(f"recipient multiview staged manifest changed: {train_path}")
        os.link(train_path, published_train)
        published_files.append((published_train, _file_identity(published_train)))
        _assert_bindings_unchanged(bindings)
        _assert_files_unchanged(stage_files, description="recipient multiview stage")
        _assert_files_unchanged(
            published_files,
            description="published recipient multiview evidence",
        )
        published_contract = output / contract_output.name
        os.link(contract_output, published_contract)
        published_files.append((published_contract, _file_identity(published_contract)))
        publication_complete = True
        _cleanup_owned_tree(files=stage_files, directories=stage_directories)
        return summary
    except BaseException:
        if not publication_complete:
            _cleanup_owned_tree(
                files=published_files,
                directories=published_directories,
            )
        _cleanup_owned_tree(files=stage_files, directories=stage_directories)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export train-only production-homologous recipient multiview teacher crops"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = export_recipient_multiview_teacher(
        manifest=args.manifest,
        dataset_contract=args.dataset_contract,
        dataset_root=args.dataset_root,
        output_dir=args.output,
    )
    print(
        "recipient_multiview_teacher_export "
        f"train_sources={summary['source_train_recipient_records']} "
        f"records={summary['output_records']} "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
