"""Train and export a Paddle-free receipt-field CTC recognizer.

The recognizer is deliberately small: the receipt detector already localises a
field, so OCR only needs to read a clean field crop.  Training uses PyTorch,
but the exported artifact is a fixed-shape ONNX graph plus a UTF-8 character
map and contract that can be consumed by a .NET ONNX Runtime worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image

from .labels import DETECTION_CLASSES
from .otherimages_paddle_teacher import canonical_paddle_color_contract


RECOGNIZER_SCHEMA_VERSION = 1
DEFAULT_IMAGE_HEIGHT = 48
DEFAULT_IMAGE_WIDTH = 768
GENERIC_TEXT_LINE_FIELD = "generic_text_line"
GENERIC_TEXT_LINE_RECORD_KIND = "otherimages_generic_text_line_record_v1"
GENERIC_TEXT_LINE_MANIFEST_NAME = "generic_text_lines.jsonl"
GENERIC_TEXT_LINE_CONTRACT_KIND = "otherimages_generic_text_line_dataset_contract_v1"
GENERIC_TEXT_LINE_RECEIPT_KIND = "otherimages_generic_text_line_dataset_receipt_v1"
SOURCE_TEACHER_CONTRACT_KIND = "otherimages_paddle_teacher_contract_v1"
SOURCE_TEACHER_RECEIPT_KIND = "otherimages_paddle_teacher_receipt_v1"
LEGACY_RECEIPT_PREPROCESS = (
    "RGB crop -> grayscale -> aspect-preserving resize -> white letterbox -> divide by 255.0"
)
GENERIC_TEXT_LINE_PREPROCESS = "opencv_exact_rgb_gray_letterbox_v1"
GENERIC_TEXT_LINE_PADDLE_COLOR_CONTRACT = canonical_paddle_color_contract()
# This is a recognizer-only vocabulary.  DETECTION_CLASSES is the detector ABI
# and must never grow a synthetic whole-document line class.
RECOGNIZER_FIELDS = (*DETECTION_CLASSES, GENERIC_TEXT_LINE_FIELD)


def _validate_recognizer_field_mode(fields: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(fields))
    invalid = sorted(set(selected) - set(RECOGNIZER_FIELDS))
    if not selected or invalid:
        raise ValueError(f"fields must be a non-empty subset of: {','.join(RECOGNIZER_FIELDS)}")
    if GENERIC_TEXT_LINE_FIELD in selected and selected != (GENERIC_TEXT_LINE_FIELD,):
        raise ValueError("generic_text_line must be trained and exported as an independent recognizer")
    return selected


def _preprocess_contract(fields: Sequence[str]) -> str:
    selected = _validate_recognizer_field_mode(fields)
    return (
        GENERIC_TEXT_LINE_PREPROCESS
        if selected == (GENERIC_TEXT_LINE_FIELD,)
        else LEGACY_RECEIPT_PREPROCESS
    )


@dataclass(frozen=True)
class RecognizerConfig:
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    base_channels: int = 64
    hidden_size: int = 128
    lstm_layers: int = 2

    def validate(self) -> None:
        if self.image_height < 16 or self.image_width < 64:
            raise ValueError("image_height must be at least 16 and image_width must be at least 64")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if self.hidden_size < 16:
            raise ValueError("hidden_size must be at least 16")
        if self.lstm_layers < 1:
            raise ValueError("lstm_layers must be at least 1")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_binding(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def _binding_matches(path: Path, value: object, *, expected_name: str) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("path") == expected_name
        and _file_binding(path) == {
            "path": value.get("path"),
            "sha256": value.get("sha256"),
            "size_bytes": value.get("size_bytes"),
            "line_count": value.get("line_count"),
        }
    )


def _json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} is forbidden")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r}")
            output[key] = value
        return output

    try:
        text = path.read_bytes().decode("utf-8")
        if text.startswith("\ufeff"):
            raise ValueError("UTF-8 BOM is forbidden")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict UTF-8 JSON: {error}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _pixel_sha256(rgb: np.ndarray) -> str:
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(pixels.shape).encode("ascii"))
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _verify_generic_line_dataset(
    *,
    records_path: Path,
    dataset_root: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    """Verify the sealed materializer closure before generic-line training/eval."""
    if records_path.name != GENERIC_TEXT_LINE_MANIFEST_NAME:
        raise ValueError(
            f"{GENERIC_TEXT_LINE_FIELD} requires records basename {GENERIC_TEXT_LINE_MANIFEST_NAME}"
        )
    publication = records_path.parent
    if dataset_root != publication:
        raise ValueError(f"{GENERIC_TEXT_LINE_FIELD} dataset_root must be the sealed manifest directory")
    contract_path = publication / "dataset.contract.json"
    receipt_path = publication / "dataset.receipt.json"
    if not contract_path.is_file() or not receipt_path.is_file():
        raise ValueError("generic_text_line requires sibling dataset.contract.json and dataset.receipt.json")
    contract = _json_object(contract_path)
    receipt = _json_object(receipt_path)
    if (
        contract.get("schema_version") != RECOGNIZER_SCHEMA_VERSION
        or contract.get("kind") != GENERIC_TEXT_LINE_CONTRACT_KIND
        or contract.get("sealed") is not True
        or contract.get("field") != GENERIC_TEXT_LINE_FIELD
        or contract.get("records") != GENERIC_TEXT_LINE_MANIFEST_NAME
        or contract.get("training_authorization") is not True
        or contract.get("training_authorization_source") != "explicit_materializer_flag"
    ):
        raise ValueError("generic_text_line dataset contract is unsupported, unsealed, or unauthorized")
    raw_output = contract.get("output_directory")
    if not isinstance(raw_output, str) or Path(raw_output).expanduser().resolve(strict=True) != publication:
        raise ValueError("generic_text_line dataset contract output_directory differs from the publication")
    if contract.get("truth_semantics") != "teacher_parity_only_not_independent_business_truth":
        raise ValueError("generic_text_line dataset contract truth semantics are invalid")
    crop_recipe = contract.get("crop_recipe")
    if (
        not isinstance(crop_recipe, Mapping)
        or crop_recipe.get("paddle_color_contract") != GENERIC_TEXT_LINE_PADDLE_COLOR_CONTRACT
    ):
        raise ValueError(
            "generic_text_line dataset does not bind the canonical RGB byte-order contract"
        )

    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("generic_text_line dataset artifacts must be an object")
    manifest_bytes = records_path.read_bytes()
    if (
        manifest_bytes.startswith(b"\xef\xbb\xbf")
        or (manifest_bytes and not manifest_bytes.endswith(b"\n"))
        or b"\n\n" in manifest_bytes
    ):
        raise ValueError("generic_text_line manifest must be canonical UTF-8 JSONL without BOM/blank lines")
    manifest_binding = artifacts.get("manifest")
    if not _binding_matches(
        records_path,
        manifest_binding,
        expected_name=GENERIC_TEXT_LINE_MANIFEST_NAME,
    ):
        raise ValueError("generic_text_line manifest differs from its sealed contract binding")
    counts = contract.get("counts")
    if not isinstance(counts, Mapping) or counts.get("line_records") != len(records):
        raise ValueError("generic_text_line manifest record count differs from its sealed contract")
    by_split = Counter(str(record["split"]) for record in records)
    if counts.get("by_split") != {name: int(by_split[name]) for name in ("train", "val", "test")}:
        raise ValueError("generic_text_line split counts differ from its sealed contract")
    if counts.get("training_eligible_lines") != by_split["train"] or counts.get(
        "evaluation_only_lines"
    ) != by_split["val"] + by_split["test"]:
        raise ValueError("generic_text_line split-use counts differ from its sealed contract")

    crop_bindings: list[dict[str, object]] = []
    declared_images: set[str] = set()
    for record in records:
        split = str(record["split"])
        training = split == "train"
        if (
            record.get("kind") != GENERIC_TEXT_LINE_RECORD_KIND
            or record.get("training_eligible") is not training
            or record.get("evaluation_only") is not (not training)
            or record.get("held_out") is not (not training)
            or record.get("truth_semantics") != "paddle_teacher_parity_not_independent_truth"
            or record.get("label_source") != "paddle_db_cls_rec_three_view_consensus"
        ):
            raise ValueError(f"generic_text_line record provenance/split flags are invalid: {record['id']}")
        image_path = Path(record["image_path"])
        image_relative = image_path.relative_to(publication).as_posix()
        declared_images.add(image_relative)
        data = image_path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        size_bytes = len(data)
        if sha256 != record.get("crop_sha256") or size_bytes != record.get("crop_size_bytes"):
            raise ValueError(f"generic_text_line crop bytes differ from manifest: {record['id']}")
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if (
            _pixel_sha256(rgb) != record.get("crop_pixel_sha256")
            or int(rgb.shape[1]) != record.get("crop_width")
            or int(rgb.shape[0]) != record.get("crop_height")
        ):
            raise ValueError(f"generic_text_line crop pixels differ from manifest: {record['id']}")
        crop_bindings.append(
            {
                "path": image_relative,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "pixel_sha256": record["crop_pixel_sha256"],
                "width": record["crop_width"],
                "height": record["crop_height"],
            }
        )
    crop_bindings.sort(key=lambda value: str(value["path"]))
    crop_artifact = artifacts.get("crops")
    if not isinstance(crop_artifact, Mapping) or crop_artifact != {
        "count": len(crop_bindings),
        "size_bytes": sum(int(value["size_bytes"]) for value in crop_bindings),
        "closure_sha256": _canonical_sha256(crop_bindings),
    }:
        raise ValueError("generic_text_line crop closure differs from its sealed contract")
    image_root = publication / "images"
    observed_images = {
        path.relative_to(publication).as_posix() for path in image_root.rglob("*") if path.is_file()
    }
    if observed_images != declared_images:
        raise ValueError("generic_text_line images directory membership differs from its sealed manifest")

    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("generic_text_line teacher inputs must be an object")
    raw_teacher = inputs.get("teacher_directory")
    if not isinstance(raw_teacher, str):
        raise ValueError("generic_text_line teacher_directory is invalid")
    teacher_root = Path(raw_teacher).expanduser().resolve(strict=True)
    for key, name in (
        ("teacher_manifest", "teacher_manifest.jsonl"),
        ("teacher_contract", "teacher.contract.json"),
        ("teacher_receipt", "teacher.receipt.json"),
    ):
        if not _binding_matches(teacher_root / name, inputs.get(key), expected_name=name):
            raise ValueError(f"generic_text_line source teacher binding changed: {name}")
    source_contract = _json_object(teacher_root / "teacher.contract.json")
    source_receipt = _json_object(teacher_root / "teacher.receipt.json")
    source_closure = inputs.get("teacher_contract_closure_sha256")
    source_closure_payload = {
        "schema_version": RECOGNIZER_SCHEMA_VERSION,
        "inputs": source_contract.get("inputs"),
        "configuration": source_contract.get("configuration"),
        "counts": source_contract.get("counts"),
        "split_use": source_contract.get("split_use"),
        "artifacts": source_contract.get("artifacts"),
    }
    if (
        source_contract.get("schema_version") != RECOGNIZER_SCHEMA_VERSION
        or source_contract.get("kind") != SOURCE_TEACHER_CONTRACT_KIND
        or source_contract.get("sealed") is not True
        or source_contract.get("training_authorization") is not False
        or source_contract.get("closure_sha256") != source_closure
        or source_contract.get("closure_sha256") != _canonical_sha256(source_closure_payload)
        or source_receipt.get("schema_version") != RECOGNIZER_SCHEMA_VERSION
        or source_receipt.get("kind") != SOURCE_TEACHER_RECEIPT_KIND
        or source_receipt.get("sealed") is not True
        or source_receipt.get("contract_closure_sha256") != source_closure
        or not _binding_matches(
            teacher_root / "teacher.contract.json",
            source_receipt.get("contract"),
            expected_name="teacher.contract.json",
        )
    ):
        raise ValueError("generic_text_line source teacher closure is invalid")

    closure_payload = {
        "schema_version": RECOGNIZER_SCHEMA_VERSION,
        "inputs": inputs,
        "crop_recipe": contract.get("crop_recipe"),
        "counts": counts,
        "split_use": contract.get("split_use"),
        "artifacts": artifacts,
    }
    if contract.get("closure_sha256") != _canonical_sha256(closure_payload):
        raise ValueError("generic_text_line dataset contract closure SHA-256 is invalid")
    if (
        receipt.get("schema_version") != RECOGNIZER_SCHEMA_VERSION
        or receipt.get("kind") != GENERIC_TEXT_LINE_RECEIPT_KIND
        or receipt.get("sealed") is not True
        or receipt.get("contract_closure_sha256") != contract.get("closure_sha256")
        or not _binding_matches(contract_path, receipt.get("contract"), expected_name="dataset.contract.json")
    ):
        raise ValueError("generic_text_line dataset receipt does not bind the sealed contract")


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OCR training requires PyTorch. Install the server's CPU/CUDA-compatible torch wheel first, "
            "then install requirements-train-ocr.txt."
        ) from error
    return torch, nn


def build_ctc_recognizer(*, vocab_size: int, config: RecognizerConfig) -> Any:
    """Create the fixed-input CNN + BiLSTM CTC recognizer without Paddle."""
    if vocab_size < 2:
        raise ValueError("vocab_size must include CTC blank plus at least one character")
    config.validate()
    _, nn = _require_torch()

    class ReceiptOcrCtc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = config.base_channels
            middle = base * 2
            high = base * 4
            self.encoder = nn.Sequential(
                nn.Conv2d(1, base, kernel_size=3, padding=1),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(base, middle, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(middle, high, kernel_size=3, padding=1),
                nn.BatchNorm2d(high),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Conv2d(high, high, kernel_size=3, padding=1),
                nn.BatchNorm2d(high),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Conv2d(high, high, kernel_size=3, padding=1),
                nn.BatchNorm2d(high),
                nn.ReLU(inplace=True),
            )
            self.sequence = nn.LSTM(
                input_size=high,
                hidden_size=config.hidden_size,
                num_layers=config.lstm_layers,
                bidirectional=True,
            )
            self.classifier = nn.Linear(config.hidden_size * 2, vocab_size)

        def forward(self, image: Any) -> Any:
            features = self.encoder(image)
            # Collapsing the remaining vertical pixels keeps the time axis tied
            # to the original horizontal text order while accepting 48px crops.
            sequence = features.mean(dim=2).permute(2, 0, 1)
            sequence, _ = self.sequence(sequence)
            return self.classifier(sequence)

    return ReceiptOcrCtc()


def _parse_record(line: str, *, records_path: Path, line_number: int, dataset_root: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} is forbidden")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r}")
            output[key] = value
        return output

    try:
        value: Any = json.loads(
            line,
            parse_constant=reject_constant,
            object_pairs_hook=object_without_duplicates,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{records_path}:{line_number}: invalid JSON: {error}") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{records_path}:{line_number}: record must be an object")
    image_value = value.get("image")
    text = value.get("text")
    field = value.get("field")
    split = value.get("split")
    group_id = value.get("group_id")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"{records_path}:{line_number}: image must be a non-empty string")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{records_path}:{line_number}: text must be a non-empty string")
    if not isinstance(field, str) or field not in RECOGNIZER_FIELDS:
        raise ValueError(f"{records_path}:{line_number}: invalid field")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"{records_path}:{line_number}: split must be train, val, or test")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError(f"{records_path}:{line_number}: group_id must be a non-empty string")
    image_path = (dataset_root / image_value).resolve()
    try:
        image_path.relative_to(dataset_root)
    except ValueError:
        raise ValueError(f"{records_path}:{line_number}: image escapes dataset root") from None
    if not image_path.is_file():
        raise FileNotFoundError(f"{records_path}:{line_number}: image not found: {image_path}")
    return {
        "image_path": image_path,
        "text": text,
        "field": field,
        "split": split,
        "id": str(value.get("id", image_value)),
        "group_id": group_id,
        "paddle_text": value.get("paddle_text") if isinstance(value.get("paddle_text"), str) else text,
        "source_text": value.get("source_text") if isinstance(value.get("source_text"), str) else None,
        "semantic_value": value.get("semantic_value") if isinstance(value.get("semantic_value"), str) else None,
        "label_source": value.get("label_source") if isinstance(value.get("label_source"), str) else "unspecified",
        "source": value.get("source") if isinstance(value.get("source"), str) else None,
        "result_json": value.get("result_json") if isinstance(value.get("result_json"), str) else None,
        "crop_sha256": value.get("crop_sha256") if isinstance(value.get("crop_sha256"), str) else None,
        "kind": value.get("kind") if isinstance(value.get("kind"), str) else None,
        "training_eligible": value.get("training_eligible"),
        "evaluation_only": value.get("evaluation_only"),
        "held_out": value.get("held_out"),
        "truth_semantics": value.get("truth_semantics") if isinstance(value.get("truth_semantics"), str) else None,
        "crop_pixel_sha256": (
            value.get("crop_pixel_sha256") if isinstance(value.get("crop_pixel_sha256"), str) else None
        ),
        "crop_size_bytes": value.get("crop_size_bytes"),
        "crop_width": value.get("crop_width"),
        "crop_height": value.get("crop_height"),
    }


def load_records(
    records_path: Path,
    *,
    fields: Sequence[str] = DETECTION_CLASSES,
    dataset_root: Path | None = None,
) -> list[dict[str, object]]:
    """Read pseudo-label records without importing PyTorch."""
    fields = _validate_recognizer_field_mode(fields)
    records_path = records_path.resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    dataset_root = (dataset_root if dataset_root is not None else records_path.parent).resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    records: list[dict[str, object]] = []
    ids: set[str] = set()
    images: set[str] = set()
    group_splits: dict[str, str] = {}
    with records_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = _parse_record(line, records_path=records_path, line_number=line_number, dataset_root=dataset_root)
            if record["field"] in fields:
                record_id = str(record["id"])
                image_key = Path(record["image_path"]).as_posix().casefold()
                group_id = str(record["group_id"])
                split = str(record["split"])
                if record_id in ids:
                    raise ValueError(f"{records_path}:{line_number}: duplicate id {record_id!r}")
                if image_key in images:
                    raise ValueError(f"{records_path}:{line_number}: duplicate image across records")
                prior_split = group_splits.get(group_id)
                if prior_split is not None and prior_split != split:
                    raise ValueError(
                        f"{records_path}:{line_number}: group_id {group_id!r} appears in both "
                        f"{prior_split} and {split} splits"
                    )
                ids.add(record_id)
                images.add(image_key)
                group_splits[group_id] = split
                records.append(record)
    if not records:
        raise ValueError(f"No records for requested fields in {records_path}")
    if fields == (GENERIC_TEXT_LINE_FIELD,):
        _verify_generic_line_dataset(
            records_path=records_path,
            dataset_root=dataset_root,
            records=records,
        )
    return records


def _charset(records: Iterable[Mapping[str, object]]) -> list[str]:
    characters = sorted({character for record in records for character in str(record["text"])})
    if not characters:
        raise ValueError("No characters found in training records")
    return characters


def _minimum_ctc_steps(text: str) -> int:
    """Return the shortest CTC alignment length for a target string.

    Consecutive duplicate characters require an intervening blank, so e.g.
    ``"111"`` needs five time steps rather than three.
    """
    return len(text) + sum(left == right for left, right in zip(text, text[1:]))


def _ctc_time_steps(config: RecognizerConfig) -> int:
    # The first two max-pooling layers halve horizontal resolution; later
    # pooling only reduces height.  Keep this alongside the architecture so a
    # configuration cannot silently produce impossible CTC samples.
    return config.image_width // 4


def _validate_ctc_capacity(records: Iterable[Mapping[str, object]], *, config: RecognizerConfig) -> None:
    available = _ctc_time_steps(config)
    for record in records:
        text = str(record["text"])
        required = _minimum_ctc_steps(text)
        if required > available:
            raise ValueError(
                "OCR target cannot fit the recognizer CTC time axis: "
                f"id={record['id']}, required={required}, available={available}, text={text!r}. "
                "Increase --image-width or exclude this record."
            )


def _validate_non_train_characters(records: Iterable[Mapping[str, object]], *, train_characters: set[str]) -> None:
    for record in records:
        unknown = sorted(set(str(record["text"])) - train_characters)
        if unknown:
            raise ValueError(
                "Validation/test record contains characters absent from the training charset: "
                f"id={record['id']}, characters={''.join(unknown)!r}. "
                "Move it to a human-held-out evaluation set with a frozen external charset, "
                "or add representative training examples."
            )


def _field_split_counts(records: Iterable[Mapping[str, object]], *, fields: Sequence[str]) -> dict[str, dict[str, int]]:
    """Return auditable per-field split sizes for the declared deployment fields."""
    counts: dict[str, Counter[str]] = {field: Counter() for field in fields}
    for record in records:
        field = str(record["field"])
        if field in counts:
            counts[field][str(record["split"])] += 1
    return {
        field: {split: int(counts[field][split]) for split in ("train", "val", "test")}
        for field in fields
    }


def _require_train_and_validation_coverage(field_counts: Mapping[str, Mapping[str, int]]) -> None:
    missing_train = [field for field, counts in field_counts.items() if counts["train"] <= 0]
    if missing_train:
        raise ValueError(
            "No training samples remain after filtering for requested field(s): "
            f"{','.join(missing_train)}. Remove those --fields or rebuild pseudo labels with more reviewed data."
        )
    missing_validation = [field for field, counts in field_counts.items() if counts["val"] <= 0]
    if missing_validation:
        raise ValueError(
            "No validation samples remain after filtering for requested field(s): "
            f"{','.join(missing_validation)}. Rebuild pseudo labels with a different --split-seed or more data."
        )


def _resolve_device(torch: Any, requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for OCR training but PyTorch CUDA is unavailable")
        return requested
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested for OCR training but is unavailable")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")


def _opencv_exact_rgb_gray_letterbox(rgb: np.ndarray, *, config: RecognizerConfig) -> np.ndarray:
    """Return the exact uint8 canvas used by the generic-line deployment ABI.

    Keep this deliberately explicit rather than relying on OpenCV's color
    conversion coefficients.  The integer Rec.601 coefficients, Python
    ties-to-even ``round``, INTER_LINEAR_EXACT resize, and centering rule are
    independently reproducible by the .NET CPU runtime.
    """
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.shape[0] <= 0 or pixels.shape[1] <= 0:
        raise ValueError("generic text-line input must be a non-empty RGB8 image")
    channels = pixels.astype(np.uint32)
    gray = (
        19_595 * channels[:, :, 0]
        + 38_470 * channels[:, :, 1]
        + 7_471 * channels[:, :, 2]
        + 32_768
    ) >> 16
    gray_u8 = np.ascontiguousarray(gray, dtype=np.uint8)
    source_height, source_width = gray_u8.shape
    scale = min(config.image_width / source_width, config.image_height / source_height)
    width = max(1, min(config.image_width, int(round(source_width * scale))))
    height = max(1, min(config.image_height, int(round(source_height * scale))))
    resized = cv2.resize(gray_u8, (width, height), interpolation=cv2.INTER_LINEAR_EXACT)
    canvas = np.full((config.image_height, config.image_width), 255, dtype=np.uint8)
    top = (config.image_height - height) // 2
    left = (config.image_width - width) // 2
    canvas[top : top + height, left : left + width] = resized
    return canvas


def preprocess_image(
    image_path: Path,
    *,
    config: RecognizerConfig,
    field: str | None = None,
) -> np.ndarray:
    """Return fixed NCHW float32 preprocessing declared in the ONNX contract.

    Omitting ``field`` preserves the existing receipt-field Pillow ABI.  The
    generic recognizer is isolated and uses its cross-runtime OpenCV-exact ABI.
    """
    if field is not None and field not in RECOGNIZER_FIELDS:
        raise ValueError(f"unsupported recognizer field for preprocessing: {field}")
    if field == GENERIC_TEXT_LINE_FIELD:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        canvas = _opencv_exact_rgb_gray_letterbox(rgb, config=config)
        return (canvas.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
    with Image.open(image_path) as image:
        gray = image.convert("L")
        scale = min(config.image_width / gray.width, config.image_height / gray.height)
        width = max(1, min(config.image_width, int(round(gray.width * scale))))
        height = max(1, min(config.image_height, int(round(gray.height * scale))))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        gray = gray.resize((width, height), resampling)
        canvas = np.full((config.image_height, config.image_width), 255, dtype=np.uint8)
        top = (config.image_height - height) // 2
        left = (config.image_width - width) // 2
        canvas[top : top + height, left : left + width] = np.asarray(gray, dtype=np.uint8)
    return (canvas.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]


def _resize_to_tensor(
    image_path: Path,
    *,
    config: RecognizerConfig,
    field: str,
    torch: Any,
) -> Any:
    """Torch wrapper for the shared train/ONNX image preprocessing."""
    return torch.from_numpy(preprocess_image(image_path, config=config, field=field)[0])


class _ReceiptOcrDataset:
    """Pickle-safe dataset for Windows DataLoader spawn workers."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        character_to_id: Mapping[str, int],
        config: RecognizerConfig,
    ) -> None:
        self._records = tuple(dict(record) for record in records)
        self._character_to_id = dict(character_to_id)
        self._config = config

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Any, Any, str, str]:
        torch, _ = _require_torch()
        record = self._records[index]
        text = str(record["text"])
        targets = torch.tensor(
            [self._character_to_id[character] for character in text],
            dtype=torch.long,
        )
        field = str(record["field"])
        image = _resize_to_tensor(
            Path(record["image_path"]),
            config=self._config,
            field=field,
            torch=torch,
        )
        return image, targets, text, field


def _make_dataset(
    records: Sequence[Mapping[str, object]],
    *,
    character_to_id: Mapping[str, int],
    config: RecognizerConfig,
    torch: Any,
) -> Any:
    del torch  # Preserve the internal call ABI while keeping the dataset pickle-safe.
    return _ReceiptOcrDataset(records, character_to_id=character_to_id, config=config)


def _collate_batch(
    batch: Sequence[tuple[Any, Any, str, str]], *, torch: Any
) -> tuple[Any, Any, Any, list[str], list[str]]:
    images = torch.stack([item[0] for item in batch])
    target_lengths = torch.tensor([item[1].numel() for item in batch], dtype=torch.long)
    targets = torch.cat([item[1] for item in batch])
    return images, targets, target_lengths, [item[2] for item in batch], [item[3] for item in batch]


def _collate_batch_worker(
    batch: Sequence[tuple[Any, Any, str, str]],
) -> tuple[Any, Any, Any, list[str], list[str]]:
    torch, _ = _require_torch()
    return _collate_batch(batch, torch=torch)


def decode_ctc_logits(logits: np.ndarray, *, characters: Sequence[str]) -> list[str]:
    """Greedily decode `[time, batch, class]` CTC logits without PyTorch."""
    values = np.asarray(logits)
    if values.ndim != 3:
        raise ValueError("CTC logits must have shape [time,batch,class]")
    if values.shape[2] != len(characters) + 1:
        raise ValueError(
            f"CTC logits class count {values.shape[2]} does not match blank plus {len(characters)} characters"
        )
    indices = values.argmax(axis=2)
    decoded: list[str] = []
    for batch_index in range(indices.shape[1]):
        previous = -1
        output: list[str] = []
        for current_value in indices[:, batch_index]:
            current = int(current_value)
            if current != 0 and current != previous:
                output.append(characters[current - 1])
            previous = current
        decoded.append("".join(output))
    return decoded


def _decode_logits(logits: Any, *, characters: Sequence[str]) -> list[str]:
    """CTC greedy decode; class 0 is blank and class N maps to characters[N-1]."""
    return decode_ctc_logits(logits.detach().cpu().numpy(), characters=characters)


def _evaluate(
    model: Any,
    loader: Any,
    *,
    criterion: Any,
    device: str,
    characters: Sequence[str],
    torch: Any,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    exact_matches = 0
    per_field: dict[str, Counter[str]] = {}
    with torch.no_grad():
        for images, targets, target_lengths, texts, fields in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(2)
            # CTCLoss accepts GPU logits/targets, while its length tensors are
            # deliberately kept on CPU for compatibility across PyTorch CUDA
            # versions.
            input_lengths = torch.full((images.shape[0],), logits.shape[0], dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += float(loss.detach().cpu()) * len(texts)
            total_items += len(texts)
            predictions = _decode_logits(logits, characters=characters)
            for field, predicted, expected in zip(fields, predictions, texts):
                counters = per_field.setdefault(field, Counter())
                counters["records"] += 1
                counters["exact_matches"] += int(predicted == expected)
                exact_matches += int(predicted == expected)
    if total_items == 0:
        raise ValueError("Validation set is empty")
    return {
        "loss": total_loss / total_items,
        "exact_match": exact_matches / total_items,
        "by_field": {
            field: {
                "records": int(counters["records"]),
                "exact_matches": int(counters["exact_matches"]),
                "exact_match": counters["exact_matches"] / counters["records"],
            }
            for field, counters in sorted(per_field.items())
        },
    }


def _write_checkpoint(path: Path, payload: Mapping[str, object], *, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _validation_due(*, epoch: int, epochs: int, validation_every: int) -> bool:
    """Validate on the requested cadence and unconditionally on the last epoch."""
    if validation_every <= 0:
        raise ValueError("validation_every must be positive")
    return epoch == epochs or epoch % validation_every == 0


def train_recognizer(
    *,
    records_path: Path,
    output_dir: Path,
    fields: Sequence[str] = DETECTION_CLASSES,
    dataset_root: Path | None = None,
    config: RecognizerConfig = RecognizerConfig(),
    device: str = "auto",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    cuda_tf32: bool = False,
    cudnn_benchmark: bool = False,
    validation_every: int = 1,
    train_progress_every: int = 0,
) -> Path:
    """Train a field-crop recognizer and return the best checkpoint path."""
    config.validate()
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if persistent_workers and num_workers <= 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if validation_every <= 0:
        raise ValueError("validation_every must be positive")
    if train_progress_every < 0:
        raise ValueError("train_progress_every cannot be negative")
    selected_fields = _validate_recognizer_field_mode(fields)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"training output already contains files: {output_dir}. Choose a new empty directory.")
    records = load_records(records_path, fields=selected_fields, dataset_root=dataset_root)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "val"]
    if not train_records:
        raise ValueError("No train split records found")
    if not validation_records:
        raise ValueError("No val split records found; rebuild pseudo labels with a non-zero --validation-ratio")
    field_counts = _field_split_counts(records, fields=selected_fields)
    _require_train_and_validation_coverage(field_counts)
    # Build the deployment alphabet from train only.  Pulling characters from
    # val/test makes an OOV metric look better than it really is.
    characters = _charset(train_records)
    _validate_non_train_characters(
        validation_records,
        train_characters=set(characters),
    )
    if selected_fields == (GENERIC_TEXT_LINE_FIELD,):
        test_records = [record for record in records if record["split"] == "test"]
        _validate_non_train_characters(
            test_records,
            train_characters=set(characters),
        )
    _validate_ctc_capacity((*train_records, *validation_records), config=config)
    character_to_id = {character: index for index, character in enumerate(characters, start=1)}
    torch, _ = _require_torch()
    target_device = _resolve_device(torch, device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if target_device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        if cuda_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    train_dataset = _make_dataset(train_records, character_to_id=character_to_id, config=config, torch=torch)
    validation_dataset = _make_dataset(validation_records, character_to_id=character_to_id, config=config, torch=torch)
    loader_worker_options = (
        {"persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor}
        if num_workers > 0
        else {}
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_batch_worker,
        pin_memory=target_device.startswith("cuda"),
        **loader_worker_options,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_batch_worker,
        pin_memory=target_device.startswith("cuda"),
        **loader_worker_options,
    )
    model = build_ctc_recognizer(vocab_size=len(characters) + 1, config=config).to(target_device)
    # Capacity is pre-validated above.  Keep infinity visible if the model is
    # changed later instead of silently treating an impossible label as zero.
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output_dir / "charset.json",
        {
            "schema_version": RECOGNIZER_SCHEMA_VERSION,
            "blank_index": 0,
            "characters": characters,
            "sha256": hashlib.sha256("".join(characters).encode("utf-8")).hexdigest(),
        },
    )

    history: list[dict[str, object]] = []
    best_loss = float("inf")
    best_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_index, (images, targets, target_lengths, texts, _fields) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(target_device)
            targets = targets.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            log_probs = logits.log_softmax(2)
            input_lengths = torch.full((images.shape[0],), logits.shape[0], dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(texts)
            total_items += len(texts)
            if train_progress_every and (
                batch_index % train_progress_every == 0 or batch_index == len(train_loader)
            ):
                print(
                    f"epoch {epoch}/{epochs} batch {batch_index}/{len(train_loader)}: "
                    f"train_loss_so_far={total_loss / max(total_items, 1):.4f}"
                )
        train_loss = total_loss / max(total_items, 1)
        validation_ran = _validation_due(
            epoch=epoch,
            epochs=epochs,
            validation_every=validation_every,
        )
        validation = (
            _evaluate(
                model,
                validation_loader,
                criterion=criterion,
                device=target_device,
                characters=characters,
                torch=torch,
            )
            if validation_ran
            else None
        )
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_ran": validation_ran,
        }
        if validation is not None:
            epoch_record.update(
                {
                    "val_loss": validation["loss"],
                    "val_exact_match": validation["exact_match"],
                    "val_by_field": validation["by_field"],
                }
            )
        history.append(epoch_record)
        training_options = {
            "seed": seed,
            "device": target_device,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
            "cuda_tf32": cuda_tf32,
            "cudnn_benchmark": cudnn_benchmark,
            "validation_every": validation_every,
            "train_progress_every": train_progress_every,
        }
        checkpoint_payload = {
            "schema_version": RECOGNIZER_SCHEMA_VERSION,
            "kind": "receipt_ocr_ctc_v1",
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "characters": characters,
            "fields": list(selected_fields),
            "field_counts": field_counts,
            "epoch": epoch,
            "metrics": epoch_record,
            "training_options": training_options,
        }
        _write_checkpoint(output_dir / "last.pt", checkpoint_payload, torch=torch)
        if validation is not None and float(validation["loss"]) < best_loss:
            best_loss = float(validation["loss"])
            _write_checkpoint(best_path, checkpoint_payload, torch=torch)
        _atomic_write_json(
            output_dir / "training_history.json",
            {
                "schema_version": RECOGNIZER_SCHEMA_VERSION,
                "records": history,
                "field_counts": field_counts,
                "training_options": training_options,
                "warning": "Validation records are PaddleOCR pseudo labels unless you replace them with reviewed labels.",
            },
        )
        if validation is None:
            print(f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} validation=skipped")
        else:
            print(
                f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                f"val_loss={float(validation['loss']):.4f} "
                f"val_exact_match={float(validation['exact_match']):.2%}"
            )
    return best_path


def _load_checkpoint(path: Path, *, torch: Any) -> Mapping[str, object]:
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was added.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("OCR checkpoint must be a mapping")
    if payload.get("schema_version") != RECOGNIZER_SCHEMA_VERSION or payload.get("kind") != "receipt_ocr_ctc_v1":
        raise ValueError("Unsupported OCR checkpoint schema")
    return payload


def export_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
) -> tuple[Path, Path, Path]:
    """Export a trained CTC model plus charset and self-describing contract."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = output_path.resolve()
    charset_path = output_path.with_suffix(".charset.json")
    contract_path = output_path.with_suffix(".contract.json")
    existing = next((path for path in (output_path, charset_path, contract_path) if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"Refusing to overwrite OCR export artifact: {existing}")
    torch, _ = _require_torch()
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    raw_config = payload.get("config")
    characters = payload.get("characters")
    fields = payload.get("fields")
    raw_field_counts = payload.get("field_counts")
    state_dict = payload.get("state_dict")
    if (
        not isinstance(raw_config, Mapping)
        or not isinstance(characters, list)
        or not isinstance(fields, list)
        or not isinstance(raw_field_counts, Mapping)
    ):
        raise ValueError("OCR checkpoint is missing config, characters, fields, or field_counts")
    if not all(isinstance(character, str) and len(character) == 1 for character in characters):
        raise ValueError("OCR checkpoint characters must be single Unicode code points")
    if not all(isinstance(field, str) and field in RECOGNIZER_FIELDS for field in fields):
        raise ValueError("OCR checkpoint fields are invalid")
    _validate_recognizer_field_mode(fields)
    field_counts: dict[str, dict[str, int]] = {}
    for field in fields:
        raw_counts = raw_field_counts.get(field)
        if not isinstance(raw_counts, Mapping):
            raise ValueError(f"OCR checkpoint field_counts are missing {field!r}")
        split_counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            value = raw_counts.get(split)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"OCR checkpoint field_counts[{field!r}][{split!r}] is invalid")
            split_counts[split] = value
        field_counts[field] = split_counts
    _require_train_and_validation_coverage(field_counts)
    try:
        config = RecognizerConfig(
            image_height=int(raw_config["image_height"]),
            image_width=int(raw_config["image_width"]),
            base_channels=int(raw_config.get("base_channels", 64)),
            hidden_size=int(raw_config["hidden_size"]),
            lstm_layers=int(raw_config["lstm_layers"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OCR checkpoint config is invalid") from error
    config.validate()
    model = build_ctc_recognizer(vocab_size=len(characters) + 1, config=config)
    if not isinstance(state_dict, Mapping):
        raise ValueError("OCR checkpoint state_dict is invalid")
    model.load_state_dict(state_dict)
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 1, config.image_height, config.image_width), dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:  # Older torch does not expose the dynamo argument.
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
        )
    with torch.no_grad():
        logits = model(dummy)
    charset_payload = {
        "schema_version": RECOGNIZER_SCHEMA_VERSION,
        "blank_index": 0,
        "characters": characters,
        "sha256": hashlib.sha256("".join(characters).encode("utf-8")).hexdigest(),
    }
    _atomic_write_json(charset_path, charset_payload)
    _atomic_write_json(
        contract_path,
        {
            "schema_version": RECOGNIZER_SCHEMA_VERSION,
            "kind": "receipt_ocr_ctc_v1",
            "onnx_file": output_path.name,
            "onnx_sha256": _sha256(output_path),
            "charset_file": charset_path.name,
            "charset_sha256": _sha256(charset_path),
            "fields": fields,
            "training_field_counts": field_counts,
            "input": {
                "name": "image",
                "dtype": "float32",
                "shape": [1, 1, config.image_height, config.image_width],
                "preprocess": _preprocess_contract(fields),
            },
            "output": {
                "name": "logits",
                "shape": [int(logits.shape[0]), 1, len(characters) + 1],
                "layout": "[time,batch,class]",
                "decoder": "ctc_greedy",
                "blank_index": 0,
            },
            "model": asdict(config),
        },
    )
    return output_path, charset_path, contract_path


def _parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(part.strip() for part in value.split(",") if part.strip())
    try:
        return _validate_recognizer_field_mode(fields)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "fields must select receipt detector fields or the independent "
            f"{GENERIC_TEXT_LINE_FIELD} recognizer; supported={','.join(RECOGNIZER_FIELDS)}"
        ) from None


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Paddle-free CTC receipt OCR recognizer")
    parser.add_argument("--records", type=Path, required=True, help="pseudo_labels.jsonl or reviewed JSONL")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root that owns image paths in --records; defaults to the records file directory",
    )
    parser.add_argument("--output", type=Path, required=True, help="New empty checkpoint output directory")
    parser.add_argument("--fields", type=_parse_fields, default=DETECTION_CLASSES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 on Windows unless your environment is configured for multiprocessing",
    )
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cuda-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument(
        "--validation-every",
        type=int,
        default=1,
        help="Validate every N epochs; the final epoch is always validated (default: 1)",
    )
    parser.add_argument(
        "--train-progress-every",
        type=int,
        default=0,
        help="Print running train loss every N batches; 0 disables batch progress output",
    )
    parser.add_argument("--onnx-output", type=Path, help="Optionally export the best checkpoint after training")
    return parser


def train_main(argv: list[str] | None = None) -> None:
    args = build_train_parser().parse_args(argv)
    config = RecognizerConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        base_channels=args.base_channels,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
    )
    try:
        checkpoint = train_recognizer(
            records_path=args.records,
            output_dir=args.output,
            fields=args.fields,
            dataset_root=args.dataset_root,
            config=config,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            cuda_tf32=args.cuda_tf32,
            cudnn_benchmark=args.cudnn_benchmark,
            validation_every=args.validation_every,
            train_progress_every=args.train_progress_every,
        )
        print(f"Best OCR checkpoint: {checkpoint}")
        if args.onnx_output is not None:
            output, charset, contract = export_onnx(checkpoint_path=checkpoint, output_path=args.onnx_output)
            print(f"Exported ONNX OCR model: {output}\nCharset: {charset}\nContract: {contract}")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OCR training failed:\n{error}") from None


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained receipt CTC OCR checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def export_main(argv: list[str] | None = None) -> None:
    args = build_export_parser().parse_args(argv)
    try:
        output, charset, contract = export_onnx(checkpoint_path=args.checkpoint, output_path=args.output)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OCR ONNX export failed:\n{error}") from None
    print(f"Exported ONNX OCR model: {output}\nCharset: {charset}\nContract: {contract}")


if __name__ == "__main__":  # pragma: no cover
    train_main()
