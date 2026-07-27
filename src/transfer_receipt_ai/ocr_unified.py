"""Train, export and evaluate one ONNX reader for four receipt fields.

The model intentionally has one shared visual encoder and one ONNX artifact,
while retaining specialised heads where the output spaces differ:

* amount/time: a shared, small numeric CTC head;
* payment method: a separate CTC head which preserves bank/card text; and
* transfer status: a finite three-class head.

That is materially different from putting all Chinese payment characters and
numeric characters in one CTC vocabulary: the latter makes the financial
fields compete with a much larger alphabet.  The exported wrapper consumes
four fixed-order crops in one call, so deployment needs one ORT session.
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
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from .onnx_runtime import _preload_cuda_dlls, onnx_providers
from .ocr import normalize_payment_method
from .ocr_unified_dataset import KIND as DATASET_KIND
from .ocr_unified_dataset import SLOT_ORDER, STATUS_CLASSES


SCHEMA_VERSION = 1
KIND = "receipt_unified_field_reader_v3"
NUMERIC_CHARACTERS = tuple("0123456789.:")
NUMERIC_BLANK_INDEX = 0
PAYMENT_BLANK_INDEX = 0


@dataclass(frozen=True)
class UnifiedReaderConfig:
    image_height: int = 48
    image_width: int = 384
    base_channels: int = 24
    numeric_hidden_size: int = 64
    payment_hidden_size: int = 96
    pooled_width: int = 8

    def validate(self) -> None:
        if self.image_height < 16 or self.image_width < 64 or self.image_width % 4:
            raise ValueError("image_height must be >=16 and image_width must be a multiple of 4 >=64")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if self.numeric_hidden_size < 16 or self.payment_hidden_size < 16:
            raise ValueError("numeric_hidden_size and payment_hidden_size must be at least 16")
        if not 1 <= self.pooled_width <= 32:
            raise ValueError("pooled_width must be between 1 and 32")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Unified OCR training requires a CUDA/CPU-compatible PyTorch wheel. "
            "Install it on the training server, then install requirements-train-ocr.txt."
        ) from error
    return torch, nn


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Unified OCR ONNX evaluation requires onnxruntime (or onnxruntime-gpu on the CUDA server)."
        ) from error
    return onnxruntime


def _resolve_device(torch: Any, requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for unified OCR training but PyTorch CUDA is unavailable")
        return requested
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested for unified OCR training but is unavailable")
        return "mps"
    raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError(channels)


def build_unified_reader(*, payment_vocab_size: int, config: UnifiedReaderConfig) -> Any:
    """Return the shared-trunk, four-slot reader used for training and ONNX export."""
    if payment_vocab_size < 2:
        raise ValueError("payment_vocab_size must include CTC blank plus at least one character")
    config.validate()
    torch, nn = _require_torch()

    class DepthwiseBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int]) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.GroupNorm(_group_count(in_channels), in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)

    class UnifiedFieldReader(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            first = config.base_channels
            second = first * 2
            third = first * 3
            fourth = first * 4
            self.stem = nn.Sequential(
                nn.Conv2d(1, first, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_group_count(first), first),
                nn.SiLU(inplace=True),
            )
            # Horizontal resolution is reduced exactly by 4, leaving 96 CTC
            # steps at the default 384px crop width.
            self.encoder = nn.Sequential(
                DepthwiseBlock(first, second, stride=(2, 2)),
                DepthwiseBlock(second, third, stride=(2, 1)),
                DepthwiseBlock(third, fourth, stride=(2, 1)),
            )
            self.slot_embedding = nn.Parameter(torch.empty(4, fourth, 1, 1))
            nn.init.normal_(self.slot_embedding, std=0.02)
            self.numeric_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
            self.numeric_classifier = nn.Linear(config.numeric_hidden_size * 2, len(NUMERIC_CHARACTERS) + 1)
            self.payment_sequence = nn.GRU(fourth, config.payment_hidden_size, bidirectional=True)
            self.payment_classifier = nn.Linear(config.payment_hidden_size * 2, payment_vocab_size)
            self.status_pool = nn.AdaptiveAvgPool2d((1, config.pooled_width))
            self.status_classifier = nn.Linear(fourth * config.pooled_width, len(STATUS_CLASSES))

        def forward(self, field_images: Any) -> tuple[Any, Any, Any]:
            # Training input: [batch, slot=4, channel=1, height, width].
            if field_images.ndim != 5 or field_images.shape[1] != len(SLOT_ORDER) or field_images.shape[2] != 1:
                raise ValueError("field_images must have shape [batch,4,1,height,width]")
            batch, slots, channels, height, width = field_images.shape
            encoded = self.encoder(self.stem(field_images.reshape(batch * slots, channels, height, width)))
            _, feature_channels, feature_height, feature_width = encoded.shape
            encoded = encoded.reshape(batch, slots, feature_channels, feature_height, feature_width)
            encoded = encoded + self.slot_embedding.unsqueeze(0)

            # amount/time slots share one numeric CTC projection but retain a
            # slot embedding, allowing the decoder to distinguish '.' and ':'.
            numeric_features = encoded[:, :2].mean(dim=3)  # [batch,2,C,T]
            numeric_sequence = numeric_features.permute(3, 0, 1, 2).reshape(feature_width, batch * 2, feature_channels)
            numeric_sequence, _ = self.numeric_sequence(numeric_sequence)
            numeric_logits = self.numeric_classifier(numeric_sequence).reshape(
                feature_width, batch, 2, len(NUMERIC_CHARACTERS) + 1
            )

            payment_features = encoded[:, 3].mean(dim=2)  # [batch,C,T]
            payment_sequence = payment_features.permute(2, 0, 1)
            payment_sequence, _ = self.payment_sequence(payment_sequence)
            payment_logits = self.payment_classifier(payment_sequence)  # [T,batch,class]

            status_features = self.status_pool(encoded[:, 2]).flatten(1)
            status_logits = self.status_classifier(status_features)
            return numeric_logits, payment_logits, status_logits

    return UnifiedFieldReader()


def preprocess_image(image_path: Path, *, config: UnifiedReaderConfig) -> np.ndarray:
    """Return one grayscale crop as ``[1,H,W]`` float32 with white letterbox."""
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
    return (canvas.astype(np.float32) / 255.0)[np.newaxis, :, :]


def _blank_image(config: UnifiedReaderConfig) -> np.ndarray:
    return np.ones((1, config.image_height, config.image_width), dtype=np.float32)


def _parse_slot(
    *,
    raw: object,
    field: str,
    records_path: Path,
    line_number: int,
    dataset_root: Path,
) -> dict[str, object] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{records_path}:{line_number}: slot {field} must be an object")
    image = raw.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"{records_path}:{line_number}: slot {field} has no image")
    image_path = (dataset_root / image).resolve()
    try:
        image_path.relative_to(dataset_root)
    except ValueError:
        raise ValueError(f"{records_path}:{line_number}: slot {field} image escapes dataset root") from None
    if not image_path.is_file():
        raise FileNotFoundError(f"{records_path}:{line_number}: slot {field} image not found: {image_path}")
    slot = dict(raw)
    slot["image_path"] = image_path
    if field in {"amount", "time", "payment_method_field"}:
        text = slot.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{records_path}:{line_number}: slot {field} must have a non-empty CTC target")
        if field == "amount" and (not all(character in "0123456789." for character in text) or text.count(".") > 1):
            raise ValueError(f"{records_path}:{line_number}: amount CTC target is invalid")
        if field == "time" and (not all(character in "0123456789:" for character in text) or text.count(":") not in {1, 2}):
            raise ValueError(f"{records_path}:{line_number}: time CTC target is invalid")
        if field == "payment_method_field" and any(not character.isprintable() for character in text):
            raise ValueError(f"{records_path}:{line_number}: payment CTC target contains a non-printable character")
    else:
        class_name = slot.get("class_name")
        if class_name not in STATUS_CLASSES:
            raise ValueError(f"{records_path}:{line_number}: status class must be one of {','.join(STATUS_CLASSES)}")
    return slot


def load_records(records_path: Path, *, dataset_root: Path | None = None) -> list[dict[str, object]]:
    """Load receipt-level records and protect train/val/test group isolation."""
    records_path = records_path.resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    contract_path = records_path.parent / "dataset.contract.json"
    if contract_path.is_file():
        contract = _load_json_object(contract_path)
        if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") != DATASET_KIND:
            raise ValueError(f"{contract_path}: unsupported unified dataset contract")
        if contract.get("slot_order") != list(SLOT_ORDER) or contract.get("status_classes") != list(STATUS_CLASSES):
            raise ValueError(f"{contract_path}: slot order or status classes do not match the unified reader")
    dataset_root = (dataset_root if dataset_root is not None else records_path.parent).resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    records: list[dict[str, object]] = []
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    crop_splits: dict[str, str] = {}
    with records_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{records_path}:{line_number}: invalid JSON: {error}") from None
            if not isinstance(raw, Mapping):
                raise ValueError(f"{records_path}:{line_number}: record must be an object")
            record_id = raw.get("id")
            group_id = raw.get("group_id")
            split = raw.get("split")
            slots = raw.get("slots")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{records_path}:{line_number}: id must be a non-empty string")
            if record_id in ids:
                raise ValueError(f"{records_path}:{line_number}: duplicate id {record_id!r}")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{records_path}:{line_number}: group_id must be a non-empty string")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{records_path}:{line_number}: split must be train, val, or test")
            if not isinstance(slots, Mapping):
                raise ValueError(f"{records_path}:{line_number}: slots must be an object")
            unknown_slots = sorted(set(slots) - set(SLOT_ORDER))
            if unknown_slots:
                raise ValueError(f"{records_path}:{line_number}: unknown unified slot(s): {','.join(unknown_slots)}")
            declared_order = raw.get("slot_order")
            if declared_order is not None and declared_order != list(SLOT_ORDER):
                raise ValueError(f"{records_path}:{line_number}: slot_order does not match the unified reader")
            prior_split = group_splits.setdefault(group_id, split)
            if prior_split != split:
                raise ValueError(
                    f"{records_path}:{line_number}: group_id {group_id!r} appears in both {prior_split} and {split}"
                )
            source = raw.get("source")
            if isinstance(source, str) and source:
                source_prior_split = source_splits.setdefault(source, split)
                if source_prior_split != split:
                    raise ValueError(
                        f"{records_path}:{line_number}: source {source!r} appears in both "
                        f"{source_prior_split} and {split} splits"
                    )
            parsed_slots = {
                field: _parse_slot(
                    raw=slots.get(field),
                    field=field,
                    records_path=records_path,
                    line_number=line_number,
                    dataset_root=dataset_root,
                )
                for field in SLOT_ORDER
            }
            if not any(value is not None for value in parsed_slots.values()):
                raise ValueError(f"{records_path}:{line_number}: receipt has no labelled slot")
            for slot in parsed_slots.values():
                if not isinstance(slot, Mapping):
                    continue
                crop_sha256 = slot.get("crop_sha256")
                if isinstance(crop_sha256, str) and crop_sha256:
                    crop_prior_split = crop_splits.setdefault(crop_sha256, split)
                    if crop_prior_split != split:
                        raise ValueError(
                            f"{records_path}:{line_number}: crop SHA-256 {crop_sha256!r} appears in both "
                            f"{crop_prior_split} and {split} splits"
                        )
            ids.add(record_id)
            records.append(
                {
                    "id": record_id,
                    "group_id": group_id,
                    "split": split,
                    "slots": parsed_slots,
                    "source": raw.get("source"),
                    "result_json": raw.get("result_json"),
                    "label_source": raw.get("label_source", "unspecified"),
                }
            )
    if not records:
        raise ValueError("No unified receipt records found")
    return records


def _payment_charset(records: Iterable[Mapping[str, object]]) -> list[str]:
    characters = sorted(
        {
            character
            for record in records
            for slot in [dict(record["slots"]).get("payment_method_field")]
            if isinstance(slot, Mapping)
            for character in str(slot["text"])
        }
    )
    if not characters:
        raise ValueError("No payment_method_field CTC labels remain in the training split")
    return characters


def _ctc_required_steps(text: str) -> int:
    return len(text) + sum(left == right for left, right in zip(text, text[1:]))


def _validate_ctc_capacity(records: Iterable[Mapping[str, object]], *, config: UnifiedReaderConfig) -> None:
    available = config.image_width // 4
    for record in records:
        for field in ("amount", "time", "payment_method_field"):
            slot = dict(record["slots"]).get(field)
            if not isinstance(slot, Mapping):
                continue
            text = str(slot["text"])
            required = _ctc_required_steps(text)
            if required > available:
                raise ValueError(
                    f"CTC target cannot fit the unified model time axis: id={record['id']}, "
                    f"field={field}, required={required}, available={available}, text={text!r}. "
                    "Increase --image-width or exclude this record."
                )


def _input_tensor(record: Mapping[str, object], *, config: UnifiedReaderConfig) -> np.ndarray:
    field_images = np.stack([_blank_image(config) for _ in SLOT_ORDER], axis=0)
    slots = dict(record["slots"])
    for index, field in enumerate(SLOT_ORDER):
        slot = slots.get(field)
        if isinstance(slot, Mapping):
            field_images[index] = preprocess_image(Path(slot["image_path"]), config=config)
    return field_images


class _UnifiedReceiptDataset:
    """A picklable dataset so Windows DataLoader workers remain usable."""

    def __init__(self, records: Sequence[Mapping[str, object]], *, config: UnifiedReaderConfig) -> None:
        self._records = list(records)
        self._config = config

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Any, Mapping[str, object]]:
        record = self._records[index]
        torch, _ = _require_torch()
        return torch.from_numpy(_input_tensor(record, config=self._config)), record


def _collate_receipts(samples: Sequence[tuple[Any, Mapping[str, object]]]) -> tuple[Any, list[Mapping[str, object]]]:
    torch_images, records = zip(*samples)
    torch, _ = _require_torch()
    return torch.stack(list(torch_images)), list(records)


def _make_dataset(records: Sequence[Mapping[str, object]], *, config: UnifiedReaderConfig, torch: Any) -> Any:
    del torch  # Kept in the signature so callers make the dependency explicit.
    return _UnifiedReceiptDataset(records, config=config)


def decode_ctc_logits(logits: np.ndarray, *, characters: Sequence[str]) -> list[str]:
    """Greedily decode a CTC ``[time,batch,class]`` tensor without Torch."""
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


def decode_ctc_logits_with_confidence(logits: np.ndarray, *, characters: Sequence[str]) -> list[tuple[str, float]]:
    """Greedily decode CTC while returning an auditable emitted-token score.

    It is a ranking signal, not a calibrated business probability.  Deployment
    must still set review thresholds from held-out data rather than assuming
    that a value such as ``0.99`` means 99 percent field accuracy.
    """
    values = np.asarray(logits, dtype=np.float64)
    texts = decode_ctc_logits(values, characters=characters)
    shifted = values - values.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    indices = values.argmax(axis=2)
    output: list[tuple[str, float]] = []
    for batch_index, text in enumerate(texts):
        previous = -1
        selected: list[float] = []
        for time_index, current_value in enumerate(indices[:, batch_index]):
            current = int(current_value)
            if current != 0 and current != previous:
                selected.append(float(probabilities[time_index, batch_index, current]))
            previous = current
        output.append((text, float(sum(selected) / len(selected)) if selected else 0.0))
    return output


def _slot_text(record: Mapping[str, object], field: str) -> str | None:
    slot = dict(record["slots"]).get(field)
    if not isinstance(slot, Mapping):
        return None
    text = slot.get("text")
    return text if isinstance(text, str) else None


def _status_name(record: Mapping[str, object]) -> str | None:
    slot = dict(record["slots"]).get("transfer_status")
    if not isinstance(slot, Mapping):
        return None
    class_name = slot.get("class_name")
    return class_name if class_name in STATUS_CLASSES else None


def _field_split_counts(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {field: Counter() for field in SLOT_ORDER}
    for record in records:
        split = str(record["split"])
        for field in SLOT_ORDER:
            if field == "transfer_status":
                labelled = _status_name(record) is not None
            else:
                labelled = _slot_text(record, field) is not None
            if labelled:
                counts[field][split] += 1
    return {
        field: {split: int(counts[field][split]) for split in ("train", "val", "test")}
        for field in SLOT_ORDER
    }


def _status_split_counts(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    for record in records:
        name = _status_name(record)
        if name is not None:
            counts[str(record["split"])][name] += 1
    return {
        split: {class_name: int(counts[split][class_name]) for class_name in STATUS_CLASSES}
        for split in ("train", "val", "test")
    }


def _require_train_and_validation_coverage(field_counts: Mapping[str, Mapping[str, int]]) -> None:
    missing_train = [field for field, counts in field_counts.items() if int(counts["train"]) <= 0]
    missing_validation = [field for field, counts in field_counts.items() if int(counts["val"]) <= 0]
    if missing_train or missing_validation:
        parts: list[str] = []
        if missing_train:
            parts.append("no train labels for " + ",".join(missing_train))
        if missing_validation:
            parts.append("no validation labels for " + ",".join(missing_validation))
        raise ValueError(
            "; ".join(parts)
            + ". Rebuild the teacher manifest with more labels or adjust its train/validation split."
        )


def _payment_oov_by_split(
    records: Iterable[Mapping[str, object]], *, payment_characters: set[str]
) -> dict[str, dict[str, object]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    examples: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}
    for record in records:
        split = str(record["split"])
        text = _slot_text(record, "payment_method_field")
        if text is None:
            continue
        unknown = sorted(set(text) - payment_characters)
        counts[split]["records"] += 1
        if unknown:
            counts[split]["oov_records"] += 1
            counts[split]["oov_characters"] += len(unknown)
            if len(examples[split]) < 20:
                examples[split].append({"id": record["id"], "characters": "".join(unknown), "text": text})
    return {
        split: {
            "records": int(counts[split]["records"]),
            "oov_records": int(counts[split]["oov_records"]),
            "oov_characters": int(counts[split]["oov_characters"]),
            "examples": examples[split],
        }
        for split in ("train", "val", "test")
    }


def _ctc_loss(
    logits: Any,
    *,
    labels: Sequence[str | None],
    character_to_id: Mapping[str, int],
    torch: Any,
) -> tuple[Any | None, int, int]:
    """Return CTC loss, used label count, and OOV-skipped label count."""
    selected: list[tuple[int, str]] = []
    skipped = 0
    for index, text in enumerate(labels):
        if text is None:
            continue
        if any(character not in character_to_id for character in text):
            skipped += 1
            continue
        selected.append((index, text))
    if not selected:
        return None, 0, skipped
    indices = torch.tensor([index for index, _ in selected], dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(1, indices)
    targets = torch.tensor(
        [character_to_id[character] for _, text in selected for character in text],
        dtype=torch.long,
        device=logits.device,
    )
    input_lengths = torch.full((len(selected),), selected_logits.shape[0], dtype=torch.long)
    target_lengths = torch.tensor([len(text) for _, text in selected], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(
        selected_logits.log_softmax(2),
        targets,
        input_lengths,
        target_lengths,
        blank=NUMERIC_BLANK_INDEX,
        reduction="mean",
        zero_infinity=False,
    )
    return loss, len(selected), skipped


def _status_loss(
    logits: Any,
    *,
    labels: Sequence[str | None],
    status_to_id: Mapping[str, int],
    criterion: Any,
    torch: Any,
) -> tuple[Any | None, int]:
    selected = [(index, label) for index, label in enumerate(labels) if label is not None]
    if not selected:
        return None, 0
    indices = torch.tensor([index for index, _ in selected], dtype=torch.long, device=logits.device)
    targets = torch.tensor([status_to_id[str(label)] for _, label in selected], dtype=torch.long, device=logits.device)
    return criterion(logits.index_select(0, indices), targets), len(selected)


def _batch_loss(
    numeric_logits: Any,
    payment_logits: Any,
    status_logits: Any,
    records: Sequence[Mapping[str, object]],
    *,
    numeric_to_id: Mapping[str, int],
    payment_to_id: Mapping[str, int],
    status_to_id: Mapping[str, int],
    status_criterion: Any,
    payment_loss_weight: float,
    torch: Any,
    allow_empty: bool = False,
) -> tuple[Any | None, dict[str, dict[str, float | int]]]:
    amount_loss, amount_used, amount_oov = _ctc_loss(
        numeric_logits[:, :, 0, :],
        labels=[_slot_text(record, "amount") for record in records],
        character_to_id=numeric_to_id,
        torch=torch,
    )
    time_loss, time_used, time_oov = _ctc_loss(
        numeric_logits[:, :, 1, :],
        labels=[_slot_text(record, "time") for record in records],
        character_to_id=numeric_to_id,
        torch=torch,
    )
    payment_loss, payment_used, payment_oov = _ctc_loss(
        payment_logits,
        labels=[_slot_text(record, "payment_method_field") for record in records],
        character_to_id=payment_to_id,
        torch=torch,
    )
    status_loss, status_used = _status_loss(
        status_logits,
        labels=[_status_name(record) for record in records],
        status_to_id=status_to_id,
        criterion=status_criterion,
        torch=torch,
    )
    pieces: list[Any] = []
    if amount_loss is not None:
        pieces.append(amount_loss)
    if time_loss is not None:
        pieces.append(time_loss)
    if payment_loss is not None:
        pieces.append(payment_loss * payment_loss_weight)
    if status_loss is not None:
        pieces.append(status_loss)
    if not pieces:
        if not allow_empty:
            raise ValueError("A training batch has no labelled unified-reader task")
        loss: Any | None = None
    else:
        loss = torch.stack(pieces).mean()
    return loss, {
        "amount": {"loss": float(amount_loss.detach().cpu()) if amount_loss is not None else math.nan, "used": amount_used, "oov": amount_oov},
        "time": {"loss": float(time_loss.detach().cpu()) if time_loss is not None else math.nan, "used": time_used, "oov": time_oov},
        "payment_method_field": {
            "loss": float(payment_loss.detach().cpu()) if payment_loss is not None else math.nan,
            "used": payment_used,
            "oov": payment_oov,
        },
        "transfer_status": {"loss": float(status_loss.detach().cpu()) if status_loss is not None else math.nan, "used": status_used, "oov": 0},
    }


def _evaluate_model(
    model: Any,
    loader: Any,
    *,
    device: str,
    numeric_characters: Sequence[str],
    numeric_to_id: Mapping[str, int],
    payment_characters: Sequence[str],
    payment_to_id: Mapping[str, int],
    status_to_id: Mapping[str, int],
    status_criterion: Any,
    payment_loss_weight: float,
    torch: Any,
) -> dict[str, object]:
    """Evaluate all four heads without discarding OOV held-out labels."""
    model.eval()
    total_loss = 0.0
    loss_receipts = 0
    exact_total = 0
    label_total = 0
    counters: dict[str, Counter[str]] = {field: Counter() for field in SLOT_ORDER}
    with torch.no_grad():
        for field_images, records in loader:
            field_images = field_images.to(device)
            numeric_logits, payment_logits, status_logits = model(field_images)
            loss, _ = _batch_loss(
                numeric_logits,
                payment_logits,
                status_logits,
                records,
                numeric_to_id=numeric_to_id,
                payment_to_id=payment_to_id,
                status_to_id=status_to_id,
                status_criterion=status_criterion,
                payment_loss_weight=payment_loss_weight,
                torch=torch,
                allow_empty=True,
            )
            if loss is not None:
                total_loss += float(loss.detach().cpu()) * len(records)
                loss_receipts += len(records)
            amount_predictions = decode_ctc_logits(
                numeric_logits[:, :, 0, :].detach().cpu().numpy(), characters=numeric_characters
            )
            time_predictions = decode_ctc_logits(
                numeric_logits[:, :, 1, :].detach().cpu().numpy(), characters=numeric_characters
            )
            payment_predictions = decode_ctc_logits(
                payment_logits.detach().cpu().numpy(), characters=payment_characters
            )
            status_predictions = status_logits.argmax(dim=1).detach().cpu().tolist()
            for index, record in enumerate(records):
                values = {
                    "amount": (_slot_text(record, "amount"), amount_predictions[index]),
                    "time": (_slot_text(record, "time"), time_predictions[index]),
                    "payment_method_field": (
                        _slot_text(record, "payment_method_field"),
                        payment_predictions[index],
                    ),
                    "transfer_status": (
                        _status_name(record),
                        STATUS_CLASSES[int(status_predictions[index])],
                    ),
                }
                for field, (expected, predicted) in values.items():
                    if expected is None:
                        continue
                    field_counter = counters[field]
                    field_counter["records"] += 1
                    if field == "payment_method_field" and any(
                        character not in payment_to_id for character in str(expected)
                    ):
                        field_counter["oov_reference"] += 1
                    matched = str(expected) == str(predicted)
                    field_counter["exact_matches"] += int(matched)
                    exact_total += int(matched)
                    label_total += 1
                    if field == "transfer_status" and expected != "success" and predicted == "success":
                        field_counter["non_success_to_success"] += 1
    if not loss_receipts or not label_total:
        raise ValueError("Validation set has no CTC/classification labels covered by the training charset")
    return {
        "loss": total_loss / loss_receipts,
        "exact_match": exact_total / label_total,
        "by_field": {
            field: {
                "records": int(counter["records"]),
                "exact_matches": int(counter["exact_matches"]),
                "exact_match": counter["exact_matches"] / max(1, counter["records"]),
                "oov_reference_records": int(counter["oov_reference"]),
                "non_success_to_success": int(counter["non_success_to_success"]),
            }
            for field, counter in counters.items()
        },
        "status_non_success_to_success": int(counters["transfer_status"]["non_success_to_success"]),
    }


def _write_checkpoint(path: Path, payload: Mapping[str, object], *, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def train_unified_reader(
    *,
    records_path: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
    config: UnifiedReaderConfig = UnifiedReaderConfig(),
    device: str = "auto",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    payment_loss_weight: float = 1.0,
    seed: int = 42,
    num_workers: int = 0,
) -> Path:
    """Train one shared-trunk reader and return the best validation checkpoint.

    The function intentionally accepts incomplete receipt rows: an absent slot
    gets a white input image but contributes no loss.  It refuses a manifest
    missing any *head* in train or validation, because such a pilot cannot
    prove that the exported ONNX interface is working end to end.
    """
    config.validate()
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0 or payment_loss_weight <= 0:
        raise ValueError("learning_rate and payment_loss_weight must be positive; weight_decay cannot be negative")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"training output already contains files: {output_dir}. Choose a new empty directory.")
    records = load_records(records_path, dataset_root=dataset_root)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "val"]
    if not train_records or not validation_records:
        raise ValueError("The unified manifest must contain non-empty train and val receipt splits")
    field_counts = _field_split_counts(records)
    _require_train_and_validation_coverage(field_counts)
    _validate_ctc_capacity(records, config=config)
    payment_characters = _payment_charset(train_records)
    payment_to_id = {character: index for index, character in enumerate(payment_characters, start=1)}
    numeric_characters = list(NUMERIC_CHARACTERS)
    numeric_to_id = {character: index for index, character in enumerate(numeric_characters, start=1)}
    status_to_id = {name: index for index, name in enumerate(STATUS_CLASSES)}
    status_counts = _status_split_counts(records)
    payment_oov = _payment_oov_by_split(records, payment_characters=set(payment_characters))

    torch, _ = _require_torch()
    target_device = _resolve_device(torch, device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if target_device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    train_dataset = _make_dataset(train_records, config=config, torch=torch)
    validation_dataset = _make_dataset(validation_records, config=config, torch=torch)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_receipts,
        pin_memory=target_device.startswith("cuda"),
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_receipts,
        pin_memory=target_device.startswith("cuda"),
    )
    model = build_unified_reader(payment_vocab_size=len(payment_characters) + 1, config=config).to(target_device)
    observed_status_classes = [name for name in STATUS_CLASSES if status_counts["train"][name] > 0]
    total_status = sum(status_counts["train"].values())
    status_weights = torch.tensor(
        [
            total_status / (len(observed_status_classes) * status_counts["train"][name])
            if status_counts["train"][name] > 0
            else 0.0
            for name in STATUS_CLASSES
        ],
        dtype=torch.float32,
        device=target_device,
    )
    status_train_criterion = torch.nn.CrossEntropyLoss(weight=status_weights)
    # Validation must not reuse zero weights for status classes unseen in
    # train.  Such a batch would otherwise have a zero denominator and return
    # NaN, hiding the very coverage gap that the summary reports.
    status_validation_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output_dir / "labels.json",
        {
            "schema_version": SCHEMA_VERSION,
            "numeric_blank_index": NUMERIC_BLANK_INDEX,
            "numeric_characters": numeric_characters,
            "payment_blank_index": PAYMENT_BLANK_INDEX,
            "payment_characters": payment_characters,
            "status_classes": list(STATUS_CLASSES),
            "payment_charset_sha256": hashlib.sha256("".join(payment_characters).encode("utf-8")).hexdigest(),
        },
    )

    history: list[dict[str, object]] = []
    # Prefer a checkpoint that never maps a held-out pending/failed label to
    # success.  Exact match and loss only break ties inside that safety rule.
    best_score = (float("-inf"), -1.0, float("-inf"))
    best_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_receipts = 0
        for field_images, batch_records in train_loader:
            field_images = field_images.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            numeric_logits, payment_logits, status_logits = model(field_images)
            loss, _ = _batch_loss(
                numeric_logits,
                payment_logits,
                status_logits,
                batch_records,
                numeric_to_id=numeric_to_id,
                payment_to_id=payment_to_id,
                status_to_id=status_to_id,
                status_criterion=status_train_criterion,
                payment_loss_weight=payment_loss_weight,
                torch=torch,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_records)
            total_receipts += len(batch_records)
        validation = _evaluate_model(
            model,
            validation_loader,
            device=target_device,
            numeric_characters=numeric_characters,
            numeric_to_id=numeric_to_id,
            payment_characters=payment_characters,
            payment_to_id=payment_to_id,
            status_to_id=status_to_id,
            status_criterion=status_validation_criterion,
            payment_loss_weight=payment_loss_weight,
            torch=torch,
        )
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_receipts, 1),
            "val_loss": validation["loss"],
            "val_exact_match": validation["exact_match"],
            "val_by_field": validation["by_field"],
            "val_status_non_success_to_success": validation["status_non_success_to_success"],
        }
        history.append(epoch_record)
        checkpoint_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "numeric_characters": numeric_characters,
            "payment_characters": payment_characters,
            "status_classes": list(STATUS_CLASSES),
            "field_counts": field_counts,
            "status_class_counts": status_counts,
            "payment_oov_by_split": payment_oov,
            "payment_loss_weight": payment_loss_weight,
            "epoch": epoch,
            "metrics": epoch_record,
        }
        _write_checkpoint(output_dir / "last.pt", checkpoint_payload, torch=torch)
        score = (
            -float(validation["status_non_success_to_success"]),
            float(validation["exact_match"]),
            -float(validation["loss"]),
        )
        if score > best_score:
            best_score = score
            _write_checkpoint(best_path, checkpoint_payload, torch=torch)
        _atomic_write_json(
            output_dir / "training_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND,
                "field_counts": field_counts,
                "status_class_counts": status_counts,
                "payment_oov_by_split": payment_oov,
                "status_classes_missing_from_train": [
                    name for name in STATUS_CLASSES if status_counts["train"][name] == 0
                ],
                "records": history,
                "warning": (
                    "Paddle teacher labels are not independent truth. A model with any status class missing from "
                    "the train split must not be used to accept that unseen class in production."
                ),
            },
        )
        print(
            f"epoch {epoch}/{epochs}: train_loss={float(epoch_record['train_loss']):.4f} "
            f"val_loss={float(validation['loss']):.4f} val_exact_match={float(validation['exact_match']):.2%}"
        )
    return best_path


def _load_checkpoint(path: Path, *, torch: Any) -> Mapping[str, object]:
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Unified OCR checkpoint must be a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != KIND:
        raise ValueError("Unsupported unified OCR checkpoint schema")
    return payload


def _checkpoint_config(payload: Mapping[str, object]) -> UnifiedReaderConfig:
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("Unified OCR checkpoint has no model config")
    try:
        config = UnifiedReaderConfig(
            image_height=int(raw["image_height"]),
            image_width=int(raw["image_width"]),
            base_channels=int(raw["base_channels"]),
            numeric_hidden_size=int(raw["numeric_hidden_size"]),
            payment_hidden_size=int(raw["payment_hidden_size"]),
            pooled_width=int(raw["pooled_width"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Unified OCR checkpoint has an invalid model config") from error
    config.validate()
    return config


def _checkpoint_labels(payload: Mapping[str, object]) -> tuple[list[str], list[str], list[str]]:
    numeric = payload.get("numeric_characters")
    payment = payload.get("payment_characters")
    status = payload.get("status_classes")
    if not isinstance(numeric, list) or not isinstance(payment, list) or not isinstance(status, list):
        raise ValueError("Unified OCR checkpoint has no label maps")
    if numeric != list(NUMERIC_CHARACTERS):
        raise ValueError("Unified OCR checkpoint numeric label map is not the supported fixed numeric charset")
    if not all(isinstance(character, str) and len(character) == 1 for character in payment):
        raise ValueError("Unified OCR checkpoint payment charset must contain single Unicode code points")
    if not payment or len(set(payment)) != len(payment):
        raise ValueError("Unified OCR checkpoint payment charset is empty or has duplicates")
    if status != list(STATUS_CLASSES):
        raise ValueError("Unified OCR checkpoint status class order is unsupported")
    return list(numeric), list(payment), list(status)


def _validate_exported_onnx(
    onnx_path: Path,
    *,
    dummy: Any,
    expected_outputs: Sequence[Any],
) -> None:
    """Require the exported graph to load and match Torch on a fixed input."""
    onnxruntime = _require_onnxruntime()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["field_images"]:
        raise ValueError("Exported unified OCR ONNX has an unexpected input name")
    output_names = ["amount_logits", "time_logits", "payment_logits", "status_logits"]
    if [item.name for item in session.get_outputs()] != output_names:
        raise ValueError("Exported unified OCR ONNX has unexpected output names")
    actual_outputs = session.run(output_names, {"field_images": dummy.detach().cpu().numpy()})
    for name, actual, expected in zip(output_names, actual_outputs, expected_outputs):
        expected_array = expected.detach().cpu().numpy()
        actual_array = np.asarray(actual)
        if list(actual_array.shape) != list(expected_array.shape):
            raise ValueError(
                f"Exported unified OCR ONNX output {name!r} has shape {list(actual_array.shape)}, "
                f"expected {list(expected_array.shape)}"
            )
        if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
            raise ValueError(f"Exported unified OCR ONNX output {name!r} contains a non-finite value")
        # CPU ORT and CPU Torch can accumulate small FP32 drift through the
        # exported GRU/normalisation sequence.  The tolerance below is still
        # far below one CTC/logit unit, and is paired with an exact argmax
        # check so a changed decoded character/status is never accepted.
        expected64 = expected_array.astype(np.float64, copy=False)
        actual64 = actual_array.astype(np.float64, copy=False)
        absolute_error = np.abs(actual64 - expected64)
        relative_error = absolute_error / np.maximum(np.abs(expected64), 1e-6)
        decision_positions = int(np.prod(actual_array.shape[:-1])) if actual_array.ndim > 1 else 1
        argmax_mismatches = int(
            np.count_nonzero(np.argmax(actual_array, axis=-1) != np.argmax(expected_array, axis=-1))
        )
        if not np.allclose(actual_array, expected_array, rtol=1e-3, atol=1e-4) or argmax_mismatches:
            raise ValueError(
                f"Exported unified OCR ONNX output {name!r} differs from Torch beyond "
                "rtol=1e-3, atol=1e-4 or changes its argmax: "
                f"max_abs={float(absolute_error.max()):.8g}, "
                f"mean_abs={float(absolute_error.mean()):.8g}, "
                f"max_rel={float(relative_error.max()):.8g}, "
                f"argmax_mismatches={argmax_mismatches}/{decision_positions}. "
                "Keep the checkpoint and report these values; do not retrain before resolving export parity."
            )


def export_unified_onnx(*, checkpoint_path: Path, output_path: Path) -> tuple[Path, Path, Path]:
    """Export a static one-receipt ONNX graph plus labels and a delivery contract."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("Unified OCR export output must end in .onnx")
    labels_path = output_path.with_suffix(".labels.json")
    contract_path = output_path.with_suffix(".contract.json")
    temporary_output = output_path.with_name(f".{output_path.stem}.exporting{output_path.suffix}")
    existing = next((path for path in (output_path, labels_path, contract_path, temporary_output) if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"Refusing to overwrite unified ONNX artifact: {existing}")
    torch, nn = _require_torch()
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    config = _checkpoint_config(payload)
    numeric_characters, payment_characters, status_classes = _checkpoint_labels(payload)
    state_dict = payload.get("state_dict")
    field_counts = payload.get("field_counts")
    status_counts = payload.get("status_class_counts")
    if not isinstance(state_dict, Mapping) or not isinstance(field_counts, Mapping) or not isinstance(status_counts, Mapping):
        raise ValueError("Unified OCR checkpoint is missing state_dict or audit counts")
    model = build_unified_reader(payment_vocab_size=len(payment_characters) + 1, config=config)
    model.load_state_dict(state_dict)
    model.eval()

    class OneReceiptExport(nn.Module):
        def __init__(self, reader: Any) -> None:
            super().__init__()
            self.reader = reader

        def forward(self, field_images: Any) -> tuple[Any, Any, Any, Any]:
            # ONNX input is one receipt in fixed field order: [4,1,H,W].
            numeric, payment, status = self.reader(field_images.unsqueeze(0))
            # Separate amount/time outputs keep the .NET decoder simple while
            # still invoking only one model/session/run.
            return numeric[:, 0, 0, :], numeric[:, 0, 1, :], payment[:, 0, :], status[0, :]

    wrapper = OneReceiptExport(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((len(SLOT_ORDER), 1, config.image_height, config.image_width), dtype=torch.float32)
    try:
        try:
            torch.onnx.export(
                wrapper,
                dummy,
                temporary_output,
                input_names=["field_images"],
                output_names=["amount_logits", "time_logits", "payment_logits", "status_logits"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:  # Older PyTorch has no dynamo argument.
            torch.onnx.export(
                wrapper,
                dummy,
                temporary_output,
                input_names=["field_images"],
                output_names=["amount_logits", "time_logits", "payment_logits", "status_logits"],
                opset_version=17,
                do_constant_folding=True,
            )
        with torch.no_grad():
            amount_logits, time_logits, payment_logits, status_logits = wrapper(dummy)
        _validate_exported_onnx(
            temporary_output,
            dummy=dummy,
            expected_outputs=(amount_logits, time_logits, payment_logits, status_logits),
        )
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    labels_payload = {
        "schema_version": SCHEMA_VERSION,
        "numeric_blank_index": NUMERIC_BLANK_INDEX,
        "numeric_characters": numeric_characters,
        "payment_blank_index": PAYMENT_BLANK_INDEX,
        "payment_characters": payment_characters,
        "status_classes": status_classes,
        "payment_charset_sha256": hashlib.sha256("".join(payment_characters).encode("utf-8")).hexdigest(),
    }
    _atomic_write_json(labels_path, labels_payload)
    _atomic_write_json(
        contract_path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "onnx_file": output_path.name,
            "onnx_sha256": _sha256(output_path),
            "labels_file": labels_path.name,
            "labels_sha256": _sha256(labels_path),
            "slot_order": list(SLOT_ORDER),
            "status_classes": status_classes,
            "training_field_counts": field_counts,
            "training_status_class_counts": status_counts,
            "input": {
                "name": "field_images",
                "dtype": "float32",
                "shape": [len(SLOT_ORDER), 1, config.image_height, config.image_width],
                "preprocess": "RGB crop -> grayscale -> aspect-preserving resize -> white letterbox -> divide by 255.0",
                "absent_slot_policy": "white_placeholder_not_decoded; emit review instead",
            },
            "outputs": {
                "amount_logits": {
                    "shape": list(amount_logits.shape),
                    "layout": "[time,class]",
                    "decoder": "ctc_greedy",
                    "blank_index": NUMERIC_BLANK_INDEX,
                    "characters": "numeric_characters",
                },
                "time_logits": {
                    "shape": list(time_logits.shape),
                    "layout": "[time,class]",
                    "decoder": "ctc_greedy",
                    "blank_index": NUMERIC_BLANK_INDEX,
                    "characters": "numeric_characters",
                },
                "payment_logits": {
                    "shape": list(payment_logits.shape),
                    "layout": "[time,class]",
                    "decoder": "ctc_greedy",
                    "blank_index": PAYMENT_BLANK_INDEX,
                    "characters": "payment_characters",
                    "target": "visible_payment_method_value",
                },
                "status_logits": {
                    "shape": list(status_logits.shape),
                    "layout": "[class]",
                    "classes": "status_classes",
                },
            },
            "model": asdict(config),
            "warning": (
                "The reader is not a detector or perspective rectifier. Delivery must use the same field crop geometry "
                "and preprocessing as the training/evaluation pipeline."
            ),
        },
    )
    return output_path, labels_path, contract_path


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_onnx_artifacts(model_path: Path) -> tuple[UnifiedReaderConfig, list[str], Mapping[str, Any]]:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    labels_path = model_path.with_suffix(".labels.json")
    contract_path = model_path.with_suffix(".contract.json")
    labels = _load_json_object(labels_path)
    contract = _load_json_object(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") != KIND:
        raise ValueError("Unified OCR ONNX contract kind/schema is unsupported")
    if contract.get("onnx_sha256") != _sha256(model_path):
        raise ValueError("Unified OCR ONNX SHA-256 does not match its contract")
    if contract.get("labels_file") != labels_path.name or contract.get("labels_sha256") != _sha256(labels_path):
        raise ValueError("Unified OCR label sidecar does not match its contract")
    if contract.get("slot_order") != list(SLOT_ORDER):
        raise ValueError("Unified OCR ONNX contract slot order is unsupported")
    raw_config = contract.get("model")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Unified OCR ONNX contract has no model config")
    try:
        config = UnifiedReaderConfig(
            image_height=int(raw_config["image_height"]),
            image_width=int(raw_config["image_width"]),
            base_channels=int(raw_config["base_channels"]),
            numeric_hidden_size=int(raw_config["numeric_hidden_size"]),
            payment_hidden_size=int(raw_config["payment_hidden_size"]),
            pooled_width=int(raw_config["pooled_width"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Unified OCR ONNX contract model config is invalid") from error
    config.validate()
    numeric = labels.get("numeric_characters")
    payment = labels.get("payment_characters")
    status = labels.get("status_classes")
    if (
        labels.get("schema_version") != SCHEMA_VERSION
        or labels.get("numeric_blank_index") != NUMERIC_BLANK_INDEX
        or labels.get("payment_blank_index") != PAYMENT_BLANK_INDEX
    ):
        raise ValueError("Unified OCR ONNX label sidecar schema or blank index is invalid")
    if numeric != list(NUMERIC_CHARACTERS):
        raise ValueError("Unified OCR ONNX numeric charset is unsupported")
    if not isinstance(payment, list) or not payment or not all(isinstance(item, str) and len(item) == 1 for item in payment):
        raise ValueError("Unified OCR ONNX payment charset is invalid")
    if len(set(payment)) != len(payment):
        raise ValueError("Unified OCR ONNX payment charset has duplicates")
    if labels.get("payment_charset_sha256") != hashlib.sha256("".join(payment).encode("utf-8")).hexdigest():
        raise ValueError("Unified OCR ONNX payment charset SHA-256 is invalid")
    if status != list(STATUS_CLASSES):
        raise ValueError("Unified OCR ONNX status class order is unsupported")
    raw_input = contract.get("input")
    outputs = contract.get("outputs")
    if not isinstance(raw_input, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("Unified OCR ONNX contract input/output schema is missing")
    expected_input = [len(SLOT_ORDER), 1, config.image_height, config.image_width]
    if raw_input.get("name") != "field_images" or raw_input.get("shape") != expected_input:
        raise ValueError("Unified OCR ONNX input must be static [4,1,H,W]")
    expected_outputs = {"amount_logits", "time_logits", "payment_logits", "status_logits"}
    if set(outputs) != expected_outputs:
        raise ValueError("Unified OCR ONNX output names are unsupported")
    time_steps = config.image_width // 4
    expected_shapes = {
        "amount_logits": [time_steps, len(NUMERIC_CHARACTERS) + 1],
        "time_logits": [time_steps, len(NUMERIC_CHARACTERS) + 1],
        "payment_logits": [time_steps, len(payment) + 1],
        "status_logits": [len(STATUS_CLASSES)],
    }
    for name, expected_shape in expected_shapes.items():
        output = outputs[name]
        if not isinstance(output, Mapping) or output.get("shape") != expected_shape:
            raise ValueError(f"Unified OCR ONNX output {name!r} has an invalid static shape")
        if name != "status_logits" and output.get("blank_index") != NUMERIC_BLANK_INDEX:
            raise ValueError(f"Unified OCR ONNX output {name!r} has an invalid CTC blank index")
    return config, list(payment), contract


def _create_onnx_session(onnxruntime: Any, model_path: Path, *, device: str) -> tuple[Any, list[str]]:
    providers = onnx_providers(device, onnxruntime)
    _preload_cuda_dlls(onnxruntime, providers)
    session = onnxruntime.InferenceSession(str(model_path), providers=providers)
    active = list(session.get_providers())
    requested_cuda = device.lower() == "cuda" or device.lower().startswith("cuda:")
    if requested_cuda and "CUDAExecutionProvider" not in active:
        raise RuntimeError("Unified OCR ONNX session did not activate CUDAExecutionProvider")
    return session, active


def levenshtein_distance(reference: str, candidate: str) -> int:
    """Unicode-character edit distance used in the payment CER report."""
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


def _semantic_value(field: str, text: str) -> str | None:
    if field == "amount":
        try:
            from .ocr import normalize_amount

            value = normalize_amount(text)
        except ValueError:
            value = None
        return str(value["normalized"]) if value is not None else None
    if field == "time":
        try:
            from .ocr import normalize_time

            return normalize_time(text)
        except ValueError:
            return None
    if field == "payment_method_field":
        return normalize_payment_method(text)["normalized"]
    if field == "transfer_status":
        return text if text in STATUS_CLASSES else None
    raise AssertionError(field)


def _ctc_single_output(logits: np.ndarray, *, characters: Sequence[str]) -> tuple[str, float]:
    values = np.asarray(logits)
    if values.ndim != 2:
        raise ValueError(f"Expected a CTC ONNX output shaped [time,class], got {list(values.shape)}")
    return decode_ctc_logits_with_confidence(values[:, np.newaxis, :], characters=characters)[0]


def _softmax_confidence(logits: np.ndarray) -> tuple[int, float]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(STATUS_CLASSES):
        raise ValueError(f"Expected status ONNX output shaped [{len(STATUS_CLASSES)}]")
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return index, float(probabilities[index])


def _comparison_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "records": 0,
            "raw_exact_match": None,
            "semantic_exact_match": None,
            "micro_cer": None,
            "oov_reference_rate": None,
            "non_success_to_success": 0,
        }
    records = len(rows)
    raw_exact = sum(bool(row["raw_exact"]) for row in rows)
    semantic_rows = [row for row in rows if row["reference_semantic"] is not None]
    semantic_exact = sum(bool(row["semantic_exact"]) for row in semantic_rows)
    edits = sum(int(row["cer_edits"]) for row in rows)
    reference_characters = sum(int(row["reference_characters"]) for row in rows)
    oov = sum(bool(row["reference_has_oov_character"]) for row in rows)
    non_success_to_success = sum(bool(row.get("non_success_to_success", False)) for row in rows)
    return {
        "records": records,
        "raw_exact_matches": raw_exact,
        "raw_exact_match": raw_exact / records,
        "semantic_exact_matches": semantic_exact,
        "semantic_exact_match": semantic_exact / max(1, len(semantic_rows)),
        "cer_edits": edits,
        "reference_characters": reference_characters,
        "micro_cer": edits / max(1, reference_characters),
        "oov_reference_records": oov,
        "oov_reference_rate": oov / records,
        "non_success_to_success": non_success_to_success,
    }


def _latency_metrics(latencies: Sequence[float]) -> dict[str, float | int]:
    if not latencies:
        return {"records": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    values = sorted(latencies)
    percentile = lambda fraction: values[min(len(values) - 1, int(math.ceil(fraction * len(values))) - 1)]
    return {
        "records": len(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "mean": sum(values) / len(values),
    }


def _finite_probability(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _unified_acceptance_failures(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    min_amount_exact_match: float | None,
    min_time_exact_match: float | None,
    min_payment_exact_match: float | None,
    min_status_exact_match: float | None,
    max_payment_oov_rate: float | None,
    max_non_success_to_success: int | None,
) -> list[str]:
    failures: list[str] = []
    desired = {
        "amount": min_amount_exact_match,
        "time": min_time_exact_match,
        "payment_method_field": min_payment_exact_match,
        "transfer_status": min_status_exact_match,
    }
    for field, threshold in desired.items():
        if threshold is not None and float(metrics[field]["raw_exact_match"]) < threshold:
            failures.append(f"{field}: raw_exact_match={float(metrics[field]['raw_exact_match']):.4f} < {threshold:.4f}")
    if max_payment_oov_rate is not None and float(metrics["payment_method_field"]["oov_reference_rate"]) > max_payment_oov_rate:
        failures.append(
            "payment_method_field: "
            f"oov_reference_rate={float(metrics['payment_method_field']['oov_reference_rate']):.4f} "
            f"> {max_payment_oov_rate:.4f}"
        )
    if max_non_success_to_success is not None:
        observed = int(metrics["transfer_status"]["non_success_to_success"])
        if observed > max_non_success_to_success:
            failures.append(
                f"transfer_status: non_success_to_success={observed} > {max_non_success_to_success}"
            )
    return failures


def evaluate_unified_onnx(
    *,
    model_path: Path,
    records_path: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
    split: str = "test",
    device: str = "auto",
    min_amount_exact_match: float | None = None,
    min_time_exact_match: float | None = None,
    min_payment_exact_match: float | None = None,
    min_status_exact_match: float | None = None,
    max_payment_oov_rate: float | None = None,
    max_non_success_to_success: int | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Compare one ONNX session run per held-out receipt with teacher labels."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test; train is not an independent teacher-parity evaluation")
    for name, value in (
        ("min_amount_exact_match", min_amount_exact_match),
        ("min_time_exact_match", min_time_exact_match),
        ("min_payment_exact_match", min_payment_exact_match),
        ("min_status_exact_match", min_status_exact_match),
        ("max_payment_oov_rate", max_payment_oov_rate),
    ):
        _finite_probability(value, name=name)
    if max_non_success_to_success is not None and max_non_success_to_success < 0:
        raise ValueError("max_non_success_to_success cannot be negative")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"evaluation output already contains files: {output_dir}. Choose a new empty directory.")
    config, payment_characters, _contract = _load_onnx_artifacts(model_path)
    records = load_records(records_path, dataset_root=dataset_root)
    evaluation_records = [record for record in records if record["split"] == split]
    if not evaluation_records:
        raise ValueError(f"No {split} receipt records found")
    for field in SLOT_ORDER:
        if not any(
            (_status_name(record) if field == "transfer_status" else _slot_text(record, field)) is not None
            for record in evaluation_records
        ):
            raise ValueError(f"No {split} labels remain for unified field {field!r}")

    onnxruntime = _require_onnxruntime()
    model_path = model_path.resolve()
    session, active_providers = _create_onnx_session(onnxruntime, model_path, device=device)
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    expected_outputs = ["amount_logits", "time_logits", "payment_logits", "status_logits"]
    if input_names != ["field_images"] or output_names != expected_outputs:
        raise ValueError(
            "Unified OCR ONNX input/output names differ from its delivery contract: "
            f"inputs={input_names}, outputs={output_names}"
        )
    expected_input_shape = [len(SLOT_ORDER), 1, config.image_height, config.image_width]
    actual_input_shape = list(session.get_inputs()[0].shape)
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"Unified OCR ONNX input shape {actual_input_shape} differs from contract {expected_input_shape}"
        )
    expected_output_shapes = {
        "amount_logits": [config.image_width // 4, len(NUMERIC_CHARACTERS) + 1],
        "time_logits": [config.image_width // 4, len(NUMERIC_CHARACTERS) + 1],
        "payment_logits": [config.image_width // 4, len(payment_characters) + 1],
        "status_logits": [len(STATUS_CLASSES)],
    }
    for output in session.get_outputs():
        actual_shape = list(output.shape)
        expected_shape = expected_output_shapes[output.name]
        if actual_shape != expected_shape:
            raise ValueError(
                f"Unified OCR ONNX output {output.name!r} shape {actual_shape} differs from contract {expected_shape}"
            )

    comparisons: list[dict[str, object]] = []
    receipt_latencies: list[float] = []
    payment_character_set = set(payment_characters)
    status_confusion: Counter[str] = Counter()
    status_reference_counts: Counter[str] = Counter()
    for record in evaluation_records:
        field_images = np.ascontiguousarray(_input_tensor(record, config=config), dtype=np.float32)
        started = perf_counter()
        amount_logits, time_logits, payment_logits, status_logits = session.run(
            expected_outputs,
            {"field_images": field_images},
        )
        latency_ms = (perf_counter() - started) * 1000.0
        receipt_latencies.append(latency_ms)
        amount_text, amount_confidence = _ctc_single_output(amount_logits, characters=NUMERIC_CHARACTERS)
        time_text, time_confidence = _ctc_single_output(time_logits, characters=NUMERIC_CHARACTERS)
        payment_text, payment_confidence = _ctc_single_output(payment_logits, characters=payment_characters)
        status_index, status_confidence = _softmax_confidence(status_logits)
        predictions: dict[str, tuple[str, float]] = {
            "amount": (amount_text, amount_confidence),
            "time": (time_text, time_confidence),
            "payment_method_field": (payment_text, payment_confidence),
            "transfer_status": (STATUS_CLASSES[status_index], status_confidence),
        }
        for field in SLOT_ORDER:
            slot = dict(record["slots"]).get(field)
            if not isinstance(slot, Mapping):
                continue
            if field == "transfer_status":
                reference_text = str(slot["class_name"])
                reference_semantic = reference_text
            else:
                reference_text = str(slot["text"])
                semantic_value = slot.get("semantic_value")
                reference_semantic = str(semantic_value) if isinstance(semantic_value, str) else _semantic_value(field, reference_text)
            candidate_text, confidence = predictions[field]
            candidate_semantic = _semantic_value(field, candidate_text)
            raw_exact = candidate_text == reference_text
            semantic_exact = reference_semantic is not None and candidate_semantic == reference_semantic
            non_success_to_success = (
                field == "transfer_status" and reference_text in {"pending", "failed"} and candidate_text == "success"
            )
            if field == "transfer_status":
                status_confusion[f"{reference_text}->{candidate_text}"] += 1
                status_reference_counts[reference_text] += 1
            comparisons.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": record["id"],
                    "field": field,
                    "split": split,
                    "group_id": record["group_id"],
                    "source": record.get("source"),
                    "result_json": record.get("result_json"),
                    "label_source": record.get("label_source"),
                    "image": Path(slot["image_path"]).as_posix(),
                    "paddle_text": slot.get("paddle_text"),
                    "reference_text": reference_text,
                    "candidate_text": candidate_text,
                    "confidence": round(confidence, 6),
                    "raw_exact": raw_exact,
                    "reference_semantic": reference_semantic,
                    "candidate_semantic": candidate_semantic,
                    "semantic_exact": semantic_exact,
                    "candidate_semantic_valid": candidate_semantic is not None,
                    "cer_edits": levenshtein_distance(reference_text, candidate_text),
                    "reference_characters": len(reference_text),
                    "reference_has_oov_character": field == "payment_method_field"
                    and bool(set(reference_text) - payment_character_set),
                    "non_success_to_success": non_success_to_success,
                    "receipt_latency_ms": round(latency_ms, 4),
                }
            )
    comparisons.sort(key=lambda row: (str(row["field"]), str(row["id"])))
    by_field = {
        field: _comparison_metrics([row for row in comparisons if row["field"] == field]) for field in SLOT_ORDER
    }
    failures = _unified_acceptance_failures(
        by_field,
        min_amount_exact_match=min_amount_exact_match,
        min_time_exact_match=min_time_exact_match,
        min_payment_exact_match=min_payment_exact_match,
        min_status_exact_match=min_status_exact_match,
        max_payment_oov_rate=max_payment_oov_rate,
        max_non_success_to_success=max_non_success_to_success,
    )
    acceptance_requested = any(
        value is not None
        for value in (
            min_amount_exact_match,
            min_time_exact_match,
            min_payment_exact_match,
            min_status_exact_match,
            max_payment_oov_rate,
            max_non_success_to_success,
        )
    )
    label_sources = sorted({str(record.get("label_source", "unspecified")) for record in evaluation_records})
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_unified_field_reader_truth_evaluation_v1"
        if label_sources == ["transaction_truth"]
        else "receipt_unified_field_reader_teacher_parity_v1",
        "model": model_path.as_posix(),
        "model_sha256": _sha256(model_path),
        "records": records_path.resolve().as_posix(),
        "evaluation_split": split,
        "label_sources": label_sources,
        "providers": active_providers,
        "slot_order": list(SLOT_ORDER),
        "by_field": by_field,
        "status_confusion": dict(sorted(status_confusion.items())),
        "status_reference_class_counts": {
            class_name: int(status_reference_counts[class_name]) for class_name in STATUS_CLASSES
        },
        "receipt_latency_ms": _latency_metrics(receipt_latencies),
        "acceptance": {
            "min_amount_exact_match": min_amount_exact_match,
            "min_time_exact_match": min_time_exact_match,
            "min_payment_exact_match": min_payment_exact_match,
            "min_status_exact_match": min_status_exact_match,
            "max_payment_oov_rate": max_payment_oov_rate,
            "max_non_success_to_success": max_non_success_to_success,
            # A report with no requested gate is informative, but it must not
            # be rendered as an accepted delivery candidate simply because no
            # threshold was supplied.
            "requested": acceptance_requested,
            "passed": (not failures) if acceptance_requested else None,
            "failures": failures,
        },
        "warning": (
            "This compares ONNX with held-out Paddle-derived teacher labels, not independently verified business truth. "
            "Do not claim production accuracy until a group-isolated human-truth holdout also passes."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    _atomic_write_jsonl(
        output_dir / "disagreements.jsonl",
        [row for row in comparisons if not bool(row["raw_exact"]) or not bool(row["semantic_exact"])],
    )
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, export, and evaluate one offline ONNX reader for amount/time/status/payment fields"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train the shared-trunk unified reader")
    train.add_argument("--records", type=Path, required=True, help="unified_fields.jsonl")
    train.add_argument(
        "--dataset-root",
        type=Path,
        help="Root that owns crop paths in the original pseudo-label manifest; defaults to --records directory",
    )
    train.add_argument("--output", type=Path, required=True, help="New empty checkpoint output directory")
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--payment-loss-weight", type=float, default=1.0)
    train.add_argument("--image-height", type=int, default=48)
    train.add_argument("--image-width", type=int, default=384)
    train.add_argument("--base-channels", type=int, default=24)
    train.add_argument("--numeric-hidden-size", type=int, default=64)
    train.add_argument("--payment-hidden-size", type=int, default=96)
    train.add_argument("--pooled-width", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 on Windows until the training environment is verified",
    )
    train.add_argument("--onnx-output", type=Path, help="Optionally export best.pt to this new ONNX path")

    export = commands.add_parser("export", help="export a trained unified reader checkpoint")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="compare an ONNX reader with held-out teacher/truth labels")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--records", type=Path, required=True, help="unified_fields.jsonl")
    evaluate.add_argument(
        "--dataset-root",
        type=Path,
        help="Root that owns crop paths in the original pseudo-label manifest; defaults to --records directory",
    )
    evaluate.add_argument("--output", type=Path, required=True, help="New empty evaluation output directory")
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--min-amount-exact-match", type=float)
    evaluate.add_argument("--min-time-exact-match", type=float)
    evaluate.add_argument("--min-payment-exact-match", type=float)
    evaluate.add_argument("--min-status-exact-match", type=float)
    evaluate.add_argument("--max-payment-oov-rate", type=float)
    evaluate.add_argument("--max-non-success-to-success", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            config = UnifiedReaderConfig(
                image_height=args.image_height,
                image_width=args.image_width,
                base_channels=args.base_channels,
                numeric_hidden_size=args.numeric_hidden_size,
                payment_hidden_size=args.payment_hidden_size,
                pooled_width=args.pooled_width,
            )
            checkpoint = train_unified_reader(
                records_path=args.records,
                output_dir=args.output,
                dataset_root=args.dataset_root,
                config=config,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                payment_loss_weight=args.payment_loss_weight,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            print(f"Best unified OCR checkpoint: {checkpoint}")
            if args.onnx_output is not None:
                output, labels, contract = export_unified_onnx(
                    checkpoint_path=checkpoint,
                    output_path=args.onnx_output,
                )
                print(f"Exported unified ONNX reader: {output}\nLabels: {labels}\nContract: {contract}")
            return
        if args.command == "export":
            output, labels, contract = export_unified_onnx(
                checkpoint_path=args.checkpoint,
                output_path=args.output,
            )
            print(f"Exported unified ONNX reader: {output}\nLabels: {labels}\nContract: {contract}")
            return
        if args.command == "evaluate":
            summary, failures = evaluate_unified_onnx(
                model_path=args.model,
                records_path=args.records,
                output_dir=args.output,
                dataset_root=args.dataset_root,
                split=args.split,
                device=args.device,
                min_amount_exact_match=args.min_amount_exact_match,
                min_time_exact_match=args.min_time_exact_match,
                min_payment_exact_match=args.min_payment_exact_match,
                min_status_exact_match=args.min_status_exact_match,
                max_payment_oov_rate=args.max_payment_oov_rate,
                max_non_success_to_success=args.max_non_success_to_success,
            )
            metrics = summary["by_field"]
            print(
                f"Wrote unified ONNX evaluation to {args.output} "
                f"(amount={float(metrics['amount']['raw_exact_match']):.2%}, "
                f"time={float(metrics['time']['raw_exact_match']):.2%}, "
                f"payment={float(metrics['payment_method_field']['raw_exact_match']):.2%}, "
                f"status={float(metrics['transfer_status']['raw_exact_match']):.2%})"
            )
            if failures:
                raise SystemExit("Unified OCR candidate did not meet the requested acceptance gate:\n- " + "\n- ".join(failures))
            return
        raise AssertionError(f"Unhandled command {args.command!r}")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Unified OCR command failed:\n{error}") from None


if __name__ == "__main__":  # pragma: no cover
    main()
