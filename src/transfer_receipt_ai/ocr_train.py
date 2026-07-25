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
from PIL import Image

from .labels import DETECTION_CLASSES


RECOGNIZER_SCHEMA_VERSION = 1
DEFAULT_IMAGE_HEIGHT = 48
DEFAULT_IMAGE_WIDTH = 768


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
    try:
        value: Any = json.loads(line)
    except json.JSONDecodeError as error:
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
    if not isinstance(field, str) or field not in DETECTION_CLASSES:
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
    }


def load_records(
    records_path: Path,
    *,
    fields: Sequence[str] = DETECTION_CLASSES,
    dataset_root: Path | None = None,
) -> list[dict[str, object]]:
    """Read pseudo-label records without importing PyTorch."""
    fields = tuple(fields)
    invalid = sorted(set(fields) - set(DETECTION_CLASSES))
    if not fields or invalid:
        raise ValueError(f"fields must be a non-empty subset of: {','.join(DETECTION_CLASSES)}")
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


def preprocess_image(image_path: Path, *, config: RecognizerConfig) -> np.ndarray:
    """Return the fixed NCHW float32 preprocessing declared in the ONNX contract."""
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


def _resize_to_tensor(image_path: Path, *, config: RecognizerConfig, torch: Any) -> Any:
    """Torch wrapper for the shared train/ONNX image preprocessing."""
    return torch.from_numpy(preprocess_image(image_path, config=config)[0])


def _make_dataset(records: Sequence[Mapping[str, object]], *, character_to_id: Mapping[str, int], config: RecognizerConfig, torch: Any) -> Any:
    class ReceiptOcrDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> tuple[Any, Any, str, str]:
            record = records[index]
            text = str(record["text"])
            targets = torch.tensor([character_to_id[character] for character in text], dtype=torch.long)
            image = _resize_to_tensor(Path(record["image_path"]), config=config, torch=torch)
            return image, targets, text, str(record["field"])

    return ReceiptOcrDataset()


def _collate_batch(
    batch: Sequence[tuple[Any, Any, str, str]], *, torch: Any
) -> tuple[Any, Any, Any, list[str], list[str]]:
    images = torch.stack([item[0] for item in batch])
    target_lengths = torch.tensor([item[1].numel() for item in batch], dtype=torch.long)
    targets = torch.cat([item[1] for item in batch])
    return images, targets, target_lengths, [item[2] for item in batch], [item[3] for item in batch]


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
) -> Path:
    """Train a field-crop recognizer and return the best checkpoint path."""
    config.validate()
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    selected_fields = tuple(dict.fromkeys(fields))
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
    _validate_ctc_capacity((*train_records, *validation_records), config=config)
    character_to_id = {character: index for index, character in enumerate(characters, start=1)}
    torch, _ = _require_torch()
    target_device = _resolve_device(torch, device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if target_device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    train_dataset = _make_dataset(train_records, character_to_id=character_to_id, config=config, torch=torch)
    validation_dataset = _make_dataset(validation_records, character_to_id=character_to_id, config=config, torch=torch)
    collate = lambda batch: _collate_batch(batch, torch=torch)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=target_device.startswith("cuda"),
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=target_device.startswith("cuda"),
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
        for images, targets, target_lengths, texts, _fields in train_loader:
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
        train_loss = total_loss / max(total_items, 1)
        validation = _evaluate(
            model,
            validation_loader,
            criterion=criterion,
            device=target_device,
            characters=characters,
            torch=torch,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_exact_match": validation["exact_match"],
            "val_by_field": validation["by_field"],
        }
        history.append(epoch_record)
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
        }
        _write_checkpoint(output_dir / "last.pt", checkpoint_payload, torch=torch)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            _write_checkpoint(best_path, checkpoint_payload, torch=torch)
        _atomic_write_json(
            output_dir / "training_history.json",
            {
                "schema_version": RECOGNIZER_SCHEMA_VERSION,
                "records": history,
                "field_counts": field_counts,
                "warning": "Validation records are PaddleOCR pseudo labels unless you replace them with reviewed labels.",
            },
        )
        print(
            f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_exact_match={validation['exact_match']:.2%}"
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
    if not all(isinstance(field, str) and field in DETECTION_CLASSES for field in fields):
        raise ValueError("OCR checkpoint fields are invalid")
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
                "preprocess": "RGB crop -> grayscale -> aspect-preserving resize -> white letterbox -> divide by 255.0",
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
    invalid = sorted(set(fields) - set(DETECTION_CLASSES))
    if not fields or invalid:
        raise argparse.ArgumentTypeError(
            f"fields must be a non-empty subset of: {','.join(DETECTION_CLASSES)}"
        )
    return fields


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
