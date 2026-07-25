"""Train, export and evaluate small field-classification ONNX models.

This is intentionally not a generic OCR replacement.  It recognises finite
business choices (transfer status, payment method, or a selected set of known
recipients) from crops already localised by the receipt detector.  The model
uses only PyTorch during training and exports a fixed-shape ONNX graph for
offline .NET/ONNX Runtime deployment.
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


SCHEMA_VERSION = 1
KIND = "receipt_field_classifier_v1"
DEFAULT_IMAGE_HEIGHT = 48
DEFAULT_IMAGE_WIDTH = 384


@dataclass(frozen=True)
class ClassifierConfig:
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    base_channels: int = 24
    pooled_width: int = 8

    def validate(self) -> None:
        if self.image_height < 16 or self.image_width < 64:
            raise ValueError("image_height must be at least 16 and image_width must be at least 64")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if self.pooled_width < 1 or self.pooled_width > 32:
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
            "Field-classifier training requires PyTorch. Install the server's CPU/CUDA-compatible torch wheel first, "
            "then install requirements-train-ocr.txt."
        ) from error
    return torch, nn


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ONNX classifier evaluation requires onnxruntime. Install the CUDA-matched onnxruntime-gpu package "
            "on a GPU server, or onnxruntime in a separate CPU environment."
        ) from error
    return onnxruntime


def _resolve_device(torch: Any, requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for field-classifier training but PyTorch CUDA is unavailable")
        return requested
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested for field-classifier training but is unavailable")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")


def build_classifier(*, class_count: int, config: ClassifierConfig) -> Any:
    """Return a small depthwise CNN with enough horizontal context for text rows."""
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    config.validate()
    _, nn = _require_torch()

    class DepthwiseBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int]) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)

    class ReceiptFieldClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            first = config.base_channels
            second = first * 2
            third = first * 3
            fourth = first * 4
            self.stem = nn.Sequential(
                nn.Conv2d(1, first, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(first),
                nn.ReLU(inplace=True),
            )
            self.encoder = nn.Sequential(
                DepthwiseBlock(first, second, stride=(2, 2)),
                DepthwiseBlock(second, third, stride=(2, 2)),
                DepthwiseBlock(third, fourth, stride=(2, 1)),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, config.pooled_width))
            self.classifier = nn.Linear(fourth * config.pooled_width, class_count)

        def forward(self, image: Any) -> Any:
            features = self.encoder(self.stem(image))
            return self.classifier(self.pool(features).flatten(1))

    return ReceiptFieldClassifier()


def preprocess_image(image_path: Path, *, config: ClassifierConfig) -> np.ndarray:
    """Return the fixed NCHW preprocessing declared by the delivery contract."""
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


def _tensor_image(image_path: Path, *, config: ClassifierConfig, torch: Any) -> Any:
    return torch.from_numpy(preprocess_image(image_path, config=config)[0])


def load_records(records_path: Path, *, dataset_root: Path | None = None) -> list[dict[str, object]]:
    """Load one task manifest and enforce group-safe, on-disk samples."""
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
    fields: set[str] = set()
    with records_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{records_path}:{line_number}: invalid JSON: {error}") from None
            if not isinstance(value, Mapping):
                raise ValueError(f"{records_path}:{line_number}: record must be an object")
            record_id = value.get("id")
            image = value.get("image")
            field = value.get("field")
            class_name = value.get("class_name")
            split = value.get("split")
            group_id = value.get("group_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{records_path}:{line_number}: id must be a non-empty string")
            if record_id in ids:
                raise ValueError(f"{records_path}:{line_number}: duplicate id {record_id!r}")
            if not isinstance(image, str) or not image:
                raise ValueError(f"{records_path}:{line_number}: image must be a non-empty relative path")
            image_path = (dataset_root / image).resolve()
            try:
                image_path.relative_to(dataset_root)
            except ValueError:
                raise ValueError(f"{records_path}:{line_number}: image escapes dataset root") from None
            if not image_path.is_file():
                raise FileNotFoundError(f"{records_path}:{line_number}: image not found: {image_path}")
            image_key = image_path.as_posix().casefold()
            if image_key in images:
                raise ValueError(f"{records_path}:{line_number}: duplicate image across records")
            if not isinstance(field, str) or not field:
                raise ValueError(f"{records_path}:{line_number}: field must be a non-empty string")
            if not isinstance(class_name, str) or not class_name:
                raise ValueError(f"{records_path}:{line_number}: class_name must be a non-empty string")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{records_path}:{line_number}: split must be train, val, or test")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{records_path}:{line_number}: group_id must be a non-empty string")
            prior_split = group_splits.setdefault(group_id, split)
            if prior_split != split:
                raise ValueError(
                    f"{records_path}:{line_number}: group_id {group_id!r} appears in both {prior_split} and {split} splits"
                )
            ids.add(record_id)
            images.add(image_key)
            fields.add(field)
            records.append(
                {
                    "id": record_id,
                    "image_path": image_path,
                    "image": image,
                    "field": field,
                    "class_name": class_name,
                    "split": split,
                    "group_id": group_id,
                    "source": value.get("source"),
                    "result_json": value.get("result_json"),
                    "crop_sha256": value.get("crop_sha256"),
                    "label_source": value.get("label_source") if isinstance(value.get("label_source"), str) else "unspecified",
                }
            )
    if not records:
        raise ValueError(f"No classification records in {records_path}")
    if len(fields) != 1:
        raise ValueError(f"Classification manifest must contain exactly one field, got: {','.join(sorted(fields))}")
    return records


def _field_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(record["split"]) for record in records)
    return {split: int(counts[split]) for split in ("train", "val", "test")}


def _make_dataset(records: Sequence[Mapping[str, object]], *, class_to_index: Mapping[str, int], config: ClassifierConfig, torch: Any) -> Any:
    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> tuple[Any, int, str, str]:
            record = records[index]
            return (
                _tensor_image(Path(record["image_path"]), config=config, torch=torch),
                class_to_index[str(record["class_name"])],
                str(record["class_name"]),
                str(record["id"]),
            )

    return Dataset()


def _collate(batch: Sequence[tuple[Any, int, str, str]], *, torch: Any) -> tuple[Any, Any, list[str], list[str]]:
    images, labels, class_names, ids = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(class_names), list(ids)


def _classification_metrics(
    *,
    targets: Sequence[int],
    predictions: Sequence[int],
    classes: Sequence[str],
) -> dict[str, object]:
    if not targets:
        raise ValueError("No validation records")
    by_class: dict[str, dict[str, object]] = {}
    recalls: list[float] = []
    for index, class_name in enumerate(classes):
        total = sum(target == index for target in targets)
        correct = sum(target == index and prediction == index for target, prediction in zip(targets, predictions))
        predicted = sum(prediction == index for prediction in predictions)
        if total:
            recall = correct / total
            recalls.append(recall)
        else:
            recall = None
        precision = correct / predicted if predicted else None
        by_class[class_name] = {
            "records": total,
            "correct": correct,
            "predicted": predicted,
            "recall": recall,
            "precision": precision,
        }
    correct = sum(target == prediction for target, prediction in zip(targets, predictions))
    return {
        "records": len(targets),
        "correct": correct,
        "accuracy": correct / len(targets),
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "by_class": by_class,
    }


def _write_checkpoint(path: Path, payload: Mapping[str, object], *, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, *, torch: Any) -> Mapping[str, object]:
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != KIND:
        raise ValueError("Unsupported field-classifier checkpoint schema")
    return payload


def train_classifier(
    *,
    records_path: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
    config: ClassifierConfig = ClassifierConfig(),
    device: str = "auto",
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 0,
) -> Path:
    """Train one status/payment/recipient classifier and return its best checkpoint."""
    config.validate()
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"training output already contains files: {output_dir}")
    records = load_records(records_path, dataset_root=dataset_root)
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    if not train_records or not val_records:
        raise ValueError("Both train and val splits are required for classifier training")
    field = str(records[0]["field"])
    classes = sorted({str(record["class_name"]) for record in train_records})
    if len(classes) < 2:
        raise ValueError("At least two training classes are required")
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    unseen_non_train_classes = sorted({str(record["class_name"]) for record in records} - set(classes))
    if unseen_non_train_classes:
        raise ValueError(
            "Validation/test contains classes absent from train: "
            f"{','.join(unseen_non_train_classes)}. Rebuild the group split or collect representative truth labels."
        )
    class_counts = Counter(str(record["class_name"]) for record in train_records)
    torch, _ = _require_torch()
    target_device = _resolve_device(torch, device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if target_device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    train_dataset = _make_dataset(train_records, class_to_index=class_to_index, config=config, torch=torch)
    val_dataset = _make_dataset(val_records, class_to_index=class_to_index, config=config, torch=torch)
    collate = lambda batch: _collate(batch, torch=torch)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=target_device.startswith("cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=target_device.startswith("cuda"),
    )
    # Square-root weighting makes rare legitimate categories visible without
    # letting a one-off pseudo label dominate every optimization step.
    raw_weights = np.asarray([math.sqrt(len(train_records) / class_counts[class_name]) for class_name in classes], dtype=np.float32)
    class_weights = torch.tensor(raw_weights / raw_weights.mean(), dtype=torch.float32, device=target_device)
    model = build_classifier(class_count=len(classes), config=config).to(target_device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output_dir / "labels.json",
        {
            "schema_version": SCHEMA_VERSION,
            "classes": classes,
            "field": field,
            "training_counts": {class_name: int(class_counts[class_name]) for class_name in classes},
        },
    )
    history: list[dict[str, object]] = []
    best_score = float("-inf")
    best_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_items = 0
        for images, labels, _class_names, _ids in train_loader:
            images = images.to(target_device)
            labels = labels.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_total += float(loss.detach().cpu()) * images.shape[0]
            train_items += images.shape[0]
        model.eval()
        val_loss_total = 0.0
        val_items = 0
        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for images, labels, _class_names, _ids in val_loader:
                images = images.to(target_device)
                labels = labels.to(target_device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_total += float(loss.detach().cpu()) * images.shape[0]
                val_items += images.shape[0]
                targets.extend(int(value) for value in labels.detach().cpu().tolist())
                predictions.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().tolist())
        metrics = _classification_metrics(targets=targets, predictions=predictions, classes=classes)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(1, train_items),
            "val_loss": val_loss_total / max(1, val_items),
            "val_accuracy": metrics["accuracy"],
            "val_macro_recall": metrics["macro_recall"],
        }
        history.append(epoch_record)
        checkpoint_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "classes": classes,
            "field": field,
            "split_counts": _field_counts(records),
            "training_counts": {class_name: int(class_counts[class_name]) for class_name in classes},
            "epoch": epoch,
            "metrics": epoch_record,
        }
        _write_checkpoint(output_dir / "last.pt", checkpoint_payload, torch=torch)
        # Macro recall is the selection criterion because the major class is
        # often "unknown" and ordinary accuracy would conceal minority errors.
        score = float(metrics["macro_recall"])
        if score > best_score:
            best_score = score
            _write_checkpoint(best_path, checkpoint_payload, torch=torch)
        _atomic_write_json(
            output_dir / "training_history.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND,
                "field": field,
                "records": history,
                "warning": "Validation labels may be OCR pseudo labels. Keep an independent human-reviewed test set for release acceptance.",
            },
        )
        print(
            f"epoch {epoch}/{epochs}: train_loss={epoch_record['train_loss']:.4f} "
            f"val_loss={epoch_record['val_loss']:.4f} val_accuracy={metrics['accuracy']:.2%} "
            f"val_macro_recall={metrics['macro_recall']:.2%}"
        )
    return best_path


def export_onnx(*, checkpoint_path: Path, output_path: Path) -> tuple[Path, Path, Path]:
    """Export a classifier checkpoint with labels and a self-verifying contract."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = output_path.resolve()
    labels_path = output_path.with_suffix(".labels.json")
    contract_path = output_path.with_suffix(".contract.json")
    existing = next((path for path in (output_path, labels_path, contract_path) if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"Refusing to overwrite classifier artifact: {existing}")
    torch, _ = _require_torch()
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    raw_config = payload.get("config")
    classes = payload.get("classes")
    field = payload.get("field")
    state_dict = payload.get("state_dict")
    if not isinstance(raw_config, Mapping) or not isinstance(classes, list) or not isinstance(field, str) or not isinstance(state_dict, Mapping):
        raise ValueError("Classifier checkpoint is missing config, classes, field, or state_dict")
    if len(classes) < 2 or not all(isinstance(value, str) and value for value in classes) or len(set(classes)) != len(classes):
        raise ValueError("Classifier checkpoint classes are invalid")
    try:
        config = ClassifierConfig(
            image_height=int(raw_config["image_height"]),
            image_width=int(raw_config["image_width"]),
            base_channels=int(raw_config["base_channels"]),
            pooled_width=int(raw_config["pooled_width"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Classifier checkpoint config is invalid") from error
    config.validate()
    model = build_classifier(class_count=len(classes), config=config)
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
    except TypeError:
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
    labels_payload = {
        "schema_version": SCHEMA_VERSION,
        "classes": classes,
        "field": field,
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
            "field": field,
            "input": {
                "name": "image",
                "dtype": "float32",
                "shape": [1, 1, config.image_height, config.image_width],
                "preprocess": "RGB crop -> grayscale -> aspect-preserving resize -> white letterbox -> divide by 255.0",
            },
            "output": {"name": "logits", "shape": [1, int(logits.shape[1])], "layout": "[batch,class]"},
            "model": asdict(config),
        },
    )
    return output_path, labels_path, contract_path


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected an object")
    return value


def _load_artifacts(model_path: Path) -> tuple[ClassifierConfig, list[str], str, str, str]:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    labels_path = model_path.with_suffix(".labels.json")
    contract_path = model_path.with_suffix(".contract.json")
    labels = _load_json(labels_path)
    contract = _load_json(contract_path)
    if contract.get("kind") != KIND:
        raise ValueError(f"Classifier contract kind must be {KIND}")
    if contract.get("onnx_sha256") != _sha256(model_path):
        raise ValueError("Classifier ONNX SHA-256 does not match contract")
    if contract.get("labels_sha256") != _sha256(labels_path):
        raise ValueError("Classifier labels SHA-256 does not match contract")
    classes = labels.get("classes")
    field = labels.get("field")
    if not isinstance(classes, list) or len(classes) < 2 or not all(isinstance(value, str) and value for value in classes):
        raise ValueError("Classifier labels are invalid")
    if not isinstance(field, str) or not field or contract.get("field") != field:
        raise ValueError("Classifier field is invalid")
    raw_config = contract.get("model")
    raw_input = contract.get("input")
    raw_output = contract.get("output")
    if not isinstance(raw_config, Mapping) or not isinstance(raw_input, Mapping) or not isinstance(raw_output, Mapping):
        raise ValueError("Classifier contract has no model/input/output configuration")
    try:
        config = ClassifierConfig(
            image_height=int(raw_config["image_height"]),
            image_width=int(raw_config["image_width"]),
            base_channels=int(raw_config["base_channels"]),
            pooled_width=int(raw_config["pooled_width"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Classifier contract model configuration is invalid") from error
    config.validate()
    if raw_input.get("name") != "image" or raw_input.get("shape") != [1, 1, config.image_height, config.image_width]:
        raise ValueError("Classifier contract input is invalid")
    if raw_output.get("name") != "logits" or raw_output.get("shape") != [1, len(classes)]:
        raise ValueError("Classifier contract output is invalid")
    return config, list(classes), field, "image", "logits"


def _create_session(onnxruntime: Any, model_path: Path, *, device: str) -> tuple[Any, list[str]]:
    providers = onnx_providers(device, onnxruntime)
    _preload_cuda_dlls(onnxruntime, providers)
    session = onnxruntime.InferenceSession(str(model_path), providers=providers)
    active = list(session.get_providers())
    requested_cuda = device.lower() == "cuda" or device.lower().startswith("cuda:")
    if requested_cuda and "CUDAExecutionProvider" not in active:
        raise RuntimeError("Classifier ONNX session did not activate CUDAExecutionProvider")
    return session, active


def _softmax_confidence(logits: np.ndarray) -> tuple[int, float]:
    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Classifier emitted empty or non-finite logits")
    winner = int(np.argmax(flat))
    shifted = flat - np.max(flat)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return winner, float(probabilities[winner])


def evaluate_onnx(
    *,
    model_path: Path,
    records_path: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
    split: str = "test",
    device: str = "auto",
    min_accuracy: float | None = None,
    min_macro_recall: float | None = None,
    min_confidence: float | None = None,
    max_non_success_to_success: int | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Evaluate a held-out classifier split and retain every prediction for review."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    for name, value in (
        ("min_accuracy", min_accuracy),
        ("min_macro_recall", min_macro_recall),
        ("min_confidence", min_confidence),
    ):
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1")
    if max_non_success_to_success is not None and max_non_success_to_success < 0:
        raise ValueError("max_non_success_to_success must be zero or a positive integer")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"evaluation output already contains files: {output_dir}")
    config, classes, field, input_name, output_name = _load_artifacts(model_path)
    records = load_records(records_path, dataset_root=dataset_root)
    if str(records[0]["field"]) != field:
        raise ValueError(f"Model field {field!r} does not match manifest field {records[0]['field']!r}")
    if max_non_success_to_success is not None and field != "transfer_status":
        raise ValueError("max_non_success_to_success is only valid for a transfer_status classifier")
    evaluation_records = [record for record in records if record["split"] == split]
    if not evaluation_records:
        raise ValueError(f"No {split} records found")
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    unknown_labels = sorted({str(record["class_name"]) for record in evaluation_records} - set(classes))
    if unknown_labels:
        raise ValueError(f"Evaluation has labels absent from model: {','.join(unknown_labels)}")
    onnxruntime = _require_onnxruntime()
    session, active_providers = _create_session(onnxruntime, model_path.resolve(), device=device)
    comparisons: list[dict[str, object]] = []
    targets: list[int] = []
    predictions: list[int] = []
    accepted_targets: list[int] = []
    accepted_predictions: list[int] = []
    latencies: list[float] = []
    for record in evaluation_records:
        started = perf_counter()
        logits = session.run([output_name], {input_name: preprocess_image(Path(record["image_path"]), config=config)})[0]
        latency_ms = (perf_counter() - started) * 1000.0
        predicted_index, confidence = _softmax_confidence(np.asarray(logits))
        target_index = class_to_index[str(record["class_name"])]
        accepted = min_confidence is None or confidence >= min_confidence
        targets.append(target_index)
        predictions.append(predicted_index)
        if accepted:
            accepted_targets.append(target_index)
            accepted_predictions.append(predicted_index)
        latencies.append(latency_ms)
        comparisons.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": str(record["id"]),
                "field": field,
                "split": split,
                "group_id": str(record["group_id"]),
                "image": Path(record["image_path"]).as_posix(),
                "reference_class": classes[target_index],
                "candidate_class": classes[predicted_index],
                "exact": target_index == predicted_index,
                "confidence": round(confidence, 6),
                "state": "read" if accepted else "review",
                "latency_ms": round(latency_ms, 4),
            }
        )
    comparisons.sort(key=lambda record: str(record["id"]))
    metrics = _classification_metrics(targets=targets, predictions=predictions, classes=classes)
    metrics["review"] = {
        "min_confidence": min_confidence,
        "accepted": len(accepted_targets),
        "review": len(targets) - len(accepted_targets),
        "accepted_accuracy": (
            _classification_metrics(
                targets=accepted_targets,
                predictions=accepted_predictions,
                classes=classes,
            )["accuracy"]
            if accepted_targets
            else None
        ),
    }
    non_success_to_success = [
        comparison
        for comparison in comparisons
        if comparison["state"] == "read"
        and comparison["reference_class"] in {"pending", "failed"}
        and comparison["candidate_class"] == "success"
    ]
    if field == "transfer_status":
        metrics["non_success_to_success"] = {
            "accepted_count": len(non_success_to_success),
            "all_raw_count": sum(
                comparison["reference_class"] in {"pending", "failed"}
                and comparison["candidate_class"] == "success"
                for comparison in comparisons
            ),
        }
    sorted_latencies = sorted(latencies)
    percentile = lambda fraction: sorted_latencies[min(len(sorted_latencies) - 1, int(math.ceil(fraction * len(sorted_latencies))) - 1)]
    metrics["latency_ms"] = {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "mean": sum(latencies) / len(latencies),
    }
    failures: list[str] = []
    if min_accuracy is not None and float(metrics["accuracy"]) < min_accuracy:
        failures.append(f"accuracy={metrics['accuracy']:.4f} < {min_accuracy:.4f}")
    if min_macro_recall is not None and float(metrics["macro_recall"]) < min_macro_recall:
        failures.append(f"macro_recall={metrics['macro_recall']:.4f} < {min_macro_recall:.4f}")
    if max_non_success_to_success is not None and len(non_success_to_success) > max_non_success_to_success:
        failures.append(
            "accepted_non_success_to_success="
            f"{len(non_success_to_success)} > {max_non_success_to_success}"
        )
    label_sources = sorted({str(record.get("label_source", "unspecified")) for record in evaluation_records})
    transaction_truth = label_sources == ["transaction_truth"]
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_field_classifier_evaluation_v1",
        "model": model_path.resolve().as_posix(),
        "model_sha256": _sha256(model_path.resolve()),
        "records": records_path.resolve().as_posix(),
        "field": field,
        "evaluation_split": split,
        "label_sources": label_sources,
        "providers": active_providers,
        "metrics": metrics,
        "acceptance": {
            "min_accuracy": min_accuracy,
            "min_macro_recall": min_macro_recall,
            "min_confidence": min_confidence,
            "max_non_success_to_success": max_non_success_to_success,
            "passed": not failures,
            "failures": failures,
        },
        "warning": (
            "This evaluates against local transaction truth. Validate receipt-key associations and keep a separate "
            "audit set before treating it as production accuracy."
            if transaction_truth
            else "Pseudo-label evaluation measures teacher parity, not independent business truth."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    _atomic_write_jsonl(output_dir / "disagreements.jsonl", [record for record in comparisons if not bool(record["exact"])])
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export/evaluate a lightweight receipt field classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train a status/payment/recipient classifier")
    train.add_argument("--records", type=Path, required=True)
    train.add_argument("--dataset-root", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    train.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    train.add_argument("--base-channels", type=int, default=24)
    train.add_argument("--pooled-width", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--onnx-output", type=Path)
    export = subparsers.add_parser("export", help="Export a classifier checkpoint to ONNX")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a classifier ONNX model")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--records", type=Path, required=True)
    evaluate.add_argument("--dataset-root", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--min-accuracy", type=float)
    evaluate.add_argument("--min-macro-recall", type=float)
    evaluate.add_argument(
        "--min-confidence",
        type=float,
        help="Treat lower-confidence classifications as review rather than accepted reads",
    )
    evaluate.add_argument(
        "--max-non-success-to-success",
        type=int,
        help="Hard maximum accepted pending/failed -> success errors; transfer_status only",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            config = ClassifierConfig(
                image_height=args.image_height,
                image_width=args.image_width,
                base_channels=args.base_channels,
                pooled_width=args.pooled_width,
            )
            checkpoint = train_classifier(
                records_path=args.records,
                dataset_root=args.dataset_root,
                output_dir=args.output,
                config=config,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            print(f"Best classifier checkpoint: {checkpoint}")
            if args.onnx_output is not None:
                model, labels, contract = export_onnx(checkpoint_path=checkpoint, output_path=args.onnx_output)
                print(f"Exported classifier ONNX: {model}\nLabels: {labels}\nContract: {contract}")
        elif args.command == "export":
            model, labels, contract = export_onnx(checkpoint_path=args.checkpoint, output_path=args.output)
            print(f"Exported classifier ONNX: {model}\nLabels: {labels}\nContract: {contract}")
        else:
            summary, failures = evaluate_onnx(
                model_path=args.model,
                records_path=args.records,
                dataset_root=args.dataset_root,
                output_dir=args.output,
                split=args.split,
                device=args.device,
                min_accuracy=args.min_accuracy,
                min_macro_recall=args.min_macro_recall,
                min_confidence=args.min_confidence,
                max_non_success_to_success=args.max_non_success_to_success,
            )
            metrics = dict(summary["metrics"])
            print(
                f"Wrote {metrics['records']} classifier comparison(s) to {args.output} "
                f"(accuracy={metrics['accuracy']:.2%}, macro_recall={metrics['macro_recall']:.2%})"
            )
            if failures:
                raise SystemExit("Classifier ONNX candidate did not meet the requested acceptance gate:\n- " + "\n- ".join(failures))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OCR lite classifier failed:\n{error}") from None


if __name__ == "__main__":  # pragma: no cover
    main()
