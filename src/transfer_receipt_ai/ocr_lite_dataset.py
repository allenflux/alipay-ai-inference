"""Build field-specialised, Paddle-free training manifests from OCR pseudo labels.

The first self-trained OCR experiment used one CTC alphabet for every receipt
field.  That makes a small recogniser spend most of its capacity on open-ended
Chinese recipient text.  This module deliberately changes the problem:

* amount and status-bar time become tiny CTC targets with numeric alphabets;
* transfer status and payment method become finite classifications; and
* recipient becomes a known-recipient classifier with an explicit ``unknown``
  class instead of a misleading attempt at arbitrary Chinese transcription.

It never imports Paddle.  Existing Paddle-derived JSON is treated only as
offline pseudo-label data and remains auditable through its source provenance.
The generated manifests reference the original crop directory rather than
copying 120k images again.  Pass that directory as ``--dataset-root`` to the
CTC/classifier trainers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .labels import DETECTION_CLASSES


SCHEMA_VERSION = 1
UNKNOWN_RECIPIENT_CLASS = "unknown"


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


def _read_records(records_path: Path) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
    """Read and validate a pseudo-label JSONL without changing its images."""
    records_path = records_path.resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    dataset_root = records_path.parent.resolve()
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    group_splits: dict[str, str] = {}
    ids: set[str] = set()
    with records_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("record must be an object")
                record = dict(value)
                record_id = record.get("id")
                image = record.get("image")
                field = record.get("field")
                semantic_value = record.get("semantic_value")
                split = record.get("split")
                group_id = record.get("group_id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError("id must be a non-empty string")
                if record_id in ids:
                    raise ValueError(f"duplicate id {record_id!r}")
                if not isinstance(image, str) or not image:
                    raise ValueError("image must be a non-empty relative path")
                image_path = (dataset_root / image).resolve()
                try:
                    image_path.relative_to(dataset_root)
                except ValueError:
                    raise ValueError("image escapes the source dataset root") from None
                if not image_path.is_file():
                    raise FileNotFoundError(f"image not found: {image_path}")
                if not isinstance(field, str) or field not in DETECTION_CLASSES:
                    raise ValueError("field is not a receipt detection class")
                if not isinstance(semantic_value, str) or not semantic_value:
                    raise ValueError("semantic_value must be a non-empty string")
                if split not in {"train", "val", "test"}:
                    raise ValueError("split must be train, val, or test")
                if not isinstance(group_id, str) or not group_id:
                    raise ValueError("group_id must be a non-empty string")
                prior_split = group_splits.setdefault(group_id, split)
                if prior_split != split:
                    raise ValueError(f"group_id {group_id!r} appears in both {prior_split} and {split}")
                ids.add(record_id)
                accepted.append(record)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                rejected.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "line_number": line_number,
                        "reason": "invalid_source_record",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
    if not accepted:
        raise ValueError("No valid pseudo-label records found")
    return dataset_root, accepted, rejected


def _record_base(record: Mapping[str, object], *, task: str) -> dict[str, object]:
    """Keep enough provenance for review without duplicating source images."""
    source_text = record.get("text") if isinstance(record.get("text"), str) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"{record['id']}:{task}",
        "image": str(record["image"]),
        "field": str(record["field"]),
        "split": str(record["split"]),
        "group_id": str(record["group_id"]),
        "source_text": source_text,
        "semantic_value": str(record["semantic_value"]),
        "paddle_confidence": record.get("paddle_confidence"),
        "detector_score": record.get("detector_score"),
        "source": record.get("source"),
        "result_json": record.get("result_json"),
        "crop_sha256": record.get("crop_sha256"),
        "label_source": record.get("label_source", "pseudo_label"),
    }


def _numeric_target(field: str, semantic_value: str) -> str | None:
    if field == "amount":
        target = semantic_value.removeprefix("¥").removeprefix("￥")
        if target and all(character in "0123456789." for character in target) and target.count(".") <= 1:
            return target
        return None
    if field == "time":
        if semantic_value and all(character in "0123456789:" for character in semantic_value) and semantic_value.count(":") in {1, 2}:
            return semantic_value
        return None
    raise ValueError(f"Unsupported numeric task field: {field}")


def _stable_bucket(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _build_recipient_catalog(
    records: list[dict[str, object]], *, top_k: int, min_train_count: int
) -> tuple[dict[str, str], list[dict[str, object]]]:
    train_counts = Counter(
        str(record["semantic_value"])
        for record in records
        if record["field"] == "recipient_field" and record["split"] == "train"
    )
    candidates = [
        (value, count)
        for value, count in train_counts.items()
        if count >= min_train_count and value.strip()
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    candidates = candidates[:top_k]
    mapping = {value: f"known_{index:04d}" for index, (value, _count) in enumerate(candidates, start=1)}
    catalog = [
        {
            "class_name": class_name,
            "recipient_value": value,
            "train_records": count,
        }
        for value, count in candidates
        for class_name in (mapping[value],)
    ]
    return mapping, catalog


def _balanced_recipient_records(
    records: list[dict[str, object]], *, known_mapping: Mapping[str, str], unknown_to_known_ratio: float
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Downsample only training unknowns so they cannot swamp known classes."""
    mapped: list[dict[str, object]] = []
    unknown_train: list[dict[str, object]] = []
    known_train_count = 0
    for record in records:
        if record["field"] != "recipient_field":
            continue
        base = _record_base(record, task="recipient_classifier")
        recipient_value = str(record["semantic_value"])
        class_name = known_mapping.get(recipient_value, UNKNOWN_RECIPIENT_CLASS)
        base["class_name"] = class_name
        base["recipient_value"] = recipient_value
        if base["split"] == "train" and class_name == UNKNOWN_RECIPIENT_CLASS:
            unknown_train.append(base)
        else:
            mapped.append(base)
            if base["split"] == "train":
                known_train_count += 1

    limit = math.ceil(known_train_count * unknown_to_known_ratio)
    unknown_train.sort(key=lambda record: (_stable_bucket(str(record["id"])), str(record["id"])))
    mapped.extend(unknown_train[:limit])
    rejected = [
        {
            "schema_version": SCHEMA_VERSION,
            "id": str(record["id"]),
            "field": "recipient_field",
            "reason": "recipient_unknown_downsampled",
        }
        for record in unknown_train[limit:]
    ]
    return mapped, rejected


def build_lite_dataset(
    *,
    records_path: Path,
    output_dir: Path,
    recipient_top_k: int = 200,
    recipient_min_train_count: int = 25,
    recipient_unknown_to_known_ratio: float = 2.0,
) -> dict[str, object]:
    """Write specialised manifests without importing Paddle or copying images."""
    if recipient_top_k <= 0:
        raise ValueError("recipient_top_k must be positive")
    if recipient_min_train_count <= 0:
        raise ValueError("recipient_min_train_count must be positive")
    if not math.isfinite(recipient_unknown_to_known_ratio) or recipient_unknown_to_known_ratio < 0.0:
        raise ValueError("recipient_unknown_to_known_ratio must be a finite non-negative number")

    dataset_root, records, rejected = _read_records(records_path)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    amount_records: list[dict[str, object]] = []
    time_records: list[dict[str, object]] = []
    status_records: list[dict[str, object]] = []
    payment_records: list[dict[str, object]] = []
    for record in records:
        field = str(record["field"])
        semantic_value = str(record["semantic_value"])
        if field in {"amount", "time"}:
            target = _numeric_target(field, semantic_value)
            if target is None:
                rejected.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": str(record["id"]),
                        "field": field,
                        "reason": "invalid_numeric_semantic_value",
                        "detail": semantic_value,
                    }
                )
                continue
            target_record = _record_base(record, task=f"{field}_ctc")
            target_record["text"] = target
            (amount_records if field == "amount" else time_records).append(target_record)
        elif field == "transfer_status":
            if semantic_value not in {"success", "pending", "failed"}:
                rejected.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": str(record["id"]),
                        "field": field,
                        "reason": "invalid_status_class",
                        "detail": semantic_value,
                    }
                )
                continue
            class_record = _record_base(record, task="transfer_status_classifier")
            class_record["class_name"] = semantic_value
            status_records.append(class_record)
        elif field == "payment_method_field":
            if semantic_value not in {"yuebao", "balance", "huabei", "bank_card", "other"}:
                rejected.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": str(record["id"]),
                        "field": field,
                        "reason": "invalid_payment_method_class",
                        "detail": semantic_value,
                    }
                )
                continue
            class_record = _record_base(record, task="payment_method_classifier")
            class_record["class_name"] = semantic_value
            payment_records.append(class_record)

    recipient_mapping, recipient_catalog = _build_recipient_catalog(
        records,
        top_k=recipient_top_k,
        min_train_count=recipient_min_train_count,
    )
    recipient_records, recipient_downsampled = _balanced_recipient_records(
        records,
        known_mapping=recipient_mapping,
        unknown_to_known_ratio=recipient_unknown_to_known_ratio,
    )
    rejected.extend(recipient_downsampled)

    manifests = {
        "amount_ctc": amount_records,
        "time_ctc": time_records,
        "transfer_status_classifier": status_records,
        "payment_method_classifier": payment_records,
        "recipient_classifier": recipient_records,
    }
    for name, task_records in manifests.items():
        task_records.sort(key=lambda record: str(record["id"]))
        _atomic_write_jsonl(output_dir / f"{name}.jsonl", task_records)

    recipient_catalog_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_recipient_catalog_v1",
        "unknown_class": UNKNOWN_RECIPIENT_CLASS,
        "entries": recipient_catalog,
        "warning": (
            "Only known classes selected from train records are in this catalog. "
            "All long-tail or unseen recipients map to unknown/review by design."
        ),
    }
    _atomic_write_json(output_dir / "recipient_catalog.json", recipient_catalog_payload)
    _atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)

    def split_counts(task_records: list[dict[str, object]]) -> dict[str, int]:
        counts = Counter(str(record["split"]) for record in task_records)
        return {split: int(counts[split]) for split in ("train", "val", "test")}

    task_counts = {name: split_counts(task_records) for name, task_records in manifests.items()}
    classifier_classes = {
        "transfer_status_classifier": sorted({str(record["class_name"]) for record in status_records if record["split"] == "train"}),
        "payment_method_classifier": sorted({str(record["class_name"]) for record in payment_records if record["split"] == "train"}),
        "recipient_classifier": [UNKNOWN_RECIPIENT_CLASS] + [entry["class_name"] for entry in recipient_catalog],
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_field_lite_dataset_v1",
        "source_records": records_path.resolve().as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "tasks": {
            name: {"records": len(task_records), "by_split": task_counts[name]}
            for name, task_records in manifests.items()
        },
        "classifier_classes": classifier_classes,
        "recipient_policy": {
            "mode": "top_k_known_plus_unknown",
            "top_k_requested": recipient_top_k,
            "min_train_count": recipient_min_train_count,
            "known_class_count": len(recipient_catalog),
            "unknown_class": UNKNOWN_RECIPIENT_CLASS,
            "train_unknown_to_known_ratio": recipient_unknown_to_known_ratio,
        },
        "rejected_records": len(rejected),
        "warning": (
            "Source labels are historical OCR pseudo labels unless independently reviewed. "
            "This builder creates an offline, Paddle-free training manifest; it does not establish business accuracy."
        ),
    }
    _atomic_write_json(output_dir / "dataset.contract.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build specialised numeric/classification manifests from existing OCR pseudo labels"
    )
    parser.add_argument("--records", type=Path, required=True, help="pseudo_labels.jsonl produced before Paddle was removed")
    parser.add_argument("--output", type=Path, required=True, help="New empty output directory for lite-v2 manifests")
    parser.add_argument("--recipient-top-k", type=int, default=200)
    parser.add_argument("--recipient-min-train-count", type=int, default=25)
    parser.add_argument("--recipient-unknown-to-known-ratio", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = build_lite_dataset(
            records_path=args.records,
            output_dir=args.output,
            recipient_top_k=args.recipient_top_k,
            recipient_min_train_count=args.recipient_min_train_count,
            recipient_unknown_to_known_ratio=args.recipient_unknown_to_known_ratio,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"OCR lite dataset build failed:\n{error}") from None
    task_summary = ", ".join(
        f"{name}={details['records']}" for name, details in dict(summary["tasks"]).items()
    )
    print(f"Wrote specialised OCR-lite manifests to {args.output} ({task_summary})")


if __name__ == "__main__":  # pragma: no cover
    main()
