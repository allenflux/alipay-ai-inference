"""Build receipt-level manifests for the unified lightweight field reader.

The detection/pseudo-label pipeline stores one crop per field.  A unified
reader, however, receives the four crops from *one receipt* in a fixed order
and returns all field outputs in one ONNX invocation.  This module turns the
flat crop manifest into that receipt-level representation without copying any
images or importing Paddle/Torch.

Only four deployable fields are included:

``amount``, ``time``, ``transfer_status`` and ``payment_method_field``.

The payment slot deliberately retains the visible payment-method value (for
example ``建设银行储蓄卡(3667)``) as a CTC target.  Its normalised business
category remains provenance only; reducing it to ``bank_card`` would make a
small model unable to match the existing Paddle output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .ocr import clean_text, extract_field_value
from .ocr_unified_targets import (
    parse_amount_aux_target,
    parse_amount_display_target,
    parse_payment_bank_prefix_target,
    parse_payment_card_tail_target,
    parse_time_aux_target,
    parse_time_display_target,
    structured_target_config,
)


SCHEMA_VERSION = 1
KIND = "receipt_unified_field_dataset_v1"
SLOT_ORDER = ("amount", "time", "transfer_status", "payment_method_field")
STATUS_CLASSES = ("success", "pending", "failed")


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


def _selection_key(value: Mapping[str, object]) -> tuple[float, float, str]:
    """Prefer the strongest teacher/detector pair for an accidental duplicate slot."""
    try:
        paddle = float(value.get("paddle_confidence", -1.0))
    except (TypeError, ValueError):
        paddle = -1.0
    try:
        detector = float(value.get("detector_score", -1.0))
    except (TypeError, ValueError):
        detector = -1.0
    return paddle, detector, str(value.get("id", ""))


def _numeric_target(field: str, semantic_value: str) -> str | None:
    if field == "amount":
        target = semantic_value
        sign = ""
        if target.startswith("-"):
            sign, target = "-", target[1:]
        target = target.removeprefix("¥").removeprefix("￥")
        target = sign + target
        if (
            target
            and all(character in "0123456789.-" for character in target)
            and target.count(".") == 1
            and target.count("-") <= 1
            and ("-" not in target or target.startswith("-"))
        ):
            return target
        return None
    if field == "time":
        target = semantic_value.replace("：", ":")
        if (
            target
            and all(character in "0123456789:- " for character in target)
            and target.count(":") in {1, 2}
        ):
            return target
        return None
    raise ValueError(f"Unsupported numeric field {field!r}")


def _receipt_key(record: Mapping[str, object]) -> str:
    """Return a per-screenshot key, never merely the transaction group id.

    One transaction can have several screenshots.  They must remain in the
    same train/val/test group, but their fields must not be combined into one
    artificial image sample.  The Python result JSON is the stable screenshot
    identity produced by the previous detector/Paddle run.
    """
    result_json = record.get("result_json")
    if isinstance(result_json, str) and result_json:
        return "result:" + result_json
    source = record.get("source")
    if isinstance(source, str) and source:
        return "source:" + source
    # Older hand-written manifests may omit both.  Keep deterministic
    # isolation rather than silently joining unrelated fields.
    return "record:" + str(record["id"])


def _read_flat_records(records_path: Path) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
    records_path = records_path.resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    dataset_root = records_path.parent.resolve()
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    with records_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw: Any = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError("record must be an object")
                record = dict(raw)
                record_id = record.get("id")
                image = record.get("image")
                field = record.get("field")
                text = record.get("text")
                semantic_value = record.get("semantic_value")
                split = record.get("split")
                group_id = record.get("group_id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError("id must be a non-empty string")
                if record_id in seen_ids:
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
                if not isinstance(field, str) or field not in SLOT_ORDER:
                    raise ValueError("field is not a unified-reader slot")
                if not isinstance(text, str) or not clean_text(text):
                    raise ValueError("text must be a non-empty string")
                if not isinstance(semantic_value, str) or not semantic_value:
                    raise ValueError("semantic_value must be a non-empty string")
                if split not in {"train", "val", "test"}:
                    raise ValueError("split must be train, val, or test")
                if not isinstance(group_id, str) or not group_id:
                    raise ValueError("group_id must be a non-empty string")
                seen_ids.add(record_id)
                record["image"] = image
                record["image_path"] = image_path
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
        raise ValueError("No valid amount/time/status/payment records found")
    return dataset_root, accepted, rejected


def _slot_payload(record: Mapping[str, object]) -> dict[str, object] | None:
    field = str(record["field"])
    semantic_value = str(record["semantic_value"])
    if field in {"amount", "time"}:
        target = _numeric_target(field, semantic_value)
        if target is None:
            return None
        payload: dict[str, object] = {
            "image": str(record["image"]),
            "text": target,
            "semantic_value": semantic_value,
            "paddle_text": record.get("paddle_text"),
            "paddle_confidence": record.get("paddle_confidence"),
            "detector_score": record.get("detector_score"),
            "crop_sha256": record.get("crop_sha256"),
        }
        # Keep the v5-compatible canonical CTC target while adding an optional
        # *visible* v6 target.  The latter is never silently repaired: CTC can
        # learn ``¥``, commas, a minus, hyphens, and the date-time space only
        # when that exact display grammar has been validated.
        source_text = clean_text(str(record["text"]))
        if field == "amount":
            amount_aux = parse_amount_aux_target(target)
            if amount_aux is not None:
                payload["amount_aux"] = amount_aux
            amount_display = parse_amount_display_target(source_text)
            if amount_display is not None:
                payload["visible_text"] = amount_display["visible_text"]
                payload["amount_display"] = amount_display
        else:
            # The legacy CTC slot may still contain a seconds-bearing value,
            # but v5's time-specific target deliberately excludes it.  Do not
            # drop the whole receipt or silently remove seconds here; the
            # contract count makes the omission auditable.
            time_aux = parse_time_aux_target(target)
            if time_aux is not None:
                payload["time_aux"] = time_aux
            time_display = parse_time_display_target(source_text)
            if time_display is not None:
                payload["visible_text"] = time_display["visible_text"]
                payload["time_display"] = time_display
        return payload
    if field == "transfer_status":
        if semantic_value not in STATUS_CLASSES:
            return None
        return {
            "image": str(record["image"]),
            "class_name": semantic_value,
            "paddle_text": record.get("paddle_text"),
            "paddle_confidence": record.get("paddle_confidence"),
            "detector_score": record.get("detector_score"),
            "crop_sha256": record.get("crop_sha256"),
        }
    if field == "payment_method_field":
        # Keep the visible value, not the broad semantic category.  A payment
        # row may contain the label itself; strip it conservatively so the CTC
        # target remains the value the business expects to show.
        source_text = clean_text(str(record["text"]))
        if record.get("label_source") == "transaction_truth" and source_text.casefold() in {
            "bank_card",
            "balance",
            "yuebao",
            "huabei",
            "other",
        }:
            # The older transaction-truth builder emits a normalised payment
            # category only.  That is valid for its legacy classifier but is
            # not visible text that a CTC reader can learn to reproduce.
            # Reject it rather than silently teaching the model English class
            # names in place of a bank/card value.
            return None
        text = extract_field_value(source_text, "payment_method")
        text = clean_text(text)
        if not text or any(not character.isprintable() for character in text):
            return None
        payload = {
            "image": str(record["image"]),
            "text": text,
            "semantic_value": semantic_value,
            "paddle_text": record.get("paddle_text"),
            "paddle_confidence": record.get("paddle_confidence"),
            "detector_score": record.get("detector_score"),
            "crop_sha256": record.get("crop_sha256"),
        }
        # The full visible text remains the CTC target.  Only an exact final
        # (ASCII-digit) card suffix is additionally split for a specialised
        # small tail head; malformed/unknown forms retain their CTC label.
        payment_card_tail = parse_payment_card_tail_target(text)
        if payment_card_tail is not None:
            payload["payment_card_tail"] = payment_card_tail
        payment_bank_prefix = parse_payment_bank_prefix_target(text)
        if payment_bank_prefix is not None:
            payload["payment_bank_prefix"] = payment_bank_prefix
        return payload
    raise AssertionError(field)


def _target_signature(field: str, slot: Mapping[str, object]) -> str:
    """Return the supervised target used to decide whether a duplicate conflicts."""
    if field == "transfer_status":
        return str(slot["class_name"])
    return str(slot["text"])


def build_unified_dataset(*, records_path: Path, output_dir: Path) -> dict[str, object]:
    """Create ``unified_fields.jsonl`` from flat Paddle/truth crop records.

    A unified record may contain fewer than four slots.  The training model
    uses a white placeholder for missing images and masks their losses.  This
    retains good labels instead of discarding an entire receipt because one
    field was below the teacher confidence threshold.
    """
    dataset_root, records, rejected = _read_flat_records(records_path)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, object]] = {}
    for record in records:
        receipt_key = _receipt_key(record)
        field = str(record["field"])
        slot = _slot_payload(record)
        if slot is None:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": str(record["id"]),
                    "field": field,
                    "reason": "invalid_unified_target",
                    "detail": str(record.get("semantic_value", "")),
                }
            )
            continue
        entry = grouped.setdefault(
            receipt_key,
            {
                "schema_version": SCHEMA_VERSION,
                "receipt_key": receipt_key,
                "group_id": str(record["group_id"]),
                "split": str(record["split"]),
                "source": record.get("source"),
                "result_json": record.get("result_json"),
                "label_source": record.get("label_source", "unspecified"),
                "slots": {},
                "_selected": {},
                "_ambiguous": set(),
            },
        )
        if entry["group_id"] != str(record["group_id"]) or entry["split"] != str(record["split"]):
            raise ValueError(
                f"Fields for one screenshot disagree on group/split: {receipt_key!r}; rebuild the source manifest"
            )
        selected = entry["_selected"]
        slots = entry["slots"]
        ambiguous = entry["_ambiguous"]
        if not isinstance(selected, dict) or not isinstance(slots, dict) or not isinstance(ambiguous, set):
            # Internal construction invariant.
            raise AssertionError("unified receipt entry has invalid slot storage")
        if field in ambiguous:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": str(record["id"]),
                    "field": field,
                    "reason": "duplicate_slot_already_ambiguous",
                    "receipt_key": receipt_key,
                }
            )
            continue
        previous = selected.get(field)
        if previous is not None:
            previous_slot = slots.get(field)
            if not isinstance(previous_slot, Mapping):
                raise AssertionError("selected duplicate slot has no payload")
            if _target_signature(field, slot) != _target_signature(field, previous_slot):
                # A high-confidence teacher conflict is still a conflict.  Do
                # not silently pick a winner and teach the student a possibly
                # wrong label; keep the other independent fields from this
                # receipt and send this slot to the review audit instead.
                slots.pop(field, None)
                selected.pop(field, None)
                ambiguous.add(field)
                for conflicting in (previous, record):
                    rejected.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "id": str(conflicting["id"]),
                            "field": field,
                            "reason": "ambiguous_duplicate_slot",
                            "receipt_key": receipt_key,
                        }
                    )
                continue
        if previous is not None and _selection_key(record) <= _selection_key(previous):
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": str(record["id"]),
                    "field": field,
                    "reason": "duplicate_slot_lower_confidence",
                    "receipt_key": receipt_key,
                }
            )
            continue
        if previous is not None:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": str(previous["id"]),
                    "field": field,
                    "reason": "duplicate_slot_replaced_by_higher_confidence",
                    "receipt_key": receipt_key,
                }
            )
        slots[field] = slot
        selected[field] = record

    unified_records: list[dict[str, object]] = []
    for receipt_key, entry in grouped.items():
        slots = dict(entry["slots"])
        if not slots:
            continue
        digest = hashlib.sha256(receipt_key.encode("utf-8")).hexdigest()
        entry.pop("_selected", None)
        ambiguous = entry.pop("_ambiguous", set())
        if ambiguous:
            entry["ambiguous_slots"] = sorted(ambiguous)
        entry["id"] = f"receipt-{digest[:24]}"
        entry["slot_order"] = list(SLOT_ORDER)
        entry["complete"] = all(field in slots for field in SLOT_ORDER)
        unified_records.append(entry)
    unified_records.sort(key=lambda value: str(value["id"]))
    if not unified_records:
        raise ValueError("No valid receipt-level records remain after target validation")

    slot_counts = Counter(
        field for record in unified_records for field in dict(record["slots"]).keys()
    )
    by_split = Counter(str(record["split"]) for record in unified_records)
    complete_by_split = Counter(
        str(record["split"]) for record in unified_records if bool(record["complete"])
    )
    ambiguous_slot_counts = Counter(
        field for record in unified_records for field in list(record.get("ambiguous_slots", []))
    )
    aux_name_by_slot = {
        "amount": ("amount_aux", "amount_display"),
        "time": ("time_aux", "time_display"),
        "payment_method_field": ("payment_card_tail", "payment_bank_prefix"),
    }
    structured_target_counts: dict[str, int] = {}
    structured_target_counts_by_split: dict[str, dict[str, int]] = {}
    for slot_name, aux_names in aux_name_by_slot.items():
        for aux_name in aux_names:
            structured_target_counts[aux_name] = sum(
                1
                for record in unified_records
                for slot in [dict(record["slots"]).get(slot_name)]
                if isinstance(slot, Mapping) and aux_name in slot
            )
            structured_target_counts_by_split[aux_name] = {
                split: sum(
                    1
                    for record in unified_records
                    if str(record["split"]) == split
                    for slot in [dict(record["slots"]).get(slot_name)]
                    if isinstance(slot, Mapping) and aux_name in slot
                )
                for split in ("train", "val", "test")
            }
            unparsed_name = f"{aux_name}_unparsed"
            structured_target_counts[unparsed_name] = int(slot_counts[slot_name]) - structured_target_counts[aux_name]
            structured_target_counts_by_split[unparsed_name] = {
                split: int(
                    sum(
                        1
                        for record in unified_records
                        if str(record["split"]) == split
                        for slot in [dict(record["slots"]).get(slot_name)]
                        if isinstance(slot, Mapping) and aux_name not in slot
                    )
                )
                for split in ("train", "val", "test")
            }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source_records": records_path.resolve().as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "slot_order": list(SLOT_ORDER),
        "status_classes": list(STATUS_CLASSES),
        "records": len(unified_records),
        "complete_records": sum(bool(record["complete"]) for record in unified_records),
        "slot_records": {field: int(slot_counts[field]) for field in SLOT_ORDER},
        "by_split": {split: int(by_split[split]) for split in ("train", "val", "test")},
        "complete_by_split": {split: int(complete_by_split[split]) for split in ("train", "val", "test")},
        "rejected_records": len(rejected),
        "ambiguous_slot_records": {field: int(ambiguous_slot_counts[field]) for field in SLOT_ORDER},
        "payment_target": "visible_payment_method_value",
        "structured_target_config": structured_target_config(),
        "structured_target_counts": structured_target_counts,
        "structured_target_counts_by_split": structured_target_counts_by_split,
        "missing_slot_policy": "white_placeholder_with_masked_loss_and_review_at_runtime",
        "warning": (
            "Paddle-derived records are teacher labels, not independent business truth. "
            "Do not claim production accuracy without a held-out teacher-parity and human-truth evaluation."
        ),
    }
    _atomic_write_jsonl(output_dir / "unified_fields.jsonl", unified_records)
    _atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)
    _atomic_write_json(output_dir / "dataset.contract.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build receipt-level manifests for the single-ONNX unified field reader"
    )
    parser.add_argument("--records", type=Path, required=True, help="Flat pseudo_labels.jsonl or transaction-truth JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New empty output directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = build_unified_dataset(records_path=args.records, output_dir=args.output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Unified OCR dataset build failed:\n{error}") from error
    slot_summary = ", ".join(f"{field}={count}" for field, count in dict(summary["slot_records"]).items())
    print(
        f"Wrote {summary['records']} unified receipt record(s) to {args.output} "
        f"(complete={summary['complete_records']}, {slot_summary})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
