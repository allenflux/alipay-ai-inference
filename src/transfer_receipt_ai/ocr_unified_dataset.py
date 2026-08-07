"""Build receipt-level manifests for the unified lightweight field reader.

The detection/pseudo-label pipeline stores one crop per field.  A unified
reader, however, receives the field crops from *one receipt* in a fixed order
and returns all field outputs in one ONNX invocation.  This module turns the
flat crop manifest into that receipt-level representation without copying any
images or importing Paddle/Torch.

The frozen v8 contract contains four slots:

``amount``, ``time``, ``transfer_status`` and ``payment_method_field``.

The v9 contract appends ``recipient_field`` as a fifth slot.  It is a distinct
dataset kind so an old four-slot model can never silently consume a five-slot
manifest.  Its recipient CTC alphabet is generated from *train only* and the
contract records validation/test OOV evidence rather than leaking those
characters into the deployable charset.

The v10 contract retains the same five input slots but changes only recipient
CTC supervision: it reads the entire visible crop line (for example
``收款方 商户甲``), then stores the extracted business value separately.  This
keeps the target geometrically aligned with a detector crop that includes the
left-side field label while preserving the value used by downstream review.

The v11 contract keeps five slots but makes the recipient data policy explicit:
only a row anchored by a recipient label and free of obvious neighbouring-row
pollution is eligible.  It stores the full visible row for audit and trains the
recipient target on the right-side value.  The paired reader contract applies a
documented value-view crop before its fifth input is resized; this is a new,
incompatible protocol rather than a silent change to v9/v10.

The v12 contract retains v11's anchored recipient labels, charset, and quality
audit verbatim.  It is nevertheless a distinct dataset kind because the paired
reader consumes that same value view through a dedicated high-resolution input
branch.  Keeping the manifest kind separate prevents a v11 artifact from
silently consuming a v12 training manifest.

The v13 contract adds a transfer-status text reader.  Its target is the visible
Chinese status line reported by Paddle OCR, not the adjacent pinyin guide and
not a semantic class name.  This keeps ``转账成功`` as pixel-grounded CTC text
while ``success`` remains a separately derived diagnostic value.

The payment slot deliberately retains the visible payment-method value (for
example ``建设银行储蓄卡(3667)``) as a CTC target.  Its normalised business
category remains provenance only; reducing it to ``bank_card`` would make a
small model unable to match the existing Paddle output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .ocr import clean_text, extract_field_value, normalize_status, parse_anchored_recipient_row
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
# ``KIND`` and ``SLOT_ORDER`` are the long-lived v8 aliases imported by the
# existing v3-v8 reader.  Do not repoint them to v9: old manifests/checkpoints
# must remain byte-for-byte compatible with the four-slot protocol.
KIND_V8 = "receipt_unified_field_dataset_v1"
KIND_V9 = "receipt_unified_field_dataset_v2"
KIND_V10 = "receipt_unified_field_dataset_v3"
KIND_V11 = "receipt_unified_field_dataset_v4"
KIND_V12 = "receipt_unified_field_dataset_v5"
KIND_V13 = "receipt_unified_field_dataset_v6"
KIND = KIND_V8
SLOT_ORDER = ("amount", "time", "transfer_status", "payment_method_field")
V9_SLOT_ORDER = (*SLOT_ORDER, "recipient_field")
V10_SLOT_ORDER = V9_SLOT_ORDER
V11_SLOT_ORDER = V9_SLOT_ORDER
V12_SLOT_ORDER = V11_SLOT_ORDER
V13_SLOT_ORDER = V12_SLOT_ORDER
ARCHITECTURE_V8 = "v8"
ARCHITECTURE_V9 = "v9"
ARCHITECTURE_V10 = "v10"
ARCHITECTURE_V11 = "v11"
ARCHITECTURE_V12 = "v12"
ARCHITECTURE_V13 = "v13"
STATUS_CLASSES = ("success", "pending", "failed")
STATUS_TEXT_TARGET = "visible_transfer_status_cjk_text"
STATUS_TEXT_CHARSET_SOURCE = "train_only_visible_transfer_status_cjk_text"
STATUS_VISIBLE_CJK_TEXTS = frozenset(
    {
        "转账成功",
        "交易成功",
        "付款成功",
        "支付成功",
        "转帐成功",
        "转账失败",
        "交易失败",
        "付款失败",
        "支付失败",
        "转帐失败",
        "转账未成功",
        "交易未成功",
        "付款未成功",
        "支付未成功",
        "转帐未成功",
        "转账已撤销",
        "交易已撤销",
        "付款已撤销",
        "支付已撤销",
        "转帐已撤销",
        "转账处理中",
        "交易处理中",
        "付款处理中",
        "支付处理中",
        "转帐处理中",
        "转账待处理",
        "交易待处理",
        "付款待处理",
        "支付待处理",
        "转帐待处理",
        "转账进行中",
        "交易进行中",
        "付款进行中",
        "支付进行中",
        "转帐进行中",
        "失败",
        "未成功",
        "已撤销",
        "处理中",
        "待处理",
        "进行中",
    }
)


def _is_cjk_ideograph(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def _visible_status_cjk_match(value: object) -> tuple[str, int]:
    """Return one exact visible status phrase and its retained match count.

    Paddle can return a pinyin guide, a neighbouring amount sentence, or the
    same Chinese status phrase twice.  A whole-string CJK projection would
    either reject valid text or join unrelated text into a made-up target.
    Instead, split on every non-CJK letter or digit, then search the remaining
    visible CJK streams for exact audited phrases.  Punctuation and whitespace
    may separate visible ideographs, but Latin text and digits are hard
    boundaries and can never be crossed.

    A shorter match is removed only when its span is contained in a longer
    match (for example ``未成功`` inside ``转账未成功``).  Repeated copies of the
    same phrase are deterministic; two distinct surviving phrases are
    ambiguous and return no target.  The semantic class never participates in
    extraction.
    """
    if not isinstance(value, str):
        return "", 0
    cleaned = unicodedata.normalize("NFC", clean_text(value))
    segments: list[str] = []
    current: list[str] = []
    for character in cleaned:
        if _is_cjk_ideograph(character):
            current.append(character)
        elif character.isalpha() or character.isdigit():
            if current:
                segments.append("".join(current))
                current = []
        else:
            # Whitespace and punctuation can appear between the visibly
            # separated ideographs of one OCR line.
            continue
    if current:
        segments.append("".join(current))

    matches: list[tuple[int, int, int, str]] = []
    for segment_index, segment in enumerate(segments):
        for phrase in STATUS_VISIBLE_CJK_TEXTS:
            start = segment.find(phrase)
            while start >= 0:
                end = start + len(phrase)
                # A success phrase directly negated in the same visible CJK
                # stream is not positive evidence (for example 未转账成功).
                if not (
                    normalize_status(phrase) == "success"
                    and start > 0
                    and segment[start - 1] in {"未", "不", "非"}
                ):
                    matches.append((segment_index, start, end, phrase))
                start = segment.find(phrase, start + 1)

    retained: list[tuple[int, int, int, str]] = []
    for match in matches:
        segment_index, start, end, _phrase = match
        contained = any(
            other_segment == segment_index
            and other_start <= start
            and end <= other_end
            and (other_end - other_start) > (end - start)
            for other_segment, other_start, other_end, _other_phrase in matches
        )
        if not contained:
            retained.append(match)

    phrases = {phrase for _segment, _start, _end, phrase in retained}
    if len(phrases) != 1:
        return "", len(retained)
    phrase = next(iter(phrases))
    return phrase, sum(match_phrase == phrase for *_span, match_phrase in retained)


def _visible_status_cjk_text(value: object) -> str:
    return _visible_status_cjk_match(value)[0]

# V12 deliberately reuses the fully-audited v11 recipient labels.  Only the
# reader-side pixels change: it receives a dedicated high-resolution value
# view, while this manifest continues to store the original crop and the
# anchored business-value target.
_ANCHORED_RECIPIENT_ARCHITECTURES = frozenset(
    (ARCHITECTURE_V11, ARCHITECTURE_V12, ARCHITECTURE_V13)
)

# These are deliberately narrow, high-signal markers of a detector crop that
# reached into the adjacent payment/balance row.  Merchant names remain open
# text: v11 does not use a bank/merchant allow-list.
RECIPIENT_POLLUTION_TOKENS = (
    "付款方式",
    "交易方式",
    "付款渠道",
    "支付方式",
    "账户余额",
    "￥",
    "¥",
)
RECIPIENT_LABELS = ("收款方", "收款人", "收款账户", "收款账号")
RECIPIENT_QUALITY_POLICY_VERSION = "anchored_value_right_crop_v1"


def _dataset_spec(architecture: str) -> tuple[str, str, tuple[str, ...]]:
    """Return the immutable manifest contract selected by an architecture.

    Keep this mapping intentionally closed.  A custom ordering might look
    harmless in a JSONL file, but it would move tensor channels and let a
    five-slot artifact be fed to an incompatible runtime.
    """
    if not isinstance(architecture, str):
        raise ValueError("architecture must be v8, v9, v10, v11, v12, or v13")
    normalized = architecture.strip().casefold()
    if normalized == ARCHITECTURE_V8:
        return ARCHITECTURE_V8, KIND_V8, SLOT_ORDER
    if normalized == ARCHITECTURE_V9:
        return ARCHITECTURE_V9, KIND_V9, V9_SLOT_ORDER
    if normalized == ARCHITECTURE_V10:
        return ARCHITECTURE_V10, KIND_V10, V10_SLOT_ORDER
    if normalized == ARCHITECTURE_V11:
        return ARCHITECTURE_V11, KIND_V11, V11_SLOT_ORDER
    if normalized == ARCHITECTURE_V12:
        return ARCHITECTURE_V12, KIND_V12, V12_SLOT_ORDER
    if normalized == ARCHITECTURE_V13:
        return ARCHITECTURE_V13, KIND_V13, V13_SLOT_ORDER
    raise ValueError("architecture must be v8, v9, v10, v11, v12, or v13")


def slot_order_for_architecture(architecture: str) -> tuple[str, ...]:
    """Return the fixed input channel order for a supported dataset contract."""
    return _dataset_spec(architecture)[2]


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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
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


def _read_flat_records(
    records_path: Path,
    *,
    slot_order: tuple[str, ...],
) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
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
                if not isinstance(field, str) or field not in slot_order:
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
        raise ValueError("No valid unified-reader slot records found")
    return dataset_root, accepted, rejected


def _recipient_crop_aspect(record: Mapping[str, object]) -> float | None:
    """Return the source crop width/height only for an opted-in geometry gate.

    The normal v11 path must remain cheap for a 120k manifest: it does not
    open every image merely to construct a JSONL file.  When a caller elects a
    non-zero aspect threshold, inspect the source crop and make an unavailable
    geometry value visible in the audit rather than guessing.
    """
    image_path = record.get("image_path")
    if not isinstance(image_path, Path):
        return None
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except (ImportError, OSError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return float(width) / float(height)


def _recipient_quality_policy_payload(
    *,
    min_crop_aspect: float,
    max_visible_chars: int,
) -> dict[str, object]:
    """Return the frozen v11/v12 recipient-label acceptance policy."""
    return {
        "version": RECIPIENT_QUALITY_POLICY_VERSION,
        "requires_leading_recipient_label": True,
        "rejects_repeated_recipient_label": True,
        "rejects_context_tokens": list(RECIPIENT_POLLUTION_TOKENS),
        "min_crop_aspect": min_crop_aspect,
        "max_visible_chars": max_visible_chars,
        "geometry_disabled_when_zero": True,
        "target": "anchored_recipient_value",
        "input_preprocess": "recipient_value_right_crop_configured_by_reader_contract",
    }


def _recipient_v11_slot_payload(
    record: Mapping[str, object],
    *,
    min_crop_aspect: float,
    max_visible_chars: int,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Create one v11/v12 recipient slot and an auditable quality decision.

    The old v9/v10 contracts deliberately accept a permissive recipient
    extraction because they preserve legacy training data.  V11/V12 are
    opt-in: this strict policy is intended to remove examples where the
    detector crop contains a neighbouring payment/balance line or a recipient
    label in an unsupported position.  A rejected fifth slot does *not* reject
    the other fields from that receipt.
    """
    source_text = clean_text(str(record["text"]))
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_recipient_quality_audit_v1",
        "policy_version": RECIPIENT_QUALITY_POLICY_VERSION,
        "id": str(record["id"]),
        "field": "recipient_field",
        "split": str(record["split"]),
        "group_id": str(record["group_id"]),
        "image": str(record["image"]),
        "source_text": source_text,
        "source_bbox_rectified": record.get("bbox_rectified"),
        "paddle_text": record.get("paddle_text"),
        "paddle_confidence": record.get("paddle_confidence"),
        "detector_score": record.get("detector_score"),
        "crop_sha256": record.get("crop_sha256"),
        "quality_decision": "rejected",
        "quality_reason": None,
        "recipient_label": None,
        "recipient_value": None,
        "observed_crop_aspect": None,
        "retained_in_unified_manifest": False,
    }

    parsed = parse_anchored_recipient_row(source_text)
    if parsed is None:
        audit["quality_reason"] = "missing_leading_recipient_label_or_value"
        return None, audit
    recipient_label, recipient_value = parsed
    audit["recipient_label"] = recipient_label
    audit["recipient_value"] = recipient_value

    if any(label in recipient_value for label in RECIPIENT_LABELS):
        audit["quality_reason"] = "repeated_or_malformed_recipient_label"
        return None, audit
    pollution = next((token for token in RECIPIENT_POLLUTION_TOKENS if token in recipient_value), None)
    if pollution is not None:
        audit["quality_reason"] = f"context_or_currency_pollution:{pollution}"
        return None, audit
    if any(not character.isprintable() for character in recipient_value):
        audit["quality_reason"] = "non_printable_recipient_value"
        return None, audit
    if max_visible_chars > 0 and len(source_text) > max_visible_chars:
        audit["quality_reason"] = "visible_row_exceeds_configured_max_chars"
        return None, audit
    if min_crop_aspect > 0.0:
        aspect = _recipient_crop_aspect(record)
        audit["observed_crop_aspect"] = aspect
        if aspect is None:
            audit["quality_reason"] = "crop_aspect_unavailable"
            return None, audit
        if aspect < min_crop_aspect:
            audit["quality_reason"] = "crop_aspect_below_configured_min"
            return None, audit

    quality_metadata = {
        "policy_version": RECIPIENT_QUALITY_POLICY_VERSION,
        "anchored_label": recipient_label,
        "visible_text": source_text,
        "value": recipient_value,
        "source_bbox_rectified": record.get("bbox_rectified"),
        "input_preprocess": "recipient_value_right_crop_configured_by_reader_contract",
    }
    audit["quality_decision"] = "accepted"
    audit["quality_reason"] = "accepted"
    return (
        {
            "image": str(record["image"]),
            # V11/V12 deliberately make both pixels and target value-aligned:
            # the paired reader crops the static left label region before
            # resizing its recipient view.  The unmodified row remains
            # provenance only.
            "text": recipient_value,
            "recipient_visible_text": source_text,
            "recipient_value": recipient_value,
            "recipient_label": recipient_label,
            "recipient_quality_policy": RECIPIENT_QUALITY_POLICY_VERSION,
            "recipient_quality": quality_metadata,
            "source_record_id": str(record["id"]),
            # Preserve the source geometry under its original field name so
            # downstream diagnostics and a .NET crop implementation do not
            # need to know about the flat-manifest source record.
            "bbox_rectified": record.get("bbox_rectified"),
            "semantic_value": recipient_value,
            "paddle_text": record.get("paddle_text"),
            "paddle_confidence": record.get("paddle_confidence"),
            "detector_score": record.get("detector_score"),
            "crop_sha256": record.get("crop_sha256"),
        },
        audit,
    )


def _slot_payload(
    record: Mapping[str, object],
    *,
    architecture: str,
) -> dict[str, object] | None:
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
        payload = {
            "image": str(record["image"]),
            "class_name": semantic_value,
            "paddle_text": record.get("paddle_text"),
            "paddle_confidence": record.get("paddle_confidence"),
            "detector_score": record.get("detector_score"),
            "crop_sha256": record.get("crop_sha256"),
        }
        if architecture == ARCHITECTURE_V13:
            audit: dict[str, object] = {
                "target": STATUS_TEXT_TARGET,
                "decision": "missing",
                "source": None,
                "reason": None,
                "semantic_value": semantic_value,
                "record_text": (
                    clean_text(record["text"]) if isinstance(record.get("text"), str) else ""
                ),
                "paddle_text": (
                    clean_text(record["paddle_text"])
                    if isinstance(record.get("paddle_text"), str)
                    else ""
                ),
            }
            # Paddle's OCR value is the only permitted v13 text truth.  The
            # record text may be a semantic label or a legacy transcription,
            # so it remains audit evidence and can never override Paddle.
            raw_text = record.get("paddle_text")
            visible_text, visible_match_count = _visible_status_cjk_match(raw_text)
            audit["visible_phrase_match_count"] = visible_match_count
            if not isinstance(raw_text, str):
                audit["reason"] = "paddle_text:not_a_string"
            elif not any(_is_cjk_ideograph(character) for character in raw_text):
                audit["reason"] = "paddle_text:no_visible_cjk_status_text"
            elif not visible_text or visible_text not in STATUS_VISIBLE_CJK_TEXTS:
                audit["reason"] = "paddle_text:unsupported_visible_cjk_status_text"
            else:
                normalized = normalize_status(visible_text)
                if normalized != semantic_value:
                    audit["reason"] = (
                        f"paddle_text_cjk:normalizes_to_{normalized}_not_{semantic_value}"
                    )
                else:
                    source = "paddle_text_cjk"
                    payload["text"] = visible_text
                    payload["status_visible_text"] = visible_text
                    payload["semantic_value"] = semantic_value
                    payload["status_text_source"] = source
                    audit.update(
                        {
                            "decision": "accepted",
                            "source": source,
                            "reason": "paddle_visible_cjk_text_normalizes_to_class_name",
                        }
                    )
            payload["status_text_audit"] = audit
        return payload
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
    if field == "recipient_field":
        # v9 keeps the recipient/business value as free-text CTC supervision.
        # v10 instead keeps the entire visible crop line as the CTC target,
        # because detector crops often contain a left-side label such as
        # ``收款方``.  Both variants retain the extracted business value for
        # semantic comparison and downstream review; neither turns merchant
        # names into a finite classifier.
        source_text = clean_text(str(record["text"]))
        recipient_value = clean_text(extract_field_value(source_text, "recipient"))
        if recipient_value in {"收款方", "收款人", "收款账户", "收款账号"}:
            # A row consisting of the label alone is not a readable recipient
            # value.  Do not teach that label as a merchant name.
            return None
        if not recipient_value or any(not character.isprintable() for character in recipient_value):
            return None
        if architecture == ARCHITECTURE_V10:
            if any(not character.isprintable() for character in source_text):
                return None
            return {
                "image": str(record["image"]),
                # ``text`` stays the canonical CTC target for all slot
                # payloads.  For v10 it is deliberately the complete visible
                # line, not just the right-side recipient value.
                "text": source_text,
                "recipient_visible_text": source_text,
                "recipient_value": recipient_value,
                "semantic_value": recipient_value,
                "paddle_text": record.get("paddle_text"),
                "paddle_confidence": record.get("paddle_confidence"),
                "detector_score": record.get("detector_score"),
                "crop_sha256": record.get("crop_sha256"),
            }
        return {
            "image": str(record["image"]),
            "text": recipient_value,
            "semantic_value": semantic_value,
            "paddle_text": record.get("paddle_text"),
            "paddle_confidence": record.get("paddle_confidence"),
            "detector_score": record.get("detector_score"),
            "crop_sha256": record.get("crop_sha256"),
        }
    raise AssertionError(field)


def _target_signature(field: str, slot: Mapping[str, object]) -> str:
    """Return the supervised target used to decide whether a duplicate conflicts."""
    if field == "transfer_status":
        # v13 adds visible-text supervision.  Two duplicate crops that agree
        # only on the broad semantic class but disagree on their OCR target
        # are not interchangeable training labels.
        return json.dumps(
            [slot["class_name"], slot.get("text")],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return str(slot["text"])


def _recipient_charset_payload(
    records: Iterable[Mapping[str, object]],
    *,
    source: str,
) -> dict[str, object]:
    """Build a v9/v10 recipient alphabet from train labels and audit held-out OOVs.

    The recipient CTC output is deliberately open text, not a finite merchant
    catalog.  Its vocabulary must therefore be frozen from train only.  A
    validation/test character cannot be silently appended just because it is
    available in the manifest: that would make held-out teacher parity look
    better than the deployable artifact can actually achieve.
    """
    stable_records = list(records)
    charset = sorted(
        {
            glyph
            for record in stable_records
            if str(record["split"]) == "train"
            for slot in [dict(record["slots"]).get("recipient_field")]
            if isinstance(slot, Mapping)
            for text in [slot.get("text")]
            if isinstance(text, str)
            for glyph in text
        }
    )
    known = set(charset)
    counters: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    examples: dict[str, list[dict[str, str]]] = {split: [] for split in ("train", "val", "test")}
    for record in stable_records:
        split = str(record["split"])
        slot = dict(record["slots"]).get("recipient_field")
        if not isinstance(slot, Mapping):
            continue
        text = slot.get("text")
        if not isinstance(text, str):
            raise AssertionError("recipient slot has a non-string CTC target")
        counters[split]["records"] += 1
        unknown = sorted(set(text) - known)
        if unknown:
            counters[split]["oov_records"] += 1
            counters[split]["oov_characters"] += len(unknown)
            if len(examples[split]) < 20:
                examples[split].append(
                    {
                        "id": str(record["id"]),
                        "characters": "".join(unknown),
                        "text": text,
                    }
                )
    return {
        "source": source,
        "characters": charset,
        "sha256": hashlib.sha256("".join(charset).encode("utf-8")).hexdigest(),
        "oov_by_split": {
            split: {
                "records": int(counters[split]["records"]),
                "oov_records": int(counters[split]["oov_records"]),
                "oov_characters": int(counters[split]["oov_characters"]),
                "examples": examples[split],
            }
            for split in ("train", "val", "test")
        },
    }


def _status_text_charset_payload(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Freeze v13's visible-status alphabet from train labels only.

    Held-out Unicode is recorded as OOV evidence and never appended to the
    deployable map.  Slots without safe visible text remain useful legacy
    status-class provenance but are excluded from CTC supervision.
    """
    stable_records = list(records)
    charset = sorted(
        {
            glyph
            for record in stable_records
            if str(record["split"]) == "train"
            for slot in [dict(record["slots"]).get("transfer_status")]
            if isinstance(slot, Mapping)
            for text in [slot.get("text")]
            if isinstance(text, str)
            for glyph in text
        }
    )
    known = set(charset)
    counters: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    source_counts: Counter[str] = Counter()
    missing_reasons: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {split: [] for split in ("train", "val", "test")}
    for record in stable_records:
        split = str(record["split"])
        slot = dict(record["slots"]).get("transfer_status")
        if not isinstance(slot, Mapping):
            continue
        audit = slot.get("status_text_audit")
        if isinstance(audit, Mapping):
            if audit.get("decision") == "accepted":
                source_counts[str(audit.get("source", "unspecified"))] += 1
            else:
                missing_reasons[str(audit.get("reason", "unspecified"))] += 1
        text = slot.get("text")
        if not isinstance(text, str):
            counters[split]["missing_text_records"] += 1
            continue
        counters[split]["records"] += 1
        unknown = sorted(set(text) - known)
        if unknown:
            counters[split]["oov_records"] += 1
            counters[split]["oov_characters"] += len(unknown)
            if len(examples[split]) < 20:
                examples[split].append(
                    {
                        "id": str(record["id"]),
                        "characters": "".join(unknown),
                        "text": text,
                    }
                )
    return {
        "source": STATUS_TEXT_CHARSET_SOURCE,
        "characters": charset,
        "sha256": hashlib.sha256("".join(charset).encode("utf-8")).hexdigest(),
        "target": STATUS_TEXT_TARGET,
        "source_counts": dict(sorted(source_counts.items())),
        "missing_reasons": dict(sorted(missing_reasons.items())),
        "oov_by_split": {
            split: {
                "records": int(counters[split]["records"]),
                "missing_text_records": int(counters[split]["missing_text_records"]),
                "oov_records": int(counters[split]["oov_records"]),
                "oov_characters": int(counters[split]["oov_characters"]),
                "examples": examples[split],
            }
            for split in ("train", "val", "test")
        },
    }


def build_unified_dataset(
    *,
    records_path: Path,
    output_dir: Path,
    architecture: str = ARCHITECTURE_V8,
    recipient_min_crop_aspect: float = 0.0,
    recipient_max_visible_chars: int = 0,
) -> dict[str, object]:
    """Create ``unified_fields.jsonl`` from flat Paddle/truth crop records.

    The default ``v8`` keeps the established four-slot protocol.  Select
    ``v9`` only with a flat manifest containing all five field labels from one
    pseudo-label export; its fifth channel is ``recipient_field``.  ``v11``
    is the strict successor for recipient data: it requires a leading
    recipient label and records every accepted/rejected source crop in a
    sidecar audit.  ``v12`` reuses that exact anchored-label policy for the
    paired reader's dedicated high-resolution recipient input.  A unified
    record may omit a slot.  The training model uses a white placeholder for
    missing images and masks its loss, preserving good labels rather than
    discarding an entire receipt because one field was below the teacher
    confidence threshold.
    """
    architecture, dataset_kind, slot_order = _dataset_spec(architecture)
    try:
        recipient_min_crop_aspect = float(recipient_min_crop_aspect)
    except (TypeError, ValueError):
        raise ValueError("recipient_min_crop_aspect must be a finite number >= 0") from None
    if not math.isfinite(recipient_min_crop_aspect) or recipient_min_crop_aspect < 0.0:
        raise ValueError("recipient_min_crop_aspect must be >= 0")
    if isinstance(recipient_max_visible_chars, bool) or not isinstance(recipient_max_visible_chars, int):
        raise ValueError("recipient_max_visible_chars must be an integer >= 0")
    if recipient_max_visible_chars < 0:
        raise ValueError("recipient_max_visible_chars must be >= 0")
    if architecture not in _ANCHORED_RECIPIENT_ARCHITECTURES and (
        recipient_min_crop_aspect != 0.0 or recipient_max_visible_chars != 0
    ):
        raise ValueError("recipient geometry options are supported only by architecture v11, v12, or v13")
    dataset_root, records, rejected = _read_flat_records(records_path, slot_order=slot_order)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, object]] = {}
    recipient_audits: list[dict[str, object]] = []
    for record in records:
        receipt_key = _receipt_key(record)
        field = str(record["field"])
        recipient_audit: dict[str, object] | None = None
        if architecture in _ANCHORED_RECIPIENT_ARCHITECTURES and field == "recipient_field":
            slot, recipient_audit = _recipient_v11_slot_payload(
                record,
                min_crop_aspect=recipient_min_crop_aspect,
                max_visible_chars=recipient_max_visible_chars,
            )
            recipient_audits.append(recipient_audit)
        else:
            slot = _slot_payload(record, architecture=architecture)
        if slot is None:
            reason = "invalid_unified_target"
            detail = str(record.get("semantic_value", ""))
            if recipient_audit is not None:
                reason = "recipient_quality_rejected"
                detail = str(recipient_audit.get("quality_reason", "unspecified"))
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": str(record["id"]),
                    "field": field,
                    "reason": reason,
                    "detail": detail,
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
    retained_recipient_ids: set[str] = set()
    for receipt_key, entry in grouped.items():
        slots = dict(entry["slots"])
        if not slots:
            continue
        digest = hashlib.sha256(receipt_key.encode("utf-8")).hexdigest()
        selected = entry.pop("_selected", None)
        if architecture in _ANCHORED_RECIPIENT_ARCHITECTURES and isinstance(selected, Mapping):
            selected_recipient = selected.get("recipient_field")
            if isinstance(selected_recipient, Mapping):
                selected_id = selected_recipient.get("id")
                if isinstance(selected_id, str):
                    retained_recipient_ids.add(selected_id)
        ambiguous = entry.pop("_ambiguous", set())
        if ambiguous:
            entry["ambiguous_slots"] = sorted(ambiguous)
        entry["id"] = f"receipt-{digest[:24]}"
        entry["slot_order"] = list(slot_order)
        entry["complete"] = all(field in slots for field in slot_order)
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
        "kind": dataset_kind,
        "source_records": records_path.resolve().as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "slot_order": list(slot_order),
        "status_classes": list(STATUS_CLASSES),
        "records": len(unified_records),
        "complete_records": sum(bool(record["complete"]) for record in unified_records),
        "slot_records": {field: int(slot_counts[field]) for field in slot_order},
        "by_split": {split: int(by_split[split]) for split in ("train", "val", "test")},
        "complete_by_split": {split: int(complete_by_split[split]) for split in ("train", "val", "test")},
        "rejected_records": len(rejected),
        "ambiguous_slot_records": {field: int(ambiguous_slot_counts[field]) for field in slot_order},
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
    recipient_charset_characters: list[str] | None = None
    if architecture in {ARCHITECTURE_V9, ARCHITECTURE_V10} | _ANCHORED_RECIPIENT_ARCHITECTURES:
        if architecture == ARCHITECTURE_V10:
            recipient_charset_source = "train_only_visible_recipient_line"
        elif architecture in _ANCHORED_RECIPIENT_ARCHITECTURES:
            recipient_charset_source = "train_only_anchored_recipient_value"
        else:
            recipient_charset_source = "train_only_visible_recipient_text"
        recipient_charset = _recipient_charset_payload(
            unified_records,
            source=recipient_charset_source,
        )
        # Keep the full audit in the contract and a tiny deterministic text
        # sidecar for training scripts that accept a character list directly.
        # The newline is a file terminator, not a trainable character.
        # Keep a v8 contract byte-for-byte shaped like the established one;
        # v9/v10 are separate kinds and carry explicit architecture markers.
        summary["architecture"] = architecture
        summary["recipient_target"] = {
            ARCHITECTURE_V9: "visible_recipient_value",
            ARCHITECTURE_V10: "visible_recipient_line_then_extract_value",
            ARCHITECTURE_V11: "anchored_recipient_value_with_value_view_crop",
            ARCHITECTURE_V12: "anchored_recipient_value_with_dedicated_high_resolution_value_view",
            ARCHITECTURE_V13: "anchored_recipient_value_with_dedicated_high_resolution_value_view",
        }[architecture]
        recipient_charset_characters = list(recipient_charset["characters"])
        summary["recipient_charset"] = recipient_charset_characters
        summary["recipient_charset_sha256"] = recipient_charset["sha256"]
        summary["recipient_charset_source"] = recipient_charset["source"]
        summary["recipient_oov_by_split"] = recipient_charset["oov_by_split"]
    if architecture in _ANCHORED_RECIPIENT_ARCHITECTURES:
        for audit in recipient_audits:
            if audit["quality_decision"] == "accepted":
                retained = str(audit["id"]) in retained_recipient_ids
                audit["retained_in_unified_manifest"] = retained
                audit["manifest_decision"] = "selected" if retained else "not_selected_after_duplicate_resolution"
            else:
                audit["manifest_decision"] = "quality_rejected"
        quality_by_split: dict[str, dict[str, int]] = {}
        for split in ("train", "val", "test"):
            split_audits = [audit for audit in recipient_audits if audit["split"] == split]
            quality_by_split[split] = {
                "source_records": len(split_audits),
                "quality_accepted": sum(audit["quality_decision"] == "accepted" for audit in split_audits),
                "quality_rejected": sum(audit["quality_decision"] == "rejected" for audit in split_audits),
                "retained_slot_records": sum(bool(audit["retained_in_unified_manifest"]) for audit in split_audits),
            }
        rejection_counts = Counter(
            str(audit["quality_reason"])
            for audit in recipient_audits
            if audit["quality_decision"] == "rejected"
        )
        summary["recipient_quality_policy"] = _recipient_quality_policy_payload(
            min_crop_aspect=recipient_min_crop_aspect,
            max_visible_chars=recipient_max_visible_chars,
        )
        summary["recipient_quality_audit"] = {
            "path": "recipient_quality_audit.jsonl",
            "source_records": len(recipient_audits),
            "quality_accepted": sum(audit["quality_decision"] == "accepted" for audit in recipient_audits),
            "quality_rejected": sum(audit["quality_decision"] == "rejected" for audit in recipient_audits),
            "retained_slot_records": sum(bool(audit["retained_in_unified_manifest"]) for audit in recipient_audits),
            "rejected_by_reason": dict(sorted(rejection_counts.items())),
            "by_split": quality_by_split,
            "scope": "valid flat recipient_field records read from the source manifest",
        }
    status_text_charset_characters: list[str] | None = None
    if architecture == ARCHITECTURE_V13:
        status_text_charset = _status_text_charset_payload(unified_records)
        status_text_charset_characters = list(status_text_charset["characters"])
        if not status_text_charset_characters:
            raise ValueError(
                "No safe visible transfer-status text remains in the train split for v13 CTC supervision"
            )
        summary["status_text_target"] = status_text_charset["target"]
        summary["status_text_charset"] = status_text_charset_characters
        summary["status_text_charset_sha256"] = status_text_charset["sha256"]
        summary["status_text_charset_source"] = status_text_charset["source"]
        summary["status_text_source_counts"] = status_text_charset["source_counts"]
        summary["status_text_missing_reasons"] = status_text_charset["missing_reasons"]
        summary["status_text_oov_by_split"] = status_text_charset["oov_by_split"]

    _atomic_write_jsonl(output_dir / "unified_fields.jsonl", unified_records)
    _atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)
    if architecture in _ANCHORED_RECIPIENT_ARCHITECTURES:
        _atomic_write_jsonl(output_dir / "recipient_quality_audit.jsonl", recipient_audits)
    if architecture in {ARCHITECTURE_V9, ARCHITECTURE_V10} | _ANCHORED_RECIPIENT_ARCHITECTURES:
        if recipient_charset_characters is None:  # Internal construction invariant.
            raise AssertionError("v9/v10/v11/v12 recipient charset was not initialized")
        _atomic_write_text(
            output_dir / "recipient_charset.txt",
            "".join(recipient_charset_characters) + "\n",
        )
    if architecture == ARCHITECTURE_V13:
        if status_text_charset_characters is None:
            raise AssertionError("v13 status-text charset was not initialized")
        _atomic_write_text(
            output_dir / "status_text_charset.txt",
            "".join(status_text_charset_characters) + "\n",
        )
    _atomic_write_json(output_dir / "dataset.contract.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build receipt-level manifests for the single-ONNX unified field reader"
    )
    parser.add_argument("--records", type=Path, required=True, help="Flat pseudo_labels.jsonl or transaction-truth JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New empty output directory")
    parser.add_argument(
        "--architecture",
        choices=(
            ARCHITECTURE_V8,
            ARCHITECTURE_V9,
            ARCHITECTURE_V10,
            ARCHITECTURE_V11,
            ARCHITECTURE_V12,
            ARCHITECTURE_V13,
        ),
        default=ARCHITECTURE_V8,
        help=(
            "v8 keeps four slots; v9 appends recipient_field with a value-only CTC target; "
            "v10 keeps five slots but trains recipient CTC on the visible full line; "
            "v11 filters recipient rows to an anchored clean value-view contract; "
            "v12 reuses v11's anchored labels for a dedicated high-resolution recipient view; "
            "v13 adds visible transfer-status CTC supervision without changing the five input slots"
        ),
    )
    parser.add_argument(
        "--recipient-min-crop-aspect",
        type=float,
        default=0.0,
        help=(
            "v11/v12 only: reject recipient crops whose width/height is below this value; "
            "0 disables the optional geometry gate"
        ),
    )
    parser.add_argument(
        "--recipient-max-visible-chars",
        type=int,
        default=0,
        help=(
            "v11/v12 only: reject recipient rows longer than this cleaned visible-text length; "
            "0 disables the optional length gate"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = build_unified_dataset(
            records_path=args.records,
            output_dir=args.output,
            architecture=args.architecture,
            recipient_min_crop_aspect=args.recipient_min_crop_aspect,
            recipient_max_visible_chars=args.recipient_max_visible_chars,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Unified OCR dataset build failed:\n{error}") from error
    slot_summary = ", ".join(f"{field}={count}" for field, count in dict(summary["slot_records"]).items())
    print(
        f"Wrote {summary['records']} unified receipt record(s) to {args.output} "
        f"(complete={summary['complete_records']}, {slot_summary})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
