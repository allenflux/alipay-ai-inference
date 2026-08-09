"""Train, export and evaluate one ONNX reader for receipt fields.

The model intentionally has one shared visual encoder and one ONNX artifact,
while retaining specialised heads where the output spaces differ:

* amount/time: independent readers.  v5 adds fixed-position digit heads
  beside the CTC readers; v6 keeps visible-format CTC and verifier paths
  separate, v7 shares the time CTC state with its format heads, and v8 keeps
  canonical amount digits in CTC while learning a tiny visible-format grammar;
* payment method: a raw CTC fallback plus a visible prefix, a finite known-bank
  classifier, and exact four-digit card-tail readers; and
* transfer status: a legacy finite three-class head plus v13's additive
  visible-Chinese CTC reader; and
* v9/v10/v11 recipient: a dedicated free-text Chinese CTC reader.  v10 learns
  the complete visible detector row; v11 learns an anchored right-side value
  after a frozen left title crop, so each target matches its input pixels.
  v12 keeps the single reader/session contract but moves that fifth CTC path
  onto a small, dedicated high-resolution value-view input.

That is materially different from putting all Chinese payment characters and
numeric characters in one CTC vocabulary: the latter makes the financial
fields compete with a much larger alphabet.  The exported wrapper consumes
fixed-order crops in one call, so deployment needs one ORT session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from .onnx_runtime import _preload_cuda_dlls, onnx_providers
from .ocr import (
    clean_text,
    extract_field_value,
    normalize_payment_method,
    normalize_status,
    parse_anchored_recipient_row,
)
from .recipient_beam import CharacterNGramLanguageModel, decode_ctc_prefix_beam
from .recipient_audit import (
    DEFAULT_CUT_RADIUS as RECIPIENT_AUDIT_DEFAULT_CUT_RADIUS,
    DEFAULT_FOREGROUND_CONTRAST_THRESHOLD as RECIPIENT_AUDIT_DEFAULT_FOREGROUND_CONTRAST_THRESHOLD,
    audit_recipient_pixels,
)
from .ocr_unified_dataset import KIND as DATASET_KIND_V8
from .ocr_unified_dataset import KIND_V9 as DATASET_KIND_V9
from .ocr_unified_dataset import KIND_V10 as DATASET_KIND_V10
from .ocr_unified_dataset import KIND_V11 as DATASET_KIND_V11
from .ocr_unified_dataset import KIND_V12 as DATASET_KIND_V12
from .ocr_unified_dataset import KIND_V13 as DATASET_KIND_V13
from .ocr_unified_dataset import (
    SLOT_ORDER,
    STATUS_CLASSES,
    STATUS_TEXT_CHARSET_SOURCE,
    STATUS_TEXT_TARGET,
    STATUS_VISIBLE_CJK_TEXTS,
    V9_SLOT_ORDER,
    _is_cjk_ideograph,
)
from .ocr_unified_targets import (
    AMOUNT_AUX_FORMAT,
    AMOUNT_CURRENCY_STYLE_CLASSES,
    AMOUNT_DISPLAY_AUX_FORMAT,
    AMOUNT_GROUPED_THOUSANDS_CLASSES,
    AMOUNT_MAX_INTEGER_DIGITS as TARGET_AMOUNT_MAX_INTEGER_DIGITS,
    AMOUNT_SIGN_CLASSES,
    AMOUNT_SIGN_POSITION_CLASSES,
    AMOUNT_VISIBLE_FORMAT_V8,
    PAYMENT_CARD_TAIL_FORMAT,
    PAYMENT_BANK_PREFIX_FORMAT,
    PARENTHESIS_STYLE_ASCII,
    PARENTHESIS_STYLE_FULLWIDTH,
    TIME_AUX_FORMAT,
    TIME_DISPLAY_AUX_FORMAT,
    TIME_DISPLAY_DIGIT_SLOTS,
    TIME_DISPLAY_FORMAT_CLASSES,
    is_structured_target,
    parse_amount_display_target,
    parse_amount_visible_format_target,
    parse_time_display_target,
    render_amount_visible_format,
    recompose_payment_card_tail_target,
)


SCHEMA_VERSION = 1
CHECKPOINT_SELECTION_BALANCED = "balanced"
CHECKPOINT_SELECTION_RECIPIENT_PRIORITY = "recipient_priority"
CHECKPOINT_SELECTION_MODES = frozenset(
    (CHECKPOINT_SELECTION_BALANCED, CHECKPOINT_SELECTION_RECIPIENT_PRIORITY)
)
INIT_CHECKPOINT_MODE_STRICT = "strict"
INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION = "recipient_only_expansion"
INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION = "recipient_input_width_expansion"
INIT_CHECKPOINT_MODE_RECIPIENT_CAPACITY_REINIT = "recipient_capacity_reinit"
INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER = "recipient_open_text_adapter"
INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT = "recipient_visual_context_reinit"
INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART = "recipient_full_crop_warmstart"
INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION = "recipient_full_crop_continuation"
FULL_CROP_CONTINUATION_AUTHORITY_KEY = "full_crop_continuation_authority"
INIT_CHECKPOINT_MODES = frozenset(
    (
        INIT_CHECKPOINT_MODE_STRICT,
        INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
        INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION,
        INIT_CHECKPOINT_MODE_RECIPIENT_CAPACITY_REINIT,
        INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER,
        INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    )
)
RECIPIENT_ONLY_INIT_CHECKPOINT_MODES = frozenset(
    (
        INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
        INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION,
        INIT_CHECKPOINT_MODE_RECIPIENT_CAPACITY_REINIT,
        INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER,
        INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    )
)
V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES = frozenset(
    (
        INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    )
)


def _has_analysis_only_full_crop_continuation_lineage(
    payload: Mapping[str, object],
) -> bool:
    """Recognize both the authorized source and every B8 child checkpoint."""

    if FULL_CROP_CONTINUATION_AUTHORITY_KEY in payload:
        return True
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping):
        return False
    return (
        initialization.get("init_checkpoint_mode")
        == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
        or isinstance(
            initialization.get("source_full_crop_continuation_authority"), Mapping
        )
    )


# These are the mature text heads that must not silently regress while a
# recipient-focused experiment chooses its checkpoint.  The values are supplied
# by the caller from a baseline measured on the same validation split.
CHECKPOINT_SELECTION_PROTECTED_FIELDS = ("amount", "time", "payment_method_field")
KIND_V3 = "receipt_unified_field_reader_v3"
KIND_V4 = "receipt_unified_field_reader_v4"
KIND_V5 = "receipt_unified_field_reader_v5"
KIND_V6 = "receipt_unified_field_reader_v6"
KIND_V7 = "receipt_unified_field_reader_v7"
KIND_V8 = "receipt_unified_field_reader_v8"
KIND_V9 = "receipt_unified_field_reader_v9"
KIND_V10 = "receipt_unified_field_reader_v10"
KIND_V11 = "receipt_unified_field_reader_v11"
KIND_V12 = "receipt_unified_field_reader_v12"
KIND_V13 = "receipt_unified_field_reader_v13"
# Keep the public/default alias on the established four-slot protocol.
# Five-slot v9 is deliberately opt-in: callers must select ``architecture=v9``
# rather than silently changing an existing v8 training or deployment path.
# Loading/export code uses SUPPORTED_KINDS so every published version remains
# independently loadable.
KIND = KIND_V8
SUPPORTED_KINDS = frozenset(
    (
        KIND_V3,
        KIND_V4,
        KIND_V5,
        KIND_V6,
        KIND_V7,
        KIND_V8,
        KIND_V9,
        KIND_V10,
        KIND_V11,
        KIND_V12,
        KIND_V13,
    )
)
# Kept as the frozen v3-v5 shared charset.  Do not append v6 symbols here:
# old ONNX sidecars/checkpoints must remain loadable byte-for-byte.
NUMERIC_CHARACTERS = tuple("0123456789.:")
V6_AMOUNT_CHARACTERS = tuple("0123456789.,-¥￥ ")
V6_TIME_CHARACTERS = tuple("0123456789:- ")
# v8 moves the amount's visible punctuation into finite format heads.  The
# raw CTC reader therefore owns only the signed canonical decimal, making its
# sequence task materially easier while retaining every visible glyph via the
# grammar-safe renderer below.
V8_AMOUNT_CHARACTERS = tuple("0123456789.-")
NUMERIC_BLANK_INDEX = 0
PAYMENT_BLANK_INDEX = 0
RECIPIENT_BLANK_INDEX = 0
STATUS_TEXT_BLANK_INDEX = 0
PAYMENT_BANK_OTHER_CLASS = "__other__"

# v5 keeps these structural pieces deliberately small and fixed.  They are
# auxiliary outputs of the *same* ONNX graph, not extra OCR models/sessions.
# Covers values through 9,999,999.99 while keeping the student head compact.
# The dataset contract rejects larger values from structural supervision rather
# than silently truncating them; their legacy CTC label remains available.
AMOUNT_MAX_INTEGER_DIGITS = TARGET_AMOUNT_MAX_INTEGER_DIGITS
AMOUNT_DIGIT_SLOTS = AMOUNT_MAX_INTEGER_DIGITS + 2  # right-aligned integer + cents
TIME_DIGIT_SLOTS = 4  # canonical HHMM; hour display width is a separate head
PAYMENT_TAIL_DIGIT_SLOTS = 4
PAYMENT_STRUCTURE_CLASSES = ("unstructured", "card_tail4")
PAYMENT_PARENTHESIS_CLASSES = (PARENTHESIS_STYLE_ASCII, PARENTHESIS_STYLE_FULLWIDTH)
STRUCTURED_IGNORE_INDEX = -100

LEGACY_ONNX_OUTPUT_NAMES = ("amount_logits", "time_logits", "payment_logits", "status_logits")
V5_ONNX_OUTPUT_NAMES = (
    "amount_logits",
    "time_logits",
    "payment_logits",
    "status_logits",
    "amount_length_logits",
    "amount_digit_logits",
    "time_digit_logits",
    "time_hour_width_logits",
    "payment_prefix_logits",
    "payment_tail_digit_logits",
    "payment_structure_logits",
    "payment_parentheses_logits",
)
V6_ONNX_OUTPUT_NAMES = (
    "amount_logits",
    "time_logits",
    "payment_logits",
    "status_logits",
    "amount_sign_logits",
    "amount_length_logits",
    "amount_digit_logits",
    "time_format_logits",
    "time_digit_logits",
    "payment_prefix_logits",
    "payment_bank_prefix_logits",
    "payment_tail_digit_logits",
    "payment_structure_logits",
    "payment_parentheses_logits",
)
V8_ONNX_OUTPUT_NAMES = (
    "amount_logits",
    "time_logits",
    "payment_logits",
    "status_logits",
    "amount_currency_style_logits",
    "amount_grouped_thousands_logits",
    "amount_sign_position_logits",
    "time_format_logits",
    "time_digit_logits",
    "payment_prefix_logits",
    "payment_bank_prefix_logits",
    "payment_tail_digit_logits",
    "payment_structure_logits",
    "payment_parentheses_logits",
)
# v9 is deliberately an additive protocol: the established v8 output order is
# frozen byte-for-byte and the fifth free-text slot is appended.  This lets a
# runtime distinguish an incompatible five-slot artifact before it ever runs
# a four-slot model.
V9_ONNX_OUTPUT_NAMES = V8_ONNX_OUTPUT_NAMES + ("recipient_logits",)
# v10 keeps the same compact five-slot graph and output order as v9.  Its
# incompatible contract is the *recipient supervision*: v9 labels only the
# merchant value, while v10 labels the complete visible row and extracts the
# value after CTC decoding.  A new kind prevents mixing those datasets or
# sidecars by accident.
V10_ONNX_OUTPUT_NAMES = V9_ONNX_OUTPUT_NAMES
# v11 keeps the exact same five-slot graph interface as v9/v10.  It changes
# only the recipient input/target contract: a quality-filtered row is cropped
# to its value area and the CTC head predicts that value directly.
V11_ONNX_OUTPUT_NAMES = V9_ONNX_OUTPUT_NAMES
# v12 deliberately keeps the v9-v11 output ABI unchanged.  The only graph
# change is an additional *input* dedicated to the recipient value view.
V12_ONNX_OUTPUT_NAMES = V9_ONNX_OUTPUT_NAMES
# v13 is the first status-text OCR protocol.  Preserve every v12 output in
# its frozen order and append one visible-text CTC tensor.
V13_ONNX_OUTPUT_NAMES = V12_ONNX_OUTPUT_NAMES + ("status_text_logits",)
# Only these output tensors use CTC.  The remaining v5 tensors are ordinary
# fixed-position / classification logits and deliberately have no blank index.
# Keeping this distinction in one place prevents the delivery contract loader
# from treating a structured digit head as a CTC sequence.
CTC_ONNX_BLANK_INDICES = {
    "amount_logits": NUMERIC_BLANK_INDEX,
    "time_logits": NUMERIC_BLANK_INDEX,
    "payment_logits": PAYMENT_BLANK_INDEX,
    "payment_prefix_logits": PAYMENT_BLANK_INDEX,
    "recipient_logits": RECIPIENT_BLANK_INDEX,
    "status_text_logits": STATUS_TEXT_BLANK_INDEX,
}

# Match the project-wide fixed-graph export tolerance.  The unified reader
# additionally requires every output position to keep the exact same argmax,
# which is the actual character/class decision boundary used by the delivery
# decoder.  This accepts harmless CPU Torch/ORT accumulation drift near zero
# without accepting a changed decoded result.
ONNX_EXPORT_RTOL = 1e-3
ONNX_EXPORT_ATOL = 1e-3
# ORT's CPU GRU kernel can differ from Torch by just under 0.002 logit on the
# exported raw payment or time CTC heads.  On the v11 five-field graph, the
# same CPU path has shown bounded CTC drift of 0.00955 (amount) and 0.02651
# (time).  Keep that wider bound scoped to v11/v12 raw CTC heads only; every
# fixed-position/classification output retains the project-wide 1e-3 absolute
# tolerance.  The exact per-position argmax check below still rejects a
# changed character or class decision.  It verifies greedy CTC text only;
# softmax confidence remains an ONNX-evaluated review/ranking signal.
ONNX_EXPORT_PAYMENT_LOGITS_ATOL = 2e-3
ONNX_EXPORT_TIME_LOGITS_ATOL = 2e-3
ONNX_EXPORT_V11_CTC_LOGITS_ATOL = 3e-2
ONNX_EXPORT_V11_CTC_LOGITS_MEAN_ABS_CAP = 1e-3

# A v5 CTC prediction and its structural prediction are deliberately exposed
# together for diagnostics, but they are not independent evidence: both are
# derived from the same student model and the same Paddle-derived labels.
# Until an artifact has a separately implemented and human-truth-calibrated
# acceptance policy, text values must stay review-only.  Keeping this as a
# contract value makes a future .NET consumer unable to mistake a diagnostic
# candidate for a safe business value.
V5_TEXT_DELIVERY_POLICY = "review_only_pending_independent_calibration"
V5_TEXT_DELIVERY_REASON = (
    "CTC and structured heads share the student model; agreement is diagnostic only. "
    "Emit review until a separately implemented policy passes group-isolated human-truth calibration."
)
V6_TEXT_DELIVERY_POLICY = "review_only_pending_independent_human_truth_calibration"
V6_TEXT_DELIVERY_REASON = (
    "Visible CTC and verifier branches are separately implemented after the shared encoder, but their "
    "teacher labels are still Paddle-derived. Emit review until a group-isolated human-truth calibration "
    "accepts their format-and-agreement policy."
)
V7_TEXT_DELIVERY_POLICY = "review_only_pending_independent_human_truth_calibration"
V7_TEXT_DELIVERY_REASON = (
    "Visible CTC and format heads share the time reader in this compact student, and all teacher labels are "
    "still Paddle-derived. Emit review until a group-isolated human-truth calibration accepts the "
    "format-and-agreement policy."
)
V8_TEXT_DELIVERY_POLICY = "review_only_pending_independent_human_truth_calibration"
V8_TEXT_DELIVERY_REASON = (
    "Canonical amount CTC and visible-format heads share the same student model, and all teacher labels are "
    "Paddle-derived. Emit review until a group-isolated human-truth calibration accepts the rendered policy."
)
V9_TEXT_DELIVERY_POLICY = "review_only_pending_independent_human_truth_calibration"
V9_TEXT_DELIVERY_REASON = (
    "Recipient CTC and the other text heads share one compact student, and all current labels are "
    "Paddle-derived. Emit review until a group-isolated human-truth calibration accepts the full five-field policy."
)
V10_TEXT_DELIVERY_POLICY = "review_only_pending_independent_human_truth_calibration"
V10_TEXT_DELIVERY_REASON = (
    "Recipient CTC is trained on the complete visible row then parsed into a merchant value, and all current "
    "labels are Paddle-derived. Emit review until a group-isolated human-truth calibration accepts the full "
    "five-field policy."
)
V11_TEXT_DELIVERY_POLICY = "review_only_pending_independent_human_truth_calibration"
V11_TEXT_DELIVERY_REASON = (
    "Recipient CTC uses an anchored value-only crop and all current labels are Paddle-derived. "
    "Emit review until a group-isolated human-truth calibration accepts the full five-field policy."
)
V12_TEXT_DELIVERY_POLICY = "review_only_pending_independent_human_truth_calibration"
V12_TEXT_DELIVERY_REASON = (
    "Recipient CTC uses a dedicated high-resolution anchored value view and all current labels are "
    "Paddle-derived. Emit review until a group-isolated human-truth calibration accepts the full five-field policy."
)
V13_TEXT_DELIVERY_POLICY = V12_TEXT_DELIVERY_POLICY
V13_TEXT_DELIVERY_REASON = (
    "Visible transfer-status CTC and all financial/recipient heads are student-model outputs trained from "
    "teacher labels. Decode and normalize status text for review, but emit review as the business value until "
    "an independent group-isolated human-truth calibration accepts status delivery."
)
STATUS_TEXT_RUNTIME_POLICY = "decode_and_normalize_review_only"


def _uses_structured_heads(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version >= 5


def _uses_v6_protocol(config: "UnifiedReaderConfig") -> bool:
    """Return whether an artifact uses v6's visible-format/bank output protocol.

    v7 intentionally keeps that external protocol (charsets, 14 ONNX
    outputs, and structured decoders) while changing only the time-reader
    training topology.  Keeping this distinct from an exact ``== 6`` test is
    what lets a v6 checkpoint continue to use its original forward path.
    """
    return config.architecture_version in {6, 7}


def _is_v7(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version == 7


def _is_v8(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version == 8


def _is_v9(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version == 9


def _is_v10(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version == 10


def _is_v11(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version == 11


def _is_v12(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version == 12


def _is_v13(config: "UnifiedReaderConfig") -> bool:
    return config.architecture_version == 13


def _uses_v12_recipient_topology(config: "UnifiedReaderConfig") -> bool:
    """Return whether the frozen v12 private recipient branch is present."""
    return _is_v12(config) or _is_v13(config)


def _uses_high_resolution_recipient_input(config: "UnifiedReaderConfig") -> bool:
    """Return whether a reader has v12's second static recipient input."""
    return _uses_v12_recipient_topology(config)


def _uses_recipient_protocol(config: "UnifiedReaderConfig") -> bool:
    """Return whether this artifact has the additive fifth CTC input/output."""
    return config.architecture_version in {9, 10, 11, 12, 13}


def _recipient_target_mode(config: "UnifiedReaderConfig") -> str | None:
    """Return the immutable recipient-label contract for five-slot artifacts."""
    if _is_v9(config):
        return "visible_recipient_value"
    if _is_v10(config):
        return "visible_recipient_line_then_extract_value"
    if _is_v11(config):
        return "anchored_recipient_value_with_value_view_crop"
    if _uses_v12_recipient_topology(config):
        return "anchored_recipient_value_with_dedicated_high_resolution_value_view"
    return None


def _recipient_charset_source(config: "UnifiedReaderConfig") -> str | None:
    """Return the immutable origin of the fifth-head training alphabet."""
    if _is_v10(config):
        return "train_only_visible_recipient_line"
    if _is_v11(config) or _uses_v12_recipient_topology(config):
        return "train_only_anchored_recipient_value"
    if _is_v9(config):
        return "train_only_visible_recipient_text"
    return None


def _recipient_input_preprocess(config: "UnifiedReaderConfig") -> str | None:
    """Return the fifth-slot visual policy recorded in an artifact contract."""
    if _is_v11(config):
        return "left_trim_then_centered_aspect_resize"
    if _uses_v12_recipient_topology(config):
        return "left_trim_then_centered_aspect_resize_high_resolution"
    if _is_v10(config):
        return "centered_aspect_resize_full_visible_row"
    if _is_v9(config):
        return "right_aligned_aspect_resize_full_visible_row"
    return None


def _uses_v8_protocol(config: "UnifiedReaderConfig") -> bool:
    """Return whether the v8 amount/time/payment output protocol is present."""
    return config.architecture_version >= 8


def _slot_order(config: "UnifiedReaderConfig") -> tuple[str, ...]:
    """Return the immutable input-channel order for this artifact version."""
    return V9_SLOT_ORDER if _uses_recipient_protocol(config) else SLOT_ORDER


def _uses_modern_protocol(config: "UnifiedReaderConfig") -> bool:
    """Return whether a reader uses the v6+ time/payment protocol."""
    return config.architecture_version >= 6


def _amount_characters(config: "UnifiedReaderConfig") -> tuple[str, ...]:
    if _uses_v8_protocol(config):
        return V8_AMOUNT_CHARACTERS
    return V6_AMOUNT_CHARACTERS if _uses_v6_protocol(config) else NUMERIC_CHARACTERS


def _time_characters(config: "UnifiedReaderConfig") -> tuple[str, ...]:
    return V6_TIME_CHARACTERS if _uses_modern_protocol(config) else NUMERIC_CHARACTERS


def _text_delivery_policy(config: "UnifiedReaderConfig") -> tuple[str, str]:
    if config.architecture_version == 6:
        return V6_TEXT_DELIVERY_POLICY, V6_TEXT_DELIVERY_REASON
    if config.architecture_version == 7:
        return V7_TEXT_DELIVERY_POLICY, V7_TEXT_DELIVERY_REASON
    if config.architecture_version == 8:
        return V8_TEXT_DELIVERY_POLICY, V8_TEXT_DELIVERY_REASON
    if config.architecture_version == 9:
        return V9_TEXT_DELIVERY_POLICY, V9_TEXT_DELIVERY_REASON
    if config.architecture_version == 10:
        return V10_TEXT_DELIVERY_POLICY, V10_TEXT_DELIVERY_REASON
    if config.architecture_version == 11:
        return V11_TEXT_DELIVERY_POLICY, V11_TEXT_DELIVERY_REASON
    if config.architecture_version == 12:
        return V12_TEXT_DELIVERY_POLICY, V12_TEXT_DELIVERY_REASON
    if config.architecture_version == 13:
        return V13_TEXT_DELIVERY_POLICY, V13_TEXT_DELIVERY_REASON
    return V5_TEXT_DELIVERY_POLICY, V5_TEXT_DELIVERY_REASON


def _onnx_output_names(config: "UnifiedReaderConfig") -> tuple[str, ...]:
    if _is_v13(config):
        return V13_ONNX_OUTPUT_NAMES
    if _uses_recipient_protocol(config):
        return V9_ONNX_OUTPUT_NAMES
    if _is_v8(config):
        return V8_ONNX_OUTPUT_NAMES
    if _uses_v6_protocol(config):
        return V6_ONNX_OUTPUT_NAMES
    return V5_ONNX_OUTPUT_NAMES if config.architecture_version == 5 else LEGACY_ONNX_OUTPUT_NAMES


def _onnx_export_atol(
    output_name: str,
    *,
    config: "UnifiedReaderConfig | None" = None,
) -> float:
    """Use narrowly validated ORT tolerances for raw CTC heads only."""
    if config is not None and (_is_v11(config) or _uses_v12_recipient_topology(config)) and output_name in CTC_ONNX_BLANK_INDICES:
        return ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    if output_name == "payment_logits":
        return ONNX_EXPORT_PAYMENT_LOGITS_ATOL
    if output_name == "time_logits":
        return ONNX_EXPORT_TIME_LOGITS_ATOL
    return ONNX_EXPORT_ATOL


def _onnx_export_max_abs_cap(
    output_name: str,
    *,
    config: "UnifiedReaderConfig | None" = None,
) -> float | None:
    """Return a hard cap when a scoped tolerance must not be expanded by rtol."""
    # The v12 private recipient branch uses a 256-step bidirectional GRU.
    # ORT's CPU GRU kernel may accumulate substantially different raw logits
    # from Torch while retaining every greedy CTC decision.  It is checked by
    # exact per-position argmax parity on both export probes plus the full
    # delivery-ONNX evaluator, rather than by a misleading raw-logit cap.
    if config is not None and _uses_v12_recipient_topology(config) and output_name == "recipient_logits":
        return None
    if config is not None and (_is_v11(config) or _uses_v12_recipient_topology(config)) and output_name in CTC_ONNX_BLANK_INDICES:
        return ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    return None


def _onnx_export_mean_abs_cap(
    output_name: str,
    *,
    config: "UnifiedReaderConfig | None" = None,
) -> float | None:
    """Reject a broad v11/v12 CTC-logit shift even when no greedy decision flips."""
    if config is not None and _uses_v12_recipient_topology(config) and output_name == "recipient_logits":
        return None
    if config is not None and (_is_v11(config) or _uses_v12_recipient_topology(config)) and output_name in CTC_ONNX_BLANK_INDICES:
        return ONNX_EXPORT_V11_CTC_LOGITS_MEAN_ABS_CAP
    return None


def _onnx_export_requires_raw_logit_close(
    output_name: str,
    *,
    config: "UnifiedReaderConfig | None" = None,
) -> bool:
    """Whether an output has a stable raw Torch/ORT logit comparison contract.

    A v12 high-resolution recipient head is the one intentional exception.
    Its 256-step bidirectional GRU can have materially different CPU Torch and
    ORT logits, even though every greedy CTC decision is identical.  Raw CTC
    confidence is review-only; delivery correctness is guarded by exact
    per-position argmax parity on two deterministic probes and by the full
    CUDA ONNX evaluator.  All other outputs retain their numeric parity
    requirement.
    """
    return not (
        config is not None
        and _uses_v12_recipient_topology(config)
        and output_name == "recipient_logits"
    )


def _v12_recipient_export_probe(recipient_value_image: Any, *, torch: Any) -> Any:
    """Build a deterministic non-blank recipient view for ONNX parity checks.

    The normal all-zero export input is useful for graph shape validation but
    is not representative of the high-resolution, ink-on-light-background
    recipient view used in production.  This synthetic probe deliberately
    contains several asymmetric black strokes on a white field, without
    depending on private receipt data or altering the exported graph.
    """
    probe = torch.ones_like(recipient_value_image)
    height = int(probe.shape[-2])
    width = int(probe.shape[-1])
    stroke = max(1, height // 12)
    left = max(1, width // 9)
    middle = max(left + 1, width // 2)
    right = max(middle + 1, (width * 8) // 9)
    upper = max(1, height // 5)
    lower = max(upper + stroke + 1, (height * 3) // 5)
    probe[..., upper : upper + stroke, left:right] = 0.0
    probe[..., lower : lower + stroke, left:middle] = 0.0
    probe[..., upper:lower, middle : middle + stroke] = 0.0
    return probe


def _kind_for_architecture(architecture_version: int) -> str:
    if architecture_version == 3:
        return KIND_V3
    if architecture_version == 4:
        return KIND_V4
    if architecture_version == 5:
        return KIND_V5
    if architecture_version == 6:
        return KIND_V6
    if architecture_version == 7:
        return KIND_V7
    if architecture_version == 8:
        return KIND_V8
    if architecture_version == 9:
        return KIND_V9
    if architecture_version == 10:
        return KIND_V10
    if architecture_version == 11:
        return KIND_V11
    if architecture_version == 12:
        return KIND_V12
    if architecture_version == 13:
        return KIND_V13
    raise ValueError(f"Unsupported unified reader architecture v{architecture_version}")


def _kind_for_config(config: "UnifiedReaderConfig") -> str:
    return _kind_for_architecture(config.architecture_version)


def _architecture_for_kind(kind: object) -> int:
    if kind == KIND_V3:
        return 3
    if kind == KIND_V4:
        return 4
    if kind == KIND_V5:
        return 5
    if kind == KIND_V6:
        return 6
    if kind == KIND_V7:
        return 7
    if kind == KIND_V8:
        return 8
    if kind == KIND_V9:
        return 9
    if kind == KIND_V10:
        return 10
    if kind == KIND_V11:
        return 11
    if kind == KIND_V12:
        return 12
    if kind == KIND_V13:
        return 13
    raise ValueError(f"Unsupported unified OCR artifact kind: {kind!r}")


@dataclass(frozen=True)
class UnifiedReaderConfig:
    # v5 uses a still-small 80x512 view so financial glyphs retain enough
    # detail for the structural heads. v3/v4 stay loadable for compatibility.
    # Keep v8 as the default for existing callers.  v9 is selected explicitly
    # by the five-field training command so an old four-slot invocation never
    # gains an uninitialised recipient channel by accident.
    architecture_version: int = 8
    image_height: int = 80
    image_width: int = 512
    base_channels: int = 32
    numeric_hidden_size: int = 96
    payment_hidden_size: int = 128
    # v11 gives the open-vocabulary recipient CTC branch a little more
    # sequence capacity without widening the shared CNN or payment branch.
    # ``None`` keeps historical checkpoint layouts unchanged; v11/v12 resolve
    # it deterministically to 192 in :func:`_recipient_hidden_size`.
    recipient_hidden_size: int | None = None
    # v11 removes the static left-side recipient label before aspect-preserving
    # resize.  This is part of the artifact model config so a deployment
    # runtime can reproduce the exact fifth-slot preprocessing.
    recipient_value_left_trim: float = 0.30
    # v12 preserves the five-slot field input for ABI continuity, but its
    # recipient CTC head receives this independent, high-resolution value view
    # as a second static ONNX input.  Its small private CNN avoids widening the
    # four-field shared trunk merely to read open Chinese merchant text.
    recipient_input_height: int = 128
    recipient_input_width: int = 1024
    recipient_branch_channels: int | None = None
    # Optional context-only refinement for the v12 character CTC stream.  It
    # is residual-gated from exactly zero so a warm-started open-text model is
    # byte/decision identical to its seed before the first optimiser step.
    recipient_open_text_layers: int = 0
    recipient_open_text_heads: int = 8
    recipient_open_text_feedforward: int | None = None
    # Train-time dropout for the private recipient Transformer.  It is zero
    # by default so every published v12/v13 checkpoint keeps its historical
    # graph and numerics.  A capacity-reinitialised recipient experiment may
    # opt into bounded dropout without changing ONNX inputs/outputs; dropout
    # is disabled by ``eval()`` before export and therefore adds no runtime op.
    recipient_open_text_dropout: float = 0.0
    # The established branch is retained as the default for byte-compatible
    # v12/v13 loading.  ``residual_positional_transformer_v2`` is a genuinely
    # different open-text recogniser: a standard residual 2-D visual encoder,
    # explicit learned sequence positions, and a direct Transformer CTC head
    # (no BiGRU and no zero-gated adapter).  It is opt-in and v13-only so the
    # existing status-text ABI remains unchanged while its private recipient
    # tensors can be reinitialised under a dedicated audited mode.
    recipient_backbone: str = "legacy_depthwise_gru_v1"
    pooled_width: int = 8
    # v8 applies the display renderer only when every finite format component
    # is confident.  This is a diagnostic-candidate gate, never a business
    # delivery gate; keeping it in the artifact config makes Python and a
    # future deployment consumer use the identical policy.
    amount_format_min_confidence: float = 0.90

    def validate(self) -> None:
        if self.architecture_version not in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}:
            raise ValueError("architecture_version must be 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, or 13")
        if self.image_height < 16 or self.image_width < 64 or self.image_width % 4:
            raise ValueError("image_height must be >=16 and image_width must be a multiple of 4 >=64")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if self.numeric_hidden_size < 16 or self.payment_hidden_size < 16:
            raise ValueError("numeric_hidden_size and payment_hidden_size must be at least 16")
        if self.recipient_hidden_size is not None and self.recipient_hidden_size < 16:
            raise ValueError("recipient_hidden_size must be at least 16 when supplied")
        # The recipient recurrent-width knob alters the fifth-head topology.
        # It is deliberately v11/v12-only: letting a caller change it for a
        # v9 or v10 artifact would produce a state dict that claims an older
        # protocol while having different learned tensors/pixels.
        if self.architecture_version not in {11, 12, 13} and self.recipient_hidden_size is not None:
            raise ValueError("recipient_hidden_size is supported only by architecture v11, v12, or v13")
        if not 1 <= self.pooled_width <= 32:
            raise ValueError("pooled_width must be between 1 and 32")
        if not math.isfinite(self.amount_format_min_confidence) or not 0.0 <= self.amount_format_min_confidence <= 1.0:
            raise ValueError("amount_format_min_confidence must be between 0 and 1")
        if not math.isfinite(self.recipient_value_left_trim) or not 0.0 <= self.recipient_value_left_trim < 1.0:
            raise ValueError("recipient_value_left_trim must be in [0, 1)")
        if self.architecture_version not in {11, 12, 13} and not math.isclose(
            self.recipient_value_left_trim, 0.30, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("recipient_value_left_trim is supported only by architecture v11, v12, or v13")
        if self.recipient_input_height < 16 or self.recipient_input_width < 64 or self.recipient_input_width % 4:
            raise ValueError("recipient_input_height must be >=16 and recipient_input_width a multiple of 4 >=64")
        if self.recipient_branch_channels is not None and self.recipient_branch_channels < 8:
            raise ValueError("recipient_branch_channels must be at least 8 when supplied")
        if self.architecture_version not in {12, 13} and (
            self.recipient_input_height != 128
            or self.recipient_input_width != 1024
            or self.recipient_branch_channels is not None
            or self.recipient_open_text_layers != 0
        ):
            raise ValueError("recipient high-resolution input settings are supported only by architecture v12 or v13")
        if self.recipient_open_text_layers < 0 or self.recipient_open_text_layers > 6:
            raise ValueError("recipient_open_text_layers must be between 0 and 6")
        if self.recipient_open_text_heads <= 0:
            raise ValueError("recipient_open_text_heads must be positive")
        recipient_width = _recipient_hidden_size(self) * 2
        if self.recipient_open_text_layers and recipient_width % self.recipient_open_text_heads:
            raise ValueError("recipient open-text width must be divisible by recipient_open_text_heads")
        if self.recipient_open_text_feedforward is not None and self.recipient_open_text_feedforward < recipient_width:
            raise ValueError("recipient_open_text_feedforward must be at least the bidirectional recipient width")
        if (
            not math.isfinite(self.recipient_open_text_dropout)
            or not 0.0 <= self.recipient_open_text_dropout <= 0.5
        ):
            raise ValueError("recipient_open_text_dropout must be between 0 and 0.5")
        if self.architecture_version not in {12, 13} and not math.isclose(
            self.recipient_open_text_dropout, 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("recipient_open_text_dropout is supported only by architecture v12 or v13")
        if self.recipient_backbone not in {
            "legacy_depthwise_gru_v1",
            "residual_positional_transformer_v2",
        }:
            raise ValueError("recipient_backbone is invalid")
        if self.recipient_backbone != "legacy_depthwise_gru_v1":
            if self.architecture_version != 13:
                raise ValueError("the residual recipient backbone is supported only by architecture v13")
            if self.recipient_open_text_layers < 2:
                raise ValueError("the residual recipient backbone requires at least two Transformer layers")
            if self.recipient_input_width % 8:
                raise ValueError("the residual recipient backbone requires recipient_input_width divisible by 8")


def _recipient_hidden_size(config: UnifiedReaderConfig) -> int:
    """Return the frozen recipient branch width for this architecture."""
    if config.recipient_hidden_size is not None:
        return int(config.recipient_hidden_size)
    return 192 if _is_v11(config) or _uses_v12_recipient_topology(config) else config.payment_hidden_size


def _recipient_branch_channels(config: UnifiedReaderConfig) -> int:
    """Return v12's deliberately narrow private recipient visual width."""
    if not _uses_v12_recipient_topology(config):
        raise ValueError("recipient_branch_channels is defined only for architecture v12 or v13")
    return int(config.recipient_branch_channels or 16)


def _recipient_time_steps(config: UnifiedReaderConfig) -> int:
    """Return the CTC sequence length for the fifth output head."""
    if config.recipient_backbone == "residual_positional_transformer_v2":
        return config.recipient_input_width // 8
    return config.recipient_input_width // 4 if _uses_high_resolution_recipient_input(config) else config.image_width // 4


def _recipient_slot(record: Mapping[str, object]) -> Mapping[str, object] | None:
    """Return the optional recipient slot without mutating a manifest row."""
    slots = record.get("slots")
    if not isinstance(slots, Mapping):
        return None
    slot = slots.get("recipient_field")
    return slot if isinstance(slot, Mapping) else None


def _recipient_training_sample_weights(
    records: Sequence[Mapping[str, object]],
    *,
    recipient_sampling_weight: float,
    recipient_rare_character_max_support: int = 0,
    recipient_rare_character_sampling_weight: float = 1.0,
    recipient_long_text_min_length: int = 0,
    recipient_long_text_sampling_weight: float = 1.0,
) -> tuple[list[float], dict[str, object]]:
    """Build bounded, train-split-only receipt sampling weights.

    The recipient CTC task has a much wider long-tail alphabet than the other
    four fields.  This helper deliberately takes the *maximum* requested
    boost instead of multiplying factors: a rare long merchant must be seen
    more often, but must never dominate an epoch just because it satisfies two
    conditions.  It is pure so the frozen policy can be tested without Torch
    or image assets.
    """
    numeric_values = {
        "recipient_sampling_weight": recipient_sampling_weight,
        "recipient_rare_character_sampling_weight": recipient_rare_character_sampling_weight,
        "recipient_long_text_sampling_weight": recipient_long_text_sampling_weight,
    }
    normalized_weights: dict[str, float] = {}
    for name, raw_value in numeric_values.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be finite and positive") from None
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        normalized_weights[name] = value
    if isinstance(recipient_rare_character_max_support, bool) or not isinstance(
        recipient_rare_character_max_support, int
    ) or recipient_rare_character_max_support < 0:
        raise ValueError("recipient_rare_character_max_support must be a non-negative integer")
    if isinstance(recipient_long_text_min_length, bool) or not isinstance(recipient_long_text_min_length, int) or recipient_long_text_min_length < 0:
        raise ValueError("recipient_long_text_min_length must be a non-negative integer")

    recipient_texts: list[str | None] = []
    character_counts: Counter[str] = Counter()
    for record in records:
        slot = _recipient_slot(record)
        text = slot.get("text") if slot is not None else None
        text = text if isinstance(text, str) and text else None
        recipient_texts.append(text)
        if text is not None:
            character_counts.update(text)

    rare_hits = 0
    long_hits = 0
    special_boost_applied = False
    result: list[float] = []
    for text in recipient_texts:
        if text is None:
            result.append(1.0)
            continue
        # Preserve the historical v11/v12 behaviour exactly: a value below
        # one deliberately *downsamples* recipient rows.  The v2 long-tail
        # options may raise that base weight for a selected row, but they do
        # not silently clamp an existing downsampling recipe back to one.
        weight = normalized_weights["recipient_sampling_weight"]
        rare = (
            recipient_rare_character_max_support > 0
            and any(character_counts[character] <= recipient_rare_character_max_support for character in text)
        )
        long = recipient_long_text_min_length > 0 and len(text) >= recipient_long_text_min_length
        if rare:
            rare_hits += 1
            rare_weight = normalized_weights["recipient_rare_character_sampling_weight"]
            if rare_weight > weight:
                special_boost_applied = True
            weight = max(weight, rare_weight)
        if long:
            long_hits += 1
            long_weight = normalized_weights["recipient_long_text_sampling_weight"]
            if long_weight > weight:
                special_boost_applied = True
            weight = max(weight, long_weight)
        result.append(float(weight))

    recipient_records = sum(text is not None for text in recipient_texts)
    sampling_is_uniform = all(math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=1e-12) for weight in result)
    if sampling_is_uniform or not special_boost_applied:
        mode = "uniform" if sampling_is_uniform else "weighted_receipt_sampler_v1"
        return result, {
            "mode": mode,
            "recipient_sampling_weight": normalized_weights["recipient_sampling_weight"],
            "recipient_train_records": int(recipient_records),
            "train_records": len(records),
        }
    return result, {
        "mode": "weighted_receipt_sampler_v2",
        "recipient_sampling_weight": normalized_weights["recipient_sampling_weight"],
        "recipient_rare_character_max_support": int(recipient_rare_character_max_support),
        "recipient_rare_character_sampling_weight": normalized_weights[
            "recipient_rare_character_sampling_weight"
        ],
        "recipient_long_text_min_length": int(recipient_long_text_min_length),
        "recipient_long_text_sampling_weight": normalized_weights["recipient_long_text_sampling_weight"],
        "recipient_rare_character_train_records": int(rare_hits),
        "recipient_long_text_train_records": int(long_hits),
        "recipient_training_character_count": len(character_counts),
        "recipient_train_records": int(recipient_records),
        "train_records": len(records),
    }


def _recipient_confidence_policy(
    *,
    low_confidence_threshold: float | None,
    low_confidence_loss_weight: float,
    curriculum_epochs: int,
) -> dict[str, object]:
    """Validate and freeze training-only Paddle-teacher confidence handling."""
    try:
        normalized_weight = float(low_confidence_loss_weight)
    except (TypeError, ValueError):
        raise ValueError("recipient_low_confidence_loss_weight must be finite and in (0, 1]") from None
    if not math.isfinite(normalized_weight) or not 0.0 < normalized_weight <= 1.0:
        raise ValueError("recipient_low_confidence_loss_weight must be finite and in (0, 1]")
    if isinstance(curriculum_epochs, bool) or not isinstance(curriculum_epochs, int) or curriculum_epochs < 0:
        raise ValueError("recipient_confidence_curriculum_epochs must be a non-negative integer")
    if low_confidence_threshold is None:
        if curriculum_epochs != 0 or not math.isclose(normalized_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "recipient_low_confidence_threshold is required when recipient confidence weighting is configured"
            )
        return {
            "mode": "none",
            "low_confidence_threshold": None,
            "low_confidence_loss_weight": 1.0,
            "curriculum_epochs": 0,
        }
    try:
        normalized_threshold = float(low_confidence_threshold)
    except (TypeError, ValueError):
        raise ValueError("recipient_low_confidence_threshold must be finite and between 0 and 1") from None
    if not math.isfinite(normalized_threshold) or not 0.0 <= normalized_threshold <= 1.0:
        raise ValueError("recipient_low_confidence_threshold must be finite and between 0 and 1")
    return {
        "mode": "teacher_confidence_curriculum_v1",
        "low_confidence_threshold": normalized_threshold,
        "low_confidence_loss_weight": normalized_weight,
        "curriculum_epochs": int(curriculum_epochs),
    }


def _validate_recipient_confidence_policy(policy: object) -> dict[str, object]:
    """Validate a persisted confidence policy without changing model ABI."""
    if not isinstance(policy, Mapping):
        raise ValueError("recipient confidence policy is missing or invalid")
    mode = policy.get("mode")
    threshold = policy.get("low_confidence_threshold")
    weight = policy.get("low_confidence_loss_weight")
    epochs = policy.get("curriculum_epochs")
    if mode == "none":
        if threshold is not None:
            raise ValueError("recipient confidence policy is invalid")
        return _recipient_confidence_policy(
            low_confidence_threshold=None,
            low_confidence_loss_weight=weight,
            curriculum_epochs=epochs,
        )
    if mode != "teacher_confidence_curriculum_v1":
        raise ValueError("recipient confidence policy is invalid")
    return _recipient_confidence_policy(
        low_confidence_threshold=threshold if threshold is not None else None,
        low_confidence_loss_weight=weight,
        curriculum_epochs=epochs,
    )


def _recipient_teacher_confidence_weights(
    records: Sequence[Mapping[str, object]],
    *,
    low_confidence_threshold: float | None,
    low_confidence_loss_weight: float = 1.0,
    curriculum_epoch: int = 1,
    curriculum_epochs: int = 0,
) -> list[float]:
    """Return one recipient CTC loss weight per receipt.

    A disabled policy is intentionally a cheap all-one fast path.  When
    enabled, low-confidence Paddle labels begin at weight one and linearly
    settle at their requested lower influence by the final curriculum epoch.
    This avoids abruptly changing the CTC loss scale in the first epoch while
    still letting high-confidence teacher rows dominate the fitted model.
    """
    policy = _recipient_confidence_policy(
        low_confidence_threshold=low_confidence_threshold,
        low_confidence_loss_weight=low_confidence_loss_weight,
        curriculum_epochs=curriculum_epochs,
    )
    if policy["mode"] == "none":
        return [1.0] * len(records)
    if isinstance(curriculum_epoch, bool) or not isinstance(curriculum_epoch, int) or curriculum_epoch <= 0:
        raise ValueError("recipient confidence curriculum epoch must be a positive integer")
    threshold = float(policy["low_confidence_threshold"])
    target_weight = float(policy["low_confidence_loss_weight"])
    epochs = int(policy["curriculum_epochs"])
    if epochs == 0:
        ramp = 1.0
    elif epochs == 1:
        ramp = 1.0
    else:
        ramp = min(1.0, float(curriculum_epoch - 1) / float(epochs - 1))
    low_confidence_weight = 1.0 - (1.0 - target_weight) * ramp
    weights: list[float] = []
    for record in records:
        slot = _recipient_slot(record)
        if slot is None:
            weights.append(1.0)
            continue
        raw_confidence = slot.get("paddle_confidence")
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            raise ValueError(
                f"recipient record {record.get('id', '<unknown>')} has invalid paddle_confidence"
            ) from None
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"recipient record {record.get('id', '<unknown>')} has invalid paddle_confidence")
        weights.append(low_confidence_weight if confidence < threshold else 1.0)
    return weights


def _recipient_tail_loss_character_counts(records: Sequence[Mapping[str, object]]) -> Counter[str]:
    """Count recipient characters from the train split only."""
    counts: Counter[str] = Counter()
    for record in records:
        slot = _recipient_slot(record)
        text = slot.get("text") if slot is not None else None
        if isinstance(text, str) and text:
            counts.update(text)
    return counts


def _recipient_tail_loss_config(
    *,
    rare_character_max_support: int,
    rare_character_loss_weight: float,
    long_text_min_length: int,
    long_text_loss_weight: float,
) -> dict[str, object]:
    """Validate the static controls for bounded recipient-tail CTC boosts."""
    integer_values = {
        "recipient_tail_rare_character_max_support": rare_character_max_support,
        "recipient_tail_long_text_min_length": long_text_min_length,
    }
    normalized_integers: dict[str, int] = {}
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        normalized_integers[name] = int(value)
    numeric_values = {
        "recipient_tail_rare_character_loss_weight": rare_character_loss_weight,
        "recipient_tail_long_text_loss_weight": long_text_loss_weight,
    }
    normalized_weights: dict[str, float] = {}
    for name, raw_value in numeric_values.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be finite and at least 1") from None
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(f"{name} must be finite and at least 1")
        normalized_weights[name] = value
    rare_enabled = (
        normalized_integers["recipient_tail_rare_character_max_support"] > 0
        and normalized_weights["recipient_tail_rare_character_loss_weight"] > 1.0
    )
    long_enabled = (
        normalized_integers["recipient_tail_long_text_min_length"] > 0
        and normalized_weights["recipient_tail_long_text_loss_weight"] > 1.0
    )
    return {
        "mode": "rare_long_tail_ctc_v1" if rare_enabled or long_enabled else "none",
        "rare_character_max_support": normalized_integers[
            "recipient_tail_rare_character_max_support"
        ],
        "rare_character_loss_weight": normalized_weights[
            "recipient_tail_rare_character_loss_weight"
        ],
        "long_text_min_length": normalized_integers["recipient_tail_long_text_min_length"],
        "long_text_loss_weight": normalized_weights["recipient_tail_long_text_loss_weight"],
    }


def _recipient_tail_loss_flags(
    text: str | None,
    *,
    policy: Mapping[str, object],
    character_counts: Mapping[str, int],
) -> tuple[bool, bool]:
    """Return active rare/long tail conditions for one recipient label."""
    if not isinstance(text, str) or not text:
        return False, False
    rare_max_support = int(policy["rare_character_max_support"])
    rare_weight = float(policy["rare_character_loss_weight"])
    long_min_length = int(policy["long_text_min_length"])
    long_weight = float(policy["long_text_loss_weight"])
    rare = (
        rare_max_support > 0
        and rare_weight > 1.0
        and any(character_counts.get(character, 0) <= rare_max_support for character in text)
    )
    long = long_min_length > 0 and long_weight > 1.0 and len(text) >= long_min_length
    return rare, long


def _recipient_tail_loss_policy(
    *,
    rare_character_max_support: int,
    rare_character_loss_weight: float,
    long_text_min_length: int,
    long_text_loss_weight: float,
    records: Sequence[Mapping[str, object]],
    character_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Freeze a tail-loss recipe plus its exact train-split hit audit.

    The static configuration records the requested loss boosts.  The audit
    counts prove how many recipient labels actually received each condition
    on this training split, which makes a later 90% decision reviewable
    without changing the DataLoader distribution.
    """
    config = _recipient_tail_loss_config(
        rare_character_max_support=rare_character_max_support,
        rare_character_loss_weight=rare_character_loss_weight,
        long_text_min_length=long_text_min_length,
        long_text_loss_weight=long_text_loss_weight,
    )
    counts = (
        _recipient_tail_loss_character_counts(records)
        if character_counts is None
        else character_counts
    )
    recipient_records = 0
    rare_hits = 0
    long_hits = 0
    combined_hits = 0
    for record in records:
        slot = _recipient_slot(record)
        text = slot.get("text") if slot is not None else None
        if not isinstance(text, str) or not text:
            continue
        recipient_records += 1
        rare, long = _recipient_tail_loss_flags(text, policy=config, character_counts=counts)
        rare_hits += int(rare)
        long_hits += int(long)
        combined_hits += int(rare or long)
    return {
        **config,
        "recipient_train_records": recipient_records,
        "rare_character_train_records": rare_hits,
        "long_text_train_records": long_hits,
        "combined_boost_train_records": combined_hits,
    }


def _validate_recipient_tail_loss_policy(policy: object) -> dict[str, object]:
    """Validate persisted recipient-tail loss provenance without an ABI change."""
    if not isinstance(policy, Mapping):
        raise ValueError("recipient tail loss policy is missing or invalid")
    try:
        config = _recipient_tail_loss_config(
            rare_character_max_support=policy.get("rare_character_max_support"),
            rare_character_loss_weight=policy.get("rare_character_loss_weight"),
            long_text_min_length=policy.get("long_text_min_length"),
            long_text_loss_weight=policy.get("long_text_loss_weight"),
        )
    except ValueError as error:
        raise ValueError("recipient tail loss policy is invalid") from error
    audit_keys = (
        "recipient_train_records",
        "rare_character_train_records",
        "long_text_train_records",
        "combined_boost_train_records",
    )
    if policy.get("mode") != config["mode"] or set(policy) != {*config, *audit_keys}:
        raise ValueError("recipient tail loss policy is invalid")
    audit: dict[str, int] = {}
    for key in audit_keys:
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("recipient tail loss policy is invalid")
        audit[key] = value
    recipient_records = audit["recipient_train_records"]
    rare_hits = audit["rare_character_train_records"]
    long_hits = audit["long_text_train_records"]
    combined_hits = audit["combined_boost_train_records"]
    rare_enabled = (
        int(config["rare_character_max_support"]) > 0
        and float(config["rare_character_loss_weight"]) > 1.0
    )
    long_enabled = (
        int(config["long_text_min_length"]) > 0
        and float(config["long_text_loss_weight"]) > 1.0
    )
    if (
        rare_hits > recipient_records
        or long_hits > recipient_records
        or combined_hits > recipient_records
        or (not rare_enabled and rare_hits != 0)
        or (not long_enabled and long_hits != 0)
        or combined_hits < max(rare_hits, long_hits)
        or combined_hits > rare_hits + long_hits
    ):
        raise ValueError("recipient tail loss policy is invalid")
    return {**config, **audit}


def _recipient_tail_loss_weights(
    records: Sequence[Mapping[str, object]],
    *,
    policy: object,
    character_counts: Mapping[str, int] | None = None,
) -> list[float]:
    """Return bounded recipient-CTC boosts for rows from one train batch.

    ``character_counts`` is normally frozen from the complete training split,
    then reused for shuffled batches.  The optional all-record fallback keeps
    this helper pure and straightforward to test.
    """
    normalized = _validate_recipient_tail_loss_policy(policy)
    if normalized["mode"] == "none":
        return [1.0] * len(records)
    counts = (
        _recipient_tail_loss_character_counts(records)
        if character_counts is None
        else character_counts
    )
    weights: list[float] = []
    for record in records:
        slot = _recipient_slot(record)
        text = slot.get("text") if slot is not None else None
        rare, long = _recipient_tail_loss_flags(text, policy=normalized, character_counts=counts)
        weights.append(
            max(
                1.0,
                float(normalized["rare_character_loss_weight"]) if rare else 1.0,
                float(normalized["long_text_loss_weight"]) if long else 1.0,
            )
        )
    return weights


def _combine_recipient_loss_weights(
    confidence_weights: Sequence[float] | None,
    tail_weights: Sequence[float] | None,
) -> list[float] | None:
    """Combine independent recipient-only loss policies without cross-field effects."""
    if confidence_weights is None and tail_weights is None:
        return None
    if confidence_weights is None:
        return [float(weight) for weight in tail_weights or ()]
    if tail_weights is None:
        return [float(weight) for weight in confidence_weights]
    if len(confidence_weights) != len(tail_weights):
        raise ValueError("recipient confidence and tail loss weights must have the same length")
    return [float(confidence) * float(tail) for confidence, tail in zip(confidence_weights, tail_weights)]


def _recipient_train_augmentation_policy(*, mode: str, seed: int) -> dict[str, object]:
    """Freeze a train-only recipient perturbation policy.

    ``robust_v2`` is still deliberately label preserving: it uses only small
    geometry/contrast/noise changes and a bounded one-dimensional blur.  The
    policy is deterministic per (seed, epoch, receipt id), so a validation or
    held-out test crop is never augmented and Windows worker scheduling cannot
    change the training stream.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("recipient train augmentation seed must be an integer")
    if mode == "none":
        return {"mode": "none"}
    if mode == "light_v1":
        return {
            "mode": "light_v1",
            "seed": int(seed),
            "horizontal_shift_px": 8,
            "vertical_shift_px": 2,
            "contrast_delta": 0.12,
            "noise_std": 0.01,
        }
    if mode == "robust_v2":
        return {
            "mode": "robust_v2",
            "seed": int(seed),
            "horizontal_shift_px": 16,
            "vertical_shift_px": 3,
            "horizontal_scale_delta": 0.06,
            "vertical_scale_delta": 0.04,
            "contrast_delta": 0.18,
            "noise_std": 0.015,
            "blur_probability": 0.25,
        }
    raise ValueError("recipient_train_augmentation must be none, light_v1, or robust_v2")


def _validate_recipient_train_augmentation_policy(policy: object) -> dict[str, object]:
    """Validate a persisted train-only v12 perturbation policy."""
    if not isinstance(policy, Mapping):
        raise ValueError("recipient train augmentation policy is missing or invalid")
    mode = policy.get("mode")
    if mode == "none":
        if set(policy) != {"mode"}:
            raise ValueError("recipient train augmentation policy is invalid")
        return _recipient_train_augmentation_policy(mode="none", seed=0)
    if mode not in {"light_v1", "robust_v2"}:
        raise ValueError("recipient train augmentation policy is invalid")
    expected = _recipient_train_augmentation_policy(mode=str(mode), seed=policy.get("seed"))
    if dict(policy) != expected:
        raise ValueError("recipient train augmentation policy is invalid")
    return expected


def _recipient_train_split_policy(splits: Sequence[str]) -> dict[str, object]:
    """Validate which manifest splits are allowed to supervise recipient CTC.

    The default remains a normal held-out validation setup: train split only.
    A recipient-only Paddle-fit run may deliberately include ``val`` (or, for a
    closed deployment fit, ``test``) as teacher-labelled recipient supervision.
    That mode is useful when Paddle is the accepted truth source, but the
    resulting validation number is transductive and must not be reported as an
    independent generalisation estimate.
    """

    if isinstance(splits, str):
        raise ValueError("recipient_train_splits must be a sequence, not a string")
    allowed = {"train", "val", "test"}
    ordered: list[str] = []
    for split in splits:
        if split not in allowed:
            raise ValueError("recipient_train_splits must contain only train, val, or test")
        if split not in ordered:
            ordered.append(split)
    if not ordered:
        raise ValueError("recipient_train_splits must not be empty")
    if "train" not in ordered:
        raise ValueError("recipient_train_splits must include train")
    if ordered == ["train"]:
        return {
            "mode": "standard_train_only",
            "splits": ordered,
            "warning": None,
        }
    return {
        "mode": "paddle_fit_transductive_v1",
        "splits": ordered,
        "warning": (
            "Recipient-only training includes non-train Paddle-labelled splits. "
            "Validation on an included split measures Paddle-fit on seen teacher targets, "
            "not independent generalisation."
        ),
    }


def _require_manifest_without_test_rows(path: Path) -> None:
    """Reject a full-crop training input before the model loader sees test labels."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}:{line_number}: record must be an object")
            split = row.get("split")
            if split not in {"train", "val"}:
                if split == "test":
                    raise ValueError(
                        "recipient_full_crop_warmstart requires a manifest that physically excludes test rows"
                    )
                raise ValueError(f"{source}:{line_number}: invalid split {split!r}")


def _validate_recipient_sampling_policy(policy: object) -> dict[str, object]:
    """Validate the provenance of v11's receipt-level oversampling.

    The sampler is a training-time choice rather than an inference input, but
    freezing it into the checkpoint and ONNX sidecars makes a candidate
    reproducible and prevents a hand-edited delivery bundle from claiming an
    unknown training distribution.
    """
    if not isinstance(policy, Mapping):
        raise ValueError("v11 recipient sampling policy is missing or invalid")
    mode = policy.get("mode")
    try:
        weight = float(policy.get("recipient_sampling_weight"))
    except (TypeError, ValueError):
        raise ValueError("v11 recipient sampling weight is invalid") from None
    recipient_records = policy.get("recipient_train_records")
    train_records = policy.get("train_records")
    if (
        mode not in {"uniform", "weighted_receipt_sampler_v1", "weighted_receipt_sampler_v2"}
        or not math.isfinite(weight)
        or weight <= 0.0
        or isinstance(recipient_records, bool)
        or not isinstance(recipient_records, int)
        or isinstance(train_records, bool)
        or not isinstance(train_records, int)
        or recipient_records < 0
        or train_records < recipient_records
    ):
        raise ValueError("v11 recipient sampling policy is invalid")
    normalized: dict[str, object] = {
        "mode": mode,
        "recipient_sampling_weight": weight,
        "recipient_train_records": recipient_records,
        "train_records": train_records,
    }
    if mode == "uniform":
        if not math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("v11 uniform recipient sampling must have weight 1.0")
        return normalized
    if policy.get("replacement") is not True:
        raise ValueError("v11 weighted recipient sampling must use replacement")
    seed = policy.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("v11 weighted recipient sampling seed is invalid")
    normalized["replacement"] = True
    normalized["seed"] = seed
    if mode == "weighted_receipt_sampler_v2":
        try:
            rare_weight = float(policy.get("recipient_rare_character_sampling_weight"))
            long_weight = float(policy.get("recipient_long_text_sampling_weight"))
        except (TypeError, ValueError):
            raise ValueError("v12 recipient sampling policy is invalid") from None
        integer_keys = (
            "recipient_rare_character_max_support",
            "recipient_long_text_min_length",
            "recipient_rare_character_train_records",
            "recipient_long_text_train_records",
            "recipient_training_character_count",
        )
        integers: dict[str, int] = {}
        for key in integer_keys:
            value = policy.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("v12 recipient sampling policy is invalid")
            integers[key] = value
        if (
            not math.isfinite(rare_weight)
            or rare_weight <= 0.0
            or not math.isfinite(long_weight)
            or long_weight <= 0.0
            or integers["recipient_rare_character_train_records"] > recipient_records
            or integers["recipient_long_text_train_records"] > recipient_records
        ):
            raise ValueError("v12 recipient sampling policy is invalid")
        normalized.update(
            {
                "recipient_rare_character_max_support": integers["recipient_rare_character_max_support"],
                "recipient_rare_character_sampling_weight": rare_weight,
                "recipient_long_text_min_length": integers["recipient_long_text_min_length"],
                "recipient_long_text_sampling_weight": long_weight,
                "recipient_rare_character_train_records": integers["recipient_rare_character_train_records"],
                "recipient_long_text_train_records": integers["recipient_long_text_train_records"],
                "recipient_training_character_count": integers["recipient_training_character_count"],
            }
        )
    return normalized


def _recipient_artifact_metadata(
    config: UnifiedReaderConfig,
    *,
    recipient_sampling_policy: object | None = None,
    recipient_confidence_policy: object | None = None,
    recipient_tail_loss_policy: object | None = None,
    recipient_train_augmentation_policy: object | None = None,
) -> dict[str, object]:
    """Build frozen fifth-slot metadata for checkpoints and ONNX sidecars."""
    if not _uses_recipient_protocol(config):
        return {}
    metadata: dict[str, object] = {
        "recipient_input_preprocess": _recipient_input_preprocess(config),
    }
    if _is_v11(config) or _uses_v12_recipient_topology(config):
        metadata.update(
            {
                "recipient_value_left_trim": config.recipient_value_left_trim,
                "recipient_hidden_size": _recipient_hidden_size(config),
                "recipient_sampling_policy": _validate_recipient_sampling_policy(recipient_sampling_policy),
            }
        )
        # These are explicit training provenance, never runtime decoding
        # inputs.  Older sidecars omit them and remain loadable.
        if recipient_confidence_policy is not None:
            metadata["recipient_confidence_policy"] = _validate_recipient_confidence_policy(
                recipient_confidence_policy
            )
        if recipient_tail_loss_policy is not None:
            metadata["recipient_tail_loss_policy"] = _validate_recipient_tail_loss_policy(
                recipient_tail_loss_policy
            )
        if recipient_train_augmentation_policy is not None:
            metadata["recipient_train_augmentation_policy"] = _validate_recipient_train_augmentation_policy(
                recipient_train_augmentation_policy
            )
    if _uses_v12_recipient_topology(config):
        metadata.update(
            {
                "recipient_input_name": "recipient_value_image",
                "recipient_input_shape": [1, 1, config.recipient_input_height, config.recipient_input_width],
                "recipient_branch_channels": _recipient_branch_channels(config),
                "recipient_time_steps": _recipient_time_steps(config),
                "recipient_backbone": config.recipient_backbone,
            }
        )
        # Keep legacy v12 sidecars loadable: this additive provenance key is
        # present only when the graph actually contains the new adapter.
        if config.recipient_open_text_layers:
            metadata["recipient_open_text_encoder"] = {
                "mode": (
                    "direct_positional_transformer_ctc_v2"
                    if config.recipient_backbone == "residual_positional_transformer_v2"
                    else "zero_gated_transformer_context_v1"
                ),
                "layers": config.recipient_open_text_layers,
                "heads": config.recipient_open_text_heads,
                "feedforward": int(
                    config.recipient_open_text_feedforward
                    or (_recipient_hidden_size(config) * 2 * 4)
                ),
            }
            if (
                config.recipient_backbone == "residual_positional_transformer_v2"
                or not math.isclose(
                    config.recipient_open_text_dropout, 0.0, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                metadata["recipient_open_text_encoder"]["dropout"] = (
                    config.recipient_open_text_dropout
                )
    return metadata


def _json_safe_value(value: object) -> object:
    """Turn non-finite metrics into standards-compliant JSON ``null`` values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe_value(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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


def build_unified_reader(
    *,
    payment_vocab_size: int,
    config: UnifiedReaderConfig,
    payment_bank_prefix_vocab_size: int | None = None,
    recipient_vocab_size: int | None = None,
    status_text_vocab_size: int | None = None,
) -> Any:
    """Return the shared-trunk reader used for training and ONNX export.

    Architecture v3 deliberately preserves the original module names and
    topology.  That is necessary for strict loading of existing v3 checkpoints.
    Architecture v4 uses a less destructive vertical downsampling path and
    independent amount/time decoders.  v5 retains that compact shared trunk,
    but replaces the destructive ``mean(height)`` text reduction with learned
    per-field vertical reducers and emits structural financial-digit heads in
    the same graph.  v6 preserves a single graph/input but branches CTC and
    format verification after the encoder, and uses a finite train-only bank
    prefix class head.  v7 keeps v6's external protocol but shares the time
    CTC reader with the fixed-format time heads, so their gradients reinforce
    the same punctuation evidence. v8 keeps that compact time/payment protocol
    but attaches finite amount-display grammar heads directly to canonical
    amount CTC state. v9 retains that 14-output protocol and appends a fifth
    free-recipient CTC head. v11 keeps that one-ONNX interface but gives the
    recipient head a larger private recurrent width. v12 retains the one
    ONNX/session delivery model but routes recipient text through a narrow
    high-resolution value-view branch. v3/v4 output tuples are intentionally
    unchanged.
    """
    if payment_vocab_size < 2:
        raise ValueError("payment_vocab_size must include CTC blank plus at least one character")
    config.validate()
    if _uses_modern_protocol(config) and (payment_bank_prefix_vocab_size is None or payment_bank_prefix_vocab_size < 2):
        raise ValueError("v6/v7/v8 needs payment_bank_prefix_vocab_size including __other__ plus one class")
    if _uses_recipient_protocol(config) and (recipient_vocab_size is None or recipient_vocab_size < 2):
        raise ValueError("v9-v13 needs recipient_vocab_size including CTC blank plus at least one character")
    if _is_v13(config) and (status_text_vocab_size is None or status_text_vocab_size < 2):
        raise ValueError("v13 needs status_text_vocab_size including CTC blank plus at least one character")
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

    class ResidualConvBlock(nn.Module):
        """Standard-convolution OCR block used only by the v2 recipient path.

        The old private branch is intentionally depthwise-separable and very
        small.  That is efficient but its held-out recipient ceiling remained
        low even after width/hidden-size sweeps.  This block spends capacity on
        neighbouring stroke interactions before sequence modelling, while
        GroupNorm keeps train/inference behaviour batch-size independent.
        """

        def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int]) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
            self.skip = (
                nn.Identity()
                if stride == (1, 1) and in_channels == out_channels
                else nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                )
            )
            self.activation = nn.SiLU(inplace=True)

        def forward(self, value: Any) -> Any:
            return self.activation(self.main(value) + self.skip(value))

    class VerticalTextReducer(nn.Module):
        """Collapse a fixed feature height without averaging away digit strokes.

        The reader has a static ONNX input shape, so v5 knows the encoder's
        post-stride height.  A depthwise full-height convolution lets each
        channel learn which vertical evidence matters before a cheap pointwise
        projection mixes channels for the recurrent decoder.
        """

        def __init__(self, channels: int, feature_height: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                channels,
                channels,
                kernel_size=(feature_height, 1),
                groups=channels,
                bias=False,
            )
            self.norm = nn.GroupNorm(_group_count(channels), channels)
            self.activation = nn.SiLU(inplace=True)
            self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        def forward(self, value: Any) -> Any:
            value = self.depthwise(value)
            value = self.norm(value)
            value = self.activation(value)
            value = self.pointwise(value)
            return value.squeeze(2)

    class UnifiedFieldReader(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture_version = config.architecture_version
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
            # steps at the default 384px crop width.  v3 reduces vertical
            # resolution by 16; v4 retains twice as much vertical evidence.
            self.encoder = nn.Sequential(
                DepthwiseBlock(first, second, stride=(2, 2)),
                DepthwiseBlock(second, third, stride=(2, 1)),
                DepthwiseBlock(third, fourth, stride=(2 if config.architecture_version == 3 else 1, 1)),
            )
            self.slot_embedding = nn.Parameter(torch.empty(len(_slot_order(config)), fourth, 1, 1))
            nn.init.normal_(self.slot_embedding, std=0.02)
            if config.architecture_version == 3:
                # Do not rename these v3 modules: their state_dict keys are
                # part of the legacy checkpoint compatibility contract.
                self.numeric_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
                self.numeric_classifier = nn.Linear(config.numeric_hidden_size * 2, len(NUMERIC_CHARACTERS) + 1)
            elif _uses_modern_protocol(config):
                feature_height = (config.image_height + 7) // 8
                # Amount/payment CTC and verifier paths deliberately remain
                # independent after the compact CNN encoder.  v6 preserves a
                # separate time verifier as part of its frozen checkpoint
                # topology; v7 instead shares time CTC state with its compact
                # format heads (constructed below without legacy modules).
                self.amount_ctc_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                self.amount_ctc_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
                self.amount_ctc_classifier = nn.Linear(
                    config.numeric_hidden_size * 2, len(_amount_characters(config)) + 1
                )
                self.time_ctc_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                self.time_ctc_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
                self.time_ctc_classifier = nn.Linear(
                    config.numeric_hidden_size * 2, len(V6_TIME_CHARACTERS) + 1
                )
                self.payment_ctc_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                self.payment_ctc_sequence = nn.GRU(fourth, config.payment_hidden_size, bidirectional=True)
                self.payment_ctc_classifier = nn.Linear(config.payment_hidden_size * 2, payment_vocab_size)

                if _is_v13(config):
                    # This is intentionally additive to the frozen v12
                    # topology.  It consumes the existing status crop feature
                    # map but owns every trainable reducer/RNN/classifier
                    # parameter, allowing a v12 -> v13 head-only warm start.
                    self.status_text_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                    self.status_text_sequence = nn.GRU(
                        fourth, config.numeric_hidden_size, bidirectional=True
                    )
                    self.status_text_classifier = nn.Linear(
                        config.numeric_hidden_size * 2, int(status_text_vocab_size)
                    )

                if _uses_recipient_protocol(config):
                    # A recipient is open Chinese text, not a closed merchant
                    # catalogue.  v9-v11 share the visual trunk.  v12 keeps
                    # the one-session graph but gives this open text a narrow
                    # high-resolution value-view branch so thin Chinese glyphs
                    # are not irreversibly compressed with the four short
                    # financial fields.
                    recipient_channels = fourth
                    recipient_feature_height = feature_height
                    if _uses_high_resolution_recipient_input(config):
                        recipient_first = _recipient_branch_channels(config)
                        recipient_second = recipient_first * 2
                        recipient_third = recipient_first * 3
                        recipient_channels = recipient_first * 4
                        self.recipient_stem = nn.Sequential(
                            nn.Conv2d(1, recipient_first, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.GroupNorm(_group_count(recipient_first), recipient_first),
                            nn.SiLU(inplace=True),
                        )
                        if config.recipient_backbone == "residual_positional_transformer_v2":
                            self.recipient_encoder = nn.Sequential(
                                ResidualConvBlock(recipient_first, recipient_second, stride=(2, 2)),
                                ResidualConvBlock(recipient_second, recipient_second, stride=(1, 1)),
                                ResidualConvBlock(recipient_second, recipient_third, stride=(2, 2)),
                                ResidualConvBlock(recipient_third, recipient_channels, stride=(1, 1)),
                            )
                        else:
                            self.recipient_encoder = nn.Sequential(
                                DepthwiseBlock(recipient_first, recipient_second, stride=(2, 2)),
                                DepthwiseBlock(recipient_second, recipient_third, stride=(2, 1)),
                                DepthwiseBlock(recipient_third, recipient_channels, stride=(1, 1)),
                            )
                        recipient_feature_height = (config.recipient_input_height + 7) // 8
                    self.recipient_ctc_vertical_reducer = VerticalTextReducer(
                        recipient_channels, recipient_feature_height
                    )
                    recipient_model_width = _recipient_hidden_size(config) * 2
                    if config.recipient_backbone == "residual_positional_transformer_v2":
                        self.recipient_input_projection = nn.Linear(
                            recipient_channels, recipient_model_width
                        )
                        self.recipient_position_embedding = nn.Parameter(
                            torch.empty(_recipient_time_steps(config), 1, recipient_model_width)
                        )
                        nn.init.normal_(self.recipient_position_embedding, std=0.02)
                        recipient_model_width = _recipient_hidden_size(config) * 2
                        recipient_feedforward = int(
                            config.recipient_open_text_feedforward or recipient_model_width * 4
                        )
                        open_text_layer = nn.TransformerEncoderLayer(
                            d_model=recipient_model_width,
                            nhead=config.recipient_open_text_heads,
                            dim_feedforward=recipient_feedforward,
                            dropout=config.recipient_open_text_dropout,
                            activation="gelu",
                            batch_first=False,
                            norm_first=True,
                        )
                        self.recipient_open_text_encoder = nn.TransformerEncoder(
                            open_text_layer,
                            num_layers=config.recipient_open_text_layers,
                            enable_nested_tensor=False,
                        )
                        self.recipient_classifier = nn.Linear(
                            recipient_model_width, int(recipient_vocab_size)
                        )
                    else:
                        self.recipient_ctc_sequence = nn.GRU(
                            recipient_channels, _recipient_hidden_size(config), bidirectional=True
                        )
                        self.recipient_classifier = nn.Linear(
                            recipient_model_width, int(recipient_vocab_size)
                        )
                    if (
                        config.recipient_backbone == "legacy_depthwise_gru_v1"
                        and config.recipient_open_text_layers
                    ):
                        recipient_feedforward = int(
                            config.recipient_open_text_feedforward or recipient_model_width * 4
                        )
                        open_text_layer = nn.TransformerEncoderLayer(
                            d_model=recipient_model_width,
                            nhead=config.recipient_open_text_heads,
                            dim_feedforward=recipient_feedforward,
                            dropout=config.recipient_open_text_dropout,
                            activation="gelu",
                            batch_first=False,
                            norm_first=True,
                        )
                        self.recipient_open_text_encoder = nn.TransformerEncoder(
                            open_text_layer,
                            num_layers=config.recipient_open_text_layers,
                            enable_nested_tensor=False,
                        )
                        # tanh keeps the residual bounded; zero makes the new
                        # graph exactly reproduce the seed at initialisation.
                        self.recipient_open_text_gate = nn.Parameter(torch.zeros(()))

                if _uses_v8_protocol(config):
                    # The v8 CTC stream owns signed canonical digits. These
                    # finite heads only choose display grammar; they cannot
                    # invent or replace a monetary digit.
                    self.amount_currency_style_classifier = nn.Linear(
                        config.numeric_hidden_size * 2, len(AMOUNT_CURRENCY_STYLE_CLASSES)
                    )
                    self.amount_grouped_thousands_classifier = nn.Linear(
                        config.numeric_hidden_size * 2, len(AMOUNT_GROUPED_THOUSANDS_CLASSES)
                    )
                    self.amount_sign_position_classifier = nn.Linear(
                        config.numeric_hidden_size * 2, len(AMOUNT_SIGN_POSITION_CLASSES)
                    )
                else:
                    # Do not rename/remove these v6/v7 modules: their
                    # state_dict keys are frozen checkpoint compatibility.
                    self.amount_verifier_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                    self.amount_verifier_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
                    self.amount_sign_classifier = nn.Linear(config.numeric_hidden_size * 2, len(AMOUNT_SIGN_CLASSES))
                    self.amount_length_classifier = nn.Linear(config.numeric_hidden_size * 2, AMOUNT_MAX_INTEGER_DIGITS)
                    self.amount_digit_classifier = nn.Linear(config.numeric_hidden_size * 2, AMOUNT_DIGIT_SLOTS * 10)

                if config.architecture_version == 6:
                    # Do not rename/remove these v6 modules: an existing v6
                    # checkpoint must reproduce its original forward/export
                    # path exactly.  v7 has its own kind and omits them.
                    self.time_verifier_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                    self.time_verifier_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
                self.time_format_classifier = nn.Linear(
                    config.numeric_hidden_size * 2, len(TIME_DISPLAY_FORMAT_CLASSES)
                )
                self.time_digit_classifier = nn.Linear(
                    config.numeric_hidden_size * 2, TIME_DISPLAY_DIGIT_SLOTS * 10
                )

                self.payment_verifier_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                self.payment_prefix_sequence = nn.GRU(fourth, config.payment_hidden_size, bidirectional=True)
                self.payment_prefix_classifier = nn.Linear(config.payment_hidden_size * 2, payment_vocab_size)
                self.payment_bank_prefix_classifier = nn.Linear(
                    config.payment_hidden_size * 2, int(payment_bank_prefix_vocab_size)
                )
                self.payment_tail_digit_classifier = nn.Linear(
                    config.payment_hidden_size * 2, PAYMENT_TAIL_DIGIT_SLOTS * 10
                )
                self.payment_structure_classifier = nn.Linear(
                    config.payment_hidden_size * 2, len(PAYMENT_STRUCTURE_CLASSES)
                )
                self.payment_parentheses_classifier = nn.Linear(
                    config.payment_hidden_size * 2, len(PAYMENT_PARENTHESIS_CLASSES)
                )
            else:
                self.amount_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
                self.amount_classifier = nn.Linear(config.numeric_hidden_size * 2, len(NUMERIC_CHARACTERS) + 1)
                self.time_sequence = nn.GRU(fourth, config.numeric_hidden_size, bidirectional=True)
                self.time_classifier = nn.Linear(config.numeric_hidden_size * 2, len(NUMERIC_CHARACTERS) + 1)
            if not _uses_modern_protocol(config):
                self.payment_sequence = nn.GRU(fourth, config.payment_hidden_size, bidirectional=True)
                self.payment_classifier = nn.Linear(config.payment_hidden_size * 2, payment_vocab_size)
            if config.architecture_version == 5:
                feature_height = (config.image_height + 7) // 8
                self.amount_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                self.time_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                self.payment_vertical_reducer = VerticalTextReducer(fourth, feature_height)
                self.payment_prefix_sequence = nn.GRU(fourth, config.payment_hidden_size, bidirectional=True)
                self.payment_prefix_classifier = nn.Linear(config.payment_hidden_size * 2, payment_vocab_size)
                self.amount_length_classifier = nn.Linear(
                    config.numeric_hidden_size * 2, AMOUNT_MAX_INTEGER_DIGITS
                )
                self.amount_digit_classifier = nn.Linear(
                    config.numeric_hidden_size * 2, AMOUNT_DIGIT_SLOTS * 10
                )
                self.time_digit_classifier = nn.Linear(
                    config.numeric_hidden_size * 2, TIME_DIGIT_SLOTS * 10
                )
                self.time_hour_width_classifier = nn.Linear(config.numeric_hidden_size * 2, 2)
                self.payment_tail_digit_classifier = nn.Linear(
                    config.payment_hidden_size * 2, PAYMENT_TAIL_DIGIT_SLOTS * 10
                )
                self.payment_structure_classifier = nn.Linear(
                    config.payment_hidden_size * 2, len(PAYMENT_STRUCTURE_CLASSES)
                )
                self.payment_parentheses_classifier = nn.Linear(
                    config.payment_hidden_size * 2, len(PAYMENT_PARENTHESIS_CLASSES)
                )
            self.status_pool = nn.AdaptiveAvgPool2d((1, config.pooled_width))
            self.status_classifier = nn.Linear(fourth * config.pooled_width, len(STATUS_CLASSES))

        def forward(self, field_images: Any, recipient_value_image: Any | None = None) -> tuple[Any, ...]:
            # Training input: [batch, fixed slot count, channel=1, height, width].
            expected_slots = len(_slot_order(config))
            if field_images.ndim != 5 or field_images.shape[1] != expected_slots or field_images.shape[2] != 1:
                raise ValueError(
                    f"field_images must have shape [batch,{expected_slots},1,height,width]"
                )
            batch, slots, channels, height, width = field_images.shape
            if _uses_high_resolution_recipient_input(config):
                if recipient_value_image is None:
                    raise ValueError("v12 field reader requires recipient_value_image")
                expected_recipient_shape = [batch, 1, config.recipient_input_height, config.recipient_input_width]
                if list(recipient_value_image.shape) != expected_recipient_shape:
                    raise ValueError(
                        "recipient_value_image must have shape "
                        f"[batch,1,{config.recipient_input_height},{config.recipient_input_width}]"
                    )
                shared_slots = 4
            else:
                shared_slots = slots
            shared_images = field_images[:, :shared_slots]
            encoded = self.encoder(
                self.stem(shared_images.reshape(batch * shared_slots, channels, height, width))
            )
            _, feature_channels, feature_height, feature_width = encoded.shape
            encoded = encoded.reshape(batch, shared_slots, feature_channels, feature_height, feature_width)
            encoded = encoded + self.slot_embedding[:shared_slots].unsqueeze(0)

            if self.architecture_version == 3:
                # amount/time slots share one numeric CTC projection but retain
                # a slot embedding, allowing the decoder to distinguish '.'
                # and ':'.  This exact v3 branch preserves checkpoint loading.
                numeric_features = encoded[:, :2].mean(dim=3)  # [batch,2,C,T]
                numeric_sequence = numeric_features.permute(3, 0, 1, 2).reshape(
                    feature_width, batch * 2, feature_channels
                )
                numeric_sequence, _ = self.numeric_sequence(numeric_sequence)
                numeric_logits = self.numeric_classifier(numeric_sequence).reshape(
                    feature_width, batch, 2, len(NUMERIC_CHARACTERS) + 1
                )
            elif self.architecture_version >= 6:
                amount_ctc_features = self.amount_ctc_vertical_reducer(encoded[:, 0]).permute(2, 0, 1)
                amount_ctc_sequence, amount_ctc_hidden = self.amount_ctc_sequence(amount_ctc_features)
                amount_logits = self.amount_ctc_classifier(amount_ctc_sequence)
                time_ctc_features = self.time_ctc_vertical_reducer(encoded[:, 1]).permute(2, 0, 1)
                time_ctc_sequence, time_ctc_hidden = self.time_ctc_sequence(time_ctc_features)
                time_logits = self.time_ctc_classifier(time_ctc_sequence)
            else:
                # v4 keeps a common visual trunk but lets amount and time
                # specialize their recurrent decoder and final character logits.
                if self.architecture_version == 5:
                    amount_features = self.amount_vertical_reducer(encoded[:, 0]).permute(2, 0, 1)
                    time_features = self.time_vertical_reducer(encoded[:, 1]).permute(2, 0, 1)
                else:
                    amount_features = encoded[:, 0].mean(dim=2).permute(2, 0, 1)
                    time_features = encoded[:, 1].mean(dim=2).permute(2, 0, 1)
                amount_sequence, amount_hidden = self.amount_sequence(amount_features)
                amount_logits = self.amount_classifier(amount_sequence)
                time_sequence, time_hidden = self.time_sequence(time_features)
                time_logits = self.time_classifier(time_sequence)
                numeric_logits = torch.stack((amount_logits, time_logits), dim=2)

            if self.architecture_version >= 6:
                payment_features = self.payment_ctc_vertical_reducer(encoded[:, 3])  # [batch,C,T]
                payment_sequence, _ = self.payment_ctc_sequence(payment_features.permute(2, 0, 1))
                payment_logits = self.payment_ctc_classifier(payment_sequence)
                if _uses_recipient_protocol(config):
                    if _uses_high_resolution_recipient_input(config):
                        assert recipient_value_image is not None
                        recipient_encoded = self.recipient_encoder(self.recipient_stem(recipient_value_image))
                        recipient_features = self.recipient_ctc_vertical_reducer(recipient_encoded)
                    else:
                        recipient_features = self.recipient_ctc_vertical_reducer(encoded[:, 4])
                    recipient_sequence = recipient_features.permute(2, 0, 1)
                    if config.recipient_backbone == "residual_positional_transformer_v2":
                        recipient_sequence = self.recipient_input_projection(recipient_sequence)
                        recipient_sequence = recipient_sequence + self.recipient_position_embedding
                        recipient_sequence = self.recipient_open_text_encoder(recipient_sequence)
                    else:
                        recipient_sequence, _ = self.recipient_ctc_sequence(recipient_sequence)
                        if config.recipient_open_text_layers:
                            refined_recipient_sequence = self.recipient_open_text_encoder(recipient_sequence)
                            recipient_sequence = recipient_sequence + torch.tanh(
                                self.recipient_open_text_gate
                            ) * (refined_recipient_sequence - recipient_sequence)
                    recipient_logits = self.recipient_classifier(recipient_sequence)
            elif self.architecture_version == 5:
                payment_features = self.payment_vertical_reducer(encoded[:, 3])  # [batch,C,T]
                payment_sequence = payment_features.permute(2, 0, 1)
                payment_sequence, payment_hidden = self.payment_sequence(payment_sequence)
                payment_logits = self.payment_classifier(payment_sequence)  # [T,batch,class]
            else:
                payment_features = encoded[:, 3].mean(dim=2)  # [batch,C,T]
                payment_sequence = payment_features.permute(2, 0, 1)
                payment_sequence, payment_hidden = self.payment_sequence(payment_sequence)
                payment_logits = self.payment_classifier(payment_sequence)  # [T,batch,class]

            status_features = self.status_pool(encoded[:, 2]).flatten(1)
            status_logits = self.status_classifier(status_features)
            if _is_v13(config):
                status_text_features = self.status_text_vertical_reducer(encoded[:, 2]).permute(2, 0, 1)
                status_text_sequence, _ = self.status_text_sequence(status_text_features)
                status_text_logits = self.status_text_classifier(status_text_sequence)
            if self.architecture_version >= 6:
                if _uses_v8_protocol(config):
                    amount_summary = torch.cat((amount_ctc_hidden[0], amount_ctc_hidden[1]), dim=1)
                else:
                    amount_verifier_features = self.amount_verifier_vertical_reducer(encoded[:, 0]).permute(2, 0, 1)
                    _, amount_verifier_hidden = self.amount_verifier_sequence(amount_verifier_features)
                    amount_summary = torch.cat((amount_verifier_hidden[0], amount_verifier_hidden[1]), dim=1)
                if self.architecture_version == 6:
                    # Frozen v6 topology: keep its checkpoint semantics and
                    # independent verifier branch exactly as trained.
                    time_verifier_features = self.time_verifier_vertical_reducer(encoded[:, 1]).permute(2, 0, 1)
                    _, time_verifier_hidden = self.time_verifier_sequence(time_verifier_features)
                    time_summary = torch.cat((time_verifier_hidden[0], time_verifier_hidden[1]), dim=1)
                else:
                    # v7: CTC and the fixed-format time heads train on the
                    # same recurrent state.  CTC supplies per-timestep
                    # gradients for thin colon/digit evidence while the
                    # separate projections retain the compact 14-output ONNX
                    # delivery interface.
                    time_summary = torch.cat((time_ctc_hidden[0], time_ctc_hidden[1]), dim=1)
                payment_verifier_features = self.payment_verifier_vertical_reducer(encoded[:, 3]).permute(2, 0, 1)
                payment_prefix_sequence, payment_prefix_hidden = self.payment_prefix_sequence(payment_verifier_features)
                payment_prefix_logits = self.payment_prefix_classifier(payment_prefix_sequence)
                payment_summary = torch.cat((payment_prefix_hidden[0], payment_prefix_hidden[1]), dim=1)
                if _uses_v8_protocol(config):
                    amount_currency_style_logits = self.amount_currency_style_classifier(amount_summary)
                    amount_grouped_thousands_logits = self.amount_grouped_thousands_classifier(amount_summary)
                    amount_sign_position_logits = self.amount_sign_position_classifier(amount_summary)
                else:
                    amount_sign_logits = self.amount_sign_classifier(amount_summary)
                    amount_length_logits = self.amount_length_classifier(amount_summary)
                    amount_digit_logits = self.amount_digit_classifier(amount_summary).reshape(batch, AMOUNT_DIGIT_SLOTS, 10)
                time_format_logits = self.time_format_classifier(time_summary)
                time_digit_logits = self.time_digit_classifier(time_summary).reshape(batch, TIME_DISPLAY_DIGIT_SLOTS, 10)
                payment_bank_prefix_logits = self.payment_bank_prefix_classifier(payment_summary)
                payment_tail_digit_logits = self.payment_tail_digit_classifier(payment_summary).reshape(
                    batch, PAYMENT_TAIL_DIGIT_SLOTS, 10
                )
                payment_structure_logits = self.payment_structure_classifier(payment_summary)
                payment_parentheses_logits = self.payment_parentheses_classifier(payment_summary)
                if _uses_v8_protocol(config):
                    v8_outputs = (
                        amount_logits,
                        time_logits,
                        payment_logits,
                        status_logits,
                        amount_currency_style_logits,
                        amount_grouped_thousands_logits,
                        amount_sign_position_logits,
                        time_format_logits,
                        time_digit_logits,
                        payment_prefix_logits,
                        payment_bank_prefix_logits,
                        payment_tail_digit_logits,
                        payment_structure_logits,
                        payment_parentheses_logits,
                    )
                    if _uses_recipient_protocol(config):
                        recipient_outputs = v8_outputs + (recipient_logits,)
                        if _is_v13(config):
                            return recipient_outputs + (status_text_logits,)
                        return recipient_outputs
                    return v8_outputs
                return (
                    amount_logits,
                    time_logits,
                    payment_logits,
                    status_logits,
                    amount_sign_logits,
                    amount_length_logits,
                    amount_digit_logits,
                    time_format_logits,
                    time_digit_logits,
                    payment_prefix_logits,
                    payment_bank_prefix_logits,
                    payment_tail_digit_logits,
                    payment_structure_logits,
                    payment_parentheses_logits,
                )
            if self.architecture_version == 5:
                amount_summary = torch.cat((amount_hidden[0], amount_hidden[1]), dim=1)
                time_summary = torch.cat((time_hidden[0], time_hidden[1]), dim=1)
                payment_prefix_sequence, payment_prefix_hidden = self.payment_prefix_sequence(payment_features.permute(2, 0, 1))
                payment_prefix_logits = self.payment_prefix_classifier(payment_prefix_sequence)
                payment_summary = torch.cat((payment_prefix_hidden[0], payment_prefix_hidden[1]), dim=1)
                amount_length_logits = self.amount_length_classifier(amount_summary)
                amount_digit_logits = self.amount_digit_classifier(amount_summary).reshape(
                    batch, AMOUNT_DIGIT_SLOTS, 10
                )
                time_digit_logits = self.time_digit_classifier(time_summary).reshape(batch, TIME_DIGIT_SLOTS, 10)
                time_hour_width_logits = self.time_hour_width_classifier(time_summary)
                payment_tail_digit_logits = self.payment_tail_digit_classifier(payment_summary).reshape(
                    batch, PAYMENT_TAIL_DIGIT_SLOTS, 10
                )
                payment_structure_logits = self.payment_structure_classifier(payment_summary)
                payment_parentheses_logits = self.payment_parentheses_classifier(payment_summary)
                return (
                    numeric_logits,
                    payment_logits,
                    status_logits,
                    amount_length_logits,
                    amount_digit_logits,
                    time_digit_logits,
                    time_hour_width_logits,
                    payment_prefix_logits,
                    payment_tail_digit_logits,
                    payment_structure_logits,
                    payment_parentheses_logits,
                )
            return numeric_logits, payment_logits, status_logits

    return UnifiedFieldReader()


def preprocess_image(
    image_path: Path,
    *,
    config: UnifiedReaderConfig,
    horizontal_alignment: str = "center",
    left_crop_fraction: float = 0.0,
    output_height: int | None = None,
    output_width: int | None = None,
) -> np.ndarray:
    """Return one grayscale crop as ``[1,H,W]`` float32 with white letterbox.

    v5 places text fields against the right edge.  That makes fixed-position
    decimal, time and card-tail auxiliary heads deterministic without adding a
    second input image or a second ONNX session.  v3/v4 keep the historical
    centred letterbox exactly.
    """
    if horizontal_alignment not in {"center", "right"}:
        raise ValueError("horizontal_alignment must be center or right")
    if not math.isfinite(left_crop_fraction) or not 0.0 <= left_crop_fraction < 1.0:
        raise ValueError("left_crop_fraction must be in [0, 1)")
    target_height = config.image_height if output_height is None else int(output_height)
    target_width = config.image_width if output_width is None else int(output_width)
    if target_height < 1 or target_width < 1:
        raise ValueError("output_height and output_width must be positive")
    with Image.open(image_path) as image:
        gray = image.convert("L")
        if left_crop_fraction:
            left = min(gray.width - 1, max(0, int(round(gray.width * left_crop_fraction))))
            gray = gray.crop((left, 0, gray.width, gray.height))
        scale = min(target_width / gray.width, target_height / gray.height)
        width = max(1, min(target_width, int(round(gray.width * scale))))
        height = max(1, min(target_height, int(round(gray.height * scale))))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        gray = gray.resize((width, height), resampling)
        canvas = np.full((target_height, target_width), 255, dtype=np.uint8)
        top = (target_height - height) // 2
        left = target_width - width if horizontal_alignment == "right" else (target_width - width) // 2
        canvas[top : top + height, left : left + width] = np.asarray(gray, dtype=np.uint8)
    return (canvas.astype(np.float32) / 255.0)[np.newaxis, :, :]


def _blank_image(config: UnifiedReaderConfig) -> np.ndarray:
    return np.ones((1, config.image_height, config.image_width), dtype=np.float32)


def _blank_recipient_value_image(config: UnifiedReaderConfig) -> np.ndarray:
    """Return v12's fixed high-resolution white placeholder."""
    return np.ones((1, config.recipient_input_height, config.recipient_input_width), dtype=np.float32)


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
    if field in {"amount", "time", "payment_method_field", "recipient_field"}:
        text = slot.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{records_path}:{line_number}: slot {field} must have a non-empty CTC target")
        if field == "amount" and (
            not all(character in V6_AMOUNT_CHARACTERS for character in text) or text.count(".") != 1
        ):
            raise ValueError(f"{records_path}:{line_number}: amount CTC target is invalid")
        if field == "time" and (
            not all(character in V6_TIME_CHARACTERS for character in text) or text.count(":") not in {1, 2}
        ):
            raise ValueError(f"{records_path}:{line_number}: time CTC target is invalid")
        if field == "payment_method_field" and any(not character.isprintable() for character in text):
            raise ValueError(f"{records_path}:{line_number}: payment CTC target contains a non-printable character")
        if field == "recipient_field" and any(not character.isprintable() for character in text):
            raise ValueError(f"{records_path}:{line_number}: recipient CTC target contains a non-printable character")
    else:
        class_name = slot.get("class_name")
        if class_name not in STATUS_CLASSES:
            raise ValueError(f"{records_path}:{line_number}: status class must be one of {','.join(STATUS_CLASSES)}")
        text = slot.get("text")
        if text is not None:
            if not isinstance(text, str) or not text or clean_text(text) != text:
                raise ValueError(f"{records_path}:{line_number}: status CTC target must be clean non-empty text")
            if any(not character.isprintable() for character in text):
                raise ValueError(f"{records_path}:{line_number}: status CTC target contains a non-printable character")
            if normalize_status(text) != class_name:
                raise ValueError(
                    f"{records_path}:{line_number}: status CTC target does not normalize to class_name"
                )
    return slot


def _validate_v10_recipient_slot(
    slot: Mapping[str, object] | None,
    *,
    records_path: Path,
    line_number: int,
) -> None:
    """Reject v10 manifests whose CTC and business-value targets disagree.

    v10's recipient CTC head learns the *whole visible row*, while downstream
    business comparison reads the merchant value extracted from that row.  A
    hand-edited or legacy value-only payload would silently recreate the v9
    label/pixel mismatch, so keep this relationship explicit at load time.
    """
    if slot is None:
        return
    text = slot.get("text")
    visible_text = slot.get("recipient_visible_text")
    recipient_value = slot.get("recipient_value")
    semantic_value = slot.get("semantic_value")
    if not isinstance(text, str) or clean_text(text) != text:
        raise ValueError(
            f"{records_path}:{line_number}: v10 recipient CTC target must be clean visible text"
        )
    if not isinstance(visible_text, str) or visible_text != text:
        raise ValueError(
            f"{records_path}:{line_number}: v10 recipient_visible_text must equal the CTC target"
        )
    expected_value = _recipient_value_from_visible_text(text)
    if expected_value is None:
        raise ValueError(
            f"{records_path}:{line_number}: v10 recipient visible row has no extractable merchant value"
        )
    if not isinstance(recipient_value, str) or recipient_value != expected_value:
        raise ValueError(
            f"{records_path}:{line_number}: v10 recipient_value must match the value extracted from visible text"
        )
    if not isinstance(semantic_value, str) or semantic_value != expected_value:
        raise ValueError(
            f"{records_path}:{line_number}: v10 recipient semantic_value must match the value extracted from visible text"
        )


def _validate_anchored_recipient_slot(
    slot: Mapping[str, object] | None,
    *,
    records_path: Path,
    line_number: int,
) -> None:
    """Validate the v11/v12 strict pixel-to-value recipient contract.

    The v11 CTC target is only the merchant value, while the fifth image is
    deterministically cropped from the right side of an anchored visible row.
    It must never load a v9 value-only label whose image still contains an
    unrelated row or a v10 full-row label.
    """
    if slot is None:
        return
    text = slot.get("text")
    visible_text = slot.get("recipient_visible_text")
    recipient_value = slot.get("recipient_value")
    recipient_label = slot.get("recipient_label")
    if not isinstance(text, str) or clean_text(text) != text:
        raise ValueError(f"{records_path}:{line_number}: anchored recipient CTC target must be clean value text")
    if not isinstance(visible_text, str) or clean_text(visible_text) != visible_text:
        raise ValueError(f"{records_path}:{line_number}: anchored recipient visible text must be clean")
    parsed = parse_anchored_recipient_row(visible_text)
    if parsed is None:
        raise ValueError(f"{records_path}:{line_number}: anchored recipient row must begin with an anchored label")
    label, expected_value = parsed
    if recipient_label != label:
        raise ValueError(f"{records_path}:{line_number}: anchored recipient label does not match visible row")
    if text != expected_value or recipient_value != expected_value:
        raise ValueError(f"{records_path}:{line_number}: anchored recipient target must match anchored row value")
    if slot.get("recipient_quality_policy") != "anchored_value_right_crop_v1":
        raise ValueError(f"{records_path}:{line_number}: anchored recipient quality policy is unsupported")


def _validate_v13_status_text_slot(
    slot: Mapping[str, object] | None,
    *,
    records_path: Path,
    line_number: int,
) -> None:
    """Lock v13's CTC target to Paddle-grounded visible Chinese text."""
    if slot is None or slot.get("text") is None:
        return
    text = slot.get("text")
    visible_text = slot.get("status_visible_text")
    if not isinstance(text, str) or visible_text != text:
        raise ValueError(
            f"{records_path}:{line_number}: v13 status_visible_text must equal the CTC target"
        )
    if not text or any(not _is_cjk_ideograph(character) for character in text):
        raise ValueError(
            f"{records_path}:{line_number}: v13 transfer-status CTC target must contain only visible CJK text"
        )
    if text not in STATUS_VISIBLE_CJK_TEXTS:
        raise ValueError(
            f"{records_path}:{line_number}: v13 transfer-status CTC target is not an audited visible status phrase"
        )


def load_records(
    records_path: Path,
    *,
    dataset_root: Path | None = None,
    config: UnifiedReaderConfig | None = None,
) -> list[dict[str, object]]:
    """Load receipt-level records and protect train/val/test group isolation."""
    if config is not None:
        config.validate()
    slot_order = _slot_order(config) if config is not None else SLOT_ORDER
    expected_dataset_kind = (
        DATASET_KIND_V13
        if config is not None and _is_v13(config)
        else DATASET_KIND_V12
        if config is not None and _is_v12(config)
        else DATASET_KIND_V11
        if config is not None and _is_v11(config)
        else DATASET_KIND_V10
        if config is not None and _is_v10(config)
        else DATASET_KIND_V9
        if slot_order == V9_SLOT_ORDER
        else DATASET_KIND_V8
    )
    records_path = records_path.resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    contract_path = records_path.parent / "dataset.contract.json"
    contract_status_text_characters: list[str] | None = None
    if contract_path.is_file():
        contract = _load_json_object(contract_path)
        if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") != expected_dataset_kind:
            raise ValueError(f"{contract_path}: unsupported unified dataset contract")
        if contract.get("slot_order") != list(slot_order) or contract.get("status_classes") != list(STATUS_CLASSES):
            raise ValueError(f"{contract_path}: slot order or status classes do not match the unified reader")
        if config is not None and _is_v13(config):
            raw_status_characters = contract.get("status_text_charset")
            if (
                contract.get("status_text_target") != STATUS_TEXT_TARGET
                or contract.get("status_text_charset_source") != STATUS_TEXT_CHARSET_SOURCE
                or not isinstance(raw_status_characters, list)
                or not raw_status_characters
                or raw_status_characters != sorted(raw_status_characters)
                or len(set(raw_status_characters)) != len(raw_status_characters)
                or not all(
                    isinstance(character, str)
                    and len(character) == 1
                    and character.isprintable()
                    for character in raw_status_characters
                )
                or contract.get("status_text_charset_sha256")
                != hashlib.sha256("".join(raw_status_characters).encode("utf-8")).hexdigest()
            ):
                raise ValueError(f"{contract_path}: v13 status-text dataset contract is invalid")
            contract_status_text_characters = list(raw_status_characters)
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
            unknown_slots = sorted(set(slots) - set(slot_order))
            if unknown_slots:
                raise ValueError(f"{records_path}:{line_number}: unknown unified slot(s): {','.join(unknown_slots)}")
            declared_order = raw.get("slot_order")
            if declared_order is not None and declared_order != list(slot_order):
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
                for field in slot_order
            }
            if config is not None and _is_v10(config):
                _validate_v10_recipient_slot(
                    parsed_slots.get("recipient_field"),
                    records_path=records_path,
                    line_number=line_number,
                )
            if config is not None and (_is_v11(config) or _uses_v12_recipient_topology(config)):
                _validate_anchored_recipient_slot(
                    parsed_slots.get("recipient_field"),
                    records_path=records_path,
                    line_number=line_number,
                )
            if config is not None and _is_v13(config):
                _validate_v13_status_text_slot(
                    parsed_slots.get("transfer_status"),
                    records_path=records_path,
                    line_number=line_number,
                )
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
    if config is not None and _is_v13(config) and contract_status_text_characters is not None:
        observed_status_characters = sorted(
            {
                character
                for record in records
                if record["split"] == "train"
                for text in [_slot_text(record, "transfer_status")]
                if text is not None
                for character in text
            }
        )
        if observed_status_characters != contract_status_text_characters:
            raise ValueError(
                f"{contract_path}: v13 status-text charset does not match train manifest labels"
            )
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


def _recipient_charset(records: Iterable[Mapping[str, object]]) -> list[str]:
    """Freeze a train-only Unicode character set for recipient CTC.

    This is deliberately not a merchant classifier.  A character seen only in
    held-out data stays out of the deployed alphabet and is recorded as OOV
    evidence instead of leaking validation/test text into the model.
    """
    characters = sorted(
        {
            character
            for record in records
            for slot in [dict(record["slots"]).get("recipient_field")]
            if isinstance(slot, Mapping)
            for character in str(slot["text"])
        }
    )
    if not characters:
        raise ValueError("No recipient_field CTC labels remain in the training split")
    return characters


def _recipient_oov_by_split(
    records: Iterable[Mapping[str, object]], *, characters: Sequence[str]
) -> dict[str, dict[str, int]]:
    known = set(characters)
    counters: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    for record in records:
        text = _slot_text(record, "recipient_field")
        if text is None:
            continue
        split = str(record["split"])
        counters[split]["records"] += 1
        if any(character not in known for character in text):
            counters[split]["oov_records"] += 1
    return {
        split: {
            "records": int(counters[split]["records"]),
            "oov_records": int(counters[split]["oov_records"]),
        }
        for split in ("train", "val", "test")
    }


def _validated_recipient_oov_audit(
    value: object, *, source: str
) -> dict[str, dict[str, int]]:
    """Validate the frozen recipient audit persisted by a seed artifact."""
    if not isinstance(value, Mapping) or set(value) != {"train", "val", "test"}:
        raise ValueError(f"{source} recipient OOV audit is invalid")
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        audit = value[split]
        if not isinstance(audit, Mapping) or set(audit) != {"records", "oov_records"}:
            raise ValueError(f"{source} recipient OOV audit is invalid")
        records = audit.get("records")
        oov_records = audit.get("oov_records")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or isinstance(oov_records, bool)
            or not isinstance(oov_records, int)
            or records < 0
            or oov_records < 0
            or oov_records > records
        ):
            raise ValueError(f"{source} recipient OOV audit is invalid")
        result[split] = {"records": records, "oov_records": oov_records}
    if result["train"]["oov_records"] != 0:
        raise ValueError(f"{source} recipient train split must not contain OOV characters")
    return result


def _status_text_charset(records: Iterable[Mapping[str, object]]) -> list[str]:
    """Freeze v13's visible transfer-status CTC alphabet from train only."""
    characters = sorted(
        {
            character
            for record in records
            for text in [_slot_text(record, "transfer_status")]
            if text is not None
            for character in text
        }
    )
    if not characters:
        raise ValueError("No visible transfer_status CTC labels remain in the training split")
    return characters


def _status_text_oov_by_split(
    records: Iterable[Mapping[str, object]], *, characters: Sequence[str]
) -> dict[str, dict[str, object]]:
    known = set(characters)
    counters: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    examples: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}
    for record in records:
        text = _slot_text(record, "transfer_status")
        if text is None:
            continue
        split = str(record["split"])
        counters[split]["records"] += 1
        unknown = sorted(set(text) - known)
        if unknown:
            counters[split]["oov_records"] += 1
            counters[split]["oov_characters"] += len(unknown)
            if len(examples[split]) < 20:
                examples[split].append(
                    {"id": record["id"], "characters": "".join(unknown), "text": text}
                )
    return {
        split: {
            "records": int(counters[split]["records"]),
            "oov_records": int(counters[split]["oov_records"]),
            "oov_characters": int(counters[split]["oov_characters"]),
            "examples": examples[split],
        }
        for split in ("train", "val", "test")
    }


def _payment_bank_prefix_classes(
    records: Iterable[Mapping[str, object]], *, min_support: int
) -> tuple[list[str], dict[str, int]]:
    """Build a train-only finite bank-prefix vocabulary for v6.

    The literal ``__other__`` is always index zero.  Low-frequency prefixes
    never leak from validation/test into the model class map; they become
    review-only ``__other__`` candidates at inference.
    """
    if min_support <= 0:
        raise ValueError("payment_bank_prefix_min_support must be positive")
    counts = Counter(
        target
        for record in records
        for target in [_payment_bank_prefix_target(record)]
        if target is not None
    )
    retained = sorted(prefix for prefix, count in counts.items() if count >= min_support)
    classes = [PAYMENT_BANK_OTHER_CLASS, *retained]
    if len(classes) < 2:
        raise ValueError(
            "No payment bank prefix has enough train samples for v6 classification; "
            "lower --payment-bank-prefix-min-support or build more teacher labels."
        )
    return classes, {prefix: int(counts[prefix]) for prefix in retained}


def _payment_bank_prefix_retained_counts(
    records: Iterable[Mapping[str, object]], *, classes: Sequence[str]
) -> dict[str, int]:
    """Count the effective non-``__other__`` bank labels in a train split."""
    counts = Counter(
        target
        for record in records
        for target in [_payment_bank_prefix_target(record)]
        if target is not None
    )
    return {prefix: int(counts[prefix]) for prefix in list(classes)[1:]}


def _payment_bank_prefix_class_target(
    record: Mapping[str, object], *, classes: Sequence[str]
) -> int | None:
    prefix = _payment_bank_prefix_target(record)
    if prefix is None:
        return None
    try:
        return list(classes).index(prefix)
    except ValueError:
        return 0  # __other__, intentionally never promoted from val/test.


def _payment_bank_prefix_oov_by_split(
    records: Iterable[Mapping[str, object]], *, classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    known = set(classes) - {PAYMENT_BANK_OTHER_CLASS}
    counters: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    for record in records:
        prefix = _payment_bank_prefix_target(record)
        if prefix is None:
            continue
        split = str(record["split"])
        counters[split]["records"] += 1
        if prefix not in known:
            counters[split]["other"] += 1
    return {
        split: {"records": int(counters[split]["records"]), "other": int(counters[split]["other"])}
        for split in ("train", "val", "test")
    }


def _payment_bank_prefix_class_weights(
    records: Iterable[Mapping[str, object]],
    *,
    classes: Sequence[str],
    torch: Any,
    device: str,
) -> tuple[Any, dict[str, int]]:
    """Return balanced train-only weights for the finite bank prefix head.

    The ``__other__`` class intentionally includes low-support bank names.
    A class absent from one pilot split gets neutral weight rather than zero:
    it has no target in that run but remains a valid static ONNX output.
    """
    target_indices = [
        target
        for record in records
        for target in [_payment_bank_prefix_class_target(record, classes=classes)]
        if target is not None
    ]
    if not target_indices:
        raise ValueError("No v6 payment bank-prefix targets remain in the training split")
    counts = Counter(target_indices)
    total = len(target_indices)
    represented = max(1, len(counts))
    values = [
        total / (represented * counts[index]) if counts[index] else 1.0
        for index in range(len(classes))
    ]
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        {str(classes[index]): int(counts[index]) for index in range(len(classes))},
    )


def _ctc_required_steps(text: str) -> int:
    return len(text) + sum(left == right for left, right in zip(text, text[1:]))


def _validate_ctc_capacity(
    records: Iterable[Mapping[str, object]],
    *,
    config: UnifiedReaderConfig,
    recipient_characters: Sequence[str] | None = None,
    status_text_characters: Sequence[str] | None = None,
    allow_frozen_recipient_train_oov: bool = False,
) -> None:
    for record in records:
        fields = ("amount", "time", "payment_method_field", "recipient_field") if _uses_recipient_protocol(config) else (
            "amount",
            "time",
            "payment_method_field",
        )
        if _is_v13(config):
            fields = (*fields, "transfer_status")
        for field in fields:
            text = _ctc_slot_text(record, field, config=config)
            if text is None:
                continue
            characters = (
                _amount_characters(config)
                if field == "amount"
                else _time_characters(config)
                if field == "time"
                else recipient_characters
                if field == "recipient_field"
                else status_text_characters
                if field == "transfer_status"
                else None
            )
            if field == "recipient_field" and characters is None:
                raise ValueError("v9-v13 recipient CTC validation needs a train-only recipient charset")
            if field == "transfer_status" and characters is None:
                raise ValueError("v13 status CTC validation needs a train-only status-text charset")
            # Validation/test recipient OOV is intentional evidence for a
            # train-only Unicode alphabet.  It must not make the manifest
            # unloadable; those rows are scored/reviewed later.  Train labels,
            # by contrast, must always be encodable.
            validate_characters = (
                field not in {"recipient_field", "transfer_status"}
                or str(record["split"]) == "train"
            )
            if field == "recipient_field" and allow_frozen_recipient_train_oov:
                # Status-text-only v13 never optimizes the frozen recipient
                # head. New train-manifest glyphs are current-data OOV audit
                # evidence, not labels that need to fit the seed's row map.
                validate_characters = False
            if characters is not None and validate_characters and any(character not in characters for character in text):
                raise ValueError(
                    f"CTC target has a character outside the architecture v{config.architecture_version} {field} charset: "
                    f"id={record['id']}, text={text!r}."
                )
            required = _ctc_required_steps(text)
            available = _recipient_time_steps(config) if field == "recipient_field" else config.image_width // 4
            if required > available:
                raise ValueError(
                    f"CTC target cannot fit the unified model time axis: id={record['id']}, "
                    f"field={field}, required={required}, available={available}, text={text!r}. "
                    "Increase the relevant input width or exclude this record."
                )


def _input_tensor(record: Mapping[str, object], *, config: UnifiedReaderConfig) -> np.ndarray:
    slot_order = _slot_order(config)
    field_images = np.stack([_blank_image(config) for _ in slot_order], axis=0)
    slots = dict(record["slots"])
    for index, field in enumerate(slot_order):
        slot = slots.get(field)
        if isinstance(slot, Mapping):
            right_align = _uses_structured_heads(config) and field in {
                "amount",
                "time",
                "payment_method_field",
                "recipient_field",
            }
            # v10 reads the complete visible recipient row.  Centre it so a
            # long left-side label and the merchant value retain balanced
            # resolution; fixed-position numeric fields keep their v8/v9
            # right alignment unchanged.
            if (
                _is_v10(config)
                or _is_v11(config)
                or _uses_v12_recipient_topology(config)
            ) and field == "recipient_field":
                right_align = False
            # v12's fifth legacy slot remains part of the static input ABI but
            # is intentionally blank.  Its recipient CTC path receives the
            # separate high-resolution value view below, so this low-res
            # image cannot accidentally influence a merchant result.
            if _uses_high_resolution_recipient_input(config) and field == "recipient_field":
                continue
            field_images[index] = preprocess_image(
                Path(slot["image_path"]),
                config=config,
                horizontal_alignment="right" if right_align else "center",
                # v11's recipient target is the value to the right of the
                # anchored field label, so the pixels and CTC target stay
                # aligned.  The trim is frozen in the ONNX model config.
                left_crop_fraction=(
                    config.recipient_value_left_trim
                    if (_is_v11(config) or _uses_v12_recipient_topology(config))
                    and field == "recipient_field"
                    else 0.0
                ),
            )
    return field_images


def _recipient_value_input_tensor(record: Mapping[str, object], *, config: UnifiedReaderConfig) -> np.ndarray:
    """Return v12's independent high-resolution recipient value view.

    The anchored left trim is intentionally applied before resizing.  Keeping
    this preprocessing outside the shared low-resolution tensor makes the
    input contract explicit and lets the fifth CTC head retain text detail
    without widening the financial-field model.
    """
    if not _uses_high_resolution_recipient_input(config):
        raise ValueError("recipient value-view input is defined only for architecture v12")
    slot = dict(record["slots"]).get("recipient_field")
    if not isinstance(slot, Mapping):
        return _blank_recipient_value_image(config)
    return preprocess_image(
        Path(slot["image_path"]),
        config=config,
        horizontal_alignment="center",
        left_crop_fraction=config.recipient_value_left_trim,
        output_height=config.recipient_input_height,
        output_width=config.recipient_input_width,
    )


def _recipient_augmentation_rng(
    record: Mapping[str, object], *, policy: Mapping[str, object], epoch: int
) -> np.random.Generator:
    """Return a stable per-record augmentation RNG.

    DataLoader workers can yield examples in a different order on different
    machines.  Deriving the generator from the frozen seed, current epoch and
    record id makes the train-only perturbation reproducible regardless of
    worker scheduling.
    """
    if policy.get("mode") not in {"light_v1", "robust_v2"}:
        raise ValueError("recipient augmentation RNG requires a non-empty recipient augmentation policy")
    seed = policy.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("recipient augmentation policy seed is invalid")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("recipient augmentation epoch must be a non-negative integer")
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("recipient augmentation record id is invalid")
    digest = hashlib.sha256(f"{seed}:{epoch}:{record_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))


def _augment_recipient_value_input(
    image: np.ndarray,
    *,
    record: Mapping[str, object],
    policy: Mapping[str, object],
    epoch: int,
) -> np.ndarray:
    """Apply a deliberately small recipient-only perturbation during train.

    This protects the shared amount/time/payment pathway and keeps the static
    v12 ONNX input contract unchanged.  White padding is used after shifts so
    the synthetic crop still resembles the source crop preprocessing.
    """
    normalized_policy = _validate_recipient_train_augmentation_policy(policy)
    if normalized_policy["mode"] == "none":
        return image
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("recipient value input must have shape [1,H,W]")
    if not np.isfinite(image).all():
        raise ValueError("recipient value input contains non-finite pixels")
    rng = _recipient_augmentation_rng(record, policy=normalized_policy, epoch=epoch)
    _, height, width = image.shape
    augmented_source = image
    if normalized_policy["mode"] == "robust_v2":
        horizontal_delta = float(normalized_policy["horizontal_scale_delta"])
        vertical_delta = float(normalized_policy["vertical_scale_delta"])
        scaled_width = max(1, int(round(width * float(rng.uniform(1.0 - horizontal_delta, 1.0)))))
        scaled_height = max(1, int(round(height * float(rng.uniform(1.0 - vertical_delta, 1.0)))))
        source_u8 = np.rint(np.clip(image[0], 0.0, 1.0) * 255.0).astype(np.uint8)
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        resized = np.asarray(
            Image.fromarray(source_u8, mode="L").resize((scaled_width, scaled_height), resampling),
            dtype=np.float32,
        ) / 255.0
        scaled = np.ones_like(image, dtype=np.float32)
        scaled_top = (height - scaled_height) // 2
        scaled_left = (width - scaled_width) // 2
        scaled[0, scaled_top : scaled_top + scaled_height, scaled_left : scaled_left + scaled_width] = resized
        augmented_source = scaled
    horizontal_limit = int(normalized_policy["horizontal_shift_px"])
    vertical_limit = int(normalized_policy["vertical_shift_px"])
    shift_x = int(rng.integers(-horizontal_limit, horizontal_limit + 1))
    shift_y = int(rng.integers(-vertical_limit, vertical_limit + 1))
    shifted = np.ones_like(image, dtype=np.float32)
    source_x_start = max(0, -shift_x)
    source_x_end = min(width, width - shift_x)
    source_y_start = max(0, -shift_y)
    source_y_end = min(height, height - shift_y)
    destination_x_start = max(0, shift_x)
    destination_x_end = destination_x_start + max(0, source_x_end - source_x_start)
    destination_y_start = max(0, shift_y)
    destination_y_end = destination_y_start + max(0, source_y_end - source_y_start)
    if source_x_end > source_x_start and source_y_end > source_y_start:
        shifted[
            :, destination_y_start:destination_y_end, destination_x_start:destination_x_end
        ] = augmented_source[:, source_y_start:source_y_end, source_x_start:source_x_end]
    # White remains white under the contrast transform; only ink strength is
    # altered. This avoids teaching the model a non-existent dark background.
    contrast_delta = float(normalized_policy["contrast_delta"])
    contrast = 1.0 + float(rng.uniform(-contrast_delta, contrast_delta))
    augmented = 1.0 - (1.0 - shifted) * contrast
    noise_std = float(normalized_policy["noise_std"])
    if noise_std > 0.0:
        augmented = augmented + rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
    if (
        normalized_policy["mode"] == "robust_v2"
        and float(rng.random()) < float(normalized_policy["blur_probability"])
    ):
        # A narrow horizontal kernel models mild screenshot/display blur while
        # preserving the vertical strokes that distinguish Chinese glyphs.
        padded = np.pad(augmented, ((0, 0), (0, 0), (1, 1)), mode="edge")
        augmented = (
            0.25 * padded[:, :, :-2]
            + 0.50 * padded[:, :, 1:-1]
            + 0.25 * padded[:, :, 2:]
        )
    return np.clip(augmented, 0.0, 1.0).astype(np.float32, copy=False)


class _UnifiedReceiptDataset:
    """A picklable dataset so Windows DataLoader workers remain usable.

    Recipient augmentation derives its perturbation from the training epoch.  A regular
    integer would be copied into each Windows-spawned worker, which previously
    made persistent workers unsafe: they would keep augmenting every epoch as
    epoch zero.  For either non-empty policy, a shared-memory CPU tensor is safely
    transported by ``torch.utils.data.DataLoader`` across both fork and spawn
    contexts, so the parent can advance the epoch without recreating workers
    or changing the per-record RNG formula.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        config: UnifiedReaderConfig,
        recipient_train_augmentation_policy: Mapping[str, object] | None = None,
        recipient_only: bool = False,
    ) -> None:
        if recipient_only and not _uses_v12_recipient_topology(config):
            raise ValueError("recipient_only dataset is supported only by architecture v12 or v13")
        self._records = list(records)
        self._config = config
        self._recipient_only = recipient_only
        self._recipient_train_augmentation_policy = _validate_recipient_train_augmentation_policy(
            {"mode": "none"}
            if recipient_train_augmentation_policy is None
            else recipient_train_augmentation_policy
        )
        self._epoch = 0
        self._shared_augmentation_epoch: Any | None = None
        if self._recipient_train_augmentation_policy["mode"] != "none":
            # Do not use ``multiprocessing.Value`` here: a Value created from
            # a Linux fork context cannot be sent to an explicitly spawned
            # worker.  A shared CPU tensor is portable across DataLoader
            # contexts and its one scalar does not change the dataset/ONNX
            # input contract.
            torch, _ = _require_torch()
            self._shared_augmentation_epoch = torch.zeros(
                (), dtype=torch.int64, device="cpu"
            ).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("dataset epoch must be a non-negative integer")
        self._epoch = int(epoch)
        if self._shared_augmentation_epoch is not None:
            self._shared_augmentation_epoch.fill_(self._epoch)

    def _augmentation_epoch(self) -> int:
        """Return the worker-visible epoch used only by recipient augmentation."""
        if self._shared_augmentation_epoch is None:
            return self._epoch
        return int(self._shared_augmentation_epoch.item())

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        record = self._records[index]
        torch, _ = _require_torch()
        if self._recipient_only:
            if _recipient_slot(record) is None:
                raise ValueError("recipient_only dataset received a record without recipient_field")
            recipient_value_image = _recipient_value_input_tensor(record, config=self._config)
            if self._recipient_train_augmentation_policy["mode"] != "none":
                recipient_value_image = _augment_recipient_value_input(
                    recipient_value_image,
                    record=record,
                    policy=self._recipient_train_augmentation_policy,
                    epoch=self._augmentation_epoch(),
                )
            return torch.from_numpy(recipient_value_image), record
        field_images = torch.from_numpy(_input_tensor(record, config=self._config))
        if _uses_high_resolution_recipient_input(self._config):
            recipient_value_image = _recipient_value_input_tensor(record, config=self._config)
            if (
                self._recipient_train_augmentation_policy["mode"] != "none"
                and _recipient_slot(record) is not None
            ):
                recipient_value_image = _augment_recipient_value_input(
                    recipient_value_image,
                    record=record,
                    policy=self._recipient_train_augmentation_policy,
                    epoch=self._augmentation_epoch(),
                )
            return (
                field_images,
                torch.from_numpy(recipient_value_image),
                record,
            )
        return field_images, record


def _collate_receipts(samples: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
    torch, _ = _require_torch()
    if len(samples[0]) == 3:
        field_images, recipient_value_images, records = zip(*samples)
        return torch.stack(list(field_images)), torch.stack(list(recipient_value_images)), list(records)
    field_images, records = zip(*samples)
    return torch.stack(list(field_images)), list(records)


def _collate_recipient_only(samples: Sequence[tuple[Any, Any]]) -> tuple[Any, list[Mapping[str, object]]]:
    """Collate v12's private recipient input without loading financial slots."""
    torch, _ = _require_torch()
    recipient_value_images, records = zip(*samples)
    return torch.stack(list(recipient_value_images)), list(records)


def _recipient_only_logits(model: Any, recipient_value_images: Any, *, config: UnifiedReaderConfig) -> Any:
    """Run only v12's private recipient branch during guarded fine-tuning.

    The financial trunk and all of its heads are frozen in this mode.  Skipping
    them is mathematically identical for recipient logits and avoids four
    unnecessary crop decodes plus shared-trunk GPU work per training receipt.
    Full five-field inference still runs for every validation epoch, where the
    protection floors are measured.
    """
    if not _uses_v12_recipient_topology(config):
        raise ValueError("recipient-only logits are supported only by architecture v12 or v13")
    expected_shape = [recipient_value_images.shape[0], 1, config.recipient_input_height, config.recipient_input_width]
    if list(recipient_value_images.shape) != expected_shape:
        raise ValueError(
            "recipient_value_images must have shape "
            f"[batch,1,{config.recipient_input_height},{config.recipient_input_width}]"
        )
    encoded = model.recipient_encoder(model.recipient_stem(recipient_value_images))
    features = model.recipient_ctc_vertical_reducer(encoded)
    sequence = features.permute(2, 0, 1)
    if config.recipient_backbone == "residual_positional_transformer_v2":
        sequence = model.recipient_input_projection(sequence)
        sequence = sequence + model.recipient_position_embedding
        sequence = model.recipient_open_text_encoder(sequence)
    else:
        sequence, _ = model.recipient_ctc_sequence(sequence)
        if config.recipient_open_text_layers:
            refined_sequence = model.recipient_open_text_encoder(sequence)
            sequence = sequence + model.recipient_open_text_gate.tanh() * (
                refined_sequence - sequence
            )
    return model.recipient_classifier(sequence)


def _unpack_receipt_batch(
    batch: tuple[Any, ...], *, config: UnifiedReaderConfig
) -> tuple[Any, Any | None, list[Mapping[str, object]]]:
    """Normalize legacy and v12 DataLoader batches for the reader call."""
    if _uses_high_resolution_recipient_input(config):
        if len(batch) != 3:
            raise ValueError("v12 DataLoader batch must contain field images, recipient value images, and records")
        field_images, recipient_value_images, records = batch
        return field_images, recipient_value_images, list(records)
    if len(batch) != 2:
        raise ValueError("legacy DataLoader batch must contain field images and records")
    field_images, records = batch
    return field_images, None, list(records)


def _make_dataset(
    records: Sequence[Mapping[str, object]],
    *,
    config: UnifiedReaderConfig,
    torch: Any,
    recipient_train_augmentation_policy: Mapping[str, object] | None = None,
    recipient_only: bool = False,
) -> Any:
    del torch  # Kept in the signature so callers make the dependency explicit.
    return _UnifiedReceiptDataset(
        records,
        config=config,
        recipient_train_augmentation_policy=recipient_train_augmentation_policy,
        recipient_only=recipient_only,
    )


def _unpack_reader_outputs(outputs: object, *, config: UnifiedReaderConfig) -> dict[str, Any]:
    """Give training/evaluation named tensors while preserving v3/v4 tuples."""
    if not isinstance(outputs, tuple):
        raise ValueError("Unified reader must return a tuple of tensors")
    if _is_v13(config):
        if len(outputs) != len(V13_ONNX_OUTPUT_NAMES):
            raise ValueError("Unified v13 reader must return sixteen output tensors")
        return {name: value for name, value in zip(V13_ONNX_OUTPUT_NAMES, outputs)}
    if _uses_recipient_protocol(config):
        if len(outputs) != len(V9_ONNX_OUTPUT_NAMES):
            raise ValueError("Unified v9-v12 reader must return fifteen output tensors")
        names = V9_ONNX_OUTPUT_NAMES
        return {name: value for name, value in zip(names, outputs)}
    if _is_v8(config):
        if len(outputs) != 14:
            raise ValueError("Unified v8 reader must return fourteen output tensors")
        (
            amount_logits,
            time_logits,
            payment_logits,
            status_logits,
            amount_currency_style_logits,
            amount_grouped_thousands_logits,
            amount_sign_position_logits,
            time_format_logits,
            time_digit_logits,
            payment_prefix_logits,
            payment_bank_prefix_logits,
            payment_tail_digit_logits,
            payment_structure_logits,
            payment_parentheses_logits,
        ) = outputs
        return {
            "amount_logits": amount_logits,
            "time_logits": time_logits,
            "payment_logits": payment_logits,
            "status_logits": status_logits,
            "amount_currency_style_logits": amount_currency_style_logits,
            "amount_grouped_thousands_logits": amount_grouped_thousands_logits,
            "amount_sign_position_logits": amount_sign_position_logits,
            "time_format_logits": time_format_logits,
            "time_digit_logits": time_digit_logits,
            "payment_prefix_logits": payment_prefix_logits,
            "payment_bank_prefix_logits": payment_bank_prefix_logits,
            "payment_tail_digit_logits": payment_tail_digit_logits,
            "payment_structure_logits": payment_structure_logits,
            "payment_parentheses_logits": payment_parentheses_logits,
        }
    if _uses_v6_protocol(config):
        if len(outputs) != 14:
            raise ValueError("Unified v6/v7 reader must return fourteen output tensors")
        (
            amount_logits,
            time_logits,
            payment_logits,
            status_logits,
            amount_sign_logits,
            amount_length_logits,
            amount_digit_logits,
            time_format_logits,
            time_digit_logits,
            payment_prefix_logits,
            payment_bank_prefix_logits,
            payment_tail_digit_logits,
            payment_structure_logits,
            payment_parentheses_logits,
        ) = outputs
        return {
            "amount_logits": amount_logits,
            "time_logits": time_logits,
            "payment_logits": payment_logits,
            "status_logits": status_logits,
            "amount_sign_logits": amount_sign_logits,
            "amount_length_logits": amount_length_logits,
            "amount_digit_logits": amount_digit_logits,
            "time_format_logits": time_format_logits,
            "time_digit_logits": time_digit_logits,
            "payment_prefix_logits": payment_prefix_logits,
            "payment_bank_prefix_logits": payment_bank_prefix_logits,
            "payment_tail_digit_logits": payment_tail_digit_logits,
            "payment_structure_logits": payment_structure_logits,
            "payment_parentheses_logits": payment_parentheses_logits,
        }
    if config.architecture_version == 5:
        if len(outputs) != 11:
            raise ValueError("Unified v5 reader must return eleven output tensors")
        (
            numeric_logits,
            payment_logits,
            status_logits,
            amount_length_logits,
            amount_digit_logits,
            time_digit_logits,
            time_hour_width_logits,
            payment_prefix_logits,
            payment_tail_digit_logits,
            payment_structure_logits,
            payment_parentheses_logits,
        ) = outputs
        return {
            "numeric_logits": numeric_logits,
            "amount_logits": numeric_logits[:, :, 0, :],
            "time_logits": numeric_logits[:, :, 1, :],
            "payment_logits": payment_logits,
            "status_logits": status_logits,
            "amount_length_logits": amount_length_logits,
            "amount_digit_logits": amount_digit_logits,
            "time_digit_logits": time_digit_logits,
            "time_hour_width_logits": time_hour_width_logits,
            "payment_prefix_logits": payment_prefix_logits,
            "payment_tail_digit_logits": payment_tail_digit_logits,
            "payment_structure_logits": payment_structure_logits,
            "payment_parentheses_logits": payment_parentheses_logits,
        }
    if len(outputs) != 3:
        raise ValueError("Unified v3/v4 reader must return three output tensors")
    numeric_logits, payment_logits, status_logits = outputs
    return {
        "numeric_logits": numeric_logits,
        "amount_logits": numeric_logits[:, :, 0, :],
        "time_logits": numeric_logits[:, :, 1, :],
        "payment_logits": payment_logits,
        "status_logits": status_logits,
    }


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


def _argmax_with_confidence(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return final-axis argmax values and their softmax probabilities."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    indices = probabilities.argmax(axis=-1)
    confidence = np.take_along_axis(probabilities, indices[..., np.newaxis], axis=-1).squeeze(-1)
    return indices.astype(np.int64), confidence.astype(np.float64)


def _structured_amount_predictions(
    amount_length_logits: np.ndarray,
    amount_digit_logits: np.ndarray,
) -> list[tuple[str | None, float]]:
    lengths, length_confidence = _argmax_with_confidence(amount_length_logits)
    digits, digit_confidence = _argmax_with_confidence(amount_digit_logits)
    if digits.ndim != 2 or digits.shape[1] != AMOUNT_DIGIT_SLOTS:
        raise ValueError("amount structured digit output has an invalid shape")
    output: list[tuple[str | None, float]] = []
    for index, raw_length in enumerate(lengths.tolist()):
        integer_length = int(raw_length) + 1
        if not 1 <= integer_length <= AMOUNT_MAX_INTEGER_DIGITS:
            output.append((None, 0.0))
            continue
        first = AMOUNT_MAX_INTEGER_DIGITS - integer_length
        integer = "".join(str(value) for value in digits[index, first:AMOUNT_MAX_INTEGER_DIGITS])
        cents = "".join(str(value) for value in digits[index, AMOUNT_MAX_INTEGER_DIGITS:])
        # The dataset parser deliberately rejects non-canonical values such as
        # 00.99.  Do the same at decode time: a structural head that produces
        # an impossible amount must lead to review/CTC diagnostics, never to a
        # silently reformatted financial value.
        if integer_length > 1 and integer.startswith("0"):
            output.append((None, 0.0))
            continue
        confidence = float(
            np.mean(
                np.concatenate(
                    (
                        np.asarray([length_confidence[index]]),
                        digit_confidence[index, first:AMOUNT_DIGIT_SLOTS],
                    )
                )
            )
        )
        output.append((f"{integer}.{cents}", confidence))
    return output


def _structured_time_predictions(
    time_digit_logits: np.ndarray,
    time_hour_width_logits: np.ndarray,
) -> list[tuple[str | None, float]]:
    digits, digit_confidence = _argmax_with_confidence(time_digit_logits)
    widths, width_confidence = _argmax_with_confidence(time_hour_width_logits)
    if digits.ndim != 2 or digits.shape[1] != TIME_DIGIT_SLOTS:
        raise ValueError("time structured digit output has an invalid shape")
    output: list[tuple[str | None, float]] = []
    for index, values in enumerate(digits):
        hour = int(values[0]) * 10 + int(values[1])
        minute = int(values[2]) * 10 + int(values[3])
        width = int(widths[index]) + 1
        if not (0 <= hour <= 23 and 0 <= minute <= 59) or width not in {1, 2} or (width == 1 and hour >= 10):
            output.append((None, 0.0))
            continue
        rendered_hour = str(hour) if width == 1 else f"{hour:02d}"
        confidence = float(np.mean(np.concatenate((digit_confidence[index], [width_confidence[index]]))))
        output.append((f"{rendered_hour}:{minute:02d}", confidence))
    return output


def _structured_payment_predictions(
    payment_prefix_logits: np.ndarray,
    payment_tail_digit_logits: np.ndarray,
    payment_structure_logits: np.ndarray,
    payment_parentheses_logits: np.ndarray,
    *,
    payment_characters: Sequence[str],
) -> list[tuple[str | None, float]]:
    prefix_predictions = decode_ctc_logits_with_confidence(
        np.asarray(payment_prefix_logits), characters=payment_characters
    )
    tail_digits, tail_confidence = _argmax_with_confidence(payment_tail_digit_logits)
    structure, structure_confidence = _argmax_with_confidence(payment_structure_logits)
    parentheses, parentheses_confidence = _argmax_with_confidence(payment_parentheses_logits)
    if tail_digits.ndim != 2 or tail_digits.shape[1] != PAYMENT_TAIL_DIGIT_SLOTS:
        raise ValueError("payment structured tail output has an invalid shape")
    output: list[tuple[str | None, float]] = []
    for index, (prefix, prefix_confidence) in enumerate(prefix_predictions):
        if int(structure[index]) != 1 or not prefix or any(character in "()（）" for character in prefix):
            output.append((None, 0.0))
            continue
        tail = "".join(str(value) for value in tail_digits[index])
        style = PAYMENT_PARENTHESIS_CLASSES[int(parentheses[index])]
        candidate = recompose_payment_card_tail_target(
            prefix_text=prefix,
            card_tail=tail,
            parentheses=style,
        )
        if candidate is None:
            output.append((None, 0.0))
            continue
        confidence = float(
            np.mean(
                np.concatenate(
                    (
                        np.asarray([prefix_confidence, structure_confidence[index], parentheses_confidence[index]]),
                        tail_confidence[index],
                    )
                )
            )
        )
        output.append((candidate, confidence))
    return output


def _structured_amount_v6_predictions(
    amount_sign_logits: np.ndarray,
    amount_length_logits: np.ndarray,
    amount_digit_logits: np.ndarray,
) -> list[tuple[str | None, float]]:
    """Decode v6's independent signed canonical-amount verifier."""
    base_predictions = _structured_amount_predictions(amount_length_logits, amount_digit_logits)
    signs, sign_confidence = _argmax_with_confidence(amount_sign_logits)
    if len(base_predictions) != len(signs):
        raise ValueError("v6 amount verifier outputs have incompatible batch sizes")
    output: list[tuple[str | None, float]] = []
    for index, (base, base_confidence) in enumerate(base_predictions):
        if base is None:
            output.append((None, 0.0))
            continue
        sign = AMOUNT_SIGN_CLASSES[int(signs[index])]
        if sign == "negative" and base == "0.00":
            output.append((None, 0.0))
            continue
        candidate = "-" + base if sign == "negative" else base
        output.append((candidate, float(np.mean((base_confidence, sign_confidence[index])))))
    return output


def _structured_amount_v8_predictions(
    amount_ctc_predictions: Sequence[tuple[str, float]],
    amount_currency_style_logits: np.ndarray,
    amount_grouped_thousands_logits: np.ndarray,
    amount_sign_position_logits: np.ndarray,
    *,
    min_confidence: float,
) -> list[tuple[str | None, float]]:
    """Render v8 display grammar only when it preserves CTC's digits.

    The finite heads are never a second number recogniser. A malformed CTC
    value or a low-confidence *relevant* format component produces ``None``
    so callers retain the canonical raw CTC diagnostic.  A component that
    cannot change the rendered text must not block that diagnostic candidate:
    positive values always have no sign, and grouping a sub-1000 integer is
    visually identical to leaving it ungrouped.  Requiring confidence from
    those irrelevant classifiers was an accidental all-three gate that
    rejected otherwise safe visible renderings.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("amount v8 format confidence must be between 0 and 1")
    currencies, currency_confidence = _argmax_with_confidence(amount_currency_style_logits)
    groupings, grouping_confidence = _argmax_with_confidence(amount_grouped_thousands_logits)
    signs, sign_confidence = _argmax_with_confidence(amount_sign_position_logits)
    batch = len(amount_ctc_predictions)
    if any(len(values) != batch for values in (currencies, groupings, signs)):
        raise ValueError("v8 amount CTC and format outputs have incompatible batch sizes")
    output: list[tuple[str | None, float]] = []
    for index, (canonical, ctc_confidence) in enumerate(amount_ctc_predictions):
        # Check the canonical text before passing it to the renderer. This
        # refuses an ambiguous leading zero or a malformed decimal without
        # accidentally converting it into a financially different number.
        parsed = parse_amount_display_target(canonical)
        if (
            parsed is None
            or parsed.get("currency") is not None
            or bool(parsed.get("grouped_thousands"))
        ):
            output.append((None, 0.0))
            continue
        integer = parsed.get("integer_digits")
        negative = parsed.get("sign") == "negative"
        if not isinstance(integer, str) or not integer.isascii() or not integer.isdigit():
            output.append((None, 0.0))
            continue

        # Currency is always visible-or-absent, so it always matters.  A
        # sign choice matters only for a negative canonical number; for a
        # positive one, force the grammar-safe "none" state instead of
        # allowing an irrelevant uncertain sign head to suppress output.  A
        # comma can only become visible for four or more integer digits.
        currency_style = AMOUNT_CURRENCY_STYLE_CLASSES[int(currencies[index])]
        grouped_thousands = (
            AMOUNT_GROUPED_THOUSANDS_CLASSES[int(groupings[index])]
            if len(integer) >= 4
            else "ungrouped"
        )
        # A bare negative amount has only one grammar-safe representation
        # (``-123.45``), so the sign-position classifier cannot add useful
        # information there either.  It is needed only to distinguish
        # ``-¥123.45`` from ``¥-123.45``.
        sign_position = (
            AMOUNT_SIGN_POSITION_CLASSES[int(signs[index])]
            if negative and currency_style != "none"
            else "before_currency_or_number"
            if negative
            else "none"
        )
        relevant_confidences = [float(currency_confidence[index])]
        if len(integer) >= 4:
            relevant_confidences.append(float(grouping_confidence[index]))
        if negative and currency_style != "none":
            relevant_confidences.append(float(sign_confidence[index]))
        component_confidence = min(relevant_confidences)
        if component_confidence < min_confidence:
            output.append((None, 0.0))
            continue
        candidate = render_amount_visible_format(
            canonical,
            currency_style=currency_style,
            grouped_thousands=grouped_thousands,
            sign_position=sign_position,
        )
        if candidate is None:
            output.append((None, 0.0))
            continue
        output.append((candidate, min(float(ctc_confidence), component_confidence)))
    return output


def _structured_time_v6_predictions(
    time_format_logits: np.ndarray,
    time_digit_logits: np.ndarray,
) -> list[tuple[str | None, float]]:
    """Decode only valid v6 clock/date-time templates from fixed digits."""
    from .ocr_unified_targets import parse_time_display_target

    formats, format_confidence = _argmax_with_confidence(time_format_logits)
    digits, digit_confidence = _argmax_with_confidence(time_digit_logits)
    if digits.ndim != 2 or digits.shape[1] != TIME_DISPLAY_DIGIT_SLOTS:
        raise ValueError("v6 time structured digit output has an invalid shape")
    output: list[tuple[str | None, float]] = []
    for index, raw_format in enumerate(formats.tolist()):
        format_name = TIME_DISPLAY_FORMAT_CLASSES[int(raw_format)]
        text_digits = "".join(str(value) for value in digits[index])
        if format_name in {"clock_h_mm", "clock_hh_mm"}:
            hour, minute = text_digits[:2], text_digits[2:4]
            candidate = f"{int(hour)}:{minute}" if format_name == "clock_h_mm" else f"{hour}:{minute}"
            used = 4
        elif format_name in {"clock_h_mm_ss", "clock_hh_mm_ss"}:
            hour, minute, second = text_digits[:2], text_digits[2:4], text_digits[4:6]
            candidate = (
                f"{int(hour)}:{minute}:{second}"
                if format_name == "clock_h_mm_ss"
                else f"{hour}:{minute}:{second}"
            )
            used = 6
        elif format_name == "date_ymd_hh_mm":
            candidate = (
                f"{text_digits[:4]}-{text_digits[4:6]}-{text_digits[6:8]} "
                f"{text_digits[8:10]}:{text_digits[10:12]}"
            )
            used = 12
        else:
            candidate = (
                f"{text_digits[:4]}-{text_digits[4:6]}-{text_digits[6:8]} "
                f"{text_digits[8:10]}:{text_digits[10:12]}:{text_digits[12:14]}"
            )
            used = 14
        if parse_time_display_target(candidate) is None:
            output.append((None, 0.0))
            continue
        confidence = float(np.mean(np.concatenate((np.asarray([format_confidence[index]]), digit_confidence[index, :used]))))
        output.append((candidate, confidence))
    return output


def _structured_payment_v6_predictions(
    payment_bank_prefix_logits: np.ndarray,
    payment_tail_digit_logits: np.ndarray,
    payment_structure_logits: np.ndarray,
    payment_parentheses_logits: np.ndarray,
    *,
    payment_bank_prefix_classes: Sequence[str],
) -> list[tuple[str | None, float]]:
    """Decode v6's finite known-bank verifier without trusting CTC text."""
    banks, bank_confidence = _argmax_with_confidence(payment_bank_prefix_logits)
    tail_digits, tail_confidence = _argmax_with_confidence(payment_tail_digit_logits)
    structure, structure_confidence = _argmax_with_confidence(payment_structure_logits)
    parentheses, parentheses_confidence = _argmax_with_confidence(payment_parentheses_logits)
    if tail_digits.ndim != 2 or tail_digits.shape[1] != PAYMENT_TAIL_DIGIT_SLOTS:
        raise ValueError("v6 payment structured tail output has an invalid shape")
    output: list[tuple[str | None, float]] = []
    for index, raw_bank in enumerate(banks.tolist()):
        bank = payment_bank_prefix_classes[int(raw_bank)]
        if int(structure[index]) != 1 or bank == PAYMENT_BANK_OTHER_CLASS:
            output.append((None, 0.0))
            continue
        tail = "".join(str(value) for value in tail_digits[index])
        candidate = recompose_payment_card_tail_target(
            prefix_text=bank,
            card_tail=tail,
            parentheses=PAYMENT_PARENTHESIS_CLASSES[int(parentheses[index])],
        )
        if candidate is None:
            output.append((None, 0.0))
            continue
        confidence = float(
            np.mean(
                np.concatenate(
                    (
                        np.asarray(
                            [bank_confidence[index], structure_confidence[index], parentheses_confidence[index]]
                        ),
                        tail_confidence[index],
                    )
                )
            )
        )
        output.append((candidate, confidence))
    return output


def _select_report_candidates(
    ctc_predictions: Mapping[str, tuple[str, float]],
    structured_predictions: Mapping[str, tuple[str | None, float]],
    *,
    config: UnifiedReaderConfig,
) -> dict[str, tuple[str, float]]:
    """Choose visible diagnostic candidates without weakening delivery policy.

    A v6/v7 time verifier explicitly predicts a valid clock/date-time template,
    so it owns separators such as ``:`` / ``-`` / the date-time space.  The
    raw CTC trace is retained separately for debugging, but showing it as the
    primary candidate would incorrectly render a correct ``02:40`` verifier
    result as ``0240``. v8 can additionally render an amount only when its
    finite format grammar has passed the digit-preserving confidence gate.
    Payment keeps CTC pending a calibrated known-bank acceptance policy.
    """
    predictions = dict(ctc_predictions)
    if config.architecture_version == 5:
        for field, (structured_text, structured_confidence) in structured_predictions.items():
            if structured_text is not None:
                predictions[field] = (str(structured_text), float(structured_confidence))
    elif _uses_v8_protocol(config):
        for field in ("amount", "time"):
            structured = structured_predictions.get(field)
            if structured is not None and structured[0] is not None:
                predictions[field] = (str(structured[0]), float(structured[1]))
    elif _uses_v6_protocol(config):
        structured_time = structured_predictions.get("time")
        if structured_time is not None and structured_time[0] is not None:
            predictions["time"] = (str(structured_time[0]), float(structured_time[1]))
    return predictions


def _slot_text(record: Mapping[str, object], field: str) -> str | None:
    slot = dict(record["slots"]).get(field)
    if not isinstance(slot, Mapping):
        return None
    text = slot.get("text")
    return text if isinstance(text, str) else None


def _recipient_value_from_visible_text(text: str | None) -> str | None:
    """Extract a merchant value from a complete visible recipient row.

    v10 deliberately trains its CTC reader on the pixels it actually sees,
    including a left-side ``收款方``/``收款人`` label when present.  The
    business value remains the right-side text, so extraction belongs after
    decoding rather than in the CTC target.
    """
    if not isinstance(text, str):
        return None
    value = clean_text(extract_field_value(clean_text(text), "recipient"))
    return value or None


def _recipient_expected_value(record: Mapping[str, object], *, config: UnifiedReaderConfig) -> str | None:
    """Return the merchant value used for candidate metrics/runtime review."""
    raw = _slot_text(record, "recipient_field")
    if not _is_v10(config):
        return raw
    slot = dict(record["slots"]).get("recipient_field")
    if isinstance(slot, Mapping):
        for key in ("recipient_value", "semantic_value"):
            value = slot.get(key)
            if isinstance(value, str) and clean_text(value):
                return clean_text(value)
    return _recipient_value_from_visible_text(raw)


def _recipient_candidate_value(text: str | None, *, config: UnifiedReaderConfig) -> str | None:
    """Convert recipient CTC output into the version's merchant candidate."""
    return _recipient_value_from_visible_text(text) if _is_v10(config) else text


def _ctc_slot_text(record: Mapping[str, object], field: str, *, config: UnifiedReaderConfig) -> str | None:
    """Return the target actually supervised by a version's CTC head.

    v3-v5 retain their old canonical amount/time targets. v6/v7 consume an
    audited ``visible_text`` where available. v8 instead puts the amount back
    into a compact signed canonical CTC alphabet and learns only the visible
    display grammar with finite heads.
    """
    slot = dict(record["slots"]).get(field)
    if not isinstance(slot, Mapping):
        return None
    if _uses_v8_protocol(config) and field == "amount":
        visible = slot.get("visible_text")
        parsed = parse_amount_display_target(visible) if isinstance(visible, str) else None
        if parsed is not None:
            canonical = parsed.get("canonical_decimal")
            return canonical if isinstance(canonical, str) else None
        return None
    if _uses_modern_protocol(config) and field == "time":
        visible = slot.get("visible_text")
        if isinstance(visible, str) and visible:
            return visible
    if _uses_v6_protocol(config) and field == "amount":
        visible = slot.get("visible_text")
        if isinstance(visible, str) and visible:
            return visible
    return _slot_text(record, field)


def _status_name(record: Mapping[str, object]) -> str | None:
    slot = dict(record["slots"]).get("transfer_status")
    if not isinstance(slot, Mapping):
        return None
    class_name = slot.get("class_name")
    return class_name if class_name in STATUS_CLASSES else None


def _amount_structured_target(record: Mapping[str, object]) -> tuple[int, list[int]] | None:
    """Return integer length and fixed right-aligned decimal digits for v5.

    The unified manifest's amount target is already semantic-normalised by the
    teacher pipeline.  We nevertheless validate it here rather than letting a
    malformed pseudo-label become a confident financial prediction.
    """
    slot = dict(record["slots"]).get("amount")
    if not isinstance(slot, Mapping):
        return None
    aux = slot.get("amount_aux")
    if not is_structured_target(aux, expected_format=AMOUNT_AUX_FORMAT):
        return None
    integer = aux.get("integer_digits")
    cents = aux.get("cents_digits")
    digits = aux.get("right_aligned_digits")
    mask = aux.get("right_aligned_mask")
    if (
        not isinstance(integer, str)
        or not isinstance(cents, str)
        or not isinstance(digits, list)
        or not isinstance(mask, list)
        or len(integer) not in range(1, AMOUNT_MAX_INTEGER_DIGITS + 1)
        or len(cents) != 2
        or len(digits) != AMOUNT_DIGIT_SLOTS
        or len(mask) != AMOUNT_DIGIT_SLOTS
        or not integer.isascii()
        or not cents.isascii()
        or not integer.isdigit()
        or not cents.isdigit()
    ):
        return None
    visible_text = slot.get("text")
    if visible_text != f"{integer}.{cents}":
        return None
    expected_digits: list[str | None] = [None] * (AMOUNT_MAX_INTEGER_DIGITS - len(integer))
    expected_digits.extend((*integer, *cents))
    expected_mask = [value is not None for value in expected_digits]
    if digits != expected_digits or mask != expected_mask:
        return None
    targets: list[int] = []
    for value, active in zip(digits, mask):
        if active is False and value is None:
            targets.append(STRUCTURED_IGNORE_INDEX)
        elif active is True and isinstance(value, str) and len(value) == 1 and value.isascii() and value.isdigit():
            targets.append(int(value))
        else:
            return None
    if "".join(str(value) for value in targets if value != STRUCTURED_IGNORE_INDEX) != integer + cents:
        return None
    return len(integer) - 1, targets


def _amount_v6_structured_target(record: Mapping[str, object]) -> tuple[int, int, list[int]] | None:
    """Return sign, integer length, and digits from an audited visible amount."""
    slot = dict(record["slots"]).get("amount")
    if not isinstance(slot, Mapping):
        return None
    aux = slot.get("amount_display")
    if not is_structured_target(aux, expected_format=AMOUNT_DISPLAY_AUX_FORMAT):
        return None
    assert isinstance(aux, Mapping)  # Established by ``is_structured_target``.
    visible_slot_text = slot.get("visible_text")
    if not isinstance(visible_slot_text, str):
        return None
    # Do not merely trust a JSON object that looks like an auxiliary target.
    # The visible CTC target and all verifier labels must come from one exact
    # strict parse; otherwise a corrupt pseudo-label could teach the two v6
    # branches contradictory amount values.
    parsed = parse_amount_display_target(visible_slot_text)
    if parsed is None:
        return None
    amount_keys = (
        "visible_text",
        "canonical_decimal",
        "sign",
        "currency",
        "currency_space",
        "grouped_thousands",
        "integer_digits",
        "cents_digits",
        "integer_digit_count",
        "right_aligned_width",
        "right_aligned_digits",
        "right_aligned_mask",
    )
    if any(aux.get(key) != parsed[key] for key in amount_keys):
        return None
    sign = aux.get("sign")
    integer = aux.get("integer_digits")
    cents = aux.get("cents_digits")
    digits = aux.get("right_aligned_digits")
    mask = aux.get("right_aligned_mask")
    visible_text = aux.get("visible_text")
    if (
        sign not in AMOUNT_SIGN_CLASSES
        or not isinstance(integer, str)
        or not isinstance(cents, str)
        or not isinstance(visible_text, str)
        or visible_slot_text != visible_text
        or not isinstance(digits, list)
        or not isinstance(mask, list)
        or len(integer) not in range(1, AMOUNT_MAX_INTEGER_DIGITS + 1)
        or len(cents) != 2
        or not integer.isascii()
        or not cents.isascii()
        or not integer.isdigit()
        or not cents.isdigit()
        or len(digits) != AMOUNT_DIGIT_SLOTS
        or len(mask) != AMOUNT_DIGIT_SLOTS
    ):
        return None
    targets: list[int] = []
    for value, active in zip(digits, mask):
        if active is False and value is None:
            targets.append(STRUCTURED_IGNORE_INDEX)
        elif active is True and isinstance(value, str) and len(value) == 1 and value.isascii() and value.isdigit():
            targets.append(int(value))
        else:
            return None
    return AMOUNT_SIGN_CLASSES.index(str(sign)), len(integer) - 1, targets


def _amount_v8_format_target(record: Mapping[str, object]) -> tuple[int, int, int] | None:
    """Return v8 finite amount-display labels from one audited visible value.

    The legacy v6 parser audit remains the source-of-truth guard. This keeps
    v8 from treating a stale ``visible_text`` or malformed format object as a
    safe grammar target, while allowing the CTC branch to remain canonical.
    """
    if _amount_v6_structured_target(record) is None:
        return None
    slot = dict(record["slots"]).get("amount")
    if not isinstance(slot, Mapping):
        return None
    visible = slot.get("visible_text")
    parsed = parse_amount_visible_format_target(visible) if isinstance(visible, str) else None
    if parsed is None:
        return None
    try:
        return (
            AMOUNT_CURRENCY_STYLE_CLASSES.index(str(parsed["currency_style"])),
            AMOUNT_GROUPED_THOUSANDS_CLASSES.index(str(parsed["grouped_thousands"])),
            AMOUNT_SIGN_POSITION_CLASSES.index(str(parsed["sign_position"])),
        )
    except ValueError:
        return None


def _time_structured_target(record: Mapping[str, object]) -> tuple[list[int], int] | None:
    """Return canonical HHMM digits plus visible one/two-digit hour width.

    We intentionally do not fabricate a leading zero in the *display* value:
    ``1:44`` and ``01:44`` remain separate raw labels.  The numeric head uses
    a zero-padded internal representation and the hour-width head restores the
    visible form.  Times with seconds remain CTC-only/review candidates.
    """
    slot = dict(record["slots"]).get("time")
    if not isinstance(slot, Mapping):
        return None
    aux = slot.get("time_aux")
    if not is_structured_target(aux, expected_format=TIME_AUX_FORMAT):
        return None
    hour_text = aux.get("hour_text")
    minute_text = aux.get("minute_text")
    if (
        not isinstance(hour_text, str)
        or not isinstance(minute_text, str)
        or len(hour_text) not in {1, 2}
        or len(minute_text) != 2
        or not hour_text.isascii()
        or not minute_text.isascii()
        or not hour_text.isdigit()
        or not minute_text.isdigit()
    ):
        return None
    hour, minute = int(hour_text), int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    if slot.get("text") != f"{hour_text}:{minute_text}":
        return None
    canonical = f"{hour:02d}{minute:02d}"
    return [int(character) for character in canonical], len(hour_text) - 1


def _time_v6_structured_target(record: Mapping[str, object]) -> tuple[int, list[int]] | None:
    """Return v6 format class and masked fixed-format time/date digits."""
    slot = dict(record["slots"]).get("time")
    if not isinstance(slot, Mapping):
        return None
    aux = slot.get("time_display")
    if not is_structured_target(aux, expected_format=TIME_DISPLAY_AUX_FORMAT):
        return None
    assert isinstance(aux, Mapping)  # Established by ``is_structured_target``.
    visible_slot_text = slot.get("visible_text")
    if not isinstance(visible_slot_text, str):
        return None
    # Date/time punctuation is part of the format contract.  Reparse the
    # visible target before using a fixed-format class/digit target so a
    # malformed or stale auxiliary object cannot contradict the CTC label.
    parsed = parse_time_display_target(visible_slot_text)
    if parsed is None:
        return None
    time_keys = ("visible_text", "format_name", "canonical_digits", "digit_slots", "digit_mask")
    if any(aux.get(key) != parsed[key] for key in time_keys):
        return None
    format_name = aux.get("format_name")
    visible_text = aux.get("visible_text")
    digits = aux.get("canonical_digits")
    mask = aux.get("digit_mask")
    if (
        format_name not in TIME_DISPLAY_FORMAT_CLASSES
        or not isinstance(visible_text, str)
        or visible_slot_text != visible_text
        or not isinstance(digits, list)
        or not isinstance(mask, list)
        or len(digits) > TIME_DISPLAY_DIGIT_SLOTS
        or len(mask) != TIME_DISPLAY_DIGIT_SLOTS
        or aux.get("digit_slots") != TIME_DISPLAY_DIGIT_SLOTS
    ):
        return None
    targets: list[int] = []
    expected_active = [True] * len(digits) + [False] * (TIME_DISPLAY_DIGIT_SLOTS - len(digits))
    if mask != expected_active:
        return None
    for value, active in zip([*digits, *([None] * (TIME_DISPLAY_DIGIT_SLOTS - len(digits)))], mask):
        if active is True and isinstance(value, str) and len(value) == 1 and value.isascii() and value.isdigit():
            targets.append(int(value))
        elif active is False and value is None:
            targets.append(STRUCTURED_IGNORE_INDEX)
        else:
            return None
    return TIME_DISPLAY_FORMAT_CLASSES.index(str(format_name)), targets


def _payment_card_tail_target(record: Mapping[str, object]) -> tuple[str, list[int], int] | None:
    """Use the manifest's audited parser result when it is a complete card tail.

    The parser never guesses an incomplete/invalid tail.  Older v3/v4
    manifests have no structured payload and therefore train the raw CTC path
    only, which is safer than inferring financial digits from arbitrary text.
    """
    slot = dict(record["slots"]).get("payment_method_field")
    if not isinstance(slot, Mapping):
        return None
    aux = slot.get("payment_card_tail")
    if not is_structured_target(aux, expected_format=PAYMENT_CARD_TAIL_FORMAT):
        return None
    prefix = aux.get("prefix_text")
    tail = aux.get("card_tail")
    parentheses = aux.get("parentheses")
    if (
        not isinstance(prefix, str)
        or not prefix
        or not isinstance(tail, str)
        or len(tail) != PAYMENT_TAIL_DIGIT_SLOTS
        or not tail.isascii()
        or not tail.isdigit()
        or parentheses not in PAYMENT_PARENTHESIS_CLASSES
    ):
        return None
    if recompose_payment_card_tail_target(
        prefix_text=prefix,
        card_tail=tail,
        parentheses=str(parentheses),
    ) != slot.get("text"):
        return None
    return prefix, [int(character) for character in tail], PAYMENT_PARENTHESIS_CLASSES.index(str(parentheses))


def _payment_bank_prefix_target(record: Mapping[str, object]) -> str | None:
    slot = dict(record["slots"]).get("payment_method_field")
    if not isinstance(slot, Mapping):
        return None
    aux = slot.get("payment_bank_prefix")
    if not is_structured_target(aux, expected_format=PAYMENT_BANK_PREFIX_FORMAT):
        return None
    prefix = aux.get("visible_prefix")
    card = _payment_card_tail_target(record)
    if not isinstance(prefix, str) or not prefix or card is None or card[0] != prefix:
        return None
    return prefix


def _v6_verifier_target_text(record: Mapping[str, object], field: str) -> str | None:
    """Return the exact target represented by a v6 verifier head.

    Amount verification intentionally compares canonical signed decimals: its
    fixed digit/sign heads validate the business value while visible CTC owns
    the currency glyph, comma grouping, and spacing.  Time owns its complete
    visible template, and payment verification covers only rows with a known
    bank prefix plus an exact four-digit tail.
    """
    if field == "amount":
        if _amount_v6_structured_target(record) is None:
            return None
        slot = dict(record["slots"]).get(field)
        aux = slot.get("amount_display") if isinstance(slot, Mapping) else None
        value = aux.get("canonical_decimal") if isinstance(aux, Mapping) else None
        return value if isinstance(value, str) else None
    if field == "time":
        if _time_v6_structured_target(record) is None:
            return None
        slot = dict(record["slots"]).get(field)
        aux = slot.get("time_display") if isinstance(slot, Mapping) else None
        value = aux.get("visible_text") if isinstance(aux, Mapping) else None
        return value if isinstance(value, str) else None
    if field == "payment_method_field":
        if _payment_bank_prefix_target(record) is None:
            return None
        card = _payment_card_tail_target(record)
        if card is None:
            return None
        prefix, tail, parentheses_index = card
        return recompose_payment_card_tail_target(
            prefix_text=prefix,
            card_tail="".join(str(digit) for digit in tail),
            parentheses=PAYMENT_PARENTHESIS_CLASSES[parentheses_index],
        )
    raise ValueError(f"Unsupported v6 verifier field: {field}")


def _structured_cross_entropy_loss(
    logits: Any,
    *,
    labels: Sequence[int | Sequence[int] | None],
    torch: Any,
    class_weight: Any | None = None,
) -> tuple[Any | None, int]:
    """CE for optional fixed-position targets, excluding absent slots safely."""
    selected = [(index, label) for index, label in enumerate(labels) if label is not None]
    if not selected:
        return None, 0
    indices = torch.tensor([index for index, _ in selected], dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(0, indices)
    first_label = selected[0][1]
    if isinstance(first_label, int):
        targets = torch.tensor([int(label) for _, label in selected], dtype=torch.long, device=logits.device)
        return torch.nn.functional.cross_entropy(selected_logits, targets, weight=class_weight), len(selected)
    targets = torch.tensor([list(label) for _, label in selected], dtype=torch.long, device=logits.device)
    if selected_logits.ndim != 3 or list(selected_logits.shape[:2]) != list(targets.shape):
        raise ValueError("Structured digit logits/targets have incompatible shapes")
    return (
        torch.nn.functional.cross_entropy(
            selected_logits.reshape(-1, selected_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=STRUCTURED_IGNORE_INDEX,
        ),
        len(selected),
    )


def _field_split_counts(
    records: Iterable[Mapping[str, object]], *, config: UnifiedReaderConfig
) -> dict[str, dict[str, int]]:
    slot_order = _slot_order(config)
    counts: dict[str, Counter[str]] = {field: Counter() for field in slot_order}
    for record in records:
        split = str(record["split"])
        for field in slot_order:
            if field == "transfer_status":
                labelled = _status_name(record) is not None
            else:
                labelled = _slot_text(record, field) is not None
            if labelled:
                counts[field][split] += 1
    return {
        field: {split: int(counts[field][split]) for split in ("train", "val", "test")}
        for field in slot_order
    }


def _structured_split_counts(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    """Audit the labels that actually supervise v5-v8 auxiliary heads."""
    targets = {
        "amount_aux": _amount_structured_target,
        "time_aux": _time_structured_target,
        "payment_card_tail": _payment_card_tail_target,
        "amount_display": _amount_v6_structured_target,
        "amount_visible_format_v8": _amount_v8_format_target,
        "time_display": _time_v6_structured_target,
        "payment_bank_prefix": _payment_bank_prefix_target,
    }
    counts: dict[str, Counter[str]] = {name: Counter() for name in (*targets, "payment_unstructured")}
    for record in records:
        split = str(record["split"])
        for name, parser in targets.items():
            if parser(record) is not None:
                counts[name][split] += 1
        if _slot_text(record, "payment_method_field") is not None and _payment_card_tail_target(record) is None:
            counts["payment_unstructured"][split] += 1
    return {
        name: {split: int(counts[name][split]) for split in ("train", "val", "test")}
        for name in counts
    }


def _require_v5_structured_coverage(counts: Mapping[str, Mapping[str, int]]) -> None:
    missing = [
        f"{name}:{split}"
        for name in ("amount_aux", "time_aux", "payment_card_tail", "payment_unstructured")
        for split in ("train", "val")
        if int(counts[name][split]) <= 0
    ]
    if missing:
        raise ValueError(
            "Unified v5 needs structured train/val labels for amount, time, exact payment card tails, and "
            "unstructured payment values; missing "
            + ", ".join(missing)
            + ". Rebuild the manifest with ocr_unified_dataset and inspect dataset.contract.json."
        )


def _require_v6_structured_coverage(counts: Mapping[str, Mapping[str, int]]) -> None:
    missing = [
        f"{name}:{split}"
        for name in ("amount_display", "time_display", "payment_bank_prefix")
        for split in ("train", "val")
        if int(counts[name][split]) <= 0
    ]
    if missing:
        raise ValueError(
            "Unified v6 needs strict visible amount/time and exact payment bank-prefix targets in both train and val; "
            "missing "
            + ", ".join(missing)
            + ". Rebuild the unified manifest from the Paddle teacher labels and inspect its structured target counts."
        )


def _require_v8_structured_coverage(counts: Mapping[str, Mapping[str, int]]) -> None:
    """Require auditable grammar labels instead of asking v8 to guess style."""
    missing = [
        f"{name}:{split}"
        for name in ("amount_visible_format_v8", "time_display", "payment_bank_prefix")
        for split in ("train", "val")
        if int(counts[name][split]) <= 0
    ]
    if missing:
        raise ValueError(
            "Unified v8 needs strict amount display grammar, visible time, and exact payment bank-prefix "
            "targets in both train and val; missing "
            + ", ".join(missing)
            + ". Rebuild the unified manifest from the Paddle teacher labels and inspect its structured target counts."
        )


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


def _require_train_and_validation_coverage(
    field_counts: Mapping[str, Mapping[str, int]], *, required_fields: Iterable[str] = SLOT_ORDER
) -> None:
    required = tuple(required_fields)
    missing_train = [field for field in required if int(field_counts[field]["train"]) <= 0]
    missing_validation = [field for field in required if int(field_counts[field]["val"]) <= 0]
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


def _status_head_policy(status_counts: Mapping[str, Mapping[str, int]]) -> dict[str, object]:
    """Return the safety policy for the optional status head.

    A data set with only ``success`` examples cannot establish that pending or
    failed receipts are safe.  In that case status loss is disabled so it does
    not consume shared-trunk capacity, and the exported delivery contract says
    that callers must emit ``review`` rather than consume status logits.
    """
    missing_by_split = {
        split: [name for name in STATUS_CLASSES if int(status_counts[split][name]) <= 0]
        for split in ("train", "val", "test")
    }
    training_enabled = not missing_by_split["train"] and not missing_by_split["val"]
    delivery_allowed = training_enabled and not missing_by_split["test"]
    return {
        "training_enabled": training_enabled,
        "delivery_allowed": delivery_allowed,
        "runtime_policy": "classify" if delivery_allowed else "review_only",
        "missing_classes_by_split": missing_by_split,
        "reason": (
            "all success/pending/failed classes are represented in train, val, and test"
            if delivery_allowed
            else "one or more status classes are absent from train, val, or test; emit review at runtime",
        ),
    }


def _status_policy_from_counts(raw_counts: object, *, source: str) -> dict[str, object]:
    """Validate persisted status audit counts and derive their safe policy."""
    if not isinstance(raw_counts, Mapping):
        raise ValueError(f"{source} is missing status class audit counts")
    counts: dict[str, dict[str, int]] = {}
    try:
        for split in ("train", "val", "test"):
            raw_split = raw_counts[split]
            if not isinstance(raw_split, Mapping):
                raise TypeError(split)
            counts[split] = {name: int(raw_split[name]) for name in STATUS_CLASSES}
            if any(value < 0 for value in counts[split].values()):
                raise ValueError(split)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} has invalid status class audit counts") from error
    return _status_head_policy(counts)


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
    sample_weights: Sequence[float] | None = None,
) -> tuple[Any | None, int, int]:
    """Return CTC loss, used label count, and OOV-skipped label count.

    ``sample_weights`` is deliberately optional.  The no-weight path keeps the
    historical PyTorch ``reduction='mean'`` call exactly, while the weighted
    path first applies the same per-target-length normalization and then takes
    a weighted mean.  That makes recipient teacher-confidence weighting a
    train-only change without perturbing existing amount/time/payment losses.
    """
    if sample_weights is not None and len(sample_weights) != len(labels):
        raise ValueError("CTC sample_weights must match labels length")
    selected: list[tuple[int, str, float | None]] = []
    skipped = 0
    for index, text in enumerate(labels):
        if text is None:
            continue
        if any(character not in character_to_id for character in text):
            skipped += 1
            continue
        if sample_weights is None:
            weight: float | None = None
        else:
            try:
                weight = float(sample_weights[index])
            except (TypeError, ValueError):
                raise ValueError("CTC sample weights must be finite and positive") from None
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("CTC sample weights must be finite and positive")
        selected.append((index, text, weight))
    if not selected:
        return None, 0, skipped
    indices = torch.tensor([index for index, _, _ in selected], dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(1, indices)
    targets = torch.tensor(
        [character_to_id[character] for _, text, _ in selected for character in text],
        dtype=torch.long,
        device=logits.device,
    )
    input_lengths = torch.full((len(selected),), selected_logits.shape[0], dtype=torch.long)
    target_lengths = torch.tensor([len(text) for _, text, _ in selected], dtype=torch.long)
    if sample_weights is None:
        loss = torch.nn.functional.ctc_loss(
            selected_logits.log_softmax(2),
            targets,
            input_lengths,
            target_lengths,
            blank=NUMERIC_BLANK_INDEX,
            reduction="mean",
            zero_infinity=False,
        )
    else:
        per_sample_loss = torch.nn.functional.ctc_loss(
            selected_logits.log_softmax(2),
            targets,
            input_lengths,
            target_lengths,
            blank=NUMERIC_BLANK_INDEX,
            reduction="none",
            zero_infinity=False,
        )
        weights = torch.tensor(
            [float(weight) for _, _, weight in selected],
            dtype=per_sample_loss.dtype,
            device=per_sample_loss.device,
        )
        normalized_loss = per_sample_loss / target_lengths.to(
            dtype=per_sample_loss.dtype,
            device=per_sample_loss.device,
        )
        loss = (normalized_loss * weights).sum() / weights.sum()
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
    amount_logits: Any,
    time_logits: Any,
    payment_logits: Any,
    status_logits: Any,
    records: Sequence[Mapping[str, object]],
    *,
    amount_to_id: Mapping[str, int],
    time_to_id: Mapping[str, int],
    payment_to_id: Mapping[str, int],
    recipient_logits: Any | None = None,
    recipient_to_id: Mapping[str, int] | None = None,
    status_text_logits: Any | None = None,
    status_text_to_id: Mapping[str, int] | None = None,
    payment_bank_prefix_classes: Sequence[str] | None,
    payment_bank_class_weights: Any | None,
    status_to_id: Mapping[str, int],
    status_criterion: Any | None,
    status_enabled: bool,
    payment_loss_weight: float,
    recipient_loss_weight: float,
    status_text_loss_weight: float = 1.0,
    config: UnifiedReaderConfig,
    structured_outputs: Mapping[str, Any] | None,
    ctc_loss_weight: float,
    structured_loss_weight: float,
    torch: Any,
    recipient_sample_weights: Sequence[float] | None = None,
    allow_empty: bool = False,
    collect_metrics: bool = True,
    recipient_only: bool = False,
    status_text_only: bool = False,
) -> tuple[Any | None, dict[str, dict[str, float | int]] | None]:
    """Return one batch loss and, when requested, detached diagnostics.

    The training loop only needs the scalar loss.  Materialising every
    diagnostic by calling ``.cpu()`` per batch forces a CUDA synchronization,
    which prevents pinned-memory transfers and GPU work from overlapping.  The
    validation/audit paths may still request the exact historical diagnostics;
    the hot training path deliberately opts out.
    """
    if recipient_only and status_text_only:
        raise ValueError("recipient_only and status_text_only are mutually exclusive")
    if status_text_only:
        if not _is_v13(config):
            raise ValueError("status_text_only loss is supported only by architecture v13")
        if status_text_logits is None or status_text_to_id is None:
            raise ValueError("status_text_only loss requires status text logits and a train-only charset")
        status_text_loss, status_text_used, status_text_oov = _ctc_loss(
            status_text_logits,
            labels=[_slot_text(record, "transfer_status") for record in records],
            character_to_id=status_text_to_id,
            torch=torch,
        )
        if status_text_loss is None:
            if not allow_empty:
                raise ValueError("A status-text-only training batch has no visible status label")
            return None, None if not collect_metrics else {
                "transfer_status_text": {
                    "loss": math.nan,
                    "used": status_text_used,
                    "oov": status_text_oov,
                }
            }
        loss = status_text_loss * status_text_loss_weight * ctc_loss_weight
        if not collect_metrics:
            return loss, None
        return loss, {
            "transfer_status_text": {
                "loss": float(status_text_loss.detach().cpu()),
                "used": status_text_used,
                "oov": status_text_oov,
            }
        }
    if recipient_only:
        if not _uses_v12_recipient_topology(config):
            raise ValueError(
                "recipient_only loss requires the architecture v12 or v13 private recipient topology"
            )
        if recipient_logits is None or recipient_to_id is None:
            raise ValueError("recipient_only loss requires recipient logits and a train-only recipient charset")
        recipient_loss, recipient_used, recipient_oov = _ctc_loss(
            recipient_logits,
            labels=[_slot_text(record, "recipient_field") for record in records],
            character_to_id=recipient_to_id,
            torch=torch,
            sample_weights=recipient_sample_weights,
        )
        if recipient_loss is None:
            if not allow_empty:
                raise ValueError("A recipient-only training batch has no labelled recipient task")
            return None, None if not collect_metrics else {
                "recipient_field": {"loss": math.nan, "used": recipient_used, "oov": recipient_oov}
            }
        # The shared v12/v13 private recipient branch normally applies its CTC
        # multiplier alongside finite structured heads. Preserve that scalar
        # so recipient-only fine-tuning keeps the guarded full-recipe scale.
        loss = recipient_loss * recipient_loss_weight * ctc_loss_weight
        if not collect_metrics:
            return loss, None
        return loss, {
            "recipient_field": {
                "loss": float(recipient_loss.detach().cpu()),
                "used": recipient_used,
                "oov": recipient_oov,
            }
        }

    amount_loss, amount_used, amount_oov = _ctc_loss(
        amount_logits,
        labels=[_ctc_slot_text(record, "amount", config=config) for record in records],
        character_to_id=amount_to_id,
        torch=torch,
    )
    time_loss, time_used, time_oov = _ctc_loss(
        time_logits,
        labels=[_ctc_slot_text(record, "time", config=config) for record in records],
        character_to_id=time_to_id,
        torch=torch,
    )
    payment_loss, payment_used, payment_oov = _ctc_loss(
        payment_logits,
        labels=[_slot_text(record, "payment_method_field") for record in records],
        character_to_id=payment_to_id,
        torch=torch,
    )
    if _uses_recipient_protocol(config):
        if recipient_logits is None or recipient_to_id is None:
            raise ValueError("Unified v9-v12 loss requires recipient logits and a train-only recipient charset")
        recipient_loss, recipient_used, recipient_oov = _ctc_loss(
            recipient_logits,
            labels=[_slot_text(record, "recipient_field") for record in records],
            character_to_id=recipient_to_id,
            torch=torch,
            sample_weights=recipient_sample_weights,
        )
    else:
        recipient_loss, recipient_used, recipient_oov = None, 0, 0
    if _is_v13(config):
        if status_text_logits is None or status_text_to_id is None:
            raise ValueError("Unified v13 loss requires status text logits and a train-only charset")
        status_text_loss, status_text_used, status_text_oov = _ctc_loss(
            status_text_logits,
            labels=[_slot_text(record, "transfer_status") for record in records],
            character_to_id=status_text_to_id,
            torch=torch,
        )
    else:
        status_text_loss, status_text_used, status_text_oov = None, 0, 0
    if status_enabled:
        if status_criterion is None:
            raise ValueError("status_criterion is required when status_enabled is true")
        status_loss, status_used = _status_loss(
            status_logits,
            labels=[_status_name(record) for record in records],
            status_to_id=status_to_id,
            criterion=status_criterion,
            torch=torch,
        )
    else:
        status_loss, status_used = None, 0

    amount_sign_loss: Any | None = None
    amount_length_loss: Any | None = None
    amount_digits_loss: Any | None = None
    amount_currency_style_loss: Any | None = None
    amount_grouped_thousands_loss: Any | None = None
    amount_sign_position_loss: Any | None = None
    time_format_loss: Any | None = None
    time_digits_loss: Any | None = None
    time_hour_width_loss: Any | None = None
    payment_prefix_loss: Any | None = None
    payment_bank_prefix_loss: Any | None = None
    payment_tail_loss: Any | None = None
    payment_structure_loss: Any | None = None
    payment_parentheses_loss: Any | None = None
    amount_structured_used = 0
    time_structured_used = 0
    payment_card_tail_used = 0
    payment_bank_prefix_used = 0
    payment_structure_used = 0
    if config.architecture_version == 5:
        if structured_outputs is None:
            raise ValueError("Unified v5 loss requires structured output tensors")
        amount_targets = [_amount_structured_target(record) for record in records]
        amount_length_loss, amount_structured_used = _structured_cross_entropy_loss(
            structured_outputs["amount_length_logits"],
            labels=[target[0] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        amount_digits_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["amount_digit_logits"],
            labels=[target[1] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        time_targets = [_time_structured_target(record) for record in records]
        time_digits_loss, time_structured_used = _structured_cross_entropy_loss(
            structured_outputs["time_digit_logits"],
            labels=[target[0] if target is not None else None for target in time_targets],
            torch=torch,
        )
        time_hour_width_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["time_hour_width_logits"],
            labels=[target[1] if target is not None else None for target in time_targets],
            torch=torch,
        )
        payment_targets = [_payment_card_tail_target(record) for record in records]
        payment_prefix_loss, payment_card_tail_used, _ = _ctc_loss(
            structured_outputs["payment_prefix_logits"],
            labels=[target[0] if target is not None else None for target in payment_targets],
            character_to_id=payment_to_id,
            torch=torch,
        )
        payment_tail_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["payment_tail_digit_logits"],
            labels=[target[1] if target is not None else None for target in payment_targets],
            torch=torch,
        )
        payment_parentheses_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["payment_parentheses_logits"],
            labels=[target[2] if target is not None else None for target in payment_targets],
            torch=torch,
        )
        payment_structure_loss, payment_structure_used = _structured_cross_entropy_loss(
            structured_outputs["payment_structure_logits"],
            labels=[1 if target is not None else 0 if _slot_text(record, "payment_method_field") is not None else None for record, target in zip(records, payment_targets)],
            torch=torch,
        )
    elif _uses_v8_protocol(config):
        if structured_outputs is None:
            raise ValueError("Unified v8 loss requires structured output tensors")
        if payment_bank_prefix_classes is None:
            raise ValueError("Unified v8 loss requires train-only payment bank-prefix classes")
        amount_targets = [_amount_v8_format_target(record) for record in records]
        amount_currency_style_loss, amount_structured_used = _structured_cross_entropy_loss(
            structured_outputs["amount_currency_style_logits"],
            labels=[target[0] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        amount_grouped_thousands_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["amount_grouped_thousands_logits"],
            labels=[target[1] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        amount_sign_position_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["amount_sign_position_logits"],
            labels=[target[2] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        time_targets = [_time_v6_structured_target(record) for record in records]
        time_format_loss, time_structured_used = _structured_cross_entropy_loss(
            structured_outputs["time_format_logits"],
            labels=[target[0] if target is not None else None for target in time_targets],
            torch=torch,
        )
        time_digits_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["time_digit_logits"],
            labels=[target[1] if target is not None else None for target in time_targets],
            torch=torch,
        )
        payment_targets = [_payment_card_tail_target(record) for record in records]
        payment_prefix_loss, payment_card_tail_used, _ = _ctc_loss(
            structured_outputs["payment_prefix_logits"],
            labels=[target[0] if target is not None else None for target in payment_targets],
            character_to_id=payment_to_id,
            torch=torch,
        )
        payment_tail_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["payment_tail_digit_logits"],
            labels=[target[1] if target is not None else None for target in payment_targets],
            torch=torch,
        )
        payment_parentheses_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["payment_parentheses_logits"],
            labels=[target[2] if target is not None else None for target in payment_targets],
            torch=torch,
        )
        payment_structure_loss, payment_structure_used = _structured_cross_entropy_loss(
            structured_outputs["payment_structure_logits"],
            labels=[
                1 if target is not None else 0 if _slot_text(record, "payment_method_field") is not None else None
                for record, target in zip(records, payment_targets)
            ],
            torch=torch,
        )
        payment_bank_prefix_loss, payment_bank_prefix_used = _structured_cross_entropy_loss(
            structured_outputs["payment_bank_prefix_logits"],
            labels=[
                _payment_bank_prefix_class_target(record, classes=payment_bank_prefix_classes)
                for record in records
            ],
            torch=torch,
            class_weight=payment_bank_class_weights,
        )
    elif _uses_v6_protocol(config):
        if structured_outputs is None:
            raise ValueError("Unified v6 loss requires structured output tensors")
        if payment_bank_prefix_classes is None:
            raise ValueError("Unified v6 loss requires train-only payment bank-prefix classes")
        amount_targets = [_amount_v6_structured_target(record) for record in records]
        amount_sign_loss, amount_structured_used = _structured_cross_entropy_loss(
            structured_outputs["amount_sign_logits"],
            labels=[target[0] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        amount_length_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["amount_length_logits"],
            labels=[target[1] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        amount_digits_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["amount_digit_logits"],
            labels=[target[2] if target is not None else None for target in amount_targets],
            torch=torch,
        )
        time_targets = [_time_v6_structured_target(record) for record in records]
        time_format_loss, time_structured_used = _structured_cross_entropy_loss(
            structured_outputs["time_format_logits"],
            labels=[target[0] if target is not None else None for target in time_targets],
            torch=torch,
        )
        time_digits_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["time_digit_logits"],
            labels=[target[1] if target is not None else None for target in time_targets],
            torch=torch,
        )
        payment_targets = [_payment_card_tail_target(record) for record in records]
        payment_prefix_loss, payment_card_tail_used, _ = _ctc_loss(
            structured_outputs["payment_prefix_logits"],
            labels=[target[0] if target is not None else None for target in payment_targets],
            character_to_id=payment_to_id,
            torch=torch,
        )
        payment_tail_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["payment_tail_digit_logits"],
            labels=[target[1] if target is not None else None for target in payment_targets],
            torch=torch,
        )
        payment_parentheses_loss, _ = _structured_cross_entropy_loss(
            structured_outputs["payment_parentheses_logits"],
            labels=[target[2] if target is not None else None for target in payment_targets],
            torch=torch,
        )
        payment_structure_loss, payment_structure_used = _structured_cross_entropy_loss(
            structured_outputs["payment_structure_logits"],
            labels=[
                1 if target is not None else 0 if _slot_text(record, "payment_method_field") is not None else None
                for record, target in zip(records, payment_targets)
            ],
            torch=torch,
        )
        payment_bank_prefix_loss, payment_bank_prefix_used = _structured_cross_entropy_loss(
            structured_outputs["payment_bank_prefix_logits"],
            labels=[
                _payment_bank_prefix_class_target(record, classes=payment_bank_prefix_classes)
                for record in records
            ],
            torch=torch,
            class_weight=payment_bank_class_weights,
        )
    pieces: list[Any] = []
    if _uses_structured_heads(config):
        if amount_loss is not None:
            pieces.append(amount_loss * ctc_loss_weight)
        if time_loss is not None:
            pieces.append(time_loss * ctc_loss_weight)
        if payment_loss is not None:
            pieces.append(payment_loss * payment_loss_weight * ctc_loss_weight)
        if recipient_loss is not None:
            pieces.append(recipient_loss * recipient_loss_weight * ctc_loss_weight)
        if status_text_loss is not None:
            pieces.append(status_text_loss * status_text_loss_weight * ctc_loss_weight)
        if amount_length_loss is not None:
            pieces.append(amount_length_loss * structured_loss_weight)
        if amount_digits_loss is not None:
            pieces.append(amount_digits_loss * structured_loss_weight)
        if amount_sign_loss is not None:
            pieces.append(amount_sign_loss * structured_loss_weight)
        amount_format_losses = [
            loss
            for loss in (
                amount_currency_style_loss,
                amount_grouped_thousands_loss,
                amount_sign_position_loss,
            )
            if loss is not None
        ]
        if amount_format_losses:
            # The three finite choices describe one display grammar. Average
            # them so v8 does not overweight amount simply because it has
            # three tiny classifiers.
            pieces.append(torch.stack(amount_format_losses).mean() * structured_loss_weight)
        if time_digits_loss is not None:
            pieces.append(time_digits_loss * structured_loss_weight)
        if time_format_loss is not None:
            pieces.append(time_format_loss * structured_loss_weight)
        if time_hour_width_loss is not None:
            pieces.append(time_hour_width_loss * structured_loss_weight)
        if payment_prefix_loss is not None:
            pieces.append(payment_prefix_loss * payment_loss_weight * structured_loss_weight)
        if payment_tail_loss is not None:
            # The exact four digits are the demonstrated v4 failure mode;
            # give their compact, directly supervised head a modest priority.
            pieces.append(payment_tail_loss * payment_loss_weight * structured_loss_weight * 1.5)
        if payment_bank_prefix_loss is not None:
            # A finite bank-name vote directly targets the most damaging long
            # Chinese CTC substitutions.  It remains an auxiliary verifier,
            # not an automatic delivery decision.
            pieces.append(payment_bank_prefix_loss * payment_loss_weight * structured_loss_weight * 1.5)
        if payment_structure_loss is not None:
            pieces.append(payment_structure_loss * payment_loss_weight * structured_loss_weight)
        if payment_parentheses_loss is not None:
            pieces.append(payment_parentheses_loss * payment_loss_weight * structured_loss_weight * 0.25)
    else:
        if amount_loss is not None:
            pieces.append(amount_loss)
        if time_loss is not None:
            pieces.append(time_loss)
        if payment_loss is not None:
            pieces.append(payment_loss * payment_loss_weight)
        if recipient_loss is not None:
            pieces.append(recipient_loss * recipient_loss_weight)
        if status_text_loss is not None:
            pieces.append(status_text_loss * status_text_loss_weight)
    if status_loss is not None:
        pieces.append(status_loss)
    if not pieces:
        if not allow_empty:
            raise ValueError("A training batch has no labelled unified-reader task")
        loss: Any | None = None
    else:
        loss = torch.stack(pieces).mean()
    if not collect_metrics:
        return loss, None
    return loss, {
        "amount": {"loss": float(amount_loss.detach().cpu()) if amount_loss is not None else math.nan, "used": amount_used, "oov": amount_oov},
        "time": {"loss": float(time_loss.detach().cpu()) if time_loss is not None else math.nan, "used": time_used, "oov": time_oov},
        "payment_method_field": {
            "loss": float(payment_loss.detach().cpu()) if payment_loss is not None else math.nan,
            "used": payment_used,
            "oov": payment_oov,
        },
        "recipient_field": {
            "loss": float(recipient_loss.detach().cpu()) if recipient_loss is not None else math.nan,
            "used": recipient_used,
            "oov": recipient_oov,
        },
        "transfer_status": {"loss": float(status_loss.detach().cpu()) if status_loss is not None else math.nan, "used": status_used, "oov": 0},
        "transfer_status_text": {
            "loss": float(status_text_loss.detach().cpu()) if status_text_loss is not None else math.nan,
            "used": status_text_used,
            "oov": status_text_oov,
        },
        "amount_structured": {
            "sign_loss": float(amount_sign_loss.detach().cpu()) if amount_sign_loss is not None else math.nan,
            "length_loss": float(amount_length_loss.detach().cpu()) if amount_length_loss is not None else math.nan,
            "digits_loss": float(amount_digits_loss.detach().cpu()) if amount_digits_loss is not None else math.nan,
            "currency_style_loss": float(amount_currency_style_loss.detach().cpu())
            if amount_currency_style_loss is not None
            else math.nan,
            "grouped_thousands_loss": float(amount_grouped_thousands_loss.detach().cpu())
            if amount_grouped_thousands_loss is not None
            else math.nan,
            "sign_position_loss": float(amount_sign_position_loss.detach().cpu())
            if amount_sign_position_loss is not None
            else math.nan,
            "used": amount_structured_used,
        },
        "time_structured": {
            "format_loss": float(time_format_loss.detach().cpu()) if time_format_loss is not None else math.nan,
            "digits_loss": float(time_digits_loss.detach().cpu()) if time_digits_loss is not None else math.nan,
            "hour_width_loss": float(time_hour_width_loss.detach().cpu()) if time_hour_width_loss is not None else math.nan,
            "used": time_structured_used,
        },
        "payment_structured": {
            "prefix_loss": float(payment_prefix_loss.detach().cpu()) if payment_prefix_loss is not None else math.nan,
            "bank_prefix_loss": float(payment_bank_prefix_loss.detach().cpu())
            if payment_bank_prefix_loss is not None
            else math.nan,
            "tail_loss": float(payment_tail_loss.detach().cpu()) if payment_tail_loss is not None else math.nan,
            "structure_loss": float(payment_structure_loss.detach().cpu()) if payment_structure_loss is not None else math.nan,
            "parentheses_loss": float(payment_parentheses_loss.detach().cpu())
            if payment_parentheses_loss is not None
            else math.nan,
            "card_tail_used": payment_card_tail_used,
            "bank_prefix_used": payment_bank_prefix_used,
            "structure_used": payment_structure_used,
        },
    }


def _evaluate_model(
    model: Any,
    loader: Any,
    *,
    config: UnifiedReaderConfig,
    device: str,
    amount_characters: Sequence[str],
    amount_to_id: Mapping[str, int],
    time_characters: Sequence[str],
    time_to_id: Mapping[str, int],
    payment_characters: Sequence[str],
    payment_to_id: Mapping[str, int],
    recipient_characters: Sequence[str] | None,
    recipient_to_id: Mapping[str, int] | None,
    status_text_characters: Sequence[str] | None,
    status_text_to_id: Mapping[str, int] | None,
    payment_bank_prefix_classes: Sequence[str] | None,
    payment_bank_class_weights: Any | None,
    status_to_id: Mapping[str, int],
    status_criterion: Any | None,
    status_enabled: bool,
    payment_loss_weight: float,
    recipient_loss_weight: float,
    status_text_loss_weight: float,
    ctc_loss_weight: float,
    structured_loss_weight: float,
    torch: Any,
    status_text_only: bool = False,
) -> dict[str, object]:
    """Evaluate every available reader head without discarding held-out OOV labels."""
    model.eval()
    uses_cuda = device.startswith("cuda")
    total_loss = 0.0
    loss_receipts = 0
    exact_total = 0
    label_total = 0
    slot_order = _slot_order(config)
    counters: dict[str, Counter[str]] = {field: Counter() for field in slot_order}
    ctc_fields = (
        ("amount", "time", "payment_method_field", "recipient_field")
        if _uses_recipient_protocol(config)
        else ("amount", "time", "payment_method_field")
    )
    if _is_v13(config):
        ctc_fields = (*ctc_fields, "transfer_status")
    ctc_counters: dict[str, Counter[str]] = {field: Counter() for field in ctc_fields}
    verifier_counters: dict[str, Counter[str]] = {
        field: Counter() for field in ("amount", "time", "payment_method_field")
    }
    with torch.no_grad():
        for batch in loader:
            field_images, recipient_value_images, records = _unpack_receipt_batch(batch, config=config)
            field_images = field_images.to(device, non_blocking=uses_cuda)
            if recipient_value_images is not None:
                recipient_value_images = recipient_value_images.to(device, non_blocking=uses_cuda)
            outputs = _unpack_reader_outputs(
                model(field_images, recipient_value_images),
                config=config,
            )
            amount_logits = outputs["amount_logits"]
            time_logits = outputs["time_logits"]
            payment_logits = outputs["payment_logits"]
            status_logits = outputs["status_logits"]
            recipient_logits = outputs.get("recipient_logits")
            status_text_logits = outputs.get("status_text_logits")
            loss, _ = _batch_loss(
                amount_logits,
                time_logits,
                payment_logits,
                status_logits,
                records,
                amount_to_id=amount_to_id,
                time_to_id=time_to_id,
                payment_to_id=payment_to_id,
                recipient_logits=recipient_logits,
                recipient_to_id=recipient_to_id,
                status_text_logits=status_text_logits,
                status_text_to_id=status_text_to_id,
                payment_bank_prefix_classes=payment_bank_prefix_classes,
                payment_bank_class_weights=payment_bank_class_weights,
                status_to_id=status_to_id,
                status_criterion=status_criterion,
                status_enabled=status_enabled,
                payment_loss_weight=payment_loss_weight,
                recipient_loss_weight=recipient_loss_weight,
                status_text_loss_weight=status_text_loss_weight,
                config=config,
                structured_outputs=outputs if _uses_structured_heads(config) else None,
                ctc_loss_weight=ctc_loss_weight,
                structured_loss_weight=structured_loss_weight,
                torch=torch,
                allow_empty=True,
                collect_metrics=False,
                status_text_only=status_text_only,
            )
            if loss is not None:
                total_loss += float(loss.detach().cpu()) * len(records)
                loss_receipts += len(records)
            amount_ctc_scored = decode_ctc_logits_with_confidence(
                amount_logits.detach().cpu().numpy(), characters=amount_characters
            )
            amount_ctc_predictions = [text for text, _ in amount_ctc_scored]
            time_ctc_predictions = decode_ctc_logits(
                time_logits.detach().cpu().numpy(), characters=time_characters
            )
            payment_ctc_predictions = decode_ctc_logits(
                payment_logits.detach().cpu().numpy(), characters=payment_characters
            )
            if _uses_recipient_protocol(config):
                if recipient_logits is None or recipient_characters is None:
                    raise AssertionError("v9-v12 evaluation requires recipient logits and charset")
                recipient_ctc_predictions = decode_ctc_logits(
                    recipient_logits.detach().cpu().numpy(), characters=recipient_characters
                )
            else:
                recipient_ctc_predictions = []
            if _is_v13(config):
                if status_text_logits is None or status_text_characters is None:
                    raise AssertionError("v13 evaluation requires status text logits and charset")
                status_text_ctc_scored = decode_ctc_logits_with_confidence(
                    status_text_logits.detach().cpu().numpy(),
                    characters=status_text_characters,
                )
                status_text_ctc_predictions = [text for text, _ in status_text_ctc_scored]
            else:
                status_text_ctc_scored = []
                status_text_ctc_predictions = []
            amount_predictions = list(amount_ctc_predictions)
            time_predictions = list(time_ctc_predictions)
            payment_predictions = list(payment_ctc_predictions)
            verifier_predictions: dict[str, list[str | None]] | None = None
            delivery_predictions: dict[str, list[str | None]] = {
                "amount": list(amount_ctc_predictions),
                "time": list(time_ctc_predictions),
                "payment_method_field": list(payment_ctc_predictions),
            }
            if _uses_recipient_protocol(config):
                # Paddle-derived recipient text must remain a diagnostic/review value
                # until human-truth calibration accepts it.
                delivery_predictions["recipient_field"] = [None] * len(records)
            if config.architecture_version == 5:
                structured_amount = _structured_amount_predictions(
                    outputs["amount_length_logits"].detach().cpu().numpy(),
                    outputs["amount_digit_logits"].detach().cpu().numpy(),
                )
                structured_time = _structured_time_predictions(
                    outputs["time_digit_logits"].detach().cpu().numpy(),
                    outputs["time_hour_width_logits"].detach().cpu().numpy(),
                )
                structured_payment = _structured_payment_predictions(
                    outputs["payment_prefix_logits"].detach().cpu().numpy(),
                    outputs["payment_tail_digit_logits"].detach().cpu().numpy(),
                    outputs["payment_structure_logits"].detach().cpu().numpy(),
                    outputs["payment_parentheses_logits"].detach().cpu().numpy(),
                    payment_characters=payment_characters,
                )
                amount_predictions = [
                    structured if structured is not None else ctc
                    for ctc, (structured, _) in zip(amount_ctc_predictions, structured_amount)
                ]
                time_predictions = [
                    structured if structured is not None else ctc
                    for ctc, (structured, _) in zip(time_ctc_predictions, structured_time)
                ]
                payment_predictions = [
                    structured if structured is not None else ctc
                    for ctc, (structured, _) in zip(payment_ctc_predictions, structured_payment)
                ]
                # Structural and CTC values are both diagnostic candidates.
                # They share the same student representation, so agreement
                # between them is not independent evidence and cannot be
                # emitted as a financial delivery value.  A later version may
                # introduce a separately calibrated acceptance policy; until
                # then every v5 text value remains review-only.
                delivery_predictions = {
                    "amount": [None] * len(records),
                    "time": [None] * len(records),
                    "payment_method_field": [None] * len(records),
                }
            elif _uses_v8_protocol(config):
                structured_amount = _structured_amount_v8_predictions(
                    amount_ctc_scored,
                    outputs["amount_currency_style_logits"].detach().cpu().numpy(),
                    outputs["amount_grouped_thousands_logits"].detach().cpu().numpy(),
                    outputs["amount_sign_position_logits"].detach().cpu().numpy(),
                    min_confidence=config.amount_format_min_confidence,
                )
                structured_time = _structured_time_v6_predictions(
                    outputs["time_format_logits"].detach().cpu().numpy(),
                    outputs["time_digit_logits"].detach().cpu().numpy(),
                )
                # Payment remains raw CTC at runtime. Its finite known-bank
                # head remains a diagnostic/auditing auxiliary only.
                amount_predictions = [
                    structured if structured is not None else ctc
                    for ctc, (structured, _) in zip(amount_ctc_predictions, structured_amount)
                ]
                time_predictions = [
                    structured if structured is not None else ctc
                    for ctc, (structured, _) in zip(time_ctc_predictions, structured_time)
                ]
                delivery_predictions = {
                    "amount": [None] * len(records),
                    "time": [None] * len(records),
                    "payment_method_field": [None] * len(records),
                }
                if _uses_recipient_protocol(config):
                    delivery_predictions["recipient_field"] = [None] * len(records)
            elif _uses_v6_protocol(config):
                structured_amount = _structured_amount_v6_predictions(
                    outputs["amount_sign_logits"].detach().cpu().numpy(),
                    outputs["amount_length_logits"].detach().cpu().numpy(),
                    outputs["amount_digit_logits"].detach().cpu().numpy(),
                )
                structured_time = _structured_time_v6_predictions(
                    outputs["time_format_logits"].detach().cpu().numpy(),
                    outputs["time_digit_logits"].detach().cpu().numpy(),
                )
                if payment_bank_prefix_classes is None:
                    raise AssertionError("v6 evaluation requires train-only bank-prefix classes")
                structured_payment = _structured_payment_v6_predictions(
                    outputs["payment_bank_prefix_logits"].detach().cpu().numpy(),
                    outputs["payment_tail_digit_logits"].detach().cpu().numpy(),
                    outputs["payment_structure_logits"].detach().cpu().numpy(),
                    outputs["payment_parentheses_logits"].detach().cpu().numpy(),
                    payment_bank_prefix_classes=payment_bank_prefix_classes,
                )
                verifier_predictions = {
                    "amount": [candidate for candidate, _ in structured_amount],
                    "time": [candidate for candidate, _ in structured_time],
                    "payment_method_field": [candidate for candidate, _ in structured_payment],
                }
                # This is the same decision rule used by the runtime helper:
                # use a valid time format candidate, retain raw amount/payment
                # CTC candidates. Keeping validation aligned prevents a best
                # checkpoint from being selected by a metric it will not use.
                time_predictions = [
                    structured if structured is not None else ctc
                    for ctc, (structured, _) in zip(time_ctc_predictions, structured_time)
                ]
                # Raw visible CTC remains separately measured.  The verifier
                # score below chooses the best v6 checkpoint, but text still
                # stays review-only until independent human-truth calibration.
                delivery_predictions = {
                    "amount": [None] * len(records),
                    "time": [None] * len(records),
                    "payment_method_field": [None] * len(records),
                }
            status_predictions = status_logits.argmax(dim=1).detach().cpu().tolist()
            for index, record in enumerate(records):
                raw_values = {
                    "amount": (_ctc_slot_text(record, "amount", config=config), amount_ctc_predictions[index]),
                    "time": (_ctc_slot_text(record, "time", config=config), time_ctc_predictions[index]),
                    "payment_method_field": (_slot_text(record, "payment_method_field"), payment_ctc_predictions[index]),
                }
                if _uses_recipient_protocol(config):
                    raw_values["recipient_field"] = (
                        _slot_text(record, "recipient_field"),
                        recipient_ctc_predictions[index],
                    )
                if _is_v13(config):
                    raw_values["transfer_status"] = (
                        _slot_text(record, "transfer_status"),
                        status_text_ctc_predictions[index],
                    )
                for field, (expected, predicted) in raw_values.items():
                    if expected is None:
                        continue
                    raw_counter = ctc_counters[field]
                    raw_counter["records"] += 1
                    raw_counter["exact_matches"] += int(str(expected) == str(predicted))
                amount_expected = _ctc_slot_text(record, "amount", config=config)
                if _uses_v8_protocol(config):
                    amount_slot = dict(record["slots"]).get("amount")
                    visible = amount_slot.get("visible_text") if isinstance(amount_slot, Mapping) else None
                    if isinstance(visible, str) and parse_amount_visible_format_target(visible) is not None:
                        amount_expected = visible
                values = {
                    "amount": (
                        amount_expected,
                        amount_predictions[index],
                        delivery_predictions["amount"][index],
                    ),
                    "time": (
                        _ctc_slot_text(record, "time", config=config),
                        time_predictions[index],
                        delivery_predictions["time"][index],
                    ),
                    "payment_method_field": (
                        _slot_text(record, "payment_method_field"),
                        payment_predictions[index],
                        delivery_predictions["payment_method_field"][index],
                    ),
                }
                if _uses_recipient_protocol(config):
                    values["recipient_field"] = (
                        _recipient_expected_value(record, config=config),
                        _recipient_candidate_value(recipient_ctc_predictions[index], config=config),
                        delivery_predictions["recipient_field"][index],
                    )
                if _is_v13(config):
                    status_candidate = normalize_status(status_text_ctc_predictions[index])
                    values["transfer_status"] = (
                        _status_name(record),
                        status_candidate,
                        None,
                    )
                elif status_enabled:
                    values["transfer_status"] = (
                        _status_name(record),
                        STATUS_CLASSES[int(status_predictions[index])],
                        STATUS_CLASSES[int(status_predictions[index])],
                    )
                for field, (expected, predicted, delivery_predicted) in values.items():
                    if expected is None:
                        continue
                    field_counter = counters[field]
                    field_counter["records"] += 1
                    if field == "payment_method_field" and any(
                        character not in payment_to_id for character in str(expected)
                    ):
                        field_counter["oov_reference"] += 1
                    if field == "recipient_field" and recipient_to_id is not None:
                        # v10's business metric compares the extracted
                        # merchant value, but CTC is trained on the complete
                        # visible row.  Audit OOV against the actual CTC
                        # target so training-time validation agrees with the
                        # exported-ONNX evaluator.
                        recipient_ctc_target = (
                            _ctc_slot_text(record, "recipient_field", config=config) or str(expected)
                        )
                        if any(character not in recipient_to_id for character in recipient_ctc_target):
                            field_counter["oov_reference"] += 1
                    if field == "transfer_status" and status_text_to_id is not None:
                        status_ctc_target = _slot_text(record, "transfer_status")
                        if status_ctc_target is not None and any(
                            character not in status_text_to_id for character in status_ctc_target
                        ):
                            field_counter["oov_reference"] += 1
                    matched = str(expected) == str(predicted)
                    field_counter["exact_matches"] += int(matched)
                    if delivery_predicted is not None:
                        delivery_matched = str(expected) == str(delivery_predicted)
                        field_counter["delivery_records"] += 1
                        field_counter["delivery_exact_matches"] += int(delivery_matched)
                        field_counter["delivery_false_accepts"] += int(not delivery_matched)
                    exact_total += int(matched)
                    label_total += 1
                    if field == "transfer_status" and expected != "success" and predicted == "success":
                        field_counter["non_success_to_success"] += 1
                if verifier_predictions is not None:
                    for field, predictions in verifier_predictions.items():
                        expected = _v6_verifier_target_text(record, field)
                        if expected is None:
                            continue
                        verifier_counter = verifier_counters[field]
                        verifier_counter["records"] += 1
                        verifier_counter["exact_matches"] += int(expected == predictions[index])
    if not loss_receipts or not label_total:
        raise ValueError("Validation set has no CTC/classification labels covered by the training charset")
    delivery_records = sum(counter["delivery_records"] for counter in counters.values())
    delivery_exact_matches = sum(counter["delivery_exact_matches"] for counter in counters.values())
    verifier_records = sum(counter["records"] for counter in verifier_counters.values())
    verifier_exact_matches = sum(counter["exact_matches"] for counter in verifier_counters.values())
    verifier_field_scores = [
        counter["exact_matches"] / counter["records"]
        for counter in verifier_counters.values()
        if counter["records"]
    ]
    candidate_text_fields = ("amount", "time", "payment_method_field") + (
        ("recipient_field",) if _uses_recipient_protocol(config) else ()
    )
    if _is_v13(config):
        candidate_text_fields = (*candidate_text_fields, "transfer_status")
    candidate_text_records = sum(counters[field]["records"] for field in candidate_text_fields)
    candidate_text_exact_matches = sum(counters[field]["exact_matches"] for field in candidate_text_fields)
    candidate_text_field_scores = [
        counters[field]["exact_matches"] / counters[field]["records"]
        for field in candidate_text_fields
        if counters[field]["records"]
    ]
    return {
        "loss": total_loss / loss_receipts,
        "exact_match": exact_total / label_total,
        "delivery_coverage": delivery_records / label_total,
        "delivery_exact_match": delivery_exact_matches / max(1, delivery_records),
        "delivery_exact_overall": delivery_exact_matches / label_total,
        "delivery_false_accepts": int(
            sum(counter["delivery_false_accepts"] for counter in counters.values())
        ),
        "verifier_exact_match": verifier_exact_matches / verifier_records if verifier_records else None,
        "verifier_macro_exact_match": sum(verifier_field_scores) / len(verifier_field_scores)
        if verifier_field_scores
        else None,
        "verifier_by_field": {
            field: {
                "records": int(counter["records"]),
                "exact_matches": int(counter["exact_matches"]),
                "exact_match": counter["exact_matches"] / counter["records"] if counter["records"] else None,
            }
            for field, counter in verifier_counters.items()
        },
        "candidate_text_exact_match": candidate_text_exact_matches / max(1, candidate_text_records),
        "candidate_text_macro_exact_match": sum(candidate_text_field_scores) / len(candidate_text_field_scores)
        if candidate_text_field_scores
        else None,
        "candidate_text_by_field": {
            field: {
                "records": int(counters[field]["records"]),
                "exact_matches": int(counters[field]["exact_matches"]),
                "exact_match": counters[field]["exact_matches"] / max(1, counters[field]["records"]),
            }
            for field in candidate_text_fields
        },
        "ctc_by_field": {
            field: {
                "records": int(counter["records"]),
                "exact_matches": int(counter["exact_matches"]),
                "exact_match": counter["exact_matches"] / max(1, counter["records"]),
            }
            for field, counter in ctc_counters.items()
        },
        "by_field": {
            field: {
                "records": int(counter["records"]),
                "exact_matches": int(counter["exact_matches"]),
                "exact_match": counter["exact_matches"] / max(1, counter["records"]),
                "oov_reference_records": int(counter["oov_reference"]),
                "non_success_to_success": int(counter["non_success_to_success"]),
                "delivery_coverage": counter["delivery_records"] / max(1, counter["records"]),
                "delivery_exact_match": counter["delivery_exact_matches"] / max(1, counter["delivery_records"]),
                "delivery_false_accepts": int(counter["delivery_false_accepts"]),
            }
            for field, counter in counters.items()
        },
        "status_non_success_to_success": int(counters["transfer_status"]["non_success_to_success"]),
        "status_training_enabled": status_enabled,
        "status_text_training_enabled": _is_v13(config),
    }


def _write_checkpoint(path: Path, payload: Mapping[str, object], *, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _checkpoint_selection_policy(
    *,
    config: UnifiedReaderConfig,
    checkpoint_selection: str,
    checkpoint_min_amount_candidate_exact: float | None,
    checkpoint_min_time_candidate_exact: float | None,
    checkpoint_min_payment_candidate_exact: float | None,
    status_text_only_fine_tune: bool = False,
) -> dict[str, object]:
    """Validate and freeze a training-only best-checkpoint policy.

    ``recipient_priority`` deliberately changes only which epoch becomes
    ``best.pt``.  It does not alter the model graph, preprocessing, decoder,
    or ONNX/session ABI.  The three mature text fields receive caller-supplied
    validation floors so an experiment cannot trade them away for a higher
    recipient score.  A v13 status-text-only run additionally places raw
    visible-status CTC exact immediately after the existing status-safety
    guard and ahead of semantically normalized metrics.
    """
    if checkpoint_selection not in CHECKPOINT_SELECTION_MODES:
        allowed = ", ".join(sorted(CHECKPOINT_SELECTION_MODES))
        raise ValueError(f"checkpoint_selection must be one of: {allowed}")
    raw_minima = {
        "amount": checkpoint_min_amount_candidate_exact,
        "time": checkpoint_min_time_candidate_exact,
        "payment_method_field": checkpoint_min_payment_candidate_exact,
    }
    if status_text_only_fine_tune and not _is_v13(config):
        raise ValueError("status-text CTC checkpoint priority is supported only by architecture v13")
    status_text_selection = (
        {
            "status_text_ctc_priority": True,
            "selection_metric": (
                "status_safety_then_transfer_status_raw_ctc_exact_then_"
                "legacy_balanced_validation_score"
            ),
        }
        if status_text_only_fine_tune
        else {}
    )
    if checkpoint_selection == CHECKPOINT_SELECTION_BALANCED:
        supplied = [field for field, value in raw_minima.items() if value is not None]
        if supplied:
            raise ValueError(
                "checkpoint protection floors require checkpoint_selection=recipient_priority"
            )
        return {
            "mode": CHECKPOINT_SELECTION_BALANCED,
            "protected_minimum_candidate_exact": {},
            "selection_metric": "legacy_balanced_validation_score",
            **status_text_selection,
        }
    if not _uses_recipient_protocol(config):
        raise ValueError(
            "checkpoint_selection=recipient_priority requires architecture v9, v10, v11, v12, or v13"
        )
    missing = [field for field, value in raw_minima.items() if value is None]
    if missing:
        raise ValueError(
            "checkpoint_selection=recipient_priority requires candidate-exact floors for: "
            + ", ".join(missing)
        )
    minima: dict[str, float] = {}
    for field, value in raw_minima.items():
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"checkpoint candidate-exact floor for {field} must be a number") from None
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"checkpoint candidate-exact floor for {field} must be between 0 and 1")
        minima[field] = normalized
    return {
        "mode": CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        "protected_minimum_candidate_exact": minima,
        "selection_metric": "recipient_exact_after_protected_candidate_exact_floors",
        **(
            {
                **status_text_selection,
                "selection_metric": (
                    "status_safety_then_transfer_status_raw_ctc_exact_then_"
                    "recipient_exact_after_protected_candidate_exact_floors"
                ),
            }
            if status_text_only_fine_tune
            else {}
        ),
    }


def _validation_candidate_exact(validation: Mapping[str, object], field: str) -> float:
    """Read a finite runtime-candidate exact score from validation metrics."""
    by_field = validation.get("candidate_text_by_field")
    if not isinstance(by_field, Mapping):
        raise ValueError("validation candidate-text metrics are missing")
    metrics = by_field.get(field)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"validation candidate-text metrics are missing for {field}")
    try:
        exact_match = float(metrics.get("exact_match"))
    except (TypeError, ValueError):
        raise ValueError(f"validation candidate-text exact metric is invalid for {field}") from None
    if not math.isfinite(exact_match) or not 0.0 <= exact_match <= 1.0:
        raise ValueError(f"validation candidate-text exact metric is invalid for {field}")
    return exact_match


def _validation_ctc_exact(validation: Mapping[str, object], field: str) -> float:
    """Read a finite raw visible-text CTC exact score from validation metrics."""
    by_field = validation.get("ctc_by_field")
    if not isinstance(by_field, Mapping):
        raise ValueError("validation raw CTC metrics are missing")
    metrics = by_field.get(field)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"validation raw CTC metrics are missing for {field}")
    try:
        exact_match = float(metrics.get("exact_match"))
    except (TypeError, ValueError):
        raise ValueError(f"validation raw CTC exact metric is invalid for {field}") from None
    if not math.isfinite(exact_match) or not 0.0 <= exact_match <= 1.0:
        raise ValueError(f"validation raw CTC exact metric is invalid for {field}")
    return exact_match


def _checkpoint_selection_score(
    validation: Mapping[str, object],
    *,
    config: UnifiedReaderConfig,
    status_policy: Mapping[str, object],
    policy: Mapping[str, object],
) -> tuple[tuple[float, ...] | None, list[str]]:
    """Return an auditable best-checkpoint score or protection failures.

    The balanced branch intentionally preserves the historical score tuple.
    Recipient priority is available only after the three protected fields pass
    their independently selected validation floors.
    """
    mode = policy.get("mode")
    if mode not in CHECKPOINT_SELECTION_MODES:
        raise ValueError("checkpoint selection policy mode is invalid")
    status_text_ctc_priority = policy.get("status_text_ctc_priority", False)
    if not isinstance(status_text_ctc_priority, bool):
        raise ValueError("checkpoint selection status-text CTC priority flag is invalid")
    status_safety = (
        -float(validation["status_non_success_to_success"])
        if bool(status_policy.get("training_enabled")) or status_text_ctc_priority
        else 0.0
    )
    status_text_raw_exact: float | None = None
    if status_text_ctc_priority:
        if not _is_v13(config):
            raise ValueError("status-text CTC checkpoint priority requires architecture v13")
        # v13 semantic normalization intentionally accepts variants such as
        # `成功` as the success class.  best.pt must nevertheless prefer the
        # epoch that transcribes the complete visible string (`转账成功`), so
        # raw CTC exact follows the existing non-success->success safety guard
        # but precedes every normalized semantic/candidate metric.
        status_text_raw_exact = _validation_ctc_exact(validation, "transfer_status")
    if mode == CHECKPOINT_SELECTION_RECIPIENT_PRIORITY:
        if not _uses_recipient_protocol(config):
            raise ValueError("recipient-priority checkpoint policy requires a recipient protocol")
        raw_minima = policy.get("protected_minimum_candidate_exact")
        if not isinstance(raw_minima, Mapping):
            raise ValueError("recipient-priority checkpoint policy has no protected candidate floors")
        failures: list[str] = []
        for field in CHECKPOINT_SELECTION_PROTECTED_FIELDS:
            try:
                minimum = float(raw_minima[field])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"recipient-priority checkpoint policy has no valid floor for {field}") from None
            if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
                raise ValueError(f"recipient-priority checkpoint policy has invalid floor for {field}")
            observed = _validation_candidate_exact(validation, field)
            if observed < minimum:
                failures.append(f"{field}={observed:.6f} < {minimum:.6f}")
        if failures:
            return None, failures
        verifier_score = validation.get("verifier_macro_exact_match")
        score = (
            status_safety,
            _validation_candidate_exact(validation, "recipient_field"),
            float(validation["candidate_text_macro_exact_match"] or -1.0),
            float(validation["candidate_text_exact_match"]),
            float(verifier_score) if verifier_score is not None else -1.0,
            -float(validation["loss"]),
        )
        return (
            (score[0], status_text_raw_exact, *score[1:])
            if status_text_raw_exact is not None
            else score,
            [],
        )
    # v6+ runtime may choose a valid structured time candidate (and v8 a
    # digit-preserving rendered amount). Select checkpoints by that exact same
    # candidate path rather than by a detached verifier-only score. This is the
    # historical score and must remain byte-for-byte equivalent in semantics.
    if _uses_modern_protocol(config):
        verifier_score = (
            float(validation["verifier_macro_exact_match"])
            if validation["verifier_macro_exact_match"] is not None
            else -1.0
        )
        score = (
            status_safety,
            float(validation["candidate_text_macro_exact_match"] or -1.0),
            float(validation["candidate_text_exact_match"]),
            verifier_score,
            -float(validation["loss"]),
        )
        return (
            (score[0], status_text_raw_exact, *score[1:])
            if status_text_raw_exact is not None
            else score,
            [],
        )
    score = (
        status_safety,
        float(validation["delivery_exact_overall"]),
        float(validation["exact_match"]),
        -1.0,
        -float(validation["loss"]),
    )
    return (
        (score[0], status_text_raw_exact, *score[1:])
        if status_text_raw_exact is not None
        else score,
        [],
    )


def _label_map_sha256(values: Sequence[str]) -> str:
    """Return an unambiguous provenance digest for an ordered label map."""
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def _label_map_provenance(
    source_values: Sequence[str], *, data_derived_values: Sequence[str]
) -> dict[str, object]:
    """Describe a frozen checkpoint map without serialising another full map.

    The deployed checkpoint already persists the effective labels.  This small
    record makes a recipient-only warm start auditable when the fresh r3
    manifest would otherwise generate a different payment or bank map.
    """
    return {
        "checkpoint_count": len(source_values),
        "checkpoint_sha256": _label_map_sha256(source_values),
        "data_derived_count": len(data_derived_values),
        "data_derived_sha256": _label_map_sha256(data_derived_values),
        "identical": list(source_values) == list(data_derived_values),
    }


def _validate_recipient_input_width_expansion_config(
    source_config: UnifiedReaderConfig,
    target_config: UnifiedReaderConfig,
) -> None:
    """Allow a recipient-only seed to gain horizontal pixels, and nothing else.

    The v12 recipient branch is fully convolutional in its horizontal axis, so
    increasing its static input width changes the available glyph resolution
    and CTC time steps without changing a learned tensor shape.  This is a
    deliberately narrow warm-start exception: all financial/shared topology,
    crop geometry, recipient CNN width, decoder width, and output semantics
    must remain byte-compatible with the seed.
    """
    if not (_is_v12(source_config) and _is_v12(target_config)):
        raise ValueError("recipient_input_width_expansion is supported only by architecture v12")
    if target_config.recipient_input_width <= source_config.recipient_input_width:
        raise ValueError(
            "recipient_input_width_expansion requires a strictly larger recipient_input_width than the seed"
        )
    source_values = asdict(source_config)
    target_values = asdict(target_config)
    changed = [
        key
        for key in sorted(source_values)
        if key != "recipient_input_width" and source_values[key] != target_values.get(key)
    ]
    if changed:
        raise ValueError(
            "recipient_input_width_expansion may change only recipient_input_width; "
            f"incompatible config fields: {', '.join(changed)}"
        )


def _validate_recipient_capacity_reinit_config(
    source_config: UnifiedReaderConfig,
    target_config: UnifiedReaderConfig,
) -> None:
    """Permit a larger private recipient branch while freezing all shared topology.

    Changing a CNN or GRU width changes learned tensor shapes, so those tensors
    cannot be warm-started honestly.  This mode therefore requires a monotonic
    capacity increase and reinitialises every ``recipient_`` tensor while
    retaining every financial/shared tensor byte-for-byte from the seed.
    """
    if not (_is_v12(source_config) and _is_v12(target_config)):
        raise ValueError("recipient_capacity_reinit is supported only by architecture v12")
    source_channels = _recipient_branch_channels(source_config)
    target_channels = _recipient_branch_channels(target_config)
    source_hidden = _recipient_hidden_size(source_config)
    target_hidden = _recipient_hidden_size(target_config)
    if target_channels < source_channels or target_hidden < source_hidden:
        raise ValueError("recipient_capacity_reinit cannot reduce recipient branch capacity")
    if target_channels == source_channels and target_hidden == source_hidden:
        raise ValueError("recipient_capacity_reinit requires a larger recipient branch or hidden size")
    allowed = {"recipient_branch_channels", "recipient_hidden_size"}
    source_values = asdict(source_config)
    target_values = asdict(target_config)
    changed = [
        key
        for key in sorted(source_values)
        if key not in allowed and source_values[key] != target_values.get(key)
    ]
    if changed:
        raise ValueError(
            "recipient_capacity_reinit may change only recipient_branch_channels and recipient_hidden_size; "
            f"incompatible config fields: {', '.join(changed)}"
        )


def _validate_recipient_open_text_adapter_config(
    source_config: UnifiedReaderConfig,
    target_config: UnifiedReaderConfig,
) -> None:
    """Allow an identity-gated contextual adapter and an optional wider view."""
    if not (_is_v12(source_config) and _is_v12(target_config)):
        raise ValueError("recipient_open_text_adapter is supported only by architecture v12")
    if source_config.recipient_open_text_layers != 0:
        raise ValueError("recipient_open_text_adapter requires a seed without an existing open-text adapter")
    if target_config.recipient_open_text_layers <= 0:
        raise ValueError("recipient_open_text_adapter requires at least one open-text layer")
    if target_config.recipient_input_width < source_config.recipient_input_width:
        raise ValueError("recipient_open_text_adapter cannot reduce recipient_input_width")
    allowed = {
        "recipient_open_text_layers",
        "recipient_open_text_heads",
        "recipient_open_text_feedforward",
        "recipient_input_width",
    }
    source_values = asdict(source_config)
    target_values = asdict(target_config)
    changed = [
        key
        for key in sorted(source_values)
        if key not in allowed and source_values[key] != target_values.get(key)
    ]
    if changed:
        raise ValueError(
            "recipient_open_text_adapter may change only its contextual encoder settings and input width; "
            f"incompatible config fields: {', '.join(changed)}"
        )


def _validate_recipient_visual_context_reinit_config(
    source_config: UnifiedReaderConfig,
    target_config: UnifiedReaderConfig,
) -> None:
    """Guard the one new recipient recogniser without changing v13's ABI.

    This is intentionally not another width/capacity retry.  The source must
    be the established depthwise+BiGRU branch and the target must replace it
    with the residual visual encoder plus direct positional Transformer.  Only
    private-recipient topology/training-time dropout fields may differ; every
    shared, financial, status, input-shape and output-policy field remains
    byte-compatible with the v13 seed.
    """

    if not (_is_v13(source_config) and _is_v13(target_config)):
        raise ValueError("recipient_visual_context_reinit requires v13 source and target configs")
    if source_config.recipient_backbone != "legacy_depthwise_gru_v1":
        raise ValueError("recipient_visual_context_reinit requires the established legacy recipient seed")
    if target_config.recipient_backbone != "residual_positional_transformer_v2":
        raise ValueError("recipient_visual_context_reinit requires the residual positional Transformer target")
    allowed = {
        "recipient_backbone",
        "recipient_branch_channels",
        "recipient_hidden_size",
        "recipient_open_text_layers",
        "recipient_open_text_heads",
        "recipient_open_text_feedforward",
        "recipient_open_text_dropout",
    }
    source_values = asdict(source_config)
    target_values = asdict(target_config)
    changed = [
        key
        for key in sorted(source_values)
        if key not in allowed and source_values[key] != target_values.get(key)
    ]
    if changed:
        raise ValueError(
            "recipient_visual_context_reinit changed a non-recipient config field: "
            + ", ".join(changed)
        )


def _validate_recipient_full_crop_warmstart_config(
    source_config: UnifiedReaderConfig,
    target_config: UnifiedReaderConfig,
) -> None:
    """Permit exactly the v13 recipient-view change proven by the pilot.

    The full-crop experiment is a preprocessing intervention, not another
    architecture sweep.  It therefore keeps the v13 ONNX ABI and every model
    field byte-compatible with the seed while changing only the recipient
    high-resolution view from the historical 30 percent left trim to the
    complete production detector crop.  Learned tensors are copied by
    character in the same way as the established recipient-only expansion.
    """

    if not (_is_v13(source_config) and _is_v13(target_config)):
        raise ValueError("recipient_full_crop_warmstart requires v13 source and target configs")
    if not math.isclose(
        source_config.recipient_value_left_trim,
        0.30,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("recipient_full_crop_warmstart requires a 0.30-trim v13 seed")
    if not math.isclose(
        target_config.recipient_value_left_trim,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("recipient_full_crop_warmstart requires target recipient_value_left_trim=0")
    source_values = asdict(source_config)
    target_values = asdict(target_config)
    changed = [
        key
        for key in sorted(source_values)
        if key != "recipient_value_left_trim" and source_values[key] != target_values.get(key)
    ]
    if changed:
        raise ValueError(
            "recipient_full_crop_warmstart may change only recipient_value_left_trim; "
            f"incompatible config fields: {', '.join(changed)}"
        )


def _validate_recipient_full_crop_continuation_config(
    source_config: UnifiedReaderConfig,
    target_config: UnifiedReaderConfig,
) -> None:
    """Require the fixed legacy trim-zero continuation topology exactly.

    Unlike the first full-crop warm start, this mode is not a topology or
    preprocessing transition.  It may only reopen the already measured v13
    legacy full-crop model, with every dataclass field identical.  Keeping the
    check separate from ``strict`` prevents an ordinary checkpoint from
    acquiring the continuation authority merely because its shapes happen to
    match.
    """

    if not (_is_v13(source_config) and _is_v13(target_config)):
        raise ValueError("recipient_full_crop_continuation requires v13 source and target configs")
    if source_config.recipient_backbone != "legacy_depthwise_gru_v1":
        raise ValueError("recipient_full_crop_continuation requires the legacy recipient backbone")
    if not math.isclose(
        source_config.recipient_value_left_trim, 0.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("recipient_full_crop_continuation requires a trim-zero source")
    source_values = asdict(source_config)
    target_values = asdict(target_config)
    if source_values != target_values:
        changed = [
            key
            for key in sorted(source_values)
            if source_values[key] != target_values.get(key)
        ]
        raise ValueError(
            "recipient_full_crop_continuation requires an exact source/target config match; "
            f"incompatible config fields: {', '.join(changed)}"
        )


def _validate_recipient_full_crop_continuation_policy(
    payload: Mapping[str, object], *, torch: Any | None = None
) -> Mapping[str, object]:
    """Reopen the embedded, content-bound B8 source authority."""

    if torch is None:
        raise ValueError("continuation source revalidation requires the active torch runtime")
    try:
        from .recipient_full_crop_continuation import (
            validate_embedded_continuation_authority,
        )

        return validate_embedded_continuation_authority(payload, torch=torch)
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            "recipient_full_crop_continuation requires a valid embedded content-bound authority"
        ) from error


def _validate_recipient_full_crop_seed_policy(
    payload: Mapping[str, object], *, torch: Any | None = None
) -> None:
    """Require the content-bound v13 recipient sanitizer attestation.

    A top-level ``standard_train_only`` value can be rewritten while leaving
    transductive recipient tensors untouched.  The sanitizer proof instead
    binds the complete recipient and non-recipient state partitions plus both
    metadata partitions.  Its builder has already verified that only the
    private ``recipient_`` state came from a compatible train-only v12 seed.
    """

    try:
        from .recipient_full_crop_seed_sanitizer import (
            validate_recipient_full_crop_seed_attestation,
            verify_recipient_full_crop_seed_source_provenance,
        )

        validate_recipient_full_crop_seed_attestation(payload)
        if torch is None:
            raise ValueError("source checkpoint revalidation requires the active torch runtime")
        verify_recipient_full_crop_seed_source_provenance(payload, torch=torch)
    except (ImportError, TypeError, ValueError) as error:
        raise ValueError(
            "recipient_full_crop_warmstart requires a valid content-bound seed sanitizer attestation"
        ) from error


def _recipient_only_expansion_label_override(
    *,
    init_checkpoint: Path,
    config: UnifiedReaderConfig,
    amount_characters: Sequence[str],
    time_characters: Sequence[str],
    payment_characters: Sequence[str],
    recipient_characters: Sequence[str] | None,
    payment_bank_prefix_classes: Sequence[str] | None,
    torch: Any,
    init_checkpoint_mode: str = INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
) -> tuple[list[str], list[str], list[str], dict[str, object]]:
    """Lock financial label semantics to a v12 seed for a recipient-only run.

    A receipt manifest can gain/reorder payment text or bank-prefix labels
    between r2 and r3.  Rebuilding those output maps would reinterpret frozen
    financial classifier rows, even though recipient-only fine-tuning never
    updates them.  This narrow preflight instead keeps the seed's financial
    maps exactly.  It takes the sorted union of the seed's recipient map and
    fresh train-only Unicode characters, so a narrower r3 training shard
    cannot accidentally discard a previously learned output row while the
    persisted Unicode-map ordering remains valid.
    """
    if init_checkpoint_mode not in RECIPIENT_ONLY_INIT_CHECKPOINT_MODES:
        raise ValueError("recipient label override requires a recipient-only expansion init mode")
    v13_recipient_private_mode = (
        _is_v13(config)
        and init_checkpoint_mode in V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES
    )
    if not (_is_v12(config) or v13_recipient_private_mode):
        raise ValueError(
            "recipient-only expansion is supported by architecture v12, or by an audited v13 private-recipient mode"
        )
    if recipient_characters is None or payment_bank_prefix_classes is None:
        raise ValueError("recipient_only_expansion requires v12 recipient and payment bank label maps")
    checkpoint_path = Path(init_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    if (
        _has_analysis_only_full_crop_continuation_lineage(payload)
        and init_checkpoint_mode != INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
    ):
        raise ValueError(
            "an analysis-only full-crop continuation authority cannot be used by another init mode"
        )
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART:
        _validate_recipient_full_crop_seed_policy(payload, torch=torch)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
        _validate_recipient_full_crop_continuation_policy(payload, torch=torch)
    source_config = _checkpoint_config(payload)
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION:
        # Validate even when the two dataclasses compare equal: this mode is a
        # strictly-wider exception, never an alias for an ordinary recipient
        # warm start.  Without the unconditional check a same-width seed could
        # accidentally enter the new pilot path.
        _validate_recipient_input_width_expansion_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_CAPACITY_REINIT:
        _validate_recipient_capacity_reinit_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER:
        _validate_recipient_open_text_adapter_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT:
        _validate_recipient_visual_context_reinit_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART:
        _validate_recipient_full_crop_warmstart_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
        _validate_recipient_full_crop_continuation_config(source_config, config)
    elif source_config != config:
        raise ValueError(
            "init checkpoint model config does not match the requested training config; "
            "use the same architecture, input sizes, head widths, and decoder policy"
        )
    (
        source_amount_characters,
        source_time_characters,
        source_payment_characters,
        source_recipient_characters,
        source_status_classes,
        source_payment_bank_prefix_classes,
    ) = _checkpoint_labels(payload, config=source_config)
    for label, source_values, current_values in (
        ("amount character map", source_amount_characters, amount_characters),
        ("time character map", source_time_characters, time_characters),
        ("status class map", source_status_classes, STATUS_CLASSES),
    ):
        if list(source_values) != list(current_values):
            raise ValueError(f"init checkpoint {label} does not match the current training data")
    if source_recipient_characters is None:
        raise ValueError("init checkpoint recipient character map does not match the current training data")
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
        for label, source_values, current_values in (
            ("payment character map", source_payment_characters, payment_characters),
            ("recipient character map", source_recipient_characters, recipient_characters),
            (
                "payment bank-prefix class map",
                source_payment_bank_prefix_classes,
                payment_bank_prefix_classes,
            ),
        ):
            if source_values is None or current_values is None or list(source_values) != list(current_values):
                raise ValueError(
                    f"recipient_full_crop_continuation requires an exact {label} match"
                )
    source_recipient_set = set(source_recipient_characters)
    fresh_train_only_recipient_map = (
        init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT
    )
    effective_recipient_characters = (
        sorted(set(recipient_characters))
        if fresh_train_only_recipient_map
        else (
            list(source_recipient_characters)
            if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
            else sorted(source_recipient_set | set(recipient_characters))
        )
    )
    if source_payment_bank_prefix_classes is None:
        raise ValueError("init checkpoint payment bank-prefix class map does not match the current training data")
    return (
        list(source_payment_characters),
        list(source_payment_bank_prefix_classes),
        effective_recipient_characters,
        {
            "mode": (
                "checkpoint_financial_label_maps_recipient_input_width_expansion_v1"
                if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION
                else (
                    "checkpoint_financial_label_maps_recipient_capacity_reinit_v1"
                    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_CAPACITY_REINIT
                    else (
                        "checkpoint_financial_label_maps_recipient_visual_context_reinit_v1"
                        if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT
                        else (
                            "checkpoint_financial_label_maps_recipient_open_text_adapter_v1"
                            if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER
                            else (
                                "checkpoint_financial_label_maps_recipient_full_crop_warmstart_v1"
                                if init_checkpoint_mode
                                == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART
                                else (
                                    "checkpoint_all_label_maps_recipient_full_crop_continuation_v1"
                                    if init_checkpoint_mode
                                    == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
                                    else "checkpoint_financial_label_maps_v1"
                                )
                            )
                        )
                    )
                )
            ),
            "reason": (
                "recipient-only v12/v13 fine-tune freezes every non-recipient parameter, so payment and bank "
                "classifier row semantics remain locked to the compatible seed checkpoint"
            ),
            **(
                {
                    "source_recipient_input_width": source_config.recipient_input_width,
                    "target_recipient_input_width": config.recipient_input_width,
                }
                if init_checkpoint_mode in {
                    INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION,
                    INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER,
                }
                else {}
            ),
            **(
                {
                    "source_recipient_value_left_trim": source_config.recipient_value_left_trim,
                    "target_recipient_value_left_trim": config.recipient_value_left_trim,
                }
                if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART
                else {}
            ),
            "payment_character_map": _label_map_provenance(
                source_payment_characters,
                data_derived_values=payment_characters,
            ),
            "payment_bank_prefix_class_map": _label_map_provenance(
                source_payment_bank_prefix_classes,
                data_derived_values=payment_bank_prefix_classes,
            ),
            "recipient_character_map": {
                "mode": (
                    "fresh_train_only_reinitialized_recipient_v1"
                    if fresh_train_only_recipient_map
                    else "checkpoint_base_plus_train_only_additions_v1"
                ),
                "checkpoint_count": len(source_recipient_characters),
                "checkpoint_sha256": _label_map_sha256(source_recipient_characters),
                "data_derived_count": len(recipient_characters),
                "data_derived_sha256": _label_map_sha256(recipient_characters),
                "effective_count": len(effective_recipient_characters),
                "effective_sha256": _label_map_sha256(effective_recipient_characters),
                "checkpoint_characters_retained_not_in_current_train_count": len(
                    set(effective_recipient_characters) - set(recipient_characters)
                ),
                "checkpoint_characters_discarded_for_blind_reinit_count": (
                    len(source_recipient_set - set(recipient_characters))
                    if fresh_train_only_recipient_map
                    else 0
                ),
                "new_data_derived_character_count": len(
                    set(recipient_characters) - source_recipient_set
                ),
            },
        },
    )


def _status_text_only_legacy_label_override(
    *,
    init_checkpoint: Path,
    config: UnifiedReaderConfig,
    amount_characters: Sequence[str],
    time_characters: Sequence[str],
    payment_characters: Sequence[str],
    recipient_characters: Sequence[str] | None,
    payment_bank_prefix_classes: Sequence[str] | None,
    torch: Any,
) -> tuple[list[str], list[str], list[str], dict[str, object]]:
    """Keep every v12-compatible label row locked during v13 head-only training.

    Rebuilding a manifest with newer filtering code can add, remove, or reorder
    payment/recipient labels even when it originates from the same flat teacher
    file.  A status-only run freezes all legacy parameters, so interpreting
    those rows with newly derived maps would corrupt the old 15 outputs before
    the optimiser even starts.  Load the exact maps from the seed instead and
    retain fresh-data differences only as OOV/evaluation evidence.
    """
    if not _is_v13(config):
        raise ValueError("status-text legacy label override requires architecture v13")
    if recipient_characters is None or payment_bank_prefix_classes is None:
        raise ValueError("status-text-only v13 requires recipient and payment bank label maps")
    checkpoint_path = Path(init_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    if _has_analysis_only_full_crop_continuation_lineage(payload):
        raise ValueError(
            "an analysis-only full-crop continuation authority cannot seed status-text training"
        )
    source_config = _checkpoint_config(payload)
    if _is_v12(source_config):
        source_values = asdict(source_config)
        source_values["architecture_version"] = 13
        if source_values != asdict(config):
            changed = [
                key
                for key in sorted(asdict(config))
                if source_values.get(key) != asdict(config).get(key)
            ]
            raise ValueError(
                "v12 status-text expansion may change only architecture_version; incompatible config fields: "
                + ", ".join(changed)
            )
    elif source_config != config:
        raise ValueError("v13 status-text seed config does not match the requested training config")
    (
        source_amount_characters,
        source_time_characters,
        source_payment_characters,
        source_recipient_characters,
        source_status_classes,
        source_payment_bank_prefix_classes,
    ) = _checkpoint_labels(payload, config=source_config)
    for label, source_values, current_values in (
        ("amount character map", source_amount_characters, amount_characters),
        ("time character map", source_time_characters, time_characters),
        ("status class map", source_status_classes, STATUS_CLASSES),
    ):
        if list(source_values) != list(current_values):
            raise ValueError(f"init checkpoint {label} does not match the current training data")
    if source_recipient_characters is None or source_payment_bank_prefix_classes is None:
        raise ValueError("status-text seed has no complete v12 legacy label maps")
    seed_recipient_oov = _validated_recipient_oov_audit(
        payload.get("recipient_oov_by_split"),
        source="status-text seed checkpoint",
    )
    return (
        list(source_payment_characters),
        list(source_payment_bank_prefix_classes),
        list(source_recipient_characters),
        {
            "mode": "checkpoint_legacy_label_maps_status_text_only_v1",
            "reason": (
                "status-text-only v13 freezes every legacy parameter, so all v12-compatible output-row "
                "semantics remain locked to the seed checkpoint"
            ),
            "seed_recipient_oov_by_split": seed_recipient_oov,
            "payment_character_map": _label_map_provenance(
                source_payment_characters,
                data_derived_values=payment_characters,
            ),
            "payment_bank_prefix_class_map": _label_map_provenance(
                source_payment_bank_prefix_classes,
                data_derived_values=payment_bank_prefix_classes,
            ),
            "recipient_character_map": {
                **_label_map_provenance(
                    source_recipient_characters,
                    data_derived_values=recipient_characters,
                ),
                "effective_count": len(source_recipient_characters),
                "effective_sha256": _label_map_sha256(source_recipient_characters),
            },
        },
    )


def _recipient_classifier_unicode_expansion_state(
    *,
    source_state_dict: Mapping[str, object],
    target_state_dict: Mapping[str, object],
    source_recipient_characters: Sequence[str],
    target_recipient_characters: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Map a compatible v12 recipient classifier by Unicode rather than row.

    Only ``recipient_classifier`` depends on the recipient Unicode table.
    Every other tensor must retain the seed's exact shape and value.  New
    target characters intentionally keep the deterministically seeded target
    initialisation until training observes them.
    """
    source_keys = set(source_state_dict)
    target_keys = set(target_state_dict)
    if source_keys != target_keys:
        missing = sorted(str(key) for key in target_keys - source_keys)
        unexpected = sorted(str(key) for key in source_keys - target_keys)
        raise ValueError(
            "init checkpoint model parameters do not match the recipient-only target model; "
            f"missing={missing}, unexpected={unexpected}"
        )
    missing_source_characters = sorted(set(source_recipient_characters) - set(target_recipient_characters))
    if missing_source_characters:
        raise ValueError(
            "recipient_only_expansion cannot discard characters from the init checkpoint recipient map; "
            f"missing={''.join(missing_source_characters)!r}"
        )
    target_indices = {character: index + 1 for index, character in enumerate(target_recipient_characters)}
    classifier_keys = ("recipient_classifier.weight", "recipient_classifier.bias")
    adapted: dict[str, object] = dict(source_state_dict)
    for key in source_keys:
        source_value = source_state_dict[key]
        target_value = target_state_dict[key]
        source_shape = tuple(getattr(source_value, "shape", ()))
        target_shape = tuple(getattr(target_value, "shape", ()))
        if key not in classifier_keys:
            if source_shape != target_shape:
                raise ValueError(
                    "init checkpoint model parameters do not match the recipient-only target model: "
                    f"{key} has source shape {source_shape} but target shape {target_shape}"
                )
            continue
        if not source_shape or not target_shape or source_shape[0] != len(source_recipient_characters) + 1:
            raise ValueError(f"init checkpoint {key} does not match its recipient character map")
        if target_shape[0] != len(target_recipient_characters) + 1 or source_shape[1:] != target_shape[1:]:
            raise ValueError(f"recipient-only target {key} does not match its recipient character map")
        if not hasattr(target_value, "detach") or not hasattr(source_value, "__getitem__"):
            raise ValueError(f"init checkpoint {key} is not a tensor")
        remapped = target_value.detach().clone()
        remapped[0].copy_(source_value[0])
        for source_index, character in enumerate(source_recipient_characters, start=1):
            remapped[target_indices[character]].copy_(source_value[source_index])
        adapted[key] = remapped
    new_target_characters = [
        character for character in target_recipient_characters if character not in set(source_recipient_characters)
    ]
    return adapted, {
        "blank_row_copied": True,
        "shared_character_rows_copied": len(source_recipient_characters),
        "new_target_character_rows_kept_at_seed": len(new_target_characters),
        "checkpoint_character_count": len(source_recipient_characters),
        "target_character_count": len(target_recipient_characters),
        "checkpoint_charset_sha256": _label_map_sha256(source_recipient_characters),
        "target_charset_sha256": _label_map_sha256(target_recipient_characters),
    }


def _recipient_capacity_reinit_state(
    *,
    source_state_dict: Mapping[str, object],
    target_state_dict: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Copy the frozen side of a v12 model and keep its private branch fresh."""
    source_keys = set(source_state_dict)
    target_keys = set(target_state_dict)
    if source_keys != target_keys:
        raise ValueError(
            "recipient_capacity_reinit requires the same parameter names; "
            f"missing={sorted(target_keys - source_keys)}, unexpected={sorted(source_keys - target_keys)}"
        )
    adapted: dict[str, object] = dict(target_state_dict)
    copied: list[str] = []
    reinitialised: list[str] = []
    for key in sorted(target_keys):
        if key.startswith("recipient_"):
            reinitialised.append(str(key))
            continue
        source_value = source_state_dict[key]
        target_value = target_state_dict[key]
        source_shape = tuple(getattr(source_value, "shape", ()))
        target_shape = tuple(getattr(target_value, "shape", ()))
        if source_shape != target_shape:
            raise ValueError(
                "recipient_capacity_reinit changed a frozen tensor shape: "
                f"{key} has source shape {source_shape} but target shape {target_shape}"
            )
        adapted[key] = source_value
        copied.append(str(key))
    if not copied or not reinitialised:
        raise ValueError("recipient_capacity_reinit found no frozen or recipient tensors")
    return adapted, {
        "frozen_tensor_count": len(copied),
        "recipient_tensor_count_reinitialized": len(reinitialised),
        "recipient_parameter_prefix": "recipient_",
    }


def _recipient_visual_context_reinit_state(
    *,
    source_state_dict: Mapping[str, object],
    target_state_dict: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Copy every frozen v13 tensor and keep the new recipient branch fresh."""

    source_frozen = {key for key in source_state_dict if not str(key).startswith("recipient_")}
    target_frozen = {key for key in target_state_dict if not str(key).startswith("recipient_")}
    source_recipient = {key for key in source_state_dict if str(key).startswith("recipient_")}
    target_recipient = {key for key in target_state_dict if str(key).startswith("recipient_")}
    if source_frozen != target_frozen:
        raise ValueError(
            "recipient_visual_context_reinit changed the frozen parameter set; "
            f"missing={sorted(target_frozen - source_frozen)}, "
            f"unexpected={sorted(source_frozen - target_frozen)}"
        )
    if not source_frozen or not source_recipient or not target_recipient:
        raise ValueError("recipient_visual_context_reinit has an incomplete source or target model")
    adapted: dict[str, object] = dict(target_state_dict)
    for key in sorted(source_frozen):
        source_value = source_state_dict[key]
        target_value = target_state_dict[key]
        source_shape = tuple(getattr(source_value, "shape", ()))
        target_shape = tuple(getattr(target_value, "shape", ()))
        if source_shape != target_shape:
            raise ValueError(
                "recipient_visual_context_reinit changed a frozen tensor shape: "
                f"{key} source={source_shape}, target={target_shape}"
            )
        adapted[key] = source_value
    return adapted, {
        "frozen_tensor_count": len(source_frozen),
        "source_recipient_tensor_count_discarded": len(source_recipient),
        "target_recipient_tensor_count_reinitialized": len(target_recipient),
        "recipient_parameter_prefix": "recipient_",
        "source_backbone": "legacy_depthwise_gru_v1",
        "target_backbone": "residual_positional_transformer_v2",
    }


def _recipient_open_text_adapter_state(
    *,
    source_state_dict: Mapping[str, object],
    target_state_dict: Mapping[str, object],
    source_recipient_characters: Sequence[str],
    target_recipient_characters: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Copy the complete seed and retain only new adapter tensors at init.

    The recipient classifier is mapped by Unicode so an additive train charset
    cannot reinterpret an existing row.  Every other legacy tensor must be
    present with an identical shape.  The scalar adapter gate is freshly zero,
    making the target's initial logits exactly equal to the source logits.
    """
    adapted: dict[str, object] = dict(target_state_dict)
    adapter_keys = {key for key in target_state_dict if key.startswith("recipient_open_text_")}
    if not adapter_keys or "recipient_open_text_gate" not in adapter_keys:
        raise ValueError("recipient_open_text_adapter target has no identity-gated adapter parameters")
    unexpected = sorted(set(source_state_dict) - set(target_state_dict))
    if unexpected:
        raise ValueError(f"recipient_open_text_adapter seed has unexpected tensors: {unexpected}")
    missing_legacy = sorted(
        key for key in target_state_dict if key not in adapter_keys and key not in source_state_dict
    )
    if missing_legacy:
        raise ValueError(f"recipient_open_text_adapter seed is missing legacy tensors: {missing_legacy}")
    target_indices = {character: index + 1 for index, character in enumerate(target_recipient_characters)}
    missing_characters = sorted(set(source_recipient_characters) - set(target_recipient_characters))
    if missing_characters:
        raise ValueError(
            "recipient_open_text_adapter cannot discard seed characters: "
            f"{''.join(missing_characters)!r}"
        )
    classifier_keys = {"recipient_classifier.weight", "recipient_classifier.bias"}
    copied = 0
    for key, source_value in source_state_dict.items():
        target_value = target_state_dict[key]
        if key in classifier_keys:
            remapped = target_value.detach().clone()
            remapped[0].copy_(source_value[0])
            for source_index, character in enumerate(source_recipient_characters, start=1):
                remapped[target_indices[character]].copy_(source_value[source_index])
            adapted[key] = remapped
            copied += 1
            continue
        source_shape = tuple(getattr(source_value, "shape", ()))
        target_shape = tuple(getattr(target_value, "shape", ()))
        if source_shape != target_shape:
            raise ValueError(
                "recipient_open_text_adapter changed a legacy tensor shape: "
                f"{key} has source shape {source_shape} but target shape {target_shape}"
            )
        adapted[key] = source_value
        copied += 1
    return adapted, {
        "legacy_tensor_count_copied": copied,
        "new_adapter_tensor_count": len(adapter_keys),
        "adapter_parameter_prefix": "recipient_open_text_",
        "identity_gate_initial_value": 0.0,
        "checkpoint_character_count": len(source_recipient_characters),
        "target_character_count": len(target_recipient_characters),
    }


def _parameter_only_initialization(
    *,
    init_checkpoint: Path | None,
    init_checkpoint_mode: str = INIT_CHECKPOINT_MODE_STRICT,
    config: UnifiedReaderConfig,
    amount_characters: Sequence[str],
    time_characters: Sequence[str],
    payment_characters: Sequence[str],
    recipient_characters: Sequence[str] | None,
    status_text_characters: Sequence[str] | None = None,
    payment_bank_prefix_classes: Sequence[str] | None,
    torch: Any,
    target_state_dict: Mapping[str, object] | None = None,
    allow_v12_status_text_expansion: bool = False,
) -> tuple[Mapping[str, object] | None, dict[str, object]]:
    """Load a compatible checkpoint as parameters only, never as a resume.

    A warm start is deliberately stricter than a shape-only ``state_dict``
    load.  CTC character rows and finite classifier rows are semantic indices;
    accepting an equal-size but reordered map would silently corrupt a field.
    The caller gets a fresh optimiser, epoch counter, sampler state, and
    best-checkpoint history in every case.
    """
    if init_checkpoint_mode not in INIT_CHECKPOINT_MODES:
        raise ValueError(
            "init_checkpoint_mode must be one of "
            f"{', '.join(sorted(INIT_CHECKPOINT_MODES))}"
        )
    if init_checkpoint is None:
        if init_checkpoint_mode != INIT_CHECKPOINT_MODE_STRICT:
            raise ValueError("recipient-only expansion init modes require a compatible --init-checkpoint")
        return None, {
            "mode": "random",
            "optimizer_restored": False,
            "epoch_reset": True,
        }
    checkpoint_path = Path(init_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    if (
        _has_analysis_only_full_crop_continuation_lineage(payload)
        and init_checkpoint_mode != INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
    ):
        raise ValueError(
            "an analysis-only full-crop continuation authority cannot be used by another init mode"
        )
    continuation_authority: Mapping[str, object] | None = None
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART:
        _validate_recipient_full_crop_seed_policy(payload, torch=torch)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
        continuation_authority = _validate_recipient_full_crop_continuation_policy(
            payload, torch=torch
        )
    source_config = _checkpoint_config(payload)
    v12_status_text_expansion = (
        allow_v12_status_text_expansion and _is_v12(source_config) and _is_v13(config)
    )
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION:
        # Keep this guard at the parameter-load boundary as well as the label
        # override boundary.  Direct callers must not be able to bypass the
        # strictly-wider v12-only preflight.
        _validate_recipient_input_width_expansion_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_CAPACITY_REINIT:
        _validate_recipient_capacity_reinit_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER:
        _validate_recipient_open_text_adapter_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT:
        _validate_recipient_visual_context_reinit_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART:
        _validate_recipient_full_crop_warmstart_config(source_config, config)
    elif init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
        _validate_recipient_full_crop_continuation_config(source_config, config)
    elif v12_status_text_expansion:
        source_values = asdict(source_config)
        target_values = asdict(config)
        source_values["architecture_version"] = 13
        if source_values != target_values:
            changed = [
                key
                for key in sorted(target_values)
                if source_values.get(key) != target_values.get(key)
            ]
            raise ValueError(
                "v12 status-text expansion may change only architecture_version; incompatible config fields: "
                + ", ".join(changed)
            )
    elif source_config != config:
        raise ValueError(
            "init checkpoint model config does not match the requested training config; "
            "use the same architecture, input sizes, head widths, and decoder policy"
        )
    (
        source_amount_characters,
        source_time_characters,
        source_payment_characters,
        source_recipient_characters,
        source_status_classes,
        source_payment_bank_prefix_classes,
    ) = _checkpoint_labels(payload, config=source_config)
    label_maps: tuple[tuple[str, Sequence[str] | None, Sequence[str] | None], ...] = (
        ("amount character map", source_amount_characters, amount_characters),
        ("time character map", source_time_characters, time_characters),
        ("payment character map", source_payment_characters, payment_characters),
        ("status class map", source_status_classes, STATUS_CLASSES),
        (
            "payment bank-prefix class map",
            source_payment_bank_prefix_classes,
            payment_bank_prefix_classes,
        ),
    )
    for label, source_values, current_values in label_maps:
        if source_values is None or current_values is None:
            if source_values is not current_values:
                raise ValueError(f"init checkpoint {label} does not match the current training data")
        elif list(source_values) != list(current_values):
            raise ValueError(f"init checkpoint {label} does not match the current training data")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("init checkpoint is missing model parameters")
    initialization: dict[str, object] = {
        "mode": "parameter_only",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_kind": payload.get("kind"),
        "source_epoch": payload.get("epoch"),
        "source_config": asdict(source_config),
        "optimizer_restored": False,
        "epoch_reset": True,
        **(
            {
                "source_recipient_train_split_policy": dict(
                    payload["recipient_train_split_policy"]
                ),
                "source_full_crop_seed_sanitizer_attestation": dict(
                    payload["full_crop_seed_sanitizer_attestation"]
                ),
            }
            if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART
            else {}
        ),
        **(
            {
                "source_full_crop_continuation_authority": dict(continuation_authority),
                "optimizer_restored": False,
                "scheduler_restored": False,
                "sampler_state_restored": False,
                "best_history_restored": False,
                "source_epoch_restored": False,
            }
            if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
            else {}
        ),
    }
    if v12_status_text_expansion:
        if target_state_dict is None:
            raise ValueError("v12 status-text expansion requires a freshly initialized v13 target state")
        if status_text_characters is None:
            raise ValueError("v12 status-text expansion requires a train-only status-text charset")
        if source_recipient_characters is None or recipient_characters is None:
            if source_recipient_characters is not recipient_characters:
                raise ValueError(
                    "v12 status-text expansion recipient character map does not match the current training data"
                )
        elif list(source_recipient_characters) != list(recipient_characters):
            raise ValueError(
                "v12 status-text expansion recipient character map does not match the current training data"
            )
        source_keys = set(state_dict)
        target_keys = set(target_state_dict)
        new_keys = {key for key in target_keys if str(key).startswith("status_text_")}
        if not new_keys or source_keys != target_keys - new_keys:
            raise ValueError(
                "v12 status-text expansion must add only status_text_ parameters; "
                f"missing_legacy={sorted((target_keys - new_keys) - source_keys)}, "
                f"unexpected={sorted(source_keys - (target_keys - new_keys))}"
            )
        adapted_state = dict(target_state_dict)
        for key, source_value in state_dict.items():
            target_value = target_state_dict[key]
            source_shape = tuple(getattr(source_value, "shape", ()))
            target_shape = tuple(getattr(target_value, "shape", ()))
            if source_shape != target_shape:
                raise ValueError(
                    "v12 status-text expansion changed a legacy tensor shape: "
                    f"{key} source={source_shape}, target={target_shape}"
                )
            adapted_state[key] = source_value
        initialization.update(
            {
                "mode": "parameter_only_v12_to_v13_status_text_expansion",
                "copied_legacy_tensor_count": len(source_keys),
                "new_status_text_tensor_count": len(new_keys),
                "new_parameter_prefix": "status_text_",
                "frozen_legacy_output_count": len(V12_ONNX_OUTPUT_NAMES),
                "target_status_text_character_count": len(status_text_characters),
                "target_status_text_charset_sha256": hashlib.sha256(
                    "".join(status_text_characters).encode("utf-8")
                ).hexdigest(),
            }
        )
        return adapted_state, initialization
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_STRICT:
        if source_recipient_characters is None or recipient_characters is None:
            if source_recipient_characters is not recipient_characters:
                raise ValueError("init checkpoint recipient character map does not match the current training data")
        elif list(source_recipient_characters) != list(recipient_characters):
            raise ValueError("init checkpoint recipient character map does not match the current training data")
        source_status_text_characters = _checkpoint_status_text_characters(
            payload, config=source_config
        )
        if source_status_text_characters is None or status_text_characters is None:
            if source_status_text_characters is not status_text_characters:
                raise ValueError("init checkpoint status-text character map does not match the current training data")
        elif list(source_status_text_characters) != list(status_text_characters):
            raise ValueError("init checkpoint status-text character map does not match the current training data")
        return state_dict, initialization

    v13_recipient_private_mode = (
        _is_v13(config)
        and init_checkpoint_mode in V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES
    )
    if not (_is_v12(config) or v13_recipient_private_mode):
        raise ValueError(
            "recipient-only expansion init modes require architecture v12, or an audited v13 private-recipient mode"
        )
    if source_recipient_characters is None or recipient_characters is None:
        raise ValueError("init checkpoint recipient character map does not match the current training data")
    if target_state_dict is None:
        raise ValueError("recipient-only expansion init modes require a freshly initialised target state")
    source_status_text_characters = _checkpoint_status_text_characters(
        payload, config=source_config
    )
    if source_status_text_characters is None or status_text_characters is None:
        if source_status_text_characters is not status_text_characters:
            raise ValueError("init checkpoint status-text character map does not match the current training data")
    elif list(source_status_text_characters) != list(status_text_characters):
        raise ValueError("init checkpoint status-text character map does not match the current training data")
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
        if list(source_recipient_characters) != list(recipient_characters):
            raise ValueError(
                "recipient_full_crop_continuation requires an exact recipient character map match"
            )
        source_keys = set(state_dict)
        target_keys = set(target_state_dict)
        if source_keys != target_keys:
            raise ValueError(
                "recipient_full_crop_continuation requires an exact all-state key match"
            )
        shape_mismatches = [
            name
            for name in sorted(source_keys)
            if (
                str(getattr(state_dict[name], "dtype", None)),
                tuple(getattr(state_dict[name], "shape", ())),
            )
            != (
                str(getattr(target_state_dict[name], "dtype", None)),
                tuple(getattr(target_state_dict[name], "shape", ())),
            )
        ]
        if shape_mismatches:
            raise ValueError(
                "recipient_full_crop_continuation changed state tensor dtype/shape: "
                + ", ".join(shape_mismatches[:5])
            )
        initialization.update(
            {
                "mode": "parameter_only_recipient_full_crop_continuation_all_state_copy",
                "init_checkpoint_mode": init_checkpoint_mode,
                "all_state_tensor_count_copied": len(source_keys),
                "all_state_key_set_exact": True,
                "all_state_dtype_shape_exact": True,
                "all_state_value_copy": "source_tensor_objects_loaded_strictly_before_cuda",
            }
        )
        return state_dict, initialization
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT:
        remapped_state, visual_context_mapping = _recipient_visual_context_reinit_state(
            source_state_dict=state_dict,
            target_state_dict=target_state_dict,
        )
        initialization.update(
            {
                "mode": "parameter_only_recipient_visual_context_reinit",
                "init_checkpoint_mode": init_checkpoint_mode,
                "recipient_visual_context_mapping": visual_context_mapping,
            }
        )
        return remapped_state, initialization
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_CAPACITY_REINIT:
        remapped_state, capacity_mapping = _recipient_capacity_reinit_state(
            source_state_dict=state_dict,
            target_state_dict=target_state_dict,
        )
        initialization.update(
            {
                "mode": "parameter_only_recipient_capacity_reinit",
                "init_checkpoint_mode": init_checkpoint_mode,
                "recipient_capacity_mapping": capacity_mapping,
                "source_recipient_branch_channels": _recipient_branch_channels(source_config),
                "target_recipient_branch_channels": _recipient_branch_channels(config),
                "source_recipient_hidden_size": _recipient_hidden_size(source_config),
                "target_recipient_hidden_size": _recipient_hidden_size(config),
            }
        )
        return remapped_state, initialization
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER:
        remapped_state, adapter_mapping = _recipient_open_text_adapter_state(
            source_state_dict=state_dict,
            target_state_dict=target_state_dict,
            source_recipient_characters=source_recipient_characters,
            target_recipient_characters=recipient_characters,
        )
        initialization.update(
            {
                "mode": "parameter_only_recipient_open_text_adapter",
                "init_checkpoint_mode": init_checkpoint_mode,
                "recipient_open_text_adapter_mapping": adapter_mapping,
                "source_recipient_input_width": source_config.recipient_input_width,
                "target_recipient_input_width": config.recipient_input_width,
            }
        )
        return remapped_state, initialization
    remapped_state, row_mapping = _recipient_classifier_unicode_expansion_state(
        source_state_dict=state_dict,
        target_state_dict=target_state_dict,
        source_recipient_characters=source_recipient_characters,
        target_recipient_characters=recipient_characters,
    )
    initialization.update(
        {
            "mode": (
                "parameter_only_recipient_input_width_expansion"
                if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION
                else (
                    "parameter_only_recipient_full_crop_warmstart"
                    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART
                    else "parameter_only_recipient_unicode_expansion"
                )
            ),
            "init_checkpoint_mode": init_checkpoint_mode,
            "recipient_classifier_row_mapping": row_mapping,
            **(
                {
                    "source_recipient_input_width": source_config.recipient_input_width,
                    "target_recipient_input_width": config.recipient_input_width,
                }
                if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION
                else {}
            ),
            **(
                {
                    "source_recipient_value_left_trim": source_config.recipient_value_left_trim,
                    "target_recipient_value_left_trim": config.recipient_value_left_trim,
                }
                if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART
                else {}
            ),
        }
    )
    return remapped_state, initialization


def _checkpoint_protection_report(
    validation: Mapping[str, object],
    *,
    policy: Mapping[str, object],
    failures: Sequence[str],
) -> dict[str, object]:
    """Capture the exact per-field evidence used by checkpoint protection."""
    by_field = validation.get("candidate_text_by_field")
    if not isinstance(by_field, Mapping):
        raise ValueError("validation candidate-text metrics are missing")
    observed: dict[str, dict[str, int | float]] = {}
    for field in CHECKPOINT_SELECTION_PROTECTED_FIELDS:
        metric = by_field.get(field)
        if not isinstance(metric, Mapping):
            raise ValueError(f"validation candidate-text metrics are missing for {field}")
        exact_match = _validation_candidate_exact(validation, field)
        records = metric.get("records")
        exact_matches = metric.get("exact_matches")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or records <= 0
            or isinstance(exact_matches, bool)
            or not isinstance(exact_matches, int)
            or not 0 <= exact_matches <= records
        ):
            raise ValueError(f"validation candidate-text counts are invalid for {field}")
        observed[field] = {
            "exact_matches": exact_matches,
            "records": records,
            "exact_match": exact_match,
        }
    raw_minima = policy.get("protected_minimum_candidate_exact")
    if not isinstance(raw_minima, Mapping):
        raise ValueError("checkpoint selection policy has invalid protected candidate floors")
    minima: dict[str, float] = {}
    for field in CHECKPOINT_SELECTION_PROTECTED_FIELDS:
        if field not in raw_minima:
            continue
        try:
            minimum = float(raw_minima[field])
        except (TypeError, ValueError):
            raise ValueError(f"checkpoint selection policy has invalid floor for {field}") from None
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError(f"checkpoint selection policy has invalid floor for {field}")
        minima[field] = minimum
    return {
        "candidate_exact": observed,
        "minimum_candidate_exact": minima,
        "margin": {
            field: observed[field]["exact_match"] - minimum
            for field, minimum in minima.items()
        },
        "failures": list(failures),
    }


def _format_checkpoint_protection_report(report: Mapping[str, object]) -> str:
    """Render a compact, screenshot-friendly guardrail line for training logs."""
    observed = report.get("candidate_exact")
    minima = report.get("minimum_candidate_exact")
    margins = report.get("margin")
    failures = report.get("failures")
    if not isinstance(observed, Mapping) or not isinstance(minima, Mapping) or not isinstance(margins, Mapping):
        raise ValueError("checkpoint protection report is invalid")
    labels = {"amount": "amount", "time": "time", "payment_method_field": "payment"}
    fields: list[str] = []
    for field in CHECKPOINT_SELECTION_PROTECTED_FIELDS:
        metric = observed.get(field)
        if not isinstance(metric, Mapping):
            raise ValueError(f"checkpoint protection report is missing {field}")
        try:
            exact_matches = int(metric["exact_matches"])
            records = int(metric["records"])
            exact_match = float(metric["exact_match"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"checkpoint protection report is invalid for {field}") from None
        text = f"{labels[field]}={exact_matches}/{records}={exact_match:.2%}"
        if field in minima:
            try:
                minimum = float(minima[field])
                margin = float(margins[field])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"checkpoint protection report has no valid floor for {field}") from None
            text += f"/floor={minimum:.2%} ({margin * 100:+.2f}pp)"
        fields.append(text)
    failure_text = "; ".join(str(value) for value in failures) if isinstance(failures, Sequence) and not isinstance(failures, str) and failures else "-"
    return "guards=" + ", ".join(fields) + f"; failures={failure_text}"


def _validate_validation_every(
    validation_every: int,
    *,
    config: UnifiedReaderConfig,
    recipient_only_fine_tune: bool,
    status_text_only_fine_tune: bool = False,
    init_checkpoint_mode: str,
) -> None:
    """Validate the deliberately narrow sparse-full-validation escape hatch.

    A full five-field validation is the evidence for the financial guardrails,
    so ordinary training continues to validate every epoch.  The only safe
    exceptions are the v12 recipient-private warm-start route and the v13
    status-text-only route.  In each case every tensor outside the private
    trainable head is frozen and separately byte-checked before every planned
    full validation.  Keeping the restriction here, before dataset/output
    work begins, prevents a typo from silently weakening another recipe.
    """
    if isinstance(validation_every, bool) or not isinstance(validation_every, int) or validation_every <= 0:
        raise ValueError("validation_every must be a positive integer")
    if validation_every == 1:
        return
    recipient_private_safe = (
        _uses_v12_recipient_topology(config)
        and recipient_only_fine_tune
        and init_checkpoint_mode in RECIPIENT_ONLY_INIT_CHECKPOINT_MODES
        and (
            _is_v12(config)
            or init_checkpoint_mode in V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES
        )
    )
    status_text_private_safe = _is_v13(config) and status_text_only_fine_tune
    if not (recipient_private_safe or status_text_private_safe):
        raise ValueError(
            "validation_every > 1 is supported only by guarded v12/v13 recipient_only_fine_tune "
            "or v13 status_text_only_fine_tune"
        )


def _is_full_validation_epoch(*, epoch: int, epochs: int, validation_every: int) -> bool:
    """Return whether this epoch must run the complete five-field validator.

    Epoch one establishes early evidence, every Nth epoch provides ongoing
    guardrail checks, and the final epoch is never allowed to remain
    unvalidated.  The input checks make this helper safe to test independently
    from Torch or a dataset.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if epoch > epochs:
        raise ValueError("epoch cannot exceed epochs")
    if isinstance(validation_every, bool) or not isinstance(validation_every, int) or validation_every <= 0:
        raise ValueError("validation_every must be a positive integer")
    return epoch == 1 or epoch == epochs or epoch % validation_every == 0


def _full_validation_epoch_reason(*, epoch: int, epochs: int, validation_every: int) -> str:
    """Return an audit-friendly reason for a planned full validation."""
    if not _is_full_validation_epoch(epoch=epoch, epochs=epochs, validation_every=validation_every):
        return "scheduled_skip"
    reasons: list[str] = []
    if epoch == 1:
        reasons.append("epoch_1")
    if epoch % validation_every == 0:
        reasons.append("every_n")
    if epoch == epochs:
        reasons.append("final_epoch")
    return "+".join(reasons)


def _non_recipient_parameter_bytes(model: Any) -> dict[str, bytes]:
    """Capture exact CPU bytes for the frozen side of a v12 recipient run.

    ``torch.equal`` compares values rather than necessarily their raw bit
    representation.  A copied contiguous CPU byte payload proves that every
    non-recipient tensor in the model state has stayed identical, including
    buffers such as future BatchNorm running statistics and details such as
    signed zero.  This snapshot is intentionally only used by the narrowly
    guarded sparse-validation recipient-only expansion route.
    """
    snapshots: dict[str, bytes] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("recipient_"):
            continue
        if not hasattr(tensor, "detach"):
            raise AssertionError(f"model state entry {name!r} is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        snapshots[name] = tensor.numpy().tobytes()
    if not snapshots:
        raise AssertionError("recipient-only v12 model has no non-recipient parameters to protect")
    return snapshots


def _state_dict_exact_bytes(state: Mapping[str, object]) -> dict[str, bytes]:
    """Snapshot every state tensor for an exact continuation-copy proof."""

    snapshots: dict[str, bytes] = {}
    for name, raw_tensor in state.items():
        if not isinstance(name, str) or not hasattr(raw_tensor, "detach"):
            raise AssertionError(f"model state entry {name!r} is not a tensor")
        tensor = raw_tensor.detach().cpu().contiguous()
        try:
            # ``view(uint8)`` also handles dtypes whose direct NumPy conversion
            # is unavailable, and preserves signed-zero/NaN payload bits.
            import torch

            snapshots[name] = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        except (ImportError, RuntimeError, TypeError, ValueError) as error:
            raise AssertionError(f"unable to snapshot model state entry {name!r}") from error
    if not snapshots:
        raise AssertionError("model has no state tensors")
    return snapshots


def _assert_state_dict_exact_copy(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    observed_bytes = _state_dict_exact_bytes(observed)
    expected_bytes = _state_dict_exact_bytes(expected)
    if set(observed_bytes) != set(expected_bytes):
        raise AssertionError("continuation all-state key set changed during strict load")
    changed = [
        name for name in sorted(observed_bytes) if observed_bytes[name] != expected_bytes[name]
    ]
    if changed:
        raise AssertionError(
            "continuation all-state copy is not byte-identical: " + ", ".join(changed[:5])
        )


def _assert_non_recipient_parameter_bytes(model: Any, expected: Mapping[str, bytes]) -> None:
    """Fail closed if a frozen financial/shared parameter has changed."""
    observed = _non_recipient_parameter_bytes(model)
    if set(observed) != set(expected):
        raise AssertionError("recipient-only frozen parameter set changed before full validation")
    changed = [name for name, value in observed.items() if value != expected[name]]
    if changed:
        preview = ", ".join(sorted(changed)[:5])
        suffix = "..." if len(changed) > 5 else ""
        raise AssertionError(
            "recipient-only fine-tune mutated frozen non-recipient parameters before full validation: "
            f"{preview}{suffix}"
        )


def _non_status_text_parameter_bytes(model: Any) -> dict[str, bytes]:
    """Capture the complete frozen v12-compatible side of a v13 model."""
    snapshots: dict[str, bytes] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("status_text_"):
            continue
        if not hasattr(tensor, "detach"):
            raise AssertionError(f"model state entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        snapshots[name] = value.numpy().tobytes()
    if not snapshots:
        raise AssertionError("v13 status-text model has no legacy parameters to protect")
    return snapshots


def _assert_non_status_text_parameter_bytes(model: Any, expected: Mapping[str, bytes]) -> None:
    observed = _non_status_text_parameter_bytes(model)
    if set(observed) != set(expected):
        raise AssertionError("v13 frozen legacy parameter set changed during status-text fine-tuning")
    changed = [name for name, value in observed.items() if value != expected[name]]
    if changed:
        preview = ", ".join(sorted(changed)[:5])
        suffix = "..." if len(changed) > 5 else ""
        raise AssertionError(
            "status-text-only fine-tune mutated frozen v12 parameters: " + preview + suffix
        )


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
    recipient_loss_weight: float = 1.0,
    status_text_loss_weight: float = 1.0,
    recipient_sampling_weight: float = 1.0,
    recipient_rare_character_max_support: int = 0,
    recipient_rare_character_sampling_weight: float = 1.0,
    recipient_long_text_min_length: int = 0,
    recipient_long_text_sampling_weight: float = 1.0,
    recipient_low_confidence_threshold: float | None = None,
    recipient_low_confidence_loss_weight: float = 1.0,
    recipient_confidence_curriculum_epochs: int = 0,
    recipient_tail_rare_character_max_support: int = 0,
    recipient_tail_rare_character_loss_weight: float = 1.0,
    recipient_tail_long_text_min_length: int = 0,
    recipient_tail_long_text_loss_weight: float = 1.0,
    recipient_train_augmentation: str = "none",
    recipient_train_splits: Sequence[str] = ("train",),
    recipient_only_fine_tune: bool = False,
    status_text_only_fine_tune: bool = False,
    recipient_open_text_unfreeze_legacy: bool = False,
    validation_every: int = 1,
    checkpoint_selection: str = CHECKPOINT_SELECTION_BALANCED,
    checkpoint_min_amount_candidate_exact: float | None = None,
    checkpoint_min_time_candidate_exact: float | None = None,
    checkpoint_min_payment_candidate_exact: float | None = None,
    init_checkpoint: Path | None = None,
    init_checkpoint_mode: str = INIT_CHECKPOINT_MODE_STRICT,
    ctc_loss_weight: float = 0.35,
    structured_loss_weight: float = 1.0,
    payment_bank_prefix_min_support: int = 3,
    seed: int = 42,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    train_progress_every: int = 0,
    cuda_tf32: bool = False,
    cudnn_benchmark: bool = False,
) -> Path:
    """Train one shared-trunk reader and return the best validation checkpoint.

    The function intentionally accepts incomplete receipt rows: an absent slot
    gets a white input image but contributes no loss.  Amount, time, and
    payment must be represented in train/validation.  A status head is trained
    only when all three status classes are represented in both splits;
    otherwise its final delivery policy is review-only.
    """
    config.validate()
    if recipient_only_fine_tune and status_text_only_fine_tune:
        raise ValueError("recipient_only_fine_tune and status_text_only_fine_tune are mutually exclusive")
    if status_text_only_fine_tune:
        if not _is_v13(config):
            raise ValueError("status_text_only_fine_tune is supported only by architecture v13")
        if init_checkpoint is None:
            raise ValueError("status_text_only_fine_tune requires a compatible v12 or v13 --init-checkpoint")
    recipient_train_split_policy = _recipient_train_split_policy(recipient_train_splits)
    if (
        init_checkpoint_mode in {
            INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
            INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
        }
        and recipient_train_split_policy["mode"] != "standard_train_only"
    ):
        raise ValueError("full-crop recipient init modes permit train-split supervision only")
    if recipient_only_fine_tune:
        if not _uses_v12_recipient_topology(config):
            raise ValueError("recipient_only_fine_tune is supported only by architecture v12 or v13")
        if init_checkpoint is None:
            raise ValueError("recipient_only_fine_tune requires a compatible --init-checkpoint")
    elif recipient_train_split_policy["mode"] != "standard_train_only":
        raise ValueError("recipient_train_splits beyond train are supported only by recipient_only_fine_tune")
    if recipient_open_text_unfreeze_legacy and (
        not recipient_only_fine_tune
        or init_checkpoint_mode != INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER
    ):
        raise ValueError(
            "recipient_open_text_unfreeze_legacy requires recipient-only fine-tuning "
            "with recipient_open_text_adapter initialisation"
        )
    if init_checkpoint_mode not in INIT_CHECKPOINT_MODES:
        raise ValueError(
            "init_checkpoint_mode must be one of "
            f"{', '.join(sorted(INIT_CHECKPOINT_MODES))}"
        )
    if (
        recipient_only_fine_tune
        and _is_v13(config)
        and init_checkpoint_mode not in V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES
    ):
        raise ValueError(
            "v13 recipient_only_fine_tune requires an audited private-recipient init mode: "
            f"{', '.join(sorted(V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES))}"
        )
    if init_checkpoint_mode in RECIPIENT_ONLY_INIT_CHECKPOINT_MODES:
        v13_recipient_private_mode = (
            _is_v13(config)
            and init_checkpoint_mode in V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES
        )
        if not recipient_only_fine_tune or not (_is_v12(config) or v13_recipient_private_mode):
            raise ValueError(
                "recipient-only expansion init modes require v12 recipient_only_fine_tune, "
                "or a compatible v13 private-recipient warm start"
            )
        if init_checkpoint is None:
            raise ValueError("recipient-only expansion init modes require a compatible --init-checkpoint")
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
        fixed_recipe_matches = (
            recipient_only_fine_tune
            and not status_text_only_fine_tune
            and device == "cuda:0"
            and epochs == 8
            and batch_size == 10
            and math.isclose(learning_rate, 0.0001, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(weight_decay, 0.0001, rel_tol=0.0, abs_tol=1e-12)
            and validation_every == 1
            and checkpoint_selection == CHECKPOINT_SELECTION_RECIPIENT_PRIORITY
            and checkpoint_min_amount_candidate_exact is not None
            and math.isclose(
                checkpoint_min_amount_candidate_exact,
                0.7885,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and checkpoint_min_time_candidate_exact is not None
            and math.isclose(
                checkpoint_min_time_candidate_exact,
                0.9840,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and checkpoint_min_payment_candidate_exact is not None
            and math.isclose(
                checkpoint_min_payment_candidate_exact,
                0.9325,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(ctc_loss_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(structured_loss_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(payment_loss_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(recipient_loss_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(status_text_loss_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(recipient_sampling_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and recipient_rare_character_max_support == 0
            and math.isclose(
                recipient_rare_character_sampling_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
            )
            and recipient_long_text_min_length == 0
            and math.isclose(
                recipient_long_text_sampling_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
            )
            and recipient_low_confidence_threshold is not None
            and math.isclose(
                recipient_low_confidence_threshold, 0.95, rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(
                recipient_low_confidence_loss_weight, 0.50, rel_tol=0.0, abs_tol=1e-12
            )
            and recipient_confidence_curriculum_epochs == 10
            and recipient_tail_rare_character_max_support == 3
            and math.isclose(
                recipient_tail_rare_character_loss_weight, 1.5, rel_tol=0.0, abs_tol=1e-12
            )
            and recipient_tail_long_text_min_length == 9
            and math.isclose(
                recipient_tail_long_text_loss_weight, 1.5, rel_tol=0.0, abs_tol=1e-12
            )
            and recipient_train_augmentation == "robust_v2"
            and seed == 42
            and payment_bank_prefix_min_support == 3
            and cuda_tf32 is True
            and cudnn_benchmark is True
            and persistent_workers is (num_workers > 0)
        )
        if not fixed_recipe_matches:
            raise ValueError(
                "recipient_full_crop_continuation is hard-locked to the audited fixed B8 recipe"
            )
        # Fail before manifest/dataset/output access.  A matching v13 shape is
        # not authority: this private mode exists only for the embedded,
        # content-bound fixed pilot source.
        assert init_checkpoint is not None
        continuation_torch, _ = _require_torch()
        try:
            continuation_cuda_available = bool(continuation_torch.cuda.is_available())
            continuation_cuda_name = (
                str(continuation_torch.cuda.get_device_name(0))
                if continuation_cuda_available
                else ""
            )
        except (AttributeError, RuntimeError) as error:
            raise ValueError(
                "recipient_full_crop_continuation requires CUDA device 0 on an RTX 4090"
            ) from error
        if not continuation_cuda_available or "4090" not in continuation_cuda_name:
            raise ValueError(
                "recipient_full_crop_continuation requires CUDA device 0 on an RTX 4090"
            )
        continuation_payload = _load_checkpoint(Path(init_checkpoint).resolve(), torch=continuation_torch)
        _validate_recipient_full_crop_continuation_policy(
            continuation_payload, torch=continuation_torch
        )
        _validate_recipient_full_crop_continuation_config(
            _checkpoint_config(continuation_payload), config
        )
    _validate_validation_every(
        validation_every,
        config=config,
        recipient_only_fine_tune=recipient_only_fine_tune,
        status_text_only_fine_tune=status_text_only_fine_tune,
        init_checkpoint_mode=init_checkpoint_mode,
    )
    recipient_sampling_weights, recipient_sampling_policy = _recipient_training_sample_weights(
        (),
        recipient_sampling_weight=recipient_sampling_weight,
        recipient_rare_character_max_support=recipient_rare_character_max_support,
        recipient_rare_character_sampling_weight=recipient_rare_character_sampling_weight,
        recipient_long_text_min_length=recipient_long_text_min_length,
        recipient_long_text_sampling_weight=recipient_long_text_sampling_weight,
    )
    del recipient_sampling_weights  # Values are rebuilt from the actual train split below.
    recipient_confidence_policy = _recipient_confidence_policy(
        low_confidence_threshold=recipient_low_confidence_threshold,
        low_confidence_loss_weight=recipient_low_confidence_loss_weight,
        curriculum_epochs=recipient_confidence_curriculum_epochs,
    )
    recipient_tail_loss_static_config = _recipient_tail_loss_config(
        rare_character_max_support=recipient_tail_rare_character_max_support,
        rare_character_loss_weight=recipient_tail_rare_character_loss_weight,
        long_text_min_length=recipient_tail_long_text_min_length,
        long_text_loss_weight=recipient_tail_long_text_loss_weight,
    )
    recipient_train_augmentation_policy = _recipient_train_augmentation_policy(
        mode=recipient_train_augmentation,
        seed=seed,
    )
    # Do not infer whether these options were requested from an empty record
    # list: ``all([])`` is true, so a pre-data policy probe would otherwise
    # incorrectly look uniform even when a legacy caller explicitly supplied
    # ``--recipient-sampling-weight 2``.  The helper above has already
    # validated every numeric value; this branch merely gives unsupported
    # architectures an immediate, actionable error before any dataset work.
    recipient_training_options_requested = (
        not math.isclose(float(recipient_sampling_weight), 1.0, rel_tol=0.0, abs_tol=1e-12)
        or (
            recipient_rare_character_max_support > 0
            and not math.isclose(
                float(recipient_rare_character_sampling_weight), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        )
        or (
            recipient_long_text_min_length > 0
            and not math.isclose(
                float(recipient_long_text_sampling_weight), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        )
        or recipient_confidence_policy["mode"] != "none"
        or recipient_tail_loss_static_config["mode"] != "none"
        or recipient_train_augmentation_policy["mode"] != "none"
    )
    if not (_is_v11(config) or _uses_v12_recipient_topology(config)) and recipient_training_options_requested:
        raise ValueError(
            "recipient sampling/confidence/tail-loss curriculum is supported only by architecture v11 or v12"
        )
    if not _uses_v12_recipient_topology(config) and recipient_train_augmentation_policy["mode"] != "none":
        raise ValueError("recipient_train_augmentation is supported only by architecture v12")
    checkpoint_selection_policy = _checkpoint_selection_policy(
        config=config,
        checkpoint_selection=checkpoint_selection,
        checkpoint_min_amount_candidate_exact=checkpoint_min_amount_candidate_exact,
        checkpoint_min_time_candidate_exact=checkpoint_min_time_candidate_exact,
        checkpoint_min_payment_candidate_exact=checkpoint_min_payment_candidate_exact,
        status_text_only_fine_tune=status_text_only_fine_tune,
    )
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if (
        learning_rate <= 0
        or weight_decay < 0
        or payment_loss_weight <= 0
        or recipient_loss_weight <= 0
        or status_text_loss_weight <= 0
        or recipient_sampling_weight <= 0
        or ctc_loss_weight <= 0
        or structured_loss_weight <= 0
    ):
        raise ValueError(
            "learning_rate, payment_loss_weight, recipient_loss_weight, status_text_loss_weight, recipient_sampling_weight, ctc_loss_weight, and structured_loss_weight must be positive; "
            "weight_decay cannot be negative"
        )
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if persistent_workers and num_workers <= 0:
        raise ValueError("persistent_workers requires num_workers to be positive")
    if train_progress_every < 0:
        raise ValueError("train_progress_every cannot be negative")
    if payment_bank_prefix_min_support <= 0:
        raise ValueError("payment_bank_prefix_min_support must be positive")
    if not math.isfinite(recipient_sampling_weight):
        raise ValueError("recipient_sampling_weight must be finite and positive")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"training output already contains files: {output_dir}. Choose a new empty directory.")
    if init_checkpoint_mode in {
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    }:
        _require_manifest_without_test_rows(records_path)
    records = load_records(records_path, dataset_root=dataset_root, config=config)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "val"]
    if not train_records or not validation_records:
        raise ValueError("The unified manifest must contain non-empty train and val receipt splits")
    field_counts = _field_split_counts(records, config=config)
    structured_counts = _structured_split_counts(records)
    status_counts = _status_split_counts(records)
    status_policy = _status_head_policy(status_counts)
    required_fields = ["amount", "time", "payment_method_field"]
    if _uses_recipient_protocol(config):
        required_fields.append("recipient_field")
    if bool(status_policy["training_enabled"]) or _is_v13(config):
        required_fields.append("transfer_status")
    _require_train_and_validation_coverage(field_counts, required_fields=required_fields)
    if config.architecture_version == 5:
        _require_v5_structured_coverage(structured_counts)
    elif _uses_v8_protocol(config):
        _require_v8_structured_coverage(structured_counts)
    elif _uses_v6_protocol(config):
        _require_v6_structured_coverage(structured_counts)
    data_derived_payment_characters = _payment_charset(train_records)
    amount_characters = list(_amount_characters(config))
    amount_to_id = {character: index for index, character in enumerate(amount_characters, start=1)}
    time_characters = list(_time_characters(config))
    time_to_id = {character: index for index, character in enumerate(time_characters, start=1)}
    recipient_train_split_set = set(recipient_train_split_policy["splits"])
    recipient_charset_records = (
        [record for record in records if str(record["split"]) in recipient_train_split_set]
        if recipient_only_fine_tune
        else train_records
    )
    if _uses_recipient_protocol(config):
        recipient_characters: list[str] | None = _recipient_charset(recipient_charset_records)
    else:
        recipient_characters = None
    if _is_v13(config):
        status_text_characters: list[str] | None = _status_text_charset(train_records)
        if not any(_slot_text(record, "transfer_status") is not None for record in validation_records):
            raise ValueError("v13 requires visible transfer-status text in both train and val splits")
    else:
        status_text_characters = None
    if _uses_modern_protocol(config):
        data_derived_payment_bank_prefix_classes, data_derived_payment_bank_prefix_counts = _payment_bank_prefix_classes(
            train_records,
            min_support=payment_bank_prefix_min_support,
        )
    else:
        data_derived_payment_bank_prefix_classes = None
        data_derived_payment_bank_prefix_counts: dict[str, int] = {}

    torch, _ = _require_torch()
    payment_characters = list(data_derived_payment_characters)
    payment_bank_prefix_classes = (
        list(data_derived_payment_bank_prefix_classes)
        if data_derived_payment_bank_prefix_classes is not None
        else None
    )
    payment_bank_prefix_counts = dict(data_derived_payment_bank_prefix_counts)
    financial_label_policy: dict[str, object] | None = None
    if status_text_only_fine_tune:
        assert init_checkpoint is not None
        assert payment_bank_prefix_classes is not None
        (
            payment_characters,
            payment_bank_prefix_classes,
            recipient_characters,
            financial_label_policy,
        ) = _status_text_only_legacy_label_override(
            init_checkpoint=init_checkpoint,
            config=config,
            amount_characters=amount_characters,
            time_characters=time_characters,
            payment_characters=data_derived_payment_characters,
            recipient_characters=recipient_characters,
            payment_bank_prefix_classes=data_derived_payment_bank_prefix_classes,
            torch=torch,
        )
        payment_bank_prefix_counts = _payment_bank_prefix_retained_counts(
            train_records,
            classes=payment_bank_prefix_classes,
        )
    elif init_checkpoint_mode in RECIPIENT_ONLY_INIT_CHECKPOINT_MODES:
        assert init_checkpoint is not None
        assert payment_bank_prefix_classes is not None
        (
            payment_characters,
            payment_bank_prefix_classes,
            recipient_characters,
            financial_label_policy,
        ) = _recipient_only_expansion_label_override(
            init_checkpoint=init_checkpoint,
            config=config,
            amount_characters=amount_characters,
            time_characters=time_characters,
            payment_characters=data_derived_payment_characters,
            recipient_characters=recipient_characters,
            payment_bank_prefix_classes=data_derived_payment_bank_prefix_classes,
            torch=torch,
            init_checkpoint_mode=init_checkpoint_mode,
        )
        payment_bank_prefix_counts = _payment_bank_prefix_retained_counts(
            train_records,
            classes=payment_bank_prefix_classes,
        )
    recipient_to_id: dict[str, int] | None = (
        {character: index for index, character in enumerate(recipient_characters, start=1)}
        if recipient_characters is not None
        else None
    )
    status_text_to_id: dict[str, int] | None = (
        {character: index for index, character in enumerate(status_text_characters, start=1)}
        if status_text_characters is not None
        else None
    )
    payment_to_id = {character: index for index, character in enumerate(payment_characters, start=1)}
    _validate_ctc_capacity(
        records,
        config=config,
        recipient_characters=recipient_characters,
        status_text_characters=status_text_characters,
        allow_frozen_recipient_train_oov=status_text_only_fine_tune,
    )
    status_to_id = {name: index for index, name in enumerate(STATUS_CLASSES)}
    payment_oov = _payment_oov_by_split(records, payment_characters=set(payment_characters))
    recipient_oov = (
        _recipient_oov_by_split(records, characters=recipient_characters)
        if recipient_characters is not None
        else None
    )
    if status_text_only_fine_tune:
        if not isinstance(financial_label_policy, Mapping) or recipient_oov is None:
            raise AssertionError("status-text-only legacy label policy was not initialized")
        seed_recipient_oov = _validated_recipient_oov_audit(
            financial_label_policy.get("seed_recipient_oov_by_split"),
            source="status-text legacy label policy",
        )
        financial_label_policy = {
            **financial_label_policy,
            "current_data_payment_oov_by_split": payment_oov,
            "current_data_recipient_oov_by_split": recipient_oov,
        }
        # The deployed recipient head and character rows are byte-for-byte
        # frozen from the seed. Preserve its already validated artifact audit
        # at the top level so Python/.NET bundle invariants stay unchanged;
        # current rebuilt-manifest differences remain explicit above.
        recipient_oov = seed_recipient_oov
    status_text_oov = (
        _status_text_oov_by_split(records, characters=status_text_characters)
        if status_text_characters is not None
        else None
    )
    payment_bank_prefix_oov = (
        _payment_bank_prefix_oov_by_split(records, classes=payment_bank_prefix_classes)
        if payment_bank_prefix_classes is not None
        else {}
    )
    training_records = train_records
    if recipient_only_fine_tune:
        training_records = [
            record
            for record in records
            if str(record["split"]) in recipient_train_split_set
            and _slot_text(record, "recipient_field") is not None
        ]
        if not training_records:
            raise ValueError("recipient_only_fine_tune requires at least one train receipt with recipient_field")
    elif status_text_only_fine_tune:
        training_records = [
            record for record in train_records if _slot_text(record, "transfer_status") is not None
        ]
        if not training_records:
            raise ValueError("status_text_only_fine_tune requires visible status text in the train split")
    recipient_tail_loss_character_counts = _recipient_tail_loss_character_counts(training_records)
    recipient_tail_loss_policy = _recipient_tail_loss_policy(
        rare_character_max_support=recipient_tail_rare_character_max_support,
        rare_character_loss_weight=recipient_tail_rare_character_loss_weight,
        long_text_min_length=recipient_tail_long_text_min_length,
        long_text_loss_weight=recipient_tail_long_text_loss_weight,
        records=training_records,
        character_counts=recipient_tail_loss_character_counts,
    )
    target_device = _resolve_device(torch, device)
    uses_cuda = target_device.startswith("cuda")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if uses_cuda:
        torch.cuda.manual_seed_all(seed)
        if cuda_tf32:
            # Ada GPUs such as the RTX 4090 can accelerate static convolution
            # and matrix kernels with TF32.  It is opt-in because it changes
            # internal multiplication precision, while guard evaluation and
            # export remain full FP32.
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
            if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
                cuda_backend.matmul.allow_tf32 = True
            cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
            if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
                cudnn_backend.allow_tf32 = True
        if cudnn_benchmark:
            cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
            if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
                cudnn_backend.benchmark = True
    training_runtime: dict[str, object] = {
        "device": target_device,
        "uses_cuda": uses_cuda,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "persistent_workers": persistent_workers,
        "train_progress_every": train_progress_every,
        "cuda_tf32_requested": cuda_tf32,
        "cudnn_benchmark_requested": cudnn_benchmark,
        "recipient_only_private_branch_training": recipient_only_fine_tune,
        "status_text_only_training": status_text_only_fine_tune,
        "recipient_train_split_policy": recipient_train_split_policy,
        "full_validation_schedule": "epoch_1_every_n_and_final_epoch",
        "validation_every": validation_every,
    }
    if uses_cuda:
        try:
            training_runtime["cuda_device_name"] = str(torch.cuda.get_device_name())
        except (AttributeError, RuntimeError):
            training_runtime["cuda_device_name"] = "unavailable"

    if payment_bank_prefix_classes is not None and not recipient_only_fine_tune:
        payment_bank_train_weights, payment_bank_train_counts = _payment_bank_prefix_class_weights(
            train_records,
            classes=payment_bank_prefix_classes,
            torch=torch,
            device=target_device,
        )
    else:
        payment_bank_train_weights = None
        payment_bank_train_counts = (
            _payment_bank_prefix_retained_counts(train_records, classes=payment_bank_prefix_classes)
            if payment_bank_prefix_classes is not None
            else {}
        )

    train_dataset = _make_dataset(
        training_records,
        config=config,
        torch=torch,
        recipient_train_augmentation_policy=(
            recipient_train_augmentation_policy if _uses_v12_recipient_topology(config) else None
        ),
        recipient_only=recipient_only_fine_tune,
    )
    validation_dataset = _make_dataset(validation_records, config=config, torch=torch)
    sample_weights, recipient_sampling_policy = _recipient_training_sample_weights(
        training_records,
        recipient_sampling_weight=recipient_sampling_weight,
        recipient_rare_character_max_support=recipient_rare_character_max_support,
        recipient_rare_character_sampling_weight=recipient_rare_character_sampling_weight,
        recipient_long_text_min_length=recipient_long_text_min_length,
        recipient_long_text_sampling_weight=recipient_long_text_sampling_weight,
    )
    train_sampler: Any | None = None
    if recipient_sampling_policy["mode"] != "uniform":
        if not (_is_v11(config) or _uses_v12_recipient_topology(config)):
            raise ValueError("recipient sampler is supported only by architecture v11 or v12")
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights_tensor,
            num_samples=len(training_records),
            replacement=True,
            generator=generator,
        )
        recipient_sampling_policy = {
            **recipient_sampling_policy,
            "replacement": True,
            "seed": int(seed),
        }
    if recipient_only_fine_tune and recipient_sampling_policy["mode"] != "uniform":
        raise ValueError(
            "recipient_only_fine_tune requires uniform receipt sampling; use recipient loss weighting rather "
            "than resampling whole receipts so protected fields remain frozen"
        )
    loader_performance_kwargs: dict[str, object] = {}
    if num_workers > 0:
        # v12's optional train augmentation reads its epoch from the shared
        # dataset counter above, so persistent Windows workers receive the
        # same deterministic (seed, epoch, record-id) perturbation without
        # respawning every epoch.  The validation dataset has no augmentation
        # and can safely share the same persistent-worker setting.
        loader_performance_kwargs = {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=_collate_recipient_only if recipient_only_fine_tune else _collate_receipts,
        pin_memory=uses_cuda,
        **loader_performance_kwargs,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_receipts,
        pin_memory=uses_cuda,
        **loader_performance_kwargs,
    )
    model = build_unified_reader(
        payment_vocab_size=len(payment_characters) + 1,
        config=config,
        payment_bank_prefix_vocab_size=(len(payment_bank_prefix_classes) if payment_bank_prefix_classes is not None else None),
        recipient_vocab_size=(len(recipient_characters) + 1 if recipient_characters is not None else None),
        status_text_vocab_size=(
            len(status_text_characters) + 1 if status_text_characters is not None else None
        ),
    )
    initialization_state, initialization = _parameter_only_initialization(
        init_checkpoint=init_checkpoint,
        init_checkpoint_mode=init_checkpoint_mode,
        config=config,
        amount_characters=amount_characters,
        time_characters=time_characters,
        payment_characters=payment_characters,
        recipient_characters=recipient_characters,
        status_text_characters=status_text_characters,
        payment_bank_prefix_classes=payment_bank_prefix_classes,
        torch=torch,
        target_state_dict=model.state_dict(),
        allow_v12_status_text_expansion=status_text_only_fine_tune,
    )
    if initialization_state is not None:
        # This is intentionally strict: equal tensor shapes are insufficient
        # when a CTC character or classifier-class ordering has changed.
        model.load_state_dict(initialization_state, strict=True)
        if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION:
            _assert_state_dict_exact_copy(model.state_dict(), initialization_state)
    if financial_label_policy is not None:
        initialization = {
            **initialization,
            "financial_label_policy": financial_label_policy,
        }
    model = model.to(target_device)
    fine_tune_policy: dict[str, object] = {
        "mode": "all_parameters",
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if recipient_only_fine_tune:
        trainable_recipient_prefix = (
            "recipient_open_text_"
            if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER
            and not recipient_open_text_unfreeze_legacy
            else "recipient_"
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_recipient_prefix))
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if trainable_parameter_count == 0:
            raise AssertionError("v12 recipient-only fine-tune found no recipient parameters")
        fine_tune_policy = {
            "mode": f"recipient_only_v{config.architecture_version}",
            "trainable_parameter_count": trainable_parameter_count,
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            ),
            "trainable_parameter_prefix": trainable_recipient_prefix,
            "open_text_legacy_recipient_unfrozen": recipient_open_text_unfreeze_legacy,
            "training_forward": f"private_recipient_branch_only_v{config.architecture_version}",
            "source_train_records": len(train_records),
            "recipient_train_records": len(training_records),
            "full_validation_schedule": "epoch_1_every_n_and_final_epoch",
            "validation_every": validation_every,
        }
    elif status_text_only_fine_tune:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("status_text_"))
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if trainable_parameter_count == 0:
            raise AssertionError("v13 status-text-only fine-tune found no status_text parameters")
        fine_tune_policy = {
            "mode": "status_text_only_v13",
            "trainable_parameter_count": trainable_parameter_count,
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            ),
            "trainable_parameter_prefix": "status_text_",
            "frozen_legacy_output_count": len(V12_ONNX_OUTPUT_NAMES),
            "source_train_records": len(train_records),
            "status_text_train_records": len(training_records),
            "full_validation_schedule": "epoch_1_every_n_and_final_epoch",
            "validation_every": validation_every,
        }
    if bool(status_policy["training_enabled"]):
        total_status = sum(status_counts["train"].values())
        status_weights = torch.tensor(
            [total_status / (len(STATUS_CLASSES) * status_counts["train"][name]) for name in STATUS_CLASSES],
            dtype=torch.float32,
            device=target_device,
        )
        status_train_criterion: Any | None = torch.nn.CrossEntropyLoss(weight=status_weights)
        status_validation_criterion: Any | None = torch.nn.CrossEntropyLoss()
    else:
        # Do not optimize an all-success status head.  Its logits are retained
        # only to preserve the stable single-ONNX interface and must be mapped
        # to review by delivery code.
        status_train_criterion = None
        status_validation_criterion = None
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    frozen_non_recipient_parameter_snapshot: dict[str, bytes] | None = None
    frozen_non_status_text_parameter_snapshot: dict[str, bytes] | None = None
    if (
        (
            validation_every > 1
            and init_checkpoint_mode in RECIPIENT_ONLY_INIT_CHECKPOINT_MODES
        )
        or init_checkpoint_mode
        in {
            INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
            INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
        }
    ):
        frozen_non_recipient_parameter_snapshot = _non_recipient_parameter_bytes(model)
        fine_tune_policy = {
            **fine_tune_policy,
            "frozen_non_recipient_byte_guard": "before_every_full_validation",
            "frozen_non_recipient_state_entry_count": len(
                frozen_non_recipient_parameter_snapshot
            ),
            **(
                {
                    "initialization_non_recipient_byte_guard":
                    "before_epoch_zero_validation"
                }
                if init_checkpoint_mode
                == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
                else {}
            ),
        }
    if status_text_only_fine_tune:
        frozen_non_status_text_parameter_snapshot = _non_status_text_parameter_bytes(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output_dir / "labels.json",
        {
            "schema_version": SCHEMA_VERSION,
            "amount_blank_index": NUMERIC_BLANK_INDEX,
            "amount_characters": amount_characters,
            "time_blank_index": NUMERIC_BLANK_INDEX,
            "time_characters": time_characters,
            "payment_blank_index": PAYMENT_BLANK_INDEX,
            "payment_characters": payment_characters,
            "status_classes": list(STATUS_CLASSES),
            **(
                {
                    "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
                    "status_text_characters": status_text_characters,
                    "status_text_charset_sha256": hashlib.sha256(
                        "".join(status_text_characters).encode("utf-8")
                    ).hexdigest(),
                    "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
                    "status_text_target": STATUS_TEXT_TARGET,
                    "status_text_oov_by_split": status_text_oov,
                    "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
                }
                if status_text_characters is not None
                else {}
            ),
            "structured_target_counts": structured_counts,
            "checkpoint_selection_policy": checkpoint_selection_policy,
            "initialization": initialization,
            "training_runtime": training_runtime,
            "fine_tune_policy": fine_tune_policy,
            "payment_charset_sha256": hashlib.sha256("".join(payment_characters).encode("utf-8")).hexdigest(),
            **(
                {
                    "recipient_blank_index": RECIPIENT_BLANK_INDEX,
                    "recipient_characters": recipient_characters,
                    "recipient_charset_sha256": hashlib.sha256(
                        "".join(recipient_characters).encode("utf-8")
                    ).hexdigest(),
                    "recipient_charset_source": _recipient_charset_source(config),
                    "recipient_target": _recipient_target_mode(config),
                    "recipient_oov_by_split": recipient_oov,
                    "recipient_sampling_policy": recipient_sampling_policy,
                    "recipient_confidence_policy": recipient_confidence_policy,
                    "recipient_tail_loss_policy": recipient_tail_loss_policy,
                    "recipient_train_augmentation_policy": recipient_train_augmentation_policy,
                    "recipient_train_split_policy": recipient_train_split_policy,
                    **_recipient_artifact_metadata(
                        config,
                        recipient_sampling_policy=recipient_sampling_policy,
                        recipient_confidence_policy=recipient_confidence_policy,
                        recipient_tail_loss_policy=recipient_tail_loss_policy,
                        recipient_train_augmentation_policy=recipient_train_augmentation_policy,
                    ),
                }
                if recipient_characters is not None
                else {}
            ),
            **(
                {
                    "payment_bank_prefix_classes": payment_bank_prefix_classes,
                    "payment_bank_prefix_min_support": payment_bank_prefix_min_support,
                    "payment_bank_prefix_class_counts": payment_bank_prefix_counts,
                    "payment_bank_prefix_train_class_counts": payment_bank_train_counts,
                    "payment_bank_prefix_oov_by_split": payment_bank_prefix_oov,
                }
                if payment_bank_prefix_classes is not None
                else {"numeric_blank_index": NUMERIC_BLANK_INDEX, "numeric_characters": amount_characters}
            ),
        },
    )

    history: list[dict[str, object]] = []
    # ``None`` remains until an epoch passes recipient-priority protection
    # floors. Balanced mode always produces the historical score on epoch one.
    best_score: tuple[float, ...] | None = None
    best_epoch: int | None = None
    best_path = output_dir / "best.pt"
    if init_checkpoint_mode in {
        INIT_CHECKPOINT_MODE_RECIPIENT_INPUT_WIDTH_EXPANSION,
        INIT_CHECKPOINT_MODE_RECIPIENT_OPEN_TEXT_ADAPTER,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    }:
        # Measure and persist the exact transplanted model as epoch zero so a
        # pilot cannot silently return a checkpoint worse than its own safe
        # starting point.  The adapter route must be decision-identical here;
        # the full-crop route records the deliberate preprocessing change as
        # its own epoch-zero baseline before any optimiser update.
        if init_checkpoint is None:
            raise AssertionError(f"{init_checkpoint_mode} has no seed checkpoint")
        initialization_started = perf_counter()
        if (
            init_checkpoint_mode
            == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
            and frozen_non_recipient_parameter_snapshot is not None
        ):
            _assert_non_recipient_parameter_bytes(
                model, frozen_non_recipient_parameter_snapshot
            )
        model.eval()
        initialization_validation = _evaluate_model(
            model,
            validation_loader,
            config=config,
            device=target_device,
            amount_characters=amount_characters,
            amount_to_id=amount_to_id,
            time_characters=time_characters,
            time_to_id=time_to_id,
            payment_characters=payment_characters,
            payment_to_id=payment_to_id,
            recipient_characters=recipient_characters,
            recipient_to_id=recipient_to_id,
            status_text_characters=status_text_characters,
            status_text_to_id=status_text_to_id,
            payment_bank_prefix_classes=payment_bank_prefix_classes,
            payment_bank_class_weights=None,
            status_to_id=status_to_id,
            status_criterion=status_validation_criterion,
            status_enabled=bool(status_policy["training_enabled"]),
            payment_loss_weight=payment_loss_weight,
            recipient_loss_weight=recipient_loss_weight,
            status_text_loss_weight=status_text_loss_weight,
            ctc_loss_weight=ctc_loss_weight,
            structured_loss_weight=structured_loss_weight,
            torch=torch,
            status_text_only=status_text_only_fine_tune,
        )
        if uses_cuda:
            torch.cuda.synchronize(target_device)
        initialization_score, initialization_failures = _checkpoint_selection_score(
            initialization_validation,
            config=config,
            status_policy=status_policy,
            policy=checkpoint_selection_policy,
        )
        if initialization_score is None:
            raise ValueError(
                f"{init_checkpoint_mode} initialization does not satisfy the protected checkpoint floors: "
                + "; ".join(initialization_failures)
            )
        initialization_record: dict[str, object] = {
            "epoch": 0,
            "train_loss": None,
            "train_seconds": 0.0,
            "validation_performed": True,
            "validation_schedule_reason": "initialization_baseline",
            "validation_seconds": perf_counter() - initialization_started,
            "epoch_seconds": perf_counter() - initialization_started,
            "val_loss": initialization_validation["loss"],
            "val_exact_match": initialization_validation["exact_match"],
            "val_delivery_coverage": initialization_validation["delivery_coverage"],
            "val_delivery_exact_match": initialization_validation["delivery_exact_match"],
            "val_delivery_exact_overall": initialization_validation["delivery_exact_overall"],
            "val_delivery_false_accepts": initialization_validation["delivery_false_accepts"],
            "val_verifier_exact_match": initialization_validation["verifier_exact_match"],
            "val_verifier_macro_exact_match": initialization_validation["verifier_macro_exact_match"],
            "val_verifier_by_field": initialization_validation["verifier_by_field"],
            "val_candidate_text_exact_match": initialization_validation["candidate_text_exact_match"],
            "val_candidate_text_macro_exact_match": initialization_validation["candidate_text_macro_exact_match"],
            "val_candidate_text_by_field": initialization_validation["candidate_text_by_field"],
            "val_ctc_by_field": initialization_validation["ctc_by_field"],
            "val_by_field": initialization_validation["by_field"],
            "val_status_non_success_to_success": initialization_validation["status_non_success_to_success"],
            "checkpoint_selection_eligible": True,
            "checkpoint_selection_protection_failures": [],
            "checkpoint_selection_score": list(initialization_score),
            "checkpoint_protection": _checkpoint_protection_report(
                initialization_validation,
                policy=checkpoint_selection_policy,
                failures=initialization_failures,
            ),
        }
        baseline_payload = _load_checkpoint(Path(init_checkpoint), torch=torch)
        baseline_payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": _kind_for_config(config),
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "amount_characters": amount_characters,
                "time_characters": time_characters,
                "payment_characters": payment_characters,
                "recipient_characters": recipient_characters,
                "recipient_blank_index": RECIPIENT_BLANK_INDEX,
                "recipient_charset_sha256": hashlib.sha256(
                    "".join(recipient_characters or []).encode("utf-8")
                ).hexdigest(),
                "recipient_charset_source": _recipient_charset_source(config),
                "recipient_target": _recipient_target_mode(config),
                "recipient_oov_by_split": recipient_oov,
                "recipient_sampling_policy": recipient_sampling_policy,
                "recipient_confidence_policy": recipient_confidence_policy,
                "recipient_tail_loss_policy": recipient_tail_loss_policy,
                "recipient_train_augmentation_policy": recipient_train_augmentation_policy,
                "recipient_train_split_policy": recipient_train_split_policy,
                "status_classes": list(STATUS_CLASSES),
                "field_counts": field_counts,
                "status_class_counts": status_counts,
                "structured_target_counts": structured_counts,
                "status_head_policy": status_policy,
                "payment_oov_by_split": payment_oov,
                "payment_bank_prefix_classes": payment_bank_prefix_classes,
                "payment_bank_prefix_min_support": payment_bank_prefix_min_support,
                "payment_bank_prefix_class_counts": payment_bank_prefix_counts,
                "payment_bank_prefix_train_class_counts": payment_bank_train_counts,
                "payment_bank_prefix_oov_by_split": payment_bank_prefix_oov,
                "payment_loss_weight": payment_loss_weight,
                "recipient_loss_weight": recipient_loss_weight,
                "checkpoint_selection_policy": checkpoint_selection_policy,
                "initialization": initialization,
                "training_runtime": training_runtime,
                "fine_tune_policy": fine_tune_policy,
                "ctc_loss_weight": ctc_loss_weight,
                "structured_loss_weight": structured_loss_weight,
                "epoch": 0,
                "metrics": initialization_record,
                **_recipient_artifact_metadata(
                    config,
                    recipient_sampling_policy=recipient_sampling_policy,
                    recipient_confidence_policy=recipient_confidence_policy,
                    recipient_tail_loss_policy=recipient_tail_loss_policy,
                    recipient_train_augmentation_policy=recipient_train_augmentation_policy,
                ),
            }
        )
        _write_checkpoint(best_path, baseline_payload, torch=torch)
        history.append(initialization_record)
        best_score = initialization_score
        best_epoch = 0
    for epoch in range(1, epochs + 1):
        epoch_started = perf_counter()
        model.train()
        train_dataset.set_epoch(epoch)
        total_loss_tensor: Any | None = None
        total_receipts = 0
        total_batches = len(train_loader)
        for batch_index, batch in enumerate(train_loader, start=1):
            if recipient_only_fine_tune:
                recipient_value_images, batch_records = batch
                recipient_value_images = recipient_value_images.to(target_device, non_blocking=uses_cuda)
                recipient_logits = _recipient_only_logits(
                    model,
                    recipient_value_images,
                    config=config,
                )
                amount_logits = time_logits = payment_logits = status_logits = None
                structured_outputs = None
            else:
                field_images, recipient_value_images, batch_records = _unpack_receipt_batch(batch, config=config)
                field_images = field_images.to(target_device, non_blocking=uses_cuda)
                if recipient_value_images is not None:
                    recipient_value_images = recipient_value_images.to(target_device, non_blocking=uses_cuda)
                outputs = _unpack_reader_outputs(
                    model(field_images, recipient_value_images),
                    config=config,
                )
                amount_logits = outputs["amount_logits"]
                time_logits = outputs["time_logits"]
                payment_logits = outputs["payment_logits"]
                status_logits = outputs["status_logits"]
                recipient_logits = outputs.get("recipient_logits")
                status_text_logits = outputs.get("status_text_logits")
                structured_outputs = outputs if _uses_structured_heads(config) else None
            if recipient_only_fine_tune:
                status_text_logits = None
            optimizer.zero_grad(set_to_none=True)
            recipient_confidence_weights = (
                _recipient_teacher_confidence_weights(
                    batch_records,
                    low_confidence_threshold=recipient_low_confidence_threshold,
                    low_confidence_loss_weight=recipient_low_confidence_loss_weight,
                    curriculum_epoch=epoch,
                    curriculum_epochs=recipient_confidence_curriculum_epochs,
                )
                if recipient_confidence_policy["mode"] != "none"
                else None
            )
            recipient_tail_weights = (
                _recipient_tail_loss_weights(
                    batch_records,
                    policy=recipient_tail_loss_policy,
                    character_counts=recipient_tail_loss_character_counts,
                )
                if recipient_tail_loss_policy["mode"] != "none"
                else None
            )
            recipient_sample_weights = _combine_recipient_loss_weights(
                recipient_confidence_weights,
                recipient_tail_weights,
            )
            loss, _ = _batch_loss(
                amount_logits,
                time_logits,
                payment_logits,
                status_logits,
                batch_records,
                amount_to_id=amount_to_id,
                time_to_id=time_to_id,
                payment_to_id=payment_to_id,
                recipient_logits=recipient_logits,
                recipient_to_id=recipient_to_id,
                status_text_logits=status_text_logits,
                status_text_to_id=status_text_to_id,
                payment_bank_prefix_classes=payment_bank_prefix_classes,
                payment_bank_class_weights=payment_bank_train_weights,
                status_to_id=status_to_id,
                status_criterion=status_train_criterion,
                status_enabled=bool(status_policy["training_enabled"]),
                payment_loss_weight=payment_loss_weight,
                recipient_loss_weight=recipient_loss_weight,
                status_text_loss_weight=status_text_loss_weight,
                config=config,
                structured_outputs=structured_outputs,
                ctc_loss_weight=ctc_loss_weight,
                structured_loss_weight=structured_loss_weight,
                torch=torch,
                recipient_sample_weights=recipient_sample_weights,
                collect_metrics=False,
                recipient_only=recipient_only_fine_tune,
                status_text_only=status_text_only_fine_tune,
            )
            if loss is None:
                raise AssertionError("a training batch must produce a loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
            optimizer.step()
            weighted_loss = loss.detach() * len(batch_records)
            if total_loss_tensor is None:
                total_loss_tensor = weighted_loss
            else:
                total_loss_tensor.add_(weighted_loss)
            total_receipts += len(batch_records)
            if train_progress_every and (
                batch_index == 1 or batch_index % train_progress_every == 0 or batch_index == total_batches
            ):
                print(
                    f"train epoch {epoch}/{epochs}: batch {batch_index}/{total_batches} "
                    f"receipts={total_receipts} elapsed_s={perf_counter() - epoch_started:.1f}",
                    flush=True,
                )
        # The hot recipient-only path intentionally avoids per-batch CPU
        # reads.  Synchronise once at the epoch boundary so the recorded
        # training/validation timings are not misleadingly split across the
        # asynchronous CUDA queue.
        if uses_cuda:
            torch.cuda.synchronize(target_device)
        train_seconds = perf_counter() - epoch_started
        full_validation_planned = _is_full_validation_epoch(
            epoch=epoch,
            epochs=epochs,
            validation_every=validation_every,
        )
        validation_reason = _full_validation_epoch_reason(
            epoch=epoch,
            epochs=epochs,
            validation_every=validation_every,
        )
        common_epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": (
                float((total_loss_tensor / total_receipts).cpu())
                if total_loss_tensor is not None and total_receipts > 0
                else math.nan
            ),
            "train_seconds": train_seconds,
            "validation_performed": full_validation_planned,
            "validation_schedule_reason": validation_reason,
        }
        if full_validation_planned:
            validation_started = perf_counter()
            if frozen_non_recipient_parameter_snapshot is not None:
                _assert_non_recipient_parameter_bytes(model, frozen_non_recipient_parameter_snapshot)
            if frozen_non_status_text_parameter_snapshot is not None:
                _assert_non_status_text_parameter_bytes(
                    model, frozen_non_status_text_parameter_snapshot
                )
            validation = _evaluate_model(
                model,
                validation_loader,
                config=config,
                device=target_device,
                amount_characters=amount_characters,
                amount_to_id=amount_to_id,
                time_characters=time_characters,
                time_to_id=time_to_id,
                payment_characters=payment_characters,
                payment_to_id=payment_to_id,
                recipient_characters=recipient_characters,
                recipient_to_id=recipient_to_id,
                status_text_characters=status_text_characters,
                status_text_to_id=status_text_to_id,
                payment_bank_prefix_classes=payment_bank_prefix_classes,
                payment_bank_class_weights=None,
                status_to_id=status_to_id,
                status_criterion=status_validation_criterion,
                status_enabled=bool(status_policy["training_enabled"]),
                payment_loss_weight=payment_loss_weight,
                recipient_loss_weight=recipient_loss_weight,
                status_text_loss_weight=status_text_loss_weight,
                ctc_loss_weight=ctc_loss_weight,
                structured_loss_weight=structured_loss_weight,
                torch=torch,
                status_text_only=status_text_only_fine_tune,
            )
            if uses_cuda:
                torch.cuda.synchronize(target_device)
            validation_seconds = perf_counter() - validation_started
            epoch_record = {
                **common_epoch_record,
                "validation_seconds": validation_seconds,
                "epoch_seconds": perf_counter() - epoch_started,
                "val_loss": validation["loss"],
                "val_exact_match": validation["exact_match"],
                "val_delivery_coverage": validation["delivery_coverage"],
                "val_delivery_exact_match": validation["delivery_exact_match"],
                "val_delivery_exact_overall": validation["delivery_exact_overall"],
                "val_delivery_false_accepts": validation["delivery_false_accepts"],
                "val_verifier_exact_match": validation["verifier_exact_match"],
                "val_verifier_macro_exact_match": validation["verifier_macro_exact_match"],
                "val_verifier_by_field": validation["verifier_by_field"],
                "val_candidate_text_exact_match": validation["candidate_text_exact_match"],
                "val_candidate_text_macro_exact_match": validation["candidate_text_macro_exact_match"],
                "val_candidate_text_by_field": validation["candidate_text_by_field"],
                "val_ctc_by_field": validation["ctc_by_field"],
                "val_by_field": validation["by_field"],
                "val_status_non_success_to_success": validation["status_non_success_to_success"],
            }
            score, protection_failures = _checkpoint_selection_score(
                validation,
                config=config,
                status_policy=status_policy,
                policy=checkpoint_selection_policy,
            )
            protection_report: Mapping[str, object] | None = _checkpoint_protection_report(
                validation,
                policy=checkpoint_selection_policy,
                failures=protection_failures,
            )
            epoch_record["checkpoint_selection_eligible"] = score is not None
            epoch_record["checkpoint_selection_protection_failures"] = protection_failures
            epoch_record["checkpoint_selection_score"] = list(score) if score is not None else None
            epoch_record["checkpoint_protection"] = protection_report
        else:
            # A skipped full validation cannot provide financial guardrail
            # evidence.  It is retained in history for progress visibility
            # but is categorically ineligible for either checkpoint.
            validation = None
            validation_seconds = None
            score = None
            protection_report = None
            epoch_record = {
                **common_epoch_record,
                "validation_seconds": None,
                "epoch_seconds": perf_counter() - epoch_started,
                "val_loss": None,
                "val_exact_match": None,
                "val_delivery_coverage": None,
                "val_delivery_exact_match": None,
                "val_delivery_exact_overall": None,
                "val_delivery_false_accepts": None,
                "val_verifier_exact_match": None,
                "val_verifier_macro_exact_match": None,
                "val_verifier_by_field": None,
                "val_candidate_text_exact_match": None,
                "val_candidate_text_macro_exact_match": None,
                "val_candidate_text_by_field": None,
                "val_ctc_by_field": None,
                "val_by_field": None,
                "val_status_non_success_to_success": None,
                "checkpoint_selection_eligible": False,
                "checkpoint_selection_protection_failures": ["full_validation_not_scheduled"],
                "checkpoint_selection_score": None,
                "checkpoint_protection": None,
            }
        history.append(epoch_record)
        checkpoint_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": _kind_for_config(config),
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "amount_characters": amount_characters,
            "time_characters": time_characters,
            **({"numeric_characters": amount_characters} if not _uses_modern_protocol(config) else {}),
            "payment_characters": payment_characters,
            **(
                {
                    "recipient_characters": recipient_characters,
                    "recipient_blank_index": RECIPIENT_BLANK_INDEX,
                    "recipient_charset_sha256": hashlib.sha256(
                        "".join(recipient_characters).encode("utf-8")
                    ).hexdigest(),
                    "recipient_charset_source": _recipient_charset_source(config),
                    "recipient_target": _recipient_target_mode(config),
                    "recipient_oov_by_split": recipient_oov,
                    "recipient_sampling_policy": recipient_sampling_policy,
                    "recipient_confidence_policy": recipient_confidence_policy,
                    "recipient_tail_loss_policy": recipient_tail_loss_policy,
                    "recipient_train_augmentation_policy": recipient_train_augmentation_policy,
                    "recipient_train_split_policy": recipient_train_split_policy,
                    **_recipient_artifact_metadata(
                        config,
                        recipient_sampling_policy=recipient_sampling_policy,
                        recipient_confidence_policy=recipient_confidence_policy,
                        recipient_tail_loss_policy=recipient_tail_loss_policy,
                        recipient_train_augmentation_policy=recipient_train_augmentation_policy,
                    ),
                }
                if recipient_characters is not None
                else {}
            ),
            "status_classes": list(STATUS_CLASSES),
            **(
                {
                    "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
                    "status_text_characters": status_text_characters,
                    "status_text_charset_sha256": hashlib.sha256(
                        "".join(status_text_characters).encode("utf-8")
                    ).hexdigest(),
                    "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
                    "status_text_target": STATUS_TEXT_TARGET,
                    "status_text_oov_by_split": status_text_oov,
                    "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
                }
                if status_text_characters is not None
                else {}
            ),
            "field_counts": field_counts,
            "status_class_counts": status_counts,
            "structured_target_counts": structured_counts,
            "status_head_policy": status_policy,
            "payment_oov_by_split": payment_oov,
            "payment_bank_prefix_classes": payment_bank_prefix_classes,
            "payment_bank_prefix_min_support": payment_bank_prefix_min_support if payment_bank_prefix_classes is not None else None,
            "payment_bank_prefix_class_counts": payment_bank_prefix_counts,
            "payment_bank_prefix_train_class_counts": payment_bank_train_counts,
            "payment_bank_prefix_oov_by_split": payment_bank_prefix_oov,
            "payment_loss_weight": payment_loss_weight,
            "recipient_loss_weight": recipient_loss_weight,
            "status_text_loss_weight": status_text_loss_weight,
            "checkpoint_selection_policy": checkpoint_selection_policy,
            "initialization": initialization,
            "training_runtime": training_runtime,
            "fine_tune_policy": fine_tune_policy,
            "ctc_loss_weight": ctc_loss_weight,
            "structured_loss_weight": structured_loss_weight,
            "epoch": epoch,
            "metrics": epoch_record,
        }
        # Leave recovery with the most recent audited model, never an
        # unvalidated interior epoch whose financial guardrail evidence was
        # intentionally skipped.
        if full_validation_planned:
            _write_checkpoint(output_dir / "last.pt", checkpoint_payload, torch=torch)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_epoch = epoch
            _write_checkpoint(best_path, checkpoint_payload, torch=torch)
        _atomic_write_json(
            output_dir / "training_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": _kind_for_config(config),
                "config": asdict(config),
                "field_counts": field_counts,
                "status_class_counts": status_counts,
                "structured_target_counts": structured_counts,
                "status_head_policy": status_policy,
                "status_text_oov_by_split": status_text_oov,
                "status_text_charset_sha256": (
                    hashlib.sha256("".join(status_text_characters).encode("utf-8")).hexdigest()
                    if status_text_characters is not None
                    else None
                ),
                "status_text_charset_source": (
                    STATUS_TEXT_CHARSET_SOURCE if status_text_characters is not None else None
                ),
                "status_text_target": STATUS_TEXT_TARGET if status_text_characters is not None else None,
                "status_text_runtime_policy": (
                    STATUS_TEXT_RUNTIME_POLICY if status_text_characters is not None else None
                ),
                "payment_oov_by_split": payment_oov,
                "payment_bank_prefix_classes": payment_bank_prefix_classes,
                "payment_bank_prefix_min_support": payment_bank_prefix_min_support
                if payment_bank_prefix_classes is not None
                else None,
                "payment_bank_prefix_class_counts": payment_bank_prefix_counts,
                "payment_bank_prefix_train_class_counts": payment_bank_train_counts,
                "payment_bank_prefix_oov_by_split": payment_bank_prefix_oov,
                "recipient_oov_by_split": recipient_oov,
                "recipient_target": _recipient_target_mode(config),
                "recipient_loss_weight": recipient_loss_weight,
                "recipient_sampling_policy": recipient_sampling_policy,
                "recipient_confidence_policy": recipient_confidence_policy,
                "recipient_tail_loss_policy": recipient_tail_loss_policy,
                "recipient_train_augmentation_policy": recipient_train_augmentation_policy,
                "recipient_train_split_policy": recipient_train_split_policy,
                "checkpoint_selection_policy": checkpoint_selection_policy,
                "initialization": initialization,
                "training_runtime": training_runtime,
                "fine_tune_policy": fine_tune_policy,
                "best_checkpoint_epoch": best_epoch,
                "best_checkpoint_score": list(best_score) if best_score is not None else None,
                "records": history,
                "warning": (
                    "Paddle teacher labels are not independent truth. v5-v12 text candidates remain review-only until "
                    "a separate acceptance policy passes group-isolated human-truth calibration. When "
                    "status_head_policy.runtime_policy is review_only, status logits are also not a delivery "
                    "decision and runtime must emit review."
                ),
            },
        )
        if validation is None:
            print(
                f"epoch {epoch}/{epochs}: train_loss={float(epoch_record['train_loss']):.4f} "
                f"validation=scheduled_skip train_s={train_seconds:.1f} "
                "checkpoint=unvalidated"
            )
        else:
            assert isinstance(protection_report, Mapping)
            assert validation_seconds is not None
            print(
                f"epoch {epoch}/{epochs}: train_loss={float(epoch_record['train_loss']):.4f} "
                f"val_loss={float(validation['loss']):.4f} val_exact_match={float(validation['exact_match']):.2%} "
                f"val_candidate={float(validation['candidate_text_macro_exact_match'] or 0.0):.2%} "
                f"val_verifier={_format_exact_match(validation['verifier_macro_exact_match'])} "
                f"val_delivery={float(validation['delivery_exact_overall']):.2%} "
                f"coverage={float(validation['delivery_coverage']):.2%} "
                f"train_s={train_seconds:.1f} val_s={validation_seconds:.1f} "
                f"{_format_checkpoint_protection_report(protection_report)} "
                f"checkpoint={'eligible' if score is not None else 'protected'}"
            )
    if best_epoch is None:
        final_report = history[-1].get("checkpoint_protection") if history else None
        if isinstance(final_report, Mapping):
            print(
                "training_complete: best_checkpoint=none "
                f"last_epoch={history[-1]['epoch']}/{epochs} "
                f"{_format_checkpoint_protection_report(final_report)}"
            )
        raise ValueError(
            "No epoch met checkpoint protection floors; best.pt was not written. "
            "Inspect training_summary.json and recalibrate the validation floors."
        )
    best_record = next(record for record in history if record["epoch"] == best_epoch)
    best_report = best_record.get("checkpoint_protection")
    last_report = history[-1].get("checkpoint_protection")
    if not isinstance(best_report, Mapping) or not isinstance(last_report, Mapping):
        raise AssertionError("training history is missing checkpoint protection evidence")
    print(
        "training_complete: "
        f"best_epoch={best_epoch}/{epochs} "
        f"best_{_format_checkpoint_protection_report(best_report)} "
        f"last_epoch={history[-1]['epoch']}/{epochs} "
        f"last_{_format_checkpoint_protection_report(last_report)}"
    )
    return best_path


def _load_checkpoint(path: Path, *, torch: Any) -> Mapping[str, object]:
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Unified OCR checkpoint must be a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") not in SUPPORTED_KINDS:
        raise ValueError("Unsupported unified OCR checkpoint schema")
    return payload


def _config_from_mapping(
    raw: Mapping[str, object], *, artifact_kind: object, source: str
) -> UnifiedReaderConfig:
    """Read a config while treating pre-v4 artifacts as explicit v3.

    Old v3 artifacts predate ``architecture_version``.  Its absence is only
    valid when the kind itself says v3; this avoids accidentally loading a v3
    state dict into the v4 decoder topology.
    """
    inferred_architecture = _architecture_for_kind(artifact_kind)
    raw_architecture = raw.get("architecture_version", inferred_architecture)
    try:
        architecture_version = int(raw_architecture)
        config = UnifiedReaderConfig(
            architecture_version=architecture_version,
            image_height=int(raw["image_height"]),
            image_width=int(raw["image_width"]),
            base_channels=int(raw["base_channels"]),
            numeric_hidden_size=int(raw["numeric_hidden_size"]),
            payment_hidden_size=int(raw["payment_hidden_size"]),
            recipient_hidden_size=(
                int(raw["recipient_hidden_size"])
                if raw.get("recipient_hidden_size") is not None
                else None
            ),
            recipient_value_left_trim=float(raw.get("recipient_value_left_trim", 0.30)),
            recipient_input_height=int(raw.get("recipient_input_height", 128)),
            recipient_input_width=int(raw.get("recipient_input_width", 1024)),
            recipient_branch_channels=(
                int(raw["recipient_branch_channels"])
                if raw.get("recipient_branch_channels") is not None
                else None
            ),
            recipient_open_text_layers=int(raw.get("recipient_open_text_layers") or 0),
            recipient_open_text_heads=int(raw.get("recipient_open_text_heads") or 8),
            recipient_open_text_feedforward=(
                int(raw["recipient_open_text_feedforward"])
                if raw.get("recipient_open_text_feedforward") is not None
                else None
            ),
            recipient_open_text_dropout=float(raw.get("recipient_open_text_dropout") or 0.0),
            recipient_backbone=str(raw.get("recipient_backbone") or "legacy_depthwise_gru_v1"),
            pooled_width=int(raw["pooled_width"]),
            amount_format_min_confidence=float(raw.get("amount_format_min_confidence", 0.90)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} has an invalid model config") from error
    config.validate()
    if _kind_for_config(config) != artifact_kind:
        raise ValueError(
            f"{source} architecture v{config.architecture_version} does not match artifact kind {artifact_kind!r}"
        )
    return config


def _checkpoint_config(payload: Mapping[str, object]) -> UnifiedReaderConfig:
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("Unified OCR checkpoint has no model config")
    return _config_from_mapping(raw, artifact_kind=payload.get("kind"), source="Unified OCR checkpoint")


def _checkpoint_labels(
    payload: Mapping[str, object], *, config: UnifiedReaderConfig
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str] | None,
    list[str],
    list[str] | None,
]:
    """Load architecture-specific CTC and finite-class label maps safely."""
    numeric = payload.get("numeric_characters")
    amount = payload.get("amount_characters")
    time = payload.get("time_characters")
    payment = payload.get("payment_characters")
    status = payload.get("status_classes")
    if not isinstance(payment, list) or not isinstance(status, list):
        raise ValueError("Unified OCR checkpoint has no label maps")
    if not all(isinstance(character, str) and len(character) == 1 for character in payment):
        raise ValueError("Unified OCR checkpoint payment charset must contain single Unicode code points")
    if not payment or len(set(payment)) != len(payment):
        raise ValueError("Unified OCR checkpoint payment charset is empty or has duplicates")
    if status != list(STATUS_CLASSES):
        raise ValueError("Unified OCR checkpoint status class order is unsupported")
    if _uses_modern_protocol(config):
        expected_amount = V8_AMOUNT_CHARACTERS if _uses_v8_protocol(config) else V6_AMOUNT_CHARACTERS
        if amount != list(expected_amount) or time != list(V6_TIME_CHARACTERS):
            raise ValueError("Unified v6-v13 OCR checkpoint amount/time label maps are unsupported")
        bank_classes = payload.get("payment_bank_prefix_classes")
        if (
            not isinstance(bank_classes, list)
            or len(bank_classes) < 2
            or bank_classes[0] != PAYMENT_BANK_OTHER_CLASS
            or not all(isinstance(value, str) and value for value in bank_classes)
            or len(set(bank_classes)) != len(bank_classes)
            or bank_classes[1:] != sorted(bank_classes[1:])
        ):
            raise ValueError("Unified v6-v13 OCR checkpoint bank-prefix class map is invalid")
        recipient: list[str] | None = None
        if _uses_recipient_protocol(config):
            raw_recipient = payload.get("recipient_characters")
            if (
                not isinstance(raw_recipient, list)
                or not raw_recipient
                or not all(
                    isinstance(character, str)
                    and len(character) == 1
                    and character.isprintable()
                    for character in raw_recipient
                )
                or len(set(raw_recipient)) != len(raw_recipient)
                or raw_recipient != sorted(raw_recipient)
                or payload.get("recipient_blank_index") != RECIPIENT_BLANK_INDEX
            ):
                raise ValueError("Unified v9-v13 OCR checkpoint recipient charset or blank index is invalid")
            recipient_sha256 = hashlib.sha256("".join(raw_recipient).encode("utf-8")).hexdigest()
            if payload.get("recipient_charset_sha256") != recipient_sha256:
                raise ValueError("Unified v9/v10/v11/v12 OCR checkpoint recipient charset SHA-256 is invalid")
            expected_recipient_charset_source = _recipient_charset_source(config)
            if payload.get("recipient_charset_source") != expected_recipient_charset_source:
                raise ValueError("Unified v9/v10/v11/v12 OCR checkpoint recipient charset source is invalid")
            if payload.get("recipient_target") != _recipient_target_mode(config):
                raise ValueError("Unified v9/v10/v11/v12 OCR checkpoint recipient target contract is invalid")
            if _is_v11(config) or _uses_v12_recipient_topology(config):
                _recipient_artifact_metadata(
                    config,
                    recipient_sampling_policy=payload.get("recipient_sampling_policy"),
                    recipient_confidence_policy=payload.get("recipient_confidence_policy"),
                    recipient_tail_loss_policy=payload.get("recipient_tail_loss_policy"),
                    recipient_train_augmentation_policy=payload.get("recipient_train_augmentation_policy"),
                )
            recipient = list(raw_recipient)
        return list(amount), list(time), list(payment), recipient, list(status), list(bank_classes)
    if numeric != list(NUMERIC_CHARACTERS):
        raise ValueError("Unified OCR checkpoint numeric label map is not the supported fixed numeric charset")
    return list(numeric), list(numeric), list(payment), None, list(status), None


def _validate_status_text_oov_audit(value: object, *, source: str) -> None:
    """Fail closed on the train-only v13 status alphabet audit."""
    if not isinstance(value, Mapping) or set(value) != {"train", "val", "test"}:
        raise ValueError(f"{source} status-text OOV audit is invalid")
    for split in ("train", "val", "test"):
        audit = value[split]
        if not isinstance(audit, Mapping) or set(audit) != {
            "records",
            "oov_records",
            "oov_characters",
            "examples",
        }:
            raise ValueError(f"{source} status-text OOV audit is invalid")
        records = audit.get("records")
        oov_records = audit.get("oov_records")
        oov_characters = audit.get("oov_characters")
        examples = audit.get("examples")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or isinstance(oov_records, bool)
            or not isinstance(oov_records, int)
            or isinstance(oov_characters, bool)
            or not isinstance(oov_characters, int)
            or records < 0
            or oov_records < 0
            or oov_records > records
            or oov_characters < 0
            or not isinstance(examples, list)
            or len(examples) > min(20, oov_records)
        ):
            raise ValueError(f"{source} status-text OOV audit is invalid")
        for example in examples:
            if (
                not isinstance(example, Mapping)
                or set(example) != {"id", "characters", "text"}
                or not isinstance(example.get("id"), str)
                or not example.get("id")
                or not isinstance(example.get("characters"), str)
                or not example.get("characters")
                or not isinstance(example.get("text"), str)
                or not example.get("text")
                or any(character not in str(example["text"]) for character in str(example["characters"]))
            ):
                raise ValueError(f"{source} status-text OOV audit is invalid")
    train_audit = value["train"]
    if (
        train_audit["oov_records"] != 0
        or train_audit["oov_characters"] != 0
        or train_audit["examples"]
    ):
        raise ValueError(f"{source} status-text train split must not contain OOV characters")


def _checkpoint_status_text_characters(
    payload: Mapping[str, object], *, config: UnifiedReaderConfig
) -> list[str] | None:
    """Validate the additive v13 visible-status CTC label contract."""
    if not _is_v13(config):
        return None
    raw_characters = payload.get("status_text_characters")
    if (
        not isinstance(raw_characters, list)
        or not raw_characters
        or not all(
            isinstance(character, str)
            and len(character) == 1
            and character.isprintable()
            for character in raw_characters
        )
        or raw_characters != sorted(raw_characters)
        or len(set(raw_characters)) != len(raw_characters)
        or payload.get("status_text_blank_index") != STATUS_TEXT_BLANK_INDEX
        or payload.get("status_text_charset_source") != STATUS_TEXT_CHARSET_SOURCE
        or payload.get("status_text_target") != STATUS_TEXT_TARGET
        or payload.get("status_text_runtime_policy") != STATUS_TEXT_RUNTIME_POLICY
    ):
        raise ValueError("Unified v13 OCR checkpoint status-text charset contract is invalid")
    expected_sha256 = hashlib.sha256("".join(raw_characters).encode("utf-8")).hexdigest()
    if payload.get("status_text_charset_sha256") != expected_sha256:
        raise ValueError("Unified v13 OCR checkpoint status-text charset SHA-256 is invalid")
    _validate_status_text_oov_audit(
        payload.get("status_text_oov_by_split"),
        source="Unified v13 OCR checkpoint",
    )
    return list(raw_characters)


def _validate_exported_onnx(
    onnx_path: Path,
    *,
    inputs: Mapping[str, Any] | None = None,
    # ``dummy`` is the historic one-input test seam.  Keep it until every
    # downstream caller has moved to the explicit multi-input protocol used
    # by v12.  A caller must choose one representation, never both.
    dummy: Any | None = None,
    output_names: Sequence[str],
    expected_outputs: Sequence[Any],
    config: "UnifiedReaderConfig | None" = None,
) -> None:
    """Require the exported graph to load and match Torch on fixed input(s)."""
    if inputs is None:
        if dummy is None:
            raise TypeError("_validate_exported_onnx requires inputs or the legacy dummy argument")
        inputs = {"field_images": dummy}
    elif dummy is not None:
        raise TypeError("_validate_exported_onnx accepts either inputs or dummy, not both")
    onnxruntime = _require_onnxruntime()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected_input_names = list(inputs)
    if [item.name for item in session.get_inputs()] != expected_input_names:
        raise ValueError(
            "Exported unified OCR ONNX has unexpected input names: "
            f"expected={expected_input_names}, actual={[item.name for item in session.get_inputs()]}"
        )
    expected_names = list(output_names)
    if [item.name for item in session.get_outputs()] != expected_names:
        raise ValueError("Exported unified OCR ONNX has unexpected output names")
    actual_outputs = session.run(
        expected_names,
        {name: value.detach().cpu().numpy() for name, value in inputs.items()},
    )
    for name, actual, expected in zip(expected_names, actual_outputs, expected_outputs):
        expected_array = expected.detach().cpu().numpy()
        actual_array = np.asarray(actual)
        if list(actual_array.shape) != list(expected_array.shape):
            raise ValueError(
                f"Exported unified OCR ONNX output {name!r} has shape {list(actual_array.shape)}, "
                f"expected {list(expected_array.shape)}"
            )
        if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
            raise ValueError(f"Exported unified OCR ONNX output {name!r} contains a non-finite value")
        # CPU ORT and CPU Torch can accumulate FP32 drift through exported
        # GRU/normalisation sequences. Every output except the v12 private
        # recipient GRU retains numeric closeness plus exact argmax parity.
        # That one 256-step branch uses exact argmax parity on both export
        # probes and is subsequently proven by full delivery-ONNX evaluation.
        expected64 = expected_array.astype(np.float64, copy=False)
        actual64 = actual_array.astype(np.float64, copy=False)
        absolute_error = np.abs(actual64 - expected64)
        relative_error = absolute_error / np.maximum(np.abs(expected64), 1e-6)
        decision_positions = int(np.prod(actual_array.shape[:-1])) if actual_array.ndim > 1 else 1
        argmax_mismatches = int(
            np.count_nonzero(np.argmax(actual_array, axis=-1) != np.argmax(expected_array, axis=-1))
        )
        atol = _onnx_export_atol(name, config=config)
        max_abs = float(absolute_error.max())
        mean_abs = float(absolute_error.mean())
        requires_raw_logit_close = _onnx_export_requires_raw_logit_close(name, config=config)
        max_abs_cap = _onnx_export_max_abs_cap(name, config=config)
        mean_abs_cap = _onnx_export_mean_abs_cap(name, config=config)
        if (
            (
                requires_raw_logit_close
                and (
                    not np.allclose(
                        actual_array,
                        expected_array,
                        rtol=ONNX_EXPORT_RTOL,
                        atol=atol,
                    )
                    or (max_abs_cap is not None and max_abs > max_abs_cap)
                    or (mean_abs_cap is not None and mean_abs > mean_abs_cap)
                )
            )
            or argmax_mismatches
        ):
            max_abs_cap_text = f", max_abs_cap={max_abs_cap:g}" if max_abs_cap is not None else ""
            mean_abs_cap_text = f", mean_abs_cap={mean_abs_cap:g}" if mean_abs_cap is not None else ""
            parity_policy = "raw_logit_close_and_argmax" if requires_raw_logit_close else "exact_argmax_only"
            raise ValueError(
                f"Exported unified OCR ONNX output {name!r} differs from Torch beyond "
                f"parity_policy={parity_policy}, rtol={ONNX_EXPORT_RTOL:g}, atol={atol:g}"
                f"{max_abs_cap_text}{mean_abs_cap_text} or changes its argmax: "
                f"max_abs={max_abs:.8g}, "
                f"mean_abs={mean_abs:.8g}, "
                f"max_rel={float(relative_error.max()):.8g}, "
                f"argmax_mismatches={argmax_mismatches}/{decision_positions}. "
                "Keep the checkpoint and report these values; do not retrain before resolving export parity."
            )


def export_unified_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
    amount_format_min_confidence: float | None = None,
) -> tuple[Path, Path, Path]:
    """Export a static one-receipt ONNX graph plus labels and a delivery contract.

    ``amount_format_min_confidence`` is a v8-v12 *bundle* override.  It
    changes the finite amount-display renderer policy recorded in the newly
    exported labels/contract, never the checkpoint or an existing ONNX
    artifact.  The neural graph is unchanged because this threshold is a
    decoder policy, not a learned parameter.
    """
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
    if _has_analysis_only_full_crop_continuation_lineage(payload):
        raise ValueError(
            "analysis-only full-crop continuation checkpoints cannot be exported to ONNX"
        )
    config = _checkpoint_config(payload)
    if amount_format_min_confidence is not None:
        # Validate an export override before creating any output.  It is
        # intentionally unsupported for historical architectures so their
        # already-published decoding protocols remain byte-for-byte
        # compatible.
        try:
            configured_threshold = float(amount_format_min_confidence)
        except (TypeError, ValueError):
            raise ValueError("amount_format_min_confidence must be between 0 and 1") from None
        if not math.isfinite(configured_threshold) or not 0.0 <= configured_threshold <= 1.0:
            raise ValueError("amount_format_min_confidence must be between 0 and 1")
        if not _uses_v8_protocol(config):
            raise ValueError("amount_format_min_confidence export override is supported only by v8-v12 checkpoints")
        config = replace(config, amount_format_min_confidence=configured_threshold)
    (
        amount_characters,
        time_characters,
        payment_characters,
        recipient_characters,
        status_classes,
        payment_bank_prefix_classes,
    ) = _checkpoint_labels(payload, config=config)
    status_text_characters = _checkpoint_status_text_characters(payload, config=config)
    recipient_artifact_metadata = (
        _recipient_artifact_metadata(
            config,
            recipient_sampling_policy=payload.get("recipient_sampling_policy"),
            recipient_confidence_policy=payload.get("recipient_confidence_policy"),
            recipient_tail_loss_policy=payload.get("recipient_tail_loss_policy"),
            recipient_train_augmentation_policy=payload.get("recipient_train_augmentation_policy"),
        )
        if recipient_characters is not None
        else {}
    )
    state_dict = payload.get("state_dict")
    field_counts = payload.get("field_counts")
    status_counts = payload.get("status_class_counts")
    if not isinstance(state_dict, Mapping) or not isinstance(field_counts, Mapping) or not isinstance(status_counts, Mapping):
        raise ValueError("Unified OCR checkpoint is missing state_dict or audit counts")
    status_policy = _status_policy_from_counts(status_counts, source="Unified OCR checkpoint")
    if _is_v13(config):
        status_policy = {
            **status_policy,
            "delivery_allowed": False,
            "runtime_policy": "review_only",
            "reason": (
                "v13 supersedes the legacy finite status classifier with visible-text CTC; "
                "legacy status_logits remain ABI-only and must not be exposed as a candidate"
            ),
        }
    model = build_unified_reader(
        payment_vocab_size=len(payment_characters) + 1,
        config=config,
        payment_bank_prefix_vocab_size=(
            len(payment_bank_prefix_classes) if payment_bank_prefix_classes is not None else None
        ),
        recipient_vocab_size=(len(recipient_characters) + 1 if recipient_characters is not None else None),
        status_text_vocab_size=(
            len(status_text_characters) + 1 if status_text_characters is not None else None
        ),
    )
    model.load_state_dict(state_dict)
    model.eval()

    class OneReceiptExport(nn.Module):
        def __init__(self, reader: Any) -> None:
            super().__init__()
            self.reader = reader

        def forward(self, field_images: Any, recipient_value_image: Any | None = None) -> tuple[Any, ...]:
            # ONNX input is one receipt in architecture-specific fixed field
            # order: v3-v8 use [4,1,H,W], v9-v12 use [5,1,H,W].  v12 has
            # one additional NCHW recipient value-view input, but still one
            # reader/session/run.
            outputs = _unpack_reader_outputs(
                self.reader(
                    field_images.unsqueeze(0),
                    recipient_value_image if _uses_high_resolution_recipient_input(config) else None,
                ),
                config=config,
            )
            payment = outputs["payment_logits"]
            status = outputs["status_logits"]
            # Separate amount/time outputs keep the .NET decoder simple while
            # still invoking only one model/session/run.
            base = (
                outputs["amount_logits"][:, 0, :],
                outputs["time_logits"][:, 0, :],
                payment[:, 0, :],
                status[0, :],
            )
            if _uses_v8_protocol(config):
                v8_outputs = base + (
                    outputs["amount_currency_style_logits"][0, :],
                    outputs["amount_grouped_thousands_logits"][0, :],
                    outputs["amount_sign_position_logits"][0, :],
                    outputs["time_format_logits"][0, :],
                    outputs["time_digit_logits"][0, :, :],
                    outputs["payment_prefix_logits"][:, 0, :],
                    outputs["payment_bank_prefix_logits"][0, :],
                    outputs["payment_tail_digit_logits"][0, :, :],
                    outputs["payment_structure_logits"][0, :],
                    outputs["payment_parentheses_logits"][0, :],
                )
                if _uses_recipient_protocol(config):
                    recipient_outputs = v8_outputs + (outputs["recipient_logits"][:, 0, :],)
                    if _is_v13(config):
                        return recipient_outputs + (outputs["status_text_logits"][:, 0, :],)
                    return recipient_outputs
                return v8_outputs
            if _uses_v6_protocol(config):
                return base + (
                    outputs["amount_sign_logits"][0, :],
                    outputs["amount_length_logits"][0, :],
                    outputs["amount_digit_logits"][0, :, :],
                    outputs["time_format_logits"][0, :],
                    outputs["time_digit_logits"][0, :, :],
                    outputs["payment_prefix_logits"][:, 0, :],
                    outputs["payment_bank_prefix_logits"][0, :],
                    outputs["payment_tail_digit_logits"][0, :, :],
                    outputs["payment_structure_logits"][0, :],
                    outputs["payment_parentheses_logits"][0, :],
                )
            if config.architecture_version != 5:
                return base
            return base + (
                outputs["amount_length_logits"][0, :],
                outputs["amount_digit_logits"][0, :, :],
                outputs["time_digit_logits"][0, :, :],
                outputs["time_hour_width_logits"][0, :],
                outputs["payment_prefix_logits"][:, 0, :],
                outputs["payment_tail_digit_logits"][0, :, :],
                outputs["payment_structure_logits"][0, :],
                outputs["payment_parentheses_logits"][0, :],
            )

    wrapper = OneReceiptExport(model)
    output_names = list(_onnx_output_names(config))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_dummy = torch.zeros(
        (len(_slot_order(config)), 1, config.image_height, config.image_width), dtype=torch.float32
    )
    if _uses_high_resolution_recipient_input(config):
        recipient_dummy = torch.zeros(
            (1, 1, config.recipient_input_height, config.recipient_input_width), dtype=torch.float32
        )
        export_args: Any = (field_dummy, recipient_dummy)
        export_input_names = ["field_images", "recipient_value_image"]
        export_inputs = {"field_images": field_dummy, "recipient_value_image": recipient_dummy}
    else:
        export_args = field_dummy
        export_input_names = ["field_images"]
        export_inputs = {"field_images": field_dummy}
    try:
        try:
            torch.onnx.export(
                wrapper,
                export_args,
                temporary_output,
                input_names=export_input_names,
                output_names=output_names,
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:  # Older PyTorch has no dynamo argument.
            torch.onnx.export(
                wrapper,
                export_args,
                temporary_output,
                input_names=export_input_names,
                output_names=output_names,
                opset_version=17,
                do_constant_folding=True,
            )
        with torch.no_grad():
            exported_outputs = wrapper(*export_args) if isinstance(export_args, tuple) else wrapper(export_args)
        _validate_exported_onnx(
            temporary_output,
            inputs=export_inputs,
            output_names=output_names,
            expected_outputs=exported_outputs,
            config=config,
        )
        if _uses_high_resolution_recipient_input(config):
            # The zero tensor above establishes the static two-input ABI.  A
            # second non-blank probe makes the recipient-head decision parity
            # check meaningful for the production high-resolution branch.
            recipient_probe = _v12_recipient_export_probe(recipient_dummy, torch=torch)
            probe_inputs = {
                "field_images": field_dummy,
                "recipient_value_image": recipient_probe,
            }
            with torch.no_grad():
                probe_outputs = wrapper(field_dummy, recipient_probe)
            _validate_exported_onnx(
                temporary_output,
                inputs=probe_inputs,
                output_names=output_names,
                expected_outputs=probe_outputs,
                config=config,
            )
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    output_values = dict(zip(output_names, exported_outputs))
    amount_logits = output_values["amount_logits"]
    time_logits = output_values["time_logits"]
    payment_logits = output_values["payment_logits"]
    status_logits = output_values["status_logits"]
    labels_payload = {
        "schema_version": SCHEMA_VERSION,
        "amount_blank_index": NUMERIC_BLANK_INDEX,
        "amount_characters": amount_characters,
        "time_blank_index": NUMERIC_BLANK_INDEX,
        "time_characters": time_characters,
        "payment_blank_index": PAYMENT_BLANK_INDEX,
        "payment_characters": payment_characters,
        "status_classes": status_classes,
        **(
            {
                "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
                "status_text_characters": status_text_characters,
                "status_text_charset_sha256": hashlib.sha256(
                    "".join(status_text_characters).encode("utf-8")
                ).hexdigest(),
                "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
                "status_text_target": STATUS_TEXT_TARGET,
                "status_text_oov_by_split": payload.get("status_text_oov_by_split"),
                "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
            }
            if status_text_characters is not None
            else {}
        ),
        "checkpoint_selection_policy": payload.get("checkpoint_selection_policy"),
        "initialization": payload.get("initialization"),
        "payment_charset_sha256": hashlib.sha256("".join(payment_characters).encode("utf-8")).hexdigest(),
        **(
            {
                "recipient_blank_index": RECIPIENT_BLANK_INDEX,
                "recipient_characters": recipient_characters,
                "recipient_charset_sha256": hashlib.sha256(
                    "".join(recipient_characters).encode("utf-8")
                ).hexdigest(),
                "recipient_charset_source": payload.get("recipient_charset_source"),
                "recipient_target": payload.get("recipient_target"),
                "recipient_oov_by_split": payload.get("recipient_oov_by_split"),
                **recipient_artifact_metadata,
            }
            if recipient_characters is not None
            else {}
        ),
        **(
            {
                "payment_bank_prefix_classes": payment_bank_prefix_classes,
                "payment_bank_prefix_min_support": payload.get("payment_bank_prefix_min_support"),
                "payment_bank_prefix_class_counts": payload.get("payment_bank_prefix_class_counts"),
                "payment_bank_prefix_train_class_counts": payload.get("payment_bank_prefix_train_class_counts"),
                "payment_bank_prefix_oov_by_split": payload.get("payment_bank_prefix_oov_by_split"),
            }
            if payment_bank_prefix_classes is not None
            else {"numeric_blank_index": NUMERIC_BLANK_INDEX, "numeric_characters": amount_characters}
        ),
    }
    if config.architecture_version == 5:
        labels_payload["structured_decoder"] = {
            "schema_version": 1,
            "amount_max_integer_digits": AMOUNT_MAX_INTEGER_DIGITS,
            "amount_digit_slots": AMOUNT_DIGIT_SLOTS,
            "time_digit_slots": TIME_DIGIT_SLOTS,
            "payment_tail_digit_slots": PAYMENT_TAIL_DIGIT_SLOTS,
            "payment_structure_classes": list(PAYMENT_STRUCTURE_CLASSES),
            "payment_parentheses_classes": list(PAYMENT_PARENTHESIS_CLASSES),
            "time_display_policy": "preserve_h_mm_or_hh_mm_via_hour_width_logits",
            "payment_card_rendering": "prefix + predicted_visible_parentheses_style + exact_four_ascii_digits",
        }
    elif _uses_v8_protocol(config):
        labels_payload["structured_decoder"] = {
            "schema_version": 1,
            "amount_visible_format": AMOUNT_VISIBLE_FORMAT_V8,
            "amount_currency_style_classes": list(AMOUNT_CURRENCY_STYLE_CLASSES),
            "amount_grouped_thousands_classes": list(AMOUNT_GROUPED_THOUSANDS_CLASSES),
            "amount_sign_position_classes": list(AMOUNT_SIGN_POSITION_CLASSES),
            "amount_format_min_confidence": config.amount_format_min_confidence,
            "amount_rendering": "canonical_amount_ctc + finite_display_grammar_only_when_all_components_confident",
            "time_digit_slots": TIME_DISPLAY_DIGIT_SLOTS,
            "time_display_format_classes": list(TIME_DISPLAY_FORMAT_CLASSES),
            "time_visible_format": TIME_DISPLAY_AUX_FORMAT,
            "payment_tail_digit_slots": PAYMENT_TAIL_DIGIT_SLOTS,
            "payment_structure_classes": list(PAYMENT_STRUCTURE_CLASSES),
            "payment_parentheses_classes": list(PAYMENT_PARENTHESIS_CLASSES),
            "payment_bank_prefix_format": PAYMENT_BANK_PREFIX_FORMAT,
            "payment_bank_prefix_other_class": PAYMENT_BANK_OTHER_CLASS,
            "payment_card_rendering": "bank_prefix_class_or_review + predicted_visible_parentheses_style + exact_four_ascii_digits",
        }
    elif _uses_v6_protocol(config):
        labels_payload["structured_decoder"] = {
            "schema_version": 1,
            "amount_max_integer_digits": AMOUNT_MAX_INTEGER_DIGITS,
            "amount_digit_slots": AMOUNT_DIGIT_SLOTS,
            "amount_sign_classes": list(AMOUNT_SIGN_CLASSES),
            "amount_visible_format": AMOUNT_DISPLAY_AUX_FORMAT,
            "time_digit_slots": TIME_DISPLAY_DIGIT_SLOTS,
            "time_display_format_classes": list(TIME_DISPLAY_FORMAT_CLASSES),
            "time_visible_format": TIME_DISPLAY_AUX_FORMAT,
            "payment_tail_digit_slots": PAYMENT_TAIL_DIGIT_SLOTS,
            "payment_structure_classes": list(PAYMENT_STRUCTURE_CLASSES),
            "payment_parentheses_classes": list(PAYMENT_PARENTHESIS_CLASSES),
            "payment_bank_prefix_format": PAYMENT_BANK_PREFIX_FORMAT,
            "payment_bank_prefix_other_class": PAYMENT_BANK_OTHER_CLASS,
            "payment_card_rendering": "bank_prefix_class_or_review + predicted_visible_parentheses_style + exact_four_ascii_digits",
        }
    _atomic_write_json(labels_path, labels_payload)
    output_contract: dict[str, object] = {
        "amount_logits": {
            "shape": list(amount_logits.shape),
            "layout": "[time,class]",
            "decoder": "ctc_greedy",
            "blank_index": NUMERIC_BLANK_INDEX,
            "characters": "amount_characters",
            "target": (
                "visible_cny_amount"
                if _uses_v6_protocol(config)
                else "canonical_amount_for_guarded_display_rendering"
                if _uses_v8_protocol(config)
                else "canonical_amount"
            ),
        },
        "time_logits": {
            "shape": list(time_logits.shape),
            "layout": "[time,class]",
            "decoder": "ctc_greedy",
            "blank_index": NUMERIC_BLANK_INDEX,
            "characters": "time_characters",
            "target": "visible_clock_or_datetime" if _uses_modern_protocol(config) else "canonical_time",
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
            "runtime_policy": status_policy["runtime_policy"],
            "review_value": "review" if status_policy["runtime_policy"] == "review_only" else None,
        },
    }
    if _uses_recipient_protocol(config):
        if recipient_characters is None:
            raise AssertionError("v9/v10/v11/v12 export requires a train-only recipient charset")
        output_contract["recipient_logits"] = {
            "shape": list(output_values["recipient_logits"].shape),
            "layout": "[time,class]",
            "decoder": "ctc_greedy",
            "blank_index": RECIPIENT_BLANK_INDEX,
            "characters": "recipient_characters",
            "target": _recipient_target_mode(config),
            "runtime_policy": "review_only",
            "input_preprocess": _recipient_input_preprocess(config),
            **(
                {
                    "left_trim_fraction": config.recipient_value_left_trim,
                    "horizontal_alignment": "center",
                }
                if _is_v11(config) or _uses_v12_recipient_topology(config)
                else {}
            ),
            **(
                {"input_name": "recipient_value_image"}
                if _uses_high_resolution_recipient_input(config)
                else {}
            ),
        }
    if _is_v13(config):
        if status_text_characters is None:
            raise AssertionError("v13 export requires a train-only status-text charset")
        output_contract["status_text_logits"] = {
            "shape": list(output_values["status_text_logits"].shape),
            "layout": "[time,class]",
            "decoder": "ctc_greedy",
            "blank_index": STATUS_TEXT_BLANK_INDEX,
            "characters": "status_text_characters",
            "target": STATUS_TEXT_TARGET,
            "runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
            "review_value": "review",
            "normalizer": "normalize_status",
        }
    if config.architecture_version == 5:
        output_contract.update(
            {
                "amount_length_logits": {
                    "shape": list(output_values["amount_length_logits"].shape),
                    "layout": "[integer_length_class]",
                    "classes": list(range(1, AMOUNT_MAX_INTEGER_DIGITS + 1)),
                    "target": "integer_digit_count",
                },
                "amount_digit_logits": {
                    "shape": list(output_values["amount_digit_logits"].shape),
                    "layout": "[right_aligned_integer_slots_plus_cents,digit]",
                    "digits": "0-9",
                    "target": "right_aligned_decimal_digits",
                },
                "time_digit_logits": {
                    "shape": list(output_values["time_digit_logits"].shape),
                    "layout": "[HHMM_position,digit]",
                    "digits": "0-9",
                    "target": "canonical_zero_padded_hhmm",
                },
                "time_hour_width_logits": {
                    "shape": list(output_values["time_hour_width_logits"].shape),
                    "layout": "[hour_width_class]",
                    "classes": [1, 2],
                    "target": "visible_hour_digit_width",
                },
                "payment_prefix_logits": {
                    "shape": list(output_values["payment_prefix_logits"].shape),
                    "layout": "[time,class]",
                    "decoder": "ctc_greedy",
                    "blank_index": PAYMENT_BLANK_INDEX,
                    "characters": "payment_characters",
                    "target": "visible_payment_prefix_before_card_tail",
                },
                "payment_tail_digit_logits": {
                    "shape": list(output_values["payment_tail_digit_logits"].shape),
                    "layout": "[tail_position,digit]",
                    "digits": "0-9",
                    "target": "exact_four_card_tail_digits",
                },
                "payment_structure_logits": {
                    "shape": list(output_values["payment_structure_logits"].shape),
                    "layout": "[class]",
                    "classes": list(PAYMENT_STRUCTURE_CLASSES),
                    "target": "payment_card_tail_format",
                },
                "payment_parentheses_logits": {
                    "shape": list(output_values["payment_parentheses_logits"].shape),
                    "layout": "[class]",
                    "classes": list(PAYMENT_PARENTHESIS_CLASSES),
                    "target": "visible_card_tail_parentheses_style",
                },
            }
        )
    elif _uses_v8_protocol(config):
        if payment_bank_prefix_classes is None:
            raise AssertionError("v8-v12 export requires payment bank-prefix classes")
        output_contract.update(
            {
                "amount_currency_style_logits": {
                    "shape": list(output_values["amount_currency_style_logits"].shape),
                    "layout": "[class]",
                    "classes": list(AMOUNT_CURRENCY_STYLE_CLASSES),
                    "target": "visible_amount_currency_symbol_and_optional_space",
                },
                "amount_grouped_thousands_logits": {
                    "shape": list(output_values["amount_grouped_thousands_logits"].shape),
                    "layout": "[class]",
                    "classes": list(AMOUNT_GROUPED_THOUSANDS_CLASSES),
                    "target": "visible_amount_thousands_grouping",
                },
                "amount_sign_position_logits": {
                    "shape": list(output_values["amount_sign_position_logits"].shape),
                    "layout": "[class]",
                    "classes": list(AMOUNT_SIGN_POSITION_CLASSES),
                    "target": "visible_amount_negative_sign_position",
                },
                "time_format_logits": {
                    "shape": list(output_values["time_format_logits"].shape),
                    "layout": "[format_class]",
                    "classes": list(TIME_DISPLAY_FORMAT_CLASSES),
                    "target": "visible_clock_or_datetime_template",
                },
                "time_digit_logits": {
                    "shape": list(output_values["time_digit_logits"].shape),
                    "layout": "[YYYYMMDDHHMMSS_position,digit]",
                    "digits": "0-9",
                    "target": "visible_time_template_digits",
                },
                "payment_prefix_logits": {
                    "shape": list(output_values["payment_prefix_logits"].shape),
                    "layout": "[time,class]",
                    "decoder": "ctc_greedy",
                    "blank_index": PAYMENT_BLANK_INDEX,
                    "characters": "payment_characters",
                    "target": "visible_payment_prefix_before_card_tail",
                },
                "payment_bank_prefix_logits": {
                    "shape": list(output_values["payment_bank_prefix_logits"].shape),
                    "layout": "[bank_prefix_class]",
                    "classes": "payment_bank_prefix_classes",
                    "other_class": PAYMENT_BANK_OTHER_CLASS,
                    "target": "train_only_known_bank_prefix_or_other",
                },
                "payment_tail_digit_logits": {
                    "shape": list(output_values["payment_tail_digit_logits"].shape),
                    "layout": "[tail_position,digit]",
                    "digits": "0-9",
                    "target": "exact_four_card_tail_digits",
                },
                "payment_structure_logits": {
                    "shape": list(output_values["payment_structure_logits"].shape),
                    "layout": "[class]",
                    "classes": list(PAYMENT_STRUCTURE_CLASSES),
                    "target": "payment_card_tail_format",
                },
                "payment_parentheses_logits": {
                    "shape": list(output_values["payment_parentheses_logits"].shape),
                    "layout": "[class]",
                    "classes": list(PAYMENT_PARENTHESIS_CLASSES),
                    "target": "visible_card_tail_parentheses_style",
                },
            }
        )
    elif _uses_v6_protocol(config):
        if payment_bank_prefix_classes is None:
            raise AssertionError("v6 export requires payment bank-prefix classes")
        output_contract.update(
            {
                "amount_sign_logits": {
                    "shape": list(output_values["amount_sign_logits"].shape),
                    "layout": "[class]",
                    "classes": list(AMOUNT_SIGN_CLASSES),
                    "target": "canonical_amount_sign",
                },
                "amount_length_logits": {
                    "shape": list(output_values["amount_length_logits"].shape),
                    "layout": "[integer_length_class]",
                    "classes": list(range(1, AMOUNT_MAX_INTEGER_DIGITS + 1)),
                    "target": "canonical_integer_digit_count",
                },
                "amount_digit_logits": {
                    "shape": list(output_values["amount_digit_logits"].shape),
                    "layout": "[right_aligned_integer_slots_plus_cents,digit]",
                    "digits": "0-9",
                    "target": "canonical_signed_decimal_digits_without_sign",
                },
                "time_format_logits": {
                    "shape": list(output_values["time_format_logits"].shape),
                    "layout": "[format_class]",
                    "classes": list(TIME_DISPLAY_FORMAT_CLASSES),
                    "target": "visible_clock_or_datetime_template",
                },
                "time_digit_logits": {
                    "shape": list(output_values["time_digit_logits"].shape),
                    "layout": "[YYYYMMDDHHMMSS_position,digit]",
                    "digits": "0-9",
                    "target": "visible_time_template_digits",
                },
                "payment_prefix_logits": {
                    "shape": list(output_values["payment_prefix_logits"].shape),
                    "layout": "[time,class]",
                    "decoder": "ctc_greedy",
                    "blank_index": PAYMENT_BLANK_INDEX,
                    "characters": "payment_characters",
                    "target": "visible_payment_prefix_before_card_tail",
                },
                "payment_bank_prefix_logits": {
                    "shape": list(output_values["payment_bank_prefix_logits"].shape),
                    "layout": "[bank_prefix_class]",
                    "classes": "payment_bank_prefix_classes",
                    "other_class": PAYMENT_BANK_OTHER_CLASS,
                    "target": "train_only_known_bank_prefix_or_other",
                },
                "payment_tail_digit_logits": {
                    "shape": list(output_values["payment_tail_digit_logits"].shape),
                    "layout": "[tail_position,digit]",
                    "digits": "0-9",
                    "target": "exact_four_card_tail_digits",
                },
                "payment_structure_logits": {
                    "shape": list(output_values["payment_structure_logits"].shape),
                    "layout": "[class]",
                    "classes": list(PAYMENT_STRUCTURE_CLASSES),
                    "target": "payment_card_tail_format",
                },
                "payment_parentheses_logits": {
                    "shape": list(output_values["payment_parentheses_logits"].shape),
                    "layout": "[class]",
                    "classes": list(PAYMENT_PARENTHESIS_CLASSES),
                    "target": "visible_card_tail_parentheses_style",
                },
            }
        )
    field_input_contract: dict[str, object] = {
        "name": "field_images",
        "dtype": "float32",
        "shape": [len(_slot_order(config)), 1, config.image_height, config.image_width],
        "preprocess": (
            "RGB crop -> grayscale -> aspect-preserving resize -> white right-aligned letterbox for "
            + (
                "amount/time/payment (right-aligned), recipient fifth slot (white reserved placeholder), status (centered) -> divide by 255.0"
                if _uses_v12_recipient_topology(config)
                else "amount/time/payment (right-aligned), recipient (left-trimmed then centered), status (centered) -> divide by 255.0"
                if _is_v11(config)
                else "amount/time/payment (right-aligned), recipient (centered), status (centered) -> divide by 255.0"
                if _is_v10(config)
                else "amount/time/payment/recipient (right-aligned), status (centered) -> divide by 255.0"
                if _is_v9(config)
                else "amount/time/payment (centered status) -> divide by 255.0"
            )
            if _uses_structured_heads(config)
            else "RGB crop -> grayscale -> aspect-preserving resize -> white centered letterbox -> divide by 255.0"
        ),
        "absent_slot_policy": "white_placeholder_not_decoded; emit review instead",
    }
    recipient_value_input_contract: dict[str, object] | None = None
    if _uses_high_resolution_recipient_input(config):
        recipient_value_input_contract = {
            "name": "recipient_value_image",
            "dtype": "float32",
            "shape": [1, 1, config.recipient_input_height, config.recipient_input_width],
            "preprocess": (
                "recipient field crop -> grayscale -> trim left round(width * "
                f"{config.recipient_value_left_trim:g}) pixels -> aspect-preserving resize -> "
                "white centered letterbox -> divide by 255.0"
            ),
            "absent_slot_policy": "white_placeholder_not_decoded; emit review instead",
        }
    _atomic_write_json(
        contract_path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": _kind_for_config(config),
            "onnx_file": output_path.name,
            "onnx_sha256": _sha256(output_path),
            "labels_file": labels_path.name,
            "labels_sha256": _sha256(labels_path),
            "slot_order": list(_slot_order(config)),
            "status_classes": status_classes,
            "training_field_counts": field_counts,
            "training_status_class_counts": status_counts,
            "training_structured_target_counts": payload.get("structured_target_counts"),
            "checkpoint_selection_policy": payload.get("checkpoint_selection_policy"),
            "training_initialization": payload.get("initialization"),
            "status_head_policy": status_policy,
            **(
                {
                    "status_text_charset_sha256": hashlib.sha256(
                        "".join(status_text_characters).encode("utf-8")
                    ).hexdigest(),
                    "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
                    "status_text_target": STATUS_TEXT_TARGET,
                    "status_text_oov_by_split": payload.get("status_text_oov_by_split"),
                    "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
                }
                if status_text_characters is not None
                else {}
            ),
            "payment_bank_prefix_classes": payment_bank_prefix_classes,
            "payment_bank_prefix_min_support": payload.get("payment_bank_prefix_min_support"),
            "payment_bank_prefix_class_counts": payload.get("payment_bank_prefix_class_counts"),
            "payment_bank_prefix_train_class_counts": payload.get("payment_bank_prefix_train_class_counts"),
            "payment_bank_prefix_oov_by_split": payload.get("payment_bank_prefix_oov_by_split"),
            **(
                {
                    "recipient_charset_source": payload.get("recipient_charset_source"),
                    "recipient_target": payload.get("recipient_target"),
                    "recipient_oov_by_split": payload.get("recipient_oov_by_split"),
                    **recipient_artifact_metadata,
                }
                if recipient_characters is not None
                else {}
            ),
            "text_delivery_policy": (
                {
                    "runtime_policy": _text_delivery_policy(config)[0],
                    "review_value": "review",
                    "reason": _text_delivery_policy(config)[1],
                }
                if _uses_structured_heads(config)
                else None
            ),
            # ``input`` remains for all published one-input bundles. v12
            # additionally records an ordered plural list because its graph
            # intentionally has exactly two static inputs in one ORT run.
            "input": field_input_contract,
            **(
                {"inputs": [field_input_contract, recipient_value_input_contract]}
                if recipient_value_input_contract is not None
                else {}
            ),
            "outputs": output_contract,
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


def _load_onnx_artifact_details(
    model_path: Path,
) -> tuple[UnifiedReaderConfig, list[str], list[str] | None, Mapping[str, Any]]:
    """Load and validate an ONNX bundle, including v9-v12 recipient sidecars.

    This is the internal detailed loader.  Keep :func:`_load_onnx_artifacts`
    below as its historic three-item compatibility wrapper for callers that
    only know about the payment reader.
    """
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    labels_path = model_path.with_suffix(".labels.json")
    contract_path = model_path.with_suffix(".contract.json")
    labels = _load_json_object(labels_path)
    contract = _load_json_object(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") not in SUPPORTED_KINDS:
        raise ValueError("Unified OCR ONNX contract kind/schema is unsupported")
    if contract.get("onnx_sha256") != _sha256(model_path):
        raise ValueError("Unified OCR ONNX SHA-256 does not match its contract")
    if contract.get("labels_file") != labels_path.name or contract.get("labels_sha256") != _sha256(labels_path):
        raise ValueError("Unified OCR label sidecar does not match its contract")
    raw_config = contract.get("model")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Unified OCR ONNX contract has no model config")
    config = _config_from_mapping(
        raw_config,
        artifact_kind=contract.get("kind"),
        source="Unified OCR ONNX contract",
    )
    if contract.get("slot_order") != list(_slot_order(config)):
        raise ValueError("Unified OCR ONNX contract slot order is unsupported")
    payment = labels.get("payment_characters")
    status = labels.get("status_classes")
    if labels.get("schema_version") != SCHEMA_VERSION or labels.get("payment_blank_index") != PAYMENT_BLANK_INDEX:
        raise ValueError("Unified OCR ONNX label sidecar schema or blank index is invalid")
    if not isinstance(payment, list) or not payment or not all(isinstance(item, str) and len(item) == 1 for item in payment):
        raise ValueError("Unified OCR ONNX payment charset is invalid")
    if len(set(payment)) != len(payment):
        raise ValueError("Unified OCR ONNX payment charset has duplicates")
    if labels.get("payment_charset_sha256") != hashlib.sha256("".join(payment).encode("utf-8")).hexdigest():
        raise ValueError("Unified OCR ONNX payment charset SHA-256 is invalid")
    if status != list(STATUS_CLASSES):
        raise ValueError("Unified OCR ONNX status class order is unsupported")
    status_text_characters: list[str] | None = None
    if _is_v13(config):
        raw_status_text = labels.get("status_text_characters")
        if (
            not isinstance(raw_status_text, list)
            or not raw_status_text
            or raw_status_text != sorted(raw_status_text)
            or len(set(raw_status_text)) != len(raw_status_text)
            or not all(
                isinstance(character, str)
                and len(character) == 1
                and character.isprintable()
                for character in raw_status_text
            )
            or labels.get("status_text_blank_index") != STATUS_TEXT_BLANK_INDEX
            or labels.get("status_text_charset_source") != STATUS_TEXT_CHARSET_SOURCE
            or labels.get("status_text_target") != STATUS_TEXT_TARGET
            or labels.get("status_text_runtime_policy") != STATUS_TEXT_RUNTIME_POLICY
        ):
            raise ValueError("Unified v13 OCR status-text label contract is invalid")
        status_text_sha256 = hashlib.sha256("".join(raw_status_text).encode("utf-8")).hexdigest()
        if (
            labels.get("status_text_charset_sha256") != status_text_sha256
            or contract.get("status_text_charset_sha256") != status_text_sha256
            or contract.get("status_text_charset_source") != STATUS_TEXT_CHARSET_SOURCE
            or contract.get("status_text_target") != STATUS_TEXT_TARGET
            or contract.get("status_text_runtime_policy") != STATUS_TEXT_RUNTIME_POLICY
            or contract.get("status_text_oov_by_split") != labels.get("status_text_oov_by_split")
        ):
            raise ValueError("Unified v13 OCR status-text labels and contract differ")
        _validate_status_text_oov_audit(
            labels.get("status_text_oov_by_split"),
            source="Unified v13 OCR label sidecar",
        )
        status_text_characters = list(raw_status_text)
    payment_bank_prefix_classes: list[str] | None = None
    recipient_characters: list[str] | None = None
    if _uses_modern_protocol(config):
        expected_amount_characters = V8_AMOUNT_CHARACTERS if _uses_v8_protocol(config) else V6_AMOUNT_CHARACTERS
        if (
            labels.get("amount_blank_index") != NUMERIC_BLANK_INDEX
            or labels.get("time_blank_index") != NUMERIC_BLANK_INDEX
            or labels.get("amount_characters") != list(expected_amount_characters)
            or labels.get("time_characters") != list(V6_TIME_CHARACTERS)
        ):
            raise ValueError("Unified v6-v12 OCR amount/time charset or blank index is unsupported")
        bank_classes = labels.get("payment_bank_prefix_classes")
        if (
            not isinstance(bank_classes, list)
            or len(bank_classes) < 2
            or bank_classes[0] != PAYMENT_BANK_OTHER_CLASS
            or not all(isinstance(value, str) and value for value in bank_classes)
            or len(set(bank_classes)) != len(bank_classes)
            or bank_classes[1:] != sorted(bank_classes[1:])
        ):
            raise ValueError("Unified v6-v12 OCR bank-prefix label map is invalid")
        if contract.get("payment_bank_prefix_classes") != bank_classes:
            raise ValueError("Unified v6-v12 OCR bank-prefix classes differ between labels and contract")
        payment_bank_prefix_classes = list(bank_classes)
        if _uses_recipient_protocol(config):
            raw_recipient = labels.get("recipient_characters")
            if (
                not isinstance(raw_recipient, list)
                or not raw_recipient
                or not all(
                    isinstance(character, str)
                    and len(character) == 1
                    and character.isprintable()
                    for character in raw_recipient
                )
                or len(set(raw_recipient)) != len(raw_recipient)
                or raw_recipient != sorted(raw_recipient)
                or labels.get("recipient_blank_index") != RECIPIENT_BLANK_INDEX
            ):
                raise ValueError("Unified v9-v12 OCR recipient charset or blank index is invalid")
            recipient_sha256 = hashlib.sha256("".join(raw_recipient).encode("utf-8")).hexdigest()
            if labels.get("recipient_charset_sha256") != recipient_sha256:
                raise ValueError("Unified v9-v12 OCR recipient charset SHA-256 is invalid")
            expected_recipient_charset_source = _recipient_charset_source(config)
            if labels.get("recipient_charset_source") != expected_recipient_charset_source:
                raise ValueError("Unified v9-v12 OCR recipient charset source is unsupported")
            if labels.get("recipient_target") != _recipient_target_mode(config):
                raise ValueError("Unified v9-v12 OCR recipient target contract is invalid")
            recipient_oov_by_split = labels.get("recipient_oov_by_split")
            if not isinstance(recipient_oov_by_split, Mapping) or set(recipient_oov_by_split) != {
                "train",
                "val",
                "test",
            }:
                raise ValueError("Unified v9-v12 OCR recipient OOV audit is invalid")
            for split in ("train", "val", "test"):
                audit = recipient_oov_by_split[split]
                if (
                    not isinstance(audit, Mapping)
                    or set(audit) != {"records", "oov_records"}
                    or isinstance(audit.get("records"), bool)
                    or not isinstance(audit.get("records"), int)
                    or isinstance(audit.get("oov_records"), bool)
                    or not isinstance(audit.get("oov_records"), int)
                    or audit["records"] < 0
                    or audit["oov_records"] < 0
                    or audit["oov_records"] > audit["records"]
                ):
                    raise ValueError("Unified v9-v12 OCR recipient OOV audit is invalid")
            if recipient_oov_by_split["train"]["oov_records"] != 0:
                raise ValueError("Unified v9-v12 OCR recipient train split must not contain OOV characters")
            if contract.get("recipient_charset_source") != labels.get("recipient_charset_source"):
                raise ValueError("Unified v9-v12 OCR recipient charset source differs between labels and contract")
            if contract.get("recipient_target") != labels.get("recipient_target"):
                raise ValueError("Unified v9-v12 OCR recipient target differs between labels and contract")
            if contract.get("recipient_oov_by_split") != labels.get("recipient_oov_by_split"):
                raise ValueError("Unified v9-v12 OCR recipient OOV audit differs between labels and contract")
            if _is_v11(config) or _uses_v12_recipient_topology(config):
                expected_recipient_metadata = _recipient_artifact_metadata(
                    config,
                    recipient_sampling_policy=labels.get("recipient_sampling_policy"),
                    recipient_confidence_policy=labels.get("recipient_confidence_policy"),
                    recipient_tail_loss_policy=labels.get("recipient_tail_loss_policy"),
                    recipient_train_augmentation_policy=labels.get("recipient_train_augmentation_policy"),
                )
                for key, expected_value in expected_recipient_metadata.items():
                    if labels.get(key) != expected_value:
                        raise ValueError(
                            f"Unified v{config.architecture_version} OCR label sidecar {key} is invalid"
                        )
                    if contract.get(key) != expected_value:
                        raise ValueError(
                            f"Unified v{config.architecture_version} OCR contract {key} differs from labels"
                        )
            recipient_characters = list(raw_recipient)
    else:
        if labels.get("numeric_blank_index") != NUMERIC_BLANK_INDEX or labels.get("numeric_characters") != list(
            NUMERIC_CHARACTERS
        ):
            raise ValueError("Unified OCR ONNX numeric charset is unsupported")
    if config.architecture_version == 5:
        structured_decoder = labels.get("structured_decoder")
        if not isinstance(structured_decoder, Mapping):
            raise ValueError("Unified v5 OCR label sidecar has no structured decoder contract")
        if (
            structured_decoder.get("schema_version") != 1
            or structured_decoder.get("amount_max_integer_digits") != AMOUNT_MAX_INTEGER_DIGITS
            or structured_decoder.get("amount_digit_slots") != AMOUNT_DIGIT_SLOTS
            or structured_decoder.get("time_digit_slots") != TIME_DIGIT_SLOTS
            or structured_decoder.get("payment_tail_digit_slots") != PAYMENT_TAIL_DIGIT_SLOTS
            or structured_decoder.get("payment_structure_classes") != list(PAYMENT_STRUCTURE_CLASSES)
            or structured_decoder.get("payment_parentheses_classes") != list(PAYMENT_PARENTHESIS_CLASSES)
        ):
            raise ValueError("Unified v5 OCR structured decoder sidecar is unsupported")
        # Older v5 artifacts did not record this policy.  The loader applies
        # the same strict review-only fallback to them, so a stale sidecar
        # cannot silently enable automatic financial-text delivery.
        raw_text_delivery_policy = contract.get("text_delivery_policy")
        if raw_text_delivery_policy is not None:
            if not isinstance(raw_text_delivery_policy, Mapping):
                raise ValueError("Unified v5 OCR text_delivery_policy is invalid")
            if raw_text_delivery_policy.get("runtime_policy") != V5_TEXT_DELIVERY_POLICY:
                raise ValueError("Unified v5 OCR text_delivery_policy must remain review-only")
            if raw_text_delivery_policy.get("review_value") != "review":
                raise ValueError("Unified v5 OCR text_delivery_policy review value is invalid")
    elif _uses_v8_protocol(config):
        structured_decoder = labels.get("structured_decoder")
        if not isinstance(structured_decoder, Mapping):
            raise ValueError("Unified v8-v12 OCR label sidecar has no structured decoder contract")
        try:
            amount_format_min_confidence = float(structured_decoder.get("amount_format_min_confidence"))
        except (TypeError, ValueError):
            raise ValueError("Unified v8-v12 OCR amount format confidence is invalid") from None
        if (
            structured_decoder.get("schema_version") != 1
            or structured_decoder.get("amount_visible_format") != AMOUNT_VISIBLE_FORMAT_V8
            or structured_decoder.get("amount_currency_style_classes") != list(AMOUNT_CURRENCY_STYLE_CLASSES)
            or structured_decoder.get("amount_grouped_thousands_classes") != list(AMOUNT_GROUPED_THOUSANDS_CLASSES)
            or structured_decoder.get("amount_sign_position_classes") != list(AMOUNT_SIGN_POSITION_CLASSES)
            or not math.isclose(
                amount_format_min_confidence,
                config.amount_format_min_confidence,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or structured_decoder.get("amount_rendering")
            != "canonical_amount_ctc + finite_display_grammar_only_when_all_components_confident"
            or structured_decoder.get("time_digit_slots") != TIME_DISPLAY_DIGIT_SLOTS
            or structured_decoder.get("time_display_format_classes") != list(TIME_DISPLAY_FORMAT_CLASSES)
            or structured_decoder.get("time_visible_format") != TIME_DISPLAY_AUX_FORMAT
            or structured_decoder.get("payment_tail_digit_slots") != PAYMENT_TAIL_DIGIT_SLOTS
            or structured_decoder.get("payment_structure_classes") != list(PAYMENT_STRUCTURE_CLASSES)
            or structured_decoder.get("payment_parentheses_classes") != list(PAYMENT_PARENTHESIS_CLASSES)
            or structured_decoder.get("payment_bank_prefix_format") != PAYMENT_BANK_PREFIX_FORMAT
            or structured_decoder.get("payment_bank_prefix_other_class") != PAYMENT_BANK_OTHER_CLASS
        ):
            raise ValueError("Unified v8-v12 OCR structured decoder sidecar is unsupported")
        raw_text_delivery_policy = contract.get("text_delivery_policy")
        if not isinstance(raw_text_delivery_policy, Mapping):
            raise ValueError("Unified v8-v12 OCR text_delivery_policy is missing")
        expected_text_delivery_policy, _ = _text_delivery_policy(config)
        if raw_text_delivery_policy.get("runtime_policy") != expected_text_delivery_policy:
            raise ValueError("Unified v8-v12 OCR text_delivery_policy is unsupported")
        if raw_text_delivery_policy.get("review_value") != "review":
            raise ValueError("Unified v8-v12 OCR text_delivery_policy review value is invalid")
    elif _uses_v6_protocol(config):
        structured_decoder = labels.get("structured_decoder")
        if not isinstance(structured_decoder, Mapping):
            raise ValueError("Unified v6/v7 OCR label sidecar has no structured decoder contract")
        if (
            structured_decoder.get("schema_version") != 1
            or structured_decoder.get("amount_max_integer_digits") != AMOUNT_MAX_INTEGER_DIGITS
            or structured_decoder.get("amount_digit_slots") != AMOUNT_DIGIT_SLOTS
            or structured_decoder.get("amount_sign_classes") != list(AMOUNT_SIGN_CLASSES)
            or structured_decoder.get("amount_visible_format") != AMOUNT_DISPLAY_AUX_FORMAT
            or structured_decoder.get("time_digit_slots") != TIME_DISPLAY_DIGIT_SLOTS
            or structured_decoder.get("time_display_format_classes") != list(TIME_DISPLAY_FORMAT_CLASSES)
            or structured_decoder.get("time_visible_format") != TIME_DISPLAY_AUX_FORMAT
            or structured_decoder.get("payment_tail_digit_slots") != PAYMENT_TAIL_DIGIT_SLOTS
            or structured_decoder.get("payment_structure_classes") != list(PAYMENT_STRUCTURE_CLASSES)
            or structured_decoder.get("payment_parentheses_classes") != list(PAYMENT_PARENTHESIS_CLASSES)
            or structured_decoder.get("payment_bank_prefix_format") != PAYMENT_BANK_PREFIX_FORMAT
            or structured_decoder.get("payment_bank_prefix_other_class") != PAYMENT_BANK_OTHER_CLASS
        ):
            raise ValueError("Unified v6/v7 OCR structured decoder sidecar is unsupported")
        raw_text_delivery_policy = contract.get("text_delivery_policy")
        if not isinstance(raw_text_delivery_policy, Mapping):
            raise ValueError("Unified v6/v7 OCR text_delivery_policy is missing")
        expected_text_delivery_policy, _ = _text_delivery_policy(config)
        if raw_text_delivery_policy.get("runtime_policy") != expected_text_delivery_policy:
            raise ValueError("Unified v6/v7 OCR text_delivery_policy must remain review-only")
        if raw_text_delivery_policy.get("review_value") != "review":
            raise ValueError("Unified v6/v7 OCR text_delivery_policy review value is invalid")
    # v3 did not record a policy; the helper derives a conservative fallback
    # from its audit counts instead of trusting raw logits.
    status_policy = _contract_status_policy(contract)
    raw_input = contract.get("input")
    raw_inputs = contract.get("inputs")
    outputs = contract.get("outputs")
    if not isinstance(raw_input, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("Unified OCR ONNX contract input/output schema is missing")
    expected_input = [len(_slot_order(config)), 1, config.image_height, config.image_width]
    if (
        raw_input.get("name") != "field_images"
        or raw_input.get("dtype") != "float32"
        or raw_input.get("shape") != expected_input
    ):
        raise ValueError(
            f"Unified OCR ONNX input must be static [{len(_slot_order(config))},1,H,W]"
        )
    if _uses_high_resolution_recipient_input(config):
        if not isinstance(raw_inputs, list) or len(raw_inputs) != 2 or raw_inputs[0] != raw_input:
            raise ValueError("Unified v12 OCR contract must declare its two ordered static inputs")
        recipient_input = raw_inputs[1]
        expected_recipient_input = [1, 1, config.recipient_input_height, config.recipient_input_width]
        if (
            not isinstance(recipient_input, Mapping)
            or recipient_input.get("name") != "recipient_value_image"
            or recipient_input.get("dtype") != "float32"
            or recipient_input.get("shape") != expected_recipient_input
            or recipient_input.get("absent_slot_policy")
            != "white_placeholder_not_decoded; emit review instead"
        ):
            raise ValueError("Unified v12 OCR recipient value input contract is unsupported")
    elif raw_inputs is not None:
        raise ValueError("Only v12 unified OCR artifacts may declare multiple ONNX inputs")
    expected_outputs = set(_onnx_output_names(config))
    if set(outputs) != expected_outputs:
        raise ValueError("Unified OCR ONNX output names are unsupported")
    time_steps = config.image_width // 4
    expected_shapes = {
        "amount_logits": [time_steps, len(_amount_characters(config)) + 1],
        "time_logits": [time_steps, len(_time_characters(config)) + 1],
        "payment_logits": [time_steps, len(payment) + 1],
        "status_logits": [len(STATUS_CLASSES)],
    }
    if config.architecture_version == 5:
        expected_shapes.update(
            {
                "amount_length_logits": [AMOUNT_MAX_INTEGER_DIGITS],
                "amount_digit_logits": [AMOUNT_DIGIT_SLOTS, 10],
                "time_digit_logits": [TIME_DIGIT_SLOTS, 10],
                "time_hour_width_logits": [2],
                "payment_prefix_logits": [time_steps, len(payment) + 1],
                "payment_tail_digit_logits": [PAYMENT_TAIL_DIGIT_SLOTS, 10],
                "payment_structure_logits": [len(PAYMENT_STRUCTURE_CLASSES)],
                "payment_parentheses_logits": [len(PAYMENT_PARENTHESIS_CLASSES)],
            }
        )
    elif _uses_v8_protocol(config):
        assert payment_bank_prefix_classes is not None
        expected_shapes.update(
            {
                "amount_currency_style_logits": [len(AMOUNT_CURRENCY_STYLE_CLASSES)],
                "amount_grouped_thousands_logits": [len(AMOUNT_GROUPED_THOUSANDS_CLASSES)],
                "amount_sign_position_logits": [len(AMOUNT_SIGN_POSITION_CLASSES)],
                "time_format_logits": [len(TIME_DISPLAY_FORMAT_CLASSES)],
                "time_digit_logits": [TIME_DISPLAY_DIGIT_SLOTS, 10],
                "payment_prefix_logits": [time_steps, len(payment) + 1],
                "payment_bank_prefix_logits": [len(payment_bank_prefix_classes)],
                "payment_tail_digit_logits": [PAYMENT_TAIL_DIGIT_SLOTS, 10],
                "payment_structure_logits": [len(PAYMENT_STRUCTURE_CLASSES)],
                "payment_parentheses_logits": [len(PAYMENT_PARENTHESIS_CLASSES)],
            }
        )
        if _uses_recipient_protocol(config):
            if recipient_characters is None:
                raise AssertionError("v9-v12 recipient characters were validated above")
            expected_shapes["recipient_logits"] = [
                _recipient_time_steps(config),
                len(recipient_characters) + 1,
            ]
        if _is_v13(config):
            if status_text_characters is None:
                raise AssertionError("v13 status-text characters were validated above")
            expected_shapes["status_text_logits"] = [
                time_steps,
                len(status_text_characters) + 1,
            ]
    elif _uses_v6_protocol(config):
        assert payment_bank_prefix_classes is not None
        expected_shapes.update(
            {
                "amount_sign_logits": [len(AMOUNT_SIGN_CLASSES)],
                "amount_length_logits": [AMOUNT_MAX_INTEGER_DIGITS],
                "amount_digit_logits": [AMOUNT_DIGIT_SLOTS, 10],
                "time_format_logits": [len(TIME_DISPLAY_FORMAT_CLASSES)],
                "time_digit_logits": [TIME_DISPLAY_DIGIT_SLOTS, 10],
                "payment_prefix_logits": [time_steps, len(payment) + 1],
                "payment_bank_prefix_logits": [len(payment_bank_prefix_classes)],
                "payment_tail_digit_logits": [PAYMENT_TAIL_DIGIT_SLOTS, 10],
                "payment_structure_logits": [len(PAYMENT_STRUCTURE_CLASSES)],
                "payment_parentheses_logits": [len(PAYMENT_PARENTHESIS_CLASSES)],
            }
        )
    for name, expected_shape in expected_shapes.items():
        output = outputs[name]
        if not isinstance(output, Mapping) or output.get("shape") != expected_shape:
            raise ValueError(f"Unified OCR ONNX output {name!r} has an invalid static shape")
        expected_blank_index = CTC_ONNX_BLANK_INDICES.get(name)
        if expected_blank_index is None:
            if "blank_index" in output:
                raise ValueError(f"Unified OCR ONNX structured output {name!r} must not declare a CTC blank index")
        elif output.get("blank_index") != expected_blank_index:
            raise ValueError(f"Unified OCR ONNX output {name!r} has an invalid CTC blank index")
    if _uses_recipient_protocol(config):
        recipient_output = outputs["recipient_logits"]
        if (
            recipient_output.get("characters") != "recipient_characters"
            or recipient_output.get("target") != _recipient_target_mode(config)
            or recipient_output.get("runtime_policy") != "review_only"
        ):
            raise ValueError("Unified v9-v12 OCR recipient output contract is unsupported")
        if _is_v11(config) or _uses_v12_recipient_topology(config):
            try:
                left_trim_fraction = float(recipient_output.get("left_trim_fraction"))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Unified v{config.architecture_version} OCR recipient crop contract is invalid"
                ) from None
            if (
                recipient_output.get("input_preprocess") != _recipient_input_preprocess(config)
                or recipient_output.get("horizontal_alignment") != "center"
                or not math.isclose(
                    left_trim_fraction,
                    config.recipient_value_left_trim,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"Unified v{config.architecture_version} OCR recipient crop contract is unsupported"
                )
        if _uses_high_resolution_recipient_input(config) and recipient_output.get("input_name") != "recipient_value_image":
            raise ValueError("Unified v12 OCR recipient output must name its high-resolution input")
    status_output = outputs["status_logits"]
    if contract.get("kind") in {
        KIND_V4,
        KIND_V5,
        KIND_V6,
        KIND_V7,
        KIND_V8,
        KIND_V9,
        KIND_V10,
        KIND_V11,
        KIND_V12,
        KIND_V13,
    }:
        if status_output.get("runtime_policy") != status_policy["runtime_policy"]:
            raise ValueError("Unified OCR ONNX status output policy differs from status_head_policy")
        expected_review = "review" if status_policy["runtime_policy"] == "review_only" else None
        if status_output.get("review_value") != expected_review:
            raise ValueError("Unified OCR ONNX status output review value is invalid")
    if _is_v13(config):
        status_text_output = outputs["status_text_logits"]
        if (
            status_text_output.get("layout") != "[time,class]"
            or status_text_output.get("normalizer") != "normalize_status"
            or status_text_output.get("characters") != "status_text_characters"
            or status_text_output.get("target") != STATUS_TEXT_TARGET
            or status_text_output.get("runtime_policy") != STATUS_TEXT_RUNTIME_POLICY
            or status_text_output.get("review_value") != "review"
            or status_text_output.get("decoder") != "ctc_greedy"
        ):
            raise ValueError("Unified v13 OCR status-text output contract is unsupported")
    return config, list(payment), recipient_characters, contract


def _load_onnx_artifacts(
    model_path: Path,
) -> tuple[UnifiedReaderConfig, list[str], Mapping[str, Any]]:
    """Load an ONNX bundle using the historic three-item return contract.

    The recipient charset is meaningful only to the v9-v12 evaluator.
    Preserving this wrapper prevents old v3-v8 integrations that import this
    private helper from failing solely because a five-slot artifact appended a
    recipient OCR field.
    """
    config, payment_characters, _, contract = _load_onnx_artifact_details(model_path)
    return config, payment_characters, contract


def _contract_status_policy(contract: Mapping[str, Any]) -> dict[str, object]:
    """Return an explicit status policy, including conservative v3 fallback."""
    raw_policy = contract.get("status_head_policy")
    if raw_policy is None:
        return _status_policy_from_counts(
            contract.get("training_status_class_counts"),
            source="Unified OCR ONNX contract",
        )
    if not isinstance(raw_policy, Mapping):
        raise ValueError("Unified OCR ONNX status_head_policy is invalid")
    derived = _status_policy_from_counts(
        contract.get("training_status_class_counts"),
        source="Unified OCR ONNX contract",
    )
    # Counts, rather than a manually edited string in a sidecar, determine
    # whether a decision may be delivered.  A stale policy can only make the
    # artifact stricter, never promote it to classify.
    requested_policy = raw_policy.get("runtime_policy")
    if requested_policy not in {"classify", "review_only"}:
        raise ValueError("Unified OCR ONNX status_head_policy runtime_policy is invalid")
    if requested_policy == "classify" and derived["runtime_policy"] != "classify":
        raise ValueError("Unified OCR ONNX status_head_policy overstates status class coverage")
    if requested_policy == "review_only":
        derived = dict(derived)
        derived["delivery_allowed"] = False
        derived["runtime_policy"] = "review_only"
        reason = raw_policy.get("reason")
        if isinstance(reason, str) and reason:
            derived["reason"] = reason
    return derived


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
    if field == "recipient_field":
        return clean_text(text) or None
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


def _delivery_text(
    *,
    architecture_version: int,
    field: str,
    candidate_text: str,
    ctc_text: str | None,
    structured_text: str | None,
) -> str:
    """Return a conservative delivery value without hiding diagnostics.

    v5's CTC and structural outputs remain useful for training/debugging, but
    they are not independent, calibrated financial delivery paths: their
    heads share the same student visual representation and teacher labels.
    Agreement therefore cannot promote a value to a business decision.  Text
    values stay ``review`` until a separate acceptance policy is implemented
    and proven on group-isolated human truth.  Status follows its separate
    coverage policy upstream.
    """
    if field == "transfer_status":
        return "review" if architecture_version >= 13 else candidate_text
    if architecture_version < 5:
        return candidate_text
    # Keep the arguments in the public helper signature: callers still record
    # them in comparisons.jsonl, which is the evidence needed to build the
    # later calibration policy.
    del ctc_text, structured_text
    return "review"


def _comparison_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "records": 0,
            "raw_exact_match": None,
            "ctc_records": 0,
            "ctc_raw_exact_match": None,
            "semantic_exact_match": None,
            "micro_cer": None,
            "oov_reference_rate": None,
            "non_success_to_success": 0,
            "delivery_coverage": None,
            "delivery_exact_match": None,
            "delivery_exact_overall": None,
            "delivery_false_accepts": 0,
        }
    records = len(rows)
    raw_exact = sum(bool(row["raw_exact"]) for row in rows)
    ctc_rows = [
        row
        for row in rows
        if row.get("ctc_reference_text") is not None
        and row.get("ctc_candidate_text") is not None
    ]
    ctc_raw_exact = sum(bool(row.get("ctc_raw_exact")) for row in ctc_rows)
    semantic_rows = [row for row in rows if row["reference_semantic"] is not None]
    semantic_exact = sum(bool(row["semantic_exact"]) for row in semantic_rows)
    edits = sum(int(row["cer_edits"]) for row in rows)
    reference_characters = sum(int(row["reference_characters"]) for row in rows)
    oov = sum(bool(row["reference_has_oov_character"]) for row in rows)
    non_success_to_success = sum(bool(row.get("non_success_to_success", False)) for row in rows)
    delivered_rows = [row for row in rows if str(row.get("delivery_text", row["candidate_text"])) != "review"]
    delivered_exact = sum(bool(row.get("delivery_raw_exact", row["raw_exact"])) for row in delivered_rows)
    return {
        "records": records,
        "raw_exact_matches": raw_exact,
        "raw_exact_match": raw_exact / records,
        "ctc_records": len(ctc_rows),
        "ctc_raw_exact_matches": ctc_raw_exact,
        "ctc_raw_exact_match": ctc_raw_exact / len(ctc_rows) if ctc_rows else None,
        "semantic_exact_matches": semantic_exact,
        "semantic_exact_match": semantic_exact / max(1, len(semantic_rows)),
        "cer_edits": edits,
        "reference_characters": reference_characters,
        "micro_cer": edits / max(1, reference_characters),
        "oov_reference_records": oov,
        "oov_reference_rate": oov / records,
        "non_success_to_success": non_success_to_success,
        "delivery_coverage": len(delivered_rows) / records,
        "delivery_exact_matches": delivered_exact,
        "delivery_exact_match": delivered_exact / max(1, len(delivered_rows)),
        "delivery_exact_overall": delivered_exact / records,
        "delivery_false_accepts": len(delivered_rows) - delivered_exact,
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


def _format_exact_match(value: object) -> str:
    """Format optional held-out metrics without treating no labels as zero."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.2%}"
    return "n/a"


def _unified_acceptance_failures(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    min_amount_exact_match: float | None,
    min_time_exact_match: float | None,
    min_payment_exact_match: float | None,
    min_recipient_exact_match: float | None,
    min_status_exact_match: float | None,
    max_payment_oov_rate: float | None,
    max_recipient_oov_rate: float | None,
    max_non_success_to_success: int | None,
    min_delivery_coverage: float | None,
    min_delivery_exact_match: float | None,
    max_delivery_false_accepts: int | None,
    status_visible_text_ctc: bool = False,
) -> list[str]:
    failures: list[str] = []
    desired = {
        "amount": min_amount_exact_match,
        "time": min_time_exact_match,
        "payment_method_field": min_payment_exact_match,
        "recipient_field": min_recipient_exact_match,
        "transfer_status": min_status_exact_match,
    }
    for field, threshold in desired.items():
        if threshold is None:
            continue
        metric_name = (
            "ctc_raw_exact_match"
            if field == "transfer_status" and status_visible_text_ctc
            else "raw_exact_match"
        )
        observed = metrics[field][metric_name]
        if observed is None:
            failures.append(f"{field}: no held-out reference labels remain for the requested acceptance gate")
        elif float(observed) < threshold:
            failures.append(f"{field}: {metric_name}={float(observed):.4f} < {threshold:.4f}")
    if max_payment_oov_rate is not None and float(metrics["payment_method_field"]["oov_reference_rate"]) > max_payment_oov_rate:
        failures.append(
            "payment_method_field: "
            f"oov_reference_rate={float(metrics['payment_method_field']['oov_reference_rate']):.4f} "
            f"> {max_payment_oov_rate:.4f}"
        )
    if max_recipient_oov_rate is not None:
        recipient_metrics = metrics.get("recipient_field")
        if recipient_metrics is None:
            failures.append("recipient_field: the artifact has no recipient CTC head")
        elif float(recipient_metrics["oov_reference_rate"]) > max_recipient_oov_rate:
            failures.append(
                "recipient_field: "
                f"oov_reference_rate={float(recipient_metrics['oov_reference_rate']):.4f} "
                f"> {max_recipient_oov_rate:.4f}"
            )
    if max_non_success_to_success is not None:
        observed = int(metrics["transfer_status"]["non_success_to_success"])
        if observed > max_non_success_to_success:
            failures.append(
                f"transfer_status: non_success_to_success={observed} > {max_non_success_to_success}"
            )
    # Candidate parity (raw_exact_match above) and safe delivery are different
    # concepts.  In v5 a structural/CTC disagreement deliberately becomes
    # review, so a model cannot pass a deployment gate merely by producing the
    # correct value in an unsafe candidate channel.
    delivery_fields = ["amount", "time", "payment_method_field"]
    if "recipient_field" in metrics:
        delivery_fields.append("recipient_field")
    for field in delivery_fields:
        if min_delivery_coverage is not None:
            observed_coverage = metrics[field]["delivery_coverage"]
            if observed_coverage is None or float(observed_coverage) < min_delivery_coverage:
                rendered = "n/a" if observed_coverage is None else f"{float(observed_coverage):.4f}"
                failures.append(
                    f"{field}: delivery_coverage={rendered} < {min_delivery_coverage:.4f}"
                )
        if min_delivery_exact_match is not None:
            observed_exact = metrics[field]["delivery_exact_match"]
            if observed_exact is None or float(observed_exact) < min_delivery_exact_match:
                rendered = "n/a" if observed_exact is None else f"{float(observed_exact):.4f}"
                failures.append(
                    f"{field}: delivery_exact_match={rendered} < {min_delivery_exact_match:.4f}"
                )
        if max_delivery_false_accepts is not None:
            observed_false_accepts = int(metrics[field]["delivery_false_accepts"])
            if observed_false_accepts > max_delivery_false_accepts:
                failures.append(
                    f"{field}: delivery_false_accepts={observed_false_accepts} > {max_delivery_false_accepts}"
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
    min_recipient_exact_match: float | None = None,
    min_status_exact_match: float | None = None,
    max_payment_oov_rate: float | None = None,
    max_recipient_oov_rate: float | None = None,
    max_non_success_to_success: int | None = None,
    min_delivery_coverage: float | None = None,
    min_delivery_exact_match: float | None = None,
    max_delivery_false_accepts: int | None = None,
    amount_format_min_confidence_override: float | None = None,
    recipient_beam_width: int = 1,
    recipient_beam_token_top_k: int = 24,
    recipient_ngram_order: int = 3,
    recipient_ngram_weight: float = 0.35,
    progress_every: int = 0,
) -> tuple[dict[str, object], list[str]]:
    """Compare one ONNX session run per held-out receipt with teacher labels."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test; train is not an independent teacher-parity evaluation")
    for name, value in (
        ("min_amount_exact_match", min_amount_exact_match),
        ("min_time_exact_match", min_time_exact_match),
        ("min_payment_exact_match", min_payment_exact_match),
        ("min_recipient_exact_match", min_recipient_exact_match),
        ("min_status_exact_match", min_status_exact_match),
        ("max_payment_oov_rate", max_payment_oov_rate),
        ("max_recipient_oov_rate", max_recipient_oov_rate),
        ("min_delivery_coverage", min_delivery_coverage),
        ("min_delivery_exact_match", min_delivery_exact_match),
    ):
        _finite_probability(value, name=name)
    _finite_probability(
        amount_format_min_confidence_override,
        name="amount_format_min_confidence_override",
    )
    for name, value in (
        ("max_non_success_to_success", max_non_success_to_success),
        ("max_delivery_false_accepts", max_delivery_false_accepts),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} cannot be negative")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"evaluation output already contains files: {output_dir}. Choose a new empty directory.")
    # Start from the historic loader so existing integrations and lightweight
    # evaluator fakes keep their original three-value seam.  Five-slot v9-v12
    # models alone need the recipient alphabet, which is read from the
    # validated detailed sidecar below.
    config, payment_characters, contract = _load_onnx_artifacts(model_path)
    recipient_characters: list[str] | None = None
    if _uses_recipient_protocol(config):
        _, _, recipient_characters, _ = _load_onnx_artifact_details(model_path)
    status_text_characters: list[str] | None = None
    if _is_v13(config):
        # The detailed loader above (or this explicit call) validates hash,
        # source, target and output contract before the list is consumed.
        _load_onnx_artifact_details(model_path)
        raw_status_characters = _load_json_object(
            Path(model_path).resolve().with_suffix(".labels.json")
        ).get("status_text_characters")
        if not isinstance(raw_status_characters, list):
            raise AssertionError("v13 status-text labels were validated above")
        status_text_characters = [str(character) for character in raw_status_characters]
    artifact_amount_format_min_confidence = (
        config.amount_format_min_confidence if _uses_v8_protocol(config) else None
    )
    if amount_format_min_confidence_override is not None:
        if not _uses_v8_protocol(config):
            raise ValueError("amount_format_min_confidence_override is supported only by v8-v12 ONNX artifacts")
        # This is deliberately evaluation-only: validate the artifact's
        # persisted sidecar/contract first, then replace the in-memory
        # renderer threshold.  It must never mutate the ONNX bundle or make a
        # diagnostic calibration result look like a delivery artifact.
        config = replace(
            config,
            amount_format_min_confidence=float(amount_format_min_confidence_override),
        )
    status_policy = _contract_status_policy(contract)
    status_delivery_allowed = status_policy["runtime_policy"] == "classify"
    records = load_records(records_path, dataset_root=dataset_root, config=config)
    if recipient_beam_width <= 0 or recipient_beam_token_top_k <= 0:
        raise ValueError("recipient beam width and token top-k must be positive")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    if recipient_beam_width > 1 and not _uses_recipient_protocol(config):
        raise ValueError("recipient n-gram beam decoding requires a v9-v12 recipient model")
    recipient_language_model = None
    if recipient_beam_width > 1:
        recipient_language_model = CharacterNGramLanguageModel.from_texts(
            (
                text
                for record in records
                if record["split"] == "train"
                for text in [_ctc_slot_text(record, "recipient_field", config=config)]
                if text is not None
            ),
            order=recipient_ngram_order,
        )
    evaluation_records = [record for record in records if record["split"] == split]
    if not evaluation_records:
        raise ValueError(f"No {split} receipt records found")
    required_evaluation_fields = ["amount", "time", "payment_method_field"]
    if _uses_recipient_protocol(config):
        required_evaluation_fields.append("recipient_field")
    if _is_v13(config) or status_delivery_allowed or min_status_exact_match is not None or max_non_success_to_success is not None:
        required_evaluation_fields.append("transfer_status")
    for field in required_evaluation_fields:
        if not any(
            (_status_name(record) if field == "transfer_status" else _slot_text(record, field)) is not None
            for record in evaluation_records
        ):
            raise ValueError(f"No {split} labels remain for unified field {field!r}")

    onnxruntime = _require_onnxruntime()
    model_path = model_path.resolve()
    session, active_providers = _create_onnx_session(onnxruntime, model_path, device=device)
    if progress_every:
        print(f"Unified ONNX providers: {', '.join(active_providers)}", flush=True)
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    expected_outputs = list(_onnx_output_names(config))
    expected_input_names = ["field_images"]
    if _uses_high_resolution_recipient_input(config):
        expected_input_names.append("recipient_value_image")
    if input_names != expected_input_names or output_names != expected_outputs:
        raise ValueError(
            "Unified OCR ONNX input/output names differ from its delivery contract: "
            f"inputs={input_names}, outputs={output_names}"
        )
    expected_input_shape = [len(_slot_order(config)), 1, config.image_height, config.image_width]
    actual_input_shape = list(session.get_inputs()[0].shape)
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"Unified OCR ONNX input shape {actual_input_shape} differs from contract {expected_input_shape}"
        )
    if _uses_high_resolution_recipient_input(config):
        expected_recipient_input_shape = [1, 1, config.recipient_input_height, config.recipient_input_width]
        actual_recipient_input_shape = list(session.get_inputs()[1].shape)
        if actual_recipient_input_shape != expected_recipient_input_shape:
            raise ValueError(
                "Unified OCR ONNX recipient input shape "
                f"{actual_recipient_input_shape} differs from contract {expected_recipient_input_shape}"
            )
    expected_output_shapes = {
        "amount_logits": [config.image_width // 4, len(_amount_characters(config)) + 1],
        "time_logits": [config.image_width // 4, len(_time_characters(config)) + 1],
        "payment_logits": [config.image_width // 4, len(payment_characters) + 1],
        "status_logits": [len(STATUS_CLASSES)],
    }
    if config.architecture_version == 5:
        expected_output_shapes.update(
            {
                "amount_length_logits": [AMOUNT_MAX_INTEGER_DIGITS],
                "amount_digit_logits": [AMOUNT_DIGIT_SLOTS, 10],
                "time_digit_logits": [TIME_DIGIT_SLOTS, 10],
                "time_hour_width_logits": [2],
                "payment_prefix_logits": [config.image_width // 4, len(payment_characters) + 1],
                "payment_tail_digit_logits": [PAYMENT_TAIL_DIGIT_SLOTS, 10],
                "payment_structure_logits": [len(PAYMENT_STRUCTURE_CLASSES)],
                "payment_parentheses_logits": [len(PAYMENT_PARENTHESIS_CLASSES)],
            }
        )
    elif _uses_v8_protocol(config):
        raw_bank_classes = contract.get("payment_bank_prefix_classes")
        if not isinstance(raw_bank_classes, list) or len(raw_bank_classes) < 2:
            raise ValueError("Unified v8-v12 OCR contract has no valid payment bank-prefix classes")
        expected_output_shapes.update(
            {
                "amount_currency_style_logits": [len(AMOUNT_CURRENCY_STYLE_CLASSES)],
                "amount_grouped_thousands_logits": [len(AMOUNT_GROUPED_THOUSANDS_CLASSES)],
                "amount_sign_position_logits": [len(AMOUNT_SIGN_POSITION_CLASSES)],
                "time_format_logits": [len(TIME_DISPLAY_FORMAT_CLASSES)],
                "time_digit_logits": [TIME_DISPLAY_DIGIT_SLOTS, 10],
                "payment_prefix_logits": [config.image_width // 4, len(payment_characters) + 1],
                "payment_bank_prefix_logits": [len(raw_bank_classes)],
                "payment_tail_digit_logits": [PAYMENT_TAIL_DIGIT_SLOTS, 10],
                "payment_structure_logits": [len(PAYMENT_STRUCTURE_CLASSES)],
                "payment_parentheses_logits": [len(PAYMENT_PARENTHESIS_CLASSES)],
            }
        )
        if _uses_recipient_protocol(config):
            if recipient_characters is None:
                raise AssertionError("v9-v12 recipient characters were validated with the ONNX sidecar")
            expected_output_shapes["recipient_logits"] = [
                _recipient_time_steps(config),
                len(recipient_characters) + 1,
            ]
        if _is_v13(config):
            if status_text_characters is None:
                raise AssertionError("v13 status-text characters were validated with the ONNX sidecar")
            expected_output_shapes["status_text_logits"] = [
                config.image_width // 4,
                len(status_text_characters) + 1,
            ]
    elif _uses_v6_protocol(config):
        raw_bank_classes = contract.get("payment_bank_prefix_classes")
        if not isinstance(raw_bank_classes, list) or len(raw_bank_classes) < 2:
            raise ValueError("Unified v6 OCR contract has no valid payment bank-prefix classes")
        expected_output_shapes.update(
            {
                "amount_sign_logits": [len(AMOUNT_SIGN_CLASSES)],
                "amount_length_logits": [AMOUNT_MAX_INTEGER_DIGITS],
                "amount_digit_logits": [AMOUNT_DIGIT_SLOTS, 10],
                "time_format_logits": [len(TIME_DISPLAY_FORMAT_CLASSES)],
                "time_digit_logits": [TIME_DISPLAY_DIGIT_SLOTS, 10],
                "payment_prefix_logits": [config.image_width // 4, len(payment_characters) + 1],
                "payment_bank_prefix_logits": [len(raw_bank_classes)],
                "payment_tail_digit_logits": [PAYMENT_TAIL_DIGIT_SLOTS, 10],
                "payment_structure_logits": [len(PAYMENT_STRUCTURE_CLASSES)],
                "payment_parentheses_logits": [len(PAYMENT_PARENTHESIS_CLASSES)],
            }
        )
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
    recipient_character_set = set(recipient_characters or ())
    status_text_character_set = set(status_text_characters or ())
    status_confusion: Counter[str] = Counter()
    status_reference_counts: Counter[str] = Counter()
    total_records = len(evaluation_records)
    for record_number, record in enumerate(evaluation_records, start=1):
        field_images = np.ascontiguousarray(_input_tensor(record, config=config), dtype=np.float32)
        input_feed: dict[str, np.ndarray] = {"field_images": field_images}
        if _uses_high_resolution_recipient_input(config):
            recipient_value_image = np.ascontiguousarray(
                _recipient_value_input_tensor(record, config=config)[np.newaxis, ...],
                dtype=np.float32,
            )
            input_feed["recipient_value_image"] = recipient_value_image
        started = perf_counter()
        runtime_outputs = dict(zip(expected_outputs, session.run(expected_outputs, input_feed)))
        latency_ms = (perf_counter() - started) * 1000.0
        receipt_latencies.append(latency_ms)
        amount_logits = runtime_outputs["amount_logits"]
        time_logits = runtime_outputs["time_logits"]
        payment_logits = runtime_outputs["payment_logits"]
        status_logits = runtime_outputs["status_logits"]
        amount_text, amount_confidence = _ctc_single_output(amount_logits, characters=_amount_characters(config))
        time_text, time_confidence = _ctc_single_output(time_logits, characters=_time_characters(config))
        payment_text, payment_confidence = _ctc_single_output(payment_logits, characters=payment_characters)
        if _uses_recipient_protocol(config):
            if recipient_characters is None:
                raise AssertionError("v9-v12 recipient characters were validated with the ONNX sidecar")
            if recipient_language_model is None:
                recipient_text, recipient_confidence = _ctc_single_output(
                    runtime_outputs["recipient_logits"], characters=recipient_characters
                )
            else:
                recipient_text, recipient_score = decode_ctc_prefix_beam(
                    runtime_outputs["recipient_logits"],
                    characters=recipient_characters,
                    language_model=recipient_language_model,
                    beam_width=recipient_beam_width,
                    token_top_k=recipient_beam_token_top_k,
                    language_model_weight=recipient_ngram_weight,
                )
                recipient_confidence = 0.0
        status_index, status_confidence = _softmax_confidence(status_logits)
        raw_status_text = STATUS_CLASSES[status_index]
        if _is_v13(config):
            if status_text_characters is None:
                raise AssertionError("v13 status-text characters were validated with the ONNX sidecar")
            visible_status_text, visible_status_confidence = _ctc_single_output(
                runtime_outputs["status_text_logits"], characters=status_text_characters
            )
        else:
            visible_status_text, visible_status_confidence = "", 0.0
        ctc_predictions: dict[str, tuple[str, float]] = {
            "amount": (amount_text, amount_confidence),
            "time": (time_text, time_confidence),
            "payment_method_field": (payment_text, payment_confidence),
        }
        if _uses_recipient_protocol(config):
            ctc_predictions["recipient_field"] = (recipient_text, recipient_confidence)
        if _is_v13(config):
            ctc_predictions["transfer_status"] = (
                visible_status_text,
                visible_status_confidence,
            )
        structured_predictions: dict[str, tuple[str | None, float]] = {}
        if config.architecture_version == 5:
            structured_predictions = {
                "amount": _structured_amount_predictions(
                    np.asarray(runtime_outputs["amount_length_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["amount_digit_logits"])[np.newaxis, :, :],
                )[0],
                "time": _structured_time_predictions(
                    np.asarray(runtime_outputs["time_digit_logits"])[np.newaxis, :, :],
                    np.asarray(runtime_outputs["time_hour_width_logits"])[np.newaxis, :],
                )[0],
                "payment_method_field": _structured_payment_predictions(
                    np.asarray(runtime_outputs["payment_prefix_logits"])[:, np.newaxis, :],
                    np.asarray(runtime_outputs["payment_tail_digit_logits"])[np.newaxis, :, :],
                    np.asarray(runtime_outputs["payment_structure_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["payment_parentheses_logits"])[np.newaxis, :],
                    payment_characters=payment_characters,
                )[0],
            }
        elif _uses_v8_protocol(config):
            raw_bank_classes = contract.get("payment_bank_prefix_classes")
            if not isinstance(raw_bank_classes, list):
                raise AssertionError("v8 bank-prefix classes were validated with the output contract")
            structured_predictions = {
                "amount": _structured_amount_v8_predictions(
                    [(amount_text, amount_confidence)],
                    np.asarray(runtime_outputs["amount_currency_style_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["amount_grouped_thousands_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["amount_sign_position_logits"])[np.newaxis, :],
                    min_confidence=config.amount_format_min_confidence,
                )[0],
                "time": _structured_time_v6_predictions(
                    np.asarray(runtime_outputs["time_format_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["time_digit_logits"])[np.newaxis, :, :],
                )[0],
                # This is retained in comparisons.jsonl for diagnostics; the
                # v8 candidate selector intentionally keeps payment CTC until
                # a separately calibrated delivery policy exists.
                "payment_method_field": _structured_payment_v6_predictions(
                    np.asarray(runtime_outputs["payment_bank_prefix_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["payment_tail_digit_logits"])[np.newaxis, :, :],
                    np.asarray(runtime_outputs["payment_structure_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["payment_parentheses_logits"])[np.newaxis, :],
                    payment_bank_prefix_classes=[str(value) for value in raw_bank_classes],
                )[0],
            }
        elif _uses_v6_protocol(config):
            raw_bank_classes = contract.get("payment_bank_prefix_classes")
            if not isinstance(raw_bank_classes, list):
                raise AssertionError("v6 bank-prefix classes were validated with the output contract")
            structured_predictions = {
                "amount": _structured_amount_v6_predictions(
                    np.asarray(runtime_outputs["amount_sign_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["amount_length_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["amount_digit_logits"])[np.newaxis, :, :],
                )[0],
                "time": _structured_time_v6_predictions(
                    np.asarray(runtime_outputs["time_format_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["time_digit_logits"])[np.newaxis, :, :],
                )[0],
                "payment_method_field": _structured_payment_v6_predictions(
                    np.asarray(runtime_outputs["payment_bank_prefix_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["payment_tail_digit_logits"])[np.newaxis, :, :],
                    np.asarray(runtime_outputs["payment_structure_logits"])[np.newaxis, :],
                    np.asarray(runtime_outputs["payment_parentheses_logits"])[np.newaxis, :],
                    payment_bank_prefix_classes=[str(value) for value in raw_bank_classes],
                )[0],
            }
        predictions = _select_report_candidates(
            ctc_predictions,
            structured_predictions,
            config=config,
        )
        if _uses_recipient_protocol(config):
            # An unseen recipient remains open-text CTC evidence.  It never
            # becomes a finite merchant-class decision or a delivered value.
            # v10 recognises the complete visible row, then derives the
            # review value from that same decoded line.
            raw_recipient_text, recipient_confidence = ctc_predictions["recipient_field"]
            predictions["recipient_field"] = (
                _recipient_candidate_value(raw_recipient_text, config=config) or "",
                recipient_confidence,
            )
        # A status head with incomplete classes must never become a business
        # decision, even if its raw argmax says success.
        predictions["transfer_status"] = (
            (
                normalize_status(visible_status_text)
                if _is_v13(config)
                else raw_status_text
                if status_delivery_allowed
                else "review"
            ),
            visible_status_confidence if _is_v13(config) else status_confidence,
        )
        for field in _slot_order(config):
            slot = dict(record["slots"]).get(field)
            if not isinstance(slot, Mapping):
                continue
            if field == "transfer_status":
                reference_text = str(slot["class_name"])
                reference_semantic = reference_text
            elif field == "recipient_field" and _is_v10(config):
                reference_text = _recipient_expected_value(record, config=config)
                if reference_text is None:
                    # The v10 dataset builder rejects label-only lines, but
                    # preserve a defensive skip for malformed external data.
                    continue
                semantic_value = slot.get("semantic_value")
                reference_semantic = (
                    str(semantic_value)
                    if isinstance(semantic_value, str)
                    else _semantic_value(field, reference_text)
                )
            else:
                if _uses_v8_protocol(config) and field == "amount":
                    visible = slot.get("visible_text")
                    reference_text = (
                        visible
                        if isinstance(visible, str) and parse_amount_visible_format_target(visible) is not None
                        else _ctc_slot_text(record, field, config=config) or str(slot["text"])
                    )
                else:
                    reference_text = _ctc_slot_text(record, field, config=config) or str(slot["text"])
                semantic_value = slot.get("semantic_value")
                reference_semantic = str(semantic_value) if isinstance(semantic_value, str) else _semantic_value(field, reference_text)
            candidate_text, confidence = predictions[field]
            ctc_candidate = ctc_predictions.get(field)
            ctc_reference_text = (
                _ctc_slot_text(record, field, config=config)
                if ctc_candidate is not None
                else None
            )
            ctc_raw_exact = (
                ctc_candidate is not None
                and ctc_reference_text is not None
                and ctc_candidate[0] == ctc_reference_text
            )
            structured_candidate = structured_predictions.get(field)
            structured_text = structured_candidate[0] if structured_candidate is not None else None
            structured_confidence = structured_candidate[1] if structured_candidate is not None else None
            decoder_agrees = (
                None
                if structured_text is None or ctc_candidate is None
                else (
                    _semantic_value(field, str(structured_text))
                    == _semantic_value(field, str(ctc_candidate[0]))
                    if _uses_modern_protocol(config) and field in {"amount", "time"}
                    else str(structured_text) == str(ctc_candidate[0])
                )
            )
            # v5 never treats agreement between a structural digit head and
            # the legacy CTC head as a probability calibration. It is merely
            # a conservative first gate: both an explicit disagreement and a
            # missing structural candidate are review. A raw CTC fallback is
            # retained in this report for diagnosis, but has no calibrated
            # delivery threshold yet and therefore cannot be emitted.
            delivery_text = _delivery_text(
                architecture_version=config.architecture_version,
                field=field,
                candidate_text=candidate_text,
                ctc_text=ctc_candidate[0] if ctc_candidate is not None else None,
                structured_text=structured_text,
            )
            candidate_semantic = _semantic_value(field, candidate_text)
            raw_exact = candidate_text == reference_text
            delivery_raw_exact = delivery_text == reference_text
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
                    # v10 recipient CTC deliberately reads the complete
                    # visible crop row, while ``reference_text`` /
                    # ``candidate_text`` compare only the extracted merchant
                    # value. Preserve both views so a review can distinguish
                    # line-recognition failure from post-decode extraction.
                    "ctc_reference_text": ctc_reference_text,
                    "ctc_candidate_text": ctc_candidate[0] if ctc_candidate is not None else None,
                    "ctc_raw_exact": ctc_raw_exact,
                    "structured_candidate_text": structured_text,
                    "structured_confidence": round(float(structured_confidence), 6)
                    if structured_confidence is not None
                    else None,
                    "decoder_agrees": decoder_agrees,
                    "delivery_text": delivery_text,
                    "delivery_raw_exact": delivery_raw_exact,
                    "raw_model_candidate_text": (
                        visible_status_text if field == "transfer_status" and _is_v13(config)
                        else raw_status_text if field == "transfer_status"
                        else None
                    ),
                    "legacy_status_classifier_candidate": (
                        raw_status_text if field == "transfer_status" and _is_v13(config) else None
                    ),
                    "confidence": round(confidence, 6),
                    "runtime_policy": (
                        STATUS_TEXT_RUNTIME_POLICY
                        if field == "transfer_status" and _is_v13(config)
                        else status_policy["runtime_policy"]
                        if field == "transfer_status"
                        else _text_delivery_policy(config)[0]
                        if _uses_structured_heads(config)
                        else "decode"
                    ),
                    "raw_exact": raw_exact,
                    "reference_semantic": reference_semantic,
                    "candidate_semantic": candidate_semantic,
                    "semantic_exact": semantic_exact,
                    "candidate_semantic_valid": candidate_semantic is not None,
                    "cer_edits": levenshtein_distance(reference_text, candidate_text),
                    "reference_characters": len(reference_text),
                    "reference_has_oov_character": (
                        field == "payment_method_field"
                        and bool(set(reference_text) - payment_character_set)
                    )
                    or (
                        field == "recipient_field"
                        and bool(
                            set(
                                _ctc_slot_text(record, "recipient_field", config=config)
                                or reference_text
                            )
                            - recipient_character_set
                        )
                    )
                    or (
                        field == "transfer_status"
                        and _is_v13(config)
                        and bool(
                            set(_slot_text(record, "transfer_status") or "")
                            - status_text_character_set
                        )
                    ),
                    "non_success_to_success": non_success_to_success,
                    "receipt_latency_ms": round(latency_ms, 4),
                }
            )
        if progress_every and (
            record_number == 1
            or record_number == total_records
            or record_number % progress_every == 0
        ):
            print(
                f"Unified ONNX evaluation: {record_number}/{total_records} receipts",
                flush=True,
            )
    comparisons.sort(key=lambda row: (str(row["field"]), str(row["id"])))
    by_field = {
        field: _comparison_metrics([row for row in comparisons if row["field"] == field])
        for field in _slot_order(config)
    }
    failures = _unified_acceptance_failures(
        by_field,
        min_amount_exact_match=min_amount_exact_match,
        min_time_exact_match=min_time_exact_match,
        min_payment_exact_match=min_payment_exact_match,
        min_recipient_exact_match=min_recipient_exact_match,
        min_status_exact_match=min_status_exact_match,
        max_payment_oov_rate=max_payment_oov_rate,
        max_recipient_oov_rate=max_recipient_oov_rate,
        max_non_success_to_success=max_non_success_to_success,
        min_delivery_coverage=min_delivery_coverage,
        min_delivery_exact_match=min_delivery_exact_match,
        max_delivery_false_accepts=max_delivery_false_accepts,
        status_visible_text_ctc=_is_v13(config),
    )
    if (
        (min_status_exact_match is not None or max_non_success_to_success is not None)
        and not status_delivery_allowed
        and not _is_v13(config)
    ):
        failures.append(
            "transfer_status: acceptance was requested, but the artifact is review_only because its status "
            "class coverage is incomplete"
        )
    acceptance_requested = any(
        value is not None
        for value in (
            min_amount_exact_match,
            min_time_exact_match,
            min_payment_exact_match,
            min_recipient_exact_match,
            min_status_exact_match,
            max_payment_oov_rate,
            max_recipient_oov_rate,
            max_non_success_to_success,
            min_delivery_coverage,
            min_delivery_exact_match,
            max_delivery_false_accepts,
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
        "records_sha256": _sha256(records_path),
        "evaluation_split": split,
        "label_sources": label_sources,
        "providers": active_providers,
        "slot_order": list(_slot_order(config)),
        "amount_format_policy": {
            "artifact_min_confidence": artifact_amount_format_min_confidence,
            "effective_min_confidence": config.amount_format_min_confidence if _uses_v8_protocol(config) else None,
            "evaluation_override": amount_format_min_confidence_override,
        },
        "recipient_decoder_policy": {
            "mode": "character_ngram_ctc_prefix_beam_v1" if recipient_language_model is not None else "ctc_greedy",
            "beam_width": recipient_beam_width,
            "token_top_k": recipient_beam_token_top_k,
            "ngram_order": recipient_ngram_order if recipient_language_model is not None else None,
            "ngram_weight": recipient_ngram_weight if recipient_language_model is not None else None,
            "open_vocabulary": True,
            "complete_name_lexicon": False,
        },
        "by_field": by_field,
        "status_confusion": dict(sorted(status_confusion.items())),
        "status_reference_class_counts": {
            class_name: int(status_reference_counts[class_name]) for class_name in STATUS_CLASSES
        },
        "status_head_policy": status_policy,
        "status_text_policy": (
            {
                "target": STATUS_TEXT_TARGET,
                "runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
                "review_value": "review",
                "characters": len(status_text_characters or ()),
                "charset_source": STATUS_TEXT_CHARSET_SOURCE,
            }
            if _is_v13(config)
            else None
        ),
        "receipt_latency_ms": _latency_metrics(receipt_latencies),
        "acceptance": {
            "min_amount_exact_match": min_amount_exact_match,
            "min_time_exact_match": min_time_exact_match,
            "min_payment_exact_match": min_payment_exact_match,
            "min_recipient_exact_match": min_recipient_exact_match,
            "min_status_exact_match": min_status_exact_match,
            "max_payment_oov_rate": max_payment_oov_rate,
            "max_recipient_oov_rate": max_recipient_oov_rate,
            "max_non_success_to_success": max_non_success_to_success,
            "min_delivery_coverage": min_delivery_coverage,
            "min_delivery_exact_match": min_delivery_exact_match,
            "max_delivery_false_accepts": max_delivery_false_accepts,
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


def _recipient_audit_trim_ratios(values: Sequence[float]) -> tuple[float, ...]:
    """Validate and canonicalise an exploratory v11 recipient trim sweep."""
    if isinstance(values, (str, bytes)):
        raise ValueError("left_trim_ratios must be a sequence of numeric ratios")
    ratios: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("left_trim_ratios must contain numeric ratios")
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            raise ValueError("left_trim_ratios must contain numeric ratios") from None
        if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
            raise ValueError("left_trim_ratios must be in [0, 1)")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("left_trim_ratios must not be empty")
    return tuple(sorted(set(ratios)))


def _recipient_audit_preprocess_gray(
    gray: np.ndarray,
    *,
    config: UnifiedReaderConfig,
    trim_px: int,
) -> np.ndarray:
    """Reproduce a v11/v12 recipient value-view transform from loaded pixels.

    The audit loads each source crop once, then produces several candidate
    value views in memory.  Its resize, centring, and trim arithmetic exactly
    mirrors :func:`preprocess_image`, including v12's independent
    high-resolution input dimensions.  It is intentionally private so the
    delivery preprocessor remains the sole public runtime contract.
    """
    if not (_is_v11(config) or _uses_v12_recipient_topology(config)):
        raise ValueError("recipient audit preprocessor supports only architecture v11 or v12")
    source = np.asarray(gray, dtype=np.uint8)
    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("recipient crop must be a non-empty grayscale image")
    width = int(source.shape[1])
    if trim_px < 0 or trim_px >= width:
        raise ValueError("recipient trim pixel is outside the source crop")
    image = Image.fromarray(source, mode="L").crop((trim_px, 0, width, int(source.shape[0])))
    output_height = (
        config.recipient_input_height if _uses_high_resolution_recipient_input(config) else config.image_height
    )
    output_width = (
        config.recipient_input_width if _uses_high_resolution_recipient_input(config) else config.image_width
    )
    scale = min(output_width / image.width, output_height / image.height)
    resized_width = max(1, min(output_width, int(round(image.width * scale))))
    resized_height = max(1, min(output_height, int(round(image.height * scale))))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((resized_width, resized_height), resampling)
    canvas = np.full((output_height, output_width), 255, dtype=np.uint8)
    top = (output_height - resized_height) // 2
    left = (output_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(image, dtype=np.uint8)
    return (canvas.astype(np.float32) / 255.0)[np.newaxis, :, :]


def _recipient_audit_rendered_width(
    *,
    source_height: int,
    retained_width: int,
    config: UnifiedReaderConfig,
) -> int:
    """Return the non-letterboxed width visible to the audited CTC head."""
    if source_height <= 0 or retained_width <= 0:
        raise ValueError("recipient audit source dimensions must be positive")
    output_height = (
        config.recipient_input_height if _uses_high_resolution_recipient_input(config) else config.image_height
    )
    output_width = (
        config.recipient_input_width if _uses_high_resolution_recipient_input(config) else config.image_width
    )
    scale = min(output_width / retained_width, output_height / source_height)
    return max(1, min(output_width, int(round(retained_width * scale))))


def _recipient_audit_bucket(value: float, *, boundaries: Sequence[float], labels: Sequence[str]) -> str:
    """Assign a numeric diagnostic value to one stable, human-readable bucket."""
    if len(labels) != len(boundaries) + 1:
        raise ValueError("recipient audit bucket labels do not match boundaries")
    for boundary, label in zip(boundaries, labels):
        if value <= boundary:
            return label
    return labels[-1]


def _recipient_audit_bucket_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    bucket_for: Any,
) -> dict[str, dict[str, object]]:
    """Group rows into explainable slices while retaining exact/CER evidence."""
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        bucket = str(bucket_for(row))
        groups.setdefault(bucket, []).append(row)
    return {bucket: _comparison_metrics(groups[bucket]) for bucket in sorted(groups)}


def _recipient_audit_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Add image-geometry and latency evidence to standard CTC comparison metrics."""
    metrics = _comparison_metrics(rows)
    if not rows:
        return {
            **metrics,
            "cut_window_ink_records": 0,
            "cut_window_ink_rate": None,
            "nearest_blank_gap_touch_records": 0,
            "nearest_blank_gap_touch_rate": None,
            "mean_rendered_width_per_character": None,
            "latency_ms": _latency_metrics(()),
        }
    cut_window_ink = 0
    nearest_blank_gap_touch = 0
    rendered_widths_per_character: list[float] = []
    latencies: list[float] = []
    for row in rows:
        geometry = row.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("recipient audit row has no geometry")
        if bool(geometry.get("cut_window_has_ink")):
            cut_window_ink += 1
        raw_gap = geometry.get("nearest_blank_gap")
        if isinstance(raw_gap, Mapping) and bool(raw_gap.get("touches_trim_boundary")):
            nearest_blank_gap_touch += 1
        value = row.get("rendered_width_per_character")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            rendered_widths_per_character.append(float(value))
        latency = row.get("inference_ms")
        if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
            latencies.append(float(latency))
    records = len(rows)
    return {
        **metrics,
        "cut_window_ink_records": cut_window_ink,
        "cut_window_ink_rate": cut_window_ink / records,
        "nearest_blank_gap_touch_records": nearest_blank_gap_touch,
        "nearest_blank_gap_touch_rate": nearest_blank_gap_touch / records,
        "mean_rendered_width_per_character": (
            sum(rendered_widths_per_character) / len(rendered_widths_per_character)
            if rendered_widths_per_character
            else None
        ),
        "latency_ms": _latency_metrics(latencies),
    }


def _recipient_trim_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Build the per-ratio report that decides whether geometry is worth changing."""
    return {
        "metrics": _recipient_audit_metrics(rows),
        "by_reference_length": _recipient_audit_bucket_metrics(
            rows,
            bucket_for=lambda row: _recipient_audit_bucket(
                float(row["reference_length"]),
                boundaries=(4, 8, 12),
                labels=("1-4", "5-8", "9-12", "13+"),
            ),
        ),
        "by_min_train_character_support": _recipient_audit_bucket_metrics(
            rows,
            bucket_for=lambda row: _recipient_audit_bucket(
                float(row["min_train_character_support"]),
                boundaries=(0, 1, 3, 9),
                labels=("0", "1", "2-3", "4-9", "10+"),
            ),
        ),
        "by_rendered_width_per_character": _recipient_audit_bucket_metrics(
            rows,
            bucket_for=lambda row: _recipient_audit_bucket(
                float(row["rendered_width_per_character"]),
                boundaries=(20, 35, 50),
                labels=("<=20", "20-35", "35-50", "50+"),
            ),
        ),
    }


def audit_recipient_trims_onnx(
    *,
    model_path: Path,
    records_path: Path,
    output_dir: Path,
    left_trim_ratios: Sequence[float],
    dataset_root: Path | None = None,
    split: str = "test",
    device: str = "auto",
    foreground_contrast_threshold: int = RECIPIENT_AUDIT_DEFAULT_FOREGROUND_CONTRAST_THRESHOLD,
    cut_radius: int = RECIPIENT_AUDIT_DEFAULT_CUT_RADIUS,
    blank_column_max_ink: int = 0,
    require_high_resolution_recipient_input: bool = False,
) -> dict[str, object]:
    """Diagnose a frozen v11/v12 recipient trim without changing an artifact.

    Every trial uses the same validated ONNX and the same held-out records.
    Only the recipient value view's *in-memory* pixel trim differs.  For v11
    that view is the fifth field slot; for v12 it is the separate static,
    high-resolution ``recipient_value_image`` ONNX input.  The result is
    diagnostic evidence for a possible future preprocessing contract, never a
    rewritten or deployable model.
    """
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test; train is not an independent teacher-parity audit")
    ratios = _recipient_audit_trim_ratios(left_trim_ratios)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"recipient audit output already contains files: {output_dir}. Choose a new empty directory.")

    # Preserve the ordinary evaluator's sidecar-validation seam.  In
    # particular, do not accept a plain ONNX file with a hand-edited labels
    # file, and never write the diagnostic trim into the artifact.
    config, _payment_characters, contract = _load_onnx_artifacts(model_path)
    if not (_is_v11(config) or _uses_v12_recipient_topology(config)):
        raise ValueError(
            "audit-recipient supports only v11 or v12 ONNX artifacts with a frozen recipient value trim"
        )
    if require_high_resolution_recipient_input and not _uses_high_resolution_recipient_input(config):
        raise ValueError(
            "audit-recipient requires a v12 ONNX artifact with a high-resolution recipient_value_image input"
        )
    _, _, recipient_characters, _ = _load_onnx_artifact_details(model_path)
    if recipient_characters is None:
        raise AssertionError("v11/v12 recipient characters were validated with the ONNX sidecar")

    records = load_records(records_path, dataset_root=dataset_root, config=config)
    train_character_support: Counter[str] = Counter()
    for record in records:
        if record["split"] != "train":
            continue
        text = _ctc_slot_text(record, "recipient_field", config=config)
        if text is not None:
            train_character_support.update(text)
    recipient_records: list[tuple[Mapping[str, object], Mapping[str, object], str]] = []
    for record in records:
        if record["split"] != split:
            continue
        slot = dict(record["slots"]).get("recipient_field")
        reference_text = _ctc_slot_text(record, "recipient_field", config=config)
        if isinstance(slot, Mapping) and reference_text is not None:
            recipient_records.append((record, slot, reference_text))
    if not recipient_records:
        raise ValueError(f"No {split} recipient labels remain for the v11/v12 trim audit")

    onnxruntime = _require_onnxruntime()
    model_path = model_path.resolve()
    session, active_providers = _create_onnx_session(onnxruntime, model_path, device=device)
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    expected_outputs = list(_onnx_output_names(config))
    expected_input_names = ["field_images"]
    if _uses_high_resolution_recipient_input(config):
        expected_input_names.append("recipient_value_image")
    if input_names != expected_input_names or output_names != expected_outputs:
        raise ValueError(
            "Unified OCR ONNX input/output names differ from its delivery contract: "
            f"inputs={input_names}, outputs={output_names}"
        )
    expected_input_shape = [len(_slot_order(config)), 1, config.image_height, config.image_width]
    actual_input_shape = list(session.get_inputs()[0].shape)
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"Unified OCR ONNX input shape {actual_input_shape} differs from contract {expected_input_shape}"
        )
    if _uses_high_resolution_recipient_input(config):
        expected_recipient_input_shape = [1, 1, config.recipient_input_height, config.recipient_input_width]
        actual_recipient_input_shape = list(session.get_inputs()[1].shape)
        if actual_recipient_input_shape != expected_recipient_input_shape:
            raise ValueError(
                "Unified OCR ONNX recipient input shape "
                f"{actual_recipient_input_shape} differs from contract {expected_recipient_input_shape}"
            )
    recipient_output = next((item for item in session.get_outputs() if item.name == "recipient_logits"), None)
    expected_recipient_shape = [_recipient_time_steps(config), len(recipient_characters) + 1]
    if recipient_output is None or list(recipient_output.shape) != expected_recipient_shape:
        actual_shape = None if recipient_output is None else list(recipient_output.shape)
        raise ValueError(
            "Unified OCR ONNX recipient output shape differs from contract: "
            f"actual={actual_shape}, expected={expected_recipient_shape}"
        )

    recipient_index = _slot_order(config).index("recipient_field")
    recipient_input_name = (
        "recipient_value_image" if _uses_high_resolution_recipient_input(config) else "field_images[recipient_field]"
    )
    recipient_input_shape = (
        [1, 1, config.recipient_input_height, config.recipient_input_width]
        if _uses_high_resolution_recipient_input(config)
        else [1, config.image_height, config.image_width]
    )
    character_set = set(recipient_characters)
    rows_by_ratio: dict[float, list[dict[str, object]]] = {ratio: [] for ratio in ratios}
    total = len(recipient_records)
    progress_interval = max(1, total // 20)
    for number, (record, slot, reference_text) in enumerate(recipient_records, start=1):
        # `_input_tensor` keeps the financial slots exactly as the delivery
        # preprocessing contract defines them.  Every trial replaces only the
        # dedicated recipient value view in memory: v11's fifth slot or v12's
        # private high-resolution input.  No model file, labels, contract, or
        # delivery evaluator input is changed.
        field_images = np.ascontiguousarray(_input_tensor(record, config=config), dtype=np.float32)
        image_path = Path(slot["image_path"])
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        reference_semantic = _semantic_value("recipient_field", reference_text)
        reference_has_oov = any(character not in character_set for character in reference_text)
        character_support = [int(train_character_support[character]) for character in reference_text]
        minimum_character_support = min(character_support, default=0)
        mean_character_support = sum(character_support) / len(character_support) if character_support else 0.0
        for ratio in ratios:
            geometry = audit_recipient_pixels(
                gray,
                left_trim_ratio=ratio,
                foreground_contrast_threshold=foreground_contrast_threshold,
                cut_radius=cut_radius,
                blank_column_max_ink=blank_column_max_ink,
            )
            recipient_value_view = _recipient_audit_preprocess_gray(
                gray,
                config=config,
                trim_px=geometry.trim_px,
            )
            input_feed: dict[str, np.ndarray] = {"field_images": field_images}
            if _uses_high_resolution_recipient_input(config):
                input_feed["recipient_value_image"] = np.ascontiguousarray(
                    recipient_value_view[np.newaxis, ...], dtype=np.float32
                )
            else:
                field_images[recipient_index] = recipient_value_view
            started = perf_counter()
            recipient_logits = session.run(["recipient_logits"], input_feed)[0]
            inference_ms = (perf_counter() - started) * 1000.0
            candidate_text, confidence = _ctc_single_output(recipient_logits, characters=recipient_characters)
            candidate_semantic = _semantic_value("recipient_field", candidate_text)
            raw_exact = candidate_text == reference_text
            semantic_exact = (
                reference_semantic is not None
                and candidate_semantic is not None
                and candidate_semantic == reference_semantic
            )
            rendered_width = _recipient_audit_rendered_width(
                source_height=geometry.height,
                retained_width=geometry.retained_width_px,
                config=config,
            )
            rows_by_ratio[ratio].append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "receipt_unified_recipient_trim_audit_row_v1",
                    "id": str(record["id"]),
                    "group_id": str(record["group_id"]),
                    "split": split,
                    "source": record.get("source"),
                    "result_json": record.get("result_json"),
                    "label_source": record.get("label_source"),
                    "field": "recipient_field",
                    "image": image_path.as_posix(),
                    "bbox_rectified": slot.get("bbox_rectified"),
                    "crop_sha256": slot.get("crop_sha256"),
                    "paddle_text": slot.get("paddle_text"),
                    "paddle_confidence": slot.get("paddle_confidence"),
                    "detector_score": slot.get("detector_score"),
                    "recipient_label": slot.get("recipient_label"),
                    "recipient_visible_text": slot.get("recipient_visible_text"),
                    "reference_text": reference_text,
                    "candidate_text": candidate_text,
                    "confidence": confidence,
                    "reference_semantic": reference_semantic,
                    "candidate_semantic": candidate_semantic,
                    "raw_exact": raw_exact,
                    "semantic_exact": semantic_exact,
                    "cer_edits": levenshtein_distance(reference_text, candidate_text),
                    "reference_characters": len(reference_text),
                    "reference_has_oov_character": reference_has_oov,
                    "delivery_text": "review",
                    "delivery_raw_exact": False,
                    "non_success_to_success": False,
                    "left_trim_fraction": ratio,
                    "artifact_left_trim_fraction": config.recipient_value_left_trim,
                    "recipient_input_name": recipient_input_name,
                    "recipient_input_shape": recipient_input_shape,
                    "recipient_input_preprocess": _recipient_input_preprocess(config),
                    "reference_length": len(reference_text),
                    "min_train_character_support": minimum_character_support,
                    "mean_train_character_support": mean_character_support,
                    "rendered_width": rendered_width,
                    "rendered_width_per_character": rendered_width / max(1, len(reference_text)),
                    "geometry": geometry.as_dict(),
                    "inference_ms": inference_ms,
                }
            )
        if number == 1 or number == total or number % progress_interval == 0:
            print(
                f"Recipient trim audit: {number}/{total} receipts, {len(ratios)} trim variants each",
                flush=True,
            )

    trial_summaries: list[dict[str, object]] = []
    flattened_rows: list[dict[str, object]] = []
    for ratio in ratios:
        rows = rows_by_ratio[ratio]
        flattened_rows.extend(rows)
        trial_summaries.append(
            {
                "left_trim_fraction": ratio,
                **_recipient_trim_summary(rows),
            }
        )
    best_trial = max(
        trial_summaries,
        key=lambda trial: (
            float(dict(trial["metrics"])["raw_exact_match"]),
            -float(dict(trial["metrics"])["micro_cer"]),
            -abs(float(trial["left_trim_fraction"]) - config.recipient_value_left_trim),
        ),
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_unified_recipient_trim_audit_v1",
        "model": model_path.as_posix(),
        "model_sha256": _sha256(model_path),
        "records": records_path.resolve().as_posix(),
        "evaluation_split": split,
        "providers": active_providers,
        "artifact_kind": contract.get("kind"),
        "architecture_version": config.architecture_version,
        "artifact_left_trim_fraction": config.recipient_value_left_trim,
        "recipient_input_name": recipient_input_name,
        "recipient_input_shape": recipient_input_shape,
        "recipient_input_preprocess": _recipient_input_preprocess(config),
        "requires_high_resolution_recipient_input": require_high_resolution_recipient_input,
        "trial_left_trim_fractions": list(ratios),
        "foreground_contrast_threshold": foreground_contrast_threshold,
        "cut_radius": cut_radius,
        "blank_column_max_ink": blank_column_max_ink,
        "recipient_records": total,
        "train_recipient_character_count": len(train_character_support),
        "trials": trial_summaries,
        "best_strict_exact_trial": {
            "left_trim_fraction": best_trial["left_trim_fraction"],
            "metrics": best_trial["metrics"],
        },
        "warning": (
            f"This is an exploratory in-memory v{config.architecture_version} recipient pixel-preprocessing sweep. "
            "It does not rewrite the ONNX model, labels, contract, deployment trim, checkpoints, or the ordinary "
            "guarded evaluator. It compares with held-out Paddle-derived teacher "
            "labels, not independently verified business truth; use it only to decide whether a new preprocessing "
            "contract is justified before retraining."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / "comparisons.jsonl", flattened_rows)
    # Keep disagreements as JSONL so large audits can be inspected without
    # loading every trial. `_atomic_write_jsonl` also preserves a valid empty
    # file when every candidate happens to match.
    _atomic_write_jsonl(
        output_dir / "disagreements.jsonl",
        [row for row in flattened_rows if not bool(row["raw_exact"])],
    )
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, export, and evaluate one offline ONNX reader for amount/time/status/payment/recipient fields"
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
    train.add_argument(
        "--recipient-loss-weight",
        type=float,
        default=1.0,
        help=(
            "v9-v12 multiplier for the open-text recipient CTC loss; use a controlled value such as 3.0 "
            "for a recipient-focused experiment"
        ),
    )
    train.add_argument(
        "--status-text-loss-weight",
        type=float,
        default=1.0,
        help="v13 multiplier for visible transfer-status CTC loss",
    )
    train.add_argument(
        "--recipient-sampling-weight",
        type=float,
        default=1.0,
        help=(
            "v11/v12 only: relative sampling weight for receipts with an anchored recipient label; "
            "2.0 is the recommended first experiment"
        ),
    )
    train.add_argument(
        "--recipient-rare-character-max-support",
        type=int,
        default=0,
        help=(
            "v11/v12 only: treat recipient characters seen at most this many times in the train split as rare; "
            "0 disables rare-character sampling"
        ),
    )
    train.add_argument(
        "--recipient-rare-character-sampling-weight",
        type=float,
        default=1.0,
        help=(
            "v11/v12 only: bounded receipt sampling weight for a recipient containing a rare train character; "
            "uses max(), never multiplicative stacking"
        ),
    )
    train.add_argument(
        "--recipient-long-text-min-length",
        type=int,
        default=0,
        help=(
            "v11/v12 only: treat a recipient value with at least this many Unicode code points as long; "
            "0 disables long-text sampling"
        ),
    )
    train.add_argument(
        "--recipient-long-text-sampling-weight",
        type=float,
        default=1.0,
        help="v11/v12 only: bounded receipt sampling weight for a long recipient value",
    )
    train.add_argument(
        "--recipient-low-confidence-threshold",
        type=float,
        help=(
            "v11/v12 only: Paddle recipient labels below this confidence receive a lower CTC loss weight; "
            "omit to preserve historical all-one weighting"
        ),
    )
    train.add_argument(
        "--recipient-low-confidence-loss-weight",
        type=float,
        default=1.0,
        help=(
            "v11/v12 only: final CTC loss weight in (0,1] for recipient teacher labels below "
            "--recipient-low-confidence-threshold"
        ),
    )
    train.add_argument(
        "--recipient-confidence-curriculum-epochs",
        type=int,
        default=0,
        help=(
            "v11/v12 only: linearly ramp low-confidence recipient label weighting over this many epochs; "
            "0 applies the configured weight immediately"
        ),
    )
    train.add_argument(
        "--recipient-tail-rare-character-max-support",
        type=int,
        default=0,
        help=(
            "v11/v12 only: recipient CTC loss boost applies when a target contains a character seen at most "
            "this many times in the train split; 0 disables the rare-character tail boost"
        ),
    )
    train.add_argument(
        "--recipient-tail-rare-character-loss-weight",
        type=float,
        default=1.0,
        help=(
            "v11/v12 only: bounded recipient CTC loss weight (at least 1) for a rare-character tail target; "
            "this does not change receipt sampling"
        ),
    )
    train.add_argument(
        "--recipient-tail-long-text-min-length",
        type=int,
        default=0,
        help=(
            "v11/v12 only: recipient CTC loss boost applies to values with at least this many Unicode code points; "
            "0 disables the long-text tail boost"
        ),
    )
    train.add_argument(
        "--recipient-tail-long-text-loss-weight",
        type=float,
        default=1.0,
        help=(
            "v11/v12 only: bounded recipient CTC loss weight (at least 1) for a long-text tail target; "
            "rare and long boosts use max(), never a product"
        ),
    )
    train.add_argument(
        "--recipient-train-augmentation",
        choices=("none", "light_v1", "robust_v2"),
        default="none",
        help=(
            "v12/v13 only: train-only recipient value-crop perturbation; robust_v2 adds bounded downscale and "
            "mild blur to deterministic shifts/contrast/noise without changing the ONNX input contract"
        ),
    )
    train.add_argument(
        "--recipient-train-splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train"],
        help=(
            "v12 recipient-only fine-tune only: manifest splits allowed to supervise recipient CTC. "
            "Default train preserves an independent val metric; adding val is a Paddle-fit/transductive recipe."
        ),
    )
    train.add_argument(
        "--recipient-only-fine-tune",
        action="store_true",
        help=(
            "v12 only: require a compatible warm checkpoint, freeze every non-recipient parameter, and optimize "
            "only the private recipient branch. Whole-receipt oversampling is rejected."
        ),
    )
    train.add_argument(
        "--status-text-only-fine-tune",
        action="store_true",
        help=(
            "v13 only: warm-start a compatible v12/v13 checkpoint, freeze the shared trunk and all 15 "
            "legacy outputs, and optimize only the additive status_text_ CTC head"
        ),
    )
    train.add_argument(
        "--recipient-open-text-unfreeze-legacy",
        action="store_true",
        help=(
            "recipient_open_text_adapter only: jointly fine-tune the legacy private recipient CNN/GRU/classifier "
            "with the new adapter while every financial/shared parameter remains frozen"
        ),
    )
    train.add_argument(
        "--validation-every",
        type=int,
        default=1,
        help=(
            "Run full five-field validation every N epochs, always including epoch 1 and the final epoch. "
            "Values above 1 are restricted to the guarded v12 recipient-only expansion or v13 "
            "status-text-only warm start."
        ),
    )
    train.add_argument(
        "--checkpoint-selection",
        choices=tuple(sorted(CHECKPOINT_SELECTION_MODES)),
        default=CHECKPOINT_SELECTION_BALANCED,
        help=(
            "How to select best.pt. balanced preserves historical behavior. recipient_priority is v9-v12 only "
            "and requires all three protected candidate-exact floors; it changes training selection only, not ONNX ABI."
        ),
    )
    train.add_argument(
        "--checkpoint-min-amount-candidate-exact",
        type=float,
        help="recipient_priority only: same-validation-split amount candidate-exact protection floor",
    )
    train.add_argument(
        "--checkpoint-min-time-candidate-exact",
        type=float,
        help="recipient_priority only: same-validation-split time candidate-exact protection floor",
    )
    train.add_argument(
        "--checkpoint-min-payment-candidate-exact",
        type=float,
        help="recipient_priority only: same-validation-split payment candidate-exact protection floor",
    )
    train.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Optional parameter-only warm start from a compatible best.pt. "
            "The optimizer, epoch counter, sampler, and checkpoint history are always reset."
        ),
    )
    train.add_argument(
        "--init-checkpoint-mode",
        choices=tuple(
            sorted(
                INIT_CHECKPOINT_MODES
                - {INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION}
            )
        ),
        default=INIT_CHECKPOINT_MODE_STRICT,
        help=(
            "strict requires every label map and model config to match the seed. "
            "recipient_only_expansion is v12 recipient-only only: it locks payment/bank maps to the seed and "
            "maps additive recipient Unicode rows by character. recipient_input_width_expansion additionally "
            "permits only a strictly wider v12 recipient input while keeping all learned tensor shapes and "
            "financial/shared topology unchanged. recipient_capacity_reinit copies every financial/shared "
            "tensor but deliberately reinitialises a monotonically larger private recipient CNN/GRU."
            " recipient_open_text_adapter copies the complete seed, adds a zero-gated Transformer context "
            "encoder, and trains only that adapter. recipient_visual_context_reinit is v13-only: it copies "
            "every non-recipient tensor from a v13 seed and freshly trains the residual visual + direct "
            "positional Transformer CTC recipient branch. recipient_full_crop_warmstart is v13-only: it "
            "copies the compatible seed and permits exactly recipient_value_left_trim 0.30 -> 0.0. "
            "recipient_full_crop_continuation requires a content-bound v13 legacy trim-zero authority, "
            "exact config/maps/all-state copy, and fresh training state."
        ),
    )
    train.add_argument(
        "--ctc-loss-weight",
        type=float,
        default=0.35,
        help="v5-v12 auxiliary raw-CTC loss weight; v3/v4 keep their historical loss composition",
    )
    train.add_argument(
        "--structured-loss-weight",
        type=float,
        default=1.0,
        help="v5-v12 structured financial-format and bank-verifier loss weight",
    )
    train.add_argument(
        "--architecture",
        choices=("v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13"),
        default="v8",
        help=(
            "v9 is the five-field value-only reader. v10 learns the complete visible recipient row. v11 keeps one "
            "five-slot ONNX but learns an anchored recipient value from a left-trimmed value view. v12 remains one "
            "ONNX/session but adds a second high-resolution recipient value input. v13 appends visible status "
            "text CTC as output 16 while preserving the v12 inputs and first 15 outputs. v8 remains the compatible "
            "four-field default; v7/v6/v5/v4/v3 remain checkpoint-compatible"
        ),
    )
    train.add_argument(
        "--amount-format-min-confidence",
        type=float,
        default=0.90,
        help=(
            "v8-v12 only: require every amount display-format head to meet this confidence before it can render "
            "currency/sign/thousands punctuation; otherwise retain the raw canonical CTC candidate"
        ),
    )
    train.add_argument(
        "--payment-bank-prefix-min-support",
        type=int,
        default=3,
        help=(
            "v6-v12 only: minimum train-split examples needed to retain a bank-prefix class; "
            "rarer/unknown prefixes map to __other__ and remain review-only"
        ),
    )
    train.add_argument("--image-height", type=int, default=80)
    train.add_argument("--image-width", type=int, default=512)
    train.add_argument("--base-channels", type=int, default=32)
    train.add_argument("--numeric-hidden-size", type=int, default=96)
    train.add_argument("--payment-hidden-size", type=int, default=128)
    train.add_argument(
        "--recipient-hidden-size",
        type=int,
        help="v11/v12 only: recipient CTC branch width; defaults to 192 without widening the shared CNN",
    )
    train.add_argument(
        "--recipient-value-left-trim",
        type=float,
        default=0.30,
        help="v11-v13 only: fraction trimmed from the left of the recipient crop before resize",
    )
    train.add_argument(
        "--recipient-input-height",
        type=int,
        default=128,
        help="v12 only: static high-resolution recipient input height (default: 128)",
    )
    train.add_argument(
        "--recipient-input-width",
        type=int,
        default=1024,
        help="v12 only: static high-resolution recipient input width, divisible by 4 (default: 1024)",
    )
    train.add_argument(
        "--recipient-branch-channels",
        type=int,
        help="v12 only: private high-resolution recipient CNN width; defaults to 16",
    )
    train.add_argument(
        "--recipient-open-text-layers",
        type=int,
        default=0,
        help="v12 only: zero-gated Transformer context layers after the warm-started recipient BiGRU",
    )
    train.add_argument(
        "--recipient-open-text-heads",
        type=int,
        default=8,
        help="v12 open-text adapter attention heads (default: 8)",
    )
    train.add_argument(
        "--recipient-open-text-feedforward",
        type=int,
        help="v12 open-text adapter feed-forward width; defaults to four times its model width",
    )
    train.add_argument(
        "--recipient-open-text-dropout",
        type=float,
        default=0.0,
        help=(
            "v12/v13 private recipient Transformer train-time dropout; 0 preserves historical checkpoints, "
            "a capacity-reinitialised open-text candidate may use a bounded value such as 0.10"
        ),
    )
    train.add_argument(
        "--recipient-backbone",
        choices=("legacy_depthwise_gru_v1", "residual_positional_transformer_v2"),
        default="legacy_depthwise_gru_v1",
        help=(
            "Private recipient recogniser. The v2 residual/positional Transformer path is v13-only and "
            "requires recipient_visual_context_reinit from a compatible v13 seed."
        ),
    )
    train.add_argument("--pooled-width", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 on Windows until the training environment is verified",
    )
    train.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per DataLoader worker; used only when --num-workers is positive",
    )
    train.add_argument(
        "--persistent-workers",
        action="store_true",
        help=(
            "Keep DataLoader workers alive between epochs. v12/v13 recipient augmentation uses a "
            "process-shared epoch counter, so its deterministic per-record perturbations remain epoch-correct."
        ),
    )
    train.add_argument(
        "--train-progress-every",
        type=int,
        default=0,
        help="Print training progress every N batches inside each epoch; 0 disables progress output",
    )
    train.add_argument(
        "--cuda-tf32",
        action="store_true",
        help="Opt in to high-precision TF32 CUDA matmul/convolution kernels (for example RTX 4090)",
    )
    train.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Opt in to cuDNN autotuning for the fixed training input shapes",
    )
    train.add_argument("--onnx-output", type=Path, help="Optionally export best.pt to this new ONNX path")

    export = commands.add_parser("export", help="export a trained unified reader checkpoint")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--amount-format-min-confidence",
        type=float,
        help=(
            "v8-v12 only: write this validated amount display-format confidence gate into a new ONNX bundle; "
            "the checkpoint and existing artifacts are never modified"
        ),
    )

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
    evaluate.add_argument(
        "--amount-format-min-confidence-override",
        type=float,
        help=(
            "v8-v12 evaluation only: temporarily override the amount visible-format confidence gate without "
            "rewriting the ONNX artifact, labels, or contract"
        ),
    )
    evaluate.add_argument("--min-amount-exact-match", type=float)
    evaluate.add_argument("--min-time-exact-match", type=float)
    evaluate.add_argument("--min-payment-exact-match", type=float)
    evaluate.add_argument("--min-recipient-exact-match", type=float)
    evaluate.add_argument("--recipient-beam-width", type=int, default=1)
    evaluate.add_argument("--recipient-beam-token-top-k", type=int, default=24)
    evaluate.add_argument("--recipient-ngram-order", type=int, default=3)
    evaluate.add_argument("--recipient-ngram-weight", type=float, default=0.35)
    evaluate.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print ONNX provider and receipt progress every N evaluated receipts; 0 disables progress output",
    )
    evaluate.add_argument("--min-status-exact-match", type=float)
    evaluate.add_argument("--max-payment-oov-rate", type=float)
    evaluate.add_argument("--max-recipient-oov-rate", type=float)
    evaluate.add_argument("--max-non-success-to-success", type=int)
    evaluate.add_argument(
        "--min-delivery-coverage",
        type=float,
        help="Require this non-review coverage for each v5-v12 text field",
    )
    evaluate.add_argument(
        "--min-delivery-exact-match",
        type=float,
        help="Require this raw exact match among non-review v5-v12 text-field deliveries",
    )
    evaluate.add_argument(
        "--max-delivery-false-accepts",
        type=int,
        help="Maximum incorrect non-review v5-v12 deliveries allowed per text field",
    )

    audit_recipient = commands.add_parser(
        "audit-recipient",
        help="diagnose a frozen v11/v12 recipient value crop with in-memory trim trials",
    )
    audit_recipient.add_argument("--model", type=Path, required=True)
    audit_recipient.add_argument("--records", type=Path, required=True, help="v11/v12 unified_fields.jsonl")
    audit_recipient.add_argument(
        "--dataset-root",
        type=Path,
        help="Root that owns crop paths in the original pseudo-label manifest; defaults to --records directory",
    )
    audit_recipient.add_argument("--output", type=Path, required=True, help="New empty diagnostic output directory")
    audit_recipient.add_argument("--split", choices=("val", "test"), default="test")
    audit_recipient.add_argument("--device", default="auto")
    audit_recipient.add_argument(
        "--left-trims",
        type=float,
        nargs="+",
        required=True,
        help="One or more exploratory v11/v12 left-crop fractions, such as 0 0.20 0.30 0.40",
    )
    audit_recipient.add_argument(
        "--foreground-contrast-threshold",
        type=int,
        default=RECIPIENT_AUDIT_DEFAULT_FOREGROUND_CONTRAST_THRESHOLD,
        help="Image-only ink detection contrast against the dominant grayscale background (default: 24)",
    )
    audit_recipient.add_argument(
        "--cut-radius",
        type=int,
        default=RECIPIENT_AUDIT_DEFAULT_CUT_RADIUS,
        help="Number of source columns inspected on each side of a trim boundary (default: 2)",
    )
    audit_recipient.add_argument(
        "--blank-column-max-ink",
        type=int,
        default=0,
        help="Maximum foreground pixels allowed in a source column treated as blank (default: 0)",
    )
    audit_recipient.add_argument(
        "--require-high-resolution-recipient-input",
        action="store_true",
        help="Reject v11 and require v12's dedicated recipient_value_image input",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            config = UnifiedReaderConfig(
                architecture_version=int(args.architecture.removeprefix("v")),
                image_height=args.image_height,
                image_width=args.image_width,
                base_channels=args.base_channels,
                numeric_hidden_size=args.numeric_hidden_size,
                payment_hidden_size=args.payment_hidden_size,
                recipient_hidden_size=args.recipient_hidden_size,
                recipient_value_left_trim=args.recipient_value_left_trim,
                recipient_input_height=args.recipient_input_height,
                recipient_input_width=args.recipient_input_width,
                recipient_branch_channels=args.recipient_branch_channels,
                recipient_open_text_layers=args.recipient_open_text_layers,
                recipient_open_text_heads=args.recipient_open_text_heads,
                recipient_open_text_feedforward=args.recipient_open_text_feedforward,
                recipient_open_text_dropout=args.recipient_open_text_dropout,
                recipient_backbone=args.recipient_backbone,
                pooled_width=args.pooled_width,
                amount_format_min_confidence=args.amount_format_min_confidence,
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
                recipient_loss_weight=args.recipient_loss_weight,
                status_text_loss_weight=args.status_text_loss_weight,
                recipient_sampling_weight=args.recipient_sampling_weight,
                recipient_rare_character_max_support=args.recipient_rare_character_max_support,
                recipient_rare_character_sampling_weight=args.recipient_rare_character_sampling_weight,
                recipient_long_text_min_length=args.recipient_long_text_min_length,
                recipient_long_text_sampling_weight=args.recipient_long_text_sampling_weight,
                recipient_low_confidence_threshold=args.recipient_low_confidence_threshold,
                recipient_low_confidence_loss_weight=args.recipient_low_confidence_loss_weight,
                recipient_confidence_curriculum_epochs=args.recipient_confidence_curriculum_epochs,
                recipient_tail_rare_character_max_support=args.recipient_tail_rare_character_max_support,
                recipient_tail_rare_character_loss_weight=args.recipient_tail_rare_character_loss_weight,
                recipient_tail_long_text_min_length=args.recipient_tail_long_text_min_length,
                recipient_tail_long_text_loss_weight=args.recipient_tail_long_text_loss_weight,
                recipient_train_augmentation=args.recipient_train_augmentation,
                recipient_train_splits=tuple(args.recipient_train_splits),
                recipient_only_fine_tune=args.recipient_only_fine_tune,
                status_text_only_fine_tune=args.status_text_only_fine_tune,
                recipient_open_text_unfreeze_legacy=args.recipient_open_text_unfreeze_legacy,
                validation_every=args.validation_every,
                checkpoint_selection=args.checkpoint_selection,
                checkpoint_min_amount_candidate_exact=args.checkpoint_min_amount_candidate_exact,
                checkpoint_min_time_candidate_exact=args.checkpoint_min_time_candidate_exact,
                checkpoint_min_payment_candidate_exact=args.checkpoint_min_payment_candidate_exact,
                init_checkpoint=args.init_checkpoint,
                init_checkpoint_mode=args.init_checkpoint_mode,
                ctc_loss_weight=args.ctc_loss_weight,
                structured_loss_weight=args.structured_loss_weight,
                payment_bank_prefix_min_support=args.payment_bank_prefix_min_support,
                seed=args.seed,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                train_progress_every=args.train_progress_every,
                cuda_tf32=args.cuda_tf32,
                cudnn_benchmark=args.cudnn_benchmark,
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
                amount_format_min_confidence=args.amount_format_min_confidence,
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
                min_recipient_exact_match=args.min_recipient_exact_match,
                min_status_exact_match=args.min_status_exact_match,
                max_payment_oov_rate=args.max_payment_oov_rate,
                max_recipient_oov_rate=args.max_recipient_oov_rate,
                max_non_success_to_success=args.max_non_success_to_success,
                min_delivery_coverage=args.min_delivery_coverage,
                min_delivery_exact_match=args.min_delivery_exact_match,
                max_delivery_false_accepts=args.max_delivery_false_accepts,
                amount_format_min_confidence_override=args.amount_format_min_confidence_override,
                recipient_beam_width=args.recipient_beam_width,
                recipient_beam_token_top_k=args.recipient_beam_token_top_k,
                recipient_ngram_order=args.recipient_ngram_order,
                recipient_ngram_weight=args.recipient_ngram_weight,
                progress_every=args.progress_every,
            )
            metrics = summary["by_field"]
            status_policy = summary.get("status_head_policy")
            status_display = (
                _format_exact_match(metrics["transfer_status"]["ctc_raw_exact_match"])
                if summary.get("status_text_policy") is not None
                else "review_only"
                if isinstance(status_policy, Mapping) and status_policy.get("runtime_policy") == "review_only"
                else _format_exact_match(metrics["transfer_status"]["raw_exact_match"])
            )
            recipient_display = (
                f", recipient={_format_exact_match(metrics['recipient_field']['raw_exact_match'])}"
                if "recipient_field" in metrics
                else ""
            )
            print(
                f"Wrote unified ONNX evaluation to {args.output} "
                f"(amount={_format_exact_match(metrics['amount']['raw_exact_match'])}, "
                f"time={_format_exact_match(metrics['time']['raw_exact_match'])}, "
                f"payment={_format_exact_match(metrics['payment_method_field']['raw_exact_match'])}, "
                f"status={status_display}{recipient_display})"
            )
            if failures:
                raise SystemExit("Unified OCR candidate did not meet the requested acceptance gate:\n- " + "\n- ".join(failures))
            return
        if args.command == "audit-recipient":
            summary = audit_recipient_trims_onnx(
                model_path=args.model,
                records_path=args.records,
                output_dir=args.output,
                dataset_root=args.dataset_root,
                split=args.split,
                device=args.device,
                left_trim_ratios=args.left_trims,
                foreground_contrast_threshold=args.foreground_contrast_threshold,
                cut_radius=args.cut_radius,
                blank_column_max_ink=args.blank_column_max_ink,
                require_high_resolution_recipient_input=args.require_high_resolution_recipient_input,
            )
            best = summary["best_strict_exact_trial"]
            if not isinstance(best, Mapping) or not isinstance(best.get("metrics"), Mapping):
                raise AssertionError("recipient trim audit summary has no best trial")
            print(
                f"Wrote recipient trim audit to {args.output} "
                f"(best_diagnostic_trim={best['left_trim_fraction']}, "
                f"recipient={_format_exact_match(best['metrics'].get('raw_exact_match'))}; "
                "diagnostic only, not a delivery artifact)"
            )
            return
        raise AssertionError(f"Unhandled command {args.command!r}")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Unified OCR command failed:\n{error}") from None


if __name__ == "__main__":  # pragma: no cover
    main()
