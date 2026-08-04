"""Train, export and evaluate one ONNX reader for receipt fields.

The model intentionally has one shared visual encoder and one ONNX artifact,
while retaining specialised heads where the output spaces differ:

* amount/time: independent readers.  v5 adds fixed-position digit heads
  beside the CTC readers; v6 keeps visible-format CTC and verifier paths
  separate, v7 shares the time CTC state with its format heads, and v8 keeps
  canonical amount digits in CTC while learning a tiny visible-format grammar;
* payment method: a raw CTC fallback plus a visible prefix, a finite known-bank
  classifier, and exact four-digit card-tail readers; and
* transfer status: a finite three-class head; and
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
from .ocr import clean_text, extract_field_value, normalize_payment_method, parse_anchored_recipient_row
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
from .ocr_unified_dataset import SLOT_ORDER, STATUS_CLASSES, V9_SLOT_ORDER
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
INIT_CHECKPOINT_MODES = frozenset(
    (INIT_CHECKPOINT_MODE_STRICT, INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION)
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
# Keep the public/default alias on the established four-slot protocol.
# Five-slot v9 is deliberately opt-in: callers must select ``architecture=v9``
# rather than silently changing an existing v8 training or deployment path.
# Loading/export code uses SUPPORTED_KINDS so every published version remains
# independently loadable.
KIND = KIND_V8
SUPPORTED_KINDS = frozenset(
    (KIND_V3, KIND_V4, KIND_V5, KIND_V6, KIND_V7, KIND_V8, KIND_V9, KIND_V10, KIND_V11, KIND_V12)
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
# The v12 recipient branch is materially wider than the shared field view and
# has a separately measured CPU Torch/ORT accumulation drift of 0.001011668
# on its fixed export probe.  Keep the existing 1e-3 guard for every other
# CTC output, but permit this narrowly bounded recipient-only drift.  The
# hard max-absolute cap and exact greedy argmax parity still apply.
ONNX_EXPORT_V12_RECIPIENT_LOGITS_MEAN_ABS_CAP = 1.05e-3

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


def _uses_high_resolution_recipient_input(config: "UnifiedReaderConfig") -> bool:
    """Return whether a reader has v12's second static recipient input."""
    return _is_v12(config)


def _uses_recipient_protocol(config: "UnifiedReaderConfig") -> bool:
    """Return whether this artifact has the additive fifth CTC input/output."""
    return config.architecture_version in {9, 10, 11, 12}


def _recipient_target_mode(config: "UnifiedReaderConfig") -> str | None:
    """Return the immutable recipient-label contract for five-slot artifacts."""
    if _is_v9(config):
        return "visible_recipient_value"
    if _is_v10(config):
        return "visible_recipient_line_then_extract_value"
    if _is_v11(config):
        return "anchored_recipient_value_with_value_view_crop"
    if _is_v12(config):
        return "anchored_recipient_value_with_dedicated_high_resolution_value_view"
    return None


def _recipient_charset_source(config: "UnifiedReaderConfig") -> str | None:
    """Return the immutable origin of the fifth-head training alphabet."""
    if _is_v10(config):
        return "train_only_visible_recipient_line"
    if _is_v11(config) or _is_v12(config):
        return "train_only_anchored_recipient_value"
    if _is_v9(config):
        return "train_only_visible_recipient_text"
    return None


def _recipient_input_preprocess(config: "UnifiedReaderConfig") -> str | None:
    """Return the fifth-slot visual policy recorded in an artifact contract."""
    if _is_v11(config):
        return "left_trim_then_centered_aspect_resize"
    if _is_v12(config):
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
    return V5_TEXT_DELIVERY_POLICY, V5_TEXT_DELIVERY_REASON


def _onnx_output_names(config: "UnifiedReaderConfig") -> tuple[str, ...]:
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
    if config is not None and (_is_v11(config) or _is_v12(config)) and output_name in CTC_ONNX_BLANK_INDICES:
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
    if config is not None and (_is_v11(config) or _is_v12(config)) and output_name in CTC_ONNX_BLANK_INDICES:
        return ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    return None


def _onnx_export_mean_abs_cap(
    output_name: str,
    *,
    config: "UnifiedReaderConfig | None" = None,
) -> float | None:
    """Reject a broad v11/v12 CTC-logit shift even when no greedy decision flips."""
    if config is not None and _is_v12(config) and output_name == "recipient_logits":
        return ONNX_EXPORT_V12_RECIPIENT_LOGITS_MEAN_ABS_CAP
    if config is not None and (_is_v11(config) or _is_v12(config)) and output_name in CTC_ONNX_BLANK_INDICES:
        return ONNX_EXPORT_V11_CTC_LOGITS_MEAN_ABS_CAP
    return None


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
    pooled_width: int = 8
    # v8 applies the display renderer only when every finite format component
    # is confident.  This is a diagnostic-candidate gate, never a business
    # delivery gate; keeping it in the artifact config makes Python and a
    # future deployment consumer use the identical policy.
    amount_format_min_confidence: float = 0.90

    def validate(self) -> None:
        if self.architecture_version not in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}:
            raise ValueError("architecture_version must be 3, 4, 5, 6, 7, 8, 9, 10, 11, or 12")
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
        if self.architecture_version not in {11, 12} and self.recipient_hidden_size is not None:
            raise ValueError("recipient_hidden_size is supported only by architecture v11 or v12")
        if not 1 <= self.pooled_width <= 32:
            raise ValueError("pooled_width must be between 1 and 32")
        if not math.isfinite(self.amount_format_min_confidence) or not 0.0 <= self.amount_format_min_confidence <= 1.0:
            raise ValueError("amount_format_min_confidence must be between 0 and 1")
        if not math.isfinite(self.recipient_value_left_trim) or not 0.0 <= self.recipient_value_left_trim < 1.0:
            raise ValueError("recipient_value_left_trim must be in [0, 1)")
        if self.architecture_version not in {11, 12} and not math.isclose(
            self.recipient_value_left_trim, 0.30, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("recipient_value_left_trim is supported only by architecture v11 or v12")
        if self.recipient_input_height < 16 or self.recipient_input_width < 64 or self.recipient_input_width % 4:
            raise ValueError("recipient_input_height must be >=16 and recipient_input_width a multiple of 4 >=64")
        if self.recipient_branch_channels is not None and self.recipient_branch_channels < 8:
            raise ValueError("recipient_branch_channels must be at least 8 when supplied")
        if self.architecture_version != 12 and (
            self.recipient_input_height != 128
            or self.recipient_input_width != 1024
            or self.recipient_branch_channels is not None
        ):
            raise ValueError("recipient high-resolution input settings are supported only by architecture v12")


def _recipient_hidden_size(config: UnifiedReaderConfig) -> int:
    """Return the frozen recipient branch width for this architecture."""
    if config.recipient_hidden_size is not None:
        return int(config.recipient_hidden_size)
    return 192 if _is_v11(config) or _is_v12(config) else config.payment_hidden_size


def _recipient_branch_channels(config: UnifiedReaderConfig) -> int:
    """Return v12's deliberately narrow private recipient visual width."""
    if not _is_v12(config):
        raise ValueError("recipient_branch_channels is defined only for architecture v12")
    return int(config.recipient_branch_channels or 16)


def _recipient_time_steps(config: UnifiedReaderConfig) -> int:
    """Return the CTC sequence length for the fifth output head."""
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


def _recipient_train_augmentation_policy(*, mode: str, seed: int) -> dict[str, object]:
    """Freeze the small v12-only recipient perturbation policy used in train."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("recipient train augmentation seed must be an integer")
    if mode == "none":
        return {"mode": "none"}
    if mode != "light_v1":
        raise ValueError("recipient_train_augmentation must be none or light_v1")
    return {
        "mode": "light_v1",
        "seed": int(seed),
        "horizontal_shift_px": 8,
        "vertical_shift_px": 2,
        "contrast_delta": 0.12,
        "noise_std": 0.01,
    }


def _validate_recipient_train_augmentation_policy(policy: object) -> dict[str, object]:
    """Validate a persisted train-only v12 perturbation policy."""
    if not isinstance(policy, Mapping):
        raise ValueError("recipient train augmentation policy is missing or invalid")
    mode = policy.get("mode")
    if mode == "none":
        if set(policy) != {"mode"}:
            raise ValueError("recipient train augmentation policy is invalid")
        return _recipient_train_augmentation_policy(mode="none", seed=0)
    if mode != "light_v1":
        raise ValueError("recipient train augmentation policy is invalid")
    expected = _recipient_train_augmentation_policy(mode="light_v1", seed=policy.get("seed"))
    if dict(policy) != expected:
        raise ValueError("recipient train augmentation policy is invalid")
    return expected


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
    recipient_train_augmentation_policy: object | None = None,
) -> dict[str, object]:
    """Build frozen fifth-slot metadata for checkpoints and ONNX sidecars."""
    if not _uses_recipient_protocol(config):
        return {}
    metadata: dict[str, object] = {
        "recipient_input_preprocess": _recipient_input_preprocess(config),
    }
    if _is_v11(config) or _is_v12(config):
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
        if recipient_train_augmentation_policy is not None:
            metadata["recipient_train_augmentation_policy"] = _validate_recipient_train_augmentation_policy(
                recipient_train_augmentation_policy
            )
    if _is_v12(config):
        metadata.update(
            {
                "recipient_input_name": "recipient_value_image",
                "recipient_input_shape": [1, 1, config.recipient_input_height, config.recipient_input_width],
                "recipient_branch_channels": _recipient_branch_channels(config),
                "recipient_time_steps": _recipient_time_steps(config),
            }
        )
    return metadata


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


def build_unified_reader(
    *,
    payment_vocab_size: int,
    config: UnifiedReaderConfig,
    payment_bank_prefix_vocab_size: int | None = None,
    recipient_vocab_size: int | None = None,
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
        raise ValueError("v9/v10/v11/v12 needs recipient_vocab_size including CTC blank plus at least one character")
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
                        self.recipient_encoder = nn.Sequential(
                            DepthwiseBlock(recipient_first, recipient_second, stride=(2, 2)),
                            DepthwiseBlock(recipient_second, recipient_third, stride=(2, 1)),
                            DepthwiseBlock(recipient_third, recipient_channels, stride=(1, 1)),
                        )
                        recipient_feature_height = (config.recipient_input_height + 7) // 8
                    self.recipient_ctc_vertical_reducer = VerticalTextReducer(
                        recipient_channels, recipient_feature_height
                    )
                    self.recipient_ctc_sequence = nn.GRU(
                        recipient_channels, _recipient_hidden_size(config), bidirectional=True
                    )
                    self.recipient_classifier = nn.Linear(
                        _recipient_hidden_size(config) * 2, int(recipient_vocab_size)
                    )

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
                    recipient_sequence, _ = self.recipient_ctc_sequence(recipient_features.permute(2, 0, 1))
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
                        return v8_outputs + (recipient_logits,)
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
        DATASET_KIND_V12
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
    if contract_path.is_file():
        contract = _load_json_object(contract_path)
        if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") != expected_dataset_kind:
            raise ValueError(f"{contract_path}: unsupported unified dataset contract")
        if contract.get("slot_order") != list(slot_order) or contract.get("status_classes") != list(STATUS_CLASSES):
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
            if config is not None and (_is_v11(config) or _is_v12(config)):
                _validate_anchored_recipient_slot(
                    parsed_slots.get("recipient_field"),
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
) -> None:
    for record in records:
        fields = ("amount", "time", "payment_method_field", "recipient_field") if _uses_recipient_protocol(config) else (
            "amount",
            "time",
            "payment_method_field",
        )
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
                else None
            )
            if field == "recipient_field" and characters is None:
                raise ValueError("v9/v10/v11/v12 recipient CTC validation needs a train-only recipient charset")
            # Validation/test recipient OOV is intentional evidence for a
            # train-only Unicode alphabet.  It must not make the manifest
            # unloadable; those rows are scored/reviewed later.  Train labels,
            # by contrast, must always be encodable.
            validate_characters = field != "recipient_field" or str(record["split"]) == "train"
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
            if (_is_v10(config) or _is_v11(config) or _is_v12(config)) and field == "recipient_field":
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
                    if (_is_v11(config) or _is_v12(config)) and field == "recipient_field"
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
    if policy.get("mode") != "light_v1":
        raise ValueError("recipient augmentation RNG requires light_v1 policy")
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
        ] = image[:, source_y_start:source_y_end, source_x_start:source_x_end]
    # White remains white under the contrast transform; only ink strength is
    # altered. This avoids teaching the model a non-existent dark background.
    contrast_delta = float(normalized_policy["contrast_delta"])
    contrast = 1.0 + float(rng.uniform(-contrast_delta, contrast_delta))
    augmented = 1.0 - (1.0 - shifted) * contrast
    noise_std = float(normalized_policy["noise_std"])
    if noise_std > 0.0:
        augmented = augmented + rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
    return np.clip(augmented, 0.0, 1.0).astype(np.float32, copy=False)


class _UnifiedReceiptDataset:
    """A picklable dataset so Windows DataLoader workers remain usable."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        config: UnifiedReaderConfig,
        recipient_train_augmentation_policy: Mapping[str, object] | None = None,
        recipient_only: bool = False,
    ) -> None:
        if recipient_only and not _is_v12(config):
            raise ValueError("recipient_only dataset is supported only by architecture v12")
        self._records = list(records)
        self._config = config
        self._recipient_only = recipient_only
        self._recipient_train_augmentation_policy = _validate_recipient_train_augmentation_policy(
            {"mode": "none"}
            if recipient_train_augmentation_policy is None
            else recipient_train_augmentation_policy
        )
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("dataset epoch must be a non-negative integer")
        self._epoch = epoch

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
                    epoch=self._epoch,
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
                    epoch=self._epoch,
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
    if not _is_v12(config):
        raise ValueError("recipient-only logits are supported only by architecture v12")
    expected_shape = [recipient_value_images.shape[0], 1, config.recipient_input_height, config.recipient_input_width]
    if list(recipient_value_images.shape) != expected_shape:
        raise ValueError(
            "recipient_value_images must have shape "
            f"[batch,1,{config.recipient_input_height},{config.recipient_input_width}]"
        )
    encoded = model.recipient_encoder(model.recipient_stem(recipient_value_images))
    features = model.recipient_ctc_vertical_reducer(encoded)
    sequence, _ = model.recipient_ctc_sequence(features.permute(2, 0, 1))
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
    payment_bank_prefix_classes: Sequence[str] | None,
    payment_bank_class_weights: Any | None,
    status_to_id: Mapping[str, int],
    status_criterion: Any | None,
    status_enabled: bool,
    payment_loss_weight: float,
    recipient_loss_weight: float,
    config: UnifiedReaderConfig,
    structured_outputs: Mapping[str, Any] | None,
    ctc_loss_weight: float,
    structured_loss_weight: float,
    torch: Any,
    recipient_sample_weights: Sequence[float] | None = None,
    allow_empty: bool = False,
    collect_metrics: bool = True,
    recipient_only: bool = False,
) -> tuple[Any | None, dict[str, dict[str, float | int]] | None]:
    """Return one batch loss and, when requested, detached diagnostics.

    The training loop only needs the scalar loss.  Materialising every
    diagnostic by calling ``.cpu()`` per batch forces a CUDA synchronization,
    which prevents pinned-memory transfers and GPU work from overlapping.  The
    validation/audit paths may still request the exact historical diagnostics;
    the hot training path deliberately opts out.
    """
    if recipient_only:
        if not _is_v12(config):
            raise ValueError("recipient_only loss is supported only by architecture v12")
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
        # v12 normally applies its CTC multiplier alongside finite structured
        # heads.  Preserve that scalar so recipient-only fine-tuning has the
        # same effective recipient-loss scale as the guarded full recipe.
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
    payment_bank_prefix_classes: Sequence[str] | None,
    payment_bank_class_weights: Any | None,
    status_to_id: Mapping[str, int],
    status_criterion: Any | None,
    status_enabled: bool,
    payment_loss_weight: float,
    recipient_loss_weight: float,
    ctc_loss_weight: float,
    structured_loss_weight: float,
    torch: Any,
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
    ctc_counters: dict[str, Counter[str]] = {
        field: Counter()
        for field in (("amount", "time", "payment_method_field", "recipient_field") if _uses_recipient_protocol(config) else (
            "amount",
            "time",
            "payment_method_field",
        ))
    }
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
                payment_bank_prefix_classes=payment_bank_prefix_classes,
                payment_bank_class_weights=payment_bank_class_weights,
                status_to_id=status_to_id,
                status_criterion=status_criterion,
                status_enabled=status_enabled,
                payment_loss_weight=payment_loss_weight,
                recipient_loss_weight=recipient_loss_weight,
                config=config,
                structured_outputs=outputs if _uses_structured_heads(config) else None,
                ctc_loss_weight=ctc_loss_weight,
                structured_loss_weight=structured_loss_weight,
                torch=torch,
                allow_empty=True,
                collect_metrics=False,
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
                if status_enabled:
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
) -> dict[str, object]:
    """Validate and freeze a training-only best-checkpoint policy.

    ``recipient_priority`` deliberately changes only which epoch becomes
    ``best.pt``.  It does not alter the model graph, preprocessing, decoder,
    or ONNX/session ABI.  The three mature text fields receive caller-supplied
    validation floors so an experiment cannot trade them away for a higher
    recipient score.
    """
    if checkpoint_selection not in CHECKPOINT_SELECTION_MODES:
        allowed = ", ".join(sorted(CHECKPOINT_SELECTION_MODES))
        raise ValueError(f"checkpoint_selection must be one of: {allowed}")
    raw_minima = {
        "amount": checkpoint_min_amount_candidate_exact,
        "time": checkpoint_min_time_candidate_exact,
        "payment_method_field": checkpoint_min_payment_candidate_exact,
    }
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
        }
    if not _uses_recipient_protocol(config):
        raise ValueError("checkpoint_selection=recipient_priority requires architecture v9, v10, v11, or v12")
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
    status_safety = (
        -float(validation["status_non_success_to_success"])
        if bool(status_policy.get("training_enabled"))
        else 0.0
    )
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
        return (
            (
                status_safety,
                _validation_candidate_exact(validation, "recipient_field"),
                float(validation["candidate_text_macro_exact_match"] or -1.0),
                float(validation["candidate_text_exact_match"]),
                float(verifier_score) if verifier_score is not None else -1.0,
                -float(validation["loss"]),
            ),
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
        return (
            (
                status_safety,
                float(validation["candidate_text_macro_exact_match"] or -1.0),
                float(validation["candidate_text_exact_match"]),
                verifier_score,
                -float(validation["loss"]),
            ),
            [],
        )
    return (
        (
            status_safety,
            float(validation["delivery_exact_overall"]),
            float(validation["exact_match"]),
            -1.0,
            -float(validation["loss"]),
        ),
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
) -> tuple[list[str], list[str], dict[str, object]]:
    """Lock financial label semantics to a v12 seed for a recipient-only run.

    A receipt manifest can gain/reorder payment text or bank-prefix labels
    between r2 and r3.  Rebuilding those output maps would reinterpret frozen
    financial classifier rows, even though recipient-only fine-tuning never
    updates them.  This narrow preflight instead keeps the seed's financial
    maps exactly, while permitting only an additive recipient Unicode charset.
    """
    if not _is_v12(config):
        raise ValueError("recipient_only_expansion is supported only by architecture v12")
    if recipient_characters is None or payment_bank_prefix_classes is None:
        raise ValueError("recipient_only_expansion requires v12 recipient and payment bank label maps")
    checkpoint_path = Path(init_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    source_config = _checkpoint_config(payload)
    if source_config != config:
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
    missing_source_characters = sorted(set(source_recipient_characters) - set(recipient_characters))
    if missing_source_characters:
        raise ValueError(
            "recipient_only_expansion cannot discard characters from the init checkpoint recipient map; "
            f"missing={''.join(missing_source_characters)!r}"
        )
    if source_payment_bank_prefix_classes is None:
        raise ValueError("init checkpoint payment bank-prefix class map does not match the current training data")
    return (
        list(source_payment_characters),
        list(source_payment_bank_prefix_classes),
        {
            "mode": "checkpoint_financial_label_maps_v1",
            "reason": (
                "recipient-only v12 fine-tune freezes every non-recipient parameter, so payment and bank "
                "classifier row semantics remain locked to the compatible seed checkpoint"
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
                "checkpoint_count": len(source_recipient_characters),
                "checkpoint_sha256": _label_map_sha256(source_recipient_characters),
                "data_derived_count": len(recipient_characters),
                "data_derived_sha256": _label_map_sha256(recipient_characters),
                "checkpoint_is_subset_of_data_derived": True,
                "new_data_derived_character_count": len(
                    set(recipient_characters) - set(source_recipient_characters)
                ),
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


def _parameter_only_initialization(
    *,
    init_checkpoint: Path | None,
    init_checkpoint_mode: str = INIT_CHECKPOINT_MODE_STRICT,
    config: UnifiedReaderConfig,
    amount_characters: Sequence[str],
    time_characters: Sequence[str],
    payment_characters: Sequence[str],
    recipient_characters: Sequence[str] | None,
    payment_bank_prefix_classes: Sequence[str] | None,
    torch: Any,
    target_state_dict: Mapping[str, object] | None = None,
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
            raise ValueError("recipient_only_expansion requires a compatible --init-checkpoint")
        return None, {
            "mode": "random",
            "optimizer_restored": False,
            "epoch_reset": True,
        }
    checkpoint_path = Path(init_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path, torch=torch)
    source_config = _checkpoint_config(payload)
    if source_config != config:
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
    }
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_STRICT:
        if source_recipient_characters is None or recipient_characters is None:
            if source_recipient_characters is not recipient_characters:
                raise ValueError("init checkpoint recipient character map does not match the current training data")
        elif list(source_recipient_characters) != list(recipient_characters):
            raise ValueError("init checkpoint recipient character map does not match the current training data")
        return state_dict, initialization

    if not _is_v12(config):
        raise ValueError("recipient_only_expansion is supported only by architecture v12")
    if source_recipient_characters is None or recipient_characters is None:
        raise ValueError("init checkpoint recipient character map does not match the current training data")
    if target_state_dict is None:
        raise ValueError("recipient_only_expansion requires a freshly initialised v12 target state")
    remapped_state, row_mapping = _recipient_classifier_unicode_expansion_state(
        source_state_dict=state_dict,
        target_state_dict=target_state_dict,
        source_recipient_characters=source_recipient_characters,
        target_recipient_characters=recipient_characters,
    )
    initialization.update(
        {
            "mode": "parameter_only_recipient_unicode_expansion",
            "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
            "recipient_classifier_row_mapping": row_mapping,
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
    recipient_sampling_weight: float = 1.0,
    recipient_rare_character_max_support: int = 0,
    recipient_rare_character_sampling_weight: float = 1.0,
    recipient_long_text_min_length: int = 0,
    recipient_long_text_sampling_weight: float = 1.0,
    recipient_low_confidence_threshold: float | None = None,
    recipient_low_confidence_loss_weight: float = 1.0,
    recipient_confidence_curriculum_epochs: int = 0,
    recipient_train_augmentation: str = "none",
    recipient_only_fine_tune: bool = False,
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
    if recipient_only_fine_tune:
        if not _is_v12(config):
            raise ValueError("recipient_only_fine_tune is supported only by architecture v12")
        if init_checkpoint is None:
            raise ValueError("recipient_only_fine_tune requires a compatible --init-checkpoint")
    if init_checkpoint_mode not in INIT_CHECKPOINT_MODES:
        raise ValueError(
            "init_checkpoint_mode must be one of "
            f"{', '.join(sorted(INIT_CHECKPOINT_MODES))}"
        )
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION:
        if not recipient_only_fine_tune or not _is_v12(config):
            raise ValueError(
                "recipient_only_expansion requires architecture v12 with recipient_only_fine_tune enabled"
            )
        if init_checkpoint is None:
            raise ValueError("recipient_only_expansion requires a compatible --init-checkpoint")
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
        or recipient_train_augmentation_policy["mode"] != "none"
    )
    if not (_is_v11(config) or _is_v12(config)) and recipient_training_options_requested:
        raise ValueError(
            "recipient sampling/confidence curriculum is supported only by architecture v11 or v12"
        )
    if not _is_v12(config) and recipient_train_augmentation_policy["mode"] != "none":
        raise ValueError("recipient_train_augmentation is supported only by architecture v12")
    checkpoint_selection_policy = _checkpoint_selection_policy(
        config=config,
        checkpoint_selection=checkpoint_selection,
        checkpoint_min_amount_candidate_exact=checkpoint_min_amount_candidate_exact,
        checkpoint_min_time_candidate_exact=checkpoint_min_time_candidate_exact,
        checkpoint_min_payment_candidate_exact=checkpoint_min_payment_candidate_exact,
    )
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if (
        learning_rate <= 0
        or weight_decay < 0
        or payment_loss_weight <= 0
        or recipient_loss_weight <= 0
        or recipient_sampling_weight <= 0
        or ctc_loss_weight <= 0
        or structured_loss_weight <= 0
    ):
        raise ValueError(
            "learning_rate, payment_loss_weight, recipient_loss_weight, recipient_sampling_weight, ctc_loss_weight, and structured_loss_weight must be positive; "
            "weight_decay cannot be negative"
        )
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if persistent_workers and num_workers <= 0:
        raise ValueError("persistent_workers requires num_workers to be positive")
    if persistent_workers and recipient_train_augmentation_policy["mode"] != "none":
        raise ValueError(
            "persistent_workers is unsafe with recipient train augmentation because worker-local epoch state "
            "would stop advancing; leave it disabled or use --recipient-train-augmentation none"
        )
    if payment_bank_prefix_min_support <= 0:
        raise ValueError("payment_bank_prefix_min_support must be positive")
    if not math.isfinite(recipient_sampling_weight):
        raise ValueError("recipient_sampling_weight must be finite and positive")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"training output already contains files: {output_dir}. Choose a new empty directory.")
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
    if bool(status_policy["training_enabled"]):
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
    if _uses_recipient_protocol(config):
        recipient_characters: list[str] | None = _recipient_charset(train_records)
        recipient_to_id: dict[str, int] | None = {
            character: index for index, character in enumerate(recipient_characters, start=1)
        }
    else:
        recipient_characters = None
        recipient_to_id = None
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
    if init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION:
        assert init_checkpoint is not None
        assert payment_bank_prefix_classes is not None
        (
            payment_characters,
            payment_bank_prefix_classes,
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
        )
        payment_bank_prefix_counts = _payment_bank_prefix_retained_counts(
            train_records,
            classes=payment_bank_prefix_classes,
        )
    payment_to_id = {character: index for index, character in enumerate(payment_characters, start=1)}
    _validate_ctc_capacity(records, config=config, recipient_characters=recipient_characters)
    status_to_id = {name: index for index, name in enumerate(STATUS_CLASSES)}
    payment_oov = _payment_oov_by_split(records, payment_characters=set(payment_characters))
    recipient_oov = (
        _recipient_oov_by_split(records, characters=recipient_characters)
        if recipient_characters is not None
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
            record for record in train_records if _slot_text(record, "recipient_field") is not None
        ]
        if not training_records:
            raise ValueError("recipient_only_fine_tune requires at least one train receipt with recipient_field")
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
        "cuda_tf32_requested": cuda_tf32,
        "cudnn_benchmark_requested": cudnn_benchmark,
        "recipient_only_private_branch_training": recipient_only_fine_tune,
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
            recipient_train_augmentation_policy if _is_v12(config) else None
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
        if not (_is_v11(config) or _is_v12(config)):
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
        # Keep worker processes non-persistent by default.  v12's optional
        # train augmentation is seeded by the epoch, and Windows-spawned
        # persistent workers otherwise retain the first epoch's dataset copy.
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
    )
    initialization_state, initialization = _parameter_only_initialization(
        init_checkpoint=init_checkpoint,
        init_checkpoint_mode=init_checkpoint_mode,
        config=config,
        amount_characters=amount_characters,
        time_characters=time_characters,
        payment_characters=payment_characters,
        recipient_characters=recipient_characters,
        payment_bank_prefix_classes=payment_bank_prefix_classes,
        torch=torch,
        target_state_dict=model.state_dict(),
    )
    if initialization_state is not None:
        # This is intentionally strict: equal tensor shapes are insufficient
        # when a CTC character or classifier-class ordering has changed.
        model.load_state_dict(initialization_state, strict=True)
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
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("recipient_"))
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if trainable_parameter_count == 0:
            raise AssertionError("v12 recipient-only fine-tune found no recipient parameters")
        fine_tune_policy = {
            "mode": "recipient_only_v12",
            "trainable_parameter_count": trainable_parameter_count,
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            ),
            "frozen_parameter_prefix_exclusion": "recipient_",
            "training_forward": "private_recipient_branch_only_v12",
            "source_train_records": len(train_records),
            "recipient_train_records": len(training_records),
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
                    "recipient_train_augmentation_policy": recipient_train_augmentation_policy,
                    **_recipient_artifact_metadata(
                        config,
                        recipient_sampling_policy=recipient_sampling_policy,
                        recipient_confidence_policy=recipient_confidence_policy,
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
    for epoch in range(1, epochs + 1):
        epoch_started = perf_counter()
        model.train()
        train_dataset.set_epoch(epoch)
        total_loss_tensor: Any | None = None
        total_receipts = 0
        for batch in train_loader:
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
                structured_outputs = outputs if _uses_structured_heads(config) else None
            optimizer.zero_grad(set_to_none=True)
            recipient_sample_weights = (
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
                payment_bank_prefix_classes=payment_bank_prefix_classes,
                payment_bank_class_weights=payment_bank_train_weights,
                status_to_id=status_to_id,
                status_criterion=status_train_criterion,
                status_enabled=bool(status_policy["training_enabled"]),
                payment_loss_weight=payment_loss_weight,
                recipient_loss_weight=recipient_loss_weight,
                config=config,
                structured_outputs=structured_outputs,
                ctc_loss_weight=ctc_loss_weight,
                structured_loss_weight=structured_loss_weight,
                torch=torch,
                recipient_sample_weights=recipient_sample_weights,
                collect_metrics=False,
                recipient_only=recipient_only_fine_tune,
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
        # The hot recipient-only path intentionally avoids per-batch CPU
        # reads.  Synchronise once at the epoch boundary so the recorded
        # training/validation timings are not misleadingly split across the
        # asynchronous CUDA queue.
        if uses_cuda:
            torch.cuda.synchronize(target_device)
        train_seconds = perf_counter() - epoch_started
        validation_started = perf_counter()
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
            payment_bank_prefix_classes=payment_bank_prefix_classes,
            payment_bank_class_weights=None,
            status_to_id=status_to_id,
            status_criterion=status_validation_criterion,
            status_enabled=bool(status_policy["training_enabled"]),
            payment_loss_weight=payment_loss_weight,
            recipient_loss_weight=recipient_loss_weight,
            ctc_loss_weight=ctc_loss_weight,
            structured_loss_weight=structured_loss_weight,
            torch=torch,
        )
        if uses_cuda:
            torch.cuda.synchronize(target_device)
        validation_seconds = perf_counter() - validation_started
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": (
                float((total_loss_tensor / total_receipts).cpu())
                if total_loss_tensor is not None and total_receipts > 0
                else math.nan
            ),
            "train_seconds": train_seconds,
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
        protection_report = _checkpoint_protection_report(
            validation,
            policy=checkpoint_selection_policy,
            failures=protection_failures,
        )
        epoch_record["checkpoint_selection_eligible"] = score is not None
        epoch_record["checkpoint_selection_protection_failures"] = protection_failures
        epoch_record["checkpoint_selection_score"] = list(score) if score is not None else None
        epoch_record["checkpoint_protection"] = protection_report
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
                    "recipient_train_augmentation_policy": recipient_train_augmentation_policy,
                    **_recipient_artifact_metadata(
                        config,
                        recipient_sampling_policy=recipient_sampling_policy,
                        recipient_confidence_policy=recipient_confidence_policy,
                        recipient_train_augmentation_policy=recipient_train_augmentation_policy,
                    ),
                }
                if recipient_characters is not None
                else {}
            ),
            "status_classes": list(STATUS_CLASSES),
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
            "checkpoint_selection_policy": checkpoint_selection_policy,
            "initialization": initialization,
            "training_runtime": training_runtime,
            "fine_tune_policy": fine_tune_policy,
            "ctc_loss_weight": ctc_loss_weight,
            "structured_loss_weight": structured_loss_weight,
            "epoch": epoch,
            "metrics": epoch_record,
        }
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
                "recipient_train_augmentation_policy": recipient_train_augmentation_policy,
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
            raise ValueError("Unified v6/v7/v8/v9/v10/v11/v12 OCR checkpoint amount/time label maps are unsupported")
        bank_classes = payload.get("payment_bank_prefix_classes")
        if (
            not isinstance(bank_classes, list)
            or len(bank_classes) < 2
            or bank_classes[0] != PAYMENT_BANK_OTHER_CLASS
            or not all(isinstance(value, str) and value for value in bank_classes)
            or len(set(bank_classes)) != len(bank_classes)
            or bank_classes[1:] != sorted(bank_classes[1:])
        ):
            raise ValueError("Unified v6/v7/v8/v9/v10/v11/v12 OCR checkpoint bank-prefix class map is invalid")
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
                raise ValueError("Unified v9/v10/v11/v12 OCR checkpoint recipient charset or blank index is invalid")
            recipient_sha256 = hashlib.sha256("".join(raw_recipient).encode("utf-8")).hexdigest()
            if payload.get("recipient_charset_sha256") != recipient_sha256:
                raise ValueError("Unified v9/v10/v11/v12 OCR checkpoint recipient charset SHA-256 is invalid")
            expected_recipient_charset_source = _recipient_charset_source(config)
            if payload.get("recipient_charset_source") != expected_recipient_charset_source:
                raise ValueError("Unified v9/v10/v11/v12 OCR checkpoint recipient charset source is invalid")
            if payload.get("recipient_target") != _recipient_target_mode(config):
                raise ValueError("Unified v9/v10/v11/v12 OCR checkpoint recipient target contract is invalid")
            if _is_v11(config) or _is_v12(config):
                _recipient_artifact_metadata(
                    config,
                    recipient_sampling_policy=payload.get("recipient_sampling_policy"),
                    recipient_confidence_policy=payload.get("recipient_confidence_policy"),
                    recipient_train_augmentation_policy=payload.get("recipient_train_augmentation_policy"),
                )
            recipient = list(raw_recipient)
        return list(amount), list(time), list(payment), recipient, list(status), list(bank_classes)
    if numeric != list(NUMERIC_CHARACTERS):
        raise ValueError("Unified OCR checkpoint numeric label map is not the supported fixed numeric charset")
    return list(numeric), list(numeric), list(payment), None, list(status), None


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
        # CPU ORT and CPU Torch can accumulate small FP32 drift through the
        # exported GRU/normalisation sequence. The value tolerance matches
        # the detector export verifier and is paired with an exact argmax
        # check, so a changed decoded character/status is never accepted.
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
        max_abs_cap = _onnx_export_max_abs_cap(name, config=config)
        mean_abs_cap = _onnx_export_mean_abs_cap(name, config=config)
        if (
            not np.allclose(
                actual_array,
                expected_array,
                rtol=ONNX_EXPORT_RTOL,
                atol=atol,
            )
            or (max_abs_cap is not None and max_abs > max_abs_cap)
            or (mean_abs_cap is not None and mean_abs > mean_abs_cap)
            or argmax_mismatches
        ):
            max_abs_cap_text = f", max_abs_cap={max_abs_cap:g}" if max_abs_cap is not None else ""
            mean_abs_cap_text = f", mean_abs_cap={mean_abs_cap:g}" if mean_abs_cap is not None else ""
            raise ValueError(
                f"Exported unified OCR ONNX output {name!r} differs from Torch beyond "
                f"rtol={ONNX_EXPORT_RTOL:g}, atol={atol:g}{max_abs_cap_text}{mean_abs_cap_text} or changes its argmax: "
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
    recipient_artifact_metadata = (
        _recipient_artifact_metadata(
            config,
            recipient_sampling_policy=payload.get("recipient_sampling_policy"),
            recipient_confidence_policy=payload.get("recipient_confidence_policy"),
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
    model = build_unified_reader(
        payment_vocab_size=len(payment_characters) + 1,
        config=config,
        payment_bank_prefix_vocab_size=(
            len(payment_bank_prefix_classes) if payment_bank_prefix_classes is not None else None
        ),
        recipient_vocab_size=(len(recipient_characters) + 1 if recipient_characters is not None else None),
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
                    return v8_outputs + (outputs["recipient_logits"][:, 0, :],)
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
                if _is_v11(config) or _is_v12(config)
                else {}
            ),
            **(
                {"input_name": "recipient_value_image"}
                if _uses_high_resolution_recipient_input(config)
                else {}
            ),
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
                if _is_v12(config)
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
            if _is_v11(config) or _is_v12(config):
                expected_recipient_metadata = _recipient_artifact_metadata(
                    config,
                    recipient_sampling_policy=labels.get("recipient_sampling_policy"),
                    recipient_confidence_policy=labels.get("recipient_confidence_policy"),
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
        if _is_v11(config) or _is_v12(config):
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
    }:
        if status_output.get("runtime_policy") != status_policy["runtime_policy"]:
            raise ValueError("Unified OCR ONNX status output policy differs from status_head_policy")
        expected_review = "review" if status_policy["runtime_policy"] == "review_only" else None
        if status_output.get("review_value") != expected_review:
            raise ValueError("Unified OCR ONNX status output review value is invalid")
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
    if architecture_version < 5 or field == "transfer_status":
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
        observed = metrics[field]["raw_exact_match"]
        if observed is None:
            failures.append(f"{field}: no held-out reference labels remain for the requested acceptance gate")
        elif float(observed) < threshold:
            failures.append(f"{field}: raw_exact_match={float(observed):.4f} < {threshold:.4f}")
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
    evaluation_records = [record for record in records if record["split"] == split]
    if not evaluation_records:
        raise ValueError(f"No {split} receipt records found")
    required_evaluation_fields = ["amount", "time", "payment_method_field"]
    if _uses_recipient_protocol(config):
        required_evaluation_fields.append("recipient_field")
    if status_delivery_allowed or min_status_exact_match is not None or max_non_success_to_success is not None:
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
    status_confusion: Counter[str] = Counter()
    status_reference_counts: Counter[str] = Counter()
    for record in evaluation_records:
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
            recipient_text, recipient_confidence = _ctc_single_output(
                runtime_outputs["recipient_logits"], characters=recipient_characters
            )
        status_index, status_confidence = _softmax_confidence(status_logits)
        raw_status_text = STATUS_CLASSES[status_index]
        ctc_predictions: dict[str, tuple[str, float]] = {
            "amount": (amount_text, amount_confidence),
            "time": (time_text, time_confidence),
            "payment_method_field": (payment_text, payment_confidence),
        }
        if _uses_recipient_protocol(config):
            ctc_predictions["recipient_field"] = (recipient_text, recipient_confidence)
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
            raw_status_text if status_delivery_allowed else "review",
            status_confidence,
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
                    "raw_model_candidate_text": raw_status_text if field == "transfer_status" else None,
                    "confidence": round(confidence, 6),
                    "runtime_policy": (
                        status_policy["runtime_policy"]
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
                    ),
                    "non_success_to_success": non_success_to_success,
                    "receipt_latency_ms": round(latency_ms, 4),
                }
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
    )
    if (
        (min_status_exact_match is not None or max_non_success_to_success is not None)
        and not status_delivery_allowed
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
        "evaluation_split": split,
        "label_sources": label_sources,
        "providers": active_providers,
        "slot_order": list(_slot_order(config)),
        "amount_format_policy": {
            "artifact_min_confidence": artifact_amount_format_min_confidence,
            "effective_min_confidence": config.amount_format_min_confidence if _uses_v8_protocol(config) else None,
            "evaluation_override": amount_format_min_confidence_override,
        },
        "by_field": by_field,
        "status_confusion": dict(sorted(status_confusion.items())),
        "status_reference_class_counts": {
            class_name: int(status_reference_counts[class_name]) for class_name in STATUS_CLASSES
        },
        "status_head_policy": status_policy,
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
    """Reproduce v11's fifth-slot image transform from an already loaded crop.

    The audit loads each source crop once, then produces several candidate
    value views in memory.  Its resize, centring, and trim arithmetic exactly
    mirrors :func:`preprocess_image`; it is intentionally private so the
    delivery preprocessor remains the sole public runtime contract.
    """
    source = np.asarray(gray, dtype=np.uint8)
    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("recipient crop must be a non-empty grayscale image")
    width = int(source.shape[1])
    if trim_px < 0 or trim_px >= width:
        raise ValueError("recipient trim pixel is outside the source crop")
    image = Image.fromarray(source, mode="L").crop((trim_px, 0, width, int(source.shape[0])))
    scale = min(config.image_width / image.width, config.image_height / image.height)
    resized_width = max(1, min(config.image_width, int(round(image.width * scale))))
    resized_height = max(1, min(config.image_height, int(round(image.height * scale))))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((resized_width, resized_height), resampling)
    canvas = np.full((config.image_height, config.image_width), 255, dtype=np.uint8)
    top = (config.image_height - resized_height) // 2
    left = (config.image_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(image, dtype=np.uint8)
    return (canvas.astype(np.float32) / 255.0)[np.newaxis, :, :]


def _recipient_audit_rendered_width(
    *,
    source_height: int,
    retained_width: int,
    config: UnifiedReaderConfig,
) -> int:
    """Return the non-letterboxed horizontal pixels visible to v11's CTC head."""
    if source_height <= 0 or retained_width <= 0:
        raise ValueError("recipient audit source dimensions must be positive")
    scale = min(config.image_width / retained_width, config.image_height / source_height)
    return max(1, min(config.image_width, int(round(retained_width * scale))))


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
) -> dict[str, object]:
    """Diagnose the frozen v11 recipient trim without changing an artifact.

    Every trial uses the same validated ONNX and the same held-out records.
    Only the fifth input slot's *in-memory* pixel trim differs.  The result is
    therefore diagnostic evidence for a potential v12 preprocessing contract,
    never a rewritten or deployable v11 model.
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
    if not _is_v11(config):
        raise ValueError("audit-recipient supports only v11 ONNX artifacts with a frozen recipient value trim")
    _, _, recipient_characters, _ = _load_onnx_artifact_details(model_path)
    if recipient_characters is None:
        raise AssertionError("v11 recipient characters were validated with the ONNX sidecar")

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
        raise ValueError(f"No {split} recipient labels remain for the v11 trim audit")

    onnxruntime = _require_onnxruntime()
    model_path = model_path.resolve()
    session, active_providers = _create_onnx_session(onnxruntime, model_path, device=device)
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    expected_outputs = list(_onnx_output_names(config))
    if input_names != ["field_images"] or output_names != expected_outputs:
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
    recipient_output = next((item for item in session.get_outputs() if item.name == "recipient_logits"), None)
    expected_recipient_shape = [config.image_width // 4, len(recipient_characters) + 1]
    if recipient_output is None or list(recipient_output.shape) != expected_recipient_shape:
        actual_shape = None if recipient_output is None else list(recipient_output.shape)
        raise ValueError(
            "Unified OCR ONNX recipient output shape differs from contract: "
            f"actual={actual_shape}, expected={expected_recipient_shape}"
        )

    recipient_index = _slot_order(config).index("recipient_field")
    character_set = set(recipient_characters)
    rows_by_ratio: dict[float, list[dict[str, object]]] = {ratio: [] for ratio in ratios}
    total = len(recipient_records)
    progress_interval = max(1, total // 20)
    for number, (record, slot, reference_text) in enumerate(recipient_records, start=1):
        # `_input_tensor` keeps the other four slots exactly as the delivery
        # preprocessing contract defines them.  Recipient `logits` depend on
        # only the fifth slot; each trial replaces just that slot in-place.
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
            field_images[recipient_index] = _recipient_audit_preprocess_gray(
                gray,
                config=config,
                trim_px=geometry.trim_px,
            )
            started = perf_counter()
            recipient_logits = session.run(["recipient_logits"], {"field_images": field_images})[0]
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
            "This is an exploratory in-memory v11 recipient pixel-preprocessing sweep. It does not rewrite the "
            "ONNX model, labels, contract, or deployment trim. It compares with held-out Paddle-derived teacher "
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
        "--recipient-train-augmentation",
        choices=("none", "light_v1"),
        default="none",
        help=(
            "v12 only: train-only recipient value-crop perturbation; light_v1 uses deterministic small shifts, "
            "contrast and noise without changing the ONNX input contract"
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
        choices=tuple(sorted(INIT_CHECKPOINT_MODES)),
        default=INIT_CHECKPOINT_MODE_STRICT,
        help=(
            "strict requires every label map to match the seed. recipient_only_expansion is v12 recipient-only "
            "only: it locks payment/bank maps to the seed and maps additive recipient Unicode rows by character."
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
        choices=("v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"),
        default="v8",
        help=(
            "v9 is the five-field value-only reader. v10 learns the complete visible recipient row. v11 keeps one "
            "five-slot ONNX but learns an anchored recipient value from a left-trimmed value view. v12 remains one "
            "ONNX/session but adds a second high-resolution recipient value input. v8 remains the compatible "
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
        help="v11/v12 only: fraction trimmed from the left of the anchored recipient crop before resize",
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
            "Keep DataLoader workers alive between epochs. Not supported with v12 recipient train augmentation "
            "until its epoch state is shared safely."
        ),
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
        help="diagnose the frozen v11 recipient value crop with in-memory trim trials",
    )
    audit_recipient.add_argument("--model", type=Path, required=True)
    audit_recipient.add_argument("--records", type=Path, required=True, help="v11 unified_fields.jsonl")
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
        help="One or more exploratory v11 left-crop fractions, such as 0 0.20 0.30 0.40",
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
                recipient_sampling_weight=args.recipient_sampling_weight,
                recipient_rare_character_max_support=args.recipient_rare_character_max_support,
                recipient_rare_character_sampling_weight=args.recipient_rare_character_sampling_weight,
                recipient_long_text_min_length=args.recipient_long_text_min_length,
                recipient_long_text_sampling_weight=args.recipient_long_text_sampling_weight,
                recipient_low_confidence_threshold=args.recipient_low_confidence_threshold,
                recipient_low_confidence_loss_weight=args.recipient_low_confidence_loss_weight,
                recipient_confidence_curriculum_epochs=args.recipient_confidence_curriculum_epochs,
                recipient_train_augmentation=args.recipient_train_augmentation,
                recipient_only_fine_tune=args.recipient_only_fine_tune,
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
            )
            metrics = summary["by_field"]
            status_policy = summary.get("status_head_policy")
            status_display = (
                "review_only"
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
