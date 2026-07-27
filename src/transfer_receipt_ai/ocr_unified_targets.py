"""Pure, auditable structured targets for the unified receipt reader.

The unified reader continues to retain the visible CTC text for every slot.
These helpers expose *additional* targets only when a value has an exact,
unambiguous structure that a specialised lightweight head can learn.  They do
not repair OCR, infer missing card digits, or normalise a one-digit hour into a
two-digit hour.

Keeping this module free of image, Paddle, Torch, and ONNX dependencies makes
the target contract testable before a costly teacher-label/training run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


STRUCTURED_TARGET_SCHEMA_VERSION = 1

AMOUNT_AUX_FORMAT = "canonical_decimal_max_7_integer_digits_2_cents_v1"
TIME_AUX_FORMAT = "clock_h_mm_or_hh_mm_no_seconds_v1"
PAYMENT_CARD_TAIL_FORMAT = "visible_prefix_exact_ascii_4_digit_card_tail_v1"

AMOUNT_MAX_INTEGER_DIGITS = 7
AMOUNT_CENTS_DIGITS = 2
AMOUNT_RIGHT_ALIGNED_WIDTH = AMOUNT_MAX_INTEGER_DIGITS + AMOUNT_CENTS_DIGITS

PARENTHESIS_STYLE_ASCII = "ascii"
PARENTHESIS_STYLE_FULLWIDTH = "fullwidth"

_AMOUNT_PATTERN = re.compile(r"^(?P<integer>0|[1-9][0-9]{0,6})\.(?P<cents>[0-9]{2})$")
_TIME_PATTERN = re.compile(r"^(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})$")
_PAYMENT_CARD_TAIL_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:"
    r"\((?P<ascii_tail>[0-9]{4})\)"
    r"|（(?P<fullwidth_tail>[0-9]{4})）"
    r")$"
)


def parse_amount_aux_target(value: str) -> dict[str, object] | None:
    """Parse a canonical CNY decimal without rounding or silently reformatting.

    The accepted form is ``0.00`` or a non-zero integer of up to seven ASCII
    digits followed by exactly two ASCII cents.  The fixed-width list is
    right-aligned and left-padded with JSON ``null`` so a future per-position
    digit head can apply a loss mask without treating a leading zero as absent.
    """
    if not isinstance(value, str):
        return None
    match = _AMOUNT_PATTERN.fullmatch(value)
    if match is None:
        return None
    integer_digits = match.group("integer")
    cents_digits = match.group("cents")
    digits = integer_digits + cents_digits
    padding = AMOUNT_RIGHT_ALIGNED_WIDTH - len(digits)
    if padding < 0:  # Defensive only; the regular expression already prevents it.
        return None
    return {
        "schema_version": STRUCTURED_TARGET_SCHEMA_VERSION,
        "format": AMOUNT_AUX_FORMAT,
        "canonical_decimal": value,
        "integer_digits": integer_digits,
        "cents_digits": cents_digits,
        "integer_digit_count": len(integer_digits),
        "right_aligned_width": AMOUNT_RIGHT_ALIGNED_WIDTH,
        "right_aligned_digits": [None] * padding + list(digits),
        "right_aligned_mask": [False] * padding + [True] * len(digits),
        "right_aligned_digit_count": len(digits),
    }


def parse_time_aux_target(value: str) -> dict[str, object] | None:
    """Parse only a valid visible ``H:MM`` or ``HH:MM`` clock value.

    A full-width colon is converted to the equivalent ASCII separator, which
    mirrors the existing pseudo-label normalisation.  Crucially, the hour text
    itself is retained verbatim: ``1:44`` remains ``1:44`` rather than being
    padded to ``01:44``.  Seconds and invalid clock ranges are rejected.
    """
    if not isinstance(value, str):
        return None
    text = value.replace("：", ":")
    match = _TIME_PATTERN.fullmatch(text)
    if match is None:
        return None
    hour_text = match.group("hour")
    minute_text = match.group("minute")
    if int(hour_text) > 23 or int(minute_text) > 59:
        return None
    return {
        "schema_version": STRUCTURED_TARGET_SCHEMA_VERSION,
        "format": TIME_AUX_FORMAT,
        "text": text,
        "hour_text": hour_text,
        "minute_text": minute_text,
        "hour_width": len(hour_text),
    }


def parse_payment_card_tail_target(value: str) -> dict[str, object] | None:
    """Split an exact visible card value into its prefix and four-digit tail.

    Only a final matched ASCII ``(...)`` or full-width ``（...）`` pair is
    accepted, with exactly four *ASCII* digits inside and a non-empty prefix.
    Full-width digits, mixed parentheses, extra trailing text, and short or
    long tails all return ``None``.  Returning ``None`` means "no specialised
    target"; callers must retain the original visible CTC text unchanged.
    """
    if not isinstance(value, str):
        return None
    match = _PAYMENT_CARD_TAIL_PATTERN.fullmatch(value)
    if match is None:
        return None
    prefix_text = match.group("prefix")
    # A separator between the prefix and suffix is visible text, not merely
    # formatting noise.  Reject it instead of dropping it and producing a
    # recomposed value that no longer exactly matches the CTC target.
    if not prefix_text or not prefix_text.strip() or prefix_text != prefix_text.strip():
        return None
    ascii_tail = match.group("ascii_tail")
    fullwidth_tail = match.group("fullwidth_tail")
    if ascii_tail is not None:
        card_tail = ascii_tail
        parentheses = PARENTHESIS_STYLE_ASCII
    elif fullwidth_tail is not None:
        card_tail = fullwidth_tail
        parentheses = PARENTHESIS_STYLE_FULLWIDTH
    else:  # Defensive only: one alternation branch must have matched.
        return None
    return {
        "schema_version": STRUCTURED_TARGET_SCHEMA_VERSION,
        "format": PAYMENT_CARD_TAIL_FORMAT,
        "visible_text": value,
        "prefix_text": prefix_text,
        "card_tail": card_tail,
        "parentheses": parentheses,
    }


def recompose_payment_card_tail_target(
    *, prefix_text: str, card_tail: str, parentheses: str
) -> str | None:
    """Return a validated visible payment value from specialised predictions.

    This is deliberately strict so a runtime can use it only after both heads
    pass their acceptance checks.  It is also a small shared oracle for tests
    that verify the parsed target round-trips to the original visible value.
    """
    if not isinstance(prefix_text, str) or not prefix_text or not prefix_text.strip():
        return None
    if not isinstance(card_tail, str) or re.fullmatch(r"[0-9]{4}", card_tail) is None:
        return None
    if parentheses == PARENTHESIS_STYLE_ASCII:
        value = f"{prefix_text}({card_tail})"
    elif parentheses == PARENTHESIS_STYLE_FULLWIDTH:
        value = f"{prefix_text}（{card_tail}）"
    else:
        return None
    # Reuse the parser as a final guard against drift between the two helpers.
    parsed = parse_payment_card_tail_target(value)
    return value if parsed is not None else None


def structured_target_config() -> dict[str, object]:
    """Return a JSON-ready description of every optional structured target."""
    return {
        "schema_version": STRUCTURED_TARGET_SCHEMA_VERSION,
        "amount_aux": {
            "format": AMOUNT_AUX_FORMAT,
            "max_integer_digits": AMOUNT_MAX_INTEGER_DIGITS,
            "cents_digits": AMOUNT_CENTS_DIGITS,
            "right_aligned_width": AMOUNT_RIGHT_ALIGNED_WIDTH,
            "left_padding": "json_null_with_mask_false",
            "right_aligned_layout": "integer_digits_then_cents",
            "accepted_decimal": "^(?:0|[1-9][0-9]{0,6})\\.[0-9]{2}$",
        },
        "time_aux": {
            "format": TIME_AUX_FORMAT,
            "accepted_time": "H:MM or HH:MM",
            "hour_range": [0, 23],
            "minute_range": [0, 59],
            "seconds_allowed": False,
            "preserve_hour_width": True,
            "fullwidth_colon_normalized_to_ascii": True,
        },
        "payment_card_tail": {
            "format": PAYMENT_CARD_TAIL_FORMAT,
            "tail_digits": 4,
            "tail_character_set": "ASCII 0-9",
            "parentheses": [PARENTHESIS_STYLE_ASCII, PARENTHESIS_STYLE_FULLWIDTH],
            "accepted_suffix_forms": ["(dddd)", "（dddd）"],
            "requires_nonempty_prefix": True,
            "requires_suffix_at_end": True,
        },
    }


def is_structured_target(value: object, *, expected_format: str) -> bool:
    """Small defensive validator for downstream manifest consumers."""
    return (
        isinstance(value, Mapping)
        and value.get("schema_version") == STRUCTURED_TARGET_SCHEMA_VERSION
        and value.get("format") == expected_format
    )
