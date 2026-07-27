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
from datetime import date
from collections.abc import Mapping


STRUCTURED_TARGET_SCHEMA_VERSION = 1

AMOUNT_AUX_FORMAT = "canonical_decimal_max_7_integer_digits_2_cents_v1"
TIME_AUX_FORMAT = "clock_h_mm_or_hh_mm_no_seconds_v1"
PAYMENT_CARD_TAIL_FORMAT = "visible_prefix_exact_ascii_4_digit_card_tail_v1"

# v6 deliberately keeps these display-format targets separate from the v5
# auxiliary targets above.  v5's ``amount_aux`` / ``time_aux`` are frozen
# compatibility contracts for already-trained checkpoints; teaching visible
# currency, grouping, signs, dates, and whitespace needs a new contract rather
# than silently broadening their meaning.
AMOUNT_DISPLAY_AUX_FORMAT = "visible_cny_amount_strict_v6"
TIME_DISPLAY_AUX_FORMAT = "visible_clock_or_datetime_strict_v6"
PAYMENT_BANK_PREFIX_FORMAT = "visible_payment_bank_prefix_v6"

AMOUNT_MAX_INTEGER_DIGITS = 7
AMOUNT_CENTS_DIGITS = 2
AMOUNT_RIGHT_ALIGNED_WIDTH = AMOUNT_MAX_INTEGER_DIGITS + AMOUNT_CENTS_DIGITS

PARENTHESIS_STYLE_ASCII = "ascii"
PARENTHESIS_STYLE_FULLWIDTH = "fullwidth"

AMOUNT_SIGN_CLASSES = ("positive", "negative")
TIME_DISPLAY_FORMAT_CLASSES = (
    "clock_h_mm",
    "clock_hh_mm",
    "clock_h_mm_ss",
    "clock_hh_mm_ss",
    "date_ymd_hh_mm",
    "date_ymd_hh_mm_ss",
)
TIME_DISPLAY_DIGIT_SLOTS = 14  # YYYYMMDDHHMMSS; masks distinguish shorter forms.

_AMOUNT_PATTERN = re.compile(r"^(?P<integer>0|[1-9][0-9]{0,6})\.(?P<cents>[0-9]{2})$")
_TIME_PATTERN = re.compile(r"^(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})$")
_PAYMENT_CARD_TAIL_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:"
    r"\((?P<ascii_tail>[0-9]{4})\)"
    r"|（(?P<fullwidth_tail>[0-9]{4})）"
    r")$"
)
_AMOUNT_DISPLAY_PATTERN = re.compile(
    r"^(?P<sign_before>-)?(?P<currency>[¥￥])?(?P<sign_after>-)?(?P<currency_space> ?)(?P<integer>"
    r"(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]*))\.(?P<cents>[0-9]{2})$"
)
_CLOCK_DISPLAY_PATTERN = re.compile(
    r"^(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})(?::(?P<second>[0-9]{2}))?$"
)
_DATETIME_DISPLAY_PATTERN = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2}) "
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})(?::(?P<second>[0-9]{2}))?$"
)


def parse_amount_display_target(value: str) -> dict[str, object] | None:
    """Parse a *visible* CNY amount without silently correcting it.

    This is intentionally stricter than the historical Paddle normaliser:
    exactly two cents are required, comma grouping must be conventional, and
    ambiguous ``O/I/l`` substitutions or rounding are never applied.  It
    accepts a bare amount as well as the visible CNY forms needed by Alipay,
    including ``-¥1,234.56`` and ``¥-1,234.56``.  The result retains both the
    original display text (for CTC) and a currency/grouping-free signed
    canonical decimal (for business comparison and the verifier head).
    """
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    match = _AMOUNT_DISPLAY_PATTERN.fullmatch(value)
    if match is None:
        return None
    sign_before = match.group("sign_before")
    sign_after = match.group("sign_after")
    currency = match.group("currency")
    currency_space = match.group("currency_space")
    # A sign has one unambiguous position: before the currency/number, or
    # directly after a currency symbol.  A bare numeric amount cannot contain
    # the latter position, and a space is visible CNY formatting only.
    if (sign_before is not None and sign_after is not None) or (sign_after is not None and currency is None):
        return None
    if currency_space and currency is None:
        return None
    integer = match.group("integer")
    cents = match.group("cents")
    integer_digits = integer.replace(",", "")
    sign = "negative" if sign_before is not None or sign_after is not None else "positive"
    # Negative zero has no useful business meaning and commonly signals a
    # misplaced OCR dash.  Keep it review-only instead of inventing a value.
    if sign == "negative" and int(integer_digits) == 0 and cents == "00":
        return None
    canonical_decimal = ("-" if sign == "negative" else "") + f"{integer_digits}.{cents}"
    padding = AMOUNT_RIGHT_ALIGNED_WIDTH - len(integer_digits + cents)
    if padding < 0:
        return None
    return {
        "schema_version": STRUCTURED_TARGET_SCHEMA_VERSION,
        "format": AMOUNT_DISPLAY_AUX_FORMAT,
        "visible_text": value,
        "canonical_decimal": canonical_decimal,
        "sign": sign,
        "currency": currency,
        "currency_space": bool(currency_space),
        "grouped_thousands": "," in integer,
        "integer_digits": integer_digits,
        "cents_digits": cents,
        "integer_digit_count": len(integer_digits),
        "right_aligned_width": AMOUNT_RIGHT_ALIGNED_WIDTH,
        "right_aligned_digits": [None] * padding + list(integer_digits + cents),
        "right_aligned_mask": [False] * padding + [True] * len(integer_digits + cents),
    }


def parse_time_display_target(value: str) -> dict[str, object] | None:
    """Parse a visible clock or Alipay-style dashed date-time exactly.

    Only documented templates are accepted.  A hyphen or space is therefore a
    meaningful part of an accepted date-time form, never an unconstrained
    extra character in a generic CTC label.  Full-width colons are normalised
    to their ASCII visual equivalent, consistent with the existing OCR path.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    text = value.replace("：", ":")
    match = _CLOCK_DISPLAY_PATTERN.fullmatch(text)
    if match is not None:
        hour_text = match.group("hour")
        minute_text = match.group("minute")
        second_text = match.group("second")
        hour = int(hour_text)
        minute = int(minute_text)
        second = int(second_text) if second_text is not None else None
        if hour > 23 or minute > 59 or (second is not None and second > 59):
            return None
        if second is None:
            format_name = "clock_h_mm" if len(hour_text) == 1 else "clock_hh_mm"
            digits = f"{hour:02d}{minute:02d}"
        else:
            format_name = "clock_h_mm_ss" if len(hour_text) == 1 else "clock_hh_mm_ss"
            digits = f"{hour:02d}{minute:02d}{second:02d}"
    else:
        match = _DATETIME_DISPLAY_PATTERN.fullmatch(text)
        if match is None:
            return None
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        second_text = match.group("second")
        second = int(second_text) if second_text is not None else None
        try:
            date(year, month, day)
        except ValueError:
            return None
        if hour > 23 or minute > 59 or (second is not None and second > 59):
            return None
        if second is None:
            format_name = "date_ymd_hh_mm"
            digits = f"{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}"
        else:
            format_name = "date_ymd_hh_mm_ss"
            digits = f"{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}"
    padding = TIME_DISPLAY_DIGIT_SLOTS - len(digits)
    if padding < 0:
        return None
    return {
        "schema_version": STRUCTURED_TARGET_SCHEMA_VERSION,
        "format": TIME_DISPLAY_AUX_FORMAT,
        "visible_text": text,
        "format_name": format_name,
        "canonical_digits": list(digits),
        "digit_slots": TIME_DISPLAY_DIGIT_SLOTS,
        "digit_mask": [True] * len(digits) + [False] * padding,
    }


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


def parse_payment_bank_prefix_target(value: str) -> dict[str, object] | None:
    """Expose an exact card prefix for v6's finite bank-name classifier."""
    card = parse_payment_card_tail_target(value)
    if card is None:
        return None
    prefix = card["prefix_text"]
    assert isinstance(prefix, str)  # Established by ``parse_payment_card_tail_target``.
    return {
        "schema_version": STRUCTURED_TARGET_SCHEMA_VERSION,
        "format": PAYMENT_BANK_PREFIX_FORMAT,
        "visible_prefix": prefix,
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
        "amount_display": {
            "format": AMOUNT_DISPLAY_AUX_FORMAT,
            "accepted_examples": ["¥1,234.56", "￥1,234.56", "-¥1,234.56", "¥-1,234.56"],
            "requires_exactly_two_cents": True,
            "requires_valid_thousands_grouping": True,
            "negative_zero_allowed": False,
            "canonical_value": "signed_decimal_without_currency_or_grouping",
        },
        "time_display": {
            "format": TIME_DISPLAY_AUX_FORMAT,
            "classes": list(TIME_DISPLAY_FORMAT_CLASSES),
            "digit_slots": TIME_DISPLAY_DIGIT_SLOTS,
            "accepted_examples": ["12:34", "01:02:03", "2026-07-27 12:34", "2026-07-27 12:34:56"],
            "requires_calendar_validation": True,
        },
        "payment_bank_prefix": {
            "format": PAYMENT_BANK_PREFIX_FORMAT,
            "source": "exact_visible_prefix_before_valid_four_digit_card_tail",
            "class_vocabulary_policy": "train_split_only; low_support_and_unknown_map_to___other__",
        },
    }


def is_structured_target(value: object, *, expected_format: str) -> bool:
    """Small defensive validator for downstream manifest consumers."""
    return (
        isinstance(value, Mapping)
        and value.get("schema_version") == STRUCTURED_TARGET_SCHEMA_VERSION
        and value.get("format") == expected_format
    )
