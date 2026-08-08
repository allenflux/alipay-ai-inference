#!/usr/bin/env python3
"""Probe frozen formal recipient failures against truth without running OCR.

This is deliberately an analysis-only reader.  It accepts the atomic output of
``receipt-mlnet-hybrid-pilot-diagnose.py`` and publishes a new, hash-bound
directory.  It never changes the diagnostic input and never invokes an OCR
runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
from uuid import uuid4


INPUT_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_diagnostic_summary_v1"
INPUT_FINDING_KIND = "receipt_mlnet_hybrid_failure_diagnostic_finding_v1"
OUTPUT_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_truth_probe_summary_v1"
OUTPUT_FINDING_KIND = "receipt_mlnet_hybrid_failure_truth_probe_finding_v1"
RECIPIENT_MISSING_FAILURE = "hybrid recipient candidate missing"
FORMAL_RECORDS = 10016
FORMAL_FAILURES = 204
ATTEMPTS = ("first", "retry", "right_value")
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
LINE_MARKER = re.compile(rf"(?:^|,)(?P<index>\d+):(?P<confidence>{FLOAT_PATTERN}):")
FAILURE_REASON_PATTERN = re.compile(
    r"^anchored_or_alternative_parse_failed;"
    r"alternative_envelope=(?P<envelope>True|False);"
    r"first=(?P<first>.*);"
    r"retry=(?P<retry>.*);"
    r"right_value=(?P<right_value>.*)$",
    re.DOTALL,
)
ATTEMPT_PATTERN = re.compile(
    r"^line_count=(?P<line_count>\d+),"
    r"alternative_route=(?P<alternative_route>[^,;]+),"
    r"geometry=(?P<geometry>True|False|not_evaluated),"
    r"lines=\[(?P<lines>.*)\]$",
    re.DOTALL,
)
TIME_PATTERN = re.compile(r"(?<!\d)\d{1,2}[:：]\d{2}(?::\d{2})?(?!\d)")
PURE_AMOUNT_PATTERN = re.compile(
    r"^[¥￥$]?\s*(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?\s*(?:元)?$"
)
NEGATIVE_TOKENS = (
    "金额",
    "付款",
    "收款",
    "支付",
    "转账",
    "转帐",
    "成功",
    "失败",
    "处理中",
    "待处理",
    "时间",
    "订单",
    "活动",
    "优惠",
    "奖励",
    "红包",
    "积分",
    "充值",
    "商品",
    "交易状态",
    "交易单号",
    "流水号",
    "银行",
    "银行卡",
    "储蓄卡",
    "信用卡",
    "借记卡",
    "银联",
    "支付宝",
    "微信",
    "余额",
    "花呗",
    "尾号",
    "合计",
    "总计",
    "实付",
    "应付",
    "人民币",
)
# Exact keys observed for the pinyin recipient-row label in frozen formal
# evidence. Do not broaden this into a generic ASCII-name filter.
RECIPIENT_LABEL_PINYIN_KEYS = frozenset(
    ("shoukuanfang", "shoukuanting", "shoukudnfang")
)
# Exact, whole-line normalized UI labels. Deliberately no substring matching
# or edit distance: unrelated opaque ASCII payee names stay eligible.
ASCII_UI_LINE_KEYS = frozenset(
    (
        "amount",
        "amountdue",
        "balance",
        "bankcard",
        "creditcard",
        "debitcard",
        "discount",
        "failed",
        "failure",
        "order",
        "orderid",
        "ordernumber",
        "payee",
        "payment",
        "paymentfailed",
        "paymentfailure",
        "paymentmethod",
        "paymentprocessing",
        "paymentstatus",
        "paymentsuccess",
        "paymentsuccessful",
        "pending",
        "processing",
        "recipient",
        "recipientaccount",
        "recipientnumber",
        "status",
        "success",
        "successful",
        "time",
        "transactionfailed",
        "transactionfailure",
        "transactionid",
        "transactionnumber",
        "transactionprocessing",
        "transactionstatus",
        "transactionsuccess",
        "transactionsuccessful",
        "transfer",
        "transferfailed",
        "transferfailure",
        "transferprocessing",
        "transferstatus",
        "transfersuccess",
        "transfersuccessful",
    )
)
CURRENCY_CODE_PATTERN = re.compile(r"(?i)(?:^|[^a-z])(?:cny|rmb|usd|hkd|eur|gbp|jpy)(?:$|[^a-z])")
ALLOWED_PUNCTUATION = frozenset("*＊()（）·•&＆_-—.．/")
MINIMUM_STRICT_LINE_CONFIDENCE = 0.80
MINIMUM_RECIPIENT_DETECTOR_SCORE = 0.68


class ProbeError(ValueError):
    """Raised when frozen diagnostic evidence violates the probe contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, description: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ProbeError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise ProbeError(f"{description} is not a file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ProbeError(f"{description} must be non-empty: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": size,
    }


def _assert_identities_current(identities: Mapping[str, Mapping[str, Any]]) -> None:
    for description, expected in identities.items():
        path = Path(str(expected.get("path") or ""))
        actual = _identity(path, description=description)
        if actual != dict(expected):
            raise ProbeError(f"{description} changed while the probe was reading it")


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _source_key(value: object) -> str:
    text = str(value or "").replace("\\", "/")
    # Formal evidence is produced on Windows.  Case-folding also makes a
    # drive-letter spelling change unable to bypass the uniqueness check.
    return os.path.normpath(text).replace("\\", "/").casefold()


def _require_int(payload: Mapping[str, Any], key: str, expected: int) -> None:
    value = payload.get(key)
    if isinstance(value, bool) or value != expected:
        raise ProbeError(f"summary.{key} must equal {expected}, got {value!r}")


def _parse_lines(body: str, *, name: str, line_count: int) -> list[dict[str, Any]]:
    if not body:
        if line_count != 0:
            raise ProbeError(f"{name} declares {line_count} lines but has no line evidence")
        return []
    matches = list(LINE_MARKER.finditer(body))
    if not matches or matches[0].start() != 0:
        raise ProbeError(f"{name} line evidence does not start with index/confidence/text")
    lines: list[dict[str, Any]] = []
    for position, match in enumerate(matches):
        index = int(match.group("index"))
        if index != position:
            raise ProbeError(
                f"{name} line indices must be contiguous from zero, got {index} at {position}"
            )
        confidence = float(match.group("confidence"))
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ProbeError(f"{name}[{index}] confidence is outside [0, 1]")
        end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        text = body[match.end() : end]
        lines.append(
            {
                "index": index,
                "confidence": confidence,
                "text": text,
                "normalized_text": _clean(text),
            }
        )
    if len(lines) != line_count:
        raise ProbeError(
            f"{name} line_count mismatch: declared={line_count} parsed={len(lines)}"
        )
    return lines


def _parse_attempt(section: str, *, name: str) -> dict[str, Any]:
    if section == "none":
        return {
            "name": name,
            "present": False,
            "line_count": None,
            "alternative_route": "none",
            "geometry": "not_evaluated",
            "lines": [],
        }
    match = ATTEMPT_PATTERN.fullmatch(section)
    if match is None:
        raise ProbeError(f"{name} evidence does not match BuildFailureReason format")
    line_count = int(match.group("line_count"))
    return {
        "name": name,
        "present": True,
        "line_count": line_count,
        "alternative_route": match.group("alternative_route"),
        "geometry": match.group("geometry"),
        "lines": _parse_lines(match.group("lines"), name=name, line_count=line_count),
    }


def _empty_attempt_from_raw(finding: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    prefix = "right_value" if name == "right_value" else name
    raw = finding.get(f"{prefix}_raw")
    count = finding.get(f"{prefix}_line_count")
    present = count is not None or raw is not None
    if count is not None and (isinstance(count, bool) or count != 0):
        raise ProbeError(f"ocr_empty {name}_line_count must be zero or null")
    if _clean(raw):
        raise ProbeError(f"ocr_empty {name}_raw must be empty or null")
    return {
        "name": name,
        "present": present,
        "line_count": 0 if present else None,
        "alternative_route": "unreported",
        "geometry": "unreported",
        "lines": [],
    }


def _parse_failure_reason(
    finding: Mapping[str, Any], *, source: str
) -> tuple[str, bool | None, dict[str, dict[str, Any]]]:
    reason = finding.get("ppocr_failure_reason")
    if reason is None:
        # One frozen formal failure has no recipient diagnostic at all.  Accept
        # only that fully unreported shape; partial evidence must not silently
        # become an empty OCR result or a pseudo-truth candidate.
        unreported_fields = (
            "ppocr_route",
            "third_route",
            "first_raw",
            "first_line_count",
            "retry_raw",
            "retry_line_count",
            "right_value_raw",
            "right_value_line_count",
            "right_value_line_confidences",
        )
        reported = [name for name in unreported_fields if finding.get(name) is not None]
        if reported:
            raise ProbeError(
                f"finding {source!r} has no ppocr_failure_reason but reports "
                + ", ".join(reported)
            )
        return (
            "unreported",
            None,
            {
                name: {
                    "name": name,
                    "present": False,
                    "line_count": None,
                    "alternative_route": "unreported",
                    "geometry": "unreported",
                    "lines": [],
                }
                for name in ATTEMPTS
            },
        )
    if reason == "ocr_empty":
        return (
            "ocr_empty",
            None,
            {
                name: _empty_attempt_from_raw(finding, name=name)
                for name in ATTEMPTS
            },
        )
    if not isinstance(reason, str) or not reason:
        raise ProbeError(f"finding {source!r} has no ppocr_failure_reason")
    match = FAILURE_REASON_PATTERN.fullmatch(reason)
    if match is None:
        raise ProbeError(
            f"finding {source!r} ppocr_failure_reason does not match BuildFailureReason"
        )
    attempts = {
        name: _parse_attempt(match.group(name), name=name) for name in ATTEMPTS
    }
    return (
        "anchored_or_alternative_parse_failed",
        match.group("envelope") == "True",
        attempts,
    )


def _cross_check_attempt(
    finding: Mapping[str, Any], attempt: Mapping[str, Any], *, source: str
) -> None:
    name = str(attempt["name"])
    prefix = "right_value" if name == "right_value" else name
    raw = finding.get(f"{prefix}_raw")
    count = finding.get(f"{prefix}_line_count")
    if attempt["present"] is False:
        if raw is not None or count is not None:
            raise ProbeError(
                f"finding {source!r} {name}=none disagrees with raw/line_count"
            )
        return
    if isinstance(count, bool) or not isinstance(count, int):
        raise ProbeError(f"finding {source!r} {name}_line_count must be an integer")
    if count != attempt["line_count"]:
        raise ProbeError(
            f"finding {source!r} {name} line_count disagrees with failure reason"
        )
    if not isinstance(raw, str):
        raise ProbeError(f"finding {source!r} {name}_raw must be a string")
    joined = " ".join(
        line["normalized_text"]
        for line in attempt["lines"]
        if line["normalized_text"]
    )
    if _clean(raw) != joined:
        raise ProbeError(
            f"finding {source!r} {name}_raw disagrees with parsed line text"
        )
    if name == "right_value":
        confidences = finding.get("right_value_line_confidences")
        if confidences is not None:
            if not isinstance(confidences, list) or len(confidences) != count:
                raise ProbeError(
                    f"finding {source!r} right_value confidences/count disagree"
                )
            for index, (actual, line) in enumerate(
                zip(confidences, attempt["lines"], strict=True)
            ):
                if (
                    isinstance(actual, bool)
                    or not isinstance(actual, (int, float))
                    or not math.isfinite(float(actual))
                    or not math.isclose(
                        float(actual), line["confidence"], rel_tol=0.0, abs_tol=1e-6
                    )
                ):
                    raise ProbeError(
                        f"finding {source!r} right_value confidence[{index}] disagrees"
                    )


def _reference_analysis(
    attempts: Mapping[str, Mapping[str, Any]], reference: str | None
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    per_attempt: dict[str, dict[str, Any]] = {}
    exact_attempts: list[str] = []
    exact_positions: list[str] = []
    for name in ATTEMPTS:
        exact_indices: list[int] = []
        substring_indices: list[int] = []
        for line in attempts[name]["lines"]:
            text = line["normalized_text"]
            if reference is not None and text == reference:
                exact_indices.append(line["index"])
                exact_positions.append(f"{name}:{line['index']}")
            if reference is not None and reference in text:
                substring_indices.append(line["index"])
        if exact_indices:
            exact_attempts.append(name)
        per_attempt[name] = {
            "exact_line": bool(exact_indices),
            "exact_line_indices": exact_indices,
            "substring": bool(substring_indices),
            "substring_line_indices": substring_indices,
        }
    return per_attempt, exact_attempts, exact_positions


def _shadow_line_allowed(value: str) -> tuple[bool, str]:
    if not value:
        return False, "empty"
    visible = value.replace(" ", "")
    if len(visible) < 2 or len(visible) > 48:
        return False, "length"
    folded = visible.casefold()
    if any(token.casefold() in folded for token in NEGATIVE_TOKENS):
        return False, "negative_token"
    if folded in RECIPIENT_LABEL_PINYIN_KEYS or folded in ASCII_UI_LINE_KEYS:
        return False, "negative_token"
    if "¥" in visible or "￥" in visible or PURE_AMOUNT_PATTERN.fullmatch(value):
        return False, "amount"
    if CURRENCY_CODE_PATTERN.search(value):
        return False, "amount"
    if TIME_PATTERN.search(value):
        return False, "time"
    has_letter = False
    for character in visible:
        category = unicodedata.category(character)
        if category.startswith("L"):
            has_letter = True
            continue
        if category.startswith("N") or character in ALLOWED_PUNCTUATION:
            continue
        return False, "character_contract"
    if not has_letter:
        return False, "no_letter"
    return True, "accepted"


def _raw_consensus(
    attempts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    crops_by_line: dict[str, dict[str, float]] = defaultdict(dict)
    for name in ATTEMPTS:
        for line in attempts[name]["lines"]:
            text = line["normalized_text"]
            if not text:
                continue
            confidence = float(line["confidence"])
            crops_by_line[text][name] = max(
                crops_by_line[text].get(name, 0.0), confidence
            )
    candidates = [
        {
            "candidate": candidate,
            "crops": [name for name in ATTEMPTS if name in crops],
            "crop_confidences": {
                name: crops[name] for name in ATTEMPTS if name in crops
            },
            "minimum_confidence": min(crops.values()),
        }
        for candidate, crops in sorted(crops_by_line.items())
        if len(crops) >= 2
    ]
    return {
        "candidates": candidates,
        "state": (
            "none"
            if not candidates
            else "one"
            if len(candidates) == 1
            else "multiple"
        ),
    }


def _strict_runtime_shadow(
    attempts: Mapping[str, Mapping[str, Any]],
    *,
    recipient_score: float | None,
    geometry_reasons: Sequence[str],
    alternative_envelope: bool | None,
) -> dict[str, Any]:
    # Truth is intentionally not an argument: candidate derivation must be
    # identical in production-shadow analysis regardless of the reference.
    crops_by_line: dict[str, dict[str, float]] = defaultdict(dict)
    rejection_reasons: Counter[str] = Counter()
    for name in ATTEMPTS:
        best_by_text: dict[str, float] = {}
        for line in attempts[name]["lines"]:
            text = line["normalized_text"]
            allowed, reason = _shadow_line_allowed(text)
            if not allowed:
                rejection_reasons[reason] += 1
                continue
            confidence = float(line["confidence"])
            if confidence < MINIMUM_STRICT_LINE_CONFIDENCE:
                rejection_reasons["low_confidence"] += 1
                continue
            best_by_text[text] = max(best_by_text.get(text, 0.0), confidence)
        for text, confidence in best_by_text.items():
            crops_by_line[text][name] = confidence
    eligible = [
        {
            "candidate": candidate,
            "crops": [name for name in ATTEMPTS if name in crops],
            "crop_confidences": {
                name: crops[name] for name in ATTEMPTS if name in crops
            },
            "minimum_confidence": min(crops.values()),
        }
        for candidate, crops in sorted(crops_by_line.items())
        if len(crops) >= 2
    ]
    global_gate_failures: list[str] = []
    if recipient_score is None:
        global_gate_failures.append("recipient_score_not_available")
    elif recipient_score < MINIMUM_RECIPIENT_DETECTOR_SCORE:
        global_gate_failures.append("recipient_score_below_0.68")
    if geometry_reasons:
        global_gate_failures.append("ordinary_25pct_geometry_not_verified")
    if alternative_envelope is not True:
        global_gate_failures.append("alternative_envelope_not_verified")
    selected: Mapping[str, Any] | None = None
    selected_consensus_route: str | None = None
    if len(eligible) == 1:
        selected = eligible[0]
        selected_consensus_route = "independent_crop_exact_consensus"
    elif len(eligible) > 1:
        dominant = [
            candidate
            for candidate in eligible
            if len(candidate["crops"]) == len(ATTEMPTS)
        ]
        if len(dominant) == 1:
            selected = dominant[0]
            selected_consensus_route = (
                "independent_crop_dominant_three_crop_consensus"
            )
    if selected is not None and not global_gate_failures:
        candidate = selected["candidate"]
        crops = selected["crops"]
        return {
            "candidate": candidate,
            "state": "candidate",
            "runtime_route": selected_consensus_route,
            "selected_consensus_route": selected_consensus_route,
            "consensus_crops": crops,
            "minimum_confidence": selected["minimum_confidence"],
            "eligible_candidates": eligible,
            "rejected_line_occurrences": dict(sorted(rejection_reasons.items())),
            "pseudo_truth_source": "ppocr_independent_crop_exact_consensus",
            "external_truth": False,
            "truth_used_for_analysis_only": True,
            "formal_delivery_gate": False,
            "recipient_detector_score": recipient_score,
            "global_gate_failures": [],
        }
    return {
        "candidate": None,
        "state": (
            "unresolved"
            if not eligible
            else "rejected_by_global_gate"
            if selected is not None and global_gate_failures
            else "ambiguous"
            if len(eligible) > 1
            else "rejected_by_global_gate"
        ),
        "runtime_route": None,
        "selected_consensus_route": selected_consensus_route,
        "consensus_crops": [],
        "minimum_confidence": None,
        "eligible_candidates": eligible,
        "rejected_line_occurrences": dict(sorted(rejection_reasons.items())),
        "pseudo_truth_source": "ppocr_independent_crop_exact_consensus",
        "external_truth": False,
        "truth_used_for_analysis_only": True,
        "formal_delivery_gate": False,
        "recipient_detector_score": recipient_score,
        "global_gate_failures": global_gate_failures,
    }


def _group_key_attempt(attempt: Mapping[str, Any]) -> str:
    return (
        f"alternative_route={attempt['alternative_route']}|"
        f"geometry={attempt['geometry']}"
    )


def _line_count_key(attempts: Mapping[str, Mapping[str, Any]]) -> str:
    return "|".join(
        f"{name}={attempts[name]['line_count'] if attempts[name]['present'] else 'none'}"
        for name in ATTEMPTS
    )


def _geometry_key(attempts: Mapping[str, Mapping[str, Any]]) -> str:
    return "|".join(f"{name}={attempts[name]['geometry']}" for name in ATTEMPTS)


def _crop_combo(crops: Iterable[str], *, empty: str = "none") -> str:
    crop_set = set(crops)
    selected = [name for name in ATTEMPTS if name in crop_set]
    return "+".join(selected) if selected else empty


def _global_gate_failure_key(failures: Sequence[str]) -> str:
    return "+".join(failures) if failures else "none"


def _geometry_reason_key(reasons: Sequence[str]) -> str:
    if not reasons:
        return "verified"
    return "failed:" + "+".join(sorted(set(reasons)))


def _score_gate_key(score: float | None) -> str:
    if score is None:
        return "unreported"
    if score < MINIMUM_RECIPIENT_DETECTOR_SCORE:
        return "below_0.68"
    return "verified_0.68_plus"


LAYOUT_GEOMETRY_REASONS = frozenset(
    (
        "recipient_left_edge",
        "recipient_right_edge",
        "recipient_width",
        "recipient_height",
        "amount_before_recipient",
        "recipient_before_payment",
        "amount_edge_overlap",
        "payment_edge_overlap",
    )
)


def _geometry_reason_category(reason: str) -> str:
    if "_score_missing" in reason or "_score_below_" in reason:
        return "detector_score"
    if reason.endswith("_box_invalid") or reason.endswith("_box_outside_source"):
        return "detector_box"
    if (
        reason
        in {
            "geometry_missing",
            "source_size_missing_or_invalid",
            "rectified_size_missing_or_invalid",
            "H_original_to_rectified_missing_or_invalid",
        }
        or reason.endswith("_box_projection_invalid")
        or reason.endswith("_box_outside_rectified")
    ):
        return "rectification_or_projection"
    if reason in LAYOUT_GEOMETRY_REASONS:
        return "layout_relation"
    return "unclassified"


def _global_gate_repair_surfaces(finding: Mapping[str, Any]) -> list[str]:
    surfaces: set[str] = set()
    if finding["alternative_envelope"] is not True:
        surfaces.add("alternative_envelope_generation_or_verification")
    score = finding["recipient_detector_score"]
    if score is None or float(score) < MINIMUM_RECIPIENT_DETECTOR_SCORE:
        surfaces.add("detector_score")
    for reason in finding["geometry_reasons"]:
        category = _geometry_reason_category(str(reason))
        surfaces.add(
            {
                "detector_score": "detector_score",
                "detector_box": "detector_box",
                "layout_relation": "detector_layout_geometry",
                "rectification_or_projection": "rectification_or_projection",
                "unclassified": "unclassified_geometry_evidence",
            }[category]
        )
    return sorted(surfaces)


def _raw_cardinality_key(state: str) -> str:
    return "unique" if state == "one" else state


def _raw_candidate_blockers(raw_consensus: Mapping[str, Any]) -> list[str]:
    blockers: set[str] = set()
    for candidate in raw_consensus["candidates"]:
        value = str(candidate["candidate"])
        allowed, reason = _shadow_line_allowed(value)
        if not allowed:
            blockers.add(f"line_contract:{reason}")
            continue
        confident_crops = sum(
            float(confidence) >= MINIMUM_STRICT_LINE_CONFIDENCE
            for confidence in candidate["crop_confidences"].values()
        )
        if confident_crops < 2:
            blockers.add("insufficient_high_confidence_crop_agreement")
            continue
        # This is an executable consistency assertion rather than a fallback:
        # a raw candidate satisfying both strict line gates must have appeared
        # in strict_runtime_shadow.eligible_candidates.
        blockers.add("unexpected_strict_eligibility_gap")
    return sorted(blockers)


def _unresolved_primary_blocker(
    *,
    failure_reason_type: str,
    attempts: Mapping[str, Mapping[str, Any]],
    raw_consensus: Mapping[str, Any],
    strict_state: str,
) -> str | None:
    if strict_state != "unresolved":
        return None
    if failure_reason_type == "unreported":
        return "failure_evidence_unreported"
    if failure_reason_type == "ocr_empty":
        return "ocr_empty"
    if raw_consensus["state"] == "none":
        nonempty_lines = sum(
            bool(line["normalized_text"])
            for attempt in attempts.values()
            for line in attempt["lines"]
        )
        return (
            "no_nonempty_ocr_line"
            if nonempty_lines == 0
            else "no_exact_cross_crop_line_consensus"
        )
    blockers = _raw_candidate_blockers(raw_consensus)
    if "unexpected_strict_eligibility_gap" in blockers:
        raise ProbeError(
            "raw consensus passed strict line gates but strict shadow reported unresolved"
        )
    return "raw_consensus_filtered:" + "+".join(blockers)


def _remaining_failure_cluster(
    *,
    failure_reason_type: str,
    alternative_envelope: bool | None,
    attempts: Mapping[str, Mapping[str, Any]],
    geometry_reasons: Sequence[str],
    recipient_score: float | None,
    raw_consensus: Mapping[str, Any],
    strict_shadow: Mapping[str, Any],
) -> dict[str, Any] | None:
    strict_state = str(strict_shadow["state"])
    if strict_state == "candidate":
        return None
    raw_candidates = raw_consensus["candidates"]
    eligible_candidates = strict_shadow["eligible_candidates"]
    raw_state = str(raw_consensus["state"])
    global_gate_failures = list(strict_shadow["global_gate_failures"])
    eligible_count = len(eligible_candidates)
    ambiguous_count = eligible_count if strict_state == "ambiguous" else 0
    envelope_key = (
        "unreported"
        if alternative_envelope is None
        else str(alternative_envelope).lower()
    )
    geometry_key = _geometry_reason_key(geometry_reasons)
    score_key = _score_gate_key(recipient_score)
    unresolved_blocker = _unresolved_primary_blocker(
        failure_reason_type=failure_reason_type,
        attempts=attempts,
        raw_consensus=raw_consensus,
        strict_state=strict_state,
    )
    return {
        "strict_state": strict_state,
        "global_gate_failures": global_gate_failures,
        "global_gate_failures_combination": _global_gate_failure_key(
            global_gate_failures
        ),
        "eligible_candidate_count": eligible_count,
        "ambiguous_candidate_count": ambiguous_count,
        "unresolved_primary_blocker": unresolved_blocker,
        "raw_consensus_state": raw_state,
        "raw_candidate_count": len(raw_candidates),
        "strict_filtered_candidate_count": len(raw_candidates) - eligible_count,
        "raw_vs_strict": (
            f"raw={_raw_cardinality_key(raw_state)}|strict={strict_state}|"
            f"raw_candidates={len(raw_candidates)}|eligible={eligible_count}"
        ),
        "alternative_envelope_geometry_score": (
            f"envelope={envelope_key}|geometry={geometry_key}|score={score_key}"
        ),
    }


def _analyze_finding(payload: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if payload.get("schema_version") != 1 or payload.get("kind") != INPUT_FINDING_KIND:
        raise ProbeError(f"finding[{index}] has an unsupported schema/kind")
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ProbeError(f"finding[{index}].source must be non-empty")
    failures = payload.get("failures")
    if failures != [RECIPIENT_MISSING_FAILURE]:
        raise ProbeError(f"finding {source!r} is not recipient-missing-only")
    if payload.get("recipient_candidate") is not None:
        raise ProbeError(f"finding {source!r} recipient_candidate must be null")
    reference = payload.get("reference")
    if not isinstance(reference, Mapping):
        raise ProbeError(f"finding {source!r} reference must be an object")
    recipient_text = _clean(reference.get("recipient"))
    recipient = recipient_text or None
    reason_type, envelope, attempts = _parse_failure_reason(payload, source=source)
    for attempt in attempts.values():
        _cross_check_attempt(payload, attempt, source=source)
    recipient_score_raw = payload.get("recipient_score")
    if recipient_score_raw is None and reason_type == "unreported":
        recipient_score = None
    elif (
        isinstance(recipient_score_raw, bool)
        or not isinstance(recipient_score_raw, (int, float))
        or not math.isfinite(float(recipient_score_raw))
        or float(recipient_score_raw) < 0.0
        or float(recipient_score_raw) > 1.0
    ):
        raise ProbeError(f"finding {source!r} recipient_score must be within [0, 1]")
    else:
        recipient_score = float(recipient_score_raw)
    geometry_reasons = payload.get("geometry_reasons")
    if (
        not isinstance(geometry_reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in geometry_reasons)
    ):
        raise ProbeError(f"finding {source!r} geometry_reasons must be a string array")
    per_attempt, exact_attempts, exact_positions = _reference_analysis(
        attempts, recipient
    )
    raw_consensus = _raw_consensus(attempts)
    shadow = _strict_runtime_shadow(
        attempts,
        recipient_score=recipient_score,
        geometry_reasons=geometry_reasons,
        alternative_envelope=envelope,
    )
    shadow_value = shadow["candidate"]
    truth_state = (
        "not_available"
        if recipient is None
        else "absent"
        if shadow_value is None
        else "exact"
        if shadow_value == recipient
        else "wrong"
    )
    shadow["truth_outcome"] = truth_state
    truth_free_shadow = {
        key: value for key, value in shadow.items() if key != "truth_outcome"
    }
    remaining_cluster = _remaining_failure_cluster(
        failure_reason_type=reason_type,
        alternative_envelope=envelope,
        attempts=attempts,
        geometry_reasons=geometry_reasons,
        recipient_score=recipient_score,
        raw_consensus=raw_consensus,
        strict_shadow=truth_free_shadow,
    )
    return {
        "schema_version": 1,
        "kind": OUTPUT_FINDING_KIND,
        "source": source,
        "reference_recipient": recipient,
        "external_reference_present": recipient is not None,
        "truth_used_for_analysis_only": True,
        "runtime_truth_lookup": False,
        "formal_delivery_gate": False,
        "failure_reason_type": reason_type,
        "alternative_envelope": envelope,
        "recipient_detector_score": recipient_score,
        "geometry_reasons": geometry_reasons,
        "attempts": attempts,
        "reference_by_attempt": per_attempt,
        "reference_exact_attempts": exact_attempts,
        "reference_exact_positions": exact_positions,
        "reference_exact_line_crop_consensus": len(exact_attempts) >= 2,
        "reference_exact_line_consensus_crops": exact_attempts
        if len(exact_attempts) >= 2
        else [],
        "raw_consensus": raw_consensus,
        "strict_runtime_shadow": shadow,
        "shadow_candidate_truth_free": truth_free_shadow,
        "paddle_teacher_consensus": dict(truth_free_shadow),
        "remaining_failure_cluster": remaining_cluster,
        "group_keys": {
            "alternative_envelope": "unreported" if envelope is None else str(envelope).lower(),
            "geometry": _geometry_key(attempts),
            "line_count_tuple": _line_count_key(attempts),
            "consensus_crop_combination": _crop_combo(
                shadow["consensus_crops"], empty=shadow["state"]
            ),
        },
    }


def _load_input(input_directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = input_directory / "summary.json"
    findings_path = input_directory / "findings.jsonl"
    identities = {
        "input_summary": _identity(summary_path, description="input summary"),
        "input_findings": _identity(findings_path, description="input findings"),
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if not isinstance(summary, Mapping):
        raise ProbeError("input summary must be an object")
    if summary.get("schema_version") != 1 or summary.get("kind") != INPUT_SUMMARY_KIND:
        raise ProbeError("input summary has an unsupported schema/kind")
    if summary.get("comparison_evaluation_mode") != "formal":
        raise ProbeError("input summary must be formal evidence")
    for key, expected in (
        ("comparison_records", FORMAL_RECORDS),
        ("invariant_failure_records", FORMAL_FAILURES),
        ("recipient_missing_records", FORMAL_FAILURES),
        ("recipient_missing_only_records", FORMAL_FAILURES),
        ("failed_records", FORMAL_FAILURES),
        ("non_missing_invariant_failure_records", 0),
        ("recipient_missing_with_additional_failures_records", 0),
    ):
        _require_int(summary, key, expected)
    if summary.get("by_comparator_failure") != [
        {"name": RECIPIENT_MISSING_FAILURE, "records": FORMAL_FAILURES}
    ]:
        raise ProbeError("input summary comparator failures are not exactly missing-only")
    findings: list[dict[str, Any]] = []
    source_keys: set[str] = set()
    for index, line in enumerate(
        findings_path.read_text(encoding="utf-8-sig").splitlines()
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ProbeError(f"finding[{index}] must be an object")
        analyzed = _analyze_finding(payload, index=index)
        source_key = _source_key(analyzed["source"])
        if source_key in source_keys:
            raise ProbeError(f"duplicate finding source: {analyzed['source']!r}")
        source_keys.add(source_key)
        findings.append(analyzed)
    if len(findings) != FORMAL_FAILURES:
        raise ProbeError(
            f"findings.jsonl must contain {FORMAL_FAILURES} records, got {len(findings)}"
        )
    _assert_identities_current(identities)
    return findings, {
        "input_summary": dict(summary),
        "source_evidence": identities,
    }


def _count_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"name": name, "records": records}
        for name, records in sorted(counter.items(), key=lambda item: item[0])
    ]


def _occurrence_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"name": name, "occurrences": occurrences}
        for name, occurrences in sorted(counter.items(), key=lambda item: item[0])
    ]


def _group_rows(groups: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "records": len(sources),
            "examples": sorted(sources, key=_source_key)[:3],
        }
        for name, sources in sorted(groups.items(), key=lambda item: item[0])
    ]


def _confidence_bucket(value: float) -> str:
    if value < 0.70:
        return "[0.50,0.70)"
    if value < 0.80:
        return "[0.70,0.80)"
    if value < 0.90:
        return "[0.80,0.90)"
    if value < 0.95:
        return "[0.90,0.95)"
    return "[0.95,1.00]"


def summarize(
    findings: Sequence[Mapping[str, Any]], *, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if len(findings) != FORMAL_FAILURES:
        raise ProbeError("cannot summarize a non-formal finding count")
    reference_attempt: dict[str, Counter[str]] = {
        name: Counter() for name in ATTEMPTS
    }
    exact_position = Counter()
    exact_consensus = Counter()
    shadow_truth = Counter()
    shadow_state = Counter()
    raw_consensus_state = Counter()
    teacher_confidence = Counter()
    teacher_crop_combinations = Counter()
    teacher_runtime_routes = Counter()
    first_alt_geometry = Counter()
    retry_alt_geometry = Counter()
    groups: dict[str, dict[str, list[str]]] = {
        "alternative_envelope": defaultdict(list),
        "geometry": defaultdict(list),
        "line_count_tuple": defaultdict(list),
        "consensus_crop_combination": defaultdict(list),
    }
    remaining_groups: dict[str, dict[str, list[str]]] = {
        "global_gate_failures_combination": defaultdict(list),
        "eligible_candidate_count": defaultdict(list),
        "ambiguous_candidate_count": defaultdict(list),
        "unresolved_primary_blocker": defaultdict(list),
        "raw_vs_strict": defaultdict(list),
        "alternative_envelope_geometry_score": defaultdict(list),
    }
    global_gate_groups: dict[str, dict[str, list[str]]] = {
        "global_gate_failures_combination": defaultdict(list),
        "geometry_reason_combination": defaultdict(list),
        "geometry_reason_category_combination": defaultdict(list),
        "geometry_reason_record_incidence": defaultdict(list),
        "alternative_envelope": defaultdict(list),
        "recipient_detector_score_gate": defaultdict(list),
        "alternative_envelope_geometry_score": defaultdict(list),
        "repair_surface_combination": defaultdict(list),
        "repair_surface_record_incidence": defaultdict(list),
    }
    unresolved_groups: dict[str, dict[str, list[str]]] = {
        "primary_filter_blocker": defaultdict(list),
        "raw_consensus_state": defaultdict(list),
        "raw_candidate_filter_reason_combination": defaultdict(list),
        "rejected_line_reason_record_incidence": defaultdict(list),
        "rejected_line_occurrence_signature": defaultdict(list),
        "failure_reason_type": defaultdict(list),
    }
    unresolved_rejected_line_occurrences: Counter[str] = Counter()
    global_gate_records = 0
    global_gate_selected_consensus_records = 0
    global_gate_single_eligible_candidate_records = 0
    unresolved_records = 0
    remaining_with_global_gate_failures = 0
    remaining_gate_overlay_groups: dict[str, dict[str, list[str]]] = {
        "strict_state_by_gate_presence": defaultdict(list),
        "strict_state_by_global_gate_failures_combination": defaultdict(list),
    }
    all_strict_states = Counter()
    all_failure_reason_types = Counter()
    remaining_records = 0
    external_reference_present = 0
    for finding in findings:
        source = str(finding["source"])
        if finding["external_reference_present"]:
            external_reference_present += 1
            for name in ATTEMPTS:
                reference = finding["reference_by_attempt"][name]
                exact_key = "exact_line" if reference["exact_line"] else "no_exact_line"
                substring_key = "substring" if reference["substring"] else "no_substring"
                reference_attempt[name][exact_key] += 1
                reference_attempt[name][substring_key] += 1
            exact_position.update(finding["reference_exact_positions"])
            exact_consensus.update(
                [
                    _crop_combo(
                        finding["reference_exact_line_consensus_crops"],
                        empty="no_2_of_3_consensus",
                    )
                ]
            )
        shadow = finding["shadow_candidate_truth_free"]
        all_strict_states.update([shadow["state"]])
        all_failure_reason_types.update([finding["failure_reason_type"]])
        raw_consensus_state.update([finding["raw_consensus"]["state"]])
        shadow_truth.update([finding["strict_runtime_shadow"]["truth_outcome"]])
        shadow_state.update([shadow["state"]])
        if shadow["state"] == "rejected_by_global_gate":
            global_gate_records += 1
            if len(shadow["eligible_candidates"]) == 1:
                global_gate_single_eligible_candidate_records += 1
            if shadow.get("selected_consensus_route") is not None:
                global_gate_selected_consensus_records += 1
            failures_key = _global_gate_failure_key(
                shadow["global_gate_failures"]
            )
            geometry_reasons = sorted(set(finding["geometry_reasons"]))
            geometry_key = _geometry_reason_key(geometry_reasons)
            geometry_categories = sorted(
                {_geometry_reason_category(reason) for reason in geometry_reasons}
            )
            geometry_category_key = (
                "+".join(geometry_categories) if geometry_categories else "verified"
            )
            envelope_key = (
                "unreported"
                if finding["alternative_envelope"] is None
                else str(finding["alternative_envelope"]).lower()
            )
            score_key = _score_gate_key(finding["recipient_detector_score"])
            combined_key = (
                f"envelope={envelope_key}|geometry={geometry_key}|score={score_key}"
            )
            repair_surfaces = _global_gate_repair_surfaces(finding)
            repair_surface_key = "+".join(repair_surfaces) or "none"
            for group_name, key in (
                ("global_gate_failures_combination", failures_key),
                ("geometry_reason_combination", geometry_key),
                ("geometry_reason_category_combination", geometry_category_key),
                ("alternative_envelope", envelope_key),
                ("recipient_detector_score_gate", score_key),
                ("alternative_envelope_geometry_score", combined_key),
                ("repair_surface_combination", repair_surface_key),
            ):
                global_gate_groups[group_name][key].append(source)
            for reason in geometry_reasons:
                global_gate_groups["geometry_reason_record_incidence"][reason].append(
                    source
                )
            for surface in repair_surfaces:
                global_gate_groups["repair_surface_record_incidence"][surface].append(
                    source
                )
        elif shadow["state"] == "unresolved":
            unresolved_records += 1
            remaining_cluster = finding["remaining_failure_cluster"]
            if not isinstance(remaining_cluster, Mapping):
                raise ProbeError("unresolved record has no remaining-failure cluster")
            primary_blocker = remaining_cluster["unresolved_primary_blocker"]
            raw_consensus = finding["raw_consensus"]
            raw_blockers = _raw_candidate_blockers(raw_consensus)
            raw_blocker_key = "+".join(raw_blockers) if raw_blockers else "none"
            rejected_occurrences = shadow["rejected_line_occurrences"]
            occurrence_signature = (
                "|".join(
                    f"{reason}={int(count)}"
                    for reason, count in sorted(rejected_occurrences.items())
                    if int(count) > 0
                )
                or "none"
            )
            for group_name, key in (
                ("primary_filter_blocker", str(primary_blocker)),
                ("raw_consensus_state", str(raw_consensus["state"])),
                ("raw_candidate_filter_reason_combination", raw_blocker_key),
                ("rejected_line_occurrence_signature", occurrence_signature),
                ("failure_reason_type", str(finding["failure_reason_type"])),
            ):
                unresolved_groups[group_name][key].append(source)
            for reason, count in rejected_occurrences.items():
                count = int(count)
                if count <= 0:
                    continue
                unresolved_rejected_line_occurrences[str(reason)] += count
                unresolved_groups["rejected_line_reason_record_incidence"][
                    str(reason)
                ].append(source)
        if shadow["candidate"] is not None:
            teacher_runtime_routes.update([str(shadow["runtime_route"])])
            teacher_confidence.update(
                [_confidence_bucket(float(shadow["minimum_confidence"]))]
            )
            teacher_crop_combinations.update(
                [_crop_combo(shadow["consensus_crops"])]
            )
        first_alt_geometry.update([_group_key_attempt(finding["attempts"]["first"])])
        retry_alt_geometry.update([_group_key_attempt(finding["attempts"]["retry"])])
        for group_name, key in finding["group_keys"].items():
            groups[group_name][str(key)].append(source)
        remaining_cluster = finding["remaining_failure_cluster"]
        if remaining_cluster is not None:
            remaining_records += 1
            global_gate_failures = shadow["global_gate_failures"]
            gate_presence = "failed" if global_gate_failures else "clear"
            remaining_with_global_gate_failures += int(bool(global_gate_failures))
            remaining_gate_overlay_groups["strict_state_by_gate_presence"][
                f"strict_state={shadow['state']}|global_gates={gate_presence}"
            ].append(source)
            remaining_gate_overlay_groups[
                "strict_state_by_global_gate_failures_combination"
            ][
                f"strict_state={shadow['state']}|failures="
                f"{_global_gate_failure_key(global_gate_failures)}"
            ].append(source)
            remaining_groups["global_gate_failures_combination"][
                str(remaining_cluster["global_gate_failures_combination"])
            ].append(source)
            remaining_groups["eligible_candidate_count"][
                str(remaining_cluster["eligible_candidate_count"])
            ].append(source)
            if remaining_cluster["strict_state"] == "ambiguous":
                remaining_groups["ambiguous_candidate_count"][
                    str(remaining_cluster["ambiguous_candidate_count"])
                ].append(source)
            if remaining_cluster["unresolved_primary_blocker"] is not None:
                remaining_groups["unresolved_primary_blocker"][
                    str(remaining_cluster["unresolved_primary_blocker"])
                ].append(source)
            remaining_groups["raw_vs_strict"][
                str(remaining_cluster["raw_vs_strict"])
            ].append(source)
            remaining_groups["alternative_envelope_geometry_score"][
                str(remaining_cluster["alternative_envelope_geometry_score"])
            ].append(source)
    consensus_coverage = sum(
        finding["reference_exact_line_crop_consensus"]
        for finding in findings
        if finding["external_reference_present"]
    )
    teacher_consensus_records = sum(
        row["paddle_teacher_consensus"]["candidate"] is not None for row in findings
    )
    return {
        "schema_version": 1,
        "kind": OUTPUT_SUMMARY_KIND,
        "read_only_existing_diagnostic": True,
        "ocr_rerun": False,
        "truth_used_for_analysis_only": True,
        "runtime_truth_lookup": False,
        "formal_delivery_gate": False,
        "formal_contract": {
            "comparison_evaluation_mode": "formal",
            "comparison_records": FORMAL_RECORDS,
            "failed_records": FORMAL_FAILURES,
            "recipient_missing_only_records": FORMAL_FAILURES,
            "recipient_missing_with_additional_failures_records": 0,
            "non_missing_invariant_failure_records": 0,
        },
        "findings_records": len(findings),
        "unique_sources": len({_source_key(row["source"]) for row in findings}),
        "external_reference": {
            "present_records": external_reference_present,
            "missing_records": len(findings) - external_reference_present,
            "external_truth": external_reference_present > 0,
            "accuracy_claimed": False,
            "by_attempt": {
                name: dict(sorted(counter.items()))
                for name, counter in reference_attempt.items()
            },
            "exact_attempt_line_index_distribution": _count_rows(exact_position),
            "exact_line_2_of_3_crop_consensus": {
                "records": consensus_coverage,
                "denominator": external_reference_present,
                "coverage": (
                    consensus_coverage / external_reference_present
                    if external_reference_present
                    else None
                ),
                "by_crop_combination": _count_rows(exact_consensus),
            },
            "teacher_consensus_truth_outcome": _count_rows(shadow_truth),
        },
        "first_alternative_route_by_geometry": _count_rows(first_alt_geometry),
        "retry_alternative_route_by_geometry": _count_rows(retry_alt_geometry),
        "paddle_teacher_consensus": {
            "pseudo_truth_source": "ppocr_independent_crop_exact_consensus",
            "external_truth": False,
            "truth_used_for_analysis_only": True,
            "formal_delivery_gate": False,
            "interpretation": "self_consistency_coverage_not_human_accuracy",
            "records": teacher_consensus_records,
            "coverage": teacher_consensus_records / len(findings),
            "by_state": _count_rows(shadow_state),
            "by_attempt_combination": _count_rows(teacher_crop_combinations),
            "by_runtime_route": _count_rows(teacher_runtime_routes),
            "by_minimum_confidence_bucket": _count_rows(teacher_confidence),
            "contract": {
                "minimum_visible_characters": 2,
                "maximum_visible_characters": 48,
                "requires_letter": True,
                "minimum_line_confidence": MINIMUM_STRICT_LINE_CONFIDENCE,
                "minimum_recipient_detector_score": MINIMUM_RECIPIENT_DETECTOR_SCORE,
                "requires_empty_geometry_reasons": True,
                "requires_verified_alternative_envelope": True,
                "requires_same_exact_line_in_independent_crops": 2,
                "dominant_fallback_requires_multiple_eligible_candidates": True,
                "dominant_fallback_requires_same_exact_line_in_all_crops": len(
                    ATTEMPTS
                ),
                "dominant_fallback_requires_unique_all_crop_candidate": True,
                "negative_tokens": list(NEGATIVE_TOKENS),
                "recipient_label_pinyin_keys": sorted(
                    RECIPIENT_LABEL_PINYIN_KEYS
                ),
                "ascii_ui_line_keys": sorted(ASCII_UI_LINE_KEYS),
                "amount_time_and_character_filters": True,
                "multiple_eligible_candidates_are_absent_unless_one_is_unique_across_all_three_crops": True,
            },
        },
        "shadow_candidate_truth_free": {
            "same_evidence_as_paddle_teacher_consensus": True,
            "by_state": _count_rows(shadow_state),
        },
        "raw_consensus": {
            "descriptive_only": True,
            "not_runtime_eligible_without_strict_gates": True,
            "by_state": _count_rows(raw_consensus_state),
        },
        "remaining_failure_analysis": {
            "scope": "strict_runtime_shadow.state != candidate",
            "records": remaining_records,
            "strict_candidate_records": len(findings) - remaining_records,
            "unreported_failure_reason_records": all_failure_reason_types[
                "unreported"
            ],
            "by_strict_state_all_records": _count_rows(all_strict_states),
            "by_failure_reason_type_all_records": _count_rows(
                all_failure_reason_types
            ),
            "groups": {
                name: _group_rows(values)
                for name, values in remaining_groups.items()
            },
        },
        "global_gate_failure_analysis": {
            "scope": "strict_runtime_shadow.state == rejected_by_global_gate",
            "records": global_gate_records,
            "selected_consensus_records": (
                global_gate_selected_consensus_records
            ),
            "single_eligible_candidate_records": (
                global_gate_single_eligible_candidate_records
            ),
            "candidate_derivation_intact": (
                global_gate_selected_consensus_records == global_gate_records
            ),
            "parser_bypass_allowed": False,
            "protection_floor_changes_allowed": False,
            "remediation_must_restore_failed_gate_evidence": True,
            "repair_surface_definitions": {
                "alternative_envelope_generation_or_verification": (
                    "repair alternative-envelope evidence"
                ),
                "detector_score": (
                    "repair detector confidence without lowering its floor"
                ),
                "detector_box": "repair detector localization",
                "detector_layout_geometry": (
                    "repair field detection or layout ordering"
                ),
                "rectification_or_projection": (
                    "repair direction, homography, or coordinate projection"
                ),
                "unclassified_geometry_evidence": (
                    "diagnose new geometry evidence before changing runtime"
                ),
            },
            "groups": {
                name: _group_rows(values)
                for name, values in global_gate_groups.items()
            },
        },
        "remaining_global_gate_overlay_analysis": {
            "scope": "strict_runtime_shadow.state != candidate",
            "records": remaining_records,
            "any_global_gate_failure_records": (
                remaining_with_global_gate_failures
            ),
            "clear_global_gate_records": (
                remaining_records - remaining_with_global_gate_failures
            ),
            "gate_failure_is_decisive_only_for_state": (
                "rejected_by_global_gate"
            ),
            "unresolved_or_ambiguous_gate_failures_are_overlays_not_parser_remediation": True,
            "groups": {
                name: _group_rows(values)
                for name, values in remaining_gate_overlay_groups.items()
            },
        },
        "unresolved_filter_analysis": {
            "scope": "strict_runtime_shadow.state == unresolved",
            "records": unresolved_records,
            "parser_bypass_allowed": False,
            "protection_floor_changes_allowed": False,
            "line_filters_remain_protective": True,
            "global_gate_bypass_is_not_a_filter_remediation": True,
            "rejected_line_occurrences": _occurrence_rows(
                unresolved_rejected_line_occurrences
            ),
            "groups": {
                name: _group_rows(values)
                for name, values in unresolved_groups.items()
            },
        },
        "groups": {
            name: _group_rows(values) for name, values in groups.items()
        },
        "artifacts": {"summary": "summary.json", "findings": "findings.jsonl"},
        "source_evidence": evidence["source_evidence"],
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def write_atomic(
    output_directory: Path,
    *,
    input_directory: Path,
    summary: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> None:
    output = output_directory.resolve()
    input_root = input_directory.resolve()
    if _is_within(output, input_root):
        raise ProbeError("output directory must not be inside the diagnostic input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite truth probe output: {output}")
    identities = summary.get("source_evidence")
    if not isinstance(identities, Mapping):
        raise ProbeError("truth probe summary has no source evidence identities")
    _assert_identities_current(identities)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        (stage / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "findings.jsonl").write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in findings
            ),
            encoding="utf-8",
        )
        _assert_identities_current(identities)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite truth probe output: {output}")
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    input_directory = args.input_directory.resolve()
    findings, evidence = _load_input(input_directory)
    summary = summarize(findings, evidence=evidence)
    write_atomic(
        args.output_directory,
        input_directory=input_directory,
        summary=summary,
        findings=findings,
    )
    print(
        json.dumps(
            {
                "kind": OUTPUT_SUMMARY_KIND,
                "records": len(findings),
                "output_directory": args.output_directory.resolve().as_posix(),
                "formal_delivery_gate": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
