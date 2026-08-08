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
    if len(eligible) == 1 and not global_gate_failures:
        candidate = eligible[0]["candidate"]
        crops = eligible[0]["crops"]
        return {
            "candidate": candidate,
            "state": "candidate",
            "consensus_crops": crops,
            "minimum_confidence": eligible[0]["minimum_confidence"],
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
            else "ambiguous"
            if len(eligible) > 1
            else "rejected_by_global_gate"
        ),
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
    first_alt_geometry = Counter()
    retry_alt_geometry = Counter()
    groups: dict[str, dict[str, list[str]]] = {
        "alternative_envelope": defaultdict(list),
        "geometry": defaultdict(list),
        "line_count_tuple": defaultdict(list),
        "consensus_crop_combination": defaultdict(list),
    }
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
        raw_consensus_state.update([finding["raw_consensus"]["state"]])
        shadow_truth.update([finding["strict_runtime_shadow"]["truth_outcome"]])
        shadow_state.update([shadow["state"]])
        if shadow["candidate"] is not None:
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
                "negative_tokens": list(NEGATIVE_TOKENS),
                "amount_time_and_character_filters": True,
                "ambiguous_consensus_is_absent": True,
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
