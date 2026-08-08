#!/usr/bin/env python3
"""Summarize fail-closed hybrid-recipient pilot records without rerunning OCR."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4


RECIPIENT_LABELS = ("收款方", "收款人", "收款账户", "收款账号")
NON_RECIPIENT_LABELS = (
    "付款方式",
    "支付方式",
    "交易方式",
    "付款渠道",
    "转账成功",
    "支付成功",
    "交易成功",
    "付款成功",
    "金额",
    "时间",
    "订单号",
    "商品",
    "优惠",
    "活动",
    "充值",
    "奖励",
    "红包",
    "积分",
    "广告",
    "推荐",
)
PAIR_PATTERN = re.compile(
    r"^(?P<merchant>.+?)\s*[¥￥]\s*(?P<amount>(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]{1,2})?)$"
)
EXPECTED_AMOUNT_PATTERN = re.compile(
    r"^(?:[¥￥]\s*)?(?P<amount>(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]{1,2})?)$"
)
DIAGNOSTIC_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_diagnostic_summary_v1"
DIAGNOSTIC_FINDING_KIND = "receipt_mlnet_hybrid_failure_diagnostic_finding_v1"
RECIPIENT_MISSING_FAILURE = "hybrid recipient candidate missing"
COMPARISON_SUMMARY_KIND = "receipt_mlnet_hybrid_recipient_cpu_ab_v1"
COMPARISON_SUMMARY_SCHEMA_VERSION = 2
FORMAL_EXPECTED_RECORDS = 10016


class DiagnosticError(ValueError):
    """Raised when the frozen A/B evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source_key(value: object) -> str:
    return os.path.normcase(
        os.path.normpath(os.path.abspath(str(value or "")))
    ).replace("\\", "/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, description: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise DiagnosticError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise DiagnosticError(f"{description} is not a file: {resolved}")
    size_bytes = resolved.stat().st_size
    if size_bytes <= 0:
        raise DiagnosticError(f"{description} must be non-empty: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": size_bytes,
    }


def _assert_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any], *, description: str
) -> None:
    if dict(before) != dict(after):
        raise DiagnosticError(f"{description} changed while diagnostics were reading it")


def _normalized_source_set_sha256(sources: Sequence[str]) -> str:
    payload = "".join(f"{source}\n" for source in sorted(sources)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonempty_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticError(f"{description} must be a non-empty string")
    return value


def _recipient_candidate_missing(value: object) -> bool:
    return not isinstance(value, str) or not value


def _comparison_rows_with_identity(
    comparison: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = comparison / "comparisons.jsonl"
    identity = _file_identity(path, description="comparison rows")
    rows: list[dict[str, Any]] = []
    sources: set[str] = set()
    for index, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines()
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise DiagnosticError(f"comparison[{index}] must be an object")
        row = dict(payload)
        source = _nonempty_string(
            row.get("source"), description=f"comparison[{index}].source"
        )
        source_key = _source_key(source)
        if source_key in sources:
            raise DiagnosticError(f"duplicate comparison source: {source!r}")
        sources.add(source_key)
        invariant = row.get("invariant")
        if not isinstance(invariant, bool):
            raise DiagnosticError(f"comparison[{index}].invariant must be a boolean")
        candidate = row.get("recipient_candidate")
        if candidate is not None and not isinstance(candidate, str):
            raise DiagnosticError(
                f"comparison[{index}].recipient_candidate must be a string or null"
            )
        failures = row.get("failures")
        if (
            not isinstance(failures, list)
            or any(not isinstance(item, str) or not item for item in failures)
        ):
            raise DiagnosticError(
                f"comparison[{index}].failures must be an array of non-empty strings"
            )
        if invariant != (len(failures) == 0):
            raise DiagnosticError(
                f"comparison[{index}] invariant/failures contract disagrees"
            )
        missing = _recipient_candidate_missing(candidate)
        if missing and invariant:
            raise DiagnosticError(
                f"comparison[{index}] missing recipient candidate cannot be invariant"
            )
        if not missing and RECIPIENT_MISSING_FAILURE in failures:
            raise DiagnosticError(
                f"comparison[{index}] present recipient candidate has missing failure"
            )
        rows.append(row)
    if not rows:
        raise DiagnosticError("comparisons.jsonl must contain at least one record")
    _assert_unchanged(
        identity,
        _file_identity(path, description="comparison rows"),
        description="comparisons.jsonl",
    )
    return rows, identity


def _comparison_rows(comparison: Path) -> list[dict[str, Any]]:
    rows, _ = _comparison_rows_with_identity(comparison)
    return rows


def _contained_result_path(raw: str, *, run_root: Path, manifest_path: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise DiagnosticError(f"manifest result file is missing: {candidate}") from error
    if not resolved.is_file():
        raise DiagnosticError(f"manifest result is not a file: {resolved}")
    try:
        resolved.relative_to(run_root)
    except ValueError as error:
        raise DiagnosticError(f"manifest result path escapes hybrid root: {resolved}") from error
    return resolved


def _manifest_result_paths(
    hybrid: Path,
    *,
    require_written: bool,
) -> tuple[dict[str, Path], dict[str, object], dict[str, Any]]:
    try:
        run_root = hybrid.resolve(strict=True)
    except FileNotFoundError as error:
        raise DiagnosticError(f"hybrid run root does not exist: {hybrid}") from error
    if not run_root.is_dir():
        raise DiagnosticError(f"hybrid run root is not a directory: {run_root}")
    manifest_path = run_root / "inference_manifest.json"
    manifest_identity = _file_identity(
        manifest_path, description="hybrid inference manifest"
    )
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise DiagnosticError("hybrid inference manifest must be a non-empty array")
    results: dict[str, Path] = {}
    recipient_candidates: dict[str, object] = {}
    result_paths: set[Path] = set()
    for index, payload in enumerate(manifest):
        if not isinstance(payload, Mapping):
            raise DiagnosticError(f"hybrid manifest[{index}] must be an object")
        source = _nonempty_string(
            payload.get("source"), description=f"hybrid manifest[{index}].source"
        )
        source_key = _source_key(source)
        if source_key in results:
            raise DiagnosticError(f"duplicate hybrid manifest source: {source!r}")
        allowed_statuses = {"written"} if require_written else {
            "written",
            "skipped_existing",
        }
        if payload.get("status") not in allowed_statuses:
            raise DiagnosticError(
                f"hybrid manifest source {source!r} has disallowed status "
                f"{payload.get('status')!r}"
            )
        raw_result = _nonempty_string(
            payload.get("result"), description=f"hybrid manifest[{index}].result"
        )
        result_path = _contained_result_path(
            raw_result, run_root=run_root, manifest_path=manifest_path
        )
        if result_path in result_paths:
            raise DiagnosticError(f"duplicate hybrid result path: {result_path}")
        result_paths.add(result_path)
        result = _load_json(result_path)
        if not isinstance(result, Mapping):
            raise DiagnosticError(f"hybrid result must be an object: {result_path}")
        result_source = _nonempty_string(
            result.get("source"), description=f"hybrid result source in {result_path}"
        )
        if _source_key(result_source) != source_key:
            raise DiagnosticError(
                f"hybrid manifest/result source mismatch: {source!r} != "
                f"{result_source!r}"
            )
        fields = result.get("fields")
        if not isinstance(fields, Mapping):
            raise DiagnosticError(f"hybrid result fields must be an object: {result_path}")
        recipient = fields.get("recipient")
        if not isinstance(recipient, Mapping):
            raise DiagnosticError(f"hybrid result recipient must be an object: {result_path}")
        recipient_candidate = recipient.get("candidate")
        if recipient_candidate is not None and not isinstance(recipient_candidate, str):
            raise DiagnosticError(
                f"hybrid result recipient candidate must be a string or null: {result_path}"
            )
        results[source_key] = result_path
        recipient_candidates[source_key] = recipient_candidate
    manifest_identity = {
        **manifest_identity,
        "records": len(results),
        "normalized_source_set_sha256": _normalized_source_set_sha256(
            tuple(results)
        ),
    }
    current_identity = _file_identity(
        manifest_path, description="hybrid inference manifest"
    )
    _assert_unchanged(
        {key: manifest_identity[key] for key in ("path", "sha256", "size_bytes")},
        current_identity,
        description="hybrid inference manifest",
    )
    return results, recipient_candidates, manifest_identity


def _comparison_summary(
    comparison: Path,
    *,
    comparison_rows: Sequence[Mapping[str, Any]],
    require_formal: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = comparison / "summary.json"
    identity = _file_identity(path, description="comparison summary")
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise DiagnosticError("comparison summary must be an object")
    summary = dict(payload)
    if type(summary.get("schema_version")) is not int or summary.get(
        "schema_version"
    ) != COMPARISON_SUMMARY_SCHEMA_VERSION:
        raise DiagnosticError("comparison summary schema_version is not supported")
    if summary.get("kind") != COMPARISON_SUMMARY_KIND:
        raise DiagnosticError("comparison summary kind is not supported")
    evaluation_mode = summary.get("evaluation_mode")
    if evaluation_mode not in {"pilot", "formal"}:
        raise DiagnosticError("comparison summary evaluation_mode must be pilot or formal")
    if require_formal and evaluation_mode != "formal":
        raise DiagnosticError("--require-formal requires a formal comparison summary")
    if require_formal and len(comparison_rows) != FORMAL_EXPECTED_RECORDS:
        raise DiagnosticError(
            f"--require-formal requires exactly {FORMAL_EXPECTED_RECORDS} records; "
            f"found {len(comparison_rows)}"
        )
    invariant_records = sum(row["invariant"] is True for row in comparison_rows)
    recipient_records = sum(
        isinstance(row.get("recipient_candidate"), str)
        and bool(row.get("recipient_candidate"))
        for row in comparison_rows
    )
    expected_coverage = recipient_records / len(comparison_rows)
    if type(summary.get("records")) is not int or summary.get("records") != len(
        comparison_rows
    ):
        raise DiagnosticError("comparison summary records differs from comparisons.jsonl")
    if type(summary.get("invariant_records")) is not int or summary.get(
        "invariant_records"
    ) != invariant_records:
        raise DiagnosticError(
            "comparison summary invariant_records differs from comparisons.jsonl"
        )
    coverage = summary.get("recipient_candidate_coverage")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or float(coverage) != expected_coverage
    ):
        raise DiagnosticError(
            "comparison summary recipient_candidate_coverage differs from "
            "comparisons.jsonl"
        )
    _assert_unchanged(
        identity,
        _file_identity(path, description="comparison summary"),
        description="comparison summary",
    )
    return summary, identity, str(evaluation_mode)


def _validate_summary_manifest_identity(
    summary: Mapping[str, Any],
    *,
    manifest_identity: Mapping[str, Any],
) -> None:
    run_manifests = summary.get("run_manifests")
    hybrid_identity = (
        run_manifests.get("hybrid") if isinstance(run_manifests, Mapping) else None
    )
    if not isinstance(hybrid_identity, Mapping):
        raise DiagnosticError("comparison summary has no hybrid run manifest identity")
    summary_path = _nonempty_string(
        hybrid_identity.get("path"),
        description="comparison summary hybrid manifest path",
    )
    try:
        summary_resolved = Path(summary_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise DiagnosticError(
            f"comparison summary hybrid manifest path is missing: {summary_path}"
        ) from error
    actual_resolved = Path(str(manifest_identity["path"])).resolve(strict=True)
    if os.path.normcase(os.path.normpath(str(summary_resolved))) != os.path.normcase(
        os.path.normpath(str(actual_resolved))
    ):
        raise DiagnosticError(
            "comparison summary hybrid manifest path does not point to --hybrid"
        )
    for key in (
        "sha256",
        "size_bytes",
        "records",
        "normalized_source_set_sha256",
    ):
        if type(hybrid_identity.get(key)) is not type(manifest_identity[key]) or (
            hybrid_identity.get(key) != manifest_identity[key]
        ):
            raise DiagnosticError(
                f"comparison summary hybrid manifest {key} differs from --hybrid"
            )


def _load_bound_evidence(
    comparison: Path,
    hybrid: Path,
    *,
    require_formal: bool,
) -> dict[str, Any]:
    comparison_rows, comparisons_identity = _comparison_rows_with_identity(comparison)
    comparison_summary, summary_identity, evaluation_mode = _comparison_summary(
        comparison,
        comparison_rows=comparison_rows,
        require_formal=require_formal,
    )
    results, result_candidates, manifest_identity = _manifest_result_paths(
        hybrid,
        require_written=evaluation_mode == "formal",
    )
    comparison_sources = {_source_key(row["source"]) for row in comparison_rows}
    if set(results) != comparison_sources:
        raise DiagnosticError(
            "hybrid manifest source set differs from comparisons: "
            f"missing={len(comparison_sources - set(results))} "
            f"extra={len(set(results) - comparison_sources)}"
        )
    for index, row in enumerate(comparison_rows):
        source_key = _source_key(row["source"])
        comparison_candidate = row.get("recipient_candidate")
        result_candidate = result_candidates[source_key]
        if type(comparison_candidate) is not type(result_candidate) or (
            comparison_candidate != result_candidate
        ):
            raise DiagnosticError(
                f"comparison[{index}] recipient_candidate differs from hybrid result"
            )
    _validate_summary_manifest_identity(
        comparison_summary,
        manifest_identity=manifest_identity,
    )
    return {
        "comparison_rows": comparison_rows,
        "result_paths": results,
        "evaluation_mode": evaluation_mode,
        "identities": {
            "comparison_summary": summary_identity,
            "comparisons": comparisons_identity,
            "hybrid_manifest": manifest_identity,
        },
    }


def _assert_source_evidence_current(source_evidence: Mapping[str, Any]) -> None:
    for name in ("comparison_summary", "comparisons", "hybrid_manifest"):
        identity = source_evidence.get(name)
        if not isinstance(identity, Mapping):
            raise DiagnosticError(f"diagnostic source evidence has no {name} identity")
        path = _nonempty_string(
            identity.get("path"), description=f"diagnostic {name} path"
        )
        expected = {
            key: identity.get(key) for key in ("path", "sha256", "size_bytes")
        }
        _assert_unchanged(
            expected,
            _file_identity(Path(path), description=f"diagnostic {name}"),
            description=f"diagnostic {name}",
        )


def _box(record: Mapping[str, Any] | None) -> list[float] | None:
    if not isinstance(record, Mapping):
        return None
    raw = record.get("bbox_image")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        values = [float(value) for value in raw[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _score(record: Mapping[str, Any] | None) -> float | None:
    if not isinstance(record, Mapping):
        return None
    try:
        value = float(record.get("score"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _geometry_reasons(
    result: Mapping[str, Any], detections: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    geometry = result.get("geometry")
    size = geometry.get("rectified_size") if isinstance(geometry, Mapping) else None
    try:
        width = int(size.get("width"))
        height = int(size.get("height"))
    except (AttributeError, TypeError, ValueError):
        return ["rectified_size_missing"]
    recipient = detections.get("recipient_field")
    amount = detections.get("amount")
    payment = detections.get("payment_method_field")
    reasons: list[str] = []
    for name, record, floor in (
        ("recipient", recipient, 0.68),
        ("amount", amount, 0.80),
        ("payment", payment, 0.80),
    ):
        score = _score(record)
        if score is None:
            reasons.append(f"{name}_score_missing")
        elif score < floor:
            reasons.append(f"{name}_score_below_{floor:.2f}")
        if _box(record) is None:
            reasons.append(f"{name}_box_invalid")
    recipient_box = _box(recipient)
    amount_box = _box(amount)
    payment_box = _box(payment)
    if recipient_box is None or amount_box is None or payment_box is None:
        return reasons
    recipient_width = recipient_box[2] - recipient_box[0]
    recipient_height = recipient_box[3] - recipient_box[1]
    recipient_center = (recipient_box[1] + recipient_box[3]) * 0.5
    amount_center = (amount_box[1] + amount_box[3]) * 0.5
    payment_center = (payment_box[1] + payment_box[3]) * 0.5
    tolerance = max(4.0, recipient_height * 0.25)
    checks = (
        (recipient_box[0] <= width * 0.20, "recipient_left_edge"),
        (recipient_box[2] >= width * 0.80, "recipient_right_edge"),
        (recipient_width >= width * 0.60, "recipient_width"),
        (recipient_height <= height * 0.15, "recipient_height"),
        (amount_center < recipient_center, "amount_before_recipient"),
        (recipient_center < payment_center, "recipient_before_payment"),
        (recipient_box[1] >= amount_box[3] - tolerance, "amount_edge_overlap"),
        (recipient_box[3] <= payment_box[1] + tolerance, "payment_edge_overlap"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return reasons


def _geometry_evidence(
    result: Mapping[str, Any], detections: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    geometry = result.get("geometry")
    size = geometry.get("rectified_size") if isinstance(geometry, Mapping) else None
    recipient = _box(detections.get("recipient_field"))
    amount = _box(detections.get("amount"))
    payment = _box(detections.get("payment_method_field"))
    try:
        width = int(size.get("width"))
        height = int(size.get("height"))
    except (AttributeError, TypeError, ValueError):
        return None
    if recipient is None or amount is None or payment is None:
        return None
    recipient_height = recipient[3] - recipient[1]
    tolerance = max(4.0, recipient_height * 0.25)
    payment_overlap = max(0.0, recipient[3] - payment[1])
    exact_cjk_tolerance = max(4.0, recipient_height * 0.45)
    return {
        "rectified_width": width,
        "rectified_height": height,
        "amount_box": amount,
        "recipient_box": recipient,
        "payment_box": payment,
        "recipient_height": round(recipient_height, 4),
        "vertical_tolerance": round(tolerance, 4),
        "amount_edge_margin": round(recipient[1] - amount[3], 4),
        "payment_edge_margin": round(payment[1] - recipient[3], 4),
        "payment_overlap_fraction": round(payment_overlap / recipient_height, 6),
        "payment_excess_overlap": round(
            max(0.0, recipient[3] - payment[1] - tolerance), 4
        ),
        "payment_exact_cjk_exception_excess": round(
            max(0.0, payment_overlap - exact_cjk_tolerance), 4
        ),
    }


def _amount_fen(value: object) -> int | None:
    match = EXPECTED_AMOUNT_PATTERN.fullmatch(str(value or "").strip())
    if match is None:
        return None
    try:
        amount = Decimal(match.group("amount").replace(",", ""))
    except InvalidOperation:
        return None
    return int((amount.quantize(Decimal("0.01")) * 100).to_integral_exact())


def _aggregate_pair_reasons(raw: object, line_count: object, expected: object) -> list[str]:
    text = " ".join(str(raw or "").split())
    reasons: list[str] = []
    if line_count != 2:
        reasons.append("line_count_not_2")
    match = PAIR_PATTERN.fullmatch(text)
    if match is None:
        reasons.append("aggregate_not_merchant_currency_amount")
        return reasons
    merchant = match.group("merchant").strip()
    if len(merchant) < 2 or len(merchant) > 64:
        reasons.append("merchant_length")
    if not any("\u3400" <= character <= "\u9fff" for character in merchant):
        reasons.append("merchant_has_no_cjk")
    if any(label in merchant for label in RECIPIENT_LABELS + NON_RECIPIENT_LABELS):
        reasons.append("merchant_blocklisted")
    observed = _amount_fen(match.group("amount"))
    expected_fen = _amount_fen(expected)
    if observed is None or expected_fen is None:
        reasons.append("amount_parse_failed")
    elif observed != expected_fen:
        reasons.append("amount_mismatch")
    return reasons


def _export_failure_crops(
    specs: list[dict[str, Any]], output: Path
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite failure crop output: {output}")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("--crop-output requires Pillow") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    manifest: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(specs, start=1):
            source = Path(str(spec["source"]))
            box = spec["recipient_box"]
            with Image.open(source) as opened:
                with opened.convert("RGB") as image:
                    width, height = image.size
                    margin_x = max(2.0, (box[2] - box[0]) * 0.08)
                    margin_y = max(2.0, (box[3] - box[1]) * 0.08)
                    left = max(0, min(width, math.floor(box[0] - margin_x)))
                    top = max(0, min(height, math.floor(box[1] - margin_y)))
                    right = max(0, min(width, math.ceil(box[2] + margin_x)))
                    bottom = max(0, min(height, math.ceil(box[3] + margin_y)))
                    if right <= left or bottom <= top:
                        raise ValueError(f"invalid recipient crop for {source}")
                    primary_name = f"{index:03d}-primary.png"
                    retry_name = f"{index:03d}-left-context.png"
                    image.crop((left, top, right, bottom)).save(stage / primary_name)
                    image.crop((0, top, right, bottom)).save(stage / retry_name)
            manifest.append(
                {
                    "source": str(source),
                    "recipient_box": box,
                    "primary": primary_name,
                    "left_context_retry": retry_name,
                }
            )
        (stage / "manifest.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in manifest
            ),
            encoding="utf-8",
        )
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _primary_blocker(
    *,
    comparison_row: Mapping[str, Any],
    failure_reason: object,
    geometry_reasons: list[str],
    likely_blockers: list[str],
) -> str:
    """Return one stable record-level bucket while retaining detailed reasons."""

    candidate = comparison_row.get("recipient_candidate")
    failures = comparison_row.get("failures")
    if isinstance(candidate, str) and candidate.strip() and isinstance(failures, list):
        return "non_recipient_invariant_change"
    if isinstance(failure_reason, str) and failure_reason.strip():
        reason_root = failure_reason.split(";", maxsplit=1)[0].strip()
        if reason_root in {"missing_detection", "invalid_standard_crop", "ocr_empty"}:
            return reason_root
    if geometry_reasons:
        return "geometry_contract"
    if likely_blockers == ["per_line_confidence_or_exact_line_split"]:
        return "per_line_confidence_or_exact_line_split"
    if isinstance(failure_reason, str) and failure_reason.strip():
        return failure_reason.split(";", maxsplit=1)[0].strip()
    return "unclassified"


def _count_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"name": name, "records": records}
        for name, records in sorted(
            counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _add_group(
    groups: dict[str, set[str]], name: str, *, source: object
) -> None:
    rendered_source = str(source) if source is not None else "<missing-source>"
    groups.setdefault(name, set()).add(rendered_source)


def _group_rows(groups: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "records": len(sources),
            "example_sources": sorted(sources)[:3],
        }
        for name, sources in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]


def _crop_text_agreement(row: Mapping[str, Any]) -> str:
    texts = [
        " ".join(str(row.get(key) or "").split())
        for key in ("first_raw", "retry_raw", "right_value_raw")
    ]
    present = sum(bool(text) for text in texts)
    if present == 0:
        return "all_empty"
    if present < 3:
        return f"incomplete_{present}_of_3_nonempty"
    if texts[0] == texts[1] == texts[2]:
        return "all_three_equal"
    if texts[0] == texts[1]:
        return "first_retry_equal"
    if texts[0] == texts[2]:
        return "first_right_equal"
    if texts[1] == texts[2]:
        return "retry_right_equal"
    return "all_three_different"


def summarize(
    diagnostics: list[dict[str, Any]],
    *,
    comparison: Path,
    hybrid: Path,
    require_formal: bool = False,
    _evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = (
        dict(_evidence)
        if _evidence is not None
        else _load_bound_evidence(
            comparison,
            hybrid,
            require_formal=require_formal,
        )
    )
    comparison_rows = evidence["comparison_rows"]
    failed_rows = [row for row in comparison_rows if row["invariant"] is False]
    failed_sources = {_source_key(row["source"]) for row in failed_rows}
    diagnostic_sources: set[str] = set()
    for index, row in enumerate(diagnostics):
        source = _nonempty_string(
            row.get("source"), description=f"diagnostic[{index}].source"
        )
        source_key = _source_key(source)
        if source_key in diagnostic_sources:
            raise DiagnosticError(f"duplicate diagnostic source: {source!r}")
        diagnostic_sources.add(source_key)
    if diagnostic_sources != failed_sources:
        raise DiagnosticError(
            "diagnostic source set differs from invariant-failure comparison set: "
            f"missing={len(failed_sources - diagnostic_sources)} "
            f"extra={len(diagnostic_sources - failed_sources)}"
        )
    invariant_failure_records = len(failed_rows)
    recipient_missing_rows = [
        row
        for row in comparison_rows
        if _recipient_candidate_missing(row.get("recipient_candidate"))
    ]
    recipient_missing_only_records = sum(
        row["failures"] == [RECIPIENT_MISSING_FAILURE]
        for row in recipient_missing_rows
    )
    recipient_missing_with_additional_failures_records = sum(
        row["failures"] != [RECIPIENT_MISSING_FAILURE]
        for row in recipient_missing_rows
    )
    non_missing_invariant_failure_records = sum(
        row["invariant"] is False
        and not _recipient_candidate_missing(row.get("recipient_candidate"))
        for row in comparison_rows
    )
    comparator_failures: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    primary_blockers: Counter[str] = Counter()
    blocker_reasons: Counter[str] = Counter()
    forensic_groups: dict[str, dict[str, set[str]]] = {
        "comparator_failure_set": {},
        "reference_availability": {},
        "ppocr_failure_root": {},
        "alternative_envelope": {},
        "crop_line_counts": {},
        "crop_text_agreement": {},
        "geometry_reason": {},
    }
    for row in diagnostics:
        source = row.get("source")
        failures = row.get("failures")
        if isinstance(failures, list):
            comparator_failures.update(
                failure for failure in failures if isinstance(failure, str) and failure
            )
            failure_set = " | ".join(
                sorted(failure for failure in failures if isinstance(failure, str))
            )
        else:
            failure_set = "<invalid>"
        _add_group(
            forensic_groups["comparator_failure_set"],
            failure_set or "<none>",
            source=source,
        )
        route = row.get("ppocr_route")
        routes.update([route if isinstance(route, str) and route else "<missing>"])
        blocker = row.get("primary_blocker")
        primary_blockers.update(
            [blocker if isinstance(blocker, str) and blocker else "unclassified"]
        )
        likely = row.get("likely_blocker")
        if isinstance(likely, list):
            blocker_reasons.update(
                reason for reason in set(likely) if isinstance(reason, str) and reason
            )
        reference = row.get("reference")
        reference_availability = (
            "not_loaded"
            if not isinstance(reference, Mapping)
            else (
                "present"
                if isinstance(reference.get("recipient"), str)
                else "missing"
            )
        )
        _add_group(
            forensic_groups["reference_availability"],
            reference_availability,
            source=source,
        )
        failure_reason = row.get("ppocr_failure_reason")
        if isinstance(failure_reason, str) and failure_reason.strip():
            failure_root = failure_reason.split(";", maxsplit=1)[0].strip()
            envelope_match = re.search(
                r"(?:^|;)alternative_envelope=(True|False)(?:;|$)",
                failure_reason,
            )
            envelope = (
                envelope_match.group(1).lower()
                if envelope_match is not None
                else "not_reported"
            )
        else:
            failure_root = "<missing>"
            envelope = "not_reported"
        _add_group(
            forensic_groups["ppocr_failure_root"], failure_root, source=source
        )
        _add_group(
            forensic_groups["alternative_envelope"], envelope, source=source
        )
        line_counts = (
            f"first={row.get('first_line_count')!r},"
            f"retry={row.get('retry_line_count')!r},"
            f"right={row.get('right_value_line_count')!r}"
        )
        _add_group(
            forensic_groups["crop_line_counts"], line_counts, source=source
        )
        _add_group(
            forensic_groups["crop_text_agreement"],
            _crop_text_agreement(row),
            source=source,
        )
        geometry_reasons = row.get("geometry_reasons")
        if isinstance(geometry_reasons, list) and geometry_reasons:
            for reason in set(geometry_reasons):
                if isinstance(reason, str) and reason:
                    _add_group(
                        forensic_groups["geometry_reason"], reason, source=source
                    )
        else:
            _add_group(
                forensic_groups["geometry_reason"], "<none>", source=source
            )
    _assert_source_evidence_current(evidence["identities"])
    return {
        "schema_version": 1,
        "kind": DIAGNOSTIC_SUMMARY_KIND,
        "read_only_existing_results": True,
        "ocr_rerun": False,
        "comparison_directory": comparison.resolve().as_posix(),
        "hybrid_directory": hybrid.resolve().as_posix(),
        "comparison_evaluation_mode": evidence["evaluation_mode"],
        "comparison_records": len(comparison_rows),
        "invariant_failure_records": invariant_failure_records,
        "recipient_missing_records": len(recipient_missing_rows),
        "non_missing_invariant_failure_records": (
            non_missing_invariant_failure_records
        ),
        "recipient_missing_only_records": recipient_missing_only_records,
        "recipient_missing_with_additional_failures_records": (
            recipient_missing_with_additional_failures_records
        ),
        "failed_records": len(diagnostics),
        "comparator_failure_occurrences": sum(comparator_failures.values()),
        "by_comparator_failure": _count_rows(comparator_failures),
        "by_ppocr_route": _count_rows(routes),
        "by_primary_blocker": _count_rows(primary_blockers),
        "by_detailed_blocker_reason": _count_rows(blocker_reasons),
        "forensic_groups": {
            name: _group_rows(groups)
            for name, groups in forensic_groups.items()
        },
        "artifacts": {
            "summary": "summary.json",
            "findings": "findings.jsonl",
        },
        "source_evidence": evidence["identities"],
    }


def write_diagnostic_atomic(
    output: Path,
    *,
    summary: Mapping[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    """Publish the derived audit as one new directory; never mutate run evidence."""

    source_evidence = summary.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise DiagnosticError("diagnostic summary has no frozen source evidence")
    _assert_source_evidence_current(source_evidence)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
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
                    {
                        "schema_version": 1,
                        "kind": DIAGNOSTIC_FINDING_KIND,
                        **row,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in diagnostics
            ),
            encoding="utf-8",
        )
        _assert_source_evidence_current(source_evidence)
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def diagnose(
    comparison: Path,
    hybrid: Path,
    records: Path | None = None,
    crop_output: Path | None = None,
    require_formal: bool = False,
    _evidence: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    evidence = (
        dict(_evidence)
        if _evidence is not None
        else _load_bound_evidence(
            comparison,
            hybrid,
            require_formal=require_formal,
        )
    )
    comparison_rows = evidence["comparison_rows"]
    failed = [row for row in comparison_rows if row.get("invariant") is False]
    results = evidence["result_paths"]
    references: dict[str, dict[str, Any]] = {}
    if records is not None:
        for line in records.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "val":
                continue
            slots = record.get("slots")
            if not isinstance(slots, Mapping):
                continue
            recipient_slot = slots.get("recipient_field")
            amount_slot = slots.get("amount")
            references[_source_key(record.get("source"))] = {
                "recipient": recipient_slot.get("text")
                if isinstance(recipient_slot, Mapping)
                else None,
                "amount": (
                    amount_slot.get("visible_text") or amount_slot.get("text")
                )
                if isinstance(amount_slot, Mapping)
                else None,
            }

    diagnostics: list[dict[str, Any]] = []
    crop_specs: list[dict[str, Any]] = []
    for comparison_row in failed:
        source = comparison_row.get("source")
        result = _load_json(results[_source_key(source)])
        if not isinstance(result, Mapping):
            raise DiagnosticError(
                f"hybrid result must remain an object for source {source!r}"
            )
        if _source_key(result.get("source")) != _source_key(source):
            raise DiagnosticError(
                f"hybrid result source changed while diagnosing {source!r}"
            )
        fields = result.get("fields") or {}
        recipient = fields.get("recipient") or {}
        if type(recipient.get("candidate")) is not type(
            comparison_row.get("recipient_candidate")
        ) or recipient.get("candidate") != comparison_row.get("recipient_candidate"):
            raise DiagnosticError(
                f"hybrid result recipient candidate changed while diagnosing {source!r}"
            )
        amount = fields.get("amount") or {}
        expected_amount = amount.get("candidate")
        detections = {
            record.get("label"): record
            for record in result.get("detections") or []
            if isinstance(record, Mapping)
        }
        geometry_reasons = _geometry_reasons(result, detections)
        first_reasons = _aggregate_pair_reasons(
            recipient.get("hybrid_ocr_first_raw"),
            recipient.get("hybrid_ocr_first_line_count"),
            expected_amount,
        )
        retry_reasons = _aggregate_pair_reasons(
            recipient.get("hybrid_ocr_retry_raw"),
            recipient.get("hybrid_ocr_retry_line_count"),
            expected_amount,
        )
        if geometry_reasons:
            likely = geometry_reasons
        elif not first_reasons or not retry_reasons:
            likely = ["per_line_confidence_or_exact_line_split"]
        else:
            likely = sorted(set(first_reasons + retry_reasons))
        failure_reason = recipient.get("hybrid_ocr_failure_reason")
        ppocr_route = recipient.get("hybrid_ocr_route")
        diagnostics.append(
            {
                "source": source,
                "reference": references.get(_source_key(source)),
                "failures": comparison_row.get("failures"),
                "recipient_candidate": comparison_row.get("recipient_candidate"),
                "amount_candidate": expected_amount,
                "recipient_score": _score(detections.get("recipient_field")),
                "amount_score": _score(detections.get("amount")),
                "payment_score": _score(detections.get("payment_method_field")),
                "ppocr_route": ppocr_route,
                "ppocr_failure_reason": failure_reason,
                "geometry_reasons": geometry_reasons,
                "geometry_evidence": _geometry_evidence(result, detections),
                "first_raw": recipient.get("hybrid_ocr_first_raw"),
                "first_line_count": recipient.get("hybrid_ocr_first_line_count"),
                "first_pair_reasons": first_reasons,
                "retry_raw": recipient.get("hybrid_ocr_retry_raw"),
                "retry_line_count": recipient.get("hybrid_ocr_retry_line_count"),
                "retry_pair_reasons": retry_reasons,
                "third_route": recipient.get("hybrid_ocr_third_route"),
                "right_value_raw": recipient.get("hybrid_ocr_right_value_raw"),
                "right_value_line_count": recipient.get(
                    "hybrid_ocr_right_value_line_count"
                ),
                "right_value_line_confidences": recipient.get(
                    "hybrid_ocr_right_value_line_confidences"
                ),
                "likely_blocker": likely,
                "primary_blocker": _primary_blocker(
                    comparison_row=comparison_row,
                    failure_reason=failure_reason,
                    geometry_reasons=geometry_reasons,
                    likely_blockers=likely,
                ),
            }
        )
        recipient_box = _box(detections.get("recipient_field"))
        if crop_output is not None:
            if recipient_box is None:
                raise ValueError(f"missing valid recipient box for failed source: {source}")
            crop_specs.append({"source": source, "recipient_box": recipient_box})
    if crop_output is not None:
        _export_failure_crops(crop_specs, crop_output)
    return diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--crop-output", type=Path)
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="require a hash-bound formal comparison with exactly 10016 fresh results",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "atomically publish compact summary.json and one failed record per "
            "findings.jsonl line; existing output is refused"
        ),
    )
    args = parser.parse_args(argv)
    comparison = args.comparison.resolve()
    hybrid = args.hybrid.resolve()
    evidence = _load_bound_evidence(
        comparison,
        hybrid,
        require_formal=args.require_formal,
    )
    rows = diagnose(
        comparison,
        hybrid,
        args.records.resolve() if args.records is not None else None,
        args.crop_output.resolve() if args.crop_output is not None else None,
        require_formal=args.require_formal,
        _evidence=evidence,
    )
    if args.output_directory is not None:
        output = args.output_directory.resolve()
        summary = summarize(
            rows,
            comparison=comparison,
            hybrid=hybrid,
            require_formal=args.require_formal,
            _evidence=evidence,
        )
        write_diagnostic_atomic(output, summary=summary, diagnostics=rows)
        print(
            json.dumps(
                {
                    "kind": DIAGNOSTIC_SUMMARY_KIND,
                    "failed_records": len(rows),
                    "output_directory": output.as_posix(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 0
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    print(json.dumps({"failed_records": len(rows)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
