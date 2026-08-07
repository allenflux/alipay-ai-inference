#!/usr/bin/env python3
"""Verify v13-only versus v13+PP-OCR-recipient ML.NET outputs.

The input set is prepared separately from the validation split.  This tool has
no manifest labels and cannot tune on test truth: it checks that routing a
recipient crop through PP-OCR changes only the recipient diagnostic candidate,
while recording the exact CPU latency delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


KIND = "receipt_mlnet_hybrid_recipient_cpu_ab_v1"
DELIVERY_CONTRACT_FILENAME = "paddle_ocr_delivery.contract.json"
PADDLE_DELIVERY_KIND = "paddle_ocr_v2_delivery"
PADDLE_MODEL_ROLES = ("det", "rec", "cls")
MAX_P95_OVERHEAD_MS = 250.0
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
INVARIANT_FIELDS = ("time", "amount", "transfer_status", "payment_method")
INVARIANT_CONTRACT_KEYS = (
    "detector",
    "detector_sha256",
    "detector_contract_sha256",
    "device",
    "device_sha256",
    "device_contract_sha256",
    "unified_ocr_model",
    "unified_ocr_contract",
    "unified_ocr_model_sha256",
    "unified_ocr_labels_sha256",
    "unified_ocr_contract_sha256",
)
ARTIFACT_HASH_KEYS = (
    "detector_sha256",
    "detector_contract_sha256",
    "device_sha256",
    "device_contract_sha256",
    "unified_ocr_model_sha256",
    "unified_ocr_labels_sha256",
    "unified_ocr_contract_sha256",
)


class ComparisonError(ValueError):
    pass


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"Invalid {description} {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ComparisonError(f"Could not hash {path}: {error}") from error
    return digest.hexdigest()


def _lower_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256.fullmatch(value) is None:
        raise ComparisonError(f"{description} must be exactly 64 lowercase hexadecimal characters")
    return value


def _integer(value: object, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ComparisonError(f"{description} must be an integer >= {minimum}")
    return value


def _verified_delivery_file(
    delivery: Path,
    record: object,
    description: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ComparisonError(f"Paddle delivery {description} record must be an object")
    relative_value = record.get("path")
    if not isinstance(relative_value, str) or not relative_value:
        raise ComparisonError(f"Paddle delivery {description} has no relative path")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ComparisonError(f"Paddle delivery {description} path must be relative: {relative_value}")
    path = (delivery / relative).resolve()
    try:
        path.relative_to(delivery)
    except ValueError as error:
        raise ComparisonError(f"Paddle delivery {description} escapes the delivery directory") from error
    if not path.is_file():
        raise ComparisonError(f"Missing Paddle delivery {description}: {path}")
    expected_size = _integer(record.get("size_bytes"), f"Paddle delivery {description} size_bytes")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ComparisonError(
            f"Paddle delivery {description} size mismatch: expected {expected_size}, got {actual_size}"
        )
    expected_sha256 = _lower_sha256(record.get("sha256"), f"Paddle delivery {description} sha256")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ComparisonError(
            f"Paddle delivery {description} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "path": relative.as_posix(),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
    }


def _verify_paddle_delivery(delivery_dir: Path) -> dict[str, Any]:
    delivery = delivery_dir.resolve()
    if not delivery.is_dir():
        raise ComparisonError(f"Paddle delivery directory does not exist: {delivery}")
    contract_path = delivery / DELIVERY_CONTRACT_FILENAME
    if not contract_path.is_file():
        raise ComparisonError(f"Missing Paddle delivery contract: {contract_path}")
    contract_sha256 = _sha256(contract_path)
    contract = _load_json(contract_path, "Paddle delivery contract")
    if not isinstance(contract, Mapping) or contract.get("kind") != PADDLE_DELIVERY_KIND:
        raise ComparisonError(f"Not a {PADDLE_DELIVERY_KIND} contract: {contract_path}")
    models = contract.get("models")
    if not isinstance(models, Mapping) or set(models) != set(PADDLE_MODEL_ROLES):
        raise ComparisonError("Paddle delivery contract must contain exactly det, rec and cls models")
    verified_models = {
        role: _verified_delivery_file(delivery, models[role], f"{role} model")
        for role in PADDLE_MODEL_ROLES
    }
    dictionary = _verified_delivery_file(delivery, contract.get("dictionary"), "dictionary")
    # Fail closed if the contract was replaced while its contents were being checked.
    if _sha256(contract_path) != contract_sha256:
        raise ComparisonError("Paddle delivery contract changed during verification")
    return {
        "directory": str(delivery),
        "contract": str(contract_path),
        "contract_sha256": contract_sha256,
        "models": verified_models,
        "dictionary": dictionary,
    }


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ComparisonError(f"{description} must be a finite number")
    return float(value)


def _summary(directory: Path) -> dict[str, Any]:
    path = directory / "inference_summary.json"
    value = _load_json(path, "inference summary")
    if not isinstance(value, dict):
        raise ComparisonError(f"Inference summary must be an object: {path}")
    return value


def _result_map(directory: Path, *, expected_count: int) -> dict[str, dict[str, Any]]:
    manifest_path = directory / "inference_manifest.json"
    manifest = _load_json(manifest_path, "inference manifest")
    if not isinstance(manifest, list) or not manifest:
        raise ComparisonError(f"Inference manifest must be a non-empty array: {manifest_path}")
    if len(manifest) != expected_count:
        raise ComparisonError(
            f"Inference manifest/result count {len(manifest)} differs from CLI summary input/written {expected_count}: "
            f"{manifest_path}"
        )
    results: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(manifest):
        if not isinstance(row, Mapping) or row.get("status") != "written":
            raise ComparisonError(f"Manifest row {index} is not a completed fresh result")
        source = row.get("source")
        result = row.get("result")
        if not isinstance(source, str) or not source or not isinstance(result, str) or not result:
            raise ComparisonError(f"Manifest row {index} has no source/result path")
        source_key = os.path.normcase(os.path.abspath(source))
        result_path = Path(result)
        if not result_path.is_absolute():
            result_path = manifest_path.parent / result_path
        result_path = result_path.resolve()
        try:
            result_path.relative_to(directory)
        except ValueError as error:
            raise ComparisonError(f"Manifest result escapes its run directory: {result_path}") from error
        payload = _load_json(result_path, "receipt result")
        if not isinstance(payload, dict):
            raise ComparisonError(f"Receipt result must be an object: {result_path}")
        result_source_key = os.path.normcase(os.path.abspath(str(payload.get("source", ""))))
        if result_source_key != source_key:
            raise ComparisonError(f"Manifest/result source mismatch for {source}")
        if source_key in results:
            raise ComparisonError(f"Duplicate result source: {source}")
        results[source_key] = payload
    if len(results) != expected_count:
        raise ComparisonError(
            f"Unique result count {len(results)} differs from CLI summary input/written {expected_count}"
        )
    return results


def _require_cpu(summary: Mapping[str, Any], *, hybrid: bool) -> int:
    if summary.get("requested_device") != "cpu" or summary.get("unified_provider") != "cpu":
        raise ComparisonError("Both A/B runs must use the CPU unified provider")
    paddle_provider = summary.get("paddle_ocr_provider")
    if hybrid and paddle_provider != "cpu":
        raise ComparisonError("Hybrid run must use the CPU PP-OCR provider")
    if not hybrid and paddle_provider is not None:
        raise ComparisonError("Baseline run unexpectedly loaded PP-OCR")
    input_count = _integer(summary.get("input"), "inference summary input", minimum=1)
    written_count = _integer(summary.get("written"), "inference summary written", minimum=1)
    skipped_count = _integer(summary.get("skipped"), "inference summary skipped")
    error_count = _integer(summary.get("errors"), "inference summary errors")
    if error_count != 0 or skipped_count != 0:
        raise ComparisonError("A/B runs must be fresh and error-free")
    if written_count != input_count:
        raise ComparisonError("A/B inference coverage is incomplete")
    return written_count


def _verify_result_contracts(
    results: Mapping[str, Mapping[str, Any]],
    *,
    hybrid: bool,
    paddle_delivery: Mapping[str, Any],
) -> dict[str, str]:
    expected_fingerprint: dict[str, str] | None = None
    expected_bundle_sha256 = str(paddle_delivery["contract_sha256"])
    expected_bundle_name = Path(str(paddle_delivery["contract"])).name
    for source, result in results.items():
        contracts = result.get("model_contracts")
        if not isinstance(contracts, Mapping):
            raise ComparisonError(f"Result has no model_contracts: {source}")
        fingerprint = {
            key: _lower_sha256(contracts.get(key), f"{source} model_contracts.{key}")
            for key in ARTIFACT_HASH_KEYS
        }
        if expected_fingerprint is None:
            expected_fingerprint = fingerprint
        elif fingerprint != expected_fingerprint:
            changed = sorted(key for key in ARTIFACT_HASH_KEYS if fingerprint[key] != expected_fingerprint[key])
            raise ComparisonError(
                f"Unified/detector/device artifact hashes vary within one A/B run ({source}): {', '.join(changed)}"
            )
        bundle_name = contracts.get("ocr_bundle")
        bundle_sha256 = contracts.get("ocr_bundle_contract_sha256")
        if hybrid:
            if bundle_name != expected_bundle_name:
                raise ComparisonError(
                    f"{source} model_contracts.ocr_bundle does not name the verified delivery contract"
                )
            observed_bundle_sha256 = _lower_sha256(
                bundle_sha256,
                f"{source} model_contracts.ocr_bundle_contract_sha256",
            )
            if observed_bundle_sha256 != expected_bundle_sha256:
                raise ComparisonError(
                    f"{source} PP-OCR contract hash does not match the verified delivery: "
                    f"expected {expected_bundle_sha256}, got {observed_bundle_sha256}"
                )
        elif bundle_name is not None or bundle_sha256 is not None:
            raise ComparisonError(f"Baseline result unexpectedly binds PP-OCR: {source}")
    if expected_fingerprint is None:  # pragma: no cover - guarded by non-empty manifest
        raise ComparisonError("A/B run contains no results")
    return expected_fingerprint


def _without_ocr(detection: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in detection.items() if key != "ocr"}


def _detections_by_label(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    detections = result.get("detections")
    if not isinstance(detections, list):
        raise ComparisonError("Result has no detections array")
    by_label: dict[str, dict[str, Any]] = {}
    for detection in detections:
        if not isinstance(detection, dict) or not isinstance(detection.get("label"), str):
            raise ComparisonError("Result contains an invalid detection")
        by_label[str(detection["label"])] = detection
    return by_label


def _canonical_detector_score(field: Mapping[str, Any], description: str) -> float | None:
    """Read the detector score across the compatible read/unreadable shapes."""
    detector_score = field.get("detector_score")
    unreadable_score = field.get("score")
    if detector_score is not None and unreadable_score is not None:
        raise ComparisonError(
            f"{description} contains both detector_score and unreadable compatibility score"
        )
    value = detector_score if detector_score is not None else unreadable_score
    return None if value is None else _finite_number(value, f"{description} detector score")


def compare(
    *,
    baseline_dir: Path,
    hybrid_dir: Path,
    paddle_delivery: Path,
    output: Path,
    max_p95_overhead_ms: float = MAX_P95_OVERHEAD_MS,
    mode: str = "pilot",
) -> tuple[dict[str, Any], bool]:
    baseline_dir = baseline_dir.resolve()
    hybrid_dir = hybrid_dir.resolve()
    output = output.resolve()
    if mode not in {"pilot", "formal"}:
        raise ComparisonError("mode must be pilot or formal")
    max_p95_overhead_ms = _finite_number(max_p95_overhead_ms, "max p95 overhead")
    if not 0.0 <= max_p95_overhead_ms <= MAX_P95_OVERHEAD_MS:
        raise ComparisonError(
            f"max p95 overhead must be between 0 and the fixed release ceiling {MAX_P95_OVERHEAD_MS:.1f} ms"
        )
    if output.exists():
        raise ComparisonError(f"Refusing to overwrite A/B report: {output}")

    paddle_identity = _verify_paddle_delivery(paddle_delivery)
    baseline_summary = _summary(baseline_dir)
    hybrid_summary = _summary(hybrid_dir)
    baseline_count = _require_cpu(baseline_summary, hybrid=False)
    hybrid_count = _require_cpu(hybrid_summary, hybrid=True)
    if baseline_count != hybrid_count:
        raise ComparisonError("Baseline and hybrid CLI summary counts differ")
    baseline_results = _result_map(baseline_dir, expected_count=baseline_count)
    hybrid_results = _result_map(hybrid_dir, expected_count=hybrid_count)
    if baseline_results.keys() != hybrid_results.keys():
        raise ComparisonError("Baseline and hybrid source sets differ")
    baseline_artifact_hashes = _verify_result_contracts(
        baseline_results,
        hybrid=False,
        paddle_delivery=paddle_identity,
    )
    hybrid_artifact_hashes = _verify_result_contracts(
        hybrid_results,
        hybrid=True,
        paddle_delivery=paddle_identity,
    )
    if baseline_artifact_hashes != hybrid_artifact_hashes:
        raise ComparisonError("Unified/detector/device artifact hashes differ between A/B runs")

    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for source in sorted(baseline_results):
        baseline = baseline_results[source]
        hybrid = hybrid_results[source]
        source_failures: list[str] = []
        for key in (
            "result_schema_version",
            "result_semantics_version",
            "source",
            "inference_engine",
            "geometry",
            "device",
        ):
            if baseline.get(key) != hybrid.get(key):
                source_failures.append(f"{key} changed")

        baseline_contracts = baseline.get("model_contracts")
        hybrid_contracts = hybrid.get("model_contracts")
        if not isinstance(baseline_contracts, Mapping) or not isinstance(hybrid_contracts, Mapping):
            source_failures.append("model contracts missing")
        else:
            for key in INVARIANT_CONTRACT_KEYS:
                if baseline_contracts.get(key) != hybrid_contracts.get(key):
                    source_failures.append(f"model_contracts.{key} changed")
            # Exact PP-OCR contract/hash validation is performed for every
            # result before semantic comparisons begin.

        baseline_fields = baseline.get("fields")
        hybrid_fields = hybrid.get("fields")
        if not isinstance(baseline_fields, Mapping) or not isinstance(hybrid_fields, Mapping):
            source_failures.append("fields missing")
            recipient_candidate = None
        else:
            for field in INVARIANT_FIELDS:
                if baseline_fields.get(field) != hybrid_fields.get(field):
                    source_failures.append(f"fields.{field} changed")
            baseline_recipient = baseline_fields.get("recipient")
            hybrid_recipient = hybrid_fields.get("recipient")
            if not isinstance(baseline_recipient, Mapping) or not isinstance(hybrid_recipient, Mapping):
                source_failures.append("recipient field missing")
                recipient_candidate = None
            else:
                recipient_candidate = hybrid_recipient.get("candidate")
                if _canonical_detector_score(
                    baseline_recipient, "baseline recipient"
                ) != _canonical_detector_score(hybrid_recipient, "hybrid recipient"):
                    source_failures.append("recipient detector_score changed")
                for key in ("delivery_policy", "delivery_value", "value"):
                    if baseline_recipient.get(key) != hybrid_recipient.get(key):
                        source_failures.append(f"recipient {key} changed")
                if not isinstance(recipient_candidate, str) or not recipient_candidate:
                    source_failures.append("hybrid recipient candidate missing")
                if hybrid_recipient.get("ctc_candidate") != recipient_candidate:
                    source_failures.append("hybrid recipient CTC candidate disagrees")

        baseline_detections = _detections_by_label(baseline)
        hybrid_detections = _detections_by_label(hybrid)
        if baseline_detections.keys() != hybrid_detections.keys():
            source_failures.append("detection labels changed")
        else:
            for label in baseline_detections:
                if label == "recipient_field":
                    equal = _without_ocr(baseline_detections[label]) == _without_ocr(hybrid_detections[label])
                else:
                    equal = baseline_detections[label] == hybrid_detections[label]
                if not equal:
                    source_failures.append(f"detection {label} changed")

        comparisons.append(
            {
                "source": baseline.get("source"),
                "recipient_candidate": recipient_candidate,
                "invariant": not source_failures,
                "failures": source_failures,
            }
        )
        failures.extend(f"{baseline.get('source')}: {failure}" for failure in source_failures)

    baseline_p95 = _finite_number(
        baseline_summary.get("inference_latency_ms", {}).get("p95"), "baseline p95"
    )
    hybrid_p95 = _finite_number(
        hybrid_summary.get("inference_latency_ms", {}).get("p95"), "hybrid p95"
    )
    overhead = hybrid_p95 - baseline_p95
    if overhead > max_p95_overhead_ms:
        failures.append(
            f"CPU p95 overhead {overhead:.4f} ms exceeds {max_p95_overhead_ms:.4f} ms"
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": KIND,
        "evaluation_mode": mode,
        "records": len(comparisons),
        "input_set_identical": True,
        "cli_summary_counts_verified": True,
        "artifact_hashes": baseline_artifact_hashes,
        "paddle_delivery": paddle_identity,
        "invariant_records": sum(bool(row["invariant"]) for row in comparisons),
        "recipient_candidate_coverage": sum(
            isinstance(row["recipient_candidate"], str) and bool(row["recipient_candidate"])
            for row in comparisons
        )
        / len(comparisons),
        "cpu": {
            "baseline_inference_latency_ms": baseline_summary.get("inference_latency_ms"),
            "hybrid_inference_latency_ms": hybrid_summary.get("inference_latency_ms"),
            "baseline_stage_latency_ms": baseline_summary.get("stage_latency_ms"),
            "hybrid_stage_latency_ms": hybrid_summary.get("stage_latency_ms"),
            "p95_overhead_ms": overhead,
            "max_p95_overhead_ms": max_p95_overhead_ms,
        },
        "failures": failures,
        "accepted": not failures,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparisons.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparisons),
        encoding="utf-8",
    )
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    return summary, not failures


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= MAX_P95_OVERHEAD_MS:
        raise argparse.ArgumentTypeError(
            f"must be finite and between 0 and the fixed release ceiling {MAX_P95_OVERHEAD_MS:.1f}"
        )
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--hybrid", required=True, type=Path)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument(
        "--max-p95-overhead-ms",
        type=_nonnegative_float,
        default=MAX_P95_OVERHEAD_MS,
        help=f"CPU p95 overhead ceiling; may be tightened but never exceed {MAX_P95_OVERHEAD_MS:.1f} ms",
    )
    args = parser.parse_args(argv)
    try:
        summary, accepted = compare(
            baseline_dir=args.baseline,
            hybrid_dir=args.hybrid,
            paddle_delivery=args.delivery,
            output=args.output,
            max_p95_overhead_ms=args.max_p95_overhead_ms,
            mode=args.mode,
        )
    except (ComparisonError, OSError, ValueError) as error:
        print(f"Hybrid recipient A/B failed: {error}")
        return 2
    print(
        f"hybrid_recipient_cpu_ab records={summary['records']} "
        f"invariant={summary['invariant_records']}/{summary['records']} "
        f"recipient_coverage={summary['recipient_candidate_coverage']:.2%} "
        f"p95_overhead_ms={summary['cpu']['p95_overhead_ms']:.2f} "
        f"accepted={summary['accepted']}"
    )
    print(f"A/B report: {args.output.resolve()}")
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
