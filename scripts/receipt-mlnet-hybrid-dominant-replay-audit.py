#!/usr/bin/env python3
"""Fail-closed audit for the 204-record dominant-recipient CPU replay.

The audit is deliberately inference-free.  It binds the rejected 10,016-image
formal A/B run, its failure diagnostic, the v4 truth-free consensus probe, and
a separately produced fresh CPU replay.  Every non-recipient result value must
remain type-sensitively identical to the old hybrid result.  The only accepted
delta is the recipient candidate and its recipient-only diagnostics.

This is targeted diagnostic evidence, not a formal delivery gate.  A passing
report always publishes ``formal_delivery_gate=false``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGETED_REPLAY_SCRIPT = REPOSITORY_ROOT / "scripts" / "receipt-mlnet-hybrid-targeted-replay.py"
_SPEC = importlib.util.spec_from_file_location(
    "receipt_mlnet_hybrid_targeted_replay_for_dominant_audit",
    TARGETED_REPLAY_SCRIPT,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - repository corruption
    raise RuntimeError(f"could not load {TARGETED_REPLAY_SCRIPT}")
REPLAY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(REPLAY)


FORMAL_RECORDS = 10016
FAILURE_RECORDS = 204
EXPECTED_CANDIDATES = 75
EXPECTED_MISSING = 129
EXPECTED_DOMINANT = 2
EXPECTED_PREEXISTING = 73
DOMINANT_ROUTE = "independent_crop_dominant_three_crop_consensus"
EXACT_ROUTE = "independent_crop_exact_consensus"
PROBE_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_truth_probe_summary_v1"
PROBE_FINDING_KIND = "receipt_mlnet_hybrid_failure_truth_probe_finding_v1"
AUDIT_SUMMARY_KIND = "receipt_mlnet_hybrid_dominant_replay_audit_v1"
AUDIT_FINDING_KIND = "receipt_mlnet_hybrid_dominant_replay_audit_finding_v1"
RECIPIENT_DIAGNOSTIC_PREFIX = "hybrid_ocr_"
MODEL_HASH_KEYS = (
    "detector_sha256",
    "detector_contract_sha256",
    "device_sha256",
    "device_contract_sha256",
    "ocr_bundle_contract_sha256",
    "unified_ocr_model_sha256",
    "unified_ocr_labels_sha256",
    "unified_ocr_contract_sha256",
)
RECIPIENT_DYNAMIC_KEYS = frozenset(
    {
        "state",
        "raw",
        "ocr_confidence",
        "detector_score",
        "score",
        "candidate",
        "ctc_candidate",
        "ctc_confidence",
        "structured_candidate",
        "structured_confidence",
    }
)


class AuditError(ValueError):
    """One or more replay-evidence contracts were not proved."""


def _fail(message: str) -> None:
    raise AuditError(message)


def _identity(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return REPLAY._file_identity(path, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _assert_identity(identity: object, *, description: str) -> dict[str, Any]:
    try:
        return REPLAY._assert_identity(identity, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _load_json(path: Path, *, description: str) -> Any:
    try:
        return REPLAY._load_json(path, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _load_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    try:
        return REPLAY._load_jsonl(path, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _source_key(value: object) -> str:
    return REPLAY._source_key(value)


def _same(left: Any, right: Any) -> bool:
    return REPLAY._type_sensitive_equal(left, right)


def _nonempty(value: object, *, description: str) -> str:
    try:
        return REPLAY._nonempty_string(value, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _field(result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        return REPLAY._field(result, name)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error


def _finite_confidence(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{description} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        _fail(f"{description} must be within [0, 1]")
    return number


def _validate_model_contracts(result: Mapping[str, Any], *, description: str) -> Mapping[str, Any]:
    contracts = result.get("model_contracts")
    if not isinstance(contracts, Mapping):
        _fail(f"{description} has no model_contracts object")
    for key in ("detector", "device", "ocr_bundle", "unified_ocr_model", "unified_ocr_contract"):
        _nonempty(contracts.get(key), description=f"{description} model_contracts.{key}")
    for key in MODEL_HASH_KEYS:
        value = contracts.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _fail(f"{description} model_contracts.{key} is not a lowercase SHA-256")
    return contracts


def _resolve_regular_directory(path: Path, *, description: str) -> Path:
    if path.is_symlink():
        _fail(f"{description} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"missing {description}: {path}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        _fail(f"{description} must be a regular directory: {resolved}")
    return resolved


def _assert_bound_identity(
    bound: object,
    actual_path: Path,
    *,
    description: str,
) -> dict[str, Any]:
    if not isinstance(bound, Mapping):
        _fail(f"{description} binding must be an object")
    actual = _identity(actual_path, description=description)
    try:
        REPLAY._assert_bound_identity(bound, actual, description=description)
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error
    return actual


def _expected_count(payload: Mapping[str, Any], key: str, expected: int) -> None:
    value = payload.get(key)
    if isinstance(value, bool) or value != expected:
        _fail(f"{key} must equal {expected}, got {value!r}")


def _load_probe(
    probe_directory: Path,
    *,
    diagnostic_directory: Path,
    missing_keys: set[str],
) -> dict[str, Any]:
    root = _resolve_regular_directory(probe_directory, description="consensus probe")
    summary_path = root / "summary.json"
    findings_path = root / "findings.jsonl"
    summary_identity = _identity(summary_path, description="consensus probe summary")
    findings_identity = _identity(findings_path, description="consensus probe findings")
    summary = _load_json(summary_path, description="consensus probe summary")
    findings = _load_jsonl(findings_path, description="consensus probe findings")
    if not isinstance(summary, Mapping):
        _fail("consensus probe summary must be an object")
    for key, expected in (
        ("schema_version", 1),
        ("kind", PROBE_SUMMARY_KIND),
        ("read_only_existing_diagnostic", True),
        ("ocr_rerun", False),
        ("truth_used_for_analysis_only", True),
        ("runtime_truth_lookup", False),
        ("formal_delivery_gate", False),
        ("findings_records", FAILURE_RECORDS),
        ("unique_sources", FAILURE_RECORDS),
    ):
        if type(summary.get(key)) is not type(expected) or summary.get(key) != expected:
            _fail(f"consensus probe summary {key} must be {expected!r}")

    formal = summary.get("formal_contract")
    if not isinstance(formal, Mapping):
        _fail("consensus probe has no formal_contract")
    for key, expected in (
        ("comparison_evaluation_mode", "formal"),
        ("comparison_records", FORMAL_RECORDS),
        ("failed_records", FAILURE_RECORDS),
        ("recipient_missing_only_records", FAILURE_RECORDS),
        ("recipient_missing_with_additional_failures_records", 0),
        ("non_missing_invariant_failure_records", 0),
    ):
        if type(formal.get(key)) is not type(expected) or formal.get(key) != expected:
            _fail(f"consensus probe formal_contract.{key} must be {expected!r}")

    source_evidence = summary.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        _fail("consensus probe has no source_evidence")
    diagnostic_summary_identity = _assert_bound_identity(
        source_evidence.get("input_summary"),
        diagnostic_directory / "summary.json",
        description="probe-bound diagnostic summary",
    )
    diagnostic_findings_identity = _assert_bound_identity(
        source_evidence.get("input_findings"),
        diagnostic_directory / "findings.jsonl",
        description="probe-bound diagnostic findings",
    )

    teacher = summary.get("paddle_teacher_consensus")
    remaining = summary.get("remaining_failure_analysis")
    if not isinstance(teacher, Mapping) or not isinstance(remaining, Mapping):
        _fail("consensus probe lacks teacher/remaining summaries")
    for key, expected in (
        ("external_truth", False),
        ("truth_used_for_analysis_only", True),
        ("formal_delivery_gate", False),
        ("records", EXPECTED_CANDIDATES),
    ):
        if type(teacher.get(key)) is not type(expected) or teacher.get(key) != expected:
            _fail(f"consensus probe paddle_teacher_consensus.{key} must be {expected!r}")
    if teacher.get("interpretation") != "self_consistency_coverage_not_human_accuracy":
        _fail("consensus probe teacher interpretation changed")
    expected_routes = [
        {"name": DOMINANT_ROUTE, "records": EXPECTED_DOMINANT},
        {"name": EXACT_ROUTE, "records": EXPECTED_PREEXISTING},
    ]
    if teacher.get("by_runtime_route") != expected_routes:
        _fail("consensus probe runtime-route counts changed")
    _expected_count(remaining, "records", EXPECTED_MISSING)
    _expected_count(remaining, "strict_candidate_records", EXPECTED_CANDIDATES)

    contract = teacher.get("contract")
    if not isinstance(contract, Mapping):
        _fail("consensus probe teacher contract is missing")
    for key, expected in (
        ("dominant_fallback_requires_multiple_eligible_candidates", True),
        ("dominant_fallback_requires_same_exact_line_in_all_crops", 3),
        ("dominant_fallback_requires_unique_all_crop_candidate", True),
        ("requires_same_exact_line_in_independent_crops", 2),
    ):
        if type(contract.get(key)) is not type(expected) or contract.get(key) != expected:
            _fail(f"consensus probe teacher contract {key} changed")

    if len(findings) != FAILURE_RECORDS:
        _fail(
            f"consensus probe findings must contain {FAILURE_RECORDS} records, got {len(findings)}"
        )
    by_source: dict[str, dict[str, Any]] = {}
    route_counts: Counter[str] = Counter()
    candidate_count = 0
    for index, finding in enumerate(findings):
        for key, expected in (
            ("schema_version", 1),
            ("kind", PROBE_FINDING_KIND),
            ("truth_used_for_analysis_only", True),
            ("runtime_truth_lookup", False),
            ("formal_delivery_gate", False),
        ):
            if type(finding.get(key)) is not type(expected) or finding.get(key) != expected:
                _fail(f"consensus probe finding {index} {key} must be {expected!r}")
        source = _nonempty(
            finding.get("source"), description=f"consensus probe finding {index} source"
        )
        source_key = _source_key(source)
        if source_key in by_source:
            _fail(f"duplicate consensus probe source: {source!r}")
        shadow = finding.get("shadow_candidate_truth_free")
        teacher_shadow = finding.get("paddle_teacher_consensus")
        if not isinstance(shadow, Mapping) or not _same(shadow, teacher_shadow):
            _fail(f"consensus probe finding {source!r} teacher/shadow mismatch")
        candidate = shadow.get("candidate")
        route = shadow.get("runtime_route")
        state = shadow.get("state")
        if isinstance(candidate, str) and candidate.strip():
            if state != "candidate" or route not in {EXACT_ROUTE, DOMINANT_ROUTE}:
                _fail(f"consensus probe candidate {source!r} has an invalid state/route")
            candidate_count += 1
            route_counts[str(route)] += 1
        else:
            if candidate is not None or route is not None or state == "candidate":
                _fail(f"consensus probe missing candidate {source!r} is inconsistent")
        by_source[source_key] = {
            "source": source,
            "candidate": candidate,
            "route": route,
            "state": state,
        }
    if set(by_source) != missing_keys:
        _fail(
            "consensus probe source set differs from the 204 formal omissions: "
            f"missing={len(missing_keys - set(by_source))} "
            f"extra={len(set(by_source) - missing_keys)}"
        )
    if candidate_count != EXPECTED_CANDIDATES:
        _fail(f"consensus probe candidate count must be {EXPECTED_CANDIDATES}")
    if route_counts != Counter(
        {EXACT_ROUTE: EXPECTED_PREEXISTING, DOMINANT_ROUTE: EXPECTED_DOMINANT}
    ):
        _fail("consensus probe per-finding route counts changed")
    return {
        "root": root,
        "summary": dict(summary),
        "by_source": by_source,
        "identities": {
            "probe_summary": summary_identity,
            "probe_findings": findings_identity,
            "diagnostic_summary": diagnostic_summary_identity,
            "diagnostic_findings": diagnostic_findings_identity,
        },
    }


def _load_replay_input(
    path: Path, *, expected_sources: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    identity = _identity(path, description="fresh replay input list")
    sources = path.read_text(encoding="utf-8-sig").splitlines()
    if len(sources) != FAILURE_RECORDS or any(not source.strip() for source in sources):
        _fail(f"fresh replay input list must contain exactly {FAILURE_RECORDS} nonblank rows")
    if any(source != source.strip() for source in sources):
        _fail("fresh replay input list contains surrounding whitespace")
    observed_keys = [_source_key(source) for source in sources]
    expected_keys = [_source_key(source) for source in expected_sources]
    if set(observed_keys) != set(expected_keys):
        _fail("fresh replay input list source set differs from the formal omission set")
    if len(set(observed_keys)) != FAILURE_RECORDS:
        _fail("fresh replay input list contains duplicate sources")
    identity.update(
        {
            "records": FAILURE_RECORDS,
            "normalized_source_set_sha256": REPLAY._normalized_source_set_sha256(
                observed_keys
            ),
        }
    )
    return identity, sources


def _load_cli_closure(directory: Path) -> dict[str, Any]:
    root = _resolve_regular_directory(directory, description="fresh replay CLI app")
    rows: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    lower_paths: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: (item.as_posix().casefold(), item.as_posix())):
        if path.is_symlink():
            _fail(f"fresh replay CLI closure contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        key = relative.casefold()
        if key in lower_paths:
            _fail(f"fresh replay CLI closure has a duplicate case-insensitive path: {relative}")
        lower_paths.add(key)
        identity = _identity(path, description=f"fresh replay CLI file {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }
        )
        identities[key] = identity
    if not rows:
        _fail("fresh replay CLI closure is empty")
    required = {"receiptmlnet.cli.dll", "onnxruntime.dll"}
    if not required.issubset(lower_paths):
        _fail("fresh replay CLI closure lacks ReceiptMlNet.Cli.dll or onnxruntime.dll")
    forbidden_tokens = ("cuda", "cudnn", "tensorrt", "onnxruntime_providers_cuda")
    forbidden = [row["path"] for row in rows if any(token in row["path"].casefold() for token in forbidden_tokens)]
    if forbidden:
        _fail(f"fresh replay CPU CLI closure contains GPU provider files: {forbidden[:3]}")
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "root": root,
        "rows": rows,
        "identities": identities,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
        "assembly": identities["receiptmlnet.cli.dll"],
        "onnxruntime": identities["onnxruntime.dll"],
    }


def _recipient_detection(result: Mapping[str, Any], *, description: str) -> Mapping[str, Any]:
    detections = result.get("detections")
    if not isinstance(detections, list):
        _fail(f"{description} has no detections array")
    matches = [
        detection
        for detection in detections
        if isinstance(detection, Mapping) and detection.get("label") == "recipient_field"
    ]
    if len(matches) != 1:
        _fail(f"{description} must have exactly one recipient_field detection")
    return matches[0]


def _effective_detector_score(field: Mapping[str, Any], *, description: str) -> float:
    values = [field.get("detector_score"), field.get("score")]
    present = [value for value in values if value is not None]
    if len(present) != 1:
        _fail(f"{description} must have exactly one detector score representation")
    return _finite_confidence(present[0], description=f"{description} detector score")


def _matches_runtime_confidence_projection(projected: float, raw: float) -> bool:
    """Validate the CLI's six-decimal MathF projection without widening it.

    ``MathF.Round`` operates in float32 and can select the opposite side of a
    decimal half-step from Python's binary64 ``round``.  The serialized field
    must itself be a six-decimal value and remain within one six-decimal
    half-step plus one float32 ULP at confidence magnitudes below one.
    """

    return projected == round(projected, 6) and math.isclose(
        projected,
        raw,
        rel_tol=0.0,
        abs_tol=5.6e-7,
    )


def _assert_nonrecipient_identical(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    source: str,
) -> None:
    old_keys = set(old)
    new_keys = set(new)
    if old_keys != new_keys:
        _fail(f"fresh replay {source!r} changed top-level result keys")
    for key in old_keys - {"fields", "detections"}:
        if not _same(old.get(key), new.get(key)):
            _fail(f"fresh replay {source!r} changed {key}")

    old_fields = old.get("fields")
    new_fields = new.get("fields")
    if not isinstance(old_fields, Mapping) or not isinstance(new_fields, Mapping):
        _fail(f"fresh replay {source!r} has no fields object")
    if set(old_fields) != set(new_fields):
        _fail(f"fresh replay {source!r} changed field keys")
    for field in old_fields:
        if field == "recipient":
            continue
        if not _same(old_fields[field], new_fields[field]):
            _fail(f"fresh replay {source!r} changed fields.{field}")

    old_detections = old.get("detections")
    new_detections = new.get("detections")
    if not isinstance(old_detections, list) or not isinstance(new_detections, list):
        _fail(f"fresh replay {source!r} has invalid detections")
    if len(old_detections) != len(new_detections):
        _fail(f"fresh replay {source!r} changed detection count")
    for index, (old_detection, new_detection) in enumerate(
        zip(old_detections, new_detections, strict=True)
    ):
        if not isinstance(old_detection, Mapping) or not isinstance(new_detection, Mapping):
            _fail(f"fresh replay {source!r} detection {index} is invalid")
        if old_detection.get("label") == "recipient_field":
            old_static = {key: value for key, value in old_detection.items() if key != "ocr"}
            new_static = {key: value for key, value in new_detection.items() if key != "ocr"}
            if not _same(old_static, new_static):
                _fail(f"fresh replay {source!r} changed recipient detector output")
        elif not _same(old_detection, new_detection):
            _fail(f"fresh replay {source!r} changed detection {index}")


def _assert_recipient_contract(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    source: str,
    expected_candidate: str | None,
    expected_route: str | None,
) -> dict[str, Any]:
    old_field = _field(old, "recipient")
    new_field = _field(new, "recipient")
    if REPLAY._candidate(old, "recipient") is not None or old_field.get("ctc_candidate") is not None:
        _fail(f"old formal omission {source!r} unexpectedly has a recipient candidate")

    for key in set(old_field) | set(new_field):
        if key in RECIPIENT_DYNAMIC_KEYS or key.startswith(RECIPIENT_DIAGNOSTIC_PREFIX):
            continue
        if not _same(old_field.get(key), new_field.get(key)):
            _fail(f"fresh replay {source!r} changed recipient non-diagnostic key {key}")

    old_detection = _recipient_detection(old, description=f"old result {source!r}")
    new_detection = _recipient_detection(new, description=f"fresh result {source!r}")
    detector_score = _finite_confidence(
        new_detection.get("score"), description=f"fresh result {source!r} detection score"
    )
    if (
        not _matches_runtime_confidence_projection(
            _effective_detector_score(
                old_field, description=f"old recipient {source!r}"
            ),
            detector_score,
        )
    ):
        _fail(f"old result {source!r} recipient detector score disagrees with detection")
    if (
        not _matches_runtime_confidence_projection(
            _effective_detector_score(
                new_field, description=f"fresh recipient {source!r}"
            ),
            detector_score,
        )
    ):
        _fail(f"fresh result {source!r} recipient detector score disagrees with detection")

    candidate = new_field.get("candidate")
    ctc_candidate = new_field.get("ctc_candidate")
    route = new_field.get("hybrid_ocr_route")
    if expected_candidate is None:
        if candidate is not None or ctc_candidate is not None:
            _fail(f"fresh replay {source!r} emitted an unexpected recipient candidate")
        if route not in (None, "none") or new_field.get("state") != "unreadable":
            _fail(f"fresh replay missing recipient {source!r} has an invalid route/state")
        return {"candidate": None, "route": route, "ctc_matches": True}

    if candidate != expected_candidate or ctc_candidate != candidate:
        _fail(f"fresh replay {source!r} candidate/ctc_candidate differs from the probe")
    if route != expected_route or new_field.get("state") != "review":
        _fail(f"fresh replay {source!r} route/state differs from the probe")
    if new_field.get("raw") != candidate:
        _fail(f"fresh replay {source!r} raw recipient differs from candidate")
    ocr_confidence = _finite_confidence(
        new_field.get("ocr_confidence"), description=f"fresh recipient {source!r} OCR confidence"
    )
    ctc_confidence = _finite_confidence(
        new_field.get("ctc_confidence"), description=f"fresh recipient {source!r} CTC confidence"
    )
    if ocr_confidence != ctc_confidence:
        _fail(f"fresh replay {source!r} OCR/CTC confidence mismatch")
    if new_field.get("structured_candidate") is not None or new_field.get("structured_confidence") is not None:
        _fail(f"fresh replay {source!r} unexpectedly emitted a structured recipient")
    ocr = new_detection.get("ocr")
    if not isinstance(ocr, Mapping) or ocr.get("text") != candidate:
        _fail(f"fresh replay {source!r} recipient detection OCR differs from candidate")
    detection_ocr_confidence = _finite_confidence(
        ocr.get("confidence"),
        description=f"fresh recipient {source!r} detection OCR confidence",
    )
    if not _matches_runtime_confidence_projection(
        ocr_confidence, detection_ocr_confidence
    ):
        _fail(f"fresh replay {source!r} detection/field OCR confidence mismatch")
    return {"candidate": candidate, "route": route, "ctc_matches": True}


def _assert_paths_do_not_overlap(output: Path, protected: Sequence[Path]) -> None:
    resolved_output = output.resolve()
    for path in protected:
        resolved = path.resolve()
        try:
            resolved_output.relative_to(resolved)
        except ValueError:
            pass
        else:
            _fail(f"audit output must not be inside protected input: {resolved}")
        try:
            resolved.relative_to(resolved_output)
        except ValueError:
            pass
        else:
            _fail(f"audit output must not contain protected input: {resolved}")


def audit(
    *,
    formal_root: Path,
    diagnostic_directory: Path,
    probe_directory: Path,
    replay_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.is_symlink():
        _fail(f"refusing a symbolic-link audit output: {output_directory}")
    output = output_directory.resolve()
    if output.exists():
        _fail(f"refusing to overwrite dominant replay audit: {output}")

    diagnostic_root = _resolve_regular_directory(
        diagnostic_directory, description="failure diagnostic"
    )
    try:
        formal = REPLAY._load_formal_ab(
            formal_root,
            diagnostic=diagnostic_root,
        )
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error
    if len(formal["missing_keys"]) != FAILURE_RECORDS:
        _fail(f"formal omission set must contain exactly {FAILURE_RECORDS} sources")
    missing_sources = [
        source
        for source in formal["input_sources"]
        if _source_key(source) in formal["missing_keys"]
    ]
    if len(missing_sources) != FAILURE_RECORDS:
        _fail("canonical formal omission order is incomplete")

    probe = _load_probe(
        probe_directory,
        diagnostic_directory=diagnostic_root,
        missing_keys=set(formal["missing_keys"]),
    )
    replay_root = _resolve_regular_directory(
        replay_directory, description="fresh replay"
    )
    replay_input_path = Path(str(replay_root) + ".inputs.txt")
    cli_directory = Path(str(replay_root) + ".cli-app")
    replay_input, replay_sources = _load_replay_input(
        replay_input_path, expected_sources=missing_sources
    )
    try:
        replay = REPLAY._load_run(
            replay_root,
            expected_sources=replay_sources,
            hybrid=True,
        )
    except REPLAY.ReplayError as error:
        raise AuditError(str(error)) from error
    if replay["root"] == formal["hybrid"]["root"]:
        _fail("fresh replay directory aliases the old formal hybrid run")

    cli = _load_cli_closure(cli_directory)
    _assert_paths_do_not_overlap(
        output,
        [
            formal["root"],
            diagnostic_root,
            probe["root"],
            replay["root"],
            replay_input_path,
            cli["root"],
        ],
    )

    findings: list[dict[str, Any]] = []
    candidate_count = 0
    route_counts: Counter[str] = Counter()
    preexisting_preserved = 0
    bound_model_contracts: Mapping[str, Any] | None = None
    all_input_identities: list[tuple[str, Mapping[str, Any]]] = []
    all_input_identities.extend(
        (f"formal {name}", identity)
        for name, identity in formal["source_evidence"].items()
    )
    all_input_identities.extend(
        (name, identity) for name, identity in probe["identities"].items()
    )
    all_input_identities.extend(
        (
            ("fresh replay summary", replay["summary_identity"]),
            ("fresh replay manifest", replay["manifest_identity"]),
            ("fresh replay input list", replay_input),
        )
    )
    all_input_identities.extend(
        (f"fresh CLI {relative}", identity)
        for relative, identity in cli["identities"].items()
    )

    for source in missing_sources:
        key = _source_key(source)
        old_entry = formal["hybrid"]["results"].get(key)
        new_entry = replay["results"].get(key)
        probe_entry = probe["by_source"].get(key)
        if old_entry is None or new_entry is None or probe_entry is None:
            _fail(f"incomplete old/new/probe binding for {source!r}")
        old = old_entry["payload"]
        new = new_entry["payload"]
        old_contracts = _validate_model_contracts(
            old, description=f"old hybrid result {source!r}"
        )
        new_contracts = _validate_model_contracts(
            new, description=f"fresh replay result {source!r}"
        )
        if not _same(old_contracts, new_contracts):
            _fail(f"fresh replay {source!r} changed model contracts")
        if bound_model_contracts is None:
            bound_model_contracts = dict(new_contracts)
        elif not _same(bound_model_contracts, new_contracts):
            _fail("fresh replay uses more than one detector/device/OCR model contract")
        _assert_nonrecipient_identical(old, new, source=source)
        recipient = _assert_recipient_contract(
            old,
            new,
            source=source,
            expected_candidate=probe_entry["candidate"],
            expected_route=probe_entry["route"],
        )
        if recipient["candidate"] is not None:
            candidate_count += 1
            route_counts[str(recipient["route"])] += 1
            if recipient["route"] != DOMINANT_ROUTE:
                preexisting_preserved += 1
        old_identity = old_entry["identity"]
        new_identity = new_entry["identity"]
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = Path(str(formal["input_identity"]["path"])).parent / source_path
        source_identity = _identity(
            source_path, description=f"replayed source image {source}"
        )
        all_input_identities.append((f"replayed source image {source}", source_identity))
        all_input_identities.append((f"old hybrid result {source}", old_identity))
        all_input_identities.append((f"fresh replay result {source}", new_identity))
        findings.append(
            {
                "schema_version": 1,
                "kind": AUDIT_FINDING_KIND,
                "source": source,
                "formal_delivery_gate": False,
                "diagnostic_only": True,
                "source_image": source_identity,
                "old_hybrid_result": old_identity,
                "fresh_replay_result": new_identity,
                "probe_candidate": probe_entry["candidate"],
                "probe_route": probe_entry["route"],
                "replay_candidate": recipient["candidate"],
                "replay_route": recipient["route"],
                "ctc_candidate_equals_candidate": recipient["ctc_matches"],
                "detector_device_geometry_and_other_fields_identical": True,
            }
        )

    if len(findings) != FAILURE_RECORDS:
        _fail(f"audit must contain exactly {FAILURE_RECORDS} findings")
    if candidate_count != EXPECTED_CANDIDATES:
        _fail(f"fresh replay candidate count must be {EXPECTED_CANDIDATES}, got {candidate_count}")
    if FAILURE_RECORDS - candidate_count != EXPECTED_MISSING:
        _fail(f"fresh replay missing count must be {EXPECTED_MISSING}")
    expected_route_counts = Counter(
        {EXACT_ROUTE: EXPECTED_PREEXISTING, DOMINANT_ROUTE: EXPECTED_DOMINANT}
    )
    if route_counts != expected_route_counts:
        _fail(f"fresh replay route counts changed: {dict(route_counts)}")
    if preexisting_preserved != EXPECTED_PREEXISTING:
        _fail(
            f"the {EXPECTED_PREEXISTING} pre-dominant recovered recipients were not all preserved"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        closure_path = stage / "cli-closure.json"
        findings_path = stage / "findings.jsonl"
        summary_path = stage / "summary.json"
        closure_path.write_text(
            json.dumps(
                cli["rows"],
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        findings_path.write_text(
            "".join(
                json.dumps(
                    finding,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for finding in findings
            ),
            encoding="utf-8",
            newline="\n",
        )
        if bound_model_contracts is None:  # pragma: no cover - count contract above
            _fail("fresh replay has no model contracts")
        summary = {
            "schema_version": 1,
            "kind": AUDIT_SUMMARY_KIND,
            "read_only_existing_results": True,
            "ocr_rerun": False,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "accepted": True,
            "counts": {
                "formal_records": FORMAL_RECORDS,
                "replay_records": len(findings),
                "candidate_records": candidate_count,
                "missing_records": FAILURE_RECORDS - candidate_count,
                "preexisting_exact_consensus_preserved": preexisting_preserved,
                "dominant_three_crop_records": route_counts[DOMINANT_ROUTE],
                "ctc_candidate_equals_candidate": candidate_count,
                "nonrecipient_invariant_records": len(findings),
                "transfer_status_non_success_to_success": 0,
            },
            "invariant_contract": {
                "detector": "type_sensitive_exact",
                "device": "type_sensitive_exact",
                "geometry": "type_sensitive_exact",
                "amount": "type_sensitive_exact",
                "time": "type_sensitive_exact",
                "payment_method": "type_sensitive_exact",
                "transfer_status": "type_sensitive_exact",
                "recipient_detection_ocr": "candidate_only",
                "recipient_candidate_route_and_diagnostics": "only_allowed_delta",
            },
            "routes": [
                {"name": name, "records": count}
                for name, count in sorted(route_counts.items())
            ],
            "cpu_contract": {
                "requested_device": "cpu",
                "unified_provider": "cpu",
                "paddle_ocr_provider": "cpu",
                "replay_errors": 0,
            },
            "model_contracts": bound_model_contracts,
            "cli_build": {
                "root": cli["root"].as_posix(),
                "assembly": cli["assembly"],
                "onnxruntime": cli["onnxruntime"],
                "file_count": len(cli["rows"]),
                "closure_sha256": cli["closure_sha256"],
                "closure_artifact": "cli-closure.json",
            },
            "source_evidence": {
                **formal["source_evidence"],
                "probe_summary": probe["identities"]["probe_summary"],
                "probe_findings": probe["identities"]["probe_findings"],
                "fresh_replay_input": replay_input,
                "fresh_replay_summary": replay["summary_identity"],
                "fresh_replay_manifest": replay["manifest_identity"],
            },
            "artifacts": {
                "summary": "summary.json",
                "findings": "findings.jsonl",
                "cli_closure": "cli-closure.json",
            },
            "warning": (
                "This 204-record replay audit is diagnostic-only and remains "
                "formal_delivery_gate=false. It cannot replace a fresh 10016-record formal run."
            ),
        }
        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for description, identity in all_input_identities:
            _assert_identity(identity, description=description)
        if output.exists() or output.is_symlink():
            _fail(f"refusing to overwrite dominant replay audit: {output}")
        try:
            REPLAY._publish_directory(stage, output, description="dominant replay audit")
        except REPLAY.ReplayError as error:
            raise AuditError(str(error)) from error
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {**summary, "output_directory": output.as_posix()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-directory", type=Path, required=True)
    parser.add_argument("--diagnostic-directory", type=Path, required=True)
    parser.add_argument("--probe-directory", type=Path, required=True)
    parser.add_argument("--replay-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = audit(
        formal_root=args.formal_directory,
        diagnostic_directory=args.diagnostic_directory,
        probe_directory=args.probe_directory,
        replay_directory=args.replay_directory,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "kind": AUDIT_SUMMARY_KIND,
                "accepted": summary["accepted"],
                "records": summary["counts"]["replay_records"],
                "candidate_records": summary["counts"]["candidate_records"],
                "missing_records": summary["counts"]["missing_records"],
                "dominant_three_crop_records": summary["counts"]["dominant_three_crop_records"],
                "formal_delivery_gate": False,
                "output_directory": summary["output_directory"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
