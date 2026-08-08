#!/usr/bin/env python3
"""Prepare and evaluate a diagnostic-only full-layout recipient shadow.

``prepare`` freezes exactly 61 unresolved records from the completed derived-
crop evaluation and 278 reference-bearing controls from the canonical formal
validation order.  ``evaluate`` accepts only a fresh 339-record CPU
``LayoutShadow`` output and extracts recipient evidence anchored by an exact
Chinese recipient label.  Neither command writes receipt fields or participates
in a formal delivery gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4


FORMAL_RECORDS = 10016
FORMAL_RECIPIENT_MISSING = 204
DERIVED_PLAN_RECORDS = 63
DERIVED_SHADOW_CANDIDATES = 2
TARGET_RECORDS = 61
CONTROL_RECORDS = 278
EXPECTED_RECORDS = TARGET_RECORDS + CONTROL_RECORDS

PLAN_SUMMARY_KIND = "receipt_mlnet_recipient_derived_crop_plan_summary_v1"
PLAN_RECORD_KIND = "receipt_mlnet_recipient_derived_crop_plan_record_v1"
DERIVED_SUMMARY_KIND = "receipt_mlnet_recipient_derived_crop_shadow_summary_v1"
DERIVED_RECORD_KIND = "receipt_mlnet_recipient_derived_crop_shadow_record_v1"
TRUTH_SUMMARY_KIND = "receipt_mlnet_hybrid_failure_truth_probe_summary_v1"
TRUTH_RECORD_KIND = "receipt_mlnet_hybrid_failure_truth_probe_finding_v1"
FORMAL_SUMMARY_KIND = "receipt_mlnet_formal_missing_fields_audit_summary_v1"
FORMAL_RECORD_KIND = "receipt_mlnet_formal_missing_fields_audit_finding_v1"
LAYOUT_SUMMARY_KIND = "receipt_ppocr_dotnet_cpu_layout_shadow_summary_v1"
LAYOUT_RECORD_KIND = "receipt_ppocr_dotnet_cpu_layout_shadow_record_v1"
SELECTION_SUMMARY_KIND = "receipt_mlnet_recipient_full_layout_shadow_selection_v1"
SELECTION_RECORD_KIND = "receipt_mlnet_recipient_full_layout_shadow_selection_record_v1"
EVALUATION_SUMMARY_KIND = "receipt_mlnet_recipient_full_layout_shadow_summary_v1"
EVALUATION_RECORD_KIND = "receipt_mlnet_recipient_full_layout_shadow_record_v1"

MINIMUM_LINE_CONFIDENCE = 0.80
LABELS = ("收款账户", "收款方", "收款人")
LABEL_ALTERNATION = "|".join(map(re.escape, LABELS))
LABEL_ONLY = re.compile(rf"^(?P<label>{LABEL_ALTERNATION})\s*[:：]?\s*$")
LABEL_RHS = re.compile(
    rf"^(?P<label>{LABEL_ALTERNATION})(?:\s*[:：]\s*|\s+)(?P<value>.+)$"
)
CONTROL_STRATA = ("existing_exact", "existing_wrong")
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FullLayoutShadowError(ValueError):
    """Raised when any frozen diagnostic input violates this contract."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _loads(text: str, *, location: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise FullLayoutShadowError(f"invalid JSON at {location}: {error}") from error


def _read_bytes(path: Path, *, description: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FullLayoutShadowError(f"missing {description}: {path}") from error
    if not resolved.is_file():
        raise FullLayoutShadowError(f"{description} is not a file: {resolved}")
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise FullLayoutShadowError(f"cannot read {description}: {resolved}: {error}") from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path, payload: bytes | None = None, *, description: str = "artifact") -> dict[str, Any]:
    data = _read_bytes(path, description=description) if payload is None else payload
    return {
        "path": path.resolve(strict=True).as_posix(),
        "sha256": _sha256(data),
        "size_bytes": len(data),
    }


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_bytes(path, description=description)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise FullLayoutShadowError(f"{description} is not UTF-8: {path}") from error
    value = _loads(text, location=str(path))
    if not isinstance(value, dict):
        raise FullLayoutShadowError(f"{description} must contain one object")
    return value, payload


def _load_jsonl(path: Path, *, description: str) -> tuple[list[dict[str, Any]], bytes]:
    payload = _read_bytes(path, description=description)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise FullLayoutShadowError(f"{description} is not UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise FullLayoutShadowError(f"blank line in {description}: {path}:{line_number}")
        value = _loads(line, location=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise FullLayoutShadowError(f"{description} row must be an object")
        rows.append(value)
    if not rows:
        raise FullLayoutShadowError(f"{description} is empty: {path}")
    return rows, payload


def _require_directory(path: Path, *, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FullLayoutShadowError(f"missing {description}: {path}") from error
    if not resolved.is_dir():
        raise FullLayoutShadowError(f"{description} is not a directory: {resolved}")
    return resolved


def _require_int(value: object, *, description: str, expected: int | None = None) -> int:
    if type(value) is not int:
        raise FullLayoutShadowError(f"{description} must be an integer")
    if expected is not None and value != expected:
        raise FullLayoutShadowError(f"{description} must be {expected}, found {value}")
    return value


def _require_number(
    value: object, *, description: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullLayoutShadowError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FullLayoutShadowError(f"{description} must be finite")
    if minimum is not None and result < minimum:
        raise FullLayoutShadowError(f"{description} is below {minimum}")
    if maximum is not None and result > maximum:
        raise FullLayoutShadowError(f"{description} is above {maximum}")
    return result


def _nonempty(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FullLayoutShadowError(f"{description} must be a non-empty string")
    return value


def _clean(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().with_name(name)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FullLayoutShadowError(f"cannot load required contract module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_key(formal_module, value: object) -> str:
    return formal_module._source_key(value)


def _same_identity(contract: object, actual: Mapping[str, Any], *, description: str) -> None:
    if not isinstance(contract, Mapping):
        raise FullLayoutShadowError(f"missing {description} identity")
    if dict(contract) != dict(actual):
        raise FullLayoutShadowError(f"{description} identity differs")


def _assert_identities_current(identities: Mapping[str, Mapping[str, Any]]) -> None:
    for description, expected in identities.items():
        actual = _identity(Path(str(expected.get("path", ""))), description=description)
        if actual != dict(expected):
            raise FullLayoutShadowError(f"{description} changed while diagnostic evidence was built")


def _contained_artifact(directory: Path, contract: Mapping[str, Any], *, description: str) -> tuple[Path, dict[str, Any]]:
    relative = contract.get("path", contract.get("relative_path"))
    relative = _nonempty(relative, description=f"{description} relative path")
    path = (directory / relative).resolve(strict=True)
    try:
        path.relative_to(directory.resolve(strict=True))
    except ValueError as error:
        raise FullLayoutShadowError(f"{description} escapes its directory") from error
    actual = _identity(path, description=description)
    if contract.get("sha256") != actual["sha256"] or contract.get("size_bytes") != actual["size_bytes"]:
        raise FullLayoutShadowError(f"{description} identity differs from summary")
    return path, actual


def _rows_by_source(rows: Sequence[Mapping[str, Any]], *, kind: str, formal_module, description: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("schema_version") != 1 or row.get("kind") != kind:
            raise FullLayoutShadowError(f"{description}[{index}] schema/kind is unsupported")
        source = _nonempty(row.get("source"), description=f"{description}[{index}].source")
        key = _source_key(formal_module, source)
        if key in output:
            raise FullLayoutShadowError(f"duplicate {description} source: {source}")
        output[key] = row
    return output


def _load_derived_closure(
    plan_directory: Path,
    evaluation_directory: Path,
    truth_probe_directory: Path,
    *,
    formal_module,
) -> dict[str, Any]:
    plan_directory = _require_directory(plan_directory, description="derived-crop plan")
    evaluation_directory = _require_directory(evaluation_directory, description="derived-crop evaluation")
    truth_probe_directory = _require_directory(truth_probe_directory, description="truth probe")
    derived_script = _script_path("receipt-mlnet-recipient-derived-crop-shadow.py")
    derived_module = _load_module(derived_script, "recipient_derived_crop_contract_for_full_layout")

    # Strictly parse first so duplicate keys cannot be hidden by the reused validator.
    strict_plan_summary, strict_plan_summary_bytes = _load_json(
        plan_directory / "summary.json", description="derived plan summary"
    )
    plan_artifacts = strict_plan_summary.get("artifacts")
    if not isinstance(plan_artifacts, Mapping) or not isinstance(plan_artifacts.get("plans"), Mapping):
        raise FullLayoutShadowError("derived plan summary has no plans artifact")
    plan_path, plan_records_identity = _contained_artifact(
        plan_directory, plan_artifacts["plans"], description="derived plans"
    )
    strict_plans, _ = _load_jsonl(plan_path, description="derived plans")
    plan_summary, plans, reused_plan_summary_identity = derived_module._load_plan(plan_directory)
    if strict_plan_summary != dict(plan_summary) or strict_plans != [dict(row) for row in plans]:
        raise FullLayoutShadowError("strict and reused derived-plan projections differ")
    plan_summary_identity = _identity(
        plan_directory / "summary.json", strict_plan_summary_bytes, description="derived plan summary"
    )
    if plan_summary_identity != reused_plan_summary_identity:
        raise FullLayoutShadowError("derived plan summary identity differs from validator")
    if (
        plan_summary.get("kind") != PLAN_SUMMARY_KIND
        or plan_summary.get("records") != DERIVED_PLAN_RECORDS
        or len(plans) != DERIVED_PLAN_RECORDS
    ):
        raise FullLayoutShadowError("derived plan is not the frozen 63-record contract")
    filter_contract = plan_summary.get("filter_contract")
    if not isinstance(filter_contract, Mapping):
        raise FullLayoutShadowError("derived plan has no frozen truth-probe filter contract")
    filter_identity = _identity(
        Path(_nonempty(filter_contract.get("path"), description="truth-probe filter path")),
        description="truth-probe filter script",
    )
    _same_identity(filter_contract, filter_identity, description="truth-probe filter script")

    truth_summary, truth_summary_bytes = _load_json(
        truth_probe_directory / "summary.json", description="truth-probe summary"
    )
    truth_rows, truth_rows_bytes = _load_jsonl(
        truth_probe_directory / "findings.jsonl", description="truth-probe findings"
    )
    if truth_summary.get("schema_version") != 1 or truth_summary.get("kind") != TRUTH_SUMMARY_KIND:
        raise FullLayoutShadowError("truth-probe summary schema/kind is unsupported")
    truth_by_source = _rows_by_source(
        truth_rows, kind=TRUTH_RECORD_KIND, formal_module=formal_module, description="truth-probe findings"
    )
    if len(truth_by_source) != FORMAL_RECIPIENT_MISSING:
        raise FullLayoutShadowError("truth probe must contain exactly 204 formal recipient failures")
    truth_summary_identity = _identity(
        truth_probe_directory / "summary.json", truth_summary_bytes, description="truth-probe summary"
    )
    truth_rows_identity = _identity(
        truth_probe_directory / "findings.jsonl", truth_rows_bytes, description="truth-probe findings"
    )
    plan_sources = plan_summary.get("source_evidence")
    if not isinstance(plan_sources, Mapping):
        raise FullLayoutShadowError("derived plan has no frozen source evidence")
    _same_identity(plan_sources.get("truth_probe_summary"), truth_summary_identity, description="truth-probe summary")
    _same_identity(plan_sources.get("truth_probe_findings"), truth_rows_identity, description="truth-probe findings")

    evaluation_summary, evaluation_summary_bytes = _load_json(
        evaluation_directory / "summary.json", description="derived evaluation summary"
    )
    if (
        evaluation_summary.get("schema_version") != 1
        or evaluation_summary.get("kind") != DERIVED_SUMMARY_KIND
        or evaluation_summary.get("diagnostic_only") is not True
        or evaluation_summary.get("formal_delivery_gate") is not False
        or evaluation_summary.get("candidate_write_enabled") is not False
        or evaluation_summary.get("production_output_changed") is not False
        or evaluation_summary.get("accuracy_claimed") is not False
        or evaluation_summary.get("records") != DERIVED_PLAN_RECORDS
        or evaluation_summary.get("shadow_candidate_records") != DERIVED_SHADOW_CANDIDATES
        or evaluation_summary.get("unresolved_records") != TARGET_RECORDS
    ):
        raise FullLayoutShadowError("derived evaluation is not the completed 2/61 shadow contract")
    _same_identity(
        evaluation_summary.get("input_plan_summary"),
        plan_summary_identity,
        description="derived evaluation input plan",
    )
    evaluation_artifacts = evaluation_summary.get("artifacts")
    if not isinstance(evaluation_artifacts, Mapping) or not isinstance(evaluation_artifacts.get("findings"), Mapping):
        raise FullLayoutShadowError("derived evaluation has no findings artifact")
    evaluation_findings_path, evaluation_findings_identity = _contained_artifact(
        evaluation_directory, evaluation_artifacts["findings"], description="derived evaluation findings"
    )
    evaluation_rows, _ = _load_jsonl(
        evaluation_findings_path, description="derived evaluation findings"
    )
    if len(evaluation_rows) != DERIVED_PLAN_RECORDS:
        raise FullLayoutShadowError("derived evaluation findings must contain exactly 63 rows")
    plan_by_source = _rows_by_source(
        plans, kind=PLAN_RECORD_KIND, formal_module=formal_module, description="derived plans"
    )
    evaluation_by_source = _rows_by_source(
        evaluation_rows,
        kind=DERIVED_RECORD_KIND,
        formal_module=formal_module,
        description="derived evaluation findings",
    )
    if set(plan_by_source) != set(evaluation_by_source):
        raise FullLayoutShadowError("derived evaluation source set differs from plan")

    targets: list[dict[str, Any]] = []
    observed_states: Counter[str] = Counter()
    for plan in plans:
        key = _source_key(formal_module, plan["source"])
        finding = evaluation_by_source[key]
        state = finding.get("state")
        if state not in {"shadow_candidate", "unresolved"}:
            raise FullLayoutShadowError(f"derived evaluation has unsupported state for {plan['source']}")
        observed_states[state] += 1
        if finding.get("plan_id") != plan.get("plan_id"):
            raise FullLayoutShadowError("derived evaluation plan_id differs from plan")
        candidate = finding.get("shadow_candidate")
        if (state == "shadow_candidate") != (isinstance(candidate, str) and bool(candidate)):
            raise FullLayoutShadowError("derived evaluation state/candidate projection differs")
        truth = truth_by_source.get(key)
        if truth is None or truth.get("remaining_failure_cluster") is None:
            raise FullLayoutShadowError("derived plan source is not a frozen truth-probe remaining failure")
        shadow = truth.get("shadow_candidate_truth_free")
        if not isinstance(shadow, Mapping) or shadow.get("global_gate_failures") != []:
            raise FullLayoutShadowError("derived plan source does not preserve clear global gates")
        if state == "unresolved":
            targets.append(
                {
                    "source": plan["source"],
                    "source_image": dict(plan["source_image"]),
                    "plan_id": plan["plan_id"],
                }
            )
    if observed_states != Counter({"shadow_candidate": DERIVED_SHADOW_CANDIDATES, "unresolved": TARGET_RECORDS}):
        raise FullLayoutShadowError("derived evaluation findings do not reproduce candidate=2/unresolved=61")

    identities = {
        "derived_contract_script": _identity(derived_script, description="derived contract script"),
        "derived_plan_summary": plan_summary_identity,
        "derived_plan_records": plan_records_identity,
        "derived_evaluation_summary": _identity(
            evaluation_directory / "summary.json",
            evaluation_summary_bytes,
            description="derived evaluation summary",
        ),
        "derived_evaluation_findings": evaluation_findings_identity,
        "truth_probe_summary": truth_summary_identity,
        "truth_probe_findings": truth_rows_identity,
        "truth_probe_filter_script": filter_identity,
    }
    _assert_identities_current(identities)
    return {"targets": targets, "identities": identities}


def _load_formal_closure(formal_audit_directory: Path, *, formal_module) -> dict[str, Any]:
    formal_audit_directory = _require_directory(formal_audit_directory, description="formal audit")
    summary_path = formal_audit_directory / "summary.json"
    findings_path = formal_audit_directory / "findings.jsonl"
    summary, summary_bytes = _load_json(summary_path, description="formal audit summary")
    findings, findings_bytes = _load_jsonl(findings_path, description="formal audit findings")
    if (
        summary.get("schema_version") != 1
        or summary.get("kind") != FORMAL_SUMMARY_KIND
        or summary.get("read_only_existing_results") is not True
        or summary.get("ocr_rerun") is not False
        or summary.get("formal_required") is not True
        or summary.get("records") != FORMAL_RECORDS
    ):
        raise FullLayoutShadowError("formal audit is not the frozen formal 10016 contract")
    recipient_missing = summary.get("missing_by_field", {}).get("recipient_field")
    if not isinstance(recipient_missing, Mapping) or recipient_missing.get("records") != FORMAL_RECIPIENT_MISSING:
        raise FullLayoutShadowError("formal audit recipient missing count must be 204")

    # Re-run the source audit and compare the complete payload.  This makes the
    # supplied audit directory a verifiable closure rather than a trusted cache.
    try:
        rebuilt_summary, rebuilt_findings, audit_bindings = formal_module.audit(
            root=Path(_nonempty(summary.get("ab_root"), description="formal audit ab_root")),
            score=Path(_nonempty(summary.get("score_directory"), description="formal audit score_directory")),
            require_formal=True,
            expected_missing=None,
        )
    except (OSError, ValueError) as error:
        raise FullLayoutShadowError(f"cannot reproduce formal audit closure: {error}") from error
    if summary != rebuilt_summary or findings != rebuilt_findings:
        raise FullLayoutShadowError("formal audit output differs from a fresh closure reconstruction")

    score_directory = Path(str(summary["score_directory"])).resolve(strict=True)
    score_summary = formal_module._load_json(
        score_directory / "summary.json", description="formal score summary"
    )
    input_path = formal_module._resolve_bound_path(
        score_summary["input_selection"]["path"],
        base=score_directory,
        description="formal input list",
    )
    input_sources, input_by_key = formal_module._input_sources(input_path)
    if len(input_sources) != FORMAL_RECORDS:
        raise FullLayoutShadowError("formal input list must contain exactly 10016 records")
    records_path = formal_module._resolve_bound_path(
        score_summary["records"], base=score_directory, description="records manifest"
    )
    references = formal_module._record_references(
        records_path,
        selected_order=[formal_module._source_key(source) for source in input_sources],
        split="val",
    )
    root = Path(str(summary["ab_root"])).resolve(strict=True)
    hybrid, hybrid_ids, hybrid_closure = formal_module._manifest_results(
        root / "hybrid-recipient", label="hybrid", require_written=True
    )
    if hybrid_closure != summary.get("result_closures", {}).get("hybrid"):
        raise FullLayoutShadowError("hybrid result closure differs from formal audit")
    if set(hybrid) != set(input_by_key):
        raise FullLayoutShadowError("hybrid result source set differs from formal input list")

    missing_keys = {
        formal_module._source_key(source)
        for source in recipient_missing.get("sources", [])
        if isinstance(source, str)
    }
    if len(missing_keys) != FORMAL_RECIPIENT_MISSING:
        raise FullLayoutShadowError("formal recipient missing source set is not exactly 204")
    finding_missing_keys: set[str] = set()
    for index, finding in enumerate(findings):
        if finding.get("schema_version") != 1 or finding.get("kind") != FORMAL_RECORD_KIND:
            raise FullLayoutShadowError(f"formal audit finding[{index}] schema/kind is unsupported")
        if "recipient_field" in finding.get("missing_fields", []):
            finding_missing_keys.add(formal_module._source_key(finding.get("source")))
    if finding_missing_keys != missing_keys:
        raise FullLayoutShadowError("formal recipient missing summary/findings sets differ")

    identities = {
        "formal_audit_script": _identity(
            _script_path("receipt-mlnet-formal-missing-fields-audit.py"),
            description="formal audit script",
        ),
        "formal_audit_summary": _identity(summary_path, summary_bytes, description="formal audit summary"),
        "formal_audit_findings": _identity(findings_path, findings_bytes, description="formal audit findings"),
    }
    _assert_identities_current(identities)
    return {
        "summary": summary,
        "input_sources": input_sources,
        "input_by_key": input_by_key,
        "references": references,
        "hybrid": hybrid,
        "hybrid_ids": hybrid_ids,
        "missing_recipient_keys": missing_keys,
        "identities": identities,
        "audit_bindings": audit_bindings,
        "formal_module": formal_module,
    }


def _allocate_strata(counts: Mapping[str, int], total: int) -> dict[str, int]:
    if total < 0 or any(type(value) is not int or value < 0 for value in counts.values()):
        raise FullLayoutShadowError("control stratum counts/total must be non-negative integers")
    active = [name for name in CONTROL_STRATA if counts.get(name, 0) > 0]
    population = sum(counts.get(name, 0) for name in active)
    if total > population:
        raise FullLayoutShadowError(f"only {population} controls are eligible, need {total}")
    if total < len(active):
        raise FullLayoutShadowError("control total cannot cover every non-empty stratum")
    allocation = {name: (1 if name in active else 0) for name in CONTROL_STRATA}
    remaining = total - len(active)
    capacities = {name: counts.get(name, 0) - allocation[name] for name in CONTROL_STRATA}
    capacity_total = sum(capacities.values())
    if remaining and capacity_total <= 0:
        raise FullLayoutShadowError("control allocation has no remaining capacity")
    raw = {
        name: (remaining * capacities[name] / capacity_total if capacity_total else 0.0)
        for name in CONTROL_STRATA
    }
    for name in CONTROL_STRATA:
        addition = min(capacities[name], math.floor(raw[name]))
        allocation[name] += addition
    while sum(allocation.values()) < total:
        eligible = [name for name in CONTROL_STRATA if allocation[name] < counts.get(name, 0)]
        if not eligible:
            raise FullLayoutShadowError("control largest-remainder allocation exhausted capacity")
        name = max(
            eligible,
            key=lambda item: (
                raw[item] - math.floor(raw[item]),
                counts.get(item, 0) - allocation[item],
                -CONTROL_STRATA.index(item),
            ),
        )
        allocation[name] += 1
        raw[name] = math.floor(raw[name])
    return allocation


def _evenly_spread(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 0 or count > len(rows):
        raise FullLayoutShadowError("invalid evenly-spread selection count")
    if count == 0:
        return []
    positions = [math.floor((2 * index + 1) * len(rows) / (2 * count)) for index in range(count)]
    if len(set(positions)) != count:
        raise FullLayoutShadowError("evenly-spread control positions are not unique")
    return [rows[position] for position in positions]


def _source_closure(identities: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(
            f"{identity['path']}\0{identity['sha256']}\0{identity['size_bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _assert_formal_bindings_current(formal: Mapping[str, Any]) -> None:
    module = formal["formal_module"]
    bindings = formal["audit_bindings"]
    module._assert_source_evidence_current(
        bindings["source_evidence"],
        bindings["baseline_result_ids"],
        bindings["hybrid_result_ids"],
    )
    for identity in formal["hybrid_ids"].values():
        actual = module._file_identity(Path(str(identity["path"])), description="hybrid result")
        if actual != identity:
            raise FullLayoutShadowError("hybrid result changed during selection")


def _write_atomic_directory(output: Path, files: Mapping[str, bytes], *, closing_check) -> None:
    output = output.resolve()
    if os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite recipient full-layout shadow output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    try:
        for relative, payload in files.items():
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        closing_check()
        if os.path.lexists(os.fspath(output)):
            raise FileExistsError(f"refusing to overwrite recipient full-layout shadow output: {output}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare(
    *,
    plan_directory: Path,
    derived_evaluation_directory: Path,
    formal_audit_directory: Path,
    truth_probe_directory: Path,
    output_directory: Path,
) -> None:
    formal_script = _script_path("receipt-mlnet-formal-missing-fields-audit.py")
    formal_module = _load_module(formal_script, "formal_audit_contract_for_recipient_full_layout")
    derived = _load_derived_closure(
        plan_directory,
        derived_evaluation_directory,
        truth_probe_directory,
        formal_module=formal_module,
    )
    formal = _load_formal_closure(formal_audit_directory, formal_module=formal_module)
    target_by_key = {
        _source_key(formal_module, target["source"]): target for target in derived["targets"]
    }
    if len(target_by_key) != TARGET_RECORDS:
        raise FullLayoutShadowError("derived unresolved target set must contain exactly 61 unique sources")
    if not set(target_by_key).issubset(formal["missing_recipient_keys"]):
        raise FullLayoutShadowError("derived unresolved targets are not all formal recipient-missing records")

    canonical_index = {
        _source_key(formal_module, source): index
        for index, source in enumerate(formal["input_sources"])
    }
    controls_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in formal["input_sources"]:
        key = _source_key(formal_module, source)
        if key in target_by_key:
            continue
        reference = formal["references"][key].get("recipient_field")
        candidate = formal_module._candidate(formal["hybrid"][key], "recipient")
        if not isinstance(reference, str) or not reference or candidate is None:
            continue
        stratum = "existing_exact" if candidate == reference else "existing_wrong"
        controls_by_stratum[stratum].append(
            {
                "source": source,
                "existing_recipient": candidate,
                "external_reference": reference,
                "control_stratum": stratum,
            }
        )
    eligible_counts = {name: len(controls_by_stratum[name]) for name in CONTROL_STRATA}
    allocation = _allocate_strata(eligible_counts, CONTROL_RECORDS)
    selected_controls = [
        row
        for name in CONTROL_STRATA
        for row in _evenly_spread(controls_by_stratum[name], allocation[name])
    ]
    selected_controls.sort(key=lambda row: canonical_index[_source_key(formal_module, row["source"])])

    selected_by_key: dict[str, dict[str, Any]] = {}
    for key, target in target_by_key.items():
        source = formal["input_by_key"].get(key)
        if source is None:
            raise FullLayoutShadowError("target is outside canonical formal input list")
        source_identity = _identity(Path(source), description="target source image")
        if source_identity != target["source_image"]:
            raise FullLayoutShadowError("target source identity differs between plan and formal input")
        selected_by_key[key] = {
            "cohort": "target_unresolved",
            "source": source,
            "source_image": source_identity,
            "target_evidence": {
                "derived_plan_id": target["plan_id"],
                "derived_state": "unresolved",
            },
        }
    for control in selected_controls:
        key = _source_key(formal_module, control["source"])
        if key in selected_by_key:
            raise FullLayoutShadowError("control overlaps unresolved target")
        selected_by_key[key] = {
            "cohort": "reference_control",
            "source": control["source"],
            "source_image": _identity(Path(control["source"]), description="control source image"),
            "control_evidence": {
                "stratum": control["control_stratum"],
                "existing_recipient": control["existing_recipient"],
                "external_reference": control["external_reference"],
            },
        }
    if len(selected_by_key) != EXPECTED_RECORDS:
        raise FullLayoutShadowError("recipient full-layout selection must contain exactly 339 records")

    ordered = [
        selected_by_key[key]
        for key in sorted(selected_by_key, key=lambda item: canonical_index[item])
    ]
    selection_rows = [
        {
            "schema_version": 1,
            "kind": SELECTION_RECORD_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "index": index,
            **row,
        }
        for index, row in enumerate(ordered)
    ]
    if sum(row["cohort"] == "target_unresolved" for row in selection_rows) != TARGET_RECORDS:
        raise FullLayoutShadowError("selection target count differs from 61")
    if any("control_evidence" in row for row in selection_rows if row["cohort"] == "target_unresolved"):
        raise FullLayoutShadowError("target rows must not contain control truth")

    selection_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in selection_rows
    )
    inputs_bytes = "".join(f"{row['source']}\n" for row in selection_rows).encode("utf-8")
    if inputs_bytes.startswith(b"\xef\xbb\xbf") or not inputs_bytes.endswith(b"\n"):
        raise FullLayoutShadowError("input-list encoding contract is invalid")
    source_identities = [row["source_image"] for row in selection_rows]
    contract_identities = {**derived["identities"], **formal["identities"]}
    summary = {
        "schema_version": 1,
        "kind": SELECTION_SUMMARY_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "production_output_changed": False,
        "ocr_rerun": False,
        "records": EXPECTED_RECORDS,
        "selection_order": "canonical_formal_first_unique_source_order_after_stratified_selection",
        "cohorts": {
            "target_unresolved": {
                "records": TARGET_RECORDS,
                "selection": "exact derived-crop evaluation state=unresolved closure",
                "external_truth_used_for_selection": False,
            },
            "reference_control": {
                "records": CONTROL_RECORDS,
                "requirements": ["existing_recipient_candidate", "external_recipient_reference"],
                "stratification": "existing candidate exactness against external reference",
                "allocation": "proportional largest remainder with one per non-empty stratum",
                "within_stratum_selection": "deterministic evenly-spread canonical formal order",
                "eligible_by_stratum": eligible_counts,
                "selected_by_stratum": allocation,
            },
        },
        "required_layout": {
            "kind": LAYOUT_SUMMARY_KIND,
            "record_kind": LAYOUT_RECORD_KIND,
            "execution_provider": "cpu",
            "rectification": "max-side-1600",
            "expected_records": EXPECTED_RECORDS,
            "fresh_atomic_output_required": True,
        },
        "recipient_shadow_contract": {
            "labels": list(LABELS),
            "same_line_rhs_requires_colon_or_whitespace_boundary": True,
            "same_row_right_neighbor_requires_unique_eligible_value": True,
            "minimum_line_confidence": MINIMUM_LINE_CONFIDENCE,
            "requires_paddle_drop_score_pass": True,
            "uses_frozen_truth_probe_negative_amount_time_character_filter": True,
            "target_truth_reporting": False,
            "candidate_write_enabled": False,
        },
        "contract_evidence": contract_identities,
        "artifacts": {
            "selection": {
                "path": "selection.jsonl",
                "sha256": _sha256(selection_bytes),
                "size_bytes": len(selection_bytes),
                "records": EXPECTED_RECORDS,
            },
            "inputs": {
                "path": "inputs.txt",
                "sha256": _sha256(inputs_bytes),
                "size_bytes": len(inputs_bytes),
                "records": EXPECTED_RECORDS,
                "encoding": "utf-8-no-bom",
                "terminal_newline": True,
            },
        },
        # LayoutShadow's existing generic validator expects this direct alias.
        "input_list": {
            "relative_path": "inputs.txt",
            "sha256": _sha256(inputs_bytes),
            "size_bytes": len(inputs_bytes),
            "records": EXPECTED_RECORDS,
        },
        "source_files": source_identities,
        "source_closure_sha256": _source_closure(source_identities),
        "source_total_bytes": sum(int(item["size_bytes"]) for item in source_identities),
    }
    summary_bytes = (json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")

    def closing_check() -> None:
        _assert_identities_current(contract_identities)
        _assert_formal_bindings_current(formal)
        _assert_identities_current(
            {f"selected_source_{index:03d}": identity for index, identity in enumerate(source_identities)}
        )

    _write_atomic_directory(
        output_directory,
        {
            "summary.json": summary_bytes,
            "selection.jsonl": selection_bytes,
            "inputs.txt": inputs_bytes,
        },
        closing_check=closing_check,
    )


def _load_selection(selection_directory: Path, *, formal_module) -> dict[str, Any]:
    selection_directory = _require_directory(selection_directory, description="full-layout selection")
    summary, summary_bytes = _load_json(selection_directory / "summary.json", description="selection summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("kind") != SELECTION_SUMMARY_KIND
        or summary.get("diagnostic_only") is not True
        or summary.get("formal_delivery_gate") is not False
        or summary.get("candidate_write_enabled") is not False
        or summary.get("production_output_changed") is not False
        or summary.get("records") != EXPECTED_RECORDS
    ):
        raise FullLayoutShadowError("selection summary violates the diagnostic-only 339 contract")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FullLayoutShadowError("selection summary has no artifacts")
    selection_path, selection_identity = _contained_artifact(
        selection_directory, artifacts.get("selection", {}), description="selection records"
    )
    input_path, input_identity = _contained_artifact(
        selection_directory, artifacts.get("inputs", {}), description="selection inputs"
    )
    rows, selection_bytes = _load_jsonl(selection_path, description="selection records")
    input_bytes = _read_bytes(input_path, description="selection inputs")
    if len(rows) != EXPECTED_RECORDS:
        raise FullLayoutShadowError("selection records must contain exactly 339 rows")
    if input_bytes.startswith(b"\xef\xbb\xbf"):
        raise FullLayoutShadowError("selection inputs must not contain a BOM")
    try:
        input_text = input_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FullLayoutShadowError("selection inputs are not strict UTF-8") from error
    if not input_text.endswith("\n"):
        raise FullLayoutShadowError("selection inputs must have a terminal newline")
    input_sources = input_text.splitlines()
    if len(input_sources) != EXPECTED_RECORDS or any(not source or source != source.strip() for source in input_sources):
        raise FullLayoutShadowError("selection inputs must contain 339 clean absolute paths")

    sources: list[Path] = []
    source_identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    cohort_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    for index, (row, input_source) in enumerate(zip(rows, input_sources, strict=True)):
        if (
            row.get("schema_version") != 1
            or row.get("kind") != SELECTION_RECORD_KIND
            or row.get("diagnostic_only") is not True
            or row.get("formal_delivery_gate") is not False
            or row.get("candidate_write_enabled") is not False
            or row.get("index") != index
            or row.get("source") != input_source
        ):
            raise FullLayoutShadowError(f"selection record[{index}] contract/order differs")
        source = Path(input_source)
        if not source.is_absolute():
            raise FullLayoutShadowError(f"selection source[{index}] is not absolute")
        source_identity = _identity(source, description=f"selection source[{index}]")
        _same_identity(row.get("source_image"), source_identity, description=f"selection source[{index}]")
        key = _source_key(formal_module, input_source)
        if key in seen:
            raise FullLayoutShadowError(f"duplicate selection source: {input_source}")
        seen.add(key)
        cohort = row.get("cohort")
        if cohort not in {"target_unresolved", "reference_control"}:
            raise FullLayoutShadowError(f"selection record[{index}] has unsupported cohort")
        cohort_counts[cohort] += 1
        if cohort == "target_unresolved":
            if "control_evidence" in row or not isinstance(row.get("target_evidence"), Mapping):
                raise FullLayoutShadowError("target selection row leaks or lacks cohort evidence")
        else:
            control = row.get("control_evidence")
            if not isinstance(control, Mapping):
                raise FullLayoutShadowError("control selection row lacks control evidence")
            stratum = control.get("stratum")
            candidate = control.get("existing_recipient")
            reference = control.get("external_reference")
            if stratum not in CONTROL_STRATA or not isinstance(candidate, str) or not candidate or not isinstance(reference, str) or not reference:
                raise FullLayoutShadowError("control selection evidence is invalid")
            if (stratum == "existing_exact") != (candidate == reference):
                raise FullLayoutShadowError("control stratum disagrees with candidate/reference exactness")
            stratum_counts[stratum] += 1
        sources.append(source.resolve(strict=True))
        source_identities.append(source_identity)
    if cohort_counts != Counter({"target_unresolved": TARGET_RECORDS, "reference_control": CONTROL_RECORDS}):
        raise FullLayoutShadowError("selection cohorts must equal target=61/control=278")
    cohorts = summary.get("cohorts")
    control_cohort = cohorts.get("reference_control") if isinstance(cohorts, Mapping) else None
    expected_selected = (
        control_cohort.get("selected_by_stratum")
        if isinstance(control_cohort, Mapping)
        else None
    )
    if not isinstance(expected_selected, Mapping):
        raise FullLayoutShadowError("selection summary has no control stratum counts")
    if dict(stratum_counts) != {name: expected_selected.get(name, 0) for name in CONTROL_STRATA}:
        raise FullLayoutShadowError("selection control stratum counts differ from summary")
    if summary.get("source_files") != source_identities:
        raise FullLayoutShadowError("selection source_files differ from records")
    if summary.get("source_closure_sha256") != _source_closure(source_identities):
        raise FullLayoutShadowError("selection source closure differs")
    if summary.get("source_total_bytes") != sum(item["size_bytes"] for item in source_identities):
        raise FullLayoutShadowError("selection source byte total differs")
    input_alias = summary.get("input_list")
    if not isinstance(input_alias, Mapping) or input_alias.get("sha256") != input_identity["sha256"] or input_alias.get("size_bytes") != input_identity["size_bytes"]:
        raise FullLayoutShadowError("selection input_list alias differs from frozen inputs")
    contracts = summary.get("contract_evidence")
    if not isinstance(contracts, Mapping):
        raise FullLayoutShadowError("selection has no contract evidence")
    _assert_identities_current(contracts)
    return {
        "directory": selection_directory,
        "summary": summary,
        "summary_identity": _identity(
            selection_directory / "summary.json", summary_bytes, description="selection summary"
        ),
        "rows": rows,
        "sources": sources,
        "source_identities": source_identities,
        "selection_identity": selection_identity,
        "input_identity": input_identity,
        "contracts": dict(contracts),
    }


def _rect_from_quad(value: object, *, description: str) -> dict[str, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise FullLayoutShadowError(f"{description} must contain four points")
    xs: list[float] = []
    ys: list[float] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise FullLayoutShadowError(f"{description} point must contain x,y")
        xs.append(_require_number(point[0], description=description, minimum=0, maximum=1))
        ys.append(_require_number(point[1], description=description, minimum=0, maximum=1))
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "x_center": (min(xs) + max(xs)) / 2,
        "y_center": (min(ys) + max(ys)) / 2,
        "height": max(ys) - min(ys),
    }


def _same_row_right(label: Mapping[str, float], candidate: Mapping[str, float]) -> bool:
    overlap = min(label["y_max"], candidate["y_max"]) - max(label["y_min"], candidate["y_min"])
    minimum_height = min(label["height"], candidate["height"])
    return (
        minimum_height > 0
        and overlap / minimum_height >= 0.50
        and candidate["x_center"] > label["x_center"]
        and candidate["x_min"] >= label["x_max"] - 0.015
    )


def _strict_candidate(value: object, filter_module) -> tuple[str | None, str]:
    cleaned = _clean(value)
    allowed, reason = filter_module._shadow_line_allowed(cleaned)
    return (cleaned, "accepted") if allowed else (None, reason)


def _recipient_shadow(lines: Sequence[Mapping[str, Any]], *, drop_score: float, filter_module) -> dict[str, Any]:
    prepared: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not isinstance(line, Mapping) or line.get("index") != index:
            raise FullLayoutShadowError("layout lines must have contiguous indices")
        confidence = _require_number(
            line.get("confidence"), description="layout line confidence", minimum=0, maximum=1
        )
        passes = line.get("passes_drop_score")
        if type(passes) is not bool or passes != (confidence >= drop_score):
            raise FullLayoutShadowError("layout line drop-score projection differs")
        prepared.append(
            {
                "index": index,
                "text": _clean(line.get("text")),
                "confidence": confidence,
                "passes": passes,
                "rect": _rect_from_quad(
                    line.get("quad_rectified_normalized"), description="layout normalized quad"
                ),
            }
        )

    evidence: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    ambiguous_anchor_indices: list[int] = []
    label_anchor_indices: list[int] = []
    for label_line in prepared:
        if not label_line["passes"] or label_line["confidence"] < MINIMUM_LINE_CONFIDENCE:
            continue
        rhs_match = LABEL_RHS.fullmatch(label_line["text"])
        if rhs_match is not None:
            label_anchor_indices.append(label_line["index"])
            candidate, reason = _strict_candidate(rhs_match.group("value"), filter_module)
            if candidate is None:
                rejected[reason] += 1
            else:
                evidence.append(
                    {
                        "route": "same_line_rhs",
                        "label": rhs_match.group("label"),
                        "label_line_index": label_line["index"],
                        "candidate_line_index": label_line["index"],
                        "candidate": candidate,
                        "minimum_confidence": label_line["confidence"],
                    }
                )
            continue
        label_match = LABEL_ONLY.fullmatch(label_line["text"])
        if label_match is None:
            continue
        label_anchor_indices.append(label_line["index"])
        neighbor_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate_line in prepared:
            if candidate_line["index"] == label_line["index"]:
                continue
            if (
                not candidate_line["passes"]
                or candidate_line["confidence"] < MINIMUM_LINE_CONFIDENCE
                or not _same_row_right(label_line["rect"], candidate_line["rect"])
            ):
                continue
            candidate, reason = _strict_candidate(candidate_line["text"], filter_module)
            if candidate is None:
                rejected[reason] += 1
                continue
            neighbor_values[candidate].append(candidate_line)
        if len(neighbor_values) > 1:
            ambiguous_anchor_indices.append(label_line["index"])
            continue
        if len(neighbor_values) == 1:
            candidate, candidate_lines = next(iter(neighbor_values.items()))
            best = max(candidate_lines, key=lambda row: (row["confidence"], -row["index"]))
            evidence.append(
                {
                    "route": "same_row_unique_right_neighbor",
                    "label": label_match.group("label"),
                    "label_line_index": label_line["index"],
                    "candidate_line_index": best["index"],
                    "candidate": candidate,
                    "minimum_confidence": min(label_line["confidence"], best["confidence"]),
                }
            )

    distinct = sorted({item["candidate"] for item in evidence})
    if ambiguous_anchor_indices or len(distinct) > 1:
        state = "ambiguous"
        candidate = None
        route = None
        confidence = None
    elif len(distinct) == 1:
        state = "shadow_candidate"
        candidate = distinct[0]
        accepted = [item for item in evidence if item["candidate"] == candidate]
        route_names = {item["route"] for item in accepted}
        if route_names == {"same_line_rhs"}:
            route = "full_layout_label_rhs_shadow"
        elif route_names == {"same_row_unique_right_neighbor"}:
            route = "full_layout_label_right_neighbor_shadow"
        else:
            route = "full_layout_label_rhs_right_neighbor_consensus_shadow"
        confidence = min(item["minimum_confidence"] for item in accepted)
    else:
        state = "unresolved"
        candidate = None
        route = None
        confidence = None
    return {
        "state": state,
        "shadow_candidate": candidate,
        "shadow_route": route,
        "minimum_confidence": confidence,
        "label_anchor_indices": sorted(set(label_anchor_indices)),
        "ambiguous_anchor_indices": sorted(set(ambiguous_anchor_indices)),
        "distinct_eligible_values": distinct,
        "evidence": evidence,
        "rejected_value_reasons": dict(sorted(rejected.items())),
    }


def _validate_layout(
    *, selection: Mapping[str, Any], layout_directory: Path, layout_module
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layout_directory = _require_directory(layout_directory, description="LayoutShadow output")
    missing_sets = {field: set() for field in layout_module.FIELD_SPECS}
    try:
        layout_module._validate_layout(
            layout_directory,
            selection["directory"],
            selection["summary"],
            selection["sources"],
            selection["source_identities"],
            missing_sets,
        )
    except (OSError, ValueError) as error:
        raise FullLayoutShadowError(f"LayoutShadow 339 closure is invalid: {error}") from error
    summary, summary_bytes = _load_json(layout_directory / "summary.json", description="LayoutShadow summary")
    records, records_bytes = _load_jsonl(layout_directory / "records.jsonl", description="LayoutShadow records")
    if (
        summary.get("kind") != LAYOUT_SUMMARY_KIND
        or summary.get("expected_records") != EXPECTED_RECORDS
        or summary.get("records") != EXPECTED_RECORDS
        or summary.get("errors") != 0
        or summary.get("execution_provider") != "cpu"
        or len(records) != EXPECTED_RECORDS
    ):
        raise FullLayoutShadowError("LayoutShadow output is not the fresh CPU339 contract")
    return records, {
        "layout_summary": _identity(
            layout_directory / "summary.json", summary_bytes, description="LayoutShadow summary"
        ),
        "layout_records": _identity(
            layout_directory / "records.jsonl", records_bytes, description="LayoutShadow records"
        ),
        "paddle_bundle": summary.get("paddle_bundle"),
        "paddle_drop_score": _require_number(
            summary.get("paddle_drop_score"), description="Paddle drop score", minimum=0, maximum=1
        ),
        "latency_ms": summary.get("latency_ms"),
    }


def evaluate(
    *, selection_directory: Path, layout_directory: Path, output_directory: Path
) -> None:
    formal_module = _load_module(
        _script_path("receipt-mlnet-formal-missing-fields-audit.py"),
        "formal_audit_contract_for_recipient_full_layout_evaluate",
    )
    layout_script = _script_path("receipt-mlnet-layout-shadow-evidence.py")
    layout_module = _load_module(layout_script, "layout_contract_for_recipient_full_layout")
    filter_script = _script_path("receipt-mlnet-hybrid-failure-truth-probe.py")
    filter_module = _load_module(filter_script, "recipient_filter_for_full_layout")
    selection = _load_selection(selection_directory, formal_module=formal_module)
    expected_filter = selection["contracts"].get("truth_probe_filter_script")
    actual_filter = _identity(filter_script, description="truth-probe filter script")
    if expected_filter is not None:
        _same_identity(expected_filter, actual_filter, description="truth-probe filter script")
    records, layout_bindings = _validate_layout(
        selection=selection, layout_directory=layout_directory, layout_module=layout_module
    )
    drop_score = layout_bindings["paddle_drop_score"]

    findings: list[dict[str, Any]] = []
    for index, (selected, layout) in enumerate(zip(selection["rows"], records, strict=True)):
        shadow = _recipient_shadow(layout.get("lines", []), drop_score=drop_score, filter_module=filter_module)
        base = {
            "schema_version": 1,
            "kind": EVALUATION_RECORD_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "production_output_changed": False,
            "index": index,
            "source": selected["source"],
            "cohort": selected["cohort"],
            **shadow,
        }
        if selected["cohort"] == "target_unresolved":
            if any(key in base for key in ("external_reference", "correct", "false_positive")):
                raise FullLayoutShadowError("target finding contains truth-derived fields")
            base["target_evidence"] = dict(selected["target_evidence"])
        else:
            control = selected["control_evidence"]
            existing = control["existing_recipient"]
            reference = control["external_reference"]
            candidate = shadow["shadow_candidate"]
            base["control_evaluation"] = {
                "stratum": control["stratum"],
                "existing_recipient": existing,
                "external_reference": reference,
                "existing_exact": existing == reference,
                "shadow_emitted": candidate is not None,
                "shadow_exact": candidate == reference if candidate is not None else False,
                "false_positive": candidate is not None and candidate != reference,
                "correct_to_wrong": existing == reference and candidate is not None and candidate != reference,
                "wrong_to_correct": existing != reference and candidate == reference,
            }
        findings.append(base)

    target_rows = [row for row in findings if row["cohort"] == "target_unresolved"]
    control_rows = [row for row in findings if row["cohort"] == "reference_control"]
    target_states = Counter(row["state"] for row in target_rows)
    target_routes = Counter(row["shadow_route"] or "none" for row in target_rows)
    control_states = Counter(row["state"] for row in control_rows)
    control_routes = Counter(row["shadow_route"] or "none" for row in control_rows)
    control_metrics = Counter()
    for row in control_rows:
        control = row["control_evaluation"]
        for name in (
            "existing_exact",
            "shadow_emitted",
            "shadow_exact",
            "false_positive",
            "correct_to_wrong",
            "wrong_to_correct",
        ):
            control_metrics[name] += bool(control[name])
    findings_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in findings
    )
    evaluation_contracts = {
        **selection["contracts"],
        "selection_summary": selection["summary_identity"],
        "selection_records": selection["selection_identity"],
        "selection_inputs": selection["input_identity"],
        "layout_validator_script": _identity(layout_script, description="layout validator script"),
        "truth_probe_filter_script": actual_filter,
        "layout_summary": layout_bindings["layout_summary"],
        "layout_records": layout_bindings["layout_records"],
    }
    summary = {
        "schema_version": 1,
        "kind": EVALUATION_SUMMARY_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "production_output_changed": False,
        "accuracy_claimed_for_targets": False,
        "target_truth_used_for_candidate_selection": False,
        "records": EXPECTED_RECORDS,
        "targets": {
            "records": TARGET_RECORDS,
            "truth_reported": False,
            "shadow_candidate_records": target_states["shadow_candidate"],
            "ambiguous_records": target_states["ambiguous"],
            "unresolved_records": target_states["unresolved"],
            "by_state": dict(sorted(target_states.items())),
            "by_shadow_route": dict(sorted(target_routes.items())),
        },
        "controls": {
            "records": CONTROL_RECORDS,
            "existing_exact_records": control_metrics["existing_exact"],
            "existing_wrong_records": CONTROL_RECORDS - control_metrics["existing_exact"],
            "shadow_candidate_records": control_metrics["shadow_emitted"],
            "shadow_exact_records": control_metrics["shadow_exact"],
            "false_positive_records": control_metrics["false_positive"],
            "correct_to_wrong_records": control_metrics["correct_to_wrong"],
            "wrong_to_correct_records": control_metrics["wrong_to_correct"],
            "shadow_exact_rate_all_controls": control_metrics["shadow_exact"] / CONTROL_RECORDS,
            "shadow_exact_rate_when_emitted": (
                control_metrics["shadow_exact"] / control_metrics["shadow_emitted"]
                if control_metrics["shadow_emitted"]
                else None
            ),
            "false_positive_rate_all_controls": control_metrics["false_positive"] / CONTROL_RECORDS,
            "correct_to_wrong_rate_among_existing_exact": (
                control_metrics["correct_to_wrong"] / control_metrics["existing_exact"]
                if control_metrics["existing_exact"]
                else None
            ),
            "by_state": dict(sorted(control_states.items())),
            "by_shadow_route": dict(sorted(control_routes.items())),
        },
        "recipient_shadow_contract": selection["summary"]["recipient_shadow_contract"],
        "execution_provider": "cpu",
        "paddle_drop_score": drop_score,
        "paddle_bundle": layout_bindings["paddle_bundle"],
        "layout_latency_ms": layout_bindings["latency_ms"],
        "input_closure": evaluation_contracts,
        "artifacts": {
            "findings": {
                "path": "findings.jsonl",
                "sha256": _sha256(findings_bytes),
                "size_bytes": len(findings_bytes),
                "records": EXPECTED_RECORDS,
            }
        },
    }
    summary_bytes = (json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")

    def closing_check() -> None:
        _assert_identities_current(evaluation_contracts)
        _assert_identities_current(
            {
                f"selected_source_{index:03d}": identity
                for index, identity in enumerate(selection["source_identities"])
            }
        )

    _write_atomic_directory(
        output_directory,
        {"summary.json": summary_bytes, "findings.jsonl": findings_bytes},
        closing_check=closing_check,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="freeze 61 targets plus 278 controls")
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--derived-evaluation", type=Path, required=True)
    prepare_parser.add_argument("--formal-audit", type=Path, required=True)
    prepare_parser.add_argument("--truth-probe", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate", help="evaluate a fresh CPU LayoutShadow339 output")
    evaluate_parser.add_argument("--selection", type=Path, required=True)
    evaluate_parser.add_argument("--layout-evidence", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(
                plan_directory=args.plan,
                derived_evaluation_directory=args.derived_evaluation,
                formal_audit_directory=args.formal_audit,
                truth_probe_directory=args.truth_probe,
                output_directory=args.output,
            )
        else:
            evaluate(
                selection_directory=args.selection,
                layout_directory=args.layout_evidence,
                output_directory=args.output,
            )
    except (FullLayoutShadowError, FileExistsError, OSError, UnicodeError) as error:
        print(f"Recipient full-layout shadow failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
