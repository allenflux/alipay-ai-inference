#!/usr/bin/env python3
"""Compare the production C# PP-OCR adapter with the frozen ONNX wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from transfer_receipt_ai.ocr import clean_text, parse_anchored_recipient_row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: row must be an object")
        rows.append(value)
    return rows


def _key(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("Parity row has no source path")
    return os.path.normcase(os.path.abspath(path))


def _anchored(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    parsed = parse_anchored_recipient_row(clean_text(text))
    return clean_text(parsed[1]) if parsed is not None else None


def compare(*, wrapper: Path, dotnet: Path, delivery: Path, output: Path) -> tuple[dict[str, object], bool]:
    wrapper = wrapper.resolve()
    dotnet = dotnet.resolve()
    delivery = delivery.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite .NET parity comparison: {output}")
    wrapper_summary = _json(wrapper / "summary.json")
    dotnet_summary = _json(dotnet / "summary.json")
    delivery_contract = delivery / "paddle_ocr_delivery.contract.json"
    if not isinstance(wrapper_summary, dict) or wrapper_summary.get("accepted") is not True:
        raise ValueError("Frozen native/ONNX wrapper parity was not accepted")
    delivery_value = _json(delivery_contract)
    if not isinstance(delivery_value, dict):
        raise ValueError("Paddle delivery contract must be an object")
    if (
        not isinstance(dotnet_summary, dict)
        or dotnet_summary.get("kind") != "receipt_ppocr_dotnet_cpu_parity_v1"
        or dotnet_summary.get("execution_provider") != "cpu"
        or dotnet_summary.get("bundle_contract_sha256") != _sha256(delivery_contract)
    ):
        raise ValueError(".NET parity summary is not CPU/hash-bound to the delivery bundle")
    delivery_native_identity = delivery_value.get("native_asset_identity")
    if not isinstance(delivery_native_identity, dict):
        raise ValueError("Paddle delivery contract has no native asset identity")
    if (
        wrapper_summary.get("bundle_contract_sha256") != delivery_value.get("source_audit_contract_sha256")
        or wrapper_summary.get("native_asset_identity_sha256")
        != delivery_native_identity.get("sha256")
        or wrapper_summary.get("comparisons_sha256") != _sha256(wrapper / "comparisons.jsonl")
    ):
        raise ValueError("Wrapper parity is not hash-bound to the exact delivery audit/native identity")

    wrapper_rows = {_key(row.get("source")): row for row in _jsonl(wrapper / "comparisons.jsonl")}
    dotnet_rows = {_key(row.get("source")): row for row in _jsonl(dotnet / "records.jsonl")}
    if not wrapper_rows or wrapper_rows.keys() != dotnet_rows.keys():
        raise ValueError("Wrapper and .NET parity source sets differ")
    if (
        int(wrapper_summary.get("records", -1)) != len(wrapper_rows)
        or int(dotnet_summary.get("records", -1)) != len(dotnet_rows)
    ):
        raise ValueError("Wrapper/.NET parity summary counts differ from their records")

    comparisons = []
    for source in sorted(wrapper_rows):
        wrapper_row = wrapper_rows[source]
        dotnet_row = dotnet_rows[source]
        onnx = wrapper_row.get("onnx")
        if not isinstance(onnx, dict):
            raise ValueError(f"Wrapper parity row has no ONNX result: {source}")
        wrapper_text = onnx.get("text")
        dotnet_text = dotnet_row.get("raw_text")
        wrapper_lines = onnx.get("lines")
        dotnet_lines = dotnet_row.get("lines")
        wrapper_line_text = [line.get("text") for line in wrapper_lines] if isinstance(wrapper_lines, list) else []
        dotnet_line_text = [line.get("text") for line in dotnet_lines] if isinstance(dotnet_lines, list) else []
        wrapper_confidence = onnx.get("confidence")
        dotnet_confidence = dotnet_row.get("confidence")
        confidence_delta = (
            None
            if wrapper_confidence is None or dotnet_confidence is None
            else abs(float(wrapper_confidence) - float(dotnet_confidence))
        )
        expected_anchored = _anchored(wrapper_text)
        failures = []
        if wrapper_text != dotnet_text:
            failures.append("raw_text")
        if wrapper_line_text != dotnet_line_text:
            failures.append("line_text")
        if expected_anchored != dotnet_row.get("candidate_anchored_value"):
            failures.append("anchored_value")
        if confidence_delta is not None and (not math.isfinite(confidence_delta) or confidence_delta > 0.01):
            failures.append("confidence")
        comparisons.append(
            {
                "source": source,
                "wrapper_text": wrapper_text,
                "dotnet_text": dotnet_text,
                "anchored_value": expected_anchored,
                "confidence_absolute_delta": confidence_delta,
                "exact": not failures,
                "failures": failures,
            }
        )

    exact = sum(bool(row["exact"]) for row in comparisons)
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": "receipt_ppocr_dotnet_wrapper_parity_v1",
        "records": len(comparisons),
        "exact_records": exact,
        "exact_match": exact / len(comparisons),
        "max_confidence_absolute_delta": max(
            (
                float(row["confidence_absolute_delta"])
                for row in comparisons
                if row["confidence_absolute_delta"] is not None
            ),
            default=None,
        ),
        "delivery_contract_sha256": _sha256(delivery_contract),
        "dotnet_latency_ms": {
            "p50": dotnet_summary.get("p50_ms"),
            "p95": dotnet_summary.get("p95_ms"),
        },
        "accepted": exact == len(comparisons),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparisons.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparisons), encoding="utf-8"
    )
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    return summary, bool(summary["accepted"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--dotnet", required=True, type=Path)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary, accepted = compare(
            wrapper=args.wrapper,
            dotnet=args.dotnet,
            delivery=args.delivery,
            output=args.output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"PP-OCR .NET parity failed: {error}")
        return 2
    print(
        f"ppocr_dotnet_parity exact={summary['exact_records']}/{summary['records']}="
        f"{summary['exact_match']:.2%} accepted={summary['accepted']}"
    )
    print(f".NET parity comparison: {args.output.resolve()}")
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
