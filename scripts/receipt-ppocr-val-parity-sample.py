#!/usr/bin/env python3
"""Create an immutable converter-parity crop set from PP-OCR val evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from PIL import Image


EVALUATION_SCHEMA_VERSION = 1
EVALUATION_KIND = "receipt_paddle_recipient_teacher_parity_v1"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "paddle_ocr_v2_bundle"
BUNDLE_CONTRACT_FILENAME = "paddle_ocr_bundle.contract.json"
EXPECTED_FULL_VAL_RECORDS = 6789


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crop_sha256(path: Path) -> str:
    """Recompute the decoded RGB pixel identity stored by the source dataset."""

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        shape = (rgb.height, rgb.width, 3)
        pixels = rgb.tobytes()
    digest = hashlib.sha256()
    digest.update(str(shape).encode("ascii"))
    digest.update(pixels)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase 64-character SHA-256")
    return value


def create_sample(
    *,
    evidence: Path,
    output: Path,
    limit: int,
    audit_bundle: Path | None = None,
    trusted_manifest_sha256: str | None = None,
    expected_records: int = EXPECTED_FULL_VAL_RECORDS,
) -> Path:
    evidence = evidence.resolve()
    output = output.resolve()
    if limit <= 0:
        raise ValueError("limit must be positive")
    if expected_records <= 0:
        raise ValueError("expected_records must be positive")
    if output.exists():
        raise ValueError(f"Refusing to overwrite parity sample: {output}")
    summary_path = evidence / "summary.json"
    comparisons_path = evidence / "comparisons.jsonl"
    if not summary_path.is_file() or not comparisons_path.is_file():
        raise ValueError(f"PP-OCR evidence is incomplete: {evidence}")
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("PP-OCR summary must be an object")
    mode = summary.get("inference_mode")
    acceptance = summary.get("acceptance")
    if (
        summary.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or summary.get("kind") != EVALUATION_KIND
        or summary.get("evaluation_split") != "val"
        or summary.get("limit") is not None
        or summary.get("records") != expected_records
        or summary.get("requested_device") not in {"cuda", "cuda:0"}
        or not isinstance(mode, dict)
        or mode.get("name") != "full_det_cls_rec"
        or mode.get("detection_enabled") is not True
        or mode.get("angle_classifier_enabled") is not True
        or mode.get("recognizer_enabled") is not True
        or not isinstance(acceptance, dict)
        or acceptance.get("passed") is not True
        or float(acceptance.get("target_anchored_value_exact_match", 0.0)) < 0.90
        or float(summary.get("anchored_value_exact_match", 0.0)) < 0.90
    ):
        raise ValueError("PP-OCR evidence is not a passing full unbounded det/cls/rec validation-split run")
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict) or not str(runtime.get("active_paddle_device", "")).startswith("gpu"):
        raise ValueError("PP-OCR evidence did not run on the requested CUDA Paddle device")
    if audit_bundle is None or trusted_manifest_sha256 is None:
        raise ValueError("Full evidence validation requires the exact audit bundle and trusted manifest SHA-256")
    audit_bundle = audit_bundle.resolve()
    contract_path = audit_bundle / BUNDLE_CONTRACT_FILENAME
    if not contract_path.is_file():
        raise ValueError(f"Missing audit bundle contract: {contract_path}")
    audit_contract = _load_json(contract_path)
    audit_onnx = audit_contract.get("onnx") if isinstance(audit_contract, dict) else None
    if (
        not isinstance(audit_contract, dict)
        or audit_contract.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or audit_contract.get("kind") != BUNDLE_KIND
        or not isinstance(audit_onnx, Mapping)
        or set(audit_onnx) != {"det", "rec", "cls"}
    ):
        raise ValueError("Audit bundle is not the fully exported det/rec/cls bundle required by this evidence")
    native_identity = audit_contract.get("native_asset_identity")
    frozen = summary.get("frozen_bundle")
    if not isinstance(native_identity, dict) or not isinstance(frozen, dict):
        raise ValueError("PP-OCR evidence is not bound to a frozen native asset identity")
    if (
        Path(str(frozen.get("path", ""))).resolve() != audit_bundle
        or frozen.get("contract_kind") != BUNDLE_KIND
        or frozen.get("contract_sha256") != _sha256(contract_path)
        or frozen.get("native_asset_identity_sha256") != native_identity.get("sha256")
        or frozen.get("native_component_sha256") != native_identity.get("components")
        or frozen.get("live_source_bytes_verified") is not True
        or frozen.get("verified_before_and_after") is not True
        or frozen.get("verified") is not True
    ):
        raise ValueError("PP-OCR evidence bundle identity does not match the supplied audit bundle")
    trusted_manifest_sha256 = _require_sha256(trusted_manifest_sha256, "trusted manifest SHA-256")
    manifest_path_value = summary.get("manifest")
    if not isinstance(manifest_path_value, str) or not Path(manifest_path_value).is_file():
        raise ValueError("PP-OCR evidence manifest is missing")
    manifest_path = Path(manifest_path_value).resolve()
    if summary.get("manifest_sha256") != trusted_manifest_sha256 or _sha256(manifest_path) != trusted_manifest_sha256:
        raise ValueError("PP-OCR evidence does not match the trusted full manifest SHA-256")
    if summary.get("comparisons_sha256") != _sha256(comparisons_path):
        raise ValueError("PP-OCR evidence comparisons SHA-256 differs from summary")

    rows = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(comparisons_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or row.get("kind") != EVALUATION_KIND
            or row.get("split") != "val"
            or row.get("inference_mode") != "full_det_cls_rec"
        ):
            raise ValueError(f"Invalid/non-val comparison at {comparisons_path}:{line_number}")
        receipt_id = row.get("id")
        if not isinstance(receipt_id, str) or not receipt_id or receipt_id in seen_ids:
            raise ValueError(f"Invalid/duplicate comparison id at {comparisons_path}:{line_number}")
        seen_ids.add(receipt_id)
        source = row.get("image")
        if not isinstance(source, str) or not Path(source).is_file():
            raise ValueError(f"Missing comparison crop at {comparisons_path}:{line_number}")
        crop_sha256 = _require_sha256(row.get("crop_sha256"), f"comparison crop SHA-256 at line {line_number}")
        crop_file_sha256 = _require_sha256(
            row.get("crop_file_sha256"),
            f"comparison crop file SHA-256 at line {line_number}",
        )
        if _sha256(Path(source)) != crop_file_sha256:
            raise ValueError(f"Comparison crop file hash mismatch at {comparisons_path}:{line_number}")
        try:
            actual_crop_sha256 = _crop_sha256(Path(source))
        except OSError as error:
            raise ValueError(f"Comparison decoded crop hash mismatch at {comparisons_path}:{line_number}") from error
        if actual_crop_sha256 != crop_sha256:
            raise ValueError(f"Comparison decoded crop hash mismatch at {comparisons_path}:{line_number}")
        rows.append(row)
    if len(rows) != expected_records or len(rows) != int(summary.get("records", -1)):
        raise ValueError("PP-OCR summary/comparison record counts differ")
    anchored_matches = sum(row.get("anchored_value_exact") is True for row in rows)
    anchored_rate = anchored_matches / len(rows)
    if (
        int(summary.get("anchored_value_exact_matches", -1)) != anchored_matches
        or abs(float(summary.get("anchored_value_exact_match", -1.0)) - anchored_rate) > 1e-12
        or bool(acceptance.get("passed")) != (anchored_rate >= float(acceptance.get("target_anchored_value_exact_match", 2.0)))
        or anchored_rate < 0.90
    ):
        raise ValueError("PP-OCR summary accuracy is not exactly derivable from its comparisons")
    rows.sort(key=lambda row: (str(row.get("id", "")), str(row["image"])))

    selected = []
    seen: set[str] = set()
    for row in rows:
        source = str(Path(str(row["image"])).resolve())
        key = os.path.normcase(source)
        if key in seen:
            continue
        seen.add(key)
        selected.append((row, Path(source)))
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(f"Only {len(selected)} unique val crops are available; requested {limit}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    records = []
    try:
        images = stage / "images"
        images.mkdir()
        for index, (row, source) in enumerate(selected, start=1):
            source_hash = _sha256(source)
            destination = images / f"{index:04d}-{source_hash[:16]}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            if _sha256(destination) != source_hash:
                raise ValueError(f"Parity crop copy hash mismatch: {source}")
            records.append(
                {
                    "id": row.get("id"),
                    "split": "val",
                    "source": source.as_posix(),
                    "source_sha256": source_hash,
                    "sample": destination.relative_to(stage).as_posix(),
                }
            )
        (stage / "sample_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "receipt_ppocr_val_converter_parity_sample_v1",
                    "source_evidence": evidence.as_posix(),
                    "source_summary_sha256": _sha256(summary_path),
                    "source_comparisons_sha256": _sha256(comparisons_path),
                    "source_manifest_sha256": trusted_manifest_sha256,
                    "source_audit_contract_sha256": _sha256(contract_path),
                    "source_native_asset_identity_sha256": native_identity.get("sha256"),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--audit-bundle", required=True, type=Path)
    parser.add_argument("--trusted-manifest-sha256", required=True)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_FULL_VAL_RECORDS)
    args = parser.parse_args()
    try:
        output = create_sample(
            evidence=args.evidence,
            output=args.output,
            limit=args.limit,
            audit_bundle=args.audit_bundle,
            trusted_manifest_sha256=args.trusted_manifest_sha256,
            expected_records=args.expected_records,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not create PP-OCR val parity sample: {error}")
        return 2
    print(f"Created immutable val-only PP-OCR parity sample: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
