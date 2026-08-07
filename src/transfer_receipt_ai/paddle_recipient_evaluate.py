"""Evaluate frozen PaddleOCR v2 against held-out unified recipient crops.

This is intentionally a *baseline evaluator*, not a teacher-label shortcut.
It reads the original field crop, runs the independently packaged PaddleOCR
detector/classifier/recognizer in its own process, strictly parses the
anchored recipient value, and compares it with the held-out v12 target.  A
lenient production-style fallback is recorded only as a diagnostic.  It
neither loads Torch nor changes the unified manifest, lightweight model, or
frozen Paddle bundle.

The labels in the current r3 manifest were derived from PaddleOCR, so a strong
result is only Paddle teacher-parity evidence.  A separate human-truth holdout
remains required before calling it production accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from .ocr import OCRResult, PaddleOCRReader, clean_text, extract_field_value, parse_anchored_recipient_row


EVALUATION_KIND = "receipt_paddle_recipient_teacher_parity_v1"
EVALUATION_SCHEMA_VERSION = 1
_RECIPIENT_FIELD = "recipient_field"
_SPLITS = frozenset(("val", "test"))
_FULL_DET_CLS_REC_MODE = "full_det_cls_rec"
_SKIP_DET_CLS_REC_EXPERIMENT_MODE = "experimental_skip_det_cls_rec"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crop_sha256(image_rgb: np.ndarray) -> str:
    """Return the dataset crop identity used by the source manifests.

    The manifest's ``crop_sha256`` is intentionally a digest of the decoded
    RGB shape and pixels, not of the PNG container bytes.  This keeps the
    identity stable across lossless PNG metadata/encoder changes.
    """

    digest = hashlib.sha256()
    digest.update(str(image_rgb.shape).encode("ascii"))
    digest.update(image_rgb.tobytes(order="C"))
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{source}: no JSONL rows")
    return rows


def _require_string(value: object, *, source: Path, line_number: int, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}:{line_number}: {key} must be a non-empty string")
    return value


def _resolve_crop(*, image: str, dataset_root: Path, source: Path, line_number: int) -> Path:
    candidate = Path(image).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (dataset_root / candidate).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"{source}:{line_number}: recipient image escapes dataset root: {image!r}") from error
    if not path.is_file():
        raise FileNotFoundError(f"{source}:{line_number}: recipient image not found: {path}")
    return path


def _load_recipient_records(
    *,
    manifest_path: Path,
    dataset_root: Path,
    split: str,
    limit: int | None,
    require_crop_hash: bool = False,
) -> list[dict[str, object]]:
    source = Path(manifest_path).expanduser().resolve()
    records = _read_jsonl(source)
    seen_ids: set[str] = set()
    selected: list[dict[str, object]] = []
    for line_number, record in enumerate(records, start=1):
        receipt_id = _require_string(record.get("id"), source=source, line_number=line_number, key="id")
        if receipt_id in seen_ids:
            raise ValueError(f"{source}:{line_number}: duplicate manifest id {receipt_id!r}")
        seen_ids.add(receipt_id)
        record_split = _require_string(record.get("split"), source=source, line_number=line_number, key="split")
        if record_split not in {"train", "val", "test"}:
            raise ValueError(f"{source}:{line_number}: split must be train, val, or test")
        slots = record.get("slots")
        if not isinstance(slots, Mapping):
            raise ValueError(f"{source}:{line_number}: slots must be an object")
        slot = slots.get(_RECIPIENT_FIELD)
        if record_split != split or not isinstance(slot, Mapping):
            continue
        reference_text = _require_string(slot.get("text"), source=source, line_number=line_number, key="recipient text")
        reference_visible_text = _require_string(
            slot.get("recipient_visible_text"),
            source=source,
            line_number=line_number,
            key="recipient_visible_text",
        )
        reference_anchored = parse_anchored_recipient_row(reference_visible_text)
        if reference_anchored is None or clean_text(reference_anchored[1]) != reference_text:
            raise ValueError(
                f"{source}:{line_number}: recipient visible row does not strictly anchor to its value target"
            )
        declared_value = slot.get("recipient_value")
        if declared_value is not None and declared_value != reference_text:
            raise ValueError(f"{source}:{line_number}: recipient_value disagrees with recipient text")
        image = _require_string(slot.get("image"), source=source, line_number=line_number, key="recipient image")
        image_path = _resolve_crop(
            image=image,
            dataset_root=dataset_root,
            source=source,
            line_number=line_number,
        )
        declared_crop_hash = slot.get("crop_sha256")
        if require_crop_hash and (
            not isinstance(declared_crop_hash, str)
            or len(declared_crop_hash) != 64
            or any(character not in "0123456789abcdef" for character in declared_crop_hash)
        ):
            raise ValueError(f"{source}:{line_number}: bundle-bound evidence requires crop_sha256")
        with Image.open(image_path) as image_handle:
            image_rgb = np.asarray(image_handle.convert("RGB")).copy()
        actual_crop_hash = _crop_sha256(image_rgb)
        if declared_crop_hash is not None and declared_crop_hash != actual_crop_hash:
            raise ValueError(f"{source}:{line_number}: recipient crop SHA-256 differs from manifest")
        actual_crop_file_hash = _sha256(image_path)
        selected.append(
            {
                "id": receipt_id,
                "split": record_split,
                "group_id": record.get("group_id"),
                "source": record.get("source"),
                "result_json": record.get("result_json"),
                "crop_sha256": actual_crop_hash,
                "crop_file_sha256": actual_crop_file_hash,
                "image": image_path.as_posix(),
                "reference_text": reference_text,
                "recipient_visible_text": reference_visible_text,
                "source_record_id": slot.get("source_record_id"),
            }
        )
    selected.sort(key=lambda row: str(row["id"]))
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"{source}: no recipient_field records for held-out split {split!r}")
    return selected


def _validate_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized == "cpu" or normalized == "cuda" or re.fullmatch(r"cuda:\d+", normalized):
        return normalized
    raise ValueError("device must be cpu, cuda, or cuda:N")


def _default_reader_factory(device: str) -> PaddleOCRReader:
    if "torch" in sys.modules:
        raise RuntimeError("Paddle recipient evaluation must start in a process that has not imported Torch")
    # This explicit cache preflight makes the evaluator fail before PaddleOCR
    # can fetch a different default model version.
    from .paddle_ocr_bundle import _preflight_default_v2_assets

    _preflight_default_v2_assets(allow_model_download=False)
    reader = PaddleOCRReader(device=device, require_v2=True)
    if "torch" in sys.modules:
        raise RuntimeError("PaddleOCR initialisation imported Torch; refuse mixed-framework GPU evaluation")
    return reader


def _default_runtime_probe(_: object) -> dict[str, object]:
    import paddle

    return {
        "paddleocr_version": metadata.version("paddleocr"),
        "paddle_version": str(getattr(paddle, "__version__", "")),
        "active_paddle_device": str(paddle.get_device()),
        "torch_imported": "torch" in sys.modules,
    }


def _verify_reader_matches_bundle(reader: object, bundle_dir: Path) -> dict[str, object]:
    """Hash-check a frozen bundle and ensure the live reader resolved the same assets."""
    from .paddle_ocr_bundle import CONTRACT_FILENAME, MODEL_ROLES, verify_bundle

    bundle = Path(bundle_dir).expanduser().resolve()
    contract = verify_bundle(bundle)
    engine = getattr(reader, "_engine", None)
    args = vars(getattr(engine, "args", None)) if getattr(engine, "args", None) is not None else None
    if not isinstance(args, dict):
        raise ValueError("PaddleOCR reader does not expose v2 effective runtime arguments")
    assets = contract.get("assets")
    dictionary = contract.get("dictionary")
    if not isinstance(assets, Mapping) or not isinstance(dictionary, Mapping):
        raise ValueError("PaddleOCR bundle contract has invalid assets")
    paths = {"det": "det_model_dir", "rec": "rec_model_dir", "cls": "cls_model_dir"}
    for role, argument in paths.items():
        asset = assets.get(role)
        expected = asset.get("source_directory") if isinstance(asset, Mapping) else None
        actual = args.get(argument)
        if not isinstance(expected, str) or not isinstance(actual, str):
            raise ValueError(f"PaddleOCR bundle/runtime lacks {role} model path")
        if Path(actual).expanduser().resolve() != Path(expected).expanduser().resolve():
            raise ValueError(
                f"PaddleOCR {role} model differs from frozen bundle: runtime={actual}, bundle={expected}"
            )
        live_directory = Path(actual).expanduser().resolve()
        bundle_directory = asset.get("bundle_directory") if isinstance(asset, Mapping) else None
        file_records = asset.get("files") if isinstance(asset, Mapping) else None
        if not isinstance(bundle_directory, str) or not isinstance(file_records, list):
            raise ValueError(f"PaddleOCR bundle has invalid frozen {role} file records")
        expected_relative_files: set[str] = set()
        bundle_prefix = Path(bundle_directory)
        for record in file_records:
            if not isinstance(record, Mapping):
                raise ValueError(f"PaddleOCR bundle has invalid frozen {role} file record")
            relative = record.get("path")
            expected_hash = record.get("sha256")
            expected_size = record.get("size_bytes")
            if not isinstance(relative, str) or not isinstance(expected_hash, str) or not isinstance(expected_size, int):
                raise ValueError(f"PaddleOCR bundle has invalid frozen {role} file record")
            try:
                live_relative = Path(relative).relative_to(bundle_prefix)
            except ValueError as error:
                raise ValueError(f"Frozen {role} file escapes its bundle directory: {relative}") from error
            expected_relative_files.add(live_relative.as_posix())
            live_file = (live_directory / live_relative).resolve()
            try:
                live_file.relative_to(live_directory)
            except ValueError as error:
                raise ValueError(f"Frozen {role} file escapes its live model directory: {relative}") from error
            if not live_file.is_file() or live_file.stat().st_size != expected_size or _sha256(live_file) != expected_hash:
                raise ValueError(f"PaddleOCR live {role} model bytes differ from frozen bundle: {live_file}")
        actual_relative_files = {
            path.relative_to(live_directory).as_posix()
            for path in live_directory.rglob("*")
            if path.is_file()
        }
        if actual_relative_files != expected_relative_files:
            raise ValueError(f"PaddleOCR live {role} model file set differs from frozen bundle")
    expected_charset = dictionary.get("source_path")
    actual_charset = args.get("rec_char_dict_path")
    if not isinstance(expected_charset, str) or not isinstance(actual_charset, str):
        raise ValueError("PaddleOCR bundle/runtime lacks recognition charset path")
    if Path(actual_charset).expanduser().resolve() != Path(expected_charset).expanduser().resolve():
        raise ValueError("PaddleOCR recognition charset differs from frozen bundle")
    charset_path = Path(actual_charset).expanduser().resolve()
    charset_hash = dictionary.get("sha256")
    charset_size = dictionary.get("size_bytes")
    if (
        not isinstance(charset_hash, str)
        or not isinstance(charset_size, int)
        or charset_path.stat().st_size != charset_size
        or _sha256(charset_path) != charset_hash
    ):
        raise ValueError("PaddleOCR live recognition charset bytes differ from frozen bundle")
    native_identity = contract.get("native_asset_identity")
    if not isinstance(native_identity, Mapping):
        raise ValueError("PaddleOCR bundle has no verified native asset identity")
    components = native_identity.get("components")
    if not isinstance(components, Mapping) or set(components) != {*MODEL_ROLES, "dictionary"}:
        raise ValueError("PaddleOCR bundle has invalid native component identities")
    return {
        "path": bundle.as_posix(),
        "contract_kind": contract.get("kind"),
        "contract_sha256": _sha256(bundle / CONTRACT_FILENAME),
        "native_asset_identity_sha256": native_identity.get("sha256"),
        "native_component_sha256": dict(components),
        "live_source_bytes_verified": True,
        "verified": True,
    }


def _recipient_values(raw_text: str) -> tuple[str | None, str | None]:
    """Return strict anchored and lenient fallback values separately.

    The v12 target is an anchored row.  The fallback has real operational value
    for diagnosis, but must never be used for the acceptance metric because it
    can turn a label-position error into a superficially correct merchant name.
    """
    cleaned = clean_text(raw_text)
    anchored = parse_anchored_recipient_row(cleaned)
    anchored_value = clean_text(anchored[1]) if anchored is not None else None
    fallback_value = clean_text(extract_field_value(cleaned, "recipient")) or None
    return anchored_value, fallback_value


def _inference_mode(*, skip_detection: bool) -> dict[str, object]:
    """Describe the exact Paddle stages used by this evaluation.

    The experiment is intentionally explicit because skip-detection on a
    recipient crop is a speed A/B, not a replacement for the standard
    detector-backed inference path.
    """
    return {
        "name": _SKIP_DET_CLS_REC_EXPERIMENT_MODE if skip_detection else _FULL_DET_CLS_REC_MODE,
        "experimental": skip_detection,
        "detection_enabled": not skip_detection,
        "angle_classifier_enabled": True,
        "recognizer_enabled": True,
    }


def _recognize_recipient(reader: object, image_rgb: np.ndarray, *, skip_detection: bool) -> OCRResult:
    """Keep the default reader invocation byte-for-byte conventional.

    Passing ``det=False`` only in the explicit experimental branch matters for
    both backwards compatibility and for fake-reader tests that deliberately
    implement only the historical ``recognize(image)`` call.
    """
    recognize = getattr(reader, "recognize", None)
    if not callable(recognize):
        raise TypeError("Paddle reader must expose recognize(image_rgb)")
    result = recognize(image_rgb, det=False) if skip_detection else recognize(image_rgb)
    if not isinstance(result, OCRResult):
        raise TypeError("Paddle reader must return OCRResult")
    return result


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(dict(value), ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_paddle_recipients(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    split: str = "val",
    device: str = "cuda",
    limit: int | None = None,
    target_value_exact_match: float = 0.90,
    progress_every: int = 25,
    bundle_dir: Path | None = None,
    skip_detection: bool = False,
    reader_factory: Callable[[str], object] = _default_reader_factory,
    runtime_probe: Callable[[object], Mapping[str, object]] = _default_runtime_probe,
) -> tuple[dict[str, object], bool]:
    """Evaluate one held-out recipient split without modifying any model input.

    ``reader_factory`` and ``runtime_probe`` are dependency-injection seams for
    deterministic tests; production callers use the real, strict v2 Paddle
    reader and record its active device.
    """
    if split not in _SPLITS:
        raise ValueError("split must be val or test")
    normalized_device = _validate_device(device)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer when present")
    if isinstance(progress_every, bool) or not isinstance(progress_every, int) or progress_every <= 0:
        raise ValueError("progress_every must be a positive integer")
    if not isinstance(skip_detection, bool):
        raise ValueError("skip_detection must be a boolean")
    try:
        target = float(target_value_exact_match)
    except (TypeError, ValueError) as error:
        raise ValueError("target_value_exact_match must be a finite probability in (0, 1]") from error
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("target_value_exact_match must be a finite probability in (0, 1]")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing Paddle recipient evaluation: {output}")
    manifest = Path(manifest_path).expanduser().resolve()
    manifest_sha256 = _sha256(manifest)
    records = _load_recipient_records(
        manifest_path=manifest_path,
        dataset_root=root,
        split=split,
        limit=limit,
        require_crop_hash=bundle_dir is not None,
    )
    reader = reader_factory(normalized_device)
    runtime = dict(runtime_probe(reader))
    active_device = str(runtime.get("active_paddle_device", ""))
    if normalized_device.startswith("cuda") and not active_device.startswith("gpu"):
        raise RuntimeError(f"Paddle CUDA was requested but active device is {active_device or 'unknown'}")
    if bool(runtime.get("torch_imported")):
        raise RuntimeError("Paddle recipient evaluation detected Torch in its worker process")
    bundle: dict[str, object] | None = None
    if bundle_dir is not None:
        bundle = _verify_reader_matches_bundle(reader, Path(bundle_dir))
    inference_mode = _inference_mode(skip_detection=skip_detection)

    comparisons: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    confidences: list[float] = []
    candidate_modes: Counter[str] = Counter()
    for number, record in enumerate(records, start=1):
        image_path = Path(str(record["image"]))
        if _sha256(image_path) != record["crop_file_sha256"]:
            raise ValueError(f"Recipient crop changed after manifest validation: {image_path}")
        with Image.open(image_path) as image:
            image_rgb = np.asarray(image.convert("RGB")).copy()
        if _crop_sha256(image_rgb) != record["crop_sha256"]:
            raise ValueError(f"Recipient crop pixels changed after manifest validation: {image_path}")
        started = perf_counter()
        result = _recognize_recipient(reader, image_rgb, skip_detection=skip_detection)
        elapsed_ms = (perf_counter() - started) * 1000.0
        candidate_anchored_value, candidate_fallback_value = _recipient_values(result.text)
        reference_text = str(record["reference_text"])
        anchored_exact = candidate_anchored_value == reference_text
        fallback_exact = candidate_fallback_value == reference_text
        raw_visible_exact = clean_text(result.text) == str(record["recipient_visible_text"])
        extraction_mode = "anchored" if candidate_anchored_value is not None else "anchor_parse_failed"
        lines = [{"text": text, "confidence": confidence} for text, confidence in result.lines]
        comparison = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": EVALUATION_KIND,
            "inference_mode": inference_mode["name"],
            **record,
            "raw_paddle_text": result.text,
            "paddle_confidence": result.confidence,
            "paddle_lines": lines,
            "candidate_anchored_value": candidate_anchored_value,
            "candidate_fallback_value": candidate_fallback_value,
            "candidate_extraction_mode": extraction_mode,
            "raw_visible_exact": raw_visible_exact,
            "anchored_value_exact": anchored_exact,
            "fallback_value_exact": fallback_exact,
            "inference_ms": round(elapsed_ms, 4),
        }
        comparisons.append(comparison)
        latencies_ms.append(elapsed_ms)
        candidate_modes[extraction_mode] += 1
        if result.confidence is not None:
            confidences.append(float(result.confidence))
        if number == 1 or number == len(records) or number % progress_every == 0:
            exact_count = sum(bool(item["anchored_value_exact"]) for item in comparisons)
            print(f"paddle recipient {number}/{len(records)} exact={exact_count}/{number}={exact_count / number:.2%}")

    raw_visible_exact_matches = sum(bool(record["raw_visible_exact"]) for record in comparisons)
    anchored_exact_matches = sum(bool(record["anchored_value_exact"]) for record in comparisons)
    fallback_exact_matches = sum(bool(record["fallback_value_exact"]) for record in comparisons)
    anchored_value_exact_match = anchored_exact_matches / len(comparisons)
    if _sha256(manifest) != manifest_sha256:
        raise ValueError(f"Manifest changed during Paddle recipient evaluation: {manifest}")
    if bundle_dir is not None:
        bundle_after = _verify_reader_matches_bundle(reader, Path(bundle_dir))
        if bundle_after != bundle:
            raise ValueError("Frozen Paddle bundle or live source assets changed during evaluation")
        bundle = {**bundle_after, "verified_before_and_after": True}

    output.mkdir(parents=True, exist_ok=False)
    comparisons_path = output / "comparisons.jsonl"
    disagreements_path = output / "disagreements.jsonl"
    _atomic_jsonl(comparisons_path, comparisons)
    _atomic_jsonl(disagreements_path, [row for row in comparisons if not bool(row["anchored_value_exact"])])
    summary: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": EVALUATION_KIND,
        "manifest": manifest.as_posix(),
        "manifest_sha256": manifest_sha256,
        "comparisons_sha256": _sha256(comparisons_path),
        "disagreements_sha256": _sha256(disagreements_path),
        "dataset_root": root.as_posix(),
        "evaluation_split": split,
        "records": len(comparisons),
        "limit": limit,
        "requested_device": normalized_device,
        "inference_mode": inference_mode,
        "runtime": runtime,
        "frozen_bundle": bundle,
        "raw_visible_exact_matches": raw_visible_exact_matches,
        "raw_visible_exact_match": raw_visible_exact_matches / len(comparisons),
        "anchored_value_exact_matches": anchored_exact_matches,
        "anchored_value_exact_match": anchored_value_exact_match,
        "fallback_value_exact_matches": fallback_exact_matches,
        "fallback_value_exact_match": fallback_exact_matches / len(comparisons),
        "anchor_parse_failure_records": int(candidate_modes["anchor_parse_failed"]),
        "anchor_parse_failure_rate": candidate_modes["anchor_parse_failed"] / len(comparisons),
        "candidate_extraction_modes": dict(sorted(candidate_modes.items())),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms),
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
            "max": max(latencies_ms),
        },
        "acceptance": {
            "target_anchored_value_exact_match": target,
            "passed": anchored_value_exact_match >= target,
        },
        "warning": (
            "This is held-out teacher-parity on Paddle-derived labels, not independent human-truth accuracy. "
            "It proves neither business accuracy nor a replacement model until a human-truth holdout is evaluated."
        ),
    }
    _atomic_json(output / "summary.json", summary)
    return summary, bool(summary["acceptance"]["passed"])


def _format_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def format_paddle_recipient_evaluation(summary: Mapping[str, object]) -> str:
    """Render a compact, screenshot-ready conclusion for the remote terminal."""
    latency = summary.get("latency_ms")
    acceptance = summary.get("acceptance")
    runtime = summary.get("runtime")
    inference_mode = summary.get("inference_mode")
    if (
        not isinstance(latency, Mapping)
        or not isinstance(acceptance, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(inference_mode, Mapping)
    ):
        raise ValueError("Paddle recipient summary is invalid")
    lines = [
        "paddle_recipient_evaluation",
        f"  mode={inference_mode.get('name')}; experimental={inference_mode.get('experimental')}; "
        f"det={inference_mode.get('detection_enabled')}; cls={inference_mode.get('angle_classifier_enabled')}; "
        f"rec={inference_mode.get('recognizer_enabled')}",
        f"  anchored={summary.get('anchored_value_exact_matches')}/{summary.get('records')}="
        f"{_format_rate(summary.get('anchored_value_exact_match'))}; split={summary.get('evaluation_split')}",
        f"  raw-visible={summary.get('raw_visible_exact_matches')}/{summary.get('records')}="
        f"{_format_rate(summary.get('raw_visible_exact_match'))}; fallback={summary.get('fallback_value_exact_matches')}/"
        f"{summary.get('records')}={_format_rate(summary.get('fallback_value_exact_match'))}; "
        f"anchor-parse-failed={_format_rate(summary.get('anchor_parse_failure_rate'))}",
        f"  device=requested:{summary.get('requested_device')}, active:{runtime.get('active_paddle_device')}; "
        f"torch_imported={runtime.get('torch_imported')}",
        f"  latency_ms=mean:{float(latency.get('mean') or 0.0):.2f}, "
        f"p50:{float(latency.get('p50') or 0.0):.2f}, p95:{float(latency.get('p95') or 0.0):.2f}, "
        f"max:{float(latency.get('max') or 0.0):.2f}",
        f"  extraction_modes={summary.get('candidate_extraction_modes')}",
        f"  target={_format_rate(acceptance.get('target_anchored_value_exact_match'))}; "
        f"accepted={acceptance.get('passed')}",
        f"  frozen_bundle={summary.get('frozen_bundle')}",
        f"  warning={summary.get('warning')}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run native PaddleOCR v2 on held-out recipient crops and compare value-only outputs"
    )
    parser.add_argument("--manifest", required=True, type=Path, help="unified_fields.jsonl with v12 recipient slots")
    parser.add_argument("--dataset-root", required=True, type=Path, help="crop root used by manifest-relative image paths")
    parser.add_argument("--output", required=True, type=Path, help="new output directory; refuses to overwrite")
    parser.add_argument("--split", choices=sorted(_SPLITS), default="val")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N; CUDA never silently falls back")
    parser.add_argument("--limit", type=int, help="optional deterministic prefix after id sorting")
    parser.add_argument("--target", type=float, default=0.90, help="value-exact acceptance target, default: 0.90")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bundle", type=Path, help="optional frozen Paddle bundle to hash-verify and match to live assets")
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="EXPERIMENTAL: call pinned PaddleOCR v2 with det=False while retaining cls+rec; use only for speed A/B",
    )
    parser.add_argument("--json", action="store_true", help="print complete summary JSON instead of compact text")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary, accepted = evaluate_paddle_recipients(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_dir=args.output,
            split=args.split,
            device=args.device,
            limit=args.limit,
            target_value_exact_match=args.target,
            progress_every=args.progress_every,
            bundle_dir=args.bundle,
            skip_detection=args.skip_detection,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"Paddle recipient evaluation failed: {error}") from error
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_paddle_recipient_evaluation(summary))
        print(f"Wrote Paddle recipient evaluation to {Path(args.output).expanduser().resolve()}")
    if not accepted:
        raise SystemExit(3)


if __name__ == "__main__":  # pragma: no cover - executed through the hyphenated checkout wrapper.
    main()
