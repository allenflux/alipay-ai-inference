"""Export and review clean transfer-status crops from inference bundles.

The five-field detector already stores everything needed to reproduce the
rectified image.  This module deliberately reconstructs that image from the
upright *source* and the recorded homography instead of cropping an annotated
preview, so the status/check-mark classifier never learns the coloured review
circles or captions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import load_upright_rgb, save_rgb

STATUS_LABELS = ("check_offset", "check_aligned", "check_absent", "unclear")
_KEY_LABELS = {
    "1": "check_offset",
    "2": "check_aligned",
    "3": "check_absent",
    "4": "unclear",
}

ManifestRecord = dict[str, str | None]
KeyProvider = Callable[[ManifestRecord, np.ndarray, int, int], str | int]


class UnsafeStatusCropOutputError(ValueError):
    """Raised for path layouts that could modify v1 results or raw sources."""


class StatusCropConfigurationError(ValueError):
    """Raised when existing crops do not match the requested export setup."""


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return true when either resolved path contains the other."""
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _export_config(results_dir: Path, margin_ratio: float) -> dict[str, object]:
    return {
        "schema": 1,
        "results": results_dir.resolve().as_posix(),
        "margin": float(margin_ratio),
    }


def _has_existing_crops(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for path in output_dir.rglob("*")
    )


def _prepare_export_config(
    *,
    config_path: Path,
    desired: Mapping[str, object],
    output_dir: Path,
    skip_existing: bool,
) -> bool:
    """Validate crop provenance and return whether config commit must wait.

    A changed configuration is committed only after a successful overwrite.
    If that recrop is interrupted, the old config remains and the next default
    run refuses to mix old and new crops instead of silently skipping them.
    """
    if config_path.is_file():
        try:
            current = _load_json_document(config_path)
        except ValueError:
            if skip_existing:
                raise StatusCropConfigurationError(
                    f"Existing crop config is invalid: {config_path}. Rerun with --overwrite to rebuild crops."
                ) from None
            return True
        if current == dict(desired):
            return False
        if skip_existing:
            raise StatusCropConfigurationError(
                "Status crop configuration changed; refusing to reuse existing crops. "
                "Rerun with --overwrite to update status_crops_config.json and recrop."
            )
        return True
    if _has_existing_crops(output_dir):
        if skip_existing:
            raise StatusCropConfigurationError(
                "Existing status crops have no status_crops_config.json; "
                "rerun with --overwrite to rebuild them safely."
            )
        return True
    # A brand-new export can record its provenance up front. Every crop written
    # afterwards uses this exact geometry configuration, including after resume.
    _atomic_write_json(config_path, desired)
    return False


def _load_json_document(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from None


def _result_payload(document: Any) -> Mapping[str, Any] | None:
    """Return a result bundle, while silently ignoring inference metadata JSON."""
    if not isinstance(document, Mapping):
        return None
    if not all(key in document for key in ("source", "geometry", "detections")):
        return None
    return document


def _selection_key(relative_path: Path) -> tuple[bytes, str]:
    relative = relative_path.as_posix()
    return hashlib.sha256(relative.encode("utf-8")).digest(), relative


_CAPTURE_TIMESTAMP_SUFFIX = re.compile(r"^(?P<base>.+?)[_-](?:19|20)\d{12}$")


def _group_id(source: Path, relative_result: Path) -> str:
    """Group repeated captures of the same receipt for leakage-safe splitting."""
    match = _CAPTURE_TIMESTAMP_SUFFIX.match(source.stem)
    group_stem = match.group("base") if match else relative_result.stem
    return (relative_result.parent / group_stem).as_posix()


def reconstruct_rectified(payload: Mapping[str, Any], source_rgb: np.ndarray) -> np.ndarray:
    """Recreate the detector/OCR view using geometry stored in a result JSON."""
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise ValueError("source_rgb must be an H×W×3 RGB array")
    geometry_value = payload.get("geometry", payload)
    if not isinstance(geometry_value, Mapping):
        raise ValueError("result has no geometry object")
    size = geometry_value.get("rectified_size")
    if not isinstance(size, Mapping):
        raise ValueError("geometry has no rectified_size")
    try:
        width = int(size["width"])
        height = int(size["height"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("rectified_size must contain integer width and height") from None
    if width < 2 or height < 2:
        raise ValueError("rectified_size width and height must be at least 2")

    source_size = geometry_value.get("source_size")
    if isinstance(source_size, Mapping):
        try:
            recorded_width = int(source_size["width"])
            recorded_height = int(source_size["height"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("source_size must contain integer width and height") from None
        actual_height, actual_width = source_rgb.shape[:2]
        if (actual_width, actual_height) != (recorded_width, recorded_height):
            raise ValueError(
                "upright source size changed since inference: "
                f"recorded={recorded_width}x{recorded_height}, actual={actual_width}x{actual_height}"
            )

    homography = np.asarray(geometry_value.get("H_original_to_rectified"), dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("H_original_to_rectified must be a finite 3×3 matrix")
    if abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("H_original_to_rectified is singular")
    return cv2.warpPerspective(
        source_rgb,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_status_region(
    rectified_rgb: np.ndarray,
    bbox_xyxy: Sequence[float],
    margin_ratio: float = 0.30,
) -> np.ndarray:
    """Crop a status box plus proportional context, clipped to image bounds."""
    if rectified_rgb.ndim != 3 or rectified_rgb.shape[2] != 3:
        raise ValueError("rectified_rgb must be an H×W×3 RGB array")
    if margin_ratio < 0:
        raise ValueError("margin_ratio must be non-negative")
    if len(bbox_xyxy) != 4:
        raise ValueError("transfer_status bbox must contain four coordinates")
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    except (TypeError, ValueError):
        raise ValueError("transfer_status bbox coordinates must be numeric") from None
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("transfer_status bbox must be a finite non-empty xyxy box")

    image_height, image_width = rectified_rgb.shape[:2]
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    left = max(0, int(np.floor(x1 - margin_x)))
    top = max(0, int(np.floor(y1 - margin_y)))
    right = min(image_width, int(np.ceil(x2 + margin_x)))
    bottom = min(image_height, int(np.ceil(y2 + margin_y)))
    if right <= left or bottom <= top:
        raise ValueError("transfer_status bbox does not intersect the rectified image")
    return rectified_rgb[top:bottom, left:right].copy()


def _status_bbox(payload: Mapping[str, Any]) -> Sequence[float]:
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ValueError("result has no detections list")
    matches = [
        item
        for item in detections
        if isinstance(item, Mapping) and item.get("label") == "transfer_status"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one transfer_status detection, found {len(matches)}")
    bbox = matches[0].get("bbox_rectified")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        raise ValueError("transfer_status detection has no bbox_rectified")
    return bbox


def _source_path(payload: Mapping[str, Any], result_json: Path) -> Path:
    value = payload.get("source")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("result has no source path")
    source = Path(value)
    if not source.is_absolute():
        source = result_json.parent / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _committed_crop_exists(crop_path: Path, result_json: Path, source: Path) -> bool:
    return (
        crop_path.is_file()
        and crop_path.stat().st_size > 0
        and crop_path.stat().st_mtime_ns >= max(result_json.stat().st_mtime_ns, source.stat().st_mtime_ns)
    )


def _cohort_work_items(
    *,
    results_dir: Path,
    manifest_path: Path,
    json_paths: Sequence[Path],
    limit: int | None,
) -> list[tuple[Path, ManifestRecord | None]]:
    """Keep an existing cohort as a frozen prefix, then append new results."""
    existing_records = _load_manifest(manifest_path) if manifest_path.is_file() else []
    if limit is not None and limit < len(existing_records):
        raise ValueError(
            f"--limit {limit} is smaller than the existing cohort of {len(existing_records)}; "
            "use at least the existing size so reviewed rows are never orphaned"
        )

    items: list[tuple[Path, ManifestRecord | None]] = []
    existing_paths: set[str] = set()
    for record in existing_records:
        result_path = Path(str(record["result_json"]))
        if not result_path.is_absolute():
            result_path = manifest_path.parent / result_path
        result_path = result_path.resolve()
        try:
            result_path.relative_to(results_dir)
        except ValueError:
            raise UnsafeStatusCropOutputError(
                f"existing cohort result is outside configured results directory: {result_path}"
            ) from None
        key = result_path.as_posix().casefold()
        existing_paths.add(key)
        # Labels live in reviewed.jsonl. The export manifest remains an
        # unlabelled cohort contract even if a hand-edited manifest is supplied.
        items.append((result_path, {**record, "label": None}))

    for result_path in json_paths:
        resolved = result_path.resolve()
        if resolved.as_posix().casefold() not in existing_paths:
            items.append((resolved, None))
    return items


def _preflight_source_directories(
    *,
    json_paths: Sequence[Path],
    output_dir: Path,
    manifest_path: Path,
    errors_path: Path,
) -> None:
    """Reject unsafe write trees before reading/writing an existing manifest."""
    protected_output_dirs = (output_dir, manifest_path.parent, errors_path.parent)
    for result_json in json_paths:
        try:
            payload = _result_payload(_load_json_document(result_json))
            if payload is None:
                continue
            source = _source_path(payload, result_json)
        except (OSError, ValueError):
            # Bad result/source inputs are reported by the normal export pass.
            continue
        if any(_paths_overlap(directory, source.parent) for directory in protected_output_dirs):
            raise UnsafeStatusCropOutputError(
                "output/manifest/errors directories and source image directory must not overlap "
                f"in either direction: output={output_dir}, source_directory={source.parent}"
            )


def export_status_crops(
    *,
    results_dir: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    errors_path: Path | None = None,
    margin_ratio: float = 0.30,
    limit: int | None = None,
    skip_existing: bool = True,
    continue_on_error: bool = False,
) -> list[ManifestRecord]:
    """Export clean status crops from result JSON bundles, preserving paths.

    Selection is SHA-256 ordered by result-relative path.  Consequently a
    limited pilot cohort is repeatable on Windows and Unix regardless of the
    filesystem's enumeration order.
    """
    if margin_ratio < 0:
        raise ValueError("margin_ratio must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)
    output_dir = output_dir.resolve()
    if _paths_overlap(output_dir, results_dir):
        raise UnsafeStatusCropOutputError(
            "output directory and results directory must not overlap in either direction"
        )
    manifest_path = (manifest_path or output_dir / "status_crops_manifest.jsonl").resolve()
    errors_path = (errors_path or output_dir / "status_crops_errors.jsonl").resolve()
    config_path = output_dir / "status_crops_config.json"
    if manifest_path == errors_path:
        raise ValueError("manifest and errors paths must be different")
    for auxiliary_path in (manifest_path, errors_path):
        if _paths_overlap(auxiliary_path.parent, results_dir):
            raise UnsafeStatusCropOutputError(
                "manifest/errors directory and results directory must not overlap in either direction"
            )
    desired_config = _export_config(results_dir, margin_ratio)
    config_prepared = False
    commit_config_after_export = False

    json_paths = sorted(
        results_dir.rglob("*.json"),
        key=lambda path: _selection_key(path.relative_to(results_dir)),
    )
    _preflight_source_directories(
        json_paths=json_paths,
        output_dir=output_dir,
        manifest_path=manifest_path,
        errors_path=errors_path,
    )
    work_items = _cohort_work_items(
        results_dir=results_dir,
        manifest_path=manifest_path,
        json_paths=json_paths,
        limit=limit,
    )
    records: list[ManifestRecord] = []
    errors: list[dict[str, str]] = []
    for result_json, previous_record in work_items:
        if limit is not None and len(records) >= limit:
            break
        relative_json = result_json.relative_to(results_dir)
        try:
            document = _load_json_document(result_json)
            payload = _result_payload(document)
            if payload is None:
                if previous_record is not None:
                    raise ValueError(f"existing cohort member is no longer an inference result: {result_json}")
                continue
            source = _source_path(payload, result_json)
            protected_output_dirs = (output_dir, manifest_path.parent, errors_path.parent)
            if any(_paths_overlap(directory, source.parent) for directory in protected_output_dirs):
                raise UnsafeStatusCropOutputError(
                    "output/manifest/errors directories and source image directory must not overlap "
                    f"in either direction: output={output_dir}, source_directory={source.parent}"
                )
            relative_crop = relative_json.with_suffix(".jpg")
            crop_path = output_dir / relative_crop
            if source in {crop_path.resolve(), manifest_path, errors_path, config_path.resolve()}:
                raise UnsafeStatusCropOutputError(f"refusing to overwrite source image: {source}")
            if not config_prepared:
                commit_config_after_export = _prepare_export_config(
                    config_path=config_path,
                    desired=desired_config,
                    output_dir=output_dir,
                    skip_existing=skip_existing,
                )
                config_prepared = True
            if not (skip_existing and _committed_crop_exists(crop_path, result_json, source)):
                source_rgb = load_upright_rgb(source)
                rectified_rgb = reconstruct_rectified(payload, source_rgb)
                crop_rgb = crop_status_region(rectified_rgb, _status_bbox(payload), margin_ratio)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                save_rgb(crop_path, crop_rgb)
            records.append(
                {
                    "crop": crop_path.resolve().as_posix(),
                    "source": source.as_posix(),
                    "result_json": result_json.resolve().as_posix(),
                    "group_id": _group_id(source, relative_json),
                    "label": None,
                }
            )
        except (UnsafeStatusCropOutputError, StatusCropConfigurationError):
            # Path/configuration failures are batch-safety failures, not bad
            # samples. --continue-on-error must never turn them into writes.
            raise
        except Exception as error:
            if previous_record is not None:
                # Even when an old result temporarily disappears or becomes
                # unreadable, retaining its row prevents reviewed.jsonl from
                # acquiring an orphan on a continue-on-error refresh.
                records.append(previous_record)
            errors.append(
                {
                    "result_json": result_json.resolve().as_posix(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            if not continue_on_error:
                raise

    if not config_prepared:
        commit_config_after_export = _prepare_export_config(
            config_path=config_path,
            desired=desired_config,
            output_dir=output_dir,
            skip_existing=skip_existing,
        )
    _atomic_write_jsonl(manifest_path, records)
    _atomic_write_jsonl(errors_path, errors)
    # Reaching this point means the overwrite pass completed. Per-sample
    # failures recorded under --continue-on-error have no committed crop in the
    # manifest, so they do not make the successfully regenerated crops stale.
    if commit_config_after_export:
        _atomic_write_json(config_path, desired_config)
    return records


def _load_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from None
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            missing = [key for key in ("crop", "source", "result_json", "group_id", "label") if key not in value]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {','.join(missing)}")
            group_id = value["group_id"]
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{path}:{line_number}: invalid group_id")
            label = value["label"]
            if label is not None and label not in STATUS_LABELS:
                raise ValueError(f"{path}:{line_number}: invalid label {label}")
            for key in ("crop", "source", "result_json"):
                if not isinstance(value[key], str) or not value[key]:
                    raise ValueError(f"{path}:{line_number}: invalid {key}")
            record_key = str(value["result_json"]).casefold()
            if record_key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate result_json {value['result_json']}")
            seen.add(record_key)
            records.append(
                {
                    "crop": value["crop"],
                    "source": value["source"],
                    "result_json": value["result_json"],
                    "group_id": group_id,
                    "label": label,
                }
            )
    return records


def prepare_review_records(manifest_path: Path, labels_path: Path) -> list[ManifestRecord]:
    """Merge existing labels into the current manifest for resumable review."""
    manifest_records = _load_manifest(manifest_path)
    existing: dict[str, ManifestRecord] = {}
    if labels_path.is_file():
        for record in _load_manifest(labels_path):
            existing[str(record["result_json"]).casefold()] = record
        manifest_ids = {str(record["result_json"]).casefold() for record in manifest_records}
        orphaned = sorted(set(existing) - manifest_ids)
        if orphaned:
            raise ValueError(f"labels file contains result_json not present in manifest: {orphaned[0]}")
    merged: list[ManifestRecord] = []
    for record in manifest_records:
        reviewed = existing.get(str(record["result_json"]).casefold())
        merged.append({**record, "label": reviewed["label"] if reviewed is not None else record["label"]})
    _atomic_write_jsonl(labels_path, merged)
    return merged


def _review_frame(crop_rgb: np.ndarray, record: ManifestRecord, index: int, total: int) -> np.ndarray:
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    height, width = crop_bgr.shape[:2]
    scale = min(5.0, 1200.0 / max(width, 1), 650.0 / max(height, 1))
    scale = max(1.0, scale)
    display = cv2.resize(crop_bgr, (max(1, int(width * scale)), max(1, int(height * scale))))
    panel_height = 92
    canvas = np.full((display.shape[0] + panel_height, max(display.shape[1], 760), 3), 245, dtype=np.uint8)
    canvas[panel_height : panel_height + display.shape[0], : display.shape[1]] = display
    cv2.putText(
        canvas,
        f"{index + 1}/{total}  {record['group_id']}  label={record['label'] or '-'}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "1 offset(Android)   2 aligned(iOS)   3 absent   4 unclear",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "B/Backspace: back    Q/Esc: save and quit",
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _opencv_key_provider(record: ManifestRecord, crop_rgb: np.ndarray, index: int, total: int) -> int:
    cv2.imshow("Transfer status review", _review_frame(crop_rgb, record, index, total))
    return cv2.waitKeyEx(0)


def _key_action(key: str | int) -> str:
    if isinstance(key, str):
        normalized = key.lower()
    else:
        # Arrow-key values differ across OpenCV/Windows builds; named keys are
        # preferred in tests, while B and Backspace are reliable in the GUI.
        normalized = chr(key & 0xFF).lower() if key >= 0 else ""
    if normalized in _KEY_LABELS:
        return _KEY_LABELS[normalized]
    if normalized in {"q", "\x1b"} or key == 27:
        return "quit"
    if normalized in {"b", "\b"} or key in {8, 2424832}:
        return "back"
    return "unknown"


def review_status_crops(
    *,
    manifest_path: Path,
    labels_path: Path,
    key_provider: KeyProvider | None = None,
) -> list[ManifestRecord]:
    """Keyboard-review status crops and atomically save after every decision."""
    records = prepare_review_records(manifest_path, labels_path)
    if not records:
        return records
    unlabelled = [index for index, record in enumerate(records) if record["label"] is None]
    if not unlabelled:
        return records
    index = unlabelled[0]
    provider = key_provider or _opencv_key_provider
    try:
        while True:
            crop_path = Path(str(records[index]["crop"]))
            crop_rgb = load_upright_rgb(crop_path)
            action = _key_action(provider(records[index], crop_rgb, index, len(records)))
            if action == "quit":
                break
            if action == "back":
                index = max(0, index - 1)
                continue
            if action not in STATUS_LABELS:
                continue
            records[index]["label"] = action
            _atomic_write_jsonl(labels_path, records)
            next_unlabelled = next(
                (candidate for candidate in range(index + 1, len(records)) if records[candidate]["label"] is None),
                None,
            )
            if next_unlabelled is None:
                next_unlabelled = next(
                    (candidate for candidate in range(0, index) if records[candidate]["label"] is None),
                    None,
                )
            if next_unlabelled is None:
                break
            index = next_unlabelled
    finally:
        if key_provider is None:
            cv2.destroyAllWindows()
    return records


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export clean transfer-status crops from inference result JSON")
    parser.add_argument("--results", type=Path, required=True, help="Inference result root containing JSON bundles")
    parser.add_argument("--output", type=Path, required=True, help="Crop output root")
    parser.add_argument("--manifest", type=Path, help="Default: OUTPUT/status_crops_manifest.jsonl")
    parser.add_argument("--errors", type=Path, help="Default: OUTPUT/status_crops_errors.jsonl")
    parser.add_argument("--margin-ratio", type=float, default=0.30)
    parser.add_argument("--limit", type=int, help="Deterministic maximum number of valid crops")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    mode.add_argument("--overwrite", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def export_main(argv: list[str] | None = None) -> None:
    args = build_export_parser().parse_args(argv)
    try:
        records = export_status_crops(
            results_dir=args.results,
            output_dir=args.output,
            manifest_path=args.manifest,
            errors_path=args.errors,
            margin_ratio=args.margin_ratio,
            limit=args.limit,
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Status crop export failed:\n{error}") from None
    print(f"Exported {len(records)} clean status crop(s) to {args.output}")


def build_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review status/check-mark crops with keys 1-4")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="Resumable reviewed JSONL output")
    return parser


def review_main(argv: list[str] | None = None) -> None:
    args = build_review_parser().parse_args(argv)
    try:
        records = review_status_crops(manifest_path=args.manifest, labels_path=args.labels)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Status crop review failed:\n{error}") from None
    counts = {label: sum(record["label"] == label for record in records) for label in STATUS_LABELS}
    remaining = sum(record["label"] is None for record in records)
    print(f"Review saved to {args.labels}: {counts}, remaining={remaining}")
