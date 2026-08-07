#!/usr/bin/env python3
"""Select a fixed CPU A/B list whose detector has every required core box."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_LABELS = frozenset(
    {"time", "amount", "transfer_status", "recipient_field", "payment_method_field"}
)


class SelectionError(RuntimeError):
    """Raised when completed-run evidence is missing, malformed, or insufficient."""


def _load_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise SelectionError(f"missing {description}: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exception:
        raise SelectionError(
            f"invalid {description} {path}: {exception.msg}"
        ) from exception


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionError(f"invalid {description}: expected one JSON object")
    return value


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def select_complete_inputs(output_directory: Path, *, limit: int) -> tuple[list[Path], int]:
    if limit <= 0:
        raise SelectionError("limit must be positive")
    root = output_directory.resolve(strict=True)
    summary = _mapping(
        _load_json(root / "inference_summary.json", "inference summary"),
        "inference summary",
    )
    if (
        summary.get("requested_device") != "cpu"
        or summary.get("unified_provider") != "cpu"
        or summary.get("errors") != 0
    ):
        raise SelectionError(
            "source run must be a completed CPU detector/device/unified run with zero errors"
        )
    manifest_value = _load_json(root / "inference_manifest.json", "inference manifest")
    if not isinstance(manifest_value, list):
        raise SelectionError("invalid inference manifest: expected one JSON array")

    selected: list[Path] = []
    seen: set[str] = set()
    scanned = 0
    for row_index, row_value in enumerate(manifest_value, start=1):
        row = _mapping(row_value, f"manifest row {row_index}")
        if row.get("status") != "written":
            continue
        scanned += 1
        result_value = row.get("result")
        if not isinstance(result_value, str) or not result_value.strip():
            raise SelectionError(f"manifest row {row_index} has no result path")
        result_path = Path(result_value)
        if not result_path.is_absolute():
            result_path = root / result_path
        result_path = result_path.resolve(strict=True)
        if not _inside(result_path, root):
            raise SelectionError(
                f"manifest row {row_index} result escapes the source run: {result_path}"
            )
        result = _mapping(
            _load_json(result_path, f"result JSON for manifest row {row_index}"),
            f"result JSON for manifest row {row_index}",
        )
        detections = result.get("detections")
        if not isinstance(detections, list):
            raise SelectionError(f"result JSON has no detections array: {result_path}")
        labels = {
            detection.get("label")
            for detection in detections
            if isinstance(detection, Mapping) and isinstance(detection.get("label"), str)
        }
        if not REQUIRED_LABELS.issubset(labels):
            continue
        source_value = row.get("source")
        if not isinstance(source_value, str) or not source_value.strip():
            source_value = result.get("source")
        if not isinstance(source_value, str) or not source_value.strip():
            raise SelectionError(f"manifest row {row_index} has no source path")
        source = Path(source_value).resolve(strict=True)
        if not source.is_file():
            raise SelectionError(f"source image is not a file: {source}")
        identity = os.path.normcase(str(source))
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(source)
        if len(selected) == limit:
            break

    if len(selected) != limit:
        raise SelectionError(
            f"only {len(selected)} complete unique input(s) found after scanning {scanned}; "
            f"required {limit}"
        )
    return selected, scanned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_selection(
    output_directory: Path,
    output_list: Path,
    *,
    limit: int,
) -> tuple[list[Path], int]:
    if output_list.exists():
        raise SelectionError(f"refusing to overwrite output list: {output_list}")
    selected, scanned = select_complete_inputs(output_directory, limit=limit)
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text(
        "".join(f"{path}\n" for path in selected),
        encoding="utf-8",
        newline="\n",
    )
    return selected, scanned


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selected, scanned = write_selection(
            args.output_directory,
            args.output_list,
            limit=args.limit,
        )
    except (OSError, SelectionError) as exception:
        print(f"selection_error={exception}")
        return 2
    print("receipt_mlnet_complete_input_selection_v1")
    print(f"source_run={args.output_directory.resolve()}")
    print(f"scanned_written_results={scanned}")
    print(f"selected_complete_inputs={len(selected)}")
    print(f"output_list={args.output_list.resolve()}")
    print(f"output_list_sha256={_sha256(args.output_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
