"""Prepare and score a weakly routed iOS/Android OCR pilot.

The platform label comes from an existing status-bar device result.  It is a
weak routing proxy, not font-family truth.  The evaluation target remains the
frozen Paddle pseudo label, not independently reviewed business truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Final
from uuid import uuid4

SCHEMA_VERSION: Final[int] = 1
PREPARE_KIND: Final[str] = "receipt_font_routed_ocr_pilot_dataset_v1"
SUMMARY_KIND: Final[str] = "receipt_font_routed_ocr_pilot_ab_v1"
RUNTIME_EVIDENCE_KIND: Final[str] = "receipt_font_routed_ocr_runtime_evidence_v1"
DEFAULT_FIELDS: Final[tuple[str, ...]] = (
    "amount",
    "time",
    "payment_method_field",
)
PLATFORMS: Final[tuple[str, ...]] = ("ios", "android")
RANDOM_ROUTES: Final[tuple[str, ...]] = ("random_a", "random_b")
DEFAULT_EVALUATIONS: Final[tuple[str, ...]] = (
    "generic_ios",
    "routed_ios",
    "wrong_ios",
    "generic_android",
    "routed_android",
    "wrong_android",
    "generic_random_a",
    "routed_random_a",
    "generic_random_b",
    "routed_random_b",
)


class RoutedOcrPilotError(ValueError):
    """Raised when routed-pilot evidence is incomplete or inconsistent."""


def _device_rejection(
    payload: Mapping[str, Any],
    *,
    minimum_confidence: float,
) -> tuple[str | None, float | None, str | None, str | None]:
    device = payload.get("device")
    if not isinstance(device, Mapping):
        return None, None, None, "device_metadata_missing_or_invalid"
    platform = device.get("platform")
    confidence_value = device.get("confidence")
    conflict = device.get("device_prior_conflict")
    source_value = device.get("source")
    if (
        not isinstance(platform, str)
        or type(conflict) is not bool
        or not isinstance(source_value, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", source_value)
        or isinstance(confidence_value, bool)
        or not isinstance(confidence_value, (int, float))
    ):
        return None, None, None, "device_metadata_missing_or_invalid"
    confidence = float(confidence_value)
    source = source_value.lower()
    platform = platform.strip().lower()
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return platform, None, source, "device_metadata_missing_or_invalid"
    if platform not in PLATFORMS:
        return platform, confidence, source, "device_platform_unknown"
    if conflict:
        return platform, confidence, source, "device_prior_conflict"
    if confidence < minimum_confidence:
        return platform, confidence, source, "device_confidence_below_threshold"
    return platform, confidence, source, None


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads(text: str, *, location: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise RoutedOcrPilotError(f"invalid JSON at {location}: {error}") from error


def _read_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise RoutedOcrPilotError(f"cannot read {description}: {path}: {error}") from error
    value = _loads(payload, location=path.as_posix())
    if not isinstance(value, Mapping):
        raise RoutedOcrPilotError(f"{description} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_unit(seed: str, token: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _rank(seed: str, token: str) -> str:
    return hashlib.sha256(f"{seed}\0{token}".encode("utf-8")).hexdigest()


def _content_stratum(row: Mapping[str, Any]) -> str:
    """Control label priors so platform experts must win beyond content mix."""

    field = str(row["field"])
    text = str(row["text"]).strip()
    semantic = row.get("semantic_value")
    if field == "time":
        return f"time:{text}"
    if field == "payment_method_field":
        value = semantic if isinstance(semantic, str) and semantic else text
        return f"payment:{value}"
    sign = text[0] if text and text[0] in "-−–－+" else "none"
    currency = "cny" if any(token in text for token in ("¥", "￥", "元")) else "none"
    grouping = "comma" if "," in text else "plain"
    digits = "".join(character for character in text if character.isdigit())
    decimal_match = re.search(r"[.．](\d+)", text)
    decimal_places = len(decimal_match.group(1)) if decimal_match else 0
    integer_digits = max(0, len(digits) - decimal_places)
    return f"amount:{sign}:{currency}:{grouping}:i{integer_digits}:d{decimal_places}"


def _split(group_id: str, *, seed: str, train_ratio: float, validation_ratio: float) -> str:
    bucket = _stable_unit(seed, group_id)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "val"
    return "test"


def _absolute_file(value: object, *, root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RoutedOcrPilotError(f"{description} must be a non-empty path")
    lexical = Path(value)
    if not lexical.is_absolute():
        lexical = root / lexical
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RoutedOcrPilotError(f"cannot resolve {description}: {lexical}: {error}") from error
    if not resolved.is_file():
        raise RoutedOcrPilotError(f"{description} is not a file: {resolved}")
    return resolved


def _relative_image(value: object, *, dataset_root: Path, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise RoutedOcrPilotError(f"{location}: image must be a non-empty string")
    lexical = Path(value)
    if lexical.is_absolute():
        raise RoutedOcrPilotError(f"{location}: image must remain relative to the source dataset root")
    resolved = (dataset_root / lexical).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError:
        raise RoutedOcrPilotError(f"{location}: image escapes the source dataset root") from None
    if not resolved.is_file():
        raise RoutedOcrPilotError(f"{location}: crop is missing: {resolved}")
    return lexical.as_posix()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _field_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        counts[str(row["routing_platform"])][str(row["field"])][str(row["split"])] += 1
    return {
        platform: {
            field: {split: int(counter[split]) for split in ("train", "val", "test")}
            for field, counter in sorted(fields.items())
        }
        for platform, fields in sorted(counts.items())
    }


def _route_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        counts[str(row["random_route"])][str(row["field"])][str(row["split"])] += 1
    return {
        route: {
            field: {split: int(counter[split]) for split in ("train", "val", "test")}
            for field, counter in sorted(fields.items())
        }
        for route, fields in sorted(counts.items())
    }


def _move_validation_groups_for_charset(
    all_rows: Sequence[dict[str, Any]],
    subset: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    train_characters = {
        character
        for row in subset
        if row["split"] == "train"
        for character in str(row["text"])
    }
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        by_group[str(row["group_id"])].append(row)
    moved_groups: set[str] = set()
    for group_id in sorted(by_group):
        group_rows = by_group[group_id]
        if not any(row["split"] == "val" for row in group_rows):
            continue
        characters = {character for row in group_rows for character in str(row["text"])}
        if characters <= train_characters:
            continue
        moved_groups.add(group_id)
        train_characters.update(characters)
    if moved_groups:
        peers: dict[str, set[str]] = defaultdict(set)
        for row in all_rows:
            group_id = str(row["group_id"])
            peer = row.get("matched_peer_group_id")
            if isinstance(peer, str) and peer:
                peers[group_id].add(peer)
                peers[peer].add(group_id)
        pending = list(moved_groups)
        while pending:
            group_id = pending.pop()
            for peer in peers.get(group_id, set()):
                if peer not in moved_groups:
                    moved_groups.add(peer)
                    pending.append(peer)
    if moved_groups:
        for row in all_rows:
            if str(row["group_id"]) in moved_groups:
                row["split"] = "train"
    return len(moved_groups), sum(str(row["group_id"]) in moved_groups for row in all_rows)


def _assert_model_subset_ready(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
    description: str,
) -> None:
    for field in fields:
        for split in ("train", "val", "test"):
            if not any(row["field"] == field and row["split"] == split for row in rows):
                raise RoutedOcrPilotError(f"{description} has no {field}/{split} records")
    train_characters = {character for row in rows if row["split"] == "train" for character in str(row["text"])}
    validation_characters = {
        character for row in rows if row["split"] == "val" for character in str(row["text"])
    }
    if not validation_characters <= train_characters:
        missing = "".join(sorted(validation_characters - train_characters))
        raise RoutedOcrPilotError(f"{description} validation charset is absent from train: {missing!r}")


def prepare_routed_pilot(
    records: Path,
    output: Path,
    *,
    fields: Sequence[str] = DEFAULT_FIELDS,
    minimum_device_confidence: float = 0.90,
    allowed_device_sources: Sequence[str] = ("resolution",),
    split_seed: str = "font-routed-ocr-pilot-v1",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    maximum_train_per_platform_field: int = 6000,
    maximum_validation_per_platform_field: int = 1000,
    maximum_test_per_platform_field: int = 1500,
) -> dict[str, Any]:
    """Build balanced platform and random-control manifests without copying crops."""

    fields = tuple(dict.fromkeys(fields))
    if not fields or any(field not in DEFAULT_FIELDS for field in fields):
        raise RoutedOcrPilotError(f"fields must be a non-empty subset of {DEFAULT_FIELDS}")
    if not math.isfinite(minimum_device_confidence) or not 0.0 <= minimum_device_confidence <= 1.0:
        raise RoutedOcrPilotError("minimum_device_confidence must be in [0,1]")
    allowed_sources = tuple(dict.fromkeys(source.strip().lower() for source in allowed_device_sources if source.strip()))
    if not allowed_sources:
        raise RoutedOcrPilotError("allowed_device_sources cannot be empty")
    if not (0.0 < train_ratio < 1.0 and 0.0 < validation_ratio < 1.0):
        raise RoutedOcrPilotError("train_ratio and validation_ratio must be in (0,1)")
    if train_ratio + validation_ratio >= 1.0:
        raise RoutedOcrPilotError("train_ratio + validation_ratio must be less than 1")
    caps = {
        "train": maximum_train_per_platform_field,
        "val": maximum_validation_per_platform_field,
        "test": maximum_test_per_platform_field,
    }
    if any(isinstance(value, bool) or value <= 0 for value in caps.values()):
        raise RoutedOcrPilotError("all per-platform field caps must be positive")

    records = records.expanduser().resolve(strict=True)
    dataset_root = records.parent
    output = Path(os.path.abspath(os.fspath(output.expanduser())))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite routed OCR pilot output: {output}")
    staging = output.with_name(f".{output.name}.staging-{uuid4().hex}")
    rejection_counts: Counter[str] = Counter()
    target_rows = 0
    result_cache: dict[Path, tuple[str | None, float | None, str | None, str | None]] = {}
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    group_platforms: dict[str, set[str]] = defaultdict(set)
    source_groups: dict[str, set[str]] = defaultdict(set)
    input_sha256 = _sha256(records)

    with records.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            location = f"{records}:{line_number}"
            raw = _loads(line, location=location)
            if not isinstance(raw, Mapping):
                raise RoutedOcrPilotError(f"{location}: record must be an object")
            field = raw.get("field")
            if field not in fields:
                continue
            target_rows += 1
            record_id = raw.get("id")
            group_id = raw.get("group_id")
            text = raw.get("text")
            if not isinstance(record_id, str) or not record_id:
                raise RoutedOcrPilotError(f"{location}: id must be a non-empty string")
            if record_id in seen_ids:
                raise RoutedOcrPilotError(f"{location}: duplicate id {record_id!r}")
            seen_ids.add(record_id)
            if not isinstance(group_id, str) or not group_id:
                raise RoutedOcrPilotError(f"{location}: group_id must be a non-empty string")
            if not isinstance(text, str) or not text:
                raise RoutedOcrPilotError(f"{location}: text must be a non-empty string")
            image = _relative_image(raw.get("image"), dataset_root=dataset_root, location=location)
            image_key = image.casefold()
            if image_key in seen_images:
                raise RoutedOcrPilotError(f"{location}: duplicate crop image {image!r}")
            seen_images.add(image_key)
            result_path = _absolute_file(raw.get("result_json"), root=dataset_root, description=f"{location}: result_json")
            device = result_cache.get(result_path)
            if device is None:
                payload = _read_json(result_path, description="OCR result JSON")
                device = _device_rejection(payload, minimum_confidence=minimum_device_confidence)
                result_cache[result_path] = device
            platform, confidence, device_source, rejection = device
            if rejection is not None:
                rejection_counts[rejection] += 1
                continue
            assert platform is not None and confidence is not None and device_source is not None
            if device_source not in allowed_sources:
                rejection_counts[f"device_source_not_allowed:{device_source}"] += 1
                continue
            row = dict(raw)
            row["image"] = image
            row["split"] = _split(
                group_id,
                seed=split_seed,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
            )
            row["original_split"] = raw.get("split")
            row["routing_platform"] = platform
            row["routing_confidence"] = confidence
            row["routing_device_source"] = device_source
            row["random_route"] = (
                "random_a" if _stable_unit(f"{split_seed}:random-route", group_id) < 0.5 else "random_b"
            )
            row["routing_label_semantics"] = "device_platform_weak_proxy_not_exact_font_identity"
            row["truth_semantics"] = "paddle_teacher_parity_not_independent_human_truth"
            row["matched_content_stratum"] = _content_stratum(row)
            candidates.append(row)
            group_platforms[group_id].add(platform)
            if isinstance(raw.get("source"), str) and raw.get("source"):
                source_groups[str(raw["source"])].add(group_id)

    conflicting_groups = {group for group, platforms in group_platforms.items() if len(platforms) != 1}
    if conflicting_groups:
        kept: list[dict[str, Any]] = []
        for row in candidates:
            if str(row["group_id"]) in conflicting_groups:
                rejection_counts["group_platform_conflict"] += 1
            else:
                kept.append(row)
        candidates = kept

    if any(len(groups) != 1 for groups in source_groups.values()):
        raise RoutedOcrPilotError(
            "source images map to multiple producer groups; group-safe re-splitting is not possible"
        )

    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[
            (
                str(row["routing_platform"]),
                str(row["field"]),
                str(row["split"]),
                str(row["matched_content_stratum"]),
            )
        ].append(row)
    selected: list[dict[str, Any]] = []
    selection_targets: dict[str, dict[str, int]] = defaultdict(dict)
    matched_strata_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for field in fields:
        for split in ("train", "val", "test"):
            strata = sorted(
                {
                    key[3]
                    for key in buckets
                    if key[1] == field
                    and key[2] == split
                    and all(buckets.get((platform, field, split, key[3])) for platform in PLATFORMS)
                }
            )
            pair_units: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
            for stratum in strata:
                ranked = {
                    platform: sorted(
                        buckets[(platform, field, split, stratum)],
                        key=lambda row: _rank(
                            f"{split_seed}:within-stratum:{platform}", str(row["id"])
                        ),
                    )
                    for platform in PLATFORMS
                }
                matched = min(len(ranked["ios"]), len(ranked["android"]))
                for index in range(matched):
                    pair_units.append((stratum, index, ranked["ios"][index], ranked["android"][index]))
            pair_units.sort(
                key=lambda unit: _rank(
                    f"{split_seed}:pair-select:{field}:{split}", f"{unit[0]}:{unit[1]}"
                )
            )
            target = min(len(pair_units), caps[split])
            selection_targets[field][split] = target
            matched_strata_counts[field][split] = len(strata)
            if target <= 0:
                raise RoutedOcrPilotError(
                    f"no matched-content {field}/{split} cross-platform records remain"
                )
            for stratum, index, ios_row, android_row in pair_units[:target]:
                pair_id = _rank(
                    f"{split_seed}:matched-pair:{field}:{split}", f"{stratum}:{index}"
                )
                ios_row["matched_pair_id"] = pair_id
                android_row["matched_pair_id"] = pair_id
                ios_row["matched_peer_group_id"] = str(android_row["group_id"])
                android_row["matched_peer_group_id"] = str(ios_row["group_id"])
                selected.extend((ios_row, android_row))

    selected.sort(key=lambda row: (str(row["split"]), str(row["field"]), str(row["id"])))
    charset_adjustments: dict[str, dict[str, int]] = {}
    subsets_for_charset = {
        "ios": [row for row in selected if row["routing_platform"] == "ios"],
        "android": [row for row in selected if row["routing_platform"] == "android"],
        "random_a": [row for row in selected if row["random_route"] == "random_a"],
        "random_b": [row for row in selected if row["random_route"] == "random_b"],
    }
    for name, subset in subsets_for_charset.items():
        groups, records_moved = _move_validation_groups_for_charset(selected, subset)
        charset_adjustments[name] = {"groups_moved": groups, "records_moved": records_moved}

    manifests = {
        "global": selected,
        "ios": [row for row in selected if row["routing_platform"] == "ios"],
        "android": [row for row in selected if row["routing_platform"] == "android"],
        "random_a": [row for row in selected if row["random_route"] == "random_a"],
        "random_b": [row for row in selected if row["random_route"] == "random_b"],
    }
    for name, subset in manifests.items():
        _assert_model_subset_ready(subset, fields=fields, description=name)

    selected_groups = len({str(row["group_id"]) for row in selected})
    accepted_groups = len({str(row["group_id"]) for row in candidates})
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREPARE_KIND,
        "completed": True,
        "fields": list(fields),
        "classification_target": "ocr_teacher_parity_by_platform_routing_proxy",
        "routing_label_semantics": "device_platform_weak_proxy_not_exact_font_identity",
        "truth_semantics": "paddle_teacher_parity_not_independent_human_truth",
        "exact_font_identity": "not_assessed",
        "business_accuracy": "not_assessed",
        "authenticity": "not_assessed",
        "input": {
            "records": records.as_posix(),
            "records_sha256": input_sha256,
            "dataset_root": dataset_root.as_posix(),
            "target_field_records": target_rows,
        },
        "parameters": {
            "minimum_device_confidence": minimum_device_confidence,
            "allowed_device_sources": list(allowed_sources),
            "split_seed": split_seed,
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": 1.0 - train_ratio - validation_ratio,
            "caps_per_platform_field": caps,
        },
        "route_independence": {
            "status": (
                "primary_resolution_route_does_not_read_time_glyph_pixels"
                if allowed_sources == ("resolution",)
                else "secondary_route_may_include_statusbar_cnn_time_pixel_signal"
            ),
            "time_circularity_controlled": allowed_sources == ("resolution",),
        },
        "coverage": {
            "eligible_records_before_balancing": len(candidates),
            "selected_records": len(selected),
            "eligible_groups_before_balancing": accepted_groups,
            "selected_groups": selected_groups,
            "eligible_record_rate": len(candidates) / target_rows if target_rows else 0.0,
            "rejection_reasons": dict(sorted(rejection_counts.items())),
        },
        "selection_targets_per_platform": {
            field: dict(splits) for field, splits in sorted(selection_targets.items())
        },
        "matched_content_strata": {
            field: dict(splits) for field, splits in sorted(matched_strata_counts.items())
        },
        "counts_by_platform_field_split": _field_counts(selected),
        "counts_by_random_route_field_split": _route_counts(selected),
        "charset_adjustments": charset_adjustments,
        "random_control": {
            "purpose": "separates platform_homogeneity_gain_from_two_model_capacity_gain",
            "assignment_input": "group_id_hash_only",
        },
        "leakage_audit": {
            "group_split_consistent": True,
            "source_to_group_unique": True,
            "crop_paths_unique": True,
            "near_duplicate_pixel_audit": "not_assessed",
        },
        "manifests": {},
        "publication": False,
        "warning": (
            "This pilot can only compare student agreement with held-out Paddle pseudo labels. "
            "It cannot establish human OCR accuracy, exact font identity, or image authenticity."
        ),
    }

    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(mode=0o700)
    published = False
    try:
        manifest_report: dict[str, Any] = {}
        for name, rows in manifests.items():
            path = staging / f"{name}.jsonl"
            records_written = _write_jsonl(path, rows)
            manifest_report[name] = {
                "path": path.name,
                "records": records_written,
                "sha256": _sha256(path),
            }
        report["manifests"] = manifest_report
        _write_json(staging / "prepare.json", report)
        os.rename(staging, output)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return report


def _read_jsonl(path: Path, *, description: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = _loads(line, location=f"{path}:{line_number}")
            if not isinstance(value, Mapping):
                raise RoutedOcrPilotError(f"{description} row must be an object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RoutedOcrPilotError(f"{description} is empty: {path}")
    return rows


def _comparison_index(path: Path, *, description: str) -> dict[str, Mapping[str, Any]]:
    rows = _read_jsonl(path.resolve(strict=True), description=description)
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in result:
            raise RoutedOcrPilotError(f"{description} has invalid or duplicate id")
        result[record_id] = row
    return result


def merge_comparisons(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    """Merge per-field evaluator JSONL files for the paired AB summarizer."""

    output = Path(os.path.abspath(os.fspath(output.expanduser())))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite merged comparisons: {output}")
    if not inputs:
        raise RoutedOcrPilotError("comparison merge requires at least one input")
    combined: dict[str, Mapping[str, Any]] = {}
    identities: list[dict[str, Any]] = []
    for source in inputs:
        source = source.resolve(strict=True)
        index = _comparison_index(source, description="per-field comparisons")
        overlap = set(combined) & set(index)
        if overlap:
            raise RoutedOcrPilotError(f"comparison merge has duplicate id={sorted(overlap)[0]!r}")
        combined.update(index)
        identities.append({"path": source.as_posix(), "sha256": _sha256(source), "records": len(index)})
    rows = sorted(combined.values(), key=lambda row: (str(row.get("field")), str(row.get("id"))))
    output.parent.mkdir(parents=True, exist_ok=True)
    records = _write_jsonl(output, rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_font_routed_ocr_merged_comparisons_v1",
        "output": output.as_posix(),
        "output_sha256": _sha256(output),
        "records": records,
        "inputs": identities,
    }


def collect_runtime_evidence(
    summaries: Sequence[Path],
    output: Path,
    *,
    expected_fields: Sequence[str] = DEFAULT_FIELDS,
    expected_evaluations: Sequence[str] = DEFAULT_EVALUATIONS,
    required_provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    """Bind every evaluator's provider, model, manifest, and comparisons artifact."""

    output = Path(os.path.abspath(os.fspath(output.expanduser())))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite runtime evidence: {output}")
    fields = tuple(dict.fromkeys(expected_fields))
    evaluations = tuple(dict.fromkeys(expected_evaluations))
    if not fields or not evaluations:
        raise RoutedOcrPilotError("runtime evidence requires fields and evaluation names")
    expected = {f"{field}/{evaluation}" for field in fields for evaluation in evaluations}
    entries: dict[str, dict[str, Any]] = {}
    for source in summaries:
        source = source.resolve(strict=True)
        if source.name != "summary.json" or len(source.parents) < 2:
            raise RoutedOcrPilotError(f"evaluation summary path has unsupported layout: {source}")
        field = source.parent.parent.name
        evaluation = source.parent.name
        key = f"{field}/{evaluation}"
        if key not in expected or key in entries:
            raise RoutedOcrPilotError(f"runtime evidence has unexpected or duplicate evaluation: {key}")
        summary = _read_json(source, description=f"evaluation summary {key}")
        if summary.get("providers") != [required_provider]:
            raise RoutedOcrPilotError(
                f"evaluation {key} did not use only {required_provider}: {summary.get('providers')!r}"
            )
        if summary.get("fields") != [field] or summary.get("evaluation_split") != "test":
            raise RoutedOcrPilotError(f"evaluation {key} has wrong field or split")
        if summary.get("kind") != "receipt_ocr_ctc_pseudo_label_evaluation_v1":
            raise RoutedOcrPilotError(f"evaluation {key} does not preserve pseudo-label semantics")
        model_value = summary.get("model")
        records_value = summary.get("records")
        model_sha256 = summary.get("model_sha256")
        if not isinstance(model_value, str) or not isinstance(records_value, str):
            raise RoutedOcrPilotError(f"evaluation {key} has invalid model or records path")
        model = Path(model_value).resolve(strict=True)
        records = Path(records_value).resolve(strict=True)
        if not isinstance(model_sha256, str) or model_sha256 != _sha256(model):
            raise RoutedOcrPilotError(f"evaluation {key} model hash is invalid")
        comparisons = (source.parent / "comparisons.jsonl").resolve(strict=True)
        entries[key] = {
            "field": field,
            "evaluation": evaluation,
            "providers": [required_provider],
            "model": model.as_posix(),
            "model_sha256": model_sha256,
            "records": records.as_posix(),
            "records_sha256": _sha256(records),
            "summary": source.as_posix(),
            "summary_sha256": _sha256(source),
            "comparisons": comparisons.as_posix(),
            "comparisons_sha256": _sha256(comparisons),
        }
    missing = sorted(expected - set(entries))
    if missing or set(entries) != expected:
        raise RoutedOcrPilotError(
            f"runtime evidence coverage is incomplete; first missing={missing[0] if missing else 'unexpected'}"
        )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_EVIDENCE_KIND,
        "completed": True,
        "required_provider": required_provider,
        "fields": list(fields),
        "evaluations": list(evaluations),
        "entries": {key: entries[key] for key in sorted(entries)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, evidence)
    return evidence


def _validate_runtime_evidence(path: Path, *, fields: Sequence[str]) -> Mapping[str, Any]:
    path = path.resolve(strict=True)
    evidence = _read_json(path, description="runtime evidence")
    if (
        evidence.get("kind") != RUNTIME_EVIDENCE_KIND
        or evidence.get("completed") is not True
        or evidence.get("required_provider") != "CPUExecutionProvider"
        or evidence.get("fields") != list(fields)
        or evidence.get("evaluations") != list(DEFAULT_EVALUATIONS)
    ):
        raise RoutedOcrPilotError("runtime evidence contract is incomplete or unsupported")
    raw_entries = evidence.get("entries")
    expected = {f"{field}/{evaluation}" for field in fields for evaluation in DEFAULT_EVALUATIONS}
    if not isinstance(raw_entries, Mapping) or set(raw_entries) != expected:
        raise RoutedOcrPilotError("runtime evidence evaluation coverage differs from the A/B plan")
    verified_hashes: dict[tuple[str, str], str] = {}
    for key in sorted(expected):
        entry = raw_entries.get(key)
        field, evaluation = key.split("/", 1)
        if (
            not isinstance(entry, Mapping)
            or entry.get("field") != field
            or entry.get("evaluation") != evaluation
            or entry.get("providers") != ["CPUExecutionProvider"]
        ):
            raise RoutedOcrPilotError(f"runtime evidence entry is invalid: {key}")
        artifact_paths: dict[str, Path] = {}
        for artifact in ("model", "records", "summary", "comparisons"):
            raw_path = entry.get(artifact)
            raw_sha256 = entry.get(f"{artifact}_sha256")
            if not isinstance(raw_path, str) or not isinstance(raw_sha256, str):
                raise RoutedOcrPilotError(f"runtime evidence {key}/{artifact} binding is invalid")
            artifact_path = Path(raw_path).resolve(strict=True)
            artifact_paths[artifact] = artifact_path
            cache_key = (artifact, artifact_path.as_posix())
            actual_sha256 = verified_hashes.get(cache_key)
            if actual_sha256 is None:
                actual_sha256 = _sha256(artifact_path)
                verified_hashes[cache_key] = actual_sha256
            if not re.fullmatch(r"[0-9a-f]{64}", raw_sha256) or raw_sha256 != actual_sha256:
                raise RoutedOcrPilotError(f"runtime evidence {key}/{artifact} hash differs")
        summary = _read_json(artifact_paths["summary"], description=f"evaluation summary {key}")
        try:
            summary_model = Path(str(summary.get("model"))).resolve(strict=True)
            summary_records = Path(str(summary.get("records"))).resolve(strict=True)
        except OSError as error:
            raise RoutedOcrPilotError(f"runtime evidence {key} summary path is invalid: {error}") from error
        if (
            summary.get("kind") != "receipt_ocr_ctc_pseudo_label_evaluation_v1"
            or summary.get("providers") != ["CPUExecutionProvider"]
            or summary.get("fields") != [field]
            or summary.get("evaluation_split") != "test"
            or summary.get("model_sha256") != entry.get("model_sha256")
            or summary_model != artifact_paths["model"]
            or summary_records != artifact_paths["records"]
            or artifact_paths["comparisons"] != artifact_paths["summary"].parent / "comparisons.jsonl"
        ):
            raise RoutedOcrPilotError(f"runtime evidence {key} summary binding differs")
    return evidence


def _mcnemar_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    statistic = (max(0, abs(wins - losses) - 1) ** 2) / discordant
    return math.erfc(math.sqrt(statistic / 2.0))


def _paired_metrics(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    fields: Sequence[str],
    metric: str,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) ^ set(candidate))
        raise RoutedOcrPilotError(f"paired comparison domains differ; first difference={missing[0]!r}")
    pairs: list[tuple[bool, bool, str]] = []
    for record_id in sorted(baseline):
        left = baseline[record_id]
        right = candidate[record_id]
        if left.get("field") != right.get("field") or left.get("reference_text") != right.get("reference_text"):
            raise RoutedOcrPilotError(f"paired comparison evidence differs for id={record_id}")
        if str(left.get("field")) not in fields:
            continue
        if metric == "semantic_exact" and (
            left.get("semantic_applicable") is not True or right.get("semantic_applicable") is not True
        ):
            continue
        if type(left.get(metric)) is not bool or type(right.get(metric)) is not bool:
            raise RoutedOcrPilotError(f"paired metric {metric!r} is missing for id={record_id}")
        group_id = left.get("group_id")
        if not isinstance(group_id, str) or not group_id or group_id != right.get("group_id"):
            raise RoutedOcrPilotError(f"paired group evidence differs for id={record_id}")
        pairs.append((bool(left[metric]), bool(right[metric]), group_id))
    if not pairs:
        raise RoutedOcrPilotError(f"no paired records remain for fields={fields}, metric={metric}")
    baseline_matches = sum(left for left, _right, _group in pairs)
    candidate_matches = sum(right for _left, right, _group in pairs)
    wins = sum((not left) and right for left, right, _group in pairs)
    losses = sum(left and (not right) for left, right, _group in pairs)
    records = len(pairs)
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for left, right, group_id in pairs:
        grouped[group_id][0] += int(right) - int(left)
        grouped[group_id][1] += 1
    group_ids = sorted(grouped)
    generator = random.Random(
        int.from_bytes(
            hashlib.sha256(f"bootstrap:{','.join(fields)}:{metric}".encode("utf-8")).digest()[:8],
            "big",
        )
    )
    bootstrap_deltas: list[float] = []
    for _iteration in range(2000):
        sampled = [group_ids[generator.randrange(len(group_ids))] for _ in group_ids]
        difference = sum(grouped[group_id][0] for group_id in sampled)
        denominator = sum(grouped[group_id][1] for group_id in sampled)
        bootstrap_deltas.append(difference / denominator)
    bootstrap_deltas.sort()
    lower = bootstrap_deltas[int(0.025 * (len(bootstrap_deltas) - 1))]
    upper = bootstrap_deltas[int(0.975 * (len(bootstrap_deltas) - 1))]
    return {
        "records": records,
        "source_groups": len(group_ids),
        "baseline_matches": baseline_matches,
        "candidate_matches": candidate_matches,
        "baseline_rate": baseline_matches / records,
        "candidate_rate": candidate_matches / records,
        "delta": (candidate_matches - baseline_matches) / records,
        "wins": wins,
        "losses": losses,
        "ties": records - wins - losses,
        "mcnemar_continuity_corrected_p_value": _mcnemar_p_value(wins, losses),
        "group_bootstrap": {
            "iterations": len(bootstrap_deltas),
            "confidence": 0.95,
            "delta_interval": [lower, upper],
        },
    }


def _combine_indexes(indexes: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, Mapping[str, Any]]:
    combined: dict[str, Mapping[str, Any]] = {}
    for index in indexes:
        overlap = set(combined) & set(index)
        if overlap:
            raise RoutedOcrPilotError(f"comparison shards overlap at id={sorted(overlap)[0]!r}")
        combined.update(index)
    return combined


def _difference_of_paired_improvements(
    baseline: Mapping[str, Mapping[str, Any]],
    platform_candidate: Mapping[str, Mapping[str, Any]],
    random_baseline: Mapping[str, Mapping[str, Any]],
    random_candidate: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    metric: str,
) -> dict[str, Any]:
    domains = [set(value) for value in (baseline, platform_candidate, random_baseline, random_candidate)]
    if any(domain != domains[0] for domain in domains[1:]):
        raise RoutedOcrPilotError("platform/random paired improvement domains differ")
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    records = 0
    total_difference = 0
    for record_id in sorted(domains[0]):
        rows = (
            baseline[record_id],
            platform_candidate[record_id],
            random_baseline[record_id],
            random_candidate[record_id],
        )
        if any(row.get("field") != field for row in rows):
            continue
        if len({row.get("reference_text") for row in rows}) != 1:
            raise RoutedOcrPilotError(f"platform/random references differ for id={record_id}")
        if any(type(row.get(metric)) is not bool for row in rows):
            raise RoutedOcrPilotError(f"platform/random metric {metric!r} is missing for id={record_id}")
        if bool(rows[0][metric]) != bool(rows[2][metric]):
            raise RoutedOcrPilotError(f"repeated global baseline prediction differs for id={record_id}")
        group_id = rows[0].get("group_id")
        if not isinstance(group_id, str) or any(row.get("group_id") != group_id for row in rows):
            raise RoutedOcrPilotError(f"platform/random group evidence differs for id={record_id}")
        difference = (
            int(bool(rows[1][metric]))
            - int(bool(rows[0][metric]))
            - int(bool(rows[3][metric]))
            + int(bool(rows[2][metric]))
        )
        grouped[group_id][0] += difference
        grouped[group_id][1] += 1
        total_difference += difference
        records += 1
    if records == 0 or not grouped:
        raise RoutedOcrPilotError(f"no platform/random paired improvements remain for {field}/{metric}")
    group_ids = sorted(grouped)
    generator = random.Random(
        int.from_bytes(hashlib.sha256(f"delta-bootstrap:{field}:{metric}".encode()).digest()[:8], "big")
    )
    deltas: list[float] = []
    for _iteration in range(2000):
        sampled = [group_ids[generator.randrange(len(group_ids))] for _ in group_ids]
        numerator = sum(grouped[group_id][0] for group_id in sampled)
        denominator = sum(grouped[group_id][1] for group_id in sampled)
        deltas.append(numerator / denominator)
    deltas.sort()
    return {
        "records": records,
        "source_groups": len(group_ids),
        "delta": total_difference / records,
        "group_bootstrap": {
            "iterations": len(deltas),
            "confidence": 0.95,
            "delta_interval": [
                deltas[int(0.025 * (len(deltas) - 1))],
                deltas[int(0.975 * (len(deltas) - 1))],
            ],
        },
    }


def summarize_routed_ab(
    *,
    prepare_report: Path,
    runtime_evidence: Path,
    generic_ios: Path,
    routed_ios: Path,
    generic_android: Path,
    routed_android: Path,
    wrong_ios: Path,
    wrong_android: Path,
    generic_random_a: Path,
    routed_random_a: Path,
    generic_random_b: Path,
    routed_random_b: Path,
    output: Path,
) -> dict[str, Any]:
    """Compare platform experts with a same-capacity random two-expert control."""

    output = Path(os.path.abspath(os.fspath(output.expanduser())))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite routed OCR summary: {output}")
    prepare = _read_json(prepare_report.resolve(strict=True), description="prepare report")
    if prepare.get("kind") != PREPARE_KIND or prepare.get("completed") is not True:
        raise RoutedOcrPilotError("prepare report is incomplete or unsupported")
    fields_value = prepare.get("fields")
    if not isinstance(fields_value, list) or not all(isinstance(field, str) for field in fields_value):
        raise RoutedOcrPilotError("prepare report fields are invalid")
    fields = tuple(fields_value)
    runtime = _validate_runtime_evidence(runtime_evidence, fields=fields)

    paths = {
        "generic_ios": generic_ios,
        "routed_ios": routed_ios,
        "generic_android": generic_android,
        "routed_android": routed_android,
        "wrong_ios": wrong_ios,
        "wrong_android": wrong_android,
        "generic_random_a": generic_random_a,
        "routed_random_a": routed_random_a,
        "generic_random_b": generic_random_b,
        "routed_random_b": routed_random_b,
    }
    indexes = {
        name: _comparison_index(path, description=name) for name, path in paths.items()
    }
    runtime_entries = runtime["entries"]
    for name in paths:
        source_indexes = [
            _comparison_index(
                Path(str(runtime_entries[f"{field}/{name}"]["comparisons"])),
                description=f"runtime {field}/{name}",
            )
            for field in fields
        ]
        if _combine_indexes(source_indexes) != indexes[name]:
            raise RoutedOcrPilotError(f"merged comparisons differ from runtime evidence for {name}")
    actual_baseline = _combine_indexes([indexes["generic_ios"], indexes["generic_android"]])
    actual_candidate = _combine_indexes([indexes["routed_ios"], indexes["routed_android"]])
    random_baseline = _combine_indexes([indexes["generic_random_a"], indexes["generic_random_b"]])
    random_candidate = _combine_indexes([indexes["routed_random_a"], indexes["routed_random_b"]])
    if set(actual_baseline) != set(random_baseline):
        raise RoutedOcrPilotError("platform and random-control test domains differ")

    metrics: dict[str, Any] = {}
    supported_fields: list[str] = []
    for field in fields:
        field_result: dict[str, Any] = {}
        for metric in ("raw_exact", "semantic_exact"):
            actual = _paired_metrics(actual_baseline, actual_candidate, fields=(field,), metric=metric)
            random_control = _paired_metrics(random_baseline, random_candidate, fields=(field,), metric=metric)
            by_platform = {
                platform: _paired_metrics(
                    indexes[f"generic_{platform}"],
                    indexes[f"routed_{platform}"],
                    fields=(field,),
                    metric=metric,
                )
                for platform in PLATFORMS
            }
            correct_over_wrong_by_platform = {
                platform: _paired_metrics(
                    indexes[f"wrong_{platform}"],
                    indexes[f"routed_{platform}"],
                    fields=(field,),
                    metric=metric,
                )
                for platform in PLATFORMS
            }
            correct_over_wrong = {
                "records": sum(int(value["records"]) for value in correct_over_wrong_by_platform.values()),
                "weighted_delta": sum(
                    float(value["delta"]) * int(value["records"])
                    for value in correct_over_wrong_by_platform.values()
                ) / sum(int(value["records"]) for value in correct_over_wrong_by_platform.values()),
                "by_platform": correct_over_wrong_by_platform,
            }
            excess_evidence = _difference_of_paired_improvements(
                actual_baseline,
                actual_candidate,
                random_baseline,
                random_candidate,
                field=field,
                metric=metric,
            )
            excess = float(excess_evidence["delta"])
            required_delta = 0.005 if field == "amount" else 0.02
            supported = (
                int(actual["records"]) >= 200
                and all(int(value["records"]) >= 200 for value in by_platform.values())
                and float(actual["delta"]) >= required_delta
                and float(actual["mcnemar_continuity_corrected_p_value"]) < 0.05
                and float(actual["group_bootstrap"]["delta_interval"][0]) > 0.0
                and excess >= 0.01
                and float(excess_evidence["group_bootstrap"]["delta_interval"][0]) > 0.0
                and all(float(value["delta"]) >= -0.005 for value in by_platform.values())
                and float(correct_over_wrong["weighted_delta"]) >= 0.01
                and all(
                    float(value["group_bootstrap"]["delta_interval"][0]) > 0.0
                    for value in correct_over_wrong_by_platform.values()
                )
            )
            field_result[metric] = {
                "platform_route": actual,
                "random_two_expert_control": random_control,
                "platform_excess_delta_over_random": excess,
                "platform_excess_over_random_evidence": excess_evidence,
                "by_platform": by_platform,
                "correct_route_over_wrong_route": correct_over_wrong,
                "preliminary_direction_supported": supported,
            }
        metrics[field] = field_result
        if field_result["semantic_exact"]["preliminary_direction_supported"]:
            supported_fields.append(field)

    amount_delta = float(metrics.get("amount", {}).get("semantic_exact", {}).get("platform_route", {}).get("delta", 0.0))
    amount_by_platform = (
        metrics.get("amount", {}).get("semantic_exact", {}).get("by_platform", {})
    )
    amount_platform_guardrail_met = bool(amount_by_platform) and all(
        float(value["delta"]) >= -0.005 for value in amount_by_platform.values()
    )
    priority_supported = any(field in supported_fields for field in ("time", "payment_method_field"))
    decision = (
        "supported_for_human_truth_followup"
        if priority_supported and amount_delta >= -0.01 and amount_platform_guardrail_met
        else "not_supported_in_pilot"
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "completed": True,
        "decision": decision,
        "supported_fields": supported_fields,
        "amount_platform_guardrail_met": amount_platform_guardrail_met,
        "metrics": metrics,
        "decision_rule": {
            "minimum_paired_records_per_platform": 200,
            "minimum_platform_delta_priority_fields": 0.02,
            "minimum_platform_delta_amount": 0.005,
            "maximum_p_value": 0.05,
            "group_bootstrap_iterations": 2000,
            "minimum_group_bootstrap_delta_lower_bound": 0.0,
            "minimum_excess_delta_over_random_two_expert_control": 0.01,
            "minimum_excess_delta_bootstrap_lower_bound": 0.0,
            "minimum_correct_route_delta_over_wrong_route": 0.01,
            "maximum_per_platform_regression": 0.005,
            "maximum_amount_guardrail_regression": 0.01,
        },
        "evidence": {
            "prepare_report": {
                "path": prepare_report.resolve().as_posix(),
                "sha256": _sha256(prepare_report.resolve()),
            },
            "comparisons": {
                name: {"path": path.resolve().as_posix(), "sha256": _sha256(path.resolve())}
                for name, path in paths.items()
            },
            "runtime_evidence": {
                "path": runtime_evidence.resolve().as_posix(),
                "sha256": _sha256(runtime_evidence.resolve()),
                "required_provider": "CPUExecutionProvider",
            },
        },
        "routing_label_semantics": "device_platform_weak_proxy_not_exact_font_identity",
        "truth_semantics": "paddle_teacher_parity_not_independent_human_truth",
        "exact_font_identity": "not_assessed",
        "business_accuracy": "not_assessed",
        "authenticity": "not_assessed",
        "publication": False,
        "next_gate": (
            "Repeat supported fields across seeds, then evaluate a frozen human-labelled test set."
            if decision == "supported_for_human_truth_followup"
            else "Do not split production OCR by platform from this evidence."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, summary)
    return summary


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not result:
        raise argparse.ArgumentTypeError("value must contain at least one token")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and score a platform-routed OCR pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--records", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--fields", type=_parse_csv, default=DEFAULT_FIELDS)
    prepare.add_argument("--minimum-device-confidence", type=float, default=0.90)
    prepare.add_argument("--allowed-device-sources", type=_parse_csv, default=("resolution",))
    prepare.add_argument("--split-seed", default="font-routed-ocr-pilot-v1")
    prepare.add_argument("--train-ratio", type=float, default=0.70)
    prepare.add_argument("--validation-ratio", type=float, default=0.15)
    prepare.add_argument("--maximum-train-per-platform-field", type=int, default=6000)
    prepare.add_argument("--maximum-validation-per-platform-field", type=int, default=1000)
    prepare.add_argument("--maximum-test-per-platform-field", type=int, default=1500)

    merge = subparsers.add_parser("merge-comparisons")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)

    runtime = subparsers.add_parser("collect-runtime-evidence")
    runtime.add_argument("--summaries", type=Path, nargs="+", required=True)
    runtime.add_argument("--expected-fields", type=_parse_csv, default=DEFAULT_FIELDS)
    runtime.add_argument("--expected-evaluations", type=_parse_csv, default=DEFAULT_EVALUATIONS)
    runtime.add_argument("--required-provider", default="CPUExecutionProvider")
    runtime.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--prepare-report", type=Path, required=True)
    summarize.add_argument("--runtime-evidence", type=Path, required=True)
    for name in (
        "generic-ios", "routed-ios", "generic-android", "routed-android",
        "wrong-ios", "wrong-android",
        "generic-random-a", "routed-random-a", "generic-random-b", "routed-random-b",
    ):
        summarize.add_argument(f"--{name}", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_routed_pilot(
                args.records,
                args.output,
                fields=args.fields,
                minimum_device_confidence=args.minimum_device_confidence,
                allowed_device_sources=args.allowed_device_sources,
                split_seed=args.split_seed,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                maximum_train_per_platform_field=args.maximum_train_per_platform_field,
                maximum_validation_per_platform_field=args.maximum_validation_per_platform_field,
                maximum_test_per_platform_field=args.maximum_test_per_platform_field,
            )
        elif args.command == "merge-comparisons":
            result = merge_comparisons(args.inputs, args.output)
        elif args.command == "collect-runtime-evidence":
            result = collect_runtime_evidence(
                args.summaries,
                args.output,
                expected_fields=args.expected_fields,
                expected_evaluations=args.expected_evaluations,
                required_provider=args.required_provider,
            )
        else:
            result = summarize_routed_ab(
                prepare_report=args.prepare_report,
                runtime_evidence=args.runtime_evidence,
                generic_ios=args.generic_ios,
                routed_ios=args.routed_ios,
                generic_android=args.generic_android,
                routed_android=args.routed_android,
                wrong_ios=args.wrong_ios,
                wrong_android=args.wrong_android,
                generic_random_a=args.generic_random_a,
                routed_random_a=args.routed_random_a,
                generic_random_b=args.generic_random_b,
                routed_random_b=args.routed_random_b,
                output=args.output,
            )
    except (OSError, RoutedOcrPilotError, ValueError) as error:
        raise SystemExit(f"Font-routed OCR pilot failed:\n{error}") from None
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
