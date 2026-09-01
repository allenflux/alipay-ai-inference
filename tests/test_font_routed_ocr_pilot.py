from __future__ import annotations

import json
from pathlib import Path

import pytest

from transfer_receipt_ai.font_routed_ocr_pilot import (
    PREPARE_KIND,
    merge_comparisons,
    prepare_routed_pilot,
    summarize_routed_ab,
)


FIELDS = ("amount", "time", "payment_method_field")


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _build_source_records(root: Path, *, groups_per_platform: int = 200) -> Path:
    rows: list[dict[str, object]] = []
    values = {
        "amount": ("100.00", "100.00"),
        "time": ("12:34", "12:34"),
        "payment_method_field": ("付款方式 余额", "balance"),
    }
    for platform in ("ios", "android"):
        for index in range(groups_per_platform):
            group_id = f"{platform}-group-{index:04d}"
            source = root / "sources" / f"{group_id}.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source")
            result = _write_json(
                root / "results" / f"{group_id}.json",
                {
                    "source": source.as_posix(),
                    "device": {
                        "platform": platform,
                        "confidence": 0.99,
                        "source": "resolution",
                        "device_prior_conflict": False,
                    },
                },
            )
            for field, (text, semantic) in values.items():
                crop = root / "images" / field / f"{group_id}.png"
                crop.parent.mkdir(parents=True, exist_ok=True)
                crop.write_bytes(f"{group_id}:{field}".encode())
                rows.append(
                    {
                        "schema_version": 1,
                        "id": f"{group_id}:{field}",
                        "image": crop.relative_to(root).as_posix(),
                        "field": field,
                        "text": text,
                        "semantic_value": semantic,
                        "group_id": group_id,
                        "split": "train",
                        "source": source.as_posix(),
                        "result_json": result.as_posix(),
                        "crop_sha256": f"{platform}-{index}-{field}",
                        "label_source": "paddle_pseudo",
                    }
                )
    # A status-bar CNN route must stay outside the primary time analysis.
    source = root / "sources" / "cnn-only.png"
    source.write_bytes(b"source")
    result = _write_json(
        root / "results" / "cnn-only.json",
        {
            "source": source.as_posix(),
            "device": {
                "platform": "ios",
                "confidence": 0.99,
                "source": "cnn",
                "device_prior_conflict": False,
            },
        },
    )
    crop = root / "images" / "time" / "cnn-only.png"
    crop.write_bytes(b"cnn")
    rows.append(
        {
            "schema_version": 1,
            "id": "cnn-only:time",
            "image": crop.relative_to(root).as_posix(),
            "field": "time",
            "text": "12:34",
            "semantic_value": "12:34",
            "group_id": "cnn-only",
            "split": "train",
            "source": source.as_posix(),
            "result_json": result.as_posix(),
            "crop_sha256": "cnn-only",
            "label_source": "paddle_pseudo",
        }
    )
    return _write_jsonl(root / "pseudo_labels.jsonl", rows)


def test_prepare_builds_matched_resolution_only_platform_and_random_manifests(tmp_path: Path) -> None:
    records = _build_source_records(tmp_path / "dataset")
    output = tmp_path / "prepared"

    report = prepare_routed_pilot(
        records,
        output,
        maximum_train_per_platform_field=20,
        maximum_validation_per_platform_field=10,
        maximum_test_per_platform_field=10,
    )

    assert report["route_independence"]["time_circularity_controlled"] is True
    assert report["coverage"]["rejection_reasons"]["device_source_not_allowed:cnn"] == 1
    assert report["leakage_audit"]["source_to_group_unique"] is True
    for field in FIELDS:
        for split, expected in (("train", 20), ("val", 10), ("test", 10)):
            assert report["selection_targets_per_platform"][field][split] == expected
            ios = report["counts_by_platform_field_split"]["ios"][field][split]
            android = report["counts_by_platform_field_split"]["android"][field][split]
            assert ios == android
    rows = [json.loads(line) for line in (output / "global.jsonl").read_text().splitlines()]
    assert {row["routing_device_source"] for row in rows} == {"resolution"}
    assert {row["random_route"] for row in rows} == {"random_a", "random_b"}
    assert all(row["truth_semantics"] == "paddle_teacher_parity_not_independent_human_truth" for row in rows)
    with pytest.raises(FileExistsError):
        prepare_routed_pilot(records, output)


def _comparison(
    *,
    platform: str,
    field: str,
    index: int,
    exact: bool,
) -> dict[str, object]:
    record_id = f"{platform}:{field}:{index:03d}"
    return {
        "schema_version": 1,
        "id": record_id,
        "field": field,
        "group_id": f"group:{record_id}",
        "reference_text": f"reference-{field}-{index}",
        "candidate_text": f"candidate-{exact}",
        "raw_exact": exact,
        "semantic_applicable": True,
        "semantic_exact": exact,
    }


def test_summary_requires_platform_gain_beyond_random_and_wrong_route(tmp_path: Path) -> None:
    prepare = _write_json(
        tmp_path / "prepare.json",
        {"schema_version": 1, "kind": PREPARE_KIND, "completed": True, "fields": list(FIELDS)},
    )
    actual: dict[str, list[dict[str, object]]] = {
        name: []
        for name in ("generic_ios", "routed_ios", "wrong_ios", "generic_android", "routed_android", "wrong_android")
    }
    random_rows: dict[str, list[dict[str, object]]] = {
        name: []
        for name in ("generic_random_a", "routed_random_a", "generic_random_b", "routed_random_b")
    }
    for platform in ("ios", "android"):
        for field in FIELDS:
            for index in range(240):
                baseline_exact = index < 144
                routed_exact = index < (192 if field != "amount" else 144)
                wrong_exact = index < 120
                actual[f"generic_{platform}"].append(
                    _comparison(platform=platform, field=field, index=index, exact=baseline_exact)
                )
                actual[f"routed_{platform}"].append(
                    _comparison(platform=platform, field=field, index=index, exact=routed_exact)
                )
                actual[f"wrong_{platform}"].append(
                    _comparison(platform=platform, field=field, index=index, exact=wrong_exact)
                )
                route = "random_a" if index % 2 == 0 else "random_b"
                row = _comparison(platform=platform, field=field, index=index, exact=baseline_exact)
                random_rows[f"generic_{route}"].append(row)
                random_rows[f"routed_{route}"].append(dict(row))
    paths: dict[str, Path] = {}
    for name, rows in {**actual, **random_rows}.items():
        paths[name] = _write_jsonl(tmp_path / f"{name}.jsonl", rows)

    output = tmp_path / "summary.json"
    result = summarize_routed_ab(
        prepare_report=prepare,
        output=output,
        **paths,
    )

    assert result["decision"] == "supported_for_human_truth_followup"
    assert set(result["supported_fields"]) == {"time", "payment_method_field"}
    assert result["metrics"]["amount"]["semantic_exact"]["preliminary_direction_supported"] is False
    assert result["metrics"]["time"]["semantic_exact"]["platform_excess_delta_over_random"] == pytest.approx(0.2)
    assert result["publication"] is False
    assert result["business_accuracy"] == "not_assessed"


def test_merge_comparisons_rejects_duplicate_ids(tmp_path: Path) -> None:
    first = _write_jsonl(
        tmp_path / "first.jsonl",
        [_comparison(platform="ios", field="time", index=1, exact=True)],
    )
    second = _write_jsonl(
        tmp_path / "second.jsonl",
        [_comparison(platform="android", field="time", index=2, exact=False)],
    )
    result = merge_comparisons((first, second), tmp_path / "merged.jsonl")
    assert result["records"] == 2
    with pytest.raises(ValueError, match="duplicate id"):
        merge_comparisons((first, first), tmp_path / "duplicate.jsonl")
