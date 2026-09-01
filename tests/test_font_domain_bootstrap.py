from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

import transfer_receipt_ai.font_domain_bootstrap as bootstrap_module
from transfer_receipt_ai.font_domain_bootstrap import bootstrap_existing_pseudolabels
from transfer_receipt_ai.font_domain_cli import main
from transfer_receipt_ai.font_domain_dataset import load_font_domain_dataset


IOS_DOMAIN = "ios_alipay_font_rendering_proxy_v1"
ANDROID_DOMAIN = "android_alipay_font_rendering_proxy_v1"
BODY_FIELDS = ("amount", "recipient_field", "transfer_status")


def _write_rgb(path: Path, seed: int, *, width: int = 96, height: int = 32) -> None:
    """Write a deterministic, non-blank image unique to ``seed``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # NumPy 2 rejects overflowing a uint16 scalar before the final modulo.
    # Keep fixture generation portable even when document-derived seeds grow.
    yy, xx = np.indices((height, width), dtype=np.uint32)
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = (xx * 7 + yy * 3 + seed * 17) % 251
    pixels[..., 1] = (xx * 2 + yy * 11 + seed * 29) % 253
    pixels[..., 2] = (xx * 13 + yy * 5 + seed * 37) % 255
    Image.fromarray(pixels, mode="RGB").save(path)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _bind_source_metadata(records: list[dict[str, object]]) -> None:
    """Add the optional producer-side source snapshot binding to every row."""

    for record in records:
        source = Path(str(record["source"]))
        source_stat = source.stat()
        record["source_size_bytes"] = source_stat.st_size
        record["source_mtime_ns"] = source_stat.st_mtime_ns


def _add_document(
    root: Path,
    records: list[dict[str, object]],
    *,
    document: str,
    platform: str = "ios",
    confidence: float = 0.99,
    conflict: bool = False,
    device_source: str = "cnn",
    fields: tuple[str, ...] = BODY_FIELDS + ("time",),
    content_key: str | None = None,
    producer_group_id: str | None = None,
) -> tuple[Path, Path]:
    """Create the artifacts already produced by the OCR pseudo-label workflow."""

    ordinal = sum(1 for row in records if row.get("group_id") == f"source-{document}")
    seed_base = sum(ord(character) for character in document) + len(records) * 31
    source = (root / "sources" / f"{document}.png").resolve()
    _write_rgb(source, seed_base, width=180, height=96)
    result = (root / "results" / f"{document}.json").resolve()
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": source.as_posix(),
                "device": {
                    "platform": platform,
                    "confidence": confidence,
                    "source": device_source,
                    "device_prior_conflict": conflict,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    visible_content = content_key or document
    texts = {
        "amount": f"-{visible_content}.83",
        "recipient_field": f"收款人{visible_content}",
        "transfer_status": "支付成功",
        "payment_method_field": "余额",
        "time": "12:34",
        "status_bar": "12:34",
    }
    for index, field in enumerate(fields):
        relative_crop = Path("images") / document / f"{field}.png"
        _write_rgb(root / relative_crop, seed_base + index + ordinal + 1)
        records.append(
            {
                "schema_version": 1,
                "id": f"pseudo-{document}-{field}",
                "image": relative_crop.as_posix(),
                "field": field,
                "text": texts[field],
                "paddle_confidence": 0.995,
                "detector_score": 0.99,
                "source": source.as_posix(),
                "result_json": result.as_posix(),
                "group_id": producer_group_id or f"source-{document}",
                "split": "train",
                "label_source": "paddle_pseudo",
            }
        )
    return source, result


def _normalise_output_paths(value: object, outputs: tuple[Path, ...]) -> object:
    """Reports may truthfully record their absolute publication directory."""

    if isinstance(value, dict):
        return {
            key: _normalise_output_paths(item, outputs)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_normalise_output_paths(item, outputs) for item in value]
    if isinstance(value, str):
        normalised = value
        for output in outputs:
            normalised = normalised.replace(output.resolve().as_posix(), "<OUTPUT>")
        return normalised
    return value


def test_bootstrap_creates_deterministic_loadable_weak_labels_and_excludes_time(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    for content_index in range(8):
        content = f"paired-{content_index}"
        _add_document(
            source_root,
            records,
            document=f"ios-{content_index}",
            platform="ios",
            content_key=content,
        )
        _add_document(
            source_root,
            records,
            document=f"android-{content_index}",
            platform="android",
            content_key=content,
        )
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    output_one = tmp_path / "bootstrap-one"
    output_two = tmp_path / "bootstrap-two"
    report_one = bootstrap_existing_pseudolabels(
        records_path,
        output_one,
        split_seed="deterministic-bootstrap-test",
    )
    report_two = bootstrap_existing_pseudolabels(
        records_path,
        output_two,
        split_seed="deterministic-bootstrap-test",
    )

    assert report_one == json.loads((output_one / "bootstrap.json").read_text(encoding="utf-8"))
    assert report_two == json.loads((output_two / "bootstrap.json").read_text(encoding="utf-8"))
    assert (output_one / "font_domain.auto.jsonl").read_bytes() == (
        output_two / "font_domain.auto.jsonl"
    ).read_bytes()
    assert (output_one / "rejected.jsonl").read_bytes() == (
        output_two / "rejected.jsonl"
    ).read_bytes()
    assert _normalise_output_paths(report_one, (output_one, output_two)) == _normalise_output_paths(
        report_two, (output_one, output_two)
    )

    dataset = load_font_domain_dataset(
        output_one / "font_domain.auto.jsonl",
        require_labels=True,
        minimum_regions=3,
        require_leakage_metadata=True,
    )
    assert len(dataset.documents) == 16
    assert {document.font_domain for document in dataset.documents} == {
        IOS_DOMAIN,
        ANDROID_DOMAIN,
    }
    assert {document.label_source for document in dataset.documents} == {
        "device_platform_proxy_font_rendering_weak_v1.cnn"
    }
    assert all(document.device_prior_domain is None for document in dataset.documents)
    assert all(len(document.regions) == 3 for document in dataset.documents)
    assert all(
        {region.role for region in document.regions} == {
            "amount",
            "recipient",
            "transfer_status",
        }
        for document in dataset.documents
    )
    assert all(region.include_in_consistency for document in dataset.documents for region in document.regions)
    assert all(region.role not in {"time", "status_bar"} for document in dataset.documents for region in document.regions)
    assert report_one["ignored_fields"] == {"time": 16}
    assert report_one["classification_target"] == "font_rendering_domain"
    assert report_one["device_prior_used"] is False
    assert report_one["exact_font_identity"] == "not_assessed"
    assert report_one["font_signal_validation"] == (
        "matched_text_balanced_within_split_before_information_gate"
    )
    assert report_one["post_information_gate_matched_balance"] == "not_assessed"
    matched = report_one["selection"]["matched_text_font_evidence"]
    assert matched["strategy"] == (
        "cross_domain_role_text_exact_match_balanced_within_split_v1"
    )
    assert matched["scope"] == "within_split"
    assert matched["matched_strata"] == 17
    assert matched["included_regions"] == 48
    assert matched["included_regions_by_domain"] == {
        ANDROID_DOMAIN: 24,
        IOS_DOMAIN: 24,
    }
    assert report_one["selection"]["selected_by_device_label_source"] == {
        "cnn": 16
    }
    assert all(
        len(set(by_domain.values())) == 1
        for by_domain in matched["included_regions_by_split_and_domain"].values()
    )

    source_splits: dict[str, set[str]] = {}
    content_splits: dict[str, set[str]] = {}
    content_documents: dict[str, int] = {}
    for document in dataset.documents:
        source_splits.setdefault(document.source_group_id, set()).add(document.split)
        assert document.content_group_id is not None
        content_splits.setdefault(document.content_group_id, set()).add(document.split)
        content_documents[document.content_group_id] = content_documents.get(document.content_group_id, 0) + 1
    assert all(len(splits) == 1 for splits in source_splits.values())
    assert all(len(splits) == 1 for splits in content_splits.values())
    assert set(content_documents.values()) == {2}

    assert report_one["publication"] is False
    assert report_one["evaluation"] == "not_assessed"
    assert report_one["authenticity"] == "not_assessed"


def test_bootstrap_balances_same_text_within_each_split_when_other_text_differs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    for index in range(18):
        _add_document(
            source_root,
            records,
            document=f"ios-unique-{index}",
            platform="ios",
        )
        _add_document(
            source_root,
            records,
            document=f"android-unique-{index}",
            platform="android",
        )
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)
    output = tmp_path / "bootstrap"

    report = bootstrap_existing_pseudolabels(
        records_path,
        output,
        split_seed="split-scoped-matched-text-regression",
    )
    dataset = load_font_domain_dataset(
        output / "font_domain.auto.jsonl",
        require_labels=True,
        minimum_regions=3,
        require_leakage_metadata=True,
    )

    assert len(dataset.documents) == 36
    assert all(
        {region.role for region in document.included_regions} == {"transfer_status"}
        for document in dataset.documents
    )
    matched = report["selection"]["matched_text_font_evidence"]
    assert set(matched["included_regions_by_split"]) == {
        "train",
        "calibration",
        "test",
    }
    assert all(
        by_domain[IOS_DOMAIN] == by_domain[ANDROID_DOMAIN]
        for by_domain in matched["included_regions_by_split_and_domain"].values()
    )
    strict_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for document in dataset.documents:
        assert document.font_domain is not None
        for region in document.included_regions:
            assert region.text is not None
            key = (document.split, region.role, region.text)
            by_domain = strict_counts.setdefault(key, {})
            by_domain[document.font_domain] = by_domain.get(document.font_domain, 0) + 1
    assert all(
        set(by_domain) == {IOS_DOMAIN, ANDROID_DOMAIN}
        and by_domain[IOS_DOMAIN] == by_domain[ANDROID_DOMAIN]
        for by_domain in strict_counts.values()
    )
    assert report["readiness"]["missing_matched_text_splits"] == []


def test_font_matching_preserves_case_and_fullwidth_glyph_identity(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    fields = BODY_FIELDS + ("payment_method_field",)
    _add_document(
        source_root,
        records,
        document="ios-glyphs",
        platform="ios",
        fields=fields,
    )
    _add_document(
        source_root,
        records,
        document="android-glyphs",
        platform="android",
        fields=fields,
    )
    for row in records:
        ios = row["group_id"] == "source-ios-glyphs"
        if row["field"] == "amount":
            row["text"] = "-1" if ios else "－１"
        elif row["field"] == "transfer_status":
            row["text"] = "PAY" if ios else "pay"
        elif row["field"] == "payment_method_field":
            row["text"] = "余额"
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)
    output = tmp_path / "bootstrap"

    bootstrap_existing_pseudolabels(records_path, output)
    dataset = load_font_domain_dataset(
        output / "font_domain.auto.jsonl",
        require_labels=True,
        minimum_regions=3,
        require_leakage_metadata=True,
    )

    assert len(dataset.documents) == 2
    assert all(
        {region.role for region in document.included_regions} == {"payment_method"}
        for document in dataset.documents
    )


def test_font_pairing_never_fans_out_one_source_content_component(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    for index in range(10):
        _add_document(
            source_root,
            records,
            document=f"ios-shared-{index}",
            platform="ios",
            content_key="shared-content",
        )
    _add_document(
        source_root,
        records,
        document="android-shared",
        platform="android",
        content_key="shared-content",
    )
    for index in range(9):
        _add_document(
            source_root,
            records,
            document=f"android-unique-{index}",
            platform="android",
        )
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)
    output = tmp_path / "bootstrap"

    report = bootstrap_existing_pseudolabels(
        records_path,
        output,
        split_seed="component-fanout-regression",
    )

    pairing = report["selection"]["font_split_pairing"]
    assert pairing["strategy"] == (
        "disjoint_source_content_component_font_pair_v1"
    )
    assert pairing["source_content_components"] == 10
    assert pairing["controlled_components"] == 1
    assert pairing["uncontrolled_components"] == 9
    assert pairing["pairs_by_role"] == {}


def test_bootstrap_keeps_repeated_capture_group_in_one_source_group_and_split(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    for platform in ("ios", "android"):
        for index in range(3):
            _add_document(
                source_root,
                records,
                document=f"{platform}-capture-{index}",
                platform=platform,
                content_key=f"different-ocr-text-{platform}-{index}",
                producer_group_id=f"same-{platform}-receipt",
            )
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)
    output = tmp_path / "bootstrap"

    report = bootstrap_existing_pseudolabels(
        records_path,
        output,
        split_seed="producer-group-regression",
    )
    dataset = load_font_domain_dataset(
        output / "font_domain.auto.jsonl",
        require_labels=True,
        minimum_regions=3,
        require_leakage_metadata=True,
    )

    document_by_id = {document.document_id: document for document in dataset.documents}
    producer_source_groups: dict[str, set[str]] = {}
    producer_splits: dict[str, set[str]] = {}
    for row in report["provenance"]:
        document = document_by_id[str(row["document_id"])]
        producer = str(row["producer_group_id"])
        producer_source_groups.setdefault(producer, set()).add(document.source_group_id)
        producer_splits.setdefault(producer, set()).add(document.split)

    assert len(dataset.documents) == 6
    assert set(producer_source_groups) == {"same-ios-receipt", "same-android-receipt"}
    assert all(len(groups) == 1 for groups in producer_source_groups.values())
    assert all(len(splits) == 1 for splits in producer_splits.values())
    assert report["selection"]["selected_source_components"] == 2


def test_bootstrap_rejects_uncertain_low_confidence_and_conflicting_device_priors(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    _add_document(source_root, records, document="uncertain", platform="uncertain")
    _add_document(source_root, records, document="low", platform="ios", confidence=0.899)
    _add_document(source_root, records, document="conflict", platform="android", conflict=True)
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    output = tmp_path / "bootstrap"
    report = bootstrap_existing_pseudolabels(records_path, output, minimum_device_confidence=0.9)
    rejected = _jsonl(output / "rejected.jsonl")

    assert report["counts"]["accepted_documents"] == 0
    assert report["counts"]["rejected_documents"] == 3
    assert {row["reason"] for row in rejected} == {
        "device_platform_unknown",
        "device_confidence_below_threshold",
        "device_prior_conflict",
    }
    assert report["publication"] is False
    assert report["evaluation"] == "not_assessed"
    assert report["authenticity"] == "not_assessed"


def test_bootstrap_requires_three_distinct_body_roles_after_ignored_fields(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    _add_document(
        source_root,
        records,
        document="too-few",
        fields=("amount", "recipient_field", "time", "status_bar"),
    )
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    output = tmp_path / "bootstrap"
    report = bootstrap_existing_pseudolabels(records_path, output, minimum_regions=3)

    assert report["counts"]["accepted_documents"] == 0
    assert [row["reason"] for row in _jsonl(output / "rejected.jsonl")] == [
        "insufficient_body_regions"
    ]
    assert report["ignored_fields"] == {"status_bar": 1, "time": 1}


def test_bootstrap_applies_cap_before_expensive_source_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    for index in range(6):
        _add_document(source_root, records, document=f"ios-{index}", platform="ios")
    for index in range(3):
        _add_document(source_root, records, document=f"android-{index}", platform="android")
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    selected_source_decodes: list[Path] = []
    original_decode = bootstrap_module._decode_rgb

    def tracking_decode(
        path: Path,
        data: bytes,
        *,
        maximum_pixels: int,
        description: str,
    ) -> np.ndarray:
        if description == "selected source image":
            selected_source_decodes.append(path)
        return original_decode(
            path,
            data,
            maximum_pixels=maximum_pixels,
            description=description,
        )

    monkeypatch.setattr(bootstrap_module, "_decode_rgb", tracking_decode)

    output = tmp_path / "bootstrap"
    report = bootstrap_existing_pseudolabels(
        records_path,
        output,
        maximum_documents_per_domain=2,
        split_seed="balanced-cap-test",
    )
    dataset = load_font_domain_dataset(
        output / "font_domain.auto.jsonl",
        require_labels=True,
        minimum_regions=3,
        require_leakage_metadata=True,
    )

    domain_counts: dict[str, int] = {}
    for document in dataset.documents:
        assert document.font_domain is not None
        domain_counts[document.font_domain] = domain_counts.get(document.font_domain, 0) + 1
    assert domain_counts == {ANDROID_DOMAIN: 2, IOS_DOMAIN: 2}
    assert len(selected_source_decodes) == 4
    assert report["counts"]["accepted_documents"] == 4
    assert report["rejection_reasons"]["domain_balancing_cap"] == 5


def test_bootstrap_caches_repeated_paths_and_retains_only_best_role_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    _add_document(source_root, records, document="ios", platform="ios")
    _add_document(source_root, records, document="android", platform="android")
    base_rows = [row for row in records if row["field"] == "amount"]
    for duplicate_index in range(10):
        for base in base_rows:
            duplicate = dict(base)
            duplicate["id"] = f"discarded-{base['id']}-{duplicate_index}"
            duplicate["image"] = f"images/discarded-{base['id']}-{duplicate_index}.png"
            duplicate["text"] = f"discarded-{duplicate_index}"
            duplicate["detector_score"] = 0.10
            records.append(duplicate)
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    resolutions: list[str] = []
    original_absolute_file = bootstrap_module._absolute_file

    def tracking_absolute_file(value: object, *, description: str) -> Path:
        resolutions.append(description.rsplit(": ", 1)[-1])
        return original_absolute_file(value, description=description)

    monkeypatch.setattr(bootstrap_module, "_absolute_file", tracking_absolute_file)
    output = tmp_path / "bootstrap"
    report = bootstrap_existing_pseudolabels(records_path, output)

    assert resolutions.count("source") == 2
    assert resolutions.count("result_json") == 2
    assert report["selection"]["duplicate_role_rows_discarded"] == 20
    manifest = _jsonl(output / "font_domain.auto.jsonl")
    assert all(
        next(region for region in document["regions"] if region["role"] == "amount")[
            "text"
        ].startswith("-")
        for document in manifest
    )


def test_bootstrap_refuses_to_clobber_completed_evidence(tmp_path: Path) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    _add_document(source_root, records, document="ios", platform="ios")
    _add_document(source_root, records, document="android", platform="android")
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)
    output = tmp_path / "bootstrap"

    bootstrap_existing_pseudolabels(records_path, output)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises((FileExistsError, ValueError), match="overwrite|output|exist|empty"):
        bootstrap_existing_pseudolabels(records_path, output)
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "failure",
    ("crop_escape", "missing_selected_crop", "source_result_mismatch"),
)
def test_bootstrap_fails_closed_before_output_on_unbound_input_paths(
    tmp_path: Path,
    failure: str,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    source, result = _add_document(source_root, records, document="unsafe", platform="ios")
    if failure == "crop_escape":
        escaped = tmp_path / "escaped.png"
        _write_rgb(escaped, 999)
        records[0]["image"] = "../escaped.png"
    elif failure == "missing_selected_crop":
        (source_root / str(records[0]["image"])).unlink()
        _add_document(
            source_root,
            records,
            document="matched-android",
            platform="android",
        )
    else:
        other_source = (source_root / "sources" / "other.png").resolve()
        _write_rgb(other_source, 1001, width=180, height=96)
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["source"] = other_source.as_posix()
        result.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        assert records[0]["source"] == source.as_posix()

    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)
    output = tmp_path / "bootstrap"
    with pytest.raises(ValueError):
        bootstrap_existing_pseudolabels(records_path, output)
    assert not output.exists()


def test_bootstrap_streams_manifest_and_hashes_exact_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    _add_document(source_root, records, document="ios", platform="ios")
    _add_document(source_root, records, document="android", platform="android")
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    # Exercise first-line BOM handling and skipped blank lines while retaining
    # an exact byte-level provenance hash for the streamed file.
    canonical_bytes = records_path.read_bytes()
    first_line, remaining_lines = canonical_bytes.split(b"\n", 1)
    manifest_bytes = b"\xef\xbb\xbf" + first_line + b"\n\n" + remaining_lines
    records_path.write_bytes(manifest_bytes)

    snapshot_descriptions: list[str] = []
    original_snapshot = bootstrap_module._read_snapshot

    def tracking_snapshot(
        path: Path,
        *,
        maximum_bytes: int,
        description: str,
        **kwargs: object,
    ) -> bytes:
        snapshot_descriptions.append(description)
        return original_snapshot(
            path,
            maximum_bytes=maximum_bytes,
            description=description,
            **kwargs,
        )

    monkeypatch.setattr(bootstrap_module, "_read_snapshot", tracking_snapshot)

    output = tmp_path / "bootstrap"
    report = bootstrap_existing_pseudolabels(records_path, output)

    assert report["counts"]["input_rows"] == len(records)
    assert report["input"]["rows"] == len(records)
    assert report["input"]["records_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert "pseudo-label manifest" not in snapshot_descriptions
    assert (output / "bootstrap.json").is_file()


def test_bootstrap_rejects_source_changed_after_scan_before_final_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    raced_source, _ = _add_document(
        source_root,
        records,
        document="ios",
        platform="ios",
    )
    _add_document(source_root, records, document="android", platform="android")
    _bind_source_metadata(records)
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    original_snapshot = bootstrap_module._read_snapshot
    mutated = False

    def racing_snapshot(
        path: Path,
        *,
        maximum_bytes: int,
        description: str,
        **kwargs: object,
    ) -> bytes:
        nonlocal mutated
        if description == "selected source image" and path == raced_source and not mutated:
            path.write_bytes(path.read_bytes() + b"source-race")
            mutated = True
        return original_snapshot(
            path,
            maximum_bytes=maximum_bytes,
            description=description,
            **kwargs,
        )

    monkeypatch.setattr(bootstrap_module, "_read_snapshot", racing_snapshot)

    output = tmp_path / "bootstrap"
    with pytest.raises(ValueError):
        bootstrap_existing_pseudolabels(records_path, output)

    assert mutated
    assert not output.exists()


def test_bootstrap_rejects_result_json_changed_after_initial_scan_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    _, raced_result = _add_document(
        source_root,
        records,
        document="ios",
        platform="ios",
    )
    _add_document(source_root, records, document="android", platform="android")
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)

    original_snapshot = bootstrap_module._read_snapshot
    mutated = False

    def racing_snapshot(
        path: Path,
        *,
        maximum_bytes: int,
        description: str,
        **kwargs: object,
    ) -> bytes:
        nonlocal mutated
        data = original_snapshot(
            path,
            maximum_bytes=maximum_bytes,
            description=description,
            **kwargs,
        )
        if path == raced_result and not mutated:
            payload = json.loads(data)
            payload["race_marker"] = "changed-after-scan"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            mutated = True
        return data

    monkeypatch.setattr(bootstrap_module, "_read_snapshot", racing_snapshot)

    output = tmp_path / "bootstrap"
    with pytest.raises(ValueError):
        bootstrap_existing_pseudolabels(records_path, output)

    assert mutated
    assert not output.exists()


def test_bootstrap_existing_cli_publishes_the_zero_touch_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "existing-pseudo"
    records: list[dict[str, object]] = []
    _add_document(source_root, records, document="ios", platform="ios")
    _add_document(source_root, records, document="android", platform="android")
    records_path = _write_jsonl(source_root / "pseudo_labels.jsonl", records)
    output = tmp_path / "bootstrap"

    code = main(
        [
            "bootstrap-existing",
            "--records",
            str(records_path),
            "--output",
            str(output),
            "--maximum-documents-per-domain",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result["counts"]["accepted_documents"] == 2
    assert result["classification_target"] == "font_rendering_domain"
    assert result["label_provenance"] == (
        "device_platform_proxy_font_rendering_weak_v1"
    )
    assert (output / "font_domain.auto.jsonl").is_file()
    assert (output / "bootstrap.json").is_file()
    dataset = load_font_domain_dataset(
        output / "font_domain.auto.jsonl",
        require_labels=True,
        minimum_regions=3,
        require_leakage_metadata=True,
    )
    assert all(
        {region.role for region in document.included_regions} == {"transfer_status"}
        for document in dataset.documents
    )
    assert result["selection"]["matched_text_font_evidence"]["included_regions"] == 2
