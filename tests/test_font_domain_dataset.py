from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from transfer_receipt_ai.font_domain_dataset import (
    DOCUMENT_KIND,
    audit_near_duplicate_splits,
    classifier_records,
    load_font_domain_dataset,
    write_classifier_manifest,
)
from transfer_receipt_ai.ocr_lite_classifier import load_records as load_classifier_records


def _write_image(path: Path, seed: int, *, pixels: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pixels is None:
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, size=(28, 72, 3), dtype=np.uint8)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _region(image: str, region_id: str = "amount", role: str = "amount", **extra: object) -> dict[str, object]:
    return {"id": region_id, "role": role, "image": image, **extra}


def _document(
    document_id: str,
    *,
    split: str = "train",
    domain: str | None = "ios_alipay",
    source_group: str | None = None,
    content_group: str | None = None,
    regions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": DOCUMENT_KIND,
        "id": document_id,
        "source_group_id": source_group or f"source-{document_id}",
        "split": split,
        "font_domain": domain,
        "regions": regions or [_region(f"images/{document_id}.png")],
    }
    if domain is not None:
        record["label_source"] = "verified_device_capture"
    if content_group is not None:
        record["content_group_id"] = content_group
    return record


def _write_manifest(root: Path, records: list[dict[str, object]]) -> Path:
    path = root / "font_domain.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _prepare_images(root: Path, records: list[dict[str, object]]) -> None:
    seed = 1
    for record in records:
        for region in record["regions"]:  # type: ignore[index]
            _write_image(root / str(region["image"]), seed)
            seed += 1


def test_load_and_export_classifier_manifest_is_sorted_bound_and_no_clobber(tmp_path: Path) -> None:
    records = [
        _document(
            "z-train",
            regions=[
                _region("images/train-status.png", "status", "status_bar"),
                _region("images/train-amount.png", "amount", "amount"),
            ],
        ),
        _document(
            "a-calibration",
            split="calibration",
            domain="android_alipay",
            regions=[_region("images/calibration.png", "recipient", "recipient")],
        ),
    ]
    _prepare_images(tmp_path, records)
    dataset = load_font_domain_dataset(_write_manifest(tmp_path, records), require_labels=True)

    assert [document.document_id for document in dataset.documents] == ["a-calibration", "z-train"]
    assert dataset.documents[1].regions[0].include_in_consistency is False
    summary = dataset.summary()
    assert summary["documents"] == 2
    assert summary["regions"] == 3
    assert summary["included_regions"] == 2
    assert summary["authenticity"] == "not_assessed"
    assert summary["dataset_snapshot_sha256"] == dataset.snapshot_sha256
    assert len(dataset.snapshot_sha256) == 64

    flattened = classifier_records(dataset)
    assert [row["id"] for row in flattened] == ["a-calibration:recipient", "z-train:amount"]
    assert flattened[0]["split"] == "val"
    assert flattened[1]["split"] == "train"
    assert all(row["field"] == "font_domain" for row in flattened)
    assert all(len(str(row["raw_sha256"])) == 64 for row in flattened)
    assert all(len(str(row["pixel_sha256"])) == 64 for row in flattened)

    output = tmp_path / "classifier-published.jsonl"
    publication = write_classifier_manifest(dataset, output)
    before = output.read_bytes()
    published_rows = [json.loads(line) for line in before.decode("utf-8").splitlines()]
    assert published_rows == flattened
    assert publication["records"] == 2
    assert publication["size_bytes"] == len(before)
    with pytest.raises(FileExistsError, match="refusing to overwrite evidence"):
        write_classifier_manifest(dataset, output)
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="written beside the source manifest"):
        write_classifier_manifest(dataset, tmp_path / "published" / "classifier.jsonl")

    consumed = load_classifier_records(output)
    assert len(consumed) == 2
    bound_image = Path(consumed[0]["image_path"])
    _write_image(bound_image, 777)
    with pytest.raises(ValueError, match="raw_sha256 differs from classifier image bytes"):
        load_classifier_records(output)
    rebound = load_font_domain_dataset(dataset.manifest_path, require_labels=True)
    assert rebound.snapshot_sha256 != dataset.snapshot_sha256


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/absolute/image.png",
        "../escape.png",
        "images/../escape.png",
        "images\\windows.png",
        "C:/windows.png",
        "images//empty-component.png",
    ),
)
def test_unsafe_image_paths_are_rejected_before_decode(tmp_path: Path, unsafe_path: str) -> None:
    record = _document("unsafe", regions=[_region(unsafe_path)])
    manifest = _write_manifest(tmp_path, [record])

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        load_font_domain_dataset(manifest, require_labels=True)


def test_symlink_image_cannot_escape_dataset_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    _write_image(outside, 41)
    link = tmp_path / "images" / "linked.png"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    manifest = _write_manifest(tmp_path, [_document("symlink", regions=[_region("images/linked.png")])])

    with pytest.raises(ValueError, match="escapes the dataset root"):
        load_font_domain_dataset(manifest, require_labels=True)


@pytest.mark.parametrize(
    ("second_overrides", "message"),
    [
        ({"split": "test", "source_group": "shared-source"}, "crosses split or font domain"),
        (
            {"split": "train", "domain": "android_alipay", "source_group": "shared-source"},
            "crosses split or font domain",
        ),
        ({"split": "test", "content_group": "shared-content"}, "content_group_id.*crosses"),
    ],
)
def test_source_and_content_groups_cannot_leak_across_bindings(
    tmp_path: Path,
    second_overrides: dict[str, object],
    message: str,
) -> None:
    if "content_group" in second_overrides:
        first = _document("first", content_group="shared-content")
    else:
        first = _document("first", source_group="shared-source")
    second = _document("second", **second_overrides)  # type: ignore[arg-type]
    records = [first, second]
    _prepare_images(tmp_path, records)
    manifest = _write_manifest(tmp_path, records)

    with pytest.raises(ValueError, match=message):
        load_font_domain_dataset(manifest, require_labels=True)


def test_repeated_capture_group_may_bind_multiple_source_image_hashes(
    tmp_path: Path,
) -> None:
    records = [
        _document("capture-one", source_group="same-receipt"),
        _document("capture-two", source_group="same-receipt"),
    ]
    records[0]["source_image_sha256"] = "1" * 64
    records[1]["source_image_sha256"] = "2" * 64
    _prepare_images(tmp_path, records)

    dataset = load_font_domain_dataset(
        _write_manifest(tmp_path, records),
        require_labels=True,
        require_leakage_metadata=False,
    )

    assert len(dataset.documents) == 2
    assert {document.source_group_id for document in dataset.documents} == {
        "same-receipt"
    }


def test_exact_decoded_pixels_cannot_cross_splits_even_with_different_files(tmp_path: Path) -> None:
    pixels = np.arange(28 * 72 * 3, dtype=np.uint16).reshape(28, 72, 3).astype(np.uint8)
    records = [
        _document("train", regions=[_region("images/train.png")]),
        _document(
            "test",
            split="test",
            domain="android_alipay",
            regions=[_region("images/test.bmp")],
        ),
    ]
    _write_image(tmp_path / "images/train.png", 0, pixels=pixels)
    _write_image(tmp_path / "images/test.bmp", 0, pixels=pixels)
    manifest = _write_manifest(tmp_path, records)

    with pytest.raises(ValueError, match="exact decoded-pixel duplicate crosses train/test splits"):
        load_font_domain_dataset(manifest, require_labels=True)


def test_perceptual_near_duplicate_audit_catches_cross_split_derivative(tmp_path: Path) -> None:
    base = np.full((48, 144, 3), 255, dtype=np.uint8)
    base[10:36, 12:28] = 0
    base[10:36, 44:60] = 0
    derivative = base.copy()
    derivative[20, 80] = 220
    records = [
        _document("train", regions=[_region("images/train.png")]),
        _document(
            "test",
            split="test",
            domain="android_alipay",
            regions=[_region("images/test.png")],
        ),
    ]
    _write_image(tmp_path / "images/train.png", 0, pixels=base)
    _write_image(tmp_path / "images/test.png", 0, pixels=derivative)
    dataset = load_font_domain_dataset(_write_manifest(tmp_path, records), require_labels=True)

    # This deliberately sparse fixture moves eight pHash bits after a one-pixel
    # edit; the audit threshold is configurable and the default is eight.
    with pytest.raises(ValueError, match="perceptual near-duplicate crosses splits"):
        audit_near_duplicate_splits(dataset, maximum_hamming_distance=8)


def test_duplicate_document_and_region_ids_are_rejected(tmp_path: Path) -> None:
    duplicate_documents = [
        _document("same", regions=[_region("images/one.png")]),
        _document("same", regions=[_region("images/two.png")]),
    ]
    _prepare_images(tmp_path, duplicate_documents)
    with pytest.raises(ValueError, match="duplicate document id"):
        load_font_domain_dataset(_write_manifest(tmp_path, duplicate_documents), require_labels=True)

    duplicate_regions = [
        _document(
            "regions",
            regions=[
                _region("images/three.png", "same", "amount"),
                _region("images/four.png", "same", "time"),
            ],
        )
    ]
    _prepare_images(tmp_path, duplicate_regions)
    with pytest.raises(ValueError, match="duplicate region id"):
        load_font_domain_dataset(_write_manifest(tmp_path, duplicate_regions), require_labels=True)


def test_reused_region_path_and_unknown_training_domain_are_rejected(tmp_path: Path) -> None:
    reused = [
        _document("first", regions=[_region("images/shared.png")]),
        _document("second", regions=[_region("images/shared.png")]),
    ]
    _write_image(tmp_path / "images/shared.png", 99)
    with pytest.raises(ValueError, match="region image is reused"):
        load_font_domain_dataset(_write_manifest(tmp_path, reused), require_labels=True)

    unknown = [_document("unknown", domain="unknown", regions=[_region("images/unknown.png")])]
    _prepare_images(tmp_path, unknown)
    with pytest.raises(ValueError, match="UNKNOWN is inference-only"):
        load_font_domain_dataset(_write_manifest(tmp_path, unknown), require_labels=True)


def test_inference_records_cannot_smuggle_labels_or_label_source(tmp_path: Path) -> None:
    labeled = [_document("labeled", split="inference", domain="ios_alipay")]
    _prepare_images(tmp_path, labeled)
    with pytest.raises(ValueError, match="cannot contain font_domain"):
        load_font_domain_dataset(_write_manifest(tmp_path, labeled), require_labels=False)

    unlabeled = [_document("source", split="inference", domain=None)]
    unlabeled[0]["label_source"] = "should-not-be-here"
    _prepare_images(tmp_path, unlabeled)
    with pytest.raises(ValueError, match="cannot contain label_source"):
        load_font_domain_dataset(_write_manifest(tmp_path, unlabeled), require_labels=False)

    mixed_labeled = [_document("mixed", split="inference", domain="ios_alipay")]
    _prepare_images(tmp_path, mixed_labeled)
    with pytest.raises(ValueError, match="inference records cannot contain font_domain"):
        load_font_domain_dataset(_write_manifest(tmp_path, mixed_labeled), require_labels=None)


def test_publication_gate_requires_leakage_metadata(tmp_path: Path) -> None:
    records = [_document("missing-bindings")]
    _prepare_images(tmp_path, records)
    manifest = _write_manifest(tmp_path, records)

    with pytest.raises(ValueError, match="supervised publication requires"):
        load_font_domain_dataset(
            manifest,
            require_labels=True,
            require_leakage_metadata=True,
        )


def test_bound_region_detects_image_changes_after_manifest_validation(tmp_path: Path) -> None:
    records = [_document("bound")]
    _prepare_images(tmp_path, records)
    dataset = load_font_domain_dataset(_write_manifest(tmp_path, records), require_labels=True)
    region = dataset.documents[0].regions[0]
    assert region.load_bound_rgb().shape == (28, 72, 3)

    _write_image(region.image_path, 999)
    with pytest.raises(ValueError, match="changed after validation"):
        region.load_bound_rgb()


def test_unknown_schema_fields_and_conflicting_duplicate_labels_fail_closed(tmp_path: Path) -> None:
    typo = _document("typo")
    typo["font_domian"] = "ios_alipay"
    _prepare_images(tmp_path, [typo])
    with pytest.raises(ValueError, match="unknown document fields: font_domian"):
        load_font_domain_dataset(_write_manifest(tmp_path, [typo]), require_labels=True)

    shared_pixels = np.full((28, 72, 3), 255, dtype=np.uint8)
    shared_pixels[6:20, 10:20] = 0
    records = [
        _document("ios", regions=[_region("images/ios.png")]),
        _document(
            "android",
            domain="android_alipay",
            regions=[_region("images/android.png")],
        ),
    ]
    _write_image(tmp_path / "images/ios.png", 0, pixels=shared_pixels)
    _write_image(tmp_path / "images/android.png", 0, pixels=shared_pixels)
    with pytest.raises(ValueError, match="conflicting font domains"):
        load_font_domain_dataset(_write_manifest(tmp_path, records), require_labels=True)
