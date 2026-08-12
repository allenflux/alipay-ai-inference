from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from transfer_receipt_ai import otherimages_inventory as inventory_module
from transfer_receipt_ai.otherimages_inventory import (
    OUTPUT_FILENAMES,
    _decoded_pixel_sha256,
    _hamming_distance,
    _perceptual_hash64,
    _top_strip_statusbar_metrics,
    build_otherimages_inventory,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _receipt_pixels(*, width: int = 240, height: int = 400, statusbar: bool = True) -> np.ndarray:
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    if statusbar:
        strip_height = max(1, round(height * 0.08))
        for left in range(8, width - 8, 24):
            pixels[4 : max(5, strip_height - 4), left : left + 8] = 0
    for top, right in ((80, 190), (125, 160), (210, 205), (290, 180)):
        pixels[top : top + 5, 28:right] = 30
    return pixels


def _write_receipt(path: Path, *, width: int = 240, height: int = 400, statusbar: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_receipt_pixels(width=width, height=height, statusbar=statusbar)).save(path)


def _source_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_inventory_publishes_bound_read_only_manifests_and_unlabeled_teacher_input(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    original = source / "nested" / "white.png"
    duplicate = source / "same-bytes.png"
    different = source / "different.jpg"
    _write_receipt(original)
    shutil.copyfile(original, duplicate)
    Image.fromarray(_receipt_pixels(statusbar=False)).save(different, quality=94)
    (source / "notes.txt").write_text("not an image\n", encoding="utf-8")
    before = _source_snapshot(source)

    output = tmp_path / "inventory-v1"
    contract = build_otherimages_inventory(
        input_dir=source,
        output_dir=output,
        layout_sample_size=10,
        split_seed="fixed-test-split",
    )

    assert _source_snapshot(source) == before
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {*OUTPUT_FILENAMES, "inventory.contract.json"}
    assert all(path.suffix in {".json", ".jsonl"} for path in output.iterdir())
    images = _read_jsonl(output / "images.jsonl")
    assert len(images) == 3
    assert contract["counts"]["images"] == 3
    assert contract["counts"]["ignored_non_images"] == 1
    assert contract["source"]["source_membership_rechecked"] is True
    assert contract["paddle_teacher_contract"]["inventory_contains_labels"] is False
    assert contract["paddle_teacher_contract"]["guessed_or_synthetic_labels_forbidden"] is True

    by_relative = {row["source"]["relative_path"]: row for row in images}
    original_row = by_relative["nested/white.png"]
    duplicate_row = by_relative["same-bytes.png"]
    assert original_row["hashes"]["raw_sha256"] == duplicate_row["hashes"]["raw_sha256"]
    assert original_row["hashes"]["decoded_pixel_sha256"] == duplicate_row["hashes"]["decoded_pixel_sha256"]
    assert original_row["group_id"] == duplicate_row["group_id"]
    assert original_row["suggested_split"] == duplicate_row["suggested_split"]
    assert original_row["top_8_percent_statusbar"]["presence_state"] == "likely_present"
    assert original_row["top_8_percent_statusbar"]["measurement_canvas_width"] == 512
    assert original_row["operations"] == {
        "image_copied": False,
        "ocr_performed": False,
        "source_mutated": False,
        "source_open_mode": "read_only",
        "training_performed": False,
    }

    teacher_rows = _read_jsonl(output / "paddle_teacher_pending.jsonl")
    assert len(teacher_rows) == 3
    assert sorted(row["teacher_state"] for row in teacher_rows) == ["pending", "pending", "quarantine"]
    for row in teacher_rows:
        assert row["labels_present"] is False
        assert row["ocr_performed"] is False
        assert row["training_eligible"] is False
        assert row["manual_review_required"] is False
        assert "text" not in row
        assert "label" not in row

    bindings = {row["path"]: row for row in contract["artifacts"]}
    assert set(bindings) == set(OUTPUT_FILENAMES)
    for name, binding in bindings.items():
        payload = (output / name).read_bytes()
        assert binding["size_bytes"] == len(payload)
        assert binding["sha256"] == hashlib.sha256(payload).hexdigest()
        assert not payload or payload.endswith(b"\n")


def test_inventory_records_exif_orientation_dimensions_and_existing_decoded_hash_abi(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    path = source / "oriented.png"
    pixels = np.zeros((20, 10, 3), dtype=np.uint8)
    pixels[:, :5] = (240, 20, 30)
    pixels[:, 5:] = (10, 100, 230)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(path, exif=exif)

    output = tmp_path / "inventory"
    build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=1)
    record = _read_jsonl(output / "images.jsonl")[0]

    assert record["container"]["format"] == "PNG"
    assert record["geometry"]["stored_width"] == 10
    assert record["geometry"]["stored_height"] == 20
    assert record["geometry"]["upright_width"] == 20
    assert record["geometry"]["upright_height"] == 10
    assert record["exif"]["orientation"] == 6
    assert record["exif"]["orientation_applied"] is True
    with Image.open(path) as opened:
        upright_rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    assert record["hashes"]["decoded_pixel_sha256"] == _decoded_pixel_sha256(upright_rgb)
    assert record["hashes"]["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_phash_near_candidate_binds_resized_images_to_one_split(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    base = _receipt_pixels(width=240, height=400)
    Image.fromarray(base).save(source / "base.png")
    Image.fromarray(base).resize((480, 800), Image.Resampling.BICUBIC).save(source / "resized.png")

    output = tmp_path / "inventory"
    build_otherimages_inventory(
        input_dir=source,
        output_dir=output,
        phash_distance=6,
        layout_sample_size=2,
    )
    images = _read_jsonl(output / "images.jsonl")
    distance = _hamming_distance(
        int(images[0]["hashes"]["phash64"], 16),
        int(images[1]["hashes"]["phash64"], 16),
    )
    assert distance <= 6
    assert images[0]["hashes"]["decoded_pixel_sha256"] != images[1]["hashes"]["decoded_pixel_sha256"]
    assert images[0]["group_id"] == images[1]["group_id"]
    assert images[0]["suggested_split"] == images[1]["suggested_split"]
    candidates = _read_jsonl(output / "near_duplicate_candidates.jsonl")
    assert len(candidates) == 1
    assert candidates[0]["distance"] == distance
    assert candidates[0]["automatic_drop_authorized"] is False


def test_cropped_near_duplicate_outside_aspect_reference_still_shares_group_and_split(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    base = _receipt_pixels(width=240, height=400)
    Image.fromarray(base).save(source / "base.png")
    Image.fromarray(base[8:-8]).save(source / "cropped.png")

    output = tmp_path / "inventory"
    build_otherimages_inventory(input_dir=source, output_dir=output, phash_distance=6, layout_sample_size=2)
    records = _read_jsonl(output / "images.jsonl")
    distance = _hamming_distance(
        int(records[0]["hashes"]["phash64"], 16),
        int(records[1]["hashes"]["phash64"], 16),
    )
    assert distance <= 6
    assert records[0]["group_id"] == records[1]["group_id"]
    assert records[0]["suggested_split"] == records[1]["suggested_split"]
    candidate = _read_jsonl(output / "near_duplicate_candidates.jsonl")[0]
    assert candidate["aspect_relative_delta"] > 0.01
    assert candidate["aspect_risk"] == "outside_1pct_conservative_same_split_union"


def test_low_information_white_images_do_not_form_false_phash_cluster(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    Image.new("RGB", (100, 200), "white").save(source / "small.png")
    Image.new("RGB", (200, 400), "white").save(source / "large.png")

    output = tmp_path / "inventory"
    contract = build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=2)
    records = _read_jsonl(output / "images.jsonl")

    assert all(record["full_image_quality"]["phash_usable"] is False for record in records)
    assert all("low_information_full" in record["quality_flags"] for record in records)
    assert records[0]["group_id"] != records[1]["group_id"]
    assert _read_jsonl(output / "near_duplicate_candidates.jsonl") == []
    assert contract["phash_candidates"]["phash_unusable_images"] == 2


def test_corrupt_image_is_quarantined_without_ocr_or_teacher_label(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    _write_receipt(source / "valid.png")
    corrupt = source / "broken.png"
    corrupt.write_bytes(b"not a png")

    output = tmp_path / "inventory"
    contract = build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=1)

    assert contract["counts"]["images"] == 1
    assert contract["counts"]["image_errors_quarantined"] == 1
    errors = _read_jsonl(output / "errors.jsonl")
    assert errors[0]["relative_path"] == "broken.png"
    assert errors[0]["raw_sha256"] == hashlib.sha256(corrupt.read_bytes()).hexdigest()
    assert errors[0]["disposition"] == "quarantine"
    assert errors[0]["ocr_performed"] is False
    assert errors[0]["training_eligible"] is False
    assert len(_read_jsonl(output / "paddle_teacher_pending.jsonl")) == 1


@pytest.mark.parametrize("bomb_dimensions", [(15, 10), (20, 20)])
def test_pillow_decompression_bomb_warning_or_error_quarantines_only_that_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bomb_dimensions: tuple[int, int],
) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    Image.new("RGB", (8, 8), "white").save(source / "valid.png")
    Image.new("RGB", bomb_dimensions, "white").save(source / "bomb.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    output = tmp_path / "inventory"
    contract = build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=1)

    assert contract["counts"]["images"] == 1
    assert contract["counts"]["image_errors_quarantined"] == 1
    error = _read_jsonl(output / "errors.jsonl")[0]
    assert error["relative_path"] == "bomb.png"
    assert error["error_type"] in {"DecompressionBombWarning", "DecompressionBombError"}
    assert error["disposition"] == "quarantine"


def test_multiframe_image_is_inventory_only_and_teacher_quarantined(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    first = Image.new("RGB", (120, 200), "white")
    second = Image.new("RGB", (120, 200), "black")
    first.save(source / "animated.gif", save_all=True, append_images=[second], duration=50, loop=0)

    output = tmp_path / "inventory"
    build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=1)
    image = _read_jsonl(output / "images.jsonl")[0]
    teacher = _read_jsonl(output / "paddle_teacher_pending.jsonl")[0]

    assert image["container"]["frame_count"] == 2
    assert image["container"]["decoded_frame_policy"] == "first_frame"
    assert teacher["teacher_state"] == "quarantine"
    assert teacher["quarantine_reason"] == "multiframe_first_frame_only"
    assert teacher["training_eligible"] is False


def test_inventory_refuses_overlap_existing_output_and_reparse_source(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    _write_receipt(source / "valid.png")

    with pytest.raises(ValueError, match="must not overlap"):
        build_otherimages_inventory(
            input_dir=source,
            output_dir=source / "inventory",
            layout_sample_size=1,
        )
    assert not (source / "inventory").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "owner.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="brand-new"):
        build_otherimages_inventory(input_dir=source, output_dir=existing, layout_sample_size=1)
    assert marker.read_text(encoding="utf-8") == "keep"

    linked = source / "linked.png"
    try:
        linked.symlink_to(source / "valid.png")
    except OSError:
        pytest.skip("symlink creation is not available")
    final = tmp_path / "must-not-publish"
    with pytest.raises(ValueError, match="symlink/junction/reparse"):
        build_otherimages_inventory(input_dir=source, output_dir=final, layout_sample_size=1)
    assert not final.exists()


def test_source_change_during_closure_is_fatal_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    target = source / "valid.png"
    _write_receipt(target)
    output = tmp_path / "must-not-publish"
    original_closure = inventory_module._assert_source_closure
    mutated = False

    def mutate_then_check(*args: object, **kwargs: object) -> None:
        nonlocal mutated
        if not mutated:
            target.write_bytes(target.read_bytes() + b"changed-during-audit")
            mutated = True
        original_closure(*args, **kwargs)

    monkeypatch.setattr(inventory_module, "_assert_source_closure", mutate_then_check)
    with pytest.raises(inventory_module.SourceChangedError, match="changed"):
        build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=1)

    assert mutated
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-publish.inventory-building-*"))


def test_phash_candidate_cap_fails_instead_of_truncating_leakage_evidence(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    base = Image.fromarray(_receipt_pixels())
    base.save(source / "a.png")
    base.resize((360, 600), Image.Resampling.BICUBIC).save(source / "b.png")
    base.resize((480, 800), Image.Resampling.BICUBIC).save(source / "c.png")
    output = tmp_path / "must-not-publish"

    with pytest.raises(ValueError, match="exceeds --max-phash-candidates"):
        build_otherimages_inventory(
            input_dir=source,
            output_dir=output,
            layout_sample_size=1,
            maximum_phash_candidates=1,
        )

    assert not output.exists()


def test_many_exact_decoded_copies_are_collapsed_before_phash_candidate_cap(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    _write_receipt(source / "copy-000.png")
    for index in range(1, 449):
        shutil.copyfile(source / "copy-000.png", source / f"copy-{index:03d}.png")

    output = tmp_path / "inventory"
    contract = build_otherimages_inventory(
        input_dir=source,
        output_dir=output,
        layout_sample_size=1,
        maximum_phash_candidates=1,
    )

    assert contract["counts"]["images"] == 449
    assert contract["phash_candidates"]["exact_decoded_copies_collapsed_before_phash_index"] == 448
    assert contract["phash_candidates"]["candidate_evidence_rows"] == 0
    assert len(_read_jsonl(output / "near_duplicate_candidates.jsonl")) == 0
    assert len(_read_jsonl(output / "paddle_teacher_pending.jsonl")) == 449


def test_atomic_no_replace_publication_refuses_destination_created_at_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    _write_receipt(source / "valid.png")
    output = tmp_path / "raced-output"
    original_publish = inventory_module._rename_directory_no_replace

    def create_competitor_then_publish(stage: Path, destination: Path) -> None:
        destination.mkdir()
        original_publish(stage, destination)

    monkeypatch.setattr(inventory_module, "_rename_directory_no_replace", create_competitor_then_publish)
    with pytest.raises(FileExistsError, match="replace existing|File exists"):
        build_otherimages_inventory(input_dir=source, output_dir=output, layout_sample_size=1)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".raced-output.inventory-building-*"))


def test_layout_sample_and_split_recommendations_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "OtherImages"
    source.mkdir()
    for index in range(8):
        pixels = _receipt_pixels(width=180 + index * 7, height=300 + index * 11, statusbar=index % 2 == 0)
        pixels[150:155, 20 : 45 + index * 3] = index * 20
        Image.fromarray(pixels).save(source / f"image-{index}.png")

    first_output = tmp_path / "inventory-a"
    second_output = tmp_path / "inventory-b"
    kwargs = {
        "input_dir": source,
        "layout_sample_size": 5,
        "split_seed": "stable-split",
        "layout_sample_seed": "stable-layout",
    }
    build_otherimages_inventory(output_dir=first_output, **kwargs)
    build_otherimages_inventory(output_dir=second_output, **kwargs)

    assert _read_jsonl(first_output / "layout_sample.jsonl") == _read_jsonl(
        second_output / "layout_sample.jsonl"
    )
    first_teacher = _read_jsonl(first_output / "paddle_teacher_pending.jsonl")
    second_teacher = _read_jsonl(second_output / "paddle_teacher_pending.jsonl")
    assert [row["record_id"] for row in first_teacher] == [row["record_id"] for row in second_teacher]
    assert [row["group_id"] for row in first_teacher] == [row["group_id"] for row in second_teacher]
    assert [row["suggested_split"] for row in first_teacher] == [row["suggested_split"] for row in second_teacher]


def test_statusbar_and_phash_helpers_are_deterministic_and_finite() -> None:
    patterned = _receipt_pixels(width=240, height=400, statusbar=True)
    blank = np.full_like(patterned, 255)

    first_hash = _perceptual_hash64(patterned)
    assert first_hash == "bf3f41494a54147e"
    assert first_hash == _perceptual_hash64(patterned.copy())
    assert len(first_hash) == 16
    assert _top_strip_statusbar_metrics(patterned)["presence_state"] == "likely_present"
    assert _top_strip_statusbar_metrics(blank)["presence_state"] == "unlikely_present"
    payload = json.dumps(_top_strip_statusbar_metrics(patterned), allow_nan=False)
    assert "NaN" not in payload


def test_checkout_wrapper_exposes_inventory_cli_without_package_installation() -> None:
    wrapper = Path(__file__).parents[1] / "scripts" / "otherimages-inventory.py"
    result = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--input INPUT" in result.stdout
    assert "--output OUTPUT" in result.stdout
    assert "Paddle teacher manifest" in result.stdout
