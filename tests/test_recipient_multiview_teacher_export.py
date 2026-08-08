from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.ocr_pseudolabels import _crop_digest
from transfer_receipt_ai.ocr_unified_dataset import (
    KIND_V13,
    RECIPIENT_QUALITY_POLICY_VERSION,
)
from transfer_receipt_ai.pipeline import crop_field_with_margin
from transfer_receipt_ai.recipient_multiview_teacher_export import (
    KIND,
    VIEWS,
    _fixed_value_view,
    _production_left_context_view,
    _production_right_value_view,
    _production_standard_view,
    build_parser,
    export_recipient_multiview_teacher,
)
import transfer_receipt_ai.recipient_multiview_teacher_export as multiview_export


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def _fixture(tmp_path: Path) -> dict[str, object]:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    source = source_dir / "train.png"
    y, x = np.mgrid[:60, :100]
    source_pixels = np.stack(
        (
            (x * 3 + y) % 256,
            (x + y * 5) % 256,
            (x * 7 + y * 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    _write_png(source, source_pixels)

    result = tmp_path / "results" / "train.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "source": source.as_posix(),
                "geometry": {
                    "source_size": {"width": 100, "height": 60},
                    "rectified_size": {"width": 100, "height": 60},
                    "H_original_to_rectified": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
                "detections": [],
            }
        ),
        encoding="utf-8",
    )

    dataset_root = tmp_path / "paddle-dataset"
    bbox = (20.0, 20.0, 80.0, 40.0)
    standard = np.ascontiguousarray(crop_field_with_margin(source_pixels, bbox))
    crop_hash = _crop_digest(standard)
    crop = dataset_root / "images" / "recipient_field" / f"{crop_hash}.png"
    _write_png(crop, standard)

    train_row = {
        "schema_version": 1,
        "id": "receipt-train",
        "group_id": "receipt:train",
        "split": "train",
        "source": source.as_posix(),
        "result_json": result.as_posix(),
        "label_source": "paddle_pseudo",
        "slots": {
            "recipient_field": {
                "image": crop.relative_to(dataset_root).as_posix(),
                "text": "商户甲",
                "recipient_visible_text": "收款方 商户甲",
                "recipient_value": "商户甲",
                "semantic_value": "商户甲",
                "recipient_quality_policy": RECIPIENT_QUALITY_POLICY_VERSION,
                "bbox_rectified": list(bbox),
                "crop_sha256": crop_hash,
            }
        },
    }
    heldout_row = {
        "schema_version": 1,
        "id": "receipt-val",
        "group_id": "receipt:val",
        "split": "val",
        "source": (tmp_path / "never-opened-val.png").as_posix(),
        "label_source": "paddle_pseudo",
        "slots": {
            "recipient_field": {
                # Deliberately invalid as a target.  The exporter may inspect
                # only this crop hash for closure; it must never read/validate
                # a held-out text target.
                "text": {"held_out_secret": "绝不能进训练"},
                "crop_sha256": "1" * 64,
            }
        },
    }
    formal_row = {
        "schema_version": 1,
        "id": "receipt-formal",
        "group_id": "receipt:formal",
        "split": "formal",
        "source": (tmp_path / "never-opened-formal.png").as_posix(),
        "slots": {"recipient_field": {"text": "正式集秘密", "crop_sha256": "2" * 64}},
    }
    input_dir = tmp_path / "input-manifest"
    input_dir.mkdir()
    manifest = input_dir / "unified_fields.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (train_row, heldout_row, formal_row)
        ),
        encoding="utf-8",
    )
    contract = input_dir / "dataset.contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": KIND_V13,
                "dataset_root": dataset_root.as_posix(),
                "recipient_charset_source": "train_only_anchored_recipient_value",
                "recipient_quality_policy": {
                    "version": RECIPIENT_QUALITY_POLICY_VERSION,
                    "requires_leading_recipient_label": True,
                    "target": "anchored_recipient_value",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "contract": contract,
        "dataset_root": dataset_root,
        "train_row": train_row,
        "heldout_row": heldout_row,
        "formal_row": formal_row,
        "standard": standard,
        "crop": crop,
    }


def _rows(output: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output / "multiview_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_export_is_train_only_and_emits_all_production_views(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "multiview"

    summary = export_recipient_multiview_teacher(
        manifest=fixture["manifest"],
        dataset_contract=fixture["contract"],
        dataset_root=fixture["dataset_root"],
        output_dir=output,
    )

    assert summary["kind"] == KIND
    assert summary["optimizer_supervision_splits"] == ["train"]
    assert summary["optimizer_input_ready"] is False
    assert summary["records_role"] == "recipient_multiview_overlay_source_only"
    assert summary["optimizer_adapter_required"].endswith("not_implemented")
    assert summary["held_out_target_values_used"] is False
    assert summary["held_out_target_values_validated"] is False
    assert summary["held_out_target_values_emitted"] is False
    assert summary["source_split_counts"] == {
        "train": 1,
        "val": 1,
        "test": 0,
        "formal": 1,
    }
    assert summary["source_train_recipient_records"] == 1
    assert summary["output_records"] == 4
    assert summary["view_counts"] == {view: 1 for view in VIEWS}
    assert summary["production_route_authorized"] is False
    assert summary["commit_marker"] == "dataset.contract.json"
    assert summary["publication_complete"] is True

    rows = _rows(output)
    assert [row["view"] for row in rows] == list(VIEWS)
    assert {row["split"] for row in rows} == {"train"}
    assert {row["text"] for row in rows} == {"商户甲"}
    assert {row["target_source"] for row in rows} == {"slots.recipient_field.text"}
    assert {row["optimizer_consumable"] for row in rows} == {False}
    assert {row["group_id"] for row in rows} == {"receipt:train"}
    assert len({row["group_closure_sha256"] for row in rows}) == 1
    assert {row["group_view_count"] for row in rows} == {4}

    sizes = {row["view"]: (row["view_width"], row["view_height"]) for row in rows}
    assert sizes == {
        "fixed_value": (49, 24),
        "standard": (70, 24),
        "left_context": (85, 24),
        "right_value": (40, 24),
    }
    for row in rows:
        image = output / str(row["image"])
        assert image.is_file()
        assert hashlib.sha256(image.read_bytes()).hexdigest() == row["view_file_sha256"]
        assert _crop_digest(np.asarray(Image.open(image).convert("RGB"))) == row["view_pixel_sha256"]

    text_outputs = (output / "multiview_train.jsonl").read_text(encoding="utf-8") + (
        output / "dataset.contract.json"
    ).read_text(encoding="utf-8")
    assert "绝不能进训练" not in text_outputs
    assert "正式集秘密" not in text_outputs
    assert summary["train_manifest_sha256"] == hashlib.sha256(
        (output / "multiview_train.jsonl").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("label_source", "transaction_truth", "paddle_pseudo"),
        ("recipient_value", "冲突商户", "conflicts with manifest target"),
        ("recipient_visible_text", "付款方式 商户甲", "anchored visible text conflicts"),
    ],
)
def test_export_rejects_any_train_target_authority_conflict(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if field == "label_source":
        records[0][field] = value
    else:
        records[0]["slots"]["recipient_field"][field] = value
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        export_recipient_multiview_teacher(
            manifest=manifest,
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_train_receipt_without_recipient_target_is_counted_but_not_fabricated(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records.append(
        {
            "schema_version": 1,
            "id": "train-without-recipient",
            "group_id": "receipt:train-without-recipient",
            "split": "train",
            "source": (tmp_path / "must-not-open.png").as_posix(),
            "slots": {"amount": {"text": "1.00"}},
        }
    )
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    output = tmp_path / "out"
    summary = export_recipient_multiview_teacher(
        manifest=manifest,
        dataset_contract=fixture["contract"],
        dataset_root=fixture["dataset_root"],
        output_dir=output,
    )
    assert summary["source_manifest_split_counts"]["train"] == 2
    assert summary["source_split_counts"]["train"] == 1
    assert summary["source_train_records_without_recipient_target"] == 1
    assert len(_rows(output)) == 4


@pytest.mark.parametrize("conflict", ["group", "source", "crop"])
def test_export_rejects_cross_split_group_source_or_crop_closure(
    tmp_path: Path,
    conflict: str,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if conflict == "group":
        records[1]["group_id"] = records[0]["group_id"]
    elif conflict == "source":
        records[1]["source"] = records[0]["source"]
    else:
        records[1]["slots"]["recipient_field"]["crop_sha256"] = records[0]["slots"][
            "recipient_field"
        ]["crop_sha256"]
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="crosses"):
        export_recipient_multiview_teacher(
            manifest=manifest,
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_heldout_row_without_recipient_cannot_bypass_source_closure(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records[1]["slots"] = {"amount": {"text": "1.00"}}
    records[1]["source"] = records[0]["source"]
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source .* crosses"):
        export_recipient_multiview_teacher(
            manifest=manifest,
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_generated_train_view_hash_cannot_cross_heldout_crop_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records[1]["slots"]["recipient_field"]["crop_sha256"] = _crop_digest(
        _fixed_value_view(fixture["standard"])
    )
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generated train view hash .* crosses"):
        export_recipient_multiview_teacher(
            manifest=manifest,
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_export_rejects_crop_pixels_that_do_not_close_to_manifest_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    crop = fixture["crop"]
    changed = np.asarray(Image.open(crop).convert("RGB")).copy()
    changed[0, 0, 0] ^= 255
    _write_png(crop, changed)

    with pytest.raises(ValueError, match="stored Paddle crop pixels differ"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_export_rejects_live_source_newer_than_paddle_result(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source = Path(str(fixture["train_row"]["source"]))
    result_json = json.loads(
        Path(fixture["manifest"]).read_text(encoding="utf-8").splitlines()[0]
    )["result_json"]
    result_mtime = Path(result_json).stat().st_mtime_ns
    os.utime(source, ns=(result_mtime + 1_000_000_000, result_mtime + 1_000_000_000))

    with pytest.raises(ValueError, match="live source is newer"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_export_rejects_non_anchored_unified_contract(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract = fixture["contract"]
    value = json.loads(contract.read_text(encoding="utf-8"))
    value["recipient_charset_source"] = "all_splits"
    contract.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="not train-only anchored"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=contract,
            dataset_root=fixture["dataset_root"],
            output_dir=tmp_path / "out",
        )


def test_export_never_overwrites_existing_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=output,
        )


def test_export_rejects_dangling_output_symlink_and_symlink_ancestor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    dangling = tmp_path / "dangling-output"
    dangling.symlink_to(tmp_path / "missing-output", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink or reparse"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=dangling,
        )

    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink or reparse"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=linked_parent / "out",
        )


@pytest.mark.parametrize("protected", ["manifest", "dataset", "source"])
def test_export_rejects_output_overlap_with_any_live_input(
    tmp_path: Path,
    protected: str,
) -> None:
    fixture = _fixture(tmp_path)
    if protected == "manifest":
        output = Path(fixture["manifest"]).parent / "out"
        match = "manifest directory"
    elif protected == "dataset":
        output = Path(fixture["dataset_root"]) / "out"
        match = "Paddle dataset root"
    else:
        train_row = fixture["train_row"]
        output = Path(str(train_row["source"])).parent / "out"
        match = "live source directory"

    with pytest.raises(ValueError, match=match):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=output,
        )


def test_failed_publish_preserves_foreign_file_and_never_commits_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "contended-output"

    def competing_link(source: object, destination: object) -> None:
        destination_path = Path(destination)
        (destination_path.parent / "foreign.txt").write_text("competitor", encoding="utf-8")
        raise OSError("simulated publication race")

    monkeypatch.setattr(multiview_export.os, "link", competing_link)
    with pytest.raises(OSError, match="publication race"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=output,
        )

    assert (output / "images" / "foreign.txt").read_text(encoding="utf-8") == "competitor"
    assert not (output / "dataset.contract.json").exists()
    assert not (output / "multiview_train.jsonl").exists()
    assert not list(tmp_path.glob(".contended-output.*.tmp"))


def test_source_change_during_publish_prevents_contract_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "contended-source-output"
    real_link = os.link
    link_count = 0

    def changing_link(source: object, destination: object) -> None:
        nonlocal link_count
        real_link(source, destination)
        link_count += 1
        if link_count == 1:
            manifest = Path(fixture["manifest"])
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(multiview_export.os, "link", changing_link)
    with pytest.raises(ValueError, match="multiview source changed"):
        export_recipient_multiview_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_dir=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".contended-source-output.*.tmp"))


def test_crop_formulas_match_csharp_production_contract_examples() -> None:
    source = np.zeros((1000, 1000, 3), dtype=np.uint8)
    y, x = np.mgrid[:1000, :1000]
    source[:, :, 0] = x % 256
    source[:, :, 1] = y % 256

    # This fractional box is a deliberate precision discriminator: Python's
    # double calculation starts at x=684, while the deployed C# float formula
    # starts at x=685.
    precision_source = np.zeros((600, 1100, 3), dtype=np.uint8)
    precision_source[:, :, 0] = np.arange(1100, dtype=np.uint16) % 256
    production_standard = _production_standard_view(
        precision_source,
        (704.189, 200.0, 944.052, 400.0),
    )
    assert production_standard.shape == (232, 279, 3)
    np.testing.assert_array_equal(production_standard[0, 0], precision_source[184, 685])

    left_context = _production_left_context_view(
        source,
        (100.0, 200.0, 800.0, 400.0),
    )
    assert left_context.shape == (232, 856, 3)
    np.testing.assert_array_equal(left_context[0, 0], source[184, 0])
    np.testing.assert_array_equal(left_context[-1, -1], source[415, 855])

    right_value = _production_right_value_view(
        source,
        (100.0, 200.0, 800.0, 400.0),
    )
    assert right_value.shape == (232, 406, 3)
    np.testing.assert_array_equal(right_value[0, 0], source[184, 450])
    np.testing.assert_array_equal(right_value[-1, -1], source[415, 855])

    box_bound = _production_right_value_view(
        source,
        (600.0, 200.0, 900.0, 400.0),
    )
    assert box_bound.shape == (232, 300, 3)
    np.testing.assert_array_equal(box_bound[0, 0], source[184, 624])
    with pytest.raises(ValueError, match="right-value view is empty"):
        _production_right_value_view(source, (100.0, 200.0, 400.0, 400.0))

    assert _fixed_value_view(np.zeros((2, 55, 3), dtype=np.uint8)).shape[1] == 39
    assert _fixed_value_view(np.zeros((2, 65, 3), dtype=np.uint8)).shape[1] == 45


def test_cli_has_no_split_or_target_override(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                str(tmp_path / "manifest.jsonl"),
                "--output",
                str(tmp_path / "out"),
                "--split",
                "val",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                str(tmp_path / "manifest.jsonl"),
                "--output",
                str(tmp_path / "out"),
                "--target",
                "伪造标签",
            ]
        )
