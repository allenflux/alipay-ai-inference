from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
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
import transfer_receipt_ai.recipient_fixed2_teacher_export as fixed2
import transfer_receipt_ai.recipient_fixed2_teacher_attestation as fixed2_attestation
import transfer_receipt_ai.recipient_multiview_teacher_export as four_view


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def _train_row(
    tmp_path: Path,
    dataset_root: Path,
    *,
    index: int,
    target: str,
    group_id: str,
    left_variant: bool,
) -> tuple[dict[str, object], str]:
    y, x = np.mgrid[:60, :100]
    pixels = np.stack(
        ((x * 3 + y) % 256, (x + y * 5) % 256, (x * 7 + y * 11) % 256),
        axis=2,
    ).astype(np.uint8)
    if left_variant:
        # The production standard crop is x=15:85.  Its fixed-value trim
        # begins at crop column 21/source x=36, so this changes standard while
        # leaving fixed_value byte-identical.
        pixels[:, 15:36, 0] ^= np.uint8(0x5A)
    source = tmp_path / "raw" / f"train-{index}.png"
    _write_png(source, pixels)
    result = tmp_path / "results" / f"train-{index}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
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
    bbox = (20.0, 20.0, 80.0, 40.0)
    standard = np.ascontiguousarray(crop_field_with_margin(pixels, bbox))
    crop_sha = _crop_digest(standard)
    crop = dataset_root / "images" / "recipient_field" / f"{crop_sha}.png"
    _write_png(crop, standard)
    fixed_sha = _crop_digest(fixed2._fixed_value_view(standard))
    return (
        {
            "schema_version": 1,
            "id": f"receipt-train-{index}",
            "group_id": group_id,
            "split": "train",
            "source": str(source.resolve()),
            "result_json": str(result.resolve()),
            "label_source": "paddle_pseudo",
            "slots": {
                "recipient_field": {
                    "image": crop.relative_to(dataset_root).as_posix(),
                    "text": target,
                    "recipient_visible_text": f"收款方 {target}",
                    "recipient_value": target,
                    "semantic_value": target,
                    "recipient_quality_policy": RECIPIENT_QUALITY_POLICY_VERSION,
                    "bbox_rectified": list(bbox),
                    "crop_sha256": crop_sha,
                }
            },
        },
        fixed_sha,
    )


def _fixture(
    tmp_path: Path,
    *,
    duplicate: bool = False,
    second_target: str = "商户甲",
    second_group: str = "receipt:train:1",
    heldout_crop_sha: str | None = None,
) -> dict[str, Path]:
    dataset_root = tmp_path / "dataset"
    first, _first_fixed_sha = _train_row(
        tmp_path,
        dataset_root,
        index=0,
        target="商户甲",
        group_id="receipt:train:0",
        left_variant=False,
    )
    rows: list[dict[str, object]] = [first]
    if duplicate:
        second, _second_fixed_sha = _train_row(
            tmp_path,
            dataset_root,
            index=1,
            target=second_target,
            group_id=second_group,
            left_variant=True,
        )
        rows.append(second)
    rows.append(
        {
            "schema_version": 1,
            "id": "receipt-val",
            "group_id": "receipt:val",
            "split": "val",
            "source": str((tmp_path / "never-open-val.png").resolve()),
            "slots": {
                "recipient_field": {
                    "text": {"held_out": "must-not-read"},
                    "crop_sha256": heldout_crop_sha or "f" * 64,
                }
            },
        }
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    manifest = evidence / "unified_fields.train-val.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    contract = evidence / "dataset.contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": KIND_V13,
                "dataset_root": str(dataset_root.resolve()),
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
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    return {
        "manifest": manifest,
        "contract": contract,
        "dataset_root": dataset_root,
        "output_parent": output_parent,
    }


def _materialize(fixture: dict[str, Path], name: str) -> dict[str, object]:
    return fixed2._materialize_recipient_fixed2_teacher_analysis_test_only(
        manifest=fixture["manifest"],
        dataset_contract=fixture["contract"],
        dataset_root=fixture["dataset_root"],
        output_root=fixture["output_parent"] / name,
    )


def _swap_path_during_snapshot_use(
    monkeypatch: pytest.MonkeyPatch,
    *,
    path: Path,
    before_checkpoint: str,
    after_checkpoint: str,
    malicious_bytes: bytes,
) -> list[str]:
    backup = path.with_name(path.name + ".snapshot-original")
    observed: list[str] = []

    def hook(
        checkpoint: str,
        *,
        snapshot: fixed2._FrozenFileSnapshot,
        description: str,
    ) -> None:
        del description
        if checkpoint == before_checkpoint:
            assert snapshot.path == path.resolve()
            path.rename(backup)
            path.write_bytes(malicious_bytes)
            observed.append("before")
        elif checkpoint == after_checkpoint:
            path.unlink()
            backup.rename(path)
            observed.append("after")

    monkeypatch.setattr(fixed2, "_snapshot_use_hook", hook)
    return observed


@pytest.mark.skipif(os.name == "nt", reason="analysis profile is POSIX-only")
@pytest.mark.parametrize(
    "artifact",
    ["manifest", "contract", "source", "result", "crop"],
)
def test_materializer_consumes_frozen_bytes_during_swap_read_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    fixture = _fixture(tmp_path)
    first = json.loads(fixture["manifest"].read_text(encoding="utf-8").splitlines()[0])
    choices = {
        "manifest": (
            fixture["manifest"],
            "preflight_manifest_before_parse",
            "preflight_manifest_after_parse",
            b"{}\n",
        ),
        "contract": (
            fixture["contract"],
            "preflight_contract_before_parse",
            "preflight_contract_after_parse",
            b"{}",
        ),
        "source": (
            Path(first["source"]),
            "preflight_receipt-train-0_source_before_decode",
            "preflight_receipt-train-0_source_after_decode",
            b"not-an-image",
        ),
        "result": (
            Path(first["result_json"]),
            "preflight_receipt-train-0_result_before_parse",
            "preflight_receipt-train-0_result_after_parse",
            b"{}",
        ),
        "crop": (
            fixture["dataset_root"] / first["slots"]["recipient_field"]["image"],
            "preflight_receipt-train-0_crop_before_decode",
            "preflight_receipt-train-0_crop_after_decode",
            b"not-an-image",
        ),
    }
    path, before, after, malicious = choices[artifact]
    original = path.read_bytes()
    observed = _swap_path_during_snapshot_use(
        monkeypatch,
        path=path,
        before_checkpoint=before,
        after_checkpoint=after,
        malicious_bytes=malicious,
    )

    published = _materialize(fixture, f"swap-{artifact}")

    assert published["output_records"] == 2
    assert observed == ["before", "after"]
    assert path.read_bytes() == original


@pytest.mark.skipif(os.name == "nt", reason="analysis profile is POSIX-only")
@pytest.mark.parametrize(
    "artifact",
    [
        "marker",
        "output_manifest",
        "source_manifest",
        "source_contract",
        "source",
        "result",
        "crop",
        "png",
    ],
)
def test_verifier_consumes_frozen_bytes_during_swap_read_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    fixture = _fixture(tmp_path)
    published = _materialize(fixture, "fixed2")
    root = fixture["output_parent"] / "fixed2"
    first = json.loads(fixture["manifest"].read_text(encoding="utf-8").splitlines()[0])
    output_rows = [
        json.loads(line)
        for line in (root / fixed2.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    ]
    standard = next(row for row in output_rows if row["view"] == "standard")
    choices = {
        "marker": (
            root / fixed2.ANALYSIS_CONTRACT_NAME,
            "analysis_contract_marker_before_parse",
            "analysis_contract_marker_after_parse",
            b"{}",
        ),
        "output_manifest": (
            root / fixed2.MANIFEST_NAME,
            "verify_output_manifest_before_parse",
            "verify_output_manifest_after_parse",
            b"{}\n",
        ),
        "source_manifest": (
            fixture["manifest"],
            "verify_source_manifest_before_parse",
            "verify_source_manifest_after_parse",
            b"{}\n",
        ),
        "source_contract": (
            fixture["contract"],
            "verify_source_contract_before_parse",
            "verify_source_contract_after_parse",
            b"{}",
        ),
        "source": (
            Path(first["source"]),
            "verify_receipt-train-0_source_before_decode",
            "verify_receipt-train-0_source_after_decode",
            b"not-an-image",
        ),
        "result": (
            Path(first["result_json"]),
            "verify_receipt-train-0_result_before_parse",
            "verify_receipt-train-0_result_after_parse",
            b"{}",
        ),
        "crop": (
            fixture["dataset_root"] / first["slots"]["recipient_field"]["image"],
            "verify_receipt-train-0_crop_before_decode",
            "verify_receipt-train-0_crop_after_decode",
            b"not-an-image",
        ),
        "png": (
            root / str(standard["image"]),
            "verify_png_receipt-train-0_standard_before_decode",
            "verify_png_receipt-train-0_standard_after_decode",
            b"not-an-image",
        ),
    }
    path, before, after, malicious = choices[artifact]
    original = path.read_bytes()
    observed = _swap_path_during_snapshot_use(
        monkeypatch,
        path=path,
        before_checkpoint=before,
        after_checkpoint=after,
        malicious_bytes=malicious,
    )

    verified = fixed2._verify_recipient_fixed2_teacher_analysis_test_only(
        export_root=root
    )

    assert verified["producer_subject_id"] == published["producer_subject_id"]
    assert observed == ["before", "after"]
    assert path.read_bytes() == original


def test_analysis_producer_emits_only_fixed2_and_has_path_stable_subject(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first = _materialize(fixture, "fixed2-a")
    second = _materialize(fixture, "fixed2-b")

    assert first["kind"] == fixed2.ANALYSIS_KIND
    assert first["publication_authority"] == fixed2.ANALYSIS_PUBLICATION_AUTHORITY
    assert first["publication_identity"] is None
    assert first["optimizer_input_ready"] is False
    assert first["view_order"] == ["standard", "fixed_value"]
    assert first["output_records"] == 2
    assert first["producer_subject_id"] == second["producer_subject_id"]
    assert first["nominal_output_root"] == str(
        fixture["output_parent"] / "fixed2-a"
    )
    assert second["nominal_output_root"] == str(
        fixture["output_parent"] / "fixed2-b"
    )
    assert first["subject_path_stable"] is True
    assert first["subject_output_stable"] is True
    assert first["subject_code_stable"] is True
    rows = [
        json.loads(line)
        for line in (fixture["output_parent"] / "fixed2-a" / fixed2.MANIFEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["view"] for row in rows] == ["standard", "fixed_value"]
    assert all(set(row) == fixed2.RECORD_KEYS for row in rows)
    assert fixed2._verify_recipient_fixed2_teacher_analysis_test_only(
        export_root=fixture["output_parent"] / "fixed2-a"
    )["producer_subject_id"] == first["producer_subject_id"]


def test_copied_publication_cannot_claim_the_declared_nominal_destination(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _materialize(fixture, "fixed2")
    original = fixture["output_parent"] / "fixed2"
    copied = fixture["output_parent"] / "fixed2-copy"
    shutil.copytree(original, copied)

    with pytest.raises(ValueError, match="nominal output root"):
        fixed2._verify_recipient_fixed2_teacher_analysis_test_only(
            export_root=copied
        )


@pytest.mark.parametrize(
    ("second_target", "second_group", "expected"),
    [
        ("商户甲", "receipt:train:1", "group_conflict=true target_conflict=false"),
        ("商户乙", "receipt:train:1", "group_conflict=true target_conflict=true"),
        ("商户乙", "receipt:train:0", "group_conflict=false target_conflict=true"),
    ],
)
def test_selected_duplicate_across_groups_fails_before_stage(
    tmp_path: Path,
    second_target: str,
    second_group: str,
    expected: str,
) -> None:
    fixture = _fixture(
        tmp_path,
        duplicate=True,
        second_target=second_target,
        second_group=second_group,
    )
    output = fixture["output_parent"] / "blocked"
    with pytest.raises(ValueError, match=expected):
        _materialize(fixture, "blocked")
    assert not output.exists()
    assert not list(fixture["output_parent"].glob(".blocked.*.tmp"))


def test_excluded_four_view_builders_are_never_called(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("excluded diagnostic view builder was called")

    monkeypatch.setattr(four_view, "_production_left_context_view", forbidden)
    monkeypatch.setattr(four_view, "_production_right_value_view", forbidden)
    published = _materialize(fixture, "fixed2")
    assert published["view_order"] == ["standard", "fixed_value"]


def test_selected_view_cannot_cross_heldout_declared_crop_boundary(tmp_path: Path) -> None:
    baseline = _fixture(tmp_path / "baseline")
    # Read the fixed-value hash without opening a held-out artifact.
    probe = _materialize(baseline, "probe")
    assert probe["output_records"] == 2
    fixed_row = next(
        json.loads(line)
        for line in (baseline["output_parent"] / "probe" / fixed2.MANIFEST_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["view"] == "fixed_value"
    )
    fixture = _fixture(
        tmp_path / "blocked",
        heldout_crop_sha=str(fixed_row["view_pixel_sha256"]),
    )
    with pytest.raises(ValueError, match="crosses declared val crop boundary"):
        _materialize(fixture, "blocked")
    assert not (fixture["output_parent"] / "blocked").exists()
    assert not list(fixture["output_parent"].glob(".blocked.*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("extra_key", "key set changed"),
        ("target_source", "target_source changed"),
        ("width", "dimensions changed"),
        ("group_closure", "group closure changed"),
        ("schema_bool_as_int", "schema_version changed"),
        ("eligible_bool_as_int", "optimizer_supervision_split_eligible changed"),
        ("consumable_bool_as_int", "optimizer_consumable changed"),
    ],
)
def test_verifier_rebuilds_exact_row_proof(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    fixture = _fixture(tmp_path)
    _materialize(fixture, "fixed2")
    root = fixture["output_parent"] / "fixed2"
    manifest = root / fixed2.MANIFEST_NAME
    analysis_manifest_bytes = manifest.read_bytes()
    rows = [
        json.loads(line)
        for line in analysis_manifest_bytes.decode("utf-8").splitlines()
    ]
    if mutation == "extra_key":
        rows[0]["forged"] = True
    elif mutation == "target_source":
        rows[0]["target_source"] = "forged"
    elif mutation == "width":
        rows[0]["view_width"] += 1
    elif mutation == "schema_bool_as_int":
        rows[0]["schema_version"] = True
    elif mutation == "eligible_bool_as_int":
        rows[0]["optimizer_supervision_split_eligible"] = 1
    elif mutation == "consumable_bool_as_int":
        rows[0]["optimizer_consumable"] = 0
    else:
        rows[0]["group_closure_sha256"] = "0" * 64
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    contract_path = root / fixed2.ANALYSIS_CONTRACT_NAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["train_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload = {key: value for key, value in contract.items() if key != "integrity_sha256"}
    contract["integrity_sha256"] = fixed2._canonical_sha256(payload)
    contract_path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        fixed2._verify_recipient_fixed2_teacher_analysis_test_only(export_root=root)


def test_full_scale_paths_do_not_repeat_global_scans_per_source() -> None:
    verifier_source = inspect.getsource(fixed2._verify_payload)
    assert "next(row for row in rows" not in verifier_source
    assert "manifest_rows_by_source[source_id][VIEWS[0]]" in verifier_source
    prepare_source = inspect.getsource(fixed2._prepare)
    assert prepare_source.count(
        "source_semantic_sha = _canonical_sha256(semantic_source_rows)"
    ) == 1
    assert (
        '"source_manifest_semantic_sha256": source_semantic_sha'
        in prepare_source
    )


def test_arbitrary_selected_png_remains_rejected_after_manifest_and_contract_reseal(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _materialize(fixture, "fixed2")
    root = fixture["output_parent"] / "fixed2"
    manifest = root / fixed2.MANIFEST_NAME
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    standard = next(row for row in rows if row["view"] == "standard")
    image = root / standard["image"]
    with Image.open(image) as opened:
        pixels = np.asarray(opened.convert("RGB")).copy()
    pixels[0, 0, 0] ^= np.uint8(0xFF)
    _write_png(image, pixels)
    standard["view_pixel_sha256"] = _crop_digest(pixels)
    standard["view_file_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    by_view = {str(row["view"]): row for row in rows}
    closure_payload = {
        "source_record_id": standard["source_record_id"],
        "source_group_id": standard["group_id"],
        "source_manifest_sha256": standard["target_source_manifest_sha256"],
        "target_sha256": standard["target_sha256"],
        "source_sha256": standard["source_sha256"],
        "result_json_sha256": standard["result_json_sha256"],
        "paddle_crop_pixel_sha256": standard["paddle_crop_pixel_sha256"],
        "views": [
            {
                "view": view,
                "pixel_sha256": by_view[view]["view_pixel_sha256"],
                "file_sha256": by_view[view]["view_file_sha256"],
            }
            for view in fixed2.VIEWS
        ],
    }
    closure = fixed2._canonical_sha256(closure_payload)
    for row in rows:
        row["group_closure_sha256"] = closure
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    contract_path = root / fixed2.ANALYSIS_CONTRACT_NAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["train_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload = {key: value for key, value in contract.items() if key != "integrity_sha256"}
    contract["integrity_sha256"] = fixed2._canonical_sha256(payload)
    contract_path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from recomputed standard view"):
        fixed2._verify_recipient_fixed2_teacher_analysis_test_only(export_root=root)


def test_no_clobber_preserves_foreign_output_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["output_parent"] / "occupied"
    output.mkdir()
    sentinel = output / "foreign.bin"
    sentinel.write_bytes(b"foreign-owner-bytes")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _materialize(fixture, "occupied")
    assert sentinel.read_bytes() == b"foreign-owner-bytes"
    assert {path.name for path in output.iterdir()} == {"foreign.bin"}


def test_late_output_race_is_no_replace_and_preserves_foreign_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["output_parent"] / "late-occupied"

    def occupy_destination(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        if checkpoint == "immediately_before_rename":
            output_root.mkdir()
            (output_root / "foreign.bin").write_bytes(b"late-foreign")

    monkeypatch.setattr(fixed2, "_publication_hook", occupy_destination)
    with pytest.raises(FileExistsError):
        _materialize(fixture, "late-occupied")
    assert (output / "foreign.bin").read_bytes() == b"late-foreign"
    assert {path.name for path in output.iterdir()} == {"foreign.bin"}
    retained = list(fixture["output_parent"].glob(".late-occupied.*.tmp"))
    assert len(retained) == 1
    assert (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).is_file()
    with pytest.raises(ValueError, match="nominal output root"):
        fixed2._verify_recipient_fixed2_teacher_analysis_test_only(
            export_root=retained[0]
        )


@pytest.mark.parametrize("artifact", ["source", "result_json", "paddle_crop"])
def test_late_bound_input_mutation_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    fixture = _fixture(tmp_path)
    first_row = json.loads(
        fixture["manifest"].read_text(encoding="utf-8").splitlines()[0]
    )
    if artifact == "source":
        bound_path = Path(first_row["source"])
    elif artifact == "result_json":
        bound_path = Path(first_row["result_json"])
    else:
        bound_path = fixture["dataset_root"] / first_row["slots"]["recipient_field"]["image"]
    original_hook = fixed2._publication_hook

    def mutate_source(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        original_hook(
            checkpoint,
            parent=parent,
            stage=stage,
            output_root=output_root,
        )
        if checkpoint == "immediately_before_rename":
            bound_path.write_bytes(bound_path.read_bytes() + b"late-mutation")

    monkeypatch.setattr(fixed2, "_publication_hook", mutate_source)
    with pytest.raises(ValueError, match="changed.*preflight") as caught:
        _materialize(fixture, f"late-{artifact}")
    assert not (fixture["output_parent"] / f"late-{artifact}").exists()
    retained = list(fixture["output_parent"].glob(f".late-{artifact}.*.tmp"))
    assert len(retained) == 1
    assert (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).is_file()
    assert "no files or directories were deleted" in caught.value.fixed2_teacher_quarantine


def test_contract_marker_is_last_and_failure_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def fail_before_marker(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        if checkpoint == "before_contract_commit":
            raise RuntimeError("injected marker boundary failure")

    monkeypatch.setattr(fixed2, "_publication_hook", fail_before_marker)
    output = fixture["output_parent"] / "marker-failure"
    with pytest.raises(RuntimeError, match="marker boundary") as caught:
        _materialize(fixture, "marker-failure")
    assert not output.exists()
    retained = list(fixture["output_parent"].glob(".marker-failure.*.tmp"))
    assert len(retained) == 1
    assert (retained[0] / fixed2.MANIFEST_NAME).is_file()
    assert (retained[0] / "images").is_dir()
    assert not (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).exists()
    assert "retained failure evidence" in caught.value.fixed2_teacher_quarantine


def test_full_stage_preverify_rejects_manifest_tamper_before_nominal_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def tamper_stage_manifest(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        del parent, output_root
        if checkpoint == "before_prepublication_verify":
            manifest = stage / fixed2.MANIFEST_NAME
            manifest.write_bytes(manifest.read_bytes() + b"{}\n")

    monkeypatch.setattr(fixed2, "_publication_hook", tamper_stage_manifest)
    output = fixture["output_parent"] / "preverify-tamper"
    with pytest.raises(ValueError, match="manifest binding changed") as caught:
        _materialize(fixture, "preverify-tamper")
    assert not output.exists()
    retained = list(fixture["output_parent"].glob(".preverify-tamper.*.tmp"))
    assert len(retained) == 1
    assert not (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).exists()
    assert "no files or directories were deleted" in caught.value.fixed2_teacher_quarantine


def test_before_contract_commit_manifest_tamper_cannot_acquire_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def tamper_renamed_manifest(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        del parent, output_root
        if checkpoint == "before_contract_commit":
            manifest = stage / fixed2.MANIFEST_NAME
            manifest.write_bytes(manifest.read_bytes() + b"{}\n")

    monkeypatch.setattr(fixed2, "_publication_hook", tamper_renamed_manifest)
    output = fixture["output_parent"] / "precommit-tamper"
    with pytest.raises(ValueError, match="manifest.*changed") as caught:
        _materialize(fixture, "precommit-tamper")
    assert not output.exists()
    retained = list(fixture["output_parent"].glob(".precommit-tamper.*.tmp"))
    assert len(retained) == 1
    assert (retained[0] / fixed2.MANIFEST_NAME).is_file()
    assert not (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).exists()
    assert "retained failure evidence" in caught.value.fixed2_teacher_quarantine


def test_manifest_tamper_inside_contract_writer_never_reaches_nominal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_write_file = fixed2._write_file

    def tampering_contract_write(
        lease: object, *, name: str, payload: bytes
    ) -> tuple[int, int, int, int, int, str]:
        if name == fixed2.ANALYSIS_CONTRACT_NAME:
            manifest = lease.path / fixed2.MANIFEST_NAME
            manifest.write_bytes(manifest.read_bytes() + b"{}\n")
        return original_write_file(lease, name=name, payload=payload)

    monkeypatch.setattr(fixed2, "_write_file", tampering_contract_write)
    output = fixture["output_parent"] / "marker-write-tamper"
    with pytest.raises(ValueError, match="manifest binding changed") as caught:
        _materialize(fixture, "marker-write-tamper")
    assert not output.exists()
    retained = list(fixture["output_parent"].glob(".marker-write-tamper.*.tmp"))
    assert len(retained) == 1
    assert (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).is_file()
    assert "retained failure evidence" in caught.value.fixed2_teacher_quarantine


def test_post_rename_manifest_tamper_is_moved_out_of_nominal_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def tamper_after_rename(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        del parent, stage
        if checkpoint == "immediately_after_rename":
            manifest = output_root / fixed2.MANIFEST_NAME
            manifest.write_bytes(manifest.read_bytes() + b"{}\n")

    monkeypatch.setattr(fixed2, "_publication_hook", tamper_after_rename)
    output = fixture["output_parent"] / "post-rename-tamper"
    with pytest.raises(ValueError, match="manifest binding changed") as caught:
        _materialize(fixture, "post-rename-tamper")
    assert not output.exists()
    retained = list(fixture["output_parent"].glob(".post-rename-tamper.*.failed"))
    assert len(retained) == 1
    assert (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).is_file()
    assert not list(fixture["output_parent"].glob(".post-rename-tamper.*.tmp"))
    assert "retained failure evidence" in caught.value.fixed2_teacher_quarantine


@pytest.mark.parametrize(
    "mutation",
    [
        "contract_extra_key",
        "hard_scheme_false",
        "hard_required_int",
        "publication_identity_false",
        "nominal_output_root",
        "view_geometry_false",
        "view_geometry_extra",
        "subject_domain",
        "subject_path_stable_false",
        "subject_output_stable_false",
        "subject_code_stable_false",
        "publication",
        "failure_policy",
        "source_split_false",
        "source_split_extra",
        "recipient_split_false",
        "output_split_false",
        "output_split_extra",
        "view_count_false",
        "view_count_extra",
        "selected_bool_as_int",
        "selected_int_as_bool",
        "selected_extra",
    ],
)
def test_contract_strict_fields_and_count_types_reject_resealed_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    _materialize(fixture, "fixed2")
    root = fixture["output_parent"] / "fixed2"
    marker = root / fixed2.ANALYSIS_CONTRACT_NAME
    contract = json.loads(marker.read_text(encoding="utf-8"))
    if mutation == "contract_extra_key":
        contract["forged"] = True
    elif mutation == "hard_scheme_false":
        contract["hard_attestation_scheme"] = False
    elif mutation == "hard_required_int":
        contract["public_verification_requires_hard_attestation"] = 0
    elif mutation == "publication_identity_false":
        contract["publication_identity"] = False
    elif mutation == "nominal_output_root":
        contract["nominal_output_root"] = str(root.with_name("forged-output"))
    elif mutation == "view_geometry_false":
        contract["view_geometry"]["standard"]["margin_ratio"] = False
    elif mutation == "view_geometry_extra":
        contract["view_geometry"]["standard"]["forged"] = 1
    elif mutation == "subject_domain":
        contract["subject_domain"] = "forged"
    elif mutation.endswith("_stable_false"):
        contract[mutation.removesuffix("_false")] = False
    elif mutation == "publication":
        contract["publication"] = "forged"
    elif mutation == "failure_policy":
        contract["failure_policy"] = "forged"
    elif mutation == "source_split_false":
        contract["source_manifest_split_counts"]["val"] = False
    elif mutation == "source_split_extra":
        contract["source_manifest_split_counts"]["forged"] = 0
    elif mutation == "recipient_split_false":
        contract["source_split_counts"]["val"] = False
    elif mutation == "output_split_false":
        contract["output_split_counts"]["train"] = False
    elif mutation == "output_split_extra":
        contract["output_split_counts"]["val"] = 0
    elif mutation == "view_count_false":
        contract["view_counts"]["standard"] = False
    elif mutation == "view_count_extra":
        contract["view_counts"]["forged"] = 0
    elif mutation == "selected_bool_as_int":
        contract["selected_view_hash_closure"]["decoded_pixels_reverified"] = 1
    elif mutation == "selected_int_as_bool":
        contract["selected_view_hash_closure"]["cross_split_conflicts"] = False
    else:
        contract["selected_view_hash_closure"]["forged"] = 0
    unsigned = {
        key: value for key, value in contract.items() if key != "integrity_sha256"
    }
    contract["integrity_sha256"] = fixed2._canonical_sha256(unsigned)
    marker.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        fixed2._verify_recipient_fixed2_teacher_analysis_test_only(export_root=root)


@pytest.mark.skipif(os.name == "nt", reason="POSIX replacement simulation")
def test_parent_replacement_never_mutates_foreign_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    parent = fixture["output_parent"]
    owned_parent = parent.with_name("outputs-owned")
    replaced = False

    def replace_parent(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        nonlocal replaced
        if checkpoint == "before_manifest_write" and not replaced:
            parent.rename(owned_parent)
            parent.mkdir()
            (parent / "foreign.bin").write_bytes(b"foreign")
            replaced = True

    monkeypatch.setattr(fixed2, "_publication_hook", replace_parent)
    with pytest.raises(ValueError, match="parent identity changed"):
        _materialize(fixture, "parent-race")
    assert (parent / "foreign.bin").read_bytes() == b"foreign"
    assert not (parent / "parent-race").exists()
    retained = list(owned_parent.glob(".parent-race.*.tmp"))
    assert len(retained) == 1
    assert not (retained[0] / fixed2.ANALYSIS_CONTRACT_NAME).exists()


@pytest.mark.skipif(os.name == "nt", reason="builds a POSIX analysis tree")
def test_analysis_profile_cannot_be_resealed_as_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _materialize(fixture, "fixed2")
    root = fixture["output_parent"] / "fixed2"
    manifest = root / fixed2.MANIFEST_NAME
    analysis_manifest_bytes = manifest.read_bytes()
    rows = [
        json.loads(line)
        for line in analysis_manifest_bytes.decode("utf-8").splitlines()
    ]
    for row in rows:
        row["kind"] = fixed2.RECORD_KIND
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    contract = json.loads(
        (root / fixed2.ANALYSIS_CONTRACT_NAME).read_text(encoding="utf-8")
    )
    analysis_contract_bytes = (root / fixed2.ANALYSIS_CONTRACT_NAME).read_bytes()
    contract.update(
        {
            "kind": fixed2.KIND,
            "record_kind": fixed2.RECORD_KIND,
            "publication_profile": "formal_windows_canonical_v1",
            "formal_windows_publication": True,
            "analysis_fixture": False,
            "publication_authority": fixed2.PUBLICATION_AUTHORITY,
            "hard_attestation_scheme": fixed2.HARD_ATTESTATION_SCHEME,
            "public_verification_requires_hard_attestation": True,
            "commit_marker": fixed2.CONTRACT_NAME,
            "train_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "publication_identity": fixed2._publication_identity(
                root_identity=fixed2._directory_identity(root),
                images_identity=fixed2._directory_identity(root / "images"),
                manifest_identity=fixed2._file_identity(manifest),
                image_identities={
                    image.name: fixed2._file_identity(image)
                    for image in (root / "images").iterdir()
                },
            ),
        }
    )
    payload = {key: value for key, value in contract.items() if key != "integrity_sha256"}
    contract["integrity_sha256"] = fixed2._canonical_sha256(payload)
    canonical = root / fixed2.CONTRACT_NAME
    canonical.write_text(json.dumps(contract), encoding="utf-8")
    canonical_snapshot, canonical_payload = fixed2._snapshot_json(
        canonical,
        description="forged formal marker",
        hook_prefix="forged_formal_marker",
    )
    (root / fixed2.ANALYSIS_CONTRACT_NAME).unlink()
    monkeypatch.setattr(fixed2, "_running_on_windows", lambda: True)
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SHA256",
        "0" * 64,
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SIZE_BYTES",
        1,
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_PRODUCER_SUBJECT_ID",
        "1" * 64,
    )

    # Even after deleting the analysis marker and rebuilding every self-signed
    # field/native identity, the tree cannot match the separately reviewed
    # raw-marker/size/semantic hard pins.
    with pytest.raises(ValueError, match="does not match hard attestation"):
        fixed2._verify_payload(
            canonical_payload,
            export_root=root,
            expected_kind=fixed2.KIND,
            expected_record_kind=fixed2.RECORD_KIND,
            expected_authority=fixed2.PUBLICATION_AUTHORITY,
            expected_contract_marker=fixed2.CONTRACT_NAME,
            require_publication_identity=True,
            require_hard_attestation=True,
            contract_snapshot=canonical_snapshot,
        )
    with pytest.raises(ValueError, match="does not match hard attestation"):
        fixed2.verify_recipient_fixed2_teacher(export_root=root)
    monkeypatch.setattr(fixed2, "_running_on_windows", lambda: False)
    canonical.unlink()
    manifest.write_bytes(analysis_manifest_bytes)
    (root / fixed2.ANALYSIS_CONTRACT_NAME).write_bytes(analysis_contract_bytes)
    assert fixed2._verify_recipient_fixed2_teacher_analysis_test_only(
        export_root=root
    )["kind"] == fixed2.ANALYSIS_KIND


def test_public_materialize_and_verify_fail_closed_off_windows(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("off-Windows boundary")
    with pytest.raises(OSError, match="requires Windows"):
        fixed2.materialize_recipient_fixed2_teacher(
            manifest=tmp_path / "must-not-open",
            output_root=tmp_path / "must-not-create",
        )
    with pytest.raises(OSError, match="requires Windows"):
        fixed2.verify_recipient_fixed2_teacher(export_root=tmp_path / "must-not-open")
    with pytest.raises(OSError, match="requires Windows"):
        fixed2.inspect_recipient_fixed2_teacher_attestation_candidate(
            export_root=tmp_path / "must-not-open"
        )


def test_analysis_materialize_and_verify_fail_closed_under_windows_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fixed2, "_running_on_windows", lambda: True)
    with pytest.raises(OSError, match="disabled on Windows"):
        fixed2._materialize_recipient_fixed2_teacher_analysis_test_only(
            manifest=tmp_path / "must-not-open",
            output_root=tmp_path / "must-not-create",
        )
    with pytest.raises(OSError, match="disabled on Windows"):
        fixed2._verify_recipient_fixed2_teacher_analysis_test_only(
            export_root=tmp_path / "must-not-open"
        )


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows NT filesystem primitives")
def test_windows_formal_materialize_and_verify_real_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["output_parent"] / "fixed2-formal"
    published = fixed2.materialize_recipient_fixed2_teacher(
        manifest=fixture["manifest"],
        dataset_contract=fixture["contract"],
        dataset_root=fixture["dataset_root"],
        output_root=output,
    )
    assert published["kind"] == fixed2.KIND
    assert published["publication_authority"] == fixed2.PUBLICATION_AUTHORITY
    assert published["publication_identity"]["scheme"] == (
        "native_directory_file_identity_and_image_closure_v1"
    )
    candidate = fixed2.inspect_recipient_fixed2_teacher_attestation_candidate(
        export_root=output
    )
    assert candidate["formal_authority_granted"] is False
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SHA256",
        None,
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SIZE_BYTES",
        None,
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_PRODUCER_SUBJECT_ID",
        None,
    )
    with pytest.raises(ValueError, match="not second-stage hard-attested"):
        fixed2.verify_recipient_fixed2_teacher(export_root=output)
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SHA256",
        candidate["contract_sha256"],
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SIZE_BYTES",
        candidate["contract_size_bytes"],
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_PRODUCER_SUBJECT_ID",
        candidate["producer_subject_id"],
    )
    assert fixed2.verify_recipient_fixed2_teacher(export_root=output) == published
    with pytest.raises(FileExistsError):
        fixed2.materialize_recipient_fixed2_teacher(
            manifest=fixture["manifest"],
            dataset_contract=fixture["contract"],
            dataset_root=fixture["dataset_root"],
            output_root=output,
        )
    selected_image = next((output / "images").iterdir())
    selected_image.write_bytes(selected_image.read_bytes() + b"identity-drift")
    with pytest.raises(ValueError, match="binding|identity|hash"):
        fixed2.verify_recipient_fixed2_teacher(export_root=output)


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows NT filesystem primitives")
def test_windows_formal_leases_deny_parent_and_stage_replacement_real_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    denied: set[str] = set()

    def probe_replacement_denials(
        checkpoint: str, *, parent: Path, stage: Path, output_root: Path
    ) -> None:
        if checkpoint != "before_manifest_write":
            return
        with pytest.raises(OSError):
            parent.rename(parent.with_name("parent-replaced"))
        denied.add("parent")
        with pytest.raises(OSError):
            stage.rename(stage.with_name(stage.name + ".replaced"))
        denied.add("stage")

    monkeypatch.setattr(fixed2, "_publication_hook", probe_replacement_denials)
    output = fixture["output_parent"] / "fixed2-formal-leases"
    fixed2.materialize_recipient_fixed2_teacher(
        manifest=fixture["manifest"],
        dataset_contract=fixture["contract"],
        dataset_root=fixture["dataset_root"],
        output_root=output,
    )
    candidate = fixed2.inspect_recipient_fixed2_teacher_attestation_candidate(
        export_root=output
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SHA256",
        candidate["contract_sha256"],
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_CONTRACT_SIZE_BYTES",
        candidate["contract_size_bytes"],
    )
    monkeypatch.setattr(
        fixed2_attestation,
        "ATTESTED_FIXED2_PRODUCER_SUBJECT_ID",
        candidate["producer_subject_id"],
    )
    assert denied == {"parent", "stage"}
    assert fixed2.verify_recipient_fixed2_teacher(export_root=output)["kind"] == fixed2.KIND


def test_ntcreatefile_pending_or_nonzero_completion_is_never_success() -> None:
    assert fixed2._failed_ntstatus(0, 0) is None
    assert fixed2._failed_ntstatus(0x103, 0) == 0x103
    assert fixed2._failed_ntstatus(0, -1073741771) == -1073741771


@pytest.mark.parametrize(
    ("returned_status", "completion_status", "translated", "error_type"),
    [
        (0x00000103, 0, 997, OSError),
        (0, -1073741771, 183, FileExistsError),
    ],
)
def test_ntcreatefile_mock_rejects_pending_and_nonzero_completion(
    monkeypatch: pytest.MonkeyPatch,
    returned_status: int,
    completion_status: int,
    translated: int,
    error_type: type[OSError],
) -> None:
    closed: list[int] = []
    translated_statuses: list[int] = []
    create_options: list[int] = []

    class FakeNtCreateFile:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            args[0]._obj.value = 733  # type: ignore[attr-defined]
            args[3]._obj.status_or_pointer.status = completion_status  # type: ignore[attr-defined]
            create_options.append(int(args[8]))
            return returned_status

    class FakeTranslate:
        argtypes: object = None
        restype: object = None

        def __call__(self, status: int) -> int:
            translated_statuses.append(status)
            return translated

    class FakeNtdll:
        NtCreateFile = FakeNtCreateFile()
        RtlNtStatusToDosError = FakeTranslate()

    monkeypatch.setattr(
        fixed2.ctypes,
        "WinDLL",
        lambda name, **_kwargs: FakeNtdll() if name == "ntdll" else None,
        raising=False,
    )
    monkeypatch.setattr(fixed2, "_windows_close", closed.append)

    with pytest.raises(error_type) as caught:
        fixed2._windows_nt_directory(
            701,
            name="stage",
            disposition=fixed2._WINDOWS_FILE_CREATE,
            desired_access=(
                fixed2._WINDOWS_DIRECTORY_ACCESS
                | fixed2._WINDOWS_DELETE
                | fixed2._WINDOWS_SYNCHRONIZE
            ),
            share_access=fixed2._WINDOWS_DIRECTORY_SHARE,
        )
    assert caught.value.errno == translated
    assert closed == [733]
    assert translated_statuses == [returned_status or completion_status]
    assert create_options[0] & fixed2._WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT


def test_cli_exposes_materialize_verify_and_candidate_inspection() -> None:
    help_text = fixed2.build_parser().format_help()
    assert "materialize" in help_text
    assert "verify" in help_text
    assert "inspect-candidate" in help_text


@pytest.mark.parametrize("command", ["materialize", "verify", "inspect-candidate"])
def test_cli_routes_materialize_and_verify(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    observed: dict[str, object] = {}
    if command == "materialize":
        monkeypatch.setattr(
            fixed2,
            "materialize_recipient_fixed2_teacher",
            lambda **kwargs: observed.update(kwargs) or {"kind": fixed2.KIND},
        )
        fixed2.main(
            [
                "materialize",
                "--manifest",
                "blind.jsonl",
                "--dataset-contract",
                "dataset.contract.json",
                "--dataset-root",
                "dataset",
                "--output",
                "fixed2",
            ]
        )
        assert observed["manifest"] == Path("blind.jsonl")
        assert observed["output_root"] == Path("fixed2")
    elif command == "verify":
        monkeypatch.setattr(
            fixed2,
            "verify_recipient_fixed2_teacher",
            lambda **kwargs: observed.update(kwargs) or {"kind": fixed2.KIND},
        )
        fixed2.main(["verify", "--export-root", "fixed2"])
        assert observed["export_root"] == Path("fixed2")
    else:
        monkeypatch.setattr(
            fixed2,
            "inspect_recipient_fixed2_teacher_attestation_candidate",
            lambda **kwargs: observed.update(kwargs)
            or {"kind": "receipt_recipient_fixed2_teacher_attestation_candidate_v1"},
        )
        fixed2.main(["inspect-candidate", "--export-root", "fixed2"])
        assert observed["export_root"] == Path("fixed2")
    expected_kind = (
        "receipt_recipient_fixed2_teacher_attestation_candidate_v1"
        if command == "inspect-candidate"
        else fixed2.KIND
    )
    assert json.loads(capsys.readouterr().out)["kind"] == expected_kind
