from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.ocr import OCRResult
import transfer_receipt_ai.recipient_value_view_teacher as teacher_module
from transfer_receipt_ai.recipient_value_view_teacher import (
    MINIMUM_CONFIDENCE,
    TARGET_EXACT_MATCH,
    VALUE_VIEW_LEFT_TRIM,
    _value_view,
    _publish_fresh_summary_last,
    build_parser,
    evaluate_value_view_teacher,
)


def _write_manifest(
    path: Path,
    image: str,
    *,
    text: str = "商户甲",
    crop_sha256: str | None = None,
) -> None:
    row = {
        "id": "val-one",
        "group_id": "group:one",
        "split": "val",
        "slots": {
            "recipient_field": {
                "text": text,
                "image": image,
                "recipient_visible_text": f"收款方 {text}",
                "recipient_value": text,
                **({"crop_sha256": crop_sha256} if crop_sha256 is not None else {}),
            }
        },
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _decoded_crop_sha256(path: Path) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    digest = hashlib.sha256()
    digest.update(str(pixels.shape).encode("ascii"))
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


class _Reader:
    def __init__(self, result: OCRResult) -> None:
        self.result = result
        self.det_calls: list[bool] = []
        self.widths: list[int] = []

    def recognize(self, image: np.ndarray, *, det: bool = True) -> OCRResult:
        self.det_calls.append(det)
        self.widths.append(int(image.shape[1]))
        return self.result


def test_value_view_uses_exact_v13_rounding_and_keeps_rgb() -> None:
    image = np.zeros((5, 11, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(11, dtype=np.uint8)

    view = _value_view(image)

    assert VALUE_VIEW_LEFT_TRIM == 0.30
    assert MINIMUM_CONFIDENCE == 0.80
    assert TARGET_EXACT_MATCH == 0.90
    assert view.shape == (5, 8, 3)
    assert view.flags.c_contiguous
    assert view[0, 0, 0] == 3


def test_teacher_ceiling_is_parser_free_exact_and_summary_last(tmp_path: Path) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((12, 20, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    reader = _Reader(
        OCRResult(text=" 商户甲 ", confidence=0.99, lines=(("商户甲", 0.99),))
    )
    output = tmp_path / "out"

    summary, accepted = evaluate_value_view_teacher(
        manifest_path=manifest,
        dataset_root=root,
        output_dir=output,
        split="val",
        device="cuda:0",
        target_exact_match=0.90,
        progress_every=1,
        limit=1,
        reader_factory=lambda _device: reader,
        runtime_probe=lambda _reader: {
            "active_paddle_device": "gpu:0",
            "torch_imported": False,
        },
    )

    assert accepted is True
    assert reader.det_calls == [False]
    assert reader.widths == [14]
    assert summary["candidate_coverage"] == pytest.approx(1.0)
    assert summary["exact_match"] == pytest.approx(1.0)
    assert summary["production_route_authorized"] is False
    assert summary["inference_mode"]["parser_enabled"] is False
    comparison = json.loads((output / "comparisons.jsonl").read_text(encoding="utf-8"))
    assert comparison["candidate_text"] == "商户甲"
    assert comparison["confidence_eligible"] is True
    assert comparison["raw_exact"] is True
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=output,
            split="val",
            device="cpu",
            limit=1,
            reader_factory=lambda _device: reader,
            runtime_probe=lambda _reader: {
                "active_paddle_device": "cpu",
                "torch_imported": False,
            },
        )


def test_teacher_ceiling_rejects_missing_candidate_and_does_not_parse_anchor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    reader = _Reader(OCRResult(text="收款方 商户甲", confidence=0.99))

    summary, accepted = evaluate_value_view_teacher(
        manifest_path=manifest,
        dataset_root=root,
        output_dir=tmp_path / "out",
        device="cpu",
        limit=1,
        reader_factory=lambda _device: reader,
        runtime_probe=lambda _reader: {
            "active_paddle_device": "cpu",
            "torch_imported": False,
        },
    )

    assert accepted is False
    assert summary["exact_match"] == 0.0
    assert summary["decision"] == "analysis_only_teacher_ceiling_fail_stop"


def test_teacher_ceiling_keeps_strict_confidence_floor(tmp_path: Path) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    reader = _Reader(OCRResult(text="商户甲", confidence=0.799999))

    summary, accepted = evaluate_value_view_teacher(
        manifest_path=manifest,
        dataset_root=root,
        output_dir=tmp_path / "out",
        device="cpu",
        limit=1,
        reader_factory=lambda _device: reader,
        runtime_probe=lambda _reader: {
            "active_paddle_device": "cpu",
            "torch_imported": False,
        },
    )

    assert accepted is False
    assert summary["candidate_records"] == 0
    comparison = json.loads((tmp_path / "out" / "comparisons.jsonl").read_text())
    assert comparison["candidate_text"] is None
    assert comparison["confidence_eligible"] is False


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.01, 1.01])
def test_teacher_ceiling_rejects_nonfinite_or_out_of_range_confidence(
    tmp_path: Path, confidence: float
) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    reader = _Reader(OCRResult(text="商户甲", confidence=confidence))

    with pytest.raises(ValueError, match=r"finite probability in \[0, 1\]"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=tmp_path / "out",
            device="cpu",
            limit=1,
            reader_factory=lambda _device: reader,
            runtime_probe=lambda _reader: {
                "active_paddle_device": "cpu",
                "torch_imported": False,
            },
        )
    assert not (tmp_path / "out").exists()


def test_teacher_ceiling_rejects_nonfinite_line_confidence(tmp_path: Path) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    reader = _Reader(
        OCRResult(
            text="商户甲",
            confidence=0.99,
            lines=(("商户甲", float("nan")),),
        )
    )
    with pytest.raises(ValueError, match="line 1 confidence must be a finite probability"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=tmp_path / "out",
            device="cpu",
            limit=1,
            reader_factory=lambda _device: reader,
            runtime_probe=lambda _reader: {
                "active_paddle_device": "cpu",
                "torch_imported": False,
            },
        )


def test_full_run_requires_bundle_and_exact_6789_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="must bind an immutable Paddle audit bundle"):
        evaluate_value_view_teacher(
            manifest_path=tmp_path / "missing-records.jsonl",
            dataset_root=tmp_path / "missing-crops",
            output_dir=tmp_path / "out",
            split="val",
            device="cuda:0",
            limit=None,
        )

    root = tmp_path / "crops"
    root.mkdir()
    crop = root / "one.png"
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(crop)
    manifest = tmp_path / "records.jsonl"
    _write_manifest(
        manifest,
        "one.png",
        crop_sha256=_decoded_crop_sha256(crop),
    )
    monkeypatch.setattr(teacher_module, "EXPECTED_FULL_VAL_RECORDS", 2)
    with pytest.raises(ValueError, match="exactly 2 records, got 1"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=tmp_path / "full-out",
            split="val",
            device="cuda:0",
            limit=None,
            bundle_dir=tmp_path / "immutable-audit-bundle",
        )


def test_split_target_and_requested_limit_are_hard_locked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hard-locked to val"):
        evaluate_value_view_teacher(
            manifest_path=tmp_path / "missing.jsonl",
            dataset_root=tmp_path / "missing-crops",
            output_dir=tmp_path / "out-test",
            split="test",
            limit=1,
        )
    with pytest.raises(ValueError, match="hard-locked to exactly 0.90"):
        evaluate_value_view_teacher(
            manifest_path=tmp_path / "missing.jsonl",
            dataset_root=tmp_path / "missing-crops",
            output_dir=tmp_path / "out-low-target",
            target_exact_match=0.10,
            limit=1,
        )
    with pytest.raises(ValueError, match="pilot limit must be smaller than 6789"):
        evaluate_value_view_teacher(
            manifest_path=tmp_path / "missing.jsonl",
            dataset_root=tmp_path / "missing-crops",
            output_dir=tmp_path / "out-full-bypass",
            limit=6789,
        )

    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    with pytest.raises(ValueError, match="Requested limit=100.*selected exactly 1"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=tmp_path / "out-short-limit",
            device="cpu",
            limit=100,
        )


def test_output_and_stage_are_isolated_from_all_inputs(tmp_path: Path) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    with pytest.raises(ValueError, match="overlaps protected dataset root"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=root / "polluting-output",
            device="cpu",
            limit=1,
        )

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(ValueError, match="overlaps protected frozen bundle"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=bundle / "polluting-output",
            device="cpu",
            limit=1,
            bundle_dir=bundle,
        )

    live_model = tmp_path / "live-rec-model"
    live_model.mkdir()
    reader = _Reader(OCRResult(text="商户甲", confidence=0.99))
    reader._engine = SimpleNamespace(
        args=SimpleNamespace(rec_model_dir=live_model.as_posix())
    )
    with pytest.raises(ValueError, match="overlaps protected live recognizer model"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=live_model / "polluting-output",
            device="cpu",
            limit=1,
            reader_factory=lambda _device: reader,
        )


def test_dangling_output_symlink_and_symlink_ancestor_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(root / "one.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")

    dangling_target = tmp_path / "not-created"
    requested = tmp_path / "requested-output"
    try:
        requested.symlink_to(dangling_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"test host cannot create a directory symlink: {error}")
    with pytest.raises(ValueError, match="existing, symlink, or reparse"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=requested,
            device="cpu",
            limit=1,
        )
    assert requested.is_symlink()
    assert not dangling_target.exists()

    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="must not traverse a symlink/junction/reparse"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=linked_parent / "out",
            device="cpu",
            limit=1,
        )
    assert not (real_parent / "out").exists()


def test_bundle_and_live_assets_are_verified_four_times_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    crop = root / "one.png"
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(crop)
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png", crop_sha256=_decoded_crop_sha256(crop))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    calls: list[int] = []

    def verify(_reader: object, path: Path) -> dict[str, object]:
        assert path == bundle.resolve()
        calls.append(len(calls) + 1)
        return {"verified": True, "identity": "same"}

    monkeypatch.setattr(teacher_module, "_verify_reader_matches_bundle", verify)
    reader = _Reader(OCRResult(text="商户甲", confidence=0.99))
    summary, accepted = evaluate_value_view_teacher(
        manifest_path=manifest,
        dataset_root=root,
        output_dir=tmp_path / "verified-output",
        device="cpu",
        limit=1,
        bundle_dir=bundle,
        reader_factory=lambda _device: reader,
        runtime_probe=lambda _reader: {
            "active_paddle_device": "cpu",
            "torch_imported": False,
        },
    )
    assert accepted is True
    assert calls == [1, 2, 3, 4]
    assert summary["frozen_bundle"]["verification_passes"] == 4


def test_all_crop_hashes_are_rechecked_after_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    crop = root / "one.png"
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(crop)
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, "one.png")
    reader = _Reader(OCRResult(text="商户甲", confidence=0.99))
    original_atomic_json = teacher_module._atomic_json

    def mutate_after_summary(path: Path, value: dict[str, object]) -> None:
        original_atomic_json(path, value)
        crop.write_bytes(b"changed-after-inference")

    monkeypatch.setattr(teacher_module, "_atomic_json", mutate_after_summary)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="crop changed during"):
        evaluate_value_view_teacher(
            manifest_path=manifest,
            dataset_root=root,
            output_dir=output,
            split="val",
            device="cpu",
            limit=1,
            reader_factory=lambda _device: reader,
            runtime_probe=lambda _reader: {
                "active_paddle_device": "cpu",
                "torch_imported": False,
            },
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".out.*.tmp"))


def test_fresh_publish_never_clobbers_and_exposes_summary_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("comparisons.jsonl", "disagreements.jsonl", "summary.json"):
        (stage / name).write_text(name, encoding="utf-8")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "sentinel.txt"
    sentinel.write_text("rival", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _publish_fresh_summary_last(stage=stage, output=occupied)
    assert sentinel.read_text(encoding="utf-8") == "rival"
    assert (stage / "summary.json").is_file()

    published = tmp_path / "published"
    links: list[str] = []
    original_link = os.link

    def recording_link(source: Path, target: Path) -> None:
        links.append(Path(source).name)
        original_link(source, target)

    monkeypatch.setattr(os, "link", recording_link)
    _publish_fresh_summary_last(stage=stage, output=published)
    assert links == ["comparisons.jsonl", "disagreements.jsonl", "summary.json"]
    assert (published / "summary.json").is_file()
    assert not stage.exists()


def test_competing_same_name_is_never_overwritten_and_cleanup_is_non_recursive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("comparisons.jsonl", "disagreements.jsonl", "summary.json"):
        (stage / name).write_text(name, encoding="utf-8")
    output = tmp_path / "output"
    original_link = os.link

    def race_on_first_link(source: Path, target: Path) -> None:
        if Path(source).name == "comparisons.jsonl":
            Path(target).write_text("rival", encoding="utf-8")
            (target.parent / "foreign.txt").write_text("foreign", encoding="utf-8")
        original_link(source, target)

    monkeypatch.setattr(os, "link", race_on_first_link)
    with pytest.raises(ValueError, match="Refusing to overwrite competing"):
        _publish_fresh_summary_last(stage=stage, output=output)
    assert (output / "comparisons.jsonl").read_text(encoding="utf-8") == "rival"
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (stage / "comparisons.jsonl").is_file()
    assert (stage / "disagreements.jsonl").is_file()
    assert (stage / "summary.json").is_file()


def test_cli_has_no_tunable_crop_or_parser_switch() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--manifest", "records.jsonl", "--dataset-root", "crops", "--output", "out"]
    )
    assert not hasattr(args, "split")
    assert not hasattr(args, "target")
    assert not hasattr(args, "left_trim")
    assert not hasattr(args, "parser")


def test_4090_launcher_fixes_cuda_target_and_value_view_policy() -> None:
    launcher = (
        Path(__file__).parents[1]
        / "scripts"
        / "receipt-ocr-recipient-value-view-teacher-4090.ps1"
    ).read_text(encoding="utf-8")
    assert '"--device", "cuda:0"' in launcher
    assert '"--split"' not in launcher
    assert '"--target"' not in launcher
    assert "left trim 30%" in launcher
    assert "confidence>=0.80" in launcher
    assert "det/parser/full-layout disabled" in launcher
    assert "left-trim" not in launcher
    assert "Refusing to reuse recipient value-view teacher output" in launcher
    assert "A full 6789-record val ceiling must bind" in launcher
    assert "[ValidateRange(0, 6788)]" in launcher
    assert "ValidateSet(\"val\", \"test\")" not in launcher
    for unsupported in ("??", "&&", "||", "ForEach-Object -Parallel"):
        assert unsupported not in launcher


def test_4090_launcher_parses_with_windows_powershell_when_available() -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is unavailable on this host")
    launcher = (
        Path(__file__).parents[1]
        / "scripts"
        / "receipt-ocr-recipient-value-view-teacher-4090.ps1"
    )
    escaped = launcher.as_posix().replace("'", "''")
    command = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped}', "
        "[ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
