from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.ocr import OCRResult, _extract_paddle_lines
from transfer_receipt_ai.paddle_recipient_evaluate import (
    build_parser,
    evaluate_paddle_recipients,
    format_paddle_recipient_evaluation,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _record(receipt_id: str, *, split: str, text: str, image: str) -> dict[str, object]:
    return {
        "id": receipt_id,
        "split": split,
        "group_id": f"group:{receipt_id}",
        "slots": {
            "recipient_field": {
                "text": text,
                "image": image,
                "recipient_visible_text": f"收款方 {text}",
                "recipient_value": text,
            }
        },
    }


class _FakeReader:
    def __init__(self, results: list[OCRResult]) -> None:
        self._results = iter(results)

    def recognize(self, _image_rgb: np.ndarray) -> OCRResult:
        return next(self._results)


class _DetRecordingFakeReader:
    def __init__(self, results: list[OCRResult]) -> None:
        self._results = iter(results)
        self.det_calls: list[bool] = []

    def recognize(self, _image_rgb: np.ndarray, *, det: bool = True) -> OCRResult:
        self.det_calls.append(det)
        return next(self._results)


def test_paddle_recipient_evaluation_uses_value_extraction_and_writes_new_output(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.fromarray(np.full((5, 8, 3), 255, dtype=np.uint8), mode="RGB").save(crops / name)
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(
        manifest,
        [
            _record("val-a", split="val", text="商户甲", image="a.png"),
            _record("val-b", split="val", text="商户乙", image="b.png"),
            _record("test-c", split="test", text="商户丙", image="c.png"),
        ],
    )
    reader = _FakeReader(
        [
            OCRResult(text="收款方 商户甲", confidence=0.99, lines=(("收款方 商户甲", 0.99),)),
            OCRResult(text="商户乙 收款方", confidence=0.91, lines=(("商户乙 收款方", 0.91),)),
        ]
    )
    output = tmp_path / "out"

    summary, accepted = evaluate_paddle_recipients(
        manifest_path=manifest,
        dataset_root=crops,
        output_dir=output,
        split="val",
        device="cuda",
        target_value_exact_match=0.50,
        progress_every=1,
        reader_factory=lambda _device: reader,
        runtime_probe=lambda _reader: {
            "paddleocr_version": "2.10.0",
            "paddle_version": "test",
            "active_paddle_device": "gpu:0",
            "torch_imported": False,
        },
    )

    assert accepted
    assert summary["records"] == 2
    assert summary["anchored_value_exact_matches"] == 1
    assert summary["anchored_value_exact_match"] == pytest.approx(0.5)
    assert summary["fallback_value_exact_matches"] == 2
    assert summary["anchor_parse_failure_records"] == 1
    assert summary["candidate_extraction_modes"] == {"anchor_parse_failed": 1, "anchored": 1}
    assert summary["inference_mode"] == {
        "name": "full_det_cls_rec",
        "experimental": False,
        "detection_enabled": True,
        "angle_classifier_enabled": True,
        "recognizer_enabled": True,
    }
    rows = [json.loads(line) for line in (output / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["candidate_anchored_value"] == "商户甲"
    assert rows[0]["anchored_value_exact"] is True
    assert rows[1]["candidate_anchored_value"] is None
    assert rows[1]["candidate_fallback_value"] == "商户乙"
    assert rows[1]["anchored_value_exact"] is False
    assert rows[1]["fallback_value_exact"] is True
    assert len((output / "disagreements.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    text = format_paddle_recipient_evaluation(summary)
    assert "anchored=1/2=50.00%" in text
    assert "mode=full_det_cls_rec; experimental=False; det=True; cls=True; rec=True" in text
    assert "active:gpu:0" in text


def test_paddle_recipient_skip_detection_is_explicit_and_keeps_strict_metrics(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.fromarray(np.full((5, 8, 3), 255, dtype=np.uint8), mode="RGB").save(crops / "a.png")
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(manifest, [_record("val-a", split="val", text="商户甲", image="a.png")])
    reader = _DetRecordingFakeReader(
        [OCRResult(text="收款方 商户甲", confidence=0.99, lines=(("收款方 商户甲", 0.99),))]
    )

    summary, accepted = evaluate_paddle_recipients(
        manifest_path=manifest,
        dataset_root=crops,
        output_dir=tmp_path / "skip-det-out",
        split="val",
        device="cuda",
        target_value_exact_match=0.90,
        progress_every=1,
        skip_detection=True,
        reader_factory=lambda _device: reader,
        runtime_probe=lambda _reader: {"active_paddle_device": "gpu:0", "torch_imported": False},
    )

    assert accepted
    assert reader.det_calls == [False]
    assert summary["inference_mode"] == {
        "name": "experimental_skip_det_cls_rec",
        "experimental": True,
        "detection_enabled": False,
        "angle_classifier_enabled": True,
        "recognizer_enabled": True,
    }
    assert summary["anchored_value_exact_match"] == pytest.approx(1.0)
    assert summary["raw_visible_exact_match"] == pytest.approx(1.0)
    assert summary["fallback_value_exact_match"] == pytest.approx(1.0)
    comparison = json.loads((tmp_path / "skip-det-out" / "comparisons.jsonl").read_text(encoding="utf-8"))
    assert comparison["inference_mode"] == "experimental_skip_det_cls_rec"
    assert (
        "mode=experimental_skip_det_cls_rec; experimental=True; det=False; cls=True; rec=True"
        in format_paddle_recipient_evaluation(summary)
    )


def test_paddle_recognizer_only_payload_lines_are_supported() -> None:
    lines = _extract_paddle_lines([[("收款方 商户甲", 0.99)]])
    assert lines[0][0] == "收款方 商户甲"
    assert lines[0][1] == pytest.approx(0.99)


def test_paddle_recipient_cli_exposes_experimental_skip_detection_without_changing_default() -> None:
    parser = build_parser()
    default_args = parser.parse_args(["--manifest", "records.jsonl", "--dataset-root", "crops", "--output", "out"])
    experimental_args = parser.parse_args(
        ["--manifest", "records.jsonl", "--dataset-root", "crops", "--output", "out", "--limit", "100", "--skip-detection"]
    )

    assert default_args.skip_detection is False
    assert default_args.limit is None
    assert experimental_args.skip_detection is True
    assert experimental_args.limit == 100


def test_paddle_recipient_evaluation_rejects_cpu_fallback_and_output_reuse(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.fromarray(np.full((5, 8, 3), 255, dtype=np.uint8), mode="RGB").save(crops / "a.png")
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(manifest, [_record("val-a", split="val", text="商户甲", image="a.png")])
    reader_factory = lambda _device: _FakeReader([OCRResult(text="商户甲", confidence=0.99)])
    with pytest.raises(RuntimeError, match="active device"):
        evaluate_paddle_recipients(
            manifest_path=manifest,
            dataset_root=crops,
            output_dir=tmp_path / "bad-device",
            device="cuda",
            reader_factory=reader_factory,
            runtime_probe=lambda _reader: {"active_paddle_device": "cpu", "torch_imported": False},
        )
    reused = tmp_path / "reuse"
    reused.mkdir()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        evaluate_paddle_recipients(
            manifest_path=manifest,
            dataset_root=crops,
            output_dir=reused,
            device="cpu",
            reader_factory=reader_factory,
            runtime_probe=lambda _reader: {"active_paddle_device": "cpu", "torch_imported": False},
        )


def test_paddle_recipient_evaluation_rejects_crop_outside_root(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    outside = tmp_path / "outside.png"
    Image.fromarray(np.full((5, 8, 3), 255, dtype=np.uint8), mode="RGB").save(outside)
    manifest = tmp_path / "unified_fields.jsonl"
    _write_jsonl(manifest, [_record("val-a", split="val", text="商户甲", image="../outside.png")])

    with pytest.raises(ValueError, match="escapes dataset root"):
        evaluate_paddle_recipients(
            manifest_path=manifest,
            dataset_root=crops,
            output_dir=tmp_path / "out",
            device="cpu",
        )


def test_paddle_recipient_evaluation_rejects_manifest_without_strict_anchor(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.fromarray(np.full((5, 8, 3), 255, dtype=np.uint8), mode="RGB").save(crops / "a.png")
    manifest = tmp_path / "unified_fields.jsonl"
    row = _record("val-a", split="val", text="商户甲", image="a.png")
    row["slots"]["recipient_field"]["recipient_visible_text"] = "商户甲 收款方"
    _write_jsonl(manifest, [row])

    with pytest.raises(ValueError, match="does not strictly anchor"):
        evaluate_paddle_recipients(
            manifest_path=manifest,
            dataset_root=crops,
            output_dir=tmp_path / "out",
            device="cpu",
        )


def test_paddle_recipient_wrapper_parses_when_powershell_is_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is not installed in this local test environment")
    wrapper = Path(__file__).parents[1] / "scripts" / "receipt-ocr-paddle-recipient-4090.ps1"
    escaped_wrapper = wrapper.as_posix().replace("'", "''")
    parser_command = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped_wrapper}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", parser_command],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert "[switch]$SkipDetection" in wrapper_text
    assert '"--skip-detection"' in wrapper_text
