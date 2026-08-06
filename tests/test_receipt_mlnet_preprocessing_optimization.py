from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_production_preprocessing_uses_row_spans_without_detector_canvas() -> None:
    program = (ROOT / "dotnet" / "ReceiptMlNet.Cli" / "Program.cs").read_text(
        encoding="utf-8"
    )
    detector = _between(
        program,
        "public static DetectorInputTensor PrepareDetectorInput(",
        "public static float[] PrepareStatusbarInput(",
    )
    statusbar = _between(
        program,
        "public static float[] PrepareStatusbarInput(",
        "internal sealed record DetectorInputTensor(",
    )

    assert "KnownResamplers.Triangle" in detector
    assert "ProcessPixelRows" in detector
    assert "GetRowSpan" in detector
    assert "new Image<Rgb24>(DetectorWidth, DetectorHeight)" not in detector
    assert ".DrawImage(" not in detector

    assert "KnownResamplers.Bicubic" in statusbar
    assert "ProcessPixelRows" in statusbar
    assert "GetRowSpan" in statusbar


def test_csharp_contract_harness_compares_legacy_and_optimized_float_bits() -> None:
    harness = (
        ROOT / "dotnet" / "ReceiptMlNet.Cli.PreprocessingContractTests" / "Program.cs"
    ).read_text(encoding="utf-8")

    assert "LegacyPrepareDetectorInput" in harness
    assert "LegacyPrepareStatusbarInput" in harness
    assert "BitConverter.SingleToInt32Bits" in harness
    assert "VerifyCase(7, 11)" in harness
    assert "VerifyCase(13, 5)" in harness
    assert "VerifyCase(9, 16)" in harness
    assert "VerifyCase(1179, 2556)" in harness
