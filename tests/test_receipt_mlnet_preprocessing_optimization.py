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
    assert "VerifyCase(7, 11, reusableDetectorBuffer, reusableStatusbarBuffer)" in harness
    assert "VerifyUnifiedFieldTensorReuse()" in harness
    assert "VerifyCase(13, 5, reusableDetectorBuffer, reusableStatusbarBuffer)" in harness
    assert "VerifyCase(9, 16, reusableDetectorBuffer, reusableStatusbarBuffer)" in harness
    assert "VerifyCase(1179, 2556, reusableDetectorBuffer, reusableStatusbarBuffer)" in harness


def test_unified_runtime_decodes_native_output_views_without_array_copies() -> None:
    engine = (
        ROOT / "dotnet" / "ReceiptMlNet.Cli" / "UnifiedOcrEngine.cs"
    ).read_text(encoding="utf-8")

    assert "ReadOutputViews(runtimeOutputs)" in engine
    assert "denseTensor.Buffer" in engine
    assert "output.Values.Span" in engine
    assert "tensor.ToArray()" not in engine


def test_detector_cpu_thread_tuning_is_explicit_and_auditable() -> None:
    program = (ROOT / "dotnet" / "ReceiptMlNet.Cli" / "Program.cs").read_text(
        encoding="utf-8"
    )

    assert 'case "--detector-intra-op-threads"' in program
    assert '--detector-intra-op-threads requires --device cpu' in program
    assert "IntraOpNumThreads = intraOpThreads" in program
    assert "options.DetectorIntraOpThreads" in program
    assert "int? DetectorIntraOpThreads" in program
