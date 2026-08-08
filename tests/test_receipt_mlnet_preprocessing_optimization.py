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
    assert "VerifyUnifiedGrayscaleRowSpanBitExactness()" in harness
    assert "new Random(0x5EED)" in harness
    assert "new Random(0xD1A5)" in harness
    assert "dimensionRandom.Next(1, 513)" in harness
    assert "dimensionRandom.Next(1, 33)" in harness
    assert "boundaryExpectedBytes" in harness
    assert "row-span grayscale rounding boundaries" in harness
    assert "VerifyCase(13, 5, reusableDetectorBuffer, reusableStatusbarBuffer)" in harness
    assert "VerifyCase(9, 16, reusableDetectorBuffer, reusableStatusbarBuffer)" in harness
    assert "VerifyCase(1179, 2556, reusableDetectorBuffer, reusableStatusbarBuffer)" in harness


def test_unified_grayscale_uses_paired_row_spans_with_frozen_luma_arithmetic() -> None:
    production = (
        ROOT / "dotnet" / "ReceiptMlNet.Cli" / "UnifiedOcrImageOps.cs"
    ).read_text(encoding="utf-8")
    grayscale = production.split(
        "private static Image<L8> ToGrayscale(Image<Rgb24> source)", 1
    )[1]
    harness = (
        ROOT / "dotnet" / "ReceiptMlNet.Cli.PreprocessingContractTests" / "Program.cs"
    ).read_text(encoding="utf-8")
    legacy = _between(
        harness,
        "private static float[] LegacyPrepareFieldTensor(",
        "private static void AssertArrayBitsEqual(",
    )

    assert "source.ProcessPixelRows(grayscale" in grayscale
    assert "sourceAccessor.GetRowSpan(y)" in grayscale
    assert "grayscaleAccessor.GetRowSpan(y)" in grayscale
    assert "source[x, y]" not in grayscale
    assert "grayscale[x, y]" not in grayscale
    assert "pixel.R * 0.299 + pixel.G * 0.587 + pixel.B * 0.114" in grayscale
    assert "MidpointRounding.ToEven" in grayscale
    assert "image[x, y]" in legacy
    assert "grayscale[x, y]" in legacy


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
