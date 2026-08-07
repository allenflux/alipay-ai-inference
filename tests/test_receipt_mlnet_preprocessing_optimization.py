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


def test_unified_runtime_reuses_bound_input_and_output_ort_values() -> None:
    engine = (
        ROOT / "dotnet" / "ReceiptMlNet.Cli" / "UnifiedOcrEngine.cs"
    ).read_text(encoding="utf-8")

    constructor = _between(
        engine,
        "public UnifiedOcrEngine(UnifiedOcrBundle bundle, DeviceSetting requestedDevice)",
        "public string ExecutionProvider",
    )
    recognize = _between(
        engine,
        "public UnifiedOcrReadResult RecognizeReceipt(",
        "private static double StopAndReadMilliseconds(",
    )
    runtime = _between(
        engine,
        "private sealed class UnifiedOcrRuntimeBuffers : IDisposable",
        "private sealed record OrtOutput",
    )

    assert constructor.count("UnifiedOcrRuntimeBuffers.Create(") == 1
    assert runtime.count("OrtValue.CreateTensorValueFromMemory(") == 3
    assert runtime.count("binding.BindInput(") == 2
    assert "binding.BindOutput(UnifiedOcrBundle.OutputNames[index], outputValues[index])" in runtime
    assert recognize.count("_session.RunWithBinding(_runtime.RunOptions, _runtime.Binding)") == 1
    assert "var outputs = _runtime.Outputs" in recognize
    assert "output.Values.Span" in engine
    assert "tensor.ToArray()" not in engine
    assert "NamedOnnxValue" not in engine
    assert "DisposableNamedOnnxValue" not in engine
    assert "DenseTensor<float>" not in engine


def test_unified_runtime_keeps_bound_values_alive_and_disposes_binding_first() -> None:
    engine = (
        ROOT / "dotnet" / "ReceiptMlNet.Cli" / "UnifiedOcrEngine.cs"
    ).read_text(encoding="utf-8")
    runtime = _between(
        engine,
        "private sealed class UnifiedOcrRuntimeBuffers : IDisposable",
        "private sealed record OrtOutput",
    )
    dispose = _between(runtime, "public void Dispose()", "private static int GetElementCount(")

    assert "private readonly OrtValue[] _inputValues;" in runtime
    assert "private readonly OrtValue[] _outputValues;" in runtime
    assert "public IReadOnlyDictionary<string, OrtOutput> Outputs { get; }" in runtime
    assert dispose.index("Binding.Dispose();") < dispose.index(
        "DisposeValues(_outputValues);"
    )
    assert dispose.index("DisposeValues(_outputValues);") < dispose.index(
        "DisposeValues(_inputValues);"
    )


def test_detector_cpu_thread_tuning_is_explicit_and_auditable() -> None:
    program = (ROOT / "dotnet" / "ReceiptMlNet.Cli" / "Program.cs").read_text(
        encoding="utf-8"
    )

    assert 'case "--detector-intra-op-threads"' in program
    assert '--detector-intra-op-threads requires --device cpu' in program
    assert "IntraOpNumThreads = intraOpThreads" in program
    assert "options.DetectorIntraOpThreads" in program
    assert "int? DetectorIntraOpThreads" in program
