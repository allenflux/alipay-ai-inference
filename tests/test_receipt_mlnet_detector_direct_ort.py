from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "dotnet" / "ReceiptMlNet.Cli" / "Program.cs"


def _program() -> str:
    return PROGRAM.read_text(encoding="utf-8")


def _detector_model(program: str) -> str:
    return program.split("internal sealed class DetectorModel", 1)[1].split(
        "internal sealed class DeviceModel", 1
    )[0]


def test_detector_uses_one_reusable_direct_ort_input() -> None:
    program = _program()
    detector = _detector_model(program)

    assert "PredictionEngine<DetectorInput, DetectorOutput>" not in detector
    assert "ApplyOnnxModel" not in detector
    assert "OrtValue.CreateTensorValueFromMemory(inputBuffer, InputShape)" in detector
    assert "inputValues = [inputValue]" in detector
    assert "_inputValues = inputValues" in detector
    assert "ReferenceEquals(tensor, _inputBuffer)" in detector
    assert (
        "_session.Run(_runOptions, InputNames, _inputValues, OutputNames)" in detector
    )


def test_detector_validates_named_typed_runtime_outputs() -> None:
    detector = _detector_model(_program())

    assert 'InputNames = ["image"]' in detector
    assert 'OutputNames = ["boxes", "labels", "scores"]' in detector
    assert "VerifyModelAbi(session)" in detector
    assert 'RequireTensorMetadata(session.OutputMetadata, "boxes", typeof(float))' in detector
    assert 'RequireTensorMetadata(session.OutputMetadata, "labels", typeof(long))' in detector
    assert 'RequireTensorMetadata(session.OutputMetadata, "scores", typeof(float))' in detector
    assert "TensorElementType.Int64" in detector
    assert "TensorElementType.Float" in detector
    assert "GetTensorDataAsSpan<float>()" in detector
    assert "GetTensorDataAsSpan<T>()" in detector
    assert "labels.Length != boxCount || scores.Length != boxCount" in detector


def test_detector_preserves_cpu_threads_cuda_fallback_and_disposal() -> None:
    program = _program()
    detector = _detector_model(program)

    assert "internal sealed class DetectorModel : IDisposable" in program
    assert "using var detector = new DetectorModel(" in program
    assert "IntraOpNumThreads = intraOpThreads.Value" in detector
    assert "options.AppendExecutionProvider_CUDA(device.GpuDeviceId.Value)" in detector
    assert "catch (OnnxRuntimeException) when (device.FallbackToCpu)" in detector
    assert "_inputValue.Dispose()" in detector
    assert "_runOptions.Dispose()" in detector
    assert "_session.Dispose()" in detector
