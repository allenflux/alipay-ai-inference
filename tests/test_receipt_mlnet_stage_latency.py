from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_stage_observation_does_not_repeat_model_inference() -> None:
    program = (ROOT / "dotnet" / "ReceiptMlNet.Cli" / "Program.cs").read_text(encoding="utf-8")
    infer_image = _between(
        program,
        "private static ReceiptResult InferImage(",
        "private static double StopAndReadMilliseconds(",
    )
    assert infer_image.count("detector.Predict(") == 1
    assert infer_image.count("deviceClassifier?.Classify(") == 1
    assert infer_image.count("unifiedOcrEngine.RecognizeReceipt(") == 1

    unified = (ROOT / "dotnet" / "ReceiptMlNet.Cli" / "UnifiedOcrEngine.cs").read_text(encoding="utf-8")
    recognize_receipt = _between(
        unified,
        "public UnifiedOcrReadResult RecognizeReceipt(",
        "private static double StopAndReadMilliseconds(",
    )
    assert (
        recognize_receipt.count(
            "_session.RunWithBinding(_runtime.RunOptions, _runtime.Binding)"
        )
        == 1
    )
    assert recognize_receipt.count("SynchronizeBoundInputs()") == 1
    assert recognize_receipt.count("SynchronizeBoundOutputs()") == 1
    assert "_session.Run(inputs)" not in recognize_receipt


def test_stage_latency_is_additive_to_existing_summary_and_manifest_fields() -> None:
    program = (ROOT / "dotnet" / "ReceiptMlNet.Cli" / "Program.cs").read_text(encoding="utf-8")
    assert "double? InferenceMs = null,\n    InferenceStageLatency? StageLatencyMs = null" in program
    assert "LatencySummary InferenceLatencyMs,\n    InferenceStageLatencySummary StageLatencyMs" in program

    for stage in (
        "ImageLoad",
        "Device",
        "DetectorPreprocess",
        "DetectorInference",
        "DetectorPostprocess",
        "UnifiedOcrPreprocess",
        "UnifiedOcrInference",
        "UnifiedOcrPostprocess",
        "ResultAssembly",
    ):
        assert f"LatencySummary {stage}" in program
