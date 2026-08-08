from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "dotnet" / "ReceiptMlNet.Cli" / "PaddleOcrEngine.cs"
ASSEMBLY = ROOT / "dotnet" / "ReceiptMlNet.Cli" / "AssemblyInfo.cs"
PROJECT = ROOT / "dotnet" / "ReceiptMlNet.Cli.LayoutShadow"
CONTRACT_PROJECT = ROOT / "dotnet" / "ReceiptMlNet.Cli.LayoutShadowContractTests"


def _method_block(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated method {signature!r}")


def test_layout_api_is_additive_and_legacy_recognize_body_is_frozen() -> None:
    source = ENGINE.read_text(encoding="utf-8")
    image_overload = _method_block(
        source,
        "    public PaddleOcrReadResult Recognize(Image<Rgb24> image)\n",
    )
    assert hashlib.sha256(image_overload.encode("utf-8")).hexdigest() == (
        "b50257034cc3a1d422234c2940dff3b181a727dea3c1b1404c99a80839b3e7cb"
    )
    legacy = _method_block(
        source,
        "    public PaddleOcrReadResult Recognize(Mat rgb)\n",
    )
    assert hashlib.sha256(legacy.encode("utf-8")).hexdigest() == (
        "fcdfa3f10279fbec78be415e8761344eb7098abe64e1ebef4cbb6fd4d0ddb9f8"
    )
    assert "RecognizeLayoutDiagnostic(Image<Rgb24> image)" in source
    assert "RecognizeLayoutDiagnostic(Mat rgb)" in source
    assert "AssembleLayoutDiagnostic(" in source
    assert "lines[batch[index].OriginalIndex]" in source
    assert "PassesDropScore" in source
    assert "PaddleOcrLayoutPoint" in source


def test_layout_shadow_is_an_independent_cpu_only_raw_diagnostic() -> None:
    source = (PROJECT / "Program.cs").read_text(encoding="utf-8")
    project = (PROJECT / "ReceiptMlNet.Cli.LayoutShadow.csproj").read_text(
        encoding="utf-8"
    )
    assembly = ASSEMBLY.read_text(encoding="utf-8")

    assert "ReceiptMlNet.Cli.LayoutShadow" in assembly
    assert "ReceiptMlNet.Cli.LayoutShadowContractTests" in assembly
    assert "..\\ReceiptMlNet.Cli\\ReceiptMlNet.Cli.csproj" in project
    assert "ExpectedRecordCount = 339" in source
    assert 'DeviceSetting.Parse("cpu")' in source
    assert 'string.Equals(engine.ExecutionProvider, "cpu"' in source
    assert "ReceiptRectifier.MaxSide1600Mode" in source
    assert 'case "--input-list-sha256"' in source
    assert "RequireLowerSha256" in source
    assert "Directory.Move(stage, output.FullPath)" in source
    assert "PaddleOcrDeliveryBundle.LoadAndVerify(options.Bundle)" in source
    assert "DiagnosticOnly: true" in source
    assert "FormalDeliveryGate: false" in source
    assert "CandidateWriteEnabled: false" in source
    assert "LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence)" in source
    assert source.count(
        "LayoutShadowInputContract.VerifyUnchanged(selection, sourceEvidence)"
    ) == 2
    assert "RequireDisjointFromBundle" in source
    assert "RequireRegularNonReparseFile" in source
    assert "Paddle OCR delivery identity changed before layout shadow publication" in source
    assert "<OnnxRuntimeFlavor Condition=" in project
    assert "RequireCpuOnnxRuntimeFlavor" in project
    assert "AdditionalProperties=\"OnnxRuntimeFlavor=cpu\"" in project

    assert 'case "--device"' not in source
    assert 'case "--input"' not in source
    assert "ReceiptFields" not in source
    assert "BuildFields" not in source
    assert "PaddleRecipient" not in source
    assert "UnifiedOcr" not in source
    assert "DetectorModel" not in source
    assert "DeviceModel" not in source


def test_layout_shadow_schema_exposes_raw_quad_without_field_candidates() -> None:
    source = (PROJECT / "Program.cs").read_text(encoding="utf-8")
    assert "QuadRectified" in source
    assert "QuadRectifiedNormalized" in source
    assert "ConfidenceSemantics" in source
    assert "AcceptedText" in source
    assert "AcceptedConfidence" in source
    assert "AcceptedLineCount" in source
    assert "RawLineCount" in source
    assert "LayoutOcr" in source
    assert "P99" in source
    assert "SourceImageSha256" in source
    assert "BuildLayoutLines(" in source
    assert "not index-bound to accepted CTC output" in source
    assert "outside rectified image bounds" in source

    forbidden_record_properties = (
        "ReceiptFieldResult",
        "DeliveryValue",
        "DetectorScore",
        "TransferStatus",
        "PaymentMethod",
    )
    for value in forbidden_record_properties:
        assert value not in source


def test_layout_shadow_contract_harness_covers_fail_closed_boundaries() -> None:
    source = (CONTRACT_PROJECT / "Program.cs").read_text(encoding="utf-8")
    project = (
        CONTRACT_PROJECT / "ReceiptMlNet.Cli.LayoutShadowContractTests.csproj"
    ).read_text(encoding="utf-8")
    assert "ReceiptMlNet.Cli.LayoutShadow.csproj" in project
    assert "VerifyAcceptedProjectionMatchesLegacySemantics" in source
    assert "LayoutShadowProgram.BuildLayoutLines" in source
    assert "LegacyProjection" in source
    assert "SingleToInt32Bits" in source
    assert "VerifyFrozenInputContract" in source
    assert "exactly 339" in source
    assert "Duplicate layout shadow source" in source
    assert "VerifyOptionsAreDiagnosticOnly" in source
    assert '"--device", "cuda:0"' in source
    assert "VerifyJsonHasNoProductionFields" in source
    assert '"fields"' in source
    assert "VerifyFreshOutputContract" in source
    assert "VerifyOutputIsDisjointFromBundle" in source
    assert "LayoutShadowInputContract.VerifyUnchanged" in source
    assert "source changed while" in source
    assert "input list changed while" in source
    assert "must be an absolute path" in source
