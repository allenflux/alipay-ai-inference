from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOTNET = ROOT / "dotnet" / "ReceiptMlNet.Cli"


def _source(name: str) -> str:
    return (DOTNET / name).read_text(encoding="utf-8")


def test_hybrid_cli_loads_both_verified_bundles_and_requires_v13() -> None:
    program = _source("Program.cs")

    assert 'options.OcrMode is "onnx" or "hybrid-recipient"' in program
    assert 'PaddleOcrDeliveryBundle.LoadAndVerify(options.OcrBundlePath!)' in program
    assert 'options.OcrMode is "unified" or "hybrid-recipient"' in program
    assert 'UnifiedOcrBundle.LoadAndVerify(options.OcrModelPath!)' in program
    assert 'unifiedOcrBundle?.ArchitectureVersion != 13' in program
    assert "requires an architecture-v13 unified OCR model" in program
    assert 'mode is "none" or "onnx" or "unified" or "hybrid-recipient"' in program
    assert "--ocr-bundle is required when --ocr onnx or hybrid-recipient" in program
    assert "--ocr-model is required when --ocr unified or hybrid-recipient" in program


def test_hybrid_routes_only_recipient_after_v13_and_keeps_review_policy() -> None:
    program = _source("Program.cs")
    router = _source("PaddleRecipientHybrid.cs")

    unified_call = program.index("unifiedOcrEngine.RecognizeReceipt(")
    hybrid_call = program.index("PaddleRecipientHybrid.OverrideRecipient(")
    field_assembly = program.index("BuildUnifiedFields(detections, unifiedOcr)")
    assert unified_call < hybrid_call < field_assembly
    assert 'includeRecipient: ocrMode != "hybrid-recipient"' in program
    unified_engine = _source("UnifiedOcrEngine.cs")
    assert '.Where(name => !string.Equals(name, "recipient_logits"' in unified_engine
    assert "if (includeRecipient)" in unified_engine
    assert "_session.Run(inputs, _outputsWithoutRecipient)" in unified_engine
    assert "ReadOutputViews(runtimeOutputs, _outputsWithoutRecipient)" in unified_engine
    assert 'ocrEngine is not null && unifiedOcrEngine is null' in program
    assert 'ocrMode == "hybrid-recipient"' in program
    assert 'candidates.Remove("recipient_field")' in router
    assert 'candidates["recipient_field"] = new UnifiedOcrCandidate(' in router
    for forbidden in (
        'candidates["amount"]',
        'candidates["time"]',
        'candidates["payment_method_field"]',
        "StatusCandidate =",
        "StatusNormalized =",
    ):
        assert forbidden not in router
    assert "unified.TextDeliveryValue" in router
    assert "never falls back to the lower-accuracy v13 recipient branch" in router


def test_hybrid_is_pure_onnx_cpu_and_binds_all_artifact_hashes() -> None:
    program = _source("Program.cs")
    engine = _source("PaddleOcrEngine.cs")
    bundle = _source("PaddleOcrDeliveryBundle.cs")

    assert "CreateCpuSessions(bundle)" in engine
    assert "new InferenceSession(bundle.DetModel.FullPath)" in engine
    assert "new InferenceSession(bundle.ClsModel.FullPath)" in engine
    assert "new InferenceSession(bundle.RecModel.FullPath)" in engine
    assert "VerifySessionContract(_detector, bundle.DetModel)" in engine
    assert "VerifySessionContract(_classifier, bundle.ClsModel)" in engine
    assert "VerifySessionContract(_recognizer, bundle.RecModel)" in engine
    assert "padRight: false" in engine
    assert "VerifyAdapterContract" in (ROOT / "dotnet" / "ReceiptMlNet.Cli" / "PaddleOcrDeliveryBundle.cs").read_text(
        encoding="utf-8"
    )
    assert engine.count("padRight: false") == 2
    assert "normalized-space 0" in engine
    assert "padRight: true" not in engine
    assert "metadata.ElementType != typeof(float)" in engine
    assert "expected.IsDynamic && actual > 0" in engine
    assert "Process.Start" not in engine
    assert "Python.Runtime" not in engine
    assert "Paddle.Inference" not in engine
    assert "VerifyFile(dictionary" in bundle
    assert "VerifyRecognizerVocabulary" in bundle
    assert "PaddleOcrSettings.Parse" in bundle
    assert "VerifyFile(file" in bundle
    assert "ParseDynamicShapeRequirement" in bundle
    assert "paddleOcr?.ContractSha256" in program
    assert "unifiedOcr?.ModelSha256" in program
    assert "unifiedOcr?.LabelsSha256" in program
    assert "unifiedOcr?.ContractSha256" in program
    assert "string? PaddleOcrProvider" in program
    assert "string? UnifiedProvider" in program


def test_recipient_parser_has_package_free_executable_contract_tests() -> None:
    parser = _source("PaddleRecipientValueParser.cs")
    project = (
        ROOT
        / "dotnet"
        / "ReceiptMlNet.Cli.PaddleRecipientContractTests"
        / "ReceiptMlNet.Cli.PaddleRecipientContractTests.csproj"
    ).read_text(encoding="utf-8")
    harness = (
        ROOT
        / "dotnet"
        / "ReceiptMlNet.Cli.PaddleRecipientContractTests"
        / "Program.cs"
    ).read_text(encoding="utf-8")

    assert "ReceiptFieldNormalizer.CleanText(rawText)" in parser
    assert "text.StartsWith(label, StringComparison.Ordinal)" in parser
    assert "TrimStart(RowSeparators)" in parser
    assert "PaddleRecipientValueParser.cs" in project
    assert "PackageReference" not in project
    assert r'PaddleRecipientValueParser.Parse("\u5546\u6237\u7532 \u6536\u6b3e\u65b9")' in harness
    assert r'PaddleRecipientValueParser.Parse("\u5907\u6ce8 \u6536\u6b3e\u65b9 \u5546\u6237\u7532")' in harness
    assert r'PaddleRecipientValueParser.Parse("\u6536\u6b3e\u65b9\uff1a")' in harness


def test_paddle_bundle_conversion_is_dynamic_and_atomic() -> None:
    bundle = (ROOT / "src" / "transfer_receipt_ai" / "paddle_ocr_bundle.py").read_text(
        encoding="utf-8"
    )

    assert 'MODEL_ROLES = ("det", "rec", "cls")' in bundle
    assert '"--enable_onnx_checker",' in bundle
    assert '"True",' in bundle
    assert "_require_dynamic_ocr_shapes(role, metadata_value)" in bundle
    assert "stage.replace(output_dir)" in bundle
    assert "verify_bundle(bundle, require_onnx=True)" in bundle
    assert "package_delivery_bundle" in bundle
    assert '"runtime_dependencies": ["ONNX Runtime",' in bundle
