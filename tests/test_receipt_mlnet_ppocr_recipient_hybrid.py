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
    assert "CropRecipientRowLeftContext" in router
    assert 'retryRoute = "left_context_retry"' in router
    assert "PaddleRecipientValueParser.Parse(retryRead.Text)" in router
    assert "ParsePinyinAnnotatedRecipient" in router
    assert "ParseUnlabelledMerchantAmountPair" in router
    assert 'route = $"primary_{alternative.Route}"' in router
    assert 'retryRoute = $"left_context_retry_{retryAlternative.Route}"' in router
    assert "HasVerifiedUnlabelledMerchantRowLayout" in router
    assert "HasVerifiedUnlabelledMerchantRowGeometry" in router
    assert "ParseCalibratedAlternative" in router
    assert 'amount.Candidate' in router
    assert "expectedReceiptAmount,\n                recipientDetectorScore);" in router
    assert "expectedReceiptAmount,\n                read.Confidence);" not in router
    parser = _source("PaddleRecipientValueParser.cs")
    assert 'recipientScore < 0.68f' in parser
    assert 'amountScore < 0.80f' in parser
    assert 'paymentScore < 0.80f' in parser
    assert 'amountCenterY < recipientCenterY' in parser
    assert 'recipientCenterY < paymentCenterY' in parser
    assert 'recipientBox[1] >= amountBox[3] - amountVerticalTolerance' in parser
    assert 'recipientBox[3] <= paymentBox[1] + paymentVerticalTolerance' in parser
    assert 'paymentOverlapFraction > 0.45f' in parser
    assert 'recipientDetectorScore >= 0.84f' in parser
    assert 'AllowsExactCjkPaymentOverlapException' in parser
    assert 'paymentOverlapFraction: 0.45f' in router
    assert 'HasVerifiedCalibratedAlternativeRowLayout' in router
    assert 'recipientDetectorScore < 0.90f' in parser
    assert '!float.IsFinite(recipientDetectorScore)' in parser
    assert 'lines[0].Confidence < 0.80f' in parser
    assert 'lines[1].Confidence < 0.70f' in parser
    assert 'lines[2].Confidence < 0.80f' in parser
    assert '"shoukuanfang"' in parser
    assert '"pinyin_annotated_three_line"' in parser
    assert '0 => 0.75f' in parser
    assert '<= 1 => 0.68f' in parser
    assert '<= 100 => 0.90f' in parser
    assert '"unlabelled_cjk_amount_exact"' in parser
    assert '"unlabelled_cjk_amount_within_one_fen"' in parser
    assert '"unlabelled_cjk_amount_within_one_yuan"' in parser
    assert '@"^[0-9]{2,8}$"' in parser
    assert 'score < 0.95f' in parser
    assert 'merchantNumber == expectedFen / 100L' in parser
    assert '"unlabelled_numeric_amount_exact"' in parser
    assert "RecipientDiagnostic = diagnostic" in router


def test_hybrid_retry_is_left_context_only_and_remains_fail_closed() -> None:
    image_ops = _source("UnifiedOcrImageOps.cs")
    router = _source("PaddleRecipientHybrid.cs")

    retry = image_ops.split("CropRecipientRowLeftContext", 1)[1]
    assert "new Rectangle(0, top, right, bottom - top)" in retry
    assert router.count("paddleOcr.Recognize(") == 2
    assert router.count("PaddleRecipientValueParser.Parse(") == 2
    assert router.count("ParseCalibratedAlternative(") == 3
    assert router.count("PaddleRecipientValueParser.ParsePinyinAnnotatedRecipient(") == 1
    assert router.count("PaddleRecipientValueParser.ParseUnlabelledMerchantAmountPair(") == 1
    assert '"anchored_or_pair_parse_failed"' in router
    assert '"ocr_empty"' in router
    assert "candidates.Remove(\"recipient_field\")" in router


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
    assert "ParsePinyinAnnotatedRecipient" in harness
    assert "ParseUnlabelledMerchantAmountPair" in harness
    assert "strict pinyin annotation route" in harness
    assert "pinyin detector below 0.90" in harness
    assert "pinyin detector is non-finite" in harness
    assert "pinyin annotation line below 0.80" in harness
    assert "pinyin merchant line at 0.70" in harness
    assert "pinyin merchant line below 0.70" in harness
    assert "pinyin wrong line order" in harness
    assert "pinyin annotation typo" in harness
    assert "pinyin annotation contains non-letter noise" in harness
    assert "pinyin value is not CJK" in harness
    assert "pinyin trailing label is not exact" in harness
    assert "pinyin extra line" in harness
    assert "pinyin missing label line" in harness
    assert "CJK exact amount at 0.75" in harness
    assert "CJK exact amount below 0.75" in harness
    assert "CJK one-fen drift at 0.68" in harness
    assert "CJK one-fen drift below 0.68" in harness
    assert "CJK one-yuan drift at 0.90" in harness
    assert "CJK one-yuan drift below 0.90" in harness
    assert "CJK drift above one yuan" in harness
    assert "reversed pair" in harness
    assert "amount without currency mark" in harness
    assert "non-recipient row label" in harness
    assert "extra line" in harness
    assert "merchant line below 0.80" in harness
    assert "amount line below 0.80" in harness
    assert "non-finite pair line confidence" in harness
    assert "pair confidence count mismatch" in harness
    assert "non-finite pair detector score" in harness
    assert "infinite pair detector score" in harness
    assert "product row" in harness
    assert "OCR-confusable amount" in harness
    assert "full four-digit amount" in harness
    assert "four-digit amount mismatch above one yuan" in harness
    assert "shared-suffix amount mismatch" in harness
    assert "numeric merchant exact amount at 0.95" in harness
    assert "numeric merchant below 0.95" in harness
    assert "numeric merchant too short" in harness
    assert "numeric merchant too long" in harness
    assert "numeric merchant not digits only" in harness
    assert "numeric merchant amount not exact" in harness
    assert "numeric merchant equals amount integer part" in harness
    assert "zero-padded numeric merchant equals amount integer part" in harness
    assert "exact CJK overlap exception at 0.84" in harness
    assert "exact CJK overlap exception below 0.84" in harness
    assert "one-fen route has no overlap exception" in harness
    assert "one-yuan route has no overlap exception" in harness
    assert "numeric route has no overlap exception" in harness
    assert "pinyin route has no overlap exception" in harness
    assert "strict row geometry at calibrated floors" in harness
    assert "amount overlap" in harness
    assert "payment overlap" in harness
    assert "45 percent payment overlap is rejected by default geometry" in harness
    assert "45 percent payment overlap at exact-route envelope" in harness
    assert "payment overlap above 45 percent" in harness
    assert "exact-route exception does not relax amount overlap" in harness
    assert "payment overlap fraction above calibrated maximum" in harness
    assert "recipient detector below 0.68 floor" in harness
    assert "non-finite recipient detector score" in harness


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
