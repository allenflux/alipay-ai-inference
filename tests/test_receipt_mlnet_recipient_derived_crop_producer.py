from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "dotnet" / "ReceiptMlNet.Cli.RecipientDerivedCropShadow"
CONTRACT_TESTS = (
    ROOT / "dotnet" / "ReceiptMlNet.Cli.RecipientDerivedCropShadowContractTests"
)
MAIN = ROOT / "dotnet" / "ReceiptMlNet.Cli"


def test_projects_are_independent_cpu_only_diagnostics() -> None:
    producer_project = (
        PRODUCER / "ReceiptMlNet.Cli.RecipientDerivedCropShadow.csproj"
    ).read_text(encoding="utf-8")
    contract_project = (
        CONTRACT_TESTS
        / "ReceiptMlNet.Cli.RecipientDerivedCropShadowContractTests.csproj"
    ).read_text(encoding="utf-8")
    for source in (producer_project, contract_project):
        assert "<OnnxRuntimeFlavor" in source
        assert ">cpu</OnnxRuntimeFlavor>" in source
        assert "RequireCpuOnnxRuntimeFlavor" in source
        assert "AdditionalProperties=\"OnnxRuntimeFlavor=cpu\"" in source
    assert "ReceiptMlNet.Cli.csproj" in producer_project
    assert "RecipientDerivedCropShadow.csproj" in contract_project

    main_friends = (MAIN / "AssemblyInfo.cs").read_text(encoding="utf-8")
    producer_friends = (PRODUCER / "AssemblyInfo.cs").read_text(encoding="utf-8")
    assert 'InternalsVisibleTo("ReceiptMlNet.Cli.RecipientDerivedCropShadow")' in main_friends
    assert (
        'InternalsVisibleTo("ReceiptMlNet.Cli.RecipientDerivedCropShadowContractTests")'
        in main_friends
    )
    assert "RecipientDerivedCropShadowContractTests" in producer_friends


def test_producer_executes_exact_frozen_crop4_crop5_layout_contract() -> None:
    source = (PRODUCER / "Program.cs").read_text(encoding="utf-8")
    assert "ExpectedRecords = 63" in source
    assert "receipt_mlnet_recipient_derived_crop_plan_summary_v1" in source
    assert "receipt_mlnet_recipient_derived_crop_plan_record_v1" in source
    assert "receipt_mlnet_recipient_derived_crop_layout_summary_v1" in source
    assert "receipt_mlnet_recipient_derived_crop_layout_record_v1" in source
    assert "crop4_interrow_value_corridor" in source
    assert "crop5_recipient_value_core" in source
    assert 'case "--plan-summary-sha256"' in source
    assert "PaddleOcrDeliveryBundle.LoadAndVerify" in source
    assert "PaddleOcrCpuModelSnapshot.Create(" in source
    assert "new PaddleOcrEngine(bundle, cpuModelSnapshot)" in source
    assert "RecipientDerivedCropBundleSnapshot.Create(" in source
    assert "privateBundleEvidence.ContentEquals(bundleEvidence)" in source
    assert "Directory.Delete(privateBundleDirectory, recursive: true)" in source
    assert "source.ContractPath" in source
    assert "source.Dictionary" in source
    engine_source = (MAIN / "PaddleOcrEngine.cs").read_text(encoding="utf-8")
    assert "PaddleOcrCpuModelSnapshot" in engine_source
    assert "new InferenceSession(snapshot.Detector)" in engine_source
    assert "new InferenceSession(snapshot.Classifier)" in engine_source
    assert "new InferenceSession(snapshot.Recognizer)" in engine_source
    assert "new InferenceSession(bundle.DetModel.FullPath)" in engine_source
    assert 'engine.ExecutionProvider, "cpu"' in source
    assert "ReceiptRectifier.MaxSide1600Mode" in source
    assert "context.Crop(" in source
    assert "engine.RecognizeLayoutDiagnostic(cropImage)" in source
    assert "BuildLines(read, cropPlan, dropScore)" in source
    assert "QuadCrop" in source
    assert "QuadRectified" in source
    assert "point.X + crop.Box.Left" in source
    assert "point.Y + crop.Box.Top" in source
    assert "line.PassesDropScore != expectedPass" in source
    assert "RequireOrderedConvexQuad(line.Quad, lineIndex)" in source


def test_existing_production_engine_paths_match_audited_ownership_revision() -> None:
    source = (MAIN / "PaddleOcrEngine.cs").read_text(encoding="utf-8")

    def section(start: str, end: str) -> str:
        left = source.index(start)
        return source[left : source.index(end, left)]

    frozen_sections = {
        "path_constructor": (
            section(
                "    public PaddleOcrEngine(PaddleOcrDeliveryBundle bundle, DeviceSetting requestedDevice)",
                "    /// <summary>\n    /// CPU-only diagnostic constructor",
            ),
            "b62fa65c11dc081d13647a009188ee202cb3a28f4e88355474f87a2a0c4c541c",
        ),
        "recognize_methods": (
            section(
                "    public PaddleOcrReadResult Recognize(Image<Rgb24> image)",
                "    /// <summary>\n    /// Run the same frozen DB/CLS/REC pipeline",
            ),
            # Frozen after the audited A3 Mat ownership-only revision.
            "9660b41dc77ed688fa87d201ef32d8e94fd0c78e21dde98bc5b22bcf2f796c26",
        ),
        "device_session_selection": (
            section(
                "    private static (InferenceSession Detector, InferenceSession Classifier, InferenceSession Recognizer) CreateSessions(",
                "    private static void VerifySessionContract(",
            ),
            "9a691e075364c2a479ae0a704809fa78123ae976e9277073196fc4b3c4876bfb",
        ),
        "path_cpu_sessions": (
            section(
                "    private static (InferenceSession Detector, InferenceSession Classifier, InferenceSession Recognizer) CreateCpuSessions(\n        PaddleOcrDeliveryBundle bundle)",
                "    private static (InferenceSession Detector, InferenceSession Classifier, InferenceSession Recognizer) CreateCpuSessions(\n        PaddleOcrCpuModelSnapshot snapshot)",
            ),
            "f5f3a72ecfc862aea7e1500affa5894141b53c2a8d983eef82222bd19d950ba4",
        ),
    }
    for name, (payload, expected_sha256) in frozen_sections.items():
        assert hashlib.sha256(payload.encode()).hexdigest() == expected_sha256, name


def test_producer_is_atomic_toctou_closed_and_never_writes_candidates() -> None:
    source = (PRODUCER / "Program.cs").read_text(encoding="utf-8")
    assert source.count("VerifyClosingEvidence(plan, options.Bundle, bundleEvidence)") == 2
    assert "RecipientDerivedCropPlanContract.VerifyUnchanged(plan)" in source
    assert "Source image differs from the frozen derived-crop plan" in source
    assert "Paddle OCR delivery identity changed" in source
    assert "Directory.Move(stage, output.FullPath)" in source
    assert "Refusing to overwrite recipient derived-crop output" in source
    assert "RequireDisjoint" in source
    assert "RequireNoReparseDirectoryChain" in source
    assert "VerifyOwnedStage(output, stage)" in source
    assert "CandidateWriteEnabled: false" in source
    assert "FormalDeliveryGate: false" in source
    assert "ProductionOutputChanged: false" in source
    assert "AccuracyClaimed: false" in source
    assert "TruthUsedForCandidateSelection: false" in source
    assert "PaddleRecipientHybrid" not in source
    assert "UnifiedOcrCandidate" not in source
    assert "ReceiptFields" not in source

    production_program = (MAIN / "Program.cs").read_text(encoding="utf-8")
    assert "RecipientDerivedCropShadow" not in production_program


def test_contract_tests_cover_positive_and_fail_closed_evidence() -> None:
    source = (CONTRACT_TESTS / "Program.cs").read_text(encoding="utf-8")
    for contract in (
        "VerifyOptionsRequireExternallyBoundPlanSummary",
        "VerifyFrozenPlanAndInputContract",
        "VerifyPlanAndSourceMutationAreRejected",
        "VerifyGlobalGateFailureIsRejected",
        "VerifyCanonicalCropGeometry",
        "VerifyPythonCanonicalPlanId",
        "VerifyModelByteSnapshotIsHashBoundAndCloned",
        "VerifyContractAndDictionarySwapAreRejected",
        "VerifyRawLayoutLineCoordinatesAndDropScore",
        "VerifyDiagnosticJsonHasNoFieldCandidate",
        "VerifyFreshDisjointOutputContract",
    ):
        assert f"{contract}();" in source
    assert '"--device", "cuda:0"' in source
    assert "does not preserve every global gate" in source
    assert "drop-score flag differs" in source
    assert "escapes crop bounds" in source
    assert "degenerate, non-convex" in source
    assert 'TryGetProperty("candidate"' in source
    assert 'TryGetProperty("shadow_candidate"' in source
