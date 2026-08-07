from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOTNET = ROOT / "dotnet" / "ReceiptMlNet.Cli"


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_unified_bundle_has_strict_versioned_status_text_abi() -> None:
    bundle = (DOTNET / "UnifiedOcrBundle.cs").read_text(encoding="utf-8")

    assert 'KindV12 = "receipt_unified_field_reader_v12"' in bundle
    assert 'KindV13 = "receipt_unified_field_reader_v13"' in bundle
    assert 'StatusTextOutputName = "status_text_logits"' in bundle
    assert 'StatusTextRuntimePolicy = "decode_and_normalize_review_only"' in bundle
    assert 'StatusTextTarget = "visible_transfer_status_cjk_text"' in bundle
    assert 'StatusTextCharsetSource = "train_only_visible_transfer_status_cjk_text"' in bundle
    assert 'StatusTextNormalizer = "normalize_status"' in bundle
    assert "12 => KindV12" in bundle
    assert "13 => KindV13" in bundle
    assert 'RequireInteger(labels, "status_text_blank_index", labelsPath, 0)' in bundle
    assert '"status_text_characters"' in bundle
    assert 'RequireSorted(statusTextCharacters, labelsPath, "status_text_characters")' in bundle
    assert 'RequireString(labels, "status_text_charset_source"' in bundle
    assert 'RequireString(contract, "status_text_charset_source"' in bundle
    assert 'RequireString(labels, "status_text_target", labelsPath, StatusTextTarget)' in bundle
    assert 'RequireString(contract, "status_text_target", contractPath, StatusTextTarget)' in bundle
    assert 'RequireString(labels, "status_text_runtime_policy", labelsPath, StatusTextRuntimePolicy)' in bundle
    assert 'RequireString(contract, "status_text_runtime_policy", contractPath, StatusTextRuntimePolicy)' in bundle
    assert 'architectureVersion == 13 && statusRuntimePolicy != "review_only"' in bundle
    assert 'RequireHash(labels, "status_text_charset_sha256"' in bundle
    assert 'RequireHash(contract, "status_text_charset_sha256"' in bundle
    assert "expectedOutputShapes[StatusTextOutputName]" in bundle
    assert 'RequireString(statusTextOutput, "layout", contractPath, "[time,class]")' in bundle
    assert 'RequireString(statusTextOutput, "decoder", contractPath, "ctc_greedy")' in bundle
    assert 'RequireString(statusTextOutput, "characters", contractPath, "status_text_characters")' in bundle
    assert 'RequireString(statusTextOutput, "target", contractPath, StatusTextTarget)' in bundle
    assert 'RequireString(statusTextOutput, "runtime_policy", contractPath, StatusTextRuntimePolicy)' in bundle
    assert 'RequireString(statusTextOutput, "review_value", contractPath, ReviewValue)' in bundle
    assert 'RequireString(statusTextOutput, "normalizer", contractPath, StatusTextNormalizer)' in bundle


def test_status_runtime_uses_ctc_text_and_never_exposes_review_only_classifier() -> None:
    engine = (DOTNET / "UnifiedOcrEngine.cs").read_text(encoding="utf-8")
    status_decode = _between(
        engine,
        "string? statusCandidate = null;",
        "result = new UnifiedOcrReadResult(",
    )

    assert "UnifiedStatusTextDecoder.Decode(" in status_decode
    assert "statusCandidate = statusText.Text;" in status_decode
    assert "statusNormalized = statusText.Normalized;" in status_decode
    assert 'else if (_bundle.StatusRuntimePolicy == "classify")' in status_decode
    assert status_decode.count('outputs["status_logits"]') == 1
    assert "!_bundle.HasStatusTextCtc && _bundle.StatusRuntimePolicy == \"classify\"" in status_decode
    assert '_bundle.HasStatusTextCtc\n                    ? "review"' in status_decode
    assert ': _bundle.StatusReviewValue ?? "review";' in status_decode


def test_status_json_keeps_visible_text_and_separate_normalized_candidate() -> None:
    program = (DOTNET / "Program.cs").read_text(encoding="utf-8")
    status_field = _between(
        program,
        "private static ReceiptFieldResult UnifiedStatusField(",
        "private static bool ExistingResultSatisfiesRequestedMode(",
    )

    assert "unifiedOcr.StatusCandidate" in status_field
    assert "unifiedOcr.StatusNormalized" in status_field
    assert 'unifiedOcr.StatusDeliveryValue == "review" ? "review" : "read"' in status_field
    assert "Candidate: unifiedOcr.StatusCandidate" in status_field
    assert "CtcCandidate: unifiedOcr.StatusRuntimePolicy == UnifiedOcrBundle.StatusTextRuntimePolicy" in status_field
    assert "DeliveryValue: unifiedOcr.StatusDeliveryValue" in status_field


def test_ctc_contract_harness_executes_status_text_normalization() -> None:
    project = (
        ROOT
        / "dotnet"
        / "ReceiptMlNet.Cli.CtcContractTests"
        / "ReceiptMlNet.Cli.CtcContractTests.csproj"
    ).read_text(encoding="utf-8")
    harness = (
        ROOT / "dotnet" / "ReceiptMlNet.Cli.CtcContractTests" / "Program.cs"
    ).read_text(encoding="utf-8")

    assert "UnifiedStatusTextDecoder.cs" in project
    assert "ReceiptFieldNormalizer.cs" in project
    assert "VerifyVisibleStatusTextNormalization();" in harness
    assert '"转账成功", "success"' in harness
    assert '"处理中", "pending"' in harness
    assert '"转账失败", "failed"' in harness
