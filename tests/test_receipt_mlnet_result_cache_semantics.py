from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "dotnet" / "ReceiptMlNet.Cli"
HARNESS = ROOT / "dotnet" / "ReceiptMlNet.Cli.CtcContractTests"


def test_each_result_persists_an_explicit_cache_semantics_contract() -> None:
    program = (CLI / "Program.cs").read_text(encoding="utf-8")
    contract = (CLI / "ReceiptResultCacheContract.cs").read_text(encoding="utf-8")

    assert 'SemanticsVersion = "status-review-only-visible-text-v1"' in contract
    assert 'root.TryGetProperty("result_schema_version"' in contract
    assert 'schemaVersion.ValueKind == JsonValueKind.Number' in contract
    assert 'root.TryGetProperty("result_semantics_version"' in contract
    assert "StringComparison.Ordinal" in contract
    assert "ReceiptResultCacheContract.SchemaVersion" in program
    assert "ReceiptResultCacheContract.SemanticsVersion" in program
    assert "int ResultSchemaVersion" in program
    assert "string ResultSemanticsVersion" in program


def test_skip_existing_requires_current_semantics_before_model_hashes() -> None:
    program = (CLI / "Program.cs").read_text(encoding="utf-8")
    validator = program.split(
        "private static bool ExistingResultSatisfiesRequestedMode(", 1
    )[1].split("private static bool HasJsonString(", 1)[0]

    semantics_check = "!ReceiptResultCacheContract.IsCurrent(document.RootElement)"
    assert semantics_check in validator
    assert validator.index(semantics_check) < validator.index(
        'HasJsonString(contracts, "detector"'
    )


def test_framework_free_csharp_harness_rejects_legacy_cache_shapes() -> None:
    project = (HARNESS / "ReceiptMlNet.Cli.CtcContractTests.csproj").read_text(
        encoding="utf-8"
    )
    harness = (HARNESS / "Program.cs").read_text(encoding="utf-8")

    assert "ReceiptResultCacheContract.cs" in project
    assert "VerifyResultCacheSemantics();" in harness
    assert '"legacy result without versions"' in harness
    assert '"missing semantics version"' in harness
    assert '"missing schema version"' in harness
    assert '"future schema version"' in harness
    assert '"legacy status-logit semantics"' in harness
    assert '"string schema version"' in harness
    assert "ReceiptResultCacheContract.IsCurrent(document.RootElement)" in harness
