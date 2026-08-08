from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "scripts" / "receipt-ocr-status-text-v13-recover-evidence.ps1"
ATTESTOR = ROOT / "scripts" / "receipt_ocr_v13_recovery_attest.py"
SIDECAR_ATTESTOR = ROOT / "scripts" / "receipt_ocr_v13_sidecar_attest.py"
GENERATOR = ROOT / "scripts" / "receipt-ocr-status-text-v13-4090.ps1"
FORMAL = ROOT / "scripts" / "receipt-mlnet-hybrid-recipient-formal-ab.ps1"
CONSUMER = ROOT / "scripts" / "v13-cpu.ps1"


def _source() -> str:
    payload = RECOVERY.read_bytes()
    assert all(byte < 128 for byte in payload), "recovery script must remain ASCII-only"
    return payload.decode("ascii").replace("\r\n", "\n")


def test_recovery_takes_only_the_run_and_recovers_recorded_provenance() -> None:
    source = _source()

    assert "[Parameter(Mandatory = $true)]\n    [string]$RunDirectory" in source
    assert "[string]$PseudoLabels" not in source
    assert "[string]$DatasetRoot" not in source
    assert "[string]$SeedCheckpoint" not in source
    assert 'Resolve-RecordedFile $datasetContract.source_records' in source
    assert 'Resolve-RecordedDirectory $datasetContract.dataset_root' in source
    assert 'Resolve-RecordedFile $initialization.checkpoint_path' in source
    assert '$initialization.checkpoint_sha256' in source
    assert '$seedCheckpointSha256 -cne $seedCheckpointExpectedSha256' in source


def test_recovery_attests_manifest_and_artifacts_without_training_or_evaluation() -> None:
    source = _source()

    assert '"-m", "transfer_receipt_ai.ocr_unified_dataset"' in source
    assert '"--records", $pseudoLabels' in source
    assert '"--architecture", "v13"' in source
    assert '(Get-Sha256 $attestedRecords) -cne $recordsSha256' in source
    assert '(Get-Sha256 $attestedContract) -cne $datasetContractSha256' in source
    assert 'Assert-FileHash $pseudoLabels $pseudoLabelsSha256' in source
    assert 'Remove-Item -LiteralPath $attestationRoot -Recurse -Force' in source
    for forbidden in (
        '"transfer_receipt_ai.ocr_unified", "train"',
        '"transfer_receipt_ai.ocr_unified", "evaluate"',
        "nvidia-smi",
    ):
        assert forbidden not in source
    assert source.count('"transfer_receipt_ai.ocr_unified", "export"') == 2
    assert 'deterministic v12 seed attestation re-export' in source
    assert 'deterministic v13 candidate attestation re-export' in source
    assert 'Deterministic re-export differs from existing' in source
    assert 'strict v12/v13 sidecar compatibility attestation' in source
    assert 'receipt_ocr_v13_sidecar_attest.py' in source
    assert 'temporary-attestation-reexport-performed=true' in source
    assert 'training-or-evaluation-performed=false' in source


def test_recovery_revalidates_dataset_training_and_additive_status_contracts() -> None:
    source = _source()

    for token in (
        'receipt_unified_field_dataset_v6',
        'visible_transfer_status_cjk_text',
        'train_only_visible_transfer_status_cjk_text',
        'teacher labels, not independent business truth',
        'missing_status_text_records -ne 0',
        'trainStatusAudit.oov_records -ne 0',
        'max_possible_exact_match',
        'parameter_only_v12_to_v13_status_text_expansion',
        'checkpoint_legacy_label_maps_status_text_only_v1',
        'status_text_only_v13',
        'trainable_parameter_prefix -ne "status_text_"',
        'frozen_legacy_output_count -ne 15',
        'epoch_1_every_n_and_final_epoch',
        'runtime.uses_cuda -ne $true',
        'runtime.status_text_only_training -ne $true',
        'status_safety_then_transfer_status_raw_ctc_exact_then_recipient_exact_after_protected_candidate_exact_floors',
        'checkpoint_selection_score[0]',
        'checkpoint_selection_score[1]',
        'val_ctc_by_field.transfer_status.exact_match',
    ):
        assert token in source
    assert '$amountFloor = 0.7885' in source
    assert '$timeFloor = 0.9840' in source
    assert '$paymentFloor = 0.9325' in source
    assert '$recipientFloor = 0.90' in source
    assert '$statusTextFloor = 0.90' in source
    for forbidden_parameter in (
        "AmountFloor",
        "TimeFloor",
        "PaymentFloor",
        "RecipientFloor",
        "StatusTextFloor",
    ):
        assert f"[double]${forbidden_parameter}" not in source


def test_recovery_requires_original_seed_and_hash_bound_candidate_sidecars() -> None:
    source = _source()

    for path in (
        '"wide1536-v12-seed.onnx"',
        '"status-text-v13.onnx"',
        '"training-v13"',
        '"best.pt"',
        '"onnx-val-gpu\\summary.json"',
        '"onnx-test-gpu\\summary.json"',
    ):
        assert path in source
    assert 'Assert-ArtifactHash $seedContract $seedModel "v12 seed"' in source
    assert 'Assert-ArtifactHash $candidateContract $candidateModel "v13 candidate"' in source
    assert '$seedOutputs.Count -ne 15' in source
    assert '$candidateOutputs.Count -ne 16' in source
    assert 'Legacy output ABI parity failed' in source
    assert 'Properties["status_text_logits"]' in source
    assert 'candidateContract.status_head_policy.runtime_policy -ne "review_only"' in source
    assert 'Get-CanonicalJson $candidateContract.training_initialization' in source
    assert 'Get-CanonicalJson $candidateLabels.initialization' in source
    assert 'Get-CanonicalJson $candidateContract.checkpoint_selection_policy' in source
    assert '$candidateContract.status_text_oov_by_split "candidate contract status-text OOV audit"' in source
    assert 'Get-StatusOovProjection' in source
    assert 'Get-CanonicalJson $datasetStatusOov' in source
    assert 'Get-CanonicalJson $trainingStatusOov' in source
    assert 'candidateContract.status_text_charset_sha256' in source
    assert 'candidateLabels.status_text_charset_sha256' in source
    assert 'trainingSummary.status_text_charset_sha256' in source


def test_recovery_revalidates_gpu_core_fields_and_delegates_only_recipient() -> None:
    source = _source()

    assert source.count("Assert-EvaluationSummary `") == 2
    for token in (
        'receipt_unified_field_reader_teacher_parity_v1',
        'not independently verified business truth',
        'CUDAExecutionProvider',
        '$requestedValue -isnot [bool]',
        '$basePassedValue -isnot [bool]',
        '$rawFailures -isnot [Array]',
        'StartsWith("recipient_field:"',
        '$nonRecipientFailures.Count -ne 0',
        'Field = "amount"; Metric = "raw_exact_match"',
        'Field = "time"; Metric = "raw_exact_match"',
        'Field = "payment_method_field"; Metric = "raw_exact_match"',
        'Field = "transfer_status"; Metric = "ctc_raw_exact_match"',
        '$ctcRecords -ne [int]$StatusAudit.visible_status_records',
        '$ctcExactMatches / [double]$ctcRecords',
        'status_reference_class_counts',
        'max_non_success_to_success',
        '$unsafeCount -ne 0',
        'base_summary_passed = $baseSummaryPassed',
        'base_summary_failures = $failures',
        'recipient_delegated_to_hybrid_formal = $recipientDelegated',
        'accepted = $baseSummaryPassed',
    ):
        assert token in source
    assert '$candidateModel $candidateModelSha256 $records $recordsSha256' in source
    assert 'Field = "recipient_field"; Metric = "raw_exact_match"' not in source


def test_recovery_emits_the_original_guarded_schema_and_cpu_binding() -> None:
    source = _source()
    generator = GENERATOR.read_text(encoding="utf-8")
    formal = FORMAL.read_text(encoding="utf-8")
    consumer = CONSUMER.read_text(encoding="utf-8")

    for token in (
        'schema_version = 1',
        'kind = "receipt_unified_status_text_v13_guarded_validation_v1"',
        'pseudo_labels_sha256 = $pseudoLabelsSha256',
        'status_text_oov = @($trainStatusAudit, $valStatusAudit, $testStatusAudit)',
        'candidate = [ordered]@{',
        'training = [ordered]@{',
        'checkpoint_attestation = [ordered]@{',
        'legacy_output_parity = [ordered]@{',
        'acceptance_floors = [ordered]@{',
        'evaluations = @($valEvidence, $testEvidence)',
        'recipient_delivery_policy = [ordered]@{',
        'cpu_packaging = [ordered]@{',
        'performed = $false',
        'required_runtime_flavor = "cpu"',
        'required_rectification = "max-side-1600"',
        'include_device_model = $true',
    ):
        assert token in source
        if token in (
            'schema_version = 1',
            'kind = "receipt_unified_status_text_v13_guarded_validation_v1"',
            'candidate = [ordered]@{',
            'training = [ordered]@{',
            'legacy_output_parity = [ordered]@{',
            'acceptance_floors = [ordered]@{',
            'cpu_packaging = [ordered]@{',
            'performed = $false',
            'required_runtime_flavor = "cpu"',
            'required_rectification = "max-side-1600"',
            'include_device_model = $true',
        ):
            assert token in generator
    assert 'receipt_unified_status_text_v13_guarded_validation_v1' in formal
    assert 'required_runtime_flavor' in formal
    assert 'required_rectification' in formal
    assert 'include_device_model' in formal
    assert 'receipt_unified_status_text_v13_guarded_validation_v1' in consumer
    assert 'evidence.acceptance_floors' in consumer
    assert 'evidence.evaluations' in consumer
    assert 'evidence.cpu_packaging' in consumer
    for shared_delegation_token in (
        'base_summary_passed = $baseSummaryPassed',
        'base_summary_failures = $failures',
        'recipient_delegated_to_hybrid_formal = $recipientDelegated',
        'core_amount_time_payment_status_accepted = $true',
        'accepted = $baseSummaryPassed',
        'recipient_delivery_policy = [ordered]@{',
        'final_gate_required = $true',
    ):
        assert shared_delegation_token in source
        assert shared_delegation_token in generator


def test_recovery_truthfully_records_exact_onnx_and_allowlisted_sidecar_attestation() -> None:
    source = _source()

    assert 'deterministic_reexport_byte_identical = $true' not in source
    assert 'deterministic_reexport_onnx_byte_identical = $true' in source
    assert 'deterministic_reexport_sidecars_byte_identical = `' in source
    assert 'deterministic_reexport_sidecars_semantically_equivalent = $true' in source
    assert 'sidecar_allowed_fresh_only_defaults' in source
    assert 'sidecar_allowed_derived_differences' in source
    assert 'sidecar_comparisons' in source
    assert 'legacy_recipient_sidecar_defaults_added_by_17bc8af_v1' in source
    assert '17bc8afca6f0a1a95b0f3a45d603d016638fbbdb' in source
    assert 'byte-identical ONNX re-export + strict allowlisted sidecar semantic attestation' in source


def test_recovery_refuses_reuse_rechecks_hashes_and_publishes_atomically() -> None:
    source = _source()

    assert 'Require-NewPath $evidencePath "guarded v13 evidence"' in source
    assert 'already exists; refusing reuse or overwrite' in source
    assert 'changed during v13 evidence recovery' in source
    assert '@{ Path = $seedContractPath; Sha256 = $seedContractSha256' in source
    assert '@{ Path = $seedLabelsPath; Sha256 = $seedLabelsSha256' in source
    assert 'temporaryEvidence = $evidencePath + ".tmp-"' in source
    write = source.index('[IO.File]::WriteAllText(')
    verify = source.index('Read-GuardedJson $temporaryEvidence')
    second_refusal = source.rindex('Require-NewPath $evidencePath')
    final_recheck = source.rindex('foreach ($binding in $sourceBindings)')
    publish = source.index('Move-Item -LiteralPath $temporaryEvidence -Destination $evidencePath')
    assert source.count('foreach ($binding in $sourceBindings)') == 2
    assert write < verify < second_refusal < final_recheck < publish
    assert 'Remove-Item -LiteralPath $temporaryEvidence -Force' in source


def test_recovery_freezes_hashes_before_first_parse_and_rechecks_each_stage() -> None:
    source = _source()

    assert source.index('$datasetContractSha256 = Get-Sha256 $datasetContractPath') < source.index(
        '$datasetContract = Read-GuardedJson $datasetContractPath'
    )
    assert source.index('$trainingSummarySha256 = Get-Sha256 $trainingSummaryPath') < source.index(
        '$trainingSummary = Read-GuardedJson $trainingSummaryPath'
    )
    assert source.index('$valSummarySha256 = Get-Sha256 $valSummaryPath') < source.index(
        '$valEvidence = Assert-EvaluationSummary'
    )
    assert '$recordsSha256 $valSummarySha256' in source
    assert '$recordsSha256 $testSummarySha256' in source
    assert 'Assert-FileHash $trainingSummaryPath $trainingSummarySha256' in source
    assert 'return ,$property.Value' in source


def _load_attestor_module():
    spec = importlib.util.spec_from_file_location("receipt_ocr_v13_recovery_attest", ATTESTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sidecar_attestor_module():
    spec = importlib.util.spec_from_file_location(
        "receipt_ocr_v13_sidecar_attest", SIDECAR_ATTESTOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar_bundle(
    root: Path,
    *,
    name: str,
    architecture_version: int,
    compatibility_defaults: bool,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model = root / name
    model.write_bytes(b"deterministic-onnx")
    labels = model.with_suffix(".labels.json")
    labels_payload: dict[str, object] = {
        "schema_version": 1,
        "payment_characters": ["a", "b"],
    }
    if compatibility_defaults:
        labels_payload["recipient_backbone"] = "legacy_depthwise_gru_v1"
    labels.write_text(
        json.dumps(labels_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contract = model.with_suffix(".contract.json")
    model_config: dict[str, object] = {
        "architecture_version": architecture_version,
        "recipient_input_width": 1536,
    }
    contract_payload: dict[str, object] = {
        "schema_version": 1,
        "kind": f"receipt_unified_field_reader_v{architecture_version}",
        "onnx_file": model.name,
        "onnx_sha256": _sha256(model),
        "labels_file": labels.name,
        "labels_sha256": _sha256(labels),
        "model": model_config,
    }
    if compatibility_defaults:
        contract_payload["recipient_backbone"] = "legacy_depthwise_gru_v1"
        model_config["recipient_open_text_dropout"] = 0.0
        model_config["recipient_backbone"] = "legacy_depthwise_gru_v1"
    contract.write_text(
        json.dumps(contract_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return model


def _rewrite_bundle_json(model: Path, *, labels_payload: dict, contract_payload: dict) -> None:
    labels = model.with_suffix(".labels.json")
    contract = model.with_suffix(".contract.json")
    labels.write_text(json.dumps(labels_payload, indent=2) + "\n", encoding="utf-8")
    contract_payload["labels_sha256"] = _sha256(labels)
    contract.write_text(json.dumps(contract_payload, indent=2) + "\n", encoding="utf-8")


def test_sidecar_attestor_accepts_only_the_exact_17bc8af_drift_for_both_models(
    tmp_path: Path,
) -> None:
    module = _load_sidecar_attestor_module()
    existing_seed = _write_sidecar_bundle(
        tmp_path / "existing-seed",
        name="wide1536-v12-seed.onnx",
        architecture_version=12,
        compatibility_defaults=False,
    )
    fresh_seed = _write_sidecar_bundle(
        tmp_path / "fresh-seed",
        name="wide1536-v12-seed.onnx",
        architecture_version=12,
        compatibility_defaults=True,
    )
    existing_candidate = _write_sidecar_bundle(
        tmp_path / "existing-candidate",
        name="status-text-v13.onnx",
        architecture_version=13,
        compatibility_defaults=False,
    )
    fresh_candidate = _write_sidecar_bundle(
        tmp_path / "fresh-candidate",
        name="status-text-v13.onnx",
        architecture_version=13,
        compatibility_defaults=True,
    )

    result = module.attest(
        existing_seed_model=existing_seed,
        fresh_seed_model=fresh_seed,
        existing_candidate_model=existing_candidate,
        fresh_candidate_model=fresh_candidate,
    )

    assert result["passed"] is True
    assert result["all_onnx_byte_identical"] is True
    assert result["all_sidecars_byte_identical"] is False
    assert result["all_sidecars_semantically_equivalent"] is True
    assert result["comparisons"]["seed"]["observed_difference_paths"] == {
        "labels": ["/recipient_backbone"],
        "contract": [
            "/labels_sha256",
            "/model/recipient_backbone",
            "/model/recipient_open_text_dropout",
            "/recipient_backbone",
        ],
    }
    assert (
        result["allowed_fresh_only_defaults"]["contract"]
        ["/model/recipient_open_text_dropout"]
        == 0.0
    )


def test_sidecar_attestor_also_accepts_fully_byte_identical_sidecars(tmp_path: Path) -> None:
    module = _load_sidecar_attestor_module()
    existing = _write_sidecar_bundle(
        tmp_path / "existing",
        name="wide1536-v12-seed.onnx",
        architecture_version=12,
        compatibility_defaults=True,
    )
    fresh = _write_sidecar_bundle(
        tmp_path / "fresh",
        name="wide1536-v12-seed.onnx",
        architecture_version=12,
        compatibility_defaults=True,
    )

    result = module.attest_pair(
        description="seed", existing_model=existing, fresh_model=fresh
    )

    assert result["onnx_byte_identical"] is True
    assert result["sidecars_byte_identical"] is True
    assert result["observed_difference_paths"] == {"labels": [], "contract": []}


def test_sidecar_attestor_rejects_status_or_any_other_semantic_drift(tmp_path: Path) -> None:
    module = _load_sidecar_attestor_module()
    existing = _write_sidecar_bundle(
        tmp_path / "existing",
        name="status-text-v13.onnx",
        architecture_version=13,
        compatibility_defaults=False,
    )
    fresh = _write_sidecar_bundle(
        tmp_path / "fresh",
        name="status-text-v13.onnx",
        architecture_version=13,
        compatibility_defaults=True,
    )
    labels_path = fresh.with_suffix(".labels.json")
    contract_path = fresh.with_suffix(".contract.json")
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    labels_payload["status_text_target"] = "visible_transfer_status_text"
    contract_payload["status_text_target"] = "visible_transfer_status_text"
    _rewrite_bundle_json(
        fresh,
        labels_payload=labels_payload,
        contract_payload=contract_payload,
    )

    with pytest.raises(ValueError, match="exact compatibility drift"):
        module.attest_pair(description="candidate", existing_model=existing, fresh_model=fresh)


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (
            lambda labels, contract: labels.__setitem__(
                "recipient_backbone", "residual_positional_transformer_v2"
            ),
            "unsafe value",
        ),
        (
            lambda labels, contract: contract.pop("recipient_backbone"),
            "exact compatibility drift",
        ),
        (
            lambda labels, contract: contract["model"].__setitem__(
                "recipient_open_text_dropout", 0
            ),
            "unsafe value",
        ),
    ],
)
def test_sidecar_attestor_rejects_wrong_defaults_partial_sets_and_numeric_type_drift(
    tmp_path: Path, mutator, error: str
) -> None:
    module = _load_sidecar_attestor_module()
    existing = _write_sidecar_bundle(
        tmp_path / "existing",
        name="wide1536-v12-seed.onnx",
        architecture_version=12,
        compatibility_defaults=False,
    )
    fresh = _write_sidecar_bundle(
        tmp_path / "fresh",
        name="wide1536-v12-seed.onnx",
        architecture_version=12,
        compatibility_defaults=True,
    )
    labels_path = fresh.with_suffix(".labels.json")
    contract_path = fresh.with_suffix(".contract.json")
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    mutator(labels_payload, contract_payload)
    _rewrite_bundle_json(
        fresh,
        labels_payload=labels_payload,
        contract_payload=contract_payload,
    )

    with pytest.raises(ValueError, match=error):
        module.attest_pair(description="seed", existing_model=existing, fresh_model=fresh)


def test_sidecar_attestor_rejects_contract_hash_binding_failure(tmp_path: Path) -> None:
    module = _load_sidecar_attestor_module()
    existing = _write_sidecar_bundle(
        tmp_path / "existing",
        name="status-text-v13.onnx",
        architecture_version=13,
        compatibility_defaults=False,
    )
    fresh = _write_sidecar_bundle(
        tmp_path / "fresh",
        name="status-text-v13.onnx",
        architecture_version=13,
        compatibility_defaults=True,
    )
    contract_path = fresh.with_suffix(".contract.json")
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_payload["labels_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(contract_payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Contract labels hash does not bind"):
        module.attest_pair(description="candidate", existing_model=existing, fresh_model=fresh)


class _FakeTensor:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values
        self.shape = (len(values),)
        self.dtype = "float32"

    def detach(self):
        return self

    def cpu(self):
        return self


def test_checkpoint_attestor_binds_best_epoch_metadata_and_exact_legacy_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_attestor_module()
    seed_path = tmp_path / "seed.pt"
    candidate_path = tmp_path / "best.pt"
    summary_path = tmp_path / "training_summary.json"
    seed_path.write_bytes(b"seed")
    candidate_path.write_bytes(b"candidate")
    common = {
        "config": {"architecture_version": 13},
        "initialization": {
            "copied_legacy_tensor_count": 1,
            "new_status_text_tensor_count": 1,
        },
        "checkpoint_selection_policy": {"selection_metric": "guarded"},
        "status_text_oov_by_split": {"train": {}, "val": {}, "test": {}},
        "status_text_charset_sha256": "a" * 64,
        "status_text_charset_source": "train_only_visible_transfer_status_cjk_text",
        "status_text_target": "visible_transfer_status_cjk_text",
        "status_text_runtime_policy": "decode_and_normalize_review_only",
        "field_counts": {"amount": 1},
        "status_class_counts": {"val": {"success": 1, "pending": 0, "failed": 0}},
    }
    summary_path.write_text(
        json.dumps({**common, "best_checkpoint_epoch": 3}), encoding="utf-8"
    )
    checkpoints = {
        seed_path.resolve(): {
            "kind": "receipt_unified_field_reader_v12",
            "state_dict": {"legacy.weight": _FakeTensor((1, 2))},
        },
        candidate_path.resolve(): {
            **common,
            "kind": "receipt_unified_field_reader_v13",
            "epoch": 3,
            "state_dict": {
                "legacy.weight": _FakeTensor((1, 2)),
                "status_text_head.weight": _FakeTensor((3, 4)),
            },
        },
    }
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda path, **_: checkpoints[Path(path).resolve()]
    fake_torch.equal = lambda left, right: left.values == right.values
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = module.attest(
        seed_checkpoint=seed_path.resolve(),
        candidate_checkpoint=candidate_path.resolve(),
        training_summary_path=summary_path.resolve(),
    )

    assert result["passed"] is True
    assert result["candidate_epoch"] == 3
    assert result["legacy_tensor_count"] == 1
    assert result["new_status_text_tensor_count"] == 1

    checkpoints[candidate_path.resolve()]["state_dict"]["legacy.weight"] = _FakeTensor((9, 9))
    with pytest.raises(ValueError, match="changed frozen v12 tensors"):
        module.attest(
            seed_checkpoint=seed_path.resolve(),
            candidate_checkpoint=candidate_path.resolve(),
            training_summary_path=summary_path.resolve(),
        )


def test_recovery_powershell_parses_when_available() -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    escaped_path = str(RECOVERY).replace("'", "''")
    parser_command = (
        "$errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped_path}',[ref]$null,[ref]$errors); "
        "if($errors.Count -gt 0){$errors | ForEach-Object { Write-Error $_ }; exit 1}"
    )
    completed = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", parser_command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
