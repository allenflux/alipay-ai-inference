from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT / "scripts" / "receipt-mlnet-hybrid-recipient-ab.py"
LAUNCHER = ROOT / "scripts" / "receipt-mlnet-hybrid-recipient-cpu-ab.ps1"


def _load_module():
    spec = importlib.util.spec_from_file_location("receipt_hybrid_ab_bindings", COMPARATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_delivery(root: Path) -> tuple[Path, str]:
    root.mkdir()
    models = {}
    package_size = 0
    for role in ("det", "rec", "cls"):
        target = root / f"{role}.onnx"
        target.write_bytes(f"model-{role}".encode())
        package_size += target.stat().st_size
        models[role] = {
            "path": target.name,
            "sha256": _sha256(target),
            "size_bytes": target.stat().st_size,
        }
    dictionary = root / "dict.txt"
    dictionary.write_text("收\n款\n方\n", encoding="utf-8")
    package_size += dictionary.stat().st_size
    contract = {
        "schema_version": 1,
        "kind": "paddle_ocr_v2_delivery",
        "package_size_bytes": package_size,
        "models": models,
        "dictionary": {
            "path": dictionary.name,
            "sha256": _sha256(dictionary),
            "size_bytes": dictionary.stat().st_size,
        },
    }
    contract_path = root / "paddle_ocr_delivery.contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return root, _sha256(contract_path)


def _write_run(root: Path, source: Path, *, hybrid: bool, delivery_sha256: str) -> None:
    root.mkdir()
    (root / "inference_summary.json").write_text(
        json.dumps(
            {
                "requested_device": "cpu",
                "paddle_ocr_provider": "cpu" if hybrid else None,
                "unified_provider": "cpu",
                "input": 1,
                "written": 1,
                "skipped": 0,
                "errors": 0,
                "inference_latency_ms": {
                    "count": 1,
                    "p50": 100.0,
                    "p95": 110.0 if not hybrid else 120.0,
                },
                "stage_latency_ms": {},
            }
        ),
        encoding="utf-8",
    )
    contracts = {
        "detector": "det.contract.json",
        "detector_sha256": "a" * 64,
        "detector_contract_sha256": "b" * 64,
        "device": "device.contract.json",
        "device_sha256": "c" * 64,
        "device_contract_sha256": "d" * 64,
        "unified_ocr_model": "v13.onnx",
        "unified_ocr_contract": "v13.contract.json",
        "unified_ocr_model_sha256": "e" * 64,
        "unified_ocr_labels_sha256": "f" * 64,
        "unified_ocr_contract_sha256": "0" * 64,
    }
    if hybrid:
        contracts.update(
            {
                "ocr_bundle": "paddle_ocr_delivery.contract.json",
                "ocr_bundle_contract_sha256": delivery_sha256,
            }
        )
    recipient_candidate = "新收款方" if hybrid else "旧收款方"
    result = {
        "result_schema_version": 1,
        "result_semantics_version": "fixture",
        "source": str(source.resolve()),
        "inference_engine": "mlnet",
        "geometry": {"rectification": "max-side-1600"},
        "device": {"platform": "android"},
        "model_contracts": contracts,
        "fields": {
            "time": {"state": "review", "candidate": "12:30"},
            "amount": {"state": "review", "candidate": "10.00"},
            "transfer_status": {"state": "review", "candidate": "转账成功", "normalized": "success"},
            "recipient": {
                "state": "review",
                "candidate": recipient_candidate,
                "ctc_candidate": recipient_candidate,
                "detector_score": 0.99,
                "delivery_policy": "review_only_pending_independent_human_truth_calibration",
                "delivery_value": "review",
                "value": "review",
            },
            "payment_method": {"state": "review", "candidate": "余额"},
        },
        "detections": [
            {"label": "amount", "score": 0.99, "bbox_image": [0, 0, 1, 1], "ocr": {"text": "10.00"}},
            {
                "label": "recipient_field",
                "score": 0.99,
                "bbox_image": [0, 1, 1, 2],
                "ocr": {"text": recipient_candidate},
            },
        ],
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (root / "inference_manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": str(source.resolve()),
                    "result": str(result_path.resolve()),
                    "status": "written",
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_cli_app(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    for name in (
        "ReceiptMlNet.Cli.exe",
        "ReceiptMlNet.Cli.dll",
        "ReceiptMlNet.Cli.deps.json",
        "ReceiptMlNet.Cli.runtimeconfig.json",
        "Microsoft.ML.OnnxRuntime.dll",
        "onnxruntime.dll",
        "OpenCvSharp.dll",
        "OpenCvSharpExtern.dll",
    ):
        (root / name).write_bytes(f"payload:{name}".encode())
    rows = [
        {
            "path": target.relative_to(root).as_posix(),
            "sha256": _sha256(target),
            "size_bytes": target.stat().st_size,
        }
        for target in sorted(root.rglob("*"), key=lambda path: (path.name.casefold(), path.name))
        if target.is_file()
    ]
    manifest = root.parent / "cli-app-closure.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return root, manifest


def test_comparison_hash_binds_input_run_manifests_and_cli_assembly(tmp_path: Path) -> None:
    module = _load_module()
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    source = tmp_path / "receipt.jpg"
    source.write_bytes(b"receipt")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_run(baseline, source, hybrid=False, delivery_sha256=delivery_sha256)
    _write_run(hybrid, source, hybrid=True, delivery_sha256=delivery_sha256)
    input_manifest = tmp_path / "fixed-inputs.txt"
    input_manifest.write_text(str(source.resolve()) + "\n", encoding="utf-8")
    cli_app, closure_manifest = _write_cli_app(tmp_path / "cli-app")
    assembly = cli_app / "ReceiptMlNet.Cli.dll"

    summary, accepted = module.compare(
        baseline_dir=baseline,
        hybrid_dir=hybrid,
        paddle_delivery=delivery,
        output=tmp_path / "comparison",
        input_manifest=input_manifest,
        input_manifest_sha256=_sha256(input_manifest),
        cli_assembly=assembly,
        cli_assembly_sha256=_sha256(assembly),
        cli_app=cli_app,
        cli_closure_manifest=closure_manifest,
        cli_closure_manifest_sha256=_sha256(closure_manifest),
        mode="pilot",
    )

    assert accepted is True
    assert summary["schema_version"] == 2
    assert summary["input_set"]["input_manifest"]["sha256"] == _sha256(input_manifest)
    assert summary["run_manifests"]["baseline"]["sha256"] == _sha256(
        baseline / "inference_manifest.json"
    )
    assert summary["run_manifests"]["hybrid"]["sha256"] == _sha256(
        hybrid / "inference_manifest.json"
    )
    assert summary["run_summaries"]["baseline"]["sha256"] == _sha256(
        baseline / "inference_summary.json"
    )
    assert summary["run_summaries"]["hybrid"]["sha256"] == _sha256(
        hybrid / "inference_summary.json"
    )
    assert summary["cli_build"]["assembly"]["sha256"] == _sha256(assembly)
    assert summary["cli_build"]["app_closure"]["closure_sha256"] == _sha256(
        closure_manifest
    )
    assert summary["cli_build"]["app_closure"]["file_count"] == 8
    assert summary["paddle_delivery"]["package_size_bytes"] > 0


def test_formal_comparison_hard_rejects_non_10016_count(tmp_path: Path) -> None:
    module = _load_module()
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    source = tmp_path / "receipt.jpg"
    source.write_bytes(b"receipt")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_run(baseline, source, hybrid=False, delivery_sha256=delivery_sha256)
    _write_run(hybrid, source, hybrid=True, delivery_sha256=delivery_sha256)

    with pytest.raises(module.ComparisonError, match="exactly 10016"):
        module.compare(
            baseline_dir=baseline,
            hybrid_dir=hybrid,
            paddle_delivery=delivery,
            output=tmp_path / "comparison",
            mode="formal",
        )


def test_comparison_rejects_latency_count_that_does_not_cover_run(tmp_path: Path) -> None:
    module = _load_module()
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    source = tmp_path / "receipt.jpg"
    source.write_bytes(b"receipt")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_run(baseline, source, hybrid=False, delivery_sha256=delivery_sha256)
    _write_run(hybrid, source, hybrid=True, delivery_sha256=delivery_sha256)
    summary_path = hybrid / "inference_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["inference_latency_ms"]["count"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(module.ComparisonError, match="latency count"):
        module.compare(
            baseline_dir=baseline,
            hybrid_dir=hybrid,
            paddle_delivery=delivery,
            output=tmp_path / "comparison",
            mode="pilot",
        )


def test_cli_app_closure_rejects_an_unlisted_runtime_file(tmp_path: Path) -> None:
    module = _load_module()
    cli_app, closure_manifest = _write_cli_app(tmp_path / "cli-app")
    manifest_sha256 = _sha256(closure_manifest)
    (cli_app / "unlisted-native.dll").write_bytes(b"unlisted")

    with pytest.raises(module.ComparisonError, match="closure is not exact"):
        module._verify_cli_app_closure(
            cli_app,
            closure_manifest,
            expected_manifest_sha256=manifest_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda contract: contract.__setitem__("package_size_bytes", contract["package_size_bytes"] + 1), "package_size_bytes mismatch"),
        (lambda contract: contract.__setitem__("schema_version", "1"), "schema_version must be an integer"),
    ],
)
def test_comparison_rejects_malformed_paddle_delivery_schema_or_size(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    module = _load_module()
    delivery, delivery_sha256 = _write_delivery(tmp_path / "delivery")
    source = tmp_path / "receipt.jpg"
    source.write_bytes(b"receipt")
    baseline = tmp_path / "baseline"
    hybrid = tmp_path / "hybrid"
    _write_run(baseline, source, hybrid=False, delivery_sha256=delivery_sha256)
    _write_run(hybrid, source, hybrid=True, delivery_sha256=delivery_sha256)
    contract_path = delivery / "paddle_ocr_delivery.contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    mutation(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(module.ComparisonError, match=message):
        module.compare(
            baseline_dir=baseline,
            hybrid_dir=hybrid,
            paddle_delivery=delivery,
            output=tmp_path / "comparison",
            mode="pilot",
        )


def test_launcher_freezes_one_published_cli_and_hard_counts_formal_set() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "$requiredFormalReceipts = 10016" in source
    assert "$preparedInputs.Count -ne $requiredFormalReceipts" in source
    assert '"prepared-full-val-inputs.txt"' in source
    assert '"fixed-selected-inputs.txt"' in source
    assert "[string[]]$selectedInputs" in source
    assert "Copy-Item -LiteralPath $preparedInputList -Destination $inputList" in source
    assert "mlnet_hybrid_ab_publish_frozen_cli" in source
    assert "Write-CliAppClosureManifest $cliPublishDirectory $cliClosureManifest" in source
    assert "& $DotnetExe $cliAssembly @baselineArguments" in source
    assert "& $DotnetExe $cliAssembly @hybridArguments" in source
    assert '"--input-manifest-sha256", $inputManifestSha256' in source
    assert '"--cli-assembly-sha256", $cliAssemblySha256' in source
    assert '"--cli-closure-manifest-sha256", $cliClosureManifestSha256' in source
    assert "[int]$score.coverage.expected_receipts -ne $requiredFormalReceipts" in source
