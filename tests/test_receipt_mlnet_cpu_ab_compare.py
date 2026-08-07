from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import receipt_mlnet_cpu_ab_compare as compare


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _app_variant(tmp_path: Path, variant: str) -> dict[str, object]:
    app_root = tmp_path / "apps" / variant
    files = {
        "ReceiptMlNet.Cli.exe": f"apphost-{variant}".encode(),
        "ReceiptMlNet.Cli.dll": f"managed-entrypoint-{variant}".encode(),
        "ReceiptMlNet.Cli.deps.json": b"{}",
        "ReceiptMlNet.Cli.runtimeconfig.json": b"{}",
        "runtimes/win-x64/native/onnxruntime.dll": b"native-runtime",
    }
    rows: list[dict[str, object]] = []
    for relative, content in sorted(files.items()):
        path = app_root / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        rows.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size})
    manifest = tmp_path / f"{variant}-app-payload.json"
    _write_json(manifest, rows)
    return {
        "app_root": str(app_root),
        "executable": _evidence(app_root / "ReceiptMlNet.Cli.exe"),
        "app_payload": _evidence(manifest),
        "executable_relative_path": "ReceiptMlNet.Cli.exe",
        "managed_entrypoint_relative_path": "ReceiptMlNet.Cli.dll",
        "deps_json_relative_path": "ReceiptMlNet.Cli.deps.json",
        "runtimeconfig_json_relative_path": "ReceiptMlNet.Cli.runtimeconfig.json",
    }


def _result(source: Path, artifact_hashes: dict[str, str]) -> dict[str, object]:
    fields = {
        "time": {"state": "review", "candidate": "12:34", "delivery_policy": "review_only", "delivery_value": "review"},
        "amount": {"state": "review", "candidate": "88.00", "delivery_policy": "review_only", "delivery_value": "review"},
        "transfer_status": {"state": "review", "candidate": "成功", "delivery_policy": "review_only", "delivery_value": "review"},
        "recipient": {"state": "review", "candidate": "测试商户", "delivery_policy": "review_only", "delivery_value": "review"},
        "payment_method": {"state": "review", "candidate": "余额", "delivery_policy": "review_only", "delivery_value": "review"},
    }
    labels = ["time", "amount", "transfer_status", "recipient_field", "payment_method_field"]
    return {
        "source": str(source),
        "inference_engine": "mlnet",
        "geometry": {
            "source_size": {"width": 100, "height": 200},
            "rectified_size": {"width": 100, "height": 200},
            "detector_canvas": {"width": 864, "height": 1536},
            "resize_mode": "letterbox",
            "rectification": "max-side-1600",
            "rotation_degrees": 0,
            "screen_detected": False,
            "screen_quad_original": [[0, 0], [99, 0], [99, 199], [0, 199]],
            "H_original_to_rectified": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "H_rectified_to_original": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "detections": [
            {"label": label, "score": 0.9, "bbox_image": [1.0, 2.0, 30.0, 40.0]}
            for label in labels
        ],
        "fields": fields,
        "device": {
            "platform": "android",
            "platform_cn": "安卓",
            "source": "cnn",
            "confidence": 0.91,
            "device_prior_conflict": False,
            "p_ios": 0.09,
        },
        "model_contracts": {
            compare.MODEL_HASH_FIELDS[name]: digest for name, digest in artifact_hashes.items()
        },
        "limitations": ["fixed contract fixture"],
    }


def _write_run(
    output: Path,
    sources: list[Path],
    artifact_hashes: dict[str, str],
    *,
    inference_ms: float,
    total_seconds: float,
) -> None:
    manifest: list[dict[str, object]] = []
    stage_template = {
        "image_load": 0.5,
        "device": 0.8,
        "detector_preprocess": 1.1,
        "detector_inference": 2.2,
        "detector_postprocess": 0.4,
        "unified_ocr_preprocess": 1.3,
        "unified_ocr_inference": 3.2,
        "unified_ocr_postprocess": 0.3,
        "result_assembly": 0.2,
    }
    for index, source in enumerate(sources):
        result_path = output / "input-list" / f"result-{index}.json"
        _write_json(result_path, _result(source, artifact_hashes))
        manifest.append(
            {
                "source": str(source),
                "result": str(result_path),
                "status": "written",
                "annotated_rectified": str(output / f"ignored-{index}.jpg"),
                "inference_ms": inference_ms + index,
                "stage_latency_ms": stage_template,
            }
        )
    inference_values = [inference_ms + index for index in range(len(sources))]
    stage_summary = {
        stage: compare._summarize(
            [float(record["stage_latency_ms"][stage]) for record in manifest]  # type: ignore[index]
        )
        for stage in compare.REQUIRED_STAGES
    }
    stage_summary["paddle_ocr"] = compare._summarize([])
    summary = {
        "requested_device": "cpu",
        "unified_provider": "cpu",
        "input": len(sources),
        "written": len(sources),
        "skipped": 0,
        "errors": 0,
        "total_seconds": total_seconds,
        "inference_latency_ms": compare._summarize(inference_values),
        "stage_latency_ms": stage_summary,
    }
    _write_json(output / "inference_manifest.json", manifest)
    _write_json(output / "inference_summary.json", summary)
    (output / "inference_errors.jsonl").write_text("", encoding="utf-8")


def _fixture_plan(tmp_path: Path) -> Path:
    sources = [tmp_path / "input-1.jpg", tmp_path / "input-2.jpg"]
    for index, source in enumerate(sources):
        source.write_bytes(f"image-{index}".encode())
    fixed_list = tmp_path / "fixed-inputs.txt"
    fixed_list.write_text("\n".join(str(item) for item in sources) + "\n", encoding="utf-8")
    input_evidence = tmp_path / "input-evidence.json"
    _write_json(
        input_evidence,
        [
            {"source": str(source), "sha256": _sha256(source), "bytes": source.stat().st_size}
            for source in sources
        ],
    )

    artifacts: dict[str, dict[str, object]] = {}
    artifact_hashes: dict[str, str] = {}
    for name in compare.MODEL_HASH_FIELDS:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"artifact-{name}".encode())
        artifacts[name] = _evidence(path)
        artifact_hashes[name] = _sha256(path)
    variants = {variant: _app_variant(tmp_path, variant) for variant in compare.VARIANTS}

    runs: list[dict[str, object]] = []
    execution_order = 0
    for phase, count, expected_sources in (
        ("warmup", 1, sources[:1]),
        ("measured", 3, sources),
    ):
        for iteration in range(1, count + 1):
            order = compare.VARIANTS if iteration % 2 else tuple(reversed(compare.VARIANTS))
            for variant in order:
                execution_order += 1
                output = tmp_path / "runs" / f"{phase}-{iteration:02d}" / variant / "output"
                speed = 8.0 if variant == "candidate" else 10.0
                seconds = 0.8 if variant == "candidate" else 1.0
                _write_run(
                    output,
                    expected_sources,
                    artifact_hashes,
                    inference_ms=speed,
                    total_seconds=seconds,
                )
                runs.append(
                    {
                        "id": f"{phase}-{iteration:02d}-{variant}",
                        "phase": phase,
                        "variant": variant,
                        "iteration": iteration,
                        "execution_order": execution_order,
                        "expected_count": len(expected_sources),
                        "output_directory": str(output),
                        "console_log": str(output.parent / "console.log"),
                    }
                )
    plan = {
        "schema_version": 1,
        "kind": compare.PLAN_KIND,
        "created_utc": "2026-08-07T00:00:00+00:00",
        "output_root": str(tmp_path),
        "input_count": len(sources),
        "input_selection": {
            "rule": "deduplicate_in_order_then_first_n",
            "source_input_list": _evidence(fixed_list),
            "input_limit_requested": 0,
            "available_count": len(sources),
            "selected_count": len(sources),
        },
        "fixed_input_list": _evidence(fixed_list),
        "input_evidence": _evidence(input_evidence),
        "warmup_runs": 1,
        "warmup_limit": 1,
        "repetitions": 3,
        "cli_contract": {
            "device": "cpu",
            "unified_provider": "cpu",
            "ocr": "unified",
            "score_threshold": 0.5,
            "rectification": "max-side-1600",
            "annotate": "none",
            "require_complete": True,
            "continue_on_error": False,
            "skip_existing": False,
            "includes_device_model": True,
        },
        "performance_gate": {
            "minimum_throughput_gain_percent": 2.0,
            "maximum_p50_regression_percent": 0.0,
            "maximum_p95_regression_percent": 0.0,
        },
        "artifacts": artifacts,
        "variants": variants,
        "runs": runs,
    }
    plan_path = tmp_path / "ab-plan.json"
    _write_json(plan_path, plan)
    return plan_path


def test_cpu_ab_accepts_exact_predictions_and_reports_pooled_performance(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)

    report, differences = compare.analyze_plan(plan_path)

    assert report["accepted"] is True
    assert differences == []
    assert report["prediction_consistency"]["compared_runs"] == 7
    assert report["performance"]["baseline"]["repetitions"] == 3
    assert report["performance"]["candidate"]["total_images"] == 6
    assert (
        report["performance"]["candidate"]["throughput_images_per_second"]["aggregate"]
        > report["performance"]["baseline"]["throughput_images_per_second"]["aggregate"]
    )
    assert report["performance"]["baseline"]["stage_latency_ms"]["device"]["count"] == 6
    assert (
        report["variant_identities"]["baseline"]["managed_entrypoint_sha256"]
        != report["variant_identities"]["candidate"]["managed_entrypoint_sha256"]
    )


def test_cpu_ab_rejects_any_candidate_business_output_difference(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidate_run = next(
        row
        for row in plan["runs"]
        if row["phase"] == "measured" and row["variant"] == "candidate" and row["iteration"] == 2
    )
    manifest_path = Path(candidate_run["output_directory"]) / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = Path(manifest[0]["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["fields"]["recipient"]["candidate"] = "被篡改商户"
    _write_json(result_path, result)

    report, differences = compare.analyze_plan(plan_path)

    assert report["accepted"] is False
    assert report["prediction_consistency"]["difference_count"] == 1
    assert differences[0]["json_pointer"] == "/fields/recipient/candidate"
    assert differences[0]["compared_run"] == "measured-02-candidate"


def test_cpu_ab_rejects_an_exact_but_slower_candidate(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for run in plan["runs"]:
        if run["phase"] != "measured" or run["variant"] != "candidate":
            continue
        output = Path(run["output_directory"])
        manifest_path = output / "inference_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest:
            record["inference_ms"] = 12.0
        _write_json(manifest_path, manifest)
        summary_path = output / "inference_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["total_seconds"] = 1.2
        summary["inference_latency_ms"] = compare._summarize(
            [float(record["inference_ms"]) for record in manifest]
        )
        _write_json(summary_path, summary)

    report, differences = compare.analyze_plan(plan_path)

    assert differences == []
    assert report["prediction_consistency"]["accepted"] is True
    assert report["performance"]["accepted"] is False
    assert report["accepted"] is False
    assert "aggregate throughput gain" in report["performance"]["gate"]["failures"][0]


def test_cpu_ab_rejects_latency_regression_even_when_throughput_improves(
    tmp_path: Path,
) -> None:
    plan_path = _fixture_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for run in plan["runs"]:
        if run["phase"] != "measured" or run["variant"] != "candidate":
            continue
        output = Path(run["output_directory"])
        manifest_path = output / "inference_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest:
            record["inference_ms"] = 12.0
        _write_json(manifest_path, manifest)
        summary_path = output / "inference_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["total_seconds"] = 0.5
        summary["inference_latency_ms"] = compare._summarize(
            [float(record["inference_ms"]) for record in manifest]
        )
        _write_json(summary_path, summary)

    report, differences = compare.analyze_plan(plan_path)

    assert differences == []
    assert report["prediction_consistency"]["accepted"] is True
    assert (
        report["performance"]["candidate_vs_baseline"]
        ["throughput_images_per_second"]["percent"]
        > 2.0
    )
    assert report["performance"]["accepted"] is False
    assert report["accepted"] is False
    failures = report["performance"]["gate"]["failures"]
    assert "inference p50 regressed" in failures
    assert "inference p95 regressed" in failures


def test_cpu_ab_rejects_a_changed_threshold_or_cpu_contract(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["cli_contract"]["score_threshold"] = 0.49
    _write_json(plan_path, plan)

    with pytest.raises(compare.ValidationError, match="score_threshold changed"):
        compare.analyze_plan(plan_path)


def test_cpu_ab_rejects_any_app_payload_change_after_freeze(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)
    native_runtime = tmp_path / "apps" / "candidate" / "runtimes" / "win-x64" / "native" / "onnxruntime.dll"
    native_runtime.write_bytes(b"changed-native-runtime")

    with pytest.raises(compare.ValidationError, match="app payload changed"):
        compare.analyze_plan(plan_path)


def test_cpu_ab_rejects_identical_managed_entrypoints(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    baseline_dll = tmp_path / "apps" / "baseline" / "ReceiptMlNet.Cli.dll"
    candidate_dll = tmp_path / "apps" / "candidate" / "ReceiptMlNet.Cli.dll"
    candidate_dll.write_bytes(baseline_dll.read_bytes())
    manifest_path = Path(plan["variants"]["candidate"]["app_payload"]["path"])
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    managed_row = next(row for row in rows if row["path"] == "ReceiptMlNet.Cli.dll")
    managed_row.update(sha256=_sha256(candidate_dll), bytes=candidate_dll.stat().st_size)
    _write_json(manifest_path, rows)
    plan["variants"]["candidate"]["app_payload"] = _evidence(manifest_path)
    _write_json(plan_path, plan)

    with pytest.raises(compare.ValidationError, match="managed entrypoints are byte-identical"):
        compare.analyze_plan(plan_path)


def test_input_limit_is_verified_as_first_n_of_the_canonical_list(tmp_path: Path) -> None:
    plan_path = _fixture_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fixed_inputs = Path(plan["fixed_input_list"]["path"]).read_text(encoding="utf-8").splitlines()
    plan["input_selection"]["input_limit_requested"] = 1
    plan["input_selection"]["selected_count"] = 1

    selection = compare._validate_input_selection(plan, fixed_inputs[:1])

    assert selection["input_limit_requested"] == 1


def test_cpu_ab_powershell_wrapper_parses_when_powershell_is_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    script = Path(__file__).resolve().parents[1] / "scripts" / "receipt-mlnet-cpu-ab-validate.ps1"
    parser_command = (
        "$errors=$null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{script}',[ref]$null,[ref]$errors); "
        "if($errors.Count -gt 0){$errors | ForEach-Object { Write-Error $_ }; exit 1}"
    )
    completed = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", parser_command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
