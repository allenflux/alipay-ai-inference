from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-layout-shadow-run.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_layout_shadow_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(MODULE, "EXPECTED_RECORDS", 3)
    app_directory = tmp_path / "app"
    bundle = tmp_path / "bundle"
    selection = tmp_path / "selection"
    for directory in (app_directory, bundle, selection):
        directory.mkdir(parents=True)
    app = app_directory / "ReceiptMlNet.Cli.LayoutShadow.exe"
    app.write_bytes(b"fixed-app")
    input_list = selection / "inputs.txt"
    sources = [(tmp_path / "images" / f"receipt-{index}.jpg").resolve() for index in range(3)]
    input_bytes = "".join(f"{source}\n" for source in sources).encode()
    input_list.write_bytes(input_bytes)
    return {
        "app": app,
        "bundle": bundle,
        "input_list": input_list,
        "input_bytes": input_bytes,
        "sources": sources,
        "output": tmp_path / "layout-output",
    }


def _publish_valid(fixture: dict[str, object]) -> None:
    output = Path(fixture["output"])
    output.mkdir()
    sources = fixture["sources"]
    records = [
        {
            "schema_version": 1,
            "kind": MODULE.RECORD_KIND,
            "diagnostic_only": True,
            "formal_delivery_gate": False,
            "candidate_write_enabled": False,
            "index": index,
            "source": str(source),
            "execution_provider": "cpu",
        }
        for index, source in enumerate(sources)
    ]
    records_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode() for row in records
    )
    (output / "records.jsonl").write_bytes(records_bytes)
    input_bytes = fixture["input_bytes"]
    summary = {
        "schema_version": 1,
        "kind": MODULE.SUMMARY_KIND,
        "diagnostic_only": True,
        "formal_delivery_gate": False,
        "candidate_write_enabled": False,
        "expected_records": MODULE.EXPECTED_RECORDS,
        "records": MODULE.EXPECTED_RECORDS,
        "errors": 0,
        "execution_provider": "cpu",
        "input_list": {
            "path": str(Path(fixture["input_list"]).resolve()),
            "sha256": _sha(input_bytes),
            "size_bytes": len(input_bytes),
            "records": MODULE.EXPECTED_RECORDS,
        },
        "artifacts": {
            "records_jsonl": {
                "relative_path": "records.jsonl",
                "sha256": _sha(records_bytes),
                "size_bytes": len(records_bytes),
            }
        },
    }
    (output / "summary.json").write_text(json.dumps(summary) + "\n")


def test_launcher_computes_lower_sha_uses_argument_array_and_verifies_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    def fake_run(command: list[str], *, check: bool, shell: bool):
        observed.update(command=command, check=check, shell=shell)
        _publish_valid(fixture)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    summary = MODULE.run_layout_shadow(
        app=fixture["app"],
        bundle=fixture["bundle"],
        input_list=fixture["input_list"],
        output=fixture["output"],
    )
    command = observed["command"]
    assert isinstance(command, list)
    assert observed["shell"] is False
    assert observed["check"] is False
    assert command[0] == str(Path(fixture["app"]).resolve())
    sha_index = command.index("--input-list-sha256") + 1
    assert command[sha_index] == _sha(fixture["input_bytes"])
    assert command[sha_index].islower()
    assert summary["records"] == 3
    assert summary["execution_provider"] == "cpu"


def test_nonzero_child_exit_is_preserved_and_never_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 7),
    )
    with pytest.raises(MODULE.LayoutShadowProcessError) as caught:
        MODULE.run_layout_shadow(
            app=fixture["app"],
            bundle=fixture["bundle"],
            input_list=fixture["input_list"],
            output=fixture["output"],
        )
    assert caught.value.returncode == 7
    monkeypatch.setattr(
        MODULE,
        "run_layout_shadow",
        lambda **kwargs: (_ for _ in ()).throw(MODULE.LayoutShadowProcessError(7)),
    )
    assert MODULE.main([
        "--app", str(fixture["app"]),
        "--bundle", str(fixture["bundle"]),
        "--input-list", str(fixture["input_list"]),
        "--output", str(fixture["output"]),
    ]) == 7


@pytest.mark.parametrize("overlap", ["existing", "bundle", "app", "input"])
def test_requires_fresh_output_disjoint_from_all_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlap: str
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if overlap == "existing":
        output = Path(fixture["output"])
        output.mkdir()
    elif overlap == "bundle":
        output = Path(fixture["bundle"]) / "output"
    elif overlap == "app":
        output = Path(fixture["app"]).parent / "output"
    else:
        output = Path(fixture["input_list"]).parent / "output"
    with pytest.raises(MODULE.LayoutShadowRunError, match="fresh|overlaps"):
        MODULE.run_layout_shadow(
            app=fixture["app"],
            bundle=fixture["bundle"],
            input_list=fixture["input_list"],
            output=output,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("summary_provider", "execution_provider must be cpu"),
        ("summary_errors", "summary errors must be 0"),
        ("summary_flag", "candidate_write_enabled must be false"),
        ("input_sha", "input-list SHA-256 differs"),
        ("artifact_hash", "records artifact SHA-256 differs"),
        ("record_source", "source/order differs"),
        ("record_flag", "formal_delivery_gate must be false"),
    ],
)
def test_post_run_verifier_rejects_tampered_summary_and_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _publish_valid(fixture)
    output = Path(fixture["output"])
    summary_path = output / "summary.json"
    records_path = output / "records.jsonl"
    summary = json.loads(summary_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    if mutation == "summary_provider":
        summary["execution_provider"] = "cuda"
    elif mutation == "summary_errors":
        summary["errors"] = 1
    elif mutation == "summary_flag":
        summary["candidate_write_enabled"] = True
    elif mutation == "input_sha":
        summary["input_list"]["sha256"] = "0" * 64
    elif mutation == "artifact_hash":
        summary["artifacts"]["records_jsonl"]["sha256"] = "0" * 64
    elif mutation == "record_source":
        records[0]["source"] = records[1]["source"]
    else:
        records[0]["formal_delivery_gate"] = True
    if mutation.startswith("record_"):
        records_bytes = b"".join(
            (json.dumps(row, separators=(",", ":")) + "\n").encode() for row in records
        )
        records_path.write_bytes(records_bytes)
        summary["artifacts"]["records_jsonl"]["sha256"] = _sha(records_bytes)
        summary["artifacts"]["records_jsonl"]["size_bytes"] = len(records_bytes)
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(MODULE.LayoutShadowRunError, match=message):
        MODULE.validate_output(
            output=output,
            input_list=fixture["input_list"],
            input_bytes=fixture["input_bytes"],
            input_sources=[str(source) for source in fixture["sources"]],
        )


def test_rejects_input_list_mutation_during_child_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    def fake_run(command: list[str], **kwargs):
        _publish_valid(fixture)
        Path(fixture["input_list"]).write_bytes(fixture["input_bytes"] + b"changed\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    with pytest.raises(MODULE.LayoutShadowRunError, match="input list changed"):
        MODULE.run_layout_shadow(
            app=fixture["app"],
            bundle=fixture["bundle"],
            input_list=fixture["input_list"],
            output=fixture["output"],
        )


def test_source_contract_has_no_shell_or_powershell_hash_bridge() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "subprocess.run(command, check=False, shell=False)" in source
    assert "Get-FileHash" not in source
    assert "powershell" not in source.casefold()
    assert '"--input-list-sha256",\n        input_sha256' in source
    assert MODULE.EXPECTED_RECORDS == 339

