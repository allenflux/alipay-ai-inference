from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "receipt-mlnet-select-complete-inputs.py"
SPEC = importlib.util.spec_from_file_location("receipt_mlnet_complete_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _fixture_run(tmp_path: Path) -> tuple[Path, list[Path]]:
    run = tmp_path / "run"
    sources = [tmp_path / "images" / f"{index}.jpg" for index in range(3)]
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"image")
    _write_json(
        run / "inference_summary.json",
        {"requested_device": "cpu", "unified_provider": "cpu", "errors": 0},
    )
    manifest = []
    for index, source in enumerate(sources):
        result = run / "results" / f"{index}.json"
        labels = sorted(selector.REQUIRED_LABELS)
        if index == 1:
            labels.remove("time")
        _write_json(
            result,
            {
                "source": str(source),
                "detections": [{"label": label} for label in labels],
            },
        )
        manifest.append(
            {"status": "written", "source": str(source), "result": str(result)}
        )
    _write_json(run / "inference_manifest.json", manifest)
    return run, sources


def test_selector_writes_only_unique_complete_detector_inputs(tmp_path: Path) -> None:
    run, sources = _fixture_run(tmp_path)
    output = tmp_path / "complete.txt"

    selected, scanned = selector.write_selection(run, output, limit=2)

    assert selected == [sources[0].resolve(), sources[2].resolve()]
    assert scanned == 3
    assert output.read_text(encoding="utf-8").splitlines() == [
        str(sources[0].resolve()),
        str(sources[2].resolve()),
    ]


def test_selector_fails_closed_when_complete_evidence_is_insufficient(
    tmp_path: Path,
) -> None:
    run, _ = _fixture_run(tmp_path)

    with pytest.raises(selector.SelectionError, match="only 2 complete"):
        selector.select_complete_inputs(run, limit=3)


def test_selector_rejects_non_cpu_or_error_run(tmp_path: Path) -> None:
    run, _ = _fixture_run(tmp_path)
    _write_json(
        run / "inference_summary.json",
        {"requested_device": "cpu", "unified_provider": "cpu", "errors": 1},
    )

    with pytest.raises(selector.SelectionError, match="zero errors"):
        selector.select_complete_inputs(run, limit=1)
