from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recipient-run-history.py"


def _module():
    spec = importlib.util.spec_from_file_location("recipient_run_history", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_reports_training_recipe_and_every_evaluation(tmp_path: Path) -> None:
    run = tmp_path / "unified-run-v12-example"
    run.mkdir()
    (run / "training_summary.json").write_text(
        json.dumps(
            {
                "best_checkpoint_epoch": 2,
                "config": {
                    "recipient_input_height": 128,
                    "recipient_input_width": 1536,
                    "recipient_branch_channels": 24,
                    "recipient_hidden_size": 256,
                    "recipient_open_text_layers": 2,
                    "recipient_open_text_heads": 8,
                    "recipient_open_text_feedforward": 2048,
                },
                "recipient_train_split_policy": {"splits": ["train", "val"]},
                "recipient_tail_loss_policy": {
                    "rare_character_max_support": 3,
                    "rare_character_loss_weight": 2.0,
                    "long_text_min_length": 9,
                    "long_text_loss_weight": 2.0,
                },
                "initialization": {"mode": "adapter"},
                "records": [
                    {"epoch": 1, "val_candidate_text_by_field": {"recipient_field": {"exact_match": 0.8}}},
                    {"epoch": 2, "val_candidate_text_by_field": {"recipient_field": {"exact_match": 0.9}}},
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluation = run / "onnx-test"
    evaluation.mkdir()
    (evaluation / "summary.json").write_text(
        json.dumps({"by_field": {"recipient_field": {"raw_exact_match": 0.7}}}),
        encoding="utf-8",
    )

    rows = _module().summarize_root(tmp_path)

    assert len(rows) == 1
    assert rows[0]["best_val"] == 0.9
    assert rows[0]["splits"] == ["train", "val"]
    assert rows[0]["evaluations"] == [("onnx-test/summary.json", 0.7)]
    rendered = _module().format_rows(rows)
    assert "best=e2:90.00%" in rendered
    assert "eval onnx-test/summary.json recipient=70.00%" in rendered


def test_history_skips_malformed_and_non_training_runs(tmp_path: Path) -> None:
    malformed = tmp_path / "unified-run-v12-bad"
    malformed.mkdir()
    (malformed / "training_summary.json").write_text("not json", encoding="utf-8")
    empty = tmp_path / "unified-run-v12-empty"
    empty.mkdir()

    assert _module().summarize_root(tmp_path) == []
