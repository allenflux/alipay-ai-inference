#!/usr/bin/env python3
"""Summarize existing recipient training/evaluation evidence without mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _recipient_metric(payload: Mapping[str, Any] | None) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    by_field = payload.get("by_field")
    if not isinstance(by_field, Mapping):
        return None
    recipient = by_field.get("recipient_field")
    if not isinstance(recipient, Mapping):
        return None
    value = recipient.get("raw_exact_match")
    return float(value) if isinstance(value, (int, float)) else None


def _training_metric(record: Mapping[str, Any] | None) -> float | None:
    if not isinstance(record, Mapping):
        return None
    by_field = record.get("val_candidate_text_by_field")
    if not isinstance(by_field, Mapping):
        return None
    recipient = by_field.get("recipient_field")
    if not isinstance(recipient, Mapping):
        return None
    value = recipient.get("exact_match")
    return float(value) if isinstance(value, (int, float)) else None


def _evaluation_rows(run: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for summary in sorted(run.rglob("summary.json")):
        if summary.parent.name == "training-v13":
            continue
        value = _recipient_metric(_read_object(summary))
        if value is None:
            continue
        rows.append((summary.relative_to(run).as_posix(), value))
    return rows


def summarize_run(run: Path) -> dict[str, Any] | None:
    summary_path = run / "training_summary.json"
    if not summary_path.is_file():
        summary_path = run / "training-v13" / "training_summary.json"
    training = _read_object(summary_path)
    if training is None:
        return None
    records = training.get("records")
    history = [row for row in records if isinstance(row, Mapping)] if isinstance(records, list) else []
    best_epoch = training.get("best_checkpoint_epoch")
    best = next((row for row in history if row.get("epoch") == best_epoch), None)
    last = history[-1] if history else None
    config = training.get("config") if isinstance(training.get("config"), Mapping) else {}
    split_policy = (
        training.get("recipient_train_split_policy")
        if isinstance(training.get("recipient_train_split_policy"), Mapping)
        else {}
    )
    tail = training.get("recipient_tail_loss_policy") if isinstance(training.get("recipient_tail_loss_policy"), Mapping) else {}
    initialization = training.get("initialization") if isinstance(training.get("initialization"), Mapping) else {}
    return {
        "run": run.name,
        "best_epoch": best_epoch,
        "best_val": _training_metric(best),
        "last_val": _training_metric(last),
        "splits": list(split_policy.get("splits") or []),
        "width": config.get("recipient_input_width"),
        "height": config.get("recipient_input_height"),
        "channels": config.get("recipient_branch_channels"),
        "hidden": config.get("recipient_hidden_size"),
        "layers": config.get("recipient_open_text_layers"),
        "heads": config.get("recipient_open_text_heads"),
        "feedforward": config.get("recipient_open_text_feedforward"),
        "init": initialization.get("mode"),
        "tail": {
            "rare_max": tail.get("rare_character_max_support"),
            "rare_weight": tail.get("rare_character_loss_weight"),
            "long_min": tail.get("long_text_min_length"),
            "long_weight": tail.get("long_text_loss_weight"),
        },
        "evaluations": _evaluation_rows(run),
    }


def summarize_root(root: Path) -> list[dict[str, Any]]:
    source = root.expanduser().resolve()
    rows = [
        row
        for run in sorted(source.glob("unified-run-v12-*"), key=lambda item: item.stat().st_mtime)
        if run.is_dir()
        for row in [summarize_run(run)]
        if row is not None
    ]
    return rows


def _rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def format_rows(rows: list[dict[str, Any]]) -> str:
    lines = [f"recipient_run_history runs={len(rows)}"]
    for row in rows:
        lines.append(
            "  "
            f"{row['run']} best=e{row['best_epoch']}:{_rate(row['best_val'])} "
            f"last={_rate(row['last_val'])} splits={','.join(row['splits']) or 'n/a'} "
            f"view={row['height']}x{row['width']} cnn={row['channels']} hidden={row['hidden']} "
            f"transformer={row['layers']}x{row['heads']} ff={row['feedforward']} init={row['init']}"
        )
        tail = row["tail"]
        lines.append(
            "    "
            f"tail rare<={tail['rare_max']} x{tail['rare_weight']} "
            f"long>={tail['long_min']} x{tail['long_weight']}"
        )
        for path, value in row["evaluations"]:
            lines.append(f"    eval {path} recipient={_rate(value)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(format_rows(summarize_root(args.root)))


if __name__ == "__main__":
    main()
