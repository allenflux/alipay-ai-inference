#!/usr/bin/env python3
"""List unified validation summaries and their recipient decoder policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUN = Path(
    "D:/alipay-ai-data/receipt-lite-teacher-120k-v1/"
    "unified-run-v12-r3-4090-paddle-fit-open-text-joint-wide1536-20260806-114954"
)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def audit(run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run.rglob("summary.json")):
        payload = _load(path)
        if payload is None:
            continue
        by_field = payload.get("by_field")
        recipient = by_field.get("recipient_field") if isinstance(by_field, dict) else None
        if not isinstance(recipient, dict):
            continue
        decoder = payload.get("recipient_decoder_policy")
        decoder = decoder if isinstance(decoder, dict) else {}
        rows.append(
            {
                "path": path.relative_to(run).as_posix(),
                "model_sha256": payload.get("model_sha256"),
                "matches": recipient.get("raw_exact_matches"),
                "records": recipient.get("records"),
                "exact_match": recipient.get("raw_exact_match"),
                "decoder_mode": decoder.get("mode", "undeclared"),
                "beam_width": decoder.get("beam_width"),
                "token_top_k": decoder.get("token_top_k"),
                "ngram_order": decoder.get("ngram_order"),
                "ngram_weight": decoder.get("ngram_weight"),
                "accepted": payload.get("acceptance", {}).get("passed")
                if isinstance(payload.get("acceptance"), dict)
                else payload.get("accepted"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        parser.error(f"run directory does not exist: {run}")
    rows = audit(run)
    if not rows:
        parser.error(f"no recipient validation summaries under: {run}")
    print("recipient_baseline_audit")
    for row in rows:
        exact = row["exact_match"]
        exact_text = "null" if not isinstance(exact, (int, float)) else f"{float(exact):.4%}"
        print(
            "  "
            f"recipient={row['matches']}/{row['records']}={exact_text}; "
            f"decoder={row['decoder_mode']}; beam={row['beam_width']}; "
            f"top-k={row['token_top_k']}; order={row['ngram_order']}; "
            f"weight={row['ngram_weight']}; accepted={row['accepted']}; "
            f"path={row['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
