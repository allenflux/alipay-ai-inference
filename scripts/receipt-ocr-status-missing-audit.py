"""Print v13 transfer-status rows that lack an accepted visible OCR target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    records_path = args.run / "manifest-v13" / "unified_fields.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"missing v13 unified manifest: {records_path}")

    missing = 0
    with records_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            status = dict(record.get("slots", {})).get("transfer_status")
            if not isinstance(status, dict):
                continue
            audit = status.get("status_text_audit")
            if not isinstance(audit, dict) or audit.get("decision") != "missing":
                continue
            missing += 1
            if args.compact:
                print(
                    "|".join(
                        (
                            str(missing),
                            str(record.get("split", "")),
                            str(status.get("class_name", "")),
                            str(audit.get("reason", "")),
                            json.dumps(audit.get("paddle_text"), ensure_ascii=True),
                            json.dumps(audit.get("record_text"), ensure_ascii=True),
                        )
                    )
                )
                continue
            print(
                json.dumps(
                    {
                        "line": line_number,
                        "receipt_key": record.get("receipt_key"),
                        "group_id": record.get("group_id"),
                        "split": record.get("split"),
                        "class_name": status.get("class_name"),
                        "reason": audit.get("reason"),
                        "record_text": audit.get("record_text"),
                        "paddle_text": audit.get("paddle_text"),
                        "image": status.get("image"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
    print(json.dumps({"missing": missing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
