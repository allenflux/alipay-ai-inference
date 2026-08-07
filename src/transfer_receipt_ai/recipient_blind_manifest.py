"""Materialise a train/validation-only OCR manifest without test-label use.

The unified OCR trainer already optimises ``train`` and selects checkpoints on
``val``.  This small boundary makes that policy physical for recipient model
development: test rows are omitted before the training process starts, while a
hash binds the blind candidate to the immutable full manifest used by a later
one-shot final gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
KIND = "receipt_recipient_blind_train_val_manifest_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_blind_manifest(
    *,
    source: Path,
    output: Path,
    contract: Path,
) -> dict[str, Any]:
    """Copy only train/val rows and persist a fail-closed split contract."""

    source = source.resolve()
    output = output.resolve()
    contract = contract.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source:
        raise ValueError("blind output must differ from its full source manifest")
    if output.exists() or contract.exists():
        raise FileExistsError("refusing to overwrite blind manifest evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    contract.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_contract = contract.with_suffix(contract.suffix + ".tmp")
    if temporary_output.exists() or temporary_contract.exists():
        raise FileExistsError("refusing to reuse temporary blind manifest paths")

    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    try:
        with source.open("r", encoding="utf-8") as reader, temporary_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as writer:
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from error
                if not isinstance(row, Mapping):
                    raise ValueError(f"{source}:{line_number}: record must be an object")
                split = row.get("split")
                if split not in {"train", "val", "test"}:
                    raise ValueError(f"{source}:{line_number}: invalid split {split!r}")
                record_id = row.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(f"{source}:{line_number}: record id is missing")
                if record_id in seen_ids:
                    raise ValueError(f"{source}:{line_number}: duplicate record id {record_id!r}")
                seen_ids.add(record_id)
                counts[str(split)] += 1
                if split in {"train", "val"}:
                    # Preserve the original JSON bytes/field order.  The blind
                    # boundary examines only split/id and never derives a test
                    # vocabulary, metric, example list, or checkpoint choice.
                    writer.write(line.rstrip("\r\n") + "\n")
        for required in ("train", "val", "test"):
            if counts[required] <= 0:
                raise ValueError(f"full manifest has no {required} records")
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "source_manifest": str(source),
            "source_manifest_sha256": _sha256(source),
            "blind_manifest": str(output),
            "blind_manifest_sha256": _sha256(temporary_output),
            "split_counts": {
                "train": int(counts["train"]),
                "val": int(counts["val"]),
                "test_excluded": int(counts["test"]),
            },
            "optimizer_supervision_splits": ["train"],
            "checkpoint_selection_splits": ["val"],
            "final_gate_only_splits": ["test"],
            "test_labels_used": False,
            "test_metrics_computed": False,
            "test_examples_emitted": False,
        }
        temporary_contract.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_output.replace(output)
        temporary_contract.replace(contract)
        return payload
    except BaseException:
        for temporary in (temporary_output, temporary_contract):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an immutable train/val-only manifest for recipient candidate development"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = build_blind_manifest(
        source=args.source,
        output=args.output,
        contract=args.contract,
    )
    print(
        "recipient_blind_manifest "
        f"train={payload['split_counts']['train']} "
        f"val={payload['split_counts']['val']} "
        f"test_excluded={payload['split_counts']['test_excluded']}"
    )


if __name__ == "__main__":
    main()
