from __future__ import annotations

from pathlib import Path

import pytest

from transfer_receipt_ai.result_semantics import (
    RECEIPT_RESULT_SCHEMA_VERSION,
    RECEIPT_RESULT_SEMANTICS_VERSION,
    has_current_result_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_python_result_semantics_are_current() -> None:
    assert has_current_result_semantics(
        {
            "result_schema_version": RECEIPT_RESULT_SCHEMA_VERSION,
            "result_semantics_version": RECEIPT_RESULT_SEMANTICS_VERSION,
        }
    )


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"result_schema_version": RECEIPT_RESULT_SCHEMA_VERSION},
        {"result_semantics_version": RECEIPT_RESULT_SEMANTICS_VERSION},
        {
            "result_schema_version": True,
            "result_semantics_version": RECEIPT_RESULT_SEMANTICS_VERSION,
        },
        {
            "result_schema_version": RECEIPT_RESULT_SCHEMA_VERSION,
            "result_semantics_version": "legacy-status-normalization-v1",
        },
    ),
)
def test_legacy_or_malformed_python_result_semantics_are_stale(payload: object) -> None:
    assert not has_current_result_semantics(payload)


def test_python_writers_and_both_resume_paths_share_the_semantics_contract() -> None:
    pipeline = (ROOT / "src" / "transfer_receipt_ai" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    inference = (ROOT / "src" / "transfer_receipt_ai" / "infer.py").read_text(
        encoding="utf-8"
    )
    enrichment = (ROOT / "src" / "transfer_receipt_ai" / "ocr_enrich.py").read_text(
        encoding="utf-8"
    )

    assert '"result_schema_version": RECEIPT_RESULT_SCHEMA_VERSION' in pipeline
    assert '"result_semantics_version": RECEIPT_RESULT_SEMANTICS_VERSION' in pipeline
    assert "has_current_result_semantics(payload)" in inference
    assert "has_current_result_semantics(payload)" in enrichment
