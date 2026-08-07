"""Versioned semantics for persisted Python receipt inference results."""

from __future__ import annotations

from collections.abc import Mapping


RECEIPT_RESULT_SCHEMA_VERSION = 1
RECEIPT_RESULT_SEMANTICS_VERSION = "python-status-normalization-negation-v2"


def has_current_result_semantics(payload: object) -> bool:
    """Return whether a persisted result is safe for ``--skip-existing`` reuse."""
    if not isinstance(payload, Mapping):
        return False
    return (
        type(payload.get("result_schema_version")) is int
        and payload.get("result_schema_version") == RECEIPT_RESULT_SCHEMA_VERSION
        and payload.get("result_semantics_version") == RECEIPT_RESULT_SEMANTICS_VERSION
    )
