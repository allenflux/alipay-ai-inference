from __future__ import annotations

import pytest

from transfer_receipt_ai.font_domain import (
    UNKNOWN_DOMAIN,
    aggregate_font_domain_predictions,
    prediction_from_probabilities,
)


def _prediction(
    region_id: str,
    role: str,
    domain: str = "ios_alipay",
    *,
    confidence: float = 0.92,
    quality: float = 0.90,
    include: bool = True,
):
    other = "android_alipay" if domain == "ios_alipay" else "ios_alipay"
    return prediction_from_probabilities(
        region_id=region_id,
        role=role,
        probabilities={domain: confidence, other: 1.0 - confidence},
        quality=quality,
        include_in_consistency=include,
        fit_p_value=0.80,
    )


def test_document_consensus_passes_and_serializes_as_non_authenticity_evidence() -> None:
    result = aggregate_font_domain_predictions(
        document_id="receipt-1",
        predictions=(
            _prediction("time", "time"),
            _prediction("status", "status_bar", include=False),
            _prediction("amount", "amount"),
            _prediction("recipient", "recipient"),
        ),
    )

    assert result.decision == "PASS"
    assert result.dominant_domain == "ios_alipay"
    assert result.included_regions == 3
    assert result.accepted_regions == 3
    assert result.known_coverage == 1.0
    assert result.roles == ("amount", "recipient", "time")
    assert [line.region_id for line in result.lines] == ["amount", "recipient", "status", "time"]
    encoded = result.as_dict()
    assert encoded["authenticity"] == "not_assessed"
    assert encoded["requires_manual_review"] is False


def test_one_strong_cross_domain_region_forces_review() -> None:
    result = aggregate_font_domain_predictions(
        document_id="receipt-conflict",
        predictions=(
            _prediction("amount", "amount", "android_alipay", confidence=0.95),
            _prediction("recipient", "recipient"),
            _prediction("time", "time"),
            _prediction("method", "payment_method"),
        ),
    )

    assert result.decision == "REVIEW"
    assert result.dominant_domain == "ios_alipay"
    assert result.conflicts == ("amount",)
    assert "CROSS_DOMAIN_REGION_CONFLICT" in result.reasons
    assert result.as_dict()["requires_manual_review"] is True


def test_any_accepted_cross_domain_line_prevents_pass() -> None:
    weaker_conflict = _prediction(
        "amount",
        "amount",
        "android_alipay",
        confidence=0.70,
        quality=0.49,
    )
    result = aggregate_font_domain_predictions(
        document_id="receipt-accepted-conflict",
        predictions=(
            weaker_conflict,
            _prediction("recipient", "recipient"),
            _prediction("time", "time"),
            _prediction("method", "payment_method"),
        ),
    )

    assert result.support_ratio == 0.75
    assert result.decision == "REVIEW"
    assert result.conflicts == ("amount",)
    assert "CROSS_DOMAIN_REGION_CONFLICT" in result.reasons
    assert "STRONG_CROSS_DOMAIN_REGION_CONFLICT" not in result.reasons


def test_strong_conflict_wins_over_insufficient_evidence_and_generic_fallback() -> None:
    fallback_conflict = prediction_from_probabilities(
        region_id="amount",
        role="amount",
        probabilities={"android_alipay": 0.95, "ios_alipay": 0.05},
        quality=0.90,
        fit_p_value=0.80,
        generic_fallback=True,
    )
    result = aggregate_font_domain_predictions(
        document_id="receipt-small-conflict",
        predictions=(fallback_conflict, _prediction("time", "time")),
    )

    assert result.decision == "REVIEW"
    assert result.conflicts
    assert "INSUFFICIENT_REGIONS" in result.reasons
    assert "CROSS_DOMAIN_REGION_CONFLICT" in result.reasons


def test_unknown_lines_reduce_coverage_without_becoming_conflicts() -> None:
    rejected = prediction_from_probabilities(
        region_id="time",
        role="time",
        probabilities={"ios_alipay": 0.51, "android_alipay": 0.49},
        quality=0.90,
        fit_p_value=0.80,
    )
    result = aggregate_font_domain_predictions(
        document_id="receipt-unknown",
        predictions=(
            _prediction("amount", "amount"),
            _prediction("recipient", "recipient"),
            rejected,
        ),
    )

    assert rejected.label == UNKNOWN_DOMAIN
    assert rejected.candidate_domain == "ios_alipay"
    assert "LOW_DOMAIN_SUPPORT" in rejected.reasons
    assert "LOW_DOMAIN_MARGIN" in rejected.reasons
    assert result.decision == "PASS"
    assert result.accepted_regions == 2
    assert result.unknown_regions == 1
    assert result.known_coverage == pytest.approx(2 / 3)
    assert result.conflicts == ()


def test_insufficient_regions_roles_and_known_coverage_return_unknown() -> None:
    low_information = prediction_from_probabilities(
        region_id="recipient",
        role="recipient",
        probabilities={"ios_alipay": 0.95, "android_alipay": 0.05},
        quality=0.10,
        fit_p_value=0.80,
    )
    result = aggregate_font_domain_predictions(
        document_id="receipt-insufficient",
        predictions=(_prediction("amount", "amount"), low_information),
    )

    assert low_information.label == UNKNOWN_DOMAIN
    assert low_information.reasons == ("LOW_INFORMATION",)
    assert result.decision == "UNKNOWN"
    assert result.dominant_domain is None
    assert result.candidate_domain == "ios_alipay"
    assert "INSUFFICIENT_REGIONS" in result.reasons
    assert "INSUFFICIENT_KNOWN_COVERAGE" in result.reasons


def test_unknown_role_does_not_satisfy_accepted_role_diversity() -> None:
    rejected = prediction_from_probabilities(
        region_id="recipient",
        role="recipient",
        probabilities={"ios_alipay": 0.51, "android_alipay": 0.49},
        quality=0.90,
        fit_p_value=0.80,
    )
    result = aggregate_font_domain_predictions(
        document_id="receipt-one-known-role",
        predictions=(
            _prediction("amount-a", "amount"),
            _prediction("amount-b", "amount"),
            rejected,
        ),
    )

    assert result.roles == ("amount",)
    assert result.decision == "UNKNOWN"
    assert "INSUFFICIENT_ROLE_DIVERSITY" in result.reasons


def test_missing_calibration_rejects_an_otherwise_confident_line() -> None:
    prediction = prediction_from_probabilities(
        region_id="amount",
        role="amount",
        probabilities={"ios_alipay": 0.95, "android_alipay": 0.05},
        quality=0.90,
    )

    assert prediction.label == UNKNOWN_DOMAIN
    assert "UNCALIBRATED_MODEL" in prediction.reasons


def test_device_prior_mismatch_is_review_not_authenticity_failure() -> None:
    result = aggregate_font_domain_predictions(
        document_id="receipt-prior",
        predictions=(
            _prediction("amount", "amount"),
            _prediction("recipient", "recipient"),
            _prediction("time", "time"),
        ),
        device_prior_domain="android_alipay",
    )

    assert result.decision == "REVIEW"
    assert result.dominant_domain == "ios_alipay"
    assert result.reasons == ("DEVICE_PRIOR_DOMAIN_CONFLICT",)


def test_predictions_are_sorted_and_duplicate_region_ids_are_rejected() -> None:
    predictions = (
        _prediction("z-last", "time"),
        _prediction("a-first", "amount"),
        _prediction("m-middle", "recipient"),
    )
    result = aggregate_font_domain_predictions(document_id="receipt-order", predictions=predictions)
    assert [line.region_id for line in result.lines] == ["a-first", "m-middle", "z-last"]

    with pytest.raises(ValueError, match="region_id values must be unique"):
        aggregate_font_domain_predictions(
            document_id="receipt-duplicate",
            predictions=(
                _prediction("same", "amount"),
                _prediction("same", "time"),
                _prediction("other", "recipient"),
            ),
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({}, "must not be empty"),
        ({"unknown": 1.0}, "cannot be a trained"),
        ({"ios_alipay": 1.0}, "at least two known domains"),
        ({"ios_alipay": 0.8, "android_alipay": 0.3}, "must sum to one"),
    ],
)
def test_probability_contract_fails_closed(probabilities: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        prediction_from_probabilities(
            region_id="amount",
            role="amount",
            probabilities=probabilities,
        )
