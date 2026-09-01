"""Pure font-domain prediction gates and document-level consistency rules.

The font-domain feature/model implementation deliberately feeds this module
through a small, model-agnostic line contract.  A future CNN/ONNX classifier
can therefore replace the classical validation baseline without changing the
document sidecar semantics.

These outputs are review evidence, not authenticity verdicts.  In particular,
``PASS`` means only that the included text regions are mutually consistent
with one known rendering domain under the supplied model.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final


SCHEMA_VERSION: Final[int] = 1
LINE_KIND: Final[str] = "receipt_font_domain_line_prediction_v1"
RESULT_KIND: Final[str] = "receipt_font_domain_consistency_v1"
UNKNOWN_DOMAIN: Final[str] = "unknown"
DECISIONS: Final[tuple[str, ...]] = ("PASS", "REVIEW", "UNKNOWN")
_DOMAIN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _finite_unit(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must be finite and between 0 and 1")
    return result


def _nonempty(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value.strip()


def _domain(value: object, *, description: str) -> str:
    domain = _nonempty(value, description=description)
    if value != domain or domain == UNKNOWN_DOMAIN or not _DOMAIN_PATTERN.fullmatch(domain):
        raise ValueError(f"{description} must be a known domain matching {_DOMAIN_PATTERN.pattern!r}")
    return domain


@dataclass(frozen=True)
class FontDomainLinePrediction:
    """One region's gated domain evidence.

    ``confidence`` is classifier support, not authenticity probability.
    ``candidate_domain`` remains visible when the accepted ``label`` is
    ``unknown`` so a reviewer can distinguish low margin from low quality.
    """

    region_id: str
    role: str
    label: str
    candidate_domain: str | None
    confidence: float
    margin: float
    fit_p_value: float | None
    quality: float
    include_in_consistency: bool
    generic_fallback: bool
    supports: Mapping[str, float]
    distances: Mapping[str, float] | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.region_id, description="region_id")
        _nonempty(self.role, description="role")
        if self.label != UNKNOWN_DOMAIN:
            _domain(self.label, description="label")
        if self.candidate_domain is not None:
            _domain(self.candidate_domain, description="candidate_domain")
        _finite_unit(self.confidence, description="confidence")
        _finite_unit(self.margin, description="margin")
        if self.fit_p_value is not None:
            _finite_unit(self.fit_p_value, description="fit_p_value")
        _finite_unit(self.quality, description="quality")
        if not isinstance(self.include_in_consistency, bool):
            raise ValueError("include_in_consistency must be boolean")
        if not isinstance(self.generic_fallback, bool):
            raise ValueError("generic_fallback must be boolean")
        if len(self.supports) < 2:
            raise ValueError("supports must contain at least two known domains")
        for domain, support in self.supports.items():
            _domain(domain, description="support domain")
            _finite_unit(support, description=f"support for {domain}")
        support_total = sum(float(value) for value in self.supports.values())
        if not math.isclose(support_total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"supports must sum to one; found {support_total}")
        if self.candidate_domain is not None:
            if self.candidate_domain not in self.supports:
                raise ValueError("candidate_domain must be present in supports")
            maximum = max(float(value) for value in self.supports.values())
            if not math.isclose(
                float(self.supports[self.candidate_domain]), maximum, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError("candidate_domain must have maximum support")
            ordered_supports = sorted(
                (float(value) for value in self.supports.values()), reverse=True
            )
            expected_margin = ordered_supports[0] - ordered_supports[1]
            if not math.isclose(self.confidence, maximum, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("confidence must equal candidate_domain support")
            if not math.isclose(self.margin, expected_margin, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("margin must equal the top-two support difference")
        if self.label != UNKNOWN_DOMAIN:
            if self.candidate_domain != self.label or self.label not in self.supports:
                raise ValueError("an accepted label must equal candidate_domain and exist in supports")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("reasons must be a non-empty tuple")
        for reason in self.reasons:
            _nonempty(reason, description="reason")
        if self.label == UNKNOWN_DOMAIN and self.reasons == ("KNOWN_DOMAIN_EVIDENCE",):
            raise ValueError("an unknown prediction must contain a rejection reason")
        if self.distances is not None:
            if set(self.distances) != set(self.supports):
                raise ValueError("distances and supports must contain the same domains")
            for domain, distance in self.distances.items():
                if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                    raise ValueError(f"distance for {domain} must be numeric")
                if not math.isfinite(float(distance)) or float(distance) < 0.0:
                    raise ValueError(f"distance for {domain} must be finite and non-negative")

    @property
    def accepted(self) -> bool:
        return self.label != UNKNOWN_DOMAIN

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": LINE_KIND,
            "region_id": self.region_id,
            "role": self.role,
            "label": self.label,
            "candidate_domain": self.candidate_domain,
            "confidence": round(float(self.confidence), 6),
            "margin": round(float(self.margin), 6),
            "fit_p_value": None if self.fit_p_value is None else round(float(self.fit_p_value), 6),
            "quality": round(float(self.quality), 6),
            "include_in_consistency": self.include_in_consistency,
            "generic_fallback": self.generic_fallback,
            "supports": {
                domain: round(float(self.supports[domain]), 6)
                for domain in sorted(self.supports)
            },
            "distances": None
            if self.distances is None
            else {
                domain: round(float(self.distances[domain]), 6)
                for domain in sorted(self.distances)
            },
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FontDomainConsistencyResult:
    document_id: str
    decision: str
    dominant_domain: str | None
    candidate_domain: str | None
    known_coverage: float
    consistency_score: float
    support_ratio: float
    included_regions: int
    accepted_regions: int
    unknown_regions: int
    roles: tuple[str, ...]
    conflicts: tuple[str, ...]
    reasons: tuple[str, ...]
    lines: tuple[FontDomainLinePrediction, ...]
    device_prior_domain: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.document_id, description="document_id")
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if self.dominant_domain is not None:
            _nonempty(self.dominant_domain, description="dominant_domain")
        if self.candidate_domain is not None:
            _nonempty(self.candidate_domain, description="candidate_domain")
        _finite_unit(self.known_coverage, description="known_coverage")
        _finite_unit(self.consistency_score, description="consistency_score")
        _finite_unit(self.support_ratio, description="support_ratio")
        for name, count in (
            ("included_regions", self.included_regions),
            ("accepted_regions", self.accepted_regions),
            ("unknown_regions", self.unknown_regions),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.accepted_regions + self.unknown_regions != self.included_regions:
            raise ValueError("accepted and unknown counts must close to included_regions")
        line_ids = {line.region_id for line in self.lines}
        if len(line_ids) != len(self.lines):
            raise ValueError("result lines must have unique region_id values")
        if any(conflict not in line_ids for conflict in self.conflicts):
            raise ValueError("every conflict must identify a result line")
        if self.decision == "PASS" and self.dominant_domain is None:
            raise ValueError("PASS requires a dominant_domain")
        if self.decision == "UNKNOWN" and self.dominant_domain is not None:
            raise ValueError("UNKNOWN cannot claim a dominant_domain")
        if self.dominant_domain is not None and self.dominant_domain != self.candidate_domain:
            raise ValueError("dominant_domain must equal candidate_domain")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": RESULT_KIND,
            "document_id": self.document_id,
            "decision": self.decision,
            "dominant_domain": self.dominant_domain,
            "candidate_domain": self.candidate_domain,
            "known_coverage": round(float(self.known_coverage), 6),
            "consistency_score": round(float(self.consistency_score), 6),
            "support_ratio": round(float(self.support_ratio), 6),
            "included_regions": self.included_regions,
            "accepted_regions": self.accepted_regions,
            "unknown_regions": self.unknown_regions,
            "roles": list(self.roles),
            "conflicts": list(self.conflicts),
            "device_prior_domain": self.device_prior_domain,
            "authenticity": "not_assessed",
            "requires_manual_review": self.decision != "PASS",
            "reasons": list(self.reasons),
            "lines": [line.as_dict() for line in self.lines],
        }


def prediction_from_probabilities(
    *,
    region_id: str,
    role: str,
    probabilities: Mapping[str, float],
    quality: float = 1.0,
    include_in_consistency: bool = True,
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.08,
    quality_threshold: float = 0.25,
    fit_p_value: float | None = None,
    fit_p_threshold: float = 0.05,
    require_fit_p_value: bool = True,
    generic_fallback: bool = False,
) -> FontDomainLinePrediction:
    """Gate a future CNN probability vector into known/UNKNOWN evidence."""

    for name, threshold in (
        ("confidence_threshold", confidence_threshold),
        ("margin_threshold", margin_threshold),
        ("quality_threshold", quality_threshold),
        ("fit_p_threshold", fit_p_threshold),
    ):
        _finite_unit(threshold, description=name)
    if not isinstance(require_fit_p_value, bool):
        raise ValueError("require_fit_p_value must be boolean")
    quality = _finite_unit(quality, description="quality")
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    values: dict[str, float] = {}
    for raw_domain, raw_probability in probabilities.items():
        domain = _nonempty(raw_domain, description="probability domain")
        if raw_domain != domain:
            raise ValueError("probability domains must use canonical unpadded names")
        if domain == UNKNOWN_DOMAIN:
            raise ValueError("unknown cannot be a trained probability class")
        if domain in values:
            raise ValueError(f"duplicate probability domain {domain!r}")
        values[domain] = _finite_unit(raw_probability, description=f"probability for {domain}")
    if len(values) < 2:
        raise ValueError("probabilities must contain at least two known domains")
    total = sum(values.values())
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(f"probabilities must sum to one; found {total}")
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    candidate, confidence = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = max(0.0, confidence - runner_up)
    reasons: list[str] = []
    if quality < quality_threshold:
        reasons.append("LOW_INFORMATION")
    if confidence < confidence_threshold:
        reasons.append("LOW_DOMAIN_SUPPORT")
    if margin < margin_threshold:
        reasons.append("LOW_DOMAIN_MARGIN")
    if fit_p_value is not None:
        fit_p_value = _finite_unit(fit_p_value, description="fit_p_value")
        if fit_p_value < fit_p_threshold:
            reasons.append("OUT_OF_DISTRIBUTION")
    elif require_fit_p_value:
        reasons.append("UNCALIBRATED_MODEL")
    label = candidate if not reasons else UNKNOWN_DOMAIN
    return FontDomainLinePrediction(
        region_id=region_id,
        role=role,
        label=label,
        candidate_domain=candidate,
        confidence=confidence,
        margin=margin,
        fit_p_value=fit_p_value,
        quality=quality,
        include_in_consistency=include_in_consistency,
        generic_fallback=generic_fallback,
        supports=values,
        distances=None,
        reasons=tuple(reasons or ("KNOWN_DOMAIN_EVIDENCE",)),
    )


def aggregate_font_domain_predictions(
    *,
    document_id: str,
    predictions: Sequence[FontDomainLinePrediction],
    minimum_regions: int = 3,
    minimum_roles: int = 2,
    minimum_known_coverage: float = 0.60,
    pass_support_ratio: float = 0.75,
    minimum_candidate_ratio: float = 0.60,
    strong_conflict_confidence: float = 0.75,
    strong_conflict_margin: float = 0.15,
    strong_conflict_quality: float = 0.50,
    device_prior_domain: str | None = None,
) -> FontDomainConsistencyResult:
    """Aggregate accepted line evidence into PASS/REVIEW/UNKNOWN.

    UNKNOWN lines reduce coverage but never create a font conflict.  A known,
    high-quality line assigned to a different domain is a review signal; this
    is intentionally fail-safe because a single edited amount line can be the
    important anomaly.
    """

    document_id = _nonempty(document_id, description="document_id")
    if minimum_regions < 1 or minimum_roles < 1:
        raise ValueError("minimum_regions and minimum_roles must be positive")
    for name, threshold in (
        ("minimum_known_coverage", minimum_known_coverage),
        ("pass_support_ratio", pass_support_ratio),
        ("minimum_candidate_ratio", minimum_candidate_ratio),
        ("strong_conflict_confidence", strong_conflict_confidence),
        ("strong_conflict_margin", strong_conflict_margin),
        ("strong_conflict_quality", strong_conflict_quality),
    ):
        _finite_unit(threshold, description=name)
    if minimum_candidate_ratio > pass_support_ratio:
        raise ValueError("minimum_candidate_ratio cannot exceed pass_support_ratio")
    if device_prior_domain is not None:
        device_prior_domain = _domain(device_prior_domain, description="device_prior_domain")

    ordered = tuple(sorted(predictions, key=lambda value: (value.region_id, value.role)))
    if len({line.region_id for line in ordered}) != len(ordered):
        raise ValueError("region_id values must be unique within one document")
    included = tuple(line for line in ordered if line.include_in_consistency)
    accepted = tuple(line for line in included if line.accepted)
    roles = tuple(sorted({line.role for line in accepted}))
    included_count = len(included)
    accepted_count = len(accepted)
    unknown_count = included_count - accepted_count
    known_coverage = accepted_count / included_count if included_count else 0.0

    reasons: list[str] = []
    candidate: str | None = None
    support_ratio = 0.0
    consistency_score = 0.0
    conflicts: list[str] = []
    strong_conflicts: list[str] = []

    if accepted:
        score_by_domain: dict[str, float] = defaultdict(float)
        count_by_domain: dict[str, int] = defaultdict(int)
        for line in accepted:
            weight = max(1e-9, line.quality * line.confidence)
            score_by_domain[line.label] += weight
            count_by_domain[line.label] += 1
        candidate = sorted(score_by_domain, key=lambda domain: (-score_by_domain[domain], domain))[0]
        total_score = sum(score_by_domain.values())
        consistency_score = score_by_domain[candidate] / total_score if total_score else 0.0
        support_ratio = count_by_domain[candidate] / accepted_count
        for line in accepted:
            if line.label == candidate:
                continue
            # Every accepted cross-domain line is inconsistent by contract.
            # The stronger thresholds describe severity; they never turn an
            # accepted conflicting domain into a document PASS.
            conflicts.append(line.region_id)
            if (
                line.quality >= strong_conflict_quality
                and line.confidence >= strong_conflict_confidence
                and line.margin >= strong_conflict_margin
            ):
                strong_conflicts.append(line.region_id)

    insufficient = False
    if included_count < minimum_regions:
        reasons.append("INSUFFICIENT_REGIONS")
        insufficient = True
    if len(roles) < minimum_roles:
        reasons.append("INSUFFICIENT_ROLE_DIVERSITY")
        insufficient = True
    if known_coverage < minimum_known_coverage:
        reasons.append("INSUFFICIENT_KNOWN_COVERAGE")
        insufficient = True
    if candidate is None or support_ratio < minimum_candidate_ratio:
        reasons.append("NO_STABLE_DOMINANT_DOMAIN")
        insufficient = True

    if conflicts:
        decision = "REVIEW"
        dominant = candidate
        reasons.append("CROSS_DOMAIN_REGION_CONFLICT")
        if strong_conflicts:
            reasons.append("STRONG_CROSS_DOMAIN_REGION_CONFLICT")
    elif insufficient:
        decision = "UNKNOWN"
        dominant = None
    else:
        dominant = candidate
        if device_prior_domain is not None and candidate != device_prior_domain:
            decision = "REVIEW"
            reasons.append("DEVICE_PRIOR_DOMAIN_CONFLICT")
        elif support_ratio < pass_support_ratio:
            decision = "REVIEW"
            reasons.append("DOMINANT_DOMAIN_SUPPORT_BELOW_PASS_GATE")
        else:
            decision = "PASS"
            reasons.append("DOMINANT_DOMAIN_CONSENSUS")

    return FontDomainConsistencyResult(
        document_id=document_id,
        decision=decision,
        dominant_domain=dominant,
        candidate_domain=candidate,
        known_coverage=known_coverage,
        consistency_score=consistency_score,
        support_ratio=support_ratio,
        included_regions=included_count,
        accepted_regions=accepted_count,
        unknown_regions=unknown_count,
        roles=roles,
        conflicts=tuple(sorted(conflicts)),
        reasons=tuple(reasons),
        lines=ordered,
        device_prior_domain=device_prior_domain,
    )
