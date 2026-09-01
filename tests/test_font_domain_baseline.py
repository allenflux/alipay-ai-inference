from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from transfer_receipt_ai.font_domain import UNKNOWN_DOMAIN
from transfer_receipt_ai.font_domain_baseline import (
    FEATURE_ABI,
    FEATURE_DIMENSION,
    FontDomainGates,
    FontDomainPublicationSafety,
    FontDomainPrototypeModel,
    extract_font_domain_features,
    fit_font_domain_model,
    load_font_domain_model,
    minimum_conformal_calibration_count,
    predict_document,
    predict_region,
    save_font_domain_model,
)
from transfer_receipt_ai.font_domain_dataset import (
    DOCUMENT_KIND,
    FontDomainDataset,
    load_font_domain_dataset,
)


_DOMAINS = ("round_synthetic", "square_synthetic")
_PERMISSIVE_POC_GATES = FontDomainGates(
    confidence=0.50,
    margin=0.0,
    quality=0.25,
    fit_p_value=0.0,
)


def _synthetic_line(domain: str, variant: int) -> np.ndarray:
    """Render pseudo-glyphs without loading a font or any private image."""

    if domain not in _DOMAINS:
        raise ValueError(domain)
    rgb = np.full((48, 180, 3), 255, dtype=np.uint8)
    seed_offset = 0 if domain == _DOMAINS[0] else 10_000
    random = np.random.default_rng(seed_offset + variant)
    for glyph_index in range(6):
        x = 10 + glyph_index * 27 + int(random.integers(-1, 2))
        y = 8 + int(random.integers(-1, 2))
        if domain == _DOMAINS[0]:
            radius = 7 + variant % 2
            cv2.circle(rgb, (x + 8, y + 15), radius, (0, 0, 0), 2, cv2.LINE_8)
            cv2.line(rgb, (x + 15, y + 8), (x + 15, y + 23), (0, 0, 0), 2)
        else:
            cv2.rectangle(
                rgb,
                (x + 2, y + 5),
                (x + 14 + variant % 2, y + 26),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(rgb, (x + 5, y + 8), (x + 11, y + 18), (255, 255, 255), -1)
    return rgb


def _document_record(
    *,
    document_id: str,
    split: str,
    domain: str,
    role: str,
    relative_image: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": DOCUMENT_KIND,
        "id": document_id,
        "source_group_id": f"source-{document_id}",
        "split": split,
        "font_domain": domain,
        "label_source": "synthetic_unit_test",
        "regions": [
            {
                "id": "line",
                "role": role,
                "image": relative_image,
            }
        ],
    }


def _build_dataset(root: Path, *, include_calibration: bool) -> FontDomainDataset:
    records: list[dict[str, object]] = []
    for domain in _DOMAINS:
        for index in range(6):
            document_id = f"{domain}-train-{index}"
            relative_image = f"images/{document_id}.png"
            image_path = root / relative_image
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_synthetic_line(domain, index), mode="RGB").save(image_path)
            records.append(
                _document_record(
                    document_id=document_id,
                    split="train",
                    domain=domain,
                    role="amount" if index < 3 else "time",
                    relative_image=relative_image,
                )
            )
        if include_calibration:
            for index in range(2):
                document_id = f"{domain}-calibration-{index}"
                relative_image = f"images/{document_id}.png"
                image_path = root / relative_image
                Image.fromarray(_synthetic_line(domain, 20 + index), mode="RGB").save(image_path)
                records.append(
                    _document_record(
                        document_id=document_id,
                        split="calibration",
                        domain=domain,
                        role="amount" if index == 0 else "time",
                        relative_image=relative_image,
                    )
                )
    manifest = root / "font-domain.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return load_font_domain_dataset(manifest, require_labels=True)


@dataclass(frozen=True)
class _CalibratedCorpus:
    dataset: FontDomainDataset
    model: FontDomainPrototypeModel


@pytest.fixture(scope="module")
def calibrated_corpus(tmp_path_factory: pytest.TempPathFactory) -> _CalibratedCorpus:
    dataset = _build_dataset(
        tmp_path_factory.mktemp("font-domain-calibrated"),
        include_calibration=True,
    )
    model = fit_font_domain_model(
        dataset,
        gates=_PERMISSIVE_POC_GATES,
        minimum_train_regions_per_domain=3,
        minimum_role_regions_per_domain=3,
        minimum_calibration_groups_per_domain=1,
    )
    return _CalibratedCorpus(dataset=dataset, model=model)


def _document(dataset: FontDomainDataset, document_id: str):
    return next(document for document in dataset.documents if document.document_id == document_id)


def test_document_prediction_ignores_legacy_device_prior(
    calibrated_corpus: _CalibratedCorpus,
) -> None:
    document = _document(
        calibrated_corpus.dataset,
        f"{_DOMAINS[0]}-calibration-0",
    )
    options = {"minimum_regions": 1, "minimum_roles": 1}

    without_prior = predict_document(
        calibrated_corpus.model,
        replace(document, device_prior_domain=None),
        **options,
    )
    with_contrary_prior = predict_document(
        calibrated_corpus.model,
        replace(document, device_prior_domain=_DOMAINS[1]),
        **options,
    )

    assert with_contrary_prior == without_prior
    assert with_contrary_prior.device_prior_domain is None
    assert "DEVICE_PRIOR_DOMAIN_CONFLICT" not in with_contrary_prior.reasons


def test_legacy_device_proxy_publication_state_still_loads_fail_closed(
    calibrated_corpus: _CalibratedCorpus,
    tmp_path: Path,
) -> None:
    legacy_model = fit_font_domain_model(
        calibrated_corpus.dataset,
        gates=_PERMISSIVE_POC_GATES,
        minimum_train_regions_per_domain=3,
        minimum_role_regions_per_domain=3,
        minimum_calibration_groups_per_domain=1,
        publication_safety=FontDomainPublicationSafety(
            leakage_metadata="device_platform_weak_pseudo",
            near_duplicate_audit="skipped",
        ),
    )
    destination = tmp_path / "legacy-device-proxy.model.json"
    save_font_domain_model(legacy_model, destination)

    loaded = load_font_domain_model(destination)

    assert loaded.publication_safety.leakage_metadata == "device_platform_weak_pseudo"
    assert loaded.publication_safety.required_checks_recorded is False


def test_classical_feature_is_deterministic_finite_and_exactly_64_dimensional() -> None:
    rgb = _synthetic_line(_DOMAINS[0], 7)

    first = extract_font_domain_features(rgb)
    second = extract_font_domain_features(rgb.copy())

    assert FEATURE_ABI == "font-domain-classical-64-v1"
    assert FEATURE_DIMENSION == 64
    assert first.values == second.values
    assert len(first.values) == FEATURE_DIMENSION
    assert np.all(np.isfinite(np.asarray(first.values)))
    assert first.usable is True
    assert first.quality >= 0.25
    assert first.reasons == ("INFORMATION_GATE_PASSED",)


def test_low_information_crop_is_rejected_but_keeps_a_finite_feature_vector(
    calibrated_corpus: _CalibratedCorpus,
    tmp_path: Path,
) -> None:
    blank = np.full((48, 180, 3), 255, dtype=np.uint8)
    feature = extract_font_domain_features(blank)
    assert feature.usable is False
    assert feature.quality < 0.25
    assert len(feature.values) == FEATURE_DIMENSION
    assert np.all(np.isfinite(np.asarray(feature.values)))
    assert "INK_BELOW_128" in feature.reasons

    image = tmp_path / "blank.png"
    Image.fromarray(blank, mode="RGB").save(image)
    manifest = tmp_path / "inference.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": DOCUMENT_KIND,
                "id": "blank-inference",
                "source_group_id": "source-blank-inference",
                "split": "inference",
                "regions": [{"id": "line", "role": "amount", "image": "blank.png"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inference = load_font_domain_dataset(manifest, require_labels=False)
    prediction = predict_region(calibrated_corpus.model, inference.documents[0].regions[0])
    assert prediction.label == UNKNOWN_DOMAIN
    assert prediction.quality < calibrated_corpus.model.gates.quality
    assert "LOW_INFORMATION" in prediction.reasons
    assert "INK_BELOW_128" in prediction.reasons

    zero_quality_gate_model = replace(
        calibrated_corpus.model,
        gates=FontDomainGates(confidence=0.0, margin=0.0, quality=0.0, fit_p_value=0.0),
        model_sha256=None,
    )
    hard_gated = predict_region(zero_quality_gate_model, inference.documents[0].regions[0])
    assert hard_gated.label == UNKNOWN_DOMAIN
    assert "LOW_INFORMATION" in hard_gated.reasons


def test_fit_builds_calibrated_generic_and_balanced_role_prototypes(
    calibrated_corpus: _CalibratedCorpus,
) -> None:
    model = calibrated_corpus.model

    assert model.domains == tuple(sorted(_DOMAINS))
    assert model.training_counts == {domain: 6 for domain in _DOMAINS}
    assert model.calibration_counts == {domain: 2 for domain in _DOMAINS}
    assert model.calibration_source == {domain: "calibration" for domain in _DOMAINS}
    assert set(model.prototypes) == set(_DOMAINS)
    assert set(model.role_prototypes) == {"amount", "time"}
    for prototypes in model.role_prototypes.values():
        assert set(prototypes) == set(_DOMAINS)
        assert all(len(vector) == FEATURE_DIMENSION for vector in prototypes.values())
    assert all(len(vector) == FEATURE_DIMENSION for vector in model.prototypes.values())


def test_conformal_threshold_raises_required_independent_calibration_groups(
    calibrated_corpus: _CalibratedCorpus,
) -> None:
    strict_tail = fit_font_domain_model(
        calibrated_corpus.dataset,
        gates=FontDomainGates(
            confidence=0.0,
            margin=0.0,
            quality=0.25,
            fit_p_value=0.01,
        ),
        minimum_train_regions_per_domain=3,
        minimum_role_regions_per_domain=3,
        minimum_calibration_groups_per_domain=1,
    )
    assert strict_tail.minimum_calibration_groups_per_domain == 100
    region = _document(
        calibrated_corpus.dataset,
        f"{_DOMAINS[0]}-calibration-0",
    ).regions[0]
    prediction = predict_region(strict_tail, region)
    assert prediction.label == UNKNOWN_DOMAIN
    assert "INSUFFICIENT_CALIBRATION_SUPPORT" in prediction.reasons


def test_conformal_threshold_rejects_unrepresentable_tiny_alpha() -> None:
    with pytest.raises(ValueError, match="conformal alpha is too small"):
        minimum_conformal_calibration_count(5e-324)


@pytest.mark.parametrize("domain", _DOMAINS)
def test_calibrated_region_prediction_uses_role_and_generic_prototypes(
    calibrated_corpus: _CalibratedCorpus,
    domain: str,
) -> None:
    document = _document(calibrated_corpus.dataset, f"{domain}-calibration-0")
    role_prediction = predict_region(calibrated_corpus.model, document.regions[0])

    assert role_prediction.label == domain
    assert role_prediction.candidate_domain == domain
    assert role_prediction.accepted is True
    assert role_prediction.generic_fallback is False
    assert role_prediction.distances is not None
    assert role_prediction.distances[domain] == min(role_prediction.distances.values())
    assert "KNOWN_DOMAIN_EVIDENCE" in role_prediction.reasons

    unseen_role_region = replace(document.regions[0], role="recipient")
    generic_prediction = predict_region(calibrated_corpus.model, unseen_role_region)
    assert generic_prediction.label == domain
    assert generic_prediction.candidate_domain == domain
    assert generic_prediction.generic_fallback is True


def test_model_save_load_no_clobber_and_tamper_detection(
    calibrated_corpus: _CalibratedCorpus,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.json"
    publication = save_font_domain_model(calibrated_corpus.model, destination)
    original = destination.read_bytes()

    assert publication["path"] == destination.resolve().as_posix()
    assert publication["size_bytes"] == len(original)
    assert publication["model_sha256"] == calibrated_corpus.model.model_sha256
    loaded = load_font_domain_model(destination)
    assert loaded.as_dict() == calibrated_corpus.model.as_dict()
    assert loaded.dataset_snapshot_sha256 == calibrated_corpus.dataset.snapshot_sha256
    assert loaded.publication_safety.near_duplicate_audit == "not_run"

    with pytest.raises(FileExistsError, match="refusing to overwrite model artifact"):
        save_font_domain_model(calibrated_corpus.model, destination)
    assert destination.read_bytes() == original

    tampered_payload = json.loads(original)
    tampered_payload["prototypes"][_DOMAINS[0]][0] += 0.125
    tampered = tmp_path / "tampered-model.json"
    tampered.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model_sha256 does not match decision fields"):
        load_font_domain_model(tampered)


def test_train_only_model_defaults_to_unknown_and_requires_explicit_poc_override(
    tmp_path: Path,
) -> None:
    dataset = _build_dataset(tmp_path / "uncalibrated", include_calibration=False)
    model = fit_font_domain_model(
        dataset,
        gates=_PERMISSIVE_POC_GATES,
        minimum_train_regions_per_domain=3,
        minimum_role_regions_per_domain=3,
    )
    assert model.calibration_source == {domain: "train_fallback" for domain in _DOMAINS}

    region = _document(dataset, f"{_DOMAINS[0]}-train-0").regions[0]
    conservative = predict_region(model, region)
    assert conservative.candidate_domain == _DOMAINS[0]
    assert conservative.label == UNKNOWN_DOMAIN
    assert "UNCALIBRATED_MODEL" in conservative.reasons

    proof_of_concept = predict_region(model, region, allow_uncalibrated=True)
    assert proof_of_concept.candidate_domain == _DOMAINS[0]
    assert proof_of_concept.label == _DOMAINS[0]
    assert "UNCALIBRATED_TRAIN_FALLBACK" in proof_of_concept.reasons
    assert "UNCALIBRATED_MODEL" not in proof_of_concept.reasons
