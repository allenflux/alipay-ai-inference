"""Command-line workflow for the isolated font-domain consistency sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .font_domain import SCHEMA_VERSION
from .font_domain_baseline import (
    FEATURE_ABI,
    MODEL_KIND,
    FontDomainGates,
    FontDomainPublicationSafety,
    fit_font_domain_model,
    load_font_domain_model,
    predict_document,
    save_font_domain_model,
)
from .font_domain_dataset import (
    audit_near_duplicate_splits,
    load_font_domain_dataset,
    write_classifier_manifest,
)


RUN_KIND = "receipt_font_domain_run_v1"
DEVICE_PLATFORM_WEAK_LABEL_SOURCE = "device_platform_weak_pseudo_v1"


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without resolving the final symlink identity."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _atomic_write_bytes_no_clobber(path: Path, data: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        try:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _print_json(value: object) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


def _load_training_dataset(args: argparse.Namespace) -> Any:
    dataset = load_font_domain_dataset(
        args.records,
        require_labels=True,
        require_leakage_metadata=not args.allow_incomplete_leakage_metadata,
    )
    near_duplicate_audit: dict[str, object] | None = None
    if not args.skip_near_duplicate_audit:
        near_duplicate_audit = audit_near_duplicate_splits(
            dataset,
            maximum_hamming_distance=args.maximum_phash_distance,
            maximum_regions=args.maximum_near_duplicate_regions,
        )
    return dataset, near_duplicate_audit


def _classification_metrics(
    labeled_predictions: list[tuple[str, str | None]],
) -> dict[str, object]:
    """Summarize selective classification without treating abstention as coverage."""

    total = len(labeled_predictions)
    covered = sum(prediction is not None for _, prediction in labeled_predictions)
    correct = sum(
        prediction is not None and prediction == reference
        for reference, prediction in labeled_predictions
    )
    prediction_counts = Counter(
        prediction if prediction is not None else "unknown"
        for _, prediction in labeled_predictions
    )
    return {
        "total": total,
        "covered": covered,
        "correct": correct,
        "classification_coverage": round(covered / total, 6) if total else 0.0,
        "selective_accuracy": round(correct / covered, 6) if covered else None,
        "overall_accuracy": round(correct / total, 6) if total else 0.0,
        "prediction_counts": {
            prediction: int(prediction_counts[prediction])
            for prediction in sorted(prediction_counts)
        },
    }


def _metrics_with_domains(
    labeled_predictions: list[tuple[str, str | None]],
) -> dict[str, object]:
    metrics = _classification_metrics(labeled_predictions)
    metrics["per_domain"] = {
        domain: _classification_metrics(
            [
                (reference, prediction)
                for reference, prediction in labeled_predictions
                if reference == domain
            ]
        )
        for domain in sorted({reference for reference, _ in labeled_predictions})
    }
    return metrics


def _evaluate_held_out(
    model: Any,
    dataset: Any,
    *,
    test_label_sources: list[str],
) -> dict[str, object]:
    """Evaluate the fitted model on labeled test documents only.

    Coverage is reported separately from selective accuracy so UNKNOWN
    abstentions cannot make the result look more accurate than it is. The
    overall accuracy denominator includes abstentions.
    """

    test_documents = [
        document
        for document in dataset.documents
        if document.split == "test" and document.font_domain is not None
    ]
    if not test_documents:
        return {
            "status": "not_available",
            "authenticity": "not_assessed",
            "device_prior_used": False,
            "reference_label_used_for_prediction": False,
            "reference_label_sources": test_label_sources,
            "documents": _metrics_with_domains([]),
            "regions": _metrics_with_domains([]),
            "decision_counts": {decision: 0 for decision in ("PASS", "REVIEW", "UNKNOWN")},
            "decision_rates": {decision: 0.0 for decision in ("PASS", "REVIEW", "UNKNOWN")},
        }

    document_predictions: list[tuple[str, str | None]] = []
    region_predictions: list[tuple[str, str | None]] = []
    decision_counts: Counter[str] = Counter()
    for document in test_documents:
        reference = document.font_domain
        # The filter above establishes this for the type checker and keeps
        # inference-only rows out of the held-out denominator.
        if reference is None:  # pragma: no cover - defensive invariant
            continue
        # ``device_prior_domain`` is derived from the same device classifier as
        # the weak reference label in bootstrap datasets.  Remove it here so
        # held-out decisions and metrics are image-feature-only evidence.
        result = predict_document(model, replace(document, device_prior_domain=None))
        document_predictions.append((reference, result.dominant_domain))
        decision_counts[result.decision] += 1
        region_predictions.extend(
            (reference, line.label if line.accepted else None)
            for line in result.lines
            if line.include_in_consistency
        )

    weak_test_labels = DEVICE_PLATFORM_WEAK_LABEL_SOURCE in test_label_sources
    status = "test_split_labels_not_independently_assessed"
    if weak_test_labels:
        status = (
            "weak_pseudo_test_split"
            if test_label_sources == [DEVICE_PLATFORM_WEAK_LABEL_SOURCE]
            else "mixed_including_weak_pseudo_test_split"
        )
    total_documents = len(document_predictions)
    return {
        "status": status,
        "authenticity": "not_assessed",
        "device_prior_used": False,
        "reference_label_used_for_prediction": False,
        "reference_label_sources": test_label_sources,
        "documents": _metrics_with_domains(document_predictions),
        "regions": _metrics_with_domains(region_predictions),
        "decision_counts": {
            decision: int(decision_counts[decision])
            for decision in ("PASS", "REVIEW", "UNKNOWN")
        },
        "decision_rates": {
            decision: round(decision_counts[decision] / total_documents, 6)
            for decision in ("PASS", "REVIEW", "UNKNOWN")
        },
    }


def _validate(args: argparse.Namespace) -> dict[str, object]:
    require_labels: bool | None
    if args.mode == "training":
        require_labels = True
    elif args.mode == "inference":
        require_labels = False
    else:
        require_labels = None
    dataset = load_font_domain_dataset(
        args.records,
        require_labels=require_labels,
        require_leakage_metadata=(
            args.mode != "inference" and not args.allow_incomplete_leakage_metadata
        ),
    )
    result: dict[str, object] = {
        "validation": "passed",
        "mode": args.mode,
        "dataset": dataset.summary(),
    }
    if args.mode != "inference" and not args.skip_near_duplicate_audit:
        result["near_duplicate_audit"] = audit_near_duplicate_splits(
            dataset,
            maximum_hamming_distance=args.maximum_phash_distance,
            maximum_regions=args.maximum_near_duplicate_regions,
        )
    return result


def _fit(args: argparse.Namespace) -> dict[str, object]:
    dataset, near_duplicate_audit = _load_training_dataset(args)
    label_sources = sorted(
        {
            document.label_source
            for document in dataset.documents
            if document.label_source is not None
        }
    )
    training_label_sources = sorted(
        {
            document.label_source
            for document in dataset.documents
            if document.split in {"train", "calibration"}
            and document.label_source is not None
        }
    )
    test_label_sources = sorted(
        {
            document.label_source
            for document in dataset.documents
            if document.split == "test" and document.label_source is not None
        }
    )
    weak_device_training_labels = (
        DEVICE_PLATFORM_WEAK_LABEL_SOURCE in training_label_sources
    )
    # A device-platform pseudo label is useful for a zero-touch pilot, but it
    # is not independent font-domain ground truth.  Persist that limitation in
    # the self-hashed publication safety state so the resulting model can only
    # run through the explicit experimental path.
    leakage_metadata_status = "required_and_present"
    if weak_device_training_labels:
        leakage_metadata_status = "device_platform_weak_pseudo"
    elif args.allow_incomplete_leakage_metadata:
        leakage_metadata_status = "incomplete_allowed"
    if near_duplicate_audit is None:
        publication_safety = FontDomainPublicationSafety(
            leakage_metadata=leakage_metadata_status,
            near_duplicate_audit="skipped",
        )
    else:
        publication_safety = FontDomainPublicationSafety(
            leakage_metadata=leakage_metadata_status,
            near_duplicate_audit="passed",
            perceptual_hash_abi=str(near_duplicate_audit["perceptual_hash_abi"]),
            maximum_hamming_distance=int(near_duplicate_audit["maximum_hamming_distance"]),
            checked_regions=int(near_duplicate_audit["checked_regions"]),
            cross_split_comparisons=int(near_duplicate_audit["cross_split_comparisons"]),
        )
    model = fit_font_domain_model(
        dataset,
        gates=FontDomainGates(
            confidence=args.confidence_threshold,
            margin=args.margin_threshold,
            quality=args.quality_threshold,
            fit_p_value=args.fit_p_threshold,
        ),
        minimum_train_regions_per_domain=args.minimum_train_regions_per_domain,
        minimum_role_regions_per_domain=args.minimum_role_regions_per_domain,
        minimum_calibration_groups_per_domain=args.minimum_calibration_groups_per_domain,
        publication_safety=publication_safety,
    )
    fallback_domains = sorted(
        domain
        for domain, source in model.calibration_source.items()
        if source == "train_fallback"
    )
    insufficient_domains = sorted(
        domain
        for domain in model.domains
        if model.calibration_source[domain] == "calibration"
        and model.calibration_group_counts[domain]
        < model.minimum_calibration_groups_per_domain
    )
    unready_domains = sorted(set(fallback_domains) | set(insufficient_domains))
    if unready_domains and not args.allow_uncalibrated_model:
        raise ValueError(
            "independent calibration source groups are insufficient for domains: "
            + ", ".join(unready_domains)
            + f" (required per domain: {model.minimum_calibration_groups_per_domain}). "
            "Use --allow-uncalibrated-model only for an UNKNOWN-by-default PoC model."
        )
    held_out_evaluation = _evaluate_held_out(
        model,
        dataset,
        test_label_sources=test_label_sources,
    )
    publication = save_font_domain_model(model, args.output)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "feature_abi": FEATURE_ABI,
        "dataset": dataset.summary(),
        "near_duplicate_audit": near_duplicate_audit,
        "model": publication,
        "domains": list(model.domains),
        "training_counts": dict(model.training_counts),
        "calibration_counts": dict(model.calibration_counts),
        "calibration_group_counts": dict(model.calibration_group_counts),
        "calibration_source": dict(model.calibration_source),
        "rejected_counts": dict(model.rejected_counts),
        "minimum_calibration_groups_per_domain": model.minimum_calibration_groups_per_domain,
        "calibration_prerequisites_met": not unready_domains,
        "ood_gate_effective": model.gates.fit_p_value > 0.0 and not unready_domains,
        "publication_safety": model.publication_safety.as_dict(),
        "label_sources": label_sources,
        "training_label_sources": training_label_sources,
        "test_label_sources": test_label_sources,
        "label_evidence_status": (
            "device_platform_weak_pseudo"
            if weak_device_training_labels
            else "not_independently_assessed"
        ),
        "held_out_evaluation": held_out_evaluation,
        "publication_prerequisites_recorded": (
            not unready_domains
            and model.gates.fit_p_value > 0.0
            and model.minimum_calibration_groups_per_domain >= 20
            and model.publication_safety.required_checks_recorded
            and (model.publication_safety.maximum_hamming_distance or -1) >= 8
        ),
        "authenticity": "not_assessed",
    }


def _export_classifier(args: argparse.Namespace) -> dict[str, object]:
    dataset, near_duplicate_audit = _load_training_dataset(args)
    publication = write_classifier_manifest(dataset, args.output)
    return {
        "dataset": dataset.summary(),
        "near_duplicate_audit": near_duplicate_audit,
        "classifier_manifest": publication,
        "dataset_root": dataset.root.as_posix(),
        "note": "The exported manifest stays beside the source manifest so image paths remain bound.",
    }


def _bootstrap_existing(args: argparse.Namespace) -> dict[str, object]:
    from .font_domain_bootstrap import bootstrap_existing_pseudolabels

    return bootstrap_existing_pseudolabels(
        args.records,
        args.output,
        minimum_device_confidence=args.minimum_device_confidence,
        minimum_regions=args.minimum_regions,
        maximum_documents_per_domain=args.maximum_documents_per_domain,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
    )


def _analyze(args: argparse.Namespace) -> dict[str, object]:
    model = load_font_domain_model(args.model)
    calibration_ready = all(
        model.calibration_source[domain] == "calibration"
        and model.calibration_group_counts[domain]
        >= model.minimum_calibration_groups_per_domain
        for domain in model.domains
    )
    publication_prerequisites_recorded = (
        calibration_ready
        and model.gates.fit_p_value > 0.0
        and model.minimum_calibration_groups_per_domain >= 20
        and model.publication_safety.required_checks_recorded
        and (model.publication_safety.maximum_hamming_distance or -1) >= 8
    )
    if not publication_prerequisites_recorded and not args.allow_experimental_model:
        raise ValueError(
            "model lacks the default calibration/publication prerequisites; "
            "use --allow-experimental-model only for an explicitly experimental sidecar run"
        )
    dataset = load_font_domain_dataset(
        args.records,
        require_labels=False,
    )
    results = [
        predict_document(
            model,
            document,
            minimum_regions=args.minimum_regions,
            minimum_roles=args.minimum_roles,
            minimum_known_coverage=args.minimum_known_coverage,
            pass_support_ratio=args.pass_support_ratio,
        )
        for document in dataset.documents
    ]
    sidecar_rows: list[dict[str, object]] = []
    for result in results:
        row = result.as_dict()
        row["model_evidence"] = {
            "model_sha256": model.model_sha256,
            "feature_abi": FEATURE_ABI,
            "publication_prerequisites_recorded": publication_prerequisites_recorded,
            "evaluation_status": "not_assessed",
        }
        if not publication_prerequisites_recorded:
            row["requires_manual_review"] = True
        sidecar_rows.append(row)
    sidecar_bytes = _jsonl_bytes(sidecar_rows)
    errors_bytes = b""
    decision_counts = Counter(result.decision for result in results)
    run: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": args.model.expanduser().resolve(strict=True).as_posix(),
            "kind": MODEL_KIND,
            "model_sha256": model.model_sha256,
            "feature_abi": FEATURE_ABI,
            "domains": list(model.domains),
            "calibration_source": dict(model.calibration_source),
            "calibration_group_counts": dict(model.calibration_group_counts),
            "minimum_calibration_groups_per_domain": model.minimum_calibration_groups_per_domain,
            "publication_safety": model.publication_safety.as_dict(),
            "publication_prerequisites_recorded": publication_prerequisites_recorded,
            "evaluation_status": "not_assessed",
        },
        "input": dataset.summary(),
        "aggregation": {
            "minimum_regions": args.minimum_regions,
            "minimum_roles": args.minimum_roles,
            "minimum_known_coverage": args.minimum_known_coverage,
            "pass_support_ratio": args.pass_support_ratio,
        },
        "documents": len(results),
        "decisions": {
            decision: int(decision_counts[decision])
            for decision in ("PASS", "REVIEW", "UNKNOWN")
        },
        "failures": 0,
        "outputs": {
            "font_domain.sidecar.jsonl": {
                "sha256": _sha256(sidecar_bytes),
                "size_bytes": len(sidecar_bytes),
                "records": len(sidecar_rows),
            },
            "errors.jsonl": {
                "sha256": _sha256(errors_bytes),
                "size_bytes": 0,
                "records": 0,
            },
        },
        "authenticity": "not_assessed",
    }
    run_bytes = _json_bytes(run, pretty=True)

    output = _absolute_lexical(args.output)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite analysis output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, mode=0o700)
    _atomic_write_bytes_no_clobber(output / "font_domain.sidecar.jsonl", sidecar_bytes)
    _atomic_write_bytes_no_clobber(output / "errors.jsonl", errors_bytes)
    # run.json is the completion marker and is intentionally published last.
    _atomic_write_bytes_no_clobber(output / "run.json", run_bytes)
    return {
        "output": output.as_posix(),
        "run": run,
        "run_sha256": _sha256(run_bytes),
    }


def _add_training_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-incomplete-leakage-metadata",
        action="store_true",
        help="Exploratory only: permit missing content_group_id/source_image_sha256 bindings",
    )
    parser.add_argument(
        "--skip-near-duplicate-audit",
        action="store_true",
        help="Skip bounded pHash cross-split audit (not suitable for model publication)",
    )
    parser.add_argument("--maximum-phash-distance", type=int, default=8)
    parser.add_argument("--maximum-near-duplicate-regions", type=int, default=5000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, fit and run the isolated receipt font-domain consistency sidecar; "
            "outputs are not authenticity verdicts"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap-existing",
        help="Build a zero-touch iOS/Android weak-label pilot from existing OCR crops",
    )
    bootstrap.add_argument("--records", type=Path, required=True, help="Existing pseudo_labels.jsonl")
    bootstrap.add_argument("--output", type=Path, required=True, help="New bootstrap dataset directory")
    bootstrap.add_argument("--minimum-device-confidence", type=float, default=0.90)
    bootstrap.add_argument("--minimum-regions", type=int, default=3)
    bootstrap.add_argument("--maximum-documents-per-domain", type=int, default=500)
    bootstrap.add_argument("--split-seed", default="font-domain-device-pseudo-v1")
    bootstrap.add_argument("--train-ratio", type=float, default=0.60)
    bootstrap.add_argument("--calibration-ratio", type=float, default=0.20)

    validate = subparsers.add_parser("validate", help="Validate a document-level manifest")
    validate.add_argument("--records", type=Path, required=True)
    validate.add_argument("--mode", choices=("training", "inference", "mixed"), default="training")
    _add_training_safety_options(validate)

    fit = subparsers.add_parser("fit", help="Fit and self-hash the 64-D prototype baseline")
    fit.add_argument("--records", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--minimum-train-regions-per-domain", type=int, default=3)
    fit.add_argument("--minimum-role-regions-per-domain", type=int, default=3)
    fit.add_argument("--minimum-calibration-groups-per-domain", type=int, default=20)
    fit.add_argument("--confidence-threshold", type=float, default=0.60)
    fit.add_argument("--margin-threshold", type=float, default=0.08)
    fit.add_argument("--quality-threshold", type=float, default=0.25)
    fit.add_argument("--fit-p-threshold", type=float, default=0.05)
    fit.add_argument(
        "--allow-uncalibrated-model",
        "--allow-train-calibration-fallback",
        dest="allow_uncalibrated_model",
        action="store_true",
        help="Exploratory only: save a model whose under-calibrated predictions remain UNKNOWN",
    )
    _add_training_safety_options(fit)

    export = subparsers.add_parser(
        "export-classifier",
        help="Export included regions for the optional CNN/ONNX training path",
    )
    export.add_argument("--records", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    _add_training_safety_options(export)

    analyze = subparsers.add_parser("analyze", help="Write an isolated sidecar evidence directory")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--records", "--inputs", dest="records", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--minimum-regions", type=int, default=3)
    analyze.add_argument("--minimum-roles", type=int, default=2)
    analyze.add_argument("--minimum-known-coverage", type=float, default=0.60)
    analyze.add_argument("--pass-support-ratio", type=float, default=0.75)
    analyze.add_argument(
        "--allow-experimental-model",
        "--allow-nonoperational-model",
        dest="allow_experimental_model",
        action="store_true",
        help="Exploratory only: run a model that lacks default publication prerequisites",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "bootstrap-existing":
            result = _bootstrap_existing(args)
        elif args.command == "validate":
            result = _validate(args)
        elif args.command == "fit":
            result = _fit(args)
        elif args.command == "export-classifier":
            result = _export_classifier(args)
        else:
            result = _analyze(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"font-domain {args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    _print_json(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
