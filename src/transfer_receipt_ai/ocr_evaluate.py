"""Evaluate a receipt CTC ONNX candidate against held-out PaddleOCR labels.

This is a *teacher-parity* evaluator: its reference text comes from the
existing PaddleOCR result bundle.  It deliberately evaluates only a held-out
group split and reports OOV characters instead of silently filtering them, so
the result is useful for deciding whether a candidate can replace Paddle in a
later .NET integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .ocr import (
    clean_text,
    extract_field_value,
    normalize_amount,
    normalize_payment_method,
    normalize_status,
    normalize_time,
)
from .onnx_runtime import _preload_cuda_dlls, onnx_providers
from .ocr_train import (
    GENERIC_TEXT_LINE_FIELD,
    RECOGNIZER_FIELDS,
    _preprocess_contract,
    RecognizerConfig,
    _validate_recognizer_field_mode,
    decode_ctc_logits,
    load_records,
    preprocess_image,
)
from .labels import DETECTION_CLASSES


EVALUATION_SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    try:
        return _validate_recognizer_field_mode(fields)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "fields must select receipt detector fields or the independent "
            f"{GENERIC_TEXT_LINE_FIELD} recognizer; supported={','.join(RECOGNIZER_FIELDS)}"
        ) from None


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    invalid = sorted(set(splits) - {"train", "val", "test"})
    if not splits or invalid:
        raise argparse.ArgumentTypeError("training-splits must be a non-empty subset of: train,val,test")
    return splits


def _finite_probability(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ONNX OCR evaluation requires onnxruntime. Install the CUDA-matched onnxruntime-gpu package "
            "on a GPU server, or onnxruntime in a separate CPU environment."
        ) from error
    return onnxruntime


def _load_artifacts(model_path: Path) -> tuple[RecognizerConfig, list[str], Mapping[str, Any], str, str]:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    charset_path = model_path.with_suffix(".charset.json")
    contract_path = model_path.with_suffix(".contract.json")
    charset = _load_json(charset_path)
    contract = _load_json(contract_path)
    if contract.get("kind") != "receipt_ocr_ctc_v1":
        raise ValueError("OCR ONNX contract kind must be receipt_ocr_ctc_v1")
    if contract.get("onnx_sha256") != _sha256(model_path):
        raise ValueError("OCR ONNX SHA-256 does not match its contract")
    if contract.get("charset_sha256") != _sha256(charset_path):
        raise ValueError("OCR charset SHA-256 does not match its contract")
    characters = charset.get("characters")
    if charset.get("blank_index") != 0 or not isinstance(characters, list):
        raise ValueError("OCR charset must contain blank_index=0 and a characters list")
    if not all(isinstance(character, str) and len(character) == 1 for character in characters):
        raise ValueError("OCR charset entries must be single Unicode code points")
    if len(set(characters)) != len(characters):
        raise ValueError("OCR charset contains duplicate characters")
    raw_model = contract.get("model")
    if not isinstance(raw_model, Mapping):
        raise ValueError("OCR ONNX contract has no model configuration")
    try:
        config = RecognizerConfig(
            image_height=int(raw_model["image_height"]),
            image_width=int(raw_model["image_width"]),
            base_channels=int(raw_model.get("base_channels", 64)),
            hidden_size=int(raw_model["hidden_size"]),
            lstm_layers=int(raw_model["lstm_layers"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OCR ONNX contract has an invalid model configuration") from error
    config.validate()
    raw_input = contract.get("input")
    raw_output = contract.get("output")
    if not isinstance(raw_input, Mapping) or not isinstance(raw_output, Mapping):
        raise ValueError("OCR ONNX contract has no input/output schema")
    expected_input_shape = [1, 1, config.image_height, config.image_width]
    if raw_input.get("name") != "image" or raw_input.get("shape") != expected_input_shape:
        raise ValueError("OCR ONNX contract input must be fixed image [1,1,H,W]")
    contract_fields = contract.get("fields")
    if not isinstance(contract_fields, list) or not all(isinstance(field, str) for field in contract_fields):
        raise ValueError("OCR ONNX contract fields are invalid")
    selected_fields = _validate_recognizer_field_mode(contract_fields)
    expected_preprocess = _preprocess_contract(selected_fields)
    if raw_input.get("preprocess") != expected_preprocess:
        raise ValueError("OCR ONNX contract preprocess does not match its declared recognizer fields")
    if raw_output.get("name") != "logits" or raw_output.get("layout") != "[time,batch,class]":
        raise ValueError("OCR ONNX contract output must be logits [time,batch,class]")
    return config, characters, contract, str(raw_input["name"]), str(raw_output["name"])


def _create_session(onnxruntime: Any, model_path: Path, *, device: str) -> tuple[Any, list[str]]:
    providers = onnx_providers(device, onnxruntime)
    # Match the production ONNX path: this discovers CUDA/cuDNN DLLs from the
    # installed PyTorch/NVIDIA packages before Windows loads the ORT CUDA DLL.
    _preload_cuda_dlls(onnxruntime, providers)
    session = onnxruntime.InferenceSession(str(model_path), providers=providers)
    active = list(session.get_providers())
    requested_cuda = device.lower() == "cuda" or device.lower().startswith("cuda:")
    if requested_cuda and "CUDAExecutionProvider" not in active:
        raise RuntimeError("OCR ONNX session did not activate CUDAExecutionProvider")
    return session, active


def levenshtein_distance(reference: str, candidate: str) -> int:
    """Unicode-character edit distance used for CER reporting."""
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


def semantic_value(field: str, text: str) -> str | None:
    """Apply the same production field extraction rules to a candidate OCR string."""
    text = clean_text(text)
    if field == GENERIC_TEXT_LINE_FIELD:
        # A generic line has no receipt-field schema.  Raw exact/CER are the
        # only meaningful teacher-parity metrics at this boundary.
        return None
    if field == "amount":
        amount = normalize_amount(text)
        return str(amount["normalized"]) if amount is not None else None
    if field == "time":
        return normalize_time(text)
    if field == "transfer_status":
        status = normalize_status(text)
        return None if status == "unknown" else status
    if field == "recipient_field":
        return extract_field_value(text, "recipient") or None
    if field == "payment_method_field":
        value = extract_field_value(text, "payment_method")
        return normalize_payment_method(value)["normalized"] if value else None
    raise ValueError(f"Unsupported OCR field: {field}")


def _assert_no_training_overlap(
    evaluation_records: Sequence[Mapping[str, object]], training_records: Sequence[Mapping[str, object]]
) -> None:
    for name in ("group_id", "crop_sha256", "source"):
        evaluation_values = {str(record[name]) for record in evaluation_records if record.get(name)}
        training_values = {str(record[name]) for record in training_records if record.get(name)}
        overlap = sorted(evaluation_values & training_values)
        if overlap:
            raise ValueError(
                f"Evaluation {name} overlaps the selected training split; first overlap={overlap[0]!r}. "
                "Rebuild the dataset with leakage-safe group splits."
            )


def _metrics(comparisons: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = len(comparisons)
    if not count:
        raise ValueError("No evaluation records")
    raw_exact = sum(bool(record["raw_exact"]) for record in comparisons)
    semantic_records = [record for record in comparisons if record.get("semantic_applicable") is True]
    semantic_exact = sum(bool(record["semantic_exact"]) for record in semantic_records)
    semantic_valid = sum(record["candidate_semantic"] is not None for record in semantic_records)
    empty = sum(not str(record["candidate_text"]) for record in comparisons)
    oov = sum(bool(record["reference_has_oov_character"]) for record in comparisons)
    seen_text = sum(bool(record["reference_text_seen_in_model_train"]) for record in comparisons)
    edits = sum(int(record["cer_edits"]) for record in comparisons)
    reference_characters = sum(int(record["reference_characters"]) for record in comparisons)
    per_record_cer = [
        int(record["cer_edits"]) / max(1, int(record["reference_characters"])) for record in comparisons
    ]
    latencies = sorted(float(record["latency_ms"]) for record in comparisons)
    percentile = lambda fraction: latencies[min(len(latencies) - 1, int(math.ceil(fraction * len(latencies))) - 1)]
    return {
        "records": count,
        "raw_exact_matches": raw_exact,
        "raw_exact_match": raw_exact / count,
        "semantic_exact_matches": semantic_exact,
        "semantic_applicable_records": len(semantic_records),
        "semantic_exact_match": semantic_exact / len(semantic_records) if semantic_records else None,
        "candidate_semantic_valid_records": semantic_valid,
        "candidate_semantic_valid_rate": semantic_valid / len(semantic_records) if semantic_records else None,
        "empty_records": empty,
        "empty_rate": empty / count,
        "oov_reference_records": oov,
        "oov_reference_rate": oov / count,
        "reference_text_seen_in_model_train_records": seen_text,
        "reference_text_seen_in_model_train_rate": seen_text / count,
        "cer_edits": edits,
        "reference_characters": reference_characters,
        "micro_cer": edits / max(1, reference_characters),
        "macro_cer": sum(per_record_cer) / count,
        "latency_ms": {"p50": percentile(0.50), "p95": percentile(0.95), "mean": sum(latencies) / count},
    }


def _acceptance_failures(
    metrics_by_field: Mapping[str, Mapping[str, object]],
    *,
    min_raw_exact_match: float | None,
    min_semantic_exact_match: float | None,
    max_micro_cer: float | None,
    max_oov_reference_rate: float | None,
) -> list[str]:
    failures: list[str] = []
    for field, metrics in metrics_by_field.items():
        raw_exact = float(metrics["raw_exact_match"])
        semantic_exact_value = metrics["semantic_exact_match"]
        micro_cer = float(metrics["micro_cer"])
        oov_rate = float(metrics["oov_reference_rate"])
        if min_raw_exact_match is not None and raw_exact < min_raw_exact_match:
            failures.append(f"{field}: raw_exact_match={raw_exact:.4f} < {min_raw_exact_match:.4f}")
        if min_semantic_exact_match is not None:
            if semantic_exact_value is None:
                failures.append(f"{field}: semantic_exact_match is not applicable")
            elif float(semantic_exact_value) < min_semantic_exact_match:
                failures.append(
                    f"{field}: semantic_exact_match={float(semantic_exact_value):.4f} "
                    f"< {min_semantic_exact_match:.4f}"
                )
        if max_micro_cer is not None and micro_cer > max_micro_cer:
            failures.append(f"{field}: micro_cer={micro_cer:.4f} > {max_micro_cer:.4f}")
        if max_oov_reference_rate is not None and oov_rate > max_oov_reference_rate:
            failures.append(
                f"{field}: oov_reference_rate={oov_rate:.4f} > {max_oov_reference_rate:.4f}"
            )
    return failures


def evaluate_onnx(
    *,
    model_path: Path,
    records_path: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
    split: str = "test",
    fields: Sequence[str] = DETECTION_CLASSES,
    training_splits: Sequence[str] = ("train", "val"),
    device: str = "auto",
    min_raw_exact_match: float | None = None,
    min_semantic_exact_match: float | None = None,
    max_micro_cer: float | None = None,
    max_oov_reference_rate: float | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Run a fixed-shape OCR ONNX model against held-out Paddle pseudo labels."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test; train evaluation would not be an independent parity check")
    fields = _validate_recognizer_field_mode(fields)
    training_splits = tuple(dict.fromkeys(training_splits))
    if split in training_splits:
        raise ValueError("evaluation split must not also be a training split")
    _finite_probability(min_raw_exact_match, name="min_raw_exact_match")
    _finite_probability(min_semantic_exact_match, name="min_semantic_exact_match")
    _finite_probability(max_micro_cer, name="max_micro_cer")
    _finite_probability(max_oov_reference_rate, name="max_oov_reference_rate")
    if fields == (GENERIC_TEXT_LINE_FIELD,) and min_semantic_exact_match is not None:
        raise ValueError("semantic exact-match is not applicable to generic_text_line teacher parity")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"evaluation output already contains files: {output_dir}. Choose a new empty directory.")

    config, characters, contract, input_name, output_name = _load_artifacts(model_path)
    contract_fields = contract.get("fields")
    if not isinstance(contract_fields, list) or not all(isinstance(field, str) for field in contract_fields):
        raise ValueError("OCR ONNX contract fields are invalid")
    unsupported = sorted(set(fields) - set(contract_fields))
    if unsupported:
        raise ValueError(f"OCR ONNX contract does not declare requested fields: {','.join(unsupported)}")
    records = load_records(records_path, fields=fields, dataset_root=dataset_root)
    evaluation_records = [record for record in records if record["split"] == split]
    development_records = [record for record in records if record["split"] in training_splits]
    model_train_records = [record for record in records if record["split"] == "train"]
    if not evaluation_records:
        raise ValueError(f"No {split} records found; rebuild pseudo labels with a non-zero --test-ratio or --validation-ratio")
    if not development_records:
        raise ValueError("No selected training records found for leakage checks")
    if not model_train_records:
        raise ValueError("No train records found for OCR model-text coverage checks")
    for field in fields:
        if not any(record["field"] == field for record in evaluation_records):
            raise ValueError(
                f"No {split} evaluation records remain for requested field {field!r}; "
                "rebuild pseudo labels with a larger held-out split before accepting this OCR candidate"
            )
        if not any(record["field"] == field for record in model_train_records):
            raise ValueError(f"No train records remain for requested field {field!r}")
    _assert_no_training_overlap(evaluation_records, development_records)

    onnxruntime = _require_onnxruntime()
    model_path = model_path.resolve()
    session, active_providers = _create_session(onnxruntime, model_path, device=device)
    training_texts = {
        field: {str(record["text"]) for record in model_train_records if record["field"] == field}
        for field in fields
    }
    character_set = set(characters)
    comparisons: list[dict[str, object]] = []
    for record in evaluation_records:
        started = perf_counter()
        logits = session.run(
            [output_name],
            {
                input_name: preprocess_image(
                    Path(record["image_path"]),
                    config=config,
                    field=str(record["field"]),
                )
            },
        )[0]
        latency_ms = (perf_counter() - started) * 1000.0
        candidate_text = clean_text(decode_ctc_logits(np.asarray(logits), characters=characters)[0])
        reference_text = str(record["text"])
        semantic_applicable = str(record["field"]) != GENERIC_TEXT_LINE_FIELD
        reference_semantic = semantic_value(str(record["field"]), reference_text)
        candidate_semantic = semantic_value(str(record["field"]), candidate_text)
        edits = levenshtein_distance(reference_text, candidate_text)
        comparisons.append(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "id": str(record["id"]),
                "field": str(record["field"]),
                "split": split,
                "group_id": str(record["group_id"]),
                "image": Path(record["image_path"]).as_posix(),
                "source": record.get("source"),
                "result_json": record.get("result_json"),
                "crop_sha256": record.get("crop_sha256"),
                "paddle_text": str(record["paddle_text"]),
                "source_text": record.get("source_text"),
                "semantic_value": record.get("semantic_value"),
                "label_source": record.get("label_source"),
                "reference_text": reference_text,
                "candidate_text": candidate_text,
                "raw_exact": candidate_text == reference_text,
                "semantic_applicable": semantic_applicable,
                "reference_semantic": reference_semantic,
                "candidate_semantic": candidate_semantic,
                "semantic_exact": (
                    reference_semantic is not None and reference_semantic == candidate_semantic
                    if semantic_applicable
                    else None
                ),
                "candidate_semantic_valid": candidate_semantic is not None if semantic_applicable else None,
                "cer_edits": edits,
                "reference_characters": len(reference_text),
                "reference_has_oov_character": bool(set(reference_text) - character_set),
                "reference_text_seen_in_model_train": reference_text in training_texts[str(record["field"])],
                "latency_ms": round(latency_ms, 4),
            }
        )

    comparisons.sort(key=lambda value: (str(value["field"]), str(value["id"])))
    by_field = {field: _metrics([record for record in comparisons if record["field"] == field]) for field in fields}
    overall = _metrics(comparisons)
    failures = _acceptance_failures(
        by_field,
        min_raw_exact_match=min_raw_exact_match,
        min_semantic_exact_match=min_semantic_exact_match,
        max_micro_cer=max_micro_cer,
        max_oov_reference_rate=max_oov_reference_rate,
    )
    label_sources = sorted({str(record.get("label_source", "unspecified")) for record in evaluation_records})
    transaction_truth = label_sources == ["transaction_truth"]
    diagnostic_only = split != "test"
    if diagnostic_only:
        failures.append("validation split is diagnostic-only; formal acceptance requires the frozen test split")
    summary: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": (
            "generic_text_line_ctc_teacher_parity_evaluation_v1"
            if fields == (GENERIC_TEXT_LINE_FIELD,)
            else ("receipt_ocr_ctc_truth_evaluation_v1" if transaction_truth else "receipt_ocr_ctc_pseudo_label_evaluation_v1")
        ),
        "model": model_path.as_posix(),
        "model_sha256": _sha256(model_path),
        "records": records_path.resolve().as_posix(),
        "evaluation_split": split,
        "training_splits": list(training_splits),
        "fields": list(fields),
        "label_sources": label_sources,
        "providers": active_providers,
        "overall": overall,
        "by_field": by_field,
        "acceptance": {
            "min_raw_exact_match": min_raw_exact_match,
            "min_semantic_exact_match": min_semantic_exact_match,
            "max_micro_cer": max_micro_cer,
            "max_oov_reference_rate": max_oov_reference_rate,
            "diagnostic_only": diagnostic_only,
            "passed": not failures and not diagnostic_only,
            "failures": failures,
        },
        "warning": (
            "This evaluates against local transaction truth. Validate the receipt-key association and keep a separate "
            "audit set before treating it as production accuracy."
            if transaction_truth
            else "This compares ONNX output to held-out pseudo labels, not to independent business truth. "
            "Exact match and CER are teacher-parity metrics only; do not make production-accuracy claims."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    _atomic_write_jsonl(
        output_dir / "disagreements.jsonl",
        [
            record
            for record in comparisons
            if not bool(record["raw_exact"])
            or (record.get("semantic_applicable") is True and not bool(record["semantic_exact"]))
        ],
    )
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a receipt OCR ONNX candidate with held-out PaddleOCR labels")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True, help="Top-level pseudo_labels.jsonl")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root that owns image paths in --records; defaults to the records file directory",
    )
    parser.add_argument("--output", type=Path, required=True, help="New empty evaluation output directory")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--fields", type=_parse_fields, default=DETECTION_CLASSES)
    parser.add_argument("--training-splits", type=_parse_splits, default=("train", "val"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-raw-exact-match", type=float)
    parser.add_argument("--min-semantic-exact-match", type=float)
    parser.add_argument("--max-micro-cer", type=float)
    parser.add_argument("--max-oov-reference-rate", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary, failures = evaluate_onnx(
            model_path=args.model,
            records_path=args.records,
            output_dir=args.output,
            dataset_root=args.dataset_root,
            split=args.split,
            fields=args.fields,
            training_splits=args.training_splits,
            device=args.device,
            min_raw_exact_match=args.min_raw_exact_match,
            min_semantic_exact_match=args.min_semantic_exact_match,
            max_micro_cer=args.max_micro_cer,
            max_oov_reference_rate=args.max_oov_reference_rate,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OCR ONNX evaluation failed:\n{error}") from None
    overall = summary["overall"]
    print(
        f"Wrote {overall['records']} ONNX/Paddle comparison(s) to {args.output} "
        f"(raw_exact_match={overall['raw_exact_match']:.2%}, micro_cer={overall['micro_cer']:.4f})"
    )
    if failures:
        raise SystemExit("OCR ONNX candidate did not meet the requested acceptance gate:\n- " + "\n- ".join(failures))


if __name__ == "__main__":  # pragma: no cover
    main()
