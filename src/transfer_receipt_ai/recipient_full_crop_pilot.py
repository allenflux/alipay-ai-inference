"""Run the narrow v13 full-production-crop recipient pilot.

The command derives its complete model configuration from an immutable v13
checkpoint and changes only ``recipient_value_left_trim`` from 0.30 to 0.0.
It never opens the test split, never exports a production artifact, and stops
after eight epochs unless the fixed validation trend justifies a later run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .ocr_unified import (
    CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
    INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
    KIND_V13,
    UnifiedReaderConfig,
    _atomic_write_json,
    _checkpoint_config,
    _load_checkpoint,
    _require_torch,
    _validate_recipient_full_crop_seed_policy,
    _validate_recipient_full_crop_warmstart_config,
    train_unified_reader,
)
from .recipient_blind_manifest import KIND as BLIND_MANIFEST_KIND


KIND = "receipt_recipient_full_crop_pilot_v1"
PILOT_EPOCHS = 8
PILOT_MINIMUM_BEST_RECIPIENT = 0.75
PILOT_MINIMUM_EPOCH4_TO_8_GAIN = 0.02
AMOUNT_FLOOR = 0.7885
TIME_FLOOR = 0.9840
PAYMENT_FLOOR = 0.9325
STATUS_TEXT_FLOOR = 0.90
BLIND_CONTRACT_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_reparse_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & 0x400)  # Windows FILE_ATTRIBUTE_REPARSE_POINT


def _fresh_output_path(path: Path) -> Path:
    """Resolve a new output without following a symlink/junction boundary."""

    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.path.lexists(os.fspath(raw)):
        raise ValueError(
            f"Refusing to reuse existing, symlink, or reparse full-crop pilot output: {raw}"
        )
    ancestor = raw.parent
    while True:
        if os.path.lexists(os.fspath(ancestor)):
            if _is_reparse_path(ancestor):
                raise ValueError(
                    "full-crop pilot output must not traverse a symlink/junction/reparse ancestor"
                )
            if not ancestor.is_dir():
                raise ValueError("full-crop pilot output ancestor is not a directory")
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    resolved = raw.resolve()
    if os.path.lexists(os.fspath(resolved)):
        raise ValueError(f"Refusing to reuse full-crop pilot output: {resolved}")
    return resolved


def verify_blind_manifest_contract(
    *, records_path: Path, blind_contract_path: Path
) -> dict[str, object]:
    """Bind training to a physical train/val-only manifest before Torch runs."""

    records = Path(records_path).expanduser().resolve()
    contract_path = Path(blind_contract_path).expanduser().resolve()
    if not records.is_file():
        raise FileNotFoundError(records)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    try:
        raw_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("full-crop blind manifest contract is invalid JSON") from error
    contract = _mapping(raw_contract, "blind manifest contract")
    blind_manifest = contract.get("blind_manifest")
    if not isinstance(blind_manifest, str) or not blind_manifest:
        raise ValueError("full-crop blind manifest contract has no bound manifest")
    try:
        bound_manifest = Path(blind_manifest).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("full-crop blind manifest contract has no valid bound manifest") from error
    try:
        same_manifest = os.path.samefile(records, bound_manifest)
    except OSError as error:
        raise ValueError("full-crop blind manifest contract binding cannot be verified") from error
    expected_sha256 = contract.get("blind_manifest_sha256")
    source_manifest = contract.get("source_manifest")
    source_manifest_sha256 = contract.get("source_manifest_sha256")
    if not isinstance(source_manifest, str) or not source_manifest:
        raise ValueError("full-crop blind manifest contract has no source manifest")
    try:
        bound_source = Path(source_manifest).expanduser().resolve(strict=True)
        source_is_blind = os.path.samefile(bound_source, records)
    except (OSError, RuntimeError) as error:
        raise ValueError("full-crop blind manifest contract has no valid source manifest") from error
    if (
        contract.get("schema_version") != BLIND_CONTRACT_SCHEMA_VERSION
        or contract.get("kind") != BLIND_MANIFEST_KIND
        or not same_manifest
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not isinstance(source_manifest_sha256, str)
        or len(source_manifest_sha256) != 64
        or source_is_blind
        or contract.get("test_labels_used") is not False
        or contract.get("test_metrics_computed") is not False
        or contract.get("test_examples_emitted") is not False
        or contract.get("optimizer_supervision_splits") != ["train"]
        or contract.get("checkpoint_selection_splits") != ["val"]
        or contract.get("final_gate_only_splits") != ["test"]
    ):
        raise ValueError("full-crop blind manifest contract is incomplete or unsafe")
    observed_sha256 = _sha256(records)
    if observed_sha256 != expected_sha256.lower():
        raise ValueError("full-crop blind manifest changed after contract creation")

    split_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    with records.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"full-crop blind manifest line {line_number} is invalid JSON"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"full-crop blind manifest line {line_number} is not an object"
                )
            split = row.get("split")
            if split not in {"train", "val"}:
                raise ValueError(
                    "full-crop pilot records must physically exclude every test row"
                )
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise ValueError("full-crop blind manifest has a missing or duplicate record id")
            seen_ids.add(record_id)
            split_counts[str(split)] += 1
    contract_counts = _mapping(contract.get("split_counts"), "blind split counts")
    try:
        expected_train = int(contract_counts.get("train", -1))
        expected_val = int(contract_counts.get("val", -1))
        excluded_test = int(contract_counts.get("test_excluded", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("full-crop blind manifest contract has invalid split counts") from error
    if (
        split_counts != Counter({"train": expected_train, "val": expected_val})
        or expected_train <= 0
        or expected_val <= 0
        or excluded_test <= 0
    ):
        raise ValueError("full-crop blind manifest split counts do not match its contract")
    return {
        "schema_version": BLIND_CONTRACT_SCHEMA_VERSION,
        "kind": BLIND_MANIFEST_KIND,
        "contract_path": str(contract_path),
        "source_manifest": str(bound_source),
        "source_manifest_sha256": source_manifest_sha256.lower(),
        "blind_manifest": str(records),
        "blind_manifest_sha256": observed_sha256,
        "split_counts": {
            "train": expected_train,
            "val": expected_val,
            "test_excluded": excluded_test,
        },
        "optimizer_supervision_splits": ["train"],
        "checkpoint_selection_splits": ["val"],
        "test_opened_by_training": False,
    }


def target_config_from_seed(checkpoint: Path, *, torch: Any) -> UnifiedReaderConfig:
    """Return the seed configuration with only the destructive trim removed."""

    payload = _load_checkpoint(Path(checkpoint), torch=torch)
    _validate_recipient_full_crop_seed_policy(payload)
    source = _checkpoint_config(payload)
    target = replace(source, recipient_value_left_trim=0.0)
    _validate_recipient_full_crop_warmstart_config(source, target)
    return target


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"full-crop pilot summary has invalid {description}")
    return value


def _finite_rate(value: object, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"full-crop pilot summary has invalid {description}")
    try:
        rate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"full-crop pilot summary has invalid {description}") from error
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"full-crop pilot summary has invalid {description}")
    return rate


def _recipient_exact(record: Mapping[str, object], description: str) -> float:
    fields = _mapping(record.get("val_candidate_text_by_field"), f"{description} fields")
    recipient = _mapping(fields.get("recipient_field"), f"{description} recipient metric")
    return _finite_rate(recipient.get("exact_match"), f"{description} recipient exact")


def evaluate_pilot_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Validate the training contract and apply the fixed eight-epoch stop gate."""

    config = _mapping(summary.get("config"), "config")
    initialization = _mapping(summary.get("initialization"), "initialization")
    source_config = _mapping(initialization.get("source_config"), "source config")
    seed_split_policy = _mapping(
        initialization.get("source_recipient_train_split_policy"),
        "seed recipient split policy",
    )
    row_mapping = _mapping(
        initialization.get("recipient_classifier_row_mapping"),
        "recipient classifier row mapping",
    )
    financial_policy = _mapping(
        initialization.get("financial_label_policy"), "financial label policy"
    )
    fine_tune = _mapping(summary.get("fine_tune_policy"), "fine-tune policy")
    runtime = _mapping(summary.get("training_runtime"), "training runtime")
    split_policy = _mapping(summary.get("recipient_train_split_policy"), "recipient split policy")
    checkpoint_policy = _mapping(summary.get("checkpoint_selection_policy"), "checkpoint policy")
    protected_minima = _mapping(
        checkpoint_policy.get("protected_minimum_candidate_exact"),
        "checkpoint protected floors",
    )
    field_counts = _mapping(summary.get("field_counts"), "field counts")
    recipient_counts = _mapping(field_counts.get("recipient_field"), "recipient field counts")
    try:
        source_reader_config = UnifiedReaderConfig(**dict(source_config))
        target_reader_config = UnifiedReaderConfig(**dict(config))
        source_reader_config.validate()
        target_reader_config.validate()
        _validate_recipient_full_crop_warmstart_config(
            source_reader_config,
            target_reader_config,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training summary does not prove the v13 full-crop config transition"
        ) from error
    if (
        summary.get("kind") != KIND_V13
        or int(config.get("architecture_version", -1)) != 13
        or not math.isclose(
            _finite_rate(config.get("recipient_value_left_trim"), "target left trim"),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(source_config.get("architecture_version", -1)) != 13
        or not math.isclose(
            _finite_rate(source_config.get("recipient_value_left_trim"), "source left trim"),
            0.30,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or initialization.get("mode") != "parameter_only_recipient_full_crop_warmstart"
        or initialization.get("init_checkpoint_mode")
        != INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART
        or initialization.get("source_kind") != KIND_V13
        or seed_split_policy.get("mode") != "standard_train_only"
        or seed_split_policy.get("splits") != ["train"]
        or row_mapping.get("blank_row_copied") is not True
        or int(row_mapping.get("shared_character_rows_copied", -1))
        != int(row_mapping.get("checkpoint_character_count", -2))
        or int(row_mapping.get("checkpoint_character_count", -1)) <= 0
        or int(row_mapping.get("target_character_count", -1))
        != int(row_mapping.get("checkpoint_character_count", -2))
        + int(row_mapping.get("new_target_character_rows_kept_at_seed", -3))
        or financial_policy.get("mode")
        != "checkpoint_financial_label_maps_recipient_full_crop_warmstart_v1"
        or not math.isclose(
            _finite_rate(
                financial_policy.get("source_recipient_value_left_trim"),
                "financial policy source left trim",
            ),
            0.30,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(
                financial_policy.get("target_recipient_value_left_trim"),
                "financial policy target left trim",
            ),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or fine_tune.get("mode") != "recipient_only_v13"
        or fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("frozen_non_recipient_byte_guard")
        != "before_every_full_validation"
        or int(fine_tune.get("frozen_non_recipient_state_entry_count", 0)) <= 0
        or runtime.get("device") != "cuda:0"
        or runtime.get("uses_cuda") is not True
        or "4090" not in str(runtime.get("cuda_device_name", ""))
        or split_policy.get("mode") != "standard_train_only"
        or list(split_policy.get("splits", [])) != ["train"]
        or checkpoint_policy.get("mode") != CHECKPOINT_SELECTION_RECIPIENT_PRIORITY
        or not math.isclose(
            _finite_rate(protected_minima.get("amount"), "amount protection floor"),
            AMOUNT_FLOOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(protected_minima.get("time"), "time protection floor"),
            TIME_FLOOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(
                protected_minima.get("payment_method_field"),
                "payment protection floor",
            ),
            PAYMENT_FLOOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("training summary does not prove the v13 full-crop blind pilot contract")
    for field, raw_counts in field_counts.items():
        counts = _mapping(raw_counts, f"{field} field counts")
        if int(counts.get("test", -1)) != 0:
            raise ValueError("training summary contains a physically included test row")

    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("full-crop pilot summary has invalid epoch records")
    records = [
        _mapping(record, "epoch record")
        for record in raw_records
    ]
    observed_epochs = [record.get("epoch") for record in records]
    if any(isinstance(epoch, bool) or not isinstance(epoch, int) for epoch in observed_epochs):
        raise ValueError("full-crop pilot epoch identifiers must be integers")
    if observed_epochs != list(range(0, PILOT_EPOCHS + 1)):
        raise ValueError("full-crop pilot requires ordered epoch-zero plus epochs 1 through 8")
    if any(record.get("validation_performed") is not True for record in records):
        raise ValueError("full-crop pilot requires validation at epoch zero and every training epoch")
    protected = {
        "amount": AMOUNT_FLOOR,
        "time": TIME_FLOOR,
        "payment_method_field": PAYMENT_FLOOR,
    }
    for record in records:
        epoch = record.get("epoch")
        fields = _mapping(record.get("val_candidate_text_by_field"), f"epoch {epoch} fields")
        for field, floor in protected.items():
            metric = _mapping(fields.get(field), f"epoch {epoch} {field} metric")
            if _finite_rate(metric.get("exact_match"), f"epoch {epoch} {field} exact") < floor:
                raise ValueError(f"full-crop pilot epoch {epoch} violated the {field} floor")
        if (
            record.get("checkpoint_selection_eligible") is not True
            or record.get("checkpoint_selection_protection_failures") != []
        ):
            raise ValueError(f"full-crop pilot epoch {epoch} did not pass checkpoint protection")
        status_errors = record.get("val_status_non_success_to_success")
        if isinstance(status_errors, bool) or not isinstance(status_errors, int) or status_errors != 0:
            raise ValueError(f"full-crop pilot epoch {epoch} violated visible-status safety")
        raw_ctc = _mapping(record.get("val_ctc_by_field"), f"epoch {epoch} raw CTC fields")
        status_metric = _mapping(
            raw_ctc.get("transfer_status"), f"epoch {epoch} visible-status metric"
        )
        if (
            _finite_rate(
                status_metric.get("exact_match"), f"epoch {epoch} visible-status exact"
            )
            < STATUS_TEXT_FLOOR
        ):
            raise ValueError(f"full-crop pilot epoch {epoch} violated visible-status floor")

    def exactly_one_epoch(epoch: int) -> Mapping[str, object]:
        matches = [record for record in records if record.get("epoch") == epoch]
        if len(matches) != 1:
            raise ValueError(f"full-crop pilot requires exactly one epoch {epoch} validation record")
        if matches[0].get("validation_performed") is not True:
            raise ValueError(f"full-crop pilot epoch {epoch} was not validated")
        return matches[0]

    epoch4 = exactly_one_epoch(4)
    epoch8 = exactly_one_epoch(8)
    best_epoch = summary.get("best_checkpoint_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 0 <= best_epoch <= PILOT_EPOCHS:
        raise ValueError("full-crop pilot summary has invalid best checkpoint epoch")
    best = exactly_one_epoch(best_epoch)
    best_recipient = _recipient_exact(best, "best checkpoint")
    maximum_recipient = max(
        _recipient_exact(record, f"epoch {record['epoch']}") for record in records
    )
    if not math.isclose(best_recipient, maximum_recipient, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("full-crop pilot best checkpoint is not recipient-optimal")
    epoch4_recipient = _recipient_exact(epoch4, "epoch 4")
    epoch8_recipient = _recipient_exact(epoch8, "epoch 8")
    gain = epoch8_recipient - epoch4_recipient
    failures: list[str] = []
    if best_recipient < PILOT_MINIMUM_BEST_RECIPIENT:
        failures.append("best_recipient_below_75_percent")
    if gain < PILOT_MINIMUM_EPOCH4_TO_8_GAIN:
        failures.append("epoch4_to_8_gain_below_2pp")
    return {
        "schema_version": 1,
        "kind": KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "epochs": PILOT_EPOCHS,
        "source_config": dict(source_config),
        "target_config": dict(config),
        "fixed_gates": {
            "minimum_best_recipient_exact": PILOT_MINIMUM_BEST_RECIPIENT,
            "minimum_epoch4_to_8_gain": PILOT_MINIMUM_EPOCH4_TO_8_GAIN,
            "amount_candidate_exact_floor": AMOUNT_FLOOR,
            "time_candidate_exact_floor": TIME_FLOOR,
            "payment_candidate_exact_floor": PAYMENT_FLOOR,
            "visible_status_raw_exact_floor": STATUS_TEXT_FLOOR,
            "status_non_success_to_success_max": 0,
        },
        "observed": {
            "best_epoch": best_epoch,
            "best_recipient_exact": best_recipient,
            "epoch4_recipient_exact": epoch4_recipient,
            "epoch8_recipient_exact": epoch8_recipient,
            "epoch4_to_8_gain": gain,
        },
        "passed": not failures,
        "failures": failures,
        "decision": (
            "analysis_only_continue_to_separate_guarded_candidate"
            if not failures
            else "analysis_only_stop_do_not_extend_epochs"
        ),
    }


def run_full_crop_pilot(
    *,
    records_path: Path,
    blind_contract_path: Path,
    dataset_root: Path,
    seed_checkpoint: Path,
    output_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 10,
    learning_rate: float = 0.0001,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    train_progress_every: int = 250,
) -> dict[str, object]:
    """Train exactly eight epochs and persist the stop decision."""

    if device != "cuda:0":
        raise ValueError("full-crop pilot is hard-locked to cuda:0")
    blind_binding = verify_blind_manifest_contract(
        records_path=records_path,
        blind_contract_path=blind_contract_path,
    )
    output = _fresh_output_path(output_dir)
    torch, _ = _require_torch()
    config = target_config_from_seed(seed_checkpoint, torch=torch)
    if not output.parent.is_dir():
        raise ValueError("full-crop pilot output parent must already exist")
    try:
        output.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"Refusing to reuse full-crop pilot output: {output}") from error
    if _is_reparse_path(output):
        raise ValueError("fresh full-crop pilot output unexpectedly became a reparse point")
    train_unified_reader(
        records_path=Path(records_path),
        dataset_root=Path(dataset_root),
        output_dir=output,
        config=config,
        device=device,
        epochs=PILOT_EPOCHS,
        batch_size=batch_size,
        learning_rate=learning_rate,
        recipient_low_confidence_threshold=0.95,
        recipient_low_confidence_loss_weight=0.50,
        recipient_confidence_curriculum_epochs=10,
        recipient_tail_rare_character_max_support=3,
        recipient_tail_rare_character_loss_weight=1.5,
        recipient_tail_long_text_min_length=9,
        recipient_tail_long_text_loss_weight=1.5,
        recipient_train_augmentation="robust_v2",
        recipient_train_splits=("train",),
        recipient_only_fine_tune=True,
        validation_every=1,
        checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        checkpoint_min_amount_candidate_exact=AMOUNT_FLOOR,
        checkpoint_min_time_candidate_exact=TIME_FLOOR,
        checkpoint_min_payment_candidate_exact=PAYMENT_FLOOR,
        init_checkpoint=Path(seed_checkpoint),
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        ctc_loss_weight=1.0,
        structured_loss_weight=1.0,
        payment_bank_prefix_min_support=3,
        seed=42,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        train_progress_every=train_progress_every,
        cuda_tf32=True,
        cudnn_benchmark=True,
    )
    final_blind_binding = verify_blind_manifest_contract(
        records_path=records_path,
        blind_contract_path=blind_contract_path,
    )
    if final_blind_binding != blind_binding:
        raise ValueError("full-crop blind manifest binding changed during training")
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("full-crop training summary must be a JSON object")
    decision = evaluate_pilot_summary(summary)
    decision = {**decision, "blind_manifest_contract": blind_binding}
    _atomic_write_json(output / "pilot_decision.json", decision)
    if not bool(decision["passed"]):
        raise ValueError(
            "PILOT STOP: " + "; ".join(str(value) for value in decision["failures"])
        )
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed v13 recipient full-crop CUDA pilot")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--blind-contract", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--train-progress-every", type=int, default=250)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    decision = run_full_crop_pilot(
        records_path=args.records,
        blind_contract_path=args.blind_contract,
        dataset_root=args.dataset_root,
        seed_checkpoint=args.seed_checkpoint,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        train_progress_every=args.train_progress_every,
    )
    observed = _mapping(decision.get("observed"), "decision observations")
    print(
        "PILOT PASS: full-crop val trend justifies a separate guarded candidate; "
        f"best={float(observed['best_recipient_exact']):.2%}, "
        f"epoch4={float(observed['epoch4_recipient_exact']):.2%}, "
        f"epoch8={float(observed['epoch8_recipient_exact']):.2%}, "
        f"gain={float(observed['epoch4_to_8_gain']):+.2%}"
    )


if __name__ == "__main__":
    main()
