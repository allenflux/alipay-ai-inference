"""Fail-closed recovery finalizer for the v12 random recipient bootstrap.

The original bootstrap input contract hash-binds its verifier, so a completed
run cannot safely be finalized by editing that verifier in place.  This module
keeps every original binding and structural check, then independently rebuilds
the candidate-metric denominators from the immutable blind manifest using the
v12 evaluator's target-eligibility rules.  It publishes analysis evidence
only; it cannot authorize ONNX export or a production route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import recipient_random_bootstrap as bootstrap
from . import ocr_unified_targets as target_helpers


RECOVERY_KIND = "receipt_recipient_random_bootstrap_recovery_decision_v1"
CANDIDATE_FIELDS = ("amount", "time", "payment_method_field", "recipient_field")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_write_json_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    """Publish via an atomic hard-link create, never a replacing rename."""

    bootstrap._require_no_reparse(path, include_leaf=False)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # link() is an atomic create-if-absent operation on the same volume.
        # Unlike replace(), it also refuses an existing broken symlink.
        os.link(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _hash_handle(handle: Any) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


class _FrozenEvidence:
    """Keep important evidence inodes open and prove paths still name them."""

    def __init__(self, paths: Mapping[str, Path]) -> None:
        self._entries: dict[str, tuple[Path, Any, os.stat_result, str]] = {}
        try:
            for name, raw_path in paths.items():
                path = bootstrap._require_file(raw_path, name)
                handle = path.open("rb")
                status = os.fstat(handle.fileno())
                self._entries[name] = (path, handle, status, _hash_handle(handle))
        except BaseException:
            self.close()
            raise

    def manifest(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "path": str(path),
                "sha256": sha256,
                "size_bytes": int(status.st_size),
            }
            for name, (path, _handle, status, sha256) in self._entries.items()
        }

    def json_object(self, name: str, *, training: bool = False) -> dict[str, Any]:
        """Decode JSON from the already-open evidence inode.

        Reading summaries or the input contract through their path before the
        inode is frozen would leave a replacement window in which the decision
        could describe one byte stream while publishing the hash of another.
        """

        try:
            _path, handle, _status, _sha256_value = self._entries[name]
        except KeyError:
            raise KeyError(f"unknown frozen evidence entry: {name}") from None
        handle.seek(0)
        raw = handle.read()
        handle.seek(0)

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON number {value!r} is forbidden")

        try:
            value = json.loads(
                raw.decode("utf-8"),
                parse_constant=(lambda _value: None) if training else reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to decode frozen JSON evidence {name}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"frozen JSON evidence must be an object: {name}")
        return value

    def verify(self) -> None:
        for name, (path, handle, initial, sha256) in self._entries.items():
            bootstrap._require_no_reparse(path)
            descriptor_status = os.fstat(handle.fileno())
            path_status = path.stat()
            if not os.path.samestat(descriptor_status, path_status):
                raise RuntimeError(f"frozen evidence path was replaced: {name}")
            if (
                descriptor_status.st_size != initial.st_size
                or descriptor_status.st_mtime_ns != initial.st_mtime_ns
                or _hash_handle(handle) != sha256
                or bootstrap._sha256(path) != sha256
            ):
                raise RuntimeError(f"frozen evidence changed during recovery: {name}")

    def close(self) -> None:
        entries = getattr(self, "_entries", {})
        for _path, handle, _status, _sha256_value in entries.values():
            try:
                handle.close()
            except Exception:
                pass
        entries.clear()

    def __del__(self) -> None:
        self.close()


def _slot(row: Mapping[str, object], field: str) -> Mapping[str, object] | None:
    slots = row.get("slots")
    if not isinstance(slots, Mapping):
        raise ValueError("blind manifest row has no slots mapping")
    value = slots.get(field)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"blind manifest {field} slot is not an object")
    return value


def _slot_text(slot: Mapping[str, object] | None, key: str = "text") -> str | None:
    if slot is None:
        return None
    value = slot.get(key)
    return value if isinstance(value, str) else None


def _candidate_reference_present(row: Mapping[str, object], field: str) -> bool:
    slot = _slot(row, field)
    if field == "amount":
        # v8-v12 candidate validation intentionally excludes rows whose
        # visible amount does not satisfy the strict display grammar.  The
        # raw slot remains part of field_counts, hence the two denominators
        # are allowed (and expected) to differ.
        visible = slot.get("visible_text") if slot is not None else None
        return (
            isinstance(visible, str)
            and target_helpers.parse_amount_display_target(visible) is not None
        )
    if field == "time":
        visible = _slot_text(slot, "visible_text")
        return bool(visible) or _slot_text(slot) is not None
    if field == "payment_method_field":
        return _slot_text(slot) is not None
    if field == "recipient_field":
        # This recovery is fixed to v12.  Unlike v10, v12 candidate
        # validation compares the anchored value in slot.text directly and
        # does not fall back to semantic_value/recipient_value extraction.
        return _slot_text(slot) is not None
    raise ValueError(f"unsupported candidate field: {field}")


def rebuild_candidate_denominators(
    *, contract: Mapping[str, object], summary_config: object
) -> dict[str, object]:
    """Rebuild val denominators from hash-bound labels, without model output."""

    bootstrap._assert_mapping_subset(summary_config, bootstrap.FIXED_TOPOLOGY, "training topology")
    if not isinstance(summary_config, Mapping) or summary_config.get("architecture_version") != 12:
        raise ValueError("candidate denominator recovery supports only the bound v12 topology")
    dataset_binding = contract.get("dataset_binding")
    if not isinstance(dataset_binding, Mapping):
        raise ValueError("input contract has no bound dataset evidence")
    field_counts = dataset_binding.get("field_counts")
    if not isinstance(field_counts, Mapping):
        raise ValueError("input contract has no bound per-field split counts")
    blind_manifest = bootstrap._require_read_only_file(
        Path(str(contract.get("blind_manifest", ""))), "blind manifest"
    )
    expected_manifest_sha = bootstrap._require_sha(
        contract.get("blind_manifest_sha256"), "blind_manifest_sha256"
    )
    if bootstrap._sha256(blind_manifest) != expected_manifest_sha:
        raise ValueError("bound blind manifest changed before denominator recovery")

    denominators = {field: 0 for field in CANDIDATE_FIELDS}
    observed_raw_counts = {field: 0 for field in CANDIDATE_FIELDS}
    eligible_ids: dict[str, list[str]] = {field: [] for field in CANDIDATE_FIELDS}
    raw_ids: dict[str, list[str]] = {field: [] for field in CANDIDATE_FIELDS}
    val_rows = 0
    with blind_manifest.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{blind_manifest}:{line_number}: invalid JSON") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{blind_manifest}:{line_number}: row must be an object")
            split = row.get("split")
            if split not in {"train", "val"}:
                raise ValueError("recovery manifest physically contains test/unknown rows")
            if split != "val":
                continue
            val_rows += 1
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{blind_manifest}:{line_number}: row has no record id")
            for field in CANDIDATE_FIELDS:
                if _slot(row, field) is not None:
                    observed_raw_counts[field] += 1
                    raw_ids[field].append(record_id)
                if _candidate_reference_present(row, field):
                    denominators[field] += 1
                    eligible_ids[field].append(record_id)
    if bootstrap._sha256(blind_manifest) != expected_manifest_sha:
        raise RuntimeError("bound blind manifest changed during denominator recovery")

    raw_val_counts: dict[str, int] = {}
    for field in CANDIDATE_FIELDS:
        raw = field_counts.get(field)
        if (
            not isinstance(raw, Mapping)
            or isinstance(raw.get("val"), bool)
            or not isinstance(raw.get("val"), int)
            or int(raw["val"]) <= 0
        ):
            raise ValueError(f"input contract has invalid raw val count for {field}")
        raw_val_counts[field] = int(raw["val"])
        if observed_raw_counts[field] != raw_val_counts[field]:
            raise ValueError(f"rebuilt raw val count differs from input contract for {field}")
        if not 0 < denominators[field] <= raw_val_counts[field]:
            raise ValueError(f"rebuilt candidate denominator is invalid for {field}")

    split_counts = dataset_binding.get("split_counts")
    if (
        not isinstance(split_counts, Mapping)
        or isinstance(split_counts.get("val"), bool)
        or not isinstance(split_counts.get("val"), int)
        or val_rows != split_counts["val"]
    ):
        raise ValueError("rebuilt val row count differs from input contract")

    return {
        "policy": "v12_candidate_reference_eligibility_v1",
        "blind_manifest": str(blind_manifest),
        "blind_manifest_sha256": expected_manifest_sha,
        "summary_config_sha256": _canonical_sha256(summary_config),
        "val_rows": val_rows,
        "raw_val_field_counts": raw_val_counts,
        "candidate_val_denominators": denominators,
        "raw_val_record_ids_sha256": {
            field: _canonical_sha256(sorted(raw_ids[field])) for field in CANDIDATE_FIELDS
        },
        "eligible_record_ids_sha256": {
            field: _canonical_sha256(sorted(eligible_ids[field])) for field in CANDIDATE_FIELDS
        },
        "excluded_from_candidate_metric": {
            field: raw_val_counts[field] - denominators[field] for field in CANDIDATE_FIELDS
        },
    }


def _candidate_metric(
    record: Mapping[str, object], field: str, *, expected_records: int
) -> float:
    by_field = record.get("val_candidate_text_by_field")
    if not isinstance(by_field, Mapping) or not isinstance(by_field.get(field), Mapping):
        raise ValueError(f"training record has no candidate metric for {field}")
    metric = by_field[field]
    assert isinstance(metric, Mapping)
    rate = bootstrap._finite_rate(metric.get("exact_match"), f"{field} exact_match")
    matches = metric.get("exact_matches")
    records = metric.get("records")
    if (
        isinstance(matches, bool)
        or not isinstance(matches, int)
        or isinstance(records, bool)
        or not isinstance(records, int)
        or records != expected_records
        or not 0 <= matches <= records
        or not math.isclose(rate, matches / records, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"training record has inconsistent candidate metric for {field}")
    return rate


def _validate_common_summary(
    summary: Mapping[str, object],
    *,
    epochs: int,
    fine_tune_mode: str,
    expected_field_counts: Mapping[str, Mapping[str, int]],
) -> list[Mapping[str, object]]:
    if summary.get("schema_version") != bootstrap.SCHEMA_VERSION or summary.get("kind") != bootstrap.CHECKPOINT_KIND:
        raise ValueError("training summary is not a v12 unified-reader artifact")
    bootstrap._assert_mapping_subset(summary.get("config"), bootstrap.FIXED_TOPOLOGY, "training topology")
    policy = summary.get("recipient_train_split_policy")
    if not isinstance(policy, Mapping) or policy.get("mode") != "standard_train_only" or policy.get("splits") != ["train"]:
        raise ValueError("recipient supervision must be train split only")
    checkpoint_policy = summary.get("checkpoint_selection_policy")
    if (
        not isinstance(checkpoint_policy, Mapping)
        or checkpoint_policy.get("mode") != "balanced"
        or checkpoint_policy.get("protected_minimum_candidate_exact") != {}
    ):
        raise ValueError("branch-source training must use the unrelaxed balanced analysis selector")
    fine_tune = summary.get("fine_tune_policy")
    if not isinstance(fine_tune, Mapping) or fine_tune.get("mode") != fine_tune_mode:
        raise ValueError(f"unexpected fine-tune policy; expected {fine_tune_mode}")
    runtime = summary.get("training_runtime")
    if (
        not isinstance(runtime, Mapping)
        or "4090" not in str(runtime.get("cuda_device_name", ""))
        or runtime.get("cuda_tf32_requested") is not True
        or runtime.get("cudnn_benchmark_requested") is not True
        or runtime.get("validation_every") != 1
        or runtime.get("recipient_only_private_branch_training") is not (fine_tune_mode == "recipient_only_v12")
    ):
        raise ValueError("training summary does not prove the fixed CUDA:0 4090 recipe")
    fields = summary.get("field_counts")
    if not isinstance(fields, Mapping) or set(fields) != set(expected_field_counts):
        raise ValueError("training summary has no field split counts")
    for field, expected in expected_field_counts.items():
        counts = fields.get(field)
        if not isinstance(counts, Mapping) or counts != {
            "train": expected["train"], "val": expected["val"], "test": 0
        }:
            raise ValueError(f"training summary field counts are not bound to blind data: {field}")
    return bootstrap._validated_records(summary, epochs=epochs)


def build_analysis_decision(
    *,
    root_summary: Mapping[str, object],
    pilot_summary: Mapping[str, object],
    expected_field_counts: Mapping[str, Mapping[str, int]],
    candidate_denominators: Mapping[str, int],
) -> dict[str, object]:
    root_records = _validate_common_summary(
        root_summary,
        epochs=bootstrap.ROOT_EPOCHS,
        fine_tune_mode="all_parameters",
        expected_field_counts=expected_field_counts,
    )
    if root_summary.get("initialization") != {
        "mode": "random", "optimizer_restored": False, "epoch_reset": True
    } or root_summary.get("best_checkpoint_epoch") != 1:
        raise ValueError("one-epoch topology root was not initialized and selected completely at random")
    pilot_records = _validate_common_summary(
        pilot_summary,
        epochs=bootstrap.PILOT_EPOCHS,
        fine_tune_mode="recipient_only_v12",
        expected_field_counts=expected_field_counts,
    )
    fine_tune = pilot_summary["fine_tune_policy"]
    assert isinstance(fine_tune, Mapping)
    if (
        fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("training_forward") != "private_recipient_branch_only_v12"
        or fine_tune.get("open_text_legacy_recipient_unfrozen") is not False
    ):
        raise ValueError("strict warm-start must train only the complete private recipient branch")

    protected_observed: dict[str, float] = {}
    for field in ("amount", "time", "payment_method_field"):
        denominator = candidate_denominators[field]
        root_value = _candidate_metric(root_records[0], field, expected_records=denominator)
        values = [_candidate_metric(record, field, expected_records=denominator) for record in pilot_records]
        if any(not math.isclose(value, root_value, rel_tol=0.0, abs_tol=0.0) for value in values):
            raise ValueError(f"frozen random-root {field} metric changed during recipient-only training")
        protected_observed[field] = root_value

    recipient_by_epoch = {
        int(record["epoch"]): _candidate_metric(
            record,
            "recipient_field",
            expected_records=candidate_denominators["recipient_field"],
        )
        for record in pilot_records
    }
    best_recipient = max(recipient_by_epoch.values())
    best_epochs = [epoch for epoch, value in recipient_by_epoch.items() if value == best_recipient]
    best_checkpoint_epoch = pilot_summary.get("best_checkpoint_epoch")
    if best_checkpoint_epoch not in best_epochs:
        raise ValueError("best.pt is not an epoch with maximum strict recipient exact accuracy")
    epoch4 = recipient_by_epoch[4]
    epoch8 = recipient_by_epoch[8]
    gain = epoch8 - epoch4
    continuation_authorized = (
        best_recipient >= bootstrap.CONTINUATION_RECIPIENT_FLOOR
        and gain >= bootstrap.CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR
    )
    observed = {**protected_observed, "recipient_field": best_recipient}
    return {
        "analysis_only": True,
        "branch_source_only": True,
        "production_route_authorized": False,
        "onnx_delivery_authorized": False,
        "delivery_gate_evaluated": False,
        "financial_delivery_checkpoint_eligible": False,
        "delivery_floor_parameters": bootstrap.DELIVERY_FLOORS,
        "nonrecipient_metrics_authoritative_for_delivery": False,
        "nonrecipient_ineligibility_reason": (
            "amount/time/payment tensors originate from a one-epoch random root and must be discarded; "
            "only recipient_* tensors may enter the later sanitizer"
        ),
        "observed_analysis_metrics": observed,
        "would_meet_delivery_floor": {
            field: observed[field] >= floor for field, floor in bootstrap.DELIVERY_FLOORS.items()
        },
        "recipient_delivery_target_reached": best_recipient >= bootstrap.DELIVERY_FLOORS["recipient_field"],
        "continuation_16_epoch_authorized": continuation_authorized,
        "continuation_gates": {
            "minimum_best_recipient_exact": bootstrap.CONTINUATION_RECIPIENT_FLOOR,
            "minimum_epoch4_to_8_gain": bootstrap.CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR,
        },
        "recipient_observed": {
            "best_exact": best_recipient,
            "best_epochs": best_epochs,
            "selected_best_epoch": best_checkpoint_epoch,
            "epoch4_exact": epoch4,
            "epoch8_exact": epoch8,
            "epoch4_to_8_gain": gain,
            "by_epoch": {str(epoch): value for epoch, value in recipient_by_epoch.items()},
        },
        "epoch4_evidence_authority": "analysis_continuation_only_no_checkpoint_or_delivery_authority",
    }


def _assert_checkpoint_metrics_match_summary(
    payload: Mapping[str, object],
    summary_record: Mapping[str, object],
    *,
    candidate_denominators: Mapping[str, int],
    description: str,
) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("epoch") != summary_record.get("epoch"):
        raise ValueError(f"{description} has no matching embedded epoch metrics")
    if metrics.get("validation_performed") is not True or summary_record.get("validation_performed") is not True:
        raise ValueError(f"{description} is not backed by complete validation")
    for field in CANDIDATE_FIELDS:
        embedded = _candidate_metric(metrics, field, expected_records=candidate_denominators[field])
        summarized = _candidate_metric(summary_record, field, expected_records=candidate_denominators[field])
        if not math.isclose(embedded, summarized, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{description} embedded {field} metric differs from training summary")


def _finalize_recovery_impl(
    *,
    input_contract: Path,
    root_output: Path,
    pilot_output: Path,
    output: Path,
    recovery_verifier: Path,
    recovery_launcher: Path,
    frozen_evidence: list[_FrozenEvidence],
) -> dict[str, object]:
    input_contract = bootstrap._require_file(input_contract, "bootstrap input contract")
    root_output = bootstrap._require_directory(root_output, "random-root output")
    pilot_output = bootstrap._require_directory(pilot_output, "strict warm-start output")
    output_root = input_contract.parent
    for stage in (root_output, pilot_output):
        try:
            stage.relative_to(output_root)
        except ValueError:
            raise ValueError(f"training output escapes the fresh bootstrap root: {stage}") from None
    bootstrap._validate_output_tree(output_root)
    output = Path(os.path.abspath(os.fspath(output)))
    bootstrap._require_no_reparse(output, include_leaf=False)
    try:
        output.relative_to(output_root)
    except ValueError:
        raise ValueError(f"decision output escapes the fresh bootstrap root: {output}") from None
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite recovery decision: {output}")

    recovery_verifier = bootstrap._require_file(recovery_verifier, "recovery verifier")
    if recovery_verifier != Path(__file__).resolve():
        raise ValueError("recovery verifier path does not identify the executing module")
    recovery_launcher = bootstrap._require_file(recovery_launcher, "recovery launcher")
    target_helper_path = bootstrap._require_file(
        Path(str(target_helpers.__file__)), "recovery amount-target helper"
    )
    root_summary_path = bootstrap._require_read_only_file(
        root_output / "training_summary.json", "random-root summary"
    )
    pilot_summary_path = bootstrap._require_read_only_file(
        pilot_output / "training_summary.json", "strict warm-start summary"
    )
    root_best_path = bootstrap._require_read_only_file(root_output / "best.pt", "random-root best checkpoint")
    root_last_path = bootstrap._require_read_only_file(root_output / "last.pt", "random-root last checkpoint")
    pilot_best_path = bootstrap._require_read_only_file(pilot_output / "best.pt", "strict warm-start best checkpoint")
    pilot_last_path = bootstrap._require_read_only_file(pilot_output / "last.pt", "strict warm-start last checkpoint")

    # Freeze every path whose bytes are parsed below before the first parse.
    # Referenced contract inputs are added to a second frozen set once the
    # contract itself has been decoded from this open inode.
    primary_frozen = _FrozenEvidence(
        {
            "input_contract": input_contract,
            "root_summary": root_summary_path,
            "pilot_summary": pilot_summary_path,
            "root_best_checkpoint": root_best_path,
            "root_last_checkpoint": root_last_path,
            "pilot_best_checkpoint": pilot_best_path,
            "pilot_last_checkpoint": pilot_last_path,
            "recovery_verifier": recovery_verifier,
            "recovery_launcher": recovery_launcher,
            "recovery_amount_target_helper": target_helper_path,
        }
    )
    frozen_evidence.append(primary_frozen)
    primary_manifest = primary_frozen.manifest()
    contract = primary_frozen.json_object("input_contract")
    root_summary = primary_frozen.json_object("root_summary", training=True)
    pilot_summary = primary_frozen.json_object("pilot_summary", training=True)
    bootstrap._verify_bound_inputs(contract, input_contract)
    if root_summary.get("config") != pilot_summary.get("config"):
        raise ValueError("strict warm-start changed the random-root model configuration")
    dataset_binding = contract.get("dataset_binding")
    if not isinstance(dataset_binding, Mapping):
        raise ValueError("input contract has no bound dataset evidence")
    expected_field_counts = dataset_binding.get("field_counts")
    if not isinstance(expected_field_counts, Mapping):
        raise ValueError("input contract has no bound per-field split counts")
    denominator_evidence = rebuild_candidate_denominators(
        contract=contract, summary_config=root_summary.get("config")
    )
    candidate_denominators = denominator_evidence["candidate_val_denominators"]
    assert isinstance(candidate_denominators, Mapping)
    decision = build_analysis_decision(
        root_summary=root_summary,
        pilot_summary=pilot_summary,
        expected_field_counts=expected_field_counts,
        candidate_denominators=candidate_denominators,
    )

    frozen_paths: dict[str, Path] = {
        "source_manifest": Path(str(contract["source_manifest"])),
        "blind_manifest": Path(str(contract["blind_manifest"])),
        "blind_contract": Path(str(contract["blind_contract"])),
    }
    original_code_inputs = contract.get("code_inputs")
    if not isinstance(original_code_inputs, Mapping):
        raise ValueError("input contract code bindings are missing")
    for name, binding in original_code_inputs.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"input contract code binding is invalid: {name}")
        frozen_paths[f"original_code_{name}"] = Path(str(binding.get("path", "")))
    referenced_frozen = _FrozenEvidence(frozen_paths)
    frozen_evidence.append(referenced_frozen)
    frozen_manifest = {**primary_manifest, **referenced_frozen.manifest()}
    recovery_code_inputs = {
        name.removeprefix("recovery_"): evidence
        for name, evidence in frozen_manifest.items()
        if name.startswith("recovery_")
    }
    root_best, torch = bootstrap._torch_load(root_best_path)
    root_last, _ = bootstrap._torch_load(root_last_path)
    pilot_best, _ = bootstrap._torch_load(pilot_best_path)
    pilot_last, _ = bootstrap._torch_load(pilot_last_path)
    for checkpoint in (root_best, root_last, pilot_best, pilot_last):
        bootstrap._validate_checkpoint_common(checkpoint)
    expected_config = root_summary.get("config")
    if any(checkpoint.get("config") != expected_config for checkpoint in (root_best, root_last, pilot_best, pilot_last)):
        raise ValueError("checkpoint and summary configurations do not match exactly")
    if root_best.get("epoch") != 1 or root_last.get("epoch") != 1:
        raise ValueError("random-root checkpoints must come from epoch one")
    if root_best.get("initialization") != {
        "mode": "random", "optimizer_restored": False, "epoch_reset": True
    }:
        raise ValueError("random-root best checkpoint has non-random ancestry")
    best_epoch = pilot_summary.get("best_checkpoint_epoch")
    if pilot_best.get("epoch") != best_epoch or pilot_last.get("epoch") != bootstrap.PILOT_EPOCHS:
        raise ValueError("strict warm-start checkpoint epochs do not match the summary")
    root_records = root_summary.get("records")
    pilot_records = pilot_summary.get("records")
    if not isinstance(root_records, list) or not isinstance(pilot_records, list):
        raise ValueError("training summaries have no epoch records")
    for checkpoint, record, description in (
        (root_best, root_records[0], "random-root best checkpoint"),
        (root_last, root_records[0], "random-root last checkpoint"),
        (pilot_best, pilot_records[int(best_epoch) - 1], "strict warm-start best checkpoint"),
        (pilot_last, pilot_records[bootstrap.PILOT_EPOCHS - 1], "strict warm-start last checkpoint"),
    ):
        _assert_checkpoint_metrics_match_summary(
            checkpoint,
            record,
            candidate_denominators=candidate_denominators,
            description=description,
        )

    expected_root_sha = str(primary_manifest["root_best_checkpoint"]["sha256"])
    for name, checkpoint in (("best", pilot_best), ("last", pilot_last)):
        initialization = checkpoint.get("initialization")
        if (
            not isinstance(initialization, Mapping)
            or initialization.get("mode") != "parameter_only"
            or initialization.get("checkpoint_sha256") != expected_root_sha
            or initialization.get("optimizer_restored") is not False
            or initialization.get("epoch_reset") is not True
        ):
            raise ValueError(f"pilot {name} checkpoint is not a strict fresh warm-start from the random root")
    pilot_summary_init = pilot_summary.get("initialization")
    if not isinstance(pilot_summary_init, Mapping) or pilot_summary_init.get("checkpoint_sha256") != expected_root_sha:
        raise ValueError("pilot summary is not hash-bound to the random root checkpoint")

    root_nonrecipient = bootstrap._nonrecipient_manifest(root_best, torch=torch)
    if bootstrap._nonrecipient_manifest(root_last, torch=torch) != root_nonrecipient:
        raise ValueError("random-root best/last nonrecipient partitions disagree")
    if bootstrap._nonrecipient_manifest(pilot_best, torch=torch) != root_nonrecipient:
        raise ValueError("strict warm-start best checkpoint changed nonrecipient tensors")
    if bootstrap._nonrecipient_manifest(pilot_last, torch=torch) != root_nonrecipient:
        raise ValueError("strict warm-start last checkpoint changed nonrecipient tensors")
    root_recipient = bootstrap._partition_manifest(root_best, prefix="recipient_", torch=torch)
    pilot_best_recipient = bootstrap._partition_manifest(pilot_best, prefix="recipient_", torch=torch)
    pilot_last_recipient = bootstrap._partition_manifest(pilot_last, prefix="recipient_", torch=torch)
    if pilot_last_recipient == root_recipient:
        raise ValueError("eight recipient-only epochs did not change any recipient tensor")
    for key in (
        "amount_characters", "time_characters", "payment_characters", "recipient_characters",
        "status_classes", "payment_bank_prefix_classes",
    ):
        if pilot_best.get(key) != root_best.get(key) or pilot_last.get(key) != root_best.get(key):
            raise ValueError(f"strict warm-start changed semantic label map {key}")

    payload: dict[str, object] = {
        "schema_version": bootstrap.SCHEMA_VERSION,
        "kind": RECOVERY_KIND,
        **decision,
        "recovery_reason": "original finalizer compared v12 amount candidate records to raw amount slot count",
        "candidate_denominator_evidence": denominator_evidence,
        "input_contract": str(input_contract),
        "input_contract_sha256": primary_manifest["input_contract"]["sha256"],
        "original_bound_code_inputs": contract["code_inputs"],
        "recovery_code_inputs": recovery_code_inputs,
        "blind_manifest_sha256": contract["blind_manifest_sha256"],
        "test_rows_physically_present_in_training_manifest": False,
        "test_labels_used_by_training": False,
        "test_metrics_computed": False,
        "random_root": {
            "output": str(root_output),
            "best_checkpoint": str(root_best_path),
            "best_checkpoint_sha256": expected_root_sha,
            "summary_sha256": primary_manifest["root_summary"]["sha256"],
            "initialization": "random",
            "epochs": bootstrap.ROOT_EPOCHS,
        },
        "strict_recipient_warmstart": {
            "output": str(pilot_output),
            "best_checkpoint": str(pilot_best_path),
            "best_checkpoint_sha256": primary_manifest["pilot_best_checkpoint"]["sha256"],
            "last_checkpoint": str(pilot_last_path),
            "last_checkpoint_sha256": primary_manifest["pilot_last_checkpoint"]["sha256"],
            "summary_sha256": primary_manifest["pilot_summary"]["sha256"],
            "epochs": bootstrap.PILOT_EPOCHS,
            "optimizer_restored": False,
            "epoch_reset": True,
            "recipient_best_tensor_manifest": pilot_best_recipient,
            "recipient_last_tensor_manifest": pilot_last_recipient,
            "nonrecipient_tensor_manifest": root_nonrecipient,
            "nonrecipient_byte_identical_to_random_root": True,
        },
        "authorized_16_epoch_warmstart_checkpoint": (
            str(pilot_best_path) if decision["continuation_16_epoch_authorized"] else None
        ),
        "notice": (
            "RECOVERED ANALYSIS ONLY. Even a 90% recipient result is not delivery acceptance. "
            "Discard every non-recipient tensor and run the later full v13 protected validation."
        ),
        "epoch4_evidence_limit": (
            "Epoch 4 is used solely for the 8-to-16 analysis continuation decision and has no "
            "checkpoint, sanitizer, model-selection, or delivery authority."
        ),
    }
    # Re-run the complete original binding, including every snapshot crop,
    # immediately before publication.  Then re-derive the denominator record
    # identities and verify every open evidence inode/path pair.  This makes a
    # long Torch checkpoint inspection fail closed on any concurrent change.
    bootstrap._verify_bound_inputs(contract, input_contract)
    final_denominator_evidence = rebuild_candidate_denominators(
        contract=contract, summary_config=root_summary.get("config")
    )
    if final_denominator_evidence != denominator_evidence:
        raise RuntimeError("candidate denominator evidence changed during recovery")
    for frozen in frozen_evidence:
        frozen.verify()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite recovery decision: {output}")
    _atomic_write_json_no_clobber(output, payload)
    return payload


def finalize_recovery(
    *,
    input_contract: Path,
    root_output: Path,
    pilot_output: Path,
    output: Path,
    recovery_verifier: Path,
    recovery_launcher: Path,
) -> dict[str, object]:
    """Finalize while releasing Windows checkpoint handles on every exit."""

    frozen_evidence: list[_FrozenEvidence] = []
    try:
        return _finalize_recovery_impl(
            input_contract=input_contract,
            root_output=root_output,
            pilot_output=pilot_output,
            output=output,
            recovery_verifier=recovery_verifier,
            recovery_launcher=recovery_launcher,
            frozen_evidence=frozen_evidence,
        )
    finally:
        for frozen in reversed(frozen_evidence):
            frozen.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover a completed v12 recipient bootstrap finalization")
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--root-output", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-verifier", type=Path, required=True)
    parser.add_argument("--recovery-launcher", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = finalize_recovery(
        input_contract=args.input_contract,
        root_output=args.root_output,
        pilot_output=args.pilot_output,
        output=args.output,
        recovery_verifier=args.recovery_verifier,
        recovery_launcher=args.recovery_launcher,
    )
    observed = result["recipient_observed"]
    assert isinstance(observed, Mapping)
    print(
        "recipient_random_bootstrap_recovery_decision "
        f"best={float(observed['best_exact']):.2%} "
        f"epoch4_to_8_gain={float(observed['epoch4_to_8_gain']):+.2%} "
        f"continuation16={result['continuation_16_epoch_authorized']} "
        "production=False"
    )


if __name__ == "__main__":
    main()
