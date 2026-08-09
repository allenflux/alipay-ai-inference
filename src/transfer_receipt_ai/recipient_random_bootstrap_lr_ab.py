"""Audit the single-variable learning-rate rescue for recipient bootstrap.

This module is intentionally separate from :mod:`recipient_random_bootstrap`.
The original 031004 run hash-binds that module and its PowerShell launcher, so
changing either file would invalidate the already-completed evidence.  The
rescue starts again from the same one-epoch random root and changes exactly one
training value: AdamW learning rate ``1e-4 -> 3e-4``.

The workflow remains analysis-only.  It physically reuses the immutable
train/validation snapshot, never opens test, never exports ONNX, freezes every
non-recipient parameter, validates every epoch, and applies the original
75-percent / epoch-4-to-8 +2pp continuation gates unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path

from . import recipient_random_bootstrap as base
from . import ocr_unified_targets as target_rules


SCHEMA_VERSION = 1
INPUT_KIND = "receipt_recipient_random_bootstrap_lr_ab_input_contract_v1"
DECISION_KIND = "receipt_recipient_random_bootstrap_lr_ab_decision_v1"
SOURCE_RECOVERY_DECISION_KIND = "receipt_recipient_random_bootstrap_recovery_decision_v1"
BASELINE_LEARNING_RATE = 0.0001
CANDIDATE_LEARNING_RATE = 0.0003
PILOT_EPOCHS = base.PILOT_EPOCHS
SEED = 424242
SOURCE_INPUT_NAME = "bootstrap-input.contract.json"
SOURCE_ROOT_OUTPUT_NAME = "random-root-1e"
SOURCE_PILOT_OUTPUT_NAME = "strict-recipient-warmstart-8e"
SOURCE_RECOVERY_DECISION_NAME = "analysis-decision.recovered.json"
CANDIDATE_OUTPUT_NAME = "strict-recipient-lr3e4-8e"
INPUT_CONTRACT_NAME = "lr-ab-input.contract.json"
DECISION_NAME = "lr-ab-decision.json"
EXPECTED_031004_INPUT_CONTRACT_SHA256 = (
    "7f6f2b07b33a5707ea376739e6853629c806675b22b320a9898c45f5bede91fc"
)
EXPECTED_031004_CANDIDATE_DENOMINATORS = {
    "amount": 1428,
    "time": 3738,
    "payment_method_field": 5242,
    "recipient_field": 6789,
}
EXPECTED_031004_RAW_VAL_COUNTS = {
    "amount": 1606,
    "time": 3738,
    "payment_method_field": 5242,
    "recipient_field": 6789,
}
EXPECTED_031004_RECIPIENT_OBSERVED = {
    "best_exact": 4467 / 6789,
    "epoch4_exact": 2651 / 6789,
    "epoch8_exact": 4467 / 6789,
    "epoch4_to_8_gain": (4467 - 2651) / 6789,
}
PUBLICATION_POLICY = "same_directory_exclusive_hardlink_closing_identity_v1"
_WINDOWS_FILE_SHARE_READ = 0x00000001
_WINDOWS_FILE_SHARE_WRITE = 0x00000002
_WINDOWS_FILE_SHARE_DELETE = 0x00000004
_WINDOWS_PUBLICATION_SHARE_MODE = _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_DELETE


def _open_publication_read_handle(path: Path) -> int:
    """Open evidence without blocking deletion of its private Windows link.

    The destination and temporary name are hard links to the same inode.  We
    must keep that inode open through publication, while also removing the
    private temporary name before applying the Windows read-only attribute.
    CRT ``os.open`` handles do not share delete access on Windows, so use a
    native handle with ``FILE_SHARE_DELETE`` there and transfer ownership to a
    Python file descriptor.  Omitting ``FILE_SHARE_WRITE`` also prevents a new
    writer from opening the prepared bytes while the publication proof is
    live.
    """

    open_flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    if os.name != "nt":
        return os.open(path, open_flags | int(getattr(os, "O_NOFOLLOW", 0)))

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    generic_read = 0x80000000
    open_existing = 3
    file_flag_open_reparse_point = 0x00200000
    handle = create_file(
        str(path),
        generic_read,
        _WINDOWS_PUBLICATION_SHARE_MODE,
        None,
        open_existing,
        file_flag_open_reparse_point,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(int(handle), open_flags)
    except BaseException:
        close_handle(handle)
        raise


def _closing_file_identity(
    path: Path,
    *,
    held_status: os.stat_result,
    expected_sha256: str,
) -> dict[str, object]:
    """Prove that ``path`` still names the inode held through publication."""

    base._require_no_reparse(path)
    path_status = path.stat()
    if not os.path.samestat(held_status, path_status):
        raise RuntimeError(f"published evidence path was replaced before close: {path}")
    observed_sha256 = base._sha256(path)
    if observed_sha256 != expected_sha256 or int(path_status.st_size) != int(held_status.st_size):
        raise RuntimeError(f"published evidence bytes changed before close: {path}")
    if path_status.st_mode & 0o222:
        raise RuntimeError(f"published evidence is not read-only at close: {path}")
    return {
        "path": str(path),
        "device": int(path_status.st_dev),
        "inode": int(path_status.st_ino),
        "size_bytes": int(path_status.st_size),
        "sha256": observed_sha256,
        "read_only": True,
        "policy": PUBLICATION_POLICY,
    }


def _atomic_write_json_no_clobber(
    path: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    """Exclusively publish JSON and verify the closing path/inode identity.

    A same-directory hard link is the commit operation.  It is atomic and
    fails when *anything* already occupies the destination, including a
    broken symlink.  Unlike ``replace()``, a racing writer can never be
    overwritten.  The temporary inode stays open until the destination has
    been sealed read-only, hash-checked, and proven to name that same inode.
    """

    path = Path(os.path.abspath(os.fspath(path)))
    base._require_no_reparse(path, include_leaf=False)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    writer: int | None = None
    held: int | None = None
    held_status: os.stat_result | None = None
    try:
        writer = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(writer, "wb", closefd=True) as handle:
            writer = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        held = _open_publication_read_handle(temporary)
        held_status = os.fstat(held)
        if not os.path.samestat(held_status, temporary.stat()):
            raise RuntimeError("temporary evidence path changed before publication")
        # link() is an atomic create-if-absent operation on the same volume.
        os.link(temporary, path, follow_symlinks=False)
        base._require_no_reparse(path)
        if not os.path.samestat(held_status, path.stat()):
            raise RuntimeError("published evidence does not name the prepared inode")
        if not os.path.samestat(held_status, temporary.stat()):
            raise RuntimeError("temporary evidence path changed during publication")
        # Remove the private link before applying the Windows read-only
        # attribute, which otherwise can make unlinking that hard link fail.
        temporary.unlink()
        os.chmod(path, 0o444)
        held_status = os.fstat(held)
        return _closing_file_identity(
            path,
            held_status=held_status,
            expected_sha256=expected_sha256,
        )
    finally:
        if writer is not None:
            os.close(writer)
        if held is not None:
            os.close(held)
        # Delete only the private inode created by this invocation.  If a
        # racing process replaced the temp name, leave its file untouched.
        try:
            temporary_status = temporary.stat()
        except FileNotFoundError:
            temporary_status = None
        if (
            temporary_status is not None
            and held_status is not None
            and os.path.samestat(temporary_status, held_status)
        ):
            temporary.unlink()


FIXED_BASELINE_RECIPE: dict[str, object] = {
    "device": "cuda:0",
    "architecture": "v12",
    "image_height": 80,
    "image_width": 512,
    "base_channels": 32,
    "numeric_hidden_size": 96,
    "payment_hidden_size": 128,
    "recipient_hidden_size": 256,
    "recipient_value_left_trim": 0.30,
    "recipient_input_height": 128,
    "recipient_input_width": 1536,
    "recipient_branch_channels": 24,
    "recipient_open_text_layers": 2,
    "recipient_open_text_heads": 8,
    "recipient_open_text_feedforward": 2048,
    "recipient_open_text_dropout": 0.0,
    "recipient_backbone": "legacy_depthwise_gru_v1",
    "pooled_width": 8,
    "batch_size": 12,
    "learning_rate": BASELINE_LEARNING_RATE,
    "payment_loss_weight": 1.0,
    "recipient_loss_weight": 4.0,
    "recipient_sampling_weight": 1.0,
    "recipient_rare_character_max_support": 0,
    "recipient_long_text_min_length": 0,
    "recipient_low_confidence_threshold": 0.98,
    "recipient_low_confidence_loss_weight": 0.35,
    "recipient_confidence_curriculum_epochs": 10,
    "recipient_tail_rare_character_max_support": 0,
    "recipient_tail_rare_character_loss_weight": 1.0,
    "recipient_tail_long_text_min_length": 0,
    "recipient_tail_long_text_loss_weight": 1.0,
    "recipient_train_augmentation": "light_v1",
    "recipient_train_splits": ["train"],
    "recipient_only_fine_tune": True,
    "init_checkpoint_mode": "strict",
    "checkpoint_selection": "balanced",
    "ctc_loss_weight": 0.75,
    "structured_loss_weight": 1.0,
    "amount_format_min_confidence": 0.80,
    "payment_bank_prefix_min_support": 3,
    "epochs": PILOT_EPOCHS,
    "seed": SEED,
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "train_progress_every": 250,
    "validation_every": 1,
    "cuda_tf32": True,
    "cudnn_benchmark": True,
    "onnx_export": False,
}
FIXED_CANDIDATE_RECIPE: dict[str, object] = {
    **FIXED_BASELINE_RECIPE,
    "learning_rate": CANDIDATE_LEARNING_RATE,
}

_PERSISTED_RECIPE_KEYS = (
    "config",
    "field_counts",
    "recipient_loss_weight",
    "recipient_sampling_policy",
    "recipient_confidence_policy",
    "recipient_tail_loss_policy",
    "recipient_train_augmentation_policy",
    "recipient_train_split_policy",
    "checkpoint_selection_policy",
    "initialization",
)
_LABEL_KEYS = (
    "amount_characters",
    "time_characters",
    "payment_characters",
    "recipient_characters",
    "status_classes",
    "payment_bank_prefix_classes",
)
_PROTECTED_FIELDS = ("amount", "time", "payment_method_field")


def _finite_rate(value: object, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{description} must be numeric") from None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must be a finite rate")
    return result


def _candidate_metric(
    record: Mapping[str, object], field: str, *, expected_records: int
) -> float:
    by_field = record.get("val_candidate_text_by_field")
    if not isinstance(by_field, Mapping) or not isinstance(by_field.get(field), Mapping):
        raise ValueError(f"training record has no candidate metric for {field}")
    metric = by_field[field]
    assert isinstance(metric, Mapping)
    rate = _finite_rate(metric.get("exact_match"), f"{field} exact_match")
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


def training_validation_candidate_denominators_v12(
    *,
    blind_manifest: Path,
    snapshot_dataset_root: Path,
    config_value: object,
) -> dict[str, int]:
    """Reproduce the v12 trainer's validation-candidate eligibility.

    ``dataset_binding.field_counts`` counts raw non-null slots.  The trainer's
    candidate metric is intentionally narrower for malformed amount display
    targets: an amount contributes only when the v8/v12 target parser can
    produce the same reference used by ``_evaluate_model``.  The other three
    candidate references follow their exact training-validation helpers.
    """

    from . import ocr_unified as unified

    if not isinstance(config_value, Mapping):
        raise ValueError("training config is missing for candidate denominator reconstruction")
    try:
        config = unified.UnifiedReaderConfig(**dict(config_value))
    except TypeError as error:
        raise ValueError("training config cannot be reconstructed") from error
    config.validate()
    if config.architecture_version != 12:
        raise ValueError("LR A/B denominator reconstruction is v12-only")
    records = unified.load_records(
        Path(blind_manifest),
        dataset_root=Path(snapshot_dataset_root),
        config=config,
    )
    counts = {field: 0 for field in (*_PROTECTED_FIELDS, "recipient_field")}
    validation_records = 0
    for record in records:
        if record.get("split") != "val":
            continue
        validation_records += 1
        amount_reference = unified._ctc_slot_text(record, "amount", config=config)
        if unified._uses_v8_protocol(config):
            slots = record.get("slots")
            amount_slot = slots.get("amount") if isinstance(slots, Mapping) else None
            visible = amount_slot.get("visible_text") if isinstance(amount_slot, Mapping) else None
            if (
                isinstance(visible, str)
                and target_rules.parse_amount_visible_format_target(visible) is not None
            ):
                amount_reference = visible
        references = {
            "amount": amount_reference,
            "time": unified._ctc_slot_text(record, "time", config=config),
            "payment_method_field": unified._slot_text(record, "payment_method_field"),
            "recipient_field": unified._recipient_expected_value(record, config=config),
        }
        for field, reference in references.items():
            if reference is not None:
                counts[field] += 1
    if validation_records <= 0 or any(value <= 0 for value in counts.values()):
        raise ValueError("blind manifest has no complete v12 candidate validation denominators")
    return counts


def _recipient_observed(
    summary: Mapping[str, object],
    *,
    expected_records: int,
) -> dict[str, object]:
    records = base._validated_records(summary, epochs=PILOT_EPOCHS)
    by_epoch = {
        int(record["epoch"]): _candidate_metric(
            record, "recipient_field", expected_records=expected_records
        )
        for record in records
    }
    best_exact = max(by_epoch.values())
    best_epochs = [epoch for epoch, value in by_epoch.items() if value == best_exact]
    selected = summary.get("best_checkpoint_epoch")
    if selected not in best_epochs:
        raise ValueError("best.pt is not an epoch with maximum strict recipient exact accuracy")
    epoch4 = by_epoch[4]
    epoch8 = by_epoch[8]
    return {
        "best_exact": best_exact,
        "best_epochs": best_epochs,
        "selected_best_epoch": selected,
        "epoch4_exact": epoch4,
        "epoch8_exact": epoch8,
        "epoch4_to_8_gain": epoch8 - epoch4,
        "by_epoch": {str(epoch): value for epoch, value in by_epoch.items()},
    }


def _recipe_difference() -> dict[str, dict[str, object]]:
    keys = set(FIXED_BASELINE_RECIPE) | set(FIXED_CANDIDATE_RECIPE)
    return {
        key: {
            "baseline": FIXED_BASELINE_RECIPE.get(key),
            "candidate": FIXED_CANDIDATE_RECIPE.get(key),
        }
        for key in sorted(keys)
        if FIXED_BASELINE_RECIPE.get(key) != FIXED_CANDIDATE_RECIPE.get(key)
    }


_PS_ARRAY_TOKEN = re.compile(
    r'(?P<quoted>"(?:[^"`]|`.)*")|(?P<variable>\$[A-Za-z_][A-Za-z0-9_]*)|(?P<separator>[\s,]+)'
)


def _powershell_scalar(source: str, name: str) -> str:
    matches = list(re.finditer(
        rf"(?m)^\s*\${re.escape(name)}\s*=\s*(?P<value>[^#\r\n]+?)\s*$",
        source,
    ))
    if len(matches) != 1:
        raise ValueError(f"PowerShell runner must assign ${name} exactly once")
    match = matches[0]
    value = match.group("value").strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _powershell_array_tokens(
    source: str,
    name: str,
    *,
    variables: Mapping[str, str],
    expected_prefix: str | None = None,
) -> list[str]:
    prefix = (
        rf"\${re.escape(expected_prefix)}\s*\+\s*"
        if expected_prefix is not None
        else ""
    )
    match = re.search(
        rf"(?ms)^\s*\${re.escape(name)}\s*=\s*{prefix}@\(\s*(?P<body>.*?)^\s*\)\s*$",
        source,
    )
    if match is None:
        raise ValueError(f"PowerShell runner has no simple ${name} argument array")
    body = match.group("body")
    tokens: list[str] = []
    cursor = 0
    while cursor < len(body):
        token = _PS_ARRAY_TOKEN.match(body, cursor)
        if token is None:
            raise ValueError(f"PowerShell ${name} contains non-literal argument syntax")
        cursor = token.end()
        if token.lastgroup == "separator":
            continue
        value = token.group(token.lastgroup or "")
        if token.lastgroup == "quoted":
            value = value[1:-1]
            if "`" in value:
                raise ValueError(f"PowerShell ${name} contains escaped argument text")
        if value.startswith("$"):
            variable_name = value[1:]
            if variable_name not in variables:
                raise ValueError(
                    f"PowerShell ${name} uses unbound argument variable ${variable_name}"
                )
            value = variables[variable_name]
        tokens.append(value)
    return tokens


def _option_names(tokens: list[str], description: str) -> list[str]:
    if tokens[:3] != ["-m", "transfer_receipt_ai.ocr_unified", "train"]:
        raise ValueError(f"{description} does not invoke the fixed unified trainer")
    names: list[str] = []
    index = 3
    while index < len(tokens):
        option = tokens[index]
        if not option.startswith("--"):
            raise ValueError(f"{description} has a positional or detached argument: {option}")
        if option in names:
            raise ValueError(f"{description} repeats training option {option}")
        names.append(option)
        index += 1
        while index < len(tokens) and not tokens[index].startswith("--"):
            index += 1
    return names


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _normalized_train_namespace(tokens: list[str], description: str) -> dict[str, object]:
    from . import ocr_unified as unified

    _option_names(tokens, description)
    try:
        namespace = unified.build_parser().parse_args(tokens[2:])
    except SystemExit as error:
        raise ValueError(f"{description} cannot be parsed by the hash-bound trainer") from error
    return {key: _jsonable(value) for key, value in sorted(vars(namespace).items())}


def _baseline_pilot_argv(source: str) -> list[str]:
    variables = {
        "blindRecords": "__BOUND_BLIND_RECORDS__",
        "snapshotRoot": "__BOUND_SNAPSHOT_ROOT__",
        "pilotOutput": "__PILOT_OUTPUT__",
        "rootCheckpoint": "__BOUND_RANDOM_ROOT_BEST__",
        "pilotEpochs": _powershell_scalar(source, "pilotEpochs"),
    }
    common = _powershell_array_tokens(source, "commonTrainArgs", variables=variables)
    pilot = _powershell_array_tokens(
        source,
        "pilotArgs",
        variables=variables,
        expected_prefix="commonTrainArgs",
    )
    # pilotArgs is declared as commonTrainArgs + @( ... ); the parser returns
    # only the literal suffix for that declaration.
    return common + pilot


def _candidate_train_argv(source: str) -> list[str]:
    variables = {
        "blindRecords": "__BOUND_BLIND_RECORDS__",
        "snapshotRoot": "__BOUND_SNAPSHOT_ROOT__",
        "pilotOutput": "__PILOT_OUTPUT__",
        "rootCheckpoint": "__BOUND_RANDOM_ROOT_BEST__",
        "pilotEpochs": _powershell_scalar(source, "pilotEpochs"),
        "seed": _powershell_scalar(source, "seed"),
        "candidateLearningRate": _powershell_scalar(source, "candidateLearningRate"),
    }
    if source.count("$trainArgs") != 2 or "Invoke-Python $trainArgs" not in source:
        raise ValueError("LR A/B runner may define and invoke $trainArgs exactly once")
    return _powershell_array_tokens(source, "trainArgs", variables=variables)


def _namespace_difference(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    return {
        key: {"baseline": baseline.get(key), "candidate": candidate.get(key)}
        for key in sorted(set(baseline) | set(candidate))
        if baseline.get(key) != candidate.get(key)
    }


def _assert_namespace_matches_recipe(
    namespace: Mapping[str, object], recipe: Mapping[str, object], description: str
) -> None:
    for key, expected in recipe.items():
        if key == "onnx_export":
            observed = namespace.get("onnx_output") is not None
        else:
            observed = namespace.get(key)
        if observed != expected:
            raise ValueError(f"{description} differs from fixed recipe field {key}")


def _actual_train_argv_evidence(
    *, baseline_proof: Mapping[str, object], candidate_runner: Path
) -> dict[str, object]:
    baseline_namespace = baseline_proof.get("normalized_train_namespace")
    baseline_options = baseline_proof.get("option_names")
    if not isinstance(baseline_namespace, Mapping) or not isinstance(baseline_options, list):
        raise ValueError("source baseline has no normalized train argv evidence")
    candidate_source = candidate_runner.read_text(encoding="utf-8")
    candidate_tokens = _candidate_train_argv(candidate_source)
    candidate_options = _option_names(candidate_tokens, "LR A/B candidate train argv")
    candidate_namespace = _normalized_train_namespace(
        candidate_tokens, "LR A/B candidate train argv"
    )
    if set(candidate_options) != set(baseline_options):
        extra = sorted(set(candidate_options) - set(baseline_options))
        missing = sorted(set(baseline_options) - set(candidate_options))
        raise ValueError(
            "LR A/B actual train option set differs from baseline "
            f"(extra={extra}, missing={missing})"
        )
    difference = _namespace_difference(baseline_namespace, candidate_namespace)
    if difference != {
        "learning_rate": {
            "baseline": BASELINE_LEARNING_RATE,
            "candidate": CANDIDATE_LEARNING_RATE,
        }
    }:
        raise ValueError("LR A/B actual train argv changes more than learning_rate")
    _assert_namespace_matches_recipe(
        baseline_namespace, FIXED_BASELINE_RECIPE, "source baseline train argv"
    )
    _assert_namespace_matches_recipe(
        candidate_namespace, FIXED_CANDIDATE_RECIPE, "LR A/B candidate train argv"
    )
    if baseline_namespace.get("weight_decay") != 0.0001:
        raise ValueError("hash-bound trainer default weight_decay is not the expected 1e-4")
    return {
        "policy": "full_hash_bound_trainer_argparse_namespace_only_learning_rate_diff_v1",
        "baseline_option_names": sorted(baseline_options),
        "candidate_option_names": sorted(candidate_options),
        "baseline_namespace": dict(baseline_namespace),
        "candidate_namespace": candidate_namespace,
        "namespace_difference": difference,
        "implicit_weight_decay": baseline_namespace["weight_decay"],
    }


def _file_descriptor(path: Path, description: str, *, read_only: bool = True) -> dict[str, object]:
    resolved = (
        base._require_read_only_file(path, description)
        if read_only
        else base._require_file(path, description)
    )
    status = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(status.st_size),
        "sha256": base._sha256(resolved),
        "read_only_required": read_only,
    }


def _verify_file_descriptor(value: object, description: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
        "read_only_required",
    }:
        raise ValueError(f"{description} descriptor is invalid")
    read_only = value.get("read_only_required") is True
    path = (
        base._require_read_only_file(Path(str(value.get("path", ""))), description)
        if read_only
        else base._require_file(Path(str(value.get("path", ""))), description)
    )
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{description} descriptor size is invalid")
    expected_sha = base._require_sha(value.get("sha256"), f"{description} SHA-256")
    if path.stat().st_size != size or base._sha256(path) != expected_sha:
        raise ValueError(f"{description} changed after LR A/B input binding")
    return path


def _checkpoint_metrics_match(
    payload: Mapping[str, object],
    summary_record: Mapping[str, object],
    *,
    expected_candidate_records: Mapping[str, int],
    description: str,
) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("epoch") != summary_record.get("epoch"):
        raise ValueError(f"{description} has no matching embedded epoch metrics")
    if metrics.get("validation_performed") is not True or summary_record.get("validation_performed") is not True:
        raise ValueError(f"{description} is not backed by complete validation")
    for field in (*_PROTECTED_FIELDS, "recipient_field"):
        expected = int(expected_candidate_records[field])
        embedded = _candidate_metric(metrics, field, expected_records=expected)
        summarized = _candidate_metric(summary_record, field, expected_records=expected)
        if not math.isclose(embedded, summarized, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{description} embedded {field} metric differs from training summary")


def _validate_checkpoint_labels(
    child: Mapping[str, object], parent: Mapping[str, object], *, description: str
) -> None:
    for key in _LABEL_KEYS:
        if child.get(key) != parent.get(key):
            raise ValueError(f"{description} changed semantic label map {key}")


def _baseline_learning_rate_proof(input_contract: Mapping[str, object]) -> dict[str, object]:
    code_inputs = input_contract.get("code_inputs")
    if not isinstance(code_inputs, Mapping) or not isinstance(code_inputs.get("runner"), Mapping):
        raise ValueError("source input contract has no bound baseline runner")
    runner_value = code_inputs["runner"]
    assert isinstance(runner_value, Mapping)
    runner = base._require_file(Path(str(runner_value.get("path", ""))), "source baseline runner")
    expected_sha = base._require_sha(
        runner_value.get("sha256"), "source baseline runner SHA-256"
    )
    if base._sha256(runner) != expected_sha:
        raise ValueError("source baseline runner changed after input binding")
    source = runner.read_text(encoding="utf-8")
    tokens = _baseline_pilot_argv(source)
    namespace = _normalized_train_namespace(tokens, "source baseline pilot train argv")
    _assert_namespace_matches_recipe(namespace, FIXED_BASELINE_RECIPE, "source baseline train argv")
    if namespace.get("weight_decay") != 0.0001:
        raise ValueError("hash-bound baseline trainer default weight_decay is not 1e-4")
    return {
        "runner_path": str(runner),
        "runner_sha256": expected_sha,
        "learning_rate": BASELINE_LEARNING_RATE,
        "proof": "hash_bound_full_train_argv_and_trainer_argparse_namespace_v1",
        "option_names": _option_names(tokens, "source baseline pilot train argv"),
        "normalized_train_namespace": namespace,
    }


def _assert_expected_031004_source_identity(
    *,
    input_contract_sha256: str,
    candidate_denominators: Mapping[str, int],
    observed: Mapping[str, object],
) -> None:
    if input_contract_sha256 != EXPECTED_031004_INPUT_CONTRACT_SHA256:
        raise ValueError("source input contract is not the hash-bound 031004 run")
    if dict(candidate_denominators) != EXPECTED_031004_CANDIDATE_DENOMINATORS:
        raise ValueError("source candidate denominators are not the real 031004 evidence")
    for key, expected in EXPECTED_031004_RECIPIENT_OBSERVED.items():
        if not math.isclose(
            float(observed.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"source recipient {key} is not the real 031004 evidence")


def _source_decision_descriptor(
    *,
    decision_path: Path | None,
    source_root: Path,
    input_contract: Path,
    candidate_denominators: Mapping[str, int],
    observed: Mapping[str, object],
) -> dict[str, object]:
    recovery_default = source_root / SOURCE_RECOVERY_DECISION_NAME
    if os.path.lexists(recovery_default):
        if decision_path is not None and not base._same_path(str(decision_path), recovery_default):
            raise ValueError("recovery decision exists but a different source decision was supplied")
        decision_path = recovery_default
    if decision_path is None:
        raise ValueError("real 031004 LR A/B requires analysis-decision.recovered.json")
    path = base._require_read_only_file(decision_path, "source recovery decision")
    if not base._same_path(str(path), recovery_default):
        raise ValueError("source decision must be the 031004 recovery decision at its canonical path")
    payload = base._json_load(path)
    recorded = payload.get("recipient_observed")
    denominator_evidence = payload.get("candidate_denominator_evidence")
    input_payload = base._json_load(input_contract)
    raw_counts = (
        input_payload.get("dataset_binding", {}).get("field_counts", {})
        if isinstance(input_payload.get("dataset_binding"), Mapping)
        else {}
    )
    observed_raw_counts = {
        field: (
            raw_counts.get(field, {}).get("val")
            if isinstance(raw_counts.get(field), Mapping)
            else None
        )
        for field in EXPECTED_031004_RAW_VAL_COUNTS
    }
    if (
        payload.get("kind") != SOURCE_RECOVERY_DECISION_KIND
        or payload.get("analysis_only") is not True
        or payload.get("production_route_authorized") is not False
        or payload.get("onnx_delivery_authorized") is not False
        or payload.get("continuation_16_epoch_authorized") is not False
        or payload.get("authorized_16_epoch_warmstart_checkpoint") is not None
        or payload.get("input_contract_sha256") != base._sha256(input_contract)
        or not isinstance(recorded, Mapping)
        or not isinstance(denominator_evidence, Mapping)
        or denominator_evidence.get("policy") != "v12_candidate_reference_eligibility_v1"
        or denominator_evidence.get("candidate_val_denominators")
        != EXPECTED_031004_CANDIDATE_DENOMINATORS
        or denominator_evidence.get("raw_val_field_counts") != EXPECTED_031004_RAW_VAL_COUNTS
        or dict(candidate_denominators) != EXPECTED_031004_CANDIDATE_DENOMINATORS
        or observed_raw_counts != EXPECTED_031004_RAW_VAL_COUNTS
    ):
        raise ValueError("source recovery decision is not compatible 031004 analysis evidence")
    for key in ("best_exact", "epoch4_exact", "epoch8_exact", "epoch4_to_8_gain"):
        if not math.isclose(
            float(recorded.get(key, math.nan)),
            float(observed[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"source recovery decision differs from reconstructed {key}")
    _assert_expected_031004_source_identity(
        input_contract_sha256=base._sha256(input_contract),
        candidate_denominators=candidate_denominators,
        observed=recorded,
    )
    return {
        "used": True,
        "kind": SOURCE_RECOVERY_DECISION_KIND,
        "evidence": _file_descriptor(path, "source recovery decision"),
        "input_contract_sha256": payload["input_contract_sha256"],
        "candidate_denominators": dict(candidate_denominators),
        "continuation_16_epoch_authorized": False,
    }


def _validate_source_closure(
    source_root: Path,
    *,
    source_decision: Path | None = None,
) -> tuple[dict[str, object], Mapping[str, object], Mapping[str, object]]:
    source_root = base._require_directory(source_root, "source bootstrap root")
    base._validate_output_tree(source_root)
    input_contract_path = base._require_read_only_file(
        source_root / SOURCE_INPUT_NAME, "source bootstrap input contract"
    )
    input_contract = base._json_load(input_contract_path)
    base._verify_bound_inputs(input_contract, input_contract_path)
    input_contract_sha256 = base._sha256(input_contract_path)
    if input_contract_sha256 != EXPECTED_031004_INPUT_CONTRACT_SHA256:
        raise ValueError("source input contract is not the hash-bound 031004 run")
    baseline_lr_proof = _baseline_learning_rate_proof(input_contract)
    dataset = input_contract.get("dataset_binding")
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("field_counts"), Mapping):
        raise ValueError("source input contract has no bound field counts")
    expected_field_counts = dataset["field_counts"]
    assert isinstance(expected_field_counts, Mapping)

    root_output = base._require_directory(
        source_root / SOURCE_ROOT_OUTPUT_NAME, "source random-root output"
    )
    pilot_output = base._require_directory(
        source_root / SOURCE_PILOT_OUTPUT_NAME, "source strict warm-start output"
    )
    root_summary_path = base._require_read_only_file(
        root_output / "training_summary.json", "source random-root summary"
    )
    pilot_summary_path = base._require_read_only_file(
        pilot_output / "training_summary.json", "source strict warm-start summary"
    )
    root_summary = base._training_json_load(root_summary_path)
    pilot_summary = base._training_json_load(pilot_summary_path)
    root_records = base._validate_common_summary(
        root_summary,
        epochs=base.ROOT_EPOCHS,
        fine_tune_mode="all_parameters",
        expected_field_counts=expected_field_counts,
    )
    pilot_records = base._validate_common_summary(
        pilot_summary,
        epochs=PILOT_EPOCHS,
        fine_tune_mode="recipient_only_v12",
        expected_field_counts=expected_field_counts,
    )
    if root_summary.get("config") != pilot_summary.get("config"):
        raise ValueError("source strict warm-start changed random-root topology")
    expected_candidate_records = training_validation_candidate_denominators_v12(
        blind_manifest=Path(str(input_contract["blind_manifest"])),
        snapshot_dataset_root=Path(str(input_contract["snapshot_dataset_root"])),
        config_value=root_summary.get("config"),
    )
    if expected_candidate_records != EXPECTED_031004_CANDIDATE_DENOMINATORS:
        raise ValueError("source candidate denominators are not the real 031004 evidence")
    for field, records in expected_candidate_records.items():
        raw_val = int(expected_field_counts[field]["val"])
        if records > raw_val:
            raise ValueError(f"candidate denominator exceeds raw val slots for {field}")
    if root_summary.get("initialization") != {
        "mode": "random",
        "optimizer_restored": False,
        "epoch_reset": True,
    }:
        raise ValueError("source root was not freshly initialized at random")
    fine_tune = pilot_summary.get("fine_tune_policy")
    if (
        not isinstance(fine_tune, Mapping)
        or fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("training_forward") != "private_recipient_branch_only_v12"
        or fine_tune.get("open_text_legacy_recipient_unfrozen") is not False
    ):
        raise ValueError("source pilot was not strict private recipient-only training")

    root_best_path = base._require_read_only_file(root_output / "best.pt", "source random-root best")
    root_last_path = base._require_read_only_file(root_output / "last.pt", "source random-root last")
    pilot_best_path = base._require_read_only_file(pilot_output / "best.pt", "source pilot best")
    pilot_last_path = base._require_read_only_file(pilot_output / "last.pt", "source pilot last")
    root_best, torch = base._torch_load(root_best_path)
    root_last, _ = base._torch_load(root_last_path)
    pilot_best, _ = base._torch_load(pilot_best_path)
    pilot_last, _ = base._torch_load(pilot_last_path)
    for payload in (root_best, root_last, pilot_best, pilot_last):
        base._validate_checkpoint_common(payload)
    expected_config = root_summary.get("config")
    if any(payload.get("config") != expected_config for payload in (root_best, root_last, pilot_best, pilot_last)):
        raise ValueError("source checkpoint and summary topology differ")
    if root_best.get("epoch") != 1 or root_last.get("epoch") != 1:
        raise ValueError("source random-root checkpoint epoch is invalid")
    if root_best.get("initialization") != root_summary.get("initialization"):
        raise ValueError("source random-root checkpoint ancestry is invalid")
    source_best_epoch = pilot_summary.get("best_checkpoint_epoch")
    if pilot_best.get("epoch") != source_best_epoch or pilot_last.get("epoch") != PILOT_EPOCHS:
        raise ValueError("source pilot checkpoint epochs do not match its summary")
    _checkpoint_metrics_match(
        root_best,
        root_records[0],
        expected_candidate_records=expected_candidate_records,
        description="source random-root best",
    )
    _checkpoint_metrics_match(
        root_last,
        root_records[0],
        expected_candidate_records=expected_candidate_records,
        description="source random-root last",
    )
    _checkpoint_metrics_match(
        pilot_best,
        pilot_records[int(source_best_epoch) - 1],
        expected_candidate_records=expected_candidate_records,
        description="source pilot best",
    )
    _checkpoint_metrics_match(
        pilot_last,
        pilot_records[-1],
        expected_candidate_records=expected_candidate_records,
        description="source pilot last",
    )
    root_sha = base._sha256(root_best_path)
    expected_initialization = pilot_summary.get("initialization")
    if not isinstance(expected_initialization, Mapping):
        raise ValueError("source pilot has no initialization provenance")
    for description, payload in (("best", pilot_best), ("last", pilot_last)):
        initialization = payload.get("initialization")
        if (
            not isinstance(initialization, Mapping)
            or initialization.get("mode") != "parameter_only"
            or not base._same_path(initialization.get("checkpoint_path"), root_best_path)
            or initialization.get("checkpoint_sha256") != root_sha
            or initialization.get("optimizer_restored") is not False
            or initialization.get("epoch_reset") is not True
        ):
            raise ValueError(f"source pilot {description} is not a fresh strict root warm-start")
    if dict(expected_initialization) != dict(pilot_best["initialization"]):
        raise ValueError("source pilot summary and best initialization differ")
    root_nonrecipient = base._nonrecipient_manifest(root_best, torch=torch)
    if base._nonrecipient_manifest(root_last, torch=torch) != root_nonrecipient:
        raise ValueError("source random-root best/last non-recipient bytes differ")
    if base._nonrecipient_manifest(pilot_best, torch=torch) != root_nonrecipient:
        raise ValueError("source pilot best changed a frozen non-recipient tensor")
    if base._nonrecipient_manifest(pilot_last, torch=torch) != root_nonrecipient:
        raise ValueError("source pilot last changed a frozen non-recipient tensor")
    _validate_checkpoint_labels(pilot_best, root_best, description="source pilot best")
    _validate_checkpoint_labels(pilot_last, root_best, description="source pilot last")

    recipient_records = int(expected_candidate_records["recipient_field"])
    observed = _recipient_observed(pilot_summary, expected_records=recipient_records)
    if float(observed["best_exact"]) >= base.CONTINUATION_RECIPIENT_FLOOR:
        raise ValueError("source bootstrap already reached the 75-percent absolute continuation gate")
    if float(observed["epoch4_to_8_gain"]) < base.CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR:
        raise ValueError("source bootstrap did not prove the required epoch-4-to-8 learning trend")
    _assert_expected_031004_source_identity(
        input_contract_sha256=input_contract_sha256,
        candidate_denominators=expected_candidate_records,
        observed=observed,
    )
    decision_descriptor = _source_decision_descriptor(
        decision_path=source_decision,
        source_root=source_root,
        input_contract=input_contract_path,
        candidate_denominators=expected_candidate_records,
        observed=observed,
    )
    closure = {
        "source_bootstrap_root": str(source_root),
        "source_input_contract": _file_descriptor(
            input_contract_path, "source bootstrap input contract"
        ),
        "source_decision": decision_descriptor,
        "baseline_learning_rate_proof": baseline_lr_proof,
        "random_root": {
            "output": str(root_output),
            "summary": _file_descriptor(root_summary_path, "source random-root summary"),
            "best_checkpoint": _file_descriptor(root_best_path, "source random-root best"),
            "last_checkpoint": _file_descriptor(root_last_path, "source random-root last"),
            "nonrecipient_tensor_manifest": root_nonrecipient,
        },
        "failed_lr1e4_pilot": {
            "output": str(pilot_output),
            "summary": _file_descriptor(pilot_summary_path, "source pilot summary"),
            "best_checkpoint": _file_descriptor(pilot_best_path, "source pilot best"),
            "last_checkpoint": _file_descriptor(pilot_last_path, "source pilot last"),
            "observed": observed,
        },
        "training_validation_candidate_denominators": expected_candidate_records,
        "candidate_denominator_policy": (
            "v12_training_validation_reconstructed_from_bound_blind_manifest_and_snapshot_v1"
        ),
        "rescue_precondition": {
            "best_below_75_percent": True,
            "epoch4_to_8_gain_at_least_2pp": True,
        },
    }
    return closure, input_contract, pilot_summary


def _validate_runner_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    required = (
        "$baselineLearningRate = 0.0001",
        "$candidateLearningRate = 0.0003",
        '"--learning-rate", "$candidateLearningRate"',
        '"--recipient-only-fine-tune"',
        '"--recipient-train-splits", "train"',
        '"--validation-every", "1"',
        '"--init-checkpoint-mode", "strict"',
        '"--epochs", "$pilotEpochs"',
        '"--seed", "$seed"',
        '$rootCheckpoint = Join-Path $SourceBootstrapRoot "random-root-1e\\best.pt"',
        '$pilotOutput = Join-Path $OutputRoot "strict-recipient-lr3e4-8e"',
        '$blindRecords = [IO.Path]::GetFullPath([string]$sourceContract.blind_manifest)',
        '$snapshotRoot = [IO.Path]::GetFullPath([string]$sourceContract.snapshot_dataset_root)',
        '$sourceRecoveryDecision = Join-Path $SourceBootstrapRoot "analysis-decision.recovered.json"',
        '"--source-decision", $SourceDecision',
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise ValueError("LR A/B runner does not contain its fixed contract: " + ", ".join(missing))
    if "--onnx-output" in source:
        raise ValueError("LR A/B runner must not contain an ONNX export argument")
    for variable in (
        "candidateLearningRate",
        "pilotEpochs",
        "seed",
        "rootCheckpoint",
        "pilotOutput",
        "blindRecords",
        "snapshotRoot",
    ):
        if len(re.findall(rf"(?m)^\s*\${re.escape(variable)}\s*=", source)) != 1:
            raise ValueError(f"LR A/B runner must assign ${variable} exactly once")


def check_source_preflight(
    *,
    source_bootstrap_root: Path,
    source_decision: Path,
    runner: Path,
) -> dict[str, object]:
    """Read-only proof that CheckOnly has the real recoverable 031004 source."""

    runner = base._require_file(runner, "LR A/B PowerShell runner")
    _validate_runner_source(runner)
    closure, _source_input, _source_pilot_summary = _validate_source_closure(
        source_bootstrap_root,
        source_decision=source_decision,
    )
    baseline_proof = closure.get("baseline_learning_rate_proof")
    failed = closure.get("failed_lr1e4_pilot")
    if not isinstance(baseline_proof, Mapping) or not isinstance(failed, Mapping):
        raise ValueError("source closure is incomplete for LR A/B preflight")
    actual_argv_evidence = _actual_train_argv_evidence(
        baseline_proof=baseline_proof,
        candidate_runner=runner,
    )
    observed = failed.get("observed")
    if not isinstance(observed, Mapping):
        raise ValueError("source closure has no fixed 031004 recipient observation")
    return {
        "source_bootstrap_root": closure["source_bootstrap_root"],
        "source_decision": closure["source_decision"],
        "candidate_denominators": closure[
            "training_validation_candidate_denominators"
        ],
        "source_observed": dict(observed),
        "actual_train_argv_evidence": actual_argv_evidence,
    }


def prepare_input_contract(
    *,
    source_bootstrap_root: Path,
    output_root: Path,
    runner: Path,
    verifier: Path,
    source_decision: Path | None = None,
) -> dict[str, object]:
    """Verify the old closure and atomically bind a fresh LR-only experiment."""

    source_bootstrap_root = base._require_directory(
        source_bootstrap_root, "source bootstrap root"
    )
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    base._require_no_reparse(output_root, include_leaf=False)
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse LR A/B output: {output_root}")
    if not output_root.parent.is_dir():
        raise ValueError("LR A/B output parent must already exist")
    if output_root == source_bootstrap_root or output_root in source_bootstrap_root.parents or source_bootstrap_root in output_root.parents:
        raise ValueError("LR A/B output must not overlap the immutable source bootstrap root")
    runner = base._require_file(runner, "LR A/B PowerShell runner")
    verifier = base._require_file(verifier, "LR A/B verifier")
    target_parser = base._require_file(
        Path(str(target_rules.__file__)), "amount target parser"
    )
    _validate_runner_source(runner)
    closure, source_input, source_pilot_summary = _validate_source_closure(
        source_bootstrap_root,
        source_decision=source_decision,
    )
    baseline_proof = closure.get("baseline_learning_rate_proof")
    if not isinstance(baseline_proof, Mapping):
        raise ValueError("source closure has no baseline train argv proof")
    actual_argv_evidence = _actual_train_argv_evidence(
        baseline_proof=baseline_proof,
        candidate_runner=runner,
    )
    difference = _recipe_difference()
    if difference != {
        "learning_rate": {
            "baseline": BASELINE_LEARNING_RATE,
            "candidate": CANDIDATE_LEARNING_RATE,
        }
    }:
        raise AssertionError("LR A/B recipe changed more than learning_rate")
    output_root.mkdir(parents=False, exist_ok=False)
    if base._is_reparse(output_root):
        raise ValueError("fresh LR A/B output unexpectedly became a reparse point")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": INPUT_KIND,
        "analysis_only": True,
        "branch_source_only": True,
        "production_route_authorized": False,
        "onnx_export_authorized": False,
        "source_closure": closure,
        "source_dataset_binding": source_input["dataset_binding"],
        "blind_manifest": source_input["blind_manifest"],
        "blind_manifest_sha256": source_input["blind_manifest_sha256"],
        "snapshot_dataset_root": source_input["snapshot_dataset_root"],
        "fixed_topology": source_input["fixed_topology"],
        "delivery_floors_unchanged": source_input["delivery_floors_unchanged"],
        "analysis_continuation_gates": source_input["analysis_continuation_gates"],
        "optimizer_supervision_splits": ["train"],
        "checkpoint_selection_splits": ["val"],
        "test_rows_physically_present_in_training_manifest": False,
        "test_labels_used_by_training": False,
        "test_metrics_computed": False,
        "baseline_recipe": FIXED_BASELINE_RECIPE,
        "candidate_recipe": FIXED_CANDIDATE_RECIPE,
        "recipe_difference": difference,
        "actual_train_argv_evidence": actual_argv_evidence,
        "fresh_start": {
            "checkpoint": closure["random_root"]["best_checkpoint"],
            "failed_pilot_checkpoint_reused": False,
            "optimizer_restored": False,
            "epoch_reset": True,
        },
        "persisted_source_recipe": {
            key: source_pilot_summary.get(key) for key in _PERSISTED_RECIPE_KEYS
        },
        "code_inputs": {
            "runner": _file_descriptor(runner, "LR A/B runner", read_only=False),
            "verifier": _file_descriptor(verifier, "LR A/B verifier", read_only=False),
            "training_target_parser": _file_descriptor(
                target_parser, "amount target parser", read_only=False
            ),
        },
        "publication_policy": PUBLICATION_POLICY,
    }
    contract = output_root / INPUT_CONTRACT_NAME
    _atomic_write_json_no_clobber(contract, payload)
    return payload


def _verify_code_inputs(contract: Mapping[str, object]) -> Path:
    code = contract.get("code_inputs")
    if not isinstance(code, Mapping) or set(code) != {
        "runner",
        "verifier",
        "training_target_parser",
    }:
        raise ValueError("LR A/B code bindings are incomplete")
    runner = _verify_file_descriptor(code["runner"], "LR A/B runner")
    _verify_file_descriptor(code["verifier"], "LR A/B verifier")
    _verify_file_descriptor(code["training_target_parser"], "amount target parser")
    _validate_runner_source(runner)
    return runner


def _verify_source_closure_from_contract(
    value: object,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("LR A/B source closure is missing")
    source_root = base._require_directory(
        Path(str(value.get("source_bootstrap_root", ""))), "source bootstrap root"
    )
    source_input_path = _verify_file_descriptor(
        value.get("source_input_contract"), "source bootstrap input contract"
    )
    source_input = base._json_load(source_input_path)
    base._verify_bound_inputs(source_input, source_input_path)
    random_root = value.get("random_root")
    failed = value.get("failed_lr1e4_pilot")
    if not isinstance(random_root, Mapping) or not isinstance(failed, Mapping):
        raise ValueError("LR A/B source closure stages are missing")
    for name, description in (
        ("summary", "source random-root summary"),
        ("best_checkpoint", "source random-root best"),
        ("last_checkpoint", "source random-root last"),
    ):
        _verify_file_descriptor(random_root.get(name), description)
    for name, description in (
        ("summary", "source pilot summary"),
        ("best_checkpoint", "source pilot best"),
        ("last_checkpoint", "source pilot last"),
    ):
        _verify_file_descriptor(failed.get(name), description)
    source_decision = value.get("source_decision")
    if (
        not isinstance(source_decision, Mapping)
        or source_decision.get("used") is not True
        or source_decision.get("kind") != SOURCE_RECOVERY_DECISION_KIND
        or source_decision.get("continuation_16_epoch_authorized") is not False
    ):
        raise ValueError("LR A/B source decision binding is invalid")
    bound_decision = _verify_file_descriptor(
        source_decision.get("evidence"), "source recovery decision"
    )
    reconstructed, reconstructed_input, source_pilot = _validate_source_closure(
        source_root,
        source_decision=bound_decision,
    )
    # A decision may have been explicitly supplied at prepare time under a
    # non-default filename.  Reconstructed stage/crop evidence must still be
    # identical; its optional decision descriptor is verified separately.
    for key in (
        "source_bootstrap_root",
        "source_input_contract",
        "baseline_learning_rate_proof",
        "random_root",
        "failed_lr1e4_pilot",
        "training_validation_candidate_denominators",
        "candidate_denominator_policy",
        "rescue_precondition",
        "source_decision",
    ):
        if reconstructed.get(key) != value.get(key):
            raise ValueError(f"source bootstrap closure changed in {key}")
    return reconstructed_input, source_pilot, value


def build_lr_ab_decision(
    *,
    source_observed: Mapping[str, object],
    candidate_observed: Mapping[str, object],
) -> dict[str, object]:
    source_best = _finite_rate(source_observed.get("best_exact"), "source best recipient")
    source_gain = float(source_observed.get("epoch4_to_8_gain", math.nan))
    if source_best >= base.CONTINUATION_RECIPIENT_FLOOR:
        raise ValueError("LR rescue source no longer proves an absolute-floor miss")
    if not math.isfinite(source_gain) or source_gain < base.CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR:
        raise ValueError("LR rescue source no longer proves a positive continuation trend")
    candidate_best = _finite_rate(
        candidate_observed.get("best_exact"), "candidate best recipient"
    )
    candidate_gain = float(candidate_observed.get("epoch4_to_8_gain", math.nan))
    if not math.isfinite(candidate_gain):
        raise ValueError("candidate epoch-4-to-8 gain must be finite")
    continuation = (
        candidate_best >= base.CONTINUATION_RECIPIENT_FLOOR
        and candidate_gain >= base.CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR
    )
    return {
        "analysis_only": True,
        "branch_source_only": True,
        "production_route_authorized": False,
        "onnx_delivery_authorized": False,
        "delivery_gate_evaluated": False,
        "financial_delivery_checkpoint_eligible": False,
        "nonrecipient_metrics_authoritative_for_delivery": False,
        "delivery_floor_parameters": dict(base.DELIVERY_FLOORS),
        "analysis_continuation_gates": {
            "minimum_best_recipient_exact": base.CONTINUATION_RECIPIENT_FLOOR,
            "minimum_epoch4_to_8_gain": base.CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR,
        },
        "source_observed": dict(source_observed),
        "candidate_observed": dict(candidate_observed),
        "candidate_best_delta": candidate_best - source_best,
        "recipient_delivery_target_reached": candidate_best >= base.DELIVERY_FLOORS["recipient_field"],
        "continuation_16_epoch_authorized": continuation,
        "decision": (
            "analysis_only_lr_rescue_pass_continue_under_separate_contract"
            if continuation
            else "analysis_only_stop_lr_rescue_failed_fixed_gates"
        ),
    }


def finalize(
    *,
    input_contract: Path,
    candidate_output: Path,
    output: Path,
) -> dict[str, object]:
    """Verify the LR-only candidate and atomically publish its decision."""

    contract_path = base._require_read_only_file(input_contract, "LR A/B input contract")
    contract = base._json_load(contract_path)
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != INPUT_KIND
        or contract.get("analysis_only") is not True
        or contract.get("production_route_authorized") is not False
        or contract.get("onnx_export_authorized") is not False
    ):
        raise ValueError("LR A/B input contract is invalid")
    if contract.get("fixed_topology") != base.FIXED_TOPOLOGY:
        raise ValueError("LR A/B topology differs from the source bootstrap")
    if contract.get("delivery_floors_unchanged") != base.DELIVERY_FLOORS:
        raise ValueError("LR A/B changed a delivery floor")
    if contract.get("analysis_continuation_gates") != {
        "minimum_best_recipient_exact": base.CONTINUATION_RECIPIENT_FLOOR,
        "minimum_epoch4_to_8_gain": base.CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR,
    }:
        raise ValueError("LR A/B changed a continuation gate")
    if (
        contract.get("baseline_recipe") != FIXED_BASELINE_RECIPE
        or contract.get("candidate_recipe") != FIXED_CANDIDATE_RECIPE
        or contract.get("recipe_difference") != _recipe_difference()
        or set(_recipe_difference()) != {"learning_rate"}
    ):
        raise ValueError("LR A/B contract changed more than learning_rate")
    runner = _verify_code_inputs(contract)
    source_input, source_pilot_summary, source_closure = _verify_source_closure_from_contract(
        contract.get("source_closure")
    )
    baseline_proof = source_closure.get("baseline_learning_rate_proof")
    if not isinstance(baseline_proof, Mapping):
        raise ValueError("source closure has no baseline train argv proof")
    actual_argv_evidence = _actual_train_argv_evidence(
        baseline_proof=baseline_proof,
        candidate_runner=runner,
    )
    if contract.get("actual_train_argv_evidence") != actual_argv_evidence:
        raise ValueError("LR A/B actual train argv evidence changed after input binding")
    if contract.get("publication_policy") != PUBLICATION_POLICY:
        raise ValueError("LR A/B contract publication policy is invalid")
    if contract.get("source_dataset_binding") != source_input.get("dataset_binding"):
        raise ValueError("LR A/B dataset binding differs from the source snapshot")
    if (
        contract.get("blind_manifest") != source_input.get("blind_manifest")
        or contract.get("blind_manifest_sha256") != source_input.get("blind_manifest_sha256")
        or contract.get("snapshot_dataset_root") != source_input.get("snapshot_dataset_root")
        or contract.get("optimizer_supervision_splits") != ["train"]
        or contract.get("checkpoint_selection_splits") != ["val"]
        or contract.get("test_rows_physically_present_in_training_manifest") is not False
        or contract.get("test_labels_used_by_training") is not False
        or contract.get("test_metrics_computed") is not False
    ):
        raise ValueError("LR A/B data isolation contract is invalid")

    candidate_output = base._require_directory(candidate_output, "LR A/B candidate output")
    output_root = contract_path.parent
    try:
        candidate_output.relative_to(output_root)
    except ValueError:
        raise ValueError("LR A/B candidate output escapes its fresh root") from None
    if candidate_output.name != CANDIDATE_OUTPUT_NAME:
        raise ValueError("LR A/B candidate output has an unexpected stage name")
    base._validate_output_tree(output_root)
    output = Path(os.path.abspath(os.fspath(output)))
    base._require_no_reparse(output, include_leaf=False)
    try:
        output.relative_to(output_root)
    except ValueError:
        raise ValueError("LR A/B decision output escapes its fresh root") from None
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite LR A/B decision: {output}")

    summary_path = base._require_read_only_file(
        candidate_output / "training_summary.json", "LR A/B training summary"
    )
    best_path = base._require_read_only_file(candidate_output / "best.pt", "LR A/B best checkpoint")
    last_path = base._require_read_only_file(candidate_output / "last.pt", "LR A/B last checkpoint")
    summary = base._training_json_load(summary_path)
    dataset_binding = source_input.get("dataset_binding")
    assert isinstance(dataset_binding, Mapping)
    expected_field_counts = dataset_binding.get("field_counts")
    if not isinstance(expected_field_counts, Mapping):
        raise ValueError("source dataset binding has no field counts")
    candidate_records = base._validate_common_summary(
        summary,
        epochs=PILOT_EPOCHS,
        fine_tune_mode="recipient_only_v12",
        expected_field_counts=expected_field_counts,
    )
    expected_candidate_records = training_validation_candidate_denominators_v12(
        blind_manifest=Path(str(source_input["blind_manifest"])),
        snapshot_dataset_root=Path(str(source_input["snapshot_dataset_root"])),
        config_value=summary.get("config"),
    )
    if expected_candidate_records != source_closure.get(
        "training_validation_candidate_denominators"
    ):
        raise ValueError("LR A/B candidate denominators differ from the bound source reconstruction")
    persisted = contract.get("persisted_source_recipe")
    if not isinstance(persisted, Mapping):
        raise ValueError("LR A/B contract has no persisted source recipe")
    for key in _PERSISTED_RECIPE_KEYS:
        if summary.get(key) != persisted.get(key) or persisted.get(key) != source_pilot_summary.get(key):
            raise ValueError(f"LR A/B changed persisted recipe field {key}")
    runtime = summary.get("training_runtime")
    source_runtime = source_pilot_summary.get("training_runtime")
    runtime_keys = (
        "device",
        "uses_cuda",
        "torch_version",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "train_progress_every",
        "cuda_tf32_requested",
        "cudnn_benchmark_requested",
        "recipient_only_private_branch_training",
        "status_text_only_training",
        "recipient_train_split_policy",
        "full_validation_schedule",
        "validation_every",
        "cuda_device_name",
    )
    if not isinstance(runtime, Mapping) or not isinstance(source_runtime, Mapping):
        raise ValueError("LR A/B runtime evidence is missing")
    if any(runtime.get(key) != source_runtime.get(key) for key in runtime_keys):
        raise ValueError("LR A/B runtime differs from the source recipe")
    fine_tune = summary.get("fine_tune_policy")
    if (
        not isinstance(fine_tune, Mapping)
        or fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("training_forward") != "private_recipient_branch_only_v12"
        or fine_tune.get("open_text_legacy_recipient_unfrozen") is not False
    ):
        raise ValueError("LR A/B candidate is not strict recipient-only training")

    random_root = source_closure.get("random_root")
    failed_source = source_closure.get("failed_lr1e4_pilot")
    if not isinstance(random_root, Mapping) or not isinstance(failed_source, Mapping):
        raise ValueError("LR A/B source stage binding is invalid")
    root_best_path = _verify_file_descriptor(
        random_root.get("best_checkpoint"), "source random-root best"
    )
    failed_paths = {
        str(_verify_file_descriptor(failed_source.get(name), f"failed pilot {name}"))
        for name in ("best_checkpoint", "last_checkpoint")
    }
    initialization = summary.get("initialization")
    root_sha = base._sha256(root_best_path)
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("mode") != "parameter_only"
        or not base._same_path(initialization.get("checkpoint_path"), root_best_path)
        or initialization.get("checkpoint_sha256") != root_sha
        or initialization.get("optimizer_restored") is not False
        or initialization.get("epoch_reset") is not True
        or str(root_best_path) in failed_paths
    ):
        raise ValueError("LR A/B did not freshly restart from the original random-root best")

    root_payload, torch = base._torch_load(root_best_path)
    best_payload, _ = base._torch_load(best_path)
    last_payload, _ = base._torch_load(last_path)
    for payload in (root_payload, best_payload, last_payload):
        base._validate_checkpoint_common(payload)
    if best_payload.get("config") != root_payload.get("config") or last_payload.get("config") != root_payload.get("config"):
        raise ValueError("LR A/B checkpoint topology differs from random root")
    best_epoch = summary.get("best_checkpoint_epoch")
    if best_payload.get("epoch") != best_epoch or last_payload.get("epoch") != PILOT_EPOCHS:
        raise ValueError("LR A/B checkpoint epochs do not match summary")
    if best_payload.get("initialization") != initialization or last_payload.get("initialization") != initialization:
        raise ValueError("LR A/B checkpoint initialization differs from summary")
    _checkpoint_metrics_match(
        best_payload,
        candidate_records[int(best_epoch) - 1],
        expected_candidate_records=expected_candidate_records,
        description="LR A/B best",
    )
    _checkpoint_metrics_match(
        last_payload,
        candidate_records[-1],
        expected_candidate_records=expected_candidate_records,
        description="LR A/B last",
    )
    root_nonrecipient = base._nonrecipient_manifest(root_payload, torch=torch)
    if root_nonrecipient != random_root.get("nonrecipient_tensor_manifest"):
        raise ValueError("source random-root non-recipient manifest changed")
    if base._nonrecipient_manifest(best_payload, torch=torch) != root_nonrecipient:
        raise ValueError("LR A/B best changed a frozen non-recipient tensor")
    if base._nonrecipient_manifest(last_payload, torch=torch) != root_nonrecipient:
        raise ValueError("LR A/B last changed a frozen non-recipient tensor")
    _validate_checkpoint_labels(best_payload, root_payload, description="LR A/B best")
    _validate_checkpoint_labels(last_payload, root_payload, description="LR A/B last")
    if base._partition_manifest(last_payload, prefix="recipient_", torch=torch) == base._partition_manifest(
        root_payload, prefix="recipient_", torch=torch
    ):
        raise ValueError("LR A/B eight epochs did not change any recipient tensor")
    root_record = base._training_json_load(
        _verify_file_descriptor(random_root.get("summary"), "source random-root summary")
    )["records"][0]
    for field in _PROTECTED_FIELDS:
        expected = int(expected_candidate_records[field])
        frozen = _candidate_metric(root_record, field, expected_records=expected)
        for record in candidate_records:
            if not math.isclose(
                _candidate_metric(record, field, expected_records=expected),
                frozen,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(f"LR A/B changed frozen {field} validation output")

    recipient_records = int(expected_candidate_records["recipient_field"])
    candidate_observed = _recipient_observed(summary, expected_records=recipient_records)
    source_observed = failed_source.get("observed")
    if not isinstance(source_observed, Mapping):
        raise ValueError("LR A/B source observations are missing")
    decision = build_lr_ab_decision(
        source_observed=source_observed,
        candidate_observed=candidate_observed,
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": DECISION_KIND,
        **decision,
        "input_contract": str(contract_path),
        "input_contract_sha256": base._sha256(contract_path),
        "source_bootstrap_root": source_closure["source_bootstrap_root"],
        "source_input_contract_sha256": source_closure["source_input_contract"]["sha256"],
        "blind_manifest_sha256": source_input["blind_manifest_sha256"],
        "snapshot_dataset_binding": source_input["dataset_binding"],
        "recipe_difference": _recipe_difference(),
        "actual_train_argv_evidence": actual_argv_evidence,
        "fresh_start": {
            "checkpoint": str(root_best_path),
            "checkpoint_sha256": root_sha,
            "failed_pilot_checkpoint_reused": False,
            "optimizer_restored": False,
            "epoch_reset": True,
        },
        "candidate": {
            "output": str(candidate_output),
            "summary": _file_descriptor(summary_path, "LR A/B summary"),
            "best_checkpoint": _file_descriptor(best_path, "LR A/B best"),
            "last_checkpoint": _file_descriptor(last_path, "LR A/B last"),
            "nonrecipient_byte_identical_to_random_root": True,
            "test_rows_used": False,
            "onnx_exported": False,
        },
        "authorized_16_epoch_warmstart_checkpoint": (
            str(best_path) if decision["continuation_16_epoch_authorized"] else None
        ),
        "notice": (
            "ANALYSIS ONLY. The LR rescue can authorize only a separately bound 16-epoch analysis; "
            "it cannot authorize production, ONNX export, parser changes, or use of held-out test data."
        ),
        "publication_policy": PUBLICATION_POLICY,
    }
    _atomic_write_json_no_clobber(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind and verify the recipient random-root LR-only A/B")
    commands = parser.add_subparsers(dest="command", required=True)
    check_source = commands.add_parser("check-source")
    check_source.add_argument("--source-bootstrap-root", type=Path, required=True)
    check_source.add_argument("--source-decision", type=Path, required=True)
    check_source.add_argument("--runner", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-bootstrap-root", type=Path, required=True)
    prepare.add_argument("--source-decision", type=Path)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--runner", type=Path, required=True)
    prepare.add_argument("--verifier", type=Path, required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--input-contract", type=Path, required=True)
    finish.add_argument("--candidate-output", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "check-source":
        result = check_source_preflight(
            source_bootstrap_root=args.source_bootstrap_root,
            source_decision=args.source_decision,
            runner=args.runner,
        )
        observed = result["source_observed"]
        assert isinstance(observed, Mapping)
        print(
            "recipient_random_bootstrap_lr_ab_source_check "
            f"best={float(observed['best_exact']):.2%} "
            f"epoch4_to_8_gain={float(observed['epoch4_to_8_gain']):+.2%} "
            "canonical_recovery=True production=False"
        )
        return
    if args.command == "prepare":
        result = prepare_input_contract(
            source_bootstrap_root=args.source_bootstrap_root,
            source_decision=args.source_decision,
            output_root=args.output_root,
            runner=args.runner,
            verifier=args.verifier,
        )
        source = result["source_closure"]["failed_lr1e4_pilot"]["observed"]
        print(
            "recipient_random_bootstrap_lr_ab_input "
            f"source_best={float(source['best_exact']):.2%} "
            f"source_gain={float(source['epoch4_to_8_gain']):+.2%} "
            f"lr={BASELINE_LEARNING_RATE:g}->{CANDIDATE_LEARNING_RATE:g}"
        )
        return
    result = finalize(
        input_contract=args.input_contract,
        candidate_output=args.candidate_output,
        output=args.output,
    )
    observed = result["candidate_observed"]
    assert isinstance(observed, Mapping)
    print(
        "recipient_random_bootstrap_lr_ab_decision "
        f"best={float(observed['best_exact']):.2%} "
        f"epoch4_to_8_gain={float(observed['epoch4_to_8_gain']):+.2%} "
        f"continuation16={result['continuation_16_epoch_authorized']} "
        "production=False"
    )


if __name__ == "__main__":
    main()
