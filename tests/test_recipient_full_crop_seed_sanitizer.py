"""Contracts for the analysis-only v13 recipient seed sanitizer."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest

from transfer_receipt_ai.ocr_unified import (
    KIND_V12,
    KIND_V13,
    STATUS_CLASSES,
    STATUS_TEXT_BLANK_INDEX,
    STATUS_TEXT_CHARSET_SOURCE,
    STATUS_TEXT_RUNTIME_POLICY,
    STATUS_TEXT_TARGET,
    V12_ONNX_OUTPUT_NAMES,
    V6_TIME_CHARACTERS,
    V8_AMOUNT_CHARACTERS,
    UnifiedReaderConfig,
    _recipient_artifact_metadata,
    _recipient_train_split_policy,
    _validate_recipient_full_crop_seed_policy,
    build_unified_reader,
)
from transfer_receipt_ai.recipient_full_crop_seed_sanitizer import (
    ATTESTATION_KEY,
    ATTESTATION_KIND,
    DISCARDED_TRAIN_PAYMENT_POLICY,
    DISCARDED_TRAIN_PAYMENT_TENSOR_KEYS,
    _build_sanitized_payload,
    _canonical_sha256,
    _metadata_partitions,
    _partition_descriptor,
    _validated_status_v12_source_config,
    sanitize_recipient_full_crop_seed,
    validate_recipient_full_crop_seed_attestation,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON_WRAPPER = ROOT / "scripts" / "receipt-ocr-recipient-full-crop-seed-sanitize.py"
POWERSHELL_WRAPPER = ROOT / "scripts" / "receipt-ocr-recipient-full-crop-seed-sanitize.ps1"


def _torch():
    return pytest.importorskip("torch")


def _config(architecture: int) -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=architecture,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_value_left_trim=0.30,
        recipient_input_height=32,
        recipient_input_width=1536,
        recipient_branch_channels=8,
        recipient_open_text_layers=2,
        recipient_open_text_heads=4,
        recipient_open_text_feedforward=64,
        pooled_width=2,
    )


def _recipient_policies() -> dict[str, object]:
    return {
        "recipient_sampling_policy": {
            "mode": "uniform",
            "recipient_sampling_weight": 1.0,
            "recipient_train_records": 2,
            "train_records": 2,
        },
        "recipient_confidence_policy": {
            "mode": "none",
            "low_confidence_threshold": None,
            "low_confidence_loss_weight": 1.0,
            "curriculum_epochs": 0,
        },
        "recipient_tail_loss_policy": {
            "mode": "none",
            "rare_character_max_support": 0,
            "rare_character_loss_weight": 1.0,
            "long_text_min_length": 0,
            "long_text_loss_weight": 1.0,
            "recipient_train_records": 2,
            "rare_character_train_records": 0,
            "long_text_train_records": 0,
            "combined_boost_train_records": 0,
        },
        "recipient_train_augmentation_policy": {"mode": "none"},
    }


def _state(
    *,
    v13: bool,
    recipient_fill: float,
    shared_fill: float = 1.0,
    recipient_character_count: int = 2,
    payment_character_count: int = 4,
):
    _torch()
    config = _config(13 if v13 else 12)
    model = build_unified_reader(
        payment_vocab_size=payment_character_count + 1,
        config=config,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=recipient_character_count + 1,
        status_text_vocab_size=5 if v13 else None,
    )
    state = {}
    for index, (name, value) in enumerate(model.state_dict().items(), start=1):
        cloned = value.detach().clone()
        if name.startswith("recipient_"):
            cloned.fill_(recipient_fill + index / 1000.0)
        elif name.startswith("status_text_"):
            cloned.fill_(7.0 + index / 1000.0)
        else:
            cloned.fill_(shared_fill + index / 1000.0)
        state[name] = cloned
    return state


def _payload(
    *,
    architecture: int,
    train_only: bool,
    recipient_fill: float,
    recipient_characters: list[str] | None = None,
    payment_characters: list[str] | None = None,
    shared_fill: float = 1.0,
) -> dict[str, object]:
    config = _config(architecture)
    policies = _recipient_policies()
    recipient_characters = recipient_characters or ["商", "户"]
    payment_characters = payment_characters or ["卡", "行", "银", "储"]
    state = _state(
        v13=architecture == 13,
        recipient_fill=recipient_fill,
        shared_fill=shared_fill,
        recipient_character_count=len(recipient_characters),
        payment_character_count=len(payment_characters),
    )
    split_policy = _recipient_train_split_policy(["train"] if train_only else ["train", "val"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": KIND_V13 if architecture == 13 else KIND_V12,
        "config": asdict(config),
        "state_dict": state,
        "amount_characters": list(V8_AMOUNT_CHARACTERS),
        "time_characters": list(V6_TIME_CHARACTERS),
        "payment_characters": payment_characters,
        "status_classes": list(STATUS_CLASSES),
        "payment_bank_prefix_classes": ["__other__", "银行"],
        "recipient_characters": recipient_characters,
        "recipient_blank_index": 0,
        "recipient_charset_sha256": hashlib.sha256(
            "".join(recipient_characters).encode("utf-8")
        ).hexdigest(),
        "recipient_charset_source": "train_only_anchored_recipient_value",
        "recipient_target": "anchored_recipient_value_with_dedicated_high_resolution_value_view",
        "recipient_oov_by_split": {
            split: {"records": 2, "oov_records": 0}
            for split in ("train", "val", "test")
        },
        "recipient_train_split_policy": split_policy,
        "recipient_loss_weight": 1.0,
        # ocr_unified persists this CLI scalar in every checkpoint, including
        # v12 where no status-text head exists.
        "status_text_loss_weight": 1.0,
        "epoch": 4,
        "metrics": {"source": "status" if architecture == 13 else "recipient"},
    }
    payload.update(policies)
    payload.update(
        _recipient_artifact_metadata(
            config,
            recipient_sampling_policy=policies["recipient_sampling_policy"],
            recipient_confidence_policy=policies["recipient_confidence_policy"],
            recipient_tail_loss_policy=policies["recipient_tail_loss_policy"],
            recipient_train_augmentation_policy=policies[
                "recipient_train_augmentation_policy"
            ],
        )
    )
    if architecture == 13:
        status_characters = sorted(set("转账成功"))
        status_keys = [key for key in state if key.startswith("status_text_")]
        source_config = asdict(config)
        source_config["architecture_version"] = 12
        # The accepted status-only v13 artifact was created before these two
        # byte-compatible defaults existed in the serialized v12 config.
        del source_config["recipient_backbone"]
        del source_config["recipient_open_text_dropout"]
        payload.update(
            {
                "status_text_characters": status_characters,
                "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
                "status_text_charset_sha256": hashlib.sha256(
                    "".join(status_characters).encode("utf-8")
                ).hexdigest(),
                "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
                "status_text_target": STATUS_TEXT_TARGET,
                "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
                "status_text_oov_by_split": {
                    split: {
                        "records": 2,
                        "oov_records": 0,
                        "oov_characters": 0,
                        "examples": [],
                    }
                    for split in ("train", "val", "test")
                },
                "initialization": {
                    "mode": "parameter_only_v12_to_v13_status_text_expansion",
                    "source_kind": KIND_V12,
                    "source_config": source_config,
                    "checkpoint_sha256": "a" * 64,
                    "optimizer_restored": False,
                    "epoch_reset": True,
                    "new_parameter_prefix": "status_text_",
                    "copied_legacy_tensor_count": len(state) - len(status_keys),
                    "new_status_text_tensor_count": len(status_keys),
                    "frozen_legacy_output_count": len(V12_ONNX_OUTPUT_NAMES),
                    "financial_label_policy": {
                        "mode": "checkpoint_legacy_label_maps_status_text_only_v1"
                    },
                },
                "fine_tune_policy": {
                    "mode": "status_text_only_v13",
                    "trainable_parameter_prefix": "status_text_",
                    "frozen_legacy_output_count": len(V12_ONNX_OUTPUT_NAMES),
                    "full_validation_schedule": "epoch_1_every_n_and_final_epoch",
                    "validation_every": 4,
                },
                "training_runtime": {
                    "status_text_only_training": True,
                    "recipient_only_private_branch_training": False,
                    "full_validation_schedule": "epoch_1_every_n_and_final_epoch",
                    "validation_every": 4,
                    "recipient_train_split_policy": split_policy,
                },
            }
        )
    else:
        payload["training_runtime"] = {
            "recipient_train_split_policy": split_policy,
        }
        payload["initialization"] = {
            "mode": "random",
            "optimizer_restored": False,
            "epoch_reset": True,
        }
    return payload


def _source(path: str, kind: str) -> dict[str, object]:
    return {
        "path": path,
        "sha256": ("b" if kind == KIND_V13 else "c") * 64,
        "size_bytes": 123,
        "kind": kind,
        "epoch": 4,
    }


def _lineage(payload: dict[str, object], source: dict[str, object]) -> dict[str, object]:
    return {
        "policy": "hash_bound_recursive_train_only_v12_to_random_v1",
        "checkpoint_count": 1,
        "root_initialization_mode": "random",
        "entries": [
            {
                "checkpoint": source,
                "config_sha256": _canonical_sha256(
                    payload["config"], description="test lineage config"
                ),
                "recipient_charset_sha256": payload["recipient_charset_sha256"],
                "recipient_train_split_policy_sha256": _canonical_sha256(
                    payload["recipient_train_split_policy"],
                    description="test lineage policy",
                ),
                "initialization_mode": "random",
                "parent_checkpoint_path": None,
                "parent_checkpoint_sha256": None,
                "parent_config_sha256": None,
                "parent_epoch": None,
                "recipient_state": _partition_descriptor(
                    payload["state_dict"], recipient=True
                ),
            }
        ],
    }


def _sanitized_payload() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    status_source = _source("/checkpoints/status.pt", KIND_V13)
    train_source = _source("/checkpoints/train-only.pt", KIND_V12)
    output = _build_sanitized_payload(
        status_payload=status,
        train_payload=recipient,
        status_source=status_source,
        train_source=train_source,
        train_lineage=_lineage(recipient, train_source),
    )
    return status, recipient, output


def _reseal_attestation(output: dict[str, object]) -> None:
    attestation = output[ATTESTATION_KEY]
    # Tests that intentionally mutate the embedded lineage need to keep every
    # public derivative hash coherent; otherwise validation correctly stops at
    # the outer dual-source metadata binding before exercising the narrower
    # lineage invariant under test.
    output["initialization"]["train_only_recipient_lineage_sha256"] = (
        _canonical_sha256(
            attestation["train_only_recipient_lineage"],
            description="test resealed initialization lineage",
        )
    )
    _, non_recipient_metadata = _metadata_partitions(output)
    attestation["metadata_proof"]["output_non_recipient_sha256"] = (
        _canonical_sha256(
            non_recipient_metadata,
            description="test resealed non-recipient metadata",
        )
    )
    attestation["metadata_proof"]["operative_metadata_sha256"] = (
        _canonical_sha256(
            {
                key: output[key]
                for key in ("initialization", "training_runtime", "metrics")
            },
            description="test resealed operative metadata",
        )
    )
    unsigned = {
        key: value for key, value in attestation.items() if key != "integrity_sha256"
    }
    attestation["integrity_sha256"] = _canonical_sha256(
        unsigned, description="test resealed attestation"
    )


def test_wrapper_declares_strict_analysis_only_sources_and_no_clobber() -> None:
    python_source = PYTHON_WRAPPER.read_text(encoding="utf-8")
    powershell = POWERSHELL_WRAPPER.read_text(encoding="utf-8")
    assert "recipient_full_crop_seed_sanitizer import main" in python_source
    for token in (
        "[string]$StatusCheckpoint",
        "[string]$TrainOnlyRecipientCheckpoint",
        "[string]$OutputCheckpoint",
        "analysis-only; production authorization remains false",
        "old status initialization/runtime/metrics: non-operative history only",
        "every v12 warmstart ancestor: recorded path/hash/config/epoch",
        "full-crop warmstart reopens A/B and every B ancestor",
        "no optimizer restore; no manifest or held-out data lookup",
        "Refusing to overwrite an existing sanitizer output",
        "ReparsePoint",
        "--status-checkpoint",
        "--train-only-recipient-checkpoint",
        "--output-checkpoint",
    ):
        assert token in powershell


def test_sanitizer_preserves_status_side_and_replaces_only_recipient_side() -> None:
    torch = _torch()
    status, recipient, output = _sanitized_payload()
    validate_recipient_full_crop_seed_attestation(output)
    assert output[ATTESTATION_KEY]["kind"] == ATTESTATION_KIND
    assert output[ATTESTATION_KEY]["analysis_only"] is True
    assert output[ATTESTATION_KEY]["production_route_authorized"] is False
    for name, value in output["state_dict"].items():
        expected = recipient["state_dict"][name] if name.startswith("recipient_") else status["state_dict"][name]
        assert torch.equal(value, expected)
    status_nonrecipient = {
        key: value
        for key, value in status.items()
        if key != "state_dict" and not key.startswith("recipient_")
    }
    output_nonrecipient = {
        key: value
        for key, value in output.items()
        if key not in {"state_dict", ATTESTATION_KEY} and not key.startswith("recipient_")
    }
    for key, value in status_nonrecipient.items():
        if key not in {"initialization", "training_runtime", "metrics"}:
            assert output_nonrecipient[key] == value
    history = output["status_source_history"]
    assert history["operative_recipient_claim"] is False
    assert history["values"] == {
        key: status[key]
        for key in ("initialization", "training_runtime", "metrics")
        if key in status
    }
    assert output["initialization"]["mode"] == (
        "analysis_only_full_crop_seed_sanitizer_dual_source_v1"
    )
    assert output["training_runtime"]["recipient_train_split_policy"] == (
        _recipient_train_split_policy(["train"])
    )
    assert output["metrics"] == {
        "mode": "seed_sanitization_only_v1",
        "training_metrics_carried_forward": False,
    }
    assert output["recipient_train_split_policy"] == _recipient_train_split_policy(["train"])
    assert output["recipient_sampling_policy"] == recipient["recipient_sampling_policy"]


def test_sanitizer_discards_train_payment_mismatch_and_attests_exact_four_heads() -> None:
    torch = _torch()
    status_payment = ["卡", "行", "银", "储", "蓄"]
    train_payment = ["卡", "行", "银", "储"]
    status = _payload(
        architecture=13,
        train_only=False,
        recipient_fill=8.0,
        payment_characters=status_payment,
    )
    recipient = _payload(
        architecture=12,
        train_only=True,
        recipient_fill=3.0,
        payment_characters=train_payment,
    )
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    output = _build_sanitized_payload(
        status_payload=status,
        train_payload=recipient,
        status_source=status_source,
        train_source=train_source,
        train_lineage=_lineage(recipient, train_source),
    )
    validate_recipient_full_crop_seed_attestation(output)
    assert output["payment_characters"] == status_payment
    for name in DISCARDED_TRAIN_PAYMENT_TENSOR_KEYS:
        assert torch.equal(output["state_dict"][name], status["state_dict"][name])
    proof = output[ATTESTATION_KEY]["compatibility"]["discarded_train_payment"]
    assert proof == {
        "policy": DISCARDED_TRAIN_PAYMENT_POLICY,
        "status_character_count": len(status_payment),
        "train_character_count": len(train_payment),
        "status_charset_sha256": hashlib.sha256(
            "".join(status_payment).encode("utf-8")
        ).hexdigest(),
        "train_charset_sha256": hashlib.sha256(
            "".join(train_payment).encode("utf-8")
        ).hexdigest(),
        "label_maps_equal": False,
        "allowed_shape_difference_keys": sorted(DISCARDED_TRAIN_PAYMENT_TENSOR_KEYS),
        "observed_shape_difference_keys": sorted(DISCARDED_TRAIN_PAYMENT_TENSOR_KEYS),
    }


def test_sanitizer_rejects_nonpayment_label_mismatch_despite_discarded_payment() -> None:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    recipient["payment_bank_prefix_classes"] = ["__other__", "另一银行"]
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    with pytest.raises(ValueError, match="bank label maps do not match"):
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


def test_sanitizer_rejects_shape_drift_outside_discarded_payment_heads() -> None:
    torch = _torch()
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    name = "payment_ctc_sequence.weight_ih_l0"
    recipient["state_dict"][name] = torch.cat(
        (recipient["state_dict"][name], recipient["state_dict"][name][:1]), dim=0
    )
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    with pytest.raises(ValueError, match="shape/dtype does not match its declared model"):
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


def test_attestation_rejects_discarded_payment_proof_substitution() -> None:
    status = _payload(
        architecture=13,
        train_only=False,
        recipient_fill=8.0,
        payment_characters=["卡", "行", "银", "储", "蓄"],
    )
    recipient = _payload(
        architecture=12,
        train_only=True,
        recipient_fill=3.0,
        payment_characters=["卡", "行", "银", "储"],
    )
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    output = _build_sanitized_payload(
        status_payload=status,
        train_payload=recipient,
        status_source=status_source,
        train_source=train_source,
        train_lineage=_lineage(recipient, train_source),
    )
    proof = output[ATTESTATION_KEY]["compatibility"]["discarded_train_payment"]
    proof["observed_shape_difference_keys"] = sorted(
        DISCARDED_TRAIN_PAYMENT_TENSOR_KEYS
    )[:-1]
    _reseal_attestation(output)
    with pytest.raises(ValueError, match="discarded train payment proof"):
        validate_recipient_full_crop_seed_attestation(output)


@pytest.mark.parametrize("representation", ["missing", "null"])
def test_status_source_config_accepts_only_the_two_legacy_aliases(
    representation: str,
) -> None:
    expected = asdict(_config(13))
    expected["architecture_version"] = 12
    legacy = dict(expected)
    if representation == "missing":
        del legacy["recipient_backbone"]
        del legacy["recipient_open_text_dropout"]
    else:
        legacy["recipient_backbone"] = None
        legacy["recipient_open_text_dropout"] = None
    assert _validated_status_v12_source_config(legacy, expected=expected) == expected


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", "outside the two legacy aliases"),
        ("extra", "outside the two legacy aliases"),
        ("other_null", "outside the two legacy null aliases"),
        ("backbone", "outside the two legacy null aliases"),
        ("dropout", "outside the two legacy null aliases"),
        ("other_value", "outside the two legacy null aliases"),
    ],
)
def test_status_source_config_rejects_missing_extra_and_other_drift(
    mutation: str, message: str
) -> None:
    expected = asdict(_config(13))
    expected["architecture_version"] = 12
    observed = dict(expected)
    del observed["recipient_backbone"]
    del observed["recipient_open_text_dropout"]
    if mutation == "missing":
        del observed["pooled_width"]
    elif mutation == "extra":
        observed["unexpected"] = 1
    elif mutation == "other_null":
        observed["image_width"] = None
    elif mutation == "backbone":
        observed["recipient_backbone"] = "residual_positional_transformer_v2"
    elif mutation == "dropout":
        observed["recipient_open_text_dropout"] = 0.1
    elif mutation == "other_value":
        observed["pooled_width"] = int(observed["pooled_width"]) + 1
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        _validated_status_v12_source_config(observed, expected=expected)


def test_warmstart_rejects_top_level_policy_laundering_without_attestation() -> None:
    raw = _payload(architecture=13, train_only=True, recipient_fill=8.0)
    with pytest.raises(ValueError, match="content-bound seed sanitizer attestation"):
        _validate_recipient_full_crop_seed_policy(raw)


def test_sanitizer_rejects_transductive_nested_policy() -> None:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    recipient["training_runtime"]["recipient_train_split_policy"] = _recipient_train_split_policy(
        ["train", "val"]
    )
    with pytest.raises(ValueError, match="does not prove train-only"):
        status_source = _source("/status.pt", KIND_V13)
        train_source = _source("/recipient.pt", KIND_V12)
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


def test_sanitizer_rejects_transductive_source_policy_provenance() -> None:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    recipient["initialization"]["source_recipient_train_split_policy"] = (
        _recipient_train_split_policy(["train", "val"])
    )
    with pytest.raises(ValueError, match="does not prove train-only"):
        status_source = _source("/status.pt", KIND_V13)
        train_source = _source("/recipient.pt", KIND_V12)
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


@pytest.mark.parametrize(
    "value",
    [None, True, "1.0", 0.0, -1.0, 2.0, float("nan"), float("inf")],
)
def test_sanitizer_rejects_invalid_v12_passive_status_text_loss_weight(
    value: object,
) -> None:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    recipient["status_text_loss_weight"] = value
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    with pytest.raises(ValueError, match="fixed v12 passive status-text loss weight"):
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


def test_sanitizer_requires_real_v12_passive_status_text_loss_weight() -> None:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    del recipient["status_text_loss_weight"]
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    with pytest.raises(ValueError, match="missing the fixed v12 passive status-text loss weight"):
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


@pytest.mark.parametrize(
    "key",
    ["status_text_characters", "status_text_runtime_policy", "status_text_target"],
)
def test_sanitizer_rejects_active_status_text_metadata_in_v12(key: str) -> None:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    recipient[key] = "forbidden"
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    with pytest.raises(ValueError, match="active status-text metadata"):
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


def test_sanitizer_rejects_status_text_state_tensor_in_v12() -> None:
    torch = _torch()
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    recipient["state_dict"]["status_text_classifier.weight"] = torch.zeros((1, 1))
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    with pytest.raises(ValueError, match="unexpectedly contains status_text_ tensors"):
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


@pytest.mark.parametrize("mutation, message", [
    ("shape", "shape/dtype does not match"),
    ("key", "tensor keys do not match"),
    ("charset", "must be a subset"),
    ("config", "configs may differ only"),
])
def test_sanitizer_rejects_incompatible_sources(mutation: str, message: str) -> None:
    torch = _torch()
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    if mutation == "shape":
        recipient["state_dict"]["recipient_classifier.bias"] = torch.zeros(4)
    elif mutation == "key":
        del recipient["state_dict"]["recipient_classifier.bias"]
    elif mutation == "charset":
        status = _payload(
            architecture=13,
            train_only=False,
            recipient_fill=8.0,
            recipient_characters=["商", "新"],
        )
    elif mutation == "config":
        recipient["config"]["amount_format_min_confidence"] = 0.85
    with pytest.raises(ValueError, match=message):
        status_source = _source("/status.pt", KIND_V13)
        train_source = _source("/recipient.pt", KIND_V12)
        _build_sanitized_payload(
            status_payload=status,
            train_payload=recipient,
            status_source=status_source,
            train_source=train_source,
            train_lineage=_lineage(recipient, train_source),
        )


def test_sanitizer_accepts_smaller_train_only_charset_and_binds_classifier_to_b() -> None:
    status = _payload(
        architecture=13,
        train_only=False,
        recipient_fill=8.0,
        recipient_characters=["商", "户", "源"],
    )
    recipient = _payload(
        architecture=12,
        train_only=True,
        recipient_fill=3.0,
        recipient_characters=["商", "户"],
    )
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    output = _build_sanitized_payload(
        status_payload=status,
        train_payload=recipient,
        status_source=status_source,
        train_source=train_source,
        train_lineage=_lineage(recipient, train_source),
    )
    validate_recipient_full_crop_seed_attestation(output)
    assert output["recipient_characters"] == recipient["recipient_characters"]
    assert int(output["state_dict"]["recipient_classifier.weight"].shape[0]) == 3
    assert output[ATTESTATION_KEY]["compatibility"]["recipient_charset_relation"] == (
        "train_only_subset_of_status_source_v1"
    )


def test_status_transductive_provenance_is_history_not_an_operative_recipient_claim() -> None:
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    transductive = _recipient_train_split_policy(["train", "val"])
    status["initialization"]["source_recipient_train_split_policy"] = transductive
    status["metrics"]["recipient_train_split_policy"] = transductive
    recipient = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    status_source = _source("/status.pt", KIND_V13)
    train_source = _source("/recipient.pt", KIND_V12)
    output = _build_sanitized_payload(
        status_payload=status,
        train_payload=recipient,
        status_source=status_source,
        train_source=train_source,
        train_lineage=_lineage(recipient, train_source),
    )
    validate_recipient_full_crop_seed_attestation(output)
    history = output["status_source_history"]
    assert history["operative_recipient_claim"] is False
    assert history["values"]["initialization"][
        "source_recipient_train_split_policy"
    ] == transductive
    assert history["values"]["metrics"]["recipient_train_split_policy"] == transductive
    assert output["training_runtime"]["recipient_train_split_policy"] == (
        _recipient_train_split_policy(["train"])
    )


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("recipient_classifier.bias", "recipient lineage leaf state does not match"),
        ("status_text_classifier.bias", "state no longer matches"),
    ],
)
def test_attestation_detects_state_tampering(name: str, message: str) -> None:
    _, _, output = _sanitized_payload()
    output["state_dict"][name] = output["state_dict"][name].clone()
    output["state_dict"][name][0] += 1.0
    with pytest.raises(ValueError, match=message):
        validate_recipient_full_crop_seed_attestation(output)


def test_attestation_integrity_binds_source_provenance_hashes() -> None:
    _, _, output = _sanitized_payload()
    output[ATTESTATION_KEY]["status_checkpoint"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="integrity hash does not match"):
        validate_recipient_full_crop_seed_attestation(output)


def test_attestation_rejects_spliced_child_parent_lineage() -> None:
    _, _, output = _sanitized_payload()
    attestation = output[ATTESTATION_KEY]
    original_root = copy.deepcopy(attestation["train_only_recipient_lineage"]["entries"][0])
    child = copy.deepcopy(original_root)
    child["initialization_mode"] = "parameter_only_recipient_open_text_adapter"
    child["parent_checkpoint_path"] = "/checkpoints/not-the-root.pt"
    child["parent_checkpoint_sha256"] = "e" * 64
    child["parent_config_sha256"] = original_root["config_sha256"]
    child["parent_epoch"] = 4
    root = copy.deepcopy(original_root)
    root["checkpoint"] = {
        "path": "/checkpoints/root.pt",
        "sha256": "e" * 64,
        "size_bytes": 321,
        "kind": KIND_V12,
        "epoch": 4,
    }
    attestation["train_only_recipient_lineage"] = {
        "policy": "hash_bound_recursive_train_only_v12_to_random_v1",
        "checkpoint_count": 2,
        "root_initialization_mode": "random",
        "entries": [child, root],
    }
    _reseal_attestation(output)
    with pytest.raises(ValueError, match="does not bind parent"):
        validate_recipient_full_crop_seed_attestation(output)


@pytest.mark.parametrize(
    "field,message",
    [
        ("recipient_train_split_policy_sha256", "canonical train-only policy"),
        ("recipient_charset_sha256", "leaf charset"),
        ("recipient_state", "leaf state"),
    ],
)
def test_attestation_binds_lineage_leaf_to_transplanted_recipient(
    field: str, message: str
) -> None:
    _, _, output = _sanitized_payload()
    leaf = output[ATTESTATION_KEY]["train_only_recipient_lineage"]["entries"][0]
    if field == "recipient_state":
        leaf[field] = copy.deepcopy(
            output[ATTESTATION_KEY]["state_proof"]["output_non_recipient"]
        )
    else:
        leaf[field] = "d" * 64
    _reseal_attestation(output)
    with pytest.raises(ValueError, match=message):
        validate_recipient_full_crop_seed_attestation(output)


def test_file_sanitizer_rejects_train_only_leaf_with_transductive_ancestor(
    tmp_path: Path,
) -> None:
    torch = _torch()
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    root = _payload(architecture=12, train_only=False, recipient_fill=2.0)
    leaf = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    status_path = tmp_path / "status.pt"
    root_path = tmp_path / "transductive-root.pt"
    leaf_path = tmp_path / "claimed-train-only-leaf.pt"
    output_path = tmp_path / "must-not-exist.pt"
    torch.save(status, status_path)
    torch.save(root, root_path)
    leaf["initialization"] = {
        "mode": "parameter_only",
        "checkpoint_path": str(root_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(root_path.read_bytes()).hexdigest(),
        "source_kind": KIND_V12,
        "source_epoch": root["epoch"],
        "source_config": root["config"],
        "optimizer_restored": False,
        "epoch_reset": True,
    }
    torch.save(leaf, leaf_path)
    with pytest.raises(ValueError, match="does not prove train-only"):
        sanitize_recipient_full_crop_seed(
            status_checkpoint=status_path,
            train_only_recipient_checkpoint=leaf_path,
            output_checkpoint=output_path,
            torch=torch,
        )
    assert not output_path.exists()


def test_file_sanitizer_rejects_hash_bound_parent_incompatible_with_declared_mode(
    tmp_path: Path,
) -> None:
    torch = _torch()
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    parent = _payload(
        architecture=12,
        train_only=True,
        recipient_fill=2.0,
        recipient_characters=["商"],
    )
    leaf = _payload(
        architecture=12,
        train_only=True,
        recipient_fill=3.0,
        recipient_characters=["商", "户"],
    )
    status_path = tmp_path / "status.pt"
    parent_path = tmp_path / "incompatible-parent.pt"
    leaf_path = tmp_path / "leaf.pt"
    output_path = tmp_path / "must-not-exist.pt"
    torch.save(status, status_path)
    torch.save(parent, parent_path)
    leaf["initialization"] = {
        "mode": "parameter_only",
        "checkpoint_path": str(parent_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "source_kind": KIND_V12,
        "source_epoch": parent["epoch"],
        "source_config": parent["config"],
        "optimizer_restored": False,
        "epoch_reset": True,
    }
    torch.save(leaf, leaf_path)
    with pytest.raises(ValueError, match="strict parameter-only config/labels"):
        sanitize_recipient_full_crop_seed(
            status_checkpoint=status_path,
            train_only_recipient_checkpoint=leaf_path,
            output_checkpoint=output_path,
            torch=torch,
        )
    assert not output_path.exists()


def test_warmstart_reopens_sources_and_rejects_coherently_resealed_lineage_splice(
    tmp_path: Path,
) -> None:
    torch = _torch()
    status = _payload(architecture=13, train_only=False, recipient_fill=8.0)
    original_root = _payload(architecture=12, train_only=True, recipient_fill=2.0)
    spliced_root = _payload(architecture=12, train_only=True, recipient_fill=4.0)
    leaf = _payload(architecture=12, train_only=True, recipient_fill=3.0)
    status_path = tmp_path / "status.pt"
    original_root_path = tmp_path / "original-root.pt"
    spliced_root_path = tmp_path / "spliced-root.pt"
    leaf_path = tmp_path / "leaf.pt"
    output_path = tmp_path / "sanitized.pt"
    torch.save(status, status_path)
    torch.save(original_root, original_root_path)
    torch.save(spliced_root, spliced_root_path)
    leaf["initialization"] = {
        "mode": "parameter_only",
        "checkpoint_path": str(original_root_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(original_root_path.read_bytes()).hexdigest(),
        "source_kind": KIND_V12,
        "source_epoch": original_root["epoch"],
        "source_config": original_root["config"],
        "optimizer_restored": False,
        "epoch_reset": True,
    }
    torch.save(leaf, leaf_path)
    sanitize_recipient_full_crop_seed(
        status_checkpoint=status_path,
        train_only_recipient_checkpoint=leaf_path,
        output_checkpoint=output_path,
        torch=torch,
    )
    try:
        tampered = torch.load(output_path, map_location="cpu", weights_only=True)
    except TypeError:
        tampered = torch.load(output_path, map_location="cpu")
    lineage = tampered[ATTESTATION_KEY]["train_only_recipient_lineage"]
    child, root = lineage["entries"]
    spliced_hash = hashlib.sha256(spliced_root_path.read_bytes()).hexdigest()
    root["checkpoint"] = {
        "path": str(spliced_root_path.resolve()),
        "sha256": spliced_hash,
        "size_bytes": spliced_root_path.stat().st_size,
        "kind": KIND_V12,
        "epoch": spliced_root["epoch"],
    }
    root["config_sha256"] = _canonical_sha256(
        spliced_root["config"], description="spliced root config"
    )
    root["recipient_charset_sha256"] = spliced_root["recipient_charset_sha256"]
    root["recipient_train_split_policy_sha256"] = _canonical_sha256(
        spliced_root["recipient_train_split_policy"], description="spliced root policy"
    )
    root["recipient_state"] = _partition_descriptor(
        spliced_root["state_dict"], recipient=True
    )
    child["parent_checkpoint_path"] = root["checkpoint"]["path"]
    child["parent_checkpoint_sha256"] = spliced_hash
    child["parent_config_sha256"] = root["config_sha256"]
    child["parent_epoch"] = spliced_root["epoch"]
    _reseal_attestation(tampered)
    # It is structurally coherent and publicly resealed, so only reopening the
    # recorded leaf and reconstructing its real initialization chain catches it.
    validate_recipient_full_crop_seed_attestation(tampered)
    with pytest.raises(ValueError, match="content-bound seed sanitizer attestation"):
        _validate_recipient_full_crop_seed_policy(tampered, torch=torch)


def test_atomic_file_publication_is_fresh_and_reloads(tmp_path: Path) -> None:
    torch = _torch()
    status = _payload(
        architecture=13,
        train_only=False,
        recipient_fill=8.0,
        payment_characters=["卡", "行", "银", "储", "蓄"],
    )
    recipient = _payload(
        architecture=12,
        train_only=True,
        recipient_fill=3.0,
        payment_characters=["卡", "行", "银", "储"],
    )
    status_path = tmp_path / "status.pt"
    recipient_path = tmp_path / "recipient.pt"
    output_path = tmp_path / "sanitized.pt"
    torch.save(status, status_path)
    torch.save(recipient, recipient_path)
    summary = sanitize_recipient_full_crop_seed(
        status_checkpoint=status_path,
        train_only_recipient_checkpoint=recipient_path,
        output_checkpoint=output_path,
        torch=torch,
    )
    assert summary["analysis_only"] is True
    assert summary["production_route_authorized"] is False
    assert summary["output_checkpoint_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    try:
        reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    except TypeError:
        reloaded = torch.load(output_path, map_location="cpu")
    validate_recipient_full_crop_seed_attestation(reloaded)
    _validate_recipient_full_crop_seed_policy(reloaded, torch=torch)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        sanitize_recipient_full_crop_seed(
            status_checkpoint=status_path,
            train_only_recipient_checkpoint=recipient_path,
            output_checkpoint=output_path,
            torch=torch,
        )


def test_reopened_policy_rejects_resealed_train_payment_tensor_substitution(
    tmp_path: Path,
) -> None:
    torch = _torch()
    status = _payload(
        architecture=13,
        train_only=False,
        recipient_fill=8.0,
        payment_characters=["卡", "行", "银", "储"],
        shared_fill=1.0,
    )
    recipient = _payload(
        architecture=12,
        train_only=True,
        recipient_fill=3.0,
        payment_characters=["卡", "行", "银", "蓄"],
        shared_fill=2.0,
    )
    status_path = tmp_path / "status.pt"
    recipient_path = tmp_path / "recipient.pt"
    output_path = tmp_path / "sanitized.pt"
    torch.save(status, status_path)
    torch.save(recipient, recipient_path)
    sanitize_recipient_full_crop_seed(
        status_checkpoint=status_path,
        train_only_recipient_checkpoint=recipient_path,
        output_checkpoint=output_path,
        torch=torch,
    )
    try:
        tampered = torch.load(output_path, map_location="cpu", weights_only=True)
    except TypeError:
        tampered = torch.load(output_path, map_location="cpu")
    name = "payment_ctc_classifier.weight"
    assert tampered["state_dict"][name].shape == recipient["state_dict"][name].shape
    assert not torch.equal(tampered["state_dict"][name], recipient["state_dict"][name])
    tampered["state_dict"][name] = recipient["state_dict"][name].clone()
    tampered[ATTESTATION_KEY]["state_proof"]["output_non_recipient"] = (
        _partition_descriptor(tampered["state_dict"], recipient=False)
    )
    _reseal_attestation(tampered)
    validate_recipient_full_crop_seed_attestation(tampered)
    with pytest.raises(ValueError, match="content-bound seed sanitizer attestation"):
        _validate_recipient_full_crop_seed_policy(tampered, torch=torch)
