"""Safety tests for v13 recipient-only white-domain warm starts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
    INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
    KIND_V13,
    STATUS_CLASSES,
    STATUS_TEXT_BLANK_INDEX,
    STATUS_TEXT_CHARSET_SOURCE,
    STATUS_TEXT_RUNTIME_POLICY,
    STATUS_TEXT_TARGET,
    V6_TIME_CHARACTERS,
    V8_AMOUNT_CHARACTERS,
    UnifiedReaderConfig,
    _parameter_only_initialization,
    _recipient_charset_source,
    _recipient_domain_adapt_state,
    _recipient_only_expansion_label_override,
    _recipient_target_mode,
    _requires_non_recipient_parameter_byte_guard,
    _state_dict_exact_bytes,
    _validate_recipient_domain_adapt_config,
    _validate_validation_every,
    build_parser,
    build_unified_reader,
    main,
    train_unified_reader,
)


PAYMENT_CHARACTERS = ["储", "卡", "行", "银"]
BANK_CLASSES = ["__other__", "银行"]
STATUS_TEXT_CHARACTERS = sorted(set("转账成功"))


def _config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=13,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_input_height=32,
        recipient_input_width=128,
        recipient_branch_channels=8,
        pooled_width=2,
    )


def _model(config: UnifiedReaderConfig, recipient_characters: list[str]):
    return build_unified_reader(
        payment_vocab_size=len(PAYMENT_CHARACTERS) + 1,
        payment_bank_prefix_vocab_size=len(BANK_CLASSES),
        recipient_vocab_size=len(recipient_characters) + 1,
        status_text_vocab_size=len(STATUS_TEXT_CHARACTERS) + 1,
        config=config,
    )


def _status_oov_audit() -> dict[str, dict[str, object]]:
    return {
        split: {
            "records": 1,
            "oov_records": 0,
            "oov_characters": 0,
            "examples": [],
        }
        for split in ("train", "val", "test")
    }


def _payload(
    *,
    config: UnifiedReaderConfig,
    state_dict: dict[str, object],
    recipient_characters: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": KIND_V13,
        "config": asdict(config),
        "state_dict": state_dict,
        "amount_characters": list(V8_AMOUNT_CHARACTERS),
        "time_characters": list(V6_TIME_CHARACTERS),
        "payment_characters": PAYMENT_CHARACTERS,
        "recipient_characters": recipient_characters,
        "recipient_blank_index": 0,
        "recipient_charset_sha256": hashlib.sha256(
            "".join(recipient_characters).encode("utf-8")
        ).hexdigest(),
        "recipient_charset_source": _recipient_charset_source(config),
        "recipient_target": _recipient_target_mode(config),
        "recipient_sampling_policy": {
            "mode": "uniform",
            "recipient_sampling_weight": 1.0,
            "recipient_train_records": 1,
            "train_records": 1,
        },
        "status_classes": list(STATUS_CLASSES),
        "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
        "status_text_characters": STATUS_TEXT_CHARACTERS,
        "status_text_charset_sha256": hashlib.sha256(
            "".join(STATUS_TEXT_CHARACTERS).encode("utf-8")
        ).hexdigest(),
        "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
        "status_text_target": STATUS_TEXT_TARGET,
        "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
        "status_text_oov_by_split": _status_oov_audit(),
        "payment_bank_prefix_classes": BANK_CLASSES,
        "epoch": 4,
    }


def _write_seed(
    tmp_path: Path, *, recipient_characters: list[str]
) -> tuple[object, Path, dict[str, object]]:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    model = _model(_config(), recipient_characters)
    state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    for index, value in enumerate(state.values(), start=1):
        value.fill_(float(index) / 10.0)
    path = tmp_path / "v13-domain-seed.pt"
    torch.save(
        _payload(
            config=_config(),
            state_dict=state,
            recipient_characters=recipient_characters,
        ),
        path,
    )
    return torch, path, state


def _initialize(
    *,
    torch: object,
    checkpoint: Path,
    recipient_characters: list[str],
    target_state: dict[str, object],
):
    return _parameter_only_initialization(
        init_checkpoint=checkpoint,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
        config=_config(),
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=PAYMENT_CHARACTERS,
        recipient_characters=recipient_characters,
        status_text_characters=STATUS_TEXT_CHARACTERS,
        payment_bank_prefix_classes=BANK_CLASSES,
        torch=torch,
        target_state_dict=target_state,
    )


def test_recipient_domain_adapt_exact_charset_copies_every_state_byte(
    tmp_path: Path,
) -> None:
    recipient_characters = ["乙", "甲"]
    torch, checkpoint, source_state = _write_seed(
        tmp_path, recipient_characters=recipient_characters
    )
    target_state = _model(_config(), recipient_characters).state_dict()

    mapped, initialization = _initialize(
        torch=torch,
        checkpoint=checkpoint,
        recipient_characters=recipient_characters,
        target_state=target_state,
    )

    assert mapped is not None
    assert _state_dict_exact_bytes(mapped) == _state_dict_exact_bytes(source_state)
    assert initialization["mode"] == "parameter_only_recipient_domain_adapt"
    mapping = initialization["recipient_domain_adapt_mapping"]
    assert mapping["recipient_charset_exact"] is True
    assert mapping["source_value_copy"] == "all_state_exact"


def test_recipient_domain_adapt_extension_maps_only_unicode_classifier_rows(
    tmp_path: Path,
) -> None:
    source_characters = ["乙", "甲"]
    target_characters = ["丙", "乙", "甲"]
    torch, checkpoint, source_state = _write_seed(
        tmp_path, recipient_characters=source_characters
    )
    torch.manual_seed(29)
    target_state = {
        key: value.detach().clone()
        for key, value in _model(_config(), target_characters).state_dict().items()
    }
    fresh_new_weight = target_state["recipient_classifier.weight"][1].clone()
    fresh_new_bias = target_state["recipient_classifier.bias"][1].clone()

    mapped, initialization = _initialize(
        torch=torch,
        checkpoint=checkpoint,
        recipient_characters=target_characters,
        target_state=target_state,
    )

    assert mapped is not None
    classifier_keys = {
        "recipient_classifier.weight",
        "recipient_classifier.bias",
    }
    for key, source_value in source_state.items():
        if key not in classifier_keys:
            assert torch.equal(mapped[key], source_value)
    assert torch.equal(mapped["recipient_classifier.weight"][0], source_state["recipient_classifier.weight"][0])
    assert torch.equal(mapped["recipient_classifier.weight"][2], source_state["recipient_classifier.weight"][1])
    assert torch.equal(mapped["recipient_classifier.weight"][3], source_state["recipient_classifier.weight"][2])
    assert torch.equal(mapped["recipient_classifier.weight"][1], fresh_new_weight)
    assert torch.equal(mapped["recipient_classifier.bias"][1], fresh_new_bias)
    mapping = initialization["recipient_domain_adapt_mapping"]
    assert mapping["recipient_charset_exact"] is False
    assert mapping["recipient_classifier_row_mapping"]["shared_character_rows_copied"] == 2
    assert mapping["recipient_classifier_row_mapping"]["new_target_character_rows_kept_at_seed"] == 1


def test_recipient_domain_adapt_locks_financial_maps_and_unions_train_unicode(
    tmp_path: Path,
) -> None:
    torch, checkpoint, _ = _write_seed(
        tmp_path, recipient_characters=["乙", "甲"]
    )

    payment, banks, recipient, policy = _recipient_only_expansion_label_override(
        init_checkpoint=checkpoint,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
        config=_config(),
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=["新"],
        recipient_characters=["丙"],
        payment_bank_prefix_classes=["__other__", "新银行"],
        torch=torch,
    )

    assert payment == PAYMENT_CHARACTERS
    assert banks == BANK_CLASSES
    assert recipient == ["丙", "乙", "甲"]
    assert policy["mode"] == "checkpoint_financial_label_maps_recipient_domain_adapt_v1"


def test_recipient_domain_adapt_fails_closed_on_topology_dtype_and_discard(
    tmp_path: Path,
) -> None:
    torch, checkpoint, source_state = _write_seed(
        tmp_path, recipient_characters=["乙", "甲"]
    )
    incompatible = replace(_config(), recipient_input_width=256)
    with pytest.raises(ValueError, match="exact v13 source/target config match"):
        _validate_recipient_domain_adapt_config(_config(), incompatible)

    target_state = _model(_config(), ["乙", "甲"]).state_dict()
    bad_state = dict(source_state)
    non_classifier = next(key for key in bad_state if not key.startswith("recipient_"))
    bad_state[non_classifier] = bad_state[non_classifier].double()
    with pytest.raises(ValueError, match="state tensor dtype"):
        _recipient_domain_adapt_state(
            source_state_dict=bad_state,
            target_state_dict=target_state,
            source_recipient_characters=["乙", "甲"],
            target_recipient_characters=["乙", "甲"],
        )

    with pytest.raises(ValueError, match="cannot discard source recipient characters"):
        _recipient_domain_adapt_state(
            source_state_dict=source_state,
            target_state_dict=_model(_config(), ["乙"]).state_dict(),
            source_recipient_characters=["乙", "甲"],
            target_recipient_characters=["乙"],
        )

    # The saved checkpoint remains usable after the isolated tamper checks.
    assert checkpoint.is_file()


def test_recipient_domain_adapt_always_uses_frozen_non_recipient_byte_guard() -> None:
    _validate_validation_every(
        4,
        config=_config(),
        recipient_only_fine_tune=True,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
    )
    assert _requires_non_recipient_parameter_byte_guard(
        validation_every=1,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
    )
    assert not _requires_non_recipient_parameter_byte_guard(
        validation_every=1,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
    )
    assert _requires_non_recipient_parameter_byte_guard(
        validation_every=4,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
    )


def test_recipient_domain_adapt_is_a_public_train_cli_choice() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--records",
            "records.jsonl",
            "--output",
            "output",
            "--architecture",
            "v13",
            "--recipient-only-fine-tune",
            "--init-checkpoint",
            "seed.pt",
            "--init-checkpoint-mode",
            INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
        ]
    )
    assert args.init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT


def test_recipient_domain_adapt_rejects_v12_before_data_or_checkpoint_io(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires architecture v13"):
        train_unified_reader(
            records_path=tmp_path / "missing-records.jsonl",
            output_dir=tmp_path / "output",
            config=replace(_config(), architecture_version=12),
            recipient_only_fine_tune=True,
            init_checkpoint=tmp_path / "missing-seed.pt",
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
        )


@pytest.mark.parametrize(
    "recipient_train_splits",
    (("train", "val"), ("train", "test")),
)
def test_recipient_domain_adapt_function_rejects_non_train_supervision_before_io(
    tmp_path: Path,
    recipient_train_splits: tuple[str, ...],
) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="train-split supervision only"):
        train_unified_reader(
            records_path=tmp_path / "not-opened.jsonl",
            output_dir=output,
            config=_config(),
            recipient_train_splits=recipient_train_splits,
            recipient_only_fine_tune=True,
            init_checkpoint=tmp_path / "not-opened.pt",
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
        )
    assert not output.exists()


def test_recipient_domain_adapt_function_accepts_train_only_until_manifest_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="not-opened.jsonl"):
        train_unified_reader(
            records_path=tmp_path / "not-opened.jsonl",
            output_dir=tmp_path / "not-created",
            config=_config(),
            recipient_train_splits=("train",),
            recipient_only_fine_tune=True,
            init_checkpoint=tmp_path / "not-opened.pt",
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
        )


def test_recipient_domain_adapt_rejects_physical_test_lineage_before_checkpoint_io(
    tmp_path: Path,
) -> None:
    records = tmp_path / "unsafe.jsonl"
    records.write_text(
        '{"id":"secret","split":"test","slots":{}}\n',
        encoding="utf-8",
    )
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="physically excludes test rows"):
        train_unified_reader(
            records_path=records,
            output_dir=output,
            config=_config(),
            recipient_train_splits=("train",),
            recipient_only_fine_tune=True,
            init_checkpoint=tmp_path / "not-opened.pt",
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
        )
    assert not output.exists()


@pytest.mark.parametrize("leaked_split", ("val", "test"))
def test_recipient_domain_adapt_cli_rejects_non_train_supervision(
    tmp_path: Path,
    leaked_split: str,
) -> None:
    with pytest.raises(SystemExit, match="train-split supervision only"):
        main(
            [
                "train",
                "--records",
                str(tmp_path / "not-opened.jsonl"),
                "--output",
                str(tmp_path / "not-created"),
                "--architecture",
                "v13",
                "--recipient-only-fine-tune",
                "--recipient-train-splits",
                "train",
                leaked_split,
                "--init-checkpoint",
                str(tmp_path / "not-opened.pt"),
                "--init-checkpoint-mode",
                INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
            ]
        )


def test_recipient_domain_adapt_cli_accepts_train_only_until_manifest_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="not-opened.jsonl") as raised:
        main(
            [
                "train",
                "--records",
                str(tmp_path / "not-opened.jsonl"),
                "--output",
                str(tmp_path / "not-created"),
                "--architecture",
                "v13",
                "--recipient-only-fine-tune",
                "--recipient-train-splits",
                "train",
                "--init-checkpoint",
                str(tmp_path / "not-opened.pt"),
                "--init-checkpoint-mode",
                INIT_CHECKPOINT_MODE_RECIPIENT_DOMAIN_ADAPT,
            ]
        )
    assert "train-split supervision only" not in str(raised.value)
