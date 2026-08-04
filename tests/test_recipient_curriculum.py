"""Pure unit tests for recipient-focused v12 training curriculum helpers.

These tests deliberately stay independent of torch, PIL, ONNX, and actual
image assets.  They protect the training-distribution and teacher-confidence
contracts that are frozen into the reader sidecars, while leaving the single
ONNX inference ABI untouched.
"""

from __future__ import annotations

import math

import pytest

from transfer_receipt_ai.ocr_unified import (
    _recipient_teacher_confidence_weights,
    _recipient_training_sample_weights,
    build_parser,
)


_MISSING = object()


def _record(text: str | None, confidence: object = _MISSING) -> dict[str, object]:
    """Build the minimal receipt shape consumed by curriculum helpers."""
    slot: dict[str, object] | None
    if text is None:
        slot = None
    else:
        slot = {"text": text}
        if confidence is not _MISSING:
            slot["paddle_confidence"] = confidence
    return {"slots": {"recipient_field": slot}}


def test_recipient_training_sample_weights_defaults_are_uniform() -> None:
    records = [_record("商户甲"), _record(None), _record("商户乙")]

    weights, policy = _recipient_training_sample_weights(
        records,
        recipient_sampling_weight=1.0,
        recipient_rare_character_max_support=0,
        recipient_rare_character_sampling_weight=1.0,
        recipient_long_text_min_length=0,
        recipient_long_text_sampling_weight=1.0,
    )

    assert weights == [1.0, 1.0, 1.0]
    assert policy["mode"] == "uniform"
    assert policy["recipient_sampling_weight"] == 1.0
    assert policy["recipient_train_records"] == 2
    assert policy["train_records"] == 3


def test_recipient_training_sample_weights_preserve_legacy_downsampling() -> None:
    """A pre-v2 recipe is allowed to intentionally downsample recipients."""
    weights, policy = _recipient_training_sample_weights(
        [_record("商户甲"), _record(None)],
        recipient_sampling_weight=0.5,
    )

    assert weights == [0.5, 1.0]
    assert policy == {
        "mode": "weighted_receipt_sampler_v1",
        "recipient_sampling_weight": 0.5,
        "recipient_train_records": 1,
        "train_records": 2,
    }


def test_recipient_training_sample_weights_do_not_enable_sampler_without_effect() -> None:
    """Unused rare/long switches must not change the historical row order."""
    weights, policy = _recipient_training_sample_weights(
        [_record("商户甲"), _record(None)],
        recipient_sampling_weight=1.0,
        recipient_rare_character_max_support=3,
        recipient_rare_character_sampling_weight=1.0,
        recipient_long_text_min_length=12,
        recipient_long_text_sampling_weight=1.0,
    )

    assert weights == [1.0, 1.0]
    assert policy == {
        "mode": "uniform",
        "recipient_sampling_weight": 1.0,
        "recipient_train_records": 1,
        "train_records": 2,
    }


def test_recipient_training_sample_weights_use_max_not_product_for_focus_rows() -> None:
    """A rare, long merchant gets the strongest single boost, not a product.

    Multiplying several oversampling factors would make a handful of noisy
    examples dominate an epoch.  The curriculum should instead cap each
    receipt at the largest explicitly requested boost.
    """
    records = [
        _record("甲"),  # rare only
        _record("常常常常"),  # long only
        _record("乙常常常"),  # rare and long
        _record("常常"),  # ordinary recipient
        _record(None),  # no recipient slot
    ]

    weights, policy = _recipient_training_sample_weights(
        records,
        recipient_sampling_weight=2.0,
        recipient_rare_character_max_support=1,
        recipient_rare_character_sampling_weight=5.0,
        recipient_long_text_min_length=4,
        recipient_long_text_sampling_weight=7.0,
    )

    assert weights == [5.0, 7.0, 7.0, 2.0, 1.0]
    assert policy["mode"] == "weighted_receipt_sampler_v2"
    assert policy["recipient_sampling_weight"] == 2.0
    assert policy["recipient_rare_character_max_support"] == 1
    assert policy["recipient_rare_character_sampling_weight"] == 5.0
    assert policy["recipient_long_text_min_length"] == 4
    assert policy["recipient_long_text_sampling_weight"] == 7.0
    assert policy["recipient_train_records"] == 4
    assert policy["train_records"] == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"recipient_sampling_weight": 0.0}, "recipient_sampling_weight"),
        ({"recipient_rare_character_max_support": -1}, "recipient_rare_character_max_support"),
        ({"recipient_rare_character_sampling_weight": 0.0}, "recipient_rare_character_sampling_weight"),
        ({"recipient_long_text_min_length": -1}, "recipient_long_text_min_length"),
        ({"recipient_long_text_sampling_weight": 0.0}, "recipient_long_text_sampling_weight"),
    ),
)
def test_recipient_training_sample_weights_reject_invalid_policy_values(
    kwargs: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "recipient_sampling_weight": 1.0,
        "recipient_rare_character_max_support": 0,
        "recipient_rare_character_sampling_weight": 1.0,
        "recipient_long_text_min_length": 0,
        "recipient_long_text_sampling_weight": 1.0,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _recipient_training_sample_weights([_record("商户甲")], **base)  # type: ignore[arg-type]


def test_recipient_teacher_confidence_weights_default_to_no_reweighting() -> None:
    records = [_record("商户甲"), _record("商户乙", "not-a-number"), _record(None)]

    weights = _recipient_teacher_confidence_weights(
        records,
        low_confidence_threshold=None,
        low_confidence_loss_weight=1.0,
        curriculum_epoch=1,
        curriculum_epochs=0,
    )

    assert weights == [1.0, 1.0, 1.0]


def test_recipient_teacher_confidence_weights_ramp_linearly() -> None:
    records = [_record("高", 0.99), _record("低", 0.97), _record(None)]

    early = _recipient_teacher_confidence_weights(
        records,
        low_confidence_threshold=0.98,
        low_confidence_loss_weight=0.20,
        curriculum_epoch=1,
        curriculum_epochs=4,
    )
    middle = _recipient_teacher_confidence_weights(
        records,
        low_confidence_threshold=0.98,
        low_confidence_loss_weight=0.20,
        curriculum_epoch=2,
        curriculum_epochs=4,
    )
    completed = _recipient_teacher_confidence_weights(
        records,
        low_confidence_threshold=0.98,
        low_confidence_loss_weight=0.20,
        curriculum_epoch=4,
        curriculum_epochs=4,
    )

    # The disabled/high-confidence examples always retain their full loss;
    # a low-confidence row starts at 1 and linearly reaches the requested
    # down-weight after the configured curriculum duration.
    assert early == pytest.approx([1.0, 1.0, 1.0])
    assert middle == pytest.approx([1.0, 0.7333333333, 1.0])
    assert completed == pytest.approx([1.0, 0.2, 1.0])


@pytest.mark.parametrize(
    "confidence",
    (_MISSING, None, "bad", math.nan, math.inf),
)
def test_recipient_teacher_confidence_weights_require_finite_confidence_when_enabled(
    confidence: object,
) -> None:
    record = _record("商户甲") if confidence is _MISSING else _record("商户甲", confidence)

    with pytest.raises(ValueError, match="paddle_confidence"):
        _recipient_teacher_confidence_weights(
            [record],
            low_confidence_threshold=0.98,
            low_confidence_loss_weight=0.5,
            curriculum_epoch=1,
            curriculum_epochs=1,
        )


def test_train_cli_forwards_recipient_curriculum_options() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--records",
            "records.jsonl",
            "--output",
            "run",
            "--recipient-rare-character-max-support",
            "3",
            "--recipient-rare-character-sampling-weight",
            "3.5",
            "--recipient-long-text-min-length",
            "12",
            "--recipient-long-text-sampling-weight",
            "2.5",
            "--recipient-low-confidence-threshold",
            "0.95",
            "--recipient-low-confidence-loss-weight",
            "0.35",
            "--recipient-confidence-curriculum-epochs",
            "8",
            "--recipient-train-augmentation",
            "light_v1",
        ]
    )

    assert args.recipient_rare_character_max_support == 3
    assert args.recipient_rare_character_sampling_weight == 3.5
    assert args.recipient_long_text_min_length == 12
    assert args.recipient_long_text_sampling_weight == 2.5
    assert args.recipient_low_confidence_threshold == 0.95
    assert args.recipient_low_confidence_loss_weight == 0.35
    assert args.recipient_confidence_curriculum_epochs == 8
    assert args.recipient_train_augmentation == "light_v1"
