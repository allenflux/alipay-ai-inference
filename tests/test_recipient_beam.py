from __future__ import annotations

import numpy as np
import pytest

from transfer_receipt_ai.recipient_beam import CharacterNGramLanguageModel, decode_ctc_prefix_beam


def test_character_ngram_beam_can_override_an_ambiguous_greedy_path() -> None:
    language_model = CharacterNGramLanguageModel.from_texts(["AA"] * 20 + ["AB"], order=2)
    probabilities = np.asarray(
        [
            [0.02, 0.55, 0.43],
            [0.90, 0.05, 0.05],
            [0.02, 0.43, 0.55],
        ],
        dtype=np.float64,
    )
    text, score = decode_ctc_prefix_beam(
        np.log(probabilities),
        characters=["A", "B"],
        language_model=language_model,
        beam_width=8,
        token_top_k=3,
        language_model_weight=1.5,
    )
    assert text == "AA"
    assert np.isfinite(score)


def test_character_ngram_model_does_not_store_complete_names() -> None:
    language_model = CharacterNGramLanguageModel.from_texts(["甲乙", "乙丙"], order=2)
    assert all(len(key) <= 2 for key in language_model.counts)
    assert language_model.log_probability("甲", "丙") < 0


def test_recipient_beam_rejects_invalid_shapes_and_weights() -> None:
    language_model = CharacterNGramLanguageModel.from_texts(["甲"], order=1)
    with pytest.raises(ValueError, match="shape"):
        decode_ctc_prefix_beam(
            np.zeros((2, 2, 2)),
            characters=["甲"],
            language_model=language_model,
        )
    with pytest.raises(ValueError, match="non-negative"):
        decode_ctc_prefix_beam(
            np.zeros((2, 2)),
            characters=["甲"],
            language_model=language_model,
            language_model_weight=-1,
        )
