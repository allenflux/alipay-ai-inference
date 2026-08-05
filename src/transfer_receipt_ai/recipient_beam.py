"""Open-vocabulary CTC prefix beam search with a character n-gram prior.

The language model stores only character-transition counts.  It never stores
or constrains decoding to complete recipient names, so unseen names remain
composable and the route is materially different from a closed lexicon.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def _logaddexp(*values: float) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


@dataclass(frozen=True)
class CharacterNGramLanguageModel:
    order: int
    vocabulary_size: int
    counts: dict[tuple[str, ...], int]
    context_counts: dict[tuple[str, ...], int]
    smoothing: float = 0.1

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        order: int = 3,
        smoothing: float = 0.1,
    ) -> "CharacterNGramLanguageModel":
        if order < 1 or order > 5:
            raise ValueError("character n-gram order must be between 1 and 5")
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("character n-gram smoothing must be finite and positive")
        materialized = [text for text in texts if isinstance(text, str) and text]
        vocabulary = sorted({character for text in materialized for character in text} | {"</s>"})
        if not vocabulary:
            raise ValueError("character n-gram language model requires non-empty text")
        counts: Counter[tuple[str, ...]] = Counter()
        contexts: Counter[tuple[str, ...]] = Counter()
        start = ("<s>",) * max(order - 1, 0)
        for text in materialized:
            history = list(start)
            for character in (*text, "</s>"):
                context = tuple(history[-(order - 1) :]) if order > 1 else ()
                counts[context + (character,)] += 1
                contexts[context] += 1
                history.append(character)
        return cls(
            order=order,
            vocabulary_size=len(vocabulary),
            counts=dict(counts),
            context_counts=dict(contexts),
            smoothing=float(smoothing),
        )

    def log_probability(self, prefix: str, character: str) -> float:
        context = tuple(prefix[-(self.order - 1) :]) if self.order > 1 else ()
        numerator = self.counts.get(context + (character,), 0) + self.smoothing
        denominator = self.context_counts.get(context, 0) + self.smoothing * self.vocabulary_size
        return math.log(numerator / denominator)

    def finish_log_probability(self, prefix: str) -> float:
        return self.log_probability(prefix, "</s>")


def decode_ctc_prefix_beam(
    logits: np.ndarray,
    *,
    characters: Sequence[str],
    language_model: CharacterNGramLanguageModel,
    beam_width: int = 10,
    token_top_k: int = 24,
    language_model_weight: float = 0.35,
) -> tuple[str, float]:
    """Decode one ``[time,class]`` CTC matrix with pruned prefix beam search."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(characters) + 1:
        raise ValueError("recipient CTC logits must have shape [time, blank plus characters]")
    if beam_width <= 0 or token_top_k <= 0:
        raise ValueError("beam_width and token_top_k must be positive")
    if not math.isfinite(language_model_weight) or language_model_weight < 0:
        raise ValueError("language_model_weight must be finite and non-negative")
    shifted = values - values.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    # prefix -> (blank score, non-blank score, accumulated LM score)
    beams: dict[str, tuple[float, float, float]] = {"": (0.0, -math.inf, 0.0)}
    lm_probability_cache: dict[tuple[tuple[str, ...], str], float] = {}

    def cached_lm_log_probability(prefix: str, character: str) -> float:
        context = tuple(prefix[-(language_model.order - 1) :]) if language_model.order > 1 else ()
        key = (context, character)
        cached = lm_probability_cache.get(key)
        if cached is None:
            cached = language_model.log_probability(prefix, character)
            lm_probability_cache[key] = cached
        return cached

    class_count = values.shape[1]
    keep = min(token_top_k, class_count)
    top_by_time = np.argpartition(log_probabilities, class_count - keep, axis=1)[:, class_count - keep :]
    for row, top in zip(log_probabilities, top_by_time):
        if 0 not in top:
            top = np.append(top, 0)
        next_beams: dict[str, tuple[float, float, float]] = {}

        def merge(prefix: str, blank: float, non_blank: float, lm_score: float) -> None:
            old_blank, old_non_blank, old_lm = next_beams.get(prefix, (-math.inf, -math.inf, lm_score))
            next_beams[prefix] = (
                _logaddexp(old_blank, blank),
                _logaddexp(old_non_blank, non_blank),
                max(old_lm, lm_score),
            )

        for prefix, (blank_score, non_blank_score, lm_score) in beams.items():
            total = _logaddexp(blank_score, non_blank_score)
            for raw_index in top:
                index = int(raw_index)
                acoustic = float(row[index])
                if index == 0:
                    merge(prefix, total + acoustic, -math.inf, lm_score)
                    continue
                character = characters[index - 1]
                if prefix and prefix[-1] == character:
                    # A repeated non-blank stays in the same collapsed prefix;
                    # only a path arriving from blank may append it again.
                    merge(prefix, -math.inf, non_blank_score + acoustic, lm_score)
                    extended = prefix + character
                    extension_lm = lm_score + cached_lm_log_probability(prefix, character)
                    merge(extended, -math.inf, blank_score + acoustic, extension_lm)
                else:
                    extended = prefix + character
                    extension_lm = lm_score + cached_lm_log_probability(prefix, character)
                    merge(extended, -math.inf, total + acoustic, extension_lm)
        ranked = sorted(
            next_beams.items(),
            key=lambda item: _logaddexp(item[1][0], item[1][1])
            + language_model_weight * item[1][2],
            reverse=True,
        )
        beams = dict(ranked[:beam_width])
    best_prefix, (best_blank, best_non_blank, best_lm) = max(
        beams.items(),
        key=lambda item: _logaddexp(item[1][0], item[1][1])
        + language_model_weight
        * (item[1][2] + cached_lm_log_probability(item[0], "</s>")),
    )
    score = _logaddexp(best_blank, best_non_blank) + language_model_weight * (
        best_lm + cached_lm_log_probability(best_prefix, "</s>")
    )
    return best_prefix, float(score)
