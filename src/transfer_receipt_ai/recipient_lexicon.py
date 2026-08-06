"""Safe optional local recipient-name lexicon support.

This module deliberately has no image, ONNX, Torch, Paddle, or network
dependency.  A caller can use it after a single-ONNX reader has produced a
small ranked list of *its own* recipient candidates.  It only ever returns a
name that already exists in that candidate list (an exact match), or one
unambiguous one-edit dictionary correction of the top, very-high-confidence
candidate.  Unknown names are therefore left for the caller's normal OCR
fallback/review path instead of being invented from a merchant catalogue.

The JSON format is intentionally small and deterministic so the same catalog
can be shipped alongside an ML.NET/ONNX delivery without becoming another
model or service.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RECIPIENT_LEXICON_SCHEMA_VERSION = 1
RECIPIENT_LEXICON_KIND = "receipt_recipient_lexicon_v1"

# A one-edit correction of a short Chinese name is far too ambiguous.  These
# conservative defaults intentionally favour review over a false merchant
# substitution.  Exact candidates do not need a confidence value.
DEFAULT_MIN_NEAR_MATCH_CONFIDENCE = 0.995
DEFAULT_MIN_NEAR_MATCH_LENGTH = 7
DEFAULT_MIN_NEAR_MATCH_SIMILARITY = 0.85
DEFAULT_NEAR_MATCH_TOP_K = 1


class RecipientLexiconError(ValueError):
    """Raised when a recipient lexicon contract is malformed or unsafe."""


@dataclass(frozen=True)
class RecipientLexiconEntry:
    """One locally known recipient string and its source occurrence count."""

    text: str
    occurrences: int


@dataclass(frozen=True)
class RecipientLexiconCandidate:
    """A ranked OCR candidate supplied by the existing single ONNX reader.

    ``confidence`` is optional because an exact candidate remains safe even
    when a particular decoder does not expose a calibrated sequence score.
    Near matching is disabled for candidates without a finite confidence.
    """

    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("recipient candidate text must be a string")
        if self.confidence is None:
            return
        if isinstance(self.confidence, bool):
            raise ValueError("recipient candidate confidence must be numeric")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("recipient candidate confidence must be finite and between 0 and 1")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class RecipientLexiconMatch:
    """An auditable lexicon decision.

    Callers should emit ``resolved_text`` only for a non-``None`` result and
    otherwise retain their original model candidate or route the receipt to
    review.  ``match_kind == 'near'`` always means a one-edit correction,
    except ``edit_distance == 0`` for Unicode compatibility formatting (for
    example full-width versus ASCII parentheses).
    """

    resolved_text: str
    input_text: str
    candidate_index: int
    candidate_confidence: float | None
    match_kind: str
    edit_distance: int
    similarity: float
    occurrences: int

    def as_dict(self) -> dict[str, object]:
        return {
            "resolved_text": self.resolved_text,
            "input_text": self.input_text,
            "candidate_index": self.candidate_index,
            "candidate_confidence": self.candidate_confidence,
            "match_kind": self.match_kind,
            "edit_distance": self.edit_distance,
            "similarity": self.similarity,
            "occurrences": self.occurrences,
        }


def _comparison_key(value: str) -> str:
    """Return a narrow comparison key without changing a delivered string.

    NFC alone does not reconcile the full-width punctuation commonly emitted
    by receipt OCR.  NFKC/casefold does, but it is used only for a guarded
    high-confidence near match; the original lexicon spelling remains the
    returned delivery text.
    """

    return unicodedata.normalize("NFKC", value.strip()).casefold()


def _one_edit_deletes(value: str) -> tuple[str, ...]:
    """Deletion signatures shared by strings at Levenshtein distance <= 1."""

    signatures = {value}
    signatures.update(value[:index] + value[index + 1 :] for index in range(len(value)))
    return tuple(signatures)


def _bounded_levenshtein(left: str, right: str, *, maximum: int) -> int | None:
    """Return the edit distance only when it is at most ``maximum``."""

    if abs(len(left) - len(right)) > maximum:
        return None
    if left == right:
        return 0
    if not left or not right:
        distance = max(len(left), len(right))
        return distance if distance <= maximum else None

    # Keep the row width small when possible.  The implementation remains
    # straightforward because this safety contract intentionally supports one
    # edit only, not broad fuzzy merchant matching.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        row_minimum = left_index
        for right_index, right_character in enumerate(right, start=1):
            cost = 0 if left_character == right_character else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > maximum:
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= maximum else None


def _similarity(left: str, right: str, distance: int) -> float:
    return 1.0 - (distance / max(len(left), len(right), 1))


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise RecipientLexiconError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise RecipientLexiconError(f"{name} must not be empty")
    return text


def _require_occurrences(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecipientLexiconError(f"{name} must be a positive integer")
    return value


class RecipientLexicon:
    """A deterministic local recipient catalogue with a safe re-ranker."""

    def __init__(self, entries: Sequence[RecipientLexiconEntry]) -> None:
        exact: dict[str, RecipientLexiconEntry] = {}
        normalized: dict[str, list[RecipientLexiconEntry]] = {}
        for entry in entries:
            text = _require_text(entry.text, name="recipient lexicon entry text")
            occurrences = _require_occurrences(entry.occurrences, name="recipient lexicon entry occurrences")
            if text in exact:
                raise RecipientLexiconError(f"duplicate recipient lexicon entry: {text!r}")
            normalized_key = _comparison_key(text)
            if not normalized_key:
                raise RecipientLexiconError("recipient lexicon comparison key must not be empty")
            normalized_entry = RecipientLexiconEntry(text=text, occurrences=occurrences)
            exact[text] = normalized_entry
            normalized.setdefault(normalized_key, []).append(normalized_entry)

        # The persisted order is predictable, and frozen tuples prevent an
        # accidental caller mutation from changing the decision surface.
        self._entries = tuple(sorted(exact.values(), key=lambda item: item.text))
        self._exact = exact
        self._normalized = {key: tuple(value) for key, value in normalized.items()}
        near_index: dict[str, set[str]] = {}
        for normalized_key in self._normalized:
            for signature in _one_edit_deletes(normalized_key):
                near_index.setdefault(signature, set()).add(normalized_key)
        self._near_index = {key: frozenset(value) for key, value in near_index.items()}

    @property
    def entries(self) -> tuple[RecipientLexiconEntry, ...]:
        return self._entries

    @classmethod
    def from_strings(cls, values: Iterable[str]) -> "RecipientLexicon":
        """Build a catalogue from receipt/merchant names.

        Leading/trailing whitespace is removed during catalogue construction,
        while interior spaces and the original visible punctuation are kept.
        Blank values are ignored; non-string values are rejected rather than
        coerced into names.
        """

        counts: Counter[str] = Counter()
        for index, value in enumerate(values):
            if not isinstance(value, str):
                raise TypeError(f"recipient lexicon value at index {index} must be a string")
            text = value.strip()
            if text:
                counts[text] += 1
        return cls([RecipientLexiconEntry(text=text, occurrences=count) for text, count in counts.items()])

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RecipientLexicon":
        if not isinstance(payload, Mapping):
            raise RecipientLexiconError("recipient lexicon JSON must contain an object")
        if payload.get("schema_version") != RECIPIENT_LEXICON_SCHEMA_VERSION:
            raise RecipientLexiconError("unsupported recipient lexicon schema_version")
        if payload.get("kind") != RECIPIENT_LEXICON_KIND:
            raise RecipientLexiconError("unsupported recipient lexicon kind")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise RecipientLexiconError("recipient lexicon entries must be a list")

        entries: list[RecipientLexiconEntry] = []
        for index, value in enumerate(raw_entries):
            if not isinstance(value, Mapping):
                raise RecipientLexiconError(f"recipient lexicon entry {index} must be an object")
            entries.append(
                RecipientLexiconEntry(
                    text=_require_text(value.get("text"), name=f"recipient lexicon entry {index} text"),
                    occurrences=_require_occurrences(
                        value.get("occurrences"), name=f"recipient lexicon entry {index} occurrences"
                    ),
                )
            )
        return cls(entries)

    @classmethod
    def load(cls, path: str | Path) -> "RecipientLexicon":
        """Load the strict JSON catalogue, accepting UTF-8 with an optional BOM."""

        source = Path(path)
        try:
            raw: Any = json.loads(source.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise RecipientLexiconError(f"{source}: invalid JSON: {error}") from None
        if not isinstance(raw, Mapping):
            raise RecipientLexiconError(f"{source}: recipient lexicon JSON must contain an object")
        return cls.from_payload(raw)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": RECIPIENT_LEXICON_SCHEMA_VERSION,
            "kind": RECIPIENT_LEXICON_KIND,
            "entries": [
                {"text": entry.text, "occurrences": entry.occurrences}
                for entry in self._entries
            ],
        }

    def save(self, path: str | Path) -> None:
        """Persist an atomically replaced JSON catalogue."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def rerank(
        self,
        candidates: Iterable[str | RecipientLexiconCandidate],
        *,
        min_near_match_confidence: float = DEFAULT_MIN_NEAR_MATCH_CONFIDENCE,
        min_near_match_length: int = DEFAULT_MIN_NEAR_MATCH_LENGTH,
        min_near_match_similarity: float = DEFAULT_MIN_NEAR_MATCH_SIMILARITY,
        near_match_top_k: int = DEFAULT_NEAR_MATCH_TOP_K,
    ) -> RecipientLexiconMatch | None:
        """Return a safe exact/near correction, or ``None`` for unknown text.

        The input ordering is treated as beam rank.  Any exact dictionary
        candidate is safe and wins in that order.  A non-exact correction is
        considered only for the top ``near_match_top_k`` candidates, requires
        a high sequence confidence, at least seven comparison characters by
        default, a unique lexicon neighbour, and edit distance no greater than
        one.  Those constraints are deliberately strict: a catalogue should
        improve recurring merchant names, never fabricate arbitrary ones.
        """

        if not isinstance(min_near_match_confidence, (int, float)) or isinstance(
            min_near_match_confidence, bool
        ):
            raise ValueError("min_near_match_confidence must be numeric")
        min_near_match_confidence = float(min_near_match_confidence)
        if not math.isfinite(min_near_match_confidence) or not 0.0 <= min_near_match_confidence <= 1.0:
            raise ValueError("min_near_match_confidence must be finite and between 0 and 1")
        if isinstance(min_near_match_length, bool) or not isinstance(min_near_match_length, int) or min_near_match_length < 1:
            raise ValueError("min_near_match_length must be a positive integer")
        if not isinstance(min_near_match_similarity, (int, float)) or isinstance(min_near_match_similarity, bool):
            raise ValueError("min_near_match_similarity must be numeric")
        min_near_match_similarity = float(min_near_match_similarity)
        if not math.isfinite(min_near_match_similarity) or not 0.0 <= min_near_match_similarity <= 1.0:
            raise ValueError("min_near_match_similarity must be finite and between 0 and 1")
        if isinstance(near_match_top_k, bool) or not isinstance(near_match_top_k, int) or near_match_top_k < 1:
            raise ValueError("near_match_top_k must be a positive integer")

        normalized_candidates: list[RecipientLexiconCandidate] = []
        for value in candidates:
            if isinstance(value, str):
                normalized_candidates.append(RecipientLexiconCandidate(value))
            elif isinstance(value, RecipientLexiconCandidate):
                normalized_candidates.append(value)
            else:
                raise TypeError("recipient candidates must be strings or RecipientLexiconCandidate values")

        # An exact result is one the existing OCR decoder already offered; no
        # confidence threshold or fuzzy dictionary substitution is involved.
        for index, candidate in enumerate(normalized_candidates):
            entry = self._exact.get(candidate.text)
            if entry is not None:
                return RecipientLexiconMatch(
                    resolved_text=entry.text,
                    input_text=candidate.text,
                    candidate_index=index,
                    candidate_confidence=candidate.confidence,
                    match_kind="exact",
                    edit_distance=0,
                    similarity=1.0,
                    occurrences=entry.occurrences,
                )

        for index, candidate in enumerate(normalized_candidates[:near_match_top_k]):
            if candidate.confidence is None or candidate.confidence < min_near_match_confidence:
                continue
            comparison_key = _comparison_key(candidate.text)
            if len(comparison_key) < min_near_match_length:
                continue
            possible_keys: set[str] = set()
            for signature in _one_edit_deletes(comparison_key):
                possible_keys.update(self._near_index.get(signature, ()))

            matches: list[tuple[RecipientLexiconEntry, int, float]] = []
            for possible_key in possible_keys:
                distance = _bounded_levenshtein(comparison_key, possible_key, maximum=1)
                if distance is None:
                    continue
                similarity = _similarity(comparison_key, possible_key, distance)
                if similarity < min_near_match_similarity:
                    continue
                for entry in self._normalized[possible_key]:
                    matches.append((entry, distance, similarity))

            # A one-character correction is allowed only when the catalogue
            # yields one unique delivery spelling.  A tie is not a reason to
            # pick the most frequent merchant: it must remain an OCR/review
            # result to avoid inventing an unknown recipient.
            unique_entries = {entry.text: (entry, distance, similarity) for entry, distance, similarity in matches}
            if len(unique_entries) != 1:
                continue
            entry, distance, similarity = next(iter(unique_entries.values()))
            return RecipientLexiconMatch(
                resolved_text=entry.text,
                input_text=candidate.text,
                candidate_index=index,
                candidate_confidence=candidate.confidence,
                match_kind="near",
                edit_distance=distance,
                similarity=similarity,
                occurrences=entry.occurrences,
            )
        return None


def build_recipient_lexicon(values: Iterable[str]) -> RecipientLexicon:
    """Build a local lexicon from strings; shorthand for ``from_strings``."""

    return RecipientLexicon.from_strings(values)


def load_recipient_lexicon(path: str | Path) -> RecipientLexicon:
    """Load a local lexicon; shorthand for :meth:`RecipientLexicon.load`."""

    return RecipientLexicon.load(path)


def rerank_recipient_candidates(
    lexicon: RecipientLexicon | None,
    candidates: Iterable[str | RecipientLexiconCandidate],
    **kwargs: object,
) -> RecipientLexiconMatch | None:
    """Safely rerank candidates when a local lexicon is configured.

    ``None`` is intentionally accepted for ``lexicon`` so callers can make
    the optional sidecar a zero-cost no-op without branching their ONNX path.
    """

    if lexicon is None:
        return None
    return lexicon.rerank(candidates, **kwargs)  # type: ignore[arg-type]
