from __future__ import annotations

import json

import pytest

from transfer_receipt_ai.recipient_lexicon import (
    RECIPIENT_LEXICON_KIND,
    RECIPIENT_LEXICON_SCHEMA_VERSION,
    RecipientLexicon,
    RecipientLexiconCandidate,
    RecipientLexiconError,
    build_recipient_lexicon,
    load_recipient_lexicon,
    rerank_recipient_candidates,
)


def test_build_save_and_load_are_deterministic_and_count_duplicate_names(tmp_path) -> None:
    lexicon = build_recipient_lexicon((" 深圳市示例商贸有限公司 ", "北京测试商店", "深圳市示例商贸有限公司"))

    assert [(entry.text, entry.occurrences) for entry in lexicon.entries] == [
        ("北京测试商店", 1),
        ("深圳市示例商贸有限公司", 2),
    ]

    path = tmp_path / "recipient_catalog.json"
    lexicon.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "entries": [
            {"occurrences": 1, "text": "北京测试商店"},
            {"occurrences": 2, "text": "深圳市示例商贸有限公司"},
        ],
        "kind": RECIPIENT_LEXICON_KIND,
        "schema_version": RECIPIENT_LEXICON_SCHEMA_VERSION,
    }
    assert load_recipient_lexicon(path).to_payload() == lexicon.to_payload()


def test_exact_candidate_is_returned_without_near_match_confidence() -> None:
    lexicon = build_recipient_lexicon(("深圳市示例商贸有限公司",))

    match = lexicon.rerank((RecipientLexiconCandidate("深圳市示例商贸有限公司", confidence=0.12),))

    assert match is not None
    assert match.resolved_text == "深圳市示例商贸有限公司"
    assert match.match_kind == "exact"
    assert match.edit_distance == 0
    assert match.candidate_confidence == pytest.approx(0.12)

    # Plain strings are supported for decoders that only expose text.  They
    # can receive an exact hit but are never eligible for a fuzzy rewrite.
    string_match = lexicon.rerank(("完全未知收款方名称", "深圳市示例商贸有限公司"))
    assert string_match is not None
    assert string_match.candidate_index == 1
    assert string_match.candidate_confidence is None


def test_high_confidence_unique_one_edit_candidate_can_use_local_catalogue() -> None:
    lexicon = build_recipient_lexicon(("深圳市示例商贸有限公司", "北京测试商店"))

    match = lexicon.rerank((RecipientLexiconCandidate("深圳市示例商贸有限公可", confidence=0.999),))

    assert match is not None
    assert match.resolved_text == "深圳市示例商贸有限公司"
    assert match.input_text == "深圳市示例商贸有限公可"
    assert match.match_kind == "near"
    assert match.edit_distance == 1
    assert match.similarity > 0.85


def test_low_confidence_or_short_near_candidate_never_rewrites_unknown_name() -> None:
    lexicon = build_recipient_lexicon(("深圳市示例商贸有限公司", "深圳商店"))

    assert lexicon.rerank((RecipientLexiconCandidate("深圳市示例商贸有限公可", confidence=0.994),)) is None
    assert lexicon.rerank((RecipientLexiconCandidate("深圳商甸", confidence=1.0),)) is None
    assert rerank_recipient_candidates(None, (RecipientLexiconCandidate("深圳市示例商贸有限公可", 1.0),)) is None


def test_ambiguous_near_match_and_unseen_text_fall_back_to_none() -> None:
    lexicon = build_recipient_lexicon(("深圳市示例商贸有限公司", "深圳市示例商贸有限公所"))

    # The candidate is one edit from both names.  The lexicon must not use
    # occurrence frequency as a tie breaker because it would invent a result.
    assert lexicon.rerank((RecipientLexiconCandidate("深圳市示例商贸有限公可", confidence=0.999),)) is None
    assert lexicon.rerank((RecipientLexiconCandidate("完全未知收款方名称", confidence=1.0),)) is None


def test_near_matching_only_considers_the_top_candidate_by_default() -> None:
    lexicon = build_recipient_lexicon(("深圳市示例商贸有限公司",))
    candidates = (
        RecipientLexiconCandidate("完全未知收款方名称", confidence=1.0),
        RecipientLexiconCandidate("深圳市示例商贸有限公可", confidence=1.0),
    )

    assert lexicon.rerank(candidates) is None
    match = lexicon.rerank(candidates, near_match_top_k=2)
    assert match is not None
    assert match.candidate_index == 1
    assert match.resolved_text == "深圳市示例商贸有限公司"


def test_fullwidth_compatibility_is_high_confidence_near_match_not_an_exact_rewrite() -> None:
    lexicon = build_recipient_lexicon(("建设银行储蓄卡(3531)",))

    assert lexicon.rerank((RecipientLexiconCandidate("建设银行储蓄卡（3531）", confidence=0.99),)) is None
    match = lexicon.rerank((RecipientLexiconCandidate("建设银行储蓄卡（3531）", confidence=0.999),))

    assert match is not None
    assert match.match_kind == "near"
    assert match.edit_distance == 0
    assert match.resolved_text == "建设银行储蓄卡(3531)"


def test_load_rejects_unknown_schema_and_invalid_entries(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": RECIPIENT_LEXICON_SCHEMA_VERSION,
                "kind": RECIPIENT_LEXICON_KIND,
                "entries": [{"text": "", "occurrences": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RecipientLexiconError, match="must not be empty"):
        RecipientLexicon.load(path)

    with pytest.raises(TypeError, match="must be a string"):
        build_recipient_lexicon(("深圳市示例商贸有限公司", 7))  # type: ignore[arg-type]
