"""Tests for the phase-1 rejection appeal classifier.

No test makes a real model call. The hermetic conftest pins
MODEL_PROVIDER=fallback, which the module turns into a no-op; tests that
exercise a provider branch patch the transport and RECORD calls rather than
asserting inside the stub (the entry point catches everything, so an assertion
raised from a stub would be swallowed into the same None a passing case returns).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from app.core.config import settings
from app.pipeline import phase1_appeal_classifier as pac

# Synthetic email: every quote below is a substring of it.
SOURCE = (
    "Subject: Appeal for submission 1234\nBody:\n"
    "Dear PC, Reviewer 2 clearly reviewed a different paper, since the review "
    "discusses graph neural networks for protein folding. My rating from "
    "Reviewer 3 says “accept” but the score shown is 2. Reviewer 1 claims "
    "there is no ablation study, yet Section 5 contains a full ablation. "
    "Our average score was above the acceptance threshold. "
    "Please reconsider the decision on our paper."
)

QUOTE = {
    "wrong_paper_review": "Reviewer 2 clearly reviewed a different paper",
    "record_error": "says \"accept\" but the score shown is 2",
    "missing_material_claim": "Reviewer 1 claims there is no ablation study",
    "decision_vs_reviews": "average score was above the acceptance threshold",
    "reconsideration_only": "Please reconsider the decision on our paper",
}

# The approved prompt, pinned by length and hash.
_APPROVED_PROMPT_CHARS = 1991
_APPROVED_PROMPT_SHA256 = "a3d6bc508da8c8d1e74be1e4d2d4db470ad3fa57145519afbdfb60a38ac8dc89"


def _answer(relation="appeal", papers=("1234",), reasons=()):
    return json.dumps({
        "relation": relation,
        "papers": list(papers),
        "reasons": [{"reason": r, "quote": q} for r, q in reasons],
    })


def _reasons(result):
    return [r.reason for r in result.reasons]


def _parse(*reasons, **kwargs):
    result = pac.parse_answer(_answer(reasons=reasons, **kwargs), SOURCE)
    assert result is not None
    return result


def _route_local(monkeypatch, text):
    """Route to the local branch and make it return `text`; returns the call log."""
    calls: list[str] = []

    async def fake_local(user):  # noqa: ANN001
        calls.append(user)
        return text

    monkeypatch.setattr(settings, "MODEL_PROVIDER", "local")
    monkeypatch.setattr(pac, "_call_local", fake_local)
    return calls


# ---------------------------------------------------------------------------
# Prompt and registry
# ---------------------------------------------------------------------------
def test_prompt_is_byte_identical_to_the_approved_version():
    """Changing the prompt requires re-approving the wording and re-running the
    eval before updating both constants; do not just re-baseline them."""
    p = pac.SYSTEM_PROMPT
    assert len(p) == _APPROVED_PROMPT_CHARS
    assert hashlib.sha256(p.encode("utf-8")).hexdigest() == _APPROVED_PROMPT_SHA256
    assert pac.PROMPT_SHA256 == _APPROVED_PROMPT_SHA256


def test_prompt_lists_exactly_the_registry_reasons_in_order():
    menu = [
        line[2:].split(":", 1)[0]
        for line in pac.SYSTEM_PROMPT.splitlines()
        if line.startswith("- ") and not line.startswith(("- relation", "- papers"))
    ]
    assert menu == list(pac.PHASE1_APPEAL_REASONS)


def test_registry_order_and_constants():
    assert pac.PHASE1_APPEAL_REASONS == (
        "wrong_paper_review", "record_error", "missing_material_claim",
        "reviewer_misconduct", "llm_generated_review", "decision_vs_reviews",
        "reviewer_misjudgment", "reconsideration_only", "other",
    )
    assert pac.MUST_VERIFY_REASONS == {"wrong_paper_review", "record_error"}
    assert pac.RELATIONS == ("appeal", "feedback_only", "not_appeal")


# ---------------------------------------------------------------------------
# Contract violations -> None
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        "{not json}",
        '["appeal"]',
        _answer(relation="maybe"),
        _answer(relation="Appeal"),
        json.dumps({"relation": "appeal", "papers": [], "reasons": "other"}),
        json.dumps({"relation": "appeal", "papers": []}),
        _answer(reasons=[("made_up_reason", QUOTE["wrong_paper_review"])]),
        # One unknown name fails the whole answer, even next to a valid reason.
        _answer(reasons=[("wrong_paper_review", QUOTE["wrong_paper_review"]),
                         ("reciprocal_dispute", "anything")]),
        json.dumps({"relation": "appeal", "papers": [], "reasons": ["other"]}),
    ],
)
def test_a_contract_violation_is_None_never_a_partial_guess(text):
    assert pac.parse_answer(text, SOURCE) is None


def test_json_is_extracted_from_surrounding_text():
    text = "Here is the answer:\n```json\n" + _answer(
        reasons=[("wrong_paper_review", QUOTE["wrong_paper_review"])]
    ) + "\n```"
    result = pac.parse_answer(text, SOURCE)
    assert result is not None
    assert _reasons(result) == ["wrong_paper_review"]


@pytest.mark.parametrize("relation", ["appeal", "feedback_only", "not_appeal"])
def test_every_relation_is_accepted(relation):
    assert _parse(relation=relation).relation == relation


# ---------------------------------------------------------------------------
# Quote check
# ---------------------------------------------------------------------------
def test_a_reason_with_an_unfound_quote_is_dropped_and_recorded():
    result = _parse(
        ("wrong_paper_review", QUOTE["wrong_paper_review"]),
        ("reviewer_misconduct", "the reviewer insulted our entire team"),
    )
    assert _reasons(result) == ["wrong_paper_review"]
    assert result.dropped_unquoted == ["reviewer_misconduct"]


@pytest.mark.parametrize("quote", [None, "", 42])
def test_a_missing_or_non_string_quote_drops_the_reason(quote):
    text = json.dumps({"relation": "appeal", "papers": [],
                       "reasons": [{"reason": "other", "quote": quote}]})
    result = pac.parse_answer(text, SOURCE)
    assert result is not None
    assert result.reasons == []
    assert result.dropped_unquoted == ["other"]


def test_quote_matching_ignores_case_whitespace_and_curly_quotes():
    # The source has curly quotes around accept; the quote uses straight ones,
    # different case and a line break.
    result = _parse(("record_error", 'SAYS "accept"   but the\nscore shown is 2'))
    assert _reasons(result) == ["record_error"]


def test_edge_punctuation_on_a_quote_is_ignored():
    result = _parse(("wrong_paper_review", '"Reviewer 2 clearly reviewed a different paper!"'))
    assert _reasons(result) == ["wrong_paper_review"]


@pytest.mark.parametrize("elision", ["...", "…", "[...]", "[sic]", " [the review] "])
def test_a_quote_is_split_at_elisions(elision):
    quote = f"Reviewer 2 clearly reviewed{elision}discusses graph neural networks"
    assert _reasons(_parse(("wrong_paper_review", quote))) == ["wrong_paper_review"]


def test_every_fragment_of_an_elided_quote_must_be_found():
    quote = "Reviewer 2 clearly reviewed ... a fabricated second fragment"
    result = _parse(("wrong_paper_review", quote))
    assert result.reasons == []
    assert result.dropped_unquoted == ["wrong_paper_review"]


def test_short_fragments_are_ignored_but_one_long_fragment_must_remain():
    # "zzz" is not in the source but is under 12 chars, so it is ignored.
    ok = _parse(("wrong_paper_review", "zzz ... Reviewer 2 clearly reviewed"))
    assert _reasons(ok) == ["wrong_paper_review"]
    # Only short scraps, all present in the source: still not evidence.
    scraps = _parse(("wrong_paper_review", "Reviewer 2 ... the review"))
    assert scraps.reasons == []
    assert scraps.dropped_unquoted == ["wrong_paper_review"]


def test_a_long_bracket_is_not_an_elision():
    """Brackets over 20 chars are kept as text, so the quote no longer matches."""
    quote = "Reviewer 2 clearly reviewed [this is a long inserted remark] a different paper"
    assert _parse(("wrong_paper_review", quote)).reasons == []


# ---------------------------------------------------------------------------
# Dedupe and ordering
# ---------------------------------------------------------------------------
def test_reasons_are_deduplicated_keeping_the_first_found_quote_in_registry_order():
    result = _parse(
        ("missing_material_claim", QUOTE["missing_material_claim"]),
        ("wrong_paper_review", "a quote that is nowhere in the email"),
        ("wrong_paper_review", QUOTE["wrong_paper_review"]),
        ("wrong_paper_review", "discusses graph neural networks for protein folding"),
    )
    assert _reasons(result) == ["wrong_paper_review", "missing_material_claim"]
    assert result.reasons[0].quote == QUOTE["wrong_paper_review"]
    # Kept through another quote, so it was not dropped.
    assert result.dropped_unquoted == []


# ---------------------------------------------------------------------------
# Fallback-only reasons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fallback", ["reconsideration_only", "other"])
def test_a_fallback_reason_is_dropped_next_to_a_specific_one(fallback):
    result = _parse(
        (fallback, QUOTE["reconsideration_only"]),
        ("missing_material_claim", QUOTE["missing_material_claim"]),
    )
    assert _reasons(result) == ["missing_material_claim"]


@pytest.mark.parametrize("fallback", ["reconsideration_only", "other"])
def test_a_fallback_reason_alone_is_kept(fallback):
    assert _reasons(_parse((fallback, QUOTE["reconsideration_only"]))) == [fallback]


def test_other_outranks_reconsideration_only():
    """`other` is a specific reason, so `reconsideration_only` yields to it."""
    result = _parse(
        ("reconsideration_only", QUOTE["reconsideration_only"]),
        ("other", QUOTE["decision_vs_reviews"]),
    )
    assert _reasons(result) == ["other"]


def test_a_fallback_reason_survives_when_the_specific_one_had_no_quote():
    result = _parse(
        ("reconsideration_only", QUOTE["reconsideration_only"]),
        ("reviewer_misjudgment", "a quote that is nowhere in the email"),
    )
    assert _reasons(result) == ["reconsideration_only"]
    assert result.dropped_unquoted == ["reviewer_misjudgment"]


# ---------------------------------------------------------------------------
# Hold rule and must_verify
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trigger", ["wrong_paper_review", "record_error"])
def test_decision_vs_reviews_is_held_behind_a_verifiable_error(trigger):
    result = _parse(
        ("decision_vs_reviews", QUOTE["decision_vs_reviews"]),
        (trigger, QUOTE[trigger]),
    )
    assert _reasons(result) == [trigger]
    assert result.must_verify is True


def test_decision_vs_reviews_is_kept_otherwise():
    result = _parse(
        ("decision_vs_reviews", QUOTE["decision_vs_reviews"]),
        ("missing_material_claim", QUOTE["missing_material_claim"]),
    )
    assert _reasons(result) == ["missing_material_claim", "decision_vs_reviews"]
    assert result.must_verify is False


def test_must_verify_is_serialized():
    result = _parse(("record_error", QUOTE["record_error"]))
    assert result.model_dump()["must_verify"] is True


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
def test_papers_are_normalized_to_bare_deduplicated_digits():
    result = _parse(papers=["#1234", "1234", "Paper 77", "not a number", 0, 56])
    assert result.papers == ["1234", "77", "56"]


def test_non_list_papers_read_as_empty():
    text = json.dumps({"relation": "appeal", "papers": "1234", "reasons": []})
    assert pac.parse_answer(text, SOURCE).papers == []


# ---------------------------------------------------------------------------
# Input shapes
# ---------------------------------------------------------------------------
def test_single_message_prompt_carries_subject_and_capped_body():
    body = "x" * (pac._BODY_CAP_CHARS + 500)
    user = pac.build_user_prompt("Subj", body)
    assert user.startswith("Subject: Subj\nBody:\n")
    assert user.count("x") == pac._BODY_CAP_CHARS


def test_the_transcript_replaces_the_body():
    user = pac.build_user_prompt("Subj", "latest only", "TURN-1\nTURN-2")
    assert user == "Subject: Subj\nConversation (oldest to newest):\nTURN-1\nTURN-2"


@pytest.mark.asyncio
async def test_quotes_are_checked_against_the_prompt_the_model_saw(monkeypatch):
    """A quote from earlier in the thread counts; the latest body alone is not the source."""
    quote = "the review discusses an unrelated benchmark entirely"
    calls = _route_local(monkeypatch, _answer(reasons=[("wrong_paper_review", quote)]))
    result = await pac.classify_phase1_appeal({
        "subject": "Appeal",
        "body": "Any update?",
        "thread_transcript": f"[author] {quote}\n[author] Any update?",
    })
    assert len(calls) == 1
    assert _reasons(result) == ["wrong_paper_review"]


# ---------------------------------------------------------------------------
# Dispatch and never-raises
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["fallback", "template", "unknown_provider"])
async def test_no_real_llm_is_None_without_a_call(monkeypatch, provider):
    monkeypatch.setattr(settings, "MODEL_PROVIDER", provider)
    calls: list[str] = []

    async def record(user):  # noqa: ANN001
        calls.append(user)
        return _answer()

    monkeypatch.setattr(pac, "_call_local", record)
    monkeypatch.setattr(pac, "_call_anthropic", record)
    assert await pac.classify_phase1_appeal({"subject": "s", "body": "b"}) is None
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "anthropic_api"])
async def test_anthropic_providers_route_to_the_anthropic_branch(monkeypatch, provider):
    local_calls: list[str] = []

    async def fake_anthropic(user):  # noqa: ANN001
        return _answer(relation="feedback_only")

    async def record_local(user):  # noqa: ANN001
        local_calls.append(user)
        return _answer()

    monkeypatch.setattr(settings, "MODEL_PROVIDER", provider)
    monkeypatch.setattr(pac, "_call_anthropic", fake_anthropic)
    monkeypatch.setattr(pac, "_call_local", record_local)
    result = await pac.classify_phase1_appeal({"subject": "s", "body": "b"})
    assert result.relation == "feedback_only"
    assert local_calls == []


@pytest.mark.asyncio
async def test_a_transport_exception_is_None_and_never_raises(monkeypatch):
    async def boom(user):  # noqa: ANN001
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(settings, "MODEL_PROVIDER", "local")
    monkeypatch.setattr(pac, "_call_local", boom)
    assert await pac.classify_phase1_appeal({"subject": "s", "body": "b"}) is None


@pytest.mark.asyncio
async def test_an_invalid_model_answer_is_None(monkeypatch):
    _route_local(monkeypatch, _answer(relation="maybe"))
    assert await pac.classify_phase1_appeal({"subject": "s", "body": "b"}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("email_data", [{}, {"subject": None, "body": None}, None])
async def test_odd_email_data_never_raises(monkeypatch, email_data):
    _route_local(monkeypatch, _answer())
    result = await pac.classify_phase1_appeal(email_data)
    assert result is None or result.relation == "appeal"


@pytest.mark.asyncio
async def test_local_request_uses_configured_model_and_params(monkeypatch):
    """The payload reaching post_chat: model id from config, system + user messages."""
    sent: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": _answer()}}]}

    async def fake_post_chat(client, url, payload, headers):  # noqa: ANN001
        sent.update(url=url, payload=payload)
        return FakeResponse()

    monkeypatch.setattr(settings, "MODEL_PROVIDER", "local")
    monkeypatch.setattr(pac, "post_chat", fake_post_chat)
    result = await pac.classify_phase1_appeal({"subject": "s", "body": "b"})
    assert result is not None and result.relation == "appeal"
    payload = sent["payload"]
    assert sent["url"].endswith("/chat/completions")
    assert payload["model"] == settings.LOCAL_MODEL_NAME
    assert payload["messages"] == [
        {"role": "system", "content": pac.SYSTEM_PROMPT},
        {"role": "user", "content": "Subject: s\nBody:\nb"},
    ]
    assert payload["max_tokens"] == 4000
    assert payload["temperature"] == settings.DRAFTER_TEMPERATURE
    assert payload["seed"] == settings.DRAFTER_SEED
