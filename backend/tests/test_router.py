"""Unit tests for the EmailRouter two-lane decision (no DB, no API).

The FAQ lane is now a property of the generated *draft* (completeness,
grounding, and the drafter's self-rated answer confidence), not the email's
classified intent — EXCEPT for ``SENSITIVE_INTENTS``, which is the one lever
that overrides that gate and now holds ``desk_reject_appeal`` until Phase 3
reply templates exist (see router.py and reject_appeal.md D13).
"""

from app.pipeline.classifier import ClassificationResult
from app.pipeline.drafter import DraftResponse
from app.pipeline.router import (
    LANE_FAQ,
    LANE_HUMAN_REVIEW,
    EmailRouter,
    RoutingDecision,
    apply_self_sufficiency_floor,
)


def _draft(placeholders=None, notes=None, citations=("policy_101",), conf=0.95):
    return DraftResponse(
        draft_text="ok",
        notes_for_chair=notes,
        placeholders=list(placeholders or []),
        citations=list(citations),
        model_used="m",
        answer_confidence=conf,
    )


def _clf(intent="submission_requirements", confidence=0.9):
    return ClassificationResult(
        intent=intent, confidence=confidence, reasoning="t", method="test"
    )


def test_complete_grounded_confident_draft_is_faq():
    r = EmailRouter().route(_clf(), ["c"], _draft())
    assert r.lane == "faq"


def test_placeholder_forces_human():
    r = EmailRouter().route(_clf(), ["c"], _draft(placeholders=["date"]))
    assert r.lane == "human_review"


def test_notes_force_human():
    r = EmailRouter().route(_clf(), ["c"], _draft(notes="verify X"))
    assert r.lane == "human_review"


def test_ungrounded_forces_human():
    r = EmailRouter().route(_clf(), [], _draft(citations=()))
    assert r.lane == "human_review"


def test_low_answer_confidence_forces_human():
    r = EmailRouter().route(_clf(), ["c"], _draft(conf=0.4))
    assert r.lane == "human_review"


def test_none_answer_confidence_forces_human():
    r = EmailRouter().route(_clf(), ["c"], _draft(conf=None))
    assert r.lane == "human_review"


# --- SENSITIVE_INTENTS hold (Step 9c) ---------------------------------------
#
# ⚠️ INVERTED. This used to assert that a complete appeal draft REACHED the FAQ
# lane, back when SENSITIVE_INTENTS was empty. `desk_reject_appeal` is now held
# until Phase 3 reply templates exist (reject_appeal.md D13): the draft may be
# complete and grounded in the E011 sense while the policy stance it takes is
# one the chairs have not signed off on.


def test_desk_reject_appeal_is_held_even_with_a_perfect_draft():
    """The hold must beat the draft-quality gate, not merely tie with it.

    Uses a draft that passes EVERY FAQ condition — no placeholders, no notes,
    grounded, high classifier confidence, high answer confidence — so the only
    thing that can produce human_review is the sensitive-intent override. A
    weaker draft would route to human review anyway and prove nothing.
    """
    r = EmailRouter().route(
        _clf(intent="desk_reject_appeal", confidence=0.99), ["c"], _draft(conf=0.99)
    )
    assert r.lane == LANE_HUMAN_REVIEW
    assert r.override_reason == (
        "Intent 'desk_reject_appeal' always requires human review"
    )


def test_the_held_reason_explains_itself_to_a_chair():
    """The rationale panel shows this text; "is force-escalated" explains nothing."""
    r = EmailRouter().route(_clf(intent="desk_reject_appeal"), ["c"], _draft())
    assert "reject appeals require chair review" in r.reason


def test_a_sensitive_intent_without_a_reason_entry_still_routes():
    """Adding to SENSITIVE_INTENTS must never be able to crash the router.

    The reason lookup falls back to generic phrasing, so a future entry added
    without a matching `_SENSITIVE_INTENT_REASONS` line degrades to a dull
    message rather than a KeyError in the routing path.
    """
    import app.pipeline.router as router_module

    original = router_module.SENSITIVE_INTENTS
    router_module.SENSITIVE_INTENTS = ["anonymity_violation"]
    try:
        r = EmailRouter().route(_clf(intent="anonymity_violation"), ["c"], _draft())
    finally:
        router_module.SENSITIVE_INTENTS = original
    assert r.lane == LANE_HUMAN_REVIEW
    assert "anonymity_violation" in r.reason


def test_non_sensitive_intents_still_reach_the_faq_lane():
    """The hold is scoped to ONE intent — it must not become a blanket block.

    `review_decision_appeal` is checked explicitly: it is the neighbouring
    appeal intent and the one most likely to be swept in by a careless edit.
    """
    for intent in ("submission_requirements", "review_decision_appeal", "cms_support"):
        r = EmailRouter().route(_clf(intent=intent), ["c"], _draft())
        assert r.lane == LANE_FAQ, intent
        assert r.override_reason is None, intent


def _faq_routing():
    return RoutingDecision(
        lane=LANE_FAQ, reason="stub faq", confidence_used=0.9, threshold_applied=0.65
    )


def test_apply_self_sufficiency_floor_demotes_faq_with_placeholders():
    routing = apply_self_sufficiency_floor(_faq_routing(), _draft(placeholders=["date"]))
    assert routing.lane == LANE_HUMAN_REVIEW
    assert routing.override_reason == (
        "draft is not self-sufficient (1 placeholder(s), notes=no, "
        "answer_confidence=set, grounded) — requires a human"
    )


def test_apply_self_sufficiency_floor_demotes_faq_with_notes():
    routing = apply_self_sufficiency_floor(_faq_routing(), _draft(notes="verify X"))
    assert routing.lane == LANE_HUMAN_REVIEW
    assert routing.override_reason == (
        "draft is not self-sufficient (0 placeholder(s), notes=yes, "
        "answer_confidence=set, grounded) — requires a human"
    )


def test_apply_self_sufficiency_floor_demotes_faq_with_no_answer_confidence():
    routing = apply_self_sufficiency_floor(_faq_routing(), _draft(conf=None))
    assert routing.lane == LANE_HUMAN_REVIEW
    assert routing.override_reason == (
        "draft is not self-sufficient (0 placeholder(s), notes=no, "
        "answer_confidence=none, grounded) — requires a human"
    )


def test_apply_self_sufficiency_floor_demotes_faq_with_no_citations():
    routing = apply_self_sufficiency_floor(_faq_routing(), _draft(citations=()))
    assert routing.lane == LANE_HUMAN_REVIEW
    assert routing.override_reason == (
        "draft is not self-sufficient (0 placeholder(s), notes=no, "
        "answer_confidence=set, ungrounded) — requires a human"
    )


def test_apply_self_sufficiency_floor_is_noop_on_self_sufficient_draft():
    routing = _faq_routing()
    result = apply_self_sufficiency_floor(routing, _draft())
    assert result.lane == LANE_FAQ
    assert result.override_reason is None
    assert result == routing
