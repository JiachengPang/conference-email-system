"""Router (the two-lane decision).

Decides whether an email is answered automatically (FAQ lane) or escalated to a
human chair (human-review lane). The FAQ lane is a property of the generated
*draft*, not the email's classified intent: auto-reply requires a COMPLETE
draft (no chair placeholders, no notes for the chair), GROUNDED in retrieved
policy (at least one citation), sufficient classifier confidence, AND
sufficient drafter self-rated answer confidence. Every condition must hold —
if any one fails, the email is escalated to a human with a specific reason.

The thresholds are read from settings (FAQ_CONFIDENCE_THRESHOLD,
FAQ_ANSWER_CONFIDENCE_THRESHOLD) so they are tunable without code changes. The
`strategy` flag is the seam for an RL router later.
"""

from pydantic import BaseModel, Field

from app.core.config import settings
from app.pipeline.classifier import ClassificationResult

# Intents that ALWAYS require a human regardless of draft quality. This is the
# one lever that overrides the draft-quality FAQ gate, so it is populated
# deliberately and sparingly. Vocabulary source: app.pipeline.taxonomy.
#
# `desk_reject_appeal` is held until Phase 3 reply templates exist
# (reject_appeal.md D13). A desk-rejection appeal is answerable in the E011
# sense — a complete, grounded "no" is a real answer — but the CONTENT of that
# answer is a policy stance the chairs have not signed off on yet, and the
# largest bucket behind this intent is reciprocal-review disputes, where the
# template would restate the rule that rejected the requester rather than
# engage the dispute.
#
# ⚠️ DELIBERATELY OVER-BROAD. The hold is keyed on the INTENT, which catches
# every desk-reject appeal (formatting, page-limit, checklist…) and not only
# the reciprocal ones. Holding on `is_reciprocal_dispute` instead is not
# possible here: `route()` receives (classification, retrieved_chunks, draft)
# and never sees `extraction`, so a flag-level hold would mean making the
# router depend on the extractor. Over-holding is the cheap, reversible side
# of that trade — this is one line to undo.
#
# `review_decision_appeal` is held for the same reason (reject_appeal.md D62):
# a post-review appeal gets a reply whose policy stance the chairs have not
# signed off on either. It is where the live appeal volume is (D73).
SENSITIVE_INTENTS: list[str] = ["desk_reject_appeal", "review_decision_appeal"]

# Why each held intent is held, in words a chair reads in the routing-rationale
# panel. A bare "is force-escalated" tells them the system did something without
# telling them why, which is the difference between a rule and an unexplained
# refusal. Falls back to the generic phrasing for any intent added without an
# entry, so adding to SENSITIVE_INTENTS can never crash the router.
_SENSITIVE_INTENT_REASONS: dict[str, str] = {
    "desk_reject_appeal": (
        "reject appeals require chair review — the reply states a policy "
        "stance that has not been signed off yet"
    ),
    "review_decision_appeal": (
        "review-decision appeals require chair review — the reply states a "
        "policy stance that has not been signed off yet"
    ),
}

LANE_FAQ = "faq"
LANE_HUMAN_REVIEW = "human_review"


class RoutingDecision(BaseModel):
    """Output of the router — the lane and a transparent rationale."""

    lane: str = Field(..., description='"faq" or "human_review".')
    reason: str = Field(..., description="Human-readable explanation of the lane.")
    confidence_used: float = Field(
        ..., description="Classifier confidence considered in the decision."
    )
    threshold_applied: float = Field(
        ..., description="FAQ confidence threshold that was applied."
    )
    override_reason: str | None = Field(
        default=None,
        description="Set when a hard rule forced the lane (e.g. sensitive intent).",
    )


def apply_self_sufficiency_floor(routing: "RoutingDecision", draft) -> "RoutingDecision":
    """Strategy-independent safety floor: a draft that is not self-sufficient
    (chair placeholders, notes-for-chair, an unrated answer confidence, or an
    ungrounded citation set) can NEVER be auto-answered, whatever the router
    returned. The rule_based router's draft-quality gate already enforces this;
    this floor also covers the RL strategy, which routes without ever seeing
    the draft. Deliberately redundant for rule_based; load-bearing for rl.
    Returns routing unchanged when the draft is self-sufficient."""
    not_self_sufficient = (
        bool(draft.placeholders)
        or bool(draft.notes_for_chair)
        or draft.answer_confidence is None
        or not draft.citations
    )
    if not_self_sufficient and routing.lane != LANE_HUMAN_REVIEW:
        return routing.model_copy(update={
            "lane": LANE_HUMAN_REVIEW,
            "override_reason": (
                "draft is not self-sufficient ("
                f"{len(draft.placeholders)} placeholder(s), "
                f"notes={'yes' if draft.notes_for_chair else 'no'}, "
                f"answer_confidence={'none' if draft.answer_confidence is None else 'set'}, "
                f"{'grounded' if draft.citations else 'ungrounded'}) — requires a human"
            ),
        })
    return routing


class EmailRouter:
    """Threshold-and-rule router for the two-lane workflow."""

    def __init__(self, strategy: str = "threshold") -> None:
        self.strategy = strategy

    def route(
        self,
        classification: ClassificationResult,
        retrieved_chunks: list,
        draft,
    ) -> RoutingDecision:
        """Pick a lane from the classification and the generated draft's quality."""
        threshold = settings.FAQ_CONFIDENCE_THRESHOLD
        answer_threshold = settings.FAQ_ANSWER_CONFIDENCE_THRESHOLD
        intent = classification.intent
        # Prefer the calibrated confidence when the classifier attached one
        # (calibration enabled + a fitted artifact exists); otherwise use the
        # raw score. This changes only WHICH confidence value is compared — the
        # threshold logic below is untouched.
        confidence = (
            classification.calibrated_confidence
            if classification.calibrated_confidence is not None
            else classification.confidence
        )

        # RL strategy: delegate the lane choice to the learning bandit. It keeps
        # the same RoutingDecision contract and its own hard safety guards
        # (sensitive intents + a low-confidence floor). Imported lazily to avoid
        # a circular import (rl_router imports RoutingDecision from this module).
        if self.strategy == "rl":
            from app.pipeline.rl_router import get_rl_router

            return get_rl_router().route(intent, confidence, threshold)

        # Seam (empty by default) — force certain intents to a human if ever needed.
        if intent in SENSITIVE_INTENTS:
            why = _SENSITIVE_INTENT_REASONS.get(
                intent, f"'{intent}' always requires human review"
            )
            return RoutingDecision(
                lane=LANE_HUMAN_REVIEW,
                reason=f"Routed to human review: {why}.",
                confidence_used=confidence,
                threshold_applied=threshold,
                override_reason=f"Intent '{intent}' always requires human review",
            )

        # Draft-quality FAQ gate: every condition must hold.
        complete = not draft.placeholders and not draft.notes_for_chair
        grounded = bool(draft.citations)
        answer_conf = draft.answer_confidence
        if (
            complete
            and grounded
            and confidence >= threshold
            and answer_conf is not None
            and answer_conf >= answer_threshold
        ):
            return RoutingDecision(
                lane=LANE_FAQ,
                reason=(
                    f"Auto-reply eligible: complete grounded draft "
                    f"(answer_confidence {answer_conf:.2f} >= {answer_threshold:.2f}, "
                    f"intent confidence {confidence:.2f} >= {threshold:.2f})."
                ),
                confidence_used=confidence,
                threshold_applied=threshold,
            )

        if draft.placeholders:
            why = f"draft has {len(draft.placeholders)} chair placeholder(s)"
        elif draft.notes_for_chair:
            why = "draft has notes for the chair"
        elif not grounded:
            why = "draft cites no policy (ungrounded)"
        elif confidence < threshold:
            why = f"intent confidence {confidence:.2f} < {threshold:.2f}"
        elif answer_conf is None:
            why = "drafter provided no answer confidence"
        else:
            why = f"answer confidence {answer_conf:.2f} < {answer_threshold:.2f}"
        return RoutingDecision(
            lane=LANE_HUMAN_REVIEW,
            reason=f"Routed to human review: {why}.",
            confidence_used=confidence,
            threshold_applied=threshold,
        )
