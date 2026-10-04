"""The appeal reply hook with the PHASE-1 reason source (reject-appeal Phase 4, P3).

``prepare_appeal_draft(..., mapped=map_phase1(outcome))`` against the REAL approved
template file. Every expected text and record is a hand-written literal. The
appeal_reason source (``mapped`` None) is covered, unchanged, by
test_appeal_reply_hook.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.pipeline.appeal_reply_hook import prepare_appeal_draft
from app.pipeline.phase1_appeal_classifier import AppealReason, Phase1AppealResult
from app.pipeline.phase1_appeal_outcome import (
    CLASSIFIED,
    FAILED,
    GATE_NOT_MET,
    NOT_APPEAL,
    Phase1Outcome,
)
from app.pipeline.phase1_reply_mapping import map_phase1

REVIEW, DESK = "review_decision_appeal", "desk_reject_appeal"
INSIDE = "2026-09-15T10:00:00Z"
WINDOW_END = datetime(2026, 11, 1, tzinfo=timezone.utc)

OPENING = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:"
)
P_SCORES = (
    "Decisions are not based solely on the visible reviewer scores. Senior program committee members "
    "evaluated both the paper and the reviews, including whether the raised concerns can be addressed "
    "with minor clarifications or require substantial revision."
)
P_REVIEWERS = "Decisions are not based on any single review; all assessments are weighed together."
P_REBUTTAL = (
    "AAAI's two-phase process forgoes rebuttal for Phase 1 papers in favor of a quicker decision. We "
    "understand this can be frustrating, but Phase 1 decisions are final and will not be revisited in "
    "response to author objections."
)
P_THANKS = (
    "Thank you for sharing your view of the review process. We will consider your input when studying "
    "possible changes for future editions."
)
CLOSING = (
    "The decision is final, but we hope the feedback will be useful in further strengthening your work "
    "and helping you secure publication in another leading venue or future AAAI edition."
)
SIGN = "Best Regards,\nAAAI 2027 PC Team"
DRAFT_T1 = (f"Dear Jane Doe,\n\n{OPENING}\n\n(1) {P_SCORES}\n\n(2) {P_REBUTTAL}\n\n{CLOSING}\n\n{SIGN}")
DRAFT_T1_T2 = (f"Dear Jane Doe,\n\n{OPENING}\n\n(1) {P_SCORES}\n\n(2) {P_REVIEWERS}\n\n"
               f"(3) {P_REBUTTAL}\n\n(4) {P_THANKS}\n\n{CLOSING}\n\n{SIGN}")
YAN_A_FIRST = ("Dear Jane Doe,\n\nThank you for taking the time to share your concerns regarding the "
               "review process for your submission.")
YAN_B_FIRST = ("Dear Jane Doe,\n\nThank you for providing the detailed information regarding your "
               "concerns about the reviews of your submission.")
T1_BLOCKS = ["opening_warm", "lead_in_concerns", "point_scores", "point_rebuttal", "closing_reviewed"]
T1_T2_BLOCKS = ["opening_warm", "lead_in_concerns", "point_scores", "point_all_assessments",
                "point_rebuttal", "point_consider_input", "closing_reviewed"]


def classified(relation, reasons, papers=("12345",)) -> Phase1Outcome:
    return Phase1Outcome(CLASSIFIED, Phase1AppealResult(
        relation=relation, papers=list(papers),
        reasons=[AppealReason(reason=r, quote=f"the author's words for {r}") for r in reasons],
        dropped_unquoted=[]))


def snap(state, relation=None, reasons=None, must_verify=None, papers=None) -> dict:
    return {"state": state, "relation": relation, "reasons": reasons,
            "must_verify": must_verify, "papers": papers}


def cls(relation, reasons, must_verify=False, papers=("12345",)) -> dict:
    return snap("classified", relation, list(reasons), must_verify, list(papers))


def run(intent, outcome, *, reciprocal=None, created=INSIDE, window_end=None, appeal_reason=None):
    data = {"subject": "s", "body": "b", "sender_name": "Jane Doe"}
    if created is not None:
        data["timestamp"] = created
    return prepare_appeal_draft(intent, appeal_reason, reciprocal, data, window_end=window_end,
                                mapped=map_phase1(outcome))


# --- composed ---------------------------------------------------------------------------------
@pytest.mark.parametrize("reasons, text, record_reasons, blocks, mode", [
    (["decision_vs_reviews"], DRAFT_T1, ["score_outcome_mismatch"], T1_BLOCKS, "merged"),
    (["decision_vs_reviews", "reviewer_misjudgment"], DRAFT_T1_T2,
     ["score_outcome_mismatch", "reviewer_misunderstanding"], T1_T2_BLOCKS, "merged"),
], ids=["T1", "T1+T2-merge"])
def test_composed_replies_from_his_reasons(reasons, text, record_reasons, blocks, mode):
    draft, rec = run(REVIEW, classified("appeal", reasons))
    assert draft.draft_text == text
    assert (draft.notes_for_chair, draft.placeholders) == (None, [])
    assert rec == {"mode": mode, "reasons": record_reasons, "block_ids": blocks,
                   "source": "phase1", "phase1": cls("appeal", reasons)}


@pytest.mark.parametrize("reason, first, block", [
    ("reconsideration_only", YAN_A_FIRST, "standalone_general_stage1"),
    ("llm_generated_review", YAN_B_FIRST, "standalone_ai_review"),
], ids=["yan-a", "yan-b"])
def test_yans_two_replies_from_his_reasons(reason, first, block):
    draft, rec = run(REVIEW, classified("appeal", [reason]))
    assert draft.draft_text.startswith(first) and draft.draft_text.endswith(f"\n\n{SIGN}")
    assert (rec["mode"], rec["block_ids"], rec["source"]) == ("standalone", [block], "phase1")


@pytest.mark.parametrize("reasons, note", [
    (["decision_vs_reviews", "llm_generated_review"],
     "Chair writes: no approved reply covers these reasons together: "
     "score_outcome_mismatch, llm_generated_review."),
    (["reviewer_misjudgment", "reconsideration_only"],
     "Chair writes: no approved reply covers these reasons together: "
     "reviewer_misunderstanding, general_dissatisfaction."),
    (["llm_generated_review", "reconsideration_only"],
     "Chair writes: no approved reply covers these reasons together: "
     "llm_generated_review, general_dissatisfaction."),
], ids=["yan-b+T1", "yan-a+T2", "yan-a+yan-b"])
def test_a_yan_reply_mixed_with_anything_goes_to_the_chair(reasons, note):
    draft, rec = run(REVIEW, classified("appeal", reasons))
    assert (draft.draft_text, draft.notes_for_chair) == ("[CHAIR: write reply]", note)
    assert (rec["mode"], rec["block_ids"]) == ("chair_writes", ["line_chair_writes"])


# --- holds ------------------------------------------------------------------------------------
HOLD_CASES = [
    # id, outcome, text, notes, mode, block_ids, phase1 snapshot
    ("flag-off", None, "[CHAIR: write reply]",
     "Chair writes: the phase-1 appeal classifier is turned off, so the appeal reasons were not "
     "determined.", "reason_unknown", [], snap("flag_off")),
    ("gate-not-met", Phase1Outcome(GATE_NOT_MET), "[CHAIR: write reply]",
     "Chair writes: this email is outside the phase-1 appeal classification (intent or ticket "
     "date), so its reasons were not determined.", "reason_unknown", [], snap("gate_not_met")),
    ("failed", Phase1Outcome(FAILED), "[CHAIR: write reply]",
     "Chair writes: the phase-1 appeal classifier did not return an answer.", "reason_unknown", [],
     snap("failed")),
    ("not-appeal", Phase1Outcome(NOT_APPEAL), "[CHAIR: write reply]",
     "Chair writes: the phase-1 classifier judged this email not to be an appeal (for example, a "
     "request to see the reviews or a reply without a request).", "not_appeal", [],
     snap("not_appeal", "not_appeal")),
    ("wrong-paper", classified("appeal", ["wrong_paper_review", "decision_vs_reviews"]),
     "[CHAIR: do not reply yet; see note]",
     "Investigate first: the author says a review is about a different paper. Do not reply to or "
     "close the ticket yet.\nAlso raised: decision_vs_reviews.", "no_draft", [],
     cls("appeal", ["wrong_paper_review", "decision_vs_reviews"], True)),
    ("record-error", classified("appeal", ["record_error"]),
     "[CHAIR: do not reply yet; see note]",
     "Investigate first: the author says a rating contradicts its own review, or that a submitted "
     "review was left out of the decision. Do not reply to or close the ticket yet.", "no_draft", [],
     cls("appeal", ["record_error"], True)),
    ("feedback-only", classified("feedback_only", ["reviewer_misjudgment"]),
     "[CHAIR: write reply]",
     "Chair writes: the author reports a review problem but says they are not asking for a "
     "change.\nRaised: reviewer_misjudgment.", "chair_writes", ["line_chair_writes"],
     cls("feedback_only", ["reviewer_misjudgment"])),
    ("chair-write-reason-with-T1-and-T2",
     classified("appeal", ["decision_vs_reviews", "reviewer_misjudgment", "missing_material_claim"]),
     "[CHAIR: write reply]",
     "Chair writes: no approved reply covers missing_material_claim.\n"
     "Also raised: decision_vs_reviews, reviewer_misjudgment.", "chair_writes", ["line_chair_writes"],
     cls("appeal", ["decision_vs_reviews", "reviewer_misjudgment", "missing_material_claim"])),
    ("two-papers", classified("appeal", ["decision_vs_reviews"], papers=("11111", "22222")),
     "[CHAIR: write reply]",
     "Chair writes: the email is about 2 papers; the approved replies are written for one paper.\n"
     "Raised: decision_vs_reviews.", "chair_writes", ["line_chair_writes"],
     cls("appeal", ["decision_vs_reviews"], papers=("11111", "22222"))),
]


@pytest.mark.parametrize("outcome, text, notes, mode, blocks, phase1",
                         [c[1:] for c in HOLD_CASES], ids=[c[0] for c in HOLD_CASES])
def test_holds(outcome, text, notes, mode, blocks, phase1):
    draft, rec = run(REVIEW, outcome)
    assert (draft.draft_text, draft.notes_for_chair) == (text, notes)
    assert draft.placeholders and draft.model_used == "none"
    assert rec == {"mode": mode, "reasons": None, "block_ids": blocks,
                   "source": "phase1", "phase1": phase1}


# --- order: reciprocal, then desk reject, then the mapping ----------------------------------------
def test_a_non_reciprocal_desk_reject_appeal_with_no_reasons_is_desk_reject_not_reason_unknown():
    """The phase-1 classifier never answers a desk-reject appeal (its gate is the
    review-decision intent), so the desk-reject rule must come before the reasons."""
    draft, rec = run(DESK, Phase1Outcome(GATE_NOT_MET), reciprocal=False)
    assert draft.draft_text == "[CHAIR: write reply]"
    assert draft.notes_for_chair == ("Chair writes: desk-rejection appeals get no composed reply, "
                                     "because the approved wording assumes the paper was reviewed.")
    assert rec == {"mode": "desk_reject", "reasons": None, "block_ids": [],
                   "source": "phase1", "phase1": snap("gate_not_met")}
    assert run(DESK, None, reciprocal=None)[1]["mode"] == "desk_reject"


def test_reciprocal_comes_first_even_over_composable_reasons():
    draft, rec = run(REVIEW, classified("appeal", ["decision_vs_reviews"]), reciprocal=True)
    assert draft.draft_text == "[CHAIR: reciprocal complaint; see note]"
    assert draft.notes_for_chair == ("Reciprocal-review complaint: tagged for Marc to review himself. "
                                     "No reply is drafted.\nAlso raised: score_outcome_mismatch.")
    assert rec == {"mode": "reciprocal_review",
                   "reasons": ["score_outcome_mismatch", "reciprocal_dispute"], "block_ids": [],
                   "source": "phase1", "phase1": cls("appeal", ["decision_vs_reviews"])}
    assert run(DESK, Phase1Outcome(GATE_NOT_MET), reciprocal=True)[1]["mode"] == "reciprocal_review"


def test_appeal_reason_is_ignored_with_the_phase1_source():
    """Our own classifier's answer says wrong paper; the phase-1 outcome says T1: T1 wins."""
    draft, _ = run(REVIEW, classified("appeal", ["decision_vs_reviews"]),
                   appeal_reason=["wrong_paper_review"])
    assert draft.draft_text == DRAFT_T1


# --- the window, last, composed text only -------------------------------------------------------
def test_the_window_replaces_composed_text_only():
    late = "2026-12-01T10:00:00Z"
    draft, rec = run(REVIEW, classified("appeal", ["decision_vs_reviews"]), created=late,
                     window_end=WINDOW_END)
    assert (draft.draft_text, draft.notes_for_chair) == (
        "[CHAIR: write reply]", "Appeal reply wording is for Phase 1 rejections only")
    assert (rec["mode"], rec["reasons"], rec["source"]) == ("window", ["score_outcome_mismatch"], "phase1")
    assert run(REVIEW, classified("appeal", ["record_error"]), created=late,
               window_end=WINDOW_END)[1]["mode"] == "no_draft"
    assert run(REVIEW, classified("appeal", ["other"]), created=late,
               window_end=WINDOW_END)[1]["mode"] == "chair_writes"
    assert run(REVIEW, classified("appeal", ["decision_vs_reviews"]), created=None,
               window_end=WINDOW_END)[1]["mode"] == "window"
    assert run(REVIEW, classified("appeal", ["decision_vs_reviews"]), created=late,
               window_end=None)[1]["mode"] == "merged", "no window set by default"


def test_a_failure_inside_keeps_the_phase1_source_on_the_record(monkeypatch):
    from app.pipeline import appeal_reply_hook as hook

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(hook, "compose_reply", boom)
    draft, rec = run(REVIEW, classified("appeal", ["decision_vs_reviews"]))
    assert draft.draft_text == "[CHAIR: write reply]"
    assert draft.notes_for_chair == "Chair writes: the appeal reply could not be prepared automatically."
    assert rec == {"mode": "failed", "reasons": ["score_outcome_mismatch"], "block_ids": [],
                   "source": "phase1", "phase1": cls("appeal", ["decision_vs_reviews"])}


def test_a_non_appeal_intent_is_still_not_handled():
    assert run("submission_requirements", classified("appeal", ["decision_vs_reviews"])) is None
