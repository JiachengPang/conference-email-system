"""Full path for the Step 2.5 reasons: phase-1 outcome -> mapping -> hook -> draft.

``prepare_appeal_draft(..., mapped=map_phase1(outcome))`` against the REAL
approved template file, outcomes built with Jiacheng's own types. Covers each
new reason alone and with decision_vs_reviews, the wrong-paper precedence, the
Yan-reply rule (D105/D162), the reason cap, ``other``, an unknown name and the
flag off. Every expected text, note and record is a hand-written literal.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.pipeline.appeal_reply_hook import prepare_appeal_draft
from app.pipeline.phase1_appeal_classifier import AppealReason, Phase1AppealResult
from app.pipeline.phase1_appeal_outcome import CLASSIFIED, Phase1Outcome
from app.pipeline.phase1_reply_mapping import map_phase1

REVIEW = "review_decision_appeal"

OPENING = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:"
)
P_SPC = (
    "Decisions are not based on any single review. Senior program committee members evaluated both "
    "the paper and the reviews to decide their recommendation."
)
P_TRACKING = (
    "We are asking SPCs and ACs to report reviewers that are being unprofessional. We keep track of "
    "the reviewer's performance and we will consider it in future editions."
)
P_ETHICS = (
    "You can also report unethical behavior through the ethics report form at "
    "https://docs.google.com/forms/d/e/1FAIpQLSdIs72RunUy5wKsOv7SdBma6A6riv3jp8lifUxlLcwhcdXMxw/viewform. "
    "This may impact our future relationship with this reviewer, but it will not change the outcome "
    "for this specific paper."
)
P_SCORES = (
    "Decisions are not based solely on the visible reviewer scores. Senior program committee members "
    "evaluated both the paper and the reviews, including whether the raised concerns can be addressed "
    "with minor clarifications or require substantial revision."
)
P_REBUTTAL = (
    "AAAI's two-phase process forgoes rebuttal for Phase 1 papers in favor of a quicker decision. We "
    "understand this can be frustrating, but Phase 1 decisions are final and will not be revisited in "
    "response to author objections."
)
CLOSING = (
    "The decision is final, but we hope the feedback will be useful in further strengthening your work "
    "and helping you secure publication in another leading venue or future AAAI edition."
)
SIGN = "Best Regards,\nAAAI 2027 PC Team"

V_MISCONDUCT = ("Before sending, check for harassment or an undisclosed conflict of interest. "
                "If either is present, do not send; forward the ticket to the Ethics Chairs.")
V_RECORD = "Before sending, check the review text against its score."
N_WRONG = ("Investigate first: the author says a review is about a different paper. "
           "Do not reply to or close the ticket yet.")

FRAME = ["opening_warm", "lead_in_concerns"]
MISCONDUCT_POINTS = ["point_spc_evaluation", "point_reviewer_tracking", "point_ethics_form"]


def draft_of(*points: str) -> str:
    numbered = "".join(f"({n}) {p}\n\n" for n, p in enumerate(points, start=1))
    return f"Dear Jane Doe,\n\n{OPENING}\n\n{numbered}{CLOSING}\n\n{SIGN}"


def classified(reasons, relation="appeal", papers=("12345",), dropped=()) -> Phase1Outcome:
    return Phase1Outcome(CLASSIFIED, Phase1AppealResult(
        relation=relation, papers=list(papers),
        reasons=[AppealReason(reason=r, quote=f"quote for {r}") for r in reasons],
        dropped_unquoted=list(dropped),
    ))


def phase1(reasons, must_verify=False) -> dict:
    return {"state": "classified", "relation": "appeal", "reasons": list(reasons),
            "must_verify": must_verify, "papers": ["12345"]}


def run(outcome, *, reciprocal=False, created="2026-09-15T10:00:00Z", window_end=None):
    data = {"subject": "s", "body": "b", "sender_name": "Jane Doe", "timestamp": created}
    return prepare_appeal_draft(REVIEW, None, reciprocal, data, window_end=window_end,
                                mapped=map_phase1(outcome))


COMPOSED = [
    # id, his reasons, text, notes, record reasons, block ids, his must_verify
    ("misconduct-alone", ["reviewer_misconduct"],
     draft_of(P_SPC, P_TRACKING, P_ETHICS), V_MISCONDUCT,
     ["reviewer_misconduct"], [*FRAME, *MISCONDUCT_POINTS, "closing_reviewed"], False),
    ("missing-material-alone", ["missing_material_claim"],
     draft_of(P_SPC), None,
     ["missing_material_claim"], [*FRAME, "point_spc_evaluation", "closing_reviewed"], False),
    ("record-error-alone", ["record_error"],
     draft_of(P_SCORES), V_RECORD,
     ["record_error"], [*FRAME, "point_scores", "closing_reviewed"], True),
    ("misconduct-with-T1", ["reviewer_misconduct", "decision_vs_reviews"],
     draft_of(P_SPC, P_TRACKING, P_ETHICS, P_SCORES, P_REBUTTAL), V_MISCONDUCT,
     ["reviewer_misconduct", "score_outcome_mismatch"],
     [*FRAME, *MISCONDUCT_POINTS, "point_scores", "point_rebuttal", "closing_reviewed"], False),
    ("missing-material-with-T1", ["decision_vs_reviews", "missing_material_claim"],
     draft_of(P_SPC, P_SCORES, P_REBUTTAL), None,
     ["score_outcome_mismatch", "missing_material_claim"],
     [*FRAME, "point_spc_evaluation", "point_scores", "point_rebuttal", "closing_reviewed"], False),
    ("record-error-with-T1", ["decision_vs_reviews", "record_error"],
     draft_of(P_SCORES, P_REBUTTAL), V_RECORD,
     ["score_outcome_mismatch", "record_error"],
     [*FRAME, "point_scores", "point_rebuttal", "closing_reviewed"], True),
    ("misconduct-and-record-error", ["record_error", "reviewer_misconduct"],
     draft_of(P_SPC, P_TRACKING, P_ETHICS, P_SCORES), f"{V_MISCONDUCT}\n{V_RECORD}",
     ["record_error", "reviewer_misconduct"],
     [*FRAME, *MISCONDUCT_POINTS, "point_scores", "closing_reviewed"], True),
]


@pytest.mark.parametrize("reasons, text, notes, rec_reasons, blocks, must_verify",
                         [c[1:] for c in COMPOSED], ids=[c[0] for c in COMPOSED])
def test_composed_replies_for_the_new_reasons(reasons, text, notes, rec_reasons, blocks, must_verify):
    draft, rec = run(classified(reasons))
    assert draft.draft_text == text
    assert draft.notes_for_chair == notes
    assert draft.placeholders == []
    assert rec == {"mode": "merged", "reasons": rec_reasons, "block_ids": blocks,
                   "source": "phase1", "phase1": phase1(reasons, must_verify)}


HELD = [
    # id, outcome, text, notes, mode, record reasons, block ids
    ("wrong-paper-beats-record-error", classified(["record_error", "wrong_paper_review"]),
     "[CHAIR: do not reply yet; see note]", f"{N_WRONG}\nAlso raised: record_error.",
     "no_draft", None, []),
    # Held with misconduct: the misconduct check is kept (answer 2), and only that one.
    ("wrong-paper-beats-misconduct", classified(["wrong_paper_review", "reviewer_misconduct"]),
     "[CHAIR: do not reply yet; see note]",
     f"{N_WRONG}\nAlso raised: reviewer_misconduct.\n{V_MISCONDUCT}",
     "no_draft", None, []),
    ("wrong-paper-with-record-error-and-misconduct",
     classified(["wrong_paper_review", "record_error", "reviewer_misconduct"]),
     "[CHAIR: do not reply yet; see note]",
     f"{N_WRONG}\nAlso raised: record_error, reviewer_misconduct.\n{V_MISCONDUCT}",
     "no_draft", None, []),
    ("misconduct-with-other", classified(["reviewer_misconduct", "other"]),
     "[CHAIR: write reply]",
     f"Chair writes: no approved reply covers other.\nAlso raised: reviewer_misconduct.\n{V_MISCONDUCT}",
     "chair_writes", None, ["line_chair_writes"]),
    ("misconduct-feedback-only", classified(["reviewer_misconduct"], relation="feedback_only"),
     "[CHAIR: write reply]",
     "Chair writes: the author reports a review problem but says they are not asking for a "
     f"change.\nRaised: reviewer_misconduct.\n{V_MISCONDUCT}",
     "chair_writes", None, ["line_chair_writes"]),
    ("misconduct-with-two-papers", classified(["reviewer_misconduct"], papers=("11111", "22222")),
     "[CHAIR: write reply]",
     "Chair writes: the email is about 2 papers; the approved replies are written for one paper.\n"
     f"Raised: reviewer_misconduct.\n{V_MISCONDUCT}",
     "chair_writes", None, ["line_chair_writes"]),
    # record_error's check never rides on a hold.
    ("record-error-feedback-only-has-no-check", classified(["record_error"], relation="feedback_only"),
     "[CHAIR: write reply]",
     "Chair writes: the author reports a review problem but says they are not asking for a "
     "change.\nRaised: record_error.",
     "chair_writes", None, ["line_chair_writes"]),
    ("record-error-with-two-papers-has-no-check",
     classified(["record_error"], papers=("11111", "22222")),
     "[CHAIR: write reply]",
     "Chair writes: the email is about 2 papers; the approved replies are written for one paper.\n"
     "Raised: record_error.",
     "chair_writes", None, ["line_chair_writes"]),
    ("record-error-with-other-has-no-check", classified(["record_error", "other"]),
     "[CHAIR: write reply]",
     "Chair writes: no approved reply covers other.\nAlso raised: record_error.",
     "chair_writes", None, ["line_chair_writes"]),
    # No check where the email is not answered as an appeal.
    ("not-appeal-with-misconduct-has-no-check",
     classified(["reviewer_misconduct"], relation="not_appeal"),
     "[CHAIR: write reply]",
     "Chair writes: the phase-1 classifier judged this email not to be an appeal (for example, a "
     "request to see the reviews or a reply without a request).",
     "not_appeal", None, []),
    ("unverified-misconduct-has-no-check", classified([], dropped=("reviewer_misconduct",)),
     "[CHAIR: write reply]",
     "Chair writes: the author appeals, but no reason could be verified in the email.\n"
     "Possibly raised (quote not verified): reviewer_misconduct.",
     "reason_unknown", None, []),
    ("misconduct-with-yan-b-goes-to-the-chair",
     classified(["reviewer_misconduct", "llm_generated_review"]),
     "[CHAIR: write reply]",
     "Chair writes: no approved reply covers these reasons together: reviewer_misconduct, "
     f"llm_generated_review.\n{V_MISCONDUCT}",
     "chair_writes", ["reviewer_misconduct", "llm_generated_review"], ["line_chair_writes"]),
    ("four-reasons-go-to-the-chair",
     classified(["record_error", "decision_vs_reviews", "missing_material_claim",
                 "reviewer_misconduct"]),
     "[CHAIR: write reply]",
     "Chair writes: more than 3 issues raised: reviewer_misconduct, missing_material_claim, "
     f"score_outcome_mismatch, record_error.\n{V_MISCONDUCT}\n{V_RECORD}",
     "chair_writes",
     ["record_error", "score_outcome_mismatch", "missing_material_claim", "reviewer_misconduct"],
     ["line_chair_writes"]),
    ("other-alone", classified(["other"]),
     "[CHAIR: write reply]", "Chair writes: no approved reply covers other.",
     "chair_writes", None, ["line_chair_writes"]),
    ("unknown-name", classified(["future_reason"]),
     "[CHAIR: write reply]", "Chair writes: no approved reply covers future_reason.",
     "chair_writes", None, ["line_chair_writes"]),
    ("flag-off", None,
     "[CHAIR: write reply]",
     "Chair writes: the phase-1 appeal classifier is turned off, so the appeal reasons were not "
     "determined.", "reason_unknown", None, []),
]


@pytest.mark.parametrize("outcome, text, notes, mode, rec_reasons, blocks",
                         [c[1:] for c in HELD], ids=[c[0] for c in HELD])
def test_the_new_reasons_never_override_the_precedence_rules(outcome, text, notes, mode,
                                                             rec_reasons, blocks):
    draft, rec = run(outcome)
    assert (draft.draft_text, draft.notes_for_chair) == (text, notes)
    assert draft.placeholders, "every held draft is blocked at approve"
    assert (rec["mode"], rec["reasons"], rec["block_ids"]) == (mode, rec_reasons, blocks)


RECIPROCAL_NOTE = "Reciprocal-review complaint: tagged for Marc to review himself. No reply is drafted."


@pytest.mark.parametrize("outcome, also", [
    (classified(["record_error"]), "\nAlso raised: record_error."),
    (classified(["reviewer_misconduct"]), "\nAlso raised: reviewer_misconduct."),
    (classified(["reviewer_misconduct", "other"]), ""),
    (classified(["wrong_paper_review", "reviewer_misconduct"]), ""),
], ids=["record-error", "misconduct", "misconduct+other", "wrong-paper+misconduct"])
def test_a_reciprocal_complaint_still_comes_first_and_carries_no_verify_note(outcome, also):
    draft, rec = run(outcome, reciprocal=True)
    assert draft.draft_text == "[CHAIR: reciprocal complaint; see note]"
    assert draft.notes_for_chair == RECIPROCAL_NOTE + also
    assert rec["mode"] == "reciprocal_review"


def test_outside_the_window_the_misconduct_check_is_kept():
    draft, rec = run(classified(["reviewer_misconduct"]), created="2026-12-01T10:00:00Z",
                     window_end=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert (draft.draft_text, rec["mode"]) == ("[CHAIR: write reply]", "window")
    assert draft.notes_for_chair == f"Appeal reply wording is for Phase 1 rejections only\n{V_MISCONDUCT}"


def test_the_verify_notes_never_reach_the_reply_text():
    for reasons in (["reviewer_misconduct"], ["record_error"]):
        draft, _ = run(classified(reasons))
        assert "Before sending" not in draft.draft_text
        assert "Ethics Chairs" not in draft.draft_text
