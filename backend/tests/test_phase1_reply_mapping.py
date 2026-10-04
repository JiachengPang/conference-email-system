"""Tests for the phase-1 outcome -> composer input mapping (reject-appeal Phase 4, P3).

Pure: no database, no model call. Outcomes are built with Jiacheng's own types
(``Phase1Outcome`` / ``Phase1AppealResult`` / ``AppealReason``), so a contract
change on his side fails here. Every expected note is a hand-written literal.
"""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings, settings
from app.pipeline import phase1_reply_mapping as m
from app.pipeline.phase1_appeal_classifier import (
    PHASE1_APPEAL_REASONS,
    AppealReason,
    Phase1AppealResult,
)
from app.pipeline.phase1_appeal_outcome import (
    CLASSIFIED,
    FAILED,
    GATE_NOT_MET,
    NOT_APPEAL,
    Phase1Outcome,
)
from app.pipeline.phase1_reply_mapping import MappedAppeal, map_phase1

QUOTE_MARK = "QUOTE-TEXT-THAT-MUST-NEVER-BE-STORED"

# --- the notes, written out ------------------------------------------------------------------
N_FLAG_OFF = ("Chair writes: the phase-1 appeal classifier is turned off, so the appeal reasons "
              "were not determined.")
N_GATE = ("Chair writes: this email is outside the phase-1 appeal classification (intent or "
          "ticket date), so its reasons were not determined.")
N_FAILED = "Chair writes: the phase-1 appeal classifier did not return an answer."
N_NOT_APPEAL = ("Chair writes: the phase-1 classifier judged this email not to be an appeal (for "
                "example, a request to see the reviews or a reply without a request).")
N_WRONG = ("Investigate first: the author says a review is about a different paper. "
           "Do not reply to or close the ticket yet.")
N_RECORD = ("Investigate first: the author says a rating contradicts its own review, or that a "
            "submitted review was left out of the decision. Do not reply to or close the ticket yet.")
N_FEEDBACK = ("Chair writes: the author reports a review problem but says they are not asking "
              "for a change.")
N_NO_VERIFIED = "Chair writes: the author appeals, but no reason could be verified in the email."
N_INTERNAL = "Chair writes: the phase-1 outcome could not be read."


def classified(relation, reasons, papers=("12345",), dropped=()) -> Phase1Outcome:
    result = Phase1AppealResult(
        relation=relation,
        papers=list(papers),
        reasons=[AppealReason(reason=r, quote=f"{QUOTE_MARK} for {r}") for r in reasons],
        dropped_unquoted=list(dropped),
    )
    return Phase1Outcome(CLASSIFIED, result)


def snap(state, relation=None, reasons=None, must_verify=None, papers=None) -> dict:
    return {"state": state, "relation": relation, "reasons": reasons,
            "must_verify": must_verify, "papers": papers}


def cls_snap(relation, reasons, must_verify=False, papers=("12345",)) -> dict:
    return snap("classified", relation, list(reasons), must_verify, list(papers))


SCORE, REVIEWER = "score_outcome_mismatch", "reviewer_misunderstanding"
LLM, GENERAL = "llm_generated_review", "general_dissatisfaction"

# (id, outcome, expected)
CASES = [
    # --- states -----------------------------------------------------------------------------
    ("flag-off", None,
     MappedAppeal(None, "reason_unknown", (N_FLAG_OFF,), snap("flag_off"))),
    ("gate-not-met", Phase1Outcome(GATE_NOT_MET),
     MappedAppeal(None, "reason_unknown", (N_GATE,), snap("gate_not_met"))),
    ("failed", Phase1Outcome(FAILED),
     MappedAppeal(None, "reason_unknown", (N_FAILED,), snap("failed"))),
    ("classified-without-result", Phase1Outcome(CLASSIFIED, None),
     MappedAppeal(None, "reason_unknown", (N_FAILED,), snap("failed"))),
    ("not-appeal", Phase1Outcome(NOT_APPEAL),
     MappedAppeal(None, "not_appeal", (N_NOT_APPEAL,), snap("not_appeal", "not_appeal"))),
    ("not-appeal-inside-a-result", classified("not_appeal", []),
     MappedAppeal(None, "not_appeal", (N_NOT_APPEAL,), cls_snap("not_appeal", []))),
    # --- investigate first: no draft (must_verify) -------------------------------------------
    ("wrong-paper", classified("appeal", ["wrong_paper_review"]),
     MappedAppeal(None, "no_draft", (N_WRONG,),
                  cls_snap("appeal", ["wrong_paper_review"], must_verify=True))),
    ("record-error", classified("appeal", ["record_error"]),
     MappedAppeal(None, "no_draft", (N_RECORD,), cls_snap("appeal", ["record_error"], True))),
    ("both-investigate", classified("appeal", ["record_error", "wrong_paper_review"]),
     MappedAppeal(None, "no_draft", (N_WRONG, N_RECORD),
                  cls_snap("appeal", ["record_error", "wrong_paper_review"], True))),
    ("record-error-with-others",
     classified("appeal", ["record_error", "reviewer_misjudgment", "other"]),
     MappedAppeal(None, "no_draft", (N_RECORD, "Also raised: reviewer_misjudgment, other."),
                  cls_snap("appeal", ["record_error", "reviewer_misjudgment", "other"], True))),
    ("investigate-beats-feedback-only", classified("feedback_only", ["wrong_paper_review"]),
     MappedAppeal(None, "no_draft", (N_WRONG,),
                  cls_snap("feedback_only", ["wrong_paper_review"], True))),
    ("investigate-beats-several-papers",
     classified("appeal", ["record_error"], papers=("11111", "22222")),
     MappedAppeal(None, "no_draft", (N_RECORD,),
                  cls_snap("appeal", ["record_error"], True, ("11111", "22222")))),
    # --- feedback only ----------------------------------------------------------------------
    ("feedback-only", classified("feedback_only", ["reviewer_misjudgment"]),
     MappedAppeal(None, "chair_writes", (N_FEEDBACK, "Raised: reviewer_misjudgment."),
                  cls_snap("feedback_only", ["reviewer_misjudgment"]))),
    ("feedback-only-no-reason", classified("feedback_only", []),
     MappedAppeal(None, "chair_writes", (N_FEEDBACK,), cls_snap("feedback_only", []))),
    # --- reasons the chair writes (even mixed with T1/T2) -------------------------------------
    ("missing-material", classified("appeal", ["missing_material_claim"]),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: no approved reply covers missing_material_claim.",),
                  cls_snap("appeal", ["missing_material_claim"]))),
    ("misconduct", classified("appeal", ["reviewer_misconduct"]),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: no approved reply covers reviewer_misconduct.",),
                  cls_snap("appeal", ["reviewer_misconduct"]))),
    ("other", classified("appeal", ["other"]),
     MappedAppeal(None, "chair_writes", ("Chair writes: no approved reply covers other.",),
                  cls_snap("appeal", ["other"]))),
    ("T1-with-missing-material",
     classified("appeal", ["missing_material_claim", "decision_vs_reviews"]),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: no approved reply covers missing_material_claim.",
                   "Also raised: decision_vs_reviews."),
                  cls_snap("appeal", ["missing_material_claim", "decision_vs_reviews"]))),
    ("T1-T2-with-misconduct",
     classified("appeal", ["reviewer_misconduct", "decision_vs_reviews", "reviewer_misjudgment"]),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: no approved reply covers reviewer_misconduct.",
                   "Also raised: decision_vs_reviews, reviewer_misjudgment."),
                  cls_snap("appeal", ["reviewer_misconduct", "decision_vs_reviews",
                                      "reviewer_misjudgment"]))),
    ("two-chair-write-reasons-and-yan",
     classified("appeal", ["missing_material_claim", "llm_generated_review", "other"]),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: no approved reply covers missing_material_claim, other.",
                   "Also raised: llm_generated_review."),
                  cls_snap("appeal", ["missing_material_claim", "llm_generated_review", "other"]))),
    ("unknown-reason-is-chair-written", classified("appeal", ["future_reason", "decision_vs_reviews"]),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: no approved reply covers future_reason.",
                   "Also raised: decision_vs_reviews."),
                  cls_snap("appeal", ["future_reason", "decision_vs_reviews"]))),
    # --- several papers ---------------------------------------------------------------------
    ("two-papers", classified("appeal", ["decision_vs_reviews"], papers=("11111", "22222")),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: the email is about 2 papers; the approved replies are written "
                   "for one paper.", "Raised: decision_vs_reviews."),
                  cls_snap("appeal", ["decision_vs_reviews"], papers=("11111", "22222")))),
    ("three-papers-yan-a",
     classified("appeal", ["reconsideration_only"], papers=("1", "2", "3")),
     MappedAppeal(None, "chair_writes",
                  ("Chair writes: the email is about 3 papers; the approved replies are written "
                   "for one paper.", "Raised: reconsideration_only."),
                  cls_snap("appeal", ["reconsideration_only"], papers=("1", "2", "3")))),
    ("same-paper-twice-is-one", classified("appeal", ["decision_vs_reviews"], papers=("123", "123")),
     MappedAppeal((SCORE,), None, (), cls_snap("appeal", ["decision_vs_reviews"], papers=("123",)))),
    ("no-paper-is-fine", classified("appeal", ["decision_vs_reviews"], papers=()),
     MappedAppeal((SCORE,), None, (), cls_snap("appeal", ["decision_vs_reviews"], papers=()))),
    # --- no verified reason -----------------------------------------------------------------
    ("appeal-no-reason", classified("appeal", []),
     MappedAppeal(None, "reason_unknown", (N_NO_VERIFIED,), cls_snap("appeal", []))),
    ("appeal-no-reason-dropped-quotes",
     classified("appeal", [], dropped=("decision_vs_reviews", "other")),
     MappedAppeal(None, "reason_unknown",
                  (N_NO_VERIFIED, "Possibly raised (quote not verified): decision_vs_reviews, other."),
                  cls_snap("appeal", []))),
    # --- composed: the four mapped reasons ----------------------------------------------------
    ("T1", classified("appeal", ["decision_vs_reviews"]),
     MappedAppeal((SCORE,), None, (), cls_snap("appeal", ["decision_vs_reviews"]))),
    ("T2", classified("appeal", ["reviewer_misjudgment"]),
     MappedAppeal((REVIEWER,), None, (), cls_snap("appeal", ["reviewer_misjudgment"]))),
    ("T1+T2", classified("appeal", ["decision_vs_reviews", "reviewer_misjudgment"]),
     MappedAppeal((SCORE, REVIEWER), None, (),
                  cls_snap("appeal", ["decision_vs_reviews", "reviewer_misjudgment"]))),
    ("yan-b", classified("appeal", ["llm_generated_review"]),
     MappedAppeal((LLM,), None, (), cls_snap("appeal", ["llm_generated_review"]))),
    ("yan-a", classified("appeal", ["reconsideration_only"]),
     MappedAppeal((GENERAL,), None, (), cls_snap("appeal", ["reconsideration_only"]))),
    ("yan-b-with-T1-left-to-the-composer",
     classified("appeal", ["decision_vs_reviews", "llm_generated_review"]),
     MappedAppeal((SCORE, LLM), None, (),
                  cls_snap("appeal", ["decision_vs_reviews", "llm_generated_review"]))),
    ("yan-a-with-yan-b-left-to-the-composer",
     classified("appeal", ["llm_generated_review", "reconsideration_only"]),
     MappedAppeal((LLM, GENERAL), None, (),
                  cls_snap("appeal", ["llm_generated_review", "reconsideration_only"]))),
    ("duplicate-reason-counts-once",
     classified("appeal", ["decision_vs_reviews", "decision_vs_reviews"]),
     MappedAppeal((SCORE,), None, (), cls_snap("appeal", ["decision_vs_reviews"]))),
]


@pytest.mark.parametrize("outcome, expected", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_mapping_table(outcome, expected):
    assert map_phase1(outcome) == expected


@pytest.mark.parametrize("outcome", [c[1] for c in CASES], ids=[c[0] for c in CASES])
def test_exactly_one_of_reasons_and_hold_is_set(outcome):
    mapped = map_phase1(outcome)
    assert (mapped.reasons is None) != (mapped.hold is None)
    assert mapped.hold is None or mapped.hold in m.HOLDS


@pytest.mark.parametrize("outcome", [c[1] for c in CASES], ids=[c[0] for c in CASES])
def test_the_snapshot_holds_names_only_never_a_quote(outcome):
    mapped = map_phase1(outcome)
    assert QUOTE_MARK not in json.dumps(mapped.snapshot)
    assert QUOTE_MARK not in " ".join(mapped.notes)
    assert set(mapped.snapshot) == {"state", "relation", "reasons", "must_verify", "papers"}


def test_every_one_of_his_reasons_has_exactly_one_rule():
    """Drift guard: each of the 9 names is composed, investigated or chair-written."""
    composable, investigate, chair = set(m.COMPOSABLE), set(m.INVESTIGATE), m.CHAIR_WRITES_REASONS
    assert composable | investigate | chair == set(PHASE1_APPEAL_REASONS)
    assert not (composable & investigate) and not (composable & chair) and not (investigate & chair)
    assert m.COMPOSABLE == {
        "decision_vs_reviews": "score_outcome_mismatch",
        "reviewer_misjudgment": "reviewer_misunderstanding",
        "llm_generated_review": "llm_generated_review",
        "reconsideration_only": "general_dissatisfaction",
    }


@pytest.mark.parametrize("junk", [object(), "classified", 42, Phase1Outcome("some_new_state")])
def test_never_raises_and_never_guesses_on_junk(junk):
    mapped = map_phase1(junk)
    assert (mapped.reasons, mapped.hold) == (None, "reason_unknown")
    assert mapped.notes in ((N_INTERNAL,), (N_FAILED,))


def test_the_phase1_intent_is_the_review_decision_appeal():
    """The hook's phase-1 branch answers review_decision_appeal; the phase-1 gate must
    ask exactly that intent, or the mapping would silently apply to nothing."""
    assert Settings.model_fields["PHASE1_APPEAL_INTENT"].default == "review_decision_appeal"
    assert settings.PHASE1_APPEAL_INTENT == "review_decision_appeal"


def test_the_reason_source_defaults_to_phase1_and_offers_the_rollback():
    field = Settings.model_fields["APPEAL_REPLY_REASON_SOURCE"]
    assert field.default == "phase1"
    assert set(field.annotation.__args__) == {"phase1", "appeal_reason"}
