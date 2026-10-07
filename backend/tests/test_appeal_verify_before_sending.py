"""The CSV verify_before_sending text (reject-appeal Step 2.5, step 6).

``appeal_reply_hook.verify_before_sending(reasons, record)`` must say exactly
what the hook's chair note says a person must check, built from the same
constants. Two layers:
  * every case of the step-5 table, with a hand-written literal cell;
  * agreement with the hook's own chair note for every subset of the nine
    phase-1 reasons, both relations, one and two papers, and reciprocal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations

import pytest

from app.pipeline import phase1_reply_mapping as m
from app.pipeline.appeal_reply_hook import prepare_appeal_draft, verify_before_sending
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
from app.pipeline.phase1_reply_mapping import map_phase1

REVIEW = "review_decision_appeal"
V_MISC = ("Before sending, check for harassment or an undisclosed conflict of interest. "
          "If either is present, do not send; forward the ticket to the Ethics Chairs.")
V_REC = "Before sending, check the review text against its score."
V_WRONG = "Do not reply or close the ticket until it is clear what we are doing with it."


def classified(reasons, relation="appeal", papers=("12345",), dropped=()) -> Phase1Outcome:
    return Phase1Outcome(CLASSIFIED, Phase1AppealResult(
        relation=relation, papers=list(papers),
        reasons=[AppealReason(reason=r, quote=f"quote for {r}") for r in reasons],
        dropped_unquoted=list(dropped),
    ))


def run(outcome, *, reciprocal=False, created="2026-09-15T10:00:00Z", window_end=None):
    data = {"subject": "s", "body": "b", "sender_name": "Jane Doe", "timestamp": created}
    return prepare_appeal_draft(REVIEW, None, reciprocal, data, window_end=window_end,
                                mapped=map_phase1(outcome))


def names_of(outcome) -> list[str]:
    result = getattr(outcome, "result", None)
    return [r.reason for r in result.reasons] if result is not None else []


def test_the_csv_texts_are_the_decided_ones():
    assert m.VERIFY_WRONG_PAPER == V_WRONG
    assert m.VERIFY_BEFORE_SENDING == {"reviewer_misconduct": V_MISC, "record_error": V_REC}


# --- the step-5 table, cell by cell -------------------------------------------------------
TABLE = [
    # id, outcome, reciprocal, window, expected cell
    ("misconduct-alone", classified(["reviewer_misconduct"]), False, False, V_MISC),
    ("missing-material-alone", classified(["missing_material_claim"]), False, False, ""),
    ("record-error-alone", classified(["record_error"]), False, False, V_REC),
    ("misconduct-with-T1", classified(["reviewer_misconduct", "decision_vs_reviews"]),
     False, False, V_MISC),
    ("missing-material-with-T1", classified(["decision_vs_reviews", "missing_material_claim"]),
     False, False, ""),
    ("record-error-with-T1", classified(["decision_vs_reviews", "record_error"]), False, False, V_REC),
    ("misconduct-and-record-error", classified(["record_error", "reviewer_misconduct"]),
     False, False, f"{V_MISC} {V_REC}"),
    ("wrong-paper-alone", classified(["wrong_paper_review"]), False, False, V_WRONG),
    ("wrong-paper-with-record-error", classified(["record_error", "wrong_paper_review"]),
     False, False, V_WRONG),
    ("wrong-paper-with-misconduct", classified(["wrong_paper_review", "reviewer_misconduct"]),
     False, False, f"{V_WRONG} {V_MISC}"),
    ("wrong-paper-with-record-error-and-misconduct",
     classified(["wrong_paper_review", "record_error", "reviewer_misconduct"]),
     False, False, f"{V_WRONG} {V_MISC}"),
    ("misconduct-with-yan-b", classified(["reviewer_misconduct", "llm_generated_review"]),
     False, False, V_MISC),
    ("four-reasons", classified(["record_error", "decision_vs_reviews", "missing_material_claim",
                                 "reviewer_misconduct"]), False, False, f"{V_MISC} {V_REC}"),
    ("misconduct-with-other", classified(["reviewer_misconduct", "other"]), False, False, V_MISC),
    ("misconduct-feedback-only", classified(["reviewer_misconduct"], relation="feedback_only"),
     False, False, V_MISC),
    ("misconduct-with-two-papers", classified(["reviewer_misconduct"], papers=("11111", "22222")),
     False, False, V_MISC),
    ("record-error-feedback-only", classified(["record_error"], relation="feedback_only"),
     False, False, V_REC),
    ("record-error-with-two-papers", classified(["record_error"], papers=("11111", "22222")),
     False, False, V_REC),
    ("record-error-with-other", classified(["record_error", "other"]), False, False, ""),
    ("not-appeal-with-misconduct", classified(["reviewer_misconduct"], relation="not_appeal"),
     False, False, ""),
    ("unverified-misconduct", classified([], dropped=("reviewer_misconduct",)), False, False, ""),
    ("other-alone", classified(["other"]), False, False, ""),
    ("unknown-name", classified(["future_reason"]), False, False, ""),
    ("flag-off", None, False, False, ""),
    ("gate-not-met", Phase1Outcome(GATE_NOT_MET), False, False, ""),
    ("failed", Phase1Outcome(FAILED), False, False, ""),
    ("not-appeal-state", Phase1Outcome(NOT_APPEAL), False, False, ""),
    ("reciprocal-record-error", classified(["record_error"]), True, False, ""),
    ("reciprocal-misconduct", classified(["reviewer_misconduct"]), True, False, ""),
    ("reciprocal-misconduct-with-other", classified(["reviewer_misconduct", "other"]), True, False, ""),
    ("reciprocal-wrong-paper-with-misconduct",
     classified(["wrong_paper_review", "reviewer_misconduct"]), True, False, ""),
    ("window-misconduct", classified(["reviewer_misconduct"]), False, True, V_MISC),
]


@pytest.mark.parametrize("outcome, reciprocal, window, expected",
                         [c[1:] for c in TABLE], ids=[c[0] for c in TABLE])
def test_the_cell_for_every_case_of_the_step5_table(outcome, reciprocal, window, expected):
    kwargs = {}
    if window:
        kwargs = {"created": "2026-12-01T10:00:00Z",
                  "window_end": datetime(2026, 11, 1, tzinfo=timezone.utc)}
    _, record = run(outcome, reciprocal=reciprocal, **kwargs)
    assert verify_before_sending(names_of(outcome), record) == expected


# --- agreement with the hook's chair note, exhaustively -------------------------------------
_CHECKS = set(m.VERIFY_BEFORE_SENDING.values())
_WRONG_NOTE = m.NOTE_INVESTIGATE["wrong_paper_review"]


def _from_the_chair_note(notes: str | None) -> str:
    lines = (notes or "").split("\n")
    items = [m.VERIFY_WRONG_PAPER] if _WRONG_NOTE in lines else []
    items += [line for line in lines if line in _CHECKS]
    return " ".join(items)


def test_the_cell_always_says_what_the_hooks_chair_note_says():
    checked = with_text = 0
    for size in range(len(PHASE1_APPEAL_REASONS) + 1):
        for subset in combinations(PHASE1_APPEAL_REASONS, size):
            variants = [("appeal", ("12345",), False), ("feedback_only", ("12345",), False),
                        ("appeal", ("11111", "22222"), False), ("appeal", ("12345",), True)]
            for relation, papers, reciprocal in variants:
                outcome = classified(list(subset), relation=relation, papers=papers)
                draft, record = run(outcome, reciprocal=reciprocal)
                cell = verify_before_sending(list(subset), record)
                assert cell == _from_the_chair_note(draft.notes_for_chair), (
                    subset, relation, papers, reciprocal, record["mode"])
                checked += 1
                with_text += bool(cell)
    assert checked == 512 * 4
    assert with_text > 0


@pytest.mark.parametrize("record", [None, {}, {"mode": None}, "merged", {"reasons": None}])
def test_no_record_gives_an_empty_cell(record):
    assert verify_before_sending(["reviewer_misconduct"], record) == ""


def test_text_never_comes_from_the_records_reasons_only_from_his_names():
    """The cell is built from his names; a composed record without them says nothing."""
    record = {"mode": "merged", "reasons": ["score_outcome_mismatch"], "block_ids": []}
    assert verify_before_sending([], record) == ""
    assert verify_before_sending(None, record) == ""
