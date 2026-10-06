"""Marc's reviewer-misconduct blocks in the REAL template file (reject-appeal Step 2.5, step 4).

Three points approved by Marc on 2026-10-05; misconduct gets all three,
missing material the first only, and record_error shares the score point
without the rebuttal point. Points follow the one global order (misconduct
points 1-3 first). Every expected text is a hand-written literal; composer
reasons use the composer's names (``score_outcome_mismatch``, not the phase-1
classifier's ``decision_vs_reviews``).
"""

from __future__ import annotations

from itertools import permutations

import pytest

from app.pipeline.appeal_ai_suggestion import build_bank
from app.pipeline.appeal_reply_composer import compose_reply
from app.pipeline.appeal_reply_lint import lint_template_body
from app.pipeline.appeal_reply_templates import load_approved_templates

MISCONDUCT, MISSING, RECORD = "reviewer_misconduct", "missing_material_claim", "record_error"
SCORE, REVIEWER = "score_outcome_mismatch", "reviewer_misunderstanding"
ROLES = "internal_roles_or_process"

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
FRAME = ("opening_warm", "lead_in_concerns")

MISCONDUCT_ALONE = (
    f"{OPENING}\n\n(1) {P_SPC}\n\n(2) {P_TRACKING}\n\n(3) {P_ETHICS}\n\n{CLOSING}"
)
MISCONDUCT_IDS = (*FRAME, "point_spc_evaluation", "point_reviewer_tracking", "point_ethics_form",
                  "closing_reviewed")


@pytest.fixture(scope="module")
def served():
    return {t.id: t for t in load_approved_templates()}


# --- the blocks ---------------------------------------------------------------------
@pytest.mark.parametrize("block_id, body, reasons, order", [
    ("point_spc_evaluation", P_SPC, (MISCONDUCT, MISSING), 1),
    ("point_reviewer_tracking", P_TRACKING, (MISCONDUCT,), 2),
    ("point_ethics_form", P_ETHICS, (MISCONDUCT,), 3),
])
def test_each_block_is_served_exactly_as_marc_approved_it(served, block_id, body, reasons, order):
    t = served[block_id]
    assert (t.kind, t.order, t.optional, t.reasons, t.body) == ("point", order, False, reasons, body)
    assert (t.approved_by, t.approved_at) == ("Marc Pujol-Gonzalez", "2026-10-05")


@pytest.mark.parametrize("block_id, rules", [
    ("point_spc_evaluation", {ROLES}),
    ("point_reviewer_tracking", {ROLES}),
    ("point_ethics_form", set()),
])
def test_each_block_trips_only_its_waived_rule(served, block_id, rules):
    t = served[block_id]
    assert {name for name, _ in lint_template_body(t.body)} == rules
    assert {w.rule for w in t.lint_waivers} == rules


def test_the_score_point_text_is_unchanged_and_now_also_serves_record_error(served):
    t = served["point_scores"]
    assert (t.order, t.reasons, t.body) == (4, (SCORE, RECORD), P_SCORES)


# --- composed replies -----------------------------------------------------------------
def test_reviewer_misconduct_alone_is_its_three_points_then_the_closing():
    result = compose_reply([MISCONDUCT])
    assert (result.mode, result.refusal, result.chair_notes) == ("merged", None, ())
    assert result.body == MISCONDUCT_ALONE
    assert result.used_ids == MISCONDUCT_IDS


def test_misconduct_and_missing_material_share_the_first_point_once():
    assert compose_reply([MISSING, MISCONDUCT]) == compose_reply([MISCONDUCT])


def test_missing_material_alone_is_misconduct_point_1_alone():
    result = compose_reply([MISSING])
    assert result.mode == "merged"
    assert result.body == f"{OPENING}\n\n(1) {P_SPC}\n\n{CLOSING}"
    assert result.used_ids == (*FRAME, "point_spc_evaluation", "closing_reviewed")


def test_record_error_alone_is_the_score_point_without_the_rebuttal_point():
    result = compose_reply([RECORD])
    assert result.mode == "merged"
    assert result.body == f"{OPENING}\n\n(1) {P_SCORES}\n\n{CLOSING}"
    assert result.used_ids == (*FRAME, "point_scores", "closing_reviewed")
    assert "point_rebuttal" not in result.used_ids


def test_record_error_with_scores_gives_the_shared_point_once():
    result = compose_reply([RECORD, SCORE])
    assert result.mode == "merged"
    assert result.body == f"{OPENING}\n\n(1) {P_SCORES}\n\n(2) {P_REBUTTAL}\n\n{CLOSING}"
    assert result.used_ids == (*FRAME, "point_scores", "point_rebuttal", "closing_reviewed")
    assert result.body == compose_reply([SCORE]).body


def test_misconduct_with_scores_puts_the_misconduct_points_first():
    expected = (f"{OPENING}\n\n(1) {P_SPC}\n\n(2) {P_TRACKING}\n\n(3) {P_ETHICS}\n\n"
                f"(4) {P_SCORES}\n\n(5) {P_REBUTTAL}\n\n{CLOSING}")
    results = {compose_reply(list(p)) for p in permutations([SCORE, MISCONDUCT])}
    assert len(results) == 1
    (result,) = results
    assert (result.mode, result.body) == ("merged", expected)
    assert result.used_ids == (*FRAME, "point_spc_evaluation", "point_reviewer_tracking",
                               "point_ethics_form", "point_scores", "point_rebuttal",
                               "closing_reviewed")


@pytest.mark.parametrize("reasons", [
    [MISCONDUCT, MISSING, SCORE, RECORD],
    [MISCONDUCT, SCORE, REVIEWER, RECORD],
])
def test_four_reasons_go_to_the_chair(reasons):
    result = compose_reply(reasons)
    assert (result.mode, result.body, result.used_ids) == (
        "chair_writes", "[CHAIR: write reply]", ("line_chair_writes",))
    assert result.chair_notes[0].startswith("Chair writes: more than 3 issues raised: ")


# --- the AI draft's sentence bank (2a) ------------------------------------------------
def test_the_bank_has_the_three_new_blocks_and_no_retired_block(served):
    bank = build_bank(served.values())
    assert [block_id for block_id, _ in bank.blocks] == [
        "opening_warm", "lead_in_concerns",
        "point_spc_evaluation", "point_reviewer_tracking", "point_ethics_form",
        "point_scores", "point_all_assessments", "point_rebuttal", "point_consider_input",
        "closing_reviewed", "standalone_general_stage1", "standalone_ai_review",
    ]
    assert sum(len(s) for _, s in bank.blocks) == 39
    assert len(bank.sources) == 39


def test_the_bank_holds_each_new_sentence_from_its_own_block(served):
    bank = build_bank(served.values())
    new = {
        "Decisions are not based on any single review.": "point_spc_evaluation",
        "Senior program committee members evaluated both the paper and the reviews to decide "
        "their recommendation.": "point_spc_evaluation",
        "We are asking SPCs and ACs to report reviewers that are being unprofessional.":
            "point_reviewer_tracking",
        "We keep track of the reviewer's performance and we will consider it in future editions.":
            "point_reviewer_tracking",
        P_ETHICS.split(". This")[0] + ".": "point_ethics_form",
        "This may impact our future relationship with this reviewer, but it will not change the "
        "outcome for this specific paper.": "point_ethics_form",
    }
    for sentence, block_id in new.items():
        assert bank.sources[sentence] == frozenset({block_id}), sentence
    assert bank.waivers["point_spc_evaluation"] == frozenset({ROLES})
    assert bank.waivers["point_reviewer_tracking"] == frozenset({ROLES})
    assert bank.waivers["point_ethics_form"] == frozenset()
