"""Composer: the three composer-only reasons and their order (reject-appeal Step 2.5, step 2).

``reviewer_misconduct``, ``missing_material_claim`` and ``record_error`` come from
the phase-1 classifier and are accepted by the composer only, never added to the
appeal-reason registry. Reasons are ordered most critical first: misconduct,
missing material, the registry in its own order, record_error, then reciprocal.

Which POINTS each new reason gets is the template file's job (step 4); these
tests pin only what step 2 changes: the accepted names, the order of reasons in
chair notes, the precedence rules, and the reason cap. Every expected value is a
hand-written literal.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from app.pipeline import appeal_reply_composer as arc
from app.pipeline.appeal_reasons import REASON_NAMES
from app.pipeline.appeal_reply_composer import compose_reply

MISCONDUCT, MISSING, RECORD = "reviewer_misconduct", "missing_material_claim", "record_error"
NEW = (MISCONDUCT, MISSING, RECORD)
WRONG, SCORE, REVIEWER = "wrong_paper_review", "score_outcome_mismatch", "reviewer_misunderstanding"
LLM, GENERAL, OTHER, RECIPROCAL = (
    "llm_generated_review", "general_dissatisfaction", "other", "reciprocal_dispute")

CANONICAL = [MISCONDUCT, MISSING, WRONG, SCORE, REVIEWER, LLM, GENERAL, OTHER, RECORD, RECIPROCAL]


def test_the_registry_is_unchanged():
    assert REASON_NAMES == (WRONG, SCORE, REVIEWER, LLM, GENERAL, OTHER)


def test_the_composer_accepts_exactly_the_registry_reciprocal_and_the_three_new_names():
    assert arc.ALLOWED_REASONS == frozenset(REASON_NAMES) | {RECIPROCAL, *NEW}
    assert not set(NEW) & set(REASON_NAMES)


def test_the_canonical_order_is_most_critical_first():
    assert sorted(arc.ALLOWED_REASONS, key=arc._ORDER.__getitem__) == CANONICAL


def test_the_order_is_visible_in_the_no_draft_note():
    """Every other reason is listed after the wrong-paper note, in canonical order."""
    result = compose_reply(list(reversed(CANONICAL)))
    assert result.mode == "no_draft"
    assert result.body is None and result.refusal is None
    assert result.chair_notes[1] == (
        "Also raised: reviewer_misconduct, missing_material_claim, score_outcome_mismatch, "
        "reviewer_misunderstanding, llm_generated_review, general_dissatisfaction, other, "
        "record_error, reciprocal_dispute."
    )


@pytest.mark.parametrize("new", NEW)
def test_wrong_paper_review_still_wins_over_each_new_reason(new):
    result = compose_reply([new, WRONG])
    assert (result.mode, result.body, result.used_ids) == ("no_draft", None, ())
    assert result.chair_notes[1] == f"Also raised: {new}."


@pytest.mark.parametrize("new", NEW)
def test_a_reciprocal_complaint_still_wins_over_each_new_reason(new):
    result = compose_reply([RECIPROCAL, new])
    assert (result.mode, result.body, result.used_ids) == ("reciprocal_review", None, ())
    assert result.chair_notes == (
        "Reciprocal-review complaint: tagged for Marc to review himself. No reply is drafted.",
        f"Also raised: {new}.",
    )


@pytest.mark.parametrize("standalone", [GENERAL])
@pytest.mark.parametrize("new", NEW)
def test_a_standalone_reply_mixed_with_a_new_reason_goes_to_the_chair(standalone, new):
    """D105 is unchanged: Yan's general reply only ever stands alone."""
    result = compose_reply([standalone, new])
    first, second = sorted([standalone, new], key=CANONICAL.index)
    assert result.mode == "chair_writes"
    assert result.body == "[CHAIR: write reply]"
    assert result.used_ids == ("line_chair_writes",)
    assert result.chair_notes == (
        f"Chair writes: no approved reply covers these reasons together: {first}, {second}.",
    )


def test_more_than_three_merged_reasons_go_to_the_chair():
    result = compose_reply([RECORD, SCORE, MISSING, MISCONDUCT])
    assert result.mode == "chair_writes"
    assert result.body == "[CHAIR: write reply]"
    assert result.used_ids == ("line_chair_writes",)
    assert result.chair_notes == (
        "Chair writes: more than 3 issues raised: reviewer_misconduct, missing_material_claim, "
        "score_outcome_mismatch, record_error.",
    )


def test_the_reason_cap_is_still_three():
    assert arc.MAX_REASONS == 3


def test_any_input_order_gives_identical_output():
    results = {compose_reply(list(p)) for p in permutations([RECORD, LLM, MISSING, MISCONDUCT])}
    assert len(results) == 1
    (only,) = results
    assert only.chair_notes == (
        "Chair writes: more than 3 issues raised: reviewer_misconduct, "
        "missing_material_claim, llm_generated_review, record_error.",
    )


@pytest.mark.parametrize("name", [
    "decision_vs_reviews", "reviewer_misjudgment", "reconsideration_only", "Record_Error",
])
def test_phase1_only_names_and_case_variants_are_still_unknown(name):
    result = compose_reply([name])
    assert (result.mode, result.refusal) == ("refused", "unknown_reason")
