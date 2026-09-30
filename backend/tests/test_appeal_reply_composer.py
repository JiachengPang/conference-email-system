"""Tests for the appeal reply composer (reject-appeal Phase 3, D96).

Golden outputs are written out as LITERAL strings — never built by calling the
composer. The blocks come from a temp copy of the real template file in which
every entry is approved with a correct hash (``approved_copy``); the real file
itself (all draft) must refuse every input.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from pathlib import Path

import pytest

from app.pipeline import appeal_reply_composer as arc
from app.pipeline import appeal_reply_templates as art
from app.pipeline.appeal_reply_composer import ComposeResult, compose_reply
from app.pipeline.appeal_reply_lint import lint_template_body
from app.pipeline.appeal_reply_templates import compute_body_sha256, expected_points_for_reasons

SCORE, REVIEWER, LLM = "score_outcome_mismatch", "reviewer_misunderstanding", "llm_generated_review"
WRONG, GENERAL, OTHER, RECIP = "wrong_paper_review", "general_dissatisfaction", "other", "reciprocal_dispute"


def approved_copy(tmp_path: Path, *, overrides: dict | None = None, draft: set | None = None,
                  unblock: set = frozenset({"full_reciprocal"})) -> Path:
    """A temp copy of the real file with every entry approved and correctly hashed.

    ``point_report_form`` stays draft (its form address is still missing).
    ``overrides`` maps id -> fields to change BEFORE hashing; ``draft`` lists ids
    to leave unapproved; ``unblock`` lists ids whose blocked_on is cleared.
    """
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    keep_draft = {"point_report_form"} | (draft or set())
    for e in data["templates"]:
        e.update((overrides or {}).get(e["id"], {}))
        if e["id"] in unblock:
            e["blocked_on"] = []
        if e["id"] in keep_draft:
            continue
        e.update(status="approved", approved_by="test", approved_at="2026-09-29",
                 approved_sha256=compute_body_sha256(e["body"]))
    p = tmp_path / "templates.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def blocks(tmp_path) -> Path:
    return approved_copy(tmp_path)


# --- golden strings (literals) --------------------------------------------------------
G_SCORE = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Final decisions are not determined solely by the reviewer scores that are visible to you. "
    "Additional input from senior members of the program committee is also considered, and all "
    "factors are carefully weighed.\n\n"
    "(2) We understand that not having a chance for rebuttal is frustrating, but this is how the "
    "two-phase process works.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)
G_REVIEWER = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Decisions are not based on any single review; all assessments are weighed together.\n\n"
    "(2) We understand that not having a chance for rebuttal is frustrating, but this is how the "
    "two-phase process works.\n\n"
    "(3) Thank you for sharing your view of the review. We will consider your input when studying "
    "possible changes for future editions.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)
G_LLM = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) The review process includes one review that is clearly identified as AI-generated. It carries "
    "no ratings and plays no part in the decision.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)
G_HOLD_WRONG = "Thank you for flagging this. We will investigate and follow up with you."
G_HOLD_SCORE = ("Thank you for flagging your concern about the consistency of the outcome. "
                "We will investigate and follow up with you.")
G_HOLD_BOTH = "Thank you for flagging these concerns. We will investigate and follow up with you."
G_GENERAL = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission.\n\n"
    "We understand your wish for the decision to be reconsidered. We also understand that not having "
    "a chance for rebuttal is frustrating, but this is how the two-phase process works.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)
G_OTHER = "[CHAIR: write reply]"
G_RECIP = (
    "We understand that this situation is disappointing, and we appreciate the effort you invested in "
    "preparing your submission. We would like to clarify the circumstances:\n\n"
    "(1) Reciprocal reviewers were informed that failure to complete their assigned reviews could "
    "result in desk rejection of the paper(s) that nominated them.\n\n"
    "(2) We expected authors to coordinate with their co-authors on reciprocal reviewing "
    "responsibilities, so first authors were not warned separately when a nominee had not completed "
    "their review.\n\n"
    "(3) In your case, the nominated reciprocal reviewer did not complete their assigned review. The "
    "desk rejection is therefore consistent with the stated policy.\n\n"
    "The desk rejection is final, but we hope to see your work at a future AAAI conference."
)
G_EXAMPLE_A = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Final decisions are not determined solely by the reviewer scores that are visible to you. "
    "Additional input from senior members of the program committee is also considered, and all "
    "factors are carefully weighed.\n\n"
    "(2) Decisions are not based on any single review; all assessments are weighed together.\n\n"
    "(3) We understand that not having a chance for rebuttal is frustrating, but this is how the "
    "two-phase process works.\n\n"
    "(4) Thank you for sharing your view of the review. We will consider your input when studying "
    "possible changes for future editions.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)
G_EXAMPLE_B = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Decisions are not based on any single review; all assessments are weighed together.\n\n"
    "(2) We understand that not having a chance for rebuttal is frustrating, but this is how the "
    "two-phase process works.\n\n"
    "(3) Thank you for sharing your view of the review. We will consider your input when studying "
    "possible changes for future editions.\n\n"
    "(4) The review process includes one review that is clearly identified as AI-generated. It carries "
    "no ratings and plays no part in the decision.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)
G_SCORE_OTHER = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Final decisions are not determined solely by the reviewer scores that are visible to you. "
    "Additional input from senior members of the program committee is also considered, and all "
    "factors are carefully weighed.\n\n"
    "(2) We understand that not having a chance for rebuttal is frustrating, but this is how the "
    "two-phase process works.\n\n"
    "[CHAIR: write reply]\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)
G_THREE = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Final decisions are not determined solely by the reviewer scores that are visible to you. "
    "Additional input from senior members of the program committee is also considered, and all "
    "factors are carefully weighed.\n\n"
    "(2) Decisions are not based on any single review; all assessments are weighed together.\n\n"
    "(3) We understand that not having a chance for rebuttal is frustrating, but this is how the "
    "two-phase process works.\n\n"
    "(4) Thank you for sharing your view of the review. We will consider your input when studying "
    "possible changes for future editions.\n\n"
    "(5) The review process includes one review that is clearly identified as AI-generated. It carries "
    "no ratings and plays no part in the decision.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your "
    "work and helping you secure publication in another leading venue or future AAAI edition."
)

GOLDEN_CASES = [
    ([SCORE], {}, "merged", G_SCORE),
    ([REVIEWER], {}, "merged", G_REVIEWER),
    ([LLM], {}, "merged", G_LLM),
    ([WRONG], {}, "holding", G_HOLD_WRONG),
    ([SCORE], {"chair_forwards_score_mismatch": True}, "holding", G_HOLD_SCORE),
    ([WRONG, SCORE], {"chair_forwards_score_mismatch": True}, "holding", G_HOLD_BOTH),
    ([GENERAL], {}, "standalone", G_GENERAL),
    ([OTHER], {}, "chair_writes", G_OTHER),
    ([RECIP], {}, "reciprocal", G_RECIP),
    ([SCORE, REVIEWER], {}, "merged", G_EXAMPLE_A),
    ([REVIEWER, LLM], {}, "merged", G_EXAMPLE_B),
    ([SCORE, OTHER], {}, "merged", G_SCORE_OTHER),
    ([SCORE, REVIEWER, LLM], {}, "merged", G_THREE),
]


# --- golden outputs --------------------------------------------------------------------------
@pytest.mark.parametrize("reasons, kwargs, mode, expected", GOLDEN_CASES,
                         ids=[f"{'+'.join(c[0])}{'-fwd' if c[1] else ''}" for c in GOLDEN_CASES])
def test_golden_output(blocks, reasons, kwargs, mode, expected):
    r = compose_reply(reasons, path=blocks, **kwargs)
    assert r.refusal is None, r.refusal
    assert r.mode == mode
    assert r.body == expected


@pytest.mark.parametrize("expected", [c[3] for c in GOLDEN_CASES],
                         ids=[f"{'+'.join(c[0])}{'-fwd' if c[1] else ''}" for c in GOLDEN_CASES])
def test_every_golden_output_passes_the_lint(expected):
    assert lint_template_body(expected) == []


def test_every_golden_body_is_clean_text(blocks):
    """Blank line between paragraphs; no trailing whitespace; no trailing newline."""
    for reasons, kwargs, _, _ in GOLDEN_CASES:
        body = compose_reply(reasons, path=blocks, **kwargs).body
        assert not body.endswith("\n") and body == body.strip()
        assert all(line == line.rstrip() for line in body.split("\n"))
        assert "\n\n\n" not in body


def test_score_without_the_forward_flag_is_the_merged_reply(blocks):
    r = compose_reply([SCORE], chair_forwards_score_mismatch=False, path=blocks)
    assert (r.mode, r.body) == ("merged", G_SCORE)


def test_example_c_wrong_paper_plus_reviewer_is_a_holding_reply_with_a_note(blocks):
    r = compose_reply([WRONG, REVIEWER], path=blocks)
    assert (r.mode, r.body) == ("holding", G_HOLD_WRONG)
    assert r.chair_notes == ("Also raised: reviewer_misunderstanding. Not answered in this reply.",)
    assert "reviewer_misunderstanding" not in r.body


def test_holding_has_no_opening_closing_or_final_wording(blocks):
    for reasons, kw in (([WRONG], {}), ([SCORE], {"chair_forwards_score_mismatch": True}),
                        ([WRONG, SCORE], {"chair_forwards_score_mismatch": True})):
        body = compose_reply(reasons, path=blocks, **kw).body
        assert "final" not in body.lower() and "disappointing" not in body


def test_wrong_paper_with_an_unforwarded_score_notes_the_score(blocks):
    r = compose_reply([WRONG, SCORE], path=blocks)
    assert (r.mode, r.body) == ("holding", G_HOLD_WRONG)
    assert r.chair_notes == ("Also raised: score_outcome_mismatch. Not answered in this reply.",)


# --- rules ------------------------------------------------------------------------------------
def test_r3_a_shared_point_appears_exactly_once(blocks):
    body = compose_reply([SCORE, REVIEWER], path=blocks).body
    assert body.count("not having a chance for rebuttal") == 1
    assert re.findall(r"(?m)^\((\d)\)", body) == ["1", "2", "3", "4"]


def test_r4_general_dissatisfaction_is_dropped_beside_another_reason(blocks):
    r = compose_reply([GENERAL, SCORE], path=blocks)
    assert (r.mode, r.body) == ("merged", G_SCORE)
    assert "reconsidered" not in r.body
    assert "body_reconsider" not in r.used_ids


def test_r5_the_chair_line_sits_after_the_last_point_and_before_the_closing(blocks):
    body = compose_reply([SCORE, OTHER], path=blocks).body
    paras = body.split("\n\n")
    assert paras[-2] == "[CHAIR: write reply]"
    assert paras[-3].startswith("(2) ")
    assert paras[-1].startswith("The decision is final")


def test_r7_reciprocal_wins_and_notes_the_rest(blocks):
    r = compose_reply([RECIP, SCORE], path=blocks)
    assert (r.mode, r.body) == ("reciprocal", G_RECIP)
    assert r.chair_notes == ("Also raised: score_outcome_mismatch. Not answered in this reply.",)


def test_r8_four_reasons_are_refused_and_three_pass(blocks):
    r = compose_reply([SCORE, REVIEWER, LLM, OTHER], path=blocks)
    assert (r.mode, r.refusal, r.body) == ("refused", "too_many_reasons", None)
    assert compose_reply([SCORE, REVIEWER, LLM], path=blocks).body == G_THREE


def test_r8_counts_after_r4(blocks):
    """general_dissatisfaction is dropped first, so it does not count toward 3."""
    assert compose_reply([GENERAL, SCORE, REVIEWER, LLM], path=blocks).body == G_THREE


@pytest.mark.parametrize("reasons, refusal", [
    (["made_up_reason"], "unknown_reason"),
    ([SCORE, "a"], "unknown_reason"),
    ([], "no_reasons"),
])
def test_unknown_or_empty_reasons_are_refused(blocks, reasons, refusal):
    r = compose_reply(reasons, path=blocks)
    assert (r.mode, r.refusal, r.body) == ("refused", refusal, None)


def test_every_ordering_of_a_three_reason_set_gives_identical_output(blocks):
    results = {compose_reply(list(p), path=blocks) for p in itertools.permutations([SCORE, REVIEWER, LLM])}
    assert len(results) == 1
    assert next(iter(results)).body == G_THREE


def test_duplicate_reasons_count_once(blocks):
    assert compose_reply([SCORE, SCORE], path=blocks).body == G_SCORE


# --- missing blocks -----------------------------------------------------------------------------
@pytest.mark.parametrize("missing, reasons", [
    ("opening_warm", [SCORE]),
    ("lead_in_concerns", [SCORE]),
    ("closing_reviewed", [SCORE]),
    ("point_rebuttal", [SCORE]),
    ("holding_wrong_paper", [WRONG]),
    ("body_reconsider", [GENERAL]),
    ("line_chair_writes", [OTHER]),
    ("full_reciprocal", [RECIP]),
])
def test_a_missing_required_block_refuses_with_its_id(tmp_path, missing, reasons):
    p = approved_copy(tmp_path, draft={missing})
    r = compose_reply(reasons, path=p)
    assert (r.mode, r.refusal, r.body) == ("refused", f"missing_approved_block:{missing}", None)


def test_an_optional_point_that_is_not_approved_is_omitted_not_refused(blocks):
    r = compose_reply([LLM], path=blocks)
    assert r.body == G_LLM
    assert "point_report_form" not in r.used_ids


def test_the_report_form_point_is_still_expected_for_llm_reviews():
    assert ("point_report_form", True) in expected_points_for_reasons([LLM])


# --- the final lint ------------------------------------------------------------------------------
def test_a_composed_body_that_fails_the_lint_is_refused_without_text(tmp_path, caplog):
    """Each block is clean alone; the JOIN forms "this year" across the opening
    and the lead-in, so only the composer's final lint can catch it."""
    p = approved_copy(tmp_path, overrides={
        "opening_warm": {"body": "We understand this outcome, and we reply this"},
        "lead_in_concerns": {"body": "year as follows:"},
    })
    with caplog.at_level(logging.WARNING):
        r = compose_reply([SCORE], path=p)
    assert (r.mode, r.refusal, r.body) == ("refused", "lint:year_or_time_specific", None)
    assert "this year" not in caplog.text and "as follows" not in caplog.text


# --- never raises -------------------------------------------------------------------------------
@pytest.mark.parametrize("junk", [None, "a string", 42, 3.5, {"score": 1}, [SCORE, None],
                                  [[SCORE]], [{"x": 1}], b"bytes", object()])
def test_never_raises_on_junk(blocks, junk):
    r = compose_reply(junk, path=blocks)
    assert isinstance(r, ComposeResult)
    assert r.mode == "refused" and r.body is None and r.refusal


def test_a_missing_file_is_refused_not_raised(tmp_path):
    r = compose_reply([SCORE], path=tmp_path / "absent.json")
    assert r.mode == "refused" and r.body is None


# --- the REAL file (all draft) ----------------------------------------------------------------------
@pytest.mark.parametrize("reasons, kwargs", [(c[0], c[1]) for c in GOLDEN_CASES])
def test_the_real_file_refuses_everything_today(reasons, kwargs):
    r = compose_reply(reasons, **kwargs)
    assert r.mode == "refused" and r.body is None
    assert r.refusal.startswith("missing_approved_block:")


# --- expected_points_for_reasons (loader helper) --------------------------------------------------
def test_expected_points_include_unapproved_points_in_the_global_order():
    """The real file is all draft, yet the helper still lists what is needed."""
    assert expected_points_for_reasons([SCORE, REVIEWER]) == [
        ("point_scores", False), ("point_all_assessments", False),
        ("point_rebuttal", False), ("point_consider_input", False),
    ]
    assert expected_points_for_reasons([LLM]) == [("point_ai_review", False), ("point_report_form", True)]
    assert expected_points_for_reasons([WRONG, GENERAL, OTHER]) == []


def test_expected_points_return_ids_and_flags_only():
    for item in expected_points_for_reasons([SCORE, REVIEWER, LLM]):
        assert isinstance(item, tuple) and len(item) == 2
        assert isinstance(item[0], str) and isinstance(item[1], bool)
        assert " " not in item[0], "an id, never body text"


@pytest.mark.parametrize("junk", [None, 5, [None], "score_outcome_mismatch"])
def test_expected_points_never_raise(junk):
    assert isinstance(expected_points_for_reasons(junk), list)


def test_expected_points_on_a_bad_file_is_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert expected_points_for_reasons([SCORE], p) == []


# --- source scan -----------------------------------------------------------------------------------
def test_the_composer_never_reads_the_template_file_itself():
    src = Path(arc.__file__).read_text(encoding="utf-8")
    for forbidden in ("appeal_reply_templates.json", "read_text", "open(", "json.load", "import json"):
        assert forbidden not in src, forbidden
