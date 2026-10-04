"""Tests for the appeal reply composer (reject-appeal Phase 3, D96; rules per D97-D111).

Golden outputs are written out as LITERAL strings — never built by calling the
composer or reading the file. Most tests use a temp copy of the real template
file in which every non-retired entry is approved with a correct hash
(``approved_copy``). The tests in the "REAL file" section run on the real file
exactly as approved in Step 3c, against the same literal goldens.
"""

from __future__ import annotations

import inspect
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
ROLES = "internal_roles_or_process"


def approved_copy(tmp_path: Path, *, overrides: dict | None = None, draft: set | None = None,
                  approve_retired: bool = False) -> Path:
    """A temp copy of the real file with every non-retired entry approved and
    correctly hashed.

    ``overrides`` maps id -> fields to change BEFORE hashing; ``draft`` lists ids
    to leave unapproved; ``approve_retired`` approves the retired entries too, to
    prove the composer never uses them. No live block is blocked in the real file
    any more (Yan's AI-review reply was approved on 2026-10-05), so nothing needs
    unblocking; a test that wants a blocker sets one through ``overrides``.
    """
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    for e in data["templates"]:
        e.update((overrides or {}).get(e["id"], {}))
        if e["id"] in (draft or set()):
            # The real file is approved since Step 3c, so leaving a block
            # unapproved must actively reset it, not just skip stamping it.
            e.update(status="draft", approved_by=None, approved_at=None, approved_sha256=None)
            continue
        if e["status"] == "retired" and not approve_retired:
            continue
        if e["status"] == "retired":
            e["blocked_on"] = []
        e.update(status="approved", approved_by="test", approved_at="2026-10-02",
                 approved_sha256=compute_body_sha256(e["body"]))
    p = tmp_path / "templates.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def blocks(tmp_path) -> Path:
    return approved_copy(tmp_path)


# --- golden strings (literals) --------------------------------------------------------
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
CHAIR_LINE = "[CHAIR: write reply]"

G_T1 = f"{OPENING}\n\n(1) {P_SCORES}\n\n(2) {P_REBUTTAL}\n\n{CLOSING}"
G_T2 = f"{OPENING}\n\n(1) {P_REVIEWERS}\n\n(2) {P_REBUTTAL}\n\n(3) {P_THANKS}\n\n{CLOSING}"
G_T1_T2 = (f"{OPENING}\n\n(1) {P_SCORES}\n\n(2) {P_REVIEWERS}\n\n(3) {P_REBUTTAL}\n\n"
           f"(4) {P_THANKS}\n\n{CLOSING}")
G_T1_OTHER = f"{OPENING}\n\n(1) {P_SCORES}\n\n(2) {P_REBUTTAL}\n\n{CHAIR_LINE}\n\n{CLOSING}"
G_T1_T2_OTHER = (f"{OPENING}\n\n(1) {P_SCORES}\n\n(2) {P_REVIEWERS}\n\n(3) {P_REBUTTAL}\n\n"
                 f"(4) {P_THANKS}\n\n{CHAIR_LINE}\n\n{CLOSING}")
G_YAN_A = (
    "Thank you for taking the time to share your concerns regarding the review process for your "
    "submission. We take concerns about fairness in the review process seriously and have carefully "
    "considered the issues you raised.\n\n"
    "We understand that aspects of the reviews may have led you to feel that your submission did not "
    "receive fair consideration. We would like to emphasize that the final decision on a submission is "
    "not determined by any single review or reviewer.\n\n"
    "To further strengthen the consistency and fairness of the decision-making process, we have "
    "introduced a new Senior Program Chair (SPC) buddy system. Under this system, each SPC is paired "
    "with another SPC who serves as a \"buddy\" and independently reviews the SPC's recommendations. "
    "This additional layer of cross-checking is intended to reduce the influence of any individual "
    "assessment and promote greater consistency and fairness across the review process. In addition, "
    "the Area Chair is asked to review the recommendations by SPCs. When concerns arise regarding the "
    "quality of a particular review, these concerns are taken into account when the reviews are "
    "weighed as part of the overall assessment.\n\n"
    "Your feedback is valuable to us, not only in reviewing the process surrounding your submission "
    "but also in identifying areas where the review process can be improved. We will document the "
    "concerns raised and share relevant information with future AAAI Program Chairs to support "
    "continued improvements in the quality, fairness, and integrity of the review process. We "
    "appreciate your engagement with the process and your effort in bringing these concerns to our "
    "attention."
)
# Yan's AI-review reply, approved 2026-10-05 exactly as written — its four
# paragraphs as literals ("the authors' responses" wording accepted as is).
YAN_B_PARAGRAPHS = (
    "Thank you for providing the detailed information regarding your concerns about the reviews of "
    "your submission. We take concerns about the integrity and quality of the review process seriously "
    "and have carefully considered the issues you raised.",
    "We recognize that some characteristics of a review may raise concerns about the possible use of "
    "AI tools. But rest assured that the decision on your submission does not rely on any single "
    "review. The Senior Program Chair and/or Area Chair have also reviewed the paper, considered the "
    "reviews and the authors' responses, and formed their own assessment of the submission. The final "
    "decision is made based on this broader evaluation rather than on the assessment or "
    "recommendation of any individual reviewer.",
    "In addition, we ask Senior Program Chairs to assess the quality of the reviews and provide "
    "feedback on the reviewers, including identifying reviews that exhibit characteristics associated "
    "with AI-generated content. Your feedback is also very valuable to us. We will document these "
    "concerns and share the relevant information with future AAAI Program Chairs to help further "
    "improve the quality and integrity of the review process.",
    "Thank you again for raising your concerns and for providing the supporting details. We "
    "appreciate your engagement with the review process.",
)
G_YAN_B = "\n\n".join(YAN_B_PARAGRAPHS)

NOTE_NO_DRAFT = (
    "Investigate first: the author says a review is about a different paper. "
    "Do not reply to or close the ticket yet."
)
NOTE_RECIP = "Reciprocal-review complaint: tagged for Marc to review himself. No reply is drafted."

# (reasons, mode, expected body, expected used ids, rules that fire on the body)
GOLDEN_CASES = [
    ([SCORE], "merged", G_T1,
     ("opening_warm", "lead_in_concerns", "point_scores", "point_rebuttal", "closing_reviewed"), {ROLES}),
    ([REVIEWER], "merged", G_T2,
     ("opening_warm", "lead_in_concerns", "point_all_assessments", "point_rebuttal",
      "point_consider_input", "closing_reviewed"), set()),
    ([SCORE, REVIEWER], "merged", G_T1_T2,
     ("opening_warm", "lead_in_concerns", "point_scores", "point_all_assessments", "point_rebuttal",
      "point_consider_input", "closing_reviewed"), {ROLES}),
    ([SCORE, OTHER], "merged", G_T1_OTHER,
     ("opening_warm", "lead_in_concerns", "point_scores", "point_rebuttal", "line_chair_writes",
      "closing_reviewed"), {ROLES}),
    ([SCORE, REVIEWER, OTHER], "merged", G_T1_T2_OTHER,
     ("opening_warm", "lead_in_concerns", "point_scores", "point_all_assessments", "point_rebuttal",
      "point_consider_input", "line_chair_writes", "closing_reviewed"), {ROLES}),
    ([GENERAL], "standalone", G_YAN_A, ("standalone_general_stage1",), {ROLES}),
    ([LLM], "standalone", G_YAN_B, ("standalone_ai_review",), {ROLES}),
    ([OTHER], "chair_writes", CHAIR_LINE, ("line_chair_writes",), set()),
]
_IDS = ["+".join(c[0]) for c in GOLDEN_CASES]


# --- golden outputs --------------------------------------------------------------------------
@pytest.mark.parametrize("reasons, mode, expected, used, _rules", GOLDEN_CASES, ids=_IDS)
def test_golden_output(blocks, reasons, mode, expected, used, _rules):
    r = compose_reply(reasons, path=blocks)
    assert r.refusal is None, r.refusal
    assert (r.mode, r.body, r.used_ids, r.chair_notes) == (mode, expected, used, ())


@pytest.mark.parametrize("reasons, _mode, expected, _used, rules", GOLDEN_CASES, ids=_IDS)
def test_every_golden_output_trips_only_the_waived_rule(reasons, _mode, expected, _used, rules):
    """The finished email lints clean except for the one rule the used blocks waive."""
    assert {name for name, _ in lint_template_body(expected)} == rules


def test_every_golden_body_is_clean_text(blocks):
    """Blank line between paragraphs; no trailing whitespace; no trailing newline."""
    for reasons, *_ in GOLDEN_CASES:
        body = compose_reply(reasons, path=blocks).body
        assert not body.endswith("\n") and body == body.strip()
        assert all(line == line.rstrip() for line in body.split("\n"))
        assert "\n\n\n" not in body


# --- merged replies (T1 / T2) ------------------------------------------------------------------
def test_t1_plus_t2_has_the_rebuttal_point_exactly_once_in_the_fixed_order(blocks):
    body = compose_reply([SCORE, REVIEWER], path=blocks).body
    assert body.count("forgoes rebuttal") == 1
    assert re.findall(r"(?m)^\((\d)\)", body) == ["1", "2", "3", "4"]
    positions = [body.index(p) for p in (P_SCORES, P_REVIEWERS, P_REBUTTAL, P_THANKS)]
    assert positions == sorted(positions)


def test_the_chair_line_sits_after_the_last_point_and_before_the_closing(blocks):
    paras = compose_reply([SCORE, OTHER], path=blocks).body.split("\n\n")
    assert paras[-2] == CHAIR_LINE
    assert paras[-3].startswith("(2) ")
    assert paras[-1].startswith("The decision is final")


def test_every_ordering_of_a_three_reason_set_gives_identical_output(blocks):
    results = {compose_reply(list(p), path=blocks) for p in itertools.permutations([SCORE, REVIEWER, OTHER])}
    assert len(results) == 1
    assert next(iter(results)).body == G_T1_T2_OTHER


def test_duplicate_reasons_count_once(blocks):
    assert compose_reply([SCORE, SCORE], path=blocks).body == G_T1


# --- Yan's standalone replies -------------------------------------------------------------------
@pytest.mark.parametrize("reason, block_id", [(GENERAL, "standalone_general_stage1"),
                                              (LLM, "standalone_ai_review")])
def test_a_standalone_reply_has_no_opening_list_or_closing(blocks, reason, block_id):
    r = compose_reply([reason], path=blocks)
    assert (r.mode, r.used_ids) == ("standalone", (block_id,))
    assert OPENING not in r.body and CLOSING not in r.body
    assert not re.search(r"(?m)^\(\d\)", r.body)


def test_an_approved_block_that_is_still_blocked_is_refused(tmp_path):
    """The blocker rule still holds even though the real file no longer blocks any
    live block: approved but carrying a blocker, the loader refuses the block and
    the composer refuses the reply. (Rewritten 2026-10-05: this used the real Yan B
    blocker, which is gone since its approval.)"""
    p = approved_copy(tmp_path, overrides={"standalone_ai_review": {"blocked_on": ["waiting"]}})
    r = compose_reply([LLM], path=p)
    assert (r.mode, r.body, r.refusal) == ("refused", None, "missing_approved_block:standalone_ai_review")


@pytest.mark.parametrize("reasons, note", [
    ([GENERAL, SCORE],
     "Chair writes: no approved reply covers these reasons together: "
     "score_outcome_mismatch, general_dissatisfaction."),
    ([LLM, SCORE],
     "Chair writes: no approved reply covers these reasons together: "
     "score_outcome_mismatch, llm_generated_review."),
    ([GENERAL, LLM],
     "Chair writes: no approved reply covers these reasons together: "
     "llm_generated_review, general_dissatisfaction."),
    ([LLM, OTHER],
     "Chair writes: no approved reply covers these reasons together: llm_generated_review, other."),
    ([GENERAL, REVIEWER, OTHER],
     "Chair writes: no approved reply covers these reasons together: "
     "reviewer_misunderstanding, general_dissatisfaction, other."),
], ids=["general+score", "llm+score", "general+llm", "llm+other", "general+reviewer+other"])
def test_a_standalone_reason_mixed_with_any_other_reason_goes_to_the_chair(blocks, reasons, note):
    r = compose_reply(reasons, path=blocks)
    assert r.refusal is None
    assert (r.mode, r.body, r.used_ids, r.chair_notes) == (
        "chair_writes", CHAIR_LINE, ("line_chair_writes",), (note,))


def test_more_than_three_issues_go_to_the_chair(blocks):
    r = compose_reply([SCORE, REVIEWER, LLM, OTHER], path=blocks)
    assert (r.mode, r.body, r.refusal) == ("chair_writes", CHAIR_LINE, None)
    assert r.chair_notes == ("Chair writes: no approved reply covers these reasons together: "
                             "score_outcome_mismatch, reviewer_misunderstanding, llm_generated_review, other.",)


def test_the_issue_count_guard_sends_a_too_large_merged_set_to_the_chair(blocks, monkeypatch):
    """With today's registry rule 3 catches every set larger than three before the
    guard; lowering the limit is the only way to reach the guard itself."""
    monkeypatch.setattr(arc, "MAX_REASONS", 2)
    r = compose_reply([SCORE, REVIEWER, OTHER], path=blocks)
    assert (r.mode, r.body, r.refusal) == ("chair_writes", CHAIR_LINE, None)
    assert r.chair_notes == ("Chair writes: more than 2 issues raised: "
                             "score_outcome_mismatch, reviewer_misunderstanding, other.",)


# --- no draft: wrong-paper review ---------------------------------------------------------------
def test_a_wrong_paper_review_alone_is_no_draft(blocks):
    r = compose_reply([WRONG], path=blocks)
    assert r == ComposeResult(body=None, mode="no_draft", used_ids=(), chair_notes=(NOTE_NO_DRAFT,),
                              refusal=None)


@pytest.mark.parametrize("reasons, also", [
    ([WRONG, SCORE], "Also raised: score_outcome_mismatch."),
    ([WRONG, REVIEWER, LLM, OTHER],
     "Also raised: reviewer_misunderstanding, llm_generated_review, other."),
    ([WRONG, RECIP], "Also raised: reciprocal_dispute."),
    ([WRONG, GENERAL, RECIP], "Also raised: general_dissatisfaction, reciprocal_dispute."),
], ids=["with-score", "with-three-more", "beats-reciprocal", "beats-reciprocal-and-general"])
def test_a_wrong_paper_review_with_other_reasons_is_no_draft_listing_them(blocks, reasons, also):
    r = compose_reply(reasons, path=blocks)
    assert r == ComposeResult(body=None, mode="no_draft", used_ids=(),
                              chair_notes=(NOTE_NO_DRAFT, also), refusal=None)


# --- reciprocal review ---------------------------------------------------------------------------
def test_a_reciprocal_complaint_alone_is_tagged_for_marc(blocks):
    r = compose_reply([RECIP], path=blocks)
    assert r == ComposeResult(body=None, mode="reciprocal_review", used_ids=(),
                              chair_notes=(NOTE_RECIP,), refusal=None)


@pytest.mark.parametrize("reasons, also", [
    ([RECIP, SCORE, GENERAL], "Also raised: score_outcome_mismatch, general_dissatisfaction."),
    ([RECIP, LLM], "Also raised: llm_generated_review."),
], ids=["with-score-and-general", "beats-the-mixing-rule"])
def test_a_reciprocal_complaint_with_other_reasons_lists_them(blocks, reasons, also):
    r = compose_reply(reasons, path=blocks)
    assert r == ComposeResult(body=None, mode="reciprocal_review", used_ids=(),
                              chair_notes=(NOTE_RECIP, also), refusal=None)


def test_full_reciprocal_is_never_served_even_when_approved(blocks):
    """D111: stored approved, never served (the approved copy approves it)."""
    assert "full_reciprocal" in {t.id for t in art.load_approved_templates(blocks)}
    r = compose_reply([RECIP], path=blocks)
    assert r.body is None and "full_reciprocal" not in r.used_ids


def test_no_draft_and_reciprocal_review_are_not_refusals_and_need_no_block(tmp_path):
    """A deliberate "no draft" is told apart from a failure by refusal is None,
    and it does not depend on any approved block existing."""
    missing = tmp_path / "absent.json"
    assert compose_reply([WRONG], path=missing).mode == "no_draft"
    assert compose_reply([RECIP], path=missing).mode == "reciprocal_review"
    assert compose_reply([SCORE], path=missing).mode == "refused"
    for reasons in ([WRONG], [RECIP]):
        assert compose_reply(reasons, path=missing).refusal is None


# --- retired blocks are never used ------------------------------------------------------------------
def test_retired_blocks_are_never_used_even_if_approved(tmp_path):
    p = approved_copy(tmp_path, approve_retired=True)
    retired = {"point_ai_review", "point_report_form", "holding_wrong_paper",
               "holding_score_mismatch", "holding_both", "body_reconsider"}
    # point_report_form still holds its [ETHICS FORM ADDRESS] placeholder, which the
    # loader refuses once unblocked, so it is the one retired block not served here.
    assert retired - {"point_report_form"} <= {t.id for t in art.load_approved_templates(p)}, \
        "the copy approves them"
    for reasons, *_ in GOLDEN_CASES:
        assert not retired & set(compose_reply(reasons, path=p).used_ids), reasons
    assert compose_reply([WRONG], path=p).mode == "no_draft"
    assert compose_reply([LLM], path=p).body == G_YAN_B


def test_the_forward_choice_is_gone():
    """D97: the score-mismatch forward reply and its parameter are removed."""
    assert "chair_forwards_score_mismatch" not in inspect.signature(compose_reply).parameters


# --- refusals --------------------------------------------------------------------------------------
@pytest.mark.parametrize("missing, reasons", [
    ("opening_warm", [SCORE]),
    ("lead_in_concerns", [SCORE]),
    ("closing_reviewed", [SCORE]),
    ("point_scores", [SCORE]),
    ("point_rebuttal", [SCORE]),
    ("point_all_assessments", [REVIEWER]),
    ("point_consider_input", [REVIEWER]),
    ("standalone_general_stage1", [GENERAL]),
    ("standalone_ai_review", [LLM]),
    ("line_chair_writes", [OTHER]),
    ("line_chair_writes", [SCORE, OTHER]),
    ("line_chair_writes", [GENERAL, SCORE]),
])
def test_a_missing_approved_block_refuses_with_its_id(tmp_path, missing, reasons):
    p = approved_copy(tmp_path, draft={missing})
    r = compose_reply(reasons, path=p)
    assert (r.mode, r.refusal, r.body) == ("refused", f"missing_approved_block:{missing}", None)


def test_the_score_point_without_its_waiver_is_not_served_so_t1_is_refused(tmp_path):
    p = approved_copy(tmp_path, overrides={"point_scores": {"lint_waivers": []}})
    r = compose_reply([SCORE], path=p)
    assert (r.mode, r.refusal, r.body) == ("refused", "missing_approved_block:point_scores", None)


@pytest.mark.parametrize("reasons, refusal", [
    (["made_up_reason"], "unknown_reason"),
    ([SCORE, "a"], "unknown_reason"),
    ([], "no_reasons"),
])
def test_unknown_or_empty_reasons_are_refused(blocks, reasons, refusal):
    r = compose_reply(reasons, path=blocks)
    assert (r.mode, r.refusal, r.body) == ("refused", refusal, None)


def test_a_composed_body_that_fails_the_lint_is_refused_without_text(tmp_path, caplog):
    """Each block is clean alone; the JOIN forms "this year" across the opening
    and the lead-in, so only the composer's final lint can catch it."""
    p = approved_copy(tmp_path, overrides={
        "opening_warm": {"body": "We understand this outcome, and we reply this"},
        "lead_in_concerns": {"body": "year as follows:"},
    })
    with caplog.at_level(logging.WARNING):
        r = compose_reply([REVIEWER], path=p)
    assert (r.mode, r.refusal, r.body) == ("refused", "lint:year_or_time_specific", None)
    assert "this year" not in caplog.text and "as follows" not in caplog.text


@pytest.mark.parametrize("junk", [None, "a string", 42, 3.5, {"score": 1}, [SCORE, None],
                                  [[SCORE]], [{"x": 1}], b"bytes", object()])
def test_never_raises_on_junk(blocks, junk):
    r = compose_reply(junk, path=blocks)
    assert isinstance(r, ComposeResult)
    assert r.mode == "refused" and r.body is None and r.refusal


def test_a_missing_file_is_refused_not_raised(tmp_path):
    r = compose_reply([SCORE], path=tmp_path / "absent.json")
    assert r.mode == "refused" and r.body is None


# --- the REAL file (approved in Step 3c; Yan's AI-review reply on 2026-10-05) -----------------------------
# No temp copies here: these run on data/reply_templates/appeal_reply_templates.json
# exactly as approved, against the hand-written literal goldens above.
@pytest.mark.parametrize("reasons, expected, used", [
    ([SCORE], G_T1,
     ("opening_warm", "lead_in_concerns", "point_scores", "point_rebuttal", "closing_reviewed")),
    ([REVIEWER], G_T2,
     ("opening_warm", "lead_in_concerns", "point_all_assessments", "point_rebuttal",
      "point_consider_input", "closing_reviewed")),
    ([SCORE, REVIEWER], G_T1_T2,
     ("opening_warm", "lead_in_concerns", "point_scores", "point_all_assessments", "point_rebuttal",
      "point_consider_input", "closing_reviewed")),
    ([GENERAL], G_YAN_A, ("standalone_general_stage1",)),
    ([LLM], G_YAN_B, ("standalone_ai_review",)),
], ids=["T1-scores", "T2-reviewer", "T1+T2", "yan-a-general", "yan-b-ai-review"])
def test_the_real_file_composes_the_approved_replies(reasons, expected, used):
    r = compose_reply(reasons)
    assert r.refusal is None, r.refusal
    assert (r.body, r.used_ids, r.chair_notes) == (expected, used, ())
    assert r.mode == ("standalone" if reasons in ([GENERAL], [LLM]) else "merged")


def test_the_real_file_composes_t1_t2_with_the_rebuttal_point_once():
    body = compose_reply([SCORE, REVIEWER]).body
    assert body.count("forgoes rebuttal") == 1
    assert re.findall(r"(?m)^\((\d)\)", body) == ["1", "2", "3", "4"]


def test_the_real_file_composes_yans_ai_review_reply_alone_as_exactly_her_four_paragraphs():
    """Approved 2026-10-05 exactly as written. (Rewritten: until then the real file
    refused this reply while Yan B was blocked.)"""
    r = compose_reply([LLM])
    assert r == ComposeResult(body=G_YAN_B, mode="standalone", used_ids=("standalone_ai_review",),
                              chair_notes=(), refusal=None)
    assert tuple(r.body.split("\n\n")) == YAN_B_PARAGRAPHS
    assert "the authors' responses" in r.body, "accepted as written, not edited"
    assert OPENING not in r.body and CLOSING not in r.body, "a standalone reply has no wrapper"


@pytest.mark.parametrize("reasons, note", [
    ([LLM, SCORE], "Chair writes: no approved reply covers these reasons together: "
                   "score_outcome_mismatch, llm_generated_review."),
    ([LLM, REVIEWER], "Chair writes: no approved reply covers these reasons together: "
                      "reviewer_misunderstanding, llm_generated_review."),
    ([LLM, GENERAL], "Chair writes: no approved reply covers these reasons together: "
                     "llm_generated_review, general_dissatisfaction."),
    ([LLM, OTHER], "Chair writes: no approved reply covers these reasons together: "
                   "llm_generated_review, other."),
    ([LLM, SCORE, REVIEWER], "Chair writes: no approved reply covers these reasons together: "
                             "score_outcome_mismatch, reviewer_misunderstanding, llm_generated_review."),
], ids=["+score", "+reviewer", "+general", "+other", "+score+reviewer"])
def test_the_real_file_still_sends_the_ai_review_reason_mixed_with_another_to_the_chair(reasons, note):
    """The mixing rule (D105) is unchanged by the approval: mixed with any reason
    that would otherwise be composed, the AI-review reason gives chair_writes."""
    r = compose_reply(reasons)
    assert r == ComposeResult(body=CHAIR_LINE, mode="chair_writes", used_ids=("line_chair_writes",),
                              chair_notes=(note,), refusal=None)


def test_the_real_file_lets_wrong_paper_and_reciprocal_beat_the_ai_review_reason():
    """The two earlier rules still win over the AI-review reason: never Yan's text."""
    assert compose_reply([LLM, WRONG]) == ComposeResult(
        body=None, mode="no_draft", used_ids=(),
        chair_notes=(NOTE_NO_DRAFT, "Also raised: llm_generated_review."), refusal=None)
    assert compose_reply([LLM, RECIP]) == ComposeResult(
        body=None, mode="reciprocal_review", used_ids=(),
        chair_notes=(NOTE_RECIP, "Also raised: llm_generated_review."), refusal=None)


def test_the_real_file_gives_no_draft_for_a_wrong_paper_review():
    assert compose_reply([WRONG]) == ComposeResult(
        body=None, mode="no_draft", used_ids=(), chair_notes=(NOTE_NO_DRAFT,), refusal=None)
    assert compose_reply([WRONG, SCORE]).chair_notes == (NOTE_NO_DRAFT, "Also raised: score_outcome_mismatch.")


def test_the_real_file_gives_reciprocal_review_and_never_serves_full_reciprocal():
    assert "full_reciprocal" in {t.id for t in art.load_approved_templates()}, "approved, D111"
    for reasons in ([RECIP], [RECIP, SCORE], [RECIP, GENERAL]):
        r = compose_reply(reasons)
        assert (r.mode, r.body, r.refusal, r.used_ids) == ("reciprocal_review", None, None, ())
        assert r.chair_notes[0] == NOTE_RECIP


def test_the_real_file_gives_the_chair_writes_line_for_other():
    r = compose_reply([OTHER])
    assert (r.mode, r.body, r.used_ids, r.chair_notes, r.refusal) == (
        "chair_writes", CHAIR_LINE, ("line_chair_writes",), (), None)


def test_the_real_file_puts_the_chair_line_inside_a_merged_reply_for_other():
    assert compose_reply([SCORE, OTHER]).body == G_T1_OTHER


@pytest.mark.parametrize("reasons, note", [
    ([GENERAL, SCORE], "Chair writes: no approved reply covers these reasons together: "
                       "score_outcome_mismatch, general_dissatisfaction."),
    ([LLM, SCORE], "Chair writes: no approved reply covers these reasons together: "
                   "score_outcome_mismatch, llm_generated_review."),
    ([GENERAL, LLM], "Chair writes: no approved reply covers these reasons together: "
                     "llm_generated_review, general_dissatisfaction."),
], ids=["yan-a+score", "yan-b+score", "yan-a+yan-b"])
def test_the_real_file_sends_a_yan_reply_mixed_with_another_reason_to_the_chair(reasons, note):
    r = compose_reply(reasons)
    assert (r.mode, r.body, r.used_ids, r.chair_notes, r.refusal) == (
        "chair_writes", CHAIR_LINE, ("line_chair_writes",), (note,), None)


# --- expected_points_for_reasons (loader helper) --------------------------------------------------
def test_expected_points_include_unapproved_points_in_the_global_order():
    """The real file is all draft or retired, yet the helper still lists what is
    needed. It lists retired points too (it does not read status); the composer
    never asks it about llm_generated_review, which is a standalone reply."""
    assert expected_points_for_reasons([SCORE]) == [("point_scores", False), ("point_rebuttal", False)]
    assert expected_points_for_reasons([REVIEWER]) == [
        ("point_all_assessments", False), ("point_rebuttal", False), ("point_consider_input", False),
    ]
    assert expected_points_for_reasons([SCORE, REVIEWER]) == [
        ("point_scores", False), ("point_all_assessments", False),
        ("point_rebuttal", False), ("point_consider_input", False),
    ]
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
