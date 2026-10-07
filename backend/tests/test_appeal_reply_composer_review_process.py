"""Composed replies under the program chairs' rewording (reject-appeal, 2026-10-06).

The review-process point opens every merged reply, so each reply states once that
no single review decides; the closing is the only place that says the decision is
final. Points follow one global order: review_process, scores, rebuttal,
reviewer_tracking, ethics_form.

Every expected body is a LITERAL string. Each case runs twice: on the real
template file (as approved on 2026-10-06) and on a temp copy in which every
non-retired block is re-approved with a correct hash (a test fixture only), so
the real file and a freshly approved copy must compose the same text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.pipeline import appeal_reply_templates as art
from app.pipeline.appeal_reply_composer import compose_reply
from app.pipeline.appeal_reply_templates import compute_body_sha256

SCORE, REVIEWER = "score_outcome_mismatch", "reviewer_misunderstanding"
MISCONDUCT, MISSING, RECORD = "reviewer_misconduct", "missing_material_claim", "record_error"

OPENING = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:"
)
REVIEW_PROCESS = (
    "Decisions are not based on any single review or on the visible scores alone. SPCs evaluated both "
    "the paper and the reviews, and all assessments were weighed together."
)
SCORES = (
    "SPCs also considered whether the concerns raised could be addressed with minor clarifications or "
    "would require substantial revision."
)
REBUTTAL = (
    "AAAI's two-phase process forgoes rebuttal for Phase 1 papers in favor of a quicker decision. We "
    "understand this can be frustrating."
)
REVIEWER_TRACKING = (
    "We ask SPCs and ACs to report reviewers who are unprofessional. We keep track of reviewers' "
    "performance and take it into account in future editions."
)
ETHICS_FORM = (
    "You can also report unethical behavior through the ethics report form at "
    "https://docs.google.com/forms/d/e/1FAIpQLSdIs72RunUy5wKsOv7SdBma6A6riv3jp8lifUxlLcwhcdXMxw/viewform. "
    "This may impact our future relationship with this reviewer, but it will not change the outcome "
    "for this specific paper."
)
CLOSING = (
    "The decision is final, but we hope the feedback will be useful in further strengthening your work "
    "and helping you secure publication in another leading venue or future AAAI edition. Thank you for "
    "raising your concerns; we will document them and help improve the future AAAI editions."
)

# The one global point order, as (block id, literal body).
POINT_ORDER = (
    ("point_review_process", REVIEW_PROCESS),
    ("point_scores", SCORES),
    ("point_rebuttal", REBUTTAL),
    ("point_reviewer_tracking", REVIEWER_TRACKING),
    ("point_ethics_form", ETHICS_FORM),
)


def _reply(*points: str) -> str:
    numbered = [f"({n}) {body}" for n, body in enumerate(points, start=1)]
    return "\n\n".join([OPENING, *numbered, CLOSING])


SCORES_ALONE = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Decisions are not based on any single review or on the visible scores alone. SPCs evaluated "
    "both the paper and the reviews, and all assessments were weighed together.\n\n"
    "(2) SPCs also considered whether the concerns raised could be addressed with minor clarifications "
    "or would require substantial revision.\n\n"
    "(3) AAAI's two-phase process forgoes rebuttal for Phase 1 papers in favor of a quicker decision. "
    "We understand this can be frustrating.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your work "
    "and helping you secure publication in another leading venue or future AAAI edition. Thank you for "
    "raising your concerns; we will document them and help improve the future AAAI editions."
)

# (case id, reasons, expected point ids in order, expected full body)
CASES = [
    ("scores", [SCORE],
     ("point_review_process", "point_scores", "point_rebuttal"),
     SCORES_ALONE),
    ("record_error", [RECORD],
     ("point_review_process", "point_scores"),
     _reply(REVIEW_PROCESS, SCORES)),
    ("misjudgment", [REVIEWER],
     ("point_review_process", "point_rebuttal"),
     _reply(REVIEW_PROCESS, REBUTTAL)),
    ("missing_material", [MISSING],
     ("point_review_process",),
     _reply(REVIEW_PROCESS)),
    ("misconduct", [MISCONDUCT],
     ("point_review_process", "point_reviewer_tracking", "point_ethics_form"),
     _reply(REVIEW_PROCESS, REVIEWER_TRACKING, ETHICS_FORM)),
    ("scores+misjudgment+misconduct", [SCORE, REVIEWER, MISCONDUCT],
     ("point_review_process", "point_scores", "point_rebuttal", "point_reviewer_tracking",
      "point_ethics_form"),
     _reply(REVIEW_PROCESS, SCORES, REBUTTAL, REVIEWER_TRACKING, ETHICS_FORM)),
]
_IDS = [c[0] for c in CASES]


def _approved_copy(tmp_path: Path) -> Path:
    """Test fixture only: a temp copy of the real file with every non-retired block
    re-approved with a correct hash. Never written back to the real file."""
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    for e in data["templates"]:
        if e["status"] == "retired":
            continue
        e.update(status="approved", approved_by="test fixture", approved_at="2026-10-06",
                 approved_sha256=compute_body_sha256(e["body"]))
    p = tmp_path / "templates.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture(params=["real_file", "approved_copy"])
def source(request, tmp_path) -> Path:
    return art.DEFAULT_PATH if request.param == "real_file" else _approved_copy(tmp_path)


def test_the_scores_literal_matches_the_helper():
    """The one fully spelled-out literal guards the helper the other cases use."""
    assert SCORES_ALONE == _reply(REVIEW_PROCESS, SCORES, REBUTTAL)


@pytest.mark.parametrize("_case, reasons, points, expected", CASES, ids=_IDS)
def test_the_composed_reply_is_exactly_the_expected_text(source, _case, reasons, points, expected):
    r = compose_reply(reasons, path=source)
    assert r.refusal is None, r.refusal
    assert r.mode == "merged"
    assert r.body == expected
    assert r.used_ids == ("opening_warm", "lead_in_concerns", *points, "closing_reviewed")
    assert r.chair_notes == ()


@pytest.mark.parametrize("_case, reasons, points, _expected", CASES, ids=_IDS)
def test_points_follow_the_one_global_order(source, _case, reasons, points, _expected):
    r = compose_reply(reasons, path=source)
    order = [block_id for block_id, _ in POINT_ORDER]
    used_points = [i for i in r.used_ids if i.startswith("point_")]
    assert used_points == list(points)
    assert used_points == sorted(used_points, key=order.index)
    bodies = dict(POINT_ORDER)
    positions = [r.body.index(bodies[i]) for i in used_points]
    assert positions == sorted(positions)
    assert re.findall(r"(?m)^\((\d)\)", r.body) == [str(n) for n in range(1, len(points) + 1)]


@pytest.mark.parametrize("_case, reasons, _points, _expected", CASES, ids=_IDS)
def test_no_single_review_is_said_exactly_once(source, _case, reasons, _points, _expected):
    body = compose_reply(reasons, path=source).body
    assert body.count("Decisions are not based on any single review") == 1


@pytest.mark.parametrize("_case, reasons, _points, _expected", CASES, ids=_IDS)
def test_final_appears_only_in_the_closing(source, _case, reasons, _points, _expected):
    body = compose_reply(reasons, path=source).body
    paragraphs = body.split("\n\n")
    assert paragraphs[-1] == CLOSING
    assert [p for p in paragraphs if re.search(r"\bfinal\b", p, re.I)] == [CLOSING]
    assert len(re.findall(r"\bfinal\b", body, re.I)) == len(re.findall(r"\bfinal\b", CLOSING, re.I)) == 1
