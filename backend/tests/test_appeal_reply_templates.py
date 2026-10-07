"""Structural checks on data/reply_templates/appeal_reply_templates.json (Phase 3, D91/D92/D95).

STRUCTURE ONLY — the wording rules are in test_appeal_reply_lint.py and the
refusal rules in test_appeal_reply_template_loader.py. Schema 2 (D95): the file
holds composable BLOCKS (opening, lead-in, numbered points, closing, holding
replies, standalone replies), not whole emails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.appeal_reasons import REASON_NAMES
from app.pipeline.appeal_reply_templates import KINDS

# Not an appeal reason: the reciprocal block serves the `is_reciprocal_dispute`
# flag, which the registry deliberately excludes (D60).
RECIPROCAL = "reciprocal_dispute"
# Not in the registry either: phase-1 classifier names the composer accepts
# (Step 2.5), served by Marc's misconduct points and the score point.
COMPOSER_ONLY = {"reviewer_misconduct", "missing_material_claim", "record_error"}

_TEMPLATES = (
    Path(__file__).resolve().parents[2] / "data" / "reply_templates" / "appeal_reply_templates.json"
)

EXPECTED_IDS = {
    "opening_warm", "opening_warm_plural", "lead_in_concerns",
    "point_review_process", "point_spc_evaluation", "point_reviewer_tracking", "point_ethics_form",
    "point_scores", "point_all_assessments", "point_rebuttal", "point_consider_input",
    "point_ai_review", "point_report_form",
    "closing_reviewed", "closing_feedback",
    "holding_wrong_paper", "holding_score_mismatch", "holding_both",
    "body_reconsider", "line_chair_writes", "full_reciprocal",
    "standalone_general_stage1", "standalone_ai_review",
}

# Who approved each block, and when. The exact hashes are pinned in
# test_appeal_reply_template_loader.py (APPROVED_PINS). The blocks whose text the
# program chairs rewrote (and the new review-process point) were approved on
# 2026-10-06 by Jiacheng Pang; their earlier approvers have not seen the new text.
# The ethics-form point kept its body, so Marc's approval of it still holds.
_MARC, _JIACHENG, _SAHIL = "Marc Pujol-Gonzalez", "Jiacheng Pang", "Sahil Satasiya"
REWORDED = (
    "point_scores", "point_rebuttal", "point_reviewer_tracking",
    "closing_reviewed", "standalone_general_stage1",
)
# Approved 2026-10-07 by Jiacheng Pang: the merged AI-review point, the
# feedback-only closing, the plural opening, and the review-process point
# (re-approved with llm_generated_review added to its reasons).
ADDED = ("point_review_process", "point_ai_review", "closing_feedback", "opening_warm_plural")
APPROVERS = {
    "opening_warm": _MARC, "lead_in_concerns": _MARC, "full_reciprocal": _MARC,
    "point_ethics_form": _MARC, "line_chair_writes": _SAHIL,
} | {block_id: _JIACHENG for block_id in (*REWORDED, *ADDED)}
APPROVAL_DATES = {
    "opening_warm": "2026-10-02", "lead_in_concerns": "2026-10-02", "full_reciprocal": "2026-10-02",
    "line_chair_writes": "2026-10-02", "point_ethics_form": "2026-10-05",
} | {block_id: "2026-10-06" for block_id in REWORDED} | {
    block_id: "2026-10-07" for block_id in ADDED}

# Kept in the file but no longer used (D97/D98/D101/D104), plus the three points the
# chairs' rewording replaced (folded into point_review_process).
RETIRED_IDS = {
    "standalone_ai_review", "point_report_form",
    "holding_wrong_paper", "holding_score_mismatch", "holding_both",
    "body_reconsider",
    "point_spc_evaluation", "point_all_assessments", "point_consider_input",
}

REQUIRED_FIELDS = {
    "id", "title", "kind", "order", "optional", "reasons", "when_used", "body", "status",
    "approved_by", "approved_at", "approved_sha256", "cycle", "scope", "basis", "blocked_on",
}
STATUSES = {"draft", "approved", "retired"}


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(_TEMPLATES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def templates(data) -> list[dict]:
    return data["templates"]


def test_the_file_is_schema_version_2(data):
    assert data["schema_version"] == 2


def test_the_file_holds_exactly_the_expected_ids(templates):
    ids = [t["id"] for t in templates]
    assert len(ids) == 23
    assert len(set(ids)) == 23, "template ids must be unique"
    assert set(ids) == EXPECTED_IDS


def test_every_entry_has_all_required_fields(templates):
    for t in templates:
        missing = REQUIRED_FIELDS - set(t)
        assert not missing, f"{t.get('id')}: missing {sorted(missing)}"


def test_every_kind_is_an_allowed_kind(templates):
    for t in templates:
        assert t["kind"] in KINDS, f"{t['id']}: kind {t['kind']!r}"


def test_the_openings_lead_in_and_closings(templates):
    """One lead-in; an opening for one paper and one for several; a closing for
    appeals and one for feedback-only emails."""
    expected = {"opening": {"opening_warm", "opening_warm_plural"}, "lead_in": {"lead_in_concerns"},
                "closing": {"closing_reviewed", "closing_feedback"}}
    for kind, ids in expected.items():
        assert {t["id"] for t in templates if t["kind"] == kind} == ids, kind


def test_every_point_has_a_unique_integer_order(templates):
    orders = [t["order"] for t in templates if t["kind"] == "point"]
    assert orders, "there must be points"
    assert all(isinstance(o, int) and not isinstance(o, bool) for o in orders), orders
    assert len(orders) == len(set(orders)), f"duplicate point order: {orders}"


def test_only_points_have_an_order(templates):
    for t in templates:
        if t["kind"] != "point":
            assert t["order"] is None, f"{t['id']}: non-point with an order"


def test_optional_is_a_bool(templates):
    for t in templates:
        assert isinstance(t["optional"], bool), t["id"]


def test_status_is_one_of_draft_approved_retired(templates):
    for t in templates:
        assert t["status"] in STATUSES, f"{t['id']}: status {t['status']!r}"


def test_an_approved_entry_carries_its_approval_record(templates):
    """Checked on the live file AND on a synthetic entry, so the rule is proven
    even while no entry is approved yet."""
    def ok(t: dict) -> bool:
        return t["status"] != "approved" or all(
            t.get(k) for k in ("approved_by", "approved_at", "approved_sha256")
        )

    for t in templates:
        assert ok(t), f"{t['id']}: approved without approved_by/approved_at/approved_sha256"
    assert not ok({"status": "approved", "approved_by": "x", "approved_at": None, "approved_sha256": "y"})
    assert ok({"status": "approved", "approved_by": "x", "approved_at": "2026-10-01", "approved_sha256": "y"})


def test_every_reason_is_a_registry_name_reciprocal_or_composer_only(templates):
    """Full registry names (D59/D92) — never the scoring letter codes — plus
    reciprocal_dispute and the three composer-only names (Step 2.5). Framing
    blocks (opening, lead-in, closing) serve every reason and list none."""
    allowed = set(REASON_NAMES) | {RECIPROCAL} | COMPOSER_ONLY
    for t in templates:
        if t["kind"] not in ("opening", "lead_in", "closing"):
            assert t["reasons"], f"{t['id']}: empty reasons"
        bad = [r for r in t["reasons"] if r not in allowed]
        assert not bad, f"{t['id']}: unknown reason value(s) {bad}"


def test_every_entry_has_the_cycle_and_scope(templates):
    for t in templates:
        assert (t["cycle"], t["scope"]) == ("AAAI-27", "phase1_reject"), t["id"]


def test_exactly_the_expected_entries_are_approved_retired_and_draft(templates):
    """Fourteen blocks are approved, each with its approver and date: five the
    chairs reworded (2026-10-06), four added or changed on 2026-10-07, and five
    whose text did not change (opening, lead-in, reciprocal reply, chair line,
    ethics-form point). The nine retired ones stay retired; no block is draft.
    Unapproved entries carry no approval record."""
    status = {t["id"]: t["status"] for t in templates}
    assert {i for i, s in status.items() if s == "approved"} == set(APPROVERS)
    assert {i for i, s in status.items() if s == "retired"} == RETIRED_IDS
    assert {i for i, s in status.items() if s == "draft"} == set()
    for t in templates:
        if t["status"] == "approved":
            assert (t["approved_by"], t["approved_at"]) == (APPROVERS[t["id"]], APPROVAL_DATES[t["id"]]), t["id"]
            assert isinstance(t["approved_sha256"], str) and len(t["approved_sha256"]) == 64, t["id"]
        else:
            assert (t["approved_by"], t["approved_at"], t["approved_sha256"]) == (None, None, None), t["id"]


def test_the_new_standalone_blocks_are_complete_middles_for_one_reason(templates):
    """Yan's two replies (D103/D104) are standalone_full blocks: no wrapper, no order."""
    by_id = {t["id"]: t for t in templates}
    for block_id, reason in (("standalone_general_stage1", "general_dissatisfaction"),
                             ("standalone_ai_review", "llm_generated_review")):
        t = by_id[block_id]
        assert (t["kind"], t["order"], t["reasons"]) == ("standalone_full", None, [reason]), block_id
        assert len(t["body"].split("\n\n")) == 4, block_id
        assert not t["body"].lstrip().lower().startswith("dear"), "middle text only"


def test_the_live_points_have_the_decided_reasons_and_order(templates):
    """The chairs' point order: the review-process point opens every merged reply
    (every merged reason needs it), then scores, rebuttal, reviewer tracking and
    the ethics form. record_error shares the score point without the rebuttal
    point; misconduct adds tracking and the ethics form."""
    live = {t["id"]: (t["order"], t["reasons"]) for t in templates
            if t["kind"] == "point" and t["status"] != "retired"}
    assert live == {
        "point_review_process": (1, ["score_outcome_mismatch", "reviewer_misunderstanding",
                                     "missing_material_claim", "record_error",
                                     "reviewer_misconduct", "llm_generated_review"]),
        "point_scores": (2, ["score_outcome_mismatch", "record_error"]),
        "point_rebuttal": (3, ["score_outcome_mismatch", "reviewer_misunderstanding"]),
        "point_ai_review": (4, ["llm_generated_review"]),
        "point_reviewer_tracking": (5, ["reviewer_misconduct"]),
        "point_ethics_form": (6, ["reviewer_misconduct"]),
    }


def test_the_retired_points_sit_after_the_live_points(templates):
    """Renumbered after the six live points so no two points share an order."""
    retired = {t["id"]: t["order"] for t in templates
               if t["kind"] == "point" and t["status"] == "retired"}
    assert retired == {"point_spc_evaluation": 7, "point_all_assessments": 8,
                       "point_consider_input": 9, "point_report_form": 10}


def test_the_blockers_are_as_decided(templates):
    """Marc answered question (d), so the reciprocal reply is unblocked (it is still
    never served, D111). Yan's AI-review reply is unblocked too: Sahil accepted its
    "the authors' responses" wording as written (2026-10-05), so the blocker that
    waited on Yan's confirmation is gone from the file entirely."""
    by_id = {t["id"]: t for t in templates}
    assert by_id["full_reciprocal"]["blocked_on"] == []
    assert by_id["standalone_ai_review"]["blocked_on"] == []
    assert by_id["standalone_general_stage1"]["blocked_on"] == []
    assert not any("yan_confirm_authors_responses" in t["blocked_on"] for t in templates)
