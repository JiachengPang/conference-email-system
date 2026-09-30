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

_TEMPLATES = (
    Path(__file__).resolve().parents[2] / "data" / "reply_templates" / "appeal_reply_templates.json"
)

EXPECTED_IDS = {
    "opening_warm", "lead_in_concerns",
    "point_scores", "point_all_assessments", "point_rebuttal", "point_consider_input",
    "point_ai_review", "point_report_form",
    "closing_reviewed",
    "holding_wrong_paper", "holding_score_mismatch", "holding_both",
    "body_reconsider", "line_chair_writes", "full_reciprocal",
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


def test_the_file_holds_exactly_the_fifteen_expected_ids(templates):
    ids = [t["id"] for t in templates]
    assert len(ids) == 15
    assert len(set(ids)) == 15, "template ids must be unique"
    assert set(ids) == EXPECTED_IDS


def test_every_entry_has_all_required_fields(templates):
    for t in templates:
        missing = REQUIRED_FIELDS - set(t)
        assert not missing, f"{t.get('id')}: missing {sorted(missing)}"


def test_every_kind_is_an_allowed_kind(templates):
    for t in templates:
        assert t["kind"] in KINDS, f"{t['id']}: kind {t['kind']!r}"


def test_exactly_one_opening_one_lead_in_and_one_closing(templates):
    for kind in ("opening", "lead_in", "closing"):
        assert sum(1 for t in templates if t["kind"] == kind) == 1, kind


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


def test_every_reason_is_a_registry_name_or_reciprocal_dispute(templates):
    """Full registry names only (D59/D92) — never the scoring letter codes.
    Framing blocks (opening, lead-in, closing) serve every reason and list none."""
    allowed = set(REASON_NAMES) | {RECIPROCAL}
    for t in templates:
        if t["kind"] not in ("opening", "lead_in", "closing"):
            assert t["reasons"], f"{t['id']}: empty reasons"
        bad = [r for r in t["reasons"] if r not in allowed]
        assert not bad, f"{t['id']}: unknown reason value(s) {bad}"


def test_every_entry_has_the_cycle_and_scope(templates):
    for t in templates:
        assert (t["cycle"], t["scope"]) == ("AAAI-27", "phase1_reject"), t["id"]


def test_all_current_entries_are_draft(templates):
    """Nothing is approved yet — Marc has not signed off any wording."""
    assert [t["id"] for t in templates if t["status"] != "draft"] == []
    for t in templates:
        assert (t["approved_by"], t["approved_at"], t["approved_sha256"]) == (None, None, None), t["id"]
