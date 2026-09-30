"""Structural checks on data/reply_templates/appeal_reply_templates.json (Phase 3, D91).

STRUCTURE ONLY — no wording rules yet (the D85 safety lint and the D81 loader
come later). Nothing reads this file at runtime; these tests keep its shape
honest so the future loader can rely on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.appeal_reasons import REASON_NAMES

# Not an appeal reason: the reciprocal template serves the `is_reciprocal_dispute`
# flag, which the registry deliberately excludes (D60).
RECIPROCAL = "reciprocal_dispute"

_TEMPLATES = (
    Path(__file__).resolve().parents[2] / "data" / "reply_templates" / "appeal_reply_templates.json"
)

REQUIRED_FIELDS = {
    "id", "title", "reasons", "when_used", "body", "status", "approved_by",
    "approved_at", "approved_sha256", "cycle", "scope", "basis", "blocked_on",
}
STATUSES = {"draft", "approved", "retired"}


@pytest.fixture(scope="module")
def templates() -> list[dict]:
    data = json.loads(_TEMPLATES.read_text(encoding="utf-8"))
    return data["templates"]


def test_the_file_parses_and_holds_exactly_eight_unique_ids(templates):
    ids = [t["id"] for t in templates]
    assert len(ids) == 8
    assert len(set(ids)) == 8, "template ids must be unique"


def test_every_entry_has_all_required_fields(templates):
    for t in templates:
        missing = REQUIRED_FIELDS - set(t)
        assert not missing, f"{t.get('id')}: missing {sorted(missing)}"


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
    """Full registry names only (D59/D92) — never the scoring letter codes."""
    allowed = set(REASON_NAMES) | {RECIPROCAL}
    for t in templates:
        assert t["reasons"], f"{t['id']}: empty reasons"
        bad = [r for r in t["reasons"] if r not in allowed]
        assert not bad, f"{t['id']}: unknown reason value(s) {bad}"


def test_all_current_entries_are_draft(templates):
    """Nothing is approved yet — Marc has not signed off any wording."""
    assert [t["id"] for t in templates if t["status"] != "draft"] == []
