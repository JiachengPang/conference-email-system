"""Tests for the appeal reply template loader (reject-appeal Phase 3, D93).

The rule tests use temp files (pytest tmp_path), never the real template file.
Three tests read the REAL file on purpose: it serves nothing today, the pin of
approved (id, sha256) pairs starts empty, and no other module names the file.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from app.pipeline import appeal_reply_templates as art
from app.pipeline.appeal_reply_templates import (
    compute_body_sha256,
    get_template,
    load_approved_templates,
    templates_for_reason,
)

CYCLE = "AAAI-27"
BODY = "We understand that this outcome may be disappointing.\n\n(1) First point.\n(2) The decision is final."

# ⚠️ Every approval must edit this constant (D93). It lists the (id, sha256)
# pairs of the entries marked "approved" in the REAL file; it is empty because
# Marc has approved nothing yet. Never re-baseline it without an approval record.
APPROVED_PINS: frozenset[tuple[str, str]] = frozenset()


def entry(**overrides) -> dict:
    """An entry that passes every rule unless overridden."""
    body = overrides.pop("body", BODY)
    base = {
        "id": "b_standard", "title": "t", "reasons": ["score_outcome_mismatch"],
        "when_used": "w", "body": body, "status": "approved",
        "approved_by": "Marc", "approved_at": "2026-10-01",
        "approved_sha256": compute_body_sha256(body),
        "cycle": CYCLE, "scope": "phase1_reject", "basis": ["policy_102"], "blocked_on": [],
    }
    base.update(overrides)
    return base


def write(tmp_path: Path, *entries: dict) -> Path:
    p = tmp_path / "templates.json"
    p.write_text(json.dumps({"schema_version": 1, "templates": list(entries)}), encoding="utf-8")
    return p


def ids(templates) -> list[str]:
    return [t.id for t in templates]


# --- compute_body_sha256 ---------------------------------------------------
def test_hash_is_sha256_of_the_utf8_bytes_with_no_normalization():
    import hashlib

    body = "Line one —\nline two  "  # em dash + trailing spaces kept
    assert compute_body_sha256(body) == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert compute_body_sha256(body) != compute_body_sha256(body.strip())


# --- refusal rules -----------------------------------------------------------
def test_a_draft_entry_is_refused(tmp_path):
    assert load_approved_templates(write(tmp_path, entry(status="draft")), cycle=CYCLE) == []


def test_an_approved_entry_with_a_matching_hash_is_returned(tmp_path):
    (t,) = load_approved_templates(write(tmp_path, entry()), cycle=CYCLE)
    assert t.id == "b_standard"
    assert t.body == BODY
    assert t.reasons == ("score_outcome_mismatch",)


def test_an_approved_entry_whose_body_changed_is_refused(tmp_path):
    changed = entry(approved_sha256=compute_body_sha256(BODY), body=BODY + " Edited later.")
    assert load_approved_templates(write(tmp_path, changed), cycle=CYCLE) == []


def test_the_wrong_cycle_is_refused(tmp_path):
    assert load_approved_templates(write(tmp_path, entry(cycle="AAAI-26")), cycle=CYCLE) == []


def test_the_default_cycle_comes_from_settings(tmp_path, monkeypatch):
    from app.core.config import settings

    p = write(tmp_path, entry(cycle="AAAI-28"))
    monkeypatch.setattr(settings, "APPEAL_REPLY_CYCLE", "AAAI-27")
    assert load_approved_templates(p) == []
    monkeypatch.setattr(settings, "APPEAL_REPLY_CYCLE", "AAAI-28")
    assert ids(load_approved_templates(p)) == ["b_standard"]


def test_a_retired_entry_is_refused(tmp_path):
    assert load_approved_templates(write(tmp_path, entry(status="retired")), cycle=CYCLE) == []


@pytest.mark.parametrize("field", ["approved_by", "approved_at", "approved_sha256"])
def test_an_incomplete_approval_record_is_refused(tmp_path, field):
    assert load_approved_templates(write(tmp_path, entry(**{field: None})), cycle=CYCLE) == []


def test_the_wrong_scope_is_refused(tmp_path):
    assert load_approved_templates(write(tmp_path, entry(scope="phase2_reject")), cycle=CYCLE) == []


def test_an_approved_but_blocked_entry_is_refused(tmp_path):
    blocked = entry(blocked_on=["ethics_form_address"])
    assert load_approved_templates(write(tmp_path, blocked), cycle=CYCLE) == []


def test_an_unknown_placeholder_is_refused(tmp_path):
    body = "Report it through the form at [ETHICS FORM ADDRESS]."
    assert load_approved_templates(write(tmp_path, entry(body=body)), cycle=CYCLE) == []


def test_a_chair_placeholder_is_allowed(tmp_path):
    (t,) = load_approved_templates(write(tmp_path, entry(body="[CHAIR: write reply]")), cycle=CYCLE)
    assert t.body == "[CHAIR: write reply]"


def test_a_chair_placeholder_does_not_excuse_another_placeholder(tmp_path):
    body = "[CHAIR: write reply] and [ETHICS FORM ADDRESS]"
    assert load_approved_templates(write(tmp_path, entry(body=body)), cycle=CYCLE) == []


def test_the_allowed_chair_form_is_one_the_drafter_blocks_at_approval():
    """What the loader lets through must be what the approve endpoint 409s on."""
    from app.pipeline.drafter import PLACEHOLDER_RE

    for s in ("[CHAIR: write reply]", "[CHAIR:x]", "[CHAIR:   name the ground ]"):
        assert art._CHAIR_RE.fullmatch(s), s
        assert PLACEHOLDER_RE.fullmatch(s), s


def test_a_duplicate_id_refuses_both_copies(tmp_path):
    p = write(tmp_path, entry(), entry(title="second copy"))
    assert load_approved_templates(p, cycle=CYCLE) == []


def test_one_bad_entry_does_not_hide_the_good_ones(tmp_path):
    p = write(tmp_path, entry(id="bad", status="draft"), entry(id="good"))
    assert ids(load_approved_templates(p, cycle=CYCLE)) == ["good"]


def test_an_entry_missing_fields_is_refused_not_raised(tmp_path):
    p = write(tmp_path, {"id": "half", "status": "approved"}, entry(id="good"))
    assert ids(load_approved_templates(p, cycle=CYCLE)) == ["good"]


# --- never raises ------------------------------------------------------------
def test_malformed_json_returns_empty_and_does_not_raise(tmp_path):
    p = tmp_path / "templates.json"
    p.write_text('{"templates": [ {"id": "x", ', encoding="utf-8")
    assert load_approved_templates(p, cycle=CYCLE) == []


@pytest.mark.parametrize("content", ['{"no_templates_key": []}', '{"templates": "nope"}', "[]"])
def test_a_wrong_top_level_shape_returns_empty(tmp_path, content):
    p = tmp_path / "templates.json"
    p.write_text(content, encoding="utf-8")
    assert load_approved_templates(p, cycle=CYCLE) == []


def test_a_missing_file_returns_empty(tmp_path):
    assert load_approved_templates(tmp_path / "absent.json", cycle=CYCLE) == []


def test_a_refusal_logs_the_id_and_rule_but_never_the_body(tmp_path, caplog):
    secret = "UNIQUE-BODY-TEXT-MUST-NOT-BE-LOGGED"
    p = write(tmp_path, entry(id="leaky", body=secret, approved_sha256="0" * 64))
    with caplog.at_level(logging.WARNING, logger=art.__name__):
        assert load_approved_templates(p, cycle=CYCLE) == []
    assert "leaky" in caplog.text
    assert "body_hash_mismatch" in caplog.text
    assert secret not in caplog.text


# --- templates_for_reason / get_template ---------------------------------------
def test_templates_for_reason_returns_both_b_templates_and_picks_neither(tmp_path):
    p = write(tmp_path, entry(id="b_standard"), entry(id="b_forward"),
              entry(id="c_standard", reasons=["reviewer_misunderstanding"]))
    got = templates_for_reason("score_outcome_mismatch", p, cycle=CYCLE)
    assert ids(got) == ["b_standard", "b_forward"]


def test_templates_for_reason_never_returns_an_unapproved_one(tmp_path):
    p = write(tmp_path, entry(id="b_standard"), entry(id="b_forward", status="draft"))
    assert ids(templates_for_reason("score_outcome_mismatch", p, cycle=CYCLE)) == ["b_standard"]


def test_get_template(tmp_path):
    p = write(tmp_path, entry(id="b_standard"), entry(id="b_forward", status="draft"))
    assert get_template("b_standard", p, cycle=CYCLE).id == "b_standard"
    assert get_template("b_forward", p, cycle=CYCLE) is None
    assert get_template("nope", p, cycle=CYCLE) is None


def test_returned_templates_are_frozen(tmp_path):
    (t,) = load_approved_templates(write(tmp_path, entry()), cycle=CYCLE)
    with pytest.raises(Exception):
        t.body = "changed"  # type: ignore[misc]


# --- the REAL file -----------------------------------------------------------------
def test_the_real_file_serves_nothing_today():
    """All eight templates are draft (D91), so nothing may be served."""
    assert art.DEFAULT_PATH.exists(), art.DEFAULT_PATH
    assert load_approved_templates() == []


def test_the_approved_set_in_the_real_file_equals_the_pin():
    """PIN (D93): any approval in the real file must also edit APPROVED_PINS."""
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    approved = {(t["id"], t["approved_sha256"]) for t in data["templates"] if t.get("status") == "approved"}
    assert approved == APPROVED_PINS


# --- source scan (D81) ----------------------------------------------------------
def test_no_other_app_module_reads_the_template_file():
    """Only the loader may name the file or build its path; everything else must
    go through the loader. (Importing the loader module is fine.)"""
    app_dir = Path(art.__file__).resolve().parents[1]  # backend/app
    loader = Path(art.__file__).resolve()
    pattern = re.compile(r"appeal_reply_templates\.json|[\"']reply_templates[\"']")
    offenders = [
        str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
        if p.resolve() != loader and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"modules reading the template file directly: {offenders}"
