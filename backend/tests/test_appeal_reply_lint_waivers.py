"""Tests for wording-check exceptions: per-block, per-rule, text-bound lint waivers
(reject-appeal Phase 3 Step 3b-1, D107/D109).

A template entry may carry ``lint_waivers`` — a list of ``{rule, approved_by,
note}`` — that tolerates one named wording-check rule for that entry's exact
text, instead of weakening the rule for everyone. Every test writes temp files;
the real template file is never edited (its pin lives in
test_appeal_reply_template_loader.py).

The bodies below are Marc's approved score point and synthetic variants of it.
No ticket text and no PII.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.pipeline import appeal_reply_templates as art
from app.pipeline.appeal_reply_composer import compose_reply
from app.pipeline.appeal_reply_lint import lint_template_body
from app.pipeline.appeal_reply_templates import (
    LintWaiver,
    compute_body_sha256,
    load_approved_templates,
)

CYCLE = "AAAI-27"
ROLES = "internal_roles_or_process"
YEAR = "year_or_time_specific"
SCORE = "score_outcome_mismatch"

# Marc's approved score point (T1 point 1). It trips ROLES on "Senior program
# committee" and nothing else.
SPC_BODY = (
    "Decisions are not based solely on the visible reviewer scores. Senior program "
    "committee members evaluated both the paper and the reviews, including whether the "
    "raised concerns can be addressed with minor clarifications or require substantial "
    "revision."
)
# The same text plus a second, unrelated violation (YEAR).
SPC_AND_YEAR_BODY = SPC_BODY + " This is how it worked this year."
WAIVER = {"rule": ROLES, "approved_by": "Marc", "note": "Approved wording names the committee."}


def entry(**overrides) -> dict:
    """An approved, hash-matching entry with SPC_BODY and a valid ROLES waiver,
    unless overridden. ``approved_sha256`` is computed from the FINAL body unless
    the override sets it explicitly."""
    body = overrides.pop("body", SPC_BODY)
    base = {
        "id": "point_scores", "title": "t", "kind": "point", "order": 1, "optional": False,
        "reasons": [SCORE], "when_used": "w", "body": body, "status": "approved",
        "approved_by": "Marc", "approved_at": "2026-10-02",
        "approved_sha256": compute_body_sha256(body),
        "cycle": CYCLE, "scope": "phase1_reject", "basis": [], "blocked_on": [],
        "lint_waivers": [dict(WAIVER)],
    }
    base.update(overrides)
    return base


def write(tmp_path: Path, *entries: dict) -> Path:
    p = tmp_path / "templates.json"
    p.write_text(json.dumps({"schema_version": 2, "templates": list(entries)}), encoding="utf-8")
    return p


def failing(e: dict) -> list[str]:
    return art._failing_rules(e, CYCLE)


# --- the wording check itself ------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Senior Program Chair",
    "Senior Program Chairs",
    "the senior program chair reviewed it",
    "In addition, we ask Senior Program Chairs to assess the quality of the reviews.",
])
def test_senior_program_chair_fires(text):
    assert ROLES in {name for name, _ in lint_template_body(text)}


def test_senior_program_committee_still_fires_without_a_waiver():
    """Not allowed globally (D109): only a reviewed, text-bound waiver tolerates it."""
    assert {name for name, _ in lint_template_body(SPC_BODY)} == {ROLES}
    assert {name for name, _ in lint_template_body("Senior program committee members")} == {ROLES}


def test_the_body_fixtures_trip_exactly_the_rules_the_tests_rely_on():
    """Guards the fixtures: if a rule changes, these tests must not pass vacuously."""
    assert {name for name, _ in lint_template_body(SPC_AND_YEAR_BODY)} == {ROLES, YEAR}


# --- loader: a valid waiver ----------------------------------------------------------------
def test_without_a_waiver_the_entry_is_refused_by_the_lint(tmp_path):
    e = entry()
    del e["lint_waivers"]
    assert failing(e) == [f"lint:{ROLES}"]
    assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []


def test_a_valid_waiver_serves_the_entry_and_carries_the_waiver(tmp_path):
    e = entry()
    assert failing(e) == []
    (t,) = load_approved_templates(write(tmp_path, e), cycle=CYCLE)
    assert t.id == "point_scores" and t.body == SPC_BODY
    assert t.lint_waivers == (LintWaiver(rule=ROLES, approved_by="Marc", note=WAIVER["note"]),)


def test_an_entry_without_waivers_loads_with_an_empty_tuple(tmp_path):
    e = entry(body="Decisions are not based on any single review.")
    del e["lint_waivers"]
    (t,) = load_approved_templates(write(tmp_path, e), cycle=CYCLE)
    assert t.lint_waivers == ()


def test_an_empty_waiver_list_is_allowed_and_waives_nothing(tmp_path):
    assert failing(entry(lint_waivers=[])) == [f"lint:{ROLES}"]


def test_a_waiver_for_one_rule_does_not_waive_another():
    e = entry(body=SPC_AND_YEAR_BODY)
    assert failing(e) == [f"lint:{YEAR}"]


def test_a_waiver_for_a_rule_that_does_not_fire_changes_nothing(tmp_path):
    e = entry(lint_waivers=[{"rule": YEAR, "approved_by": "Marc", "note": "n"}])
    assert failing(e) == [f"lint:{ROLES}"]


# --- loader: approver and note --------------------------------------------------------------
@pytest.mark.parametrize("field", ["approved_by", "note"])
@pytest.mark.parametrize("value", ["", "   ", None, 5, ["Marc"]])
def test_an_empty_or_non_string_approver_or_note_refuses_the_entry(tmp_path, field, value):
    e = entry(lint_waivers=[{**WAIVER, field: value}])
    assert failing(e) == ["bad_lint_waivers", f"lint:{ROLES}"]
    assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []


@pytest.mark.parametrize("field", ["approved_by", "note", "rule"])
def test_a_missing_waiver_key_refuses_the_entry(tmp_path, field):
    waiver = dict(WAIVER)
    del waiver[field]
    e = entry(lint_waivers=[waiver])
    assert failing(e) == ["bad_lint_waivers", f"lint:{ROLES}"]
    assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []


# --- loader: the waiver is bound to the approved text --------------------------------------
def test_a_changed_body_voids_the_approval_and_its_waiver(tmp_path):
    e = entry(approved_sha256=compute_body_sha256(SPC_BODY),
              body=SPC_BODY.replace("revision.", "revisions."))
    rules = failing(e)
    assert "body_hash_mismatch" in rules
    assert f"lint:{ROLES}" in rules, "the waiver must have no effect on a mismatched body"
    assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []


@pytest.mark.parametrize("overrides, expected_rule", [
    ({"status": "draft"}, "not_approved"),
    ({"status": "retired"}, "not_approved"),
    ({"approved_by": None}, "approval_record_incomplete"),
    ({"approved_at": None}, "approval_record_incomplete"),
    ({"approved_sha256": None}, "approval_record_incomplete"),
    ({"cycle": "AAAI-26"}, "wrong_cycle"),
    ({"scope": "elsewhere"}, "wrong_scope"),
    ({"blocked_on": ["something"]}, "blocked"),
])
def test_a_waiver_on_an_entry_that_is_not_otherwise_approved_has_no_effect(
    tmp_path, overrides, expected_rule
):
    e = entry(**overrides)
    rules = failing(e)
    assert expected_rule in rules
    assert f"lint:{ROLES}" in rules, "the waiver must have no effect"
    assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []


# --- loader: unknown rule names -------------------------------------------------------------
def test_an_unknown_rule_name_refuses_the_whole_entry(tmp_path, caplog):
    e = entry(lint_waivers=[dict(WAIVER), {"rule": "made_up_rule", "approved_by": "Marc", "note": "n"}])
    assert failing(e) == ["waiver_unknown_rule:made_up_rule", f"lint:{ROLES}"]
    with caplog.at_level(logging.WARNING):
        assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []
    assert "'point_scores'" in caplog.text and "waiver_unknown_rule:made_up_rule" in caplog.text
    assert "Senior program committee" not in caplog.text, "body text must never be logged"
    assert WAIVER["note"] not in caplog.text


def test_an_unknown_rule_name_alone_refuses_a_clean_entry():
    e = entry(body="Decisions are not based on any single review.",
              lint_waivers=[{"rule": "made_up_rule", "approved_by": "Marc", "note": "n"}])
    assert failing(e) == ["waiver_unknown_rule:made_up_rule"]


def test_a_long_unknown_rule_name_is_logged_truncated(tmp_path):
    e = entry(lint_waivers=[{"rule": "x" * 500, "approved_by": "Marc", "note": "n"}])
    (rule, *_rest) = failing(e)
    assert rule == "waiver_unknown_rule:" + "x" * 64


# --- loader: malformed waivers --------------------------------------------------------------
@pytest.mark.parametrize("value", [
    "internal_roles_or_process",
    {"rule": ROLES, "approved_by": "Marc", "note": "n"},
    None,
    5,
    ["internal_roles_or_process"],
    [None],
    [[ROLES, "Marc", "n"]],
    [{**WAIVER, "rule": 5}],
    [{**WAIVER, "extra": "x"}],
    [dict(WAIVER), dict(WAIVER)],
], ids=["string", "dict", "null", "int", "list-of-string", "list-of-null", "list-of-list",
        "non-string-rule", "extra-key", "duplicate-rule"])
def test_a_malformed_waiver_value_refuses_the_entry_without_raising(tmp_path, value):
    e = entry(lint_waivers=value)
    assert failing(e) == ["bad_lint_waivers", f"lint:{ROLES}"]
    assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []


def test_a_bad_waiver_entry_does_not_hide_a_good_entry(tmp_path):
    good = entry(id="good_one", body="Decisions are not based on any single review.")
    del good["lint_waivers"]
    bad = entry(id="bad_one", lint_waivers="junk")
    assert [t.id for t in load_approved_templates(write(tmp_path, bad, good), cycle=CYCLE)] == ["good_one"]


def test_loaded_waivers_are_frozen(tmp_path):
    (t,) = load_approved_templates(write(tmp_path, entry()), cycle=CYCLE)
    assert isinstance(t.lint_waivers, tuple)
    with pytest.raises(Exception):
        t.lint_waivers[0].rule = "other"  # type: ignore[misc]


# --- composer -----------------------------------------------------------------------------
def approved_copy(tmp_path: Path, overrides: dict) -> Path:
    """A temp copy of the real file with every entry approved and correctly hashed,
    after applying ``overrides`` (id -> fields). Mirrors the composer tests' helper."""
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    for e in data["templates"]:
        e.update(overrides.get(e["id"], {}))
        if e["id"] == "point_report_form":  # its form address is still missing
            continue
        if e["id"] == "full_reciprocal":
            e["blocked_on"] = []
        e.update(status="approved", approved_by="test", approved_at="2026-10-02",
                 approved_sha256=compute_body_sha256(e["body"]))
    p = tmp_path / "templates.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_the_composer_honors_the_waiver_of_a_block_it_used(tmp_path):
    p = approved_copy(tmp_path, {"point_scores": {"body": SPC_BODY, "lint_waivers": [dict(WAIVER)]}})
    r = compose_reply([SCORE], path=p)
    assert r.refusal is None, r.refusal
    assert r.mode == "merged" and "point_scores" in r.used_ids
    # The review-process point is (1); the score point follows it as (2).
    assert f"(2) {SPC_BODY}" in r.body
    # The finished email really does contain the waived wording.
    assert {name for name, _ in lint_template_body(r.body)} == {ROLES}


def test_the_composer_refuses_when_the_used_block_has_no_waiver(tmp_path):
    # The real point_scores carries a waiver since Step 3b-2, so remove it explicitly.
    p = approved_copy(tmp_path, {"point_scores": {"body": SPC_BODY, "lint_waivers": []}})
    r = compose_reply([SCORE], path=p)
    assert (r.mode, r.body, r.refusal) == ("refused", None, "missing_approved_block:point_scores")


@pytest.mark.parametrize("bad_waiver", [
    {**WAIVER, "note": ""},
    {**WAIVER, "approved_by": ""},
    {**WAIVER, "rule": "made_up_rule"},
    {**WAIVER, "rule": YEAR},
], ids=["empty-note", "empty-approver", "unknown-rule", "wrong-rule"])
def test_the_composer_refuses_when_the_used_blocks_waiver_is_invalid(tmp_path, bad_waiver):
    p = approved_copy(tmp_path, {"point_scores": {"body": SPC_BODY, "lint_waivers": [bad_waiver]}})
    r = compose_reply([SCORE], path=p)
    assert (r.mode, r.body, r.refusal) == ("refused", None, "missing_approved_block:point_scores")


def test_the_composer_ignores_waivers_on_blocks_it_did_not_use(tmp_path, caplog):
    """The JOIN of the opening and the lead-in forms "this year" (each block is
    clean alone). A YEAR waiver on point_ai_review — a block a score reply does
    not use — must not excuse it."""
    p = approved_copy(tmp_path, {
        "opening_warm": {"body": "We understand this outcome, and we reply this"},
        "lead_in_concerns": {"body": "year as follows:"},
        "point_ai_review": {"lint_waivers": [dict(WAIVER),
                                             {"rule": YEAR, "approved_by": "Marc", "note": "n"}]},
    })
    assert "point_ai_review" in {t.id for t in load_approved_templates(p)}, "the unused block is served"
    with caplog.at_level(logging.WARNING):
        r = compose_reply([SCORE], path=p)
    assert (r.mode, r.body, r.refusal) == ("refused", None, f"lint:{YEAR}")
    assert "this year" not in caplog.text


def test_a_used_blocks_waiver_does_not_excuse_a_different_rule_formed_at_a_join(tmp_path):
    """point_scores waives ROLES; the join of the opening and lead-in forms a YEAR
    violation. The waiver names ROLES only, so the reply is refused on YEAR."""
    p = approved_copy(tmp_path, {
        "opening_warm": {"body": "We understand this outcome, and we reply this"},
        "lead_in_concerns": {"body": "year as follows:"},
        "point_scores": {"body": SPC_BODY, "lint_waivers": [dict(WAIVER)]},
    })
    r = compose_reply([SCORE], path=p)
    assert (r.mode, r.body, r.refusal) == ("refused", None, f"lint:{YEAR}")


def test_the_composer_honors_a_used_blocks_waiver_for_a_rule_formed_at_a_join(tmp_path):
    """The union is over USED blocks: a YEAR waiver on the lead-in (used) does
    tolerate a YEAR violation in the finished email. This is the documented
    boundary of a per-block union, pinned so it cannot change silently."""
    p = approved_copy(tmp_path, {
        "opening_warm": {"body": "We understand this outcome, and we reply this"},
        "lead_in_concerns": {"body": "year as follows:",
                             "lint_waivers": [{"rule": YEAR, "approved_by": "Marc", "note": "n"}]},
    })
    r = compose_reply([SCORE], path=p)
    assert r.refusal is None and r.mode == "merged"
