"""Tests for the appeal reply wording check (reject-appeal Phase 3, D94).

The regression strings are chair-written phrases from real replies (no PII),
each of which the check must catch. The real template file is read to prove
every current body is clean.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.pipeline import appeal_reply_templates as art
from app.pipeline.appeal_reply_lint import RULES, lint_template_body
from app.pipeline.appeal_reply_templates import compute_body_sha256, load_approved_templates


def fired(text: str, blocked_on=()) -> set[str]:
    return {name for name, _ in lint_template_body(text, blocked_on)}


# --- the real file ------------------------------------------------------------
def _real_templates() -> list[dict]:
    return json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))["templates"]


@pytest.mark.parametrize("entry", _real_templates(), ids=lambda e: e["id"])
def test_every_current_template_body_is_clean(entry):
    assert lint_template_body(entry["body"], tuple(entry["blocked_on"])) == []


# --- one test per rule category -----------------------------------------------------
@pytest.mark.parametrize("text", [
    "see policy_104 for details", "as policy 104 states", "your ticket #21567",
    "paper 12345 was rejected", "submission number 4821", "reviewer 3 wrote",
    "the rule (2025) applies", "[source: call_for_papers]",
])
def test_identifiers(text):
    assert "identifiers" in fired(text)


@pytest.mark.parametrize("text", [
    "with a record number of submissions this year",   # real reply (regression)
    "the message sent on 12 September",                 # real reply (regression)
    "as last year", "maybe next year", "since 2025", "under AAAI26 rules",
    "under AAAI-27 rules", "on Sept 24",
])
def test_year_or_time_specific(text):
    assert "year_or_time_specific" in fired(text)


@pytest.mark.parametrize("text", [
    "Senior Program Committee (SPC) member and an Area Chair (AC)",  # real reply (regression)
    "the SPC decided", "your AC agreed", "to detect collusion",
    "flag unprofessional reviews",
])
def test_internal_roles_or_process(text):
    assert "internal_roles_or_process" in fired(text)


def test_the_role_tokens_are_case_sensitive():
    """'ac' / 'spc' inside ordinary words or lower case must not fire."""
    assert "internal_roles_or_process" not in fired("We act on each account; the spc of it.")


@pytest.mark.parametrize("text", [
    "Your paper should not have been rejected.",         # real reply (regression)
    "We apologize for this.", "we apologise", "Sorry for the delay.",
    "that was our mistake", "this was our error", "we will reinstate it",
    "your paper is reinstated", "you are right", "we agree with you",
])
def test_concession(text):
    assert "concession" in fired(text)


@pytest.mark.parametrize("text", [
    "we will both track and take action in the future against such reviewers",  # real reply
    "we will reconsider", "it will be reconsidered", "we will change the policy",
    "we will reinstate your paper",
])
def test_promise(text):
    assert "promise" in fired(text)


@pytest.mark.parametrize("text", [
    "posting it as a comment so you cannot see it",     # real reply (regression)
    "the SPC asked them to extend their review", "it was posted as a comment",
    "the deliberation notes", "the meta-review says", "the metareview",
])
def test_disclosure(text):
    assert "disclosure" in fired(text)


@pytest.mark.parametrize("text", [
    "Dear Author,", "Best regards,", "AAAI Team", "[Author name]", "[Sender name]",
    "see https://example.org/form", "visit www.example.org", "write to chairs@example.org",
    "[SOME PLACEHOLDER]",
])
def test_structure(text):
    assert "structure" in fired(text)


def test_every_rule_category_has_a_test():
    tested = {"identifiers", "year_or_time_specific", "internal_roles_or_process", "concession",
              "promise", "disclosure", "structure"}
    assert set(RULES) == tested


# --- deliberate non-flags --------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "We will investigate and follow up with you.",
    "We will consider your input when studying possible changes for future editions.",
    "senior members of the program committee",
    "another leading venue or future AAAI edition.",
    "(1) First point.\n(2) Second point.\n(9) Ninth point.",
    "papers not advanced to Phase 2 do not have an author response period",
    "[CHAIR: write reply]",
])
def test_approved_wording_does_not_trip(text):
    assert lint_template_body(text) == []


def test_the_allowed_point_markers_are_only_at_a_line_start():
    assert "identifiers" in fired("as noted in (1) above")


# --- blocked_on tolerance --------------------------------------------------------------------
def test_the_ethics_form_placeholder_depends_on_blocked_on():
    body = "you can report it through the form at [ETHICS FORM ADDRESS]."
    assert "structure" in fired(body, ())
    assert lint_template_body(body, ("ethics_form_address",)) == []


def test_blocked_on_tolerates_only_its_own_placeholder():
    body = "see [ETHICS FORM ADDRESS] and [OTHER THING]"
    assert fired(body, ("ethics_form_address",)) == {"structure"}


# --- the loader enforces the lint --------------------------------------------------------------
def _entry(body: str) -> dict:
    return {
        "id": "b_standard", "title": "t", "kind": "holding", "order": None, "optional": False,
        "reasons": ["score_outcome_mismatch"], "when_used": "w",
        "body": body, "status": "approved", "approved_by": "Marc", "approved_at": "2026-10-01",
        "approved_sha256": compute_body_sha256(body), "cycle": "AAAI-27", "scope": "phase1_reject",
        "basis": [], "blocked_on": [],
    }


def _write(tmp_path, entry: dict):
    p = tmp_path / "templates.json"
    p.write_text(json.dumps({"schema_version": 2, "templates": [entry]}), encoding="utf-8")
    return p


def test_the_loader_refuses_an_approved_hash_matching_body_that_fails_the_lint(tmp_path, caplog):
    body = "With a record number of submissions this year, the decision is final."
    with caplog.at_level(logging.WARNING, logger=art.__name__):
        assert load_approved_templates(_write(tmp_path, _entry(body)), cycle="AAAI-27") == []
    assert "lint:year_or_time_specific" in caplog.text
    assert "record number" not in caplog.text, "matched text must never be logged"
    assert body not in caplog.text


def test_the_loader_still_serves_a_clean_body(tmp_path):
    body = "We understand that this outcome may be disappointing.\n\n(1) The decision is final."
    (t,) = load_approved_templates(_write(tmp_path, _entry(body)), cycle="AAAI-27")
    assert t.body == body
