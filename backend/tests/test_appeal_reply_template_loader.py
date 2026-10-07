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
    templates_of_kind,
)

CYCLE = "AAAI-27"
BODY = "We understand that this outcome may be disappointing.\n\n(1) First point.\n(2) The decision is final."

# ⚠️ Every approval must edit this constant (D93). It lists the (id, sha256)
# pairs of the entries marked "approved" in the REAL file. Never re-baseline it
# without an approval record. Recorded by scripts/approve_appeal_reply_blocks.py:
# on 2026-10-02 the opening, lead-in and reciprocal reply (Marc Pujol-Gonzalez)
# and the chair-writes line (Sahil Satasiya); on 2026-10-05 the ethics-form point
# (Marc Pujol-Gonzalez); on 2026-10-06 the seven blocks the program chairs reworded
# or added (Jiacheng Pang). The ethics-form point's text did not change, so its
# 2026-10-05 approval still holds.
APPROVED_PINS: frozenset[tuple[str, str]] = frozenset({
    ("opening_warm", "539d321e8a99d158a9e950413b2830fa0ff9af588ac02e7a28f29baad104ff48"),
    ("lead_in_concerns", "ca08da45a100b7feb534760966895a417a7b5d5114212f3db31a49abefd3ed07"),
    ("full_reciprocal", "ec881ab040b76d457efdbbcbfe9a62765f172072ba1f9dfeb8b2675f488d1788"),
    ("line_chair_writes", "b13e731e4f37105818e9b7c888bc3ccb2c97e1b3a3fd95dbdc88516dc403c698"),
    ("point_ethics_form", "aa88ba283d71b5929a31a50d018705f2090db554bbdf209f6730bd69ca5ff0c0"),
    # The program chairs' rewording (2026-10-06).
    ("point_review_process", "e06a0d424b5f36fba6095e7e69e10d43386385aafc35d921fd99aa64f4e98ebd"),
    ("point_scores", "720e6a3c3c1195c18bec946e435ea0a3c62e1729f98f1f67559bb7b0065502c3"),
    ("point_rebuttal", "38598a26d59a6f06535f8d79b3610c8a34a3e8f214e336a9beeb595c24b7a557"),
    ("point_reviewer_tracking", "04d9b68604cba39c2911ed4c04284de647b1d7b1098be54913e7cc279ab00dce"),
    ("closing_reviewed", "3cb72e53cb6bd5fbf66c6b82fadee41efb02421530ed1fa94bd2f1838a3028f9"),
    ("standalone_ai_review", "f08fd646f89f697805416626fc8146fb456db2390b905e94beb9dd42a4a1036f"),
    ("standalone_general_stage1", "594e67ba852f185b35af0fa535bfd867fd23f1331eb3ea1012191192b9f3c59b"),
})

# ⚠️ Every lint waiver must edit this constant too (D107/D109). It lists, for each
# entry in the REAL file that carries a non-empty `lint_waivers`, the triple
# (id, sha256 of its body as stored, sorted waived rule names) — whatever the
# entry's status, so even a waiver on a draft cannot be added unnoticed. Never
# re-baseline it without a reviewed exception. Today: the five approved blocks of
# the chairs' rewording that name SPCs / ACs / the Area Chair (in effect), and the
# retired misconduct point 1, which keeps Marc's waiver for its unchanged text
# (no effect while retired). The ethics-form point has NO waiver: its link passes
# through the wording check's URL allow-list.
WAIVER_PINS: frozenset[tuple[str, str, tuple[str, ...]]] = frozenset({
    ("point_review_process",
     "e06a0d424b5f36fba6095e7e69e10d43386385aafc35d921fd99aa64f4e98ebd",
     ("internal_roles_or_process",)),
    ("point_scores",
     "720e6a3c3c1195c18bec946e435ea0a3c62e1729f98f1f67559bb7b0065502c3",
     ("internal_roles_or_process",)),
    ("point_reviewer_tracking",
     "04d9b68604cba39c2911ed4c04284de647b1d7b1098be54913e7cc279ab00dce",
     ("internal_roles_or_process",)),
    ("standalone_general_stage1",
     "594e67ba852f185b35af0fa535bfd867fd23f1331eb3ea1012191192b9f3c59b",
     ("internal_roles_or_process",)),
    ("standalone_ai_review",
     "f08fd646f89f697805416626fc8146fb456db2390b905e94beb9dd42a4a1036f",
     ("internal_roles_or_process",)),
    ("point_spc_evaluation",
     "258bcd58b5c310ea6968207bd01a6074294b9234cff171211e1d5673dd6a5068",
     ("internal_roles_or_process",)),
})


def entry(**overrides) -> dict:
    """An entry that passes every rule unless overridden."""
    body = overrides.pop("body", BODY)
    base = {
        "id": "b_standard", "title": "t", "kind": "holding", "order": None, "optional": False,
        "reasons": ["score_outcome_mismatch"],
        "when_used": "w", "body": body, "status": "approved",
        "approved_by": "Marc", "approved_at": "2026-10-01",
        "approved_sha256": compute_body_sha256(body),
        "cycle": CYCLE, "scope": "phase1_reject", "basis": ["policy_102"], "blocked_on": [],
    }
    base.update(overrides)
    return base


def write(tmp_path: Path, *entries: dict, schema_version: int = 2) -> Path:
    p = tmp_path / "templates.json"
    p.write_text(json.dumps({"schema_version": schema_version, "templates": list(entries)}),
                 encoding="utf-8")
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


# --- schema 2: kind / order / optional (D95) -------------------------------------
@pytest.mark.parametrize("version", [1, 3, None])
def test_any_schema_version_other_than_2_serves_nothing(tmp_path, version):
    assert load_approved_templates(write(tmp_path, entry(), schema_version=version), cycle=CYCLE) == []


def test_an_unknown_kind_is_refused(tmp_path):
    assert load_approved_templates(write(tmp_path, entry(kind="paragraph")), cycle=CYCLE) == []


@pytest.mark.parametrize("order", [None, "1", 1.5, True])
def test_a_point_without_an_integer_order_is_refused(tmp_path, order):
    p = write(tmp_path, entry(kind="point", order=order))
    assert load_approved_templates(p, cycle=CYCLE) == []


def test_a_non_point_with_an_order_is_refused(tmp_path):
    assert load_approved_templates(write(tmp_path, entry(kind="holding", order=1)), cycle=CYCLE) == []


@pytest.mark.parametrize("optional", [None, "false", 0])
def test_a_non_bool_optional_is_refused(tmp_path, optional):
    assert load_approved_templates(write(tmp_path, entry(optional=optional)), cycle=CYCLE) == []


@pytest.mark.parametrize("field", ["kind", "order", "optional"])
def test_a_v2_entry_missing_a_new_field_is_refused(tmp_path, field):
    e = entry()
    del e[field]
    assert load_approved_templates(write(tmp_path, e), cycle=CYCLE) == []


def test_approved_blocks_carry_kind_order_and_optional(tmp_path):
    (t,) = load_approved_templates(write(tmp_path, entry(kind="point", order=4, optional=True)), cycle=CYCLE)
    assert (t.kind, t.order, t.optional) == ("point", 4, True)


def test_templates_of_kind_sorts_points_by_order_then_id(tmp_path):
    p = write(tmp_path,
              entry(id="p_z", kind="point", order=2), entry(id="p_b", kind="point", order=1),
              entry(id="p_a", kind="point", order=2), entry(id="h_1", kind="holding"))
    assert ids(templates_of_kind("point", p, cycle=CYCLE)) == ["p_b", "p_a", "p_z"]
    assert ids(templates_of_kind("holding", p, cycle=CYCLE)) == ["h_1"]
    assert templates_of_kind("closing", p, cycle=CYCLE) == []


def test_templates_of_kind_sorts_unordered_kinds_by_id(tmp_path):
    p = write(tmp_path, entry(id="h_b", kind="holding"), entry(id="h_a", kind="holding"))
    assert ids(templates_of_kind("holding", p, cycle=CYCLE)) == ["h_a", "h_b"]


def test_templates_for_reason_returns_blocks_of_every_kind_and_picks_none(tmp_path):
    p = write(tmp_path,
              entry(id="point_scores", kind="point", order=1),
              entry(id="holding_score_mismatch", kind="holding"),
              entry(id="point_other", kind="point", order=2, reasons=["reviewer_misunderstanding"]))
    got = templates_for_reason("score_outcome_mismatch", p, cycle=CYCLE)
    assert ids(got) == ["point_scores", "holding_score_mismatch"]


def test_the_real_file_is_schema_2_and_every_entry_would_pass_the_shape_rules():
    """Guards the real file against the loader's shape rules; nothing is approved,
    so the only rules any entry may fail are the approval/blocked ones."""
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == art.SCHEMA_VERSION
    for e in data["templates"]:
        allowed = {"not_approved", "approval_record_incomplete", "blocked"}
        if e["blocked_on"]:
            # A blocked entry may still hold its pending placeholder (e.g. the
            # ethics form address); the loader refuses it until it is filled.
            allowed.add("unknown_placeholder")
        # A waiver has no effect until the entry is approved (D107/D109), so on a
        # draft the rule it names still shows. Only that entry's OWN waived rules
        # are tolerated here; any other lint rule still fails this test.
        allowed |= {f"lint:{w['rule']}" for w in e.get("lint_waivers", [])}
        assert set(art._failing_rules(e, CYCLE)) <= allowed, (e["id"], art._failing_rules(e, CYCLE))


def test_returned_templates_are_frozen(tmp_path):
    (t,) = load_approved_templates(write(tmp_path, entry()), cycle=CYCLE)
    with pytest.raises(Exception):
        t.body = "changed"  # type: ignore[misc]


# --- the REAL file -----------------------------------------------------------------
def test_the_real_file_serves_exactly_the_approved_blocks():
    """The real file serves exactly the twelve approved blocks, each with its
    recorded approver and date, and nothing else: no retired block. The seven
    blocks the chairs reworded or added were approved by Jiacheng Pang on
    2026-10-06; Marc and Yan have not approved the new text."""
    assert art.DEFAULT_PATH.exists(), art.DEFAULT_PATH
    served = {t.id: (t.approved_by, t.approved_at, t.approved_sha256) for t in load_approved_templates()}
    assert len(served) == 12
    assert {(i, sha) for i, (_, _, sha) in served.items()} == APPROVED_PINS
    reworded = {"point_review_process", "point_scores", "point_rebuttal", "point_reviewer_tracking",
                "closing_reviewed", "standalone_ai_review", "standalone_general_stage1"}
    assert {i for i, (by, _, _) in served.items() if by == "Jiacheng Pang"} == reworded
    assert {i for i, (by, _, _) in served.items() if by == "Marc Pujol-Gonzalez"} == {
        "opening_warm", "lead_in_concerns", "full_reciprocal", "point_ethics_form"}
    assert {i for i, (by, _, _) in served.items() if by == "Prof. Yan"} == set()
    assert {i for i, (by, _, _) in served.items() if by == "Sahil Satasiya"} == {"line_chair_writes"}
    assert {i for i, (_, at, _) in served.items() if at == "2026-10-06"} == reworded
    assert {i for i, (_, at, _) in served.items() if at == "2026-10-05"} == {"point_ethics_form"}
    assert {at for _, at, _ in served.values()} == {"2026-10-02", "2026-10-05", "2026-10-06"}


def test_one_changed_character_in_an_approved_body_makes_the_loader_refuse_it(tmp_path, caplog):
    """The approval is bound to the exact text: edit one character of an approved
    body and the block is no longer served (body_hash_mismatch)."""
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    for e in data["templates"]:
        if e["id"] == "point_rebuttal":
            assert e["status"] == "approved"
            e["body"] = e["body"].replace("quicker", "quickest", 1)
    p = tmp_path / "templates.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        served = {t.id for t in load_approved_templates(p)}
    assert "point_rebuttal" not in served
    assert "'point_rebuttal' refused: body_hash_mismatch" in caplog.text
    assert {i for i, _ in APPROVED_PINS} - served == {"point_rebuttal"}


def test_the_approved_set_in_the_real_file_equals_the_pin():
    """PIN (D93): any approval in the real file must also edit APPROVED_PINS.

    Extended for lint waivers (D107/D109): any waiver in the real file must also
    edit WAIVER_PINS, bound to the exact body text it was reviewed for.
    """
    data = json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))
    approved = {(t["id"], t["approved_sha256"]) for t in data["templates"] if t.get("status") == "approved"}
    assert approved == APPROVED_PINS

    waived = set()
    for t in data["templates"]:
        if "lint_waivers" not in t:
            continue
        assert isinstance(t["lint_waivers"], list), f"{t['id']}: lint_waivers is not a list"
        if t["lint_waivers"]:
            rules = tuple(sorted(w["rule"] for w in t["lint_waivers"]))
            waived.add((t["id"], compute_body_sha256(t["body"]), rules))
    assert waived == WAIVER_PINS


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
