"""Tests for scripts/approve_appeal_reply_blocks.py (reject-appeal Phase 3, Step 3c).

Every test works on files under tmp_path — synthetic ones, or a copy of the real
template file. The real file is never written. No ticket text, no PII: the
bodies are synthetic or the approved block wording.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline import appeal_reply_templates as art
from app.pipeline.appeal_reply_templates import compute_body_sha256, load_approved_templates
from scripts import approve_appeal_reply_blocks as aarb
from scripts.approve_appeal_reply_blocks import Refused, Stamp, approve

CYCLE = "AAAI-27"
DATE = "2026-10-02"
CLEAN = "Decisions are not based on any single review; all assessments are weighed together."
SPC_BODY = (
    "Decisions are not based solely on the visible reviewer scores. Senior program committee "
    "members evaluated both the paper and the reviews."
)
WAIVER = {"rule": "internal_roles_or_process", "approved_by": "Marc", "note": "names the committee"}


def _entry(block_id: str, body: str = CLEAN, **overrides) -> dict:
    e = {
        "id": block_id, "title": "t", "kind": "point", "order": None, "optional": False,
        "reasons": ["reviewer_misunderstanding"], "when_used": "w", "body": body,
        "status": "draft", "approved_by": None, "approved_at": None, "approved_sha256": None,
        "cycle": CYCLE, "scope": "phase1_reject", "basis": [], "blocked_on": [],
    }
    e.update(overrides)
    return e


def _entries() -> list[dict]:
    order = iter(range(1, 100))
    return [
        _entry("clean", order=next(order)),
        _entry("other_clean", order=next(order)),
        _entry("blocked", order=next(order), blocked_on=["waiting"]),
        _entry("retired", order=next(order), status="retired"),
        _entry("already", order=next(order), status="approved", approved_by="Someone",
               approved_at="2026-09-30", approved_sha256=compute_body_sha256(CLEAN)),
        _entry("this_year", "This is how it worked this year.", order=next(order)),
        _entry("spc_waived", SPC_BODY, order=next(order), lint_waivers=[dict(WAIVER)]),
        _entry("spc_unwaived", SPC_BODY, order=next(order)),
        _entry("placeholder", "Please use the form at [FORM ADDRESS].", order=next(order)),
        _entry("wrong_cycle", order=next(order), cycle="AAAI-26"),
        _entry("unordered_point", order=None),
    ]


def write(tmp_path: Path, entries: list[dict] | None = None, *, crlf: bool = False) -> Path:
    text = json.dumps({"schema_version": 2, "templates": entries or _entries()}, indent=2) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    p = tmp_path / "templates.json"
    p.write_bytes(text.encode("utf-8"))
    return p


def by_id(path: Path) -> dict[str, dict]:
    return {e["id"]: e for e in json.loads(path.read_text(encoding="utf-8"))["templates"]}


# --- dry run and apply ---------------------------------------------------------------------
def test_a_dry_run_writes_nothing_and_reports_the_hash(tmp_path):
    p = write(tmp_path)
    before = p.read_bytes()
    stamps = approve(p, ["clean"], "Marc", DATE, apply=False, cycle=CYCLE)
    assert stamps == [Stamp("clean", compute_body_sha256(CLEAN))]
    assert p.read_bytes() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["templates.json"], "no temp files left"


def test_apply_stamps_the_approval_with_the_loaders_own_hash(tmp_path):
    p = write(tmp_path)
    stamps = approve(p, ["clean"], "Marc Pujol-Gonzalez", DATE, apply=True, cycle=CYCLE)
    e = by_id(p)["clean"]
    assert (e["status"], e["approved_by"], e["approved_at"]) == ("approved", "Marc Pujol-Gonzalez", DATE)
    assert e["approved_sha256"] == compute_body_sha256(e["body"]) == stamps[0].sha256
    assert "clean" in {t.id for t in load_approved_templates(p, cycle=CYCLE)}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["templates.json"], "no temp files left"


def test_apply_changes_only_the_four_approval_values_of_the_named_block(tmp_path):
    p = write(tmp_path)
    before_text = p.read_text(encoding="utf-8")
    before = by_id(p)
    approve(p, ["clean"], "Marc", DATE, apply=True, cycle=CYCLE)
    after = by_id(p)
    # Every other entry is untouched, and the named one differs only in four fields.
    assert {k: v for k, v in after.items() if k != "clean"} == {k: v for k, v in before.items() if k != "clean"}
    changed = {f for f in after["clean"] if after["clean"][f] != before["clean"][f]}
    assert changed == {"status", "approved_by", "approved_at", "approved_sha256"}
    assert after["clean"]["body"] == before["clean"]["body"]
    # Formatting preserved: same line count, and exactly four lines differ.
    old_lines, new_lines = before_text.split("\n"), p.read_text(encoding="utf-8").split("\n")
    assert len(old_lines) == len(new_lines)
    diff = [(o, n) for o, n in zip(old_lines, new_lines) if o != n]
    assert len(diff) == 4
    assert all(o.split(":")[0] == n.split(":")[0] for o, n in diff), "same keys, same indentation"


def test_crlf_line_endings_are_preserved(tmp_path):
    p = write(tmp_path, crlf=True)
    before = p.read_bytes()
    approve(p, ["clean", "other_clean"], "Marc", DATE, apply=True, cycle=CYCLE)
    after = p.read_bytes()
    assert after.count(b"\r\n") == before.count(b"\r\n")
    assert after.count(b"\n") == after.count(b"\r\n"), "no bare LF introduced"


def test_several_ids_are_stamped_in_one_run(tmp_path):
    p = write(tmp_path)
    stamps = approve(p, ["clean", "other_clean"], "Marc", DATE, apply=True, cycle=CYCLE)
    assert [s.block_id for s in stamps] == ["clean", "other_clean"]
    assert {by_id(p)[i]["status"] for i in ("clean", "other_clean")} == {"approved"}


# --- the block's own waiver ------------------------------------------------------------------
def test_a_block_whose_own_waiver_covers_its_wording_can_be_approved(tmp_path):
    p = write(tmp_path)
    approve(p, ["spc_waived"], "Marc Pujol-Gonzalez", DATE, apply=True, cycle=CYCLE)
    e = by_id(p)["spc_waived"]
    assert e["status"] == "approved" and e["approved_by"] == "Marc Pujol-Gonzalez"
    assert e["lint_waivers"] == [WAIVER], "the nested waiver's own approved_by is untouched"
    assert "spc_waived" in {t.id for t in load_approved_templates(p, cycle=CYCLE)}


# --- refusals: nothing is written --------------------------------------------------------------
@pytest.mark.parametrize("block_id, reason", [
    ("blocked", "blocked: blocked"),
    ("retired", "retired: status_is_retired"),
    ("already", "already: status_is_approved"),
    ("no_such_block", "no_such_block: missing"),
    ("this_year", "this_year: lint:year_or_time_specific"),
    ("spc_unwaived", "spc_unwaived: lint:internal_roles_or_process"),
    ("placeholder", "unknown_placeholder"),
    ("wrong_cycle", "wrong_cycle: wrong_cycle"),
    ("unordered_point", "unordered_point: bad_order"),
])
def test_a_block_the_loader_would_refuse_is_refused_and_nothing_is_written(tmp_path, block_id, reason):
    p = write(tmp_path)
    before = p.read_bytes()
    with pytest.raises(Refused) as exc:
        approve(p, [block_id], "Marc", DATE, apply=True, cycle=CYCLE)
    assert reason in str(exc.value)
    assert p.read_bytes() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["templates.json"]


def test_one_bad_id_refuses_the_whole_run(tmp_path):
    p = write(tmp_path)
    before = p.read_bytes()
    with pytest.raises(Refused, match="blocked: blocked"):
        approve(p, ["clean", "blocked"], "Marc", DATE, apply=True, cycle=CYCLE)
    assert p.read_bytes() == before


def test_a_refusal_never_contains_body_text(tmp_path):
    p = write(tmp_path)
    with pytest.raises(Refused) as exc:
        approve(p, ["this_year"], "Marc", DATE, apply=True, cycle=CYCLE)
    assert "this year" not in str(exc.value) and "worked" not in str(exc.value)


@pytest.mark.parametrize("ids, approved_by, when, reason", [
    (["clean"], "", DATE, "empty_approved_by"),
    (["clean"], "   ", DATE, "empty_approved_by"),
    (["clean"], "Marc", "2026-13-01", "bad_date"),
    (["clean"], "Marc", "02/10/2026", "bad_date"),
    (["clean"], "Marc", "", "bad_date"),
    ([], "Marc", DATE, "no_ids"),
    (["clean", "clean"], "Marc", DATE, "duplicate_ids_requested"),
])
def test_bad_arguments_are_refused_and_nothing_is_written(tmp_path, ids, approved_by, when, reason):
    p = write(tmp_path)
    before = p.read_bytes()
    with pytest.raises(Refused, match=reason):
        approve(p, ids, approved_by, when, apply=True, cycle=CYCLE)
    assert p.read_bytes() == before


def test_a_block_id_that_appears_twice_in_the_file_is_refused(tmp_path):
    p = write(tmp_path, _entries() + [_entry("clean", order=50)])
    before = p.read_bytes()
    with pytest.raises(Refused, match="clean: duplicate_id"):
        approve(p, ["clean"], "Marc", DATE, apply=True, cycle=CYCLE)
    assert p.read_bytes() == before


# --- the CLI ------------------------------------------------------------------------------------
def test_the_cli_dry_run_prints_the_pairs_and_writes_nothing(tmp_path, capsys):
    p = write(tmp_path)
    before = p.read_bytes()
    code = aarb.main(["--ids", "clean", "--approved-by", "Marc", "--date", DATE, "--path", str(p)])
    out = capsys.readouterr().out
    assert code == 0 and "DRY RUN" in out
    assert f'("clean", "{compute_body_sha256(CLEAN)}")' in out
    assert p.read_bytes() == before


def test_the_cli_apply_writes(tmp_path, capsys):
    p = write(tmp_path)
    code = aarb.main(["--ids", "clean", "--approved-by", "Marc", "--date", DATE, "--path", str(p), "--apply"])
    assert code == 0 and "STAMPED" in capsys.readouterr().out
    assert by_id(p)["clean"]["status"] == "approved"


def test_the_cli_refusal_exits_non_zero_and_writes_nothing(tmp_path, capsys):
    p = write(tmp_path)
    before = p.read_bytes()
    code = aarb.main(["--ids", "blocked", "--approved-by", "Marc", "--date", DATE, "--path", str(p), "--apply"])
    assert code == 1
    assert "REFUSED" in capsys.readouterr().err
    assert p.read_bytes() == before


# --- against a copy of the REAL file ---------------------------------------------------------------
@pytest.fixture
def real_copy(tmp_path) -> Path:
    p = tmp_path / "templates.json"
    p.write_bytes(art.DEFAULT_PATH.read_bytes())
    return p


@pytest.mark.parametrize("block_id, reason", [
    # Approved 2026-10-05, so re-approving it is now refused as already approved
    # (it used to be refused as blocked; the blocked rule is covered on synthetic
    # entries above, since no live block in the real file is blocked any more).
    ("standalone_ai_review", "standalone_ai_review: status_is_approved"),
    ("holding_wrong_paper", "holding_wrong_paper: status_is_retired"),
    ("point_scores", "point_scores: status_is_approved"),
])
def test_the_real_file_refuses_retired_and_already_approved_blocks(real_copy, block_id, reason):
    before = real_copy.read_bytes()
    with pytest.raises(Refused, match=reason):
        approve(real_copy, [block_id], "Prof. Yan", DATE, apply=True)
    assert real_copy.read_bytes() == before


def test_every_approved_block_in_the_real_file_carries_the_scripts_hash():
    for e in json.loads(art.DEFAULT_PATH.read_text(encoding="utf-8"))["templates"]:
        if e["status"] == "approved":
            assert e["approved_sha256"] == compute_body_sha256(e["body"]), e["id"]
