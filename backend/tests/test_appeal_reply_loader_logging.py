"""Loader refusal logging (reject-appeal Phase 4, Step 4).

Retired entries are intentional and log nothing; a draft entry (blocked or not)
logs at most once per process; an approved entry that fails a rule logs every
time, by id and rule name only. Refusal BEHAVIOUR is unchanged: none of these
entries is ever served. Each test uses its own tmp_path file, so the
once-per-process memory (keyed by file, id and rules) cannot leak between tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.pipeline.appeal_reply_templates import compute_body_sha256, load_approved_templates

CYCLE = "AAAI-27"
BODY = "Decisions are not based on any single review; all assessments are weighed together."
LOGGER = "app.pipeline.appeal_reply_templates"


def _entry(block_id, **over):
    e = {
        "id": block_id, "title": "t", "kind": "point", "order": 1, "optional": False,
        "reasons": ["reviewer_misunderstanding"], "when_used": "w", "body": BODY,
        "status": "approved", "approved_by": "Marc", "approved_at": "2026-10-02",
        "approved_sha256": compute_body_sha256(BODY), "cycle": CYCLE, "scope": "phase1_reject",
        "basis": [], "blocked_on": [],
    }
    e.update(over)
    return e


def write(tmp_path: Path, *entries) -> Path:
    p = tmp_path / "templates.json"
    p.write_text(json.dumps({"schema_version": 2, "templates": list(entries)}), encoding="utf-8")
    return p


def refusals(caplog, block_id: str) -> int:
    return sum(1 for r in caplog.records
               if r.name == LOGGER and f"'{block_id}' refused" in r.getMessage())


def load_twice(path):
    return [t.id for t in load_approved_templates(path, cycle=CYCLE)], \
           [t.id for t in load_approved_templates(path, cycle=CYCLE)]


def test_a_retired_entry_logs_nothing_and_is_still_not_served(tmp_path, caplog):
    p = write(tmp_path, _entry("old", status="retired", approved_by=None, approved_at=None,
                               approved_sha256=None))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert load_twice(p) == ([], [])
    assert refusals(caplog, "old") == 0


def test_a_draft_entry_logs_once_per_process_and_is_still_not_served(tmp_path, caplog):
    p = write(tmp_path, _entry("pending", status="draft", approved_by=None, approved_at=None,
                               approved_sha256=None))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert load_twice(p) == ([], [])
        load_approved_templates(p, cycle=CYCLE)
    assert refusals(caplog, "pending") == 1


def test_a_blocked_draft_entry_logs_once_per_process(tmp_path, caplog):
    p = write(tmp_path, _entry("waiting", status="draft", approved_by=None, approved_at=None,
                               approved_sha256=None, blocked_on=["yan_confirm"]))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert load_twice(p) == ([], [])
    assert refusals(caplog, "waiting") == 1


def test_an_approved_entry_that_fails_a_rule_logs_every_time(tmp_path, caplog):
    p = write(tmp_path, _entry("broken", approved_sha256="0" * 64))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert load_twice(p) == ([], [])
        load_approved_templates(p, cycle=CYCLE)
    assert refusals(caplog, "broken") == 3
    assert "body_hash_mismatch" in caplog.text
    assert BODY not in caplog.text, "never the body"


def test_a_duplicated_retired_id_is_still_logged(tmp_path, caplog):
    retired = _entry("twice", status="retired")
    p = write(tmp_path, retired, dict(retired))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        load_approved_templates(p, cycle=CYCLE)
    assert refusals(caplog, "twice") == 2


def test_the_once_per_process_memory_is_per_file(tmp_path, caplog):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    draft = _entry("pending", status="draft", approved_by=None, approved_at=None, approved_sha256=None)
    pa, pb = write(a, draft), write(b, draft)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        load_approved_templates(pa, cycle=CYCLE)
        load_approved_templates(pb, cycle=CYCLE)
    assert refusals(caplog, "pending") == 2


def test_the_good_entries_are_served_exactly_as_before(tmp_path):
    p = write(tmp_path, _entry("good"), _entry("old", status="retired"),
              _entry("pending", id="pending", status="draft"))
    assert [t.id for t in load_approved_templates(p, cycle=CYCLE)] == ["good"]


def test_the_real_file_logs_nothing_for_its_retired_blocks(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        load_approved_templates()
        load_approved_templates()
    for retired in ("point_ai_review", "point_report_form", "holding_wrong_paper",
                    "holding_score_mismatch", "holding_both", "body_reconsider"):
        assert refusals(caplog, retired) == 0, retired
    # Approved 2026-10-05 (it used to be draft and blocked, logged at most once): an
    # approved block that passes every rule is never logged at all.
    assert refusals(caplog, "standalone_ai_review") == 0, "approved: never logged"
