"""Tests for the labeling CLI: Phase 0 regression pins + the --phase2 mode.

SCOPE LIMIT: SYNTHETIC records only — nothing here reads data/labeling/.

`scripts/labeling/` is not in the backend image (the Dockerfile copies only
backend/ and data/), so this whole file SKIPS in the container and runs on the
host and in CI, where the repo root is present. Same pattern as the drift test
in test_appeal_reasons.py.

label_appeals.py had NO tests before this file. The Phase 0 cases below pin its
current behavior, so the --phase2 addition is provably not a change to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_LABELING_DIR = Path(__file__).resolve().parents[2] / "scripts" / "labeling"

pytestmark = pytest.mark.skipif(
    not (_LABELING_DIR / "label_appeals.py").exists(),
    reason="scripts/labeling/ is not in the backend image (the Dockerfile copies "
    "only backend/ and data/); these tests run on the host and in CI",
)

if (_LABELING_DIR / "label_appeals.py").exists():
    sys.path.insert(0, str(_LABELING_DIR))
    import label_appeals as la  # noqa: E402
    import label_phase2 as lp  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_AGENT_BODY = "AGENT-ONLY-TEXT-must-never-be-shown"
_MARC_BODY = "MARC-REPLY-TEXT-must-never-be-shown"
_PRIVATE_AUTHOR_BODY = "NON-PUBLIC-AUTHOR-TEXT-must-never-be-shown"


def _phase0_row(tid: int) -> dict:
    """Field order matches the real Phase 0 file (appeal fields mid-record)."""
    return {
        "ticket_id": tid,
        "created_at": "2025-09-21T10:00:00Z",
        "channel": "email",
        "status": "solved",
        "subject": f"subject {tid}",
        "initial_message_body": f"body {tid}",
        "marc_reply_body": _MARC_BODY,
        "submission_numbers_mentioned": None,
        "is_reject_appeal": None,
        "appeal_reason": None,
        "thread": [],
        "marc_replies": [],
    }


def _pool_row(tid: int, **extra) -> dict:
    """A Phase 2 pool row: Phase 0 record shape WITHOUT the Phase 0 label fields."""
    row = {
        "ticket_id": tid,
        "created_at": "2025-09-16T10:00:00Z",
        "channel": "email",
        "status": "solved",
        "subject": f"subject {tid}",
        "initial_message_body": f"author body {tid}",
        "marc_reply_body": _MARC_BODY,
        "submission_numbers_mentioned": None,
        "thread": [
            {"sender_type": "requester", "sender_email_or_null": None,
             "body": f"author body {tid}", "created_at": "t1", "is_public": True},
            {"sender_type": "agent", "sender_email_or_null": None,
             "body": _AGENT_BODY, "created_at": "t2", "is_public": True},
            {"sender_type": "other_end_user", "sender_email_or_null": None,
             "body": f"coauthor {tid}", "created_at": "t3", "is_public": True},
            {"sender_type": "unknown", "sender_email_or_null": None,
             "body": "UNKNOWN-SENDER-TEXT", "created_at": "t4", "is_public": False},
            {"sender_type": "agent", "sender_email_or_null": None,
             "body": _AGENT_BODY, "created_at": "t5", "is_public": True},
            # Non-public AUTHOR messages: production never shows them to the
            # model (thread_transcript.build_transcript keeps public turns only).
            {"sender_type": "other_end_user", "sender_email_or_null": None,
             "body": _PRIVATE_AUTHOR_BODY, "created_at": "t6", "is_public": False},
            {"sender_type": "requester", "sender_email_or_null": None,
             "body": _PRIVATE_AUTHOR_BODY, "created_at": "t7", "is_public": False},
        ],
        "marc_replies": [{"body": _MARC_BODY, "created_at": "t2"}],
    }
    row.update(extra)
    return row


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _feed(monkeypatch, answers: list[str]) -> None:
    """Drive builtins.input; running out raises EOFError, which the tools treat as quit."""
    it = iter(answers)

    def fake_input(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)


class _Harness:
    """run_phase2 with injected I/O and a persist counter."""

    def __init__(self, records, answers):
        self.records = records
        self.lines: list[str] = []
        self.persists = 0
        self._answers = iter(answers)

    def _input(self, prompt=""):
        self.lines.append(prompt)  # the [] confirmation is asked via the prompt
        try:
            return next(self._answers)
        except StopIteration:
            raise EOFError

    def _persist(self):
        self.persists += 1
        self.lines.append("<persist>")

    def run(self, **kw):
        return lp.run_phase2(self.records, self._persist, input_fn=self._input,
                             out=self.lines.append, **kw)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _load_rows(rows):
    """The in-memory defaults `lp.load_phase2` applies, without a file."""
    for r in rows:
        for key in lp.PHASE2_FIELDS:
            r.setdefault(key, False if key == la.DEFERRED_KEY else None)
    return rows, 0


def _one(answers: list[str], **row_extra) -> tuple[dict, _Harness]:
    records, _ = _load_rows([_pool_row(1, **row_extra)])
    h = _Harness(records, answers)
    h.run()
    return records[0], h


# ---------------------------------------------------------------------------
# Phase 0 regression pins — behavior and format must not move
# ---------------------------------------------------------------------------
def test_phase0_label_writes_single_code_and_keeps_field_order(tmp_path, monkeypatch):
    path = _write(tmp_path / "p0.jsonl", [_phase0_row(1)])
    original_keys = list(_phase0_row(1))
    _feed(monkeypatch, ["r"])
    assert la.main([str(path)]) == 0

    row = _read(path)[0]
    assert row["is_reject_appeal"] is True
    assert row["appeal_reason"] == "r"
    assert row["deferred"] is False
    # Existing field order intact; `deferred` appended at the END.
    assert list(row) == original_keys + ["deferred"]
    # No Phase 2 field leaks into a Phase 0 file.
    for key in ("appeal_reason_codes", "is_reciprocal_label"):
        assert key not in row


def test_phase0_not_an_appeal(tmp_path, monkeypatch):
    path = _write(tmp_path / "p0.jsonl", [_phase0_row(1)])
    _feed(monkeypatch, ["n"])
    la.main([str(path)])
    row = _read(path)[0]
    assert row["is_reject_appeal"] is False
    assert row["appeal_reason"] is None


def test_phase0_quit_without_change_leaves_file_byte_identical(tmp_path, monkeypatch):
    path = _write(tmp_path / "p0.jsonl", [_phase0_row(1), _phase0_row(2)])
    before = path.read_bytes()
    _feed(monkeypatch, ["q"])
    la.main([str(path)])
    assert path.read_bytes() == before


def test_phase0_defer_then_review_deferred(tmp_path, monkeypatch):
    path = _write(tmp_path / "p0.jsonl", [_phase0_row(1)])
    _feed(monkeypatch, ["s"])
    la.main([str(path)])
    assert _read(path)[0]["deferred"] is True

    _feed(monkeypatch, ["b"])
    la.main([str(path), "--review-deferred"])
    row = _read(path)[0]
    assert (row["is_reject_appeal"], row["appeal_reason"], row["deferred"]) == (True, "b", False)


def test_phase0_refuses_a_phase2_file(tmp_path, monkeypatch, capsys):
    rows = [dict(_pool_row(1), is_reject_appeal=None, appeal_reason_codes=None,
                 is_reciprocal_label=None, deferred=False)]
    path = _write(tmp_path / "pool.jsonl", rows)
    before = path.read_bytes()
    _feed(monkeypatch, ["r"])
    assert la.main([str(path)]) == 2
    assert "Phase 2 file" in capsys.readouterr().err
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Phase 2 — reason codes
# ---------------------------------------------------------------------------
def test_phase2_reason_codes_pinned_in_registry_order():
    """Same literal as test_appeal_reasons.py's container-side pin."""
    assert lp.REASON_CODES == ("a", "b", "c", "d", "e", "o")


# ---------------------------------------------------------------------------
# Phase 2 — requester-only view (D63, D72 amendment)
# ---------------------------------------------------------------------------
def test_author_view_shows_only_the_requesters_public_messages():
    shown, hidden = lp.author_view(_pool_row(1))
    assert [(e["sender_type"], e["body"]) for e in shown] == [("requester", "author body 1")]
    assert hidden == 6  # 2 agent, 2 other_end_user, 1 unknown, 1 non-public requester


def test_co_author_messages_are_hidden_even_when_public():
    """Roles come from the CURRENT users pull, so a past-year chair whose account
    is now `end-user` looks exactly like a co-author. Only the requester shows."""
    row = _pool_row(1)
    row["thread"] = [{"sender_type": "other_end_user", "sender_email_or_null": None,
                      "body": "coauthor 1", "created_at": "t1", "is_public": True}]
    assert lp.author_view(row) == ([], 1)


@pytest.mark.parametrize("sender", ["agent", "other_end_user", "unknown", None, "Requester"])
def test_every_non_requester_sender_is_hidden(sender):
    row = _pool_row(1)
    row["thread"] = [{"sender_type": sender, "sender_email_or_null": None,
                      "body": "x", "created_at": "t1", "is_public": True}]
    assert lp.author_view(row) == ([], 1)


@pytest.mark.parametrize("flag", [False, None, "true", 1])
def test_non_public_requester_message_is_hidden_like_production(flag):
    """Matches thread_transcript.build_transcript: only a real public turn shows.

    Anything but `is_public is True` is hidden (and counted), so an odd or
    missing flag can never surface a message the model would not have seen.
    """
    row = _pool_row(1)
    row["thread"] = [{"sender_type": "requester", "sender_email_or_null": None,
                      "body": _PRIVATE_AUTHOR_BODY, "created_at": "t1", "is_public": flag}]
    assert lp.author_view(row) == ([], 1)


def test_ticket_display_shows_only_requester_text_and_one_hidden_count():
    lines: list[str] = []
    lp.show_ticket_phase2(_pool_row(1), 1, 1, lines.append)
    text = "\n".join(lines)
    assert "author body 1" in text
    assert "coauthor 1" not in text
    assert _AGENT_BODY not in text
    assert _MARC_BODY not in text  # marc_reply_body and marc_replies never shown
    assert "UNKNOWN-SENDER-TEXT" not in text
    assert _PRIVATE_AUTHOR_BODY not in text
    assert "6 other messages hidden" in text


def test_no_visibility_marker_is_ever_shown():
    """A marker on a shown message could bias the labeler; every shown message is public."""
    lines: list[str] = []
    lp.show_ticket_phase2(_pool_row(1), 1, 1, lines.append)
    assert "[INTERNAL]" not in "\n".join(lines)


def test_ticket_display_never_shows_why_a_ticket_was_picked():
    """No bucket or pattern hint reaches the labeler, even if one leaked into a row."""
    lines: list[str] = []
    lp.show_ticket_phase2(_pool_row(1, bucket="a_pattern", status="PICKED-BY-REGEX"),
                          1, 1, lines.append)
    text = "\n".join(lines)
    assert "a_pattern" not in text and "bucket" not in text
    assert "PICKED-BY-REGEX" not in text


def test_no_requester_messages_says_so():
    row = _pool_row(1)
    row["thread"] = [e for e in row["thread"] if e["sender_type"] != "requester"]
    lines: list[str] = []
    lp.show_ticket_phase2(row, 1, 1, lines.append)
    assert "no requester messages" in "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2 — multi-label selection
# ---------------------------------------------------------------------------
def test_multi_label_saves_codes_in_registry_order():
    row, h = _one(["o", "c", "a", ""])
    assert row["is_reject_appeal"] is True
    assert row["appeal_reason_codes"] == ["a", "c", "o"]
    assert row["is_reciprocal_label"] is False
    assert row["deferred"] is False
    # Saved AT LABEL TIME (before end-of-input triggers the Phase 0-style
    # final save on the way out).
    recorded = next(i for i, l in enumerate(h.lines) if l.startswith("recorded:"))
    assert h.lines[recorded - 1] == "<persist>"


def test_toggling_twice_removes_a_reason():
    row, _ = _one(["a", "b", "a", ""])
    assert row["appeal_reason_codes"] == ["b"]


def test_r_is_a_separate_box_not_a_reason():
    row, h = _one(["r", ""])
    assert row["is_reject_appeal"] is True
    assert row["appeal_reason_codes"] == []
    assert row["is_reciprocal_label"] is True
    assert "NO listed reason" not in h.text  # r ticked => no [] confirmation


def test_r_combines_with_reasons():
    row, _ = _one(["b", "r", ""])
    assert (row["appeal_reason_codes"], row["is_reciprocal_label"]) == (["b"], True)


def test_n_clears_reasons_and_reciprocal():
    row, _ = _one(["a", "r", "n", ""])
    assert row["is_reject_appeal"] is False
    assert row["appeal_reason_codes"] == []  # D67: n scores as []
    assert row["is_reciprocal_label"] is False


def test_n_really_clears_the_box_so_it_cannot_come_back():
    """Saving right after `n` writes False regardless, which would hide a stale
    box. Toggling a reason AFTER `n` is where a leftover tick would reappear."""
    row, _ = _one(["r", "n", "d", ""])
    assert (row["is_reject_appeal"], row["appeal_reason_codes"], row["is_reciprocal_label"]) == (True, ["d"], False)


def test_toggling_after_n_makes_it_an_appeal_again():
    row, _ = _one(["n", "d", ""])
    assert (row["is_reject_appeal"], row["appeal_reason_codes"]) == (True, ["d"])


def test_empty_appeal_needs_confirmation_and_y_saves_it():
    row, h = _one(["", "y"])
    assert "NO listed reason" in h.text
    assert row["is_reject_appeal"] is True
    assert row["appeal_reason_codes"] == []
    assert row["is_reciprocal_label"] is False


def test_empty_appeal_not_confirmed_is_not_saved():
    row, h = _one(["", "", "a", ""])
    assert "not saved" in h.text
    assert row["appeal_reason_codes"] == ["a"]
    assert sum(l.startswith("recorded:") for l in h.lines) == 1


def test_selection_is_shown_before_saving():
    _, h = _one(["a", "r", ""])
    current = [i for i, l in enumerate(h.lines)
               if l.startswith("CURRENT:") and "[a]" in l and "reciprocal: YES" in l]
    recorded = [i for i, l in enumerate(h.lines) if l.startswith("recorded:")]
    assert current and recorded and current[-1] < recorded[0]


def test_unknown_key_is_rejected_and_nothing_saved():
    row, h = _one(["x"])
    assert "'x' is not one of" in h.text
    assert row["is_reject_appeal"] is None
    assert h.persists == 0


def test_defer_leaves_labels_null():
    row, h = _one(["a", "s"])
    assert row["deferred"] is True
    assert (row["is_reject_appeal"], row["appeal_reason_codes"], row["is_reciprocal_label"]) == (None, None, None)
    deferred = next(i for i, l in enumerate(h.lines) if l.startswith("deferred -"))
    assert h.lines[deferred - 1] == "<persist>"


def test_review_deferred_shows_only_deferred_and_labeling_clears_it():
    records, _ = _load_rows([_pool_row(1), _pool_row(2, deferred=True)])
    h = _Harness(records, ["b", ""])
    h.run(review_deferred=True)
    by = {r["ticket_id"]: r for r in records}
    assert by[2]["appeal_reason_codes"] == ["b"] and by[2]["deferred"] is False
    assert by[1]["is_reject_appeal"] is None  # untouched ticket not offered


def test_default_mode_never_offers_a_deferred_ticket():
    """Order-independent: a file holding ONLY a deferred ticket has nothing to label."""
    records, _ = _load_rows([_pool_row(1, deferred=True)])
    h = _Harness(records, ["a", ""])
    h.run()
    assert "Nothing left to label" in h.text
    assert records[0]["is_reject_appeal"] is None and h.persists == 0


def test_review_mode_never_offers_an_untouched_ticket():
    records, _ = _load_rows([_pool_row(1)])
    h = _Harness(records, ["a", ""])
    h.run(review_deferred=True)
    assert "Nothing deferred to review" in h.text
    assert records[0]["is_reject_appeal"] is None and h.persists == 0


def test_labeled_rows_are_skipped_on_resume():
    records, _ = _load_rows([_pool_row(1, is_reject_appeal=True, appeal_reason_codes=["a"],
                                       is_reciprocal_label=False)])
    h = _Harness(records, [])
    h.run()
    assert "Nothing left to label" in h.text
    assert h.persists == 0


# ---------------------------------------------------------------------------
# Phase 2 — file handling through the real entry point
# ---------------------------------------------------------------------------
def test_phase2_end_to_end_writes_the_four_fields_atomically(tmp_path, monkeypatch):
    path = _write(tmp_path / "pool.jsonl", [_pool_row(1), _pool_row(2)])
    _feed(monkeypatch, ["a", "b", "", "q"])
    assert la.main([str(path), "--phase2"]) == 0

    rows = {r["ticket_id"]: r for r in _read(path)}
    labeled = [r for r in rows.values() if r["is_reject_appeal"] is not None]
    assert len(labeled) == 1
    assert labeled[0]["appeal_reason_codes"] == ["a", "b"]
    assert labeled[0]["is_reciprocal_label"] is False
    for r in rows.values():
        assert "appeal_reason" not in r  # never the Phase 0 field
        for key in lp.PHASE2_FIELDS:
            assert key in r
    assert not list(tmp_path.glob("*.tmp"))  # atomic save left no temp file


def test_phase2_quit_without_change_leaves_file_byte_identical(tmp_path, monkeypatch):
    """Lazy defaults never mark the session dirty."""
    path = _write(tmp_path / "pool.jsonl", [_pool_row(1)])
    before = path.read_bytes()
    _feed(monkeypatch, ["a", "q"])  # a toggle is not a change until Enter
    la.main([str(path), "--phase2"])
    assert path.read_bytes() == before


def test_phase2_refuses_a_phase0_file(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path / "p0.jsonl", [_phase0_row(1)])
    before = path.read_bytes()
    _feed(monkeypatch, ["a", ""])
    assert la.main([str(path), "--phase2"]) == 2
    assert "Phase 0 file" in capsys.readouterr().err
    assert path.read_bytes() == before


def test_phase2_resumes_after_interruption(tmp_path, monkeypatch):
    path = _write(tmp_path / "pool.jsonl", [_pool_row(1), _pool_row(2)])
    _feed(monkeypatch, ["c", ""])  # label one, then EOF
    la.main([str(path), "--phase2"])
    _feed(monkeypatch, ["e", ""])
    la.main([str(path), "--phase2"])
    codes = sorted(tuple(r["appeal_reason_codes"]) for r in _read(path))
    assert codes == [("c",), ("e",)]
