"""Tests for scripts/labeling/ticket_extraction.py + the Phase 0 verifier.

SCOPE LIMIT: SYNTHETIC archive data only — nothing here reads data/tickets/ or
data/labeling/. `scripts/labeling/` is not in the backend image, so this file
SKIPS in the container and runs on the host and in CI (same pattern as
test_label_appeals_phase2.py).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_LABELING_DIR = Path(__file__).resolve().parents[2] / "scripts" / "labeling"

pytestmark = pytest.mark.skipif(
    not (_LABELING_DIR / "ticket_extraction.py").exists(),
    reason="scripts/labeling/ is not in the backend image (the Dockerfile copies "
    "only backend/ and data/); these tests run on the host and in CI",
)

if (_LABELING_DIR / "ticket_extraction.py").exists():
    sys.path.insert(0, str(_LABELING_DIR))
    import ticket_extraction as te  # noqa: E402
    import verify_phase0_extraction as vp  # noqa: E402

REQUESTER, COAUTHOR, CHAIR, ADMIN, MARC, GHOST = 1, 2, 3, 4, 5, 99


def _users():
    return {
        REQUESTER: {"id": REQUESTER, "role": "end-user", "email": "req@example.org"},
        COAUTHOR: {"id": COAUTHOR, "role": "end-user", "email": "co@example.org"},
        CHAIR: {"id": CHAIR, "role": "agent", "email": "chair@example.org"},
        ADMIN: {"id": ADMIN, "role": "admin", "email": None},
        MARC: {"id": MARC, "role": "admin", "email": "marc@example.org"},
        # GHOST is deliberately absent from users -> "unknown"
    }


def _ticket(tid=100, created="2025-09-21T10:00:00Z", requester=REQUESTER):
    return {"id": tid, "created_at": created, "status": "closed", "subject": f"subj {tid}",
            "description": "DESCRIPTION-differs-from-first-comment",
            "requester_id": requester, "via": {"channel": "email"}}


def _c(author, created, body, public=True, tid=100):
    return {"ticket_id": tid, "created_at": created, "author_id": author,
            "public": public, "body": body, "via": "email"}


def _comments():
    # Deliberately NOT in time order in the "file".
    return [
        _c(CHAIR, "2025-09-21T12:00:00Z", "chair reply"),
        _c(REQUESTER, "2025-09-21T10:00:00Z", "  first message  \n"),
        _c(MARC, "2025-09-21T13:00:00Z", "marc private", public=False),
        _c(MARC, "2025-09-21T14:00:00Z", "marc public 1"),
        _c(COAUTHOR, "2025-09-21T11:00:00Z", "coauthor"),
        _c(GHOST, "2025-09-21T15:00:00Z", "ghost", public=False),
        _c(ADMIN, "2025-09-21T16:00:00Z", "admin note", public=False),
        _c(MARC, "2025-09-21T17:00:00Z", "marc public 2"),
    ]


def _no_numbers(subject, body):
    return []


def _record(**kw):
    kw.setdefault("find_numbers", _no_numbers)
    return te.build_record(_ticket(), _comments(), _users(), MARC, **kw)


# ---------------------------------------------------------------------------
# sender_type / thread
# ---------------------------------------------------------------------------
def test_sender_type_rules():
    t, u = _ticket(), _users()
    assert te.sender_type(_c(REQUESTER, "x", ""), t, u) == "requester"
    assert te.sender_type(_c(COAUTHOR, "x", ""), t, u) == "other_end_user"
    assert te.sender_type(_c(CHAIR, "x", ""), t, u) == "agent"
    assert te.sender_type(_c(ADMIN, "x", ""), t, u) == "agent"
    assert te.sender_type(_c(GHOST, "x", ""), t, u) == "unknown"


def test_requester_wins_over_an_agent_role():
    """A requester who carries an agent role is still the requester."""
    assert te.sender_type(_c(CHAIR, "x", ""), _ticket(requester=CHAIR), _users()) == "requester"


def test_thread_is_every_comment_in_time_order_with_the_exact_entry_shape():
    thread = _record()["thread"]
    assert [e["body"] for e in thread] == [
        "  first message  \n", "coauthor", "chair reply", "marc private",
        "marc public 1", "ghost", "admin note", "marc public 2",
    ]
    assert [e["sender_type"] for e in thread] == [
        "requester", "other_end_user", "agent", "agent", "agent", "unknown", "agent", "agent",
    ]
    assert [e["is_public"] for e in thread] == [True, True, True, False, True, False, False, True]
    assert all(list(e) == ["sender_type", "sender_email_or_null", "body", "created_at", "is_public"]
               for e in thread)


def test_email_is_null_for_unknown_authors_and_users_without_one():
    by_body = {e["body"]: e["sender_email_or_null"] for e in _record()["thread"]}
    assert by_body["ghost"] is None
    assert by_body["admin note"] is None
    assert by_body["coauthor"] == "co@example.org"


def test_equal_timestamps_keep_file_order():
    comments = [_c(REQUESTER, "2025-09-21T10:00:00Z", "b"), _c(REQUESTER, "2025-09-21T10:00:00Z", "a")]
    rec = te.build_record(_ticket(), comments, _users(), MARC, find_numbers=_no_numbers)
    assert [e["body"] for e in rec["thread"]] == ["b", "a"]


def test_bodies_are_verbatim_not_stripped():
    assert _record()["thread"][0]["body"] == "  first message  \n"


# ---------------------------------------------------------------------------
# initial message / Marc / submission numbers
# ---------------------------------------------------------------------------
def test_initial_message_is_the_first_comment_not_the_description():
    rec = _record()
    assert rec["initial_message_body"] == "  first message  \n"
    assert rec["initial_message_body"] != _ticket()["description"]


def test_no_comments_gives_an_empty_initial_message():
    rec = te.build_record(_ticket(), [], _users(), MARC, find_numbers=_no_numbers)
    assert rec["initial_message_body"] == "" and rec["thread"] == []


def test_marc_replies_are_public_only_in_thread_order():
    rec = _record()
    assert rec["marc_replies"] == [
        {"body": "marc public 1", "created_at": "2025-09-21T14:00:00Z"},
        {"body": "marc public 2", "created_at": "2025-09-21T17:00:00Z"},
    ]
    assert rec["marc_reply_body"] == "marc public 1"


def test_no_marc_reply_gives_none_and_an_empty_list():
    comments = [c for c in _comments() if c["author_id"] != MARC]
    rec = te.build_record(_ticket(), comments, _users(), MARC, find_numbers=_no_numbers)
    assert rec["marc_reply_body"] is None and rec["marc_replies"] == []


def test_submission_numbers_come_from_subject_and_initial_message_only():
    calls = []

    def spy(subject, body):
        calls.append((subject, body))
        return ["12345", "6789"]

    rec = _record(find_numbers=spy)
    assert calls == [("subj 100", "  first message  \n")]
    assert rec["submission_numbers_mentioned"] == ["12345", "6789"]


def test_no_submission_numbers_is_none_not_an_empty_list():
    assert _record()["submission_numbers_mentioned"] is None


def test_default_uses_the_production_extractor():
    """Reused, not copied: the default is the live extractor's answer."""
    from app.pipeline.extractor import EmailExtractor

    subject, body = "Appeal for paper 12345", "Our submission #23456 was rejected."
    expected = EmailExtractor().extract(subject, body, "", "", None).submission_numbers
    assert expected  # the fixture must exercise a real hit
    assert te.production_submission_numbers(subject, body) == expected


# ---------------------------------------------------------------------------
# record shape / label fields
# ---------------------------------------------------------------------------
_PHASE0_KEYS = ["ticket_id", "created_at", "channel", "status", "subject",
                "initial_message_body", "marc_reply_body", "submission_numbers_mentioned",
                "is_reject_appeal", "appeal_reason", "thread", "marc_replies"]


def test_phase0_label_fields_sit_in_the_middle_in_the_exact_key_order():
    rec = _record(label_fields=te.PHASE0_LABEL_FIELDS)
    assert list(rec) == _PHASE0_KEYS
    assert rec["is_reject_appeal"] is None and rec["appeal_reason"] is None


def test_without_label_fields_the_record_has_none():
    """The Phase 2 pool: label fields are added by the labeling tool itself."""
    rec = _record()
    assert list(rec) == [k for k in _PHASE0_KEYS if k not in ("is_reject_appeal", "appeal_reason")]


def test_label_fields_are_copied_not_shared():
    a = _record(label_fields=te.PHASE0_LABEL_FIELDS)
    a["is_reject_appeal"] = True
    assert te.PHASE0_LABEL_FIELDS == {"is_reject_appeal": None, "appeal_reason": None}


def test_channel_comes_from_via():
    assert _record()["channel"] == "email"


# ---------------------------------------------------------------------------
# window selection + loading
# ---------------------------------------------------------------------------
def _archive():
    tickets = {
        1: _ticket(1, "2025-09-20T00:00:00Z"),
        2: _ticket(2, "2025-09-30T23:59:59Z"),
        3: _ticket(3, "2025-09-19T23:59:59Z"),
        4: _ticket(4, "2025-10-01T00:00:00Z"),
        5: _ticket(5, "2025-09-25T08:00:00Z"),
        6: _ticket(6, "2025-09-25T08:00:00Z"),
    }
    return te.Archive(tickets, [2, 6, 1, 5, 3, 4], _users(), {}, MARC)


def test_window_is_date_inclusive_and_time_ordered_with_stable_ties():
    assert te.select_window(_archive(), "2025-09-20", "2025-09-30") == [1, 6, 5, 2]


def test_load_archive_reads_the_four_files(tmp_path):
    (tmp_path / "tickets.jsonl").write_text(json.dumps(_ticket(7)) + "\n", encoding="utf-8")
    (tmp_path / "users.jsonl").write_text(
        "".join(json.dumps(u) + "\n" for u in _users().values()), encoding="utf-8")
    (tmp_path / "comment_events.jsonl").write_text(
        "".join(json.dumps(dict(c, ticket_id=7)) + "\n" for c in _comments()), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"marc_threads": {"user_id": MARC}}),
                                            encoding="utf-8")
    arc = te.load_archive(tmp_path)
    assert arc.ticket_order == [7] and arc.marc_user_id == MARC
    assert len(arc.comments[7]) == len(_comments()) and set(arc.users) == set(_users())


# ---------------------------------------------------------------------------
# the verifier itself — a zero must be earned, not structural
# ---------------------------------------------------------------------------
def _pair():
    regen = [_record(label_fields=te.PHASE0_LABEL_FIELDS)]
    original = copy.deepcopy(regen)
    original[0]["deferred"] = False  # label fields are ignored
    original[0]["appeal_reason"] = "r"
    return regen, original


def test_verifier_reports_zero_on_identical_records(capsys):
    assert vp.compare(*_pair()) == 0


@pytest.mark.parametrize("mutate, field", [
    (lambda r: r.__setitem__("status", "open"), "status"),
    (lambda r: r["thread"][0].__setitem__("sender_type", "agent"), "thread.sender_type"),
    (lambda r: r["thread"][1].__setitem__("is_public", False), "thread.public"),
    (lambda r: r["thread"].reverse(), "thread order"),
    (lambda r: r["thread"].pop(), "thread count"),
    (lambda r: r.__setitem__("submission_numbers_mentioned", ["11111"]), "submission_numbers_mentioned"),
])
def test_verifier_catches_each_kind_of_drift(mutate, field, capsys):
    regen, original = _pair()
    mutate(original[0])
    assert vp.compare(regen, original) > 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.strip().startswith(field))
    assert " 0 mismatching" not in line


def test_verifier_refuses_to_write_the_phase0_file(tmp_path, monkeypatch):
    """Points PHASE0_PATH at a temp stand-in: aiming this test at the REAL file
    would truncate it the moment the guard broke."""
    stand_in = tmp_path / "phase1_appeals_stand_in.jsonl"
    stand_in.write_text('{"ticket_id": 1}\n', encoding="utf-8")
    monkeypatch.setattr(vp, "PHASE0_PATH", stand_in)
    with pytest.raises(SystemExit):
        vp.write_jsonl(stand_in, [])
    assert stand_in.read_text(encoding="utf-8") == '{"ticket_id": 1}\n'
