"""Reject Appeals queue, read side (Z3a): /appeals/config, /queue, /queue/counts, /apcs.

A throwaway in-memory SQLite database (and Postgres too when TEST_DATABASE_URL
names a disposable database) seeded with made-up rows. Nothing here posts to
Zendesk: conftest makes ZendeskSender.add_comment raise. Chair names are made
up ("APC North", "APC South").

Seeded view members, newest first: e8, e7, e5, e4, e3, e2, e1.
e6 (not an appeal, no note) and e9 (toy) are NOT members.

Phase-1 reasons (P4): e1 has stored phase1_appeals rows AND a draft snapshot that
disagree (the rows must win); e4 has only a snapshot (the fallback); e8 has rows
for two papers that the extraction never names (the classifier_papers_differ
warning); every other member has neither (phase1 is null). e1's extraction still
carries an old appeal_reason that must never be read.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import main
from app.core.config import settings
from app.db.database import Base, get_db
from app.db.models import Email, PaperAssignment, Phase1Appeal, ZendeskChairNote
from app.models import appeal_queue

T0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
NAMES = ("APC North", "APC South", "apc lowercase")
# A synthetic author quote longer than the 240-character cap (264 characters).
LONG_QUOTE = "synthetic quote words " * 12
assert len(LONG_QUOTE) == 264
# Its first 240 characters, written out: ten full repeats (220) + 20 more.
TRUNCATED_QUOTE = "synthetic quote words " * 10 + "synthetic quote word"


def _snapshot(relation, reasons, papers, must_verify=False, state="classified"):
    return {"state": state, "relation": relation, "reasons": reasons,
            "must_verify": must_verify, "papers": papers}

_PG = os.environ.get("TEST_DATABASE_URL", "")
_PG = _PG if _PG.startswith("postgresql") else None


def _at(hours: int) -> datetime:
    return T0 + timedelta(hours=hours)


def _email(n: int, **kw) -> Email:
    base = dict(
        sender=f"author{n}@example.org",
        sender_name=f"Author {n}",
        subject=f"subject {n}",
        body="body",
        status="DRAFT_GENERATED",
        routing={"lane": "human_review"},
        received_at=_at(n),
        created_at=_at(n),
        updated_at=_at(n),
    )
    base.update(kw)
    return Email(**base)


def _draft(mode, text="Dear Author,\n\nReply.\n\nBest Regards,\nAAAI 2027 PC Team", reply=None,
           **extra):
    draft = {"draft_text": text, "notes_for_chair": None, **extra}
    if mode is not None:
        draft["appeal_reply"] = {"mode": mode, "reasons": [], "block_ids": [], **(reply or {})}
    return draft


def seed() -> list:
    rows = [
        _email(1, subject="appeal one", classification={"intent": "review_decision_appeal"},
               draft=_draft("merged", reply={
                   "reasons": ["score_outcome_mismatch"], "source": "phase1",
                   # Deliberately NOT what the stored rows say: the rows must win.
                   "phase1": _snapshot("appeal", ["reviewer_misjudgment"], ["99999"]),
               }),
               extraction={"submission_numbers": ["12345"], "openreview_forum_ids": ["Ab3xY9kLm2"],
                           # An old answer of the dormant classifier: never read any more.
                           "appeal_reason": ["wrong_paper_review"], "is_reciprocal_dispute": False},
               source="zendesk", zendesk_ticket_id=9001, zendesk_status="open"),
        _email(2, subject="desk two", classification={"intent": "desk_reject_appeal"},
               draft={"draft_text": "A model draft.", "notes_for_chair": None},
               extraction={"submission_number": "24680", "openreview_forum_id": None},
               source="zendesk", zendesk_ticket_id=9002, zendesk_status="new"),
        _email(3, subject="reciprocal three", classification={"intent": "desk_reject_appeal"},
               draft=_draft("reciprocal_review", "[CHAIR: reciprocal complaint; see note]"),
               extraction={"submission_numbers": ["777"], "is_reciprocal_dispute": True},
               source="zendesk", zendesk_ticket_id=9003, zendesk_status="pending"),
        _email(4, subject="chair writes four", classification={"intent": "review_decision_appeal"},
               draft=_draft("chair_writes", "[CHAIR: write reply]", reply={
                   "reasons": None, "source": "phase1",
                   # Only a snapshot (no stored rows): the fallback.
                   "phase1": _snapshot("feedback_only", ["reviewer_misjudgment"], ["99999"]),
               }),
               extraction={"submission_numbers": ["67890"], "appeal_reason": []},
               source="zendesk", zendesk_ticket_id=9004, zendesk_status="solved"),
        _email(5, subject="cms five", classification={"intent": "cms_support"},
               source="zendesk", zendesk_ticket_id=9005, zendesk_status="open"),
        _email(6, subject="cms six", classification={"intent": "cms_support"},
               source="zendesk", zendesk_ticket_id=9006, zendesk_status="open"),
        _email(7, subject="investigate seven", status="approved",
               classification={"intent": "review_decision_appeal"},
               draft=_draft("no_draft", "[CHAIR: do not reply yet; see note]", is_edited=True),
               extraction={"submission_numbers": ["8"]},
               source="zendesk", zendesk_ticket_id=9007, zendesk_status="hold"),
        _email(8, subject="future eight", classification={"intent": "review_decision_appeal"},
               draft=_draft("a_mode_added_later"), extraction={},
               source="toy_dataset"),
        _email(9, subject="toy nine", classification={"intent": "submission_requirements"},
               source="toy_dataset"),
    ]
    sheet = [
        PaperAssignment(paper_number="12345", apc_name="APC North",
                        openreview_url="u", openreview_forum_id="Ab3xY9kLm2"),
        PaperAssignment(paper_number="24680", apc_name="APC North",
                        openreview_url="u", openreview_forum_id="Qq1wW2eE3r"),
        PaperAssignment(paper_number="67890", apc_name="APC South",
                        openreview_url="u", openreview_forum_id="Zz9yY8xX77"),
        PaperAssignment(paper_number="777", apc_name="APC South",
                        openreview_url="u", openreview_forum_id="Sv7nUmb3r0"),
        PaperAssignment(paper_number="8", apc_name=" apc lowercase ",
                        openreview_url="u", openreview_forum_id="Sh0rtNumb8"),
        PaperAssignment(paper_number="55555", apc_name="   ",
                        openreview_url="u", openreview_forum_id="Bl4nkApc00"),
    ]
    return rows, sheet


async def _make_ctx(kind: str):
    if kind == "postgres":
        url = _PG if "+asyncpg" in _PG else _PG.replace("postgresql://", "postgresql+asyncpg://", 1)
        engine = create_async_engine(url)
    else:
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    rows, sheet = seed()
    async with factory() as session:
        session.add_all(rows + sheet)
        await session.commit()
        ids = {e.subject.split()[-1]: e.id for e in rows}
        session.add_all([
            ZendeskChairNote(email_id=ids["four"], zendesk_ticket_id=9004, status="posted",
                             attempts=1, posted_at=_at(20), mode="chair_writes"),
            ZendeskChairNote(email_id=ids["five"], zendesk_ticket_id=9005, status="failed",
                             attempts=2),
        ])
        session.add_all([
            # e1: one row, a long quote (cut to 240 characters in the response).
            Phase1Appeal(email_id=ids["one"], zendesk_ticket_id=9001, submission_number="12345",
                         relation="appeal", must_verify=False, prompt_sha256="0" * 64,
                         reasons=[{"reason": "decision_vs_reviews", "quote": LONG_QUOTE}]),
            # e8: two papers, inserted out of order; the extraction names neither.
            Phase1Appeal(email_id=ids["eight"], submission_number="24680", relation="appeal",
                         must_verify=True, prompt_sha256="0" * 64,
                         reasons=[{"reason": "record_error", "quote": "short quote"}]),
            Phase1Appeal(email_id=ids["eight"], submission_number="12345", relation="appeal",
                         must_verify=True, prompt_sha256="0" * 64,
                         reasons=[{"reason": "record_error", "quote": "short quote"}]),
        ])
        await session.commit()

    async def _override_get_db():
        async with factory() as session:
            yield session

    main.app.dependency_overrides[get_db] = _override_get_db
    client = httpx.AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test")
    return SimpleNamespace(client=client, engine=engine, ids=ids, kind=kind)


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def ctx(request, monkeypatch):
    if request.param == "postgres" and not _PG:
        pytest.skip("set TEST_DATABASE_URL to a disposable postgresql:// database")
    monkeypatch.setattr(settings, "REJECT_APPEALS_QUEUE_ENABLED", True)
    monkeypatch.setattr(settings, "ZENDESK_SUBDOMAIN", "example")
    c = await _make_ctx(request.param)
    yield c
    await c.client.aclose()
    main.app.dependency_overrides.clear()
    if request.param == "postgres":
        async with c.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    await c.engine.dispose()


async def _subjects(ctx, query: str = "") -> list[str]:
    resp = await ctx.client.get(f"/api/v1/appeals/queue{query}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    subjects = [e["subject"].split()[-1] for e in body["emails"]]
    assert body["total"] == len(subjects) or "limit=" in query
    return subjects


ALL_MEMBERS = ["eight", "seven", "five", "four", "three", "two", "one"]


# --- membership and ordering --------------------------------------------------------


async def test_members_are_appeal_intents_or_emails_with_a_note_newest_first(ctx):
    assert await _subjects(ctx) == ALL_MEMBERS


async def test_pages_add_up_to_the_whole_view(ctx):
    seen = []
    for offset in (0, 3, 6):
        resp = await ctx.client.get(f"/api/v1/appeals/queue?limit=3&offset={offset}")
        body = resp.json()
        assert body["total"] == 7
        seen += [e["subject"].split()[-1] for e in body["emails"]]
    assert seen == ALL_MEMBERS


@pytest.mark.parametrize("limit", ["0", "201"])
async def test_limit_bounds(ctx, limit):
    assert (await ctx.client.get(f"/api/v1/appeals/queue?limit={limit}")).status_code == 422


# --- filters --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group, expected",
    [
        ("composed", ["one"]),
        ("chair_writes", ["eight", "four"]),
        ("investigate", ["seven"]),
        ("reciprocal", ["three"]),
        ("not_drafted", ["five", "two"]),
    ],
)
async def test_mode_group_filter(ctx, group, expected):
    assert await _subjects(ctx, f"?mode_group={group}") == expected


async def test_unknown_mode_group_is_rejected(ctx):
    assert (await ctx.client.get("/api/v1/appeals/queue?mode_group=bogus")).status_code == 422


@pytest.mark.parametrize(
    "value, expected",
    [("true", ["three"]), ("false", ["eight", "seven", "five", "four", "two", "one"])],
)
async def test_reciprocal_filter(ctx, value, expected):
    assert await _subjects(ctx, f"?reciprocal={value}") == expected


@pytest.mark.parametrize(
    "state, expected",
    [
        ("none", ["eight", "seven", "three", "two", "one"]),
        ("posted", ["four"]),
        ("failed", ["five"]),
        ("pending", []),
        ("posting", []),
    ],
)
async def test_note_status_filter(ctx, state, expected):
    assert await _subjects(ctx, f"?note_status={state}") == expected


async def test_email_status_filter(ctx):
    assert await _subjects(ctx, "?status=approved") == ["seven"]


async def test_zendesk_status_filter(ctx):
    assert await _subjects(ctx, "?zendesk_status=open") == ["five", "one"]
    # "solved" is the main queue's combined solved+closed bucket, reused as is.
    assert await _subjects(ctx, "?zendesk_status=solved") == ["four"]


async def test_received_range_filter(ctx):
    assert await _subjects(ctx, "?received_after=2026-09-10T15:00:00Z") == [
        "eight", "seven", "five", "four", "three",
    ]
    assert await _subjects(ctx, "?received_before=2026-09-10T13:00:00Z") == ["one"]


async def test_search_filter(ctx):
    assert await _subjects(ctx, "?search=%239002") == ["two"]
    assert await _subjects(ctx, "?search=reciprocal") == ["three"]


async def test_filters_compose(ctx):
    assert await _subjects(ctx, "?zendesk_status=open&note_status=none") == ["one"]


# --- row fields ---------------------------------------------------------------------------


async def _row(ctx, name):
    body = (await ctx.client.get("/api/v1/appeals/queue?limit=200")).json()
    return next(e for e in body["emails"] if e["subject"].endswith(name))


async def test_row_keeps_the_email_payload_and_adds_the_appeal_fields(ctx):
    row = await _row(ctx, "one")
    assert row["zendesk_ticket_url"] == "https://example.zendesk.com/agent/tickets/9001"
    assert {k: row[k] for k in (
        "appeal", "suggested_apcs", "chair_source", "chair_numbers", "chair_warnings",
        "chair_note", "eligibility",
    )} == {
        "appeal": {
            "intent": "review_decision_appeal",
            # From draft.appeal_reply.reasons — NOT extraction.appeal_reason.
            "reasons": ["score_outcome_mismatch"],
            "is_reciprocal_dispute": False,
            "mode": "merged",
            "mode_group": "composed",
            "has_placeholders": False,
            "is_edited": False,
            "submission_numbers": ["12345"],
            # From the stored rows — NOT the (different) draft snapshot.
            "phase1": {
                "source": "rows",
                "relation": "appeal",
                "must_verify": False,
                "papers": ["12345"],
                "reasons": [{"reason": "decision_vs_reviews", "quote": TRUNCATED_QUOTE}],
            },
        },
        "suggested_apcs": ["APC North"],
        "chair_source": "both",
        "chair_numbers": ["12345"],
        "chair_warnings": [],
        "chair_note": None,
        "eligibility": {"eligible": False, "reason": "flag_off"},
    }


async def test_old_shape_desk_reject_row_is_refused_by_the_guard(ctx):
    row = await _row(ctx, "two")
    assert row["appeal"] == {
        "intent": "desk_reject_appeal",
        "reasons": None,
        "is_reciprocal_dispute": None,
        "mode": None,
        "mode_group": "not_drafted",
        "has_placeholders": False,
        "is_edited": False,
        "submission_numbers": ["24680"],
        "phase1": None,
    }
    assert (row["suggested_apcs"], row["chair_source"], row["chair_numbers"], row["chair_warnings"]) == (
        [], "none", [], ["desk_reject_not_in_sheet"],
    )


async def test_posted_note_and_short_number_rows(ctx):
    four = await _row(ctx, "four")
    assert four["chair_note"]["status"] == "posted"
    assert four["chair_note"]["attempts"] == 1
    assert four["chair_note"]["posted_at"].startswith("2026-09-11T08:00:00")
    # A hold records no composer input: reasons None (read from the draft).
    assert (four["appeal"]["reasons"], four["chair_source"], four["suggested_apcs"]) == (
        None, "number", ["APC South"],
    )
    seven = await _row(ctx, "seven")
    assert (seven["appeal"]["has_placeholders"], seven["appeal"]["is_edited"]) == (True, True)
    assert (seven["chair_source"], seven["suggested_apcs"], seven["chair_warnings"]) == (
        "number", ["apc lowercase"], ["short_number"],
    )


async def test_note_only_member_has_no_appeal_mode(ctx):
    five = await _row(ctx, "five")
    assert five["appeal"]["intent"] == "cms_support"
    assert five["appeal"]["mode_group"] == "not_drafted"
    assert five["chair_note"] == {"status": "failed", "attempts": 2, "posted_at": None}


# --- phase-1 reasons (P4) ---------------------------------------------------------------------


async def test_stored_rows_win_over_the_draft_snapshot(ctx):
    phase1 = (await _row(ctx, "one"))["appeal"]["phase1"]
    assert phase1["source"] == "rows"
    assert [r["reason"] for r in phase1["reasons"]] == ["decision_vs_reviews"]
    assert phase1["papers"] == ["12345"], "not the snapshot's 99999"


async def test_without_rows_the_names_only_snapshot_is_used(ctx):
    four = await _row(ctx, "four")
    assert four["appeal"]["phase1"] == {
        "source": "snapshot",
        "relation": "feedback_only",
        "must_verify": False,
        "papers": ["99999"],
        "reasons": [{"reason": "reviewer_misjudgment", "quote": None}],
    }


@pytest.mark.parametrize("name", ["two", "three", "five", "seven"])
async def test_without_rows_or_snapshot_phase1_is_null(ctx, name):
    assert (await _row(ctx, name))["appeal"]["phase1"] is None


async def test_rows_for_several_papers_are_ordered_and_keep_must_verify(ctx):
    eight = await _row(ctx, "eight")
    assert eight["appeal"]["phase1"] == {
        "source": "rows",
        "relation": "appeal",
        "must_verify": True,
        "papers": ["12345", "24680"],
        "reasons": [{"reason": "record_error", "quote": "short quote"}],
    }


async def test_quotes_are_cut_to_240_characters(ctx):
    quote = (await _row(ctx, "one"))["appeal"]["phase1"]["reasons"][0]["quote"]
    assert len(quote) == 240
    assert quote == TRUNCATED_QUOTE


async def test_appeal_reasons_come_from_the_draft_never_from_the_extraction(ctx):
    one = await _row(ctx, "one")
    assert one["appeal"]["reasons"] == ["score_outcome_mismatch"]
    assert one["extraction"]["appeal_reason"] == ["wrong_paper_review"], "still stored, ignored"
    assert (await _row(ctx, "two"))["appeal"]["reasons"] is None, "no appeal_reply at all"
    assert (await _row(ctx, "seven"))["appeal"]["reasons"] == [], "an empty composer input"


async def test_the_phase1_rows_of_a_whole_page_are_one_query(ctx):
    from sqlalchemy import event

    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if "phase1_appeals" in statement.lower():
            statements.append(statement)

    event.listen(ctx.engine.sync_engine, "before_cursor_execute", record)
    try:
        resp = await ctx.client.get("/api/v1/appeals/queue?limit=200")
    finally:
        event.remove(ctx.engine.sync_engine, "before_cursor_execute", record)
    assert resp.status_code == 200
    assert len(statements) == 1, statements


async def test_classifier_papers_differ_is_a_warning_only(ctx):
    # e4: snapshot papers 99999, the extraction matched 67890 -> warning, chair unchanged.
    four = await _row(ctx, "four")
    assert (four["suggested_apcs"], four["chair_source"], four["chair_numbers"], four["chair_warnings"]) == (
        ["APC South"], "number", ["67890"], ["classifier_papers_differ"],
    )
    # e8: rows name 12345/24680, the extraction names nothing -> warning, still no chair.
    eight = await _row(ctx, "eight")
    assert (eight["suggested_apcs"], eight["chair_source"], eight["chair_numbers"], eight["chair_warnings"]) == (
        [], "none", [], ["classifier_papers_differ"],
    )
    # e1: rows name 12345, the extraction matched 12345 -> no warning.
    assert (await _row(ctx, "one"))["chair_warnings"] == []
    # e7: no phase-1 papers at all -> nothing to compare, the existing warning only.
    assert (await _row(ctx, "seven"))["chair_warnings"] == ["short_number"]


async def test_quotes_never_reach_the_logs(ctx, caplog):
    caplog.set_level(logging.DEBUG)
    assert (await ctx.client.get("/api/v1/appeals/queue?limit=200")).status_code == 200
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "synthetic quote" not in logged and "short quote" not in logged


def test_phase1_block_rules_without_a_database():
    """The pure builder: rows first, then the snapshot, else None; junk-safe."""
    row = SimpleNamespace(relation="appeal", must_verify=0, submission_number="7",
                          reasons=[{"reason": "other", "quote": 5}, "junk", {"quote": "no name"}])
    snap_draft = {"appeal_reply": {"source": "phase1",
                                   "phase1": _snapshot("not_appeal", None, None, None)}}
    assert appeal_queue.phase1_block(snap_draft, [row]) == {
        "source": "rows", "relation": "appeal", "must_verify": False, "papers": ["7"],
        "reasons": [{"reason": "other", "quote": None}],
    }
    assert appeal_queue.phase1_block(snap_draft, []) == {
        "source": "snapshot", "relation": "not_appeal", "must_verify": None, "papers": [],
        "reasons": [],
    }
    for draft in (None, {}, {"appeal_reply": {"mode": "merged"}},
                  {"appeal_reply": {"source": "appeal_reason", "phase1": {}}},
                  {"appeal_reply": {"source": "phase1", "phase1": "junk"}}):
        assert appeal_queue.phase1_block(draft, []) is None, draft


async def test_eligibility_uses_the_z2a_check(ctx, monkeypatch):
    monkeypatch.setattr(settings, "CHAIR_NOTE_ENABLED", True)
    monkeypatch.setattr(settings, "CHAIR_NOTE_INTENTS", "review_decision_appeal,desk_reject_appeal")
    monkeypatch.setattr(settings, "CHAIR_NOTE_TICKET_IDS", "")
    assert (await _row(ctx, "one"))["eligibility"] == {"eligible": True, "reason": "eligible"}
    assert (await _row(ctx, "four"))["eligibility"] == {
        "eligible": False, "reason": "ticket_status_not_allowed",
    }
    assert (await _row(ctx, "two"))["eligibility"] == {"eligible": False, "reason": "no_appeal_reply"}


# --- counts ---------------------------------------------------------------------------------


async def test_counts(ctx):
    resp = await ctx.client.get("/api/v1/appeals/queue/counts")
    assert resp.status_code == 200
    assert resp.json() == {
        "needs_note": 3,
        "total": 7,
        "by_mode_group": {
            "composed": 1, "chair_writes": 2, "investigate": 1, "reciprocal": 1, "not_drafted": 2,
        },
        "by_note_status": {"none": 5, "pending": 0, "posting": 0, "posted": 1, "failed": 1},
        "without_approved_draft": 2,
    }


async def test_needs_note_excludes_posted_resolved_and_ticketless(ctx):
    """Only drafted appeals on an open ticket whose note is not posted need one."""
    factory = async_sessionmaker(ctx.engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        extra = [
            _email(30, subject="open posted", classification={"intent": "review_decision_appeal"},
                   draft=_draft("merged"), source="zendesk", zendesk_ticket_id=9030,
                   zendesk_status="open"),
            _email(31, subject="open none", classification={"intent": "review_decision_appeal"},
                   draft=_draft("merged"), source="zendesk", zendesk_ticket_id=9031,
                   zendesk_status="open"),
            _email(32, subject="closed none", classification={"intent": "review_decision_appeal"},
                   draft=_draft("merged"), source="zendesk", zendesk_ticket_id=9032,
                   zendesk_status="closed"),
            _email(33, subject="open pending", classification={"intent": "review_decision_appeal"},
                   draft=_draft("merged"), source="zendesk", zendesk_ticket_id=9033,
                   zendesk_status="open"),
            _email(34, subject="open posting", classification={"intent": "review_decision_appeal"},
                   draft=_draft("merged"), source="zendesk", zendesk_ticket_id=9034,
                   zendesk_status="open"),
            _email(35, subject="open failed", classification={"intent": "review_decision_appeal"},
                   draft=_draft("merged"), source="zendesk", zendesk_ticket_id=9035,
                   zendesk_status="open"),
        ]
        session.add_all(extra)
        await session.commit()
        session.add_all([
            ZendeskChairNote(email_id=extra[0].id, zendesk_ticket_id=9030, status="posted", attempts=1),
            ZendeskChairNote(email_id=extra[3].id, zendesk_ticket_id=9033, status="pending"),
            ZendeskChairNote(email_id=extra[4].id, zendesk_ticket_id=9034, status="posting", attempts=1),
            ZendeskChairNote(email_id=extra[5].id, zendesk_ticket_id=9035, status="failed", attempts=1),
        ])
        await session.commit()
    counts = (await ctx.client.get("/api/v1/appeals/queue/counts")).json()
    # The seeded 3, plus "open none", "open pending" and "open failed" (a retry).
    # NOT "open posted", NOT "open posting" (in flight: Zendesk may have it), NOT closed.
    assert (counts["needs_note"], counts["total"]) == (6, 13)


# --- the chair dropdown ------------------------------------------------------------------------


async def test_apcs_are_distinct_trimmed_sorted(ctx):
    resp = await ctx.client.get("/api/v1/appeals/apcs")
    assert resp.json() == {"apcs": ["apc lowercase", "APC North", "APC South"]}


# --- the flag -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["/api/v1/appeals/queue", "/api/v1/appeals/queue/counts", "/api/v1/appeals/apcs"]
)
async def test_endpoints_answer_404_when_the_flag_is_off(ctx, monkeypatch, path):
    monkeypatch.setattr(settings, "REJECT_APPEALS_QUEUE_ENABLED", False)
    resp = await ctx.client.get(path)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Reject Appeals queue is turned off"}


@pytest.mark.parametrize("enabled", [True, False])
async def test_config_always_answers(ctx, monkeypatch, enabled):
    monkeypatch.setattr(settings, "REJECT_APPEALS_QUEUE_ENABLED", enabled)
    resp = await ctx.client.get("/api/v1/appeals/config")
    assert (resp.status_code, resp.json()) == (200, {"enabled": enabled})


def test_flag_defaults_off():
    from app.core.config import Settings

    assert Settings.model_fields["REJECT_APPEALS_QUEUE_ENABLED"].default is False


# --- chair names never reach logs or errors ------------------------------------------------------


async def test_chair_names_never_appear_in_logs_or_errors(ctx, caplog):
    caplog.set_level(logging.DEBUG)
    bodies = []
    for path in (
        "/api/v1/appeals/queue?limit=200",
        "/api/v1/appeals/queue/counts",
        "/api/v1/appeals/apcs",
        "/api/v1/appeals/queue?mode_group=bogus",
        "/api/v1/appeals/queue?received_after=not-a-date",
    ):
        resp = await ctx.client.get(path)
        if resp.status_code >= 400:
            bodies.append(resp.text)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for name in ("APC North", "APC South", "apc lowercase"):
        assert name not in logged
        assert all(name not in body for body in bodies)


# --- SQL and Python agree on mode groups -----------------------------------------------------------


def test_python_mode_groups():
    expected = {
        "merged": "composed", "standalone": "composed", "no_draft": "investigate",
        "reciprocal_review": "reciprocal", "chair_writes": "chair_writes", "refused": "chair_writes",
        "reason_unknown": "chair_writes", "desk_reject": "chair_writes", "window": "chair_writes",
        "failed": "chair_writes", "a_mode_added_later": "chair_writes", "": "chair_writes",
        None: "not_drafted",
    }
    assert {m: appeal_queue.mode_group(m) for m in expected} == expected


def test_composed_modes_match_the_hook():
    from app.pipeline.appeal_reply_hook import COMPOSED_MODES

    assert set(appeal_queue.COMPOSED_MODES) == set(COMPOSED_MODES)


async def test_sql_mode_groups_match_python_for_every_mode(ctx):
    """Every appeal mode, filtered in SQL, lands in the group Python computes."""
    from sqlalchemy import select as sa_select

    from app.repositories.email_repository import _mode_group_condition

    modes = ["merged", "standalone", "no_draft", "reciprocal_review", "chair_writes", "refused",
             "reason_unknown", "desk_reject", "window", "failed", "a_mode_added_later", ""]
    drafts = [_draft(m) for m in modes] + [
        {"draft_text": "x"}, None, {"appeal_reply": None}, {"appeal_reply": {"mode": None}},
    ]
    factory = async_sessionmaker(ctx.engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        new = [_email(100 + i, subject=f"mode probe {i}", draft=d,
                      classification={"intent": "review_decision_appeal"})
               for i, d in enumerate(drafts)]
        session.add_all(new)
        await session.commit()
        for group in appeal_queue.MODE_GROUPS:
            result = await session.execute(
                sa_select(Email.id).where(
                    Email.id.in_([e.id for e in new]), _mode_group_condition(group)
                )
            )
            in_sql = {row[0] for row in result.all()}
            in_python = {
                e.id for e in new
                if appeal_queue.mode_group(appeal_queue.appeal_mode(e.draft)) == group
            }
            assert in_sql == in_python, group
