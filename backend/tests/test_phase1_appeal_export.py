"""Phase-1 appeal CSV export: columns, content, ordering, the route and the CLI.

All data is synthetic. The route and the CLI must produce the same file because
they share one builder; both are checked against it here.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import main
from app.core.config import settings
from app.db.database import get_db
from app.db.models import Base
from app.exports.phase1_appeals import EXPORT_COLUMNS, build_phase1_export_csv
from app.repositories.email_repository import EmailRepository
from app.repositories.phase1_appeal_repository import Phase1AppealRepository
from scripts import export_phase1_appeals as export_cli

REQUESTER = 9001
AGENT = 9002

EXPECTED_HEADER = [
    "ticket", "zendesk_link", "submission_number", "apc", "openreview_link",
    "relation", "appeal_reasons", "must_verify", "same_paper_tickets",
    "zendesk_status", "created_utc", "subject", "email_body",
]


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


def _row(number, *, relation="appeal", reasons=("wrong_paper_review",), must_verify=True,
         ticket=None, apc=None, url=None):
    return {
        "zendesk_ticket_id": ticket,
        "submission_number": number,
        "apc_name": apc,
        "openreview_url": url,
        "relation": relation,
        "reasons": [{"reason": r, "quote": "a synthetic quote"} for r in reasons],
        "must_verify": must_verify,
        "prompt_sha256": "a" * 64,
        "model": "synthetic-model",
    }


async def _email(db, *, ticket, subject="Synthetic subject", body="Stored body.",
                 requester=REQUESTER, created=None, status="open", messages=()):
    email = await EmailRepository().create_email(db, {
        "sender": "author@example.edu",
        "subject": subject,
        "body": body,
        "status": "draft_generated",
        "zendesk_ticket_id": ticket,
        "zendesk_requester_id": requester,
        "zendesk_status": status,
        "zendesk_created_at": created,
    })
    if messages:
        await EmailRepository().add_thread_messages(db, str(email.id), list(messages))
    return email


def _msg(comment_id, author, text, minute, public=True):
    return {
        "zendesk_comment_id": comment_id,
        "public": public,
        "author_id": author,
        "author_role": "end-user",
        "plain_body": text,
        "created_at": datetime(2026, 9, 10, 8, minute, tzinfo=timezone.utc),
    }


def _parse(text):
    return list(csv.DictReader(io.StringIO(text)))


async def _export(factory):
    async with factory() as db:
        return await build_phase1_export_csv(db)


async def test_header_is_exactly_the_specified_columns(factory):
    assert list(EXPORT_COLUMNS) == EXPECTED_HEADER
    text = await _export(factory)
    assert next(csv.reader(io.StringIO(text))) == EXPECTED_HEADER
    assert _parse(text) == []


async def test_row_content(factory, monkeypatch):
    monkeypatch.setattr(settings, "ZENDESK_SUBDOMAIN", "example")
    async with factory() as db:
        email = await _email(
            db, ticket=501, subject="Appeal for my paper",
            created=datetime(2026, 9, 5, 14, 30, tzinfo=timezone.utc), status="pending",
        )
        await Phase1AppealRepository().replace_for_email(db, email.id, [
            _row("12345", reasons=("wrong_paper_review", "decision_vs_reviews"),
                 apc="Synthetic APC", url="https://openreview.net/forum?id=Ab3xY9kLm2"),
        ])
    (row,) = _parse(await _export(factory))
    assert row == {
        "ticket": "501",
        "zendesk_link": "https://example.zendesk.com/agent/tickets/501",
        "submission_number": "12345",
        "apc": "Synthetic APC",
        "openreview_link": "https://openreview.net/forum?id=Ab3xY9kLm2",
        "relation": "appeal",
        "appeal_reasons": "wrong_paper_review; decision_vs_reviews",
        "must_verify": "true",
        "same_paper_tickets": "",
        "zendesk_status": "pending",
        "created_utc": "2026-09-05T14:30:00Z",
        "subject": "Appeal for my paper",
        "email_body": "Stored body.",
    }


async def test_zendesk_link_is_empty_without_a_subdomain(factory, monkeypatch):
    monkeypatch.setattr(settings, "ZENDESK_SUBDOMAIN", None)
    async with factory() as db:
        email = await _email(db, ticket=501)
        await Phase1AppealRepository().replace_for_email(db, email.id, [_row("12345")])
    (row,) = _parse(await _export(factory))
    assert row["zendesk_link"] == ""


async def test_null_fields_export_as_empty(factory):
    async with factory() as db:
        email = await _email(db, ticket=None, requester=None)
        await Phase1AppealRepository().replace_for_email(
            db, email.id, [_row(None, reasons=(), must_verify=False)]
        )
    (row,) = _parse(await _export(factory))
    assert (row["ticket"], row["submission_number"], row["apc"], row["openreview_link"]) == (
        "", "", "", ""
    )
    assert (row["appeal_reasons"], row["must_verify"]) == ("", "false")


async def test_email_body_is_the_requesters_public_messages_oldest_first(factory):
    async with factory() as db:
        email = await _email(db, ticket=501, messages=[
            _msg(2, REQUESTER, "Second requester message.", 20),
            _msg(1, REQUESTER, "First requester message.", 10),
            _msg(3, AGENT, "An agent reply.", 15),
            _msg(4, REQUESTER, "A note the requester cannot see.", 25, public=False),
            _msg(5, REQUESTER, "   ", 30),
        ])
        await Phase1AppealRepository().replace_for_email(db, email.id, [_row("12345")])
    (row,) = _parse(await _export(factory))
    assert row["email_body"] == "First requester message.\n\n---\n\nSecond requester message."


async def test_email_body_falls_back_to_the_stored_body(factory):
    async with factory() as db:
        email = await _email(db, ticket=501, body="Only the stored body.",
                             messages=[_msg(3, AGENT, "An agent reply.", 15)])
        await Phase1AppealRepository().replace_for_email(db, email.id, [_row("12345")])
    (row,) = _parse(await _export(factory))
    assert row["email_body"] == "Only the stored body."


async def test_sorted_by_ticket_then_submission_numerically_nulls_last(factory):
    async with factory() as db:
        e10 = await _email(db, ticket=10)
        e9 = await _email(db, ticket=9)
        e_none = await _email(db, ticket=None)
        e100 = await _email(db, ticket=100)
        await Phase1AppealRepository().replace_for_email(
            db, e10.id, [_row("1000"), _row(None), _row("999")]
        )
        await Phase1AppealRepository().replace_for_email(db, e9.id, [_row("5")])
        await Phase1AppealRepository().replace_for_email(db, e_none.id, [_row("1")])
        await Phase1AppealRepository().replace_for_email(db, e100.id, [_row("2")])
    rows = _parse(await _export(factory))
    assert [(r["ticket"], r["submission_number"]) for r in rows] == [
        ("9", "5"), ("10", "999"), ("10", "1000"), ("10", ""), ("100", "2"), ("", "1"),
    ]


async def test_same_paper_tickets_lists_the_other_tickets(factory):
    async with factory() as db:
        for ticket, numbers in [(30, ["12345"]), (4, ["12345", "67890"]), (200, ["12345"]),
                                (8, ["67890"]), (9, [None])]:
            email = await _email(db, ticket=ticket)
            await Phase1AppealRepository().replace_for_email(
                db, email.id, [_row(n) for n in numbers]
            )
    rows = {(r["ticket"], r["submission_number"]): r["same_paper_tickets"]
            for r in _parse(await _export(factory))}
    assert rows[("4", "12345")] == "30; 200"
    assert rows[("30", "12345")] == "4; 200"
    assert rows[("4", "67890")] == "8"
    assert rows[("9", "")] == ""


async def test_the_ticket_comes_from_the_email_when_the_row_lacks_it(factory):
    async with factory() as db:
        email = await _email(db, ticket=777)
        await Phase1AppealRepository().replace_for_email(db, email.id, [_row("1", ticket=None)])
    (row,) = _parse(await _export(factory))
    assert row["ticket"] == "777"


async def test_the_route_serves_the_builders_csv(factory):
    async with factory() as db:
        email = await _email(db, ticket=501, body="Line one.\nLine two, with a comma.")
        await Phase1AppealRepository().replace_for_email(db, email.id, [_row("12345")])

    async def _override_get_db():
        async with factory() as session:
            yield session

    main.app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/appeals/phase1/export.csv")
    finally:
        main.app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.text == await _export(factory)
    (row,) = _parse(resp.text)
    assert row["email_body"] == "Line one.\nLine two, with a comma."


async def test_the_cli_writes_the_builders_csv(factory, tmp_path):
    async with factory() as db:
        email = await _email(db, ticket=501)
        await Phase1AppealRepository().replace_for_email(db, email.id, [_row("12345")])
    text = await export_cli.export(factory)
    assert text == await _export(factory)


@pytest.mark.parametrize("use_out", [True, False])
def test_cli_main_writes_to_out_or_stdout(monkeypatch, tmp_path, capsys, use_out):
    async def fake_export(session_factory):
        return "ticket\r\n1\r\n"

    monkeypatch.setattr(export_cli, "export", fake_export)
    out = tmp_path / "export.csv"
    assert export_cli.main(["--out", str(out)] if use_out else []) == 0
    if use_out:
        assert out.read_bytes() == b"ticket\r\n1\r\n"
    else:
        assert capsys.readouterr().out == "ticket\r\n1\r\n"
