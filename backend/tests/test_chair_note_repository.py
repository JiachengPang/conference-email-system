"""ChairNoteRepository's claim protocol and the zendesk_chair_notes migration (Z2a).

Repository tests run on in-memory SQLite with ``PRAGMA foreign_keys=ON`` so the
ON DELETE CASCADE from ``emails`` fires. The migration test uses a throwaway
SQLite file, never a real database. All data is synthetic; nothing touches
Zendesk.
"""

import asyncio
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import delete, event, insert, inspect
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Email, ZendeskChairNote
from app.repositories.chair_note_repository import (
    FAILED,
    MAX_ATTEMPTS,
    PENDING,
    POSTED,
    POSTING,
    ChairNoteRepository,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REVISION = "8d2f6c1a9b3e"
_PREV_REVISION = "4faaa7e50e0a"

T0 = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
repo = ChairNoteRepository()


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _record):  # pragma: no cover - trivial
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


@pytest_asyncio.fixture
async def session(factory):
    async with factory() as s:
        yield s


async def _email(session, ticket: int = 21567) -> int:
    email = Email(sender="a@example.org", subject="s", body="b", zendesk_ticket_id=ticket)
    session.add(email)
    await session.commit()
    return email.id


async def _row(session, email_id):
    session.expire_all()
    return await repo.get_by_email_id(session, email_id)


# --- enqueue ------------------------------------------------------------------


async def test_enqueue_creates_one_pending_row(session):
    email_id = await _email(session)
    assert await repo.enqueue(session, email_id, 21567, mode="merged") is True
    row = await _row(session, email_id)
    assert (row.status, row.attempts, row.zendesk_ticket_id, row.mode) == (
        PENDING, 0, 21567, "merged"
    )
    assert row.claimed_at is None and row.posted_at is None


async def test_second_enqueue_is_a_no_op_whatever_the_state(session):
    email_id = await _email(session)
    assert await repo.enqueue(session, email_id, 21567) is True
    assert await repo.claim(session, email_id, now=T0) is True
    assert await repo.mark_posted(session, email_id, mode="merged", body_sha256="a" * 64)

    assert await repo.enqueue(session, email_id, 21567, mode="standalone") is False
    row = await _row(session, email_id)
    assert (row.status, row.attempts, row.mode) == (POSTED, 1, "merged")


# --- claim ----------------------------------------------------------------------


async def test_claim_moves_pending_to_posting_and_counts_the_attempt(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    assert await repo.claim(session, email_id, now=T0) is True
    row = await _row(session, email_id)
    assert (row.status, row.attempts) == (POSTING, 1)
    assert row.claimed_at.replace(tzinfo=timezone.utc) == T0


async def test_second_claim_loses(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    assert await repo.claim(session, email_id, now=T0) is True
    assert await repo.claim(session, email_id, now=T0) is False
    assert (await _row(session, email_id)).attempts == 1


async def test_two_concurrent_claims_exactly_one_wins(factory):
    async with factory() as s:
        email_id = await _email(s)
        await repo.enqueue(s, email_id, 21567)

    async def attempt():
        async with factory() as s:
            return await repo.claim(s, email_id, now=T0)

    results = await asyncio.gather(attempt(), attempt())
    assert sorted(results) == [False, True]


async def test_claim_without_a_row_is_false(session):
    email_id = await _email(session)
    assert await repo.claim(session, email_id) is False


async def test_a_posted_note_is_never_claimed_again(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    await repo.claim(session, email_id, now=T0)
    await repo.mark_posted(session, email_id, mode="merged", body_sha256="b" * 64,
                           zendesk_audit_id=987654321, now=T0)
    assert await repo.claim(session, email_id) is False
    row = await _row(session, email_id)
    assert (row.status, row.body_sha256, row.zendesk_audit_id) == (POSTED, "b" * 64, 987654321)
    assert row.posted_at.replace(tzinfo=timezone.utc) == T0


async def test_failures_are_retried_until_three_attempts_then_never(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    for attempt in (1, 2, 3):
        assert await repo.claim(session, email_id, now=T0) is True
        assert await repo.mark_failed(session, email_id, error=f"ZendeskSendError HTTP 503 #{attempt}")
        row = await _row(session, email_id)
        assert (row.status, row.attempts) == (FAILED, attempt)
    assert await repo.claim(session, email_id, now=T0) is False
    row = await _row(session, email_id)
    assert (row.status, row.attempts, row.last_error) == (FAILED, MAX_ATTEMPTS, "ZendeskSendError HTTP 503 #3")


async def test_a_failure_then_success_ends_posted(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    await repo.claim(session, email_id, now=T0)
    await repo.mark_failed(session, email_id, error="ZendeskSendError HTTP 502")
    assert await repo.claim(session, email_id, now=T0) is True
    assert await repo.mark_posted(session, email_id, mode="no_draft", body_sha256="c" * 64)
    row = await _row(session, email_id)
    assert (row.status, row.attempts, row.last_error) == (POSTED, 2, None)


# --- mark_posted / mark_failed ----------------------------------------------------


async def test_marks_only_change_a_posting_row(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    assert await repo.mark_posted(session, email_id, mode="merged", body_sha256="d" * 64) is False
    assert await repo.mark_failed(session, email_id, error="x") is False
    assert (await _row(session, email_id)).status == PENDING


async def test_stored_error_is_cut_to_500_characters(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    await repo.claim(session, email_id, now=T0)
    await repo.mark_failed(session, email_id, error="E" * 900)
    assert (await _row(session, email_id)).last_error == "E" * 500


# --- stale posting rows ------------------------------------------------------------


async def test_stale_posting_rows_are_listed_but_never_claimed(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    await repo.claim(session, email_id, now=T0)

    fresh = await repo.find_stale_posting(session, older_than=timedelta(minutes=15),
                                          now=T0 + timedelta(minutes=10))
    assert fresh == []
    stale = await repo.find_stale_posting(session, older_than=timedelta(minutes=15),
                                          now=T0 + timedelta(minutes=20))
    assert [r.email_id for r in stale] == [email_id]

    # Even a long-stale posting row stays where it is.
    assert await repo.claim(session, email_id, now=T0 + timedelta(days=2)) is False
    row = await _row(session, email_id)
    assert (row.status, row.attempts) == (POSTING, 1)


async def test_only_posting_rows_are_ever_stale(session):
    ids = [await _email(session, ticket=t) for t in (1, 2, 3)]
    for email_id, ticket in zip(ids, (1, 2, 3)):
        await repo.enqueue(session, email_id, ticket)
    await repo.claim(session, ids[1], now=T0)
    await repo.mark_posted(session, ids[1], mode="merged", body_sha256="e" * 64)
    await repo.claim(session, ids[2], now=T0)
    await repo.mark_failed(session, ids[2], error="x")
    assert await repo.find_stale_posting(session, older_than=timedelta(0),
                                         now=T0 + timedelta(days=1)) == []


# --- table constraints ---------------------------------------------------------------


async def test_one_row_per_email_is_enforced_by_the_table(session):
    email_id = await _email(session)
    session.add(ZendeskChairNote(email_id=email_id, zendesk_ticket_id=1))
    await session.commit()
    session.add(ZendeskChairNote(email_id=email_id, zendesk_ticket_id=1))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


@pytest.mark.parametrize("values", [{"status": "sent"}, {"attempts": 4}, {"attempts": -1}])
async def test_check_constraints_reject_bad_state(session, values):
    email_id = await _email(session)
    with pytest.raises(IntegrityError):
        await session.execute(
            insert(ZendeskChairNote).values(email_id=email_id, zendesk_ticket_id=1, **values)
        )
    await session.rollback()


async def test_rows_die_with_their_email(session):
    email_id = await _email(session)
    await repo.enqueue(session, email_id, 21567)
    await session.execute(delete(Email).where(Email.id == email_id))
    await session.commit()
    assert await _row(session, email_id) is None


# --- portability ---------------------------------------------------------------------


@pytest.mark.parametrize("dialect", [postgresql.dialect(), sqlite.dialect()])
def test_enqueue_statement_compiles_to_on_conflict_do_nothing(dialect):
    insert_cls = (
        postgresql.insert if dialect.name == "postgresql" else sqlite.insert
    )
    stmt = (
        insert_cls(ZendeskChairNote)
        .values(email_id=1, zendesk_ticket_id=2, status=PENDING, attempts=0)
        .on_conflict_do_nothing(index_elements=["email_id"])
        .returning(ZendeskChairNote.id)
    )
    sql = str(stmt.compile(dialect=dialect))
    assert "ON CONFLICT (email_id) DO NOTHING" in sql
    assert "RETURNING" in sql


# --- the migration -----------------------------------------------------------------------


def _run_alembic(args, db_url: str):
    env = {**os.environ, "DATABASE_URL": db_url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(BACKEND_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _tables(db_file) -> set[str]:
    con = sqlite3.connect(db_file)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def test_migration_is_the_single_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(Config(str(BACKEND_ROOT / "alembic.ini")))
    assert script.get_heads() == [_REVISION]
    assert script.get_revision(_REVISION).down_revision == _PREV_REVISION


def test_migration_upgrade_downgrade_and_re_upgrade(tmp_path):
    """Runs against a throwaway SQLite file, never a real database."""
    db_file = tmp_path / "chair_notes.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    up = _run_alembic(["upgrade", _REVISION], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}"
    assert "zendesk_chair_notes" in _tables(db_file)

    con = sqlite3.connect(db_file)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        columns = {r[1] for r in con.execute("PRAGMA table_info(zendesk_chair_notes)")}
        assert columns == {c.name for c in ZendeskChairNote.__table__.columns}
        indexes = {r[1]: r for r in con.execute("PRAGMA index_list(zendesk_chair_notes)")}
        assert {"ix_zendesk_chair_notes_zendesk_ticket_id", "ix_zendesk_chair_notes_status"} <= set(indexes)
        unique_cols = [
            [c[2] for c in con.execute(f"PRAGMA index_info('{name}')")]
            for name, row in indexes.items()
            if row[2] == 1
        ]
        assert ["email_id"] in unique_cols
        fks = list(con.execute("PRAGMA foreign_key_list(zendesk_chair_notes)"))
        assert [(r[2], r[3], r[6]) for r in fks] == [("emails", "email_id", "CASCADE")]

        con.execute(
            "INSERT INTO emails (sender, subject, body, status, source, openreview_candidate_dismissed, redrafting) "
            "VALUES ('a@example.org', 's', 'b', 'PENDING', 'zendesk', 0, 0)"
        )
        email_id = con.execute("SELECT id FROM emails").fetchone()[0]
        con.execute(
            "INSERT INTO zendesk_chair_notes (email_id, zendesk_ticket_id) VALUES (?, 21567)",
            (email_id,),
        )
        con.commit()
        assert con.execute("SELECT status, attempts FROM zendesk_chair_notes").fetchone() == ("pending", 0)
        for bad in ("UPDATE zendesk_chair_notes SET status = 'sent'",
                    "UPDATE zendesk_chair_notes SET attempts = 4"):
            with pytest.raises(sqlite3.IntegrityError):
                con.execute(bad)
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO zendesk_chair_notes (email_id, zendesk_ticket_id) VALUES (?, 1)",
                (email_id,),
            )
    finally:
        con.close()

    down = _run_alembic(["downgrade", _PREV_REVISION], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}"
    tables = _tables(db_file)
    assert "zendesk_chair_notes" not in tables
    assert {"emails", "paper_assignments", "phase1_appeals"} <= tables

    again = _run_alembic(["upgrade", _REVISION], db_url)
    assert again.returncode == 0, f"re-upgrade failed:\n{again.stderr}"
    assert "zendesk_chair_notes" in _tables(db_file)


def test_model_and_migration_declare_the_same_columns():
    names = [c.name for c in ZendeskChairNote.__table__.columns]
    assert names == [
        "id", "email_id", "zendesk_ticket_id", "mode", "body_sha256", "status",
        "attempts", "last_error", "zendesk_audit_id", "claimed_at", "posted_at",
        "created_at", "updated_at",
    ]
    assert inspect(ZendeskChairNote).local_table.name == "zendesk_chair_notes"
