"""PaperAssignmentRepository, Phase1AppealRepository, the assignment loader,
and the migration that creates their tables.

Repository tests run on in-memory SQLite with ``PRAGMA foreign_keys=ON`` so the
ON DELETE CASCADE from ``emails`` actually fires. All data is synthetic.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Email, PaperAssignment, Phase1Appeal
from app.repositories.phase1_appeal_repository import (
    PaperAssignmentRepository,
    Phase1AppealRepository,
)
from scripts import load_paper_assignments as loader

BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REVISION = "4faaa7e50e0a"
_PREV_REVISION = "c9f3a1b7d204"


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


def _assignment(number: str, apc: str = "APC One", forum_id: str | None = None) -> dict:
    forum_id = forum_id or f"Fid{number}xx"
    return {
        "paper_number": number,
        "apc_name": apc,
        "openreview_url": f"https://openreview.net/forum?id={forum_id}",
        "openreview_forum_id": forum_id,
    }


def _appeal(submission: str | None, *, ticket: int | None = 1001, **overrides) -> dict:
    row = {
        "zendesk_ticket_id": ticket,
        "submission_number": submission,
        "apc_name": None,
        "openreview_url": None,
        "relation": "appeal",
        "reasons": [{"reason": "reviewer_misjudgment", "quote": "the reviewer misread it"}],
        "must_verify": False,
        "prompt_sha256": "0" * 64,
        "model": "test-model",
    }
    row.update(overrides)
    return row


async def _email(db, ticket: int | None = None) -> Email:
    e = Email(sender="author@example.org", subject="s", body="b", zendesk_ticket_id=ticket)
    db.add(e)
    await db.commit()
    await db.refresh(e)
    return e


async def _appeal_rows(db, email_id: int | None = None) -> list[Phase1Appeal]:
    q = select(Phase1Appeal).order_by(Phase1Appeal.id)
    if email_id is not None:
        q = q.where(Phase1Appeal.email_id == email_id)
    return list((await db.execute(q)).scalars().all())


async def _count(db, model) -> int:
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


# --- PaperAssignmentRepository -------------------------------------------------


async def test_upsert_inserts_and_returns_count(session):
    repo = PaperAssignmentRepository()
    written = await repo.upsert_many(session, [_assignment("101"), _assignment("102")])
    assert written == 2
    assert await _count(session, PaperAssignment) == 2


async def test_upsert_is_idempotent(session):
    repo = PaperAssignmentRepository()
    rows = [_assignment("101", "APC One"), _assignment("102", "APC Two")]
    await repo.upsert_many(session, rows)
    await repo.upsert_many(session, rows)

    assert await _count(session, PaperAssignment) == 2
    found = await repo.get_by_numbers(session, ["101", "102"])
    assert found["101"].apc_name == "APC One"
    assert found["102"].apc_name == "APC Two"


async def test_upsert_updates_existing_row(session):
    repo = PaperAssignmentRepository()
    await repo.upsert_many(session, [_assignment("101", "APC One", "OldForumId")])
    await repo.upsert_many(session, [_assignment("101", "APC Two", "NewForumId")])

    row = (await repo.get_by_numbers(session, ["101"]))["101"]
    assert row.apc_name == "APC Two"
    assert row.openreview_forum_id == "NewForumId"
    assert await repo.get_by_forum_ids(session, ["OldForumId"]) == {}
    assert await _count(session, PaperAssignment) == 1


async def test_upsert_absent_key_keeps_stored_value(session):
    repo = PaperAssignmentRepository()
    await repo.upsert_many(session, [{**_assignment("101"), "cycle": "AAAI-28"}])
    await repo.upsert_many(session, [{"paper_number": "101", "apc_name": "APC Two"}])

    row = (await repo.get_by_numbers(session, ["101"]))["101"]
    assert row.apc_name == "APC Two"
    assert row.cycle == "AAAI-28"
    assert row.openreview_forum_id == "Fid101xx"


async def test_upsert_defaults_cycle(session):
    await PaperAssignmentRepository().upsert_many(session, [_assignment("101")])
    row = (await session.execute(select(PaperAssignment))).scalar_one()
    assert row.cycle == "AAAI-27"


async def test_upsert_duplicate_in_batch_last_wins_and_blank_skipped(session):
    repo = PaperAssignmentRepository()
    written = await repo.upsert_many(
        session,
        [_assignment("101", "First"), {"paper_number": "", "apc_name": "x"}, _assignment("101", "Last")],
    )
    assert written == 1
    assert (await repo.get_by_numbers(session, ["101"]))["101"].apc_name == "Last"


async def test_upsert_empty_is_noop(session):
    assert await PaperAssignmentRepository().upsert_many(session, []) == 0


async def test_get_by_numbers_found_and_missing(session):
    repo = PaperAssignmentRepository()
    await repo.upsert_many(session, [_assignment("101"), _assignment("102")])

    found = await repo.get_by_numbers(session, ["102", "999", "", "102"])
    assert set(found) == {"102"}
    assert found["102"].paper_number == "102"
    assert await repo.get_by_numbers(session, []) == {}


async def test_lookups_span_more_than_one_in_chunk(session):
    repo = PaperAssignmentRepository()
    rows = [_assignment(str(n), forum_id=f"F{n:09d}") for n in range(1, 1201)]
    assert await repo.upsert_many(session, rows) == 1200

    numbers = await repo.get_by_numbers(session, [str(n) for n in range(1, 1201)])
    assert len(numbers) == 1200
    forums = await repo.get_by_forum_ids(session, [f"F{n:09d}" for n in range(1, 1201)])
    assert len(forums) == 1200


async def test_get_by_forum_ids_maps_forum_id_to_paper(session):
    repo = PaperAssignmentRepository()
    await repo.upsert_many(
        session,
        [_assignment("101", forum_id="AbC123xyZ9"), _assignment("102", forum_id="QwErTy0987")],
    )

    found = await repo.get_by_forum_ids(session, ["AbC123xyZ9", "Unknown000"])
    assert set(found) == {"AbC123xyZ9"}
    assert found["AbC123xyZ9"].paper_number == "101"


async def test_get_by_forum_ids_is_case_sensitive(session):
    repo = PaperAssignmentRepository()
    await repo.upsert_many(session, [_assignment("101", forum_id="AbC123xyZ9")])
    assert await repo.get_by_forum_ids(session, ["abc123xyz9"]) == {}


async def test_get_by_forum_ids_never_matches_null_forum_id(session):
    repo = PaperAssignmentRepository()
    await repo.upsert_many(
        session, [{**_assignment("101"), "openreview_forum_id": None}]
    )
    assert await repo.get_by_forum_ids(session, ["", "None"]) == {}


# --- Phase1AppealRepository ----------------------------------------------------


async def test_replace_inserts_rows(session):
    email = await _email(session, ticket=1001)
    repo = Phase1AppealRepository()

    n = await repo.replace_for_email(session, email.id, [_appeal("101"), _appeal("102")])
    assert n == 2
    rows = await _appeal_rows(session, email.id)
    assert [r.submission_number for r in rows] == ["101", "102"]
    assert rows[0].reasons == [
        {"reason": "reviewer_misjudgment", "quote": "the reviewer misread it"}
    ]
    assert rows[0].classified_at is not None


async def test_replace_removes_old_rows_and_keeps_new(session):
    email = await _email(session, ticket=1001)
    repo = Phase1AppealRepository()
    await repo.replace_for_email(session, email.id, [_appeal("101"), _appeal("102")])

    n = await repo.replace_for_email(
        session, email.id, [_appeal("103", relation="feedback_only")]
    )
    assert n == 1
    rows = await _appeal_rows(session, email.id)
    assert [(r.submission_number, r.relation) for r in rows] == [("103", "feedback_only")]


async def test_replace_leaves_other_emails_untouched(session):
    a = await _email(session, ticket=1001)
    b = await _email(session, ticket=1002)
    repo = Phase1AppealRepository()
    await repo.replace_for_email(session, a.id, [_appeal("101")])
    await repo.replace_for_email(session, b.id, [_appeal("201", ticket=1002)])

    await repo.replace_for_email(session, a.id, [_appeal("102")])
    assert [r.submission_number for r in await _appeal_rows(session, b.id)] == ["201"]


async def test_replace_with_empty_list_clears(session):
    email = await _email(session)
    repo = Phase1AppealRepository()
    await repo.replace_for_email(session, email.id, [_appeal("101")])

    assert await repo.replace_for_email(session, email.id, []) == 0
    assert await _appeal_rows(session, email.id) == []


async def test_replace_takes_email_id_from_argument(session):
    a = await _email(session)
    b = await _email(session)
    await Phase1AppealRepository().replace_for_email(
        session, a.id, [{**_appeal("101"), "email_id": b.id}]
    )
    assert len(await _appeal_rows(session, a.id)) == 1
    assert await _appeal_rows(session, b.id) == []


async def test_replace_allows_null_submission_number(session):
    email = await _email(session)
    await Phase1AppealRepository().replace_for_email(session, email.id, [_appeal(None)])
    (row,) = await _appeal_rows(session, email.id)
    assert row.submission_number is None


async def test_replace_failure_keeps_old_rows(session):
    email_id = (await _email(session)).id  # rollback expires the instance
    repo = Phase1AppealRepository()
    await repo.replace_for_email(session, email_id, [_appeal("101")])

    # relation is NOT NULL: the insert fails, so the delete must roll back too.
    with pytest.raises(IntegrityError):
        await repo.replace_for_email(session, email_id, [_appeal("102", relation=None)])

    rows = await _appeal_rows(session, email_id)
    assert [r.submission_number for r in rows] == ["101"]


async def test_delete_for_email(session):
    a = await _email(session)
    b = await _email(session)
    repo = Phase1AppealRepository()
    await repo.replace_for_email(session, a.id, [_appeal("101"), _appeal("102")])
    await repo.replace_for_email(session, b.id, [_appeal("201")])

    assert await repo.delete_for_email(session, a.id) == 2
    assert await _appeal_rows(session, a.id) == []
    assert len(await _appeal_rows(session, b.id)) == 1
    assert await repo.delete_for_email(session, a.id) == 0


async def test_rows_cascade_on_email_delete(session):
    a = await _email(session)
    b = await _email(session)
    repo = Phase1AppealRepository()
    await repo.replace_for_email(session, a.id, [_appeal("101"), _appeal("102")])
    await repo.replace_for_email(session, b.id, [_appeal("201")])

    await session.delete(a)
    await session.commit()

    remaining = await _appeal_rows(session)
    assert [r.email_id for r in remaining] == [b.id]


async def test_list_all_orders_by_ticket_then_submission(session):
    e1 = await _email(session, ticket=3000)
    e2 = await _email(session, ticket=1000)
    e3 = await _email(session)
    repo = Phase1AppealRepository()
    await repo.replace_for_email(
        session, e1.id, [_appeal("300", ticket=3000), _appeal(None, ticket=3000), _appeal("100", ticket=3000)]
    )
    await repo.replace_for_email(session, e2.id, [_appeal("200", ticket=1000)])
    await repo.replace_for_email(session, e3.id, [_appeal("050", ticket=None)])

    ordered = [(r.zendesk_ticket_id, r.submission_number) for r in await repo.list_all(session)]
    assert ordered == [
        (1000, "200"),
        (3000, "100"),
        (3000, "300"),
        (3000, None),
        (None, "050"),
    ]


async def test_list_all_empty(session):
    assert await Phase1AppealRepository().list_all(session) == []


# --- loader --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://openreview.net/forum?id=AbC123xyZ9", "AbC123xyZ9"),
        ("  https://openreview.net/forum?id=AbC123xyZ9  ", "AbC123xyZ9"),
        ("https://openreview.net/forum?id=AbC123xyZ9&noteId=Zz99", "AbC123xyZ9"),
        ("https://openreview.net/forum?noteId=Zz99&id=AbC123xyZ9", "AbC123xyZ9"),
        ("https://openreview.net/pdf?id=AbC123xyZ9", "AbC123xyZ9"),
        ("https://openreview.net/forum", None),
        ("https://openreview.net/forum?id=", None),
        ("", None),
        (None, None),
        ("https://openreview.net/forum?id=" + "x" * 33, None),
    ],
)
def test_parse_forum_id(url, expected):
    assert loader.parse_forum_id(url) == expected


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_sheet_strips_and_parses(tmp_path):
    csv_path = _write_csv(
        tmp_path / "sheet.csv",
        "﻿Assignment,Paper number,Paper URL\n"
        "  APC One  , 101 , https://openreview.net/forum?id=AbC123xyZ9 \n"
        "APC Two,102,https://openreview.net/forum?id=QwErTy0987\n"
        "APC Three,103,https://openreview.net/forum\n"
        ",104,https://openreview.net/forum?id=Missing000\n"
        "APC Four,,https://openreview.net/forum?id=Missing001\n"
        "APC Five,105,\n",
    )
    sheet = loader.parse_sheet(csv_path, cycle="AAAI-28")

    assert sheet.read == 6
    assert sheet.skipped == 3
    assert sheet.without_forum_id == 1
    assert sheet.rows == [
        {
            "paper_number": "101",
            "apc_name": "APC One",
            "openreview_url": "https://openreview.net/forum?id=AbC123xyZ9",
            "openreview_forum_id": "AbC123xyZ9",
            "cycle": "AAAI-28",
        },
        {
            "paper_number": "102",
            "apc_name": "APC Two",
            "openreview_url": "https://openreview.net/forum?id=QwErTy0987",
            "openreview_forum_id": "QwErTy0987",
            "cycle": "AAAI-28",
        },
        {
            "paper_number": "103",
            "apc_name": "APC Three",
            "openreview_url": "https://openreview.net/forum",
            "openreview_forum_id": None,
            "cycle": "AAAI-28",
        },
    ]


def test_parse_sheet_default_cycle(tmp_path):
    csv_path = _write_csv(
        tmp_path / "sheet.csv",
        "Assignment,Paper number,Paper URL\nAPC One,101,https://openreview.net/forum?id=AbC123xyZ9\n",
    )
    assert loader.parse_sheet(csv_path).rows[0]["cycle"] == "AAAI-27"


def test_parse_sheet_missing_column_raises(tmp_path):
    csv_path = _write_csv(tmp_path / "sheet.csv", "Assignment,Paper number\nAPC One,101\n")
    with pytest.raises(ValueError, match="Paper URL"):
        loader.parse_sheet(csv_path)


async def test_load_upserts_and_is_rerunnable(tmp_path, factory):
    csv_path = _write_csv(
        tmp_path / "sheet.csv",
        "Assignment,Paper number,Paper URL\n"
        "APC One,101,https://openreview.net/forum?id=AbC123xyZ9\n"
        "APC Two,102,https://openreview.net/forum?id=QwErTy0987\n",
    )
    sheet, written = await loader.load(csv_path, "AAAI-27", factory)
    assert (sheet.read, written) == (2, 2)
    _, written_again = await loader.load(csv_path, "AAAI-27", factory)
    assert written_again == 2

    async with factory() as db:
        assert await _count(db, PaperAssignment) == 2
        found = await PaperAssignmentRepository().get_by_forum_ids(db, ["QwErTy0987"])
        assert found["QwErTy0987"].paper_number == "102"
        assert found["QwErTy0987"].apc_name == "APC Two"


# --- migration -----------------------------------------------------------------


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


def test_migration_upgrade_and_downgrade(tmp_path):
    """Runs against a throwaway SQLite file, never a real database."""
    db_file = tmp_path / "phase1.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    up = _run_alembic(["upgrade", _REVISION], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}"
    assert {"paper_assignments", "phase1_appeals"} <= _tables(db_file)

    con = sqlite3.connect(db_file)
    try:
        con.execute(
            "INSERT INTO paper_assignments (paper_number, apc_name, openreview_url) "
            "VALUES ('101', 'APC One', 'https://openreview.net/forum?id=AbC123xyZ9')"
        )
        con.commit()
        assert con.execute("SELECT cycle FROM paper_assignments").fetchone()[0] == "AAAI-27"
        indexes = {r[1] for r in con.execute("PRAGMA index_list(phase1_appeals)")}
        assert {
            "ix_phase1_appeals_email_id",
            "ix_phase1_appeals_zendesk_ticket_id",
            "ix_phase1_appeals_submission_number",
        } <= indexes
        fks = list(con.execute("PRAGMA foreign_key_list(phase1_appeals)"))
        assert [(r[2], r[3], r[6]) for r in fks] == [("emails", "email_id", "CASCADE")]
    finally:
        con.close()

    down = _run_alembic(["downgrade", _PREV_REVISION], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}"
    tables = _tables(db_file)
    assert "paper_assignments" not in tables
    assert "phase1_appeals" not in tables
    assert "emails" in tables
