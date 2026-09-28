"""The pipeline's extraction is persisted on Email and EmailProcessingResult.

Covers the subtask-4 DB layer: the migration round-trips, a processed email's
``extraction`` column survives a write + read, per-follow-up rows carry their
own, and NULL (every row predating the column) breaks nothing on read.

SCOPE LIMIT: persistence only — the DB layer, up to and including the JSON
column. An endpoint DOES serve ``extraction`` (it did even when this note first
claimed otherwise), so nothing here should be read as covering the wire:
pipeline-to-HTTP coverage lives in ``test_email_extraction_api.py``.
"""

import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base, Email, EmailProcessingResult, EmailThreadMessage
from app.pipeline.distiller import DistillResult
from app.pipeline.orchestrator import EmailPipeline
from app.repositories.email_repository import EmailRepository

BACKEND_ROOT = Path(__file__).resolve().parents[1]


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


class _StubRetriever:
    async def retrieve(self, query, intent, top_k=3, *, prior_intent=""):
        return []


class _StubDistiller:
    def __init__(self, result):
        self.result = result

    async def distill(self, subject, body, *, transcript=None):
        return self.result


def _pipeline():
    pipeline = EmailPipeline()
    pipeline.retriever = _StubRetriever()
    return pipeline


_EMAIL = {
    "from": "jane@example.edu",
    "sender_name": "Jane Roe",
    "subject": "Re: Desk Rejection of Your Submission 22336",
    "body": "Dear chairs, we would like to appeal.",
}


# ---------------------------------------------------------------------------
# Migration round-trip (subprocess, temp SQLite file)
# ---------------------------------------------------------------------------
_PREV_REVISION = "f8b2c4d6e0a1"  # suggestion_audit_logs — this migration's down_revision.
# The revision under test. ⚠️ PINNED, NOT "head": this exercises the round-trip
# of the EXTRACTION migration specifically, and it used to say "head" only
# because that migration happened to be head when it was written. Any later
# revision that adds an index to `emails` then breaks it for the wrong reason —
# `before` is snapshotted at head and includes the newer index, while the
# downgrade to _PREV_REVISION necessarily unwinds that revision too, so the
# index sets cannot match. Caught exactly that way when
# c9f3a1b7d204 (openreview_candidate_dismissed) landed.
_REVISION = "57b59f3ef990"


def _run_alembic(args, db_url: str):
    env = {**os.environ, "DATABASE_URL": db_url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(BACKEND_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _columns(db_file, table: str) -> list[str]:
    con = sqlite3.connect(db_file)
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


def _indexes(db_file, table: str) -> list[str]:
    con = sqlite3.connect(db_file)
    try:
        return sorted(
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
                (table,),
            )
        )
    finally:
        con.close()


def test_extraction_migration_round_trips(tmp_path):
    """upgrade → downgrade → upgrade, on BOTH tables.

    The downgrade is genuinely exercised rather than assumed from the column
    type: dropping a column under SQLite goes through ``batch_alter_table``,
    which RECREATES the table, so the reversal is only safe if the rebuilt
    table keeps its rows and indexes. Both are asserted below.
    """
    db_file = tmp_path / "extraction_roundtrip.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    up = _run_alembic(["upgrade", _REVISION], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}"
    for table in ("emails", "email_processing_results"):
        assert "extraction" in _columns(db_file, table), table

    before = {t: _indexes(db_file, t) for t in ("emails", "email_processing_results")}
    con = sqlite3.connect(db_file)
    con.execute(
        "INSERT INTO emails (sender, subject, body, status, extraction) "
        "VALUES ('a@x.edu', 's', 'b', 'draft_generated', '{\"method\":\"none\"}')"
    )
    con.commit()
    con.close()

    down = _run_alembic(["downgrade", _PREV_REVISION], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}"
    for table in ("emails", "email_processing_results"):
        assert "extraction" not in _columns(db_file, table), table
        # The batch recreate must not silently drop indexes or rows.
        assert _indexes(db_file, table) == before[table], table
    con = sqlite3.connect(db_file)
    assert con.execute("SELECT count(*) FROM emails").fetchone()[0] == 1
    con.close()

    up2 = _run_alembic(["upgrade", _REVISION], db_url)
    assert up2.returncode == 0, f"re-upgrade failed:\n{up2.stderr}"
    for table in ("emails", "email_processing_results"):
        assert "extraction" in _columns(db_file, table), table
        assert _indexes(db_file, table) == before[table], table


# ---------------------------------------------------------------------------
# Email.extraction round-trips through a write + read
# ---------------------------------------------------------------------------
async def test_processed_email_persists_extraction(session):
    result = await _pipeline().process_email(_EMAIL, session)

    email = await EmailRepository().get_email_by_id(session, result.email_id)
    assert email.extraction is not None
    assert email.extraction["submission_numbers"] == ["22336"]
    assert email.extraction["openreview_forum_ids"] == []
    assert email.extraction["method"] == "regex_fallback"
    assert email.extraction["authors"] == [
        {"name": "Jane Roe", "email": "jane@example.edu", "affiliation": None}
    ]


async def test_extraction_round_trips_every_field_from_the_llm_path(
    session, monkeypatch
):
    """Every field survives the JSON write + read, populated.

    ``openreview_note_id`` and ``openreview_notification_sender`` are both None
    here only because this fixture's body carries neither a forum link nor an
    OpenReview address — NOT because the LLM path withholds them, which it does
    not: both are read from the raw text on either path. Pinned as explicit keys
    rather than omitted so this stays an EXACT-shape check. The test below feeds
    a body that does carry the address.
    """
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    pipeline = _pipeline()
    pipeline.distiller = _StubDistiller(
        DistillResult(
            queries=["q"],
            intent="desk_reject_appeal",
            confidence=0.9,
            submission_numbers_raw=["99999"],
            openreview_ids_raw=["Ab3xY9kLm2"],
            authors_raw=[
                "Jane Roe | jane@example.edu | Example University",
                "John Doe | NONE | NONE",
            ],
            # `is_reciprocal_dispute` is NOT set here any more: the field was
            # removed from DistillResult when the question moved to its own
            # conditional call. The snapshot below therefore pins it as None —
            # which is the real contract now, since `extract()` never sets it
            # on either path and `_compute` attaches the detector's answer
            # after extraction.
        )
    )
    result = await pipeline.process_email(_EMAIL, session)

    stored = (await EmailRepository().get_email_by_id(session, result.email_id)).extraction
    assert stored == {
        "submission_numbers": ["99999"],
        "openreview_forum_ids": ["Ab3xY9kLm2"],
        "openreview_note_id": None,
        "openreview_notification_sender": None,
        # Derived (note_id AND sender). False here because the LLM path never
        # reports a note id, so the left operand is unreachable on this path.
        "openreview_reply_candidate": False,
        # The fixture body carries no quote, so the whole trimmed body is the
        # reply — this field is NOT empty here, unlike the two scalars above.
        "extracted_reply_text": "Dear chairs, we would like to appeal.",
        "authors": [
            {
                "name": "Jane Roe",
                "email": "jane@example.edu",
                "affiliation": "Example University",
            },
            {"name": "John Doe", "email": None, "affiliation": None},
        ],
        "method": "llm_distiller",
        # Copied verbatim from the DistillResult above and persisted. The only
        # field here the regex path can never produce, so a True in the stored
        # JSON is proof the LLM path ran end to end.
        # None, not True: the extractor no longer sets this on any path (the
        # detector does, after extraction — commit 4). Kept as an EXPLICIT key
        # so this stays an exact-shape check and the field's removal from the
        # serialized record would fail here rather than pass silently.
        "is_reciprocal_dispute": None,
        # None: set only by `_compute` from the appeal-reason classifier, which
        # is OFF by default (D65/D76). Explicit for the same exact-shape reason.
        "appeal_reason": None,
    }


async def test_notification_sender_survives_the_distill_path_end_to_end(
    session, monkeypatch
):
    """The production path actually stores it — the point of the carve-out.

    Every other extraction field is withheld on the distill path unless the
    distiller itself reported it, and `QUERY_STRATEGY=distill` is what runs in
    production. So a unit test proving `EmailExtractor` populates this field is
    not enough on its own: if the orchestrator did not hand the raw body through
    on this path, the field would be permanently None in production and every
    extractor-level test would still pass. This drives the real orchestrator
    with a stub distiller and reads the value back out of the JSON column.
    """
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    pipeline = _pipeline()
    pipeline.distiller = _StubDistiller(
        DistillResult(queries=["q"], intent="desk_reject_appeal", confidence=0.9)
    )
    address = "aaai2027-notifications@openreview.net"
    email_data = {
        **_EMAIL,
        "body": f'Please advise.\n\n发件人:"AAAI 2027" <{address}>\n主题: commented',
    }

    result = await pipeline.process_email(email_data, session)

    stored = (
        await EmailRepository().get_email_by_id(session, result.email_id)
    ).extraction
    assert stored["method"] == "llm_distiller"
    assert stored["openreview_notification_sender"] == address


async def test_reply_candidate_is_true_end_to_end_on_the_distill_path(
    session, monkeypatch
):
    """The whole feature, proven where it has to work: production.

    `QUERY_STRATEGY=distill` is what runs in production, and until the note-id
    correction this flag could not be True there at all — its left operand was
    hardcoded unreachable on that path. A unit test on `EmailExtractor` alone
    would not catch a regression in the orchestrator's hand-off of the raw body,
    so this drives the real pipeline with a stub distiller and reads the flag
    back out of the JSON column.

    The body is the real reported shape: a Chinese-labelled quoted header from
    the venue's notification address, plus the forum link carrying the noteId.
    """
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    pipeline = _pipeline()
    # The model reports the forum id, as its prompt asks it to; the note id and
    # the sender address are read from the raw text on this path.
    pipeline.distiller = _StubDistiller(
        DistillResult(
            queries=["q"],
            intent="cms_support",
            confidence=0.9,
            openreview_ids_raw=["ll0avn6ylq"],
        )
    )
    address = "aaai2027-notifications@openreview.net"
    link = "https://openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm"
    email_data = {
        **_EMAIL,
        "body": (
            "老师您好，请见下方邮件。\n\n"
            f'发件人:"AAAI 2027" <{address}>\n'
            "主题: [AAAI 2027] Senior Program Committee 6UDQ commented on a "
            "paper you are reviewing. Paper Number: 1030\n"
            f"{link}"
        ),
    }

    result = await pipeline.process_email(email_data, session)

    stored = (
        await EmailRepository().get_email_by_id(session, result.email_id)
    ).extraction
    assert stored["method"] == "llm_distiller"
    assert stored["openreview_reply_candidate"] is True
    # Both operands really are present, so the flag is not True by accident.
    assert stored["openreview_note_id"] == "jnHgRMHgrm"
    assert stored["openreview_notification_sender"] == address
    # ...and the model's own forum-id answer is still the model's.
    assert stored["openreview_forum_ids"] == ["ll0avn6ylq"]


async def test_multi_value_identifier_lists_round_trip(session, monkeypatch):
    """The list shape's whole point: >1 entry must survive the JSON column.

    A single-element list would round-trip even if something upstream collapsed
    lists to scalars, so this uses several of each and asserts ORDER too.
    """
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    pipeline = _pipeline()
    pipeline.distiller = _StubDistiller(
        DistillResult(
            queries=["q"],
            intent="desk_reject_appeal",
            confidence=0.9,
            submission_numbers_raw=["11111", "22222", "33333"],
            openreview_ids_raw=["Ab3xY9kLm2", "Zz9QwErTy1"],
        )
    )
    result = await pipeline.process_email(_EMAIL, session)

    stored = (
        await EmailRepository().get_email_by_id(session, result.email_id)
    ).extraction
    assert stored["submission_numbers"] == ["11111", "22222", "33333"]
    assert stored["openreview_forum_ids"] == ["Ab3xY9kLm2", "Zz9QwErTy1"]


async def test_examined_but_empty_persists_as_empty_lists_not_null(session):
    """"Examined, found none" must be storable and distinguishable from NULL."""
    email = Email(
        sender="a@example.edu",
        subject="No identifiers here",
        body="Just a general question.",
        status="draft_generated",
        extraction={
            "submission_numbers": [],
            "openreview_forum_ids": [],
            "authors": [],
            "method": "llm_distiller",
        },
    )
    session.add(email)
    await session.commit()
    await session.refresh(email)

    fetched = await EmailRepository().get_email_by_id(session, str(email.id))
    assert fetched.extraction is not None  # NOT collapsed to NULL
    assert fetched.extraction["submission_numbers"] == []
    assert fetched.extraction["method"] == "llm_distiller"


async def test_extraction_is_serialized_with_model_dump(session):
    """Same pattern as classification/routing/draft — a plain dict, not a model."""
    result = await _pipeline().process_email(_EMAIL, session)
    email = await EmailRepository().get_email_by_id(session, result.email_id)
    assert isinstance(email.extraction, dict)
    assert set(email.extraction) == {
        "submission_numbers",
        "openreview_forum_ids",
        "openreview_note_id",
        "openreview_notification_sender",
        "openreview_reply_candidate",
        "extracted_reply_text",
        "authors",
        "method",
        "is_reciprocal_dispute",
        "appeal_reason",
    }


async def test_reprocess_updates_the_stored_extraction(session):
    """The reprocess path's allowlist must include extraction, or a redraft
    would leave a stale extraction beside a fresh draft."""
    pipeline = _pipeline()
    created = await pipeline.process_email(_EMAIL, session)
    email = await EmailRepository().get_email_by_id(session, created.email_id)
    email.extraction = {"submission_numbers": ["00000"], "method": "none",
                        "openreview_forum_ids": [], "authors": []}
    await session.commit()

    await pipeline.reprocess_email(session, email)

    refreshed = await EmailRepository().get_email_by_id(session, created.email_id)
    assert refreshed.extraction["submission_numbers"] == ["22336"]
    assert refreshed.extraction["method"] == "regex_fallback"


# ---------------------------------------------------------------------------
# EmailProcessingResult carries its own extraction
# ---------------------------------------------------------------------------
async def test_processing_result_persists_extraction(session):
    """A follow-up message's own result row stores its own extraction."""
    pipeline = _pipeline()
    created = await pipeline.process_email(_EMAIL, session)

    message = EmailThreadMessage(
        email_id=int(created.email_id),
        public=True,
        author_role="end-user",
        plain_body="Following up about submission 44444.",
        created_at=datetime.now(timezone.utc),
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)

    computed = await pipeline.compute(
        {
            "from": "jane@example.edu",
            "sender_name": "Jane Roe",
            "subject": "Follow-up",
            "body": "Following up about submission 44444.",
        },
        session,
    )
    row = await EmailRepository().add_processing_result(
        session, message.id, computed.record
    )

    assert row.extraction is not None
    assert row.extraction["submission_numbers"] == ["44444"]
    assert row.extraction["method"] == "regex_fallback"


async def test_processing_result_extraction_is_independent_of_the_parent(session):
    """A follow-up can name a different submission than the opening inquiry."""
    pipeline = _pipeline()
    created = await pipeline.process_email(_EMAIL, session)
    parent = await EmailRepository().get_email_by_id(session, created.email_id)

    message = EmailThreadMessage(
        email_id=int(created.email_id),
        public=True,
        author_role="end-user",
        plain_body="Also about submission 44444.",
        created_at=datetime.now(timezone.utc),
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)

    computed = await pipeline.compute(
        {"from": "jane@example.edu", "subject": "F", "body": "About submission 44444."},
        session,
    )
    row = await EmailRepository().add_processing_result(
        session, message.id, computed.record
    )

    assert parent.extraction["submission_numbers"] == ["22336"]
    assert row.extraction["submission_numbers"] == ["44444"]


# ---------------------------------------------------------------------------
# NULL extraction (every row predating the column) must break nothing
# ---------------------------------------------------------------------------
async def test_legacy_email_with_null_extraction_reads_fine(session):
    """A pre-migration row has extraction NULL; reads must not assume presence."""
    email = Email(
        sender="legacy@example.edu",
        subject="Legacy row",
        body="Predates the extraction column.",
        status="draft_generated",
    )
    session.add(email)
    await session.commit()
    await session.refresh(email)

    fetched = await EmailRepository().get_email_by_id(session, str(email.id))
    assert fetched is not None
    assert fetched.extraction is None
    # The queue read path must tolerate it too (no KeyError / no filtering out).
    queue = await EmailRepository().get_email_queue(session)
    assert any(e.id == email.id for e in queue)


async def test_legacy_processing_result_with_null_extraction_reads_fine(session):
    email = Email(
        sender="legacy@example.edu", subject="s", body="b", status="draft_generated"
    )
    session.add(email)
    await session.commit()
    await session.refresh(email)

    message = EmailThreadMessage(
        email_id=email.id,
        public=True,
        created_at=datetime.now(timezone.utc),
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)

    row = EmailProcessingResult(thread_message_id=message.id)
    session.add(row)
    await session.commit()

    found = (
        await session.execute(
            select(EmailProcessingResult).where(
                EmailProcessingResult.thread_message_id == message.id
            )
        )
    ).scalar_one()
    assert found.extraction is None


async def test_add_processing_result_tolerates_a_record_without_extraction(session):
    """A record built before this column existed must still persist."""
    email = Email(
        sender="a@example.edu", subject="s", body="b", status="draft_generated"
    )
    session.add(email)
    await session.commit()
    await session.refresh(email)

    message = EmailThreadMessage(
        email_id=email.id,
        public=True,
        created_at=datetime.now(timezone.utc),
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)

    row = await EmailRepository().add_processing_result(
        session, message.id, {"classification": {}, "routing": {}, "draft": {}}
    )
    assert row.extraction is None
