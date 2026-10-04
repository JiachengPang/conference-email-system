"""Persistence for Zendesk chair notes (``zendesk_chair_notes``, Z2).

The claim protocol that keeps an email to ONE internal note, even when two
reprocesses overlap. Every state change is a single conditional statement, so
it is atomic on SQLite and Postgres alike without row locks:

1. :meth:`ChairNoteRepository.enqueue` inserts a ``pending`` row, or does
   nothing when the email already has one (whatever its state).
2. :meth:`ChairNoteRepository.claim` moves ``pending``, or ``failed`` with
   fewer than ``MAX_ATTEMPTS`` attempts, to ``posting`` and counts the attempt.
   Exactly one caller can win; everyone else gets ``False``.
3. :meth:`ChairNoteRepository.mark_posted` / :meth:`mark_failed` close a
   claimed (``posting``) row.

A ``posting`` row is never claimed again: if a process died after Zendesk may
have accepted the note, retrying could post it twice.
:meth:`ChairNoteRepository.find_stale_posting` lists such rows for a manual
check instead.

Writes commit before returning (rolling back and re-raising on failure), like
the other repositories. Timestamps are written as aware UTC values, because
SQLite stores the wall-clock fields and drops the offset.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ZendeskChairNote

PENDING = "pending"
POSTING = "posting"
POSTED = "posted"
FAILED = "failed"
STATUSES = (PENDING, POSTING, POSTED, FAILED)

# Matches the table's CHECK constraint (attempts <= 3).
MAX_ATTEMPTS = 3

# Bound on the stored error text: an exception type and HTTP status fit easily,
# and a long upstream message (which could quote the note) is cut off.
_ERROR_MAX_CHARS = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """An aware UTC datetime; a naive value is read as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _insert_for(db: AsyncSession):
    """The dialect's INSERT construct, which supports ON CONFLICT DO NOTHING."""
    name = db.bind.dialect.name
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # pragma: no cover - only SQLite and Postgres are supported
        raise NotImplementedError(f"chair notes do not support the {name!r} dialect")
    return insert


class ChairNoteRepository:
    """Async data access for the ``zendesk_chair_notes`` table."""

    async def enqueue(
        self,
        db: AsyncSession,
        email_id: int,
        zendesk_ticket_id: int,
        *,
        mode: str | None = None,
    ) -> bool:
        """Create the email's ``pending`` row. Returns ``False`` if it already has one.

        An existing row is left exactly as it is, whatever its state: an email
        gets one note, so a later draft never re-queues it.
        """
        insert = _insert_for(db)
        stmt = (
            insert(ZendeskChairNote)
            .values(
                email_id=email_id,
                zendesk_ticket_id=zendesk_ticket_id,
                mode=mode,
                status=PENDING,
                attempts=0,
            )
            .on_conflict_do_nothing(index_elements=["email_id"])
            .returning(ZendeskChairNote.id)
        )
        try:
            created = (await db.execute(stmt)).scalar_one_or_none()
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        return created is not None

    async def claim(
        self, db: AsyncSession, email_id: int, *, now: datetime | None = None
    ) -> bool:
        """Claim the email's note for posting. ``True`` only for the one caller that wins.

        Claimable: ``pending``, or ``failed`` with fewer than ``MAX_ATTEMPTS``
        attempts. ``posting`` and ``posted`` rows are never claimable. The
        attempt is counted here, at the claim, so a crash mid-post still uses
        one up.
        """
        stmt = (
            update(ZendeskChairNote)
            .where(
                ZendeskChairNote.email_id == email_id,
                ZendeskChairNote.attempts < MAX_ATTEMPTS,
                or_(
                    ZendeskChairNote.status == PENDING,
                    ZendeskChairNote.status == FAILED,
                ),
            )
            .values(
                status=POSTING,
                attempts=ZendeskChairNote.attempts + 1,
                claimed_at=_as_utc(now or _utcnow()),
            )
            .execution_options(synchronize_session=False)
        )
        return await self._update_one(db, stmt)

    async def mark_posted(
        self,
        db: AsyncSession,
        email_id: int,
        *,
        mode: str | None,
        body_sha256: str | None,
        zendesk_audit_id: int | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Record a successful post. Only a ``posting`` row changes; returns whether it did."""
        stmt = (
            update(ZendeskChairNote)
            .where(
                ZendeskChairNote.email_id == email_id,
                ZendeskChairNote.status == POSTING,
            )
            .values(
                status=POSTED,
                mode=mode,
                body_sha256=body_sha256,
                zendesk_audit_id=zendesk_audit_id,
                posted_at=_as_utc(now or _utcnow()),
                last_error=None,
            )
            .execution_options(synchronize_session=False)
        )
        return await self._update_one(db, stmt)

    async def mark_failed(
        self,
        db: AsyncSession,
        email_id: int,
        *,
        error: str,
        mode: str | None = None,
        body_sha256: str | None = None,
    ) -> bool:
        """Record a failed post. Only a ``posting`` row changes; returns whether it did.

        ``error`` should be an exception type and HTTP status, never the note
        body; it is cut to 500 characters regardless.
        """
        stmt = (
            update(ZendeskChairNote)
            .where(
                ZendeskChairNote.email_id == email_id,
                ZendeskChairNote.status == POSTING,
            )
            .values(
                status=FAILED,
                mode=mode,
                body_sha256=body_sha256,
                last_error=(error or "")[:_ERROR_MAX_CHARS],
            )
            .execution_options(synchronize_session=False)
        )
        return await self._update_one(db, stmt)

    async def find_stale_posting(
        self,
        db: AsyncSession,
        *,
        older_than: timedelta,
        now: datetime | None = None,
    ) -> list[ZendeskChairNote]:
        """``posting`` rows claimed more than ``older_than`` ago, oldest first.

        For a manual check only. Nothing here or in :meth:`claim` ever moves
        them on: Zendesk may already have accepted that note.
        """
        cutoff = _as_utc(now or _utcnow()) - older_than
        result = await db.execute(
            select(ZendeskChairNote)
            .where(
                ZendeskChairNote.status == POSTING,
                or_(
                    ZendeskChairNote.claimed_at.is_(None),
                    ZendeskChairNote.claimed_at < cutoff,
                ),
            )
            .order_by(ZendeskChairNote.claimed_at, ZendeskChairNote.id)
        )
        return list(result.scalars().all())

    async def get_by_email_id(
        self, db: AsyncSession, email_id: int
    ) -> ZendeskChairNote | None:
        """The email's row, or ``None``."""
        result = await db.execute(
            select(ZendeskChairNote).where(ZendeskChairNote.email_id == email_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email_ids(
        self, db: AsyncSession, email_ids: list[int]
    ) -> dict[int, ZendeskChairNote]:
        """Each found email's row, keyed by email id. One query for a whole page."""
        wanted = list(dict.fromkeys(email_ids))
        if not wanted:
            return {}
        result = await db.execute(
            select(ZendeskChairNote).where(ZendeskChairNote.email_id.in_(wanted))
        )
        return {row.email_id: row for row in result.scalars().all()}

    async def _update_one(self, db: AsyncSession, stmt) -> bool:
        try:
            result = await db.execute(stmt)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        return (result.rowcount or 0) == 1
