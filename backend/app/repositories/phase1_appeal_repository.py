"""Persistence for phase-1 rejection appeals and the paper-assignment sheet.

All access to ``paper_assignments`` and ``phase1_appeals`` goes through these
repositories; pipeline modules never touch SQLAlchemy directly. Reads return
``{}`` / ``[]`` on miss rather than raising, matching the other repositories.
Writes commit before returning (rolling back on failure), so each call is one
transaction.
"""

from collections.abc import Iterable, Iterator

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PaperAssignment, Phase1Appeal

# Bound on the size of one ``IN (...)`` list, well under SQLite's bind-variable
# limit, so a full assignment sheet can be looked up or upserted in one call.
_IN_CHUNK = 500

_ASSIGNMENT_FIELDS = (
    "apc_name",
    "openreview_url",
    "openreview_forum_id",
    "cycle",
)


def _chunks(values: list[str], size: int = _IN_CHUNK) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _distinct_nonblank(values: Iterable[str | None]) -> list[str]:
    """Order-preserving distinct list of the non-empty values."""
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return list(seen)


class PaperAssignmentRepository:
    """Async data access for the ``paper_assignments`` table."""

    async def upsert_many(self, db: AsyncSession, rows: list[dict]) -> int:
        """Insert or update assignment rows keyed by ``paper_number``.

        Each dict carries ``paper_number`` plus any of ``apc_name``,
        ``openreview_url``, ``openreview_forum_id``, ``cycle``. A key absent
        from the dict leaves the stored value unchanged on update (and takes
        the column default on insert). When the same paper appears more than
        once in ``rows`` the last occurrence wins. Returns the number of
        distinct papers written.
        """
        by_number: dict[str, dict] = {}
        for row in rows:
            number = row.get("paper_number")
            if not number:
                continue
            by_number[number] = row
        if not by_number:
            return 0

        try:
            existing = await self._load(db, list(by_number))
            for number, row in by_number.items():
                values = {k: row[k] for k in _ASSIGNMENT_FIELDS if k in row}
                current = existing.get(number)
                if current is None:
                    db.add(PaperAssignment(paper_number=number, **values))
                else:
                    for key, value in values.items():
                        setattr(current, key, value)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        return len(by_number)

    async def get_by_numbers(
        self, db: AsyncSession, numbers: list[str]
    ) -> dict[str, PaperAssignment]:
        """Map each found submission number to its assignment row."""
        return await self._load(db, _distinct_nonblank(numbers))

    async def get_by_forum_ids(
        self, db: AsyncSession, forum_ids: list[str]
    ) -> dict[str, PaperAssignment]:
        """Map each found OpenReview forum id to its assignment row.

        Matching is exact (forum ids are case-sensitive). Should two papers
        share a forum id, the lowest paper number is returned.
        """
        wanted = _distinct_nonblank(forum_ids)
        found: dict[str, PaperAssignment] = {}
        for chunk in _chunks(wanted):
            result = await db.execute(
                select(PaperAssignment)
                .where(PaperAssignment.openreview_forum_id.in_(chunk))
                .order_by(PaperAssignment.paper_number)
            )
            for row in result.scalars().all():
                found.setdefault(row.openreview_forum_id, row)
        return found

    async def _load(
        self, db: AsyncSession, numbers: list[str]
    ) -> dict[str, PaperAssignment]:
        found: dict[str, PaperAssignment] = {}
        for chunk in _chunks(numbers):
            result = await db.execute(
                select(PaperAssignment).where(PaperAssignment.paper_number.in_(chunk))
            )
            for row in result.scalars().all():
                found[row.paper_number] = row
        return found


class Phase1AppealRepository:
    """Async data access for the ``phase1_appeals`` table."""

    async def replace_for_email(
        self, db: AsyncSession, email_id: int, rows: list[dict]
    ) -> int:
        """Replace every appeal row of ``email_id`` with ``rows``, atomically.

        Existing rows for the email are deleted and ``rows`` inserted in one
        transaction, so a reader never sees a mix of old and new rows. Each
        dict holds ``Phase1Appeal`` column values; ``email_id`` is taken from
        the argument, never from the dict. An empty ``rows`` simply clears the
        email's rows. Returns the number of rows inserted.
        """
        try:
            await db.execute(
                delete(Phase1Appeal).where(Phase1Appeal.email_id == email_id)
            )
            for row in rows:
                values = {k: v for k, v in row.items() if k not in ("id", "email_id")}
                db.add(Phase1Appeal(email_id=email_id, **values))
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        return len(rows)

    async def delete_for_email(self, db: AsyncSession, email_id: int) -> int:
        """Delete every appeal row of ``email_id``; return how many were removed."""
        try:
            result = await db.execute(
                delete(Phase1Appeal).where(Phase1Appeal.email_id == email_id)
            )
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        return result.rowcount or 0

    async def list_all(self, db: AsyncSession) -> list[Phase1Appeal]:
        """Every appeal row, ordered by ticket then submission number.

        NULL ticket ids and NULL submission numbers sort last on every
        dialect (SQLite and Postgres disagree by default); ``id`` breaks ties
        so the order is fully deterministic.
        """
        result = await db.execute(
            select(Phase1Appeal).order_by(
                Phase1Appeal.zendesk_ticket_id.asc().nulls_last(),
                Phase1Appeal.submission_number.asc().nulls_last(),
                Phase1Appeal.id,
            )
        )
        return list(result.scalars().all())
