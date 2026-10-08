"""Persistence for chair Zendesk accounts (``zendesk_chair_accounts``).

One row per chair, keyed by the chair name exactly as the paper-to-chair sheet
has it. :meth:`ChairAccountRepository.upsert_many` is idempotent: loading the
same file twice writes nothing the second time. Reads return ``None`` / ``[]``
on a miss rather than raising, like the other repositories. Writes commit
before returning (rolling back and re-raising on failure).

Chair names and Zendesk user ids are personal data: nothing here logs them.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ZendeskChairAccount


@dataclass(frozen=True)
class ChairAccountRow:
    """One chair account as the loader hands it over (already validated)."""

    chair_name: str
    zendesk_user_id: int
    active: bool = True


@dataclass(frozen=True)
class UpsertStats:
    """What one upsert did. ``unchanged`` rows were already exactly as given."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0


class ChairAccountRepository:
    """Async data access for the ``zendesk_chair_accounts`` table."""

    async def upsert_many(self, db: AsyncSession, rows: list[ChairAccountRow]) -> UpsertStats:
        """Insert or update rows keyed by ``chair_name``; one transaction.

        A row whose stored values already equal the given ones is left alone
        (so ``updated_at`` does not move and a re-run reports it ``unchanged``).
        Rows not named in ``rows`` are never touched or deactivated. When the
        same name appears twice in ``rows`` the last one wins (the loader
        refuses duplicates before it gets here).
        """
        by_name: dict[str, ChairAccountRow] = {}
        for row in rows:
            by_name[row.chair_name] = row
        if not by_name:
            return UpsertStats()

        inserted = updated = unchanged = 0
        try:
            result = await db.execute(
                select(ZendeskChairAccount).where(
                    ZendeskChairAccount.chair_name.in_(list(by_name))
                )
            )
            existing = {acc.chair_name: acc for acc in result.scalars().all()}
            for name, row in by_name.items():
                current = existing.get(name)
                if current is None:
                    db.add(
                        ZendeskChairAccount(
                            chair_name=name,
                            zendesk_user_id=row.zendesk_user_id,
                            active=row.active,
                        )
                    )
                    inserted += 1
                elif (current.zendesk_user_id, current.active) == (
                    row.zendesk_user_id,
                    row.active,
                ):
                    unchanged += 1
                else:
                    current.zendesk_user_id = row.zendesk_user_id
                    current.active = row.active
                    updated += 1
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        return UpsertStats(inserted=inserted, updated=updated, unchanged=unchanged)

    async def get_by_name(self, db: AsyncSession, chair_name: str) -> ZendeskChairAccount | None:
        """The account for this exact chair name, or ``None``."""
        if not isinstance(chair_name, str) or not chair_name:
            return None
        result = await db.execute(
            select(ZendeskChairAccount).where(ZendeskChairAccount.chair_name == chair_name)
        )
        return result.scalar_one_or_none()

    async def list_all(self, db: AsyncSession) -> list[ZendeskChairAccount]:
        """Every account, ordered by chair name."""
        result = await db.execute(
            select(ZendeskChairAccount).order_by(ZendeskChairAccount.chair_name)
        )
        return list(result.scalars().all())
