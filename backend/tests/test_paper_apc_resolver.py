"""The APC resolver for chair notes (Z2a): forum-id route only.

The point these tests guard above all: paper NUMBERS are never used. Whether the
number an author writes equals the sheet's ``paper_number`` is still unverified,
so a number must not be able to pick an APC, even one that would match a row.
Runs on in-memory SQLite; every value is synthetic.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.pipeline import paper_apc_resolver
from app.pipeline.paper_apc_resolver import (
    ApcResolution,
    resolve_apcs_by_forum_id,
    resolve_paper_apcs,
    usable_forum_ids,
)
from app.repositories.phase1_appeal_repository import PaperAssignmentRepository

SHEET = [
    # paper_number, apc_name, forum id
    ("12345", "APC North", "Ab3xY9kLm2"),
    ("67890", "APC South", "Zz9yY8xX77"),
    ("24680", "APC North", "Qq1wW2eE3r"),
    ("13579", "   ", "Bl4nkApc00"),
]


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
        await PaperAssignmentRepository().upsert_many(
            s,
            [
                {
                    "paper_number": number,
                    "apc_name": apc,
                    "openreview_url": f"https://openreview.net/forum?id={fid}",
                    "openreview_forum_id": fid,
                }
                for number, apc, fid in SHEET
            ],
        )
        yield s
    await engine.dispose()


class SpyAssignments(PaperAssignmentRepository):
    """Records forum-id lookups; any paper-number lookup fails the test."""

    def __init__(self):
        self.forum_calls: list[list[str]] = []

    async def get_by_forum_ids(self, db, forum_ids):
        self.forum_calls.append(list(forum_ids))
        return await super().get_by_forum_ids(db, forum_ids)

    async def get_by_numbers(self, db, numbers):  # pragma: no cover - must never run
        raise AssertionError("the resolver looked up a paper NUMBER")


async def test_numbers_alone_resolve_nothing_and_query_nothing(session):
    spy = SpyAssignments()
    result = await resolve_apcs_by_forum_id(
        session, {"submission_numbers": ["12345", "67890"]}, assignments=spy
    )
    assert result == ApcResolution()
    assert spy.forum_calls == []


async def test_a_number_matching_another_row_never_changes_the_apc(session):
    # 67890 is APC South's paper number; the forum id is APC North's paper.
    spy = SpyAssignments()
    result = await resolve_apcs_by_forum_id(
        session,
        {"submission_numbers": ["67890"], "openreview_forum_ids": ["Ab3xY9kLm2"]},
        assignments=spy,
    )
    assert result == ApcResolution(
        apc_names=("APC North",), forum_ids_matched=("Ab3xY9kLm2",)
    )
    assert spy.forum_calls == [["Ab3xY9kLm2"]]


async def test_several_forum_ids_list_every_apc_once_in_email_order(session):
    result = await resolve_apcs_by_forum_id(
        session,
        {
            "openreview_forum_ids": [
                "Zz9yY8xX77",
                "bad id",
                "Ab3xY9kLm2",
                "Qq1wW2eE3r",
                "NotInSheet",
                "Zz9yY8xX77",
            ]
        },
    )
    assert result == ApcResolution(
        apc_names=("APC South", "APC North"),
        forum_ids_matched=("Zz9yY8xX77", "Ab3xY9kLm2", "Qq1wW2eE3r"),
        forum_ids_unmatched=("NotInSheet",),
    )


async def test_forum_ids_match_case_sensitively(session):
    result = await resolve_apcs_by_forum_id(
        session, {"openreview_forum_ids": ["ab3xy9klm2"]}
    )
    assert result == ApcResolution(forum_ids_unmatched=("ab3xy9klm2",))


async def test_surrounding_whitespace_is_trimmed(session):
    result = await resolve_apcs_by_forum_id(
        session, {"openreview_forum_ids": ["  Ab3xY9kLm2 "]}
    )
    assert result.apc_names == ("APC North",)


async def test_a_blank_apc_name_is_not_listed_but_the_forum_id_matched(session):
    result = await resolve_apcs_by_forum_id(
        session, {"openreview_forum_ids": ["Bl4nkApc00"]}
    )
    assert result == ApcResolution(forum_ids_matched=("Bl4nkApc00",))


@pytest.mark.parametrize(
    "extraction",
    [
        None,
        "not a dict",
        {},
        {"openreview_forum_ids": None},
        {"openreview_forum_ids": "Ab3xY9kLm2"},
        {"openreview_forum_ids": [None, 123, "", "short", "Ab3xY9kLm2X", "Ab3x-9kLm2"]},
        {"openreview_forum_id": "Ab3xY9kLm2"},
    ],
)
async def test_unusable_extractions_resolve_nothing_without_a_query(session, extraction):
    spy = SpyAssignments()
    assert await resolve_apcs_by_forum_id(session, extraction, assignments=spy) == ApcResolution()
    assert spy.forum_calls == []


def test_usable_forum_ids_shape_check():
    assert usable_forum_ids(
        {"openreview_forum_ids": ["Ab3xY9kLm2", "Ab3xY9kLm2", "Ab3xY9kLm", "x" * 11, "Zz9yY8xX77"]}
    ) == ["Ab3xY9kLm2", "Zz9yY8xX77"]


def test_the_single_entry_point_is_the_forum_id_route():
    assert resolve_paper_apcs is resolve_apcs_by_forum_id
    assert ApcResolution().route == paper_apc_resolver.ROUTE_FORUM_ID == "forum_id"
