"""The APC resolver (Z3a rule): number route first, link route as a cross-check.

The rule came from the Z1d check on 184 real September appeals (the number an
author writes equalled the sheet's number 17 of 17 times a link could confirm
it; desk-rejected papers are almost never in the sheet). It REPLACES the Z2a
forum-id-only rule, so the Z2a tests asserting "paper numbers are never used"
were rewritten deliberately.

``combine`` is pure and tested with literals. The database tests run on a
throwaway in-memory SQLite database, and on Postgres too when TEST_DATABASE_URL
names a disposable database. Every chair name is made up.
"""

import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.pipeline import paper_apc_resolver as r
from app.pipeline.paper_apc_resolver import (
    ApcResolution,
    SheetRow,
    combine,
    email_forum_ids,
    email_numbers,
    normalize_paper_number,
    resolve_paper_apcs,
    resolve_paper_apcs_many,
)
from app.repositories.phase1_appeal_repository import PaperAssignmentRepository

NORTH = SheetRow("12345", "APC North")
NORTH_2 = SheetRow("24680", "APC North")
SOUTH = SheetRow("67890", "APC South")
SHORT = SheetRow("8", "APC South")


# --- combine: the rule, pure -------------------------------------------------


def test_both_routes_agree():
    assert combine(
        "review_decision_appeal",
        ["12345"], {"12345": [NORTH]},
        ["Ab3xY9kLm2"], {"Ab3xY9kLm2": NORTH},
    ) == ApcResolution(
        apc_names=("APC North",), source="both", paper_numbers=("12345",),
        warnings=(), forum_ids_matched=("Ab3xY9kLm2",), forum_ids_unmatched=(),
    )


def test_routes_disagree_is_a_conflict_with_no_chair():
    assert combine(
        "review_decision_appeal",
        ["12345"], {"12345": [NORTH]},
        ["Zz9yY8xX77"], {"Zz9yY8xX77": SOUTH},
    ) == ApcResolution(
        apc_names=(), source="conflict", paper_numbers=("12345", "67890"),
        warnings=("conflict", "several_papers"),
        forum_ids_matched=("Zz9yY8xX77",), forum_ids_unmatched=(),
    )


def test_number_route_only():
    assert combine(
        "review_decision_appeal", ["12345"], {"12345": [NORTH]}, [], {}
    ) == ApcResolution(apc_names=("APC North",), source="number", paper_numbers=("12345",))


def test_link_route_only():
    assert combine(
        "review_decision_appeal",
        ["99999"], {},
        ["Zz9yY8xX77", "NotInSheet"], {"Zz9yY8xX77": SOUTH},
    ) == ApcResolution(
        apc_names=("APC South",), source="link", paper_numbers=("67890",),
        forum_ids_matched=("Zz9yY8xX77",), forum_ids_unmatched=("NotInSheet",),
    )


def test_neither_route():
    assert combine("review_decision_appeal", ["99999"], {}, [], {}) == ApcResolution()


def test_desk_reject_number_only_match_is_refused():
    assert combine(
        "desk_reject_appeal", ["12345"], {"12345": [NORTH]}, [], {}
    ) == ApcResolution(apc_names=(), source="none", paper_numbers=(),
                       warnings=("desk_reject_not_in_sheet",))


def test_desk_reject_link_match_is_accepted():
    assert combine(
        "desk_reject_appeal", [], {}, ["Ab3xY9kLm2"], {"Ab3xY9kLm2": NORTH}
    ) == ApcResolution(apc_names=("APC North",), source="link", paper_numbers=("12345",),
                       forum_ids_matched=("Ab3xY9kLm2",))


def test_desk_reject_number_confirmed_by_link_is_accepted():
    assert combine(
        "desk_reject_appeal", ["12345"], {"12345": [NORTH]}, ["Ab3xY9kLm2"], {"Ab3xY9kLm2": NORTH}
    ).source == "both"


def test_several_papers_list_every_chair_and_paper():
    assert combine(
        "review_decision_appeal",
        ["12345", "67890", "24680"],
        {"12345": [NORTH], "67890": [SOUTH], "24680": [NORTH_2]},
        [], {},
    ) == ApcResolution(
        apc_names=("APC North", "APC South"), source="number",
        paper_numbers=("12345", "67890", "24680"), warnings=("several_papers",),
    )


def test_same_chair_through_different_papers_is_flagged():
    # Two routes naming the same chair for DIFFERENT papers: "both", but flagged.
    assert combine(
        "review_decision_appeal",
        ["12345"], {"12345": [NORTH]},
        ["Qq1wW2eE3r"], {"Qq1wW2eE3r": NORTH_2},
    ) == ApcResolution(
        apc_names=("APC North",), source="both", paper_numbers=("12345", "24680"),
        warnings=("several_papers",), forum_ids_matched=("Qq1wW2eE3r",),
    )


def test_short_matched_number_is_flagged():
    assert combine("review_decision_appeal", ["8"], {"8": [SHORT]}, [], {}) == ApcResolution(
        apc_names=("APC South",), source="number", paper_numbers=("8",), warnings=("short_number",),
    )


@pytest.mark.parametrize("number, flagged", [("42", True), ("777", False)])
def test_short_number_boundary_is_three_digits(number, flagged):
    row = SheetRow(number, "APC South")
    warnings = combine("review_decision_appeal", [number], {number: [row]}, [], {}).warnings
    assert ("short_number" in warnings) is flagged


def test_short_unmatched_number_is_not_flagged():
    assert combine("review_decision_appeal", ["8"], {}, [], {}).warnings == ()


def test_short_number_is_flagged_even_when_the_desk_guard_refuses_it():
    assert combine("desk_reject_appeal", ["8"], {"8": [SHORT]}, [], {}).warnings == (
        "desk_reject_not_in_sheet", "short_number",
    )


def test_blank_chair_names_are_not_suggested():
    assert combine(
        "review_decision_appeal", ["12345"], {"12345": [SheetRow("12345", "   ")]}, [], {}
    ) == ApcResolution(apc_names=(), source="number", paper_numbers=("12345",))


# --- normalisation and both stored shapes -----------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [("12345", "12345"), (" 12345 ", "12345"), ("#12345", "12345"), ("# 012345", "12345"),
     ("00777", "777"), ("0", None), ("#", None), ("", None), (None, None)],
)
def test_normalize_paper_number(raw, expected):
    assert normalize_paper_number(raw) == expected


def test_email_numbers_read_the_list_shape_and_normalise():
    assert email_numbers({"submission_numbers": ["#012345", "12345", " 777 "]}) == ["12345", "777"]


@pytest.mark.parametrize(
    "extraction, expected",
    [
        ({"submission_number": "12345"}, ["12345"]),
        ({"submission_number": 12345}, ["12345"]),
        ({"submission_number": "#0777"}, ["777"]),
        ({"submission_number": None}, []),
        ({"submission_number": True}, []),
        ({"submission_numbers": [], "submission_number": "12345"}, []),
        ({}, []),
        (None, []),
    ],
)
def test_email_numbers_read_the_old_single_value_shape(extraction, expected):
    assert email_numbers(extraction) == expected


@pytest.mark.parametrize(
    "extraction, expected",
    [
        ({"openreview_forum_ids": ["Ab3xY9kLm2", "bad id", "Ab3xY9kLm2", " Zz9yY8xX77 "]},
         ["Ab3xY9kLm2", "Zz9yY8xX77"]),
        ({"openreview_forum_id": "Ab3xY9kLm2"}, ["Ab3xY9kLm2"]),
        ({"openreview_forum_id": "short"}, []),
        ({"openreview_forum_ids": "Ab3xY9kLm2"}, []),
        ({"openreview_forum_ids": [None, 123, "Ab3xY9kLm2X", "Ab3x-9kLm2"]}, []),
    ],
)
def test_email_forum_ids_both_shapes_and_shape_check(extraction, expected):
    assert email_forum_ids(extraction) == expected


# --- against a database --------------------------------------------------------------

_PG = os.environ.get("TEST_DATABASE_URL", "")
_PG = _PG if _PG.startswith("postgresql") else None

SHEET = [
    # paper_number, apc_name, forum id
    ("12345", "APC North", "Ab3xY9kLm2"),
    ("67890", "APC South", "Zz9yY8xX77"),
    ("24680", "APC North", "Qq1wW2eE3r"),
    ("0777", "APC South", "Sv7nUmb3r0"),
    ("13579", "   ", "Bl4nkApc00"),
]


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def session(request):
    if request.param == "postgres":
        if not _PG:
            pytest.skip("set TEST_DATABASE_URL to a disposable postgresql:// database")
        url = _PG.replace("postgresql://", "postgresql+asyncpg://", 1) if "+asyncpg" not in _PG else _PG
        engine = create_async_engine(url)
    else:
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        await PaperAssignmentRepository().upsert_many(
            s,
            [
                {"paper_number": n, "apc_name": a,
                 "openreview_url": f"https://openreview.net/forum?id={f}", "openreview_forum_id": f}
                for n, a, f in SHEET
            ],
        )
        yield s
    if request.param == "postgres":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


class CountingAssignments(PaperAssignmentRepository):
    def __init__(self):
        self.number_calls = 0
        self.forum_calls = 0

    async def get_by_normalized_numbers(self, db, numbers):
        self.number_calls += 1
        return await super().get_by_normalized_numbers(db, numbers)

    async def get_by_forum_ids(self, db, forum_ids):
        self.forum_calls += 1
        return await super().get_by_forum_ids(db, forum_ids)


async def test_number_route_matches_with_normalisation_on_both_sides(session):
    # "#012345" → 12345; the sheet stores "0777", the email writes "777".
    result = await resolve_paper_apcs(
        session, {"submission_numbers": ["#012345", "777"]}, intent="review_decision_appeal"
    )
    assert result == ApcResolution(
        apc_names=("APC North", "APC South"), source="number",
        paper_numbers=("12345", "0777"), warnings=("several_papers",),
    )


async def test_old_single_value_shape_resolves_by_number_and_by_link(session):
    by_number = await resolve_paper_apcs(
        session, {"submission_number": "67890"}, intent="review_decision_appeal"
    )
    assert (by_number.source, by_number.apc_names) == ("number", ("APC South",))
    by_link = await resolve_paper_apcs(
        session, {"openreview_forum_id": "Zz9yY8xX77"}, intent="review_decision_appeal"
    )
    assert (by_link.source, by_link.apc_names) == ("link", ("APC South",))
    both = await resolve_paper_apcs(
        session,
        {"submission_number": 67890, "openreview_forum_id": "Zz9yY8xX77"},
        intent="review_decision_appeal",
    )
    assert (both.source, both.apc_names, both.warnings) == ("both", ("APC South",), ())


async def test_a_number_and_a_link_to_different_chairs_is_a_conflict(session):
    result = await resolve_paper_apcs(
        session,
        {"submission_numbers": ["67890"], "openreview_forum_ids": ["Ab3xY9kLm2"]},
        intent="review_decision_appeal",
    )
    assert (result.source, result.apc_names, result.warnings) == (
        "conflict", (), ("conflict", "several_papers"),
    )


async def test_desk_reject_guard_against_the_database(session):
    refused = await resolve_paper_apcs(
        session, {"submission_numbers": ["12345"]}, intent="desk_reject_appeal"
    )
    assert (refused.source, refused.apc_names, refused.warnings) == (
        "none", (), ("desk_reject_not_in_sheet",),
    )
    accepted = await resolve_paper_apcs(
        session, {"openreview_forum_ids": ["Ab3xY9kLm2"]}, intent="desk_reject_appeal"
    )
    assert (accepted.source, accepted.apc_names) == ("link", ("APC North",))


async def test_forum_ids_match_case_sensitively(session):
    result = await resolve_paper_apcs(
        session, {"openreview_forum_ids": ["ab3xy9klm2"]}, intent="review_decision_appeal"
    )
    assert result == ApcResolution(forum_ids_unmatched=("ab3xy9klm2",))


async def test_a_whole_page_is_one_query_per_route(session):
    spy = CountingAssignments()
    results = await resolve_paper_apcs_many(
        session,
        {
            1: ("review_decision_appeal", {"submission_numbers": ["12345"]}),
            2: ("review_decision_appeal", {"submission_number": "67890"}),
            3: ("desk_reject_appeal", {"openreview_forum_ids": ["Qq1wW2eE3r"]}),
            4: ("review_decision_appeal", {"submission_numbers": ["99999"]}),
        },
        assignments=spy,
    )
    assert (spy.number_calls, spy.forum_calls) == (1, 1)
    assert {k: (v.source, v.apc_names) for k, v in results.items()} == {
        1: ("number", ("APC North",)),
        2: ("number", ("APC South",)),
        3: ("link", ("APC North",)),
        4: ("none", ()),
    }


async def test_nothing_to_look_up_means_no_query(session):
    spy = CountingAssignments()
    results = await resolve_paper_apcs_many(
        session, {1: ("review_decision_appeal", {}), 2: (None, None)}, assignments=spy
    )
    assert (spy.number_calls, spy.forum_calls) == (0, 0)
    assert set(results.values()) == {ApcResolution()}


async def test_distinct_apc_names_are_trimmed_sorted_and_blank_free(session):
    await PaperAssignmentRepository().upsert_many(
        session,
        [{"paper_number": "11111", "apc_name": " apc lowercase ",
          "openreview_url": "https://openreview.net/forum?id=Lw3rC4s3aa",
          "openreview_forum_id": "Lw3rC4s3aa"}],
    )
    assert await PaperAssignmentRepository().list_distinct_apc_names(session) == [
        "apc lowercase", "APC North", "APC South",
    ]


async def test_single_email_entry_point_matches_the_page_function(session):
    extraction = {"submission_numbers": ["12345"], "openreview_forum_ids": ["Ab3xY9kLm2"]}
    single = await r.resolve_paper_apcs(session, extraction, intent="desk_reject_appeal")
    page = await r.resolve_paper_apcs_many(session, {"k": ("desk_reject_appeal", extraction)})
    assert single == page["k"]
