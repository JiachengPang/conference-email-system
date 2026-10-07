"""Reject Appeals queue: the ai_suggestion group (reject-appeal Phase 4, 2c).

A flagged AI suggestion has its own mode group: its own filter value and its own
count, never inside chair_writes, and no row counted twice. Reuses the queue
tests' ``ctx`` (throwaway in-memory SQLite; the Postgres variant skips itself
unless TEST_DATABASE_URL points at a disposable database — never set here).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.models import appeal_queue
from tests.test_reject_appeals_queue import _draft, _email, ctx  # noqa: F401 - ctx is a fixture

FLAG = "[CHAIR: AI-written suggestion, not approved wording; review and edit before sending]"
BASE_COUNTS = {"composed": 1, "chair_writes": 2, "investigate": 1, "reciprocal": 1,
               "not_drafted": 2, "ai_suggestion": 0}


async def _add_ai_suggestion(ctx, n=40, **kw):
    factory = async_sessionmaker(ctx.engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        base = dict(subject=f"ai suggestion {n}", classification={"intent": "review_decision_appeal"},
                    draft=_draft("ai_suggestion", f"{FLAG}\n\nDear Author,\n\nReply.",
                                 reply={"source": "phase1",
                                        "ai_suggestion": {"base_mode": "chair_writes"}}),
                    source="zendesk", zendesk_ticket_id=9000 + n, zendesk_status="open")
        base.update(kw)
        session.add(_email(n, **base))
        await session.commit()


async def _counts(ctx) -> dict:
    resp = await ctx.client.get("/api/v1/appeals/queue/counts")
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _subjects(ctx, group) -> list[str]:
    resp = await ctx.client.get(f"/api/v1/appeals/queue?mode_group={group}")
    assert resp.status_code == 200, resp.text
    return [e["subject"] for e in resp.json()["emails"]]


def test_the_group_exists_in_python_and_the_api():
    from app.api.v1.appeals import ModeGroup

    assert "ai_suggestion" in appeal_queue.MODE_GROUPS
    assert "ai_suggestion" in ModeGroup.__args__
    assert appeal_queue.mode_group("ai_suggestion") == "ai_suggestion"
    assert appeal_queue.mode_group("chair_writes") == "chair_writes"


async def test_the_counts_key_is_present_at_zero(ctx):
    assert (await _counts(ctx))["by_mode_group"] == BASE_COUNTS


async def test_an_ai_suggestion_is_counted_in_its_own_group_only(ctx):
    await _add_ai_suggestion(ctx)
    counts = await _counts(ctx)
    assert counts["by_mode_group"] == {**BASE_COUNTS, "ai_suggestion": 1}
    assert counts["total"] == 8
    assert sum(counts["by_mode_group"].values()) == counts["total"]
    # Drafted (not not_drafted), on an open ticket, no note yet: it needs one.
    assert counts["needs_note"] == 4
    assert counts["without_approved_draft"] == 2


async def test_the_filter_returns_only_the_ai_suggestion_and_chair_writes_excludes_it(ctx):
    await _add_ai_suggestion(ctx, 40)
    await _add_ai_suggestion(ctx, 41)
    assert await _subjects(ctx, "ai_suggestion") == ["ai suggestion 41", "ai suggestion 40"]
    chair = await _subjects(ctx, "chair_writes")
    assert chair == ["future eight", "chair writes four"]
    assert not any(s.startswith("ai suggestion") for s in chair)


async def test_no_row_is_in_two_groups(ctx):
    await _add_ai_suggestion(ctx)
    seen: dict[str, str] = {}
    for group in appeal_queue.MODE_GROUPS:
        for subject in await _subjects(ctx, group):
            assert subject not in seen, (subject, seen.get(subject), group)
            seen[subject] = group
    total = (await ctx.client.get("/api/v1/appeals/queue")).json()["total"]
    assert len(seen) == total == 8
    assert seen["ai suggestion 40"] == "ai_suggestion"


async def test_a_row_reports_its_group(ctx):
    await _add_ai_suggestion(ctx)
    rows = (await ctx.client.get("/api/v1/appeals/queue?mode_group=ai_suggestion")).json()["emails"]
    assert [r["appeal"]["mode_group"] for r in rows] == ["ai_suggestion"]
    assert [r["appeal"]["mode"] for r in rows] == ["ai_suggestion"]


async def test_counts_still_404_with_the_queue_flag_off(ctx, monkeypatch):
    monkeypatch.setattr(settings, "REJECT_APPEALS_QUEUE_ENABLED", False)
    for path in ("/api/v1/appeals/queue/counts", "/api/v1/appeals/queue?mode_group=ai_suggestion"):
        assert (await ctx.client.get(path)).status_code == 404
