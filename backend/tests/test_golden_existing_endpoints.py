"""Golden no-regression check for existing read endpoints (Z3a).

The Reject Appeals queue is a new VIEW and must not change any existing
response. The response bodies below were captured BEFORE the Z3a edits, on the
seeded throwaway SQLite database built here, and are compared byte for byte.

Covered: /emails/queue (three parameter sets), /emails/queue/openreview,
/emails/queue/facets, /emails/{id} (non-candidates only, so the OpenReview
readers lookup never runs) and /config.

To re-capture, the golden file must be regenerated on the code BEFORE a change,
never after it; otherwise this test proves nothing.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import main
from app.core.config import settings
from app.db.database import Base, get_db
from app.db.models import Email

GOLDEN_PATH = Path(__file__).parent / "golden" / "existing_endpoints_z3a.json"

T0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _at(hours: int) -> datetime:
    return T0.replace(hour=12 + hours)


def _email(**kw) -> Email:
    received = kw.pop("received")
    base = dict(
        sender="a@example.org",
        sender_name="Author Example",
        subject="subject",
        body="body",
        status="DRAFT_GENERATED",
        routing={"lane": "human_review"},
        received_at=received,
        created_at=received,
        updated_at=received,
    )
    base.update(kw)
    return Email(**base)


def seed_rows() -> list[Email]:
    """Deterministic rows covering the shapes the appeals queue reads."""
    return [
        _email(  # 1: toy row, no Zendesk fields
            received=_at(0),
            subject="toy deadline question",
            classification={"intent": "submission_requirements", "confidence": 0.9},
            source="toy_dataset",
        ),
        _email(  # 2: review appeal, composed reply, list-shape extraction
            received=_at(1),
            subject="appeal of rejection",
            classification={"intent": "review_decision_appeal", "confidence": 0.8},
            draft={
                "draft_text": "Dear Author Example,\n\nReply.\n\nBest Regards,\nAAAI 2027 PC Team",
                "notes_for_chair": None,
                "appeal_reply": {"mode": "merged", "reasons": ["wrong_paper_review"], "block_ids": ["b1"]},
            },
            extraction={
                "submission_numbers": ["12345"],
                "openreview_forum_ids": ["Ab3xY9kLm2"],
                "authors": [],
                "method": "llm_distiller",
                "appeal_reason": ["wrong_paper_review"],
                "is_reciprocal_dispute": False,
            },
            source="zendesk",
            zendesk_ticket_id=9001,
            zendesk_status="open",
            zendesk_created_at=_at(1),
        ),
        _email(  # 3: desk-reject appeal, OLD single-value extraction, no appeal_reply
            received=_at(2),
            subject="desk reject question",
            classification={"intent": "desk_reject_appeal", "confidence": 0.7},
            draft={"draft_text": "A model draft.", "notes_for_chair": None},
            extraction={
                "submission_number": "4427",
                "openreview_forum_id": None,
                "authors": [],
                "method": "llm_distiller",
            },
            source="zendesk",
            zendesk_ticket_id=9002,
            zendesk_status="new",
            zendesk_created_at=_at(2),
        ),
        _email(  # 4: solved non-appeal
            received=_at(3),
            subject="cms login",
            classification={"intent": "cms_support", "confidence": 0.9},
            source="zendesk",
            zendesk_ticket_id=9003,
            zendesk_status="solved",
            zendesk_created_at=_at(3),
        ),
        _email(  # 5: OpenReview reply candidate (no note id, so no live lookup)
            received=_at(4),
            subject="Re: OpenReview notification",
            classification={"intent": "review_submission_help", "confidence": 0.6},
            extraction={
                "submission_numbers": [],
                "openreview_forum_ids": ["Zz9yY8xX77"],
                "openreview_reply_candidate": True,
                "method": "regex_fallback",
            },
            source="zendesk",
            zendesk_ticket_id=9004,
            zendesk_status="open",
            zendesk_created_at=_at(4),
        ),
        _email(  # 6: reciprocal desk-reject appeal, placeholder draft
            received=_at(5),
            subject="reciprocal reviewer duty",
            classification={"intent": "desk_reject_appeal", "confidence": 0.9},
            draft={
                "draft_text": "[CHAIR: reciprocal complaint; see note]",
                "notes_for_chair": "Reciprocal complaint.",
                "appeal_reply": {"mode": "reciprocal_review", "reasons": ["reciprocal_dispute"], "block_ids": []},
            },
            extraction={
                "submission_numbers": ["777", "778"],
                "openreview_forum_ids": [],
                "is_reciprocal_dispute": True,
                "method": "llm_distiller",
            },
            source="zendesk",
            zendesk_ticket_id=9005,
            zendesk_status="pending",
            zendesk_created_at=_at(5),
        ),
    ]


REQUESTS = [
    ("queue_default", "/api/v1/emails/queue"),
    ("queue_open_paged", "/api/v1/emails/queue?zendesk_status=open&limit=2&offset=0"),
    ("queue_search", "/api/v1/emails/queue?search=appeal"),
    ("queue_openreview", "/api/v1/emails/queue/openreview"),
    ("queue_facets", "/api/v1/emails/queue/facets"),
    ("email_1", "/api/v1/emails/1"),
    ("email_2", "/api/v1/emails/2"),
    ("email_3", "/api/v1/emails/3"),
    ("email_6", "/api/v1/emails/6"),
    ("config", "/api/v1/config"),
]


def pin_settings(setter) -> None:
    """Fix every setting the covered responses read. ``setter(name, value)``."""
    setter("ZENDESK_SUBDOMAIN", "example")
    setter("ALLOW_AUTO_SEND", False)
    setter("OPENREVIEW_USERNAME", None)
    setter("OPENREVIEW_PASSWORD", None)


async def collect_responses() -> dict:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with factory() as session:
        for row in seed_rows():
            session.add(row)
        await session.commit()

    async def _override_get_db():
        async with factory() as session:
            yield session

    main.app.dependency_overrides[get_db] = _override_get_db
    out = {}
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            for name, path in REQUESTS:
                resp = await client.get(path)
                out[name] = {"status": resp.status_code, "body": resp.content.decode("utf-8")}
    finally:
        main.app.dependency_overrides.clear()
        await engine.dispose()
    return out


@pytest.mark.parametrize("name", [n for n, _ in REQUESTS])
async def test_existing_endpoint_response_is_byte_identical(monkeypatch, name):
    pin_settings(lambda key, value: monkeypatch.setattr(settings, key, value))
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    current = await collect_responses()
    assert current[name] == golden[name]
