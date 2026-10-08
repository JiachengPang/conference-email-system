"""The per-email assignment service: chair account lookup + the one update.

In-memory SQLite for the chair accounts, a fake ``httpx`` transport for
Zendesk (zero real calls), synthetic emails only.
"""

import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base
from app.integrations.zendesk import ticket_assignment as ta
from app.integrations.zendesk.chair_assignment import assign_email_to_chair, is_no_chair_case
from app.integrations.zendesk.sender import ZendeskSender
from app.repositories.chair_account_repository import ChairAccountRepository, ChairAccountRow

BASE = "https://example.zendesk.com/api/v2"
NOTE = "<p>Chair note</p>"


class FakeProvider:
    base_url = BASE

    def get_auth_header(self):
        return {"Authorization": "Bearer test-token"}


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        await ChairAccountRepository().upsert_many(
            session,
            [ChairAccountRow("Chair Alpha", 1001, True), ChairAccountRow("Chair Retired", 1002, False)],
        )
        yield session
    await engine.dispose()


class Calls:
    def __init__(self, status=200):
        self.requests: list[httpx.Request] = []
        self.status = status

    def client(self):
        def handler(request):
            self.requests.append(request)
            return httpx.Response(self.status, json={"ticket": {"assignee_id": 1001, "updated_at": "t"}})
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _email(*, ticket=5151, intent="review_decision_appeal", reciprocal=None, mode="merged"):
    return SimpleNamespace(
        zendesk_ticket_id=ticket,
        classification={"intent": intent},
        extraction={"is_reciprocal_dispute": reciprocal},
        draft={"appeal_reply": {"mode": mode}},
    )


async def _run(db, email, chair, *, dry_run, calls):
    async with calls.client() as client:
        return await assign_email_to_chair(
            db, email, chair, NOTE, dry_run=dry_run,
            sender=ZendeskSender(provider=FakeProvider()), client=client,
        )


@pytest.mark.zendesk_transport
@pytest.mark.asyncio
async def test_dry_run_uses_the_chairs_user_id_and_sends_nothing(db):
    calls = Calls()
    result = await _run(db, _email(), "Chair Alpha", dry_run=True, calls=calls)
    assert result.ok and result.dry_run
    assert result.request.body == {"ticket": {"assignee_id": 1001, "comment": {"html_body": NOTE, "public": False}}}
    assert result.request.path == "/tickets/5151.json"
    assert calls.requests == []


@pytest.mark.zendesk_transport
@pytest.mark.asyncio
async def test_real_update_with_the_flag_on(db, monkeypatch):
    monkeypatch.setattr(settings, "ZENDESK_APPEAL_WRITE_ENABLED", True)
    calls = Calls()
    result = await _run(db, _email(), "Chair Alpha", dry_run=False, calls=calls)
    assert result.ok and not result.dry_run
    assert len(calls.requests) == 1
    assert json.loads(calls.requests[0].content)["ticket"]["assignee_id"] == 1001


@pytest.mark.zendesk_transport
@pytest.mark.asyncio
async def test_flag_off_refuses_a_real_update(db):
    calls = Calls()
    result = await _run(db, _email(), "Chair Alpha", dry_run=False, calls=calls)
    assert result.failure.kind == ta.WRITE_DISABLED and calls.requests == []


@pytest.mark.zendesk_transport
@pytest.mark.asyncio
@pytest.mark.parametrize("chair, kind", [("Chair Nobody", ta.CHAIR_MISSING), ("chair alpha", ta.CHAIR_MISSING),
                                         ("Chair Retired", ta.CHAIR_INACTIVE)])
async def test_unknown_and_inactive_chairs_are_refused(db, monkeypatch, chair, kind):
    monkeypatch.setattr(settings, "ZENDESK_APPEAL_WRITE_ENABLED", True)
    calls = Calls()
    for dry_run in (True, False):
        result = await _run(db, _email(), chair, dry_run=dry_run, calls=calls)
        assert result.failure.kind == kind
    assert calls.requests == []


@pytest.mark.zendesk_transport
@pytest.mark.asyncio
@pytest.mark.parametrize("email", [
    _email(intent="desk_reject_appeal"),
    _email(reciprocal=True),
    _email(mode="reciprocal_review"),
    _email(intent="desk_reject_appeal", reciprocal=True),
], ids=["desk-reject", "reciprocal-flag", "reciprocal-mode", "reciprocal-desk-reject"])
async def test_desk_reject_and_reciprocal_cases_are_never_assigned(db, monkeypatch, email):
    monkeypatch.setattr(settings, "ZENDESK_APPEAL_WRITE_ENABLED", True)
    calls = Calls()
    for dry_run in (True, False):
        result = await _run(db, email, "Chair Alpha", dry_run=dry_run, calls=calls)
        assert result.failure.kind == ta.NO_CHAIR_CASE and result.request is None
    assert calls.requests == []


@pytest.mark.zendesk_transport
@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", [None, 0])
async def test_an_email_without_a_ticket_is_refused(db, ticket):
    calls = Calls()
    result = await _run(db, _email(ticket=ticket), "Chair Alpha", dry_run=True, calls=calls)
    assert result.failure.kind == ta.INVALID_INPUT and calls.requests == []


@pytest.mark.zendesk_transport
@pytest.mark.asyncio
async def test_a_rejected_assignment_comes_back_typed(db, monkeypatch):
    monkeypatch.setattr(settings, "ZENDESK_APPEAL_WRITE_ENABLED", True)

    calls = Calls()

    def client():
        def handler(request):
            calls.requests.append(request)
            return httpx.Response(422, json={"details": {"assignee": [{"description": "x"}]}})
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    calls.client = client
    result = await _run(db, _email(), "Chair Alpha", dry_run=False, calls=calls)
    assert result.failure.kind == ta.REJECTED_ASSIGNMENT and len(calls.requests) == 1


def test_no_chair_case_reads_intent_flag_and_mode():
    assert not is_no_chair_case(_email())
    assert not is_no_chair_case(_email(reciprocal=False))
    assert not is_no_chair_case(SimpleNamespace())
    assert is_no_chair_case(_email(intent="desk_reject_appeal"))
    assert is_no_chair_case(_email(reciprocal=True))
    assert is_no_chair_case(_email(mode="reciprocal_review"))


@pytest.mark.asyncio
async def test_unmarked_tests_cannot_reach_the_assignment_transport(db):
    """conftest's guard: without the zendesk_transport marker the client raises."""
    with pytest.raises(AssertionError, match="assign_with_note was called in a test"):
        await assign_email_to_chair(db, _email(), "Chair Alpha", NOTE, dry_run=True)
    with pytest.raises(AssertionError, match="get_ticket_state was called in a test"):
        await ZendeskSender(provider=FakeProvider()).get_ticket_state(5151)


def test_the_write_flag_defaults_to_off():
    from app.core.config import Settings

    assert Settings.model_fields["ZENDESK_APPEAL_WRITE_ENABLED"].default is False
