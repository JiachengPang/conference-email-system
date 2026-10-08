"""The reject-appeal ticket read and the one assign-and-note update (client layer).

Fake transport only (``httpx.MockTransport``): no request ever leaves the
process, and every test can count the requests made. Credentials are a stub.
Synthetic data only.
"""

import json
import logging

import httpx
import pytest

from app.core.config import settings
from app.integrations.zendesk import ticket_assignment as ta
from app.integrations.zendesk.sender import ZendeskSender

# These exercise the real client methods against a fake transport, so they opt
# out of conftest's guard that makes them raise.
pytestmark = pytest.mark.zendesk_transport

BASE = "https://example.zendesk.com/api/v2"
TICKET = 4242
USER = 777001
NOTE = "<p><strong>ConfMail draft (not sent to the author)</strong></p><p>Note body</p>"
# Text that only the "live ticket" holds: it must never come back or be logged.
LEAK = "SECRET-REQUESTER-TEXT"
ACTIVE = ta.AssigneeTarget(chair_name="Chair Alpha", zendesk_user_id=USER, active=True)


class FakeProvider:
    base_url = BASE

    def get_auth_header(self):
        return {"Authorization": "Bearer test-token"}


class Recorder:
    """A fake transport: answers with ``respond(request)`` and records every request."""

    def __init__(self, respond):
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def _sender() -> ZendeskSender:
    return ZendeskSender(provider=FakeProvider())


def _ticket_json(**overrides) -> dict:
    ticket = {
        "id": TICKET, "status": "new", "assignee_id": None, "group_id": 360001,
        "updated_at": "2026-10-07T10:00:00Z",
        "subject": LEAK, "description": LEAK, "requester_id": 55, "via": {"source": {"from": {"name": LEAK}}},
        "custom_fields": [{"id": 1, "value": LEAK}],
    }
    ticket.update(overrides)
    return {"ticket": ticket}


@pytest.fixture
def write_on(monkeypatch):
    monkeypatch.setattr(settings, "ZENDESK_APPEAL_WRITE_ENABLED", True)


# --- the request body ------------------------------------------------------------------------


def test_the_body_has_exactly_one_assignee_and_one_private_comment():
    req = ta.build_assign_request(TICKET, USER, NOTE)
    assert (req.method, req.path) == ("PUT", f"/tickets/{TICKET}.json")
    assert req.body == {"ticket": {"assignee_id": USER, "comment": {"html_body": NOTE, "public": False}}}
    assert set(req.body) == {"ticket"}
    assert set(req.body["ticket"]) == {"assignee_id", "comment"}
    assert set(req.body["ticket"]["comment"]) == {"html_body", "public"}
    assert req.body["ticket"]["comment"]["public"] is False


# --- the reader ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reader_returns_only_the_four_fields_and_no_ticket_text(caplog):
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json(assignee_id=12, group_id=34, status="open")))
    caplog.set_level(logging.DEBUG)
    async with rec.client() as client:
        result = await _sender().get_ticket_state(TICKET, client=client)
    assert result.ok and result.failure is None
    assert result.state == ta.TicketState(
        ticket_id=TICKET, status="open", assignee_id=12, group_id=34, updated_at="2026-10-07T10:00:00Z"
    )
    assert [(r.method, str(r.url)) for r in rec.requests] == [("GET", f"{BASE}/tickets/{TICKET}.json")]
    assert rec.requests[0].headers["Authorization"] == "Bearer test-token"
    assert LEAK not in repr(result) and LEAK not in caplog.text


@pytest.mark.asyncio
async def test_reader_keeps_unassigned_and_ignores_wrong_types():
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json(assignee_id=None, group_id="x", status=5)))
    async with rec.client() as client:
        result = await _sender().get_ticket_state(TICKET, client=client)
    assert result.state == ta.TicketState(TICKET, None, None, None, "2026-10-07T10:00:00Z")


@pytest.mark.asyncio
@pytest.mark.parametrize("status, kind", [(404, ta.NOT_FOUND), (429, ta.RATE_LIMIT), (500, ta.OTHER),
                                          (403, ta.OTHER)])
async def test_reader_failures_are_typed_and_carry_no_body_text(status, kind, caplog):
    rec = Recorder(lambda r: httpx.Response(status, json={"error": LEAK, "description": LEAK},
                                            headers={"Retry-After": "7"}))
    async with rec.client() as client:
        result = await _sender().get_ticket_state(TICKET, client=client)
    assert not result.ok and result.state is None
    assert (result.failure.kind, result.failure.http_status) == (kind, status)
    assert LEAK not in repr(result) and LEAK not in caplog.text
    assert len(rec.requests) == 1, "no retry inside the reader"


@pytest.mark.asyncio
async def test_reader_network_failure():
    def boom(request):
        raise httpx.ConnectError(f"cannot reach {LEAK}", request=request)

    rec = Recorder(boom)
    async with rec.client() as client:
        result = await _sender().get_ticket_state(TICKET, client=client)
    assert result.failure.kind == ta.NETWORK and LEAK not in result.failure.message


@pytest.mark.asyncio
async def test_reader_unexpected_shape_is_other():
    rec = Recorder(lambda r: httpx.Response(200, json={"tickets": []}))
    async with rec.client() as client:
        result = await _sender().get_ticket_state(TICKET, client=client)
    assert result.failure.kind == ta.OTHER


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1, None, "4242", True])
async def test_reader_refuses_a_bad_ticket_id_without_a_call(bad):
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json()))
    async with rec.client() as client:
        result = await _sender().get_ticket_state(bad, client=client)
    assert result.failure.kind == ta.INVALID_INPUT and rec.requests == []


# --- the update: success -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_sends_one_put_with_the_exact_body(write_on, caplog):
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json(assignee_id=USER, status="open",
                                                                    updated_at="2026-10-07T10:05:00Z")))
    caplog.set_level(logging.DEBUG)
    async with rec.client() as client:
        result = await _sender().assign_with_note(TICKET, ACTIVE, NOTE, client=client)
    assert result.ok and not result.dry_run and result.failure is None
    assert (result.assignee_id, result.ticket_updated_at) == (USER, "2026-10-07T10:05:00Z")
    assert len(rec.requests) == 1
    sent = rec.requests[0]
    assert (sent.method, str(sent.url)) == ("PUT", f"{BASE}/tickets/{TICKET}.json")
    assert json.loads(sent.content) == {
        "ticket": {"assignee_id": USER, "comment": {"html_body": NOTE, "public": False}}
    }
    assert json.loads(sent.content) == result.request.body
    assert LEAK not in repr(result) and LEAK not in caplog.text
    assert "Chair Alpha" not in caplog.text and str(USER) not in caplog.text


# --- the update: refusals (nothing sent) ---------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_refuses_and_sends_nothing():
    assert settings.ZENDESK_APPEAL_WRITE_ENABLED is False  # conftest keeps it off
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json()))
    async with rec.client() as client:
        result = await _sender().assign_with_note(TICKET, ACTIVE, NOTE, client=client)
    assert not result.ok and result.failure.kind == ta.WRITE_DISABLED
    assert rec.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("target, kind", [
    (None, ta.CHAIR_MISSING),
    (ta.AssigneeTarget("Chair Alpha", None, True), ta.CHAIR_MISSING),
    (ta.AssigneeTarget("Chair Alpha", 0, True), ta.CHAIR_MISSING),
    (ta.AssigneeTarget("Chair Alpha", USER, False), ta.CHAIR_INACTIVE),
], ids=["no-account", "no-user-id", "zero-user-id", "inactive"])
async def test_missing_or_inactive_chairs_are_refused(write_on, target, kind, dry_run):
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json()))
    async with rec.client() as client:
        result = await _sender().assign_with_note(TICKET, target, NOTE, dry_run=dry_run, client=client)
    assert not result.ok and result.failure.kind == kind and result.request is None
    assert rec.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket_id, note", [(0, NOTE), (None, NOTE), (True, NOTE), (TICKET, ""),
                                             (TICKET, "   "), (TICKET, None)])
async def test_bad_inputs_are_refused(write_on, ticket_id, note):
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json()))
    async with rec.client() as client:
        result = await _sender().assign_with_note(ticket_id, ACTIVE, note, client=client)
    assert result.failure.kind == ta.INVALID_INPUT and rec.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
async def test_dry_run_builds_the_exact_request_and_sends_nothing(monkeypatch, flag):
    monkeypatch.setattr(settings, "ZENDESK_APPEAL_WRITE_ENABLED", flag)
    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json()))
    async with rec.client() as client:
        result = await _sender().assign_with_note(TICKET, ACTIVE, NOTE, dry_run=True, client=client)
    assert result.ok and result.dry_run and result.failure is None
    assert result.request == ta.build_assign_request(TICKET, USER, NOTE)
    assert rec.requests == [], "a dry run must make zero calls"


@pytest.mark.asyncio
async def test_dry_run_without_a_client_never_opens_one(monkeypatch):
    def no_client(*args, **kwargs):
        raise AssertionError("a dry run opened an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", no_client)
    result = await _sender().assign_with_note(TICKET, ACTIVE, NOTE, dry_run=True)
    assert result.ok and result.dry_run


# --- the update: typed failures ------------------------------------------------------------------


def _error(status, body=None, headers=None):
    return lambda r: httpx.Response(status, json=body if body is not None else {"error": LEAK},
                                    headers=headers or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("respond, kind, status, retry", [
    (_error(404), ta.NOT_FOUND, 404, None),
    (_error(422, {"error": "RecordInvalid", "description": LEAK,
                  "details": {"assignee": [{"description": f"{LEAK} is not a member of the group"}]}}),
     ta.REJECTED_ASSIGNMENT, 422, None),
    (_error(422, {"error": "RecordInvalid", "details": {"group_id": [{"description": LEAK}]}}),
     ta.REJECTED_ASSIGNMENT, 422, None),
    (_error(422, {"error": "RecordInvalid", "details": {"status": [{"description": LEAK}]}}),
     ta.OTHER, 422, None),
    (_error(429, headers={"Retry-After": "30"}), ta.RATE_LIMIT, 429, 30),
    (_error(429, headers={"Retry-After": "soon"}), ta.RATE_LIMIT, 429, None),
    (_error(500), ta.OTHER, 500, None),
    (_error(401), ta.OTHER, 401, None),
    (lambda r: httpx.Response(502, text=f"<html>{LEAK}</html>"), ta.OTHER, 502, None),
], ids=["not-found", "assignee-rejected", "group-rejected", "closed-ticket-422", "rate-limit",
        "rate-limit-bad-header", "server-error", "unauthorized", "non-json"])
async def test_update_failures_are_typed_with_safe_messages(write_on, caplog, respond, kind, status, retry):
    rec = Recorder(respond)
    caplog.set_level(logging.DEBUG)
    async with rec.client() as client:
        result = await _sender().assign_with_note(TICKET, ACTIVE, NOTE, client=client)
    assert not result.ok and not result.dry_run
    assert (result.failure.kind, result.failure.http_status, result.failure.retry_after_seconds) == (
        kind, status, retry)
    assert result.request == ta.build_assign_request(TICKET, USER, NOTE)
    assert LEAK not in result.failure.message and LEAK not in repr(result.failure)
    assert LEAK not in caplog.text
    assert len(rec.requests) == 1, "no retry inside the update"


@pytest.mark.asyncio
async def test_rejected_assignment_message_names_the_field_only(write_on):
    rec = Recorder(_error(422, {"details": {"assignee": [{"description": LEAK}]}}))
    async with rec.client() as client:
        result = await _sender().assign_with_note(TICKET, ACTIVE, NOTE, client=client)
    assert result.failure.message == "Zendesk assignment: the assignment was rejected (HTTP 422; fields: assignee)."


@pytest.mark.asyncio
@pytest.mark.parametrize("exc, kind", [(httpx.ConnectError, ta.NETWORK), (httpx.ReadTimeout, ta.NETWORK)])
async def test_network_failures(write_on, exc, kind):
    def boom(request):
        raise exc(f"failed on {LEAK}", request=request)

    rec = Recorder(boom)
    async with rec.client() as client:
        result = await _sender().assign_with_note(TICKET, ACTIVE, NOTE, client=client)
    assert result.failure.kind == kind and LEAK not in result.failure.message
    assert len(rec.requests) == 1


@pytest.mark.asyncio
async def test_a_credential_error_is_other_and_never_raises(write_on):
    class BrokenProvider(FakeProvider):
        def get_auth_header(self):
            raise RuntimeError(LEAK)

    rec = Recorder(lambda r: httpx.Response(200, json=_ticket_json()))
    async with rec.client() as client:
        result = await ZendeskSender(provider=BrokenProvider()).assign_with_note(
            TICKET, ACTIVE, NOTE, client=client)
    assert result.failure.kind == ta.OTHER and LEAK not in result.failure.message
    assert rec.requests == []
