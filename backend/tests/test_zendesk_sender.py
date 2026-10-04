"""Tests for the Zendesk write-back transport (Piece 5).

Fully hermetic: no real Zendesk. A fake async httpx client records every PUT and
returns canned responses; credentials are a stub provider. These assert the
exact payload shapes (§4) and the safe_update / 409 tag behavior.
"""

from types import SimpleNamespace

import pytest

from app.integrations.zendesk.sender import (
    ZendeskConflictError,
    ZendeskSender,
    ZendeskSendError,
)

# These tests exercise the REAL ZendeskSender.add_comment against a fake HTTP
# client, so they opt out of conftest's guard that makes add_comment raise.
pytestmark = pytest.mark.zendesk_transport


class FakeProvider:
    base_url = "https://aaai.zendesk.com/api/v2"

    def get_auth_header(self):
        return {"Authorization": "Bearer test-token"}


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "body"

    def json(self):
        return self._payload


class FakeAsyncClient:
    """Records PUTs and returns responses chosen by a responder(url, json)."""

    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    async def put(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._responder(url, json)

    async def aclose(self):
        return None

    # convenience views
    def ticket_calls(self):
        return [c for c in self.calls if c["url"].endswith("/tickets/123.json")]

    def tag_calls(self):
        return [c for c in self.calls if c["url"].endswith("/tickets/123/tags.json")]


def _sender():
    return ZendeskSender(provider=FakeProvider())


def _ok_responder(comment_updated_at="2026-07-15T10:00:00Z", tag_status=200):
    def responder(url, _json):
        if url.endswith("/tags.json"):
            return FakeResponse(tag_status, {})
        return FakeResponse(200, {"ticket": {"updated_at": comment_updated_at}})
    return responder


@pytest.mark.asyncio
async def test_internal_note_payload_shape_and_tags():
    client = FakeAsyncClient(_ok_responder())
    outcome = await _sender().send_reply(
        ticket_id=123,
        html_body="<p>hi</p>",
        public=False,
        set_status=None,
        tags=["ai_drafted"],
        updated_stamp="2026-07-15T09:00:00Z",
        client=client,
    )

    ticket_call = client.ticket_calls()[0]
    assert ticket_call["json"] == {
        "ticket": {"comment": {"html_body": "<p>hi</p>", "public": False}}
    }
    # No status set for an internal note.
    assert "status" not in ticket_call["json"]["ticket"]

    tag_call = client.tag_calls()[0]
    # safe_update uses the POST-comment updated_at (from the comment response),
    # not our stale stamp, so it can't 409 on our own write.
    assert tag_call["json"] == {
        "tags": ["ai_drafted"],
        "updated_stamp": "2026-07-15T10:00:00Z",
        "safe_update": "true",
    }
    assert outcome.mode == "internal_note"
    assert outcome.public is False
    assert outcome.tags_added == ["ai_drafted"]
    assert outcome.tag_conflict is False


@pytest.mark.asyncio
async def test_public_reply_sets_status_solved():
    client = FakeAsyncClient(_ok_responder())
    outcome = await _sender().send_reply(
        ticket_id=123,
        html_body="<p>reply</p>",
        public=True,
        set_status="solved",
        tags=["ai_auto_replied"],
        updated_stamp="2026-07-15T09:00:00Z",
        client=client,
    )
    ticket = client.ticket_calls()[0]["json"]["ticket"]
    assert ticket["comment"]["public"] is True
    assert ticket["status"] == "solved"
    assert outcome.mode == "public_reply"
    assert outcome.tags_added == ["ai_auto_replied"]


@pytest.mark.asyncio
async def test_tag_conflict_409_is_surfaced_not_overwritten():
    client = FakeAsyncClient(_ok_responder(tag_status=409))
    outcome = await _sender().send_reply(
        ticket_id=123,
        html_body="<p>hi</p>",
        public=False,
        set_status=None,
        tags=["ai_drafted"],
        updated_stamp="2026-07-15T09:00:00Z",
        client=client,
    )
    # Reply still sent; tag conflict surfaced; NOT retried/overwritten.
    assert outcome.tag_conflict is True
    assert outcome.tags_added == []
    assert len(client.tag_calls()) == 1  # exactly one attempt, no overwrite


@pytest.mark.asyncio
async def test_comment_write_failure_raises_and_skips_tags():
    def responder(url, _json):
        return FakeResponse(500, {})  # comment write fails

    client = FakeAsyncClient(responder)
    with pytest.raises(ZendeskSendError) as exc:
        await _sender().send_reply(
            ticket_id=123,
            html_body="<p>hi</p>",
            public=False,
            set_status=None,
            tags=["ai_drafted"],
            updated_stamp="2026-07-15T09:00:00Z",
            client=client,
        )
    assert exc.value.status_code == 500
    # A failed comment write must not proceed to tag the ticket.
    assert client.tag_calls() == []


@pytest.mark.asyncio
async def test_add_tags_omits_safe_update_without_stamp():
    client = FakeAsyncClient(_ok_responder())
    await _sender().add_tags(client, 123, ["ai_drafted"], None)
    assert client.tag_calls()[0]["json"] == {"tags": ["ai_drafted"]}


@pytest.mark.asyncio
async def test_add_tags_409_raises_conflict():
    client = FakeAsyncClient(_ok_responder(tag_status=409))
    with pytest.raises(ZendeskConflictError):
        await _sender().add_tags(client, 123, ["ai_drafted"], "2026-07-15T10:00:00Z")


@pytest.mark.asyncio
async def test_set_status_only_posts_status_and_no_comment():
    client = FakeAsyncClient(_ok_responder())
    outcome = await _sender().set_status_only(
        ticket_id=123,
        status="solved",
        tags=["ai_solved_no_reply"],
        updated_stamp="2026-07-15T09:00:00Z",
        client=client,
    )

    ticket_call = client.ticket_calls()[0]
    # Status-only: the ticket update carries ONLY status — never a comment.
    assert ticket_call["json"] == {"ticket": {"status": "solved"}}
    assert "comment" not in ticket_call["json"]["ticket"]

    tag_call = client.tag_calls()[0]
    # safe_update uses the POST-update updated_at, same as the reply path.
    assert tag_call["json"] == {
        "tags": ["ai_solved_no_reply"],
        "updated_stamp": "2026-07-15T10:00:00Z",
        "safe_update": "true",
    }
    assert outcome.mode == "status_only"
    assert outcome.public is False
    assert outcome.status_set == "solved"
    assert outcome.tags_added == ["ai_solved_no_reply"]
    assert outcome.tag_conflict is False


@pytest.mark.asyncio
async def test_set_status_only_tag_conflict_surfaced_not_overwritten():
    client = FakeAsyncClient(_ok_responder(tag_status=409))
    outcome = await _sender().set_status_only(
        ticket_id=123,
        status="solved",
        tags=["ai_solved_no_reply"],
        updated_stamp="2026-07-15T09:00:00Z",
        client=client,
    )
    # Status already set; tag conflict surfaced; NOT retried/overwritten.
    assert outcome.tag_conflict is True
    assert outcome.tags_added == []
    assert len(client.tag_calls()) == 1


@pytest.mark.asyncio
async def test_set_status_write_failure_raises_and_skips_tags():
    def responder(url, _json):
        return FakeResponse(500, {})  # status write fails

    client = FakeAsyncClient(responder)
    with pytest.raises(ZendeskSendError) as exc:
        await _sender().set_status_only(
            ticket_id=123,
            status="solved",
            tags=["ai_solved_no_reply"],
            updated_stamp="2026-07-15T09:00:00Z",
            client=client,
        )
    assert exc.value.status_code == 500
    # A failed status write must not proceed to tag the ticket.
    assert client.tag_calls() == []
