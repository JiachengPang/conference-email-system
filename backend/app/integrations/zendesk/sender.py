"""Zendesk write-back transport (Piece 5) — comments, status, and tags.

Transport-only: this module makes the real Zendesk API calls and nothing else.
It owns NO database state and NO send-policy decisions — the send gate
(`app/core/send_gate.py`) and the `/emails/{id}/send` endpoint decide whether
and how to send, then hand a finished HTML body here. Keeping it pure transport
mirrors the read adapter (Piece 4) and keeps the policy logic in one place.

Per ZENDESK_API.md §4:
- A reply is a ticket update carrying a ``comment`` (``html_body`` preferred).
  ``public: false`` is an internal note (does not notify the requester);
  ``public: true`` is a real reply, paired with ``status: "solved"``.
- State tags are written through the dedicated tag endpoint (merge, not the
  overwrite-prone ``ticket.tags``), guarded by ``safe_update`` + ``updated_stamp``
  so a concurrent change surfaces as 409 instead of clobbering another writer.

Credentials come from the same config-driven provider factory the read path
uses, but this sender requests ``ZENDESK_OAUTH_SCOPE`` (``read write``) while the
ingest adapter requests the narrower ``ZENDESK_SYNC_OAUTH_SCOPE`` (``read``). The
OAuth client already has ``read write`` scope (verified in Piece 2).
"""

from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.integrations.zendesk.credential_provider import (
    ZendeskCredentialProvider,
    get_zendesk_credential_provider,
)

DEFAULT_TIMEOUT_SECONDS = 30
TICKET_PATH = "/tickets/{ticket_id}.json"
TAGS_PATH = "/tickets/{ticket_id}/tags.json"


class ZendeskSendError(RuntimeError):
    """A Zendesk write failed (network, 4xx, or 5xx). Carries status/body."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ZendeskConflictError(ZendeskSendError):
    """A ``safe_update`` tag write hit 409 — the ticket changed since we read it.

    The caller must re-fetch and decide; it must NOT silently retry-and-overwrite
    (that is exactly the race the dedicated tag endpoint + ``safe_update`` guard
    against, per ZENDESK_API.md §4).
    """


class SendOutcome(BaseModel):
    """Structured result of a write-back, returned to the endpoint layer."""

    mode: str = Field(..., description='"internal_note", "public_reply", or "status_only".')
    public: bool
    status_set: str | None = None
    tags_added: list[str] = Field(default_factory=list)
    # True when the reply landed but the follow-up tag write hit a 409 and was
    # deliberately NOT overwritten — the reply is sent; the tag is re-triable.
    tag_conflict: bool = False
    ticket_updated_at: str | None = None


def _safe_text(resp: httpx.Response) -> str:
    try:
        return resp.text[:2000]
    except Exception:  # noqa: BLE001 - never let error-formatting raise
        return "<unreadable body>"


class ZendeskSender:
    """Posts internal notes / public replies and merges state tags to Zendesk."""

    def __init__(self, *, provider: ZendeskCredentialProvider | None = None) -> None:
        # Built lazily so constructing the sender (e.g. at import) never triggers
        # credential setup; only a real send does.
        self._provider = provider

    def _provider_obj(self) -> ZendeskCredentialProvider:
        if self._provider is None:
            self._provider = get_zendesk_credential_provider(settings)
        return self._provider

    async def _put(self, client: httpx.AsyncClient, path: str, json: dict) -> httpx.Response:
        # get_auth_header may do a blocking token refresh — run it off the loop,
        # same as the read adapter.
        headers = await asyncio.to_thread(self._provider_obj().get_auth_header)
        base = self._provider_obj().base_url
        return await client.put(base + path, json=json, headers=headers)

    async def add_comment(
        self,
        client: httpx.AsyncClient,
        ticket_id: int,
        *,
        html_body: str,
        public: bool,
        set_status: str | None = None,
    ) -> dict:
        """Add a comment (internal note or public reply) via a ticket update.

        Never sets ``ticket.tags`` here (that overwrites the array — §4); tags go
        through :meth:`add_tags`. Returns the updated ticket JSON.
        """
        ticket: dict = {"comment": {"html_body": html_body, "public": public}}
        if set_status is not None:
            ticket["status"] = set_status
        resp = await self._put(client, TICKET_PATH.format(ticket_id=ticket_id), {"ticket": ticket})
        if resp.status_code >= 400:
            raise ZendeskSendError(
                f"Zendesk comment write failed (HTTP {resp.status_code}).",
                status_code=resp.status_code,
                body=_safe_text(resp),
            )
        return resp.json()

    async def set_status(
        self,
        client: httpx.AsyncClient,
        ticket_id: int,
        *,
        status: str,
    ) -> dict:
        """Update ONLY the ticket status — no comment is posted.

        Used by the "mark solved, no reply" path: the ticket is resolved without
        sending anything to the requester. Never sets ``ticket.tags`` here (that
        overwrites the array — §4); tags go through :meth:`add_tags`. Returns the
        updated ticket JSON.
        """
        resp = await self._put(
            client, TICKET_PATH.format(ticket_id=ticket_id), {"ticket": {"status": status}}
        )
        if resp.status_code >= 400:
            raise ZendeskSendError(
                f"Zendesk status write failed (HTTP {resp.status_code}).",
                status_code=resp.status_code,
                body=_safe_text(resp),
            )
        return resp.json()

    async def add_tags(
        self,
        client: httpx.AsyncClient,
        ticket_id: int,
        tags: list[str],
        updated_stamp: str | None,
    ) -> dict:
        """Merge ``tags`` via the dedicated tag endpoint (PUT = add, not replace).

        With ``updated_stamp`` set, sends ``safe_update: "true"`` so a change
        since we last read the ticket fails with 409 (raised as
        :class:`ZendeskConflictError`) rather than clobbering a concurrent writer.
        """
        body: dict = {"tags": tags}
        if updated_stamp:
            body["updated_stamp"] = updated_stamp
            body["safe_update"] = "true"
        resp = await self._put(client, TAGS_PATH.format(ticket_id=ticket_id), body)
        if resp.status_code == 409:
            raise ZendeskConflictError(
                "Tag write conflict (safe_update): ticket changed since last read.",
                status_code=409,
                body=_safe_text(resp),
            )
        if resp.status_code >= 400:
            raise ZendeskSendError(
                f"Zendesk tag write failed (HTTP {resp.status_code}).",
                status_code=resp.status_code,
                body=_safe_text(resp),
            )
        return resp.json()

    async def send_reply(
        self,
        *,
        ticket_id: int,
        html_body: str,
        public: bool,
        set_status: str | None,
        tags: list[str],
        updated_stamp: str | None,
        client: httpx.AsyncClient | None = None,
    ) -> SendOutcome:
        """Post the reply, then merge the state tag. Returns a :class:`SendOutcome`.

        The reply is the primary action and goes first; a comment-write failure
        raises :class:`ZendeskSendError` before any tag is touched. The tag write
        uses ``safe_update`` against the ticket's *post-comment* ``updated_at``
        (from the comment response) so it can't 409 on our own just-made change,
        while still guarding the tiny window against another writer. A tag 409 is
        reported (``tag_conflict=True``), never silently overwritten — the reply
        has already been sent.
        """
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        try:
            ticket = await self.add_comment(
                client, ticket_id, html_body=html_body, public=public, set_status=set_status
            )
            fresh_stamp = (ticket.get("ticket") or {}).get("updated_at") or updated_stamp
            outcome = SendOutcome(
                mode="public_reply" if public else "internal_note",
                public=public,
                status_set=set_status,
                ticket_updated_at=fresh_stamp,
            )
            if tags:
                try:
                    await self.add_tags(client, ticket_id, tags, fresh_stamp)
                    outcome.tags_added = list(tags)
                except ZendeskConflictError:
                    # Reply already sent; do NOT retry-overwrite. Surface it.
                    outcome.tag_conflict = True
            return outcome
        finally:
            if owns_client:
                await client.aclose()

    async def set_status_only(
        self,
        *,
        ticket_id: int,
        status: str,
        tags: list[str],
        updated_stamp: str | None,
        client: httpx.AsyncClient | None = None,
    ) -> SendOutcome:
        """Resolve a ticket by status change alone — NO comment is posted.

        The "mark solved, no reply" transport. Sets the status first (the primary
        action; a failure raises :class:`ZendeskSendError` before any tag is
        touched), then merges the state tag with ``safe_update`` against the
        ticket's *post-update* ``updated_at``. A tag 409 is reported
        (``tag_conflict=True``), never silently overwritten — the status is
        already set. Returns a :class:`SendOutcome` with ``mode="status_only"``.
        """
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        try:
            ticket = await self.set_status(client, ticket_id, status=status)
            fresh_stamp = (ticket.get("ticket") or {}).get("updated_at") or updated_stamp
            outcome = SendOutcome(
                mode="status_only",
                public=False,
                status_set=status,
                ticket_updated_at=fresh_stamp,
            )
            if tags:
                try:
                    await self.add_tags(client, ticket_id, tags, fresh_stamp)
                    outcome.tags_added = list(tags)
                except ZendeskConflictError:
                    # Status already set; do NOT retry-overwrite. Surface it.
                    outcome.tag_conflict = True
            return outcome
        finally:
            if owns_client:
                await client.aclose()
