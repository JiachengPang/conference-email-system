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

Reject-appeal assignment (``get_ticket_state``, ``assign_with_note``): a read
of four ticket fields, and ONE update that sets the assignee and adds a private
note together, gated by ``ZENDESK_APPEAL_WRITE_ENABLED`` (typed results in
``ticket_assignment``; the per-email service is ``chair_assignment``).

Credentials come from the same config-driven provider factory the read path
uses, but this sender requests ``ZENDESK_OAUTH_SCOPE`` (``read write``) while the
ingest adapter requests the narrower ``ZENDESK_SYNC_OAUTH_SCOPE`` (``read``). The
OAuth client already has ``read write`` scope (verified in Piece 2).
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.integrations.zendesk import ticket_assignment as ta
from app.integrations.zendesk.credential_provider import (
    ZendeskCredentialProvider,
    get_zendesk_credential_provider,
)

logger = logging.getLogger(__name__)

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

    async def _get(self, client: httpx.AsyncClient, path: str) -> httpx.Response:
        # Same auth handling as _put. No retry here: the caller decides.
        headers = await asyncio.to_thread(self._provider_obj().get_auth_header)
        base = self._provider_obj().base_url
        return await client.get(base + path, headers=headers)

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

    # --- reject-appeal assignment (read side + one write) -----------------------
    # Results and the request body live in ticket_assignment. Neither method
    # retries, logs, or returns any ticket text: only ids, statuses, HTTP codes
    # and Zendesk field names.

    async def get_ticket_state(
        self,
        ticket_id: int,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> ta.TicketStateResult:
        """Read the live status, assignee id, group id and updated_at of a ticket.

        Read only. The ticket body, subject, requester and every other field of
        the response are dropped here and never logged. No retry: a 429 comes
        back as a ``rate_limit`` failure for the caller to handle.
        """
        if not ta.is_positive_int(ticket_id):
            return ta.TicketStateResult(
                failure=ta.refusal(ta.INVALID_INPUT, "Not a valid Zendesk ticket id.")
            )
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        action = "ticket read"
        try:
            try:
                resp = await self._get(client, ta.TICKET_PATH.format(ticket_id=ticket_id))
            except httpx.TransportError as exc:
                return self._failed_read(ticket_id, ta.failure_from_exception(exc, action=action, network=True))
            except Exception as exc:  # noqa: BLE001 - never raise into the caller
                return self._failed_read(ticket_id, ta.failure_from_exception(exc, action=action, network=False))
            data = _json_or_none(resp)
            if resp.status_code >= 400:
                return self._failed_read(
                    ticket_id,
                    ta.failure_from_response(resp.status_code, resp.headers, data, action=action),
                )
            state = ta.state_from_ticket_json(ticket_id, data)
            if state is None:
                return self._failed_read(
                    ticket_id, ta.ZendeskFailure(ta.OTHER, "Zendesk ticket read: unexpected response shape.")
                )
            return ta.TicketStateResult(state=state)
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _failed_read(ticket_id: int, failure: ta.ZendeskFailure) -> ta.TicketStateResult:
        logger.warning("Zendesk ticket %s read failed: %s (HTTP %s)", ticket_id, failure.kind, failure.http_status)
        return ta.TicketStateResult(failure=failure)

    async def assign_with_note(
        self,
        ticket_id: int,
        target: ta.AssigneeTarget | None,
        note_html: str,
        *,
        dry_run: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> ta.AssignResult:
        """Assign the ticket to the chair AND add the private note, in ONE update.

        One PUT carrying exactly ``assignee_id`` and a ``public: false``
        comment, so Zendesk applies both or neither. The comment carries no
        author id (the API account posts it); no status, group or tags are set
        (Zendesk itself may move a New ticket to Open). Refusals, in order:
        bad ticket id or empty note (``invalid_input``); no chair or no user id
        (``chair_missing``); inactive chair (``chair_inactive``). Then a dry run
        returns the exact request WITHOUT sending (works with the write flag
        off); a real call refuses unless ``ZENDESK_APPEAL_WRITE_ENABLED`` is
        True (``write_disabled``). No retry: a failure is returned typed
        (not_found / rejected_assignment / rate_limit / network / other) for the
        caller to show and retry.
        """
        if not ta.is_positive_int(ticket_id) or not isinstance(note_html, str) or not note_html.strip():
            return self._refused(dry_run, ta.INVALID_INPUT, "A valid ticket id and a non-empty note are required.")
        if target is None or not ta.is_positive_int(target.zendesk_user_id):
            return self._refused(dry_run, ta.CHAIR_MISSING, "No Zendesk account is set up for this chair.")
        if target.active is not True:
            return self._refused(dry_run, ta.CHAIR_INACTIVE, "This chair's Zendesk account is inactive.")

        request = ta.build_assign_request(ticket_id, target.zendesk_user_id, note_html)
        if dry_run:
            return ta.AssignResult(ok=True, dry_run=True, request=request)
        if not settings.ZENDESK_APPEAL_WRITE_ENABLED:
            return self._refused(False, ta.WRITE_DISABLED, "Zendesk appeal writes are turned off.")

        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        action = "assignment"
        try:
            try:
                resp = await self._put(client, request.path, request.body)
            except httpx.TransportError as exc:
                return self._failed_write(ticket_id, request, ta.failure_from_exception(exc, action=action, network=True))
            except Exception as exc:  # noqa: BLE001 - never raise into the caller
                return self._failed_write(ticket_id, request, ta.failure_from_exception(exc, action=action, network=False))
            data = _json_or_none(resp)
            if resp.status_code >= 400:
                return self._failed_write(
                    ticket_id,
                    request,
                    ta.failure_from_response(resp.status_code, resp.headers, data, action=action),
                )
            ticket = data.get("ticket") if isinstance(data, dict) else None
            ticket = ticket if isinstance(ticket, dict) else {}
            logger.info("Zendesk ticket %s assigned with a chair note.", ticket_id)
            return ta.AssignResult(
                ok=True,
                dry_run=False,
                request=request,
                ticket_updated_at=ticket.get("updated_at") if isinstance(ticket.get("updated_at"), str) else None,
                assignee_id=ticket.get("assignee_id") if ta.is_positive_int(ticket.get("assignee_id")) else None,
            )
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _refused(dry_run: bool, kind: str, message: str) -> ta.AssignResult:
        logger.info("Zendesk assignment refused: %s", kind)
        return ta.AssignResult(ok=False, dry_run=dry_run, failure=ta.refusal(kind, message))

    @staticmethod
    def _failed_write(ticket_id: int, request: ta.AssignRequest, failure: ta.ZendeskFailure) -> ta.AssignResult:
        logger.warning(
            "Zendesk ticket %s assignment failed: %s (HTTP %s)", ticket_id, failure.kind, failure.http_status
        )
        return ta.AssignResult(ok=False, dry_run=False, request=request, failure=failure)


def _json_or_none(resp) -> object | None:
    """The response JSON, or None when it is not JSON. Never raises."""
    try:
        return resp.json()
    except Exception:  # noqa: BLE001 - a non-JSON error page is just "no details"
        return None
