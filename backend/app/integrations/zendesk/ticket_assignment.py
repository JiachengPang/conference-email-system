"""Typed results and pure helpers for the reject-appeal ticket assignment.

The Zendesk calls themselves live on the one Zendesk client,
:class:`app.integrations.zendesk.sender.ZendeskSender`
(``get_ticket_state`` and ``assign_with_note``). This module holds only what
those calls return and the pure pieces they use, so the request body and the
failure mapping can be tested without any HTTP:

- :func:`build_assign_request`: THE request body. One ticket update that sets
  the assignee AND adds a private (``public: false``) comment, so both happen
  or neither does. No author id (the API account posts it), no status, no
  group, no tags.
- :func:`state_from_ticket_json`: the four fields the reader keeps.
- :func:`failure_from_response` / :func:`failure_from_exception`: a typed
  failure with a SAFE message (fixed text, HTTP status and Zendesk field names
  only). The response body, the ticket text and the requester are never copied
  into a result, a message or a log line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TICKET_PATH = "/tickets/{ticket_id}.json"

# Transport failures (the call was made, or tried, and did not succeed).
NOT_FOUND = "not_found"
REJECTED_ASSIGNMENT = "rejected_assignment"
RATE_LIMIT = "rate_limit"
NETWORK = "network"
OTHER = "other"
TRANSPORT_FAILURES: tuple[str, ...] = (NOT_FOUND, REJECTED_ASSIGNMENT, RATE_LIMIT, NETWORK, OTHER)

# Refusals (nothing was sent).
WRITE_DISABLED = "write_disabled"
CHAIR_MISSING = "chair_missing"
CHAIR_INACTIVE = "chair_inactive"
INVALID_INPUT = "invalid_input"
NO_CHAIR_CASE = "no_chair_case"
REFUSALS: tuple[str, ...] = (WRITE_DISABLED, CHAIR_MISSING, CHAIR_INACTIVE, INVALID_INPUT, NO_CHAIR_CASE)

# Zendesk's 422 ``details`` keys that mean "this assignment is not allowed"
# (e.g. the agent is not in the ticket's group). Any other 422 is ``other``.
_ASSIGNMENT_FIELDS = frozenset({"assignee", "assignee_id", "group", "group_id"})


@dataclass(frozen=True)
class ZendeskFailure:
    """Why a call did not succeed. ``message`` is safe to show and to log."""

    kind: str
    message: str
    http_status: int | None = None
    retry_after_seconds: int | None = None


@dataclass(frozen=True)
class TicketState:
    """The live ticket fields the assignment needs. Nothing else is kept."""

    ticket_id: int
    status: str | None
    assignee_id: int | None
    group_id: int | None
    updated_at: str | None


@dataclass(frozen=True)
class TicketStateResult:
    state: TicketState | None = None
    failure: ZendeskFailure | None = None

    @property
    def ok(self) -> bool:
        return self.state is not None and self.failure is None


@dataclass(frozen=True)
class AssigneeTarget:
    """The chair to assign, as the client needs it (decoupled from the DB row)."""

    chair_name: str
    zendesk_user_id: int | None
    active: bool


@dataclass(frozen=True)
class AssignRequest:
    """The exact HTTP request the update sends (or would send, in a dry run)."""

    method: str
    path: str
    body: dict


@dataclass(frozen=True)
class AssignResult:
    """The outcome of one assign-and-note call.

    ``ok`` True: sent and accepted, or (``dry_run``) built and NOT sent.
    ``request`` is the built request whenever the inputs were valid (also on a
    transport failure, so a caller can show what was attempted).
    """

    ok: bool
    dry_run: bool
    request: AssignRequest | None = None
    failure: ZendeskFailure | None = None
    ticket_updated_at: str | None = None
    assignee_id: int | None = None


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def build_assign_request(ticket_id: int, zendesk_user_id: int, note_html: str) -> AssignRequest:
    """The single ticket update: one assignee, one private comment. Pure."""
    body = {
        "ticket": {
            "assignee_id": zendesk_user_id,
            "comment": {"html_body": note_html, "public": False},
        }
    }
    return AssignRequest(method="PUT", path=TICKET_PATH.format(ticket_id=ticket_id), body=body)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def state_from_ticket_json(ticket_id: int, data: Any) -> TicketState | None:
    """The four kept fields from a ``GET /tickets/{id}.json`` body, or None."""
    ticket = data.get("ticket") if isinstance(data, dict) else None
    if not isinstance(ticket, dict):
        return None
    return TicketState(
        ticket_id=ticket_id,
        status=_as_str(ticket.get("status")),
        assignee_id=_as_int(ticket.get("assignee_id")),
        group_id=_as_int(ticket.get("group_id")),
        updated_at=_as_str(ticket.get("updated_at")),
    )


def _retry_after(headers: Any) -> int | None:
    try:
        raw = headers.get("Retry-After") if headers is not None else None
        value = int(str(raw).strip()) if raw is not None else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and value >= 0 else None


def _detail_fields(data: Any) -> list[str]:
    """The field NAMES in a Zendesk 422 ``details`` object (never their text)."""
    details = data.get("details") if isinstance(data, dict) else None
    if not isinstance(details, dict):
        return []
    return sorted(k for k in details if isinstance(k, str))


def failure_from_response(status_code: int, headers: Any, data: Any, *, action: str) -> ZendeskFailure:
    """A typed failure for a non-2xx response. Only names and numbers, never text."""
    if status_code == 404:
        return ZendeskFailure(NOT_FOUND, f"Zendesk {action}: ticket not found (HTTP 404).", 404)
    if status_code == 429:
        wait = _retry_after(headers)
        hint = f"; retry after {wait}s" if wait is not None else ""
        return ZendeskFailure(
            RATE_LIMIT, f"Zendesk {action}: rate limited (HTTP 429{hint}).", 429, wait
        )
    fields = _detail_fields(data)
    if status_code == 422 and _ASSIGNMENT_FIELDS.intersection(fields):
        return ZendeskFailure(
            REJECTED_ASSIGNMENT,
            f"Zendesk {action}: the assignment was rejected (HTTP 422; fields: {', '.join(fields)}).",
            422,
        )
    suffix = f"; fields: {', '.join(fields)}" if fields else ""
    return ZendeskFailure(OTHER, f"Zendesk {action} failed (HTTP {status_code}{suffix}).", status_code)


def failure_from_exception(exc: BaseException, *, action: str, network: bool) -> ZendeskFailure:
    """A typed failure for an exception: its type name only, never its text."""
    kind = NETWORK if network else OTHER
    label = "network error" if network else "error"
    return ZendeskFailure(kind, f"Zendesk {action}: {label} ({type(exc).__name__}).")


def refusal(kind: str, message: str) -> ZendeskFailure:
    return ZendeskFailure(kind, message)
