"""CSV export of the ``phase1_appeals`` rows, one line per appealed paper.

One builder, shared by ``GET /api/v1/appeals/phase1/export.csv`` and
``scripts/export_phase1_appeals.py`` so the two can never produce different
files. Reads go through the repositories only.
"""

import csv
import io
from datetime import timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.zendesk.links import zendesk_ticket_url
from app.pipeline.phase1_appeal_outcome import as_utc, ticket_created_at
from app.repositories.email_repository import EmailRepository
from app.repositories.phase1_appeal_repository import Phase1AppealRepository

EXPORT_COLUMNS: tuple[str, ...] = (
    "ticket",
    "zendesk_link",
    "submission_number",
    "apc",
    "openreview_link",
    "relation",
    "appeal_reasons",
    "must_verify",
    "same_paper_tickets",
    "zendesk_status",
    "created_utc",
    "subject",
    "email_body",
)

# Between the requester's own messages in ``email_body``.
_MESSAGE_SEPARATOR = "\n\n---\n\n"
_LIST_SEPARATOR = "; "


def _numeric_key(value) -> tuple:
    """Sort key: numbers numerically, then any non-numeric text, then NULL last."""
    if value is None or value == "":
        return (2, 0, "")
    text = str(value)
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def _iso_z(value) -> str:
    value = as_utc(value)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if value else ""


def _requester_body(email, messages: list[dict]) -> str:
    """The requester's own public messages, oldest first; else the stored body."""
    requester_id = email.zendesk_requester_id
    parts = []
    if requester_id is not None:
        parts = [
            (m.get("plain_body") or "").strip()
            for m in messages
            if m.get("public") and m.get("author_id") == requester_id
        ]
        parts = [p for p in parts if p]
    return _MESSAGE_SEPARATOR.join(parts) if parts else (email.body or "")


async def build_phase1_export_rows(
    db: AsyncSession,
    *,
    appeals: Phase1AppealRepository | None = None,
    emails: EmailRepository | None = None,
) -> list[dict[str, str]]:
    """Every appeal row as a dict keyed by ``EXPORT_COLUMNS``, sorted.

    Sorted by ticket id numerically, then submission number numerically, NULLs
    last. The ticket id is the email's current ``zendesk_ticket_id`` (the
    source of truth), else the one stored on the appeal row.
    """
    appeals = appeals or Phase1AppealRepository()
    emails = emails or EmailRepository()
    rows = await appeals.list_all(db)

    by_email: dict[int, tuple] = {}
    for row in rows:
        if row.email_id in by_email:
            continue
        email = await emails.get_email_by_id(db, str(row.email_id))
        messages = (
            await emails.get_thread_messages(db, str(row.email_id)) if email else []
        )
        by_email[row.email_id] = (email, messages)

    def ticket_of(row):
        email = by_email[row.email_id][0]
        if email is not None and email.zendesk_ticket_id is not None:
            return email.zendesk_ticket_id
        return row.zendesk_ticket_id

    tickets_by_paper: dict[str, set] = {}
    for row in rows:
        ticket = ticket_of(row)
        if row.submission_number and ticket is not None:
            tickets_by_paper.setdefault(row.submission_number, set()).add(ticket)

    out = []
    for row in rows:
        email, messages = by_email[row.email_id]
        ticket = ticket_of(row)
        others = sorted(
            tickets_by_paper.get(row.submission_number, set()) - {ticket}
            if row.submission_number
            else set(),
            key=_numeric_key,
        )
        out.append(
            {
                "ticket": "" if ticket is None else str(ticket),
                "zendesk_link": zendesk_ticket_url(ticket) or "",
                "submission_number": row.submission_number or "",
                "apc": row.apc_name or "",
                "openreview_link": row.openreview_url or "",
                "relation": row.relation,
                "appeal_reasons": _LIST_SEPARATOR.join(
                    r.get("reason", "") for r in (row.reasons or [])
                ),
                "must_verify": "true" if row.must_verify else "false",
                "same_paper_tickets": _LIST_SEPARATOR.join(str(t) for t in others),
                "zendesk_status": (email.zendesk_status or "") if email else "",
                "created_utc": _iso_z(ticket_created_at(email)) if email else "",
                "subject": (email.subject or "") if email else "",
                "email_body": _requester_body(email, messages) if email else "",
            }
        )
    out.sort(
        key=lambda r: (_numeric_key(r["ticket"]), _numeric_key(r["submission_number"]))
    )
    return out


async def build_phase1_export_csv(db: AsyncSession, **repos) -> str:
    """The export as CSV text: a header row of ``EXPORT_COLUMNS``, then the rows."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(await build_phase1_export_rows(db, **repos))
    return buffer.getvalue()
