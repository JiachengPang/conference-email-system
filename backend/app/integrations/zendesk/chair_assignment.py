"""Assign one reject-appeal email's ticket to its chair, with the chair note.

The service layer between the endpoint (next piece) and the Zendesk client:

1. refuse an email that has no Zendesk ticket (``invalid_input``);
2. refuse a desk-reject or reciprocal case (``no_chair_case``): these have no
   chair and are never assigned;
3. look up the chosen chair's account by its exact sheet name
   (``ChairAccountRepository``) — missing or inactive is refused by the client;
4. call :meth:`ZendeskSender.assign_with_note`, as a dry run (exact request,
   nothing sent) or for real (only with ``ZENDESK_APPEAL_WRITE_ENABLED``).

It does NOT decide once-per-email, the preview, which chair to pick, or the
note's text: the caller passes ``chair_name`` and the finished ``note_html``
(built with ``chair_note.build_chair_note_html``). The HTTP layer is swappable:
pass ``sender`` and/or ``client`` (tests inject a fake transport). Nothing here
logs a chair name, a user id or any ticket text.
"""

from __future__ import annotations

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.zendesk import ticket_assignment as ta
from app.integrations.zendesk.sender import ZendeskSender
from app.repositories.chair_account_repository import ChairAccountRepository

DESK_REJECT_INTENT = "desk_reject_appeal"
RECIPROCAL_MODE = "reciprocal_review"


def is_no_chair_case(email) -> bool:
    """Desk-reject and reciprocal cases have no chair and are never assigned."""
    classification = getattr(email, "classification", None)
    intent = classification.get("intent") if isinstance(classification, dict) else None
    if intent == DESK_REJECT_INTENT:
        return True
    extraction = getattr(email, "extraction", None)
    if isinstance(extraction, dict) and extraction.get("is_reciprocal_dispute") is True:
        return True
    draft = getattr(email, "draft", None)
    appeal_reply = draft.get("appeal_reply") if isinstance(draft, dict) else None
    return isinstance(appeal_reply, dict) and appeal_reply.get("mode") == RECIPROCAL_MODE


async def assign_email_to_chair(
    db: AsyncSession,
    email,
    chair_name: str,
    note_html: str,
    *,
    dry_run: bool,
    sender: ZendeskSender | None = None,
    client: httpx.AsyncClient | None = None,
    accounts: ChairAccountRepository | None = None,
) -> ta.AssignResult:
    """Assign ``email``'s ticket to ``chair_name`` and add ``note_html`` (one update)."""
    ticket_id = getattr(email, "zendesk_ticket_id", None)
    if not ta.is_positive_int(ticket_id):
        return ta.AssignResult(
            ok=False, dry_run=dry_run,
            failure=ta.refusal(ta.INVALID_INPUT, "This email has no Zendesk ticket."),
        )
    if is_no_chair_case(email):
        return ta.AssignResult(
            ok=False, dry_run=dry_run,
            failure=ta.refusal(ta.NO_CHAIR_CASE, "Desk-reject and reciprocal cases are never assigned."),
        )

    account = await (accounts or ChairAccountRepository()).get_by_name(db, chair_name)
    target = (
        ta.AssigneeTarget(
            chair_name=account.chair_name,
            zendesk_user_id=account.zendesk_user_id,
            active=account.active,
        )
        if account is not None
        else None
    )
    return await (sender or ZendeskSender()).assign_with_note(
        ticket_id, target, note_html, dry_run=dry_run, client=client
    )
