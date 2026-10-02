"""Links into the Zendesk agent UI, built from the configured subdomain."""

from app.core.config import settings


def zendesk_ticket_url(ticket_id: int | None) -> str | None:
    """The agent-UI URL of a ticket, or ``None`` without a ticket id or subdomain."""
    if ticket_id is None or not settings.ZENDESK_SUBDOMAIN:
        return None
    return f"https://{settings.ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{ticket_id}"
