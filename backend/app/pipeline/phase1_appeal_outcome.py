"""Phase-1 rejection appeal: the gate, the outcome of one run, and its persistence.

The classifier itself (``phase1_appeal_classifier``) only answers. This module
holds the rules around it, shared by every caller so they cannot drift apart:
the orchestrator's create and update paths, and
``scripts/backfill_phase1_appeals.py``.

Outcome semantics (what happens to an email's ``phase1_appeals`` rows):

- flag off: no call, no row changes (the caller never builds an outcome).
- gate not met (intent is not ``PHASE1_APPEAL_INTENT``, or the ticket was created
  before ``PHASE1_APPEAL_START``): delete the email's rows. A reprocess can move
  an email off the gate, so rows from an earlier run must not linger.
- the classifier failed (``None``): keep existing rows unchanged. A failure is not
  an answer and must never erase one.
- relation ``not_appeal``: delete the email's rows.
- otherwise: replace the email's rows, one per appealed paper.

Persistence is best-effort, like chair assignment: a failure is logged (ids and
states only, never email text) and the pipeline carries on.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.pipeline.phase1_appeal_classifier import PROMPT_SHA256, Phase1AppealResult
from app.repositories.phase1_appeal_repository import (
    PaperAssignmentRepository,
    Phase1AppealRepository,
)

logger = logging.getLogger(__name__)

# Outcome states. Also the values logged, so a log line names the rule applied.
GATE_NOT_MET = "gate_not_met"
FAILED = "failed"
NOT_APPEAL = "not_appeal"
CLASSIFIED = "classified"


@dataclass(frozen=True)
class Phase1Outcome:
    """What one run decided for an email. ``result`` is set only when CLASSIFIED."""

    state: str
    result: Phase1AppealResult | None = None


def as_utc(value: datetime | None) -> datetime | None:
    """An aware UTC datetime; a naive value is read as UTC (SQLite reads back naive)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def ticket_created_at(email) -> datetime | None:
    """When the ticket behind a STORED email row was created.

    ``zendesk_created_at`` is the ticket's own creation time and is preferred.
    Rows without it (``/ingest``, seeded rows) fall back to ``received_at``,
    which holds the inbound timestamp when one was supplied and the insert time
    otherwise.
    """
    return getattr(email, "zendesk_created_at", None) or getattr(email, "received_at", None)


def gate_met(intent: str | None, created_at: datetime | None) -> bool:
    """Whether an email qualifies for the phase-1 appeal call.

    The intent must equal ``PHASE1_APPEAL_INTENT``. When ``PHASE1_APPEAL_START``
    is set, the ticket must have been created at or after it; an unknown
    creation time counts as meeting the date gate.
    """
    if intent != settings.PHASE1_APPEAL_INTENT:
        return False
    start = as_utc(settings.PHASE1_APPEAL_START)
    created = as_utc(created_at)
    if start is None or created is None:
        return True
    return created >= start


def outcome_for(result: Phase1AppealResult | None) -> Phase1Outcome:
    """The outcome of a run whose gate was met."""
    if result is None:
        return Phase1Outcome(FAILED)
    if result.relation == "not_appeal":
        return Phase1Outcome(NOT_APPEAL)
    return Phase1Outcome(CLASSIFIED, result)


def active_model_id() -> str | None:
    """The configured model id of the active provider (what the classifier calls)."""
    provider = settings.MODEL_PROVIDER
    if provider == "local":
        return settings.LOCAL_MODEL_NAME
    if provider in ("anthropic", "anthropic_api"):
        return settings.DRAFT_MODEL
    return None


async def _resolve_papers(
    db: AsyncSession,
    result: Phase1AppealResult,
    extraction: dict | None,
    assignments: PaperAssignmentRepository,
) -> list[tuple[str | None, object | None]]:
    """``(submission_number, assignment row or None)`` for each appealed paper.

    Order: the classifier's ``papers``; else the extraction's
    ``submission_numbers``; else the extraction's ``openreview_forum_ids``
    mapped through the assignment sheet; else one paper with no number.
    """
    extraction = extraction or {}
    numbers = list(dict.fromkeys(result.papers))
    if not numbers:
        numbers = list(dict.fromkeys(extraction.get("submission_numbers") or []))
    if numbers:
        found = await assignments.get_by_numbers(db, numbers)
        return [(n, found.get(n)) for n in numbers]

    forum_ids = list(dict.fromkeys(extraction.get("openreview_forum_ids") or []))
    if forum_ids:
        by_forum = await assignments.get_by_forum_ids(db, forum_ids)
        papers: dict[str, object] = {}
        for forum_id in forum_ids:
            row = by_forum.get(forum_id)
            if row is not None:
                papers.setdefault(row.paper_number, row)
        if papers:
            return list(papers.items())
    return [(None, None)]


async def persist_phase1_outcome(
    db: AsyncSession,
    *,
    email_id: int,
    zendesk_ticket_id: int | None,
    extraction: dict | None,
    outcome: Phase1Outcome,
    model: str | None,
    assignments: PaperAssignmentRepository | None = None,
    appeals: Phase1AppealRepository | None = None,
) -> str:
    """Apply ``outcome`` to the email's rows. Never raises.

    Call only after the email row is committed. Returns what was done
    (``deleted`` / ``kept`` / ``replaced`` / ``error``) for logging and counts.
    """
    assignments = assignments or PaperAssignmentRepository()
    appeals = appeals or Phase1AppealRepository()
    rows = 0
    try:
        if outcome.state in (GATE_NOT_MET, NOT_APPEAL):
            rows = await appeals.delete_for_email(db, email_id)
            action = "deleted"
        elif outcome.state == CLASSIFIED and outcome.result is not None:
            result = outcome.result
            reasons = [r.model_dump() for r in result.reasons]
            payload = [
                {
                    "zendesk_ticket_id": zendesk_ticket_id,
                    "submission_number": number,
                    "apc_name": getattr(row, "apc_name", None),
                    "openreview_url": getattr(row, "openreview_url", None),
                    "relation": result.relation,
                    "reasons": reasons,
                    "must_verify": result.must_verify,
                    "prompt_sha256": PROMPT_SHA256,
                    "model": model,
                }
                for number, row in await _resolve_papers(db, result, extraction, assignments)
            ]
            rows = await appeals.replace_for_email(db, email_id, payload)
            action = "replaced"
        else:
            action = "kept"
    except Exception as exc:  # noqa: BLE001 - persistence must not break the pipeline
        # Exception type only: a database error message can carry bound values.
        logger.warning(
            "Phase-1 appeal persistence failed (email=%s, ticket=%s, state=%s, %s); "
            "rows left as they were.",
            email_id,
            zendesk_ticket_id,
            outcome.state,
            type(exc).__name__,
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return "error"
    # Ids and states only — never subject, body or quotes.
    logger.info(
        "Phase-1 appeal: %s -> %s (email=%s, ticket=%s, rows=%d)",
        outcome.state,
        action,
        email_id,
        zendesk_ticket_id,
        rows,
    )
    return action
