"""Appeals API (v1): the phase-1 export and the Reject Appeals queue.

Mounted under ``/api/v1``:

- ``GET /appeals/phase1/export.csv``: the phase-1 appeal rows as CSV, built by
  ``app.exports.phase1_appeals`` (the same builder the CLI uses).
- The Reject Appeals queue (Z3a), a read-only VIEW of the reject-appeal emails
  with the suggested chair for each paper. The emails stay in the main queue;
  nothing here changes an email, posts to Zendesk or calls a model.

  - ``GET /appeals/config``: ``{"enabled": bool}``; always answers.
  - ``GET /appeals/queue``: one page, newest first, ``{emails, total, page_info}``.
  - ``GET /appeals/queue/counts``: whole-view counts for the nav badge.
  - ``GET /appeals/apcs``: the distinct chair names from the sheet (dropdown).

  With ``REJECT_APPEALS_QUEUE_ENABLED`` off, the last three answer 404 "Reject
  Appeals queue is turned off".

Chair names are personal data from the assignment sheet: nothing here logs them
or puts them in an error message. The chair-note, resolver and note-repository
modules are imported inside the handlers, after the flag check, so app startup
still loads none of them (Z2a, D118).
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.emails import (
    _RECEIVED_PARAM_CONTRACT,
    _email_to_dict,
    _received_range,
)
from app.core.config import settings
from app.db.database import get_db
from app.exports.phase1_appeals import build_phase1_export_csv
from app.models.appeal_queue import (
    MODE_GROUPS,
    NOTE_STATES,
    appeal_mode,
    mode_group,
    submission_numbers,
)
from app.pipeline.drafter import find_placeholders
from app.repositories.email_repository import EmailRepository

router = APIRouter(prefix="/appeals", tags=["appeals"])

_EXPORT_FILENAME = "phase1_appeals.csv"
QUEUE_OFF_DETAIL = "Reject Appeals queue is turned off"

ModeGroup = Literal["composed", "chair_writes", "investigate", "reciprocal", "not_drafted"]
NoteState = Literal["none", "pending", "posting", "posted", "failed"]
assert set(ModeGroup.__args__) == set(MODE_GROUPS)
assert set(NoteState.__args__) == set(NOTE_STATES)

email_repo = EmailRepository()


@router.get("/phase1/export.csv")
async def export_phase1_appeals(db: AsyncSession = Depends(get_db)) -> Response:
    """Every phase-1 appeal row as CSV, one line per appealed paper."""
    return Response(
        content=await build_phase1_export_csv(db),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{_EXPORT_FILENAME}"'},
    )


def _require_queue_enabled() -> None:
    if not settings.REJECT_APPEALS_QUEUE_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=QUEUE_OFF_DETAIL)


@router.get("/config")
async def appeals_config() -> dict:
    """Whether the Reject Appeals queue is on. Always answers, flag on or off."""
    return {"enabled": settings.REJECT_APPEALS_QUEUE_ENABLED}


def _appeal_block(email) -> dict:
    classification = email.classification if isinstance(email.classification, dict) else {}
    extraction = email.extraction if isinstance(email.extraction, dict) else {}
    draft = email.draft if isinstance(email.draft, dict) else {}
    mode = appeal_mode(draft)
    text = draft.get("draft_text") if isinstance(draft.get("draft_text"), str) else ""
    return {
        "intent": classification.get("intent"),
        # None when the reason classifier never ran on this email (no key),
        # which is NOT the same as [] ("asked, no reason applies").
        "reasons": extraction.get("appeal_reason") if "appeal_reason" in extraction else None,
        "is_reciprocal_dispute": extraction.get("is_reciprocal_dispute"),
        "mode": mode,
        "mode_group": mode_group(mode),
        "has_placeholders": bool(find_placeholders(text)),
        "is_edited": bool(draft.get("is_edited")),
        "submission_numbers": submission_numbers(extraction),
    }


def _note_block(row) -> dict | None:
    if row is None:
        return None
    return {
        "status": row.status,
        "attempts": row.attempts,
        "posted_at": row.posted_at.isoformat() if row.posted_at else None,
    }


@router.get("/queue")
async def get_reject_appeals_queue(
    status_filter: str | None = Query(None, alias="status"),
    zendesk_status: str | None = None,
    received_after: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    received_before: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    search: str | None = None,
    mode_group_filter: ModeGroup | None = Query(None, alias="mode_group"),
    reciprocal: bool | None = None,
    note_status: NoteState | None = None,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """One page of the Reject Appeals view, newest first.

    Members: emails whose intent is a reject-appeal intent, plus any email that
    already has a chair note. Each row is the usual email payload plus:

    - ``appeal``: intent, reasons (null when never classified), the reciprocal
      flag, the reply mode and its group, whether placeholders remain, whether
      a chair edited the draft, and the paper numbers (either stored shape).
    - ``suggested_apcs`` / ``chair_source`` / ``chair_numbers`` /
      ``chair_warnings``: the chair lookup (``paper_apc_resolver``).
    - ``chair_note``: the note's state, or null.
    - ``eligibility``: whether a note may be posted, and why not.
    """
    _require_queue_enabled()
    from app.integrations.zendesk.chair_note import check_eligibility
    from app.pipeline.paper_apc_resolver import resolve_paper_apcs_many
    from app.repositories.chair_note_repository import ChairNoteRepository

    after, before = _received_range(received_after, received_before)
    filters = dict(
        status=status_filter,
        zendesk_status=zendesk_status,
        received_after=after,
        received_before=before,
        search=search,
        mode_group=mode_group_filter,
        reciprocal=reciprocal,
        note_status=note_status,
    )
    emails = await email_repo.get_reject_appeal_queue(db, limit=limit, offset=offset, **filters)
    total = await email_repo.count_reject_appeal_queue(db, **filters)

    resolutions = await resolve_paper_apcs_many(
        db,
        {
            e.id: (
                (e.classification or {}).get("intent") if isinstance(e.classification, dict) else None,
                e.extraction,
            )
            for e in emails
        },
    )
    notes = await ChairNoteRepository().get_by_email_ids(db, [e.id for e in emails])

    rows = []
    for e in emails:
        resolution = resolutions[e.id]
        eligibility = check_eligibility(e)
        rows.append(
            {
                **_email_to_dict(e),
                "appeal": _appeal_block(e),
                "suggested_apcs": list(resolution.apc_names),
                "chair_source": resolution.source,
                "chair_numbers": list(resolution.paper_numbers),
                "chair_warnings": list(resolution.warnings),
                "chair_note": _note_block(notes.get(e.id)),
                "eligibility": {"eligible": eligibility.eligible, "reason": eligibility.reason},
            }
        )
    return {
        "emails": rows,
        "total": total,
        "page_info": {
            "limit": limit,
            "offset": offset,
            "status": status_filter,
            "zendesk_status": zendesk_status,
            "received_after": after.isoformat() if after else None,
            "received_before": before.isoformat() if before else None,
            "search": search,
            "mode_group": mode_group_filter,
            "reciprocal": reciprocal,
            "note_status": note_status,
        },
    }


@router.get("/queue/counts")
async def get_reject_appeals_counts(db: AsyncSession = Depends(get_db)) -> dict:
    """Whole-view counts: needs_note, total, by_mode_group, by_note_status,
    without_approved_draft. Unfiltered, never a tally over a page."""
    _require_queue_enabled()
    return await email_repo.reject_appeal_counts(db)


@router.get("/apcs")
async def get_apc_names(db: AsyncSession = Depends(get_db)) -> dict:
    """The distinct chair names in the assignment sheet, for the chair dropdown."""
    _require_queue_enabled()
    from app.repositories.phase1_appeal_repository import PaperAssignmentRepository

    return {"apcs": await PaperAssignmentRepository().list_distinct_apc_names(db)}
