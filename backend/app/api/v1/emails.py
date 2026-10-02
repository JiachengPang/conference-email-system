"""Email API (v1) — ingest, queue, detail, and chair actions.

Thin HTTP layer over the pipeline and repositories. Follows the app's existing
router pattern: a module-level ``router = APIRouter(...)`` mounted by main.py,
and the ``get_db`` dependency from ``app.db.database`` for the async session.
No SQLAlchemy is touched directly here — all persistence goes through the
repositories, all processing through EmailPipeline.
"""

import asyncio
import html as _html
import json
import logging
import re

import bleach
from datetime import datetime, time, timezone
from typing import Any, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.events import get_event_broker
from app.core.openreview_gate import authorize_openreview_post
from app.core.send_gate import authorize_send
from app.core.tracing import read_traces
from app.db.database import async_session_factory, get_db
from app.integrations.openreview import (
    OpenReviewNote,
    OpenReviewNoteError,
    OpenReviewThreadMismatchError,
    get_note as openreview_get_note,
    get_openreview_client,
    post_comment_reply as openreview_post_comment_reply,
)
from app.integrations.openreview.client import (
    OpenReviewAuthError,
    OpenReviewCredentialError,
    OpenReviewDependencyError,
)
from app.integrations.zendesk.adapter import ZendeskIngestAdapter
from app.integrations.zendesk.links import zendesk_ticket_url
from app.integrations.zendesk.sender import (
    ZendeskSender,
    ZendeskSendError,
)
from app.models.enums import EmailSource, EmailStatus
from app.pipeline.active_learning import build_flag_events
from app.db.models import AuditLog, Email
from app.pipeline.drafter import find_placeholders
from app.pipeline.orchestrator import EmailPipeline, resolve_lineage_roots
from app.pipeline.rl_router import get_rl_router
from app.repositories.audit_repository import AuditRepository
from app.repositories.chair_repository import ChairRepository
from app.repositories.email_repository import EmailRepository
from app.repositories.policy_repository import PolicyRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emails", tags=["emails"])

email_repo = EmailRepository()
audit_repo = AuditRepository()
chair_repo = ChairRepository()
policy_repo = PolicyRepository()
# Module-level so tests can monkeypatch the transport without real HTTP.
zendesk_sender = ZendeskSender()
# Same — the post-send reconcile (single-ticket re-sync) is best-effort and
# stubbed in tests so no real HTTP happens.
zendesk_adapter = ZendeskIngestAdapter()


class SendRequest(BaseModel):
    """Options for releasing a draft (Zendesk write-back)."""

    # Default is the safe internal note, so a caller that omits this never
    # notifies a requester by accident. NOT gated on ALLOW_AUTO_SEND: the send
    # gate (app/core/send_gate.py) decides WHETHER this draft may be sent at
    # all, and a chair-APPROVED draft is authorized regardless of the flag — so
    # public=True on an approved draft does reach the requester with
    # ALLOW_AUTO_SEND=False. The flag only unlocks the separate unreviewed
    # FAQ-lane auto path. See ZENDESK_API.md §4.
    public: bool = Field(
        default=False,
        description="True = public reply to the requester; False = internal note "
        "(default, safe). Whether the draft may be sent at all is decided by the "
        "send gate, not by this field.",
    )
    sent_by: str = Field(default="chair", description="Actor recorded in the audit log.")
    # Consumed: an explicit value wins and is passed straight to the ticket
    # update, independently of public/internal (e.g. "pending" on an internal
    # note). When omitted, the §4 default applies instead: a public reply →
    # "solved", an internal note → status left unchanged. Mirrors the frontend
    # SendRequest.target_status naming/values (types/index.ts).
    target_status: Literal["open", "pending", "solved"] | None = Field(
        default=None,
        description="Zendesk ticket status to set on send. Omit to keep the "
        "default (public reply → 'solved'; internal note → unchanged).",
    )


class RedraftRequest(BaseModel):
    """Options for a manual re-draft (all optional — an absent body is a plain retry)."""

    forced_policy_key: str | None = Field(
        default=None,
        description="Policy key the chair wants grounded in the new draft, in "
        "ADDITION to normal retrieval (a guaranteed extra slot, never evicted by "
        "re-ranking). Must be an ACTIVE policy; an unknown, retired, or superseded "
        "key is logged and ignored rather than failing the re-draft.",
    )
    excluded_policy_ids: list[str] | None = Field(
        default=None,
        max_length=10,
        description="Policy ids the chair removed from this draft's grounding. "
        "Applied as a filter to the RANKED chunks before the forced-policy slot "
        "is appended, so the drafter simply never sees them — there is no prompt "
        "representation of an exclusion. Capped at 10: the ranked set is bounded "
        "by MAX_RETRIEVED_CHUNKS, so a longer list can only be malformed input.",
    )


class PostOpenReviewReplyRequest(BaseModel):
    """The chair's decision to relay a detected reply onward to OpenReview."""

    # Mirrors ApproveRequest.final_text: the chair may have edited the extracted
    # text in the review UI, and it is the EDITED text that must be posted.
    # Required rather than optional — the endpoint must never fall back to
    # `extraction["extracted_reply_text"]` silently, because that would post
    # something the chair did not read in the box they were looking at.
    reply_text: str = Field(
        ...,
        min_length=1,
        description="The final reply text to post, exactly as the chair "
        "approved it. Sent verbatim; never substituted with the originally "
        "extracted text.",
    )
    submission_number: int = Field(
        ...,
        gt=0,
        description="The submission this comment belongs to, used to build the "
        "Official_Comment invitation. Supplied by the caller because the "
        "extractor reports submission_numbers as a LIST (an email may name "
        "several) and choosing one is not this endpoint's decision to make.",
    )
    visibility: Literal["public", "internal"] = Field(
        default="public",
        description="Who may read the relayed comment. 'public' (the default) "
        "gives it the SAME readers as the parent note, so it reaches exactly "
        "the audience of the comment it answers. 'internal' restricts it to the "
        "Program Chairs and the paper's own Authors group.",
    )
    posted_by: str = Field(
        default="chair", description="Actor recorded in the audit log."
    )


class DismissOpenReviewCandidateRequest(BaseModel):
    """A chair's judgment that a detected OpenReview reply is a false positive."""

    reason: str = Field(
        ...,
        min_length=1,
        description="Why this email is not actually a reply to an OpenReview "
        "notification. Required — the detection is text-based and its false "
        "positives are the feedback signal for improving it, so a dismissal "
        "with no stated reason records that it happened and nothing about why.",
    )
    dismissed_by: str = Field(
        default="chair", description="Actor recorded in the audit log."
    )


class SetStatusRequest(BaseModel):
    """Options for setting a ticket's Zendesk status WITHOUT sending a reply."""

    status: Literal["new", "open", "pending", "solved"] = Field(
        default="solved",
        description="The Zendesk status to set (no reply is sent). "
        "Only 'solved' has a keyboard shortcut; 'new'/'open'/'pending' are "
        "button-only.",
    )
    set_by: str = Field(
        default="chair", description="Actor recorded in the audit log."
    )


def _text_to_html(text: str) -> str:
    """Render a plain-text draft as minimal safe HTML (preferred body per §4).

    Escapes the text, then maps blank lines to paragraph breaks and single
    newlines to ``<br>`` so the reply keeps its shape in Agent Workspace.
    """
    escaped = _html.escape(text or "").strip()
    if not escaped:
        return "<p></p>"
    paragraphs = [p.replace("\n", "<br>") for p in escaped.split("\n\n")]
    return "".join(f"<p>{p}</p>" for p in paragraphs)


# Allowlist for sanitizing Zendesk-authored comment HTML before it reaches the
# chair's browser. Comment bodies are requester/agent-authored → an XSS surface,
# so only formatting tags survive; bleach strips scripts, <style>, inline event
# handlers, style attributes, and any tag/attr/protocol not listed here.
_HTML_ALLOWED_TAGS = [
    "p", "br", "div", "span", "a", "ul", "ol", "li", "b", "strong", "i", "em",
    "u", "s", "blockquote", "pre", "code", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "table", "thead", "tbody", "tfoot", "tr", "td", "th", "img",
]
_HTML_ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
    "img": ["src", "alt", "title", "width", "height"],
    "td": ["colspan", "rowspan"],
    "th": ["colspan", "rowspan"],
}
# http/https/mailto/tel/cid are allowed anywhere; ``data:`` is permitted ONLY on
# <img src> and ONLY for safe raster types (see ``_allow_attribute``) — this lets
# inline screenshots embedded as base64 render while blocking data:svg (which can
# carry script) and data: on links.
_HTML_ALLOWED_PROTOCOLS = ["http", "https", "mailto", "tel", "cid", "data"]
_IMG_DATA_URI_RE = re.compile(r"^data:image/(png|jpe?g|gif|webp);base64,", re.I)
# Drop <script>/<style> blocks WITH their contents first: bleach removes the
# tags but would otherwise leave their inner JS/CSS as visible literal text.
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1>")


def _allow_attribute(tag: str, name: str, value: str) -> bool:
    """Attribute allowlist for bleach, with a guarded exception for data: URIs.

    Enforces :data:`_HTML_ALLOWED_ATTRS`, and additionally permits a ``data:``
    URI only on ``<img src>`` and only for safe raster image types — never on
    links or any other attribute (blocks data:text/html, data:image/svg+xml).
    """
    if name not in _HTML_ALLOWED_ATTRS.get(tag, ()):  # noqa: SIM118 - dict.get
        return False
    v = (value or "").strip()
    if v[:5].lower() == "data:":
        return tag == "img" and name == "src" and bool(_IMG_DATA_URI_RE.match(v))
    return True


def _sanitize_html(raw: str | None) -> str | None:
    """Sanitize Zendesk comment HTML for safe in-browser rendering.

    Returns cleaned HTML (formatting allowlist only), or ``None`` when there is
    no HTML so the caller can fall back to the plain-text body.
    """
    if not raw:
        return None
    stripped = _SCRIPT_STYLE_RE.sub("", raw)
    return bleach.clean(
        stripped,
        tags=_HTML_ALLOWED_TAGS,
        attributes=_allow_attribute,
        protocols=_HTML_ALLOWED_PROTOCOLS,
        strip=True,
    )


def _iso_z(dt) -> str | None:
    """Format an aware datetime as a Zendesk ISO-8601 ``...Z`` stamp, or None."""
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_rl_feedback(email: Email, lane: str | None, outcome: str) -> None:
    """Feed a chair decision to the RL bandit. Never raises.

    The bandit learns from real approve/reroute signals; a failure here must
    not break the chair's action, so everything is best-effort.
    """
    try:
        intent = (email.classification or {}).get("intent")
        if intent and lane:
            get_rl_router().record_feedback(intent=intent, action=lane, outcome=outcome)
    except Exception:  # noqa: BLE001 - feedback is best-effort
        logger.warning("RL feedback recording failed (%s).", outcome, exc_info=True)


async def _record_flag_events(
    db: AsyncSession,
    email_id: str,
    actor: str,
    classification,
    *,
    was_edited: bool = False,
    original_text: str = "",
    edited_text: str = "",
) -> None:
    """Write active-learning candidate flags to the audit log (best-effort).

    Each fired signal becomes its own audit entry with a distinct action type
    (flagged_low_confidence / flagged_meaningful_edit) so the two stay separate.
    Flags candidates for future human labeling only — no retraining is triggered.
    """
    try:
        events = build_flag_events(
            classification,
            was_edited=was_edited,
            original_text=original_text,
            edited_text=edited_text,
        )
        for action, details in events:
            await audit_repo.log_action(db, email_id, action, actor, details)
    except Exception:  # noqa: BLE001 - flagging must never break the chair action
        logger.warning("Active-learning flagging failed.", exc_info=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class IngestEmailRequest(BaseModel):
    """Inbound email payload. ``from``/``to`` are reserved-ish words, so they
    arrive under aliases and bind to safe attribute names."""

    model_config = ConfigDict(populate_by_name=True)

    from_email: str = Field(alias="from")
    to_email: str = Field(alias="to")
    subject: str
    body: str
    timestamp: str = ""


class ApproveRequest(BaseModel):
    approved_by: str
    final_text: str | None = None


class RerouteRequest(BaseModel):
    rerouted_by: str
    reason: str
    new_lane: str


class ReassignChairRequest(BaseModel):
    reassigned_by: str
    new_chair_id: int
    reason: str = ""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _forced_policy_applied(email: Email) -> bool | None:
    """Did the chair's forced policy make it into the current draft's grounding?

    DERIVED, never stored as its own column: the answer is exactly
    ``forced_policy_key in retrieved_ids``, both of which already live in
    ``retrieval_context``. Keeping it computed means it can never drift out of
    sync with the grounding set it describes (same principle as hydrating
    ``retrieved_chunks`` on read rather than duplicating chunk text).

    ``None``  — no forced policy was requested for this draft (a plain redraft,
                or any draft generated before manual invoke existed). This is the
                unchanged default, so old rows are unaffected.
    ``True``  — requested and present in the grounding set.
    ``False`` — requested but skipped: unknown key, not active, or the lookup
                failed. The chair asked for something they did not get, which is
                the case the UI needs to surface.
    """
    ctx = email.retrieval_context or {}
    key = ctx.get("forced_policy_key")
    if not key:
        return None
    return key in (ctx.get("retrieved_ids") or [])


def _email_to_dict(email: Email) -> dict:
    """Serialize an Email ORM row (including its JSON pipeline columns)."""
    return {
        "id": email.id,
        "sender": email.sender,
        "sender_name": email.sender_name,
        "subject": email.subject,
        "body": email.body,
        "status": email.status,
        "received_at": email.received_at.isoformat() if email.received_at else None,
        "assigned_chair_id": email.assigned_chair_id,
        "source": email.source,
        "zendesk_ticket_id": email.zendesk_ticket_id,
        # Deep link to the ticket in the Zendesk agent UI. Built from the
        # existing ZENDESK_SUBDOMAIN config so the frontend needs no Zendesk env
        # var. Null unless this row has a ticket id AND a subdomain is configured.
        "zendesk_ticket_url": zendesk_ticket_url(email.zendesk_ticket_id),
        "zendesk_status": email.zendesk_status,
        "classification": email.classification,
        "routing": email.routing,
        "draft": email.draft,
        # Passed through verbatim, NULL included — no key is read here, so the
        # stored dict's shape is this layer's concern only insofar as it is
        # forwarded intact. A row processed before the extraction column existed
        # serializes as null, which a consumer must read as "never examined" —
        # distinct from a present object whose identifier lists are EMPTY
        # ("examined, found nothing"). Coercing null to {} here would erase that
        # difference at the API boundary.
        "extraction": email.extraction,
        # Whether a chair has ruled the OpenReview detection a false positive.
        # Serialized beside `extraction` but sourced from its own column, which
        # is the distinction that matters: this one survives a reprocess.
        "openreview_candidate_dismissed": bool(email.openreview_candidate_dismissed),
        "redrafting": bool(email.redrafting),
        "retrieval_context": email.retrieval_context,
        # Derived from retrieval_context (no column, no duplicated state). Sits
        # beside `redrafting` on purpose: the frontend already refetches the email
        # when that flag clears, so the manual-invoke outcome arrives on the very
        # same read — no extra endpoint and no separate poll.
        "forced_policy_applied": _forced_policy_applied(email),
        "created_at": email.created_at.isoformat() if email.created_at else None,
        "updated_at": email.updated_at.isoformat() if email.updated_at else None,
    }


async def _email_detail_dict(db: AsyncSession, email: Email) -> dict:
    """``_email_to_dict`` plus the hydrated grounding set (detail views only).

    ``retrieved_chunks`` is NOT a stored column. The pipeline persists only
    ``retrieval_context.retrieved_ids`` (rank-ordered), so the full chunks are
    resolved on READ by joining those ids against ``policy_documents``. That
    keeps a single source of truth for policy text — an edited policy's chunk
    body is never a stale copy frozen into the email row.

    Distinct from ``draft.citations`` (what the model CLAIMS it cited, often
    empty). This is what retrieval actually grounded the draft on.

    Deliberately NOT folded into ``_email_to_dict``: that serializer also runs
    per-row over the queue page, where a lookup would be an N+1 across up to
    ``limit`` emails. Hydration is one extra query, on single-email reads only.

    ``openreview_readers`` is here for the SAME reason and one more: it is an
    outbound network call, and ``/queue/openreview`` is a page of nothing but
    reply candidates, so on the row serializer every list render would become up
    to ``limit`` live OpenReview round-trips. See :func:`_openreview_readers` —
    it is skipped entirely for anything that is not a reply candidate, so this
    helper still performs no I/O beyond the DB for an ordinary email.
    """
    data = _email_to_dict(email)
    data["retrieved_chunks"] = await _hydrate_retrieved_chunks(db, email)
    data["openreview_readers"] = await _openreview_readers(email)
    return data


# How long a detail read waits on OpenReview before reporting the readers as
# unavailable.
#
# This endpoint had NO external dependency before this field existed, so the
# bound matters more than the number: an unbounded SDK call would leave the
# chair's detail page hanging indefinitely on a third party, which is strictly
# worse than showing the audience as unavailable and letting the rest of the
# email render. Note that ``asyncio.wait_for`` bounds the RESPONSE, not the
# work — the worker thread is not cancelled and runs to completion. That is an
# acceptable leak here because the call is a read whose result is advisory.
_OPENREVIEW_READERS_TIMEOUT_SECONDS = 10.0


def _openreview_readers_result(
    state: str,
    *,
    readers: list[str] | None = None,
    note_id: str | None = None,
    error: str | None = None,
    error_type: str | None = None,
) -> dict:
    """One shape for all three states, so no key is ever merely absent."""
    return {
        "state": state,
        "readers": readers,
        "note_id": note_id,
        "error": error,
        "error_type": error_type,
    }


async def _openreview_readers(email: Email) -> dict:
    """Who would see this reply if it were relayed — read LIVE from OpenReview.

    WHY LIVE, NOT STORED AT DETECTION TIME
    --------------------------------------
    Caching was the cheaper option and was rejected on correctness, not effort:

    * ``post_comment_reply`` posts to the readers of the parent note as fetched
      AT POST TIME, and never to a list ConfMail computed. A stored copy could
      therefore show a chair one audience while the post reaches another — and
      the whole reason this field exists is that a chair approving a public
      relay should see who will see it. A display that can disagree with the
      action it is authorising is worse than no display.
    * Filling it at detection time would put an OpenReview API call on the
      INGEST path — inside the Zendesk poller, across every synced ticket,
      requiring OpenReview credentials to be configured before ordinary email
      processing works. Extraction is deliberately text-only (it has no
      OpenReview API integration at all, by design), and this commit is not the
      place to reverse that.

    WHAT LIVE COSTS, AND WHAT PAYS IT
    ---------------------------------
    The call is made ONLY for a genuine reply candidate carrying a note id, so
    every other email — the overwhelming majority, and every row of the main
    queue — pays exactly nothing and this endpoint behaves as it always did.
    For a candidate it is one bounded call on a page a chair opened deliberately,
    one at a time.

    Failure NEVER propagates. Any exception becomes the ``failed`` state and the
    rest of the email still renders; a broken OpenReview must not be able to
    take the detail page down with it.

    THE THREE STATES ARE THE POINT. ``not_applicable`` (not a candidate, so
    there is no parent comment and no audience to show) and ``failed`` (there IS
    an audience, but we could not read it) are completely different facts, and
    collapsing them would hide a real error behind a UI that simply shows
    nothing. ``readers`` is ``None`` in both, never ``[]`` — an empty LIST is a
    fourth, genuine fact ("fetched; the note names no readers") and must stay
    distinguishable, exactly as a null ``extraction`` stays distinct from an
    examined-but-empty one.
    """
    extraction = dict(getattr(email, "extraction", None) or {})
    note_id = (extraction.get("openreview_note_id") or "").strip()

    # Mirrors the first two conditions of ``authorize_openreview_post``, and
    # deliberately does not call it: that gate needs the chair's final reply
    # text and enforces idempotency, neither of which a read has or wants. An
    # already-posted reply still has an audience worth showing.
    if not extraction.get("openreview_reply_candidate") or not note_id:
        return _openreview_readers_result("not_applicable")

    def _fetch() -> OpenReviewNote:
        # Client construction performs a real LOGIN over the network, so it runs
        # in the worker thread alongside the fetch rather than blocking the
        # event loop ahead of it.
        client = get_openreview_client(settings)
        return openreview_get_note(client, note_id)

    try:
        note = await asyncio.wait_for(
            asyncio.to_thread(_fetch),
            timeout=_OPENREVIEW_READERS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "OpenReview readers lookup timed out for email %s (note %s)",
            email.id,
            note_id,
        )
        return _openreview_readers_result(
            "failed",
            note_id=note_id,
            error=(
                "OpenReview did not respond within "
                f"{_OPENREVIEW_READERS_TIMEOUT_SECONDS:g}s."
            ),
            error_type="TimeoutError",
        )
    except Exception as exc:  # noqa: BLE001 - reported in-band, never swallowed
        # Deliberately broad. The specific credential/auth/note errors are all
        # caught here, and so is anything the SDK raises that this code has not
        # anticipated — because on a READ the correct response to an unknown
        # failure is the same as to a known one: say the readers are unavailable
        # and render everything else.
        logger.warning(
            "OpenReview readers lookup failed for email %s (note %s): %s",
            email.id,
            note_id,
            exc,
        )
        return _openreview_readers_result(
            "failed",
            note_id=note_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )

    return _openreview_readers_result(
        "fetched", readers=list(note.readers), note_id=note.id
    )


async def _hydrate_retrieved_chunks(db: AsyncSession, email: Email) -> list[dict] | None:
    """Resolve ``retrieval_context.retrieved_ids`` into full chunk dicts.

    Returns None when the email has no retrieval context (never processed, or a
    legacy row back-filled NULL by migration ``c1d2e3f4a5b6``) — the absence of
    a grounding set is different from an empty one, and the UI already treats
    null as "nothing to show". Rank order follows ``retrieved_ids``.

    ``score`` is intentionally OMITTED: the pipeline never persisted per-chunk
    scores, so there is no honest value to serve for existing rows. Rank is
    conveyed by list position instead of a fabricated number.

    Best-effort — a lookup failure logs and yields None rather than 500-ing an
    email the chair is trying to read.
    """
    ctx = email.retrieval_context
    if not ctx:
        return None
    ids = [pid for pid in (ctx.get("retrieved_ids") or []) if pid]
    if not ids:
        return []

    try:
        rows = await policy_repo.get_by_keys(db, ids)
    except Exception:  # noqa: BLE001 - never break a detail read over grounding
        logger.exception("Failed to hydrate retrieved chunks for email %s", email.id)
        return None

    # Preserve retrieval rank; skip ids with no row (hard-deleted out of band —
    # retire/edit are soft, so this is not reachable through the KB API today).
    return [
        {
            "policy_id": row.policy_key,
            "title": row.title or "",
            "content": row.content or "",
            "category": row.category or "",
        }
        for pid in ids
        if (row := rows.get(pid)) is not None
    ]


def _audit_to_dict(entry: AuditLog) -> dict:
    return {
        "id": entry.id,
        "email_id": str(entry.email_id),
        "action": entry.action,
        "actor": entry.actor,
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "metadata": entry.extra_metadata,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/ingest")
async def ingest_email(
    payload: IngestEmailRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    """Run an inbound email through the full pipeline and persist it."""
    email_data = {
        "from": payload.from_email,
        "to": payload.to_email,
        "subject": payload.subject,
        "body": payload.body,
        "timestamp": payload.timestamp,
    }
    pipeline = EmailPipeline()
    try:
        result = await pipeline.process_email(email_data, db)
    except Exception as exc:  # noqa: BLE001 - surface pipeline failure as 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {exc}",
        ) from exc
    return result.model_dump()


# --- received_at range parsing -------------------------------------------
# Contract for the date-range filter, single-sourced here because BOTH /queue
# and /queue/facets must interpret the values identically or the counts beside
# the list would describe a different window than the list itself.
#
# A bare date is DELIBERATELY accepted (chairs pick days, not instants) and is
# expanded to cover the whole day: ``received_after`` -> 00:00:00.000000,
# ``received_before`` -> 23:59:59.999999. Requiring full timestamps was the
# alternative and was rejected — it pushes the same expansion onto every caller,
# and a client that forgets it silently loses almost a full day of tickets
# against the repository's INCLUSIVE ``<=``, with no error to notice.
#
# A value carrying an explicit time is used verbatim (never re-expanded), so a
# caller who wants a precise instant still gets one.
_RECEIVED_PARAM_CONTRACT = (
    "Accepts a bare date (YYYY-MM-DD) or a full ISO-8601 timestamp. A bare date "
    "covers the WHOLE day in local-to-UTC terms: `received_after` starts at "
    "00:00:00 and `received_before` ends at 23:59:59.999999. A timestamp is used "
    "exactly as given. Naive values (no offset) are read as UTC. Both bounds are "
    "inclusive."
)


def _parse_received_bound(
    raw: str | None, *, param: str, end_of_day: bool
) -> datetime | None:
    """Parse a ``received_after`` / ``received_before`` query value.

    Returns ``None`` for an absent/blank value (filter not applied). Raises 422
    for anything unparseable, rather than degrading to "no filter" — a chair who
    mistypes a date must see an error, not a silently unfiltered queue.

    Date-only input is widened to the correct end of the day (see the contract
    above). Naive input is stamped UTC, matching the convention the ingest path
    already uses (``orchestrator._parse_received_at``), so a value means the same
    instant no matter which entry point produced it.
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            # Literal 422, not a starlette constant: HTTP_422_UNPROCESSABLE_ENTITY
            # is deprecated on current starlette (warns on every call) while its
            # replacement HTTP_422_UNPROCESSABLE_CONTENT does not exist on older
            # versions that `fastapi>=0.111` still permits. The integer is correct
            # on both.
            status_code=422,
            detail=(
                f"Invalid {param}: {raw!r}. Expected YYYY-MM-DD or an ISO-8601 "
                "timestamp such as 2026-01-20T14:30:00Z."
            ),
        ) from None
    # "date only" is decided on the RAW TEXT, not on the parsed value: a caller
    # who explicitly asked for 2026-01-20T00:00:00 means midnight and must not
    # have it silently widened to the end of the day.
    is_date_only = "T" not in text and " " not in text
    if is_date_only and end_of_day:
        parsed = datetime.combine(parsed.date(), time.max, tzinfo=parsed.tzinfo)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _received_range(
    received_after: str | None, received_before: str | None
) -> tuple[datetime | None, datetime | None]:
    """Parse both bounds and reject an inverted range with a 422.

    Without this check an inverted range is indistinguishable from a genuinely
    empty window: the repository returns 0 rows either way, so a chair who typed
    the dates backwards would conclude there are no tickets rather than that the
    request was wrong.
    """
    after = _parse_received_bound(
        received_after, param="received_after", end_of_day=False
    )
    before = _parse_received_bound(
        received_before, param="received_before", end_of_day=True
    )
    if after is not None and before is not None and after > before:
        raise HTTPException(
            status_code=422,  # see _parse_received_bound on the literal
            detail=(
                f"received_after ({after.isoformat()}) must be on or before "
                f"received_before ({before.isoformat()})."
            ),
        )
    return after, before


@router.get("/queue")
async def get_queue(
    lane: str | None = None,
    chair_id: int | None = None,
    status: str | None = None,
    search: str | None = None,
    unassigned: bool = False,
    source: str | None = None,
    zendesk_status: str | None = None,
    received_after: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    received_before: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the email queue, filtered server-side by any combination of
    lane / chair / unassigned / status / source / zendesk_status / search /
    received_at range.

    ``total`` is the count for the SAME filter set (not the whole table), so a
    scoped caller gets an accurate total independent of ``limit``/``offset`` and
    the returned rows are the full server-side slice — callers never filter or
    count a truncated page client-side.

    ``limit`` is bounded 1..200 and ``offset`` >= 0 (FastAPI returns 422 for
    out-of-range values), so no caller can request an unbounded page or a
    negative offset.

    ``received_after`` / ``received_before`` bound ``received_at`` inclusively
    (see :data:`_RECEIVED_PARAM_CONTRACT`); an inverted range is a 422, not an
    empty page. The RESOLVED datetimes are echoed in ``page_info``, so a client
    can see exactly which window was applied after any end-of-day expansion.

    EMAILS DETECTED AS OPENREVIEW REPLIES ARE EXCLUDED — but only while they are
    still unresolved. They need a different action (relay the reply onward) and
    live in their own queue at ``/queue/openreview``, so mixing them in here
    would put two unrelated decisions in one list. A candidate that is already
    solved/closed is NOT pulled out: its visibility is exactly what it was before
    this split existed, so nothing a chair already resolved moves anywhere.

    The two queues are exact complements of ONE predicate
    (``_unresolved_openreview_candidate``), so no email can be hidden from both
    or appear in both.
    """
    after, before = _received_range(received_after, received_before)
    kwargs = dict(
        lane=lane,
        chair_id=chair_id,
        status=status,
        search=search,
        unassigned=unassigned,
        source=source,
        zendesk_status=zendesk_status,
        received_after=after,
        received_before=before,
    )
    emails = await email_repo.get_email_queue(
        db, limit=limit, offset=offset, openreview_candidates="exclude", **kwargs
    )
    total = await email_repo.count_email_queue(
        db, openreview_candidates="exclude", **kwargs
    )
    return {
        "emails": [_email_to_dict(e) for e in emails],
        "total": total,
        "page_info": {"limit": limit, "offset": offset, **kwargs},
    }


@router.get("/queue/facets")
async def get_queue_facets(
    lane: str | None = None,
    chair_id: int | None = None,
    status: str | None = None,
    search: str | None = None,
    unassigned: bool = False,
    received_after: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    received_before: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return grouped facet counts for the queue's status bar + source toggle.

    A dedicated server-side aggregate (see EmailRepository.count_queue_facets),
    NOT a tally over a capped queue page — that page-derived pattern drops
    out-of-window rows (Phase 6C bug class). The context filters
    (lane / chair / unassigned / status / search / received_at range) are honored
    so the facets compose with the queue's other filters; the facet dimensions
    themselves (source, zendesk_status) are intentionally not applied so the bar
    always shows every status and the toggle always sees every source.

    The date range is parsed by the SAME helper as ``/queue``
    (:func:`_received_range`), so the counts always describe the identical
    window as the list they sit beside — including the end-of-day expansion.

    Response shape::

        {
          "by_zendesk_status": {"new": 3, "open": 2, "solved": 1},
          "by_source": {"zendesk": 6, "toy_dataset": 47},
          "sources": ["toy_dataset", "zendesk"]
        }
    """
    after, before = _received_range(received_after, received_before)
    return await email_repo.count_queue_facets(
        db,
        lane=lane,
        chair_id=chair_id,
        status=status,
        search=search,
        unassigned=unassigned,
        received_after=after,
        received_before=before,
        # The SAME exclusion ``/queue`` applies. These counts sit directly beside
        # that list, so leaving it off would make the status bar describe a
        # larger set than the rows underneath it — the "page and total disagree"
        # bug class this aggregate exists to avoid in the first place.
        openreview_candidates="exclude",
    )


@router.get("/queue/openreview")
async def get_openreview_queue(
    lane: str | None = None,
    chair_id: int | None = None,
    status: str | None = None,
    search: str | None = None,
    unassigned: bool = False,
    source: str | None = None,
    zendesk_status: str | None = None,
    received_after: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    received_before: str | None = Query(None, description=_RECEIVED_PARAM_CONTRACT),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the OpenReview-replies queue — the exact inverse of ``/queue``.

    Emails detected as a reviewer or author replying to an OpenReview
    notification, still awaiting chair action. They are excluded from the main
    queue and shown here instead, because the action is different: relay the
    reply onward to OpenReview rather than answer it by email.

    Identical in every other respect to ``/queue`` — same filters, same
    pagination bounds, same ``{emails, total, page_info}`` envelope, same
    newest-first ordering — so a client can reuse the queue's existing patterns
    unchanged and only swap the URL.

    "Unresolved" is not redefined here. Both queues are built from the SINGLE
    predicate ``_unresolved_openreview_candidate`` in the repository, one
    asserting it and one negating it, which is what guarantees they stay exact
    complements rather than two definitions that can drift apart.

    Named ``/queue/openreview`` to sit alongside the existing ``/queue`` and
    ``/queue/facets`` rather than as a detached ``/openreview-queue`` sibling; it
    is a two-segment static path declared BEFORE the ``/{email_id}`` catch-all,
    exactly like ``/queue/facets``, so it cannot be shadowed by it.
    """
    after, before = _received_range(received_after, received_before)
    kwargs = dict(
        lane=lane,
        chair_id=chair_id,
        status=status,
        search=search,
        unassigned=unassigned,
        source=source,
        zendesk_status=zendesk_status,
        received_after=after,
        received_before=before,
    )
    emails = await email_repo.get_email_queue(
        db, limit=limit, offset=offset, openreview_candidates="only", **kwargs
    )
    total = await email_repo.count_email_queue(
        db, openreview_candidates="only", **kwargs
    )
    return {
        "emails": [_email_to_dict(e) for e in emails],
        "total": total,
        "page_info": {"limit": limit, "offset": offset, **kwargs},
    }


# Seconds between SSE heartbeat comments when no events are flowing — keeps the
# connection (and any intermediary proxies) from idling out, and lets the client
# notice a dropped connection promptly.
_SSE_HEARTBEAT_SECONDS = 15.0


@router.get("/stream")
async def stream_emails(request: Request) -> StreamingResponse:
    """Server-Sent Events stream of email lifecycle changes.

    Emits one ``data:`` event per audit-logged state change (created,
    classified/routed, drafted, approved, rerouted) so the review queue can
    update live instead of waiting for its 15s poll. A heartbeat comment is sent
    when idle. One-directional and in-process — no WebSocket, no broker.
    """
    broker = get_event_broker()
    queue = broker.add_subscriber()

    async def event_generator():
        # Opening comment so the client's onopen fires immediately.
        yield ": connected\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=_SSE_HEARTBEAT_SECONDS
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # No events for a while — send a heartbeat comment.
                    yield ": ping\n\n"
        finally:
            broker.remove_subscriber(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (e.g. nginx)
        },
    )


@router.get("/by-ticket/{ticket_id}")
async def get_email_by_ticket(
    ticket_id: int, db: AsyncSession = Depends(get_db)
) -> dict:
    """Return one email (by its Zendesk ticket id) with its full audit trail.

    A two-segment static path, so it MUST be declared before the ``/{email_id}``
    catch-all below — otherwise ``by-ticket`` is captured as an ``email_id`` and
    this route never matches. ``ticket_id`` is typed ``int``, so a non-numeric
    value is rejected by FastAPI with a 422 before this handler runs. The
    response shape mirrors ``GET /emails/{email_id}`` exactly (same helpers).

    404s when no email maps to ``ticket_id`` — including a row whose
    ``zendesk_ticket_id`` is NULL (non-Zendesk), which the repository query never
    matches.
    """
    email = await email_repo.get_email_by_zendesk_ticket_id(db, ticket_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No email found for ticket id {ticket_id}",
        )
    trail = await audit_repo.get_audit_trail(db, str(email.id))
    return {
        "email": await _email_detail_dict(db, email),
        "audit_trail": [_audit_to_dict(a) for a in trail],
    }


@router.get("/{email_id}")
async def get_email(
    email_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    """Return one email together with its full audit trail."""
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )
    trail = await audit_repo.get_audit_trail(db, email_id)
    return {
        "email": await _email_detail_dict(db, email),
        "audit_trail": [_audit_to_dict(a) for a in trail],
    }


@router.get("/{email_id}/thread")
async def get_email_thread(
    email_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    """Return one ticket's full multi-turn conversation, oldest-first.

    Includes internal notes (``public`` False) so the review UI can show them
    distinctly. Non-Zendesk emails simply have no thread rows → ``[]``. 404s if
    the email itself is unknown.
    """
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )
    messages = await email_repo.get_thread_messages(db, email_id)
    # The requester is identified by author_id == ticket requester_id — NOT by
    # Zendesk's ``role``, since chairs/agents often have role "end-user" in this
    # account (so role alone mislabels their replies as the requester). None when
    # there is no Zendesk requester (non-Zendesk email) → UI falls back to role.
    requester_id = email.zendesk_requester_id
    return {
        "messages": [
            {
                **m,
                "created_at": m["created_at"].isoformat() if m["created_at"] else None,
                # Server-sanitized HTML for rich rendering; None → UI falls back
                # to plain_body. Requester-authored, so sanitize before exposing.
                "html_body": _sanitize_html(m.get("html_body")),
                "is_requester": (
                    (m.get("author_id") == requester_id)
                    if requester_id is not None else None
                ),
            }
            for m in messages
        ]
    }


@router.get("/{email_id}/trace")
async def get_email_trace(
    email_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    """Return the ordered per-stage pipeline trace for one email.

    The trace records (classify → retrieve → route → draft) are read from the
    structured trace log, oldest first. 404s if the email itself is unknown.
    """
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )
    entries = read_traces(str(email.id))
    return {
        "email_id": str(email.id),
        "count": len(entries),
        "trace": entries,
    }


async def _learn_from_edit_bg(email_id: str) -> None:
    """Background CEL entry point: learn a candidate policy from a chair's
    [CHAIR:]-gap-filling edit.

    ``learn_from_edit`` opens its own session and never raises (best-effort),
    so this wrapper exists only to give ``background_tasks.add_task`` a plain
    ``(email_id)`` callable — no session/error handling needed here.
    """
    from app.pipeline.experience_learning import learn_from_edit

    await learn_from_edit(email_id)


@router.patch("/{email_id}/approve")
async def approve_email(
    email_id: str,
    payload: ApproveRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Approve an email's draft, preserving the diff when the chair edited it.

    When ``final_text`` differs from the current draft, the original AI/template
    draft is preserved (``draft.original_draft_text``), the edited text becomes
    the new ``draft.draft_text``, and the audit entry captures BOTH full texts so
    the diff can be reconstructed later (Phase 5G active-learning signal).
    Approving unchanged text is NOT recorded as an edit (identical ≠ an edit).
    """
    existing = await email_repo.get_email_by_id(db, email_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )

    draft = dict(existing.draft or {})
    current_text = draft.get("draft_text", "") or ""
    # The true original is the first AI/template draft — preserved across edits.
    original_text = draft.get("original_draft_text") or current_text
    final_text = payload.final_text
    edited = final_text is not None and final_text.strip() != current_text.strip()

    # Send-gate: a reply may not go out while [CHAIR: ...] placeholders remain
    # — the chair must replace each one with real content (or delete it) first.
    unresolved = find_placeholders(final_text if final_text is not None else current_text)
    if unresolved:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Draft still contains unresolved [CHAIR: ...] "
                "placeholders; fill them in before approving.",
                "placeholders": unresolved,
            },
        )

    updates: dict = {}
    details: dict = {"edited": edited}
    if edited:
        draft["original_draft_text"] = original_text
        draft["draft_text"] = final_text
        draft["is_edited"] = True
        draft["edited_by"] = payload.approved_by
        updates["draft"] = draft
        # Keep BOTH full texts so the diff is reconstructable (never lose either).
        details["original_draft"] = original_text
        details["edited_draft"] = final_text
    elif final_text is not None:
        details["final_text"] = final_text

    updated = await email_repo.update_email_status(
        db, email_id, "approved", updates
    )
    await audit_repo.log_action(
        db, email_id, "approved", payload.approved_by, details
    )
    # The approved lane was the right call → reward that (intent, lane) arm.
    _record_rl_feedback(updated, (updated.routing or {}).get("lane"), "approved")
    # Flag active-learning candidates (near-miss confidence and/or a meaningful edit).
    await _record_flag_events(
        db,
        email_id,
        payload.approved_by,
        existing.classification,
        was_edited=edited,
        original_text=original_text,
        edited_text=final_text or "",
    )
    # CEL: an edit that resolved a [CHAIR: ...] gap is exactly the signal the
    # experience-learning stage looks for (a chair supplying knowledge the AI
    # didn't have) — schedule the best-effort background learner. A plain
    # unchanged approve, or an edit with no such gap, schedules nothing.
    if edited and find_placeholders(original_text):
        background_tasks.add_task(_learn_from_edit_bg, str(updated.id))
    return _email_to_dict(updated)


@router.post("/{email_id}/send")
async def send_email_reply(
    email_id: str,
    payload: SendRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Release an email's draft to the outbound transport — gate enforced.

    The send gate (app/core/send_gate.py) always decides first, and both
    outcomes are audited. For a Zendesk-sourced email the authorized draft is
    then written back to the ticket (internal note by default; public reply only
    when ALLOW_AUTO_SEND is on AND explicitly requested — §4). Non-Zendesk emails
    have no transport yet and still answer 501. A Zendesk write failure marks the
    email ``send_failed`` (re-triable) rather than falsely showing it as sent.
    """
    payload = payload or SendRequest()
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )

    decision = authorize_send(email)
    await audit_repo.log_action(
        db, email_id,
        "send_authorized" if decision.authorized else "send_blocked",
        "send_gate",
        {"mode": decision.mode, "reason": decision.reason},
    )
    if not decision.authorized:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Send refused by the send gate.",
                    "reason": decision.reason},
        )

    # Non-Zendesk emails: no outbound transport exists yet — behavior unchanged.
    if (email.source or "") != EmailSource.ZENDESK.value or not email.zendesk_ticket_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"message": "Send authorized, but no outbound transport is "
                    "configured for this source; the draft remains queued.",
                    "mode": decision.mode},
        )

    # Closed tickets are immutable (§2) — never attempt a write; report clearly.
    if (email.zendesk_status or "").lower() == "closed":
        await audit_repo.log_action(
            db, email_id, "send_blocked_closed", payload.sent_by,
            {"zendesk_ticket_id": email.zendesk_ticket_id},
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Zendesk ticket is closed and immutable; cannot "
                    "write. Annotate via a follow-up ticket instead.",
                    "zendesk_status": email.zendesk_status},
        )

    # Public reply vs internal note = whether the requester is notified. WHO may
    # send has already been enforced by the send gate above (authorize_send): an
    # approved draft may go public regardless of ALLOW_AUTO_SEND, while an
    # unreviewed draft_generated draft is refused there (409) unless the FAQ-lane
    # auto path is unlocked. So no additional visibility gate is needed here.
    want_public = bool(payload.public)

    draft_text = (email.draft or {}).get("draft_text", "") or ""
    html_body = _text_to_html(draft_text)
    # Status: an explicit chair-chosen target_status wins (independent of the
    # public/internal decision — e.g. "pending" on an internal note). When none
    # is supplied, fall back to the §4 default: public reply → "solved"; internal
    # note → leave status unchanged. Tags track state via the dedicated tag endpoint.
    set_status = payload.target_status if payload.target_status is not None else (
        "solved" if want_public else None
    )
    # Tag by the AI draft's fate, not merely by visibility: ``ai_auto_replied``
    # ONLY when the AI's words reached the requester UNTOUCHED — a public reply
    # the chair did not edit. A human-edited reply, or ANY internal note (never
    # sent to the requester), is ``ai_drafted``. ``is_edited`` is set by the
    # approve endpoint when the chair's final text differs from the AI draft;
    # automatic sanitization (e.g. the [Sender name] fill) is part of the AI
    # draft and so is not counted as an edit.
    is_edited = bool((email.draft or {}).get("is_edited"))
    tags = ["ai_auto_replied"] if (want_public and not is_edited) else ["ai_drafted"]
    updated_stamp = _iso_z(email.zendesk_updated_at)

    try:
        outcome = await zendesk_sender.send_reply(
            ticket_id=int(email.zendesk_ticket_id),
            html_body=html_body,
            public=want_public,
            set_status=set_status,
            tags=tags,
            updated_stamp=updated_stamp,
        )
    except ZendeskSendError as exc:
        # Transport failed — record the failure locally so it never reads as
        # "sent", and keep the draft intact so the chair can retry.
        send_meta = {
            "state": "failed",
            "public": want_public,
            "error": str(exc),
            "status_code": exc.status_code,
        }
        failed_draft = {**(email.draft or {}), "send": send_meta}
        await email_repo.update_email_status(
            db, email_id, EmailStatus.SEND_FAILED.value, {"draft": failed_draft}
        )
        await audit_repo.log_action(
            db, email_id, "send_failed", payload.sent_by, send_meta
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": "Zendesk write failed; email marked send_failed "
                    "and left re-triable.", "error": str(exc)},
        ) from exc

    # Success — record what was sent and flip status to sent.
    send_meta = {
        "state": "sent",
        "mode": outcome.mode,
        "public": outcome.public,
        "status_set": outcome.status_set,
        "tags_added": outcome.tags_added,
        "tag_conflict": outcome.tag_conflict,
    }
    sent_draft = {**(email.draft or {}), "send": send_meta}
    await email_repo.update_email_status(
        db, email_id, EmailStatus.SENT.value, {"draft": sent_draft}
    )
    await audit_repo.log_action(
        db, email_id, "zendesk_sent", payload.sent_by, send_meta
    )

    # Reconcile the local Zendesk state so the ticket moves buckets immediately
    # and its thread shows the reply we just posted, without waiting for the next
    # background poll. First optimistically mirror the status we set (this alone
    # guarantees the bucket move even if the re-sync below can't reach Zendesk),
    # then best-effort re-sync THIS ticket for the authoritative status + to append
    # the outbound comment to the thread. A re-sync failure must NEVER turn a
    # successful send into an error — the reply is already posted to Zendesk.
    try:
        if set_status is not None:
            await email_repo.apply_zendesk_fields(
                db, email_id, {"zendesk_status": set_status}
            )
        await zendesk_adapter.refresh_ticket(db, int(email.zendesk_ticket_id))
    except Exception as exc:  # noqa: BLE001 - reconcile is best-effort
        logger.warning(
            "post-send reconcile of ticket %s failed (reply was sent): %s",
            email.zendesk_ticket_id, exc,
        )
        # A failed DB write inside the reconcile leaves the session in a
        # pending-rollback state; clear it so the read below (and the rest of
        # this request) can't raise PendingRollbackError. Any already-committed
        # optimistic status write stands, so the bucket move persists.
        await db.rollback()

    updated = await email_repo.get_email_by_id(db, email_id)
    result = _email_to_dict(updated)
    result["send"] = send_meta
    if outcome.tag_conflict:
        result["warning"] = (
            "Reply sent, but the state-tag write hit a 409 (ticket changed "
            "concurrently); the tag was NOT overwritten. Re-tag or re-sync."
        )
    return result


@router.post("/{email_id}/set-status")
async def set_ticket_status_no_reply(
    email_id: str,
    payload: SetStatusRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Set a Zendesk ticket's status (new / open / pending / solved) WITHOUT
    sending a reply.

    For the "no response warranted — just set the state" case: the ticket status
    flips and nothing goes to the requester (no comment). ``solved`` is the common
    "resolve it" action (and the only one with a keyboard shortcut); ``new`` /
    ``open`` re-open or reset the ticket, and ``pending`` parks it awaiting the
    requester. Unlike ``/send`` there is no reply to authorize, so the send gate
    does not apply — but the same closed-ticket guard, audit trail, and
    best-effort re-sync do. A transport failure marks the email ``send_failed``
    (re-triable), mirroring ``/send``.

    Local workflow status: a no-reply ``solved`` is terminal → ``SOLVED``; ``new`` /
    ``open`` / ``pending`` do not resolve the email, so its workflow status is left
    unchanged (only ``zendesk_status`` — and thus the queue bucket — moves).
    """
    payload = payload or SetStatusRequest()
    set_status = payload.status
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )

    # No outbound transport for non-Zendesk emails (same as /send).
    if (email.source or "") != EmailSource.ZENDESK.value or not email.zendesk_ticket_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"message": "Status change authorized, but no Zendesk transport "
                    "is configured for this source; nothing changed.",
                    "source": email.source},
        )

    # Closed tickets are immutable (§2) — never attempt a write; report clearly.
    if (email.zendesk_status or "").lower() == "closed":
        await audit_repo.log_action(
            db, email_id, "status_set_blocked_closed", payload.set_by,
            {"zendesk_ticket_id": email.zendesk_ticket_id, "requested_status": set_status},
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Zendesk ticket is closed and immutable; cannot "
                    "write. Annotate via a follow-up ticket instead.",
                    "zendesk_status": email.zendesk_status},
        )

    tags = [f"ai_status_{set_status}"]
    updated_stamp = _iso_z(email.zendesk_updated_at)

    try:
        outcome = await zendesk_sender.set_status_only(
            ticket_id=int(email.zendesk_ticket_id),
            status=set_status,
            tags=tags,
            updated_stamp=updated_stamp,
        )
    except ZendeskSendError as exc:
        # Transport failed — record the failure locally so it never reads as
        # done, and keep the email re-triable (mirrors /send).
        send_meta = {
            "state": "failed",
            "public": False,
            "requested_status": set_status,
            "error": str(exc),
            "status_code": exc.status_code,
        }
        failed_draft = {**(email.draft or {}), "send": send_meta}
        await email_repo.update_email_status(
            db, email_id, EmailStatus.SEND_FAILED.value, {"draft": failed_draft}
        )
        await audit_repo.log_action(
            db, email_id, "status_set_failed", payload.set_by, send_meta
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": "Zendesk status write failed; email marked "
                    "send_failed and left re-triable.", "error": str(exc)},
        ) from exc

    # Success — record what happened (no reply). A no-reply "solved" resolves the
    # email locally (SOLVED); "new"/"open" don't, so leave the workflow status be.
    send_meta = {
        "state": "status_set_no_reply",
        "mode": outcome.mode,
        "public": outcome.public,
        "status_set": outcome.status_set,
        "tags_added": outcome.tags_added,
        "tag_conflict": outcome.tag_conflict,
    }
    local_status = (
        EmailStatus.SOLVED.value if set_status == "solved" else email.status
    )
    new_draft = {**(email.draft or {}), "send": send_meta}
    await email_repo.update_email_status(
        db, email_id, local_status, {"draft": new_draft}
    )
    await audit_repo.log_action(
        db, email_id, "status_set_no_reply", payload.set_by, send_meta
    )

    # Reconcile local Zendesk state so the ticket moves buckets immediately
    # (optimistic mirror), then best-effort re-sync for the authoritative state.
    # A re-sync failure must NEVER fail the request — the status is already set
    # in Zendesk.
    try:
        await email_repo.apply_zendesk_fields(
            db, email_id, {"zendesk_status": set_status}
        )
        await zendesk_adapter.refresh_ticket(db, int(email.zendesk_ticket_id))
    except Exception as exc:  # noqa: BLE001 - reconcile is best-effort
        logger.warning(
            "post-status-set reconcile of ticket %s failed (status was set): %s",
            email.zendesk_ticket_id, exc,
        )
        await db.rollback()

    updated = await email_repo.get_email_by_id(db, email_id)
    result = _email_to_dict(updated)
    result["send"] = send_meta
    if outcome.tag_conflict:
        result["warning"] = (
            f"Ticket set to {set_status}, but the state-tag write hit a 409 "
            "(ticket changed concurrently); the tag was NOT overwritten. Re-sync."
        )
    return result


# Outcomes of the auto-solve half of /post-openreview-reply. Four values rather
# than a boolean because a chair acts differently on each: a transport failure
# wants a retry, a closed ticket wants nothing, and a non-Zendesk email has no
# ticket at all. Collapsing them would force the frontend to re-derive the
# difference from an error string.
OPENREVIEW_SOLVE_SOLVED = "solved"
OPENREVIEW_SOLVE_FAILED = "solve_failed"
OPENREVIEW_SOLVE_SKIPPED_NO_TICKET = "skipped_no_ticket"
OPENREVIEW_SOLVE_SKIPPED_CLOSED = "skipped_closed"


async def _auto_solve_after_openreview_post(
    db: AsyncSession, email: Email, post_meta: dict, actor: str
) -> dict:
    """Resolve the Zendesk ticket after a successful OpenReview post.

    NEVER RAISES. It is called only once the comment is already public, so any
    exception escaping here would turn a completed action into a failed request.
    Every path returns a ``ticket_resolution`` dict instead, and the caller
    surfaces it as-is.

    Reuses ``ZendeskSender.set_status_only`` — the same transport ``/set-status``
    drives, called directly rather than through an HTTP hop, which is how the
    other endpoints already share the sender.

    Deliberately does NOT mark the email ``SEND_FAILED`` when the solve fails,
    which is where it parts company with ``/set-status``. There, a failed status
    write is the whole action failing. Here the action succeeded; flagging the
    row ``SEND_FAILED`` would put an email whose reply is live into the queue's
    failed-send bucket and invite someone to retry the send. The failure is
    recorded on ``draft["ticket_resolution"]`` and in the audit trail instead.
    """
    base = {
        "attempted": False,
        "ticket_id": email.zendesk_ticket_id,
        "zendesk_status": None,
        "error": None,
        "error_type": None,
        "recovery": None,
    }
    audit_context = {
        "note_id": post_meta.get("note_id"),
        "parent_note_id": post_meta.get("parent_note_id"),
        "forum_id": post_meta.get("forum_id"),
        "zendesk_ticket_id": email.zendesk_ticket_id,
    }

    # No ticket to solve — the post still happened, so this is not a failure.
    if (email.source or "") != EmailSource.ZENDESK.value or not email.zendesk_ticket_id:
        resolution = {
            **base,
            "outcome": OPENREVIEW_SOLVE_SKIPPED_NO_TICKET,
            "recovery": "No Zendesk ticket is associated with this email, so "
            "there was nothing to resolve.",
        }
        await audit_repo.log_action(
            db, str(email.id), "zendesk_auto_solve_skipped", actor,
            {**audit_context, "reason": "not a Zendesk-sourced email"},
        )
        return resolution

    # Closed tickets are immutable (§2) — the same guard /set-status applies,
    # except here it cannot fail the request.
    if (email.zendesk_status or "").lower() == "closed":
        resolution = {
            **base,
            "outcome": OPENREVIEW_SOLVE_SKIPPED_CLOSED,
            "zendesk_status": email.zendesk_status,
            "recovery": "The Zendesk ticket is already closed and immutable; it "
            "needs no action.",
        }
        await audit_repo.log_action(
            db, str(email.id), "zendesk_auto_solve_skipped", actor,
            {**audit_context, "reason": "ticket is closed"},
        )
        return resolution

    try:
        outcome = await zendesk_sender.set_status_only(
            ticket_id=int(email.zendesk_ticket_id),
            status="solved",
            tags=["ai_status_solved"],
            updated_stamp=_iso_z(email.zendesk_updated_at),
        )
    except ZendeskSendError as exc:
        # THE partial-success path. The comment is live; the ticket is not.
        resolution = {
            **base,
            "attempted": True,
            "outcome": OPENREVIEW_SOLVE_FAILED,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "recovery": "The reply WAS posted to OpenReview, but the Zendesk "
            "ticket could not be auto-solved and is still open. Resolve it with "
            "the mark-solved action (POST /emails/{id}/set-status); do not "
            "re-post the reply.",
        }
        failed_draft = {
            **(email.draft or {}),
            "ticket_resolution": {
                "state": "solve_failed",
                "error": str(exc),
                "status_code": getattr(exc, "status_code", None),
            },
        }
        # Workflow status passed back UNCHANGED — see the docstring on why this
        # is not SEND_FAILED.
        await email_repo.update_email_status(
            db, str(email.id), email.status, {"draft": failed_draft}
        )
        await audit_repo.log_action(
            db, str(email.id), "zendesk_auto_solve_failed", actor,
            {
                **audit_context,
                "requested_status": "solved",
                "error": str(exc),
                "status_code": getattr(exc, "status_code", None),
                # Spelled out so the audit trail alone answers "what state is
                # this in?" without cross-referencing another entry.
                "openreview_post_state": "posted",
                "ticket_state": "not solved",
            },
        )
        return resolution

    # Solved. The email is terminal now, exactly as a no-reply solve leaves it.
    solved_draft = {
        **(email.draft or {}),
        "ticket_resolution": {"state": "solved", "status_set": outcome.status_set},
    }
    await email_repo.update_email_status(
        db, str(email.id), EmailStatus.SOLVED.value, {"draft": solved_draft}
    )
    await audit_repo.log_action(
        db, str(email.id), "zendesk_auto_solved", actor,
        {
            **audit_context,
            "status_set": outcome.status_set,
            "tags_added": outcome.tags_added,
            "tag_conflict": outcome.tag_conflict,
        },
    )

    # Mirror the status locally so the queue bucket moves at once, then re-sync
    # best-effort. A reconcile failure must not downgrade a real solve.
    try:
        await email_repo.apply_zendesk_fields(
            db, str(email.id), {"zendesk_status": "solved"}
        )
        await zendesk_adapter.refresh_ticket(db, int(email.zendesk_ticket_id))
    except Exception as exc:  # noqa: BLE001 - reconcile is best-effort
        logger.warning(
            "post-OpenReview solve reconcile of ticket %s failed (it WAS solved): %s",
            email.zendesk_ticket_id, exc,
        )
        await db.rollback()

    return {
        **base,
        "attempted": True,
        "outcome": OPENREVIEW_SOLVE_SOLVED,
        "zendesk_status": "solved",
    }


def _internal_reply_readers(venue_id: str, submission_number: int) -> list[str]:
    """The audience for a relay the chair marked ``internal``.

    A FIXED list, built from the venue and the submission — deliberately NOT
    derived from, filtered from, or merged with the parent note's readers. The
    point of the internal option is to be narrower than the thread, and a
    computation that started from the parent's list could only ever be narrower
    BY ACCIDENT: any group the parent happens to carry would carry over.
    Constructing the audience from scratch means the result cannot depend on
    what was in the thread.

    Both entries are load-bearing and neither is decoration:

    * ``Program_Chairs`` is the group this account posts AS (``_role_signature``
      in notes.py builds the same string), so omitting it would post a comment
      the poster cannot read.
    * ``Submission{n}/Authors`` is here because the reply concerns the authors'
      OWN paper. Narrower than the thread is the intent; invisible to the people
      it is about is not.

    Lives in the endpoint rather than in ``notes.py`` because that module states
    its own boundary outright — "No readers computation. ``post_comment_reply``
    posts to the readers it is GIVEN. Deriving them belongs to the caller."
    Putting it there would contradict the contract the transport documents.

    ⚠️ The group names share ``_role_signature``'s PENDING CONFIRMATION from
    AAAI/OpenReview. A wrong group name is rejected by OpenReview at post time
    rather than silently widening the audience, so the failure is loud — but
    confirm both before the first live internal post.
    """
    return [
        f"{venue_id}/Program_Chairs",
        f"{venue_id}/Submission{submission_number}/Authors",
    ]


@router.post("/{email_id}/post-openreview-reply")
async def post_openreview_reply(
    email_id: str,
    payload: PostOpenReviewReplyRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Relay a detected reply onward to OpenReview as a threaded Official Comment.

    For the case where a reviewer or author replied to an OpenReview
    notification and their answer landed in the chair inbox instead of on the
    forum. The chair reviews it and, if it should go where it was meant to go,
    calls this — and the reply is posted under the ORIGINAL comment.

    Ordering is deliberate and every step is a precondition of the next:

    1. The gate (``authorize_openreview_post``) decides on the email row alone —
       no I/O — and both outcomes are audited, mirroring ``/send``.
    2. The parent note is FETCHED. This proves it exists and is not deleted, it
       pins the thread the reply must land in, and — for a ``public`` relay — it
       is where the readers come from.
    3. The reply is posted, threaded under that note.

    VISIBILITY IS A SEPARATE AXIS FROM THE TARGET THREAD, and step 2 runs in
    full either way. ``public`` (the default, and the only behaviour before this
    option existed) reuses the parent's readers, so the reply reaches exactly
    the audience of the comment it answers. ``internal`` replaces them with the
    Program Chairs and the paper's own Authors group. Neither value changes
    WHERE the comment is posted, and neither relaxes the forum-id verification:
    the wrong submission's discussion is the wrong place for a narrow comment
    just as much as for a wide one.

    4. The Zendesk ticket is auto-solved. Relaying the reply IS the resolution,
       so the chair does not click twice.

    THREE OUTCOMES, not two, and the response says which. The OpenReview comment
    is public the instant step 3 returns and cannot be un-posted from here, so a
    Zendesk failure afterwards must never be reported as though the whole action
    failed — a chair who read "failed" and retried would either be blocked by the
    idempotency gate or, worse, post twice. Nor may it be swallowed: the ticket
    is genuinely still open and somebody has to close it. See
    ``ticket_resolution`` in the response.
    """
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )

    # --- 1. gate ----------------------------------------------------------
    decision = authorize_openreview_post(email, payload.reply_text)
    await audit_repo.log_action(
        db,
        email_id,
        "openreview_post_authorized" if decision.authorized else "openreview_post_blocked",
        "openreview_gate",
        {"reason": decision.reason},
    )
    if not decision.authorized:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Refused by the OpenReview post gate.",
                "reason": decision.reason,
            },
        )

    venue_id = (settings.OPENREVIEW_VENUE_ID or "").strip()
    if not venue_id:
        # Config, not caller error — 501 matches how /send reports "authorized,
        # but this deployment has no transport for it".
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "message": "OPENREVIEW_VENUE_ID is not configured, so the "
                "Official_Comment invitation cannot be built. Set it to the "
                "venue's OpenReview group id (e.g. AAAI.org/2027/Conference).",
            },
        )

    # --- 2. fetch the parent note ----------------------------------------
    # Its readers are the audience of the comment being answered, and reusing
    # them is what keeps the reply visible to exactly that audience. They are
    # never recomputed here.
    #
    # BOTH the connect and the fetch run in ONE worker thread. Building the
    # client is not a cheap local construction: openreview-py's
    # ``OpenReviewClient.__init__`` performs a real LOGIN over the network when
    # given a username and password (see client.py), so calling it on the event
    # loop stalls every other request in this worker for the duration of that
    # round-trip. Threading only the fetch that follows left the two halves of
    # one remote conversation inconsistently isolated.
    #
    # They share a thread rather than taking one each so that the SAME client is
    # reused for the post in step 3 — one login per relay, exactly as before.
    def _connect_and_fetch() -> tuple[Any, OpenReviewNote]:
        openreview_client = get_openreview_client(settings)
        return openreview_client, openreview_get_note(
            openreview_client, decision.note_id
        )

    try:
        client, parent = await asyncio.to_thread(_connect_and_fetch)
    except (OpenReviewCredentialError, OpenReviewDependencyError) as exc:
        await audit_repo.log_action(
            db, email_id, "openreview_post_failed", payload.posted_by,
            {"stage": "client", "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"message": "OpenReview access is not configured.",
                    "error": str(exc)},
        ) from exc
    except OpenReviewAuthError as exc:
        await audit_repo.log_action(
            db, email_id, "openreview_post_failed", payload.posted_by,
            {"stage": "auth", "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": "OpenReview login failed.", "error": str(exc)},
        ) from exc
    except OpenReviewNoteError as exc:
        # Not found / deleted / permission / API — surfaced with its own type
        # name so the caller keeps the distinction commit 10 drew.
        await audit_repo.log_action(
            db, email_id, "openreview_post_failed", payload.posted_by,
            {"stage": "get_note", "note_id": decision.note_id,
             "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Could not fetch the OpenReview comment being "
                "replied to; nothing was posted.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc

    # --- 3. post ----------------------------------------------------------
    # WHO reads the reply and WHICH THREAD it lands in are independent, and the
    # ordering here keeps them that way. ``parent`` was fetched above and is
    # passed through to ``post_comment_reply`` in BOTH branches, so the
    # forum-id mismatch verification runs unconditionally — an internal comment
    # posted into another submission's discussion is a leak, not a lesser one,
    # so narrowing the audience is never a reason to relax the target check.
    readers = (
        _internal_reply_readers(venue_id, payload.submission_number)
        if payload.visibility == "internal"
        else list(parent.readers)
    )
    try:
        posted = await asyncio.to_thread(
            lambda: openreview_post_comment_reply(
                client,
                venue_id=venue_id,
                submission_number=payload.submission_number,
                parent_note_id=decision.note_id,
                forum_id=decision.forum_id,
                comment_text=payload.reply_text,
                readers=readers,
                parent_note=parent,
            )
        )
    except (OpenReviewThreadMismatchError, OpenReviewNoteError) as exc:
        # The email row is left EXACTLY as it was. A failed post must not leave
        # anything looking posted, or a retry becomes impossible to reason about.
        await audit_repo.log_action(
            db, email_id, "openreview_post_failed", payload.posted_by,
            {"stage": "post", "note_id": decision.note_id,
             "forum_id": decision.forum_id,
             "visibility": payload.visibility,
             "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Posting the reply to OpenReview failed; nothing "
                "was posted and the email is unchanged.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc

    # --- success ----------------------------------------------------------
    # Recorded on the email as well as in the audit log. The audit trail is the
    # history; this is the STATE the gate reads to refuse a duplicate post, and
    # what the follow-up ticket-resolution piece will key on.
    post_meta = {
        "state": "posted",
        "note_id": posted.note_id,
        "edit_id": posted.edit_id,
        "parent_note_id": decision.note_id,
        "forum_id": decision.forum_id,
        "venue_id": venue_id,
        "submission_number": payload.submission_number,
        # Both, not one or the other. `readers` is the audience that was
        # actually used and is the fact that matters; `visibility` records the
        # chair's INTENT, which a reader list alone cannot be reverse-engineered
        # into — an internal list and a parent thread that happened to hold the
        # same two groups are indistinguishable after the fact.
        "visibility": payload.visibility,
        "readers": list(readers),
    }
    updated_draft = {**(email.draft or {}), "openreview_post": post_meta}
    # The email's own status is passed back UNCHANGED, the same way /set-status
    # leaves it for a non-terminal transition. This is not an email send, and the
    # ticket's fate is the next commit's decision, not this one's.
    #
    # `update_email_status` rather than `update_email_outputs` specifically
    # because the latter unconditionally sets `redrafting = False`, which would
    # silently clear an in-flight redraft flag this endpoint has no business
    # touching.
    await email_repo.update_email_status(
        db, email_id, email.status, {"draft": updated_draft}
    )
    await audit_repo.log_action(
        db, email_id, "openreview_comment_posted", payload.posted_by, post_meta
    )

    # --- 4. auto-solve the ticket -----------------------------------------
    # Everything past this point is ABOUT the ticket, never about the post. The
    # post has already succeeded and been audited, and no branch below is allowed
    # to raise: an exception here would surface as a failed request for an action
    # that demonstrably worked.
    resolution = await _auto_solve_after_openreview_post(
        db, email, post_meta, payload.posted_by
    )

    refreshed = await email_repo.get_email_by_id(db, email_id)
    result = _email_to_dict(refreshed)
    result["openreview_post"] = post_meta
    result["ticket_resolution"] = resolution
    # Mirrors the partial-success convention /send already uses for a tag 409:
    # a 200 carrying a human-readable warning beside the machine-readable field.
    result["warning"] = resolution.get("recovery") if resolution["outcome"] != "solved" else None
    return result


@router.post("/{email_id}/dismiss-openreview-candidate")
async def dismiss_openreview_candidate(
    email_id: str,
    payload: DismissOpenReviewCandidateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Mark a detected OpenReview reply as a false positive.

    The detection in the extraction pipeline is text-based: an email carrying
    both a note id and a venue notification address is treated as a reply to
    that notification. That is a heuristic, and a chair reading the email is the
    authority on whether it is actually one. This records that judgment, after
    which the email falls through the normal ``/queue`` predicate instead of
    ``/queue/openreview``.

    DELIBERATELY NOT AN EXTENSION OF ``/reroute``, despite both being "this went
    to the wrong place". That endpoint calls ``_record_rl_feedback(existing,
    original_lane, "rerouted")``, which penalises the ``(intent, lane)`` arm of
    the RL bandit on the premise that the LANE was wrong, and
    ``_record_flag_events``, which feeds active learning on classifier
    confidence. A false-positive OpenReview detection says nothing about either:
    the lane router and the classifier may both have been perfectly correct on
    this email. Firing those here would train two models against decisions they
    got right. Neither is called below, and ``routing`` is not touched at all.

    WHAT IT WRITES is a dedicated column, not a key inside ``extraction`` — see
    ``Email.openreview_candidate_dismissed``. Every pipeline pass rewrites
    ``extraction`` from the email text, so a dismissal stored there would be
    recomputed away by the next follow-up reply or re-draft.

    409 if the email was never a candidate: there is nothing to dismiss, and
    silently succeeding would tell a caller it had changed something it had not.
    """
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )

    extraction = dict(email.extraction or {})
    if not extraction.get("openreview_reply_candidate"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "This email was never detected as an OpenReview "
                "reply candidate, so there is nothing to dismiss.",
                "reason": "openreview_reply_candidate is not true"
                if email.extraction is not None
                else "this email has no extraction record",
            },
        )

    # Idempotent, and quiet about it. A second dismissal changes nothing, so it
    # writes no second audit entry: the audit log records state CHANGES, and two
    # entries would read as two separate chair decisions when the row was
    # already in that state. The response says which happened, so a caller can
    # tell "I did this" from "this was already done" without guessing.
    if email.openreview_candidate_dismissed:
        result = _email_to_dict(email)
        result["already_dismissed"] = True
        return result

    updated = await email_repo.dismiss_openreview_candidate(db, email_id)
    await audit_repo.log_action(
        db,
        email_id,
        "openreview_candidate_dismissed",
        payload.dismissed_by,
        {
            "reason": payload.reason,
            "openreview_note_id": extraction.get("openreview_note_id"),
            "openreview_notification_sender": extraction.get(
                "openreview_notification_sender"
            ),
            # The lane is recorded but NOT changed — so a later reader can see
            # it was left alone rather than wonder whether it was.
            "lane": (email.routing or {}).get("lane"),
        },
    )

    result = _email_to_dict(updated)
    result["already_dismissed"] = False
    return result


@router.patch("/{email_id}/reroute")
async def reroute_email(
    email_id: str, payload: RerouteRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    """Reroute an email to a different lane and record the reason."""
    existing = await email_repo.get_email_by_id(db, email_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )
    original_lane = (existing.routing or {}).get("lane")
    new_routing = dict(existing.routing or {})
    new_routing["lane"] = payload.new_lane
    updated = await email_repo.update_email_status(
        db, email_id, "rerouted", {"routing": new_routing}
    )
    await audit_repo.log_action(
        db, email_id, "rerouted", payload.rerouted_by,
        {"reason": payload.reason, "new_lane": payload.new_lane},
    )
    # The original lane was wrong → penalize that (intent, lane) arm (no win).
    _record_rl_feedback(existing, original_lane, "rerouted")
    # A reroute involves no draft edit, so only the low-confidence signal applies.
    await _record_flag_events(
        db, email_id, payload.rerouted_by, existing.classification, was_edited=False
    )
    return _email_to_dict(updated)


@router.patch("/{email_id}/reassign-chair")
async def reassign_chair(
    email_id: str, payload: ReassignChairRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    """Reassign a human-review email to a different chair (Phase 6A).

    Updates ``assigned_chair_id`` and writes a ``chair_reassigned`` audit entry
    through the EXISTING audit mechanism (no new table). The entry captures the
    original + new chair ids and the intent/confidence recorded at assignment
    time (read off the email's stored classification) — the training signal a
    learned chair-routing strategy will later consume to learn from human
    corrections.
    """
    existing = await email_repo.get_email_by_id(db, email_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )
    # The target chair must exist (a reassignment can target an inactive chair —
    # that's a deliberate human override — but not a nonexistent one).
    target = await chair_repo.get_chair_by_id(db, payload.new_chair_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chair {payload.new_chair_id} not found",
        )

    original_chair_id = existing.assigned_chair_id
    classification = existing.classification or {}
    updated = await email_repo.assign_chair(db, email_id, payload.new_chair_id)
    await audit_repo.log_action(
        db, email_id, "chair_reassigned", payload.reassigned_by,
        {
            "original_chair_id": original_chair_id,
            "new_chair_id": payload.new_chair_id,
            "reason": payload.reason,
            # Intent + confidence AT ASSIGNMENT TIME (from the stored
            # classification) — the signal a reroute is a correction against.
            "intent": classification.get("intent"),
            "confidence": classification.get("confidence"),
        },
    )
    return _email_to_dict(updated)


async def _would_empty_the_grounding(
    db: AsyncSession,
    email: Email,
    excluded_policy_ids: list[str] | None,
    forced_policy_key: str | None,
) -> bool:
    """Would this request strip the draft of ALL policy context?

    Checked HERE rather than in the pipeline because the re-draft is async: the
    endpoint returns 202 and the real filtering happens in a background task,
    which has no HTTP response left to reject with. Refusing after scheduling
    could only be a silent no-op or a stranded ticket.

    It compares against the email's CURRENT ``retrieval_context.retrieved_ids``
    — the exact cards the chair is looking at and removing from — rather than
    dry-running retrieval. A dry run would re-issue the whole retrieval stage
    (a distiller model call under QUERY_STRATEGY=distill, plus embeddings), and
    being non-deterministic it could disagree with the run that follows.

    A still-standing forced policy rescues an otherwise-empty set: "remove all
    three and ground on this one instead" is the swap workflow this feature
    exists for, and must not be rejected. Lineage roots are resolved with the
    same helper the pipeline filter uses, so the two can never disagree about
    what an exclusion covers.

    Fails OPEN: a legacy row with no stored context has nothing to compare
    against, so the request proceeds (the pipeline degrades safely on its own).
    """
    if not excluded_policy_ids:
        return False
    current = (email.retrieval_context or {}).get("retrieved_ids") or []
    if not current:
        return False

    roots = await resolve_lineage_roots(
        db,
        list(current)
        + list(excluded_policy_ids)
        + ([forced_policy_key] if forced_policy_key else []),
        policy_repo,
    )
    excluded_roots = {roots.get(x, x) for x in excluded_policy_ids}

    if any(roots.get(i, i) not in excluded_roots for i in current):
        return False  # something survives the removal
    # Nothing of the current set survives — only a non-excluded forced pick saves it.
    return not (
        forced_policy_key
        and roots.get(forced_policy_key, forced_policy_key) not in excluded_roots
    )


def _redraft_audit_extra(
    forced_policy_key: str | None, excluded_policy_ids: list[str] | None
) -> dict:
    """Audit payload for a manual re-draft — only the knobs actually used.

    A plain retry stays ``{}`` exactly as before, so existing audit rows and the
    tests asserting on them are unaffected.
    """
    extra: dict = {}
    if forced_policy_key:
        extra["forced_policy_key"] = forced_policy_key
    if excluded_policy_ids:
        extra["excluded_policy_ids"] = excluded_policy_ids
    return extra


async def _redraft_email_bg(
    email_id: str,
    forced_policy_key: str | None = None,
    excluded_policy_ids: list[str] | None = None,
) -> None:
    """Re-run the full pipeline for one email in its OWN session (retry action).

    Scheduled after the endpoint returns (the request's session is closed by
    then). On success the fresh draft overwrites the row and ``redrafting`` is
    cleared by ``reprocess_email``. On failure, clear the flag so the ticket is
    not stranded showing "re-drafting…".

    ``forced_policy_key`` (manual invoke) is passed straight through to the
    pipeline, which grounds on that policy in ADDITION to normal retrieval.
    ``excluded_policy_ids`` (chair removals) likewise rides through to the
    pipeline; the filtering itself lands in a later step.
    """
    pipeline = EmailPipeline()
    try:
        async with async_session_factory() as db:
            email = await email_repo.get_email_by_id(db, email_id)
            if email is None:
                return
            await pipeline.reprocess_email(
                db,
                email,
                forced_policy_key=forced_policy_key,
                excluded_policy_ids=excluded_policy_ids,
            )
            await audit_repo.log_action(
                db,
                email_id,
                "email_retried",
                "chair",
                _redraft_audit_extra(forced_policy_key, excluded_policy_ids),
            )
    except Exception:  # noqa: BLE001 - a failed retry must not crash the worker
        logger.exception("Retry re-draft failed for email %s; clearing flag.", email_id)
        async with async_session_factory() as db:
            await email_repo.set_redrafting(db, email_id, False)


@router.post("/{email_id}/redraft", status_code=status.HTTP_202_ACCEPTED)
async def redraft_email(
    email_id: str,
    background_tasks: BackgroundTasks,
    payload: RedraftRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Retry: re-run the full pipeline on this email and overwrite its draft.

    Marks the ticket ``redrafting`` (surfaced live as the "re-drafting…" badge,
    exactly like a policy-change sweep), then re-classifies → re-retrieves →
    re-routes → re-drafts in the background, clearing the flag when the new draft
    lands. 404 if the email is unknown.

    The body is OPTIONAL: no body (or an empty one) is the unchanged retry. A
    ``forced_policy_key`` grounds the new draft on that policy in addition to
    whatever retrieval ranks — see ``EmailPipeline._force_policy_chunk`` for the
    active-only rule and the ignore-on-miss behaviour. ``excluded_policy_ids``
    are policies the chair removed from the grounding; the value is threaded to
    the pipeline here, and the filter that acts on it lands in a later step.
    """
    email = await email_repo.get_email_by_id(db, email_id)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email {email_id} not found",
        )
    forced_policy_key = payload.forced_policy_key if payload else None
    excluded_policy_ids = payload.excluded_policy_ids if payload else None

    if await _would_empty_the_grounding(
        db, email, excluded_policy_ids, forced_policy_key
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Removing these policies would leave the draft with no policy "
                "context to work from. Keep at least one policy, or add a "
                "replacement before removing the rest."
            ),
        )

    await email_repo.set_redrafting(db, email_id, True)
    # The audit write publishes an SSE event → queue/detail flip to "re-drafting…".
    await audit_repo.log_action(
        db,
        email_id,
        "email_retry_started",
        "chair",
        _redraft_audit_extra(forced_policy_key, excluded_policy_ids),
    )
    background_tasks.add_task(
        _redraft_email_bg, email_id, forced_policy_key, excluded_policy_ids
    )
    return {
        "email_id": email_id,
        "redrafting": True,
        "forced_policy_key": forced_policy_key,
        "excluded_policy_ids": excluded_policy_ids,
    }
