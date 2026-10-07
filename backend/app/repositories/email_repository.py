"""Email persistence (the Email aggregate of the persistence layer).

All access to the `emails` table goes through this repository — the pipeline
and API layers never touch SQLAlchemy directly. Every method is async and uses
the 2.0-style `select()` API. Reads return ``None`` / ``[]`` on miss rather
than raising; writes commit then refresh so callers get a live, populated row.

Note on ids: the ``Email`` primary key is an integer (autoincrement). The
method signatures accept ``str`` (matching the API/spec contract, where ids
arrive as path/query strings) and coerce internally; a non-numeric id resolves
to "not found" rather than an error.
"""

from datetime import datetime

from typing import Literal

from sqlalchemy import String, and_, cast, exists, func, not_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, selectinload

from app.db.models import (
    Email,
    EmailProcessingResult,
    EmailThreadMessage,
    ZendeskChairNote,
)
from app.models.appeal_queue import (
    AI_SUGGESTION_MODES,
    COMPOSED_MODES,
    INVESTIGATE_MODES,
    MODE_GROUP_AI_SUGGESTION,
    MODE_GROUP_CHAIR_WRITES,
    MODE_GROUP_COMPOSED,
    MODE_GROUP_INVESTIGATE,
    MODE_GROUP_NOT_DRAFTED,
    MODE_GROUP_RECIPROCAL,
    MODE_GROUPS,
    NOTE_STATE_NONE,
    NOTE_STATES,
    RECIPROCAL_MODES,
    mode_group,
)
from app.models.enums import EmailSource, EmailStatus
from app.pipeline.taxonomy import REJECT_APPEAL_INTENTS

# Solved/closed are ONE bucket for queue filtering: a Zendesk ticket
# auto-transitions solved -> closed over time with no further chair action, so
# the two are operationally identical. "solved" is REUSED as the combined bucket
# alias (chosen over a new "solved_or_closed" value so the frontend's existing
# "Solved" status row needs no new value plumbing): filtering the queue by
# zendesk_status="solved" matches BOTH solved and closed rows, and the facets
# report a single "solved" entry. There is deliberately no separate "closed"
# filter value. Every other status (new/open/pending/hold) stays exact-match.
_SOLVED_BUCKET_ALIAS = "solved"
_SOLVED_BUCKET_STATUSES = ("solved", "closed")

# Zendesk statuses that mean "resolved — do not re-draft on a policy change".
# Deliberately a SEPARATE constant from _SOLVED_BUCKET_STATUSES above even though
# the values coincide today: that one is a display decision (which statuses the
# queue's "Solved / Closed" row folds together), this one is a pipeline gate
# (which tickets a KB-change sweep may re-draft). Splitting "closed" back out as
# its own queue filter must not silently change what the sweep re-drafts, so the
# two concerns stay independently changeable.
_RESOLVED_ZENDESK_STATUSES = ("solved", "closed")

# The extraction key that marks an email as a reply to an OpenReview
# notification (a computed field on the pipeline's ExtractionResult, persisted
# inside the `extraction` JSON column).
_OPENREVIEW_CANDIDATE_KEY = "openreview_reply_candidate"

# How a caller wants the OpenReview split applied. Tri-state rather than a bool
# because "no opinion" is a REAL and load-bearing third case: analytics
# aggregates over `get_email_queue` and must keep counting every email, so the
# default must leave the set untouched. Both queue endpoints opt in explicitly.
OpenReviewQueueMode = Literal["exclude", "only"]


def _unresolved_openreview_candidate():
    """Emails detected as an OpenReview reply that still need chair action.

    These are pulled OUT of the main queue and shown in their own queue instead,
    so this one predicate is the SINGLE definition of the split — the main queue
    negates it, the OpenReview queue asserts it. Deriving both from one
    expression is what makes them exact complements: no email can be hidden from
    both queues, and none can appear in both.

    ⚠️ ``.as_boolean().is_(True)``, NEVER ``== True``. The two are not
    interchangeable here. ``.is_(True)`` renders ``JSON_EXTRACT(...) IS 1`` on
    SQLite and ``CAST(... AS BOOLEAN) IS true`` on Postgres — ``IS`` comparisons
    are TOTAL, so a row whose ``extraction`` is NULL (every row processed before
    that column existed) yields FALSE and is correctly kept by the negation.
    ``== True`` renders ``!= 1`` / ``!= true`` under negation, which evaluates to
    NULL for those rows and would SILENTLY DROP every legacy email from the main
    queue. Same three-valued-logic trap as the ``NULL NOT IN (...)`` note in
    :meth:`EmailRepository.get_open_tickets`, and verified empirically on both
    dialects rather than assumed.

    "Unresolved" reuses :data:`_RESOLVED_ZENDESK_STATUSES` — the codebase's
    existing PIPELINE gate ("is there still work to do on this ticket"), the same
    constant ``get_open_tickets`` uses — deliberately NOT the display-only
    ``_SOLVED_BUCKET_STATUSES``. The explicit ``IS NULL`` branch is load-bearing
    for the same reason it is there: non-Zendesk rows carry a NULL status, and a
    bare ``NOT IN`` would evaluate to NULL and drop them.

    A resolved candidate is therefore NOT pulled out of the main queue — its
    visibility is exactly what it was before this split existed.
    """
    return and_(
        Email.extraction[_OPENREVIEW_CANDIDATE_KEY].as_boolean().is_(True),
        # A chair's dismissal of a false positive. Read from a COLUMN, never
        # from the extraction dict: that dict is rewritten from the email text on
        # every pipeline pass, so a dismissal stored there would be recomputed
        # away on the next follow-up or re-draft. See the note on
        # ``Email.openreview_candidate_dismissed``.
        #
        # ``.is_(False)`` for the same total-comparison reason as the line above,
        # even though the column is NOT NULL: the negated form used by the main
        # queue must stay NULL-safe by construction rather than by relying on a
        # constraint elsewhere continuing to hold.
        Email.openreview_candidate_dismissed.is_(False),
        or_(
            Email.zendesk_status.is_(None),
            Email.zendesk_status.not_in(_RESOLVED_ZENDESK_STATUSES),
        ),
    )


# Zendesk-origin columns the ingest adapter may set/patch on an Email row.
# Kept as an allow-list so ``apply_zendesk_fields`` can never write an arbitrary
# attribute, mirroring the guarded key set in ``update_email_status``.
_ZENDESK_FIELDS = frozenset(
    {
        "source",
        "zendesk_ticket_id",
        "zendesk_requester_id",
        "zendesk_status",
        "zendesk_created_at",
        "zendesk_updated_at",
        "last_processed_comment_id",
    }
)


def _queue_conditions(
    lane: str | None,
    chair_id: int | None,
    status: str | None,
    search: str | None,
    unassigned: bool,
    source: str | None = None,
    zendesk_status: str | None = None,
    received_after: datetime | None = None,
    received_before: datetime | None = None,
    *,
    openreview_candidates: OpenReviewQueueMode | None = None,
) -> list:
    """Build the shared WHERE conditions for the queue list AND its count.

    Kept in one place so ``get_email_queue`` and ``count_email_queue`` filter
    identically — the page and the total can never disagree. All filters are
    server-side: the lane lives in the ``routing`` JSON column, ``chair_id`` /
    ``unassigned`` on the ``assigned_chair_id`` FK, ``status`` on the column,
    ``source`` / ``zendesk_status`` on their columns, and ``search`` is a
    case-insensitive substring match on subject, sender, OR zendesk_ticket_id
    (a leading ``#`` is stripped first, so a ``#21567`` pasted from the UI badge
    matches the numeric ticket id). ``zendesk_status="solved"`` is
    the combined solved+closed bucket (see _SOLVED_BUCKET_*); other statuses are
    exact-match.

    ``openreview_candidates`` applies the OpenReview split (see
    :func:`_unresolved_openreview_candidate`): ``"exclude"`` removes unresolved
    candidates (the main queue), ``"only"`` keeps nothing else (the OpenReview
    queue), ``None`` leaves the set untouched. It is threaded through HERE rather
    than applied at the call sites so the list, its ``total`` and the facet
    counts are filtered identically — the same reason every other filter lives
    here. Keyword-only, defaulting to ``None``, so the existing positional calls
    and the analytics full-table read are byte-for-byte unaffected.

    ``received_after`` / ``received_before`` bound ``received_at``, INCLUSIVE at
    both ends (``>=`` / ``<=``), and are independent: either may be given alone
    for an open-ended range. They are the "when did this arrive" filter, so they
    read ``received_at`` and NOTHING ELSE — deliberately never falling back to
    ``zendesk_created_at`` for non-Zendesk rows. That column is NULL for
    ``/ingest`` and toy rows, and under SQL three-valued logic every comparison
    against NULL yields NULL, so such a fallback would silently drop those rows
    from every range query. ``received_at`` is NOT NULL for every row, which is
    what makes it safe to compare directly.
    """
    conditions: list = []
    if lane is not None:
        # Dialect-agnostic JSON access: SQLAlchemy renders JSON_EXTRACT on
        # SQLite and the ->> / #>> operators on PostgreSQL. A bare
        # func.json_extract() is SQLite-only and raises UndefinedFunctionError
        # on Postgres.
        conditions.append(Email.routing["lane"].as_string() == lane)
    if chair_id is not None:
        conditions.append(Email.assigned_chair_id == chair_id)
    if unassigned:
        conditions.append(Email.assigned_chair_id.is_(None))
    if status is not None:
        conditions.append(Email.status == status)
    if source is not None:
        conditions.append(Email.source == source)
    if zendesk_status is not None:
        # "solved" is the combined solved+closed bucket (see _SOLVED_BUCKET_*);
        # any other value is a plain exact-match.
        if zendesk_status == _SOLVED_BUCKET_ALIAS:
            conditions.append(Email.zendesk_status.in_(_SOLVED_BUCKET_STATUSES))
        else:
            conditions.append(Email.zendesk_status == zendesk_status)
    if received_after is not None:
        conditions.append(Email.received_at >= received_after)
    if received_before is not None:
        conditions.append(Email.received_at <= received_before)
    if search:
        # Strip a single leading "#" so a chair pasting "#21567" from the ticket
        # badge matches the numeric zendesk_ticket_id. Applied to ALL three
        # clauses (one shared pattern): a subject/sender that starts with "#" is
        # effectively nonexistent, and stripping only broadens their substring
        # match (never breaks it), so it's harmless there.
        term = search[1:] if search.startswith("#") else search
        pattern = f"%{term}%"
        conditions.append(
            or_(
                Email.subject.ilike(pattern),
                Email.sender.ilike(pattern),
                # zendesk_ticket_id is a BigInteger — cast to text so partial
                # ticket-number typing does substring matching like the others.
                # cast(..., String).ilike() is dialect-agnostic (SQLite default +
                # Postgres): renders CAST(... AS VARCHAR) with LIKE/ILIKE. NULL
                # ids cast to NULL and simply don't match.
                cast(Email.zendesk_ticket_id, String).ilike(pattern),
            )
        )
    if openreview_candidates == "exclude":
        conditions.append(not_(_unresolved_openreview_candidate()))
    elif openreview_candidates == "only":
        conditions.append(_unresolved_openreview_candidate())
    return conditions


# --- Reject Appeals queue (Z3a) --------------------------------------------
# A VIEW: these conditions only ever narrow a read. Nothing here touches the
# main queue's conditions, so its membership is unchanged.
_REJECT_APPEAL_INTENTS: tuple[str, ...] = tuple(sorted(REJECT_APPEAL_INTENTS))
# Note states that take an email OUT of needs_note: posted, and posting (in flight,
# or stale after a crash — Zendesk may already have the note, D125). "failed" is
# deliberately absent: a failed note still needs one (a retry).
_NOTE_STATES_NOT_NEEDING_A_NOTE: tuple[str, ...] = ("posting", "posted")


def _appeal_mode_expr():
    """``draft.appeal_reply.mode`` as text (NULL when absent), on both dialects.

    The tuple index renders JSON_EXTRACT(draft, '$."appeal_reply"."mode"') on
    SQLite and ``draft #>> '{appeal_reply,mode}'`` on Postgres.
    """
    return Email.draft[("appeal_reply", "mode")].as_string()


def _has_chair_note(note_status: str | None = None):
    """EXISTS a chair-note row for the outer email (optionally in one state).

    Uses its OWN alias of the note table: a query that also joins
    ``zendesk_chair_notes`` (the counts query does) would otherwise auto-correlate
    the subquery's table away and leave it with no FROM clause.
    """
    note = aliased(ZendeskChairNote)
    conditions = [note.email_id == Email.id]
    if note_status is not None:
        conditions.append(note.status == note_status)
    return exists(select(note.id).where(*conditions))


def _reject_appeal_member():
    """Membership: an appeal intent, OR an email that already has a chair note.

    The second arm keeps an email in the view after a follow-up re-classifies
    it away from the appeal intent, once a note exists for it.
    """
    return or_(
        Email.classification["intent"].as_string().in_(_REJECT_APPEAL_INTENTS),
        _has_chair_note(),
    )


def _mode_group_condition(group: str):
    """SQL for one mode group; mirrors ``app.models.appeal_queue.mode_group``."""
    mode = _appeal_mode_expr()
    if group == MODE_GROUP_COMPOSED:
        return mode.in_(COMPOSED_MODES)
    if group == MODE_GROUP_INVESTIGATE:
        return mode.in_(INVESTIGATE_MODES)
    if group == MODE_GROUP_RECIPROCAL:
        return mode.in_(RECIPROCAL_MODES)
    if group == MODE_GROUP_NOT_DRAFTED:
        return mode.is_(None)
    if group == MODE_GROUP_AI_SUGGESTION:
        return mode.in_(AI_SUGGESTION_MODES)
    if group == MODE_GROUP_CHAIR_WRITES:
        return and_(
            mode.is_not(None),
            mode.not_in(
                COMPOSED_MODES + INVESTIGATE_MODES + RECIPROCAL_MODES + AI_SUGGESTION_MODES
            ),
        )
    raise ValueError(f"unknown mode group {group!r}")


def _reject_appeal_conditions(
    status: str | None = None,
    zendesk_status: str | None = None,
    received_after: datetime | None = None,
    received_before: datetime | None = None,
    search: str | None = None,
    mode_group: str | None = None,
    reciprocal: bool | None = None,
    note_status: str | None = None,
) -> list:
    """The WHERE conditions shared by the appeals list AND its total.

    Email status, Zendesk status, the received range and search are the main
    queue's own conditions (:func:`_queue_conditions`), reused as they are.
    ``reciprocal`` uses an IS comparison so a row without the flag counts as
    not reciprocal rather than dropping out under NOT. ``note_status`` "none"
    means no ``zendesk_chair_notes`` row.
    """
    conditions = _queue_conditions(
        lane=None,
        chair_id=None,
        status=status,
        search=search,
        unassigned=False,
        source=None,
        zendesk_status=zendesk_status,
        received_after=received_after,
        received_before=received_before,
    )
    conditions.append(_reject_appeal_member())
    if mode_group is not None:
        conditions.append(_mode_group_condition(mode_group))
    if reciprocal is not None:
        flag = Email.extraction["is_reciprocal_dispute"].as_boolean().is_(True)
        conditions.append(flag if reciprocal else not_(flag))
    if note_status is not None:
        if note_status == NOTE_STATE_NONE:
            conditions.append(not_(_has_chair_note()))
        else:
            conditions.append(_has_chair_note(note_status))
    return conditions


def _coerce_id(email_id: str) -> int | None:
    """Best-effort coercion of a string id to the integer PK.

    Returns ``None`` when the value cannot be an integer key, so callers can
    treat it as a clean not-found instead of raising.
    """
    try:
        return int(email_id)
    except (TypeError, ValueError):
        return None


class EmailRepository:
    """Async data-access methods for the ``emails`` table."""

    # --- writes -----------------------------------------------------------
    async def create_email(self, db: AsyncSession, email_data: dict) -> Email:
        """Insert a new email row and return the persisted instance."""
        email = Email(**email_data)
        db.add(email)
        await db.commit()
        await db.refresh(email)
        return email

    async def update_email_status(
        self,
        db: AsyncSession,
        email_id: str,
        status: str,
        metadata: dict = {},
    ) -> Email | None:
        """Update an email's status (and optional pipeline-output columns).

        ``metadata`` keys that match real ``Email`` columns (e.g.
        ``classification``, ``routing``, ``draft``) are applied as a convenience
        so a status transition and its produced artifact can be persisted in one
        call. Unknown keys are ignored. Returns ``None`` if the email is absent.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None

        result = await db.execute(select(Email).where(Email.id == pk))
        email = result.scalar_one_or_none()
        if email is None:
            return None

        email.status = status
        for key, value in metadata.items():
            if key in {"classification", "routing", "draft"}:
                setattr(email, key, value)

        await db.commit()
        await db.refresh(email)
        return email

    async def dismiss_openreview_candidate(
        self, db: AsyncSession, email_id: str
    ) -> Email | None:
        """Mark an email as NOT an OpenReview reply candidate. Returns the row.

        Writes exactly one column and nothing else. Deliberately NOT routed
        through ``update_email_status``, which would also overwrite ``status``:
        a dismissal is a correction to the DETECTION, not a lifecycle
        transition, and moving the email's workflow status would change what the
        main queue shows about work still to be done on it.

        ``routing`` is untouched for the same reason. The email keeps whatever
        lane normal routing assigned it and simply resumes being an ordinary
        email — there is nothing to re-route, because the lane was never the
        thing that was wrong.

        Idempotent: dismissing an already-dismissed row is a no-op write. The
        caller decides whether that is worth auditing.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None

        result = await db.execute(select(Email).where(Email.id == pk))
        email = result.scalar_one_or_none()
        if email is None:
            return None

        email.openreview_candidate_dismissed = True
        await db.commit()
        await db.refresh(email)
        return email

    async def assign_chair(
        self, db: AsyncSession, email_id: str, chair_id: int | None
    ) -> Email | None:
        """Set an email's ``assigned_chair_id`` (a chair (re)assignment).

        Kept separate from ``update_email_status`` because a chair reassignment
        is not a lifecycle-status change — the email stays in the human-review
        lane, only its owning chair changes. Returns ``None`` if the email is
        absent or the id is non-numeric.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None
        result = await db.execute(select(Email).where(Email.id == pk))
        email = result.scalar_one_or_none()
        if email is None:
            return None
        email.assigned_chair_id = chair_id
        await db.commit()
        await db.refresh(email)
        return email

    async def clear_all_redrafting_flags(self, db: AsyncSession) -> int:
        """Clear the ``redrafting`` flag on every row; returns rows cleared.

        Used at process startup to recover flags stranded by a crash mid-sweep — a
        fresh process has no in-flight sweep, so any set flag is stale.
        """
        result = await db.execute(
            update(Email).where(Email.redrafting.is_(True)).values(redrafting=False)
        )
        await db.commit()
        return result.rowcount or 0

    async def update_email_outputs(
        self, db: AsyncSession, email_id: str, record: dict
    ) -> Email | None:
        """Overwrite an existing email's pipeline outputs in place (retry/reprocess).

        Applies the fresh status + classification/routing/draft/assigned_chair_id/
        retrieval_context/extraction from a re-run, and clears the transient ``redrafting``
        flag. Leaves id, received_at, and sender/subject/body untouched. Returns
        the refreshed row, or ``None`` if the id is missing/non-numeric.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None
        result = await db.execute(select(Email).where(Email.id == pk))
        email = result.scalar_one_or_none()
        if email is None:
            return None
        for key in (
            "status", "classification", "routing", "draft",
            "assigned_chair_id", "retrieval_context", "extraction",
        ):
            if key in record:
                setattr(email, key, record[key])
        email.redrafting = False
        await db.commit()
        await db.refresh(email)
        return email

    async def apply_zendesk_fields(
        self, db: AsyncSession, email_id: str, fields: dict
    ) -> Email | None:
        """Patch the Zendesk-origin columns on an existing Email row.

        Only keys in :data:`_ZENDESK_FIELDS` are applied (others ignored), so a
        caller can't smuggle in an arbitrary column write. Used by the ingest
        adapter to decorate the row the orchestrator created (source, ticket id,
        status, timestamps, last-processed comment). Returns ``None`` if absent.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None
        result = await db.execute(select(Email).where(Email.id == pk))
        email = result.scalar_one_or_none()
        if email is None:
            return None
        for key, value in fields.items():
            if key in _ZENDESK_FIELDS:
                setattr(email, key, value)
        await db.commit()
        await db.refresh(email)
        return email

    async def claim_for_redraft(self, db: AsyncSession, email_id: str) -> bool:
        """Atomically claim an open ticket for re-drafting.

        A single conditional UPDATE flips ``redrafting`` False→True only while the
        ticket is still an open auto-draft (status draft_generated) and not already
        being re-drafted. Returns True iff THIS call won the claim. Because the flip
        is one SQL statement, two overlapping sweeps can never both claim the same
        ticket, and a ticket approved since the sweep started is not claimed.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return False
        result = await db.execute(
            update(Email)
            .where(
                Email.id == pk,
                Email.status == EmailStatus.DRAFT_GENERATED.value,
                Email.redrafting.is_(False),
            )
            .values(redrafting=True)
        )
        await db.commit()
        return (result.rowcount or 0) == 1

    async def set_redrafting(
        self, db: AsyncSession, email_id: str, value: bool
    ) -> Email | None:
        """Set/clear the transient ``redrafting`` flag unconditionally.

        A Core UPDATE (not load-set-commit) so it always writes even when the
        session's cached instance is stale — otherwise clearing a flag set by a
        prior Core UPDATE in the same session would be a silent no-op. Returns the
        refreshed row, or None if the id is missing/non-numeric.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None
        result = await db.execute(
            update(Email).where(Email.id == pk).values(redrafting=value)
        )
        await db.commit()
        if (result.rowcount or 0) != 1:
            return None
        refreshed = await db.execute(
            select(Email).where(Email.id == pk).execution_options(populate_existing=True)
        )
        return refreshed.scalar_one_or_none()

    async def save_redraft(
        self,
        db: AsyncSession,
        email_id: str,
        *,
        draft: dict,
        routing: dict,
        retrieval_context: dict,
    ) -> Email | None:
        """Persist a re-drafted ticket IFF it is still a claimed open draft.

        A single conditional UPDATE overwrites draft + routing + retrieval_context
        and clears ``redrafting`` only while status is still draft_generated AND the
        ticket is still claimed (``redrafting`` True). If the chair approved/changed
        the ticket between claim and save, 0 rows match and this returns ``None``
        WITHOUT clobbering their work (the caller then clears the stray flag).
        Status is left unchanged. Returns ``None`` if the id is missing/non-numeric
        or the precondition no longer holds.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None
        result = await db.execute(
            update(Email)
            .where(
                Email.id == pk,
                Email.status == EmailStatus.DRAFT_GENERATED.value,
                Email.redrafting.is_(True),
            )
            .values(
                draft=draft,
                routing=routing,
                retrieval_context=retrieval_context,
                redrafting=False,
            )
        )
        await db.commit()
        if (result.rowcount or 0) != 1:
            return None
        refreshed = await db.execute(
            select(Email).where(Email.id == pk).execution_options(populate_existing=True)
        )
        return refreshed.scalar_one_or_none()

    async def get_thread_comment_ids(
        self, db: AsyncSession, email_id: str
    ) -> set[int]:
        """Return the set of Zendesk comment ids already stored for an email.

        Lets the adapter add only genuinely new comments (dedup by
        ``zendesk_comment_id``) instead of re-inserting the whole thread each poll.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return set()
        result = await db.execute(
            select(EmailThreadMessage.zendesk_comment_id).where(
                EmailThreadMessage.email_id == pk,
                EmailThreadMessage.zendesk_comment_id.is_not(None),
            )
        )
        return {cid for cid in result.scalars().all() if cid is not None}

    async def add_thread_messages(
        self, db: AsyncSession, email_id: str, messages: list[dict]
    ) -> list[EmailThreadMessage]:
        """Bulk-insert thread messages for an email; return the persisted rows.

        Each dict is an ``EmailThreadMessage`` field mapping (without
        ``email_id``, which is set here). Commits once for the batch. The
        returned rows carry their populated primary keys (the session factory
        uses ``expire_on_commit=False``), so a caller can link follow-up
        artifacts (e.g. an ``EmailProcessingResult``) to a specific message.
        Returns ``[]`` when there is nothing to add.
        """
        pk = _coerce_id(email_id)
        if pk is None or not messages:
            return []
        # Idempotent: drop comments already stored so a re-fetch of the same
        # ticket (the background poll cycle vs the post-send single-ticket
        # reconcile) can't double-insert and trip the unique zendesk_comment_id
        # constraint. Callers already pre-filter; this backstops that and makes
        # the method safe to call with an unfiltered batch. (NULL comment ids —
        # not from Zendesk — are kept; NULLs don't collide under UNIQUE.)
        existing = await self.get_thread_comment_ids(db, email_id)
        fresh = [
            data
            for data in messages
            if data.get("zendesk_comment_id") is None
            or data.get("zendesk_comment_id") not in existing
        ]
        if not fresh:
            return []
        rows = [EmailThreadMessage(email_id=pk, **data) for data in fresh]
        db.add_all(rows)
        try:
            await db.commit()
            return rows
        except IntegrityError:
            # Ultra-rare genuine concurrent insert of the same comment (separate
            # DB connections committing at once). The other writer won and already
            # stored it, so recover the session and yield nothing new.
            await db.rollback()
            return []

    async def add_processing_result(
        self, db: AsyncSession, thread_message_id: int, record: dict
    ) -> EmailProcessingResult:
        """Persist a per-message pipeline result (Piece T2) and return it.

        A follow-up requester message gets its own classify→retrieve→route→draft
        cycle stored HERE — a NEW ``EmailProcessingResult`` row linked to the
        message — leaving the parent ``Email``'s own classification/routing/draft
        untouched. ``record`` is the ``_Computed.record`` dict the orchestrator
        builds; the pipeline-output columns are lifted from it, with ``lane`` and
        ``confidence`` denormalized from the routing/classification sub-dicts.
        """
        classification = record.get("classification") or {}
        routing = record.get("routing") or {}
        row = EmailProcessingResult(
            thread_message_id=thread_message_id,
            classification=classification or None,
            routing=routing or None,
            draft=record.get("draft"),
            retrieval_context=record.get("retrieval_context"),
            extraction=record.get("extraction"),
            lane=routing.get("lane"),
            confidence=classification.get("confidence"),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row

    # --- reads ------------------------------------------------------------
    async def get_by_zendesk_ticket_id(
        self, db: AsyncSession, ticket_id: int
    ) -> Email | None:
        """Return the Email mapped to a Zendesk ticket id, or ``None``.

        The dedup lookup (ZENDESK_API.md §10): a ticket upserts to exactly one
        Email row via the unique ``zendesk_ticket_id`` column.
        """
        result = await db.execute(
            select(Email).where(Email.zendesk_ticket_id == ticket_id)
        )
        return result.scalar_one_or_none()

    async def get_thread_messages(
        self, db: AsyncSession, email_id: str
    ) -> list[dict]:
        """Return a ticket's thread messages oldest-first (ALL of them).

        Includes internal notes (``public`` False) so the UI can show the full
        conversation; callers that must hide internal notes (the transcript
        builder) filter them out. ``created_at`` is returned as a datetime for
        ordering/transcript use; the API layer serializes it.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return []
        result = await db.execute(
            select(EmailThreadMessage)
            .where(EmailThreadMessage.email_id == pk)
            .order_by(
                EmailThreadMessage.created_at.asc(), EmailThreadMessage.id.asc()
            )
        )
        return [
            {
                "comment_id": r.zendesk_comment_id,
                "public": r.public,
                "author_id": r.author_id,
                "author_role": r.author_role,
                "plain_body": r.plain_body,
                "html_body": r.html_body,
                "created_at": r.created_at,
                "via_channel": r.via_channel,
            }
            for r in result.scalars().all()
        ]

    async def get_email_by_id(
        self, db: AsyncSession, email_id: str
    ) -> Email | None:
        """Return a single email by id, or ``None`` if not found."""
        pk = _coerce_id(email_id)
        if pk is None:
            return None
        result = await db.execute(select(Email).where(Email.id == pk))
        return result.scalar_one_or_none()

    async def get_email_by_zendesk_ticket_id(
        self, db: AsyncSession, ticket_id: int
    ) -> Email | None:
        """Return a single email by its Zendesk ticket id, or ``None``.

        Mirrors ``get_email_by_id`` in style. ``zendesk_ticket_id`` is unique and
        nullable — a row with a NULL ticket id can never match an integer
        ``ticket_id`` (``NULL == n`` is never true in SQL), so non-Zendesk rows
        are inherently excluded.
        """
        result = await db.execute(
            select(Email).where(Email.zendesk_ticket_id == ticket_id)
        )
        return result.scalar_one_or_none()

    async def get_email_with_thread(
        self, db: AsyncSession, email_id: str
    ) -> Email | None:
        """Return an email with its thread messages + per-message results loaded.

        Eager-loads ``thread_messages`` and each message's ``processing_results``
        (Piece T2) in one round-trip, so the thread endpoint can serialize them
        without triggering an async lazy-load. Kept SEPARATE from
        ``get_email_by_id`` on purpose: that method is on the hot queue/detail/
        action paths and must not pay the eager-load cost. Both relationships
        keep their model-defined ordering (thread messages oldest-first by
        ``created_at``; results oldest-first by ``created_at``/``id``). Returns
        ``None`` if the id is missing/non-numeric.
        """
        pk = _coerce_id(email_id)
        if pk is None:
            return None
        result = await db.execute(
            select(Email)
            .where(Email.id == pk)
            .options(
                selectinload(Email.thread_messages).selectinload(
                    EmailThreadMessage.processing_results
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_emails_by_status(
        self,
        db: AsyncSession,
        status: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Email]:
        """Return emails in a given status, newest first, paginated."""
        result = await db.execute(
            select(Email)
            .where(Email.status == status)
            .order_by(Email.received_at.desc(), Email.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_emails_by_intent(
        self, db: AsyncSession, intent: str
    ) -> list[Email]:
        """Return every email whose stored classification intent is ``intent``.

        Oldest id first. Uses the dialect-agnostic JSON accessor, like the lane
        filter in ``_queue_conditions``; rows with no classification never match.
        """
        result = await db.execute(
            select(Email)
            .where(Email.classification["intent"].as_string() == intent)
            .order_by(Email.id.asc())
        )
        return list(result.scalars().all())

    async def get_email_queue(
        self,
        db: AsyncSession,
        lane: str | None = None,
        chair_id: int | None = None,
        status: str | None = None,
        search: str | None = None,
        unassigned: bool = False,
        source: str | None = None,
        zendesk_status: str | None = None,
        received_after: datetime | None = None,
        received_before: datetime | None = None,
        limit: int = 20,
        offset: int = 0,
        *,
        openreview_candidates: OpenReviewQueueMode | None = None,
    ) -> list[Email]:
        """Return the email queue, filtered server-side by any combination of
        lane / chair / unassigned / status / source / zendesk_status / search /
        received_at range.

        Every filter is applied in SQL (via :func:`_queue_conditions`), so the
        returned page is a slice of the FULL matching set — callers never filter
        a truncated page client-side. Ordered newest first, which is the same
        column ``received_after`` / ``received_before`` bound, so a date range
        narrows this ordering rather than fighting it.
        """
        conditions = _queue_conditions(
            lane, chair_id, status, search, unassigned, source, zendesk_status,
            received_after, received_before,
            openreview_candidates=openreview_candidates,
        )
        stmt = (
            select(Email)
            .where(*conditions)
            .order_by(Email.received_at.desc(), Email.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_emails_by_status(self, db: AsyncSession) -> dict[str, int]:
        """Return a mapping of status -> count across all emails."""
        result = await db.execute(
            select(Email.status, func.count(Email.id)).group_by(Email.status)
        )
        return {status: count for status, count in result.all()}

    async def count_email_queue(
        self,
        db: AsyncSession,
        lane: str | None = None,
        chair_id: int | None = None,
        status: str | None = None,
        search: str | None = None,
        unassigned: bool = False,
        source: str | None = None,
        zendesk_status: str | None = None,
        received_after: datetime | None = None,
        received_before: datetime | None = None,
        *,
        openreview_candidates: OpenReviewQueueMode | None = None,
    ) -> int:
        """Return the total number of emails matching the queue filters.

        Uses the SAME :func:`_queue_conditions` as ``get_email_queue`` so the
        count and the page always agree. With no filters this is the whole table;
        with filters it is that slice's true total — page-size independent, so
        callers can show an accurate count regardless of ``limit``/``offset``.
        """
        conditions = _queue_conditions(
            lane, chair_id, status, search, unassigned, source, zendesk_status,
            received_after, received_before,
            openreview_candidates=openreview_candidates,
        )
        stmt = select(func.count()).select_from(Email).where(*conditions)
        result = await db.execute(stmt)
        return int(result.scalar_one())

    # --- Reject Appeals queue (Z3a) ---------------------------------------
    async def get_reject_appeal_queue(
        self,
        db: AsyncSession,
        *,
        limit: int = 20,
        offset: int = 0,
        **filters,
    ) -> list[Email]:
        """One page of the Reject Appeals view, newest first.

        ``filters`` are those of :func:`_reject_appeal_conditions`, the SAME
        conditions :meth:`count_reject_appeal_queue` uses, so the page and its
        total can never disagree.
        """
        stmt = (
            select(Email)
            .where(*_reject_appeal_conditions(**filters))
            .order_by(Email.received_at.desc(), Email.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_reject_appeal_queue(self, db: AsyncSession, **filters) -> int:
        """The total number of Reject Appeals view rows matching ``filters``."""
        stmt = (
            select(func.count())
            .select_from(Email)
            .where(*_reject_appeal_conditions(**filters))
        )
        return int((await db.execute(stmt)).scalar_one())

    async def reject_appeal_counts(self, db: AsyncSession) -> dict:
        """Whole-view counts for the nav badge and the filter chips. Unfiltered.

        One grouped query over every view member (never a tally over a page):
        ``total``; ``by_mode_group`` (every group present, zeros included);
        ``by_note_status`` ("none" plus every note state); ``without_approved_draft``
        (the ``not_drafted`` group); and ``needs_note``: drafted by the approved
        rules, on a Zendesk ticket that is not solved or closed, whose note is
        neither posted nor in flight (``posting`` — Zendesk may already have it,
        D125). A ``failed`` note still counts: it is waiting for a retry.
        """
        mode = _appeal_mode_expr().label("mode")
        stmt = (
            select(
                mode,
                ZendeskChairNote.status,
                Email.zendesk_status,
                Email.zendesk_ticket_id.is_not(None),
                func.count(Email.id),
            )
            .select_from(Email)
            .outerjoin(ZendeskChairNote, ZendeskChairNote.email_id == Email.id)
            .where(_reject_appeal_member())
            # Grouped by the LABEL, not a second copy of the JSON expression:
            # the JSON path is a bound parameter, so two copies render as two
            # different parameters and Postgres would not see them as one.
            .group_by(
                mode,
                ZendeskChairNote.status,
                Email.zendesk_status,
                Email.zendesk_ticket_id.is_not(None),
            )
        )
        by_mode = {group: 0 for group in MODE_GROUPS}
        by_note = {state: 0 for state in NOTE_STATES}
        total = needs_note = 0
        for mode_value, note_status, zstatus, has_ticket, count in (await db.execute(stmt)).all():
            total += count
            group = mode_group(mode_value)
            by_mode[group] += count
            note_key = note_status if note_status in by_note else NOTE_STATE_NONE
            by_note[note_key] += count
            if (
                group != MODE_GROUP_NOT_DRAFTED
                and has_ticket
                and (zstatus or "").lower() not in _RESOLVED_ZENDESK_STATUSES
                and note_status not in _NOTE_STATES_NOT_NEEDING_A_NOTE
            ):
                needs_note += count
        return {
            "needs_note": needs_note,
            "total": total,
            "by_mode_group": by_mode,
            "by_note_status": by_note,
            "without_approved_draft": by_mode[MODE_GROUP_NOT_DRAFTED],
        }

    async def count_queue_facets(
        self,
        db: AsyncSession,
        lane: str | None = None,
        chair_id: int | None = None,
        status: str | None = None,
        search: str | None = None,
        unassigned: bool = False,
        received_after: datetime | None = None,
        received_before: datetime | None = None,
        *,
        openreview_candidates: OpenReviewQueueMode | None = None,
    ) -> dict:
        """Return grouped facet counts for the queue's status bar + source toggle.

        A single dedicated aggregate (three grouped queries), NOT a tally over a
        capped queue page — the Phase 6C rule that page-derived aggregates drop
        out-of-window rows. The context filters (lane / chair / unassigned /
        status / search / received_at range) are honored so the facets COMPOSE
        with the queue's other filters, but the facet dimensions themselves
        (source, zendesk_status) are deliberately NOT applied — selecting one
        status must not zero out the others in the bar.

        The date range is a CONTEXT filter, not a facet dimension: it is not one
        of the values the bar or toggle renders, so narrowing to a date window
        should narrow the status/source counts shown alongside it, exactly as
        ``status`` and ``search`` already do.

        Returns:
          - ``by_zendesk_status``: {zendesk_status -> count} over the context,
            scoped to ``source='zendesk'`` (only Zendesk rows carry a meaningful
            zendesk_status). Rows with a NULL status are omitted. "closed" is
            folded into the "solved" bucket, so it never appears as its own entry
            (they are one filter value — see _SOLVED_BUCKET_*).
          - ``by_source``: {source -> count} over the same context.
          - ``sources``: sorted distinct non-null sources across the WHOLE table
            (unfiltered). This drives the self-hiding source toggle — it must
            reflect what exists in the data, not the current filter view.
        """
        context = _queue_conditions(
            lane,
            chair_id,
            status,
            search,
            unassigned,
            received_after=received_after,
            received_before=received_before,
            openreview_candidates=openreview_candidates,
        )

        zs_stmt = (
            select(Email.zendesk_status, func.count(Email.id))
            .where(
                *context,
                Email.source == EmailSource.ZENDESK.value,
                Email.zendesk_status.is_not(None),
            )
            .group_by(Email.zendesk_status)
        )
        by_zendesk_status = {
            zs: int(count) for zs, count in (await db.execute(zs_stmt)).all() if zs
        }
        # Fold "closed" into the "solved" bucket so the bar shows ONE combined
        # entry (they are a single filter value — see _SOLVED_BUCKET_*). Done
        # here rather than in SQL to keep the grouped query dialect-simple.
        closed_count = by_zendesk_status.pop("closed", 0)
        if closed_count:
            by_zendesk_status[_SOLVED_BUCKET_ALIAS] = (
                by_zendesk_status.get(_SOLVED_BUCKET_ALIAS, 0) + closed_count
            )

        src_stmt = (
            select(Email.source, func.count(Email.id))
            .where(*context)
            .group_by(Email.source)
        )
        by_source = {
            src: int(count) for src, count in (await db.execute(src_stmt)).all() if src
        }

        dist_stmt = (
            select(Email.source).where(Email.source.is_not(None)).distinct()
        )
        sources = sorted(
            src for src in (await db.execute(dist_stmt)).scalars().all() if src
        )

        return {
            "by_zendesk_status": by_zendesk_status,
            "by_source": by_source,
            "sources": sources,
        }

    async def get_open_tickets(self, db: AsyncSession) -> list[Email]:
        """Return every open ticket (status draft_generated), oldest id first.

        "Open" = has an auto-draft awaiting chair action and not yet approved or
        sent. These are the only tickets a KB-change sweep re-evaluates.

        A ticket already solved/closed in Zendesk is excluded even when its LOCAL
        status is still draft_generated: the two statuses are orthogonal, and
        re-drafting a resolved ticket on every policy edit is wasted model spend
        on a conversation nobody will read. This gate is scoped to the sweep only
        — a NEW customer comment on a solved ticket still reprocesses via the
        ingest adapter's follow-up path, which is intentionally unaffected here.

        NULL ``zendesk_status`` (non-Zendesk rows, e.g. toy_dataset) is allowed
        through. The NULL check is explicit because SQL three-valued logic makes
        ``NULL NOT IN (...)`` evaluate to NULL, which would silently drop every
        non-Zendesk ticket from the sweep.
        """
        result = await db.execute(
            select(Email)
            .where(
                Email.status == EmailStatus.DRAFT_GENERATED.value,
                or_(
                    Email.zendesk_status.is_(None),
                    Email.zendesk_status.not_in(_RESOLVED_ZENDESK_STATUSES),
                ),
            )
            .order_by(Email.id)
        )
        return list(result.scalars().all())

    async def count_by_chair(self, db: AsyncSession) -> dict[int, int]:
        """Return a mapping of ``assigned_chair_id`` -> count over ALL emails.

        A single grouped aggregate (not a paginated scan) so per-chair volume is
        accurate regardless of how many emails exist or how they are ordered.
        Only rows with a chair assigned are counted (NULL FKs are excluded).
        """
        stmt = (
            select(Email.assigned_chair_id, func.count(Email.id))
            .where(Email.assigned_chair_id.is_not(None))
            .group_by(Email.assigned_chair_id)
        )
        result = await db.execute(stmt)
        return {int(chair_id): int(count) for chair_id, count in result.all()}
