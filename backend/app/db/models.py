"""SQLAlchemy ORM models (the persistence layer).

These tables back the domain. Nested pipeline outputs (classification, routing,
draft) are stored as JSON on the Email row for the MVP; they can be normalized
into their own tables later without changing the Pydantic contracts.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.models.enums import EmailSource, EmailStatus


class Email(Base):
    """An incoming conference email and its full lifecycle state."""

    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    sender: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    sender_name: Mapped[str | None] = mapped_column(String(320), nullable=True)
    subject: Mapped[str] = mapped_column(String(998), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=EmailStatus.PENDING.value,
        index=True,
    )

    # Pipeline outputs, serialized from their Pydantic models.
    classification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    routing: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    draft: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Which submissions the email refers to and who it names —
    # {"submission_numbers": [str], "openreview_forum_ids": [str],
    #  "authors": [{"name", "email", "affiliation"}], "method": str}.
    # The identifier fields are LISTS: an email may name several submissions.
    # ``method`` records WHICH path produced the values (llm_distiller /
    # regex_fallback / none) so a weaker regex guess is never read as the
    # model's answer. NULL for every row processed before this column existed;
    # readers must treat NULL as "never looked", which is NOT the same as an
    # ``extraction`` whose lists are EMPTY ("looked, found none").
    extraction: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # A chair's judgment that this email is NOT an OpenReview reply after all —
    # a false positive from the text-based detection above. Set by
    # POST /emails/{id}/dismiss-openreview-candidate; makes the row fall through
    # the normal /queue predicate instead of /queue/openreview. Its routing lane
    # is untouched, so it simply resumes being an ordinary email.
    #
    # ⚠️ ITS OWN COLUMN, DELIBERATELY NOT A KEY INSIDE `extraction`, and this is
    # the whole reason the column exists. `extraction` records what the pipeline
    # OBSERVED in the text; every pipeline pass overwrites it wholesale with
    # `ExtractionResult.model_dump()` (orchestrator `_compute`), which recomputes
    # `openreview_reply_candidate` from the note id and notification sender. The
    # body does not change, so it recomputes to True. A dismissal written inside
    # that dict would therefore be silently undone by the next follow-up reply,
    # manual re-draft, or KB-change sweep — reappearing days later with nothing
    # in the audit trail to explain it.
    #
    # The two are different KINDS of fact with different lifetimes: an
    # observation should be recomputed whenever the text is reprocessed, and a
    # human decision must survive exactly that. Keeping the decision on its own
    # column means `_compute` can never reach it, with no carry-forward logic to
    # remember. Same separation-of-concerns rule as `_RESOLVED_ZENDESK_STATUSES`
    # vs `_SOLVED_BUCKET_STATUSES` in the email repository.
    openreview_candidate_dismissed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false(), index=True
    )

    # Re-evaluation (Phase G). ``retrieval_context`` captures the exact retriever
    # inputs at ingest — {"query": str, "intent": str, "retrieved_ids": [...]} —
    # so a KB-change sweep can re-run retrieval with no model call and compare the
    # grounding set. ``redrafting`` is the transient in-progress flag surfaced as
    # the "re-drafting…" badge; set when queued, cleared when the new draft lands.
    retrieval_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    redrafting: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false(), index=True
    )

    # Which chair a human-review email is assigned to (Phase 6A). Nullable:
    # FAQ-lane emails are never assigned, and the column stays empty until the
    # chair router runs. FK is nullable so deleting a chair does not cascade-
    # delete its emails (SET NULL semantics are enforced at the app layer).
    assigned_chair_id: Mapped[int | None] = mapped_column(
        ForeignKey("chairs.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # --- Zendesk origin fields (Piece 3) ----------------------------------
    # A Zendesk ticket maps 1:1 onto an Email row; these columns hold the
    # ticket-specific state. All are nullable so pre-Zendesk (toy/synthetic)
    # rows are unaffected. See ZENDESK_API.md §10 (dedup) / §2 (status).
    #
    # Origin discriminator: "zendesk" vs "toy_dataset". Server default backfills
    # existing rows as toy_dataset; Python default keeps ORM inserts in agreement.
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=EmailSource.TOY_DATASET.value,
        server_default=EmailSource.TOY_DATASET.value,
        index=True,
    )
    # Canonical Zendesk dedup key (§10). Unique so a ticket upserts to one row;
    # multiple NULLs are allowed (SQLite & Postgres) for non-Zendesk rows.
    zendesk_ticket_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, unique=True, index=True
    )
    # Zendesk user id of the requester (join to users); sender/sender_name hold
    # the email + display name.
    zendesk_requester_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Zendesk lifecycle status (new/open/pending/hold/solved/closed) — distinct
    # from our internal EmailStatus. Lets us treat "closed" as read-only.
    zendesk_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # The ticket's own timestamps in Zendesk (kept separate from created_at /
    # updated_at below, which are when WE created/updated this row).
    zendesk_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Compared on each poll to skip re-processing an unchanged ticket (§10).
    zendesk_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Highest end-user comment id already processed, so the poller only drafts on
    # a genuinely new author comment rather than every ticket update (§10).
    last_processed_comment_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )

    # Thread messages (Zendesk comments), ordered oldest-first. The initial
    # inquiry is the first public end-user message in this order (see §3/§5) —
    # derived by query, not stored, per the agreed design.
    thread_messages: Mapped[list["EmailThreadMessage"]] = relationship(
        "EmailThreadMessage",
        back_populates="email",
        order_by="EmailThreadMessage.created_at",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EmailThreadMessage(Base):
    """One message (Zendesk comment) in an email/ticket thread (Piece 3).

    A child of :class:`Email`. The ``public`` flag is load-bearing: ``True`` is a
    real reply visible to the requester, ``False`` an internal note (ZENDESK_API.md
    §3). ``author_role`` (end-user/agent/admin) tells the author's message apart
    from a chair's reply. Only the initial inquiry — the first ``public`` end-user
    message ordered by ``created_at`` — is (re)classified; the rest are context.
    """

    __tablename__ = "email_thread_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(
        ForeignKey("emails.id", ondelete="CASCADE"), nullable=False
    )
    # Zendesk comment id — globally unique (§3), so a comment upserts to one row.
    zendesk_comment_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, unique=True, index=True
    )
    # True = public reply to the requester; False = internal note.
    public: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # Zendesk user id + role of the comment author (§6).
    author_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    author_role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Sanitized plain text (safest for the classifier/drafter) + rendered HTML.
    plain_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    html_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The comment's own timestamp in Zendesk — the thread ordering key.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Channel the comment arrived through, e.g. "email" (§3 via.channel).
    via_channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # When WE stored the row (distinct from the comment's created_at above).
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    email: Mapped["Email"] = relationship("Email", back_populates="thread_messages")

    # Per-message pipeline results (Piece T1 / T1b). A follow-up message gets
    # its own classify -> retrieve -> route -> draft cycle stored in its own row
    # here, separately from the parent Email's result. A message may be
    # reprocessed (mirroring the parent Email's manual redraft), so this is
    # one-to-many: each reprocess appends a new row, keeping the full history.
    # Ordered chronologically (oldest-first, latest last) to match the parent
    # Email.thread_messages relationship and the audit trail. Starts empty until
    # a later piece populates it.
    processing_results: Mapped[list["EmailProcessingResult"]] = relationship(
        "EmailProcessingResult",
        back_populates="thread_message",
        order_by="EmailProcessingResult.created_at, EmailProcessingResult.id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        # Serves both parent-lookup and the oldest-first ordering scan used to
        # find the initial inquiry, in one index.
        Index(
            "ix_email_thread_messages_email_id_created_at", "email_id", "created_at"
        ),
    )


class EmailProcessingResult(Base):
    """A pipeline result for a single thread follow-up message (Piece T1 / T1b).

    The parent :class:`Email` row stores the pipeline outputs for the *initial*
    inquiry. When a requester posts a follow-up — a later public end-user
    :class:`EmailThreadMessage` — that message gets its own
    classify -> retrieve -> route -> draft cycle, and its result is stored HERE,
    separately, so the original Email result is never overwritten.

    Multiple rows may reference the same ``thread_message_id`` (T1b): a message
    can be reprocessed (mirroring the parent Email's manual redraft), and each
    reprocess appends a new row so the full history is retained — the latest is
    the newest by ``created_at`` / ``id``. This table starts empty; only new
    processing (wired in a later piece) ever populates it. The JSON columns
    deliberately mirror the shapes the parent Email stores (``classification`` /
    ``routing`` / ``draft`` / ``retrieval_context`` / ``extraction``) so the same Pydantic
    serialization can be reused; ``lane`` and ``confidence`` are denormalized
    scalars for cheap filtering/sorting.
    """

    __tablename__ = "email_processing_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Non-unique FK (T1b): a message may have several results over time. CASCADE
    # so results die with their message (and, transitively, the parent Email).
    thread_message_id: Mapped[int] = mapped_column(
        ForeignKey("email_thread_messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Pipeline outputs — same JSON shapes the parent Email row stores.
    classification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    routing: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    draft: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # {"query": str, "intent": str, "retrieved_ids": [...]} — matches Email.
    retrieval_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # {"submission_numbers": [str], "openreview_forum_ids": [str], "authors",
    # "method"} — matches Email. A follow-up turn can name different submissions
    # than the opening inquiry, so this is stored per-message rather than
    # inherited. Empty lists mean "examined, found none"; NULL means never
    # examined.
    extraction: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Denormalized scalars (also derivable from the JSON above): the routing lane
    # ("AUTO_REPLY" / "HUMAN_REVIEW") and the classifier confidence.
    lane: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    thread_message: Mapped["EmailThreadMessage"] = relationship(
        "EmailThreadMessage", back_populates="processing_results"
    )


class ZendeskSyncState(Base):
    """Checkpoint for the Zendesk incremental-export poller (Piece 4).

    One row per Zendesk account (keyed by ``subdomain``). Holds the resume
    ``cursor`` (the incremental export's ``after_cursor``; NULL → the next poll
    starts from ``start_time``) plus light bookkeeping. Persisting the cursor in
    the DB (not a local file like the one-off pull script) is what lets the poll
    survive restarts and, later, be made multi-instance-safe by row-locking this
    single well-known row (``SELECT ... FOR UPDATE``). See ZENDESK_API.md §2/§10.
    """

    __tablename__ = "zendesk_sync_state"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    subdomain: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    # Incremental export resume point; NULL means "first run — use start_time".
    cursor: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Unix epoch for the initial call before any cursor exists.
    start_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # When the last successful cycle completed, and the last error (if any).
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Cumulative count of tickets seen across cycles (bookkeeping only).
    tickets_seen: Mapped[int] = mapped_column(
        nullable=False, default=0, server_default="0"
    )

    # --- Overlap guard (single-flight lock) -------------------------------
    # A cycle sets ``is_running`` True (stamping ``running_since``) while it
    # holds the row; a second trigger sees the flag and skips instead of racing.
    # ``running_since`` also powers a staleness takeover: if a claimed run went
    # quiet longer than the stale window (a crash mid-cycle), a new run may
    # reclaim the lock rather than staying blocked forever.
    is_running: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    running_since: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Chair(Base):
    """A conference chair who can be assigned human-review emails (Phase 6A).

    ``areas`` is a JSON list of intent/topic strings the chair owns; the chair
    router matches a classified email's intent against these. A chair with an
    empty ``areas`` list acts as the catch-all fallback (the General Chair),
    receiving anything no other active chair claims. ``active`` gates a chair
    out of routing without deleting the row (and its assignment history).
    """

    __tablename__ = "chairs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role_title: Mapped[str] = mapped_column(String(255), nullable=False)
    # List of intent/topic strings this chair handles (empty = fallback).
    areas: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class AuditLog(Base):
    """An append-only record of actions taken on an email."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(
        ForeignKey("emails.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Free-form structured context. Column is named "metadata"; the Python
    # attribute is renamed because `metadata` is reserved on the declarative base.
    extra_metadata: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True
    )


class PolicyDocument(Base):
    """A FAQ / policy knowledge-base entry used for grounding replies."""

    __tablename__ = "policy_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    policy_key: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # [tags-dropped E007] The ``tags`` column is dropped by migration
    # e7a9c1f2b3d4. E007 (docs/exp_tracking/E007_policy_tag_ablation.md) showed the
    # auto-generated tags carry no retrieval signal; the whole tag path is commented
    # out (not deleted) so B4b (a controlled facet vocab) can restore it — uncomment
    # this column + re-add it in an up-migration.
    # tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Intents this chunk can answer (Task B, controlled vocab from
    # taxonomy.VALID_INTENTS).
    intents: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Phase F (layered KB): the layer/trust axis and the soft on/off lifecycle.
    # visibility: "public" (official corpus, freely citable) | "internal"
    # (chair-authored, not on the public site — retrievable & citable, marked for
    # provenance). status: "active" (indexed) | "inactive" (retired, not indexed).
    # Both DB + Python defaults so raw migrations and ORM inserts agree.
    visibility: Mapped[str] = mapped_column(
        String(16), nullable=False, default="public", server_default="public", index=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", server_default="active", index=True
    )

    # Versioning / lineage (chair edit-a-copy). An edit inserts a new active row
    # that ``supersedes`` the retired base; ``superseded_by`` is the reverse
    # pointer (NULL ⇒ this row is the live tip). ``root_key`` groups a lineage
    # (NULL ⇒ the row is its own root — true for every original/imported policy;
    # it cannot be a column default because it derives from ``policy_key``).
    # ``version`` is the 1-based depth for display.
    supersedes: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    superseded_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    root_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )

    # Conflict detection (2e): the last LLM conflict report computed when this
    # policy was created / edited / reactivated / re-checked. Shape mirrors
    # app.pipeline.policy_conflict.ConflictReport. NULL ⇒ never checked (e.g.
    # legacy rows, or checked while no LLM was configured and skipped).
    conflict_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PolicyAuditLog(Base):
    """Append-only record of KB governance actions (create / edit / retire)."""

    __tablename__ = "policy_audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    policy_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    before: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PolicySuggestion(Base):
    """A chair-gated candidate internal policy learned from a [CHAIR:]-gap edit (CEL)."""
    __tablename__ = "policy_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_email_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(16), index=True, nullable=False, default="pending")
    experience_summary: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    intents: Mapped[list | None] = mapped_column(JSON, nullable=True)
    generalizable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    conflict_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    reviewed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    resulting_policy_key: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SuggestionAuditLog(Base):
    """Append-only record of chair review actions on a CEL policy suggestion.

    ``PolicySuggestion.reviewed_by`` / ``reviewed_reason`` hold CURRENT state and
    are overwritten on every review action; this table is the history they lose.
    One row per accept/reject, never updated.

    ``suggestion_id`` carries NO ForeignKey, matching ``policy_audit_logs``
    (``policy_key`` is a plain string) and ``policy_suggestions.source_email_id``.
    An FK would be enforced only on Postgres: SQLite — the default for local dev,
    tests and CI — does not enforce foreign keys unless ``PRAGMA foreign_keys=ON``
    is set, and nothing here sets it. Integrity is upheld by the endpoints, which
    404 before writing. Deliberately NOT ``audit_logs``: that table is email-
    scoped with an ON DELETE CASCADE, so a deleted email would take the
    governance trail with it, and every write there publishes an SSE event.
    """

    __tablename__ = "suggestion_audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    suggestion_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    # Client-supplied and UNVERIFIED — there is no auth; today every caller
    # sends the placeholder "Chair1". Recorded as given, implying no identity.
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    resulting_policy_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PaperAssignment(Base):
    """Which Associate Program Chair (APC) owns a submission, and its forum link.

    Loaded from the program committee's assignment sheet by
    ``scripts/load_paper_assignments.py``; read by the phase-1 appeal path to
    fill ``apc_name`` / ``openreview_url`` on each appeal row. Keyed by the
    submission number as a string (the same form ``extraction`` stores).
    ``openreview_forum_id`` is the ``?id=`` of ``openreview_url``, kept as its
    own indexed column so an email that names only a forum link can still be
    mapped back to a submission number. Nullable because a URL may carry no id.
    """

    __tablename__ = "paper_assignments"

    paper_number: Mapped[str] = mapped_column(String(16), primary_key=True)
    apc_name: Mapped[str] = mapped_column(String(255), nullable=False)
    openreview_url: Mapped[str] = mapped_column(String(512), nullable=False)
    openreview_forum_id: Mapped[str | None] = mapped_column(
        String(32), nullable=True, index=True
    )
    cycle: Mapped[str] = mapped_column(
        String(32), nullable=False, default="AAAI-27", server_default="AAAI-27"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Phase1Appeal(Base):
    """One appealed paper in one phase-1 rejection appeal email.

    Written by the pipeline from the phase-1 appeal classifier's result: one row
    per paper the author disputes, or a single row with ``submission_number``
    NULL when no paper could be identified. The set of rows for an email is
    replaced wholesale on every reprocess (``Phase1AppealRepository.
    replace_for_email``), so rows never accumulate across runs. ``apc_name`` /
    ``openreview_url`` are denormalized from ``paper_assignments`` at write time
    (NULL when the paper is not in the sheet) so the CSV export is a plain read.
    ``reasons`` is a JSON list of ``{"reason", "quote"}``. ``prompt_sha256`` and
    ``model`` record which prompt and model produced the row.

    Rows die with their email (ON DELETE CASCADE, enforced on Postgres and on
    SQLite when ``PRAGMA foreign_keys=ON``).
    """

    __tablename__ = "phase1_appeals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(
        ForeignKey("emails.id", ondelete="CASCADE"), nullable=False, index=True
    )
    zendesk_ticket_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, index=True
    )
    submission_number: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )
    apc_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    openreview_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    relation: Mapped[str] = mapped_column(String(16), nullable=False)
    reasons: Mapped[list] = mapped_column(JSON, nullable=False)
    must_verify: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prompt_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
