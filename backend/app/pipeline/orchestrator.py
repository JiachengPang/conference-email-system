"""Pipeline orchestrator (classify → retrieve → draft → route → persist).

Wires the four pipeline modules together and the persistence layer behind a
single entry point, `process_email`. It owns sequencing, timing, persistence,
and audit logging — the modules themselves stay unaware of each other and of
the database.

The router runs AFTER the drafter: the FAQ-vs-human_review lane is a property
of the generated draft's completeness/groundedness/self-rated confidence, not
of the classification alone (see `app.pipeline.router`).

Failure policy:
- A failure in classify / retrieve is fatal: it is logged and re-raised with
  status "error" (no partial record is persisted).
- A failed draft is non-fatal: the partial result is persisted with status
  "draft_failed" and returned (the drafter itself never raises). Routing then
  runs on that (possibly failed) draft.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.chair_router import ChairAssignment, ChairInfo, get_chair_router
from app.core.config import settings
from app.core.tracing import PipelineTracer
from app.models.enums import EmailStatus
from app.pipeline.classifier import ClassificationResult, IntentClassifier
from app.pipeline.distiller import EmailDistiller
from app.pipeline.drafter import DraftResponse, ResponseDrafter
from app.pipeline.extractor import EmailExtractor
from app.pipeline.reciprocal_detector import detect_reciprocal_dispute
from app.pipeline.appeal_reason_classifier import classify_appeal_reason
from app.pipeline.appeal_reasons import is_valid_stored
from app.pipeline.phase1_appeal_classifier import classify_phase1_appeal
from app.pipeline import phase1_appeal_outcome
from app.pipeline.taxonomy import REJECT_APPEAL_INTENTS
from app.pipeline.appeal_reply_hook import prepare_appeal_draft
from app.pipeline.phase1_reply_mapping import map_phase1
from app.pipeline.retriever import (
    RetrievedChunk,
    get_retriever,
    grounded_chunks_hash,
)
from app.pipeline.router import (
    LANE_HUMAN_REVIEW,
    EmailRouter,
    RoutingDecision,
    apply_self_sufficiency_floor,
)
from app.pipeline.thread_transcript import build_transcript
from app.repositories.audit_repository import AuditRepository
from app.repositories.chair_repository import ChairRepository
from app.repositories.email_repository import EmailRepository
from app.repositories.policy_repository import PolicyRepository

logger = logging.getLogger(__name__)

# Map a terminal pipeline status to the email's persisted lifecycle status.
# Consumed by direct index (_LIFECYCLE_STATUS[status]) — every internal status
# the draft step can produce MUST have an entry here or the pipeline KeyErrors.
_LIFECYCLE_STATUS = {
    "complete": EmailStatus.DRAFT_GENERATED.value,
    "draft_failed": EmailStatus.ROUTED.value,
    # The drafter succeeded but was cut off before writing any visible reply —
    # no draft exists to review, and it is retryable. See EmailStatus docstring.
    "draft_truncated": EmailStatus.DRAFT_TRUNCATED.value,
}

# How many leading characters of the body to use as the retrieval query
# (legacy "prefix" strategy).
# The ONLY intent that triggers the reciprocal-dispute call. Deliberately the
# narrow equality rather than `taxonomy.is_reject_appeal`, which would also
# cover `review_decision_appeal`: measured on the 50-ticket Phase 1 sample,
# widening reaches the SAME 27 of 28 r tickets (96%) while asking 2 more
# non-r ones — zero recall for extra spend (reject_appeal.md D40).
_RECIPROCAL_GATE_INTENT = "desk_reject_appeal"


def _prior_dispute_flag(email) -> bool | None:
    """The stored ``is_reciprocal_dispute``, or None when there isn't a real one.

    Reads defensively: the column is nullable JSON, legacy rows predate the
    field entirely, and only a genuine ``True``/``False`` counts as a prior
    answer — anything else (missing, null, a stringy legacy value) is "not
    answered", which under the tri-state is NOT the same as ``False`` (D7).
    """
    extraction = getattr(email, "extraction", None) or {}
    value = extraction.get("is_reciprocal_dispute")
    return value if isinstance(value, bool) else None


# The intents that trigger the appeal-reason call (reject_appeal.md D58): BOTH
# appeal intents, unlike the reciprocal detector's narrow gate, because
# post-review complaints land on `review_decision_appeal`. One constant, shared
# with `taxonomy.is_reject_appeal`, so the gate and the taxonomy cannot drift.
_APPEAL_REASON_GATE_INTENTS = REJECT_APPEAL_INTENTS


def _prior_appeal_reason(email):
    """The RAW stored ``appeal_reason`` (D66), or None when the row has none.

    Deliberately not validated here: `_compute` validates ``prior_appeal_reason``
    once, for every caller — including the public ``compute`` seam, whose caller
    fills the key itself. A second check here would make each of the two
    redundant, so deleting either one could never fail a test.
    """
    extraction = getattr(email, "extraction", None) or {}
    return extraction.get("appeal_reason")

_RETRIEVAL_QUERY_CHARS = 300
# Fallback query when QUERY_STRATEGY == "distill" but distillation failed:
# subject + this much body (E003 arm B — best non-distilled formulation).
_FALLBACK_QUERY_BODY_CHARS = 600


class PipelineResult(BaseModel):
    """End-to-end result of processing one email through the pipeline."""

    email_id: str
    classification: ClassificationResult
    retrieved_chunks: list[RetrievedChunk]
    routing: RoutingDecision
    draft: DraftResponse
    processing_time_ms: float
    status: str  # "complete" | "draft_failed" | "error"


@dataclass
class _Computed:
    """Everything one pipeline run produces, before it is persisted.

    Lets ``process_email`` (create a new row) and ``reprocess_email`` (update an
    existing row in place) share the exact same classify→retrieve→draft→route
    compute and audit/trace finalization.
    """

    classification: ClassificationResult
    retrieved_chunks: list
    routing: RoutingDecision
    draft: DraftResponse
    chair_assignment: ChairAssignment | None
    status: str
    record: dict
    tracer: PipelineTracer
    start: float
    # The phase-1 appeal outcome, persisted only by the callers that write the
    # email row (never by the public ``compute`` seam). None when the flag is
    # off, which means: leave the rows alone.
    phase1: phase1_appeal_outcome.Phase1Outcome | None = None


def _with_appeal_reply(draft: dict, appeal_reply: dict | None) -> dict:
    """The stored draft dict, with ``appeal_reply`` added ONLY when the appeal
    reply hook ran. Absent — not null — otherwise, so a flag-off or non-appeal
    record is byte-for-byte what it always was."""
    if appeal_reply is not None:
        draft["appeal_reply"] = appeal_reply
    return draft


def _append_draft_history(
    record: dict, old_draft: dict | None, triggering_comment_ids: list | None
) -> None:
    """Push the outgoing draft into ``record['draft']['history']`` before it is
    overwritten, carrying any prior history forward (D4). No-op when there was
    no prior draft text. Prior history entries are copied without their own
    nested history to avoid unbounded growth.
    """
    new_draft = record.get("draft") or {}
    prior = (old_draft or {}).get("history") or []
    history = list(prior)
    if old_draft and old_draft.get("draft_text"):
        history.append(
            {
                "draft_text": old_draft.get("draft_text"),
                "notes_for_chair": old_draft.get("notes_for_chair"),
                "citations": old_draft.get("citations", []),
                "answer_confidence": old_draft.get("answer_confidence"),
                "is_edited": bool(old_draft.get("is_edited")),
                "superseded_at": datetime.now(timezone.utc).isoformat(),
                "reason": "followup",
                "triggering_comment_ids": triggering_comment_ids or [],
            }
        )
    new_draft["history"] = history
    record["draft"] = new_draft


def _parse_received_at(value) -> datetime | None:
    """Coerce an inbound email's ``timestamp`` into a ``received_at`` datetime.

    ``received_at`` is meant to hold when the REQUESTER wrote in, but the column
    only carries a ``server_default``, so without this it silently records when
    our row was inserted. Callers that know the real instant pass it as
    ``email_data["timestamp"]`` (a Zendesk ticket's ``created_at``, or an
    ``/ingest`` payload's timestamp).

    Returns ``None`` — meaning "leave ``received_at`` alone" — for anything we
    cannot read as a real instant: a missing key, an empty string (the
    ``/ingest`` default and the adapter's ``or ""`` fallback), or an unparseable
    one. Never raises: a malformed timestamp must degrade to the insert-time
    default, not fail the pipeline run.

    Zendesk sends ISO-8601 with a ``Z`` suffix, so the suffix is normalized
    before parsing. A naive value is assumed UTC, then the result is CONVERTED to
    UTC — see the note on the return below, this is a correctness requirement and
    not just tidiness.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            logger.warning(
                "Unparseable email timestamp %r; falling back to insert time.", value
            )
            return None
    else:
        return None
    # Attach UTC to a naive value FIRST: ``astimezone`` on a naive datetime
    # assumes system-local time, which would make the stored instant depend on
    # the server's timezone.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    # Then convert to UTC, because storage is NOT offset-preserving on every
    # dialect: SQLite's DATETIME writes the wall-clock fields and silently
    # DISCARDS tzinfo, so "2026-02-09T14:45:00-05:00" would persist as 14:45
    # rather than 19:45 — wrong by five hours, with no error, and only on SQLite
    # (asyncpg converts correctly for timestamptz). Normalizing here makes the
    # stored instant identical on both. A no-op for the UTC/``Z`` timestamps
    # Zendesk sends, so it changes nothing on the poller path.
    return parsed.astimezone(timezone.utc)


async def resolve_lineage_roots(
    db: AsyncSession,
    keys,
    policy_repo: PolicyRepository | None = None,
) -> dict[str, str]:
    """Map each policy key to its LINEAGE ROOT (``key -> root``).

    Exclusions are matched on the root, not the literal ``policy_key``, because
    ``edit_policy`` mints a brand-new key for an edited policy
    (``{root}__v{n+1}``) and retires the old row. Keyed on the exact string, an
    exclusion would go stale the moment someone edits that policy — and the
    KB-edit sweep is precisely when that happens, so the chunk the chair removed
    would quietly reappear. Rooting the comparison makes "I removed this policy"
    survive its next revision.

    A key with no row maps to ITSELF, so an unknown/hard-deleted id still
    matches itself and never silently widens the exclusion. Best-effort: a
    lookup failure logs and yields an empty map, which degrades to "nothing
    excluded" rather than failing the re-draft.
    """
    repo = policy_repo or PolicyRepository()
    wanted = [k for k in dict.fromkeys(keys) if k]  # de-dup, drop falsy, keep order
    if not wanted:
        return {}
    try:
        rows = await repo.get_by_keys(db, wanted)
    except Exception:  # noqa: BLE001 - never fail a re-draft over exclusion metadata
        logger.exception("Lineage-root lookup failed for %s; treating as no-op.", wanted)
        return {}
    return {
        key: (repo._root_of(rows[key]) if key in rows else key) for key in wanted
    }


async def resolve_forced_chunk(
    db: AsyncSession,
    policy_key: str,
    already: list[RetrievedChunk],
    policy_repo: PolicyRepository | None = None,
) -> RetrievedChunk | None:
    """Resolve a chair-forced policy key into an extra grounding chunk.

    Module-level so the orchestrator (first draft / manual redraft) and the
    re-evaluation sweep share ONE implementation of the forcing rule — the
    active-status check below must never drift between those two paths.

    Returns None (never raises) when the key is unknown, is not ACTIVE, or is
    already in the retrieved set — the caller then proceeds with normal
    retrieval only, so a bad key degrades the draft's grounding rather than
    failing the whole redraft.

    ACTIVE is required on purpose. This creates NEW grounding for a reply that
    may be sent to a requester, so a retired or superseded policy must not be
    forced in — the chairs withdrew that text. (``status != "active"`` also
    covers superseded rows: ``edit_policy`` marks the old version inactive.)
    This is the OPPOSITE of the read-time hydration in the emails API, which
    deliberately ignores status because it reports the HISTORICAL grounding of
    an already-generated draft rather than creating new grounding.

    Visibility is deliberately NOT filtered: internal policies are part of the
    retrieval corpus (``list_for_index`` indexes public + internal), and
    forcing an internal chair-authored policy is the primary use case.
    """
    repo = policy_repo or PolicyRepository()
    if any(c.policy_id == policy_key for c in already):
        logger.info("Forced policy %s already retrieved; not duplicating.", policy_key)
        return None
    try:
        rows = await repo.get_by_keys(db, [policy_key])
    except Exception:  # noqa: BLE001 - a lookup failure must not kill the redraft
        logger.exception("Forced-policy lookup failed for %s.", policy_key)
        return None

    row = rows.get(policy_key)
    if row is None:
        logger.warning("Forced policy %s not found; ignoring.", policy_key)
        return None
    if row.status != "active":
        logger.warning(
            "Forced policy %s is %s (not active); ignoring.", policy_key, row.status
        )
        return None

    return RetrievedChunk(
        policy_id=row.policy_key,
        title=row.title or "",
        content=row.content or "",
        # Not a retrieval score — this chunk was never ranked. 0.0 marks it as
        # chair-forced rather than implying "least relevant".
        score=0.0,
        category=row.category or "",
        intents=list(row.intents) if getattr(row, "intents", None) else [],
    )


class EmailPipeline:
    """Orchestrates the full classify → retrieve → draft → route → save flow."""

    def __init__(self) -> None:
        # Each module is instantiated from its swappable settings flag.
        self.classifier = IntentClassifier(strategy=settings.CLASSIFIER_BACKEND)
        self.retriever = get_retriever()
        self.router = EmailRouter(strategy=settings.ROUTING_STRATEGY)
        # The SECOND routing decision (Phase 6A): which chair a human_review
        # email goes to. Independent of the lane router above and swappable via
        # its own flag.
        self.chair_router = get_chair_router(settings.CHAIR_ROUTING_STRATEGY)
        # One model call producing retrieval queries + intent (E003). Only
        # consulted when QUERY_STRATEGY == "distill"; always best-effort.
        self.distiller = EmailDistiller()
        # Which submission the email is about, and who it names. Pure CPU —
        # reuses the distiller's output when there is one and falls back to
        # regex over subject/body, so it adds NO model call and no I/O.
        self.extractor = EmailExtractor()
        self.drafter = ResponseDrafter(provider=settings.MODEL_PROVIDER)
        self.email_repo = EmailRepository()
        self.chair_repo = ChairRepository()
        self.audit_repo = AuditRepository()
        # Only used to resolve a chair-forced policy key (manual invoke); the
        # normal retrieval path never touches it.
        self.policy_repo = PolicyRepository()

    async def _force_policy_chunk(
        self, db: AsyncSession, policy_key: str, already: list[RetrievedChunk]
    ) -> RetrievedChunk | None:
        """Resolve a chair-forced policy key (see :func:`resolve_forced_chunk`)."""
        return await resolve_forced_chunk(db, policy_key, already, self.policy_repo)

    async def _assign_chair(
        self,
        db: AsyncSession,
        classification: ClassificationResult,
        routing: RoutingDecision,
    ) -> ChairAssignment | None:
        """Pick a chair for a human-review email. Best-effort — never raises.

        Runs ONLY when the lane router chose human_review (FAQ-lane emails are
        auto-replied and never assigned to a chair). Fetches the active chair
        roster via the repository, projects it onto DB-free ``ChairInfo`` objects,
        and delegates the decision to the swappable strategy. A failure here (no
        chairs, DB hiccup) leaves the email unassigned rather than breaking the
        pipeline — chair assignment is additive to the core classify→retrieve→
        draft→route flow.
        """
        if routing.lane != LANE_HUMAN_REVIEW:
            return None
        try:
            chairs = await self.chair_repo.get_active_chairs(db)
            if not chairs:
                return None
            candidates = [
                ChairInfo(
                    id=c.id,
                    name=c.name,
                    role_title=c.role_title,
                    areas=list(c.areas or []),
                    active=c.active,
                )
                for c in chairs
            ]
            return self.chair_router.assign(classification, candidates)
        except Exception:  # noqa: BLE001 - assignment must not break the pipeline
            logger.warning("Chair assignment failed; leaving email unassigned.", exc_info=True)
            return None

    async def _compute(
        self,
        email_data: dict,
        db: AsyncSession,
        *,
        forced_policy_key: str | None = None,
        excluded_policy_ids: list[str] | None = None,
    ) -> _Computed:
        """Run classify → retrieve → draft → route → chair-assign (no persistence).

        Shared by ``process_email`` (creates a new row) and ``reprocess_email``
        (updates an existing row). Builds the persistence ``record`` but does not
        write it; the caller decides create vs. update.

        ``forced_policy_key`` is a chair-specified policy to ground on in ADDITION
        to whatever retrieval ranked (manual invoke). It occupies a guaranteed
        EXTRA slot — ``MAX_RETRIEVED_CHUNKS`` still governs the ranked chunks, so
        the forced policy can never be evicted by re-ranking. Default ``None``
        leaves every existing caller bit-for-bit unchanged.

        ``excluded_policy_ids`` are chunks the chair removed from the grounding.
        They are dropped from the RANKED set before the forced slot is applied,
        matched on LINEAGE ROOT so a removal survives the policy being edited into
        a new key. If the forced key is itself excluded, exclusion wins and no
        forced chunk is appended. Excluded chunks never reach the drafter, so
        there is no prompt representation of a removal.

        The phase-1 appeal outcome is decided here and carried in the result;
        only the callers that write the email row persist it.
        """
        start = time.perf_counter()
        subject = email_data.get("subject", "")
        body = email_data.get("body", "")
        transcript_text = email_data.get("thread_transcript")
        # Hoisted so the extractor and the persistence record cannot drift apart.
        # Kept UNDEFAULTED here: the record's "unknown@unknown" is a storage
        # placeholder for a NOT NULL column, and feeding it to identity
        # extraction would fabricate an author out of a sentinel.
        sender = email_data.get("from") or email_data.get("sender") or ""
        sender_name = email_data.get("sender_name")

        # Per-email tracer: buffers a record per stage and flushes them once the
        # email is persisted (its id is known then). Additive only.
        tracer = PipelineTracer()

        # --- classify → retrieve (fatal on failure) ------------------------
        try:
            with tracer.stage(
                "classifier",
                {"subject_length": len(subject), "body_length": len(body)},
            ) as st:
                # E003: one model call distills retrieval queries AND classifies
                # intent. Best-effort — on any distiller failure the keyword/
                # trainable backend classifies as before.
                distilled = None
                if settings.QUERY_STRATEGY == "distill":
                    distilled = await self.distiller.distill(
                        subject, body, transcript=transcript_text
                    )
                if distilled is not None and distilled.intent is not None:
                    classification = ClassificationResult(
                        intent=distilled.intent,
                        confidence=(
                            distilled.confidence
                            if distilled.confidence is not None
                            else 0.5
                        ),
                        reasoning="Distiller call (query distillation + intent).",
                        method="llm_distiller",
                    )
                else:
                    classification = await self.classifier.classify(body, subject)
                st.output_summary = {
                    "intent": classification.intent,
                    "confidence": round(float(classification.confidence), 4),
                    "method": classification.method,
                }
                # Surface both confidences in the trace when calibration is active.
                if classification.calibrated_confidence is not None:
                    st.output_summary["raw_confidence"] = round(
                        float(classification.raw_confidence), 4
                    )
                    st.output_summary["calibrated_confidence"] = round(
                        float(classification.calibrated_confidence), 4
                    )

            # Which submission this email is about, and who it names. Reuses the
            # distiller output already in hand — no second model call — and falls
            # back to regex over subject/body when distillation did not run.
            #
            # Deliberately OUTSIDE the classifier tracer stage: folding it in
            # would inflate that stage's measured duration, and the trace
            # contract is exactly classify→retrieve→draft→route. This follows the
            # chair-assignment precedent, which is kept out of the tracer for the
            # same reason.
            #
            # `extraction` has nowhere to go yet — persistence is a later piece.
            # It is computed here because this is the only point where `distilled`
            # and the raw sender fields are both in scope, and because every
            # entry point (process_email / reprocess_email /
            # reprocess_email_with_thread / the public `compute` seam used by
            # per-follow-up processing) funnels through this method, so wiring it
            # once here covers them all without duplicating the call.
            extraction = self.extractor.extract(
                subject, body, sender, sender_name, distilled
            )
            logger.debug(
                "Extraction (%s): submissions=%d forum_ids=%d authors=%d",
                extraction.method,
                len(extraction.submission_numbers),
                len(extraction.openreview_forum_ids),
                len(extraction.authors),
            )

            # --- reciprocal-review dispute flag ---------------------------
            # A SEPARATE, conditional model call. It lives here rather than in
            # `extract()` because `extract` is synchronous, pure and
            # never-raises, and a network call would end all three.
            #
            # PRESERVE RULE (reject_appeal.md D42): only a real YES/NO may
            # overwrite a stored value. If the detector does not run, or runs
            # and cannot answer, whatever was already on the row is carried
            # forward. Without this a follow-up whose intent drifts off the
            # appeal family ("any update?") would silently wipe a True — the
            # flag is a property of the WHOLE conversation (D9) while intent
            # deliberately follows the latest turn, so the two must not share
            # a lifetime. A prior False is preserved for the same reason a
            # True is: under the tri-state (D7) "ruled out" and "never asked"
            # are different claims, and letting False decay to None would
            # quietly downgrade one into the other.
            prior_dispute = email_data.get("prior_is_reciprocal_dispute")
            # `isinstance(..., bool)` also rejects a stringy legacy value; only
            # a genuine True/False counts as a prior answer.
            prior_dispute = prior_dispute if isinstance(prior_dispute, bool) else None
            dispute_flag = prior_dispute
            if not settings.RECIPROCAL_DETECTOR_ENABLED:
                detector_status = "flag_off"
            elif classification.intent != _RECIPROCAL_GATE_INTENT:
                detector_status = "gate_not_met"
            else:
                verdict = await detect_reciprocal_dispute(
                    subject=subject, body=body, transcript=transcript_text
                )
                if verdict is None:
                    detector_status = "ran_no_answer"
                else:
                    detector_status = "ran"
                    dispute_flag = verdict
            # Ids and states only — never subject, body or transcript.
            logger.debug(
                "Reciprocal detector: %s (intent=%s, prior=%s, final=%s)",
                detector_status,
                classification.intent,
                prior_dispute,
                dispute_flag,
            )
            extraction = extraction.model_copy(
                update={"is_reciprocal_dispute": dispute_flag}
            )

            # --- appeal reasons (reject-appeal Phase 2) -------------------
            # A SEPARATE, conditional model call, after the detector because it
            # needs the detector's EFFECTIVE flag. Off by default (D65).
            #
            # PRESERVE RULE (D66): only a real answer — any valid list,
            # INCLUDING [] — overwrites. `classify_appeal_reason` applies it
            # itself, so it must be handed the STORED value as the prior; in
            # every branch that does not call it, the stored value is carried
            # forward unchanged (never None, never []).
            #
            # SKIP (D68): when the EFFECTIVE `is_reciprocal_dispute` is True —
            # `dispute_flag` above, i.e. after D42's preserve rule, so a stored
            # True skips even on a run where the detector did not fire. The
            # reciprocal ground is owned by that flag (D60).
            #
            # Only a canonical registry list (including []) counts as a prior;
            # anything else in storage (a letter code, a reordered list) is not
            # ours and reads as "no prior". The ONE validation point (D66).
            prior_reason_raw = email_data.get("prior_appeal_reason")
            prior_reason = (
                list(prior_reason_raw) if is_valid_stored(prior_reason_raw) else None
            )
            appeal_reason = prior_reason
            if not settings.APPEAL_REASON_CLASSIFIER_ENABLED:
                reason_status = "flag_off"
            elif classification.intent not in _APPEAL_REASON_GATE_INTENTS:
                reason_status = "gate_not_met"
            elif dispute_flag is True:
                reason_status = "skipped_reciprocal"
            else:
                appeal_reason = await classify_appeal_reason(email_data, prior_reason)
                reason_status = "ran"
            # Ids and states only — never subject, body or transcript.
            logger.debug(
                "Appeal-reason classifier: %s (intent=%s, prior=%s, final=%s)",
                reason_status,
                classification.intent,
                prior_reason,
                appeal_reason,
            )
            extraction = extraction.model_copy(update={"appeal_reason": appeal_reason})

            # --- phase-1 rejection appeal ---------------------------------
            # A SEPARATE, conditional model call; its outcome is only carried
            # here and persisted by the callers that write the email row (see
            # app.pipeline.phase1_appeal_outcome for the rules). Off by default.
            #
            # Ticket creation time for the date gate. The update paths put the
            # stored row's value in email_data["ticket_created_at"]
            # (zendesk_created_at, else received_at). The ingest paths have no
            # row yet, so it is email_data["timestamp"]: the Zendesk ticket's
            # created_at from the adapter, or the /ingest payload's timestamp.
            # Unknown counts as meeting the gate.
            ticket_created_at = email_data.get("ticket_created_at")
            if ticket_created_at is None:
                ticket_created_at = _parse_received_at(email_data.get("timestamp"))
            phase1_outcome = None
            if not settings.PHASE1_APPEAL_ENABLED:
                phase1_status = "flag_off"
            elif not phase1_appeal_outcome.gate_met(
                classification.intent, ticket_created_at
            ):
                phase1_outcome = phase1_appeal_outcome.Phase1Outcome(
                    phase1_appeal_outcome.GATE_NOT_MET
                )
                phase1_status = phase1_outcome.state
            else:
                phase1_outcome = phase1_appeal_outcome.outcome_for(
                    await classify_phase1_appeal(email_data)
                )
                phase1_status = phase1_outcome.state
            # Ids and states only — never subject, body or quotes.
            logger.debug(
                "Phase-1 appeal classifier: %s (intent=%s)",
                phase1_status,
                classification.intent,
            )

            # Query formulation (E003): distilled queries joined into one string,
            # with NO intent token (it hurts dense retrieval — E001). Distill-mode
            # fallback is subject+body[:600]; the legacy prefix strategy keeps
            # body[:300] + intent token, bit-for-bit.
            if distilled is not None and distilled.queries:
                query = " ".join(distilled.queries)
                retrieval_intent = ""
            elif settings.QUERY_STRATEGY == "distill":
                # Anchor the fallback query to the latest requester turn when a
                # thread is present, so retrieval follows the current ask.
                anchor = email_data.get("latest_requester_message") or body
                query = f"{subject} {anchor[:_FALLBACK_QUERY_BODY_CHARS]}".strip()
                retrieval_intent = ""
            else:
                query = body[:_RETRIEVAL_QUERY_CHARS]
                retrieval_intent = classification.intent

            # Soft intent prior (B5): the classified intent feeds a fusion-only
            # score boost as METADATA on a SEPARATE channel from ``retrieval_intent``.
            # This is why it does NOT reintroduce E001 — ``query`` and
            # ``retrieval_intent`` (the args that shape the query text, "" in distill
            # mode) are unchanged; ``prior_intent`` never enters the query string.
            # Gated by INTENT_PRIOR_ENABLED (default False, B7): E010 showed this
            # boost badly regresses fusion retrieval (hit@1 .730→.243) as currently
            # sized, so production withholds the intent unless explicitly opted in
            # (docs/exp_tracking/E010_intent_prior.md).
            prior_intent = (
                (classification.intent or "") if settings.INTENT_PRIOR_ENABLED else ""
            )

            with tracer.stage("retriever", {"query": query}) as st:
                retrieved_chunks = await self.retriever.retrieve(
                    query,
                    retrieval_intent,
                    top_k=settings.MAX_RETRIEVED_CHUNKS,
                    prior_intent=prior_intent,
                )
                # Chair removals: drop excluded chunks from the RANKED set before
                # the forced slot is considered. Matched on lineage root, so an
                # exclusion survives the excluded policy being edited into a new
                # key (see resolve_lineage_roots). Entirely skipped — not one
                # extra query — when the chair excluded nothing, so the default
                # path stays byte-identical.
                excluded_roots: set[str] = set()
                roots: dict[str, str] = {}
                if excluded_policy_ids:
                    roots = await resolve_lineage_roots(
                        db,
                        [c.policy_id for c in retrieved_chunks]
                        + list(excluded_policy_ids)
                        + ([forced_policy_key] if forced_policy_key else []),
                        self.policy_repo,
                    )
                    excluded_roots = {
                        roots.get(x, x) for x in excluded_policy_ids
                    }
                    retrieved_chunks = [
                        c
                        for c in retrieved_chunks
                        if roots.get(c.policy_id, c.policy_id) not in excluded_roots
                    ]

                # Exclusion beats forcing. Without this the append would silently
                # undo the removal: the chair's own pick, filtered out above,
                # would be resolved and put straight back — a contradictory
                # request resolved by accident of ordering rather than by rule.
                forced_excluded = bool(forced_policy_key) and (
                    roots.get(forced_policy_key, forced_policy_key) in excluded_roots
                )
                if forced_excluded:
                    logger.info(
                        "Forced policy %s is also excluded; exclusion wins.",
                        forced_policy_key,
                    )

                # Chair-forced policy (manual invoke): appended as a GUARANTEED
                # EXTRA slot, after ranking, so top_k still governs the ranked
                # chunks and the forced one can never be crowded out. No-op (and
                # never raises) when the key is absent, unknown, inactive, or
                # already retrieved.
                forced_chunk = (
                    await self._force_policy_chunk(db, forced_policy_key, retrieved_chunks)
                    if forced_policy_key and not forced_excluded
                    else None
                )
                if forced_chunk is not None:
                    retrieved_chunks = [*retrieved_chunks, forced_chunk]

                st.output_summary = {
                    "chunk_ids": [c.policy_id for c in retrieved_chunks],
                    "scores": [round(float(c.score), 4) for c in retrieved_chunks],
                    "backend": settings.RETRIEVAL_BACKEND,
                    "forced_policy_key": forced_policy_key,
                    "forced_applied": forced_chunk is not None,
                    "excluded_policy_ids": excluded_policy_ids,
                    "forced_excluded": forced_excluded,
                }

        except Exception:
            logger.exception("Pipeline failed before drafting; aborting.")
            raise

        # --- draft (non-fatal: failure downgrades status) -----------------
        # Appeal reply hook (reject-appeal Phase 4). With the flag ON, a reject-
        # appeal email's draft comes from the approved reply blocks — composed text
        # or a [CHAIR: ...] placeholder — and the model drafter is NOT called for
        # it. prepare_appeal_draft never raises (a failure is the chair-writes
        # placeholder). With the flag OFF, or for any other intent, the drafter is
        # called exactly as before.
        #
        # Reasons (P3): with APPEAL_REPLY_REASON_SOURCE="phase1" the hook gets the
        # phase-1 outcome computed ABOVE in this same run (``phase1_outcome``,
        # None when PHASE1_APPEAL_ENABLED is off), translated by map_phase1 — never
        # the stored phase1_appeals rows. With "appeal_reason" it gets ``mapped``
        # None and reads extraction.appeal_reason exactly as before.
        appeal_reply = None
        with tracer.stage("drafter", {}) as st:
            if (
                settings.APPEAL_REPLY_COMPOSER_ENABLED
                and classification.intent in REJECT_APPEAL_INTENTS
            ):
                mapped = (
                    map_phase1(phase1_outcome)
                    if settings.APPEAL_REPLY_REASON_SOURCE == "phase1"
                    else None
                )
                draft, appeal_reply = prepare_appeal_draft(
                    classification.intent,
                    extraction.appeal_reason,
                    dispute_flag,
                    email_data,
                    window_end=settings.APPEAL_REPLY_WINDOW_END,
                    mapped=mapped,
                )
            else:
                draft = await self.drafter.draft(
                    email_data, classification, retrieved_chunks, forced_policy_key
                )
            st.output_summary = {
                "draft_length": len(draft.draft_text),
                "provider": draft.generation_metadata.get("provider", self.drafter.provider),
                "model_used": draft.model_used,
                "placeholders": len(draft.placeholders),
                "answer_confidence": draft.answer_confidence,
                # Surfaced in the trace so a budget-exhausted draft is greppable,
                # not just inferable from draft_length == 0.
                "truncated": bool(draft.generation_metadata.get("truncated")),
            }
        # A real error outranks truncation: "error" means the call itself failed,
        # while "truncated" means it succeeded but produced no usable reply.
        if draft.generation_metadata.get("error"):
            status = "draft_failed"
        elif draft.generation_metadata.get("truncated"):
            status = "draft_truncated"
        else:
            status = "complete"

        # --- route (now draft-aware): lane = FAQ iff the draft is self-sufficient.
        # The strategy-independent safety floor (chair placeholders or
        # notes-for-chair can NEVER be auto-answered) is applied INSIDE this stage,
        # before output_summary is set, so the persisted trace's lane always
        # matches the final lane (email.routing / audit log) — load-bearing for
        # the RL strategy, which routes without ever seeing the draft.
        with tracer.stage(
            "router",
            {
                "intent": classification.intent,
                "confidence": round(float(classification.confidence), 4),
            },
        ) as st:
            routing = self.router.route(classification, retrieved_chunks, draft)
            routing = apply_self_sufficiency_floor(routing, draft)
            st.output_summary = {
                "lane": routing.lane,
                "reason": routing.reason,
            }

        # --- chair assignment (the second routing decision) ---------------
        # Human-review only; returns None for FAQ-lane emails. Kept out of the
        # tracer stages (the trace contract is exactly classify→retrieve→draft→
        # route) and best-effort so it never breaks the core flow.
        chair_assignment = await self._assign_chair(db, classification, routing)

        record = {
            "sender": sender or "unknown@unknown",
            "sender_name": sender_name,
            "subject": subject,
            "body": body,
            "status": _LIFECYCLE_STATUS[status],
            "classification": classification.model_dump(),
            "routing": routing.model_dump(),
            "draft": _with_appeal_reply(draft.model_dump(), appeal_reply),
            "extraction": extraction.model_dump(),
            "assigned_chair_id": (
                chair_assignment.chair_id if chair_assignment else None
            ),
            # Exact retriever inputs + the grounding set, so a later KB-change
            # sweep can re-run retrieval with no model call and compare. ``prior_intent``
            # is stored so re-eval reproduces the SAME boosted ranking from ctx alone
            # (legacy rows lack it → re-eval passes "" → their unboosted ids still
            # match). ``chunk_hash`` fingerprints (id, intents) so a KB intent re-label
            # that leaves the id-set unchanged is still detected (B5).
            "retrieval_context": {
                "query": query,
                "intent": retrieval_intent,
                "prior_intent": prior_intent,
                "retrieved_ids": [c.policy_id for c in retrieved_chunks],
                "chunk_hash": grounded_chunks_hash(retrieved_chunks),
                # The policy the chair asked to force into this draft, if any
                # (manual invoke). Stored because it is NOT recoverable from
                # retrieved_ids: a forced id that resolved looks identical to one
                # that merely ranked, and a forced id that did NOT resolve leaves
                # no trace at all. With it, "did it apply" is derived at
                # serialization time (``forced_policy_key in retrieved_ids``)
                # rather than stored as a second, drift-prone boolean.
                "forced_policy_key": forced_policy_key,
                # Policies the chair removed from this draft. Stored for the same
                # reason as forced_policy_key — it is NOT recoverable from
                # retrieved_ids, where an excluded policy is indistinguishable
                # from one retrieval simply never ranked. The RAW ids the chair
                # sent are kept, not their resolved lineage roots: roots are
                # re-derived from live rows at apply time, which is exactly what
                # lets an exclusion keep matching after the policy is edited into
                # a new key. Normalized to None so an empty list and "none sent"
                # persist identically.
                "excluded_policy_ids": (
                    list(excluded_policy_ids) if excluded_policy_ids else None
                ),
            },
        }
        # When the caller knows when the requester actually wrote in, store it as
        # received_at instead of letting the column's server_default record our
        # insert time. Added CONDITIONALLY, so the key is ABSENT rather than None
        # when there is no real value:
        #   * Defensive, not load-bearing: ``create_email`` does
        #     ``Email(**record)``, and SQLAlchemy 2.0 does tolerate an explicit
        #     None here (a None-valued attribute on a server_default column is
        #     omitted from the INSERT, so the default still fires — verified).
        #     The conditional avoids DEPENDING on that leniency, and keeps a
        #     meaningless key out of ``record``, which is a shared contract also
        #     read by ``update_email_outputs`` and ``add_processing_result``.
        #   * ``reprocess_email`` / ``reprocess_email_with_thread`` build
        #     ``email_data`` with no "timestamp" key at all, so they add nothing
        #     here and an existing row keeps its received_at — unchanged
        #     behaviour (``update_email_outputs`` also allowlists the columns it
        #     writes, so received_at could not leak through even if present).
        received_at = _parse_received_at(email_data.get("timestamp"))
        if received_at is not None:
            record["received_at"] = received_at
        return _Computed(
            classification=classification,
            retrieved_chunks=retrieved_chunks,
            routing=routing,
            draft=draft,
            chair_assignment=chair_assignment,
            status=status,
            record=record,
            tracer=tracer,
            start=start,
            phase1=phase1_outcome,
        )

    async def _persist_phase1(
        self,
        db: AsyncSession,
        email_id: int,
        zendesk_ticket_id: int | None,
        c: _Computed,
    ) -> None:
        """Apply the run's phase-1 appeal outcome, after the email row is committed.

        No-op when there is no outcome (flag off). Best-effort and never raises,
        like ``_assign_chair``: the repository commits on its own, so a failure
        there cannot touch the email row already written.
        """
        if c.phase1 is None:
            return
        await phase1_appeal_outcome.persist_phase1_outcome(
            db,
            email_id=email_id,
            zendesk_ticket_id=zendesk_ticket_id,
            extraction=c.record.get("extraction"),
            outcome=c.phase1,
            model=phase1_appeal_outcome.active_model_id(),
        )

    async def _finalize(
        self, db: AsyncSession, email_id: str, c: _Computed
    ) -> PipelineResult:
        """Flush the per-stage trace + write audit rows for a persisted email."""
        # Now that the id exists, write the buffered per-stage trace records.
        c.tracer.flush(email_id)

        # --- audit each stage --------------------------------------------
        await self.audit_repo.log_action(
            db, email_id, "classified", "pipeline",
            {"intent": c.classification.intent, "confidence": c.classification.confidence},
        )
        await self.audit_repo.log_action(
            db, email_id, "retrieved", "pipeline",
            {"chunk_ids": [ch.policy_id for ch in c.retrieved_chunks]},
        )
        await self.audit_repo.log_action(
            db, email_id, "routed", "pipeline",
            {"lane": c.routing.lane, "override_reason": c.routing.override_reason},
        )
        # Record the chair assignment (human-review only) — captures the intent +
        # confidence at assignment time, which is exactly the signal a later
        # reroute compares against (Phase 6A step 3 / future training data).
        if c.chair_assignment and c.chair_assignment.chair_id is not None:
            await self.audit_repo.log_action(
                db, email_id, "chair_assigned", "pipeline",
                {
                    "chair_id": c.chair_assignment.chair_id,
                    "chair_name": c.chair_assignment.chair_name,
                    "intent": c.classification.intent,
                    "confidence": c.classification.confidence,
                    "is_fallback": c.chair_assignment.is_fallback,
                    "matched_area": c.chair_assignment.matched_area,
                    "strategy": c.chair_assignment.strategy,
                    "reason": c.chair_assignment.reason,
                },
            )
        await self.audit_repo.log_action(
            db, email_id, "drafted", "pipeline",
            {"model_used": c.draft.model_used, "status": c.status},
        )

        elapsed_ms = (time.perf_counter() - c.start) * 1000.0
        return PipelineResult(
            email_id=email_id,
            classification=c.classification,
            retrieved_chunks=c.retrieved_chunks,
            routing=c.routing,
            draft=c.draft,
            processing_time_ms=elapsed_ms,
            status=c.status,
        )

    async def compute(
        self,
        email_data: dict,
        db: AsyncSession,
        *,
        forced_policy_key: str | None = None,
        excluded_policy_ids: list[str] | None = None,
    ) -> "_Computed":
        """Run classify → retrieve → draft → route WITHOUT persisting anything.

        The stable public seam over ``_compute``: callers that persist the
        result somewhere other than the ``emails`` table (e.g. a per-follow-up
        ``EmailProcessingResult``) use this instead of creating/updating an
        Email row. The returned ``_Computed.record`` mirrors the Email JSON
        column shapes (classification/routing/draft/retrieval_context). Reads
        the chair roster (best-effort) but performs no writes; the caller owns
        persistence.

        ``forced_policy_key`` forwards to ``_compute`` (extra grounding slot), as
        does ``excluded_policy_ids`` (chair removals; filter pending).

        ⚠️ The CALLER owns ``prior_is_reciprocal_dispute`` on this seam (D42).
        This method has no Email row to read it from — that is the whole point
        of the seam — so a caller that persists over an existing row must put
        the prior value into ``email_data`` itself, exactly as
        ``reprocess_email`` does. Omitting it reads as "no prior", which is
        correct for a first computation and WRONG for a re-computation. The same
        holds for ``prior_appeal_reason`` (D66).

        The phase-1 appeal outcome is computed and carried in the result like
        everything else, but this seam never writes it: its rows are keyed to an
        Email row, which this seam does not write either.
        """
        return await self._compute(
            email_data,
            db,
            forced_policy_key=forced_policy_key,
            excluded_policy_ids=excluded_policy_ids,
        )

    async def process_email(
        self, email_data: dict, db: AsyncSession
    ) -> PipelineResult:
        """Run an email through the full pipeline and persist it as a NEW row.

        Does NOT thread ``prior_is_reciprocal_dispute`` (D42) or
        ``prior_appeal_reason`` (D66) — by definition a new row has no prior
        answer, so the absent key correctly reads as None.
        """
        c = await self._compute(email_data, db)
        email = await self.email_repo.create_email(db, c.record)
        # Plain values, read now: a failed phase-1 write rolls the session back,
        # which expires ``email``, and an async lazy reload would raise.
        email_pk = email.id
        # A Zendesk ticket's id is applied to the row by the adapter only after
        # this returns, so the adapter also passes it in email_data.
        ticket_id = getattr(email, "zendesk_ticket_id", None) or email_data.get(
            "zendesk_ticket_id"
        )
        await self._persist_phase1(db, email_pk, ticket_id, c)
        return await self._finalize(db, str(email_pk), c)

    async def reprocess_email(
        self,
        db: AsyncSession,
        email,
        *,
        forced_policy_key: str | None = None,
        excluded_policy_ids: list[str] | None = None,
    ) -> PipelineResult:
        """Re-run the full pipeline for an EXISTING email and update it in place.

        Backs the per-email "retry" action: the same classify → retrieve → draft
        → route flow as ``process_email``, but the fresh outputs overwrite the
        existing row (its id, ``received_at`` and any chair-edit history stay put)
        instead of creating a new one. ``retrieval_context`` is refreshed too, and
        the transient ``redrafting`` flag is cleared as the new draft lands.

        ``forced_policy_key`` (manual invoke) adds one chair-chosen policy to the
        grounding set as an extra slot; ``None`` is the unchanged retry behaviour.
        ``excluded_policy_ids`` rides through to ``_compute`` (filter pending).
        """
        email_data = {
            "from": email.sender,
            "sender_name": email.sender_name,
            "subject": email.subject,
            "body": email.body,
            # D42: carried so a re-run cannot wipe a stored answer when the
            # detector does not run or cannot answer.
            "prior_is_reciprocal_dispute": _prior_dispute_flag(email),
            # D66: the same, for the appeal-reason classifier.
            "prior_appeal_reason": _prior_appeal_reason(email),
            # The phase-1 appeal date gate reads the stored creation time.
            "ticket_created_at": phase1_appeal_outcome.ticket_created_at(email),
        }
        # Plain values, read now: a failed phase-1 write rolls the session back,
        # which expires ``email``, and an async lazy reload would raise.
        email_pk = email.id
        ticket_id = getattr(email, "zendesk_ticket_id", None)
        c = await self._compute(
            email_data,
            db,
            forced_policy_key=forced_policy_key,
            excluded_policy_ids=excluded_policy_ids,
        )
        await self.email_repo.update_email_outputs(db, str(email_pk), c.record)
        await self._persist_phase1(db, email_pk, ticket_id, c)
        return await self._finalize(db, str(email_pk), c)

    async def reprocess_email_with_thread(
        self,
        db: AsyncSession,
        email,
        messages: list[dict],
        *,
        triggering_comment_ids: list | None = None,
    ) -> PipelineResult:
        """Re-run the full pipeline for an existing ticket over its thread.

        Used when a requester follows up: builds a transcript from the stored
        thread (internal notes excluded, latest requester turn anchored),
        re-runs classify → retrieve → draft → route, preserves the outgoing
        draft in ``draft["history"]`` (D4), and updates the row in place
        (status → draft_generated). ``received_at`` / id / stored body are kept.
        """
        transcript = build_transcript(
            messages,
            char_budget=settings.THREAD_TRANSCRIPT_MAX_CHARS,
            requester_id=getattr(email, "zendesk_requester_id", None),
        )
        email_data = {
            "from": email.sender,
            "sender_name": email.sender_name,
            "subject": email.subject,
            # Persistence ignores body on update (update_email_outputs skips it);
            # the current ask is the sensible in-pipeline value.
            "body": transcript.latest_requester_message or email.body,
            "thread_transcript": transcript.text,
            "latest_requester_message": transcript.latest_requester_message,
            # D42, and this is the path the rule EXISTS for: a follow-up
            # re-classifies on the latest turn, so an appeal thread whose newest
            # message is "any update?" leaves the gate — without this the stored
            # True would be overwritten with None.
            "prior_is_reciprocal_dispute": _prior_dispute_flag(email),
            # D66: the same, for the appeal-reason classifier.
            "prior_appeal_reason": _prior_appeal_reason(email),
            # The phase-1 appeal date gate reads the stored creation time.
            "ticket_created_at": phase1_appeal_outcome.ticket_created_at(email),
        }
        # Plain values, read now: a failed phase-1 write rolls the session back,
        # which expires ``email``, and an async lazy reload would raise.
        email_pk = email.id
        ticket_id = getattr(email, "zendesk_ticket_id", None)
        c = await self._compute(email_data, db)
        _append_draft_history(c.record, email.draft, triggering_comment_ids)
        await self.email_repo.update_email_outputs(db, str(email_pk), c.record)
        await self._persist_phase1(db, email_pk, ticket_id, c)
        return await self._finalize(db, str(email_pk), c)
