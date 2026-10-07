"""Appeal reply hook (reject-appeal Phase 4, Step 4).

Decides the draft for a reject-appeal email from the APPROVED reply blocks, and
builds it. Called by ``orchestrator._compute`` only when
``settings.APPEAL_REPLY_COMPOSER_ENABLED`` is True and the intent is one of the
two reject-appeal intents; with the flag off nothing here runs.

The result is either composed email text (greeting + approved middle + the
common sign-off) or a single ``[CHAIR: ...]`` placeholder line with a note for
the chair. It is NEVER a model-written draft: for an appeal email the model
drafter is not called at all while the flag is on, and any failure in here
becomes the chair-writes placeholder (``prepare_appeal_draft`` never raises).

WHERE THE REASONS COME FROM (``settings.APPEAL_REPLY_REASON_SOURCE``):

- ``"phase1"`` (the default): the outcome of Jiacheng's phase-1 classifier
  (``app.pipeline.phase1_appeal_classifier``) that ``_compute`` produced in the
  SAME run, translated by ``app.pipeline.phase1_reply_mapping.map_phase1`` and
  passed in as ``mapped``. Never the stored ``phase1_appeals`` rows; a missing,
  failed or gated-out outcome gives a placeholder. ``appeal_reason`` is ignored.
  It therefore depends on PHASE1_APPEAL_ENABLED: with that flag off every
  review-decision appeal gets the "reason not determined" placeholder.
- ``"appeal_reason"`` (the rollback): our own appeal-reason classifier's
  ``extraction.appeal_reason``, exactly as before P3 (it needs
  APPEAL_REASON_CLASSIFIER_ENABLED); ``mapped`` is None.

Decision order with ``mapped`` (phase-1 source; ``_decide_from_phase1``):
  1. ``is_reciprocal_dispute is True`` -> compose with ``reciprocal_dispute``
     (mode ``reciprocal_review``: no draft, tagged for Marc).
  2. ``desk_reject_appeal`` (not reciprocal) -> placeholder: the approved
     post-review wording assumes a reviewed paper (D110). The phase-1
     classifier only answers review-decision appeals, so a desk-reject appeal
     never has its reasons; it is decided before them.
  3. ``review_decision_appeal``: the mapping's hold (``no_draft``,
     ``chair_writes``, ``not_appeal`` or ``reason_unknown``), else
     ``compose_reply(mapped.reasons, feedback_only=..., multiple_papers=...)``
     (both from the mapping), with the mapping's verify-before-sending notes
     (record_error, reviewer_misconduct) kept after the composer's own notes.
  4. The Phase 1 window, as below.

Decision order without ``mapped`` (appeal_reason source; unchanged since Step 4):
  1. ``is_reciprocal_dispute is True`` -> compose with ``reciprocal_dispute``.
  2. ``appeal_reason`` None or [] -> placeholder, reason not determined.
  3. ``desk_reject_appeal`` (not reciprocal) -> placeholder (D110).
  4. ``review_decision_appeal`` with reasons -> ``compose_reply(reasons)``.
  5. The Phase 1 window, as below.

The Phase 1 window is applied LAST and ONLY to outcomes that would be composed
text (modes ``merged`` / ``standalone``): with a window set, a ticket created
after it — or with an unknown creation time — gets the placeholder and the note
"Appeal reply wording is for Phase 1 rejections only". No-draft,
reciprocal-review and chair-writes outcomes keep their own notes.

With the phase-1 source the stored record also carries ``"source": "phase1"``
and ``"phase1"``: the names-only snapshot (state, relation, reasons,
must_verify, papers) — never the author's quotes. With the old source the
record is exactly what it was.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from app.pipeline.appeal_reply_composer import OTHER, RECIPROCAL, compose_reply
from app.pipeline.appeal_reply_templates import DEFAULT_PATH
from app.pipeline.drafter import DraftResponse, find_placeholders
from app.pipeline.phase1_reply_mapping import (
    HOLD_CHAIR_WRITES,
    HOLD_NO_DRAFT,
    VERIFY_WRONG_PAPER,
    MappedAppeal,
    verify_notes,
)
from app.pipeline.taxonomy import REJECT_APPEAL_INTENTS

logger = logging.getLogger(__name__)

DESK_REJECT = "desk_reject_appeal"
# The value of draft["appeal_reply"]["source"] when the phase-1 classifier gave the reasons.
PHASE1_SOURCE = "phase1"

# The common sign-off for every composed appeal reply (D106 corrected: no
# sender-name line). Deliberately OUTSIDE the block file and the wording check:
# it contains "2027", which the block wording check's year rule would refuse.
SIGN_OFF = "Best Regards,\nAAAI 2027 PC Team"
# Every reply goes to all authors of the paper, and requester display names are
# often usernames or in another script, so the greeting names no one.
GREETING = "Dear Authors,"

# Placeholders. All match drafter.PLACEHOLDER_RE, so the approve endpoint returns
# 409 and the send gate refuses until the chair replaces them.
CHAIR_WRITES = "[CHAIR: write reply]"
NO_DRAFT_PLACEHOLDER = "[CHAIR: do not reply yet; see note]"
RECIPROCAL_PLACEHOLDER = "[CHAIR: reciprocal complaint; see note]"

NOTE_REASON_UNKNOWN = (
    "Chair writes: the appeal reason was not determined, so no approved reply could be chosen."
)
NOTE_DESK_REJECT = (
    "Chair writes: desk-rejection appeals get no composed reply, because the approved "
    "wording assumes the paper was reviewed."
)
NOTE_WINDOW = "Appeal reply wording is for Phase 1 rejections only"
NOTE_FAILED = "Chair writes: the appeal reply could not be prepared automatically."

COMPOSED_MODES = frozenset({"merged", "standalone"})
PROVIDER = "appeal_reply_composer"


@dataclass(frozen=True)
class AppealReplyDecision:
    # A composer mode (merged | standalone | chair_writes | no_draft |
    # reciprocal_review | refused) or one of this hook's own:
    # reason_unknown | desk_reject | window | failed | not_appeal.
    mode: str
    reasons: tuple[str, ...] | None
    block_ids: tuple[str, ...]
    middle: str | None          # composed middle; set only for merged / standalone
    notes: tuple[str, ...]
    # Set only with the phase-1 source: "phase1" and the mapping's names-only snapshot.
    source: str | None = None
    phase1: dict | None = None

    def record(self) -> dict:
        """The small record stored as ``draft["appeal_reply"]``.

        ``source`` / ``phase1`` are added ONLY with the phase-1 source, so a
        record made from ``appeal_reason`` is exactly what it was.
        """
        record = {
            "mode": self.mode,
            "reasons": None if self.reasons is None else list(self.reasons),
            "block_ids": list(self.block_ids),
        }
        if self.source is not None:
            record["source"] = self.source
            record["phase1"] = self.phase1
        return record


def _as_utc(value) -> datetime | None:
    """An aware UTC datetime from a datetime or an ISO-8601 string, else None.
    A value without an offset is read as UTC."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def ticket_created_at(email_data: dict) -> datetime | None:
    """When the ticket was created, from the pipeline's own ``email_data``.

    The update paths put the stored row's creation time in
    ``email_data["ticket_created_at"]`` (set by the orchestrator for every
    reprocess, whatever PHASE1_APPEAL_ENABLED says); the ingest paths carry the
    ticket's own ``timestamp`` (the Zendesk ticket's created_at from the
    adapter, or the /ingest payload's). None when neither parses.
    """
    created = _as_utc(email_data.get("ticket_created_at"))
    return created if created is not None else _as_utc(email_data.get("timestamp"))


def _from_compose(result, reasons: list[str]) -> AppealReplyDecision:
    if result.mode == "refused":
        notes = (f"Chair writes: no approved reply could be composed ({result.refusal}).",)
    else:
        notes = tuple(result.chair_notes)
    return AppealReplyDecision(
        mode=result.mode,
        reasons=tuple(reasons),
        block_ids=tuple(result.used_ids),
        middle=result.body if result.mode in COMPOSED_MODES else None,
        notes=notes,
    )


def _outside_window(decision: AppealReplyDecision, created_at, window_end) -> bool:
    """Whether the Phase 1 window replaces this decision: composed text only."""
    if decision.mode not in COMPOSED_MODES or window_end is None:
        return False
    created, end = _as_utc(created_at), _as_utc(window_end)
    return created is None or end is None or created > end


def _decide_from_phase1(
    intent: str,
    mapped: MappedAppeal,
    is_reciprocal_dispute,
    created_at,
    *,
    window_end,
    path: Path | str,
) -> AppealReplyDecision:
    """The decision when the reasons come from the phase-1 classifier (P3)."""
    composer_reasons = list(mapped.reasons or ())
    if is_reciprocal_dispute is True:
        with_reciprocal = [*composer_reasons, RECIPROCAL]
        decision = _from_compose(compose_reply(with_reciprocal, path=path), with_reciprocal)
    elif intent == DESK_REJECT:
        decision = AppealReplyDecision("desk_reject", None, (), None, (NOTE_DESK_REJECT,))
    elif mapped.hold == HOLD_CHAIR_WRITES:
        # The whole reply is the chair's: the approved chair-writes line.
        result = compose_reply([OTHER], path=path)
        if result.mode == "chair_writes":
            decision = AppealReplyDecision("chair_writes", None, tuple(result.used_ids), None,
                                           mapped.notes)
        else:
            refused = _from_compose(result, [OTHER])
            decision = replace(refused, reasons=None, notes=refused.notes + mapped.notes)
    elif mapped.hold is not None:
        decision = AppealReplyDecision(mapped.hold, None, (), None, mapped.notes)
    else:
        decision = _from_compose(
            compose_reply(composer_reasons, path=path, feedback_only=mapped.feedback_only,
                          multiple_papers=mapped.multiple_papers),
            composer_reasons)
        if _outside_window(decision, created_at, window_end):
            decision = AppealReplyDecision("window", tuple(composer_reasons), (), None, (NOTE_WINDOW,))
        # A composable outcome's only mapping notes are the verify-before-sending
        # checks (record_error, reviewer_misconduct). They follow whatever the
        # composer decided, after its own notes.
        decision = replace(decision, notes=decision.notes + mapped.notes)
    return replace(decision, source=PHASE1_SOURCE, phase1=dict(mapped.snapshot))


def decide_appeal_reply(
    intent: str | None,
    appeal_reason,
    is_reciprocal_dispute,
    created_at,
    *,
    window_end=None,
    path: Path | str = DEFAULT_PATH,
    mapped: MappedAppeal | None = None,
) -> AppealReplyDecision | None:
    """The appeal-reply decision, or None when ``intent`` is not an appeal intent.

    ``mapped`` set = the phase-1 source (``appeal_reason`` is then ignored);
    ``mapped`` None = the appeal_reason source, decided exactly as before P3.
    """
    if intent not in REJECT_APPEAL_INTENTS:
        return None
    if mapped is not None:
        return _decide_from_phase1(intent, mapped, is_reciprocal_dispute, created_at,
                                   window_end=window_end, path=path)
    reasons = list(appeal_reason) if isinstance(appeal_reason, list) else None

    if is_reciprocal_dispute is True:
        with_reciprocal = [*(reasons or []), RECIPROCAL]
        return _from_compose(compose_reply(with_reciprocal, path=path), with_reciprocal)
    if not reasons:
        return AppealReplyDecision("reason_unknown", None if reasons is None else (), (), None,
                                   (NOTE_REASON_UNKNOWN,))
    if intent == DESK_REJECT:
        return AppealReplyDecision("desk_reject", tuple(reasons), (), None, (NOTE_DESK_REJECT,))

    decision = _from_compose(compose_reply(reasons, path=path), reasons)
    if decision.mode in COMPOSED_MODES and window_end is not None:
        created, end = _as_utc(created_at), _as_utc(window_end)
        if created is None or end is None or created > end:
            return AppealReplyDecision("window", tuple(reasons), (), None, (NOTE_WINDOW,))
    return decision


# Hook outcomes that carry no verify-before-sending text (Step 2.5).
NO_VERIFY_MODES = frozenset({
    "reciprocal_review", "not_appeal", "reason_unknown", "desk_reject", "failed",
})
VERIFY_SEPARATOR = " "


def verify_before_sending(reasons, record) -> str:
    """The CSV ``verify_before_sending`` text for one phase-1 email (Step 2.5).

    ``reasons`` are the phase-1 classifier's reason names; ``record`` is the
    hook's stored ``appeal_reply`` record for the same run. Mirrors the checks
    the hook puts in the chair note, from the same constants:
      * no text for reciprocal, not_appeal, reason unknown, desk reject, failed;
      * a no-draft hold with a wrong-paper review: VERIFY_WRONG_PAPER first;
      * a hold (the record's ``reasons`` is None): the misconduct check only;
      * otherwise (the composer decided): every check of the reasons present.
    Pure. Deliberately NOT wrapped in a catch-all: an empty cell would silently
    hide a required check, so a broken input must fail the export loudly.
    """
    if not isinstance(record, dict):
        return ""
    mode = record.get("mode")
    if mode is None or mode in NO_VERIFY_MODES:
        return ""
    names = [r for r in (reasons or []) if isinstance(r, str)]
    items: list[str] = []
    if mode == HOLD_NO_DRAFT and "wrong_paper_review" in names:
        items.append(VERIFY_WRONG_PAPER)
    items.extend(verify_notes(names, held=record.get("reasons") is None))
    return VERIFY_SEPARATOR.join(items)


def build_appeal_draft(decision: AppealReplyDecision, sender_name) -> DraftResponse:
    """The draft for a decision: composed text with greeting and sign-off for
    ``merged`` / ``standalone``, a single placeholder line otherwise."""
    if decision.mode in COMPOSED_MODES and decision.middle:
        text = f"{GREETING}\n\n{decision.middle}\n\n{SIGN_OFF}"
    elif decision.mode == "no_draft":
        text = NO_DRAFT_PLACEHOLDER
    elif decision.mode == "reciprocal_review":
        text = RECIPROCAL_PLACEHOLDER
    else:
        text = CHAIR_WRITES
    return DraftResponse(
        draft_text=text,
        notes_for_chair="\n".join(decision.notes) or None,
        placeholders=find_placeholders(text),
        citations=[],
        answer_confidence=None,
        model_used="none",
        generation_metadata={"provider": PROVIDER, "appeal_mode": decision.mode},
    )


def _failed(appeal_reason, mapped=None) -> tuple[DraftResponse, dict]:
    if mapped is not None:
        # Phase-1 source: appeal_reason is not this draft's input, so it is not recorded.
        reasons = list(mapped.reasons) if isinstance(getattr(mapped, "reasons", None), tuple) else None
        snapshot = getattr(mapped, "snapshot", None)
        decision = AppealReplyDecision(
            "failed", None if reasons is None else tuple(reasons), (), None, (NOTE_FAILED,),
            source=PHASE1_SOURCE, phase1=dict(snapshot) if isinstance(snapshot, dict) else None)
        return build_appeal_draft(decision, None), decision.record()
    reasons = list(appeal_reason) if isinstance(appeal_reason, list) else None
    decision = AppealReplyDecision(
        "failed", None if reasons is None else tuple(r for r in reasons if isinstance(r, str)),
        (), None, (NOTE_FAILED,))
    return build_appeal_draft(decision, None), decision.record()


def prepare_appeal_draft(
    intent: str | None,
    appeal_reason,
    is_reciprocal_dispute,
    email_data: dict,
    *,
    window_end=None,
    path: Path | str = DEFAULT_PATH,
    mapped: MappedAppeal | None = None,
) -> tuple[DraftResponse, dict] | None:
    """``(draft, appeal_reply record)`` for an appeal email, None for any other
    intent. NEVER raises: any failure gives the chair-writes placeholder.

    ``mapped`` is the phase-1 outcome translated by ``map_phase1`` (phase-1
    source); None means the reasons come from ``appeal_reason`` as before."""
    try:
        decision = decide_appeal_reply(
            intent, appeal_reason, is_reciprocal_dispute, ticket_created_at(email_data),
            window_end=window_end, path=path, mapped=mapped,
        )
        if decision is None:
            return None
        return build_appeal_draft(decision, email_data.get("sender_name")), decision.record()
    except Exception as exc:  # noqa: BLE001 - must never break the pipeline
        logger.warning("Appeal reply hook failed (%s); chair-writes placeholder.", type(exc).__name__)
        if intent not in REJECT_APPEAL_INTENTS:
            return None
        return _failed(appeal_reason, mapped)
