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

Decision order (``decide_appeal_reply``):
  1. ``is_reciprocal_dispute is True`` -> compose with ``reciprocal_dispute``
     (mode ``reciprocal_review``: no draft, tagged for Marc).
  2. ``appeal_reason`` None or [] -> placeholder, reason not determined.
  3. ``desk_reject_appeal`` (not reciprocal) -> placeholder: the approved
     post-review wording assumes a reviewed paper (D110).
  4. ``review_decision_appeal`` with reasons -> ``compose_reply(reasons)``.
  5. Phase 1 window, applied LAST and ONLY to outcomes that would be composed
     text (modes ``merged`` / ``standalone``): with a window set, a ticket
     created after it — or with an unknown creation time — gets the placeholder
     and the note "Appeal reply wording is for Phase 1 rejections only".
     No-draft, reciprocal-review and chair-writes outcomes keep their own notes.

Inputs come only from the pipeline's own values: the classified intent, the
extraction's ``appeal_reason`` and ``is_reciprocal_dispute``, and the ticket
creation time from ``email_data``. Nothing here reads the phase-1 appeals table
or depends on PHASE1_APPEAL_ENABLED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.pipeline.appeal_reply_composer import RECIPROCAL, compose_reply
from app.pipeline.appeal_reply_templates import DEFAULT_PATH
from app.pipeline.drafter import DraftResponse, find_placeholders
from app.pipeline.taxonomy import REJECT_APPEAL_INTENTS

logger = logging.getLogger(__name__)

DESK_REJECT = "desk_reject_appeal"

# The common sign-off for every composed appeal reply (D106 corrected: no
# sender-name line). Deliberately OUTSIDE the block file and the wording check:
# it contains "2027", which the block wording check's year rule would refuse.
SIGN_OFF = "Best Regards,\nAAAI 2027 PC Team"
FALLBACK_NAME = "Author"

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
    # reason_unknown | desk_reject | window | failed.
    mode: str
    reasons: tuple[str, ...] | None
    block_ids: tuple[str, ...]
    middle: str | None          # composed middle; set only for merged / standalone
    notes: tuple[str, ...]

    def record(self) -> dict:
        """The small record stored as ``draft["appeal_reply"]``."""
        return {
            "mode": self.mode,
            "reasons": None if self.reasons is None else list(self.reasons),
            "block_ids": list(self.block_ids),
        }


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


def decide_appeal_reply(
    intent: str | None,
    appeal_reason,
    is_reciprocal_dispute,
    created_at,
    *,
    window_end=None,
    path: Path | str = DEFAULT_PATH,
) -> AppealReplyDecision | None:
    """The appeal-reply decision, or None when ``intent`` is not an appeal intent."""
    if intent not in REJECT_APPEAL_INTENTS:
        return None
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


def _greeting_name(sender_name) -> str:
    name = " ".join(sender_name.split()) if isinstance(sender_name, str) else ""
    return name or FALLBACK_NAME


def build_appeal_draft(decision: AppealReplyDecision, sender_name) -> DraftResponse:
    """The draft for a decision: composed text with greeting and sign-off for
    ``merged`` / ``standalone``, a single placeholder line otherwise."""
    if decision.mode in COMPOSED_MODES and decision.middle:
        text = f"Dear {_greeting_name(sender_name)},\n\n{decision.middle}\n\n{SIGN_OFF}"
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


def _failed(appeal_reason) -> tuple[DraftResponse, dict]:
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
) -> tuple[DraftResponse, dict] | None:
    """``(draft, appeal_reply record)`` for an appeal email, None for any other
    intent. NEVER raises: any failure gives the chair-writes placeholder."""
    try:
        decision = decide_appeal_reply(
            intent, appeal_reason, is_reciprocal_dispute, ticket_created_at(email_data),
            window_end=window_end, path=path,
        )
        if decision is None:
            return None
        return build_appeal_draft(decision, email_data.get("sender_name")), decision.record()
    except Exception as exc:  # noqa: BLE001 - must never break the pipeline
        logger.warning("Appeal reply hook failed (%s); chair-writes placeholder.", type(exc).__name__)
        if intent not in REJECT_APPEAL_INTENTS:
            return None
        return _failed(appeal_reason)
