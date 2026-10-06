"""Flagged AI suggestion in the appeal reply hook (reject-appeal Phase 4, 2b).

When ``APPEAL_AI_SUGGESTION_ENABLED`` is True and the hook could only give a
review-decision appeal a placeholder, one model call
(``appeal_ai_suggestion.suggest_appeal_middle``) may build the middle of a reply
from approved sentences only, every one checked. Called once from
``orchestrator._compute`` right after ``prepare_appeal_draft``.

WHEN (``should_suggest``), decided from the hook's stored record only:
  * intent ``review_decision_appeal`` and the phase-1 reason source
    (``record["source"] == "phase1"``; never the rollback source);
  * hook mode ``chair_writes`` or ``refused`` with at least one verified reason,
    or ``reason_unknown`` ONLY when the classifier failed (snapshot state
    ``failed``) or found no verified reason (state ``classified``, no reasons);
  * never when the relation is not ``appeal`` (feedback only), when any reason is
    ``other`` or a name the mapping does not compose, or when two or more papers
    are involved;
  * never for no_draft, reciprocal_review, desk_reject, not_appeal, merged,
    standalone, window, failed — or for flag off / gate not met.

THE DRAFT: the flag line ``AI_FLAG_LINE`` (a ``[CHAIR: ...]`` placeholder, so the
approve endpoint answers 409 and the send gate refuses until a chair removes
it), a blank line, ``Dear {name},``, the checked middle, the common sign-off.
The hook's chair notes (including the verify-before-sending checks) are kept as
they are. The record's mode becomes ``ai_suggestion``, its ``block_ids`` the
source blocks of the sentences used, and it gains a small ``ai_suggestion``
record (base mode, prompt sha256, model id) — ids only, never text.

FAILURE: any failed check, a NONE answer, a timeout, an error, no model — the
plain hook result stays unchanged; the failure is logged by NAME only. One log
line per triggered run gives the number of model calls made (0 or 1). Never
raises.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.pipeline.appeal_ai_suggestion import NO_BANK, NO_MODEL, suggest_appeal_middle
from app.pipeline.appeal_ai_suggestion_prompt import PROMPT_SHA256
from app.pipeline.appeal_reply_hook import PHASE1_SOURCE, PROVIDER, SIGN_OFF, _greeting_name
from app.pipeline.drafter import DraftResponse, find_placeholders
from app.pipeline.phase1_appeal_outcome import active_model_id
from app.pipeline.phase1_reply_mapping import COMPOSABLE

logger = logging.getLogger(__name__)

REVIEW_DECISION_APPEAL = "review_decision_appeal"
MODE_AI_SUGGESTION = "ai_suggestion"
TRIGGER_MODES = frozenset({"chair_writes", "refused", "reason_unknown"})
AI_FLAG_LINE = (
    "[CHAIR: AI-written suggestion, not approved wording; review and edit before sending]"
)
_STATE_FAILED = "failed"
_STATE_CLASSIFIED = "classified"
# Failures that mean no model was called at all.
_NO_CALL = frozenset({NO_BANK, NO_MODEL})


def should_suggest(intent, record) -> bool:
    """Whether this hook outcome may get the AI suggestion. Pure; never raises."""
    if intent != REVIEW_DECISION_APPEAL or not isinstance(record, dict):
        return False
    if record.get("source") != PHASE1_SOURCE:
        return False
    mode = record.get("mode")
    snapshot = record.get("phase1")
    if mode not in TRIGGER_MODES or not isinstance(snapshot, dict):
        return False
    state = snapshot.get("state")
    if mode == "reason_unknown" and state == _STATE_FAILED:
        return True
    if state != _STATE_CLASSIFIED or snapshot.get("relation") != "appeal":
        return False  # flag off, gate not met, not_appeal, feedback only
    names = [n for n in (snapshot.get("reasons") or []) if isinstance(n, str)]
    papers = {p for p in (snapshot.get("papers") or []) if isinstance(p, str)}
    if any(n not in COMPOSABLE for n in names) or len(papers) >= 2:
        return False  # other, an unknown name, several papers
    if mode == "reason_unknown":
        return not names  # no verified reason
    return bool(names)


def build_ai_draft(middle: str, base: DraftResponse, sender_name) -> DraftResponse:
    """The flagged draft around a checked middle; the hook's notes are kept."""
    text = f"{AI_FLAG_LINE}\n\nDear {_greeting_name(sender_name)},\n\n{middle}\n\n{SIGN_OFF}"
    return DraftResponse(
        draft_text=text,
        notes_for_chair=base.notes_for_chair,
        placeholders=find_placeholders(text),
        citations=[],
        answer_confidence=None,
        model_used=active_model_id() or "none",
        generation_metadata={"provider": PROVIDER, "appeal_mode": MODE_AI_SUGGESTION},
    )


def build_ai_record(record: dict, block_ids) -> dict:
    """The stored record for an AI suggestion: ids only, never text."""
    return {
        **record,
        "mode": MODE_AI_SUGGESTION,
        "block_ids": list(block_ids),
        "ai_suggestion": {
            "base_mode": record.get("mode"),
            "prompt_sha256": PROMPT_SHA256,
            "model": active_model_id(),
        },
    }


async def apply_ai_suggestion(
    intent, draft: DraftResponse, record, email_data: dict, *, timeout: float | None = None
) -> tuple[DraftResponse, dict | None]:
    """The hook's (draft, record), replaced by a flagged AI suggestion when one
    is allowed and passes every check; otherwise returned unchanged. Never raises."""
    if not settings.APPEAL_AI_SUGGESTION_ENABLED:
        return draft, record
    try:
        if not should_suggest(intent, record):
            return draft, record
    except Exception as exc:  # noqa: BLE001 - must never break the pipeline
        logger.warning("Appeal AI suggestion skipped: error (%s)", type(exc).__name__)
        return draft, record

    calls = 1
    try:
        # Names only (the snapshot never holds quotes); empty -> "not determined".
        reasons = [n for n in (record["phase1"].get("reasons") or []) if isinstance(n, str)]
        kwargs = {} if timeout is None else {"timeout": timeout}
        outcome = await suggest_appeal_middle(email_data, reasons, **kwargs)
        if outcome.failure in _NO_CALL:
            calls = 0
        if outcome.middle is None:
            return draft, record  # the failure name is already logged
        return (build_ai_draft(outcome.middle, draft, email_data.get("sender_name")),
                build_ai_record(record, outcome.block_ids))
    except Exception as exc:  # noqa: BLE001 - must never break the pipeline
        logger.warning("Appeal AI suggestion dropped: error (%s)", type(exc).__name__)
        return draft, record
    finally:
        logger.info("Appeal AI suggestion model calls: %d", calls)
