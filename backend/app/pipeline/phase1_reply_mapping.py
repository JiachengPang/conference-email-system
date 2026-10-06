"""Phase-1 classifier outcome -> appeal reply composer input (reject-appeal Phase 4, P3).

The appeal reply hook can take its reasons from Jiacheng's phase-1 rejection
appeal classifier (``app.pipeline.phase1_appeal_classifier``) instead of our own
appeal-reason classifier. This module is the ONLY place that translates his
outcome into what the hook needs. Pure: no I/O, no model call, never raises.

His contract (read, never changed here): a ``Phase1Outcome`` whose ``state`` is
gate_not_met / failed / not_appeal / classified; when classified, a
``Phase1AppealResult`` with ``relation`` (appeal / feedback_only / not_appeal),
``papers`` (the author's own submission numbers), ``reasons`` (each a name from
his 9 plus a quote) and the derived ``must_verify``. Any later version with the
same contract can be dropped in; a reason name this module does not know is
treated as one the chair must write (fail-safe).

RULES, first match wins (Marc's defaults, decided 2026-10-04; rules 5, 7 and
10 changed in Step 2.5, 2026-10-05):

  1. no outcome (the phase-1 flag is off)        -> hold ``reason_unknown``
  2. gate not met (intent or date)                -> hold ``reason_unknown``
  3. the classifier failed                        -> hold ``reason_unknown``
  4. relation not_appeal                          -> hold ``not_appeal``
  5. wrong_paper_review present                   -> hold ``no_draft`` (investigate first)
     (Step 2.5: decided by the name alone; his ``must_verify`` flag is no
     longer read for routing and record_error no longer holds.)
  6. relation feedback_only                       -> hold ``chair_writes``
  7. any reason the chair writes (other, or an
     unknown name) — even when mixed with
     reasons that could be composed               -> hold ``chair_writes``
  8. two or more distinct papers                  -> hold ``chair_writes``
  9. relation appeal with no verified reason      -> hold ``reason_unknown``
 10. otherwise compose, mapping each reason:
       decision_vs_reviews    -> score_outcome_mismatch    (Marc's T1)
       reviewer_misjudgment   -> reviewer_misunderstanding (Marc's T2)
       llm_generated_review   -> llm_generated_review      (Yan's AI-review reply)
       reconsideration_only   -> general_dissatisfaction   (Yan's general reply)
       reviewer_misconduct    -> reviewer_misconduct       (Marc's 3 points, Step 2.5)
       missing_material_claim -> missing_material_claim    (misconduct point 1, Step 2.5)
       record_error           -> record_error              (the score point, Step 2.5)
     The composer then applies its own rules unchanged: merged points in one
     order, either Yan reply mixed with any other reason goes to the chair
     (D105), more than three reasons go to the chair. A composed outcome
     carries the VERIFY_BEFORE_SENDING notes of the reasons it contains.

A held outcome from rules 5-8 also carries the misconduct check when
reviewer_misconduct is among the reasons (HOLD_VERIFY_REASONS; record_error's
check is never added to a hold). Rules 1-4 and 9 never carry a check.

A hold ``chair_writes`` is answered by the hook with the approved chair-writes
line; the other holds become the hook's own placeholders. Chair notes are plain
text for the chair, never part of the reply. The snapshot stored with the draft
holds NAMES ONLY — never the author's quotes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.pipeline import phase1_appeal_outcome as outcome_rules

logger = logging.getLogger(__name__)

# Hold modes. ``no_draft`` and ``chair_writes`` reuse the composer's mode names so
# the Reject Appeals queue groups them as before; ``not_appeal`` and
# ``reason_unknown`` are the hook's own.
HOLD_REASON_UNKNOWN = "reason_unknown"
HOLD_NOT_APPEAL = "not_appeal"
HOLD_NO_DRAFT = "no_draft"
HOLD_CHAIR_WRITES = "chair_writes"
HOLDS = frozenset({HOLD_REASON_UNKNOWN, HOLD_NOT_APPEAL, HOLD_NO_DRAFT, HOLD_CHAIR_WRITES})

# His reason -> the appeal-reason registry name the composer understands.
COMPOSABLE: dict[str, str] = {
    "decision_vs_reviews": "score_outcome_mismatch",
    "reviewer_misjudgment": "reviewer_misunderstanding",
    "llm_generated_review": "llm_generated_review",
    "reconsideration_only": "general_dissatisfaction",
    # Step 2.5: the composer's own names for these, passed through unchanged.
    "reviewer_misconduct": "reviewer_misconduct",
    "missing_material_claim": "missing_material_claim",
    "record_error": "record_error",
}
# A review about a different paper: no draft, the chair investigates first.
INVESTIGATE: tuple[str, ...] = ("wrong_paper_review",)
# Reasons no approved reply covers: the chair writes the whole reply.
CHAIR_WRITES_REASONS: frozenset[str] = frozenset({"other"})

# What a person must check before a composed reply is sent (Step 2.5). The one
# source for these texts: the chair note here and the CSV export. Most critical
# first, which is the order the notes appear in.
VERIFY_BEFORE_SENDING: dict[str, str] = {
    "reviewer_misconduct": (
        "Before sending, check for harassment or an undisclosed conflict of interest. "
        "If either is present, do not send; forward the ticket to the Ethics Chairs."
    ),
    "record_error": "Before sending, check the review text against its score.",
}
# Reasons whose check is ALSO added when the email is held for the chair (rules
# 5-8: no draft, feedback only, a chair-written reason, several papers). Only
# misconduct: a chair writing that reply still has to look for harassment or a
# conflict of interest. Never on reciprocal, not_appeal or "reason unknown".
HOLD_VERIFY_REASONS: tuple[str, ...] = ("reviewer_misconduct",)
# CSV only (verify_before_sending column), for a wrong-paper ticket held with no
# draft. The chair note for that case is NOTE_INVESTIGATE, unchanged.
VERIFY_WRONG_PAPER = "Do not reply or close the ticket until it is clear what we are doing with it."


def verify_notes(names, *, held: bool) -> tuple[str, ...]:
    """The verify-before-sending texts for these reason names, most critical first.

    ``held`` False (a composable outcome): every reason in VERIFY_BEFORE_SENDING.
    ``held`` True (rules 5-8): only HOLD_VERIFY_REASONS. The one rule shared by
    the mapping's notes and the CSV helper in ``appeal_reply_hook``.
    """
    keys = HOLD_VERIFY_REASONS if held else tuple(VERIFY_BEFORE_SENDING)
    return tuple(VERIFY_BEFORE_SENDING[n] for n in keys if n in names)

# Snapshot states (what the phase-1 side decided), stored with the draft.
STATE_FLAG_OFF = "flag_off"

NOTE_FLAG_OFF = (
    "Chair writes: the phase-1 appeal classifier is turned off, so the appeal reasons "
    "were not determined."
)
NOTE_GATE_NOT_MET = (
    "Chair writes: this email is outside the phase-1 appeal classification (intent or "
    "ticket date), so its reasons were not determined."
)
NOTE_FAILED = "Chair writes: the phase-1 appeal classifier did not return an answer."
NOTE_NOT_APPEAL = (
    "Chair writes: the phase-1 classifier judged this email not to be an appeal (for "
    "example, a request to see the reviews or a reply without a request)."
)
NOTE_INVESTIGATE = {
    "wrong_paper_review": (
        "Investigate first: the author says a review is about a different paper. "
        "Do not reply to or close the ticket yet."
    ),
}
NOTE_FEEDBACK_ONLY = (
    "Chair writes: the author reports a review problem but says they are not asking for "
    "a change."
)
NOTE_NO_VERIFIED_REASON = (
    "Chair writes: the author appeals, but no reason could be verified in the email."
)
NOTE_INTERNAL_ERROR = "Chair writes: the phase-1 outcome could not be read."


@dataclass(frozen=True)
class MappedAppeal:
    """What the hook needs from one phase-1 outcome.

    Exactly one of ``reasons`` (the composer input, registry names, in his
    reasons' order) and ``hold`` (a mode in :data:`HOLDS`) is set. ``notes`` are
    chair notes added by this mapping. ``snapshot`` is the names-only record of
    what the phase-1 side decided: ``state``, ``relation``, ``reasons``,
    ``must_verify`` and ``papers`` (``None`` where it gave no answer).
    """

    reasons: tuple[str, ...] | None
    hold: str | None
    notes: tuple[str, ...] = ()
    snapshot: dict = field(default_factory=dict)


def _snapshot(state, relation=None, reasons=None, must_verify=None, papers=None) -> dict:
    return {
        "state": state,
        "relation": relation,
        "reasons": reasons,
        "must_verify": must_verify,
        "papers": papers,
    }


def _raised(prefix: str, names: list[str]) -> tuple[str, ...]:
    return (f"{prefix}: {', '.join(names)}.",) if names else ()


def _hold_checks(names: list[str]) -> tuple[str, ...]:
    """The verify-before-sending notes a HELD email keeps (misconduct only)."""
    return verify_notes(names, held=True)


def map_phase1(outcome) -> MappedAppeal:
    """Map one phase-1 outcome (``None`` when the flag is off). Never raises."""
    try:
        return _map(outcome)
    except Exception as exc:  # noqa: BLE001 - must never break the pipeline
        logger.warning("Phase-1 reply mapping failed (%s); reason unknown.", type(exc).__name__)
        return MappedAppeal(None, HOLD_REASON_UNKNOWN, (NOTE_INTERNAL_ERROR,),
                            _snapshot(outcome_rules.FAILED))


def _map(outcome) -> MappedAppeal:
    if outcome is None:
        return MappedAppeal(None, HOLD_REASON_UNKNOWN, (NOTE_FLAG_OFF,), _snapshot(STATE_FLAG_OFF))
    state = outcome.state
    if state == outcome_rules.GATE_NOT_MET:
        return MappedAppeal(None, HOLD_REASON_UNKNOWN, (NOTE_GATE_NOT_MET,), _snapshot(state))
    if state == outcome_rules.NOT_APPEAL:
        return MappedAppeal(None, HOLD_NOT_APPEAL, (NOTE_NOT_APPEAL,),
                            _snapshot(state, relation="not_appeal"))
    result = outcome.result
    if state != outcome_rules.CLASSIFIED or result is None:
        # FAILED, or any state this module does not know: never a guess.
        return MappedAppeal(None, HOLD_REASON_UNKNOWN, (NOTE_FAILED,),
                            _snapshot(outcome_rules.FAILED))

    names = list(dict.fromkeys(r.reason for r in result.reasons))
    papers = list(dict.fromkeys(result.papers))
    snapshot = _snapshot(state, relation=result.relation, reasons=names,
                         must_verify=bool(result.must_verify), papers=papers)

    if result.relation == "not_appeal":  # defensive: his outcome rules drop it earlier
        return MappedAppeal(None, HOLD_NOT_APPEAL, (NOTE_NOT_APPEAL,), snapshot)
    investigate = [n for n in INVESTIGATE if n in names]
    if investigate:
        notes = tuple(NOTE_INVESTIGATE[n] for n in investigate)
        others = [n for n in names if n not in investigate]
        return MappedAppeal(None, HOLD_NO_DRAFT,
                            notes + _raised("Also raised", others) + _hold_checks(names), snapshot)
    if result.relation == "feedback_only":
        return MappedAppeal(None, HOLD_CHAIR_WRITES,
                            (NOTE_FEEDBACK_ONLY,) + _raised("Raised", names) + _hold_checks(names),
                            snapshot)
    chair_written = [n for n in names if n in CHAIR_WRITES_REASONS or n not in COMPOSABLE]
    if chair_written:
        note = f"Chair writes: no approved reply covers {', '.join(chair_written)}."
        others = [n for n in names if n not in chair_written]
        return MappedAppeal(None, HOLD_CHAIR_WRITES,
                            (note,) + _raised("Also raised", others) + _hold_checks(names), snapshot)
    if len(papers) >= 2:
        note = (f"Chair writes: the email is about {len(papers)} papers; the approved replies "
                "are written for one paper.")
        return MappedAppeal(None, HOLD_CHAIR_WRITES,
                            (note,) + _raised("Raised", names) + _hold_checks(names), snapshot)
    if not names:
        dropped = list(dict.fromkeys(getattr(result, "dropped_unquoted", None) or []))
        return MappedAppeal(None, HOLD_REASON_UNKNOWN,
                            (NOTE_NO_VERIFIED_REASON,)
                            + _raised("Possibly raised (quote not verified)", dropped),
                            snapshot)
    composer_input = tuple(dict.fromkeys(COMPOSABLE[n] for n in names))
    return MappedAppeal(composer_input, None, verify_notes(names, held=False), snapshot)
