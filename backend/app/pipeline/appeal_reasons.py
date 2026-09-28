"""Appeal-reason registry — the single source of truth for ``appeal_reason``.

Reject-appeal Phase 2 (docs/exp_tracking/reject_appeal.md D57–D68). This module
is a pure registry plus helpers: no I/O, no model calls, and nothing imports it
yet. The reason classifier and its wiring come later.

WIRE FORMAT: full names only (``wrong_paper_review``, …), per D4/D59. The
``label_code`` letters exist ONLY to score against the hand-labeled sets, which
used them; they must never appear in a model prompt's answer contract, the
extraction JSON, the API, or the frontend. ``normalize_reasons`` rejects a
letter code exactly as it rejects any other unknown value.

NOT REASONS, deliberately:
  * ``r`` (reciprocal-review duty dispute) is owned by ``is_reciprocal_dispute``
    (D60). Listing it here would create a second answer to the same question,
    and the two could drift. An old ``r`` label is gold for that flag.
  * ``n`` (not a reject appeal) is not a reason either. When an ``n`` ticket
    reaches the gate, the correct answer is ``[]`` — "asked, no reason fits"
    (D67).

TRI-STATE (D59), carried by the value of ``appeal_reason``:
  * ``None`` — not asked, skipped, or failed. Never "no reason applies".
  * ``[]``   — asked, and no listed reason applies.
  * a list   — the reasons that apply, in registry order. ``other`` is a real
    reason (an appeal on a ground not listed here), distinct from ``[]``.

Every description states the author's CLAIM, not whether the claim is true —
the classifier reads what the author argues, and chairs judge the merits.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class AppealReason:
    name: str  # on the wire
    label_code: str  # scoring only — never on the wire
    escalates: bool  # an escalation ground for Phase 3 replies
    description: str


# Order is part of the contract: `normalize_reasons` returns reasons in this
# order, so stored lists are canonical and comparable.
APPEAL_REASONS: tuple[AppealReason, ...] = (
    AppealReason(
        "wrong_paper_review", "a", True,
        "The author claims a review discusses a different paper than theirs.",
    ),
    AppealReason(
        "score_outcome_mismatch", "b", True,
        "The author claims their scores or ratings do not support the "
        "rejection decision.",
    ),
    AppealReason(
        "reviewer_misunderstanding", "c", False,
        "The author claims a reviewer misread, misunderstood, or did not read "
        "the paper.",
    ),
    AppealReason(
        "llm_generated_review", "d", False,
        "The author claims a review was written by an LLM or other AI tool.",
    ),
    AppealReason(
        "general_dissatisfaction", "e", False,
        "The author objects to the review or decision without a specific "
        "ground, such as calling it unfair and asking for reconsideration.",
    ),
    # Prompt-injection desk-reject complaints land here for now (D61); give
    # them their own entry once there are enough to label and evaluate.
    AppealReason(
        "other", "o", False,
        "The author appeals on a specific ground not listed here, such as a "
        "desk rejection for an unrelated policy reason, or a claim about a "
        "hidden prompt injection in the paper.",
    ),
)

REASON_NAMES: tuple[str, ...] = tuple(r.name for r in APPEAL_REASONS)

# For the eval and label conversion ONLY (old single-code labels map to
# one-element lists, D63). Not for anything on the wire.
LABEL_CODE_TO_NAME: Mapping[str, str] = MappingProxyType(
    {r.label_code: r.name for r in APPEAL_REASONS}
)

_ORDER: Mapping[str, int] = MappingProxyType(
    {name: i for i, name in enumerate(REASON_NAMES)}
)
_ESCALATING: frozenset[str] = frozenset(r.name for r in APPEAL_REASONS if r.escalates)


def normalize_reasons(values: object) -> list[str] | None:
    """Canonical reason list, or ``None`` when ``values`` is not a valid answer.

    * Only a ``list`` of ``str`` is accepted; any other type (``None``, a tuple,
      a bare string, …) and any non-``str`` item give ``None``.
    * ``[]`` gives ``[]`` — asked, no listed reason applies.
    * Duplicates are removed and the result is in registry order.
    * If ANY value is unknown the whole answer is ``None``: a failed answer.
      Unknown values are never silently dropped and never guessed at — no case
      folding, no trimming, no letter-code lookup. Cleaning up raw model text
      is the parser's job, before this is called.

    Always returns a NEW list; the input is never mutated.
    """
    if not isinstance(values, list):
        return None
    if not all(isinstance(v, str) for v in values):
        return None
    if any(v not in _ORDER for v in values):
        return None
    return sorted(set(values), key=_ORDER.__getitem__)


def is_valid_stored(value: object) -> bool:
    """Whether a stored ``appeal_reason`` counts as a prior answer (D66).

    True only for a list already in canonical form — known names, no
    duplicates, registry order — including ``[]``. We only ever write
    ``normalize_reasons`` output, so anything else in storage (``None``, a
    letter code, a stale or reordered list) is not ours to trust, and reads as
    "no prior". ``None`` is False: under D59 it is not an answer at all.
    """
    return normalize_reasons(value) == value if isinstance(value, list) else False


def escalation_reasons(reasons: list[str] | None) -> list[str]:
    """The escalating reasons in ``reasons``, in registry order.

    ``None`` (no answer) gives ``[]``. Anything else must be a valid reason list
    — pass ``normalize_reasons`` output. An invalid list raises ``ValueError``
    rather than returning ``[]``, because ``[]`` would silently read as "no
    escalation ground".
    """
    if reasons is None:
        return []
    normalized = normalize_reasons(reasons)
    if normalized is None:
        raise ValueError("escalation_reasons() needs a valid reason list")
    return [name for name in normalized if name in _ESCALATING]
