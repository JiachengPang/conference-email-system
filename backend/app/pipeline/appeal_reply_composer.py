"""Appeal reply composer (reject-appeal Phase 3, D96).

Joins APPROVED reply blocks into the MIDDLE of an email by fixed rules — no
greeting, no sign-off (the drafter adds those in Phase 4). Pure: it reads blocks
only through the loader's approved-only functions plus its id-only
``expected_points_for_reasons``, and never opens the template file itself.

⚠️ CALLED BY NOTHING YET. Phase 4 maps the classifier's ``appeal_reason`` plus
``is_reciprocal_dispute`` into ``reasons`` and passes the chair's "forward"
choice for score complaints.

Rules, applied in this order (reasons are registry names plus
``reciprocal_dispute``):
  * R7 reciprocal — ``reciprocal_dispute`` present: the body is
    ``full_reciprocal`` alone. Other reasons go to a chair note.
  * R6 holding — ``wrong_paper_review`` present, or ``score_outcome_mismatch``
    present AND the chair forwards it: a holding reply (``holding_both`` /
    ``holding_wrong_paper`` / ``holding_score_mismatch``). No opening, closing
    or "final" wording. Other reasons go to a chair note.
  * Otherwise, with S = the remaining reasons:
      R4 drop ``general_dissatisfaction`` if S has any other reason;
      R8 refuse if S has more than 3 distinct reasons;
      S == {other} -> the chair line alone;
      S == {general_dissatisfaction} -> opening + reconsider body + closing;
      else MERGED: opening + lead-in, the needed points numbered "(n)" in the
      ONE global point order, deduplicated by id (R3), the chair line after the
      list when ``other`` is in S (R5), then the closing.
  * A needed block that is not approved refuses the reply
    (``missing_approved_block:<id>``) — except an OPTIONAL point, which is
    simply omitted.
  * The composed body is finally linted; any violation refuses it
    (``lint:<rule_name>``, never the matched text). The only rules tolerated are
    those waived by the blocks this reply actually USED (the union of their
    ``lint_waivers``, D107/D109). Waivers come from the loader's approved blocks,
    which validated them; a block whose waiver is missing or invalid is either
    not served at all or fails here, so the reply is refused as before. Waivers
    on blocks the reply did not use are ignored.

Chair notes are plain text for the chair and NEVER part of the body. Never
raises for any input; the same set of reasons in any order gives identical
output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.pipeline.appeal_reasons import REASON_NAMES
from app.pipeline.appeal_reply_lint import lint_template_body
from app.pipeline.appeal_reply_templates import (
    DEFAULT_PATH,
    expected_points_for_reasons,
    load_approved_templates,
)

logger = logging.getLogger(__name__)

RECIPROCAL = "reciprocal_dispute"
WRONG_PAPER = "wrong_paper_review"
SCORE = "score_outcome_mismatch"
GENERAL = "general_dissatisfaction"
OTHER = "other"
MAX_REASONS = 3                                 # R8
SEP = "\n\n"                                    # paragraph separator everywhere

# Registry order + reciprocal last: the one canonical order for sets of reasons,
# so any input ordering gives identical output.
_ORDER = {name: i for i, name in enumerate(REASON_NAMES)} | {RECIPROCAL: len(REASON_NAMES)}
ALLOWED_REASONS = frozenset(_ORDER)

OPENING, LEAD_IN, CLOSING = "opening_warm", "lead_in_concerns", "closing_reviewed"
RECONSIDER, CHAIR_LINE, FULL_RECIPROCAL = "body_reconsider", "line_chair_writes", "full_reciprocal"
HOLDING_BOTH, HOLDING_WRONG_PAPER, HOLDING_SCORE = (
    "holding_both", "holding_wrong_paper", "holding_score_mismatch",
)


@dataclass(frozen=True)
class ComposeResult:
    body: str | None
    mode: str  # merged | holding | standalone | reciprocal | chair_writes | refused
    used_ids: tuple[str, ...] = ()
    chair_notes: tuple[str, ...] = ()
    refusal: str | None = None


class _Refused(Exception):
    """Internal: carries a refusal reason out of the rule code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _refused(reason: str) -> ComposeResult:
    # The reason only — never body text.
    logger.warning("Appeal reply not composed: %s", reason)
    return ComposeResult(body=None, mode="refused", refusal=reason)


def _canonical(reasons) -> list[str]:
    """Validated, deduplicated reasons in the canonical order."""
    if isinstance(reasons, (str, bytes)) or not isinstance(reasons, (list, tuple, set, frozenset)):
        raise _Refused("invalid_input")
    if not all(isinstance(r, str) for r in reasons):
        raise _Refused("invalid_input")
    distinct = set(reasons)
    if not distinct:
        raise _Refused("no_reasons")
    if distinct - ALLOWED_REASONS:
        raise _Refused("unknown_reason")
    return sorted(distinct, key=_ORDER.__getitem__)


def _note(unanswered: list[str]) -> tuple[str, ...]:
    if not unanswered:
        return ()
    return (f"Also raised: {', '.join(unanswered)}. Not answered in this reply.",)


def compose_reply(
    reasons, *, chair_forwards_score_mismatch: bool = False, path: Path | str = DEFAULT_PATH
) -> ComposeResult:
    """Compose the middle of a reply from approved blocks. Never raises."""
    try:
        return _compose(reasons, chair_forwards_score_mismatch is True, path)
    except _Refused as r:
        return _refused(r.reason)
    except Exception as exc:  # noqa: BLE001 - must never raise
        logger.warning("Appeal reply composer failed (%s).", type(exc).__name__)
        return ComposeResult(body=None, mode="refused", refusal="internal_error")


def _compose(reasons, forwards_score: bool, path) -> ComposeResult:
    ordered = _canonical(reasons)
    blocks = {t.id: t for t in load_approved_templates(path)}

    def need(block_id: str) -> str:
        if block_id not in blocks:
            raise _Refused(f"missing_approved_block:{block_id}")
        return blocks[block_id].body

    present = set(ordered)

    # R7 — reciprocal.
    if RECIPROCAL in present:
        body = need(FULL_RECIPROCAL)
        return _finish(body, "reciprocal", (FULL_RECIPROCAL,),
                       _note([r for r in ordered if r != RECIPROCAL]), blocks=blocks)

    # R6 — holding.
    wrong_paper = WRONG_PAPER in present
    score_forwarded = SCORE in present and forwards_score
    if wrong_paper or score_forwarded:
        hold_id = HOLDING_BOTH if (wrong_paper and score_forwarded) else (
            HOLDING_WRONG_PAPER if wrong_paper else HOLDING_SCORE)
        answered = {WRONG_PAPER} | ({SCORE} if score_forwarded else set())
        return _finish(need(hold_id), "holding", (hold_id,),
                       _note([r for r in ordered if r not in answered]), blocks=blocks)

    s = list(ordered)
    # R4 — general dissatisfaction only on its own.
    if GENERAL in s and len(s) > 1:
        s.remove(GENERAL)
    # R8 — too many reasons.
    if len(s) > MAX_REASONS:
        raise _Refused("too_many_reasons")

    if s == [OTHER]:
        return _finish(need(CHAIR_LINE), "chair_writes", (CHAIR_LINE,), (), blocks=blocks)
    if s == [GENERAL]:
        body = SEP.join([need(OPENING), need(RECONSIDER), need(CLOSING)])
        return _finish(body, "standalone", (OPENING, RECONSIDER, CLOSING), (), blocks=blocks)

    # MERGED. Points are gathered per reason, then deduplicated (R3) and put in
    # the one global order — so a point shared by two reasons appears once and
    # the numbering never depends on which reason brought it in.
    point_reasons = [r for r in s if r != OTHER]
    gathered: list[tuple[str, bool]] = []
    for r in point_reasons:
        gathered.extend(expected_points_for_reasons([r], path))
    rank = {pid: i for i, (pid, _) in enumerate(expected_points_for_reasons(point_reasons, path))}
    seen: set[str] = set()
    points: list[tuple[str, bool]] = []
    for pid, optional in gathered:
        if pid not in seen:
            seen.add(pid)
            points.append((pid, optional))
    points.sort(key=lambda p: rank.get(p[0], len(rank)))

    texts: list[str] = []
    used = [OPENING, LEAD_IN]
    for pid, optional in points:
        if pid not in blocks:
            if optional:
                continue  # an optional point that is not approved is simply omitted
            raise _Refused(f"missing_approved_block:{pid}")
        texts.append(blocks[pid].body)
        used.append(pid)
    if not texts:
        raise _Refused("no_points")

    parts = [f"{need(OPENING)} {need(LEAD_IN)}"]
    parts += [f"({n}) {text}" for n, text in enumerate(texts, start=1)]
    if OTHER in s:  # R5 — after the list, before the closing
        parts.append(need(CHAIR_LINE))
        used.append(CHAIR_LINE)
    parts.append(need(CLOSING))
    used.append(CLOSING)
    return _finish(SEP.join(parts), "merged", tuple(used), (), blocks=blocks)


def _finish(
    body: str, mode: str, used: tuple[str, ...], notes: tuple[str, ...], *, blocks: dict
) -> ComposeResult:
    """Final lint of the composed body, then the result.

    Approved blocks can never carry a ``blocked_on`` (the loader refuses them),
    so the union of the used blocks' ``blocked_on`` is empty by construction.

    The only rules tolerated are those waived by a block in ``used`` (D107/D109).
    ``blocks`` holds the loader's approved blocks only, whose waivers the loader
    already validated; every id in ``used`` is one of them, because a block is
    added to ``used`` only after ``need`` / the point loop found it there.
    """
    body = "\n".join(line.rstrip() for line in body.split("\n")).rstrip()
    waived = frozenset(w.rule for block_id in used for w in blocks[block_id].lint_waivers)
    violations = [v for v in lint_template_body(body, ()) if v[0] not in waived]
    if violations:
        raise _Refused("lint:" + ",".join(sorted({name for name, _ in violations})))
    return ComposeResult(body=body, mode=mode, used_ids=used, chair_notes=notes)
