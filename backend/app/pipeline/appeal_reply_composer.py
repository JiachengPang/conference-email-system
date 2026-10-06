"""Appeal reply composer (reject-appeal Phase 3, D96; rules per D97-D111).

Joins APPROVED reply blocks into the MIDDLE of an email by fixed rules — no
greeting, no sign-off (the drafter hook adds those in Phase 4). Pure: it reads
blocks only through the loader's approved-only functions plus its id-only
``expected_points_for_reasons``, and never opens the template file itself.

⚠️ CALLED BY NOTHING YET. Phase 4 maps the classifier's ``appeal_reason`` plus
``is_reciprocal_dispute`` into ``reasons`` (``is_reciprocal_dispute == True`` ->
add ``reciprocal_dispute``).

Rules, applied in this order (reasons are registry names plus
``reciprocal_dispute`` plus the composer-only ``reviewer_misconduct``,
``missing_material_claim`` and ``record_error``, Step 2.5):
  1. NO DRAFT — ``wrong_paper_review`` present, alone or with others (D98/D108):
     mode ``no_draft``, body None, refusal None. The chair note says to
     investigate first and not to reply to or close the ticket yet; any other
     reasons are listed. Needs no block.
  2. RECIPROCAL REVIEW — ``reciprocal_dispute`` present (D99/D111): mode
     ``reciprocal_review``, body None, refusal None. The chair note tags it for
     Marc; any other reasons are listed. ``full_reciprocal`` is NEVER served,
     even when approved. Needs no block.
  3. STANDALONE ONLY ALONE — ``general_dissatisfaction`` and
     ``llm_generated_review`` are each answered by ONE complete middle
     (``standalone_general_stage1`` / ``standalone_ai_review``, D103/D104), mode
     ``standalone``. Combined with each other or with ANY other reason, the
     chair writes (mode ``chair_writes``, the chair line, a note naming the
     reasons) — D105.
  4. Then, with only ``score_outcome_mismatch``, ``reviewer_misunderstanding``,
     ``other`` and the composer-only reasons left:
       more than ``MAX_REASONS`` -> the chair writes (reachable since Step 2.5:
       four or more of the merged reasons go to a person);
       {other} -> the chair line alone (mode ``chair_writes``);
       else MERGED: opening + lead-in, the needed points numbered "(n)" in the
       ONE global point order (the file's ``order``), each point once even when
       two reasons need it, the chair line after the list when ``other`` is
       present, then ``closing_reviewed``.
  * Every block a reply needs must be approved, else the reply is refused
    (mode ``refused``, ``missing_approved_block:<id>``). There are no optional
    points (the report-form point is retired, D101).
  * The composed body is finally linted; any violation refuses it
    (``lint:<rule_name>``, never the matched text). The only rules tolerated are
    those waived by the blocks this reply actually USED (the union of their
    ``lint_waivers``, D107/D109). Waivers on blocks it did not use are ignored.

``no_draft`` and ``reciprocal_review`` are deliberate answers, not failures:
they have ``refusal is None``, which is how a caller tells them from
``refused``. Chair notes are plain text for the chair and NEVER part of the
body. Never raises for any input; the same set of reasons in any order gives
identical output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
GENERAL = "general_dissatisfaction"
LLM = "llm_generated_review"
OTHER = "other"
# Composer-only reasons (Step 2.5): names from the phase-1 classifier with an
# approved reply, passed through unchanged by phase1_reply_mapping. Deliberately
# NOT in the appeal-reason registry, whose names are our own classifier's answer
# contract.
MISCONDUCT = "reviewer_misconduct"
MISSING_MATERIAL = "missing_material_claim"
RECORD_ERROR = "record_error"
MAX_REASONS = 3                                 # rule 4 guard
SEP = "\n\n"                                    # paragraph separator everywhere

# The one canonical order for sets of reasons, so any input ordering gives
# identical output. Most critical first (Step 2.5): misconduct, then missing
# material, then the registry in its own order, then record_error, with
# reciprocal last as before. The registry names keep their relative order, so
# every reply without a new reason is unchanged. This orders reasons (notes);
# the POINTS keep the one global order from the template file.
_REASON_ORDER = (MISCONDUCT, MISSING_MATERIAL, *REASON_NAMES, RECORD_ERROR, RECIPROCAL)
_ORDER = {name: i for i, name in enumerate(_REASON_ORDER)}
ALLOWED_REASONS = frozenset(_ORDER)

OPENING, LEAD_IN, CLOSING = "opening_warm", "lead_in_concerns", "closing_reviewed"
CHAIR_LINE = "line_chair_writes"
# Reasons answered by ONE complete middle, never merged with anything (D103-D105).
STANDALONE = {GENERAL: "standalone_general_stage1", LLM: "standalone_ai_review"}

NOTE_NO_DRAFT = (
    "Investigate first: the author says a review is about a different paper. "
    "Do not reply to or close the ticket yet."
)
NOTE_RECIPROCAL = (
    "Reciprocal-review complaint: tagged for Marc to review himself. No reply is drafted."
)


@dataclass(frozen=True)
class ComposeResult:
    body: str | None
    # merged | standalone | chair_writes | no_draft | reciprocal_review | refused
    mode: str
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


def _also_raised(reasons: list[str]) -> tuple[str, ...]:
    return (f"Also raised: {', '.join(reasons)}.",) if reasons else ()


def compose_reply(reasons, *, path: Path | str = DEFAULT_PATH) -> ComposeResult:
    """Compose the middle of a reply from approved blocks. Never raises."""
    try:
        return _compose(reasons, path)
    except _Refused as r:
        return _refused(r.reason)
    except Exception as exc:  # noqa: BLE001 - must never raise
        logger.warning("Appeal reply composer failed (%s).", type(exc).__name__)
        return ComposeResult(body=None, mode="refused", refusal="internal_error")


def _compose(reasons, path) -> ComposeResult:
    ordered = _canonical(reasons)
    present = set(ordered)

    # 1. Wrong-paper review: no draft at all; the chair investigates first.
    if WRONG_PAPER in present:
        return ComposeResult(
            body=None, mode="no_draft",
            chair_notes=(NOTE_NO_DRAFT,) + _also_raised([r for r in ordered if r != WRONG_PAPER]),
        )

    # 2. Reciprocal complaint: no draft; tagged for Marc. full_reciprocal is never served.
    if RECIPROCAL in present:
        return ComposeResult(
            body=None, mode="reciprocal_review",
            chair_notes=(NOTE_RECIPROCAL,) + _also_raised([r for r in ordered if r != RECIPROCAL]),
        )

    blocks = {t.id: t for t in load_approved_templates(path)}

    def need(block_id: str) -> str:
        if block_id not in blocks:
            raise _Refused(f"missing_approved_block:{block_id}")
        return blocks[block_id].body

    def chair_writes(note: str) -> ComposeResult:
        return _finish(need(CHAIR_LINE), "chair_writes", (CHAIR_LINE,), (note,), blocks=blocks)

    # 3. A standalone reply only ever stands alone.
    if present & STANDALONE.keys():
        if len(ordered) > 1:
            return chair_writes(
                "Chair writes: no approved reply covers these reasons together: "
                f"{', '.join(ordered)}."
            )
        block_id = STANDALONE[ordered[0]]
        return _finish(need(block_id), "standalone", (block_id,), (), blocks=blocks)

    # 4. Only merged reasons remain: score_outcome_mismatch,
    # reviewer_misunderstanding, other, and the composer-only reasons.
    if len(ordered) > MAX_REASONS:
        return chair_writes(
            f"Chair writes: more than {MAX_REASONS} issues raised: {', '.join(ordered)}."
        )
    if ordered == [OTHER]:
        return _finish(need(CHAIR_LINE), "chair_writes", (CHAIR_LINE,), (), blocks=blocks)

    # MERGED. The file says which points each reason needs, and in which order;
    # the helper returns each point once, so a point two reasons share (the
    # rebuttal point) appears once. Every needed point must be approved.
    point_ids = [pid for pid, _ in expected_points_for_reasons([r for r in ordered if r != OTHER], path)]
    if not point_ids:
        raise _Refused("no_points")
    parts = [f"{need(OPENING)} {need(LEAD_IN)}"]
    parts += [f"({n}) {need(pid)}" for n, pid in enumerate(point_ids, start=1)]
    used = [OPENING, LEAD_IN, *point_ids]
    if OTHER in present:  # after the list, before the closing
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
    already validated; every id in ``used`` is one of them, because each was
    fetched with ``need`` before the body was built.
    """
    body = "\n".join(line.rstrip() for line in body.split("\n")).rstrip()
    waived = frozenset(w.rule for block_id in used for w in blocks[block_id].lint_waivers)
    violations = [v for v in lint_template_body(body, ()) if v[0] not in waived]
    if violations:
        raise _Refused("lint:" + ",".join(sorted({name for name, _ in violations})))
    return ComposeResult(body=body, mode=mode, used_ids=used, chair_notes=notes)
