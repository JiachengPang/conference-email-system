"""Golden no-regression check for appeal replies (reject-appeal Step 2.5).

Step 2.5 adds Marc's reviewer-misconduct blocks and new routing for
``record_error``, ``reviewer_misconduct`` and ``missing_material_claim``. Every
reply that does NOT involve those three reasons must stay byte-identical. The
golden file was captured BEFORE the Step 2.5 edits, from the real approved
template file, and is compared exactly here.

Covered (case lists are literals, never read from the modules under test, so a
new reason added there cannot silently widen or shrink them):
  * ``compose_reply`` for every non-empty subset of the seven composer reasons
    that existed before Step 2.5 (127 cases);
  * the appeal reply hook with the phase-1 source (``map_phase1`` then
    ``prepare_appeal_draft``) for every subset of Jiacheng's six pre-existing
    composable/hold reasons, for relation appeal and feedback_only, with one and
    with two papers (256 cases);
  * the hook with the appeal_reason rollback source for every subset of the six
    registry reasons (64 cases).

To re-capture, regenerate on the code BEFORE a change, never after it,
otherwise this test proves nothing:
    python tests/test_appeal_reply_golden_step25.py --write

Deliberately re-captured once AFTER a change: the program chairs' rewording
(approved 2026-10-06). Before writing, the new snapshot was diffed against the
old one: only the 8 compose cases, 5 phase-1 hook cases and 8 rollback hook
cases that use a reworded or retired block changed, and only in body / draft
text and block ids; every mode, chair note and refusal stayed identical.

Re-captured again AFTER a change (approved 2026-10-07): the AI-review reason
merges as a point (standalone_ai_review retired), feedback-only emails and emails
about two or more papers are composed. Diffed before writing: 8 compose cases
(every set with llm_generated_review), 8 rollback hook cases (the same sets) and
82 phase-1 hook cases (an AI-review reason, relation feedback_only, or two
papers) changed; no other case changed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden" / "appeal_replies_step25.json"

REVIEW = "review_decision_appeal"
EMAIL_DATA = {"sender_name": "Jane Doe", "timestamp": "2026-09-15T10:00:00Z"}

# The composer's reasons before Step 2.5: the six registry names + reciprocal.
COMPOSER_REASONS = (
    "wrong_paper_review",
    "score_outcome_mismatch",
    "reviewer_misunderstanding",
    "llm_generated_review",
    "general_dissatisfaction",
    "other",
    "reciprocal_dispute",
)
# Jiacheng's reasons whose routing Step 2.5 does not change.
PHASE1_REASONS = (
    "wrong_paper_review",
    "llm_generated_review",
    "decision_vs_reviews",
    "reviewer_misjudgment",
    "reconsideration_only",
    "other",
)
REGISTRY_REASONS = COMPOSER_REASONS[:-1]
RELATIONS = ("appeal", "feedback_only")
PAPER_SETS = (("12345",), ("11111", "22222"))


def _subsets(names, *, include_empty: bool):
    start = 0 if include_empty else 1
    for size in range(start, len(names) + 1):
        yield from combinations(names, size)


def _draft_dict(draft) -> dict:
    return {
        "draft_text": draft.draft_text,
        "notes_for_chair": draft.notes_for_chair,
        "placeholders": list(draft.placeholders),
        "citations": list(draft.citations),
        "answer_confidence": draft.answer_confidence,
        "model_used": draft.model_used,
        "generation_metadata": dict(draft.generation_metadata),
    }


def _hook(result) -> dict | None:
    if result is None:
        return None
    draft, record = result
    return {"draft": _draft_dict(draft), "record": record}


def build_snapshot() -> dict:
    """Every covered output, keyed by a readable case id. Deterministic."""
    from app.pipeline.appeal_reply_composer import compose_reply
    from app.pipeline.appeal_reply_hook import prepare_appeal_draft
    from app.pipeline.phase1_appeal_classifier import AppealReason, Phase1AppealResult
    from app.pipeline.phase1_appeal_outcome import CLASSIFIED, Phase1Outcome
    from app.pipeline.phase1_reply_mapping import map_phase1

    compose = {}
    for subset in _subsets(COMPOSER_REASONS, include_empty=False):
        compose["+".join(subset)] = asdict(compose_reply(list(subset)))

    phase1 = {}
    for relation in RELATIONS:
        for papers in PAPER_SETS:
            for subset in _subsets(PHASE1_REASONS, include_empty=True):
                result = Phase1AppealResult(
                    relation=relation,
                    papers=list(papers),
                    reasons=[AppealReason(reason=r, quote=f"quote for {r}") for r in subset],
                    dropped_unquoted=[],
                )
                mapped = map_phase1(Phase1Outcome(CLASSIFIED, result))
                key = f"{relation}|{','.join(papers)}|{'+'.join(subset) or '-'}"
                phase1[key] = _hook(prepare_appeal_draft(
                    REVIEW, None, False, dict(EMAIL_DATA), mapped=mapped))

    rollback = {}
    for subset in _subsets(REGISTRY_REASONS, include_empty=True):
        rollback["+".join(subset) or "-"] = _hook(prepare_appeal_draft(
            REVIEW, list(subset), False, dict(EMAIL_DATA), mapped=None))

    return {"compose": compose, "hook_phase1": phase1, "hook_appeal_reason": rollback}


def _dump(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def test_case_counts_are_the_documented_ones():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert len(golden["compose"]) == 127
    assert len(golden["hook_phase1"]) == 256
    assert len(golden["hook_appeal_reason"]) == 64


def test_every_covered_output_is_byte_identical_to_the_golden_file():
    expected = GOLDEN_PATH.read_text(encoding="utf-8")
    assert _dump(build_snapshot()) == expected


def test_score_and_reviewer_merged_reply_is_in_the_golden_file():
    """The scores + reviewer reply is covered, composed, not refused."""
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    entry = golden["compose"]["score_outcome_mismatch+reviewer_misunderstanding"]
    assert entry["mode"] == "merged"
    assert entry["used_ids"] == [
        "opening_warm", "lead_in_concerns", "point_review_process", "point_scores",
        "point_rebuttal", "closing_reviewed",
    ]


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: python tests/test_appeal_reply_golden_step25.py --write")
    GOLDEN_PATH.write_text(_dump(build_snapshot()), encoding="utf-8", newline="\n")
    print(f"wrote {GOLDEN_PATH}")
