"""Tests for the flagged AI suggestion's checks and model call (reject-appeal Phase 4, step 2a).

Every model call is mocked: ``_call_model`` / ``_call_local`` are monkeypatched,
and the hermetic conftest pins MODEL_PROVIDER=fallback otherwise. The real
template file is used where the approved wording matters; small synthetic
blocks are used where a check needs a sentence the real file cannot supply.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.core.config import settings
from app.pipeline import appeal_ai_suggestion as ais
from app.pipeline.appeal_ai_suggestion import (
    EXCLUDED_BLOCK_IDS,
    MAX_WORDS,
    build_bank,
    check_answer,
    normalize,
    render_middle,
    split_sentences,
    suggest_appeal_middle,
)
from app.pipeline.appeal_ai_suggestion_prompt import (
    EMAIL_END,
    EMAIL_START,
    SYSTEM_PROMPT,
    build_user_message,
)
from app.pipeline.appeal_reply_composer import compose_reply
from app.pipeline.appeal_reply_templates import (
    ApprovedTemplate,
    LintWaiver,
    load_approved_templates,
)

ROLES = "internal_roles_or_process"
EMAIL_SENTINEL = "EMAIL-SENTINEL-7f3a"
MODEL_SENTINEL = "MODEL-SENTINEL-c91e"


# --- helpers ----------------------------------------------------------------
def _block(block_id: str, body: str, waivers: tuple[str, ...] = ()) -> ApprovedTemplate:
    return ApprovedTemplate(
        id=block_id, title=block_id, kind="point", order=1, optional=False,
        reasons=(), when_used="", body=body, approved_by="test", approved_at="2026-10-04",
        approved_sha256="x", cycle="AAAI-27", scope="phase1_reject", basis=(),
        lint_waivers=tuple(LintWaiver(rule=r, approved_by="test", note="n") for r in waivers),
    )


def _answer(intro: list[str], points: list[list[str]], outro: list[str]) -> str:
    lines = [f"INTRO: {' '.join(intro)}"]
    lines += [f"POINT: {' '.join(p)}" for p in points]
    lines.append(f"OUTRO: {' '.join(outro)}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def real_blocks() -> dict[str, ApprovedTemplate]:
    return {b.id: b for b in load_approved_templates()}


@pytest.fixture(scope="module")
def bank(real_blocks):
    return build_bank(real_blocks.values())


def _s(real_blocks, block_id: str) -> list[str]:
    return split_sentences(real_blocks[block_id].body)


def _t1_answer(real_blocks) -> str:
    """Marc's T1 (score) reply as the model would return it, tagged."""
    return _answer(
        _s(real_blocks, "opening_warm") + _s(real_blocks, "lead_in_concerns"),
        [_s(real_blocks, "point_scores"), _s(real_blocks, "point_rebuttal")],
        _s(real_blocks, "closing_reviewed"),
    )


# --- splitting and normalising ------------------------------------------------
def test_split_keeps_the_lead_in_colon_as_one_sentence_and_drops_point_markers():
    text = "(2) One thing. Two things?\n\nWe would like to respond to your concerns: Next one!"
    assert split_sentences(text) == [
        "One thing.", "Two things?", "We would like to respond to your concerns:", "Next one!",
    ]


def test_normalize_straightens_quotes_and_collapses_whitespace():
    assert normalize("  It’s   a “buddy”\n system ") == "It's a \"buddy\" system"


# --- the sentence bank --------------------------------------------------------
def test_bank_holds_every_approved_block_except_the_excluded_two(real_blocks, bank):
    listed = {block_id for block_id, _ in bank.blocks}
    assert EXCLUDED_BLOCK_IDS == {"full_reciprocal", "line_chair_writes"}
    assert listed == set(real_blocks) - EXCLUDED_BLOCK_IDS
    assert "full_reciprocal" in real_blocks and "line_chair_writes" in real_blocks


def test_no_bank_sentence_carries_a_chair_marker(bank):
    assert not any("[chair" in s.lower() for s in bank.sources)


def test_a_reciprocal_sentence_is_foreign(real_blocks, bank):
    reciprocal = _s(real_blocks, "full_reciprocal")[1]
    answer = _answer(
        _s(real_blocks, "opening_warm") + _s(real_blocks, "lead_in_concerns"),
        [[reciprocal]],
        _s(real_blocks, "closing_reviewed"),
    )
    assert check_answer(answer, bank).failure == ais.FOREIGN_SENTENCE


# --- a valid answer -------------------------------------------------------------
def test_code_numbers_the_points_and_reproduces_the_composed_t1_reply(real_blocks, bank):
    result = check_answer(_t1_answer(real_blocks), bank)
    assert result.failure is None
    assert result.middle == compose_reply(["score_outcome_mismatch"]).body
    assert result.block_ids == (
        "opening_warm", "lead_in_concerns", "point_scores", "point_rebuttal", "closing_reviewed",
    )


def test_render_middle_numbers_points_from_one():
    assert render_middle(["A."], [["B."], ["C.", "D."]], ["E."]) == "A.\n\n(1) B.\n\n(2) C. D.\n\nE."


def test_yan_sentences_may_appear_in_a_merged_reply(real_blocks, bank):
    yan_b = _s(real_blocks, "standalone_ai_review")
    answer = _answer(
        _s(real_blocks, "opening_warm") + _s(real_blocks, "lead_in_concerns"),
        [yan_b[2:5]],
        _s(real_blocks, "closing_reviewed"),
    )
    result = check_answer(answer, bank)
    assert result.failure is None
    assert "standalone_ai_review" in result.block_ids


# --- check 1: NONE --------------------------------------------------------------
@pytest.mark.parametrize("text", ["NONE", "none", "  NONE \n", "", "   ", None])
def test_none_or_empty_is_dropped(text, bank):
    assert check_answer(text, bank) == ais.CheckResult(None, failure=ais.NONE_ANSWER_FAILURE)


# --- check 2: [CHAIR inside the model text ---------------------------------------
@pytest.mark.parametrize("marker", ["[CHAIR: write the point]", "[chair: x]", "[Chair"])
def test_a_chair_marker_is_dropped_as_such(marker, real_blocks, bank):
    answer = _t1_answer(real_blocks).replace("POINT: ", f"POINT: {marker} ", 1)
    assert check_answer(answer, bank).failure == ais.CHAIR_MARKER


# --- check 3: format --------------------------------------------------------------
@pytest.mark.parametrize("mutate", [
    lambda a: a.replace("INTRO: ", "OPENING: ", 1),             # unknown tag
    lambda a: a.split("\n", 1)[1],                              # no INTRO
    lambda a: a.rsplit("\n", 1)[0],                             # no OUTRO
    lambda a: "\n".join(l for l in a.split("\n") if not l.startswith("POINT")),  # no POINT
    lambda a: a + "\nOUTRO: The decision is final.",            # two OUTRO lines
    lambda a: "\n".join([a.split("\n")[1], a.split("\n")[0], *a.split("\n")[2:]]),  # POINT first
    lambda a: a + "\nsome untagged text",                       # untagged line
    lambda a: a.replace("POINT: ", "POINT:   \nPOINT: ", 1),    # an empty POINT
])
def test_a_wrong_shape_is_bad_format(mutate, real_blocks, bank):
    assert check_answer(mutate(_t1_answer(real_blocks)), bank).failure == ais.BAD_FORMAT


# --- check 4: every sentence approved, once ---------------------------------------
@pytest.mark.parametrize("edit", [
    lambda s: s.replace("solely", "only"),        # one word changed
    lambda s: s.replace("Decisions", "decisions"),  # capital letter changed
    lambda s: s.rstrip("."),                      # punctuation dropped
    lambda s: s + " We will review it again.",    # a new sentence added
])
def test_a_changed_or_new_sentence_is_foreign(edit, real_blocks, bank):
    first = _s(real_blocks, "point_scores")[0]
    answer = _t1_answer(real_blocks).replace(first, edit(first), 1)
    assert check_answer(answer, bank).failure == ais.FOREIGN_SENTENCE


def test_the_rebuttal_point_twice_is_a_duplicate(real_blocks, bank):
    answer = _answer(
        _s(real_blocks, "opening_warm") + _s(real_blocks, "lead_in_concerns"),
        [_s(real_blocks, "point_rebuttal"), _s(real_blocks, "point_rebuttal")],
        _s(real_blocks, "closing_reviewed"),
    )
    assert check_answer(answer, bank).failure == ais.DUPLICATE_SENTENCE


# --- check 5: word cap --------------------------------------------------------------
def _words_block(n: int) -> list[ApprovedTemplate]:
    """Three synthetic blocks whose rendered middle has exactly ``n`` words."""
    filler = " ".join(["word"] * (n - 3))
    return [_block("i", "Intro."), _block("p", f"Point {filler}."), _block("o", "Outro.")]


@pytest.mark.parametrize("n,ok", [(MAX_WORDS, True), (MAX_WORDS + 1, False)])
def test_the_word_cap_is_exactly_250(n, ok):
    blocks = _words_block(n)
    answer = _answer(["Intro."], [[blocks[1].body]], ["Outro."])
    result = check_answer(answer, build_bank(blocks))
    assert (result.failure is None) is ok
    if not ok:
        assert result.failure == ais.TOO_LONG


def test_the_point_markers_are_not_counted_as_words():
    blocks = _words_block(MAX_WORDS)
    assert ais._word_count(render_middle(["Intro."], [[blocks[1].body]], ["Outro."])) == MAX_WORDS


def test_all_of_yans_general_reply_is_too_long(real_blocks, bank):
    answer = _answer(
        _s(real_blocks, "opening_warm") + _s(real_blocks, "lead_in_concerns"),
        [_s(real_blocks, "standalone_general_stage1")],
        _s(real_blocks, "closing_reviewed"),
    )
    assert check_answer(answer, bank).failure == ais.TOO_LONG


# --- check 6: wording check ---------------------------------------------------------
def test_the_wording_check_drops_a_violation_no_used_block_waives():
    blocks = [_block("i", "Intro."), _block("p", "Senior program committee members decide."),
              _block("o", "Outro.")]
    answer = _answer(["Intro."], [["Senior program committee members decide."]], ["Outro."])
    assert check_answer(answer, build_bank(blocks)).failure == f"lint:{ROLES}"


def test_a_rule_waived_by_a_used_block_is_tolerated():
    blocks = [_block("i", "Intro."), _block("p", "Senior program committee members decide.",
                                            waivers=(ROLES,)), _block("o", "Outro.")]
    answer = _answer(["Intro."], [["Senior program committee members decide."]], ["Outro."])
    assert check_answer(answer, build_bank(blocks)).failure is None


def test_a_waiver_on_an_unused_block_does_not_count():
    blocks = [_block("i", "Intro."), _block("p", "Senior program committee members decide."),
              _block("w", "Unused.", waivers=(ROLES,)), _block("o", "Outro.")]
    answer = _answer(["Intro."], [["Senior program committee members decide."]], ["Outro."])
    assert check_answer(answer, build_bank(blocks)).failure == f"lint:{ROLES}"


def test_the_real_score_point_passes_through_its_own_waiver(real_blocks, bank):
    assert ROLES in {w.rule for w in real_blocks["point_scores"].lint_waivers}
    assert check_answer(_t1_answer(real_blocks), bank).failure is None


def test_a_year_in_an_otherwise_bank_sentence_fails_the_wording_check():
    blocks = [_block("i", "Intro."), _block("p", "Reviews close in 2026."), _block("o", "Outro.")]
    answer = _answer(["Intro."], [["Reviews close in 2026."]], ["Outro."])
    assert check_answer(answer, build_bank(blocks)).failure.startswith("lint:")


# --- the prompt ---------------------------------------------------------------------
# Approved by Sahil 2026-10-04 with two edits ("at most 250 words"; no cycle names).
# Any change to the wording must be re-approved and these two values updated.
PROMPT_LENGTH = 1665
PROMPT_SHA256 = "9c44c910fef767847410d01c9a5b277aa7bb4649c49ad70fd23c16f227aeb131"


def test_the_system_prompt_is_pinned_by_length_and_hash():
    import hashlib

    from app.pipeline import appeal_ai_suggestion_prompt as prompt

    assert len(SYSTEM_PROMPT) == PROMPT_LENGTH
    assert hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() == PROMPT_SHA256
    assert prompt.PROMPT_SHA256 == PROMPT_SHA256


def test_the_system_prompt_names_no_cycle_so_the_pin_survives_a_new_cycle():
    import re

    assert not re.search(r"AAAI-?\d|\b(?:19|20)\d{2}\b|Best Regards|PC Team", SYSTEM_PROMPT)
    assert "the AAAI program committee" in SYSTEM_PROMPT
    assert "the standard sign-off" in SYSTEM_PROMPT


def test_the_prompt_word_limit_matches_the_code():
    assert f"Keep the reply to at most {MAX_WORDS} words." in SYSTEM_PROMPT


def test_the_user_message_lists_the_bank_and_fences_the_email(bank):
    msg = build_user_message("Subj", EMAIL_SENTINEL, None, ["decision_vs_reviews"], bank.blocks)
    assert "[point_scores]" in msg and "[full_reciprocal]" not in msg
    assert "[chair" not in msg.lower()
    assert msg.index(EMAIL_START) < msg.index(EMAIL_SENTINEL) < msg.index(EMAIL_END)
    assert "decision_vs_reviews" in msg


def test_no_reasons_reads_not_determined(bank):
    assert "not determined" in build_user_message("s", "b", None, None, bank.blocks)


# --- the model call: never raises, failures by name only -----------------------------
def _email() -> dict:
    return {"subject": "Appeal", "body": f"My paper. {EMAIL_SENTINEL}"}


def _fake_model(answer):
    async def fake(user):  # noqa: ANN001
        if isinstance(answer, BaseException):
            raise answer
        return answer
    return fake


async def test_a_valid_answer_gives_the_middle(monkeypatch, real_blocks):
    monkeypatch.setattr(ais, "_call_model", _fake_model(_t1_answer(real_blocks)))
    out = await suggest_appeal_middle(_email(), ["decision_vs_reviews"])
    assert out.failure is None
    assert out.middle == compose_reply(["score_outcome_mismatch"]).body


@pytest.mark.parametrize("answer,failure", [
    (None, ais.NO_MODEL),
    ("NONE", ais.NONE_ANSWER_FAILURE),
    (f"INTRO: {MODEL_SENTINEL}.\nPOINT: x.\nOUTRO: y.", ais.FOREIGN_SENTENCE),
    (RuntimeError(f"boom {EMAIL_SENTINEL}"), ais.ERROR),
    (ValueError("bad"), ais.ERROR),
])
async def test_failures_drop_the_suggestion_and_log_a_name_only(monkeypatch, caplog, answer, failure):
    monkeypatch.setattr(ais, "_call_model", _fake_model(answer))
    with caplog.at_level(logging.DEBUG):
        out = await suggest_appeal_middle(_email(), None)
    assert out == ais.SuggestionOutcome(None, failure=failure)
    assert failure in caplog.text
    assert EMAIL_SENTINEL not in caplog.text and MODEL_SENTINEL not in caplog.text


async def test_a_timeout_drops_the_suggestion(monkeypatch, caplog):
    async def slow(user):  # noqa: ANN001
        await asyncio.sleep(5)
        return "NONE"
    monkeypatch.setattr(ais, "_call_model", slow)
    with caplog.at_level(logging.WARNING):
        out = await suggest_appeal_middle(_email(), None, timeout=0.01)
    assert out.failure == ais.TIMEOUT
    assert ais.TIMEOUT in caplog.text


async def test_the_hermetic_fallback_provider_makes_no_call_and_drops(monkeypatch):
    assert settings.MODEL_PROVIDER == "fallback"
    called = []

    async def must_not_run(user):  # noqa: ANN001
        called.append(user)
        return "NONE"
    monkeypatch.setattr(ais, "_call_local", must_not_run)
    monkeypatch.setattr(ais, "_call_anthropic", must_not_run)
    out = await suggest_appeal_middle(_email(), None)
    assert out.failure == ais.NO_MODEL and called == []


async def test_an_unreadable_template_file_gives_no_bank(tmp_path):
    out = await suggest_appeal_middle(_email(), None, path=tmp_path / "missing.json")
    assert out.failure == ais.NO_BANK


async def test_the_local_call_sends_the_system_prompt_and_config_model(monkeypatch, real_blocks):
    seen = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": _t1_answer(real_blocks)}}]}

    async def fake_post_chat(client, url, payload, headers):  # noqa: ANN001
        seen.update(url=url, payload=payload)
        return _Resp()

    monkeypatch.setattr(settings, "MODEL_PROVIDER", "local")
    monkeypatch.setattr(ais, "post_chat", fake_post_chat)
    out = await suggest_appeal_middle(_email(), None)
    assert out.failure is None
    assert seen["url"].endswith("/chat/completions")
    assert seen["payload"]["model"] == settings.LOCAL_MODEL_NAME
    assert seen["payload"]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert EMAIL_SENTINEL in seen["payload"]["messages"][1]["content"]
