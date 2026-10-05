"""Flagged AI suggestion for review-decision appeals (reject-appeal Phase 4).

When the appeal reply hook can only give the chair a placeholder, one model call
may suggest the MIDDLE of a reply built from approved sentences. This module
makes that call and enforces the result; the hook integration (behind
``APPEAL_AI_SUGGESTION_ENABLED``) lives elsewhere.

THE SENTENCE BANK: every approved block (``load_approved_templates``) except
``EXCLUDED_BLOCK_IDS`` (the reciprocal reply, D111, and the chair-writes line),
split into sentences by :func:`split_sentences`. A sentence ends at ``.`` ``?``
``!`` or ``:`` followed by whitespace or the end of the text; a leading point
marker such as ``(1) `` is removed first. Sentences are normalised with
:func:`normalize` (NFKC, curly quotes made straight, whitespace collapsed) and
otherwise compared EXACTLY, case included.

THE MODEL'S ANSWER: tagged lines (``INTRO:`` once, ``POINT:`` one or more,
``OUTRO:`` once, in that order) or the single word ``NONE``. The CODE renders
it (:func:`render_middle`): the intro paragraph, the points numbered ``(1)``,
``(2)``... (the model never writes numbers), then the outro paragraph, joined
by blank lines, each sentence written in its approved form.

THE CHECKS, in this order; the first failure drops the suggestion:
  1. ``none_answer``  — the model answered NONE (or nothing).
  2. ``chair_marker`` — "[CHAIR" anywhere in the model text (any case).
  3. ``bad_format``   — not the tagged-line shape above.
  4. ``foreign_sentence`` / ``duplicate_sentence`` — every sentence must be an
     approved sentence, each used at most once.
  5. ``too_long``     — more than ``MAX_WORDS`` words (point markers not counted).
  6. ``lint:<rules>`` — the wording check on the rendered middle, tolerating only
     the rules waived by the blocks its sentences came from (the composer's
     D109 rule).
Transport outcomes: ``no_model`` (no real LLM configured), ``timeout``,
``error``. Every failure is logged as its NAME only — never email or model text.
:func:`suggest_appeal_middle` never raises.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.core.config import settings
from app.pipeline.appeal_ai_suggestion_prompt import (
    NONE_ANSWER,
    SYSTEM_PROMPT,
    TAG_INTRO,
    TAG_OUTRO,
    TAG_POINT,
    build_user_message,
)
from app.pipeline.appeal_reply_lint import lint_template_body
from app.pipeline.appeal_reply_templates import (
    DEFAULT_PATH,
    ApprovedTemplate,
    load_approved_templates,
)
from app.pipeline.openai_compat import post_chat

logger = logging.getLogger(__name__)

# Never offered to the model (Sahil, 2a decision 1): the reciprocal reply is
# never served automatically (D111); the chair-writes line is a placeholder.
EXCLUDED_BLOCK_IDS: frozenset[str] = frozenset({"full_reciprocal", "line_chair_writes"})
MAX_WORDS = 250
SEP = "\n\n"
_TIMEOUT_SECONDS = 90.0

# Failure names (the only thing ever logged about a dropped suggestion).
NONE_ANSWER_FAILURE = "none_answer"
CHAIR_MARKER = "chair_marker"
BAD_FORMAT = "bad_format"
FOREIGN_SENTENCE = "foreign_sentence"
DUPLICATE_SENTENCE = "duplicate_sentence"
TOO_LONG = "too_long"
LINT_PREFIX = "lint:"
NO_MODEL = "no_model"
TIMEOUT = "timeout"
ERROR = "error"
NO_BANK = "no_bank"

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_SENTENCE_END_RE = re.compile(r"(?<=[.!?:])\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_POINT_MARKER_RE = re.compile(r"^\(\d+\)\s*")
_LINE_RE = re.compile(rf"^({TAG_INTRO}|{TAG_POINT}|{TAG_OUTRO}):\s*(.*)$")
_CHAIR_MARK = "[chair"


@dataclass(frozen=True)
class SentenceBank:
    """Approved sentences: normalised text -> the block ids it appears in.

    ``blocks`` keeps file order for the prompt: ``[(block_id, [sentence...])]``.
    ``waivers`` maps a block id to the wording-check rules it may waive.
    """

    sources: dict[str, frozenset[str]]
    blocks: tuple[tuple[str, tuple[str, ...]], ...]
    waivers: dict[str, frozenset[str]]


@dataclass(frozen=True)
class CheckResult:
    """A checked answer: ``middle`` and ``block_ids`` on success, else ``failure``."""

    middle: str | None
    block_ids: tuple[str, ...] = ()
    failure: str | None = None


def normalize(text: str) -> str:
    """NFKC, curly quotes made straight, whitespace collapsed, ends stripped."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(_QUOTES).split())


def split_sentences(text: str) -> list[str]:
    """Normalised sentences, paragraph by paragraph; a leading ``(n)`` marker dropped."""
    out: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text or ""):
        paragraph = _POINT_MARKER_RE.sub("", paragraph.strip(), count=1)
        out.extend(s for s in _SENTENCE_END_RE.split(normalize(paragraph)) if s)
    return out


def build_bank(blocks: Iterable[ApprovedTemplate]) -> SentenceBank:
    """The sentence bank from approved blocks, minus the excluded ones."""
    sources: dict[str, set[str]] = {}
    listed: list[tuple[str, tuple[str, ...]]] = []
    waivers: dict[str, frozenset[str]] = {}
    for block in blocks:
        if block.id in EXCLUDED_BLOCK_IDS:
            continue
        sentences = tuple(s for s in split_sentences(block.body) if _CHAIR_MARK not in s.lower())
        if not sentences:
            continue
        listed.append((block.id, sentences))
        waivers[block.id] = frozenset(w.rule for w in block.lint_waivers)
        for s in sentences:
            sources.setdefault(s, set()).add(block.id)
    return SentenceBank(
        sources={s: frozenset(ids) for s, ids in sources.items()},
        blocks=tuple(listed),
        waivers=waivers,
    )


def _parse(text: str) -> tuple[list[str], list[list[str]], list[str]] | None:
    """``(intro, points, outro)`` sentence lists, or None when the shape is wrong."""
    tagged: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line.strip())
        if m is None or not m.group(2).strip():
            return None
        tagged.append((m.group(1), m.group(2)))
    tags = [t for t, _ in tagged]
    if (
        len(tags) < 3
        or tags[0] != TAG_INTRO
        or tags[-1] != TAG_OUTRO
        or any(t != TAG_POINT for t in tags[1:-1])
    ):
        return None
    intro = split_sentences(tagged[0][1])
    points = [split_sentences(content) for _, content in tagged[1:-1]]
    outro = split_sentences(tagged[-1][1])
    return intro, points, outro


def render_middle(intro: Sequence[str], points: Sequence[Sequence[str]], outro: Sequence[str]) -> str:
    """The reply middle: intro, numbered points, outro, joined by blank lines."""
    parts = [" ".join(intro)]
    parts += [f"({n}) {' '.join(p)}" for n, p in enumerate(points, start=1)]
    parts.append(" ".join(outro))
    return SEP.join(parts)


def _word_count(middle: str) -> int:
    return sum(len(_POINT_MARKER_RE.sub("", p.strip()).split()) for p in middle.split(SEP))


def check_answer(text: str | None, bank: SentenceBank) -> CheckResult:
    """Apply checks 1-6 to the model's raw answer. Pure; never raises."""
    if not isinstance(text, str) or not text.strip() or text.strip().upper() == NONE_ANSWER:
        return CheckResult(None, failure=NONE_ANSWER_FAILURE)
    if _CHAIR_MARK in text.lower():
        return CheckResult(None, failure=CHAIR_MARKER)
    parsed = _parse(text)
    if parsed is None:
        return CheckResult(None, failure=BAD_FORMAT)
    intro, points, outro = parsed

    seen: set[str] = set()
    used: list[str] = []
    for sentence in [*intro, *(s for p in points for s in p), *outro]:
        ids = bank.sources.get(sentence)
        if ids is None:
            return CheckResult(None, failure=FOREIGN_SENTENCE)
        if sentence in seen:
            return CheckResult(None, failure=DUPLICATE_SENTENCE)
        seen.add(sentence)
        used.extend(i for i in sorted(ids) if i not in used)

    middle = render_middle(intro, points, outro)
    if _word_count(middle) > MAX_WORDS:
        return CheckResult(None, failure=TOO_LONG)

    waived = frozenset(rule for block_id in used for rule in bank.waivers.get(block_id, ()))
    violations = sorted({name for name, _ in lint_template_body(middle, ()) if name not in waived})
    if violations:
        return CheckResult(None, failure=LINT_PREFIX + ",".join(violations))
    return CheckResult(middle, block_ids=tuple(used))


# --- model call (same dispatch pattern as the other enrichment modules) -----
async def _call_local(user: str) -> str | None:
    base = settings.LOCAL_MODEL_BASE_URL.rstrip("/")
    payload = {
        "model": settings.LOCAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "max_tokens": settings.DRAFTER_MAX_TOKENS,
        "temperature": settings.DRAFTER_TEMPERATURE,
        "seed": settings.DRAFTER_SEED,
        "stream": False,
    }
    headers = (
        {"Authorization": f"Bearer {settings.LOCAL_MODEL_API_KEY}"}
        if settings.LOCAL_MODEL_API_KEY
        else None
    )
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        resp = await post_chat(client, f"{base}/chat/completions", payload, headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def _call_anthropic(user: str) -> str | None:
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        return None
    from anthropic import AsyncAnthropic  # lazy — SDK optional at import time

    client = AsyncAnthropic(api_key=api_key)
    message = await client.messages.create(
        model=settings.DRAFT_MODEL,
        max_tokens=settings.DRAFTER_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in message.content if b.type == "text")


async def _call_model(user: str) -> str | None:
    """Raw model text, or ``None`` when no real LLM is configured."""
    provider = settings.MODEL_PROVIDER
    if provider in ("anthropic", "anthropic_api"):
        return await _call_anthropic(user)
    if provider == "local":
        return await _call_local(user)
    return None


@dataclass(frozen=True)
class SuggestionOutcome:
    """``middle`` set on success; otherwise ``failure`` names why it was dropped."""

    middle: str | None
    block_ids: tuple[str, ...] = ()
    failure: str | None = None


def _dropped(failure: str) -> SuggestionOutcome:
    # The failure NAME only — never email text or model text.
    logger.warning("Appeal AI suggestion dropped: %s", failure)
    return SuggestionOutcome(None, failure=failure)


async def suggest_appeal_middle(
    email_data: dict,
    reasons: Iterable[str] | None,
    *,
    path: Path | str = DEFAULT_PATH,
    timeout: float = _TIMEOUT_SECONDS,
) -> SuggestionOutcome:
    """One model call, then every check. NEVER raises."""
    try:
        bank = build_bank(load_approved_templates(path))
        if not bank.sources:
            return _dropped(NO_BANK)
        user = build_user_message(
            email_data.get("subject") or "",
            email_data.get("body") or "",
            email_data.get("thread_transcript"),
            reasons,
            bank.blocks,
        )
        try:
            text = await asyncio.wait_for(_call_model(user), timeout=timeout)
        except asyncio.TimeoutError:
            return _dropped(TIMEOUT)
        if text is None:  # no real LLM configured (or no API key): nothing answered
            return _dropped(NO_MODEL)
        result = check_answer(text, bank)
        if result.failure is not None:
            return _dropped(result.failure)
        return SuggestionOutcome(result.middle, result.block_ids)
    except Exception as exc:  # noqa: BLE001 - must never raise
        logger.warning("Appeal AI suggestion dropped: %s (%s)", ERROR, type(exc).__name__)
        return SuggestionOutcome(None, failure=ERROR)
