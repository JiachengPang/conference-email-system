"""Candidate-SAMPLING patterns for the Phase 2 pool — the one shared place.

These are the rough regexes from the Phase 2 Step 0 investigation
(reject_appeal.md, "Phase 2 Step 0 findings"), moved here VERBATIM so the pool
builder and anything later share one definition rather than retyping it.

⚠️ FOR SAMPLING ONLY — never for classification. Step 0 measured them against
the 200 gold labels: every a/c/d/e hit there was noise, b missed both real b
tickets, and e fires on almost every appeal. They find places worth looking,
nothing more.

Pure, stdlib only, no I/O.
"""

from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern] = {
    "a_wrong_paper": re.compile(
        r"(different|wrong|another|other|unrelated)\s+(paper|submission|manuscript|work)"
        r"|not\s+(about|for|related to|relevant to)\s+(our|my|the)\s+(paper|submission|work)"
        r"|review\w*\s+(of|for|belongs? to|refers? to)\s+(a|an)\s+(different|another|other)"
        r"|(mixed[- ]?up|mix[- ]?up|mismatch\w*)\s+(review|paper)", re.I),
    "b_score_vs_decision": re.compile(
        r"(scores?|ratings?)\b[^.\n]{0,80}\b(reject\w*|decision)"
        r"|(reject\w*|decision)\b[^.\n]{0,80}\b(scores?|ratings?)\b"
        r"|positive\s+(reviews?|scores?|ratings?)|average\s+(score|rating)", re.I),
    "c_misunderstood": re.compile(
        r"misunderst\w+|misread|mis-read|misinterpret\w*"
        r"|(did\s*n[o']t|not|never)\s+(read|understand|understood)\s+(the|our|my)\s+(paper|work|submission|method)"
        r"|failed\s+to\s+(read|understand|notice|recogni[sz]e)|factual(ly)?\s+(error|incorrect|wrong)", re.I),
    "d_llm_generated": re.compile(
        r"chat\s?-?gpt|\bgpt[- ]?\d|\bllm[- ]?(generated|written)|\bai[- ]?(generated|written)"
        r"|(generated|written)\s+(by|with|using)\s+(an?\s+)?(ai|llm|chatgpt|gpt|large language model)", re.I),
    "e_generic_reconsider": re.compile(
        r"reconsider\w*|unfair\w*|unjust\w*|re-?evaluat\w*|re-?assess\w*|\bappeal\w*|not\s+fair", re.I),
    "ctx_reciprocal": re.compile(r"reciprocal", re.I),
}

RECIPROCAL_KEY = "ctx_reciprocal"
REASON_PATTERN_KEYS: tuple[str, ...] = tuple(k for k in PATTERNS if k != RECIPROCAL_KEY)

SEP_15_18 = "sep15-18"
SEP_20_30 = "sep20-30"
OTHER = "other"


def match_text(subject: str | None, initial_message: str | None) -> str:
    """What Step 0 matched against: subject + the first message, newline-joined."""
    return f"{subject or ''}\n{initial_message or ''}"


def pattern_hits(text: str) -> set[str]:
    """Names of every pattern (including ctx_reciprocal) that matches."""
    return {name for name, rx in PATTERNS.items() if rx.search(text)}


def september_window(created_at: str) -> str:
    """Step 0's windows, by the date part of an ISO timestamp (any year)."""
    md = created_at[5:10]
    if "09-15" <= md <= "09-18":
        return SEP_15_18
    if "09-20" <= md <= "09-30":
        return SEP_20_30
    return OTHER
