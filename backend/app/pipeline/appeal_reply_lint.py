"""Wording check for appeal reply template bodies (reject-appeal Phase 3, D85/D86/D94).

``lint_template_body(body, blocked_on=())`` returns a list of violations, each a
``(rule_name, matched_text_short)`` pair; ``[]`` means clean. It turns the
failure kinds found in real replies (Phase 3 Step 1, F1/F2/F7) into a check:
identifiers, year- or time-specific wording, internal roles and process,
concessions, promised outcomes, disclosures, and email structure that belongs
to the style guide rather than the body.

Pure: no I/O, no model calls. Enforced by the loader (an approved entry whose
body fails is refused) and by tests over every template in the file. The future
approval step must call it too.

TO ADD A RULE: add one pattern line to ``RULES``. All patterns are compiled
case-INSENSITIVE; wrap a pattern in ``(?-i:...)`` to make it case-sensitive
(as the SPC / AC tokens are).

EXCEPTIONS: the rules are never relaxed. A template entry may instead carry a
reviewed ``lint_waivers`` list that tolerates named rules for that entry's exact
text only (D107/D109). The loader and the composer apply it; this module stays a
plain check with no knowledge of waivers.

DELIBERATE NON-FLAGS (test-pinned): "senior members of the program committee"
(an open question for Marc, not a rule yet), and the two approved sentences
"We will investigate and follow up with you." and "We will consider your input
when studying possible changes for future editions."
"""

from __future__ import annotations

import re

_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

# ---------------------------------------------------------------------------
# THE RULES — one named list of patterns per category.
# ---------------------------------------------------------------------------
RULES: dict[str, list[str]] = {
    "identifiers": [
        r"\bpolicy[_ ]?\d+",                                               # policy_104, policy 104
        r"\b(?:ticket|request|paper|submission|reviewer)s?\s*(?:#|no\.?|number|id)?\s*\d+",
        r"#\s*\d+",                                                        # #21567
        r"\(\s*\d{2,}\s*\)",                                               # (2025), (12345)
        r"\[\s*(?:source|src|ref|policy)\b[^\]]*\]",                       # [source: ...]
        r"\d+",                                                            # any other digit (allowed ones are masked first)
    ],
    "year_or_time_specific": [
        r"\bthis year\b",
        r"\blast year\b",
        r"\bnext year\b",
        r"\brecord number\b",
        r"\b(?:19|20)\d{2}\b",                                             # any 4-digit year
        r"\bAAAI-?\d+",                                                    # AAAI26, AAAI-27
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b",            # 12 September
        rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b",                   # Sept 24
    ],
    "internal_roles_or_process": [
        r"(?-i:\bSPCs?\b)",                                                # token, case-sensitive
        r"\bsenior program committee\b",
        r"\bsenior program chairs?\b",                                     # D109
        r"\barea chairs?\b",
        r"(?-i:\bACs?\b)",                                                 # token, case-sensitive
        r"\bcollusion\b",
        r"\bunprofessional review",
    ],
    "concession": [
        r"\bapologi[sz]",
        r"\bsorry\b",
        r"\bour (?:mistake|error)\b",
        r"\bshould not have been rejected\b",
        r"\breinstat",
        r"\byou are right\b",
        r"\bwe agree with you\b",
    ],
    "promise": [
        r"\btake action\b",
        r"\bwill reconsider\b",
        r"\bwill be reconsidered\b",
        r"\bwe will change\b",
        r"\bwe will reinstate\b",
    ],
    "disclosure": [
        r"\bextend(?:ed)? their review\b",
        r"\bcannot see\b",
        r"\bpost(?:ed|ing)? (?:it )?as a comment\b",
        r"\bdeliberation notes\b",
        r"\bmeta[- ]?reviews?\b",
    ],
    "structure": [
        r"\bdear\b",
        r"\bbest regards\b",
        r"\bAAAI Team\b",
        r"\[\s*(?:author|sender)[ _]name\s*\]",
        r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",                                    # email address
    ],
}

# Digits allowed anywhere: numbered-point markers "(1)"-"(9)" at a line start,
# and "Phase 1" / "Phase 2". Masked out before the identifier rules run.
ALLOWED_DIGIT_SPANS: list[str] = [
    r"(?m)^\([1-9]\)",
    r"\bPhase [12]\b",
]

# Structure, beyond RULES: URLs are refused unless listed EXACTLY here (empty for
# now), and square-bracket placeholders are refused except "[CHAIR: ...]" and a
# placeholder whose blocker is listed in the entry's blocked_on.
ALLOWED_URLS: frozenset[str] = frozenset()
TOLERATED_WHEN_BLOCKED: dict[str, str] = {"[ETHICS FORM ADDRESS]": "ethics_form_address"}

_URL_RE = re.compile(r"https?://[^\s)\]>]+|\bwww\.[^\s)\]>]+", re.I)
_BRACKET_RE = re.compile(r"\[[^\[\]\n]*\]")
_CHAIR_RE = re.compile(r"\[CHAIR:\s*[^\]]*\]")  # mirrors drafter.PLACEHOLDER_RE

_COMPILED = {name: [re.compile(p, re.I) for p in pats] for name, pats in RULES.items()}
_ALLOWED_DIGITS = [re.compile(p) for p in ALLOWED_DIGIT_SPANS]
_SHORT = 40


def _mask(text: str, patterns) -> str:
    """Blank out allowed spans (same length, so match positions stay honest)."""
    for rx in patterns:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _short(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= _SHORT else s[: _SHORT - 1] + "…"


def lint_template_body(body: str, blocked_on=()) -> list[tuple[str, str]]:
    """Every violation in ``body`` as ``(rule_name, matched_text_short)``; ``[]`` = clean.

    ``blocked_on`` is the entry's own list: a placeholder in
    ``TOLERATED_WHEN_BLOCKED`` passes only while its blocker is listed there.
    """
    violations: list[tuple[str, str]] = []
    urls = _URL_RE.findall(body)
    masked = _mask(body, _ALLOWED_DIGITS)
    masked = _URL_RE.sub(lambda m: " " * len(m.group(0)), masked)  # judged by the URL rule alone
    for name, patterns in _COMPILED.items():
        for rx in patterns:
            for m in rx.finditer(masked):
                violations.append((name, _short(m.group(0))))
    for url in urls:
        if url.rstrip(".,;") not in ALLOWED_URLS:
            violations.append(("structure", _short(url)))
    blocked = set(blocked_on or ())
    for ph in _BRACKET_RE.findall(body):
        if _CHAIR_RE.fullmatch(ph):
            continue
        if TOLERATED_WHEN_BLOCKED.get(ph) in blocked:
            continue
        violations.append(("structure", _short(ph)))
    return violations
