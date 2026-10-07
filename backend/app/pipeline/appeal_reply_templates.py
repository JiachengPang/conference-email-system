"""Appeal reply templates — the ONLY reader of the template file (reject-appeal Phase 3, D81).

The eight pre-written reply templates live in
``data/reply_templates/appeal_reply_templates.json`` (D91/D92). This module is
the single way code may read them, and it hands out **approved templates only**.
A source-scan test fails if any other module under ``backend/app`` names that
file, so a caller cannot bypass the rules below by reading the JSON itself.

⚠️ CALLED BY NOTHING YET. Wiring into the drafter / pipeline / UI is Phase 4.

Schema 2 (D95): each entry is a BLOCK with a ``kind`` (opening, lead_in, point,
closing, holding, standalone_body, standalone_full, chair_line), an ``order``
(an int for points, null otherwise) and an ``optional`` flag. A file with any
other ``schema_version`` is refused as a whole.

An entry is returned only if it passes EVERY rule (``_failing_rules``):
  * ``kind`` is a known kind, ``order`` fits the kind, ``optional`` is a bool;
  * ``status == "approved"``;
  * ``approved_by``, ``approved_at`` and ``approved_sha256`` are all set;
  * ``approved_sha256`` equals ``compute_body_sha256(body)`` — so editing an
    approved body without re-approving makes it unusable;
  * ``cycle`` equals ``settings.APPEAL_REPLY_CYCLE`` — last cycle's wording is
    never served;
  * ``scope == "phase1_reject"``;
  * ``blocked_on`` is empty;
  * the body holds no square-bracket placeholder except ``[CHAIR: ...]``, which
    the approve endpoint already blocks until the chair fills it (D87);
  * the body passes the wording check ``appeal_reply_lint.lint_template_body``
    with the entry's own ``blocked_on`` (D94) — except for rules the entry's
    own ``lint_waivers`` names (D107/D109, below).

Lint waivers (optional ``lint_waivers``, a list of ``{rule, approved_by, note}``)
tolerate a wording-check rule for that entry's exact text, instead of weakening
the rule for everyone. A waiver counts ONLY when the entry passes every other
rule above — approved, hash-matching, right cycle and scope, unblocked — so a
waiver on a draft entry, or on a body edited after approval, has no effect. It
names exactly one rule and waives that rule only; ``approved_by`` and ``note``
must be non-empty strings; the three keys are the only keys; a rule appears at
most once. A malformed list refuses the entry (``bad_lint_waivers``), and a rule
name that is not in ``RULES`` refuses the whole entry
(``waiver_unknown_rule:<name>``). Valid waivers are carried on the returned
block as ``ApprovedTemplate.lint_waivers`` so the composer can honor them on the
finished email.

Refusals are logged with the entry id and the failing rule names ONLY — never
the body text. Retired entries log nothing; a draft entry logs at most once per
process; an approved entry that fails a rule (or a malformed one) logs every
time (``_log_refusal``). Bad content never raises: an unreadable or malformed file is
logged and yields ``[]``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.pipeline.appeal_reply_lint import RULES, lint_template_body

logger = logging.getLogger(__name__)

# backend/app/pipeline/<this file> -> repo root (/app in the container).
DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "reply_templates" / "appeal_reply_templates.json"
)

REQUIRED_SCOPE = "phase1_reject"
# Schema 2 (D95): the file holds composable BLOCKS, not whole emails. Any other
# schema_version is refused as a whole — a v1 file has no `kind` to compose by.
SCHEMA_VERSION = 2
KINDS = frozenset({
    "opening", "lead_in", "point", "closing", "holding",
    "standalone_body", "standalone_full", "chair_line",
})
_REQUIRED_FIELDS = (
    "id", "title", "kind", "order", "optional", "reasons", "when_used", "body",
    "status", "approved_by", "approved_at", "approved_sha256", "cycle", "scope",
    "basis", "blocked_on",
)
# Any square-bracket placeholder; `[CHAIR: ...]` is the one allowed kind. The
# CHAIR form mirrors `drafter.PLACEHOLDER_RE` (pinned by test), so a CHAIR
# placeholder that passes here is always one the approve endpoint 409s on.
_BRACKET_RE = re.compile(r"\[[^\[\]\n]*\]")
_CHAIR_RE = re.compile(r"\[CHAIR:\s*[^\]]*\]")
# The exact keys of one lint waiver (D107/D109). Nothing more, nothing less.
_WAIVER_KEYS = frozenset({"rule", "approved_by", "note"})
# An unknown waiver rule name is logged; cap it so a junk value stays a short tag.
_WAIVER_NAME_MAX = 64


@dataclass(frozen=True)
class LintWaiver:
    """One reviewed exception to one wording-check rule, for one entry's text."""

    rule: str
    approved_by: str
    note: str


@dataclass(frozen=True)
class ApprovedTemplate:
    """One approved template. Frozen: callers cannot alter what was approved."""

    id: str
    title: str
    kind: str
    order: int | None
    optional: bool
    reasons: tuple[str, ...]
    when_used: str
    body: str
    approved_by: str
    approved_at: str
    approved_sha256: str
    cycle: str
    scope: str
    basis: tuple[str, ...]
    # The entry's valid waivers. Empty for an entry with none; only ever set on a
    # block that passed every rule, so it is safe for the composer to honor.
    lint_waivers: tuple[LintWaiver, ...] = ()


def compute_body_sha256(body: str) -> str:
    """sha256 hex of ``body`` exactly as stored: UTF-8 bytes, no normalization.

    The approval step will record this value in ``approved_sha256``.
    """
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _parse_waivers(entry: dict) -> tuple[tuple[LintWaiver, ...], list[str]]:
    """The entry's lint waivers, or the rule names that refuse them.

    Returns ``(waivers, [])`` when the list is valid (an absent key is an empty
    list), else ``((), failing_rules)``. Never raises. The caller decides whether
    a valid waiver may apply; this only checks its shape.
    """
    if "lint_waivers" not in entry:
        return (), []
    value = entry["lint_waivers"]
    if not isinstance(value, list):
        return (), ["bad_lint_waivers"]
    waivers: list[LintWaiver] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _WAIVER_KEYS:
            return (), ["bad_lint_waivers"]
        rule, approved_by, note = item["rule"], item["approved_by"], item["note"]
        if not all(isinstance(v, str) and v.strip() for v in (rule, approved_by, note)):
            return (), ["bad_lint_waivers"]
        if rule in seen:
            return (), ["bad_lint_waivers"]
        seen.add(rule)
        if rule not in RULES:
            unknown.append(rule[:_WAIVER_NAME_MAX])
            continue
        waivers.append(LintWaiver(rule=rule, approved_by=approved_by, note=note))
    if unknown:
        return (), [f"waiver_unknown_rule:{name}" for name in unknown]
    return tuple(waivers), []


def _failing_rules(entry: dict, cycle: str) -> list[str]:
    """Names of every rule ``entry`` breaks (empty = approved and usable)."""
    missing = [f for f in _REQUIRED_FIELDS if f not in entry]
    if missing:
        return ["missing_fields"]
    body = entry["body"]
    if not isinstance(body, str) or not isinstance(entry["reasons"], list) \
            or not isinstance(entry["basis"], list) or not isinstance(entry["blocked_on"], list):
        return ["wrong_field_types"]
    rules = []
    if entry["kind"] not in KINDS:
        rules.append("bad_kind")
    # `order` is an int for points and null for everything else (bool is an
    # int subclass in Python, so it is excluded explicitly).
    order = entry["order"]
    is_int = isinstance(order, int) and not isinstance(order, bool)
    if (entry["kind"] == "point" and not is_int) or (entry["kind"] != "point" and order is not None):
        rules.append("bad_order")
    if not isinstance(entry["optional"], bool):
        rules.append("bad_optional")
    if entry["status"] != "approved":
        rules.append("not_approved")
    if not (entry["approved_by"] and entry["approved_at"] and entry["approved_sha256"]):
        rules.append("approval_record_incomplete")
    if entry["approved_sha256"] and entry["approved_sha256"] != compute_body_sha256(body):
        rules.append("body_hash_mismatch")
    if entry["cycle"] != cycle:
        rules.append("wrong_cycle")
    if entry["scope"] != REQUIRED_SCOPE:
        rules.append("wrong_scope")
    if entry["blocked_on"]:
        rules.append("blocked")
    if any(not _CHAIR_RE.fullmatch(m) for m in _BRACKET_RE.findall(body)):
        rules.append("unknown_placeholder")
    waivers, waiver_rules = _parse_waivers(entry)
    rules.extend(waiver_rules)
    # Wording check (D94). Rule NAMES only — the matched text is never kept here,
    # so it can never reach the refusal log.
    lint_rules = sorted({name for name, _ in lint_template_body(body, tuple(entry["blocked_on"]))})
    # A waiver applies ONLY to an entry that passes every other rule (D107/D109):
    # approved, hash-matching, right cycle and scope, unblocked, well-formed
    # waivers. Otherwise it has no effect and the lint names stay in the refusal.
    waived = {w.rule for w in waivers} if not rules else set()
    rules.extend(f"lint:{name}" for name in lint_rules if name not in waived)
    return rules


# (file, id, rules) of draft entries already logged in this process.
_LOGGED_DRAFT_REFUSALS: set[tuple[str, str, str]] = set()


def _log_refusal(path, entry, entry_id, rules: list[str]) -> None:
    """Log a refusal by entry id and rule names only — never the body.

    Retired entries are intentional and log nothing. A draft entry (blocked or
    not) logs at most once per process for the same file, id and rules: it is
    refused on every load by design, so repeating it only buries real problems.
    Everything else — an approved entry that fails a rule, a malformed or
    duplicated entry — logs every time, as before.
    """
    status = entry.get("status") if isinstance(entry, dict) else None
    if status == "retired" and "duplicate_id" not in rules:
        return
    if status == "draft" and "duplicate_id" not in rules:
        key = (str(path), str(entry_id), ",".join(rules))
        if key in _LOGGED_DRAFT_REFUSALS:
            return
        _LOGGED_DRAFT_REFUSALS.add(key)
    logger.warning("Appeal reply template %r refused: %s", entry_id, ", ".join(rules))


def load_approved_templates(
    path: Path | str = DEFAULT_PATH, *, cycle: str | None = None
) -> list[ApprovedTemplate]:
    """Every approved, usable template, in file order. Never raises.

    ``cycle`` defaults to ``settings.APPEAL_REPLY_CYCLE``.
    """
    cycle = settings.APPEAL_REPLY_CYCLE if cycle is None else cycle
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        entries = data["templates"]
        if not isinstance(entries, list):
            raise TypeError("'templates' is not a list")
    except Exception as exc:  # noqa: BLE001 - bad content must never raise
        logger.warning(
            "Appeal reply templates unreadable (%s); serving none.", type(exc).__name__
        )
        return []

    ids = [e.get("id") for e in entries if isinstance(e, dict)]
    duplicates = {i for i in ids if ids.count(i) > 1}
    approved: list[ApprovedTemplate] = []
    for entry in entries:
        entry_id = entry.get("id", "<no id>") if isinstance(entry, dict) else "<not an object>"
        rules = ["not_an_object"] if not isinstance(entry, dict) else _failing_rules(entry, cycle)
        if entry_id in duplicates:
            rules.append("duplicate_id")
        if rules:
            _log_refusal(path, entry, entry_id, rules)
            continue
        approved.append(ApprovedTemplate(
            id=entry["id"], title=entry["title"], kind=entry["kind"], order=entry["order"],
            optional=entry["optional"], reasons=tuple(entry["reasons"]),
            when_used=entry["when_used"], body=entry["body"],
            approved_by=entry["approved_by"], approved_at=entry["approved_at"],
            approved_sha256=entry["approved_sha256"], cycle=entry["cycle"],
            scope=entry["scope"], basis=tuple(entry["basis"]),
            lint_waivers=_parse_waivers(entry)[0],
        ))
    return approved


def templates_for_reason(
    reason: str, path: Path | str = DEFAULT_PATH, *, cycle: str | None = None
) -> list[ApprovedTemplate]:
    """ALL approved blocks that serve ``reason``, of any kind, in file order.

    Deliberately never chooses between them — e.g. a score point and a score
    holding reply are both returned; the chair picks (D88).
    """
    return [t for t in load_approved_templates(path, cycle=cycle) if reason in t.reasons]


def templates_of_kind(
    kind: str, path: Path | str = DEFAULT_PATH, *, cycle: str | None = None
) -> list[ApprovedTemplate]:
    """Approved blocks of ``kind``, sorted by ``order`` (points) then ``id``.

    Non-point kinds have no order, so they sort by id alone.
    """
    blocks = [t for t in load_approved_templates(path, cycle=cycle) if t.kind == kind]
    return sorted(blocks, key=lambda t: (t.order is None, t.order or 0, t.id))


def expected_points_for_reasons(
    reasons, path: Path | str = DEFAULT_PATH
) -> list[tuple[str, bool]]:
    """``(id, optional)`` for every POINT in the file whose reasons intersect
    ``reasons`` — INCLUDING points that are not approved, but EXCLUDING retired
    points — in the global point order (``order``, then ``id``). Never raises.

    Ids and flags only, never body text: the composer uses it to know which
    points a reply NEEDS, so a required point that is not approved is refused
    rather than silently dropped. A retired point is no longer part of any
    reply, so it is never needed. Malformed point entries are skipped.
    """
    try:
        wanted = {r for r in reasons if isinstance(r, str)}
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            return []
        entries = data["templates"]
        found = []
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict) or e.get("kind") != "point" or e.get("status") == "retired":
                continue
            eid, order, optional, e_reasons = e.get("id"), e.get("order"), e.get("optional"), e.get("reasons")
            if not (isinstance(eid, str) and isinstance(order, int) and not isinstance(order, bool)
                    and isinstance(optional, bool) and isinstance(e_reasons, list)):
                continue
            if wanted & set(r for r in e_reasons if isinstance(r, str)):
                found.append((order, eid, optional))
        return [(eid, optional) for _, eid, optional in sorted(found)]
    except Exception as exc:  # noqa: BLE001 - never raises
        logger.warning("Appeal reply points unreadable (%s); expecting none.", type(exc).__name__)
        return []


def get_template(
    template_id: str, path: Path | str = DEFAULT_PATH, *, cycle: str | None = None
) -> ApprovedTemplate | None:
    """The approved template with ``template_id``, or ``None``."""
    for t in load_approved_templates(path, cycle=cycle):
        if t.id == template_id:
            return t
    return None
