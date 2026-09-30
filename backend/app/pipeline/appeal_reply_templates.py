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
    with the entry's own ``blocked_on`` (D94).

Refusals are logged with the entry id and the failing rule names ONLY — never
the body text. Bad content never raises: an unreadable or malformed file is
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
from app.pipeline.appeal_reply_lint import lint_template_body

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


def compute_body_sha256(body: str) -> str:
    """sha256 hex of ``body`` exactly as stored: UTF-8 bytes, no normalization.

    The approval step will record this value in ``approved_sha256``.
    """
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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
    # Wording check (D94). Rule NAMES only — the matched text is never kept here,
    # so it can never reach the refusal log.
    lint_rules = sorted({name for name, _ in lint_template_body(body, tuple(entry["blocked_on"]))})
    rules.extend(f"lint:{name}" for name in lint_rules)
    return rules


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
            # Id + rule names only — never the body.
            logger.warning("Appeal reply template %r refused: %s", entry_id, ", ".join(rules))
            continue
        approved.append(ApprovedTemplate(
            id=entry["id"], title=entry["title"], kind=entry["kind"], order=entry["order"],
            optional=entry["optional"], reasons=tuple(entry["reasons"]),
            when_used=entry["when_used"], body=entry["body"],
            approved_by=entry["approved_by"], approved_at=entry["approved_at"],
            approved_sha256=entry["approved_sha256"], cycle=entry["cycle"],
            scope=entry["scope"], basis=tuple(entry["basis"]),
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


def get_template(
    template_id: str, path: Path | str = DEFAULT_PATH, *, cycle: str | None = None
) -> ApprovedTemplate | None:
    """The approved template with ``template_id``, or ``None``."""
    for t in load_approved_templates(path, cycle=cycle):
        if t.id == template_id:
            return t
    return None
