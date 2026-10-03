"""Approve appeal reply template blocks in the template file (reject-appeal Phase 3, Step 3c).

Records an approval — ``status``, ``approved_by``, ``approved_at`` and
``approved_sha256`` — on named draft blocks of
``data/reply_templates/appeal_reply_templates.json``. Git history is the approval
record (D80); this script is the one mechanical way to write it.

    cd backend && python scripts/approve_appeal_reply_blocks.py \\
        --ids opening_warm,lead_in_concerns --approved-by "Name" --date 2026-10-02 [--apply]

DRY RUN BY DEFAULT: it checks everything and prints the (id, sha256) pairs it
would stamp, and writes nothing. ``--apply`` writes.

It refuses — writing NOTHING, for ANY id — if a block is missing, appears more
than once, is not ``draft`` (retired, or already approved), has a non-empty
``blocked_on``, or would fail any of the loader's own checks once approved
(cycle, scope, kind and order, placeholder rule, and the wording check with that
block's OWN ``lint_waivers``). Those checks are the loader's ``_failing_rules``,
called directly, so the script can never approve a block the loader would then
refuse (D94: the approval step must call the wording check).

The hash is the loader's ``compute_body_sha256(body)``. Bodies are never
edited and no other entry is touched: only the four approval values of each
named block are rewritten in place, in the raw text, so the file's formatting
and line endings are preserved. Before writing, the result is re-parsed and must
equal the original with exactly those fields changed, and the loader must serve
every approved block from it. The write is atomic (temp file + rename).

⚠️ Approving a block also means editing ``APPROVED_PINS`` in
``tests/test_appeal_reply_template_loader.py`` (D93), or the pin test fails.

Refusals print the block id and rule names only — never body text.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# scripts/approve_appeal_reply_blocks.py -> parents[1] is backend/ (put it on
# sys.path so `app` imports work when run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.pipeline import appeal_reply_templates as art  # noqa: E402

APPROVAL_FIELDS = ("status", "approved_by", "approved_at", "approved_sha256")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Refused(Exception):
    """Raised with a short, body-free reason; nothing has been written."""


@dataclass(frozen=True)
class Stamp:
    block_id: str
    sha256: str


# --- a minimal JSON scanner: value spans in the RAW text --------------------------------
def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _string_end(text: str, i: int) -> int:
    """``text[i]`` is an opening quote; return the index just past the closing one."""
    i += 1
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        i += 1
    raise Refused("malformed_json")


def _value_end(text: str, i: int) -> int:
    """Index just past the JSON value starting at ``text[i]``."""
    c = text[i]
    if c == '"':
        return _string_end(text, i)
    if c in "{[":
        depth = 0
        while i < len(text):
            c = text[i]
            if c == '"':
                i = _string_end(text, i)
                continue
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise Refused("malformed_json")
    j = i
    while j < len(text) and text[j] not in ",}] \t\r\n":
        j += 1
    return j


def _members(text: str, start: int) -> dict[str, tuple[int, int]]:
    """Top-level members of the object starting at ``text[start] == '{'``:
    key -> (value_start, value_end). Nested objects are skipped as whole values,
    so a key inside a nested object (e.g. a waiver's ``approved_by``) is never
    mistaken for the entry's own."""
    members: dict[str, tuple[int, int]] = {}
    i = _skip_ws(text, start + 1)
    while text[i] != "}":
        key_end = _string_end(text, i)
        key = json.loads(text[i:key_end])
        i = _skip_ws(text, key_end)
        if text[i] != ":":
            raise Refused("malformed_json")
        value_start = _skip_ws(text, i + 1)
        value_end = _value_end(text, value_start)
        if key in members:
            raise Refused("duplicate_key")
        members[key] = (value_start, value_end)
        i = _skip_ws(text, value_end)
        if text[i] == ",":
            i = _skip_ws(text, i + 1)
    return members


def _entry_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every entry object in the top-level ``templates`` array."""
    root = _skip_ws(text, 0)
    top = _members(text, root)
    if "templates" not in top:
        raise Refused("no_templates")
    arr_start, arr_end = top["templates"]
    spans = []
    i = _skip_ws(text, arr_start + 1)
    while i < arr_end - 1:
        if text[i] == "{":
            end = _value_end(text, i)
            spans.append((i, end))
            i = end
        i = _skip_ws(text, i)
        if i < len(text) and text[i] == ",":
            i = _skip_ws(text, i + 1)
        elif text[i] == "]":
            break
    return spans


# --- the approval -------------------------------------------------------------------------
def plan(text: str, ids: list[str], approved_by: str, approved_at: str, cycle: str
         ) -> tuple[str, list[Stamp]]:
    """The new file text and the stamps, or ``Refused``. Pure: writes nothing."""
    if not ids:
        raise Refused("no_ids")
    if len(set(ids)) != len(ids):
        raise Refused("duplicate_ids_requested")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise Refused("empty_approved_by")
    if not _DATE_RE.match(approved_at or ""):
        raise Refused("bad_date")
    try:
        date.fromisoformat(approved_at)
    except ValueError:
        raise Refused("bad_date") from None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise Refused("malformed_json") from None
    if data.get("schema_version") != art.SCHEMA_VERSION:
        raise Refused("unsupported_schema_version")

    spans = _entry_spans(text)
    entries = data["templates"]
    if len(spans) != len(entries):
        raise Refused("malformed_json")
    by_id: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        by_id.setdefault(entry.get("id") if isinstance(entry, dict) else None, []).append(index)

    expected = copy.deepcopy(data)
    edits: list[tuple[int, int, str]] = []
    stamps: list[Stamp] = []
    for block_id in ids:
        found = by_id.get(block_id, [])
        if not found:
            raise Refused(f"{block_id}: missing")
        if len(found) > 1:
            raise Refused(f"{block_id}: duplicate_id")
        index = found[0]
        entry = entries[index]
        if entry.get("status") != "draft":
            raise Refused(f"{block_id}: status_is_{entry.get('status')}")
        if entry.get("blocked_on"):
            raise Refused(f"{block_id}: blocked")
        body = entry.get("body")
        if not isinstance(body, str):
            raise Refused(f"{block_id}: wrong_field_types")
        sha = art.compute_body_sha256(body)
        new_values = {"status": "approved", "approved_by": approved_by,
                      "approved_at": approved_at, "approved_sha256": sha}
        candidate = {**entry, **new_values}
        rules = art._failing_rules(candidate, cycle)
        if rules:
            raise Refused(f"{block_id}: {', '.join(rules)}")

        start, end = spans[index]
        members = _members(text, start)
        for field in APPROVAL_FIELDS:
            if field not in members:
                raise Refused(f"{block_id}: missing_fields")
            value_start, value_end = members[field]
            edits.append((value_start, value_end, json.dumps(new_values[field], ensure_ascii=False)))
        expected["templates"][index].update(new_values)
        stamps.append(Stamp(block_id, sha))

    new_text = text
    for value_start, value_end, replacement in sorted(edits, reverse=True):
        new_text = new_text[:value_start] + replacement + new_text[value_end:]

    # The result must be exactly the original with only the approval fields changed.
    if json.loads(new_text) != expected:
        raise Refused("unexpected_change")
    if new_text.count("\r\n") != text.count("\r\n"):
        raise Refused("line_endings_changed")
    return new_text, stamps


def _atomic_write(path: Path, content: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def approve(path: Path, ids: list[str], approved_by: str, approved_at: str, *,
            apply: bool, cycle: str | None = None) -> list[Stamp]:
    """Check and (with ``apply``) record the approvals. Raises ``Refused``."""
    cycle = settings.APPEAL_REPLY_CYCLE if cycle is None else cycle
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    new_text, stamps = plan(text, ids, approved_by, approved_at, cycle)
    content = new_text.encode("utf-8")

    # The loader must serve every approved block from the new file. Checked on a
    # temp copy BEFORE the real file is replaced.
    # The loader logs a refusal for every OTHER entry (draft, retired, ...); those
    # lines are expected here and would only read as failures, so they are muted
    # for the duration of the check.
    fd, check = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.check.", suffix=".tmp")
    loader_log = logging.getLogger(art.__name__)
    previous_level = loader_log.level
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        loader_log.setLevel(logging.ERROR)
        served = {t.id for t in art.load_approved_templates(check, cycle=cycle)}
    finally:
        loader_log.setLevel(previous_level)
        os.unlink(check)
    not_served = [s.block_id for s in stamps if s.block_id not in served]
    if not_served:
        raise Refused(f"loader_would_not_serve: {', '.join(not_served)}")

    if apply:
        _atomic_write(path, content)
    return stamps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ids", required=True, help="Comma-separated block ids.")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--date", required=True, help="Approval date, YYYY-MM-DD.")
    parser.add_argument("--apply", action="store_true", help="Write the file (default: dry run).")
    parser.add_argument("--path", type=Path, default=art.DEFAULT_PATH, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    try:
        stamps = approve(args.path, ids, args.approved_by, args.date, apply=args.apply)
    except Refused as r:
        print(f"REFUSED — nothing written: {r}", file=sys.stderr)
        return 1
    verb = "STAMPED" if args.apply else "DRY RUN — would stamp (nothing written)"
    print(f"{verb}: approved_by={args.approved_by!r} approved_at={args.date}")
    for s in stamps:
        print(f'  ("{s.block_id}", "{s.sha256}")')
    return 0


if __name__ == "__main__":
    sys.exit(main())
