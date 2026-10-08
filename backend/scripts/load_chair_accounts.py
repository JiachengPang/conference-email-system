"""Load the chairs' Zendesk accounts into ``zendesk_chair_accounts``.

The file is PRIVATE and must live OUTSIDE the repo; pass its path, or set
``CHAIR_ACCOUNTS_FILE``. There is no default path. A path inside the repo is
refused. Expected columns (header row; other columns, e.g. an email, are
ignored and never printed):

    chair_name        the chair EXACTLY as the paper-to-chair sheet names them
    zendesk_user_id   the chair's Zendesk agent user id (a positive integer)
    active            optional: true/false, yes/no, 1/0; blank = true

Validation, all against the loaded paper-to-chair sheet
(``paper_assignments``): every name must be a chair the lookup knows
(trimmed, exact case); no name twice; no empty or non-numeric user id; a valid
``active`` value. ANY error writes nothing. Known chairs without a row are
reported as missing (not an error). Output is counts plus the missing and
unknown NAMES only: never a user id, an email or any other column.

Re-running is safe: rows upsert by chair name, and an unchanged row is not
rewritten. ``--check`` validates without writing.

Run with:  cd backend && python scripts/load_chair_accounts.py <path> [--check]
"""

import argparse
import asyncio
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# scripts/load_chair_accounts.py -> parents[1] is backend/, parents[2] the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.repositories.chair_account_repository import (  # noqa: E402
    ChairAccountRepository,
    ChairAccountRow,
    UpsertStats,
)
from app.repositories.phase1_appeal_repository import (  # noqa: E402
    PaperAssignmentRepository,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR = "CHAIR_ACCOUNTS_FILE"

COL_NAME = "chair_name"
COL_USER_ID = "zendesk_user_id"
COL_ACTIVE = "active"
REQUIRED_COLUMNS = (COL_NAME, COL_USER_ID)

# ASCII digits only (str.isdigit also accepts e.g. superscripts, which int() rejects).
_DIGITS = re.compile(r"[0-9]+")
_TRUE = frozenset({"", "true", "yes", "1"})
_FALSE = frozenset({"false", "no", "0"})


class LoaderUsageError(Exception):
    """The path is missing, not a file, or inside the repo. Nothing was read."""


@dataclass
class RawRow:
    chair_name: str
    user_id: str
    active: str


@dataclass
class Validation:
    """The outcome of validating a file. ``rows`` is filled only when there are no errors."""

    rows_read: int = 0
    rows: list[ChairAccountRow] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    unknown_names: list[str] = field(default_factory=list)
    missing_names: list[str] = field(default_factory=list)
    duplicate_names: int = 0
    empty_names: int = 0
    empty_user_ids: int = 0
    bad_user_ids: int = 0
    bad_active_values: int = 0
    shared_user_ids: int = 0
    sheet_empty: bool = False

    @property
    def error_count(self) -> int:
        return (
            len(self.missing_columns)
            + len(self.unknown_names)
            + self.duplicate_names
            + self.empty_names
            + self.empty_user_ids
            + self.bad_user_ids
            + self.bad_active_values
            + int(self.sheet_empty)
        )

    @property
    def ok(self) -> bool:
        return self.error_count == 0


def resolve_path(arg: str | None, environ=os.environ) -> Path:
    """The private file's path: the argument, else ``CHAIR_ACCOUNTS_FILE``.

    Refused (``LoaderUsageError``): no path at all, a path that is not a file,
    or a path inside the repo — the file holds personal data and must never be
    where git could pick it up.
    """
    raw = arg or environ.get(ENV_VAR) or ""
    if not raw.strip():
        raise LoaderUsageError(f"Pass the private file's path, or set {ENV_VAR}.")
    path = Path(raw.strip()).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise LoaderUsageError("Refusing a file inside the repo: keep it outside the repo.")
    if not path.is_file():
        raise LoaderUsageError("No file at that path.")
    return path


def read_rows(path: Path) -> tuple[list[RawRow], list[str]]:
    """The file's rows (stripped) and the required columns it lacks. No DB access."""
    # utf-8-sig: spreadsheet exports often lead with a byte-order mark.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = {(name or "").strip() for name in (reader.fieldnames or [])}
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            return [], missing
        rows = []
        for raw in reader:
            record = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k is not None}
            rows.append(
                RawRow(
                    chair_name=record.get(COL_NAME, ""),
                    user_id=record.get(COL_USER_ID, ""),
                    active=record.get(COL_ACTIVE, ""),
                )
            )
        return rows, []


def validate(rows: list[RawRow], missing_columns: list[str], known_names: list[str]) -> Validation:
    """Check the rows against the sheet's chair names. Pure."""
    v = Validation(rows_read=len(rows), missing_columns=list(missing_columns))
    if missing_columns:
        return v
    known = {n.strip() for n in known_names if isinstance(n, str) and n.strip()}
    v.sheet_empty = not known

    seen: dict[str, int] = {}
    for row in rows:
        if row.chair_name:
            seen[row.chair_name] = seen.get(row.chair_name, 0) + 1
    v.duplicate_names = sum(1 for count in seen.values() if count > 1)

    unknown: list[str] = []
    valid: list[ChairAccountRow] = []
    user_ids: dict[int, int] = {}
    for row in rows:
        if not row.chair_name:
            v.empty_names += 1
            continue
        if known and row.chair_name not in known and row.chair_name not in unknown:
            unknown.append(row.chair_name)
        if not row.user_id:
            v.empty_user_ids += 1
            continue
        if not _DIGITS.fullmatch(row.user_id) or int(row.user_id) <= 0:
            v.bad_user_ids += 1
            continue
        active_raw = row.active.lower()
        if active_raw not in _TRUE and active_raw not in _FALSE:
            v.bad_active_values += 1
            continue
        user_id = int(row.user_id)
        user_ids[user_id] = user_ids.get(user_id, 0) + 1
        valid.append(ChairAccountRow(row.chair_name, user_id, active_raw in _TRUE))

    v.unknown_names = sorted(unknown, key=lambda n: (n.casefold(), n))
    named = set(seen)
    v.missing_names = sorted((known - named), key=lambda n: (n.casefold(), n))
    v.shared_user_ids = sum(1 for count in user_ids.values() if count > 1)
    if v.ok:
        v.rows = valid
    return v


async def load(path: Path, session_factory, *, check: bool = False) -> tuple[Validation, UpsertStats | None]:
    """Validate the file against the loaded sheet; upsert unless ``check`` or invalid."""
    rows, missing_columns = read_rows(path)
    async with session_factory() as db:
        known = await PaperAssignmentRepository().list_distinct_apc_names(db)
        validation = validate(rows, missing_columns, known)
        if check or not validation.ok:
            return validation, None
        stats = await ChairAccountRepository().upsert_many(db, validation.rows)
    return validation, stats


def report(validation: Validation, stats: UpsertStats | None, *, check: bool) -> list[str]:
    """The printed lines: counts, plus missing / unknown NAMES only."""
    lines = [
        f"{'Rows read:':<34}{validation.rows_read}",
        f"{'Errors:':<34}{validation.error_count}",
    ]
    if validation.missing_columns:
        lines.append(f"{'Missing columns:':<34}{', '.join(validation.missing_columns)}")
    if validation.sheet_empty:
        lines.append("The paper-to-chair sheet is not loaded: no chair names to check against.")
    for label, value in (
        ("Unknown names (not in the sheet):", len(validation.unknown_names)),
        ("Duplicate names:", validation.duplicate_names),
        ("Rows with an empty name:", validation.empty_names),
        ("Rows with an empty user id:", validation.empty_user_ids),
        ("Rows with a non-numeric user id:", validation.bad_user_ids),
        ("Rows with a bad active value:", validation.bad_active_values),
        ("User ids shared by several chairs:", validation.shared_user_ids),
        ("Chairs without an account:", len(validation.missing_names)),
    ):
        lines.append(f"{label:<34}{value}")
    if validation.unknown_names:
        lines.append("Unknown names: " + "; ".join(validation.unknown_names))
    if validation.missing_names:
        lines.append("Missing names: " + "; ".join(validation.missing_names))
    if not validation.ok:
        lines.append("Nothing written: fix the errors above.")
    elif check:
        lines.append("Check only: nothing written.")
    elif stats is not None:
        lines.append(
            f"Written: {stats.inserted} inserted, {stats.updated} updated, {stats.unchanged} unchanged."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--check", action="store_true", help="validate only; write nothing")
    args = parser.parse_args(argv)
    try:
        path = resolve_path(args.path)
    except LoaderUsageError as exc:
        print(str(exc))
        return 2

    from app.db.database import async_session_factory

    validation, stats = asyncio.run(load(path, async_session_factory, check=args.check))
    for line in report(validation, stats, check=args.check):
        print(line)
    return 0 if validation.ok else 1


if __name__ == "__main__":
    sys.exit(main())
