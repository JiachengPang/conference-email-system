"""Chair Zendesk accounts: the table, its migration, the repository and the loader.

Synthetic data only. Repository tests run on in-memory SQLite, and on Postgres
too when TEST_DATABASE_URL points at a DISPOSABLE postgresql:// database (never
the local one, which holds real tickets). The loader's private file is always a
temp file outside the repo.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base, PaperAssignment, ZendeskChairAccount
from app.repositories.chair_account_repository import (
    ChairAccountRepository,
    ChairAccountRow,
    UpsertStats,
)
from scripts import load_chair_accounts as loader

BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REVISION = "3b7e9d2a5c41"
_PREV_REVISION = "8d2f6c1a9b3e"

_PG = os.environ.get("TEST_DATABASE_URL", "")
_PG = _PG if _PG.startswith("postgresql") else None

# Synthetic sheet chairs (one with edge whitespace in the sheet, as real exports have).
SHEET = {"10001": "Chair Alpha", "10002": "Chair Beta", "10003": "  Chair Gamma ", "10004": "Chair Beta"}
SECRET_EMAIL = "secret.person@example.org"
SECRET_ID = "987654321"


def _engine(kind: str):
    if kind == "postgres":
        url = _PG if "+asyncpg" in _PG else _PG.replace("postgresql://", "postgresql+asyncpg://", 1)
        return create_async_engine(url)
    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def factory(request):
    if request.param == "postgres" and not _PG:
        pytest.skip("set TEST_DATABASE_URL to a disposable postgresql:// database")
    engine = _engine(request.param)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    made = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with made() as db:
        db.add_all(
            PaperAssignment(paper_number=n, apc_name=name, openreview_url="u")
            for n, name in SHEET.items()
        )
        await db.commit()
    yield made
    if request.param == "postgres":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def session(factory):
    async with factory() as s:
        yield s


async def _rows(session) -> list[tuple]:
    result = await session.execute(
        select(ZendeskChairAccount.chair_name, ZendeskChairAccount.zendesk_user_id,
               ZendeskChairAccount.active).order_by(ZendeskChairAccount.chair_name)
    )
    return [tuple(r) for r in result.all()]


# --- repository -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_inserts_then_is_idempotent(session):
    repo = ChairAccountRepository()
    rows = [ChairAccountRow("Chair Alpha", 11, True), ChairAccountRow("Chair Beta", 22, False)]
    assert await repo.upsert_many(session, rows) == UpsertStats(inserted=2)
    first = await _rows(session)
    stamps = {a.chair_name: a.updated_at for a in await repo.list_all(session)}

    assert await repo.upsert_many(session, rows) == UpsertStats(unchanged=2)
    assert await _rows(session) == first == [("Chair Alpha", 11, True), ("Chair Beta", 22, False)]
    assert {a.chair_name: a.updated_at for a in await repo.list_all(session)} == stamps


@pytest.mark.asyncio
async def test_upsert_updates_changed_rows_and_leaves_others_alone(session):
    repo = ChairAccountRepository()
    await repo.upsert_many(session, [ChairAccountRow("Chair Alpha", 11), ChairAccountRow("Chair Beta", 22)])
    stats = await repo.upsert_many(session, [ChairAccountRow("Chair Alpha", 33, False)])
    assert stats == UpsertStats(updated=1)
    assert await _rows(session) == [("Chair Alpha", 33, False), ("Chair Beta", 22, True)]


@pytest.mark.asyncio
async def test_get_by_name_is_exact(session):
    repo = ChairAccountRepository()
    await repo.upsert_many(session, [ChairAccountRow("Chair Alpha", 11)])
    assert (await repo.get_by_name(session, "Chair Alpha")).zendesk_user_id == 11
    for miss in ("chair alpha", " Chair Alpha", "Nobody", "", None):
        assert await repo.get_by_name(session, miss) is None


@pytest.mark.asyncio
async def test_the_table_refuses_a_duplicate_name_and_a_non_positive_id(session):
    session.add(ZendeskChairAccount(chair_name="Chair Alpha", zendesk_user_id=1))
    await session.commit()
    session.add(ZendeskChairAccount(chair_name="Chair Alpha", zendesk_user_id=2))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
    session.add(ZendeskChairAccount(chair_name="Chair Zero", zendesk_user_id=0))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


@pytest.mark.asyncio
async def test_active_defaults_to_true_in_the_database(session):
    from sqlalchemy import text

    await session.execute(
        text("INSERT INTO zendesk_chair_accounts (chair_name, zendesk_user_id) VALUES ('Chair X', 5)")
    )
    await session.commit()
    assert await _rows(session) == [("Chair X", 5, True)]


# --- loader: path guard ----------------------------------------------------------------------


def _write(tmp_path: Path, text: str, name: str = "accounts.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_no_path_and_no_env_is_refused():
    with pytest.raises(loader.LoaderUsageError, match="CHAIR_ACCOUNTS_FILE"):
        loader.resolve_path(None, environ={})


def test_the_env_var_is_used_when_no_argument_is_given(tmp_path):
    path = _write(tmp_path, "chair_name,zendesk_user_id\n")
    assert loader.resolve_path(None, environ={"CHAIR_ACCOUNTS_FILE": str(path)}) == path.resolve()


def test_a_path_inside_the_repo_is_refused():
    inside = loader.REPO_ROOT / "backend" / "pyproject.toml"
    with pytest.raises(loader.LoaderUsageError, match="inside the repo"):
        loader.resolve_path(str(inside), environ={})
    with pytest.raises(loader.LoaderUsageError, match="inside the repo"):
        loader.resolve_path(None, environ={"CHAIR_ACCOUNTS_FILE": str(inside)})


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(loader.LoaderUsageError, match="No file"):
        loader.resolve_path(str(tmp_path / "absent.csv"), environ={})


def test_main_exits_2_without_a_path(capsys, monkeypatch):
    monkeypatch.delenv("CHAIR_ACCOUNTS_FILE", raising=False)
    assert loader.main([]) == 2
    assert "CHAIR_ACCOUNTS_FILE" in capsys.readouterr().out


# --- loader: validation (pure) ---------------------------------------------------------------

KNOWN = ["Chair Alpha", "Chair Beta", "Chair Gamma"]


def _validate(text: str, tmp_path: Path, known=KNOWN):
    rows, missing = loader.read_rows(_write(tmp_path, text))
    return loader.validate(rows, missing, known)


def test_a_clean_file_validates_and_extra_columns_are_ignored(tmp_path):
    v = _validate(
        f"chair_name,zendesk_user_id,active,email\n"
        f"Chair Alpha,11,,{SECRET_EMAIL}\n Chair Beta , 22 ,no,x@example.org\n",
        tmp_path,
    )
    assert v.ok and v.error_count == 0
    assert v.rows == [ChairAccountRow("Chair Alpha", 11, True), ChairAccountRow("Chair Beta", 22, False)]
    assert v.missing_names == ["Chair Gamma"]
    assert v.unknown_names == []


@pytest.mark.parametrize("value, active", [("", True), ("TRUE", True), ("yes", True), ("1", True),
                                           ("false", False), ("No", False), ("0", False)])
def test_active_values(tmp_path, value, active):
    v = _validate(f"chair_name,zendesk_user_id,active\nChair Alpha,11,{value}\n", tmp_path)
    assert v.ok and v.rows == [ChairAccountRow("Chair Alpha", 11, active)]


@pytest.mark.parametrize("text, field, count", [
    ("chair_name,zendesk_user_id\nChair Alpha,11\nChair Alpha,12\n", "duplicate_names", 1),
    ("chair_name,zendesk_user_id\nChair Alpha,\n", "empty_user_ids", 1),
    ("chair_name,zendesk_user_id\nChair Alpha,  \n", "empty_user_ids", 1),
    ("chair_name,zendesk_user_id\nChair Alpha,12a\n", "bad_user_ids", 1),
    ("chair_name,zendesk_user_id\nChair Alpha,-5\n", "bad_user_ids", 1),
    ("chair_name,zendesk_user_id\nChair Alpha,0\n", "bad_user_ids", 1),
    ("chair_name,zendesk_user_id\nChair Alpha,²\n", "bad_user_ids", 1),
    ("chair_name,zendesk_user_id,active\nChair Alpha,11,maybe\n", "bad_active_values", 1),
    ("chair_name,zendesk_user_id\n,11\n", "empty_names", 1),
], ids=["duplicate", "empty-id", "blank-id", "letters", "negative", "zero", "superscript",
        "bad-active", "empty-name"])
def test_each_error_refuses_the_whole_file(tmp_path, text, field, count):
    v = _validate(text, tmp_path)
    assert getattr(v, field) == count
    assert not v.ok and v.rows == []


def test_unknown_names_are_errors_and_listed(tmp_path):
    v = _validate("chair_name,zendesk_user_id\nChair Alpha,11\nChair Delta,44\nchair beta,22\n", tmp_path)
    assert v.unknown_names == ["chair beta", "Chair Delta"]  # exact case: "chair beta" is unknown
    assert not v.ok and v.rows == []
    assert v.missing_names == ["Chair Beta", "Chair Gamma"]


def test_a_sheet_name_with_edge_whitespace_matches_trimmed(tmp_path):
    v = _validate("chair_name,zendesk_user_id\nChair Gamma,33\n", tmp_path, known=["  Chair Gamma "])
    assert v.ok and v.rows == [ChairAccountRow("Chair Gamma", 33, True)]


def test_missing_columns_are_errors(tmp_path):
    v = _validate("name,id\nChair Alpha,11\n", tmp_path)
    assert v.missing_columns == ["chair_name", "zendesk_user_id"]
    assert not v.ok and v.rows == []


def test_an_empty_sheet_is_an_error(tmp_path):
    v = _validate("chair_name,zendesk_user_id\nChair Alpha,11\n", tmp_path, known=[])
    assert v.sheet_empty and not v.ok and v.rows == []


def test_a_shared_user_id_is_counted_but_not_an_error(tmp_path):
    v = _validate("chair_name,zendesk_user_id\nChair Alpha,11\nChair Beta,11\n", tmp_path)
    assert v.shared_user_ids == 1 and v.ok


# --- loader: end to end against the database --------------------------------------------------


@pytest.mark.asyncio
async def test_load_writes_valid_rows_and_a_second_load_changes_nothing(factory, tmp_path):
    path = _write(tmp_path, "chair_name,zendesk_user_id,active\nChair Alpha,11,\nChair Gamma,33,false\n")
    validation, stats = await loader.load(path, factory)
    assert validation.ok and stats == UpsertStats(inserted=2)
    validation, stats = await loader.load(path, factory)
    assert validation.ok and stats == UpsertStats(unchanged=2)
    async with factory() as db:
        assert await _rows(db) == [("Chair Alpha", 11, True), ("Chair Gamma", 33, False)]


@pytest.mark.asyncio
async def test_an_invalid_file_writes_nothing(factory, tmp_path):
    path = _write(tmp_path, "chair_name,zendesk_user_id\nChair Alpha,11\nChair Nobody,22\n")
    validation, stats = await loader.load(path, factory)
    assert not validation.ok and stats is None
    async with factory() as db:
        assert await _rows(db) == []


@pytest.mark.asyncio
async def test_check_mode_writes_nothing(factory, tmp_path):
    path = _write(tmp_path, "chair_name,zendesk_user_id\nChair Alpha,11\n")
    validation, stats = await loader.load(path, factory, check=True)
    assert validation.ok and stats is None
    async with factory() as db:
        assert await _rows(db) == []


@pytest.mark.asyncio
async def test_the_report_prints_names_and_counts_but_never_ids_or_emails(factory, tmp_path, capsys):
    path = _write(
        tmp_path,
        "chair_name,zendesk_user_id,email\n"
        f"Chair Alpha,{SECRET_ID},{SECRET_EMAIL}\n"
        f"Chair Nobody,{SECRET_ID}1,{SECRET_EMAIL}\n"
        f"Chair Beta,,{SECRET_EMAIL}\n",
    )
    validation, stats = await loader.load(path, factory)
    for line in loader.report(validation, stats, check=False):
        print(line)
    out = capsys.readouterr().out
    assert "Unknown names: Chair Nobody" in out
    assert "Missing names: Chair Gamma" in out
    assert "Nothing written" in out
    assert SECRET_ID not in out and SECRET_EMAIL not in out and "@" not in out


@pytest.mark.asyncio
async def test_the_success_report_has_counts_only(factory, tmp_path):
    path = _write(tmp_path, f"chair_name,zendesk_user_id\nChair Alpha,{SECRET_ID}\n")
    validation, stats = await loader.load(path, factory)
    text = "\n".join(loader.report(validation, stats, check=False))
    assert "Written: 1 inserted, 0 updated, 0 unchanged." in text
    assert SECRET_ID not in text


# --- the migration ---------------------------------------------------------------------------


def _run_alembic(args, db_url: str):
    env = {**os.environ, "DATABASE_URL": db_url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(BACKEND_ROOT), env=env, capture_output=True, text=True,
    )


def _tables(db_file) -> set[str]:
    con = sqlite3.connect(db_file)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def test_migration_is_the_single_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(Config(str(BACKEND_ROOT / "alembic.ini")))
    assert script.get_heads() == [_REVISION]
    assert script.get_revision(_REVISION).down_revision == _PREV_REVISION


def test_migration_upgrade_downgrade_and_re_upgrade(tmp_path):
    """Runs against a throwaway SQLite file, never a real database."""
    db_file = tmp_path / "chair_accounts.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    up = _run_alembic(["upgrade", _REVISION], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}"
    con = sqlite3.connect(db_file)
    try:
        columns = {r[1] for r in con.execute("PRAGMA table_info(zendesk_chair_accounts)")}
        assert columns == {c.name for c in ZendeskChairAccount.__table__.columns}
        con.execute("INSERT INTO zendesk_chair_accounts (chair_name, zendesk_user_id) VALUES ('A', 1)")
        con.commit()
        assert con.execute("SELECT active FROM zendesk_chair_accounts").fetchone() == (1,)
        for bad in ("INSERT INTO zendesk_chair_accounts (chair_name, zendesk_user_id) VALUES ('A', 2)",
                    "INSERT INTO zendesk_chair_accounts (chair_name, zendesk_user_id) VALUES ('B', 0)"):
            with pytest.raises(sqlite3.IntegrityError):
                con.execute(bad)
    finally:
        con.close()

    down = _run_alembic(["downgrade", _PREV_REVISION], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}"
    tables = _tables(db_file)
    assert "zendesk_chair_accounts" not in tables
    assert {"emails", "paper_assignments", "zendesk_chair_notes"} <= tables

    again = _run_alembic(["upgrade", _REVISION], db_url)
    assert again.returncode == 0, f"re-upgrade failed:\n{again.stderr}"
    assert "zendesk_chair_accounts" in _tables(db_file)


def test_model_columns():
    assert [c.name for c in ZendeskChairAccount.__table__.columns] == [
        "id", "chair_name", "zendesk_user_id", "active", "created_at", "updated_at",
    ]
    assert ZendeskChairAccount.__tablename__ == "zendesk_chair_accounts"
