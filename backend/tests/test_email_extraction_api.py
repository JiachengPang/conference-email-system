"""`extraction` is served by the email API, with its NULL semantics intact.

The load-bearing distinction carried all the way from the DB layer:

  * ``extraction: null``              — the row was never examined (it predates
                                        the column). Nothing is known.
  * ``{"submission_numbers": [],      — the row WAS examined and no submission
     "method": "llm_distiller"}``       was found. That is a real finding.

Coercing the first into the second (or into ``{}``) at any layer destroys it,
so these tests assert through the REAL endpoints via ASGITransport rather than
calling ``_email_to_dict`` directly — a serializer can be correct while the
route around it is not.

The serving path does NOT go through the schemas.py mirror: `_email_to_dict`
forwards `email.extraction` (the raw JSON column) verbatim, and no route on this
router declares a `response_model`. The mirror is therefore a declared contract
that nothing enforces at runtime, which is exactly why the end-to-end tests at
the bottom of this file assert against real HTTP bytes and then validate those
bytes through the mirror, rather than trusting either one alone.

SCOPE LIMIT: reading only. No endpoint consumes or filters on `extraction`, so
nothing here asserts on querying or filtering by it.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import main
from app.core.config import settings
from app.db.database import get_db
from app.db.models import Base, Email
from app.pipeline.distiller import DistillResult
from app.pipeline.orchestrator import EmailPipeline

QUEUE = "/api/v1/emails/queue"
OPENREVIEW_QUEUE = "/api/v1/emails/queue/openreview"

# A fully populated extraction, exactly as the pipeline stores it.
_FULL = {
    "submission_numbers": ["22336", "44444"],
    "openreview_forum_ids": ["Ab3xY9kLm2", "Zz9QwErTy1"],
    "authors": [
        {
            "name": "Jane Roe",
            "email": "jane@example.edu",
            "affiliation": "Example University",
        },
        {"name": "John Doe", "email": None, "affiliation": None},
    ],
    "method": "llm_distiller",
}

# Examined, nothing found — every list empty but `method` says the LLM ran.
_EXAMINED_EMPTY = {
    "submission_numbers": [],
    "openreview_forum_ids": [],
    "authors": [],
    "method": "llm_distiller",
}


@pytest.fixture(autouse=True)
def _no_openreview_network(monkeypatch):
    """⚠️ THIS FILE WAS MAKING REAL NETWORK CALLS TO OPENREVIEW. Not theoretical:
    caught in captured logs as a 429 ``RateLimitError`` from
    ``api2.dev.openreview.net``, using the live credentials in the environment.

    The fixtures here seed genuine reply candidates (a note id AND a venue
    notification address — that pairing is the point of several tests), and
    ``GET /emails/{id}`` fetches the parent note's readers LIVE for exactly those
    rows. The lookup arrived with that endpoint's readers field and predates
    nothing in this file, so the two met without anyone noticing.

    Failing the client factory keeps it in-process: the helper catches broadly
    and reports its ``failed`` state, which no assertion here looks at.
    """
    from app.api.v1 import emails as emails_module

    def _no_client(*_args, **_kwargs):
        raise RuntimeError("OpenReview access is blocked in tests")

    monkeypatch.setattr(emails_module, "get_openreview_client", _no_client)


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with factory() as session:
            yield session

    async with factory() as session:
        session.add_all(
            [
                Email(
                    sender="jane@example.edu",
                    subject="full",
                    body="b",
                    status="DRAFT_GENERATED",
                    routing={"lane": "human_review"},
                    classification={"intent": "desk_reject_appeal", "confidence": 0.9},
                    draft={"draft_text": "Hello"},
                    extraction=_FULL,
                ),
                Email(
                    sender="legacy@example.edu",
                    subject="legacy-null",
                    body="b",
                    status="DRAFT_GENERATED",
                    routing={"lane": "human_review"},
                    # extraction deliberately unset -> NULL, as for every row
                    # processed before the column existed.
                ),
                Email(
                    sender="empty@example.edu",
                    subject="examined-empty",
                    body="b",
                    status="DRAFT_GENERATED",
                    routing={"lane": "human_review"},
                    extraction=_EXAMINED_EMPTY,
                ),
            ]
        )
        await session.commit()

    main.app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    main.app.dependency_overrides.clear()
    await engine.dispose()


async def _by_subject(client) -> dict[str, dict]:
    response = await client.get(QUEUE)
    assert response.status_code == 200, response.text
    return {e["subject"]: e for e in response.json()["emails"]}


async def _detail(client, subject: str) -> dict:
    rows = await _by_subject(client)
    response = await client.get(f"/api/v1/emails/{rows[subject]['id']}")
    assert response.status_code == 200, response.text
    return response.json()["email"]


# ---------------------------------------------------------------------------
# Fully populated extraction survives the wire
# ---------------------------------------------------------------------------
async def test_queue_serializes_a_full_extraction(client):
    row = (await _by_subject(client))["full"]
    assert row["extraction"] == _FULL


async def test_detail_serializes_a_full_extraction(client):
    assert (await _detail(client, "full"))["extraction"] == _FULL


async def test_authors_survive_as_a_list_of_objects(client):
    """Nested author objects must not be flattened or stringified in transit."""
    authors = (await _by_subject(client))["full"]["extraction"]["authors"]
    assert isinstance(authors, list) and len(authors) == 2
    assert authors[0] == {
        "name": "Jane Roe",
        "email": "jane@example.edu",
        "affiliation": "Example University",
    }
    # A partial mention keeps its nulls rather than dropping the keys.
    assert authors[1] == {"name": "John Doe", "email": None, "affiliation": None}


async def test_multiple_identifiers_survive_as_lists(client):
    """The list shape's point: several ids per email reach the client, in order.

    Single-element lists would survive even if a layer collapsed lists to
    scalars, so this fixture carries two of each.
    """
    extraction = (await _by_subject(client))["full"]["extraction"]
    assert extraction["submission_numbers"] == ["22336", "44444"]
    assert extraction["openreview_forum_ids"] == ["Ab3xY9kLm2", "Zz9QwErTy1"]


async def test_identifier_lists_are_json_arrays_not_strings(client):
    """A list must not be stringified anywhere on the way out."""
    extraction = (await _detail(client, "full"))["extraction"]
    assert isinstance(extraction["submission_numbers"], list)
    assert isinstance(extraction["openreview_forum_ids"], list)
    assert all(isinstance(v, str) for v in extraction["submission_numbers"])


async def test_method_reaches_the_client(client):
    """`method` is contract, not decoration — it separates a model answer from
    a regex guess, so a consumer cannot weigh the values without it."""
    assert (await _by_subject(client))["full"]["extraction"]["method"] == "llm_distiller"


# ---------------------------------------------------------------------------
# NULL vs examined-but-empty — the distinction must reach the client
# ---------------------------------------------------------------------------
async def test_null_extraction_serializes_as_null_not_an_empty_object(client):
    row = (await _by_subject(client))["legacy-null"]
    assert "extraction" in row, "the key must be present even when null"
    assert row["extraction"] is None
    assert row["extraction"] != {}


async def test_examined_but_empty_is_distinct_from_null(client):
    """THE distinction this piece exists to preserve.

    Both rows have no submission number, but only one of them was ever looked
    at. If some layer coerced null -> {} or {} -> null, these two assertions
    could not both hold.
    """
    rows = await _by_subject(client)
    never_looked = rows["legacy-null"]["extraction"]
    looked_found_nothing = rows["examined-empty"]["extraction"]

    assert never_looked is None
    assert looked_found_nothing is not None
    assert looked_found_nothing["submission_numbers"] == []
    assert looked_found_nothing["method"] == "llm_distiller"
    assert never_looked != looked_found_nothing


async def test_examined_but_empty_survives_the_detail_endpoint_too(client):
    extraction = (await _detail(client, "examined-empty"))["extraction"]
    assert extraction == _EXAMINED_EMPTY
    assert extraction is not None


async def test_null_extraction_survives_the_detail_endpoint_too(client):
    assert (await _detail(client, "legacy-null"))["extraction"] is None


# ---------------------------------------------------------------------------
# Purely additive
# ---------------------------------------------------------------------------
async def test_existing_pipeline_fields_are_unchanged(client):
    """classification/routing/draft pass through exactly as before."""
    row = (await _by_subject(client))["full"]
    assert row["classification"] == {
        "intent": "desk_reject_appeal",
        "confidence": 0.9,
    }
    assert row["routing"] == {"lane": "human_review"}
    assert row["draft"] == {"draft_text": "Hello"}


async def test_extraction_sits_beside_the_other_pipeline_outputs(client):
    row = (await _by_subject(client))["full"]
    for key in ("classification", "routing", "draft", "extraction"):
        assert key in row, key


async def test_schema_model_accepts_the_served_shape():
    """The schemas.py mirror must actually validate what the API serves.

    A mirror that has drifted from the wire format is worse than no mirror, so
    this pins them together at the one point that matters.

    SCOPE LIMIT — this test CANNOT catch a missing field, and did not: the
    mirror silently lacked three of them for three commits while this stayed
    green. Two reasons, both structural. ``model_validate`` ignores extra keys
    by default, so a fixture carrying a field the mirror lacks still validates;
    and ``_FULL`` is hand-written, so a field nobody remembered to add here is a
    field nobody remembered to add there either — the fixture drifts in lockstep
    with the thing it is supposed to police.

    The tests below close that gap by comparing the mirror against the PIPELINE
    MODEL ITSELF rather than against a hand-maintained dict. Put new
    drift-detection assertions there, not here.
    """
    from app.models.schemas import ExtractionResult

    parsed = ExtractionResult.model_validate(_FULL)
    assert parsed.submission_numbers == ["22336", "44444"]
    assert parsed.openreview_forum_ids == ["Ab3xY9kLm2", "Zz9QwErTy1"]
    assert parsed.method == "llm_distiller"
    assert parsed.authors[1].name == "John Doe"
    assert parsed.authors[1].email is None

    empty = ExtractionResult.model_validate(_EXAMINED_EMPTY)
    assert empty.submission_numbers == []
    assert empty.authors == []


# ---------------------------------------------------------------------------
# Mirror-vs-pipeline drift detection
#
# Every assertion here derives its expectation from the PIPELINE model at
# runtime, never from a literal written by hand. That is the point: a field
# added to one model and not the other fails these immediately, with no fixture
# to remember to update. The pipeline model is the authority; schemas.py mirrors
# it.
# ---------------------------------------------------------------------------
def _pipeline_result_with_every_field_populated():
    """A REAL extractor result, not a hand-built one.

    Driven through the actual regex path on a body carrying all four signals —
    a submission number, a forum+note link, and the venue notification address —
    so every field including the derived flag is non-default. A hand-built
    instance would only prove the models agree about values someone chose to
    type out.
    """
    from app.pipeline.extractor import EmailExtractor

    body = (
        "老师您好，请见下方邮件。\n\n"
        '发件人:"AAAI 2027" <aaai2027-notifications@openreview.net>\n'
        "主题: [AAAI 2027] SPC commented on submission 1030\n"
        "https://openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm"
    )
    result = EmailExtractor().extract("", body, "peng@iscas.ac.cn", "Peng", None)
    # Guard the fixture itself: if the extractor ever stops populating one of
    # these, the round-trip below would still pass while testing much less.
    assert result.submission_numbers and result.openreview_forum_ids
    assert result.openreview_note_id and result.openreview_notification_sender
    assert result.authors and result.openreview_reply_candidate is True
    # Non-empty on this fixture: the quoted header block starts below the
    # greeting, so the greeting is the reply. A fixture whose reply text went
    # empty would still satisfy the round-trip (both sides would carry "") while
    # proving nothing about the field, so it is guarded like the rest.
    assert result.extracted_reply_text
    return result


async def test_wire_mirror_serializes_exactly_the_pipeline_fields():
    """THE drift guard: identical serialized key sets, both directions.

    Catches a field added to the pipeline and forgotten on the wire (the bug
    this commit fixes) AND the reverse, a wire field with nothing behind it.
    Computed fields are included in ``model_dump``, so the derived flag is
    covered here too.
    """
    from app.models.schemas import ExtractionResult as Wire
    from app.pipeline.extractor import ExtractionResult as Pipeline

    pipeline_keys = set(Pipeline().model_dump())
    wire_keys = set(Wire().model_dump())

    assert pipeline_keys - wire_keys == set(), (
        f"schemas.py mirror is MISSING: {sorted(pipeline_keys - wire_keys)}"
    )
    assert wire_keys - pipeline_keys == set(), (
        f"schemas.py mirror has EXTRA fields: {sorted(wire_keys - pipeline_keys)}"
    )


async def test_wire_mirror_agrees_about_which_fields_are_nullable():
    """Key parity is not enough — the two must agree on OPTIONALITY too.

    Added after a mutation exposed a real gap: changing the mirror's
    `extracted_reply_text` from `str` to `str | None` left every existing guard
    green. Key sets matched, and a populated value still round-tripped, because
    nothing ever fed a null. Yet the mutated mirror ACCEPTED a document with
    `extracted_reply_text: null` that the pipeline model REJECTS — a mirror able
    to represent something its source cannot, which is exactly the drift this
    file exists to catch.

    Compares whether each shared field's annotation admits None, which
    deliberately ignores the nested-class difference on `authors`
    (`list[extractor.AuthorMention]` vs `list[schemas.AuthorMention]`): that is
    the intended mirror convention, and neither side is nullable.
    """
    from typing import get_args

    from app.models.schemas import ExtractionResult as Wire
    from app.pipeline.extractor import ExtractionResult as Pipeline

    def admits_none(annotation) -> bool:
        return type(None) in get_args(annotation)

    shared = set(Pipeline.model_fields) & set(Wire.model_fields)
    assert shared, "the two models share no fields at all"

    for name in sorted(shared):
        pipeline_optional = admits_none(Pipeline.model_fields[name].annotation)
        wire_optional = admits_none(Wire.model_fields[name].annotation)
        assert pipeline_optional == wire_optional, (
            f"`{name}` nullability differs: pipeline optional="
            f"{pipeline_optional}, wire optional={wire_optional}"
        )


async def test_wire_mirror_author_mention_matches_the_pipeline():
    """The nested model is part of the contract too."""
    from app.models.schemas import AuthorMention as Wire
    from app.pipeline.extractor import AuthorMention as Pipeline

    assert set(Wire().model_dump()) == set(Pipeline().model_dump())


async def test_wire_mirror_round_trips_a_real_populated_result():
    """Serialize the pipeline model, validate through the mirror, compare dumps.

    Equality of the two dumps is what makes this a real check rather than a
    smoke test: a field the mirror lacks is simply absent from its dump, so the
    comparison fails loudly instead of being silently tolerated the way bare
    ``model_validate`` tolerates it.
    """
    from app.models.schemas import ExtractionResult as Wire

    pipeline_result = _pipeline_result_with_every_field_populated()
    served = pipeline_result.model_dump()

    assert Wire.model_validate(served).model_dump() == served


async def test_wire_mirror_round_trips_the_examined_but_empty_shape():
    """The other documented state — examined, nothing found — must survive too."""
    from app.models.schemas import ExtractionResult as Wire
    from app.pipeline.extractor import ExtractionResult as Pipeline

    served = Pipeline(method="llm_distiller").model_dump()
    assert Wire.model_validate(served).model_dump() == served


async def test_wire_mirror_carries_the_openreview_reply_signals_by_value():
    """Not just present as keys — carrying the right VALUES.

    A mirror could satisfy the key-set test with fields of the wrong type or a
    default that swallows the value, so the scalars are read back explicitly.
    """
    from app.models.schemas import ExtractionResult as Wire

    parsed = Wire.model_validate(_pipeline_result_with_every_field_populated().model_dump())
    assert parsed.openreview_note_id == "jnHgRMHgrm"
    assert parsed.openreview_notification_sender == "aaai2027-notifications@openreview.net"
    assert parsed.openreview_reply_candidate is True
    # Carried by VALUE, not merely present as a key — a mirror field of the
    # wrong type or with a swallowing default would pass the key-set test.
    assert parsed.extracted_reply_text == "老师您好，请见下方邮件。"


async def test_wire_mirror_derives_reply_candidate_identically():
    """The AND expression is DUPLICATED across the two models, so it is its own
    drift surface — covered by comparison rather than by inspection.

    Both are asserted against the same operands over the full truth table, so
    changing one and not the other fails here.
    """
    from app.models.schemas import ExtractionResult as Wire
    from app.pipeline.extractor import ExtractionResult as Pipeline

    for note in ("jnHgRMHgrm", None):
        for sender in ("aaai2027-notifications@openreview.net", None):
            kwargs = {
                "openreview_note_id": note,
                "openreview_notification_sender": sender,
            }
            expected = note is not None and sender is not None
            assert Pipeline(**kwargs).openreview_reply_candidate is expected
            assert Wire(**kwargs).openreview_reply_candidate is expected, kwargs


async def test_wire_mirror_reply_candidate_is_derived_not_stored():
    """Mirrored as DERIVED, matching the pipeline model.

    A plain stored field here could be handed a value contradicting the two
    fields beside it, letting the wire model represent a state the pipeline can
    never produce — which is precisely the drift this class is supposed to
    prevent. So a corrupt persisted value must be recomputed, not echoed.
    """
    from app.models.schemas import ExtractionResult as Wire

    corrupt = {
        "submission_numbers": [],
        "openreview_forum_ids": [],
        "openreview_note_id": None,
        "openreview_notification_sender": None,
        "authors": [],
        "method": "regex_fallback",
        "openreview_reply_candidate": True,
    }
    assert Wire.model_validate(corrupt).openreview_reply_candidate is False
    assert Wire(openreview_reply_candidate=True).openreview_reply_candidate is False


# ---------------------------------------------------------------------------
# Pipeline -> DB -> HTTP, for all three OpenReview reply fields together
#
# Prior commits proved the pieces separately: the extractor produces the fields,
# the JSON column stores them, and the schemas.py mirror declares them. None of
# that proves a client can actually READ them — the serving path could drop a
# field between the column and the wire and every earlier test would stay green.
# These drive the real pipeline, persist a real row, and read it back over HTTP.
# ---------------------------------------------------------------------------
_ADDRESS = "aaai2027-notifications@openreview.net"
_LINK = "https://openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm"

# The real reported shape: a Chinese-labelled quoted notification header, plus
# the forum link carrying the noteId. Carries all three signals at once.
_REAL_BODY = (
    "老师您好，请见下方邮件。\n\n"
    f'发件人:"AAAI 2027" <{_ADDRESS}>\n'
    "发送时间:2026-08-27 15:28:28 (星期四)\n"
    "收件人: pengshaohui@iscas.ac.cn\n"
    "主题: [AAAI 2027] Senior Program Committee 6UDQ commented on a paper you "
    "are reviewing. Paper Number: 1030\n"
    f"{_LINK}"
)


class _StubRetriever:
    async def retrieve(self, query, intent, top_k=3, *, prior_intent=""):
        return []


class _StubDistiller:
    """Stands in for the model, reporting only what its prompt asks for.

    ``openreview_ids_raw`` is supplied because the prompt DOES ask for forum
    ids; the note id and the sender address are not in its contract and are read
    from the raw text by the extractor. Reporting the forum id also satisfies
    the note-id coherence gate, which on this path checks the MODEL's list.
    """

    def __init__(self, result):
        self.result = result

    async def distill(self, subject, body, *, transcript=None):
        return self.result


@pytest_asyncio.fixture
async def pipeline_client():
    """A client and a session factory sharing ONE in-memory database.

    The `client` fixture above seeds rows by hand; this one hands back the
    factory as well, so a test can drive the real pipeline into the same DB the
    endpoints read from. Without the shared factory the row would be written to
    a different database than the request reads.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with factory() as session:
            yield session

    main.app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, factory
    main.app.dependency_overrides.clear()
    await engine.dispose()


async def _process_real_body(factory, monkeypatch) -> str:
    """Run the REAL pipeline on the real body, production-style, return its id.

    ``QUERY_STRATEGY=distill`` matches production, so this exercises the LLM
    path — the one where all three of these fields were at some point unreachable.
    """
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    pipeline = EmailPipeline()
    pipeline.retriever = _StubRetriever()
    pipeline.distiller = _StubDistiller(
        DistillResult(
            queries=["official comment reply"],
            intent="cms_support",
            confidence=0.9,
            openreview_ids_raw=["ll0avn6ylq"],
        )
    )
    async with factory() as session:
        result = await pipeline.process_email(
            {
                "from": "pengshaohui@iscas.ac.cn",
                "sender_name": "Peng",
                "subject": "Re: [AAAI 2027] SPC commented on a paper",
                "body": _REAL_BODY,
            },
            session,
        )
    return result.email_id


async def test_all_three_openreview_fields_reach_the_detail_endpoint(
    pipeline_client, monkeypatch
):
    """THE end-to-end check: real pipeline -> real row -> real HTTP response.

    Asserted on the parsed HTTP JSON, not on the ORM row or an intermediate
    object, because everything between the column and the wire is exactly what
    has never been covered before.
    """
    client, factory = pipeline_client
    email_id = await _process_real_body(factory, monkeypatch)

    response = await client.get(f"/api/v1/emails/{email_id}")
    assert response.status_code == 200

    extraction = response.json()["email"]["extraction"]
    assert extraction is not None, "extraction missing from the HTTP response"

    # The three fields this workstream added, all present and correctly valued.
    assert extraction["openreview_note_id"] == "jnHgRMHgrm"
    assert extraction["openreview_notification_sender"] == _ADDRESS
    assert extraction["openreview_reply_candidate"] is True

    # ...and the production path really was exercised, so this is not a regex
    # fallback quietly standing in for the distill path.
    assert extraction["method"] == "llm_distiller"


async def test_all_three_openreview_fields_reach_the_openreview_queue_endpoint(
    pipeline_client, monkeypatch
):
    """The list endpoint is a SEPARATE serialization call site from the detail
    view, so it gets its own assertion rather than being assumed to match.

    Points at ``/queue/openreview`` rather than ``/queue``, and the RENAME is the
    substance of the change, not cosmetic: this fixture is a real, unresolved
    OpenReview candidate, so the queue split now routes it here. The subject
    under test is unchanged — that a LIST endpoint serializes all three fields —
    only the list it lives in moved. Both halves of that move are asserted below,
    so this now also guards the split itself from the serialization side.
    """
    client, factory = pipeline_client
    email_id = await _process_real_body(factory, monkeypatch)

    response = await client.get(OPENREVIEW_QUEUE)
    assert response.status_code == 200

    rows = [r for r in response.json()["emails"] if str(r["id"]) == str(email_id)]
    assert len(rows) == 1, "the processed email is not in the OpenReview queue"
    extraction = rows[0]["extraction"]

    assert extraction["openreview_note_id"] == "jnHgRMHgrm"
    assert extraction["openreview_notification_sender"] == _ADDRESS
    assert extraction["openreview_reply_candidate"] is True

    # ...and it is NOT in the main queue, which is the whole point of the split.
    main = await client.get(QUEUE)
    assert [r for r in main.json()["emails"] if str(r["id"]) == str(email_id)] == []


async def test_served_extraction_validates_through_the_schemas_mirror(
    pipeline_client, monkeypatch
):
    """Ties commit 4a's mirror to the ACTUAL wire bytes.

    The mirror is not on the serving path (see the module note below), so it can
    only be trusted against real served output rather than against a fixture.
    Round-tripping what the endpoint really sent proves the two agree about the
    shape a client receives.
    """
    from app.models.schemas import ExtractionResult as Wire

    client, factory = pipeline_client
    email_id = await _process_real_body(factory, monkeypatch)

    served = (await client.get(f"/api/v1/emails/{email_id}")).json()["email"]["extraction"]
    parsed = Wire.model_validate(served)

    assert parsed.openreview_note_id == "jnHgRMHgrm"
    assert parsed.openreview_notification_sender == _ADDRESS
    assert parsed.openreview_reply_candidate is True
    # No key the endpoint sent is unknown to the mirror, and none it declares is
    # missing from the wire — extra-key tolerance would hide both directions.
    assert set(parsed.model_dump()) == set(served)



# --- is_reciprocal_dispute: present on both sides, written by nothing --------
#
# The field was added to the wire mirror AND the pipeline model in one commit,
# deliberately. The mirror's drift guard asserts key parity in BOTH directions,
# so a field landing on one side alone is red -- these two models cannot be
# changed in separate commits, and that is the guard working, not a nuisance.


async def test_wire_mirror_accepts_all_three_states():
    """bool | None is a genuine tri-state on the wire, not a bool with a hole."""
    from app.models.schemas import ExtractionResult as Wire

    for value in (True, False, None):
        parsed = Wire.model_validate({"is_reciprocal_dispute": value})
        assert parsed.is_reciprocal_dispute is value, value
        # Round-trips as itself -- False must not be serialized away as absent.
        assert parsed.model_dump()["is_reciprocal_dispute"] is value, value


async def test_wire_mirror_validates_a_legacy_row_without_the_key():
    """Every row persisted before this field lacks the key entirely.

    Those rows must keep validating, and must read as None -- "never asked" --
    rather than being coerced to False, which would be a positive claim the
    stored data never made.
    """
    from app.models.schemas import ExtractionResult as Wire

    legacy = {
        "submission_numbers": ["12345"],
        "openreview_forum_ids": ["Ab3xY9kLm2"],
        "openreview_note_id": None,
        "openreview_notification_sender": None,
        "extracted_reply_text": "Please reconsider.",
        "authors": [{"name": "A. Person", "email": None, "affiliation": None}],
        "method": "regex_fallback",
    }
    assert "is_reciprocal_dispute" not in legacy

    parsed = Wire.model_validate(legacy)
    assert parsed.is_reciprocal_dispute is None
    # The absent key does not disturb anything that WAS stored.
    assert parsed.submission_numbers == ["12345"]
    assert parsed.method == "regex_fallback"
    # ...and it is emitted on the way out, so a consumer sees a uniform shape.
    assert "is_reciprocal_dispute" in parsed.model_dump()


async def test_pipeline_model_defaults_to_none():
    from app.pipeline.extractor import ExtractionResult as Pipeline

    assert Pipeline().is_reciprocal_dispute is None


_DISPUTE_SUBJECT = "Appeal: rejection over reciprocal review duty"
_DISPUTE_BODY = (
    "My paper 12345 was rejected because my reciprocal reviewers did not "
    "submit their reviews. Please reconsider."
)


def _distilled():
    from app.pipeline.distiller import DistillResult

    return DistillResult(
        queries=["desk rejection appeal"],
        intent="desk_reject_appeal",
        confidence=0.9,
        submission_numbers_raw=["12345"],
    )


async def test_extractor_never_sets_the_flag_on_the_distiller_path():
    """REPLACES a pass-through test, because the pass-through is gone.

    It used to assert all three tri-state values survived distiller ->
    extractor verbatim. The question moved to its own conditional call
    (`reciprocal_detector`), so `DistillResult` no longer carries the field and
    `extract()` must leave it None even on a fully-populated LLM result —
    `orchestrator._compute` attaches the answer afterwards.

    Asserting it HERE, on the LLM path, is the discriminating case: the regex
    path never had access to `distilled` at all, so only this branch could have
    re-introduced the coupling.
    """
    from app.pipeline.extractor import EmailExtractor

    result = EmailExtractor().extract(
        _DISPUTE_SUBJECT, _DISPUTE_BODY, "", None, _distilled()
    )
    assert result.method == "llm_distiller"
    assert result.is_reciprocal_dispute is None


async def test_extractor_never_sets_the_flag_on_the_regex_path():
    """The other branch, for completeness — None on BOTH paths, always."""
    from app.pipeline.extractor import EmailExtractor

    result = EmailExtractor().extract(_DISPUTE_SUBJECT, _DISPUTE_BODY, "", None, None)
    assert result.is_reciprocal_dispute is None


async def test_regex_path_never_answers_the_flag():
    """Regex must not guess at a judgment, and structurally cannot.

    The body here is a textbook reciprocal-duty appeal -- if any regex were ever
    added that tried to answer this field, this is the text it would fire on.
    """
    from app.pipeline.extractor import EmailExtractor

    result = EmailExtractor().extract(_DISPUTE_SUBJECT, _DISPUTE_BODY, "", None, None)
    assert result.method == "regex_fallback"
    assert result.is_reciprocal_dispute is None


async def test_regex_helper_cannot_see_a_distill_result_at_all():
    """"Somehow present" is unrepresentable: the helper takes no `distilled`.

    Asserted on the SIGNATURE rather than by passing a value, because there is
    no way to pass one -- which is the actual guarantee. A future refactor that
    threads `distilled` into the fallback would make regex able to answer this
    field, and that is what this pins against.
    """
    import inspect

    from app.pipeline.extractor import EmailExtractor

    params = inspect.signature(EmailExtractor._extract_by_regex).parameters
    assert "distilled" not in params, (
        "the regex fallback must not be able to see the distiller's answer"
    )


async def test_failed_extraction_reports_no_flag():
    """The never-raises path yields `method="none"` and a None flag.

    Its original framing — "must not leak the True that was on its way in" —
    no longer applies: `DistillResult` cannot carry a True any more, since the
    question moved to its own call. The assertion is still worth keeping as the
    empty-result contract for the crash path, which is why this was narrowed
    rather than deleted.
    """
    from unittest.mock import patch

    from app.pipeline.extractor import EmailExtractor

    with patch(
        "app.pipeline.extractor._dedupe_identifiers", side_effect=RuntimeError("boom")
    ):
        result = EmailExtractor().extract(
            _DISPUTE_SUBJECT, _DISPUTE_BODY, "", None, _distilled()
        )

    assert result.method == "none"
    assert result.is_reciprocal_dispute is None
