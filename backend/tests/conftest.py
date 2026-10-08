"""Shared pytest fixtures.

Tests run with no real database and no Anthropic API calls. ``mock_db_session``
stands in for an AsyncSession; the sample-data fixtures provide realistic,
self-contained inputs so each test stays independent.
"""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.pipeline.classifier import ClassificationResult
from app.pipeline.retriever import RetrievedChunk


@pytest.fixture(autouse=True)
def _hermetic_model_settings(monkeypatch):
    """Tests never call a hosted model, whatever the developer's backend/.env
    says: force the deterministic fallback provider, drop the API key, and
    keep the legacy prefix query so the distiller stays out of the pipeline.
    Individual tests opt back in explicitly (monkeypatch or direct
    construction, e.g. ResponseDrafter(provider="local") with mocked httpx).
    """
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "fallback")
    monkeypatch.setattr(settings, "LOCAL_MODEL_API_KEY", None)
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "prefix")
    monkeypatch.setattr(settings, "WARM_RETRIEVER_ON_STARTUP", False)
    # Keep retrieval on BM25 in tests: the default is now "fusion", whose
    # rebuild_index() loads the dense embedding model — tests must stay offline
    # and fast (endpoint tests trigger rebuilds via the policies mutations).
    monkeypatch.setattr(settings, "RETRIEVAL_BACKEND", "bm25")


def pytest_configure(config):
    # Registered here rather than in pyproject.toml: pyproject is the manifest
    # the backend image installs from, and editing it rebuilds every dependency.
    config.addinivalue_line(
        "markers",
        "zendesk_transport: exercises the real ZendeskSender.add_comment / "
        "get_ticket_state / assign_with_note against a fake HTTP client "
        "(opts out of the _hermetic_chair_notes guard)",
    )


@pytest.fixture(autouse=True)
def _hermetic_chair_notes(request, monkeypatch):
    """Chair notes (Z2) stay off in every test, and no test can post a Zendesk
    comment by accident: ``ZendeskSender.add_comment`` is replaced by a stub that
    raises, whatever the developer's backend/.env says. Tests that exercise the
    real transport against a fake HTTP client opt out with the
    ``zendesk_transport`` marker; chair-note tests turn the flag on explicitly.
    """
    monkeypatch.setattr(settings, "CHAIR_NOTE_ENABLED", False)
    # The reject-appeal assignment write stays off too; its tests turn it on.
    monkeypatch.setattr(settings, "ZENDESK_APPEAL_WRITE_ENABLED", False)
    if request.node.get_closest_marker("zendesk_transport"):
        return
    from app.integrations.zendesk.sender import ZendeskSender

    def _refuse(name):
        async def _refuse_call(self, *args, **kwargs):
            raise AssertionError(
                f"ZendeskSender.{name} was called in a test. Stub the transport, "
                "or mark a transport test with @pytest.mark.zendesk_transport."
            )
        return _refuse_call

    # The assignment's read and write are guarded the same way as add_comment.
    for name in ("add_comment", "get_ticket_state", "assign_with_note"):
        monkeypatch.setattr(ZendeskSender, name, _refuse(name))


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """An AsyncMock standing in for an async SQLAlchemy session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def sample_email_dict() -> dict:
    """A realistic inbound email dict (pipeline / orchestrator input shape)."""
    return {
        "from": "author@university.edu",
        "to": "chairs@conference.org",
        "subject": "Deadline question",
        "body": "When is the paper submission deadline this year?",
        "timestamp": "2025-10-15T09:23:00Z",
    }


@pytest.fixture
def sample_classification_result() -> ClassificationResult:
    """A high-confidence deadline classification."""
    return ClassificationResult(
        intent="submission_requirements",
        confidence=0.85,
        reasoning="Matched deadline keywords.",
        secondary_intents=[],
    )


@pytest.fixture
def sample_retrieved_chunk() -> RetrievedChunk:
    """A single retrieved policy chunk with a positive relevance score."""
    return RetrievedChunk(
        policy_id="policy_002",
        title="Full Paper Submission Deadline",
        content="The full paper deadline is specified in Anywhere on Earth time.",
        score=5.0,
        category="submission_deadlines",
        tags=["deadline", "full-paper", "aoe"],
    )
