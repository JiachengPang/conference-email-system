"""Tests for the epsilon-greedy RL router and its analytics endpoint.

The bandit's state file is redirected to a per-test ``tmp_path`` (by monkeypatching
the class-level ``STATE_PATH``) and the module singleton is reset, so tests never
touch the repo's ``backend/models/`` and don't bleed state into each other.
"""

import random

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

import main
import app.pipeline.rl_router as rlmod
from app.pipeline.rl_router import RLRouter
from app.pipeline.router import RoutingDecision


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    """Point STATE_PATH at a tmp file and reset the singleton for isolation."""
    monkeypatch.setattr(RLRouter, "STATE_PATH", str(tmp_path / "rl_state.json"))
    monkeypatch.setattr(rlmod, "_INSTANCE", None)
    return tmp_path


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def test_route_returns_routing_decision_with_lane(isolated_state):
    router = RLRouter()
    decision = router.route("submission_requirements", 0.8, 0.65)
    assert isinstance(decision, RoutingDecision)
    assert decision.lane in {"faq", "human_review"}


def test_approved_feedback_increments_wins(isolated_state):
    router = RLRouter()
    router.record_feedback("submission_requirements", "auto_reply", "approved")
    stats = router.state["submission_requirements"]["auto_reply"]
    assert stats["wins"] == 1
    assert stats["trials"] == 1


def test_rerouted_feedback_increments_trials_not_wins(isolated_state):
    router = RLRouter()
    router.record_feedback("submission_requirements", "auto_reply", "rerouted")
    stats = router.state["submission_requirements"]["auto_reply"]
    assert stats["wins"] == 0
    assert stats["trials"] == 1


def test_exploits_auto_reply_after_repeated_approvals(isolated_state):
    random.seed(0)  # deterministic exploration
    router = RLRouter()
    for _ in range(10):
        router.record_feedback("submission_requirements", "auto_reply", "approved")

    # Over many routes, the high-win-rate arm (auto_reply → "faq") dominates.
    lanes = [router.route("submission_requirements", 0.8, 0.65).lane for _ in range(100)]
    faq = lanes.count("faq")
    assert faq > lanes.count("human_review")
    assert faq >= 70  # ~85% exploit + half of 15% exploration


def test_confidence_below_floor_forces_human_review(isolated_state):
    random.seed(0)
    router = RLRouter()
    # Even with auto_reply heavily rewarded, sub-0.4 confidence must escalate.
    for _ in range(20):
        router.record_feedback("submission_requirements", "auto_reply", "approved")
    for _ in range(20):
        decision = router.route("submission_requirements", 0.3, 0.65)
        assert decision.lane == "human_review"


@pytest.mark.parametrize("intent", ["desk_reject_appeal", "review_decision_appeal"])
def test_sensitive_intent_holds_on_the_RL_path_too(isolated_state, intent):
    """The hold is STRATEGY-INDEPENDENT — one constant, both routers.

    `rl_router` imports `SENSITIVE_INTENTS` from `router` and checks it BEFORE
    the bandit is consulted, so populating that list in commit 9c held this path
    with no change to this file. Worth an explicit test rather than trusting the
    import: the RL router returns from `EmailRouter.route` EARLY, before the
    rule-based sensitive-intent check, so it does not inherit that guard — it
    has its own, and only a test proves the two agree.

    Rewarding auto_reply 20× first means a bandit that ignored the hold would
    confidently choose the FAQ lane, so a pass here cannot be luck.

    Parametrized over both held appeal intents (D55 + D62).
    """
    random.seed(0)
    router = RLRouter()
    for _ in range(20):
        router.record_feedback(intent, "auto_reply", "approved")
    for _ in range(20):
        decision = router.route(intent, 0.99, 0.65)
        assert decision.lane == "human_review"
        assert decision.override_reason == (
            f"Intent '{intent}' always requires human review"
        )


async def test_rl_stats_endpoint_returns_200_with_strategy_key(client):
    resp = await client.get("/api/v1/analytics/rl-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "routing_strategy" in body
    assert "stats" in body
