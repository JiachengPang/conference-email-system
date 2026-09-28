"""RL routing layer — an epsilon-greedy multi-armed bandit over routing lanes.

This is an *additive* learning layer on top of the rule-based router, not a
replacement: the same ``RoutingDecision`` contract is returned. Two hard rules
run before the bandit is ever consulted:

- Sensitive intent → always human_review. ``SENSITIVE_INTENTS``
  (``app.pipeline.router.SENSITIVE_INTENTS``) currently holds
  ``desk_reject_appeal`` and ``review_decision_appeal`` (reject_appeal.md D55,
  D62), so both are forced to a human under any strategy.
- Confidence below a hard floor → always human_review (see
  ``_CONFIDENCE_FLOOR`` below; still enforced and accurate).

This router decides the lane from intent + confidence BEFORE a draft exists —
it is intentionally NOT draft-quality-aware (making the bandit draft-aware is
out of scope; the rule-based router's draft-quality gate is not replicated
here). Because of that, the bandit could otherwise pick "auto_reply" for a
draft that is not self-sufficient — unresolved ``[CHAIR: ...]`` placeholders,
notes-for-chair, an unrated answer confidence, or no citations at all. That
case is caught by a SEPARATE, strategy-independent safety floor,
``app.pipeline.router.apply_self_sufficiency_floor``, which both
``app.pipeline.orchestrator`` and ``app.pipeline.reevaluation`` call right
after ``router.route()`` returns, whatever the routing strategy — so a draft
that is not self-sufficient can never reach the FAQ lane even under ``rl``.

Arms (actions):     "auto_reply" | "human_review"
Reward signal:      approved → +1 win (the lane was right)
                    rerouted → trial recorded, no win (the lane was wrong)
State:              per-intent {action: {wins, trials}}
Persistence:        JSON (human-readable) under backend/models/

The stored lane vocabulary is "faq"/"human_review"; the bandit's arm vocabulary
is "auto_reply"/"human_review". ``faq`` and ``auto_reply`` are the same action —
``_normalize_action`` maps ``faq`` → ``auto_reply`` so feedback and routing agree.
"""

import json
import logging
import os
import random
from pathlib import Path

from app.core.config import settings
from app.pipeline.router import (
    LANE_FAQ,
    LANE_HUMAN_REVIEW,
    SENSITIVE_INTENTS,
    RoutingDecision,
)

logger = logging.getLogger(__name__)

# Project root (.../conference-email-system) so a relative STATE_PATH like
# "backend/models/..." resolves correctly regardless of the process cwd.
#   .../conference-email-system/backend/app/pipeline/rl_router.py
#   parents: [0]=pipeline [1]=app [2]=backend [3]=project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

ACTION_AUTO_REPLY = "auto_reply"
ACTION_HUMAN_REVIEW = "human_review"
_ARMS = (ACTION_AUTO_REPLY, ACTION_HUMAN_REVIEW)

# Below this confidence we never auto-reply, whatever the bandit has learned.
_CONFIDENCE_FLOOR = 0.4
# Optimistic initialization: an untried arm is assumed decent so it gets explored.
_OPTIMISTIC_WIN_RATE = 0.5


def _action_to_lane(action: str) -> str:
    """Map a bandit arm to the stored lane vocabulary."""
    return LANE_FAQ if action == ACTION_AUTO_REPLY else LANE_HUMAN_REVIEW


def _normalize_action(action: str) -> str:
    """Map a lane/action string to a canonical bandit arm.

    Accepts "faq" (stored lane) and treats it as "auto_reply".
    """
    if action in (LANE_FAQ, ACTION_AUTO_REPLY):
        return ACTION_AUTO_REPLY
    return ACTION_HUMAN_REVIEW


class RLRouter:
    """Epsilon-greedy bandit router that learns lanes from chair feedback."""

    STATE_PATH = "backend/models/rl_router_state.json"
    EPSILON = 0.15  # 15% exploration

    def __init__(self) -> None:
        # state: {intent: {action: {"wins": int, "trials": int}}}
        self.state: dict[str, dict[str, dict[str, int]]] = {}
        self._load_if_exists()

    # --- persistence ------------------------------------------------------
    def _resolve_state_path(self) -> Path:
        """Resolve STATE_PATH to an absolute path (read fresh each call).

        Relative paths are anchored to the project root; absolute paths (e.g. a
        monkeypatched tmp file in tests) are used as-is.
        """
        p = Path(self.STATE_PATH)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    def _load_if_exists(self) -> None:
        """Load saved bandit state if present; start empty otherwise."""
        path = self._resolve_state_path()
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self.state = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - corrupt state → start fresh
            logger.warning("Failed to load RL router state (%s); starting fresh.", exc)
            self.state = {}

    def _save(self) -> None:
        """Persist bandit state as human-readable JSON."""
        path = self._resolve_state_path()
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2, sort_keys=True)

    # --- bandit internals -------------------------------------------------
    def _win_rate(self, intent: str, action: str) -> float:
        """wins/trials for (intent, action), or optimistic 0.5 if untried."""
        stats = self.state.get(intent, {}).get(action)
        if not stats or stats.get("trials", 0) == 0:
            return _OPTIMISTIC_WIN_RATE
        return stats["wins"] / stats["trials"]

    def _choose_arm(self, intent: str) -> str:
        """Epsilon-greedy arm selection for an intent."""
        if random.random() < self.EPSILON:
            return random.choice(_ARMS)  # explore
        # Exploit: highest win-rate arm (ties → auto_reply, the first arm).
        return max(_ARMS, key=lambda a: self._win_rate(intent, a))

    # --- public API -------------------------------------------------------
    def route(
        self,
        intent: str,
        confidence: float,
        existing_threshold: float,
    ) -> RoutingDecision:
        """Choose a lane: rule guards first, then the bandit.

        Order of decision:
        1. Sensitive intent → human_review (hard rule, bandit never consulted).
        2. confidence < 0.4 → human_review (hard floor).
        3. confidence < threshold → human_review (below the auto-reply gate).
        4. Otherwise consult the bandit (epsilon-greedy) for this intent.
        """
        if intent in SENSITIVE_INTENTS:
            return RoutingDecision(
                lane=LANE_HUMAN_REVIEW,
                reason=f"RL router: '{intent}' is a sensitive intent — always human review.",
                confidence_used=confidence,
                threshold_applied=existing_threshold,
                override_reason=f"Intent '{intent}' always requires human review",
            )

        if confidence < _CONFIDENCE_FLOOR:
            return RoutingDecision(
                lane=LANE_HUMAN_REVIEW,
                reason=(
                    f"RL router: confidence {confidence:.2f} below hard floor "
                    f"{_CONFIDENCE_FLOOR:.2f} — forced human review."
                ),
                confidence_used=confidence,
                threshold_applied=existing_threshold,
                override_reason=f"Confidence below hard floor {_CONFIDENCE_FLOOR}",
            )

        if confidence < existing_threshold:
            return RoutingDecision(
                lane=LANE_HUMAN_REVIEW,
                reason=(
                    f"RL router: confidence {confidence:.2f} below threshold "
                    f"{existing_threshold:.2f} — human review."
                ),
                confidence_used=confidence,
                threshold_applied=existing_threshold,
                override_reason=None,
            )

        arm = self._choose_arm(intent)
        lane = _action_to_lane(arm)
        win_rate = self._win_rate(intent, arm)
        return RoutingDecision(
            lane=lane,
            reason=(
                f"RL router: bandit chose '{arm}' for intent '{intent}' "
                f"(win-rate {win_rate:.2f}, epsilon {self.EPSILON})."
            ),
            confidence_used=confidence,
            threshold_applied=existing_threshold,
            override_reason=None,
        )

    def record_feedback(self, intent: str, action: str, outcome: str) -> None:
        """Update bandit state from a chair action and persist.

        approved → +1 win, +1 trial for (intent, action).
        rerouted → +1 trial only (the lane was wrong).
        """
        arm = _normalize_action(action)
        intent_stats = self.state.setdefault(intent, {})
        arm_stats = intent_stats.setdefault(arm, {"wins": 0, "trials": 0})
        arm_stats["trials"] += 1
        if outcome == "approved":
            arm_stats["wins"] += 1
        self._save()

    def get_stats(self) -> dict:
        """Per-intent stats with derived win rates, for the analytics endpoint."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        for intent, arms in self.state.items():
            out[intent] = {}
            for action, stats in arms.items():
                trials = stats.get("trials", 0)
                wins = stats.get("wins", 0)
                out[intent][action] = {
                    "wins": wins,
                    "trials": trials,
                    "win_rate": (wins / trials) if trials > 0 else _OPTIMISTIC_WIN_RATE,
                }
        return out


# Module-level singleton — one bandit per process so its learned state and the
# JSON file stay consistent across requests. Accessed via get_rl_router().
_INSTANCE: RLRouter | None = None


def get_rl_router() -> RLRouter:
    """Return the process-wide RLRouter singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RLRouter()
    return _INSTANCE
