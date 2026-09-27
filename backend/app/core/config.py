"""Application configuration.

Centralized, typed settings loaded from environment / .env. The four "swappable
flags" below are the architectural seams that let us replace pipeline modules
(classifier, retriever, router, drafter) without rewriting the app.
"""

from functools import lru_cache
from typing import ClassVar, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Values are read from environment variables first, then a local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Canonical set of ingestable Zendesk ticket statuses ("deleted" is excluded
    # — it's always skipped by the adapter). Used to validate ZENDESK_SYNC_STATUSES.
    ZENDESK_VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"new", "open", "pending", "hold", "solved", "closed"}
    )

    # --- Swappable architecture flags -------------------------------------
    # Drafter backend. "anthropic_api"/"anthropic" → hosted API; "local" →
    # OpenAI-compatible local endpoint (e.g. self-hosted inference server);
    # "template" → zero-dependency drafter that fills a response template from
    # retrieved policy text with no model call (safest offline fallback);
    # "fallback" → deterministic no-network stub. Both anthropic spellings are
    # accepted so the historical default ("anthropic_api") and the shorter
    # "anthropic" both work.
    MODEL_PROVIDER: Literal[
        "anthropic_api", "anthropic", "local", "template", "fallback"
    ] = "anthropic_api"
    CONFIDENCE_THRESHOLD: float = 0.75
    # Retriever backend: "bm25" → keyword BM25 over the KB; "faiss" → dense
    # sentence-embedding retrieval (FAISS IndexFlatIP, cosine); "fusion" →
    # Reciprocal Rank Fusion over both. Default is "fusion": E003 validated the
    # distill+fusion recipe (hit@3 .649 → .892 on real tickets).
    RETRIEVAL_BACKEND: Literal["bm25", "faiss", "fusion"] = "fusion"
    ROUTING_STRATEGY: Literal["rule_based", "rl"] = "rule_based"
    # Chair router backend (Phase 6A): the SECOND routing decision — which chair
    # a human_review email is assigned to, distinct from the lane decision above.
    # "intent_mapping" matches the classified intent against each active chair's
    # areas (rule-based). The Literal is the swap seam: adding "learned"/"rl"
    # later is a one-line widening + a factory branch, with callers unchanged.
    CHAIR_ROUTING_STRATEGY: Literal["intent_mapping"] = "intent_mapping"
    # Classifier backend. "keyword" → dependency-free baseline; "trainable" →
    # sentence-embedding + LogisticRegression model (auto-falls back to keyword
    # until a model is trained). Default stays "keyword".
    CLASSIFIER_BACKEND: Literal["keyword", "trainable"] = "keyword"

    # Soft intent→KB retrieval prior. OFF by default — E010
    # (docs/exp_tracking/E010_intent_prior.md) showed the current boost
    # regresses fusion retrieval (hit@1 .730→.243); re-enable only after
    # magnitude tuning.
    INTENT_PRIOR_ENABLED: bool = False

    # Reciprocal-review dispute detector (reject-appeal Phase 1). A SEPARATE,
    # conditional model call that runs only when the classified intent is
    # `desk_reject_appeal` — replacing the RECIPROCAL_DISPUTE block that used to
    # sit in the distiller's main system prompt. It was moved out because that
    # block perturbed the QUERY lines the same prompt produces: old-vs-new
    # top-k Jaccard 0.445 against a same-prompt noise floor of 0.710, i.e. a
    # real retrieval shift, not run-to-run variance
    # (docs/exp_tracking/reject_appeal.md D27/D35).
    #
    # ON by default as of the final commit of the rebuild. Verified before the
    # switch by a real `detect` run over the Phase 1 sample: **within-gate
    # precision 0.931 / recall 1.000 over 36 calls**, with the main distiller
    # prompt byte-identical to the pre-series original `8c6eb49` (4,573 chars,
    # sha256 7d944e3459b5…) — so the detector earns its place WITHOUT the
    # existing pipeline changing by a single byte.
    #
    # ⚠️ Recall is within-gate. End-to-end recall is bounded by intent accuracy:
    # an r ticket the gate never reaches is a miss no detector quality can
    # recover, and the `beb5cf6` definition revert lowered that ceiling by ~7pp
    # on purpose (reject_appeal.md D26/D52). Widening the gate is a change to
    # the detector's own code, never to the prompt.
    #
    # Gating stays TWO-LEVEL: this flag is read at the call site in
    # `orchestrator._compute` (so no call is attempted and the reason is visible
    # there), and the detector additionally self-gates on MODEL_PROVIDER (so it
    # is a no-op under template/fallback and throughout the test suite).
    #
    # To switch OFF: set RECIPROCAL_DETECTOR_ENABLED=False in the environment.
    # That stops the call only — a stored answer is PRESERVED, never wiped, so
    # turning the feature off cannot destroy data (D42).
    RECIPROCAL_DETECTOR_ENABLED: bool = True

    # Confidence calibration (Phase 5B). When True AND a fitted calibrator
    # artifact exists for the active CLASSIFIER_BACKEND, the router uses the
    # calibrated confidence instead of the raw classifier score. Off by default
    # so behaviour is unchanged for anyone not opting in; the calibrator maps
    # raw confidence → P(correct) and never changes intent predictions or the
    # FAQ threshold itself.
    CALIBRATION_ENABLED: bool = False

    # Warm the retriever during app startup (build its index — and for the
    # faiss/fusion backends, load the dense embedding model) so the first real
    # request doesn't pay the cold-start, which can exceed the frontend's
    # request timeout. Tests set this False so the suite never loads embeddings.
    WARM_RETRIEVER_ON_STARTUP: bool = True

    # Retrieval-query strategy (E003). "prefix" → legacy body[:300] query with
    # the intent token appended (no model call). "distill" → one model call
    # per email rewrites it into 1-3 compact policy-vocabulary queries AND
    # classifies intent (hit@3 .649 → .892 on real tickets); on any distiller
    # failure the pipeline falls back to the keyword classifier and a
    # subject+body[:600] query. Default is "distill" (E003-validated; pairs with
    # RETRIEVAL_BACKEND=fusion).
    QUERY_STRATEGY: Literal["prefix", "distill"] = "distill"
    # Max characters of a multi-turn conversation transcript fed to the
    # distiller/drafter when re-processing a follow-up. Bounded recent-turns-
    # first (newest kept whole; older dropped with a marker; the latest
    # requester turn is never dropped). Set high — threads are usually short.
    THREAD_TRANSCRIPT_MAX_CHARS: int = 16000

    # --- Outbound send policy (transport gate) -----------------------------
    # Load-bearing precondition for ANY outbound transport (none exists yet;
    # Zendesk write-back must go through app/core/send_gate.authorize_send).
    # False (default) → every send requires a chair's explicit approval
    # (status "approved"), REGARDLESS of lane. True → complete FAQ-lane
    # drafts (no [CHAIR: ...] placeholders, no leak flags) may be released
    # without approval; everything else still requires it. Human approval is
    # policy, not accident — flipping this is an explicit product decision.
    ALLOW_AUTO_SEND: bool = False

    # --- Pipeline tuning --------------------------------------------------
    # Minimum classifier confidence for an email to qualify for the FAQ
    # auto-reply lane. Kept distinct from CONFIDENCE_THRESHOLD so the FAQ
    # gate can be tuned independently of the general confidence floor.
    FAQ_CONFIDENCE_THRESHOLD: float = 0.65
    FAQ_ANSWER_CONFIDENCE_THRESHOLD: float = 0.85  # min drafter self-rated answer confidence for the FAQ lane
    # Max policy chunks the retriever returns as grounding context.
    MAX_RETRIEVED_CHUNKS: int = 5

    # --- Active learning (Phase 5G) ---------------------------------------
    # A chair-approved/rerouted email whose router-used confidence sat within
    # this margin BELOW FAQ_CONFIDENCE_THRESHOLD is flagged as a near-miss
    # candidate for future labeling (band = [threshold - margin, threshold)).
    AL_CONFIDENCE_MARGIN: float = 0.15
    # A chair edit is "meaningful" (flagged) when the word-level change ratio
    # between the original and edited draft exceeds this (a typo fix stays below).
    AL_EDIT_RATIO: float = 0.15
    # Sentence-transformers model used by the FAISS retriever (CPU). Only read
    # when RETRIEVAL_BACKEND == "faiss".
    FAISS_MODEL_NAME: str = "all-MiniLM-L6-v2"
    # Max tokens the drafter may generate for a reply. Sized for REASONING
    # models, which spend completion budget on hidden reasoning before emitting
    # any visible text: at the former 500 the budget could be exhausted before
    # the reply started, returning an empty draft (see EmailStatus.DRAFT_TRUNCATED
    # and the guard in drafter._draft_local). 2500 is the value the drafter eval
    # harnesses already use (scripts/{judge_testset,draft_eval,testset,
    # placeholder_eval}.py) — the only place this module was measured against a
    # reasoning model. Raising the cap also raises the truncation guard's
    # threshold automatically; both read this setting.
    DRAFTER_MAX_TOKENS: int = 2500
    # Determinism for the OpenAI-compatible drafter/distiller. temperature 0 =
    # greedy (most reproducible). Reasoning models that only allow the default
    # temperature reject 0 with a 400 — the caller drops temperature and retries,
    # so this stays safe across models. A fixed seed makes outputs reproducible
    # best-effort for the same input+model (not a hard guarantee).
    DRAFTER_TEMPERATURE: float = 0.0
    DRAFTER_SEED: int = 7
    # Model the drafter calls when MODEL_PROVIDER == "anthropic_api". Never
    # hardcode a model id in source — read it from here so it stays swappable.
    DRAFT_MODEL: str = "claude-sonnet-5"

    # --- Local model provider (OpenAI-compatible, e.g. Ollama) ------------
    # Used only when MODEL_PROVIDER == "local". Base URL of the OpenAI-style
    # API; the drafter POSTs to {base}/chat/completions. Model name and URL
    # are read here — never hardcoded in source.
    LOCAL_MODEL_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_MODEL_NAME: str = "llama3.1:8b"
    # Optional bearer token for the endpoint above. Leave unset for
    # unauthenticated local servers; set it when LOCAL_MODEL_BASE_URL points at
    # a hosted keyed service that speaks the same chat-completions protocol.
    LOCAL_MODEL_API_KEY: str | None = None

    # --- Style guide (Phase 7D) -------------------------------------------
    # Path to a reply style/instruction guide (markdown). When set and readable,
    # its contents are appended to the drafter's system prompt for the AI
    # providers; the guide is style/behavior only and stays subordinate to the
    # grounding rules. Path is resolved relative to the app's working directory
    # (backend/), so the repo-root data/ dir is reached via "../data" — same
    # convention as the SQLite DATABASE_URL default below. Set to None (or an
    # unreadable path) to leave the prompt unchanged.
    STYLE_GUIDE_PATH: str | None = "../data/style_guide/style_guide_v2.md"

    # --- Zendesk integration (credential layer) ---------------------------
    # Auth mode selects the credential provider via
    # app.integrations.zendesk.get_zendesk_credential_provider — the same
    # config-flag swap convention as the pipeline seams. "token" → API-token
    # (HTTP Basic) auth; "oauth" → client_credentials OAuth (proven against the
    # incremental ticket pull). Callers depend only on the provider interface,
    # never on how credentials are obtained.
    ZENDESK_AUTH_MODE: Literal["token", "oauth"] = "token"
    # Account subdomain, e.g. "aaai" → https://aaai.zendesk.com. Required by both
    # auth modes (it forms the REST base URL and the OAuth token endpoint host).
    ZENDESK_SUBDOMAIN: str | None = None
    # API-token (Basic) auth fields — required only when ZENDESK_AUTH_MODE=token.
    # Zendesk Basic auth uses username "{email}/token" with the API token as the
    # password.
    ZENDESK_EMAIL: str | None = None
    ZENDESK_API_TOKEN: str | None = None
    # OAuth client_credentials fields — required only when
    # ZENDESK_AUTH_MODE=oauth. The client secret is read here (Settings/.env),
    # never from a checked-in secrets file. Scope defaults to read-only.
    ZENDESK_OAUTH_CLIENT_ID: str | None = None
    ZENDESK_OAUTH_CLIENT_SECRET: str | None = None
    ZENDESK_OAUTH_SCOPE: str = "read"

    # --- Zendesk ingest poller (Piece 4, read-only) -----------------------
    # Master switch for the background polling loop. Default False so the loop
    # NEVER starts on its own (tests, CI, or any environment) unless explicitly
    # enabled — the manual POST /api/v1/zendesk/sync endpoint works regardless.
    ZENDESK_POLLING_ENABLED: bool = False
    # Seconds between background poll cycles (incremental export tolerates a
    # 2–5 min cadence comfortably within its 10 req/min ceiling; see §7).
    ZENDESK_POLL_INTERVAL_SECONDS: int = 300
    # Unix epoch for the VERY FIRST incremental call (must be ≥ 1 min in the
    # past). Later calls use the stored cursor. Default 1 = "everything the
    # account has"; set a recent epoch to bound an initial live pull.
    ZENDESK_SYNC_START_TIME: int = 1
    # Page size for the incremental export (max 1000; kept modest by default).
    ZENDESK_SYNC_PER_PAGE: int = 100
    # Safety bound on pages fetched per cycle so a cold start can't pull the
    # whole account (and hammer per-ticket comment fetches) in one pass.
    ZENDESK_MAX_PAGES_PER_CYCLE: int = 10
    # Comma-separated allow-list of Zendesk ticket statuses to INGEST. The
    # incremental export is time/cursor-based (no server-side status filter), so
    # this is applied client-side by the adapter (a later piece) to skip tickets
    # whose status is not listed. Default = every ingestable status, so unset
    # behavior is unchanged. "deleted" is always skipped regardless (not a valid
    # value here). Parsed via `zendesk_sync_statuses` (splits/trims/lowercases).
    # Note: "solved" and "closed" are the SAME bucket operationally — a solved
    # ticket auto-transitions to closed over time with no further chair action —
    # so list both (or neither) to keep that bucket whole.
    ZENDESK_SYNC_STATUSES: str = "new,open,pending,hold,solved,closed"

    # --- OpenReview (app.integrations.openreview) -------------------------
    # Username (the account email) and password for the OpenReview account
    # ConfMail posts as. Both are required to build a client; a missing or blank
    # value raises OpenReviewCredentialError at construction naming exactly
    # which one, rather than surfacing later as an opaque login failure.
    # Read here (Settings/.env) and never from a checked-in file.
    OPENREVIEW_USERNAME: str | None = None
    OPENREVIEW_PASSWORD: str | None = None
    # API base. Defaults to the v2 production API, which is the version that
    # supports Official_Comment note posting.
    #
    # This default is LOAD-BEARING, not cosmetic: openreview-py's
    # OpenReviewClient falls back to `http://localhost:3001` when no baseurl is
    # given, so an unset value would silently point a live action at a local dev
    # server instead of failing. The client is therefore always constructed with
    # an explicit baseurl. Point this at https://api2.dev.openreview.net to
    # exercise the API against OpenReview's sandbox.
    OPENREVIEW_BASE_URL: str = "https://api2.openreview.net"
    # The OpenReview venue this deployment posts into, in OpenReview's own group
    # form, e.g. "AAAI.org/2027/Conference". It builds both the Official_Comment
    # invitation and the role signature.
    #
    # ⚠️ DECISION POINT, flagged rather than guessed. Nothing in the email data
    # yields this: the extractor's notification sender carries a venue PREFIX
    # ("aaai2027-notifications@...") which is not the group id and cannot be
    # mapped to one without a lookup table. It is config because ConfMail is
    # deployed per conference — one deployment, one venue — the same reasoning as
    # ZENDESK_SUBDOMAIN. Required at post time; None fails loudly there rather
    # than defaulting to a guess.
    OPENREVIEW_VENUE_ID: str | None = None

    # --- Secrets / connections --------------------------------------------
    ANTHROPIC_API_KEY: str | None = None
    # Primary async connection string used by BOTH the app's async engine and
    # Alembic. migrations/env.py injects this same async URL into Alembic's
    # sqlalchemy.url (see backend/migrations/env.py), so migrations run over the
    # async driver too — there is NO separate sync connection string.
    #   PostgreSQL: postgresql+asyncpg://user:password@localhost:5432/confmail
    #   SQLite (tests): sqlite+aiosqlite:///./test.db
    # Defaults to local SQLite so dev/tests work with no .env present.
    DATABASE_URL: str = "sqlite:///./conference_email.db"

    @property
    def zendesk_sync_statuses(self) -> list[str]:
        """The parsed ZENDESK_SYNC_STATUSES allow-list (see parse_zendesk_statuses)."""
        return parse_zendesk_statuses(self.ZENDESK_SYNC_STATUSES)


def parse_zendesk_statuses(raw: str | None) -> list[str]:
    """Parse a comma-separated Zendesk status string into a clean allow-list.

    Shared by both the ZENDESK_SYNC_STATUSES config default and the per-call
    override on POST /api/v1/zendesk/sync, so the two apply IDENTICAL rules.
    Splits on commas, trims whitespace, lowercases, drops empties, and
    de-duplicates while preserving first-seen order. Only values in
    ``Settings.ZENDESK_VALID_STATUSES`` are kept (unknown tokens are ignored, so
    a stray typo can't silently widen the filter). If the parse yields nothing
    (empty, all-blank, or all-invalid), falls back to every valid status — the
    safe "ingest everything" default that matches unset behavior.
    """
    seen: list[str] = []
    for token_raw in (raw or "").split(","):
        token = token_raw.strip().lower()
        if token and token in Settings.ZENDESK_VALID_STATUSES and token not in seen:
            seen.append(token)
    return seen or sorted(Settings.ZENDESK_VALID_STATUSES)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (single source of truth)."""
    return Settings()


settings = get_settings()
