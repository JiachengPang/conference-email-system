"""FastAPI application entry point for the Conference Email System.

This is the composition root: it wires together middleware, lifespan, and the
API routers. Business logic lives in the pipeline / db modules — not here.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import audit, auto_replies, dashboard, emails
from app.api.routes.training import router as training_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.appeals import router as appeals_router
from app.api.v1.chairs import router as chairs_router
from app.api.v1.emails import router as emails_router
from app.api.v1.policies import router as policies_router
from app.api.v1.retrieval import router as retrieval_router
from app.api.v1.zendesk import router as zendesk_router
from app.core.config import settings

logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"
SERVICE_NAME = "conference-email-system"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    On startup, optionally warm the retriever (build its index / load the dense
    model for faiss/fusion) so the first real request doesn't pay the cold-start
    latency. Guarded by ``WARM_RETRIEVER_ON_STARTUP`` and non-fatal.
    """
    # --- startup ---
    if settings.WARM_RETRIEVER_ON_STARTUP:
        try:
            from app.pipeline.retriever import get_retriever

            await get_retriever().retrieve("warmup", intent="", top_k=1)
            logger.info("Retriever warmed at startup.")
        except Exception as exc:  # non-fatal: fall back to lazy load on first request
            logger.warning("Retriever warm-up skipped: %s", exc)

    # Clear any redrafting flags stranded by a process that died mid-sweep (a fresh
    # process has no in-flight sweep, so any redrafting=True is stale). Non-fatal.
    try:
        from app.pipeline.reevaluation import clear_stale_redrafting_flags

        cleared = await clear_stale_redrafting_flags()
        if cleared:
            logger.info("Cleared %d stale redrafting flag(s) at startup.", cleared)
    except Exception as exc:  # non-fatal
        logger.warning("Could not clear stale redrafting flags: %s", exc)

    # Background Zendesk poll loop — started ONLY when explicitly enabled, so it
    # never runs in tests/CI or a default deployment. Both this loop and the
    # manual /api/v1/zendesk/sync endpoint call the same run_sync_cycle.
    zendesk_stop: asyncio.Event | None = None
    zendesk_task: asyncio.Task | None = None
    if settings.ZENDESK_POLLING_ENABLED:
        from app.integrations.zendesk.adapter import zendesk_poll_loop

        zendesk_stop = asyncio.Event()
        zendesk_task = asyncio.create_task(zendesk_poll_loop(zendesk_stop))
        logger.info("Zendesk background polling enabled.")

    yield
    # --- shutdown ---
    if zendesk_task is not None:
        zendesk_stop.set()
        zendesk_task.cancel()
        try:
            await zendesk_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Conference Email System",
    version=API_VERSION,
    description="Automated conference email reply & routing system.",
    lifespan=lifespan,
)

# CORS — permissive for local development. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register router stubs. Routes themselves arrive in Phase 1.
app.include_router(emails.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(auto_replies.router, prefix="/api")

# v1 API — implemented endpoints.
app.include_router(emails_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(training_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(chairs_router, prefix="/api/v1")
app.include_router(policies_router, prefix="/api/v1")
app.include_router(zendesk_router, prefix="/api/v1")
app.include_router(appeals_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {
        "status": "ok",
        "version": API_VERSION,
        "service": SERVICE_NAME,
    }


# How long to wait when probing a local model endpoint's readiness.
_MODEL_PROBE_TIMEOUT_SECONDS = 3.0


@app.get("/api/v1/health/model", tags=["system"])
async def model_health() -> dict:
    """Report the configured drafter provider and its reachability.

    For the local provider, performs a quick GET to ``{base}/models`` (3s
    timeout) so the frontend can show a live model-status badge. For any other
    provider there is nothing to probe, so status is always ``configured``.
    """
    # Normalize the historical "anthropic_api" spelling to the badge's enum.
    provider = "anthropic" if settings.MODEL_PROVIDER == "anthropic_api" else settings.MODEL_PROVIDER

    # The template provider has no external dependency, so it is always healthy.
    if provider == "template":
        return {
            "provider": "template",
            "local_model_url": None,
            "local_model_name": None,
            "status": "healthy",
        }

    if provider != "local":
        return {
            "provider": provider,
            "local_model_url": None,
            "local_model_name": None,
            "status": "configured",
        }

    base = settings.LOCAL_MODEL_BASE_URL.rstrip("/")
    status_value = "configured"
    # Authenticate the probe. Hosted endpoints (e.g. OpenAI) require the bearer
    # token — an unauthenticated GET /models returns 401, which would make a
    # perfectly reachable, working endpoint report "unreachable". Unauthenticated
    # local servers (LOCAL_MODEL_API_KEY unset) send no header, as before.
    headers = (
        {"Authorization": f"Bearer {settings.LOCAL_MODEL_API_KEY}"}
        if settings.LOCAL_MODEL_API_KEY
        else None
    )
    try:
        async with httpx.AsyncClient(timeout=_MODEL_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base}/models", headers=headers)
        if response.status_code != 200:
            status_value = "unreachable"
    except Exception:  # noqa: BLE001 - any failure means the endpoint is unreachable
        status_value = "unreachable"

    return {
        "provider": provider,
        "local_model_url": settings.LOCAL_MODEL_BASE_URL,
        "local_model_name": settings.LOCAL_MODEL_NAME,
        "status": status_value,
    }


@app.get("/api/v1/config", tags=["system"])
async def app_config() -> dict:
    """UI-relevant runtime flags.

    ``allow_auto_send`` is the transport gate: when False (the default), every
    outbound email — FAQ lane included — waits on a chair decision, so the review
    UI shows the approve/send controls rather than an 'auto-replied' state. Only
    when True may a complete FAQ draft be released without human approval.
    """
    return {"allow_auto_send": settings.ALLOW_AUTO_SEND}
