"""Re-evaluate open tickets after a KB change (the chair's "re-evaluate" button).

One sweep re-runs retrieval for every open ticket using the query captured at
ingest (no model call) and re-drafts only the tickets whose grounding set moved.
Chair-edited drafts are never clobbered. The sweep runs in its own DB session so
it can be scheduled as a FastAPI background task after the request returns.

Gate (design §3): a ticket is *affected* iff the set of fresh top-k policy ids
differs from the set stored at draft time. Comparing sets (not ordered lists)
means a pure score reshuffle — same chunks, different order — does not force a
needless re-draft.

Two passes so the queue reflects the whole batch, not one ticket at a time:
- Pass 1 (gate + claim): fast, no model calls. Every affected ticket is claimed
  (``redrafting=True``) and audited ``ticket_redrafting`` up front, so the UI
  shows the entire batch "re-drafting…" at once.
- Pass 2 (draft): the slow per-ticket model calls. Each ticket's new draft is
  saved and its flag cleared as it lands, so the chair watches them resolve one
  by one and can see which are still in progress.
"""

import logging

from app.core.config import settings
from app.db.database import async_session_factory
from app.pipeline.classifier import ClassificationResult
from app.pipeline.drafter import ResponseDrafter
from app.pipeline.orchestrator import resolve_forced_chunk, resolve_lineage_roots
from app.pipeline.retriever import get_retriever, grounded_chunks_hash
from app.pipeline.router import EmailRouter, apply_self_sufficiency_floor
from app.pipeline.taxonomy import REJECT_APPEAL_INTENTS
from app.repositories.audit_repository import AuditRepository
from app.repositories.email_repository import EmailRepository

logger = logging.getLogger(__name__)

_ACTOR = "reevaluation"


async def _fresh_topk_ids(retriever, ctx: dict, top_k: int, db=None):
    """Re-run retrieval from a ticket's stored context; return (ids, chunks).

    ``prior_intent`` is replayed from the stored context so the fresh ranking
    reproduces the SAME soft intent-prior boost (B5) applied at ingest — otherwise
    an unboosted re-run could shuffle the top-k and force a spurious re-draft.
    Legacy rows lack ``prior_intent`` → "" → unboosted, matching their stored ids.

    A chair-forced policy (``forced_policy_key``) is RE-APPLIED here, through the
    same ``resolve_forced_chunk`` the orchestrator uses, so the sweep cannot
    silently drop a manual invoke. Re-resolving rather than blindly copying the id
    means the active-status rule is re-checked: a policy retired since it was
    forced is correctly dropped, and that drop shows up in the gate below as a
    genuine grounding change.

    Chair EXCLUSIONS (``excluded_policy_ids``) are likewise re-applied, matched on
    LINEAGE ROOT rather than the literal key. This matters most here of all: the
    sweep is triggered BY a KB edit, and ``edit_policy`` mints a new key for an
    edited policy (``int_foo`` → ``int_foo__v2``). An exact-key exclusion would go
    stale on exactly the event that runs this code, and the chunk the chair
    removed would quietly reappear under its new name.

    Order mirrors ``EmailPipeline._compute`` exactly — filter, then force, with
    exclusion beating a conflicting forced key — so a swept re-draft and a manual
    one can never disagree about the same stored context.

    This also keeps the gate honest. The stored ids reflect both the forced policy
    and the removals, so a fresh run applying neither would differ from the stored
    set on every sweep and re-draft the ticket forever. ``db`` is optional purely
    so legacy/unit callers that pass no session keep the old pure-retrieval
    behaviour.
    """
    query = (ctx or {}).get("query") or ""
    intent = (ctx or {}).get("intent") or ""
    prior_intent = (ctx or {}).get("prior_intent") or ""
    chunks = await retriever.retrieve(
        query, intent, top_k=top_k, prior_intent=prior_intent
    )

    forced_key = (ctx or {}).get("forced_policy_key")
    excluded_ids = (ctx or {}).get("excluded_policy_ids") or []

    excluded_roots: set[str] = set()
    roots: dict[str, str] = {}
    if excluded_ids and db is not None:
        roots = await resolve_lineage_roots(
            db,
            [c.policy_id for c in chunks]
            + list(excluded_ids)
            + ([forced_key] if forced_key else []),
        )
        excluded_roots = {roots.get(x, x) for x in excluded_ids}
        chunks = [
            c
            for c in chunks
            if roots.get(c.policy_id, c.policy_id) not in excluded_roots
        ]

    forced_excluded = bool(forced_key) and (
        roots.get(forced_key, forced_key) in excluded_roots
    )
    if forced_key and db is not None and not forced_excluded:
        forced = await resolve_forced_chunk(db, forced_key, chunks)
        if forced is not None:
            chunks = [*chunks, forced]

    return [c.policy_id for c in chunks], chunks


async def clear_stale_redrafting_flags(session_factory=async_session_factory) -> int:
    """Clear every ``redrafting`` flag (call once at process startup).

    A freshly-started process has no in-flight sweep, so any ``redrafting=True`` is a
    flag stranded by a previous process that died mid-sweep. Left set, it shows a
    permanent "re-drafting…" badge and the sweep's atomic claim would skip that ticket
    on every future run. Returns the number of rows cleared.
    """
    async with session_factory() as db:
        return await EmailRepository().clear_all_redrafting_flags(db)


async def reevaluate_open_tickets(session_factory=async_session_factory) -> dict:
    """Sweep open tickets; re-draft the ones whose retrieval changed.

    Returns a summary: {"open", "redrafted", "skipped_edited", "skipped_no_context",
    "skipped_contended", "unaffected"}, plus "skipped_appeal" ONLY when
    APPEAL_REPLY_COMPOSER_ENABLED is on. Runs in two passes (see module docstring):
    all affected tickets are marked "re-drafting" first, then each is re-drafted.
    Best-effort per ticket — a failure on one ticket is logged, its ``redrafting``
    flag cleared, and the sweep continues.
    """
    email_repo = EmailRepository()
    audit_repo = AuditRepository()
    retriever = get_retriever()
    router = EmailRouter(strategy=settings.ROUTING_STRATEGY)
    drafter = ResponseDrafter(provider=settings.MODEL_PROVIDER)
    top_k = settings.MAX_RETRIEVED_CHUNKS

    stats = {
        "open": 0,
        "redrafted": 0,
        "skipped_edited": 0,
        "skipped_no_context": 0,
        "skipped_contended": 0,
        "unaffected": 0,
    }
    # Appeal reply hook (reject-appeal Phase 4): with the flag ON, an appeal
    # ticket's draft comes from the approved reply blocks, which do not use the
    # knowledge base — so a KB edit has nothing to change in it, and re-drafting
    # it here would replace it with a MODEL draft. Such tickets are skipped. The
    # counter exists only when the flag is on, so flag-off output is unchanged.
    skip_appeals = settings.APPEAL_REPLY_COMPOSER_ENABLED
    if skip_appeals:
        stats["skipped_appeal"] = 0

    async with session_factory() as db:
        tickets = await email_repo.get_open_tickets(db)
        stats["open"] = len(tickets)

        # --- Pass 1: gate + claim (fast, no model calls). Marks EVERY affected
        # ticket "re-drafting" up front so the queue shows the whole batch as
        # in-progress at once; the work list is drafted in Pass 2.
        work: list[dict] = []
        for email in tickets:
            if skip_appeals and (email.classification or {}).get("intent") in REJECT_APPEAL_INTENTS:
                stats["skipped_appeal"] += 1
                continue
            # Legacy / never-captured ticket (retrieval_context back-filled NULL by the
            # migration): no basis to compare, and an empty query retrieves arbitrary
            # policies — re-drafting would clobber a good draft with irrelevant
            # grounding. Skip until it is (re)captured at a future ingest. NOTE:
            # discriminate on the context being ABSENT, not on retrieved_ids being
            # empty — a real query that matched nothing yet is legitimately eligible so
            # a later KB addition can fill it.
            if email.retrieval_context is None:
                stats["skipped_no_context"] += 1
                continue
            ctx = email.retrieval_context
            stored_ids = set(ctx.get("retrieved_ids") or [])
            stored_hash = ctx.get("chunk_hash")

            fresh_ids_list, fresh_chunks = await _fresh_topk_ids(retriever, ctx, top_k, db)
            fresh_hash = grounded_chunks_hash(fresh_chunks)
            # Two independent axes make a ticket "affected":
            #  (1) the grounded id-SET moved (a different chunk now ranks top-k), or
            #  (2) a grounded chunk was re-labelled — same ids, different intents —
            #      caught by the (id, intents) hash. Legacy rows have no stored hash;
            #      treat that axis as unchanged so they never spuriously re-draft (the
            #      id-set gate still applies to them exactly as before).
            ids_changed = set(fresh_ids_list) != stored_ids
            hash_changed = stored_hash is not None and stored_hash != fresh_hash
            if not ids_changed and not hash_changed:
                stats["unaffected"] += 1
                continue

            email_id = str(email.id)

            # Affected but chair-edited → never clobber; audit that it *would* have
            # changed so the chair knows their edit was preserved. (Currently
            # unreachable in production: is_edited is only set by approve_email, which
            # moves status off draft_generated; kept as forward-compatible defense.)
            if (email.draft or {}).get("is_edited"):
                await audit_repo.log_action(
                    db, email_id, "ticket_redraft_skipped_edited", _ACTOR,
                    {"stored_ids": sorted(stored_ids), "fresh_ids": fresh_ids_list},
                )
                stats["skipped_edited"] += 1
                continue

            # Atomically claim the ticket. This is the ONLY guard against a second
            # overlapping sweep re-drafting it, and it refuses the claim if the chair
            # approved the ticket since the sweep started (status no longer
            # draft_generated). No separate snapshot check — the claim is authoritative.
            if not await email_repo.claim_for_redraft(db, email_id):
                stats["skipped_contended"] += 1
                continue

            await audit_repo.log_action(
                db, email_id, "ticket_redrafting", _ACTOR,
                {"stored_ids": sorted(stored_ids), "fresh_ids": fresh_ids_list},
            )
            work.append({
                "email_id": email_id,
                "ctx": ctx,
                "stored_ids": stored_ids,
                "fresh_ids_list": fresh_ids_list,
                "fresh_hash": fresh_hash,
                "fresh_chunks": fresh_chunks,
                "classification": ClassificationResult(**(email.classification or {})),
                "email_data": {
                    "from": email.sender,
                    "sender_name": email.sender_name,
                    "subject": email.subject,
                    "body": email.body,
                },
                "before_ph": len((email.draft or {}).get("placeholders") or []),
            })

        # --- Pass 2: draft each claimed ticket (the slow model calls) and clear
        # its flag as the new draft lands, so tickets resolve one by one.
        for item in work:
            email_id = item["email_id"]
            try:
                classification = item["classification"]
                fresh_chunks = item["fresh_chunks"]
                # Re-mark the chair's forced policy so a swept re-draft keeps
                # the same must-address instruction the original draft had.
                draft = await drafter.draft(
                    item["email_data"], classification, fresh_chunks,
                    item["ctx"].get("forced_policy_key"),
                )
                routing = router.route(classification, fresh_chunks, draft)

                # Safety floor (strategy-independent, shared with the orchestrator's
                # Pass-1 draft): a draft that is not self-sufficient can NEVER be
                # auto-answered, whatever the router returned. Same floor as
                # orchestrator.py's — this is Pass-2's re-draft, so it needs the
                # identical guard (defense-in-depth, deliberately redundant for the
                # rule_based router).
                routing = apply_self_sufficiency_floor(routing, draft)

                new_ctx = {
                    "query": item["ctx"].get("query", ""),
                    "intent": item["ctx"].get("intent", ""),
                    "prior_intent": item["ctx"].get("prior_intent", ""),
                    "retrieved_ids": item["fresh_ids_list"],
                    "chunk_hash": item["fresh_hash"],
                }
                # Carry a chair's manual invoke across the recompute. Added ONLY
                # when the ticket already had one, so a normal ticket's context
                # keeps exactly the five keys it has always had. The value is the
                # requested key, not proof it applied — whether it survived is
                # derived from retrieved_ids (see emails API
                # ``_forced_policy_applied``), which is why a since-retired policy
                # correctly reads as false here rather than being forgotten.
                forced_key = item["ctx"].get("forced_policy_key")
                if forced_key:
                    new_ctx["forced_policy_key"] = forced_key
                # Same carry-forward for chair removals — added ONLY when the
                # ticket had some, so a normal ticket's context keeps exactly the
                # five keys it has always had. Dropping these would silently undo
                # the chair's curation on the next KB edit.
                excluded_ids = item["ctx"].get("excluded_policy_ids")
                if excluded_ids:
                    new_ctx["excluded_policy_ids"] = excluded_ids
                saved = await email_repo.save_redraft(
                    db, email_id,
                    draft=draft.model_dump(),
                    routing=routing.model_dump(),
                    retrieval_context=new_ctx,
                )
                if saved is None:
                    # Approved/changed between claim and save: its content was left
                    # intact. Clear the flag we set so the ticket isn't stranded, and
                    # count it as contended rather than redrafted.
                    await email_repo.set_redrafting(db, email_id, False)
                    stats["skipped_contended"] += 1
                    continue

                await audit_repo.log_action(
                    db, email_id, "ticket_redrafted", _ACTOR,
                    {
                        "stored_ids": sorted(item["stored_ids"]),
                        "fresh_ids": item["fresh_ids_list"],
                        "placeholders_before": item["before_ph"],
                        "placeholders_after": len(draft.placeholders),
                        "lane": routing.lane,
                    },
                )
                stats["redrafted"] += 1
            except Exception:  # noqa: BLE001 - one bad ticket must not stop the sweep
                logger.exception("Re-draft failed for email %s; clearing flag.", email_id)
                await email_repo.set_redrafting(db, email_id, False)

    logger.info("Re-evaluation sweep complete: %s", stats)
    return stats
