# Reject-Appeal Handling: Decision Log & Status

Owner: Sahil · Last updated: 2026-09-25 · **Current position: Phase 1, Step 9b (next)**

This file is the source of truth for the reject-appeal workstream. Update it after every task: status tables, new decisions, and a changelog entry.

## Goal

Detect desk-reject appeal emails for AAAI-27, especially reciprocal-review duty disputes (the largest bucket), so they can eventually be answered with policy-consistent templates and bulk replies.

## Overall plan

| Phase | Goal | Status |
|---|---|---|
| 0 | Ground-truth labeling of appeal tickets | Done |
| 1 | Detection: reject-appeal intent + `is_reciprocal_dispute` flag | **In progress (Step 9)** |
| 2 | Reason classification (`appeal_reason[]`, full names, validated) | Pending |
| 3 | Reply templates per reason. `r` gets a policy-stance reply; only `a`/`b` may escalate. Needs Marc's sign-off | Pending |
| 4 | Drafter integration | Pending |
| 5 | Bulk-reply queue for reciprocal disputes (dedicated DB column, deterministic `event_tag`, UI surface) | Pending |

## Phase 0 results

- 200 of 530 tickets labeled, window 2025-09-20 to 2025-09-30.
- Distribution: n=84 (42%), r=112 (56%), b=2, o=2, a/c/d/e=0. This window is essentially binary (r vs. n): Phase 1 rejections this cycle were reciprocal-review-policy driven.
- Taxonomy committed in 9fb6b83: n, a, b, c, d, e, r (reciprocal-review duty dispute: disputing the facts or conceding and asking leniency), o.
- Labeling data lives in data/labeling/, gitignored (165f07a2) for PII. Never commit, search, grep, count, or quote it.

## Phase 1 plan & status

| # | Step | Commit | Status |
|---|---|---|---|
| 1 | `is_reject_appeal` helper + `REJECT_APPEAL_INTENTS` (taxonomy.py) | 8c52d2f | Done (inert) |
| 2 | `desk_reject_appeal` definition covers reciprocal-review grounds + doc mirror | beb5cf6 | Done (**changes live intent prompt**) |
| 3 | Keyword fallback for reciprocal cases | — | Parked (see D8) |
| 4 | Distiller parser for `RECIPROCAL_DISPUTE` | df60479 | Done (inert) |
| 5 | Field on both sides of the extraction mirror | 77a6038 | Done (inert) |
| 6 | Extractor passes distiller value through | b14f590 | Done (inert) |
| 7 | Frontend `ExtractionData` type | 77f72bb | Done (inert) |
| 8 | Prompt asks for `RECIPROCAL_DISPUTE` (the switch) | 11cb466 | Done (**activates detection**) |
| 9a | Read-only report: template impact + eval plan | — | **Done** (see D13–D17) |
| 9b | Build and run eval: flag accuracy, intent + retrieval regression, keyword fallback | — | **Harness built + dry-run green; BLOCKED on a stale backend image (D18) before any real call** |
| 9c | Hold reciprocal cases from auto-template, if 9a recommends | — | Pending — **9a recommends YES** (D13) |
| 10 | Deploy to AAAI server (only after eval clears; optional, see D11) | — | Pending |

## Decisions

**D1. `r` gets its own flag, `is_reciprocal_dispute`.** 56% of labeled appeals, phrased distinctively, and enables Phase 5 bulk replies without waiting on Phase 2. Rejected: folding `r` into `appeal_reason[]` only.

**D2. `is_reject_appeal` is derived from intent, not asked of the model.** The taxonomy already has `desk_reject_appeal` and `review_decision_appeal`; asking twice creates two sources that drift. Returns None for None intent (unclassified ≠ not an appeal).

**D3. `desk_reject_appeal` definition amended.** Added "waive" (covers conceding and asking leniency) and "reciprocal-review duty" grounds. Wording matches the label vocabulary. Mirrored in docs/INTENT_TAXONOMY.md.

**D4. `appeal_reason[]` deferred to Phase 2.** It is Phase 2's job; adding it now only lengthens the prompt. When added: full names on the wire (not letter codes), validated against a single source of truth in app code, with a drift test.

**D5. `event_tag` deferred to Phase 5 and never produced by the LLM.** The model can't know which event an email belongs to. Derive it deterministically from `received_at` plus a configured date window.

**D6. Flag stored in the extraction JSON for Phase 1; no migration.** A dedicated column comes in Phase 5, because the queue needs to filter on it and chair corrections in JSON would be wiped when follow-ups trigger reprocessing (same reason `openreview_candidate_dismissed` got its own column).

**D7. Tri-state `bool | None`; LLM-only.** None means unknown, never "no"; only False is a ruling-out. The regex fallback never answers it. Frontend must test `=== true` / `=== false`, never truthiness.

**D8. Keyword fallback parked.** Substring conjunctions like ("reciprocal", "reject") prove co-occurrence, not what was rejected. Two adversarial cases leaked: a rejected reviewer *application*, and a pre-submission *waiver* request. 3a (AND-group matcher support) is stashed as `parked: 3a keyword AND-group support`. Revisit only if the Step 9 eval on real labels shows fallback misses matter. The fallback runs ~1% of the time (distiller timeouts/failures).

**D9. Flag judged over the whole conversation; INTENT stays latest-message.** An appeal thread whose latest turn is "any update?" is still a reciprocal dispute. Consequence: intent (and so `is_reject_appeal`) can be False while the flag is True, so **the Phase 5 queue must key off the flag, not intent.**

**D10. Prompt placement.** The flag gets its own block after the identification lines and before the injection guard. It uses YES/NO, not the identification block's NONE convention (NONE parses as unknown). Four confusable NO cases are spelled out. Prompt grew **4,573 → 5,371 chars (+17.4%)**. *(Corrected 2026-09-25: the earlier "4,604" was the baseline prompt STRUCTURE measured with today's intent menu. The true `8c6eb49` prompt is 4,573; the 31-char gap is exactly the `beb5cf6` definition edit, which reaches the prompt only through the menu. 4,604 is still a real number — it is what the stale container currently runs, i.e. new defs, no flag block.)*

**D11. No deploy before the Step 9 eval clears.** The deploy is invisible to users: it records the flag on new emails and reroutes reciprocal complaints to `desk_reject_appeal`. Its value is collecting real data. Deploying Phases 1 and 2 together is an acceptable alternative.

**D12. Commits 5/6 re-scoped.** The drift guards require both sides of the extraction mirror to change together, so commit 5 added the field to both sides and commit 6 did the pass-through.

**D13. Hold reciprocal cases in human review via `SENSITIVE_INTENTS` (9c = yes).** Set `router.SENSITIVE_INTENTS = ["desk_reject_appeal"]` until Phase 3 templates exist. It is the only existing seam: `route()` receives `(classification, retrieved_chunks, draft)` and **never sees `extraction`**, so there is no way to hold on the flag itself without new plumbing. Consequence, accepted: this over-holds — it holds *every* desk-reject appeal, including formatting/page-limit ones, not just reciprocal. Cheap at this stage (Phase 0 says the window is r-dominated) and reversible in one line. Both strategies honor it: `rl_router.py` imports the same constant, so there is one source. Rejected for Phase 1: threading `extraction` into `route()` — a real new seam, and it would make the router depend on the extractor.

**D14. The template itself is NOT the live risk — the LLM drafter is.** `template_drafter.py` is reached only under `MODEL_PROVIDER=template`; `.env` runs `local`. And a template draft can never auto-send anyway: `TemplateDrafter` never sets `answer_confidence`, so it is `None`, which fails the router's FAQ gate *and* trips `apply_self_sufficiency_floor`, with `ALLOW_AUTO_SEND=False` as a third guard. The path that can actually reach the FAQ lane for a reciprocal appeal is the **LLM drafter**, which does self-rate. So D13's hold is aimed at that path; the template analysis is about *wording suitability*, not auto-send risk.

**D15. `data/eval/ground_truth.json` cannot measure intent — every label is dead vocabulary.** All 67 rows carry pre-2026-07-20 intents (`submission_deadline`, `general_inquiry`, …); **0 of 11 distinct labels are in today's 14-intent `VALID_INTENTS`**, and `desk_reject_appeal` appears nowhere. Intent accuracy against it is structurally 0. It also poisons the retrieval-only section, which passes `ground_truth_intent` as the query intent (run_eval.py:328). **Step 9b needs intent gold before it can report an intent regression at all.**

**D16. `run_eval.py` is blind to both prompt changes and cannot be the before/after harness as-is.** It calls `IntentClassifier` (keyword) directly and **never imports the distiller** — no `QUERY_STRATEGY`, no transcript. Both shipped changes are distiller-only: `beb5cf6` edits `INTENT_DEFS` → `_INTENT_MENU` → the distiller prompt, and the keyword classifier imports only `VALID_INTENTS`/`FALLBACK_INTENT`, never `INTENT_DEFS`. So a clean before/after run of `run_eval.py` would show **no diff by construction** — a false all-clear. The distiller-based harnesses (`e005_embed_repr.py`, `e010_intent_prior_ablation.py`, `query_distill_ablation.py`) are the right base, but they read `data/eval_real/`, which is **absent on this machine**.

**D17. Eval must use the SINGLE-MESSAGE distiller path, not the transcript path.** Production classifies a new ticket at ingest from subject + the initial public end-user comment only: `adapter._ingest_new_ticket` builds `email_data` with no `thread_transcript` key, so `orchestrator._compute` passes `transcript=None`. The conversation branch runs only on follow-up (`reprocess_email_with_thread`). ⚠️ This **collides with D9** (flag judged over the whole conversation): at first ingest there *is* no conversation, so the prompt's whole-conversation rule is inert on exactly the path that classifies most tickets. **Open question for 9b: were the 200 labels assigned per ticket (whole thread) or per first message?** If per-thread, scoring the single-message path will under-count recall for reasons that are not the flag's fault, and the two must be reported separately.

**D18. ⚠️ The running backend image PREDATES `11cb466` — no real call may be made until it is rebuilt.** Verified in the container: `_SYSTEM_PROMPT` is 4,604 chars and `"RECIPROCAL_DISPUTE" in prompt` is **False**; `distiller.py` has the parser and the new taxonomy definition but not the prompt block. The image bakes source with **no volume mount**, so this is the project's recurring trap, third occurrence. Running the eval here would return the flag as None for all 100 tickets and read as *"the flag does not work"* rather than *"the image is stale"*. Guarded by `test_live_module_is_at_head_not_a_stale_image`, which is currently the suite's one intended failure. Unblock with `docker compose build backend` — ⚠️ note that recreating the backend container auto-runs `alembic upgrade head` on the demo DB, so that rebuild is Sahil's call, not the harness's.

**D19. Eval arms are swapped IN-PROCESS, not via a worktree.** `git diff 8c6eb49..HEAD -- distiller.py` is additive only (prompt strings, `_RECIPROCAL_DISPUTE_RE`, the `DistillResult` field, the parse block) — the HTTP payload, model params and retrieval path are untouched. So overriding `_SYSTEM_PROMPT` isolates the independent variable exactly, while a worktree would also vary retriever code, config defaults and installed deps — the confounds the plan explicitly excludes. Running HEAD's parser over a baseline prompt is correct, not a shortcut: no flag line → `None`, the same observable the baseline code gave with no parser. Both arms' prompt sha256 + length land in the manifest, so "the arms really differed" is auditable; `compare` warns loudly if the two shas match.

**D20. The container has no `git` and no work tree, so git work is split out.** `prepare-baseline` runs on the **host** (writes `baseline_prompt.txt` + sha), the file is copied in, and `run --arm baseline --baseline-prompt-file` consumes it. The gitignore guard treats "no work tree above this path" as satisfied-by-construction — the guard exists to prevent a *commit*, and there is nothing to commit into. App imports in the script are **lazy** so the host-only subcommand does not need the container's ML deps. Found by running the suite in the container, which is the only place it surfaces.

**D21. The harness refuses to run under a non-`local` `MODEL_PROVIDER`.** `distill()` returns None immediately otherwise, so the eval would emit a full set of all-None rows having made **zero** calls — indistinguishable from both a real negative result and a working call cap. Caught when the hermetic conftest (which pins `fallback`) made a dry-run report 0 calls against 100 rows.

**D22. ⚠️ Label/production information asymmetry — a recall CEILING, not a flag defect.** `label_appeals.py::show_record` prints subject + `initial_message_body` + **`marc_reply_body`** (the chair's reply), with `[t]` optionally showing the full thread. Production sees only subject + initial message at ingest (D17), and the label record stores **no provenance** for whether `[t]` was pressed — so per-ticket it is unknowable which tickets were labeled with extra context. A miss may therefore mean "the answer was only visible in Marc's reply", not "the model failed". Report recall with this stated; do not tune the prompt against these misses without re-reading them.

## Known risks

- Reciprocal complaints now classify as `desk_reject_appeal`. 9a assessed the template: the opening line fits, but the body is **verbatim policy text**, which for a requester *disputing the facts* ("my reviewers did submit") restates the rule that rejected them rather than answering — non-responsive, and readable as dismissive. Mitigated by D13.
- Follow-up comments reprocess the thread and overwrite extraction wholesale.
- The prompt drives QUERY, INTENT and the flag together; any change needs retrieval (hit@k) and intent regression checks. ⚠️ Per D15/D16 **neither check is currently runnable** — no valid intent gold, and no distiller in `run_eval.py`.
- `SubmissionDetails.tsx` renders nothing for a ticket whose only signal is the flag (fine until Phase 5 UI).
- `KEYWORD_RULES["desk_reject_appeal"]` is 4 literal phrases (`desk reject`, `desk-reject`, `appeal the desk`, `rejected for formatting`) and **no rule anywhere mentions "reciprocal"** — so on the ~1% distiller-failure path a reciprocal dispute almost certainly misses the intent entirely and falls to `cms_support` (`FALLBACK_INTENT`).

## Backlog / cleanup

- Stash: `parked: 3a keyword AND-group support` (restore with `git stash list`, then `git stash pop`).
- CLAUDE.md's config table is stale on **four** flags, not just one: `QUERY_STRATEGY` (says `prefix`, config.py is `distill`), `RETRIEVAL_BACKEND` (`bm25` → `fusion`), `MAX_RETRIEVED_CHUNKS` (`3` → `5`), and it omits `FAQ_ANSWER_CONFIDENCE_THRESHOLD` (0.85) entirely.
- `scripts/label_real_tickets.py` has its own duplicate `INTENT_DEFS` (line 74), and line 85 still carries the **pre-`beb5cf6`** `desk_reject_appeal` wording — no "waive", no "reciprocal-review duty grounds". If it is used to produce intent gold for Step 9b, it will label against a definition the live distiller no longer uses. Fix before 9b, not after.
- `data/eval/ground_truth.json` is 100% dead-vocabulary intents (D15) — needs relabeling or replacing before any intent metric means anything.
- 25 pre-existing test failures (test_received_at_source, test_tracing, test_routing_safety_floor, test_env_example_config, test_redraft_endpoint).
- `merge_batches.py` doesn't validate `appeal_reason` codes against the taxonomy.
- Offline mining scripts consume intent definitions; their outputs would change if re-run.

## Working rules

- Investigation first; stop-and-report gates before any behavior change. Wording gates for any prompt or definition edit.
- Claude Code never commits and never proposes git commands; Sahil commits manually on main.
- Tests only via `docker compose exec backend python -m pytest` (PowerShell). pytest may need ephemeral install in the container.
- Mutation testing: verify every mutation actually landed. Files are CRLF, so `$`-anchored sed silently no-ops.
- Heredocs can mangle backslashes in prompt strings; verify string content after writing.
- All searches respect .gitignore; never touch data/labeling/.
- Never hardcode model names.

## Changelog

- 8c52d2f: Step 1, `is_reject_appeal` helper.
- beb5cf6: Step 2, `desk_reject_appeal` definition + doc mirror.
- Step 3: 3b rejected after adversarial leaks; 3a stashed.
- df60479: Step 4, `RECIPROCAL_DISPUTE` parser.
- 77a6038: Step 5, field on both mirror sides.
- b14f590: Step 6, extractor pass-through.
- 77f72bb: Step 7, frontend type.
- 11cb466: Step 8, prompt activation.
- 2026-09-23: this log created.
- 2026-09-25: Step 9a read-only report (no code, no model calls). Added D13–D17; recorded the template assessment, the `SENSITIVE_INTENTS` hold mechanism, and three blockers found for 9b (dead intent gold, `run_eval.py` has no distiller, `data/eval_real/` absent here). Baseline commit for before/after = **8c6eb49** (last commit before `8c52d2f`).
- 2026-09-25: Step 9b harness built — `backend/scripts/reciprocal_dispute_eval.py` (+ `backend/tests/test_reciprocal_dispute_eval.py`, 25 tests). Added D18–D22 and corrected D10's baseline figure to 4,573. Dry run green end to end on SYNTHETIC labels: both arms 100 tickets / 100 mocked calls / cap enforced, compare printed all five metric blocks. **Zero real model calls made.** Blocked at the gate on D18 (stale backend image).
