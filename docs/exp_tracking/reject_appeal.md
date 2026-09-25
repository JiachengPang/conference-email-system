# Reject-Appeal Handling: Decision Log & Status

Owner: Sahil · Last updated: 2026-09-23 · **Current position: Phase 1, Step 9a (next)**

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
| 9a | Read-only report: template impact + eval plan | — | Next |
| 9b | Build and run eval: flag accuracy, intent + retrieval regression, keyword fallback | — | Pending |
| 9c | Hold reciprocal cases from auto-template, if 9a recommends | — | Pending |
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

**D10. Prompt placement.** The flag gets its own block after the identification lines and before the injection guard. It uses YES/NO, not the identification block's NONE convention (NONE parses as unknown). Four confusable NO cases are spelled out. Prompt grew 4,604 → 5,371 chars (+16.7%).

**D11. No deploy before the Step 9 eval clears.** The deploy is invisible to users: it records the flag on new emails and reroutes reciprocal complaints to `desk_reject_appeal`. Its value is collecting real data. Deploying Phases 1 and 2 together is an acceptable alternative.

**D12. Commits 5/6 re-scoped.** The drift guards require both sides of the extraction mirror to change together, so commit 5 added the field to both sides and commit 6 did the pass-through.

## Known risks

- Reciprocal complaints now classify as `desk_reject_appeal` and get that intent's template. Step 9a checks whether the template suits them.
- Follow-up comments reprocess the thread and overwrite extraction wholesale.
- The prompt drives QUERY, INTENT and the flag together; any change needs retrieval (hit@k) and intent regression checks.
- `SubmissionDetails.tsx` renders nothing for a ticket whose only signal is the flag (fine until Phase 5 UI).

## Backlog / cleanup

- Stash: `parked: 3a keyword AND-group support` (restore with `git stash list`, then `git stash pop`).
- CLAUDE.md lists `QUERY_STRATEGY` default as `prefix`; config.py is now `distill`.
- `scripts/label_real_tickets.py` has its own duplicate `INTENT_DEFS`, now out of sync with taxonomy.py.
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
