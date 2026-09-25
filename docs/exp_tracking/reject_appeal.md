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
| 9b | Build and run eval: flag accuracy, intent + retrieval regression, keyword fallback | — | **Half 1 RUN (100+100 calls). Flag P=.871/R=.964, but see D26 — the flag never fires outside `desk_reject_appeal`. Analysis done; numbers NOT yet interpretable (D29/D30: noise baseline outstanding)** |
| 9b-noise | Noise baseline: current arm re-run, same prompt/seed | — | **Harness ready** (`--sample 50`, D31/D32). Commands in D34. **Not yet run** — required by D30 before any half-1 number is trusted |
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

**D18. ~~⚠️ The running backend image PREDATES `11cb466`~~ — RESOLVED 2026-09-25.** Rebuilt (after the `sqlalchemy[asyncio]` fix in Backlog); the container now reports a **5,371-char** prompt with `asks flag: True`, and the staleness detector passes. The original finding is kept below because the *mechanism* recurs on every rebuild-less deploy. Original: Verified in the container: `_SYSTEM_PROMPT` is 4,604 chars and `"RECIPROCAL_DISPUTE" in prompt` is **False**; `distiller.py` has the parser and the new taxonomy definition but not the prompt block. The image bakes source with **no volume mount**, so this is the project's recurring trap, third occurrence. Running the eval here would return the flag as None for all 100 tickets and read as *"the flag does not work"* rather than *"the image is stale"*. Guarded by `test_live_module_is_at_head_not_a_stale_image`, which is currently the suite's one intended failure. Unblock with `docker compose build backend` — ⚠️ note that recreating the backend container auto-runs `alembic upgrade head` on the demo DB, so that rebuild is Sahil's call, not the harness's.

**D19. Eval arms are swapped IN-PROCESS, not via a worktree.** `git diff 8c6eb49..HEAD -- distiller.py` is additive only (prompt strings, `_RECIPROCAL_DISPUTE_RE`, the `DistillResult` field, the parse block) — the HTTP payload, model params and retrieval path are untouched. So overriding `_SYSTEM_PROMPT` isolates the independent variable exactly, while a worktree would also vary retriever code, config defaults and installed deps — the confounds the plan explicitly excludes. Running HEAD's parser over a baseline prompt is correct, not a shortcut: no flag line → `None`, the same observable the baseline code gave with no parser. Both arms' prompt sha256 + length land in the manifest, so "the arms really differed" is auditable; `compare` warns loudly if the two shas match.

**D20. The container has no `git` and no work tree, so git work is split out.** `prepare-baseline` runs on the **host** (writes `baseline_prompt.txt` + sha), the file is copied in, and `run --arm baseline --baseline-prompt-file` consumes it. The gitignore guard treats "no work tree above this path" as satisfied-by-construction — the guard exists to prevent a *commit*, and there is nothing to commit into. App imports in the script are **lazy** so the host-only subcommand does not need the container's ML deps. Found by running the suite in the container, which is the only place it surfaces.

**D21. The harness refuses to run under a non-`local` `MODEL_PROVIDER`.** `distill()` returns None immediately otherwise, so the eval would emit a full set of all-None rows having made **zero** calls — indistinguishable from both a real negative result and a working call cap. Caught when the hermetic conftest (which pins `fallback`) made a dry-run report 0 calls against 100 rows.

**D22. ⚠️ Label/production information asymmetry — a recall CEILING, not a flag defect.** `label_appeals.py::show_record` prints subject + `initial_message_body` + **`marc_reply_body`** (the chair's reply), with `[t]` optionally showing the full thread. Production sees only subject + initial message at ingest (D17), and the label record stores **no provenance** for whether `[t]` was pressed — so per-ticket it is unknowable which tickets were labeled with extra context. A miss may therefore mean "the answer was only visible in Marc's reply", not "the model failed". Report recall with this stated; do not tune the prompt against these misses without re-reading them.

**D23. The labels file is JSONL, not a JSON array, and the loader accepts both.** The first dry run on the real file died in `load_labels` with `Extra data: line 2 column 1` — `json.load` had been written against the synthetic array fixtures. `_parse_labels_text` now sniffs one character (`[` → array, else JSONL), skips blank lines, and handles CRLF via `splitlines()`. A malformed line raises `MalformedLabelsFile` naming **only the line number** — the line itself is a real ticket record, and a parse error is exactly when someone pastes the message into a bug report.

**D24. ⚠️ `raise ... from None` does NOT protect PII — raise AFTER the handler exits.** `JSONDecodeError` keeps the entire input on its `.doc` attribute. `from None` only sets `__suppress_context__`, which suppresses *display* of the chain; `err.__context__.doc` still hands out the whole labels file to anything that introspects exception attributes (error reporters, `pytest --showlocals`, a debugger). Raising once the `except` block has exited leaves `__context__` itself `None`, so the document is unreachable. **Measured, not argued:** with `from None` the secret was still in `__context__.doc`; a formatted traceback leaked it in *neither* case, so the original "chained traceback is a PII leak" comment in the code was simply wrong and has been corrected. **Use this pattern for any future parse error over PII input.**

**D25. A mutation SURVIVED and exposed a vacuous test — worth recording as method.** The first `from None` test asserted `__cause__ is None`, which is true whether or not the protection is present: implicit chaining sets `__context__`, never `__cause__`. Its backup assertion checked `str(__context__)`, which never contains the document either. So the test passed both ways and the mutation removing the protection survived. Fixed by asserting `__context__ is None` on both raise sites; both mutations now fail, with disjoint failures. Second instance in this project of "mutation-verified" meaning *found during the pass* rather than *confirmed after* (see 2026-08-06) — treat a clean first mutation pass as the exception.

**D26. ⚠️ THE FLAG NEVER FIRES OUTSIDE `desk_reject_appeal` — it is bounded by intent accuracy, not independent of it.** Cross-tab on half 1: **62 flag=True, ALL 62 classified `desk_reject_appeal`; zero True on any other intent.** 69 tickets were DRA, so the flag discriminates *within* DRA (62 yes / 7 no) but never *outside* it. Consequence: **both error modes are intent errors, not judgment errors.** All **8 FPs** were DRA in BOTH arms (the intent was wrong; the flag then agreed with it), and both **FNs** (18718, 19147) were `reviewer_assignment` in the current arm, so the flag could not have fired whatever the model thought. Precision 0.871 / recall 0.964 therefore measure the *intent* classifier's behavior on DRA as much as the flag. **This weakens D1's premise** that the flag gives Phase 5 a signal independent of intent — on this evidence it is close to a strict refinement of `intent == desk_reject_appeal`. Re-test on half 2 before building the bulk-reply queue on it.

**D27. ⚠️ Retrieval churn is NOT explained by intent change — it is the prompt perturbing the QUERY lines.** Mean top-k Jaccard 0.445, but the split by intent stability is flat: **changed 0.414 vs unchanged 0.450**, and **70 of the 86 same-intent tickets still moved chunks**. The premise that "retrieval is intent-aware so an intent change could explain it" does **not hold in distill mode**: `orchestrator._compute` sets `retrieval_intent = ""` when distilled queries exist (E001), `INTENT_PRIOR_ENABLED=False` (E010), and the harness mirrors that exactly (`retrieve(query, "", prior_intent="")`). Retrieval depends **only on the distilled QUERY text**, so the churn means the +798-char prompt changed what queries the model writes — a real side effect of a block that was scoped to add a judgment line, not to touch QUERY. Churn is concentrated where the new block is most relevant: **mean Jaccard r=0.336 vs non-r=0.584**. ⚠️ At `top_k=3` Jaccard can only be 0, 0.2, 0.5 or 1.0, so the metric is coarse.

**D28. ⚠️ `MAX_RETRIEVED_CHUNKS=3` in this eval — prod runs 5.** `config.py` defaults to 5 (2026-07-29) but `backend/.env` line 20 pins **3**, and both arms' manifests record `top_k: 3`. The numbers above are therefore NOT at the production retrieval breadth, and the Jaccard buckets are coarser than they would be at k=5. Decide deliberately whether half 2 runs at 3 (comparable to half 1) or 5 (comparable to prod) — it cannot be both.

**D29. ⚠️ Determinism is REQUESTED but UNPROVEN — `post_chat` can silently drop `temperature`.** Both arms pinned `temperature=0.0`, `seed=7`, same model, same backend/top-k/strategy (manifests match on every field). But `app/pipeline/openai_compat.post_chat` retries a 400 that names `temperature` by **deleting the parameter entirely** — precisely the reasoning-model path — and it mutates the payload *after* the manifest values are read. **So the manifest records what was requested, not what went on the wire.** `seed` is never dropped, but honoring it is best-effort on an OpenAI-compatible endpoint. Proving which happened needs a real call, so it is unresolved here. This is why D30 is not optional.

**D30. A noise baseline IS required before any of these numbers are interpreted — recommend re-running the CURRENT arm, not the baseline.** Every delta so far is prompt-effect and run-to-run variance confounded, and D29 means variance cannot be assumed to be zero. Re-running the **current** arm (not the old prompt, as first suggested) costs the same and yields strictly more: it measures **flag reproducibility** — whether a ticket's YES/NO flips between identical runs — which a baseline re-run cannot, since the baseline prompt never emits the line. Size: **50 tickets** (~50 Jaccard samples gives SE ≈ 0.05 on the mean, enough to separate 0.445 from ~0.9; the non-r intent subgroup will be thin at ~22, so treat the overall churn rate as the primary read). ⚠️ Two operational gotchas: (a) `run` writes `{arm}_half{half}.json`, so a second current run **must** use a different `--out` or it overwrites the existing result; (b) 50 requires a small `--sample N` addition, because the harness deliberately refuses to truncate a half (the guard that stops a silent partial sample) — **without that flag the smallest no-code-change run is the full 100.** To diff run-1 against run-2, copy run-1's `current_half1.json` into the new dir as `baseline_half1.json`; `compare` will emit "both arms ran the SAME prompt (identical sha256)", which for a noise run is the **expected confirmation**, not an error.

**D31. `--sample N` added for the noise run; default is unchanged.** Deterministic stratified subset of the chosen half, own `SAMPLE_SEED = 20260926` (**separate from `SPLIT_SEED`** so re-drawing a sample can never move the halves and vice versa), same r/non-r ratio as the half — 50 of half 1 = **28 r / 22 non-r**. Applied **BEFORE** the call-cap check, so `--sample 50 --max-calls 50` is legal; without that ordering the flag would abort on exactly the run it exists for. Oversize `N` fails loudly. Manifest records `sample`, `sample_seed`, `half_size`, `sample_positives`, so a 50-ticket noise run is distinguishable from a truncated 100. `--sample` omitted ⇒ the whole half, byte-for-byte as before (pinned by test). `top_k` stays **3** (D28) so results remain comparable to half 1.

**D32. `compare` now scores the INTERSECTION only, and prints how many tickets that was.** With `--sample` the arms cover different ticket sets; the flag-accuracy block iterated every current row while the intent/retrieval blocks used the intersection, so one report's sections would have described different populations. Header now prints `COMPARED: N tickets (baseline arm A, current arm B; …)` plus each arm's `sample` value.

**D33. ⚠️ TWO more mutations survived — one dead branch, one nested-fixture blind spot.** (a) `stratified_sample`'s "repair" branch for a rounding overflow was **unreachable**: an exhaustive sweep of every `(total, positives, n)` up to 200 found **zero** inputs that reach it, because `n_pos` is clamped to `len(positives)` and `n <= total` is enforced, so `n_neg <= len(negatives)` always. Replaced with an `assert` carrying the proof; the "always returns exactly N" property is pinned by test instead of by a branch that can never run. (b) `test_compare_scores_only_the_intersection` originally made the current arm a **strict subset** of the baseline arm, which makes "iterate the intersection" and "iterate all current rows" produce identical output — the mutation swapping them survived. Fixed by making the sets **overlap without nesting** (60/60 with 20 shared) and asserting `TP+FP == 20`. Both now fail. **Third instance of a surviving mutation exposing a weak test rather than weak code (see D25, 2026-08-06) — when a fixture nests, it cannot discriminate.**

**D34. Exact noise-run commands (50 calls, NOT yet run).** The container holds the only copy of the `--sample` script (no source mount; `docker cp`'d, sha256-verified) — **do not rebuild or restart the backend before running these**, or the flag disappears.
```
# 1. 50-ticket current-arm re-run into a NEW dir (never reuse /tmp/eval_out —
#    `run` writes {arm}_half{half}.json and would overwrite the half-1 result)
docker compose exec -T backend python scripts/reciprocal_dispute_eval.py run \
  --labels /tmp/labels.json --out /tmp/eval_noise --half 1 --arm current \
  --sample 50 --max-calls 50

# 2. Bring half 1's current output in as the comparison arm. Naming it
#    "baseline_*" is what `compare` expects; both arms are the SAME prompt, so
#    its "identical sha256" warning is the EXPECTED confirmation, not an error.
docker compose exec -T backend cp /tmp/eval_out/current_half1.json \
  /tmp/eval_noise/baseline_half1.json

# 3. Compare — 0 model calls. Expect "COMPARED: 50 tickets (baseline arm 100,
#    current arm 50; 50 not in both, excluded)".
docker compose exec -T backend python scripts/reciprocal_dispute_eval.py compare \
  --labels /tmp/labels.json --out /tmp/eval_noise --half 1
```
Read it as: **intent churn and mean Jaccard here are the NOISE FLOOR.** Compare against half 1's 20.5% non-r intent churn and 0.445 mean Jaccard — if the floor is close to those, the prompt effect claimed in D27 is not established. The flag's own run-to-run flips are visible only because the re-run is the *current* arm (D30).

## Known risks

- Reciprocal complaints now classify as `desk_reject_appeal`. 9a assessed the template: the opening line fits, but the body is **verbatim policy text**, which for a requester *disputing the facts* ("my reviewers did submit") restates the rule that rejected them rather than answering — non-responsive, and readable as dismissive. Mitigated by D13.
- Follow-up comments reprocess the thread and overwrite extraction wholesale.
- The prompt drives QUERY, INTENT and the flag together; any change needs retrieval (hit@k) and intent regression checks. ⚠️ Per D15/D16 **neither check is currently runnable** — no valid intent gold, and no distiller in `run_eval.py`.
- `SubmissionDetails.tsx` renders nothing for a ticket whose only signal is the flag (fine until Phase 5 UI).
- `KEYWORD_RULES["desk_reject_appeal"]` is 4 literal phrases (`desk reject`, `desk-reject`, `appeal the desk`, `rejected for formatting`) and **no rule anywhere mentions "reciprocal"** — so on the ~1% distiller-failure path a reciprocal dispute almost certainly misses the intent entirely and falls to `cms_support` (`FALLBACK_INTENT`).

## Backlog / cleanup

- ⚠️ **(1) THE SERVER NEEDS THE SQLALCHEMY FIX BEFORE ITS NEXT REBUILD.** `pyproject.toml` asked for bare `sqlalchemy>=2.0`. SQLAlchemy **2.0.52** shipped `greenlet` unconditionally on x86_64 via a `platform_machine` marker; **2.1.1 removed that line**, leaving greenlet only under the `asyncio` extra. The Dockerfile runs an unpinned `pip install -e .`, so a rebuild floated 2.0.52 → 2.1.1, greenlet vanished, and the container crashlooped on `alembic upgrade head` with *"The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed."* Fixed as **`sqlalchemy[asyncio]>=2.0,<2.1`** — the extra declares the real dependency (it was only ever arriving by accident), the ceiling keeps the resolve on the 2.0.x line that has been in production all along. **AAAI prod is still on an old image and will hit this the moment it rebuilds** — ship the pyproject change before any prod rebuild, not after.
- ⚠️ **(2) No lockfile — any rebuild can float any dependency.** The image installs from `pyproject.toml` ranges with no lock and no hashes, so the dependency set is whatever PyPI resolves on build day. Item (1) is one instance of that, caught only because it crashed loudly on startup; a float that changes *behavior* rather than breaking imports would ship silently. The `<2.1` ceiling closes this one package, not the class. A real fix is a lockfile (pip-tools / uv / `pip freeze` constraints) referenced from the Dockerfile. Until then, treat every `docker compose build backend` as potentially changing dependency versions, and re-run the suite after one.
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
- 2026-09-25: backend startup crash fixed — `sqlalchemy` → **`sqlalchemy[asyncio]>=2.0,<2.1`** in `pyproject.toml` (greenlet stopped arriving transitively when SQLAlchemy floated 2.0.52 → 2.1.1). Rebuilt: resolves to **2.0.54 + greenlet 3.5.6**, alembic + uvicorn start clean, 21 passed / 4 skipped (the 4 are `@needs_git`, unrunnable in a git-less container by design — see D20). D18 resolved as a side effect (image now at HEAD: 5,371-char prompt, `asks flag: True`). Two backlog items added: the server fix, and the no-lockfile exposure.
- 2026-09-25: `--sample N` added (D31) + `compare` made intersection-safe (D32); commands for the 50-call noise run recorded in D34, **not run**. Tests 36 → **48 collected: 44 passed / 4 skipped**. Two mutations survived and were fixed (D33): an unreachable rounding-repair branch, and a nested-set compare fixture that could not discriminate. `top_k` held at 3. No model calls; container not rebuilt.
- 2026-09-25: half-1 results analysed (**zero model calls**; read only the two result JSONs). Added D26–D30. `compare` extended to print **false-positive and non-r ticket ids (ids only)** so FPs are reviewable without opening labels. Headline: flag P=.871/R=.964/F1=.915, 0 in the None bucket — but all 8 FPs and both FNs trace to the **intent**, not the judgment (D26). Keyword fallback on r tickets: **17.9%** (10/56) land on `desk_reject_appeal`. Intent shift on r tickets 89.3% → 96.4% (+7.1pp).
- 2026-09-25: `load_labels` fixed for JSONL (the real file's format; the array-only loader died on line 2). Added D23–D25. Tests 25 → **36 collected: 32 passed / 4 skipped** (`@needs_git`, unrunnable in the container by design). Script + tests `docker cp`'d into the running container — **image deliberately NOT rebuilt**; host and container copies verified identical by sha256. Real labels file never opened.
- 2026-09-25: Step 9b harness built — `backend/scripts/reciprocal_dispute_eval.py` (+ `backend/tests/test_reciprocal_dispute_eval.py`, 25 tests). Added D18–D22 and corrected D10's baseline figure to 4,573. Dry run green end to end on SYNTHETIC labels: both arms 100 tickets / 100 mocked calls / cap enforced, compare printed all five metric blocks. **Zero real model calls made.** Blocked at the gate on D18 (stale backend image).
