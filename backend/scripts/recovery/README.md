# `scripts/recovery/` — one-off recovery & data-fix scripts

Scripts in this folder are **operational one-offs**: incident recovery, data
repair, and diagnostic sweeps run by hand when something went wrong. They are
deliberately kept apart from `backend/scripts/` (seeding, eval harnesses,
retrieval ablations, corpus tooling), which is *regular pipeline tooling* that
gets run repeatedly as part of normal development.

## Rules for anything added here

1. **Not part of the running app.** Nothing in `app/` imports from this folder.
   These are standalone `python scripts/recovery/<name>.py` entry points; the
   FastAPI app, the pipeline modules (classifier / retriever / router / drafter),
   and the scheduler never reference them.
2. **Reuse, never re-implement.** Auth, clients and config come from the real
   modules (e.g. `app.integrations.zendesk.credential_provider`,
   `app.core.config.settings`). A recovery script must not carry its own copy of
   OAuth logic or credentials.
3. **Read-only by default.** A diagnostic/discovery script must not write to any
   external system. Write-capable repair scripts are separate files, are named
   so the intent is obvious, and are gated on a human having reviewed the
   discovery report first.
4. **Output is untracked.** Reports land in `scripts/recovery/output/`, which is
   gitignored — these reports quote real ticket content (PII) and must never be
   committed.
5. **Tested.** Even a one-off gets a test under
   `backend/tests/recovery/` with the external API mocked, because these
   scripts run against production data exactly once, under time pressure, and
   there is no staging rehearsal.

## Current contents

| Script | Mode | Purpose |
|---|---|---|
| `find_internal_notes.py` | **read-only** | Find comments an agent posted as *internal notes* (`public: false`) that were meant to be public replies, within the last N days. Produces a Markdown report for human review plus a JSON report a future republish script can consume without re-scraping. Writes nothing to Zendesk. |
| `republish_internal_notes.py` | **writes to Zendesk** — gated on `--confirm-author-id` | Turns the confirmed findings from a `find_internal_notes.py` JSON report into real public replies (Zendesk cannot flip an existing comment's visibility, so the fix is a new public comment). A bare run only resolves and prints `whoami` and stops; passing `--confirm-author-id <id>` matching that identity processes the batch. Comment text is always re-fetched fresh (the report stores only a 200-char preview). Resumable via a JSONL state file so a re-run never double-posts; never touches ticket status. |
| `backfill_received_at.py` | **dry-run by default** — `--execute` to write, `--yes` to skip the typed confirmation | Repairs `emails.received_at`, which recorded when our poller **inserted the row** rather than when the requester opened the ticket, so historical tickets all appeared to arrive on the day we imported them. Copies the already-correct `zendesk_created_at` into it — a pure in-place UPDATE, no Zendesk API calls. The dry run reports the affected count, a drift histogram and every long-drift outlier for eyeball re-confirmation. `--execute` writes and `fsync`s a timestamped rollback JSON (`id` + prior `received_at`) **before** the UPDATE, then verifies the rowcount and drift inside the same transaction and commits only if both are clean. Rows with a NULL `zendesk_created_at` are never matched, and no column other than `received_at` is written. |
| `appeal_sheet_rules.py` | **pure — no I/O, no Zendesk, no database** | Rules for importing the APC-approved Reject Appeals sheet (reject_appeal.md D195–D201): strict CSV parser, row rules, text guards, status and changed-ticket rules, the ids-and-hashes manifest and plan file, and the reconcile matcher. The import script that uses it (Part 2b) is not built yet. Nothing here prints text; text fields are excluded from `repr`. |
