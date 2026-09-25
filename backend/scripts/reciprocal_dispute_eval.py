"""Reciprocal-dispute eval (reject-appeal Phase 1, Step 9b).

Scores ``is_reciprocal_dispute`` against the hand-labeled appeal set, and
measures whether the two live prompt changes (`beb5cf6` desk_reject_appeal
definition, `11cb466` RECIPROCAL_DISPUTE block) regressed intent or retrieval.

Design notes that are load-bearing, not decoration:

INPUT SHAPE -- the distiller is fed ``subject`` + ``initial_message_body`` and
NOTHING else. That mirrors production ingest exactly: ``_ingest_new_ticket``
builds ``email_data`` with no ``thread_transcript`` key, so ``_compute`` passes
``transcript=None`` and the conversation branch never runs for a new ticket.
See reject_appeal.md D17. The labeler saw MORE than this (it also printed the
chair's reply, and optionally the full thread), so recall here has a ceiling
that is a property of the label process, not of the flag -- read misses with
that in mind.

TWO ARMS, ONE PROCESS -- the baseline arm does NOT check out `8c6eb49`. It
reconstructs that commit's ``_SYSTEM_PROMPT`` (including its own
``_INTENT_MENU``, built from that commit's ``INTENT_DEFS``) and swaps it into
the live module for the duration of the run. This is deliberate: a `git diff`
of `distiller.py` over the series shows the ONLY changes are the prompt
strings, the ``_RECIPROCAL_DISPUTE_RE`` pattern, the ``DistillResult`` field
and the parse block -- the HTTP payload, model params and retrieval path are
untouched. Swapping the prompt therefore isolates the independent variable to
exactly what changed, while a worktree would also vary the retriever code,
config defaults and installed deps. Running HEAD's parser against a baseline
prompt is correct, not a shortcut: a baseline prompt never emits a
RECIPROCAL_DISPUTE line, so the parser yields ``None`` -- the same observable
the baseline code produced when it had no parser at all.

The manifest records the resolved prompt's sha256 and length for each arm, so
"the arms really were different" is auditable after the fact rather than
assumed.

PII -- output carries ticket ids, the intent label, the flag, and retrieved
policy chunk ids. No subject, no body, no query text (distilled queries are
derived from the email and are treated as content), no names, no addresses.
Nothing is logged that is not written.

Usage (see reject_appeal.md for the approved command lines):
    python scripts/reciprocal_dispute_eval.py run --labels <path> \\
        --out <gitignored dir> --half 1 --arm current --dry-run
    python scripts/reciprocal_dispute_eval.py compare --labels <path> \\
        --out <gitignored dir> --half 1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# App imports are deliberately LAZY (inside the functions that need them).
# `prepare-baseline` has to run on the host, where git lives but the container's
# ML deps (sentence_transformers, faiss) do not -- a module-level
# `from app.pipeline.retriever import get_retriever` would make the host-only
# subcommand unimportable on the host. Nothing else here needs app code.

# The commit immediately BEFORE the reject-appeal series (before 8c52d2f).
BASELINE_REV = "8c6eb49"

# Fixed forever: changing it re-draws the halves and silently invalidates any
# comparison against an earlier run.
SPLIT_SEED = 20260925

# Separate from SPLIT_SEED on purpose: re-drawing a --sample must never move the
# halves, and changing the halves must never silently re-draw a sample. Fixed
# forever for the same reason SPLIT_SEED is.
SAMPLE_SEED = 20260926

# The label code that counts as a positive. Every other code (n/a/b/c/d/e/o) is
# a negative. Source: scripts/labeling/label_appeals.py REASONS.
POSITIVE_CODE = "r"

_DEFAULT_MAX_CALLS = 100


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a call that would exceed the cap is dispatched."""


class CallBudget:
    """Counts real model calls and refuses to let the run exceed ``max_calls``.

    The count is incremented at the transport seam (``post_chat``), not at the
    loop, so it measures calls actually dispatched rather than calls intended.
    ``spend`` raises before the call goes out -- a cap that trips after the
    request is already on the wire would not be a cap.
    """

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0

    def spend(self) -> None:
        if self.used + 1 > self.max_calls:
            raise BudgetExceeded(
                f"call cap reached: {self.used} call(s) made, "
                f"--max-calls={self.max_calls}; aborting before dispatch"
            )
        self.used += 1


# ---------------------------------------------------------------------------
# Labels + deterministic split
# ---------------------------------------------------------------------------
class MalformedLabelsFile(ValueError):
    """Parse failure reported by LINE NUMBER only.

    The offending line is deliberately never included. These are real ticket
    records, so a malformed one is still PII -- and a parse error is exactly the
    moment someone pastes the whole message into a bug report or a terminal
    someone else is watching. The line number is enough to find it.
    """


def _parse_labels_text(text: str) -> list[dict]:
    """Parse the labels file as JSONL, or as a JSON array if it looks like one.

    The real file is JSONL (one object per line). A JSON array is also accepted
    because the sniff is one character and the synthetic fixtures in the test
    suite are arrays -- keeping both means the harness reads whatever it is
    handed rather than making the caller convert.
    """
    # WHY THE ERRORS ARE RAISED OUTSIDE THE `except` BLOCKS, not with
    # `from None`: JSONDecodeError carries the ENTIRE document on its `.doc`
    # attribute. `from None` only sets __suppress_context__ (it stops the chain
    # being DISPLAYED) -- it leaves `.doc` reachable via `err.__context__.doc`,
    # so anything that introspects exception attributes (error reporters,
    # pytest --showlocals, a debugger) can still read every ticket record.
    # Raising after the handler has exited leaves __context__ itself None, so
    # the document is not reachable at all. Verified, not assumed: with
    # `from None` the secret was still in `__context__.doc`; this way there is
    # no __context__. (A formatted traceback never leaked it either way -- the
    # earlier comment claiming that was wrong.)
    if text.lstrip().startswith("["):
        records = None
        bad_lineno = None
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            bad_lineno = exc.lineno
        if bad_lineno is not None:
            raise MalformedLabelsFile(
                f"labels file: malformed JSON array at line {bad_lineno} "
                "(content withheld — it is ticket PII)"
            )
        if not isinstance(records, list):
            raise MalformedLabelsFile(
                "labels file: top-level JSON is not a list of records"
            )
        return records

    records = []
    # splitlines() handles \n, \r\n and a trailing newline identically, which
    # matters here: this repo is CRLF-prone (see the mutation-testing rule).
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = None
        decode_failed = False
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            decode_failed = True
        # Outside the handler -- see the note above on `.doc`.
        if decode_failed:
            raise MalformedLabelsFile(
                f"labels file: malformed JSON on line {lineno} "
                "(content withheld — it is ticket PII)"
            )
        if not isinstance(record, dict):
            raise MalformedLabelsFile(
                f"labels file: line {lineno} is not a JSON object "
                "(content withheld — it is ticket PII)"
            )
        records.append(record)
    return records


def load_labels(path: Path) -> list[dict]:
    """Read the labels file named on the command line. Never a hardcoded path.

    Accepts JSONL (the real file's format) or a JSON array; see
    ``_parse_labels_text``.

    Drops records that carry no usable label (deferred, or never labeled): they
    are neither positive nor negative, and folding them into "negative" would
    manufacture true negatives out of unlabeled rows.
    """
    records = _parse_labels_text(path.read_text(encoding="utf-8"))
    usable = [
        r
        for r in records
        if not r.get("deferred") and r.get("is_reject_appeal") is not None
    ]
    return usable


def is_positive(record: dict) -> bool:
    return record.get("appeal_reason") == POSITIVE_CODE


def stratified_halves(
    records: list[dict], seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """Split ticket ids into two halves with the same positive/negative mix.

    Deterministic by construction: each stratum is sorted by ticket id BEFORE
    shuffling, so the result depends only on the id set and the seed -- never on
    the order the records happen to sit in the file. Half 2 is the exact
    complement of half 1 (same stratum, everything after the split point), so
    the two are disjoint and together cover every usable record.
    """
    positives = sorted(str(r["ticket_id"]) for r in records if is_positive(r))
    negatives = sorted(str(r["ticket_id"]) for r in records if not is_positive(r))

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    p_cut = len(positives) // 2
    n_cut = len(negatives) // 2
    half1 = positives[:p_cut] + negatives[:n_cut]
    half2 = positives[p_cut:] + negatives[n_cut:]
    return sorted(half1), sorted(half2)


def stratified_sample(
    ids: list[str],
    by_id: dict[str, dict],
    n: int,
    seed: int = SAMPLE_SEED,
) -> list[str]:
    """Take ``n`` ids from ``ids``, keeping the positive/negative ratio.

    Same discipline as ``stratified_halves``: sort each stratum BEFORE shuffling
    so the result depends only on the id set, the count and the seed -- never on
    input order. Uses its OWN seed so that re-drawing a sample cannot silently
    move the halves, and vice versa.

    Allocation is proportional, with the positive count clamped to what exists.
    The negative count then cannot overflow its own stratum -- ``n_pos`` is at
    most ``len(positives)``, and ``n <= total`` is enforced above, so
    ``n_neg = n - n_pos <= total - n_pos <= len(negatives)``. An earlier version
    carried a "repair" branch for that case; an exhaustive check over every
    (total, positives, n) up to 200 found **zero** inputs that reach it, so it
    was dead code that a mutation could silently delete. The invariant is
    asserted instead, and the "always returns exactly n" property is pinned by
    test rather than by an unreachable branch.
    """
    positives = sorted(t for t in ids if is_positive(by_id[t]))
    negatives = sorted(t for t in ids if not is_positive(by_id[t]))
    total = len(positives) + len(negatives)
    if n > total:
        raise SystemExit(
            f"REFUSING TO RUN: --sample {n} exceeds the {total} tickets in this "
            "half. A sample cannot be larger than the set it is drawn from."
        )

    n_pos = round(n * len(positives) / total) if total else 0
    n_pos = min(n_pos, len(positives))
    n_neg = n - n_pos
    assert n_neg <= len(negatives), (
        f"allocation overflowed the negative stratum ({n_neg} > "
        f"{len(negatives)}) -- the proportional-allocation invariant broke"
    )

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return sorted(positives[:n_pos] + negatives[:n_neg])


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def git_available() -> bool:
    """Is there a usable ``git`` binary here?

    There is NOT one in the backend container: the image bakes the source and
    installs no git, and ``/app/backend`` has no work tree above it. Every
    git-dependent step therefore has to be done on the host and carried in --
    see ``prepare-baseline``. Discovered by running the suite in the container
    rather than on the host, which is the only place it would have shown up.
    """
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def inside_work_tree(path: Path) -> bool:
    """Is ``path`` inside a git work tree (i.e. could it ever be committed)?"""
    if not git_available():
        return False
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _git_show(rev: str, repo_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{rev}:{repo_path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_BACKEND_DIR.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show {rev}:{repo_path} failed: {result.stderr.strip()}")
    return result.stdout


def baseline_system_prompt() -> str:
    """Rebuild ``_SYSTEM_PROMPT`` exactly as it stood at ``BASELINE_REV``.

    The menu is rebuilt from that commit's own ``INTENT_DEFS`` -- using today's
    would leak half the change under test (`beb5cf6` edited the
    desk_reject_appeal definition, which reaches the prompt only via the menu).
    ``taxonomy.py`` has no imports, so exec-ing it is safe and cheap.
    """
    taxonomy_src = _git_show(BASELINE_REV, "backend/app/pipeline/taxonomy.py")
    namespace: dict = {}
    exec(compile(taxonomy_src, "<baseline taxonomy>", "exec"), namespace)  # noqa: S102
    intent_defs = namespace["INTENT_DEFS"]
    valid_intents = namespace["VALID_INTENTS"]
    menu = "\n".join(f"  - {i}: {intent_defs[i]}" for i in valid_intents)

    distiller_src = _git_show(BASELINE_REV, "backend/app/pipeline/distiller.py")
    match = re.search(r"_SYSTEM_PROMPT = \((.*?)\n\)\n", distiller_src, re.S)
    if match is None:
        raise RuntimeError(f"could not locate _SYSTEM_PROMPT at {BASELINE_REV}")
    prompt = eval(  # noqa: S307 - evaluating a string-literal concatenation we just read from git
        "(" + match.group(1) + ")", {"_INTENT_MENU": menu, "__builtins__": {}}
    )
    if not isinstance(prompt, str):
        raise RuntimeError("baseline _SYSTEM_PROMPT did not evaluate to a string")
    return prompt


def resolve_arm(arm: str, baseline_prompt_file: str | None = None) -> dict:
    """Return the system prompt for ``arm`` plus auditable provenance.

    ``baseline_prompt_file`` is how the baseline arm reaches a git-less
    container: ``prepare-baseline`` derives the prompt on the host and writes
    it, the file is copied in, and this reads it back. When git IS available the
    prompt is derived directly and no file is needed.
    """
    from app.pipeline import distiller as distiller_module

    if arm == "current":
        prompt = distiller_module._SYSTEM_PROMPT
        source = "HEAD (live module)"
    elif arm == "baseline":
        if baseline_prompt_file:
            prompt = Path(baseline_prompt_file).read_text(encoding="utf-8")
            source = f"{BASELINE_REV} (via --baseline-prompt-file)"
        elif git_available():
            prompt = baseline_system_prompt()
            source = f"{BASELINE_REV} (reconstructed in-process)"
        else:
            raise SystemExit(
                "REFUSING TO RUN: the baseline arm needs the prompt as it stood "
                f"at {BASELINE_REV}, but there is no git here (this is normal in "
                "the backend container). Run `prepare-baseline` on the host, copy "
                "the file in, and pass --baseline-prompt-file."
            )
        if "RECIPROCAL_DISPUTE" in prompt:
            raise SystemExit(
                "REFUSING TO RUN: the supplied baseline prompt ASKS for "
                "RECIPROCAL_DISPUTE, so it is not a baseline prompt. Both arms "
                "would measure the same thing."
            )
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown arm: {arm}")
    return {
        "arm": arm,
        "prompt_source": source,
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "asks_reciprocal_dispute": "RECIPROCAL_DISPUTE" in prompt,
        "_prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Output-directory guard
# ---------------------------------------------------------------------------
def assert_gitignored(out_dir: Path) -> None:
    """Refuse to write anywhere git would track.

    Output rows carry real ticket ids, so an un-ignored directory is one
    ``git add -A`` away from committing them.

    A path with no git work tree above it cannot be committed at all, so the
    guard is satisfied by construction there (this is the container's case:
    /app/backend has no .git). That is a real exemption, not a loophole -- the
    guard exists to stop a commit, and there is nothing to commit into.
    """
    if not inside_work_tree(out_dir if out_dir.exists() else out_dir.parent):
        print(
            f"  [gitignore guard] {out_dir} is not inside a git work tree — "
            "nothing here can be committed; guard satisfied."
        )
        return
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=str(_BACKEND_DIR.parent),
    )
    if result.returncode != 0:
        raise SystemExit(
            f"REFUSING TO RUN: output dir {out_dir} is not gitignored "
            "(git check-ignore says it is tracked or untracked-but-visible). "
            "Results carry real ticket ids -- point --out at an ignored path."
        )


# ---------------------------------------------------------------------------
# Mock transport (--dry-run)
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._text}}]}


def _make_dry_run_post_chat(prompt: str):
    """A stand-in that answers the way the ACTIVE prompt asks it to.

    It emits a RECIPROCAL_DISPUTE line only when the arm's prompt actually asks
    for one, so a dry run exercises the real per-arm difference (current arm
    parses a flag; baseline arm yields None) instead of faking both identically.
    """
    asks_flag = "RECIPROCAL_DISPUTE" in prompt

    async def _fake(client, url, payload, headers=None):  # noqa: ANN001
        lines = [
            "INTENT: desk_reject_appeal",
            "CONFIDENCE: 0.82",
            "QUERY: desk rejection appeal reciprocal review duty",
        ]
        if asks_flag:
            lines.append("RECIPROCAL_DISPUTE: YES")
        return _FakeResponse("\n".join(lines) + "\n")

    return _fake


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
async def run_arm(args: argparse.Namespace) -> int:
    from app.core.config import settings
    from app.pipeline import distiller as distiller_module
    from app.pipeline.retriever import get_retriever

    labels_path = Path(args.labels).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_gitignored(out_dir)

    records = load_labels(labels_path)
    half1, half2 = stratified_halves(records)
    half_ids = sorted(half1 if args.half == 1 else half2)
    by_id = {str(r["ticket_id"]): r for r in records}

    # --sample narrows the half BEFORE the cap check, so `--sample 50
    # --max-calls 50` is a legal pair. Default (None) = the whole half, so the
    # no-flag behavior is byte-for-byte what it was.
    sample_n = getattr(args, "sample", None)
    if sample_n is not None:
        selected_ids = stratified_sample(half_ids, by_id, sample_n)
    else:
        selected_ids = half_ids
    selected = [by_id[t] for t in selected_ids]

    if len(selected) > args.max_calls:
        raise SystemExit(
            f"REFUSING TO RUN: this run covers {len(selected)} tickets but "
            f"--max-calls={args.max_calls}. Raise the cap deliberately, or use "
            "--sample to narrow the set on purpose; this script will not "
            "silently truncate a sample."
        )

    # FOOTGUN GUARD. `EmailDistiller.distill` returns None immediately unless
    # MODEL_PROVIDER == "local". Without this check the eval would happily
    # produce a full set of all-None rows having made ZERO calls, which reads
    # exactly like "the flag never fires" and also like "the cap worked".
    # Applies to --dry-run too: a rehearsal that skips the code path under test
    # rehearses nothing.
    if settings.MODEL_PROVIDER != "local":
        raise SystemExit(
            "REFUSING TO RUN: MODEL_PROVIDER is "
            f"'{settings.MODEL_PROVIDER}', but the distiller only runs under "
            "'local'. Every row would come back None with 0 calls made, which "
            "is indistinguishable from a real negative result."
        )

    arm = resolve_arm(args.arm, getattr(args, "baseline_prompt_file", None))
    budget = CallBudget(args.max_calls)

    real_post_chat = distiller_module.post_chat
    fake_post_chat = _make_dry_run_post_chat(arm["_prompt"]) if args.dry_run else None

    async def _counted(client, url, payload, headers=None):  # noqa: ANN001
        budget.spend()
        if fake_post_chat is not None:
            return await fake_post_chat(client, url, payload, headers)
        return await real_post_chat(client, url, payload, headers)

    original_prompt = distiller_module._SYSTEM_PROMPT
    distiller_module._SYSTEM_PROMPT = arm["_prompt"]
    distiller_module.post_chat = _counted

    distiller = distiller_module.EmailDistiller()
    retriever = get_retriever()
    rows: list[dict] = []
    aborted = None

    try:
        for record in selected:
            ticket_id = str(record["ticket_id"])
            subject = record.get("subject") or ""
            body = record.get("initial_message_body") or ""
            try:
                # Production ingest shape: single message, transcript=None (D17).
                result = await distiller.distill(subject, body, transcript=None)
            except BudgetExceeded as exc:
                aborted = str(exc)
                break

            intent = result.intent if result else None
            flag = result.is_reciprocal_dispute if result else None
            method = "llm_distiller" if result and result.intent else "none"

            chunk_ids: list[str] = []
            if result is not None and result.queries:
                query = " ".join(result.queries)
                try:
                    chunks = await retriever.retrieve(
                        query, "", top_k=settings.MAX_RETRIEVED_CHUNKS, prior_intent=""
                    )
                    chunk_ids = [c.policy_id for c in chunks]
                except Exception:  # noqa: BLE001 - retrieval is not under test here
                    chunk_ids = []

            rows.append(
                {
                    "ticket_id": ticket_id,
                    "intent": intent,
                    "is_reciprocal_dispute": flag,
                    "retrieved_chunk_ids": chunk_ids,
                    "method": method,
                }
            )
    finally:
        distiller_module._SYSTEM_PROMPT = original_prompt
        distiller_module.post_chat = real_post_chat

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "half": args.half,
        "arm": arm["arm"],
        "prompt_source": arm["prompt_source"],
        "prompt_chars": arm["prompt_chars"],
        "prompt_sha256": arm["prompt_sha256"],
        "asks_reciprocal_dispute": arm["asks_reciprocal_dispute"],
        "dry_run": bool(args.dry_run),
        "calls_made": budget.used,
        "max_calls": args.max_calls,
        "aborted": aborted,
        "tickets_selected": len(selected),
        "tickets_scored": len(rows),
        "split_seed": SPLIT_SEED,
        # None = the whole half. Recorded so a later reader can tell a 50-ticket
        # noise run from a truncated 100-ticket one.
        "sample": sample_n,
        "sample_seed": SAMPLE_SEED if sample_n is not None else None,
        "half_size": len(half_ids),
        "sample_positives": sum(1 for t in selected_ids if is_positive(by_id[t])),
        # Pinned so a later reader can prove both arms ran the same config.
        "model_provider": settings.MODEL_PROVIDER,
        "model_name": settings.LOCAL_MODEL_NAME,
        "temperature": settings.DRAFTER_TEMPERATURE,
        "seed": settings.DRAFTER_SEED,
        "retrieval_backend": settings.RETRIEVAL_BACKEND,
        "top_k": settings.MAX_RETRIEVED_CHUNKS,
        "query_strategy": settings.QUERY_STRATEGY,
    }

    out_path = out_dir / f"{args.arm}_half{args.half}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"manifest": manifest, "rows": rows}, handle, indent=2)

    print(f"arm={args.arm} half={args.half} dry_run={bool(args.dry_run)}")
    print(f"  prompt: {arm['prompt_chars']} chars  sha256={arm['prompt_sha256'][:12]}")
    print(f"  asks RECIPROCAL_DISPUTE: {arm['asks_reciprocal_dispute']}")
    if sample_n is not None:
        pos = sum(1 for t in selected_ids if is_positive(by_id[t]))
        print(
            f"  SAMPLE: {sample_n} of {len(half_ids)} "
            f"({pos} r / {len(selected_ids) - pos} non-r, seed {SAMPLE_SEED})"
        )
    print(f"  tickets selected: {len(selected)}  scored: {len(rows)}")
    print(f"  CALLS MADE: {budget.used} / cap {args.max_calls}")
    if aborted:
        print(f"  ABORTED: {aborted}")
    print(f"  wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
def _load_arm(out_dir: Path, arm: str, half: int) -> dict:
    path = out_dir / f"{arm}_half{half}.json"
    if not path.exists():
        raise SystemExit(f"missing arm output: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return precision, recall, f1


async def compare(args: argparse.Namespace) -> int:
    from app.pipeline.classifier import IntentClassifier

    out_dir = Path(args.out).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()

    base = _load_arm(out_dir, "baseline", args.half)
    curr = _load_arm(out_dir, "current", args.half)

    if base["manifest"]["prompt_sha256"] == curr["manifest"]["prompt_sha256"]:
        print(
            "WARNING: both arms ran the SAME prompt (identical sha256). "
            "Any 'no regression' result here is vacuous."
        )

    records = load_labels(labels_path)
    truth = {str(r["ticket_id"]): is_positive(r) for r in records}

    base_rows = {r["ticket_id"]: r for r in base["rows"]}
    curr_rows = {r["ticket_id"]: r for r in curr["rows"]}
    shared = sorted(set(base_rows) & set(curr_rows))

    # --- flag accuracy (current arm) ------------------------------------
    tp = fp = fn = tn = 0
    none_pos = none_neg = 0
    missed: list[str] = []
    # IDS ONLY, never content -- these name real tickets for manual review.
    false_positives: list[str] = []
    non_r_ids: list[str] = []
    # INTERSECTION ONLY. With --sample the two arms can cover different ticket
    # sets (e.g. a 50-ticket noise run against the 100-ticket half-1 result).
    # Scoring every current row would then report flag accuracy over a different
    # population than the intent/retrieval sections, and the blocks below would
    # silently describe different tickets.
    for tid in shared:
        row = curr_rows[tid]
        gold = truth.get(tid)
        if gold is None:
            continue
        if not gold:
            non_r_ids.append(tid)
        flag = row["is_reciprocal_dispute"]
        if flag is None:
            # Tri-state: "asked, did not answer" is NOT a negative (D7).
            if gold:
                none_pos += 1
                missed.append(tid)
            else:
                none_neg += 1
            continue
        if flag and gold:
            tp += 1
        elif flag and not gold:
            fp += 1
            false_positives.append(tid)
        elif not flag and gold:
            fn += 1
            missed.append(tid)
        else:
            tn += 1
    precision, recall, f1 = _prf(tp, fp, fn)

    print("=" * 66)
    print(f"FLAG ACCURACY (current arm, half {args.half})")
    print("=" * 66)
    print(
        f"  COMPARED: {len(shared)} tickets "
        f"(baseline arm {len(base_rows)}, current arm {len(curr_rows)}"
        + (
            f"; {len(set(base_rows) ^ set(curr_rows))} not in both, excluded)"
            if set(base_rows) != set(curr_rows)
            else "; identical sets)"
        )
    )
    print(
        f"  sample: baseline={base['manifest'].get('sample')} "
        f"current={curr['manifest'].get('sample')}  (None = whole half)"
    )
    print(f"  precision {precision:.3f}   recall {recall:.3f}   F1 {f1:.3f}")
    print(f"  confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  None bucket (reported separately, NOT negatives):")
    print(f"    None on a true r ticket : {none_pos}")
    print(f"    None on a non-r ticket  : {none_neg}")
    print(f"  missed r ticket ids ({len(missed)}): {sorted(missed)}")
    # Added for the half-1 post-mortem: a false positive is a non-r ticket the
    # model called reciprocal, and the ONLY way to review those is by id.
    print(f"  false-positive ticket ids ({len(false_positives)}): {sorted(false_positives)}")
    print(f"  non-r ticket ids ({len(non_r_ids)}): {sorted(non_r_ids)}")

    # --- intent stability on non-r tickets -------------------------------
    non_r = [t for t in shared if not truth.get(t, False)]
    same = [t for t in non_r if base_rows[t]["intent"] == curr_rows[t]["intent"]]
    changed = sorted(set(non_r) - set(same))
    pct = 100.0 * len(same) / len(non_r) if non_r else 0.0
    print()
    print(f"INTENT STABILITY (non-r tickets, n={len(non_r)})")
    print(f"  unchanged: {len(same)} ({pct:.1f}%)")
    print(f"  changed ids ({len(changed)}): {changed}")

    # --- intent shift on r tickets ---------------------------------------
    r_ids = [t for t in shared if truth.get(t, False)]
    b_dra = sum(1 for t in r_ids if base_rows[t]["intent"] == "desk_reject_appeal")
    c_dra = sum(1 for t in r_ids if curr_rows[t]["intent"] == "desk_reject_appeal")
    bp = 100.0 * b_dra / len(r_ids) if r_ids else 0.0
    cp = 100.0 * c_dra / len(r_ids) if r_ids else 0.0
    print()
    print(f"INTENT SHIFT (r tickets, n={len(r_ids)}) -> desk_reject_appeal")
    print(f"  baseline: {b_dra} ({bp:.1f}%)")
    print(f"  current : {c_dra} ({cp:.1f}%)   delta {cp - bp:+.1f} pp")

    # --- retrieval stability ---------------------------------------------
    overlaps = []
    for tid in shared:
        a = base_rows[tid]["retrieved_chunk_ids"]
        b = curr_rows[tid]["retrieved_chunk_ids"]
        if not a and not b:
            continue
        union = set(a) | set(b)
        overlaps.append(len(set(a) & set(b)) / len(union) if union else 1.0)
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else float("nan")
    print()
    print(f"RETRIEVAL STABILITY (top-k Jaccard, n={len(overlaps)})")
    print(f"  mean overlap: {mean_overlap:.3f}")

    # --- keyword fallback, zero model calls -------------------------------
    classifier = IntentClassifier()
    hits = 0
    r_records = [r for r in records if is_positive(r) and str(r["ticket_id"]) in shared]
    for record in r_records:
        result = await classifier.classify(
            record.get("initial_message_body") or "", record.get("subject") or ""
        )
        if result.intent == "desk_reject_appeal":
            hits += 1
    kp = 100.0 * hits / len(r_records) if r_records else 0.0
    print()
    print(f"KEYWORD FALLBACK (zero model calls, r tickets n={len(r_records)})")
    print(f"  landed on desk_reject_appeal: {hits} ({kp:.1f}%)")
    print()
    print(f"calls made this compare: 0 (keyword classifier is offline)")
    return 0


# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Score one arm over one half.")
    run_p.add_argument("--labels", required=True, help="Path to the labels JSON.")
    run_p.add_argument("--out", required=True, help="Output dir (MUST be gitignored).")
    run_p.add_argument("--half", type=int, choices=(1, 2), required=True)
    run_p.add_argument("--arm", choices=("baseline", "current"), required=True)
    run_p.add_argument("--max-calls", type=int, default=_DEFAULT_MAX_CALLS)
    run_p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Score a deterministic stratified subset of N tickets from the "
        "half (same r/non-r ratio, own fixed seed). Default: the whole half.",
    )
    run_p.add_argument("--dry-run", action="store_true", help="Mocked model, 0 calls.")
    run_p.add_argument(
        "--baseline-prompt-file",
        default=None,
        help="Baseline prompt from `prepare-baseline` (required for --arm "
        "baseline where there is no git, e.g. inside the backend container).",
    )

    prep_p = sub.add_parser(
        "prepare-baseline",
        help=f"Host-only: write the {BASELINE_REV} prompt to a file for transport.",
    )
    prep_p.add_argument("--out", required=True, help="Output dir (MUST be gitignored).")

    cmp_p = sub.add_parser("compare", help="Aggregate metrics across both arms.")
    cmp_p.add_argument("--labels", required=True)
    cmp_p.add_argument("--out", required=True)
    cmp_p.add_argument("--half", type=int, choices=(1, 2), required=True)

    return parser.parse_args(argv)


def prepare_baseline(args: argparse.Namespace) -> int:
    """Derive the baseline prompt on the host so a git-less box can use it."""

    if not git_available():
        raise SystemExit(
            "prepare-baseline needs git. Run it on the host, not in the container."
        )
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_gitignored(out_dir)

    prompt = baseline_system_prompt()
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    path = out_dir / "baseline_prompt.txt"
    path.write_text(prompt, encoding="utf-8")

    print(f"baseline prompt from {BASELINE_REV}")
    print(f"  chars : {len(prompt)}")
    print(f"  sha256: {digest}")
    print(f"  asks RECIPROCAL_DISPUTE: {'RECIPROCAL_DISPUTE' in prompt}")
    print(f"  wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return asyncio.run(run_arm(args))
    if args.command == "prepare-baseline":
        return prepare_baseline(args)
    return asyncio.run(compare(args))


if __name__ == "__main__":
    raise SystemExit(main())
