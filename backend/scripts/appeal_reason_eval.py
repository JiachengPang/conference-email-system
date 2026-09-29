"""Appeal-reason classifier eval (reject-appeal Phase 2) + the D64 pool slice.

Scores ``classify_appeal_reason`` against Sahil's 124 Phase 2 labels, and — in
the SAME run, per D64's same-session rule — scores the reciprocal detector on
the pool tickets the LIVE distiller routes to ``desk_reject_appeal``.

A SIBLING of ``reciprocal_dispute_eval.py`` (Step 0 recommendation): it imports
that harness's ``CallBudget`` / ``BudgetExceeded`` / ``assert_gitignored`` /
``load_labels`` / ``_prf`` rather than copying them, the labeling tool's
``author_view`` (the exact requester-only view the labels were made on), and
production's ``build_transcript``.

INPUT SHAPE — the requester's PUBLIC messages only (``label_phase2.author_view``),
i.e. what the labeler saw (D63/D72). One message → the single-message path
(subject + body). Several → a transcript built by production's
``build_transcript``, which is what the classifier sees after follow-ups
(D66/D68). ⚠️ Production's transcript ALSO carries public agent replies; the
labels were made without them, so the eval matches the LABELS, not production
byte-for-byte. The same input goes to the distiller, detector and classifier.

PER TICKET (production order, D58/D68):
  1. distiller  → intent                                    (1 call)
  2. detector   → only if intent == desk_reject_appeal      (1 call)
  3. classifier → on EVERY ticket, "ungated", so the classifier's own quality
     is scored on all 124                                   (1 call)
The production gate is then SIMULATED from 1–2: the classifier would be called
only if intent ∈ REJECT_APPEAL_INTENTS and the detector did not say True. Both
the ungated and the gate-simulated predictions are scored.

GOLD — ``appeal_reason_codes`` mapped to registry names; ``n`` → ``[]`` (D67);
``r`` → ``[]`` for reasons (D60; the reciprocal box is gold for the DETECTOR).

PII — output carries ticket ids, label codes, intents, verdicts and reason
names. No subject, no body, no transcript, no model output text.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
for _p in (_BACKEND_DIR, _BACKEND_DIR / "scripts", _REPO_ROOT / "scripts" / "labeling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reciprocal_dispute_eval import (  # noqa: E402
    BudgetExceeded,
    CallBudget,
    _prf,
    assert_gitignored,
    load_labels,
)

_DEFAULT_MAX_CALLS = 300
NONE_LABEL = "(none)"


# ---------------------------------------------------------------------------
# Pure helpers — unit-tested, no app imports
# ---------------------------------------------------------------------------
def gold_reasons(record: dict, code_to_name: dict[str, str], order: list[str]) -> list[str]:
    """The gold reason list: codes → registry names, registry order. `n` → []."""
    if not record.get("is_reject_appeal"):
        return []
    names = {code_to_name[c] for c in (record.get("appeal_reason_codes") or [])}
    return [n for n in order if n in names]


def score_multilabel(
    pairs: list[tuple[list[str], list[str] | None]], reasons: list[str]
) -> dict:
    """Per-reason P/R/F1, exact match, and failures over (gold, pred) pairs.

    A ``None`` prediction (no usable answer) is scored as the empty set for the
    per-reason counts — it asserted no reason — but is ALSO counted in
    ``failed``, and never counts as an exact match, even against gold ``[]``:
    "failed" and "answered NONE" must not collapse (D59).
    """
    per = {r: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for r in reasons}
    exact = failed = 0
    for gold, pred in pairs:
        g = set(gold)
        p = set(pred or [])
        if pred is None:
            failed += 1
        elif g == p:
            exact += 1
        for r in reasons:
            if r in g:
                per[r]["support"] += 1
            if r in g and r in p:
                per[r]["tp"] += 1
            elif r in p:
                per[r]["fp"] += 1
            elif r in g:
                per[r]["fn"] += 1
    for r in reasons:
        c = per[r]
        c["precision"], c["recall"], c["f1"] = (round(x, 3) for x in _prf(c["tp"], c["fp"], c["fn"]))
    n = len(pairs)
    return {
        "n": n,
        "exact_match": exact,
        "exact_match_rate": round(exact / n, 3) if n else 0.0,
        "failed": failed,
        "per_reason": per,
    }


def confusion_pairs(pairs: list[tuple[list[str], list[str] | None]]) -> Counter:
    """(gold → predicted) disagreements, on tickets that got a real answer.

    For each ticket: every missed gold reason pairs with every spurious
    predicted reason. A miss with nothing spurious pairs with "(none)"
    (under-assignment); a spurious reason with no gold pairs "(none)" → it
    (over-assignment). Failed answers (None) are excluded — they are counted
    separately and would otherwise read as "predicted nothing".
    """
    out: Counter = Counter()
    for gold, pred in pairs:
        if pred is None:
            continue
        missed = sorted(set(gold) - set(pred))
        spurious = sorted(set(pred) - set(gold))
        if missed and spurious:
            for g in missed:
                for p in spurious:
                    out[(g, p)] += 1
        elif missed:
            for g in missed:
                out[(g, NONE_LABEL)] += 1
        elif spurious:
            for p in spurious:
                out[(NONE_LABEL, p)] += 1
    return out


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_input(record: dict, author_view, build_transcript, char_budget: int) -> dict:
    """email_data for one ticket from the requester-only view (see module doc)."""
    shown, _hidden = author_view(record)
    subject = record.get("subject") or ""
    if len(shown) <= 1:
        body = (shown[0].get("body") or "") if shown else ""
        return {"subject": subject, "body": body, "_shape": "single", "_shown": len(shown)}
    messages = [
        {
            "author_id": 1,
            "public": True,
            "plain_body": m.get("body") or "",
            "created_at": _parse_dt(m.get("created_at")),
        }
        for m in shown
    ]
    t = build_transcript(messages, char_budget=char_budget, requester_id=1)
    return {
        "subject": subject,
        "body": t.latest_requester_message,
        "thread_transcript": t.text,
        "_shape": "thread",
        "_shown": len(shown),
    }


# ---------------------------------------------------------------------------
# Transport: one budget + per-stage call and token counts
# ---------------------------------------------------------------------------
class _Meter:
    def __init__(self, budget: CallBudget) -> None:
        self.budget = budget
        self.calls: Counter = Counter()
        self.tokens: dict[str, Counter] = {}

    def wrap(self, stage: str, real, fake=None):
        async def counted(client, url, payload, headers=None):  # noqa: ANN001
            self.budget.spend()
            self.calls[stage] += 1
            resp = await (fake or real)(client, url, payload, headers)
            try:
                usage = resp.json().get("usage") or {}
            except Exception:  # noqa: BLE001 - metering must not break a call
                usage = {}
            bucket = self.tokens.setdefault(stage, Counter())
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), int):
                    bucket[key] += usage[key]
            return resp

        return counted


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": self._text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }


def _make_fake(stage: str):
    """Deterministic, deliberately varied answers per stage (dry run only)."""
    intents = ["desk_reject_appeal", "review_decision_appeal", "cms_support"]
    answers = [
        "APPEAL_REASONS: NONE",
        "APPEAL_REASONS: wrong_paper_review, reviewer_misunderstanding",
        "APPEAL_REASONS: general_dissatisfaction",
        "APPEAL_REASONS: made_up_reason",
    ]

    async def fake(client, url, payload, headers=None):  # noqa: ANN001
        user = payload["messages"][-1]["content"]
        k = int(hashlib.sha256(user.encode("utf-8")).hexdigest(), 16)
        if stage == "distiller":
            text = f"INTENT: {intents[k % 3]}\nCONFIDENCE: 0.8\nQUERY: appeal of a decision\n"
        elif stage == "detector":
            text = "RECIPROCAL_DISPUTE: " + ("YES" if k % 2 else "NO")
        else:
            text = answers[k % 4]
        return _FakeResponse(text)

    return fake


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
    from app.core.config import settings
    from app.pipeline import appeal_reason_classifier as arc_mod
    from app.pipeline import distiller as distiller_mod
    from app.pipeline import reciprocal_detector as det_mod
    from app.pipeline.appeal_reasons import LABEL_CODE_TO_NAME, REASON_NAMES
    from app.pipeline.taxonomy import REJECT_APPEAL_INTENTS
    from app.pipeline.thread_transcript import build_transcript
    from label_phase2 import author_view

    labels_path = Path(args.labels).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_gitignored(out_dir)

    if settings.MODEL_PROVIDER != "local":
        raise SystemExit(
            f"REFUSING TO RUN: MODEL_PROVIDER is '{settings.MODEL_PROVIDER}'; every "
            "stage would return None with 0 calls, indistinguishable from a result."
        )
    # Forced ON for this process only — never written to .env (the eval calls
    # classify_appeal_reason directly, but the state is recorded honestly).
    settings.APPEAL_REASON_CLASSIFIER_ENABLED = True

    records = load_labels(labels_path)
    if len(records) != 124 and not args.allow_partial:
        raise SystemExit(
            f"REFUSING TO RUN: {len(records)} usable labeled records, expected 124 "
            "(pass --allow-partial to score a partial file on purpose)."
        )
    reasons = list(REASON_NAMES)
    code_to_name = dict(LABEL_CODE_TO_NAME)

    # Worst case: every ticket reaches the detector.
    worst = 3 * len(records)
    if worst > args.max_calls:
        raise SystemExit(
            f"REFUSING TO RUN: up to {worst} calls possible but --max-calls="
            f"{args.max_calls}. Raise the cap deliberately."
        )

    meter = _Meter(CallBudget(args.max_calls))
    patched = {
        distiller_mod: ("distiller", distiller_mod.post_chat),
        det_mod: ("detector", det_mod.post_chat),
        arc_mod: ("classifier", arc_mod.post_chat),
    }
    for mod, (stage, real) in patched.items():
        mod.post_chat = meter.wrap(stage, real, _make_fake(stage) if args.dry_run else None)

    distiller = distiller_mod.EmailDistiller()
    rows: list[dict] = []
    aborted = None
    try:
        for i, record in enumerate(records, start=1):
            tid = str(record["ticket_id"])
            data = build_input(
                record, author_view, build_transcript, settings.THREAD_TRANSCRIPT_MAX_CHARS
            )
            email_data = {k: v for k, v in data.items() if not k.startswith("_")}
            transcript = email_data.get("thread_transcript")
            try:
                dres = await distiller.distill(
                    email_data["subject"], email_data["body"], transcript=transcript
                )
                intent = dres.intent if dres else None
                verdict = None
                asked_detector = intent == "desk_reject_appeal"
                if asked_detector:
                    verdict = await det_mod.detect_reciprocal_dispute(
                        subject=email_data["subject"], body=email_data["body"],
                        transcript=transcript,
                    )
                answer = await arc_mod.classify_appeal_reason(email_data, None)
            except BudgetExceeded as exc:
                aborted = str(exc)
                break
            gate_would_call = intent in REJECT_APPEAL_INTENTS and verdict is not True
            rows.append({
                "ticket_id": tid,
                "gold_codes": list(record.get("appeal_reason_codes") or []),
                "gold_is_appeal": bool(record.get("is_reject_appeal")),
                "gold_reciprocal": bool(record.get("is_reciprocal_label")),
                "gold_reasons": gold_reasons(record, code_to_name, reasons),
                "input_shape": data["_shape"],
                "requester_messages": data["_shown"],
                "intent": intent,
                "detector_asked": asked_detector,
                "detector_verdict": verdict,
                "gate_would_call": gate_would_call,
                "answer_ungated": answer,
                "answer_production": answer if gate_would_call else None,
            })
            print(f"  [{i}/{len(records)}] {tid}: intent={intent} det={verdict} "
                  f"gate={'call' if gate_would_call else 'skip'} answer={answer}", flush=True)
    finally:
        for mod, (_stage, real) in patched.items():
            mod.post_chat = real

    report = summarize(rows, reasons)
    report["manifest"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "aborted": aborted,
        "labels_file": labels_path.name,
        "labeled_records": len(records),
        "scored": len(rows),
        "calls_by_stage": dict(meter.calls),
        "calls_total": meter.budget.used,
        "max_calls": args.max_calls,
        "tokens_by_stage": {k: dict(v) for k, v in meter.tokens.items()},
        "model_provider": settings.MODEL_PROVIDER,
        "model_name": settings.LOCAL_MODEL_NAME,
        "temperature": settings.DRAFTER_TEMPERATURE,
        "seed": settings.DRAFTER_SEED,
        "query_strategy": settings.QUERY_STRATEGY,
        "classifier_prompt_sha256": hashlib.sha256(
            arc_mod._SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "detector_prompt_sha256": hashlib.sha256(
            det_mod._SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "distiller_prompt_sha256": hashlib.sha256(
            distiller_mod._SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
    path = out_dir / ("appeal_reason_eval_dryrun.json" if args.dry_run else "appeal_reason_eval.json")
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"report": report, "rows": rows}, fh, indent=2)
    print_report(report)
    print(f"\n  wrote {path}")
    return 0


def rescore(args: argparse.Namespace) -> int:
    """Re-score a SAVED run with the current post-processing — ZERO model calls.

    Valid only for rules applied AFTER parsing (e.g. D78's
    ``drop_redundant_fallback``): the saved ``answer_*`` fields are already the
    parsed, normalized lists, so re-applying a pure function of that list gives
    exactly what the run would have stored under the new code. Refuses if the
    saved run used a different classifier prompt — then the saved answers are
    not what the current prompt would have produced.
    """
    from app.pipeline import appeal_reason_classifier as arc_mod
    from app.pipeline.appeal_reasons import REASON_NAMES

    src = Path(args.rescore).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_gitignored(out_dir)
    saved = json.loads(src.read_text(encoding="utf-8"))
    manifest = saved["report"]["manifest"]
    current_sha = hashlib.sha256(arc_mod._SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if manifest.get("classifier_prompt_sha256") != current_sha:
        raise SystemExit(
            "REFUSING TO RESCORE: the saved run used a different classifier prompt "
            f"({str(manifest.get('classifier_prompt_sha256'))[:12]} vs {current_sha[:12]}); "
            "its answers are not what the current prompt would produce."
        )
    if manifest.get("dry_run"):
        raise SystemExit("REFUSING TO RESCORE: the saved run is a dry run (mocked answers).")

    rows = []
    changed = []
    for row in saved["rows"]:
        new = dict(row)
        new["answer_ungated"] = arc_mod.drop_redundant_fallback(row["answer_ungated"])
        new["answer_production"] = arc_mod.drop_redundant_fallback(row["answer_production"])
        if new["answer_ungated"] != row["answer_ungated"]:
            changed.append(row["ticket_id"])
        rows.append(new)

    report = summarize(rows, list(REASON_NAMES))
    report["manifest"] = {
        **manifest,
        "rescored_at": datetime.now(timezone.utc).isoformat(),
        "rescored_from": src.name,
        "rescore_rule": "drop_redundant_fallback (D78)",
        "rescore_calls": 0,
        "tickets_changed_by_rule": changed,
    }
    path = out_dir / "appeal_reason_eval_rescored_d78.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"report": report, "rows": rows}, fh, indent=2)
    print(f"RESCORE (0 model calls) of {src.name} with drop_redundant_fallback (D78)")
    print(f"  tickets whose answer changed: {len(changed)} {changed}")
    print_report(report)
    print(f"\n  wrote {path}")
    return 0


def summarize(rows: list[dict], reasons: list[str]) -> dict:
    ungated = [(r["gold_reasons"], r["answer_ungated"]) for r in rows]
    production = [(r["gold_reasons"], r["answer_production"]) for r in rows]
    appeals = [(r["gold_reasons"], r["answer_ungated"]) for r in rows
               if r["gold_is_appeal"] and not r["gold_reciprocal"]]
    r_rows = [r for r in rows if r["gold_reciprocal"]]
    det_rows = [r for r in rows if r["detector_asked"]]
    tp = sum(1 for r in det_rows if r["detector_verdict"] is True and r["gold_reciprocal"])
    fp = sum(1 for r in det_rows if r["detector_verdict"] is True and not r["gold_reciprocal"])
    fn = sum(1 for r in det_rows if r["detector_verdict"] is False and r["gold_reciprocal"])
    tn = sum(1 for r in det_rows if r["detector_verdict"] is False and not r["gold_reciprocal"])
    dp, dr, df = _prf(tp, fp, fn)
    return {
        "ungated": score_multilabel(ungated, reasons),
        "ungated_reason_coded_appeals_only": score_multilabel(appeals, reasons),
        "production_gate_simulated": score_multilabel(
            [(g, p) for (g, p), r in zip(production, rows) if r["gate_would_call"]], reasons),
        "gate": {
            "would_call": sum(r["gate_would_call"] for r in rows),
            "skipped_not_appeal_intent": sum(
                1 for r in rows if not r["gate_would_call"] and not r["detector_verdict"]),
            "skipped_reciprocal_true": sum(1 for r in rows if r["detector_verdict"] is True),
            "intent_counts": dict(Counter(str(r["intent"]) for r in rows)),
            "gold_reason_appeals_not_gated": sorted(
                r["ticket_id"] for r in rows
                if r["gold_reasons"] and not r["gate_would_call"]),
        },
        "confusion_ungated": {f"{g} -> {p}": n for (g, p), n in
                              confusion_pairs(ungated).most_common()},
        "reciprocal_check": {
            "r_tickets": len(r_rows),
            "r_nonempty_ungated_answer": sorted(
                r["ticket_id"] for r in r_rows if r["answer_ungated"]),
            "r_gate_would_call": sorted(r["ticket_id"] for r in r_rows if r["gate_would_call"]),
            "r_gate_would_call_nonempty": sorted(
                r["ticket_id"] for r in r_rows if r["gate_would_call"] and r["answer_ungated"]),
            "r_intents": dict(Counter(str(r["intent"]) for r in r_rows)),
            "detector_true_nonempty_answer": sorted(
                r["ticket_id"] for r in rows
                if r["detector_verdict"] is True and r["answer_ungated"]),
        },
        "detector_live_gate_pool": {
            "asked": len(det_rows),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "none": sum(1 for r in det_rows if r["detector_verdict"] is None),
            "precision": round(dp, 3), "recall": round(dr, 3), "f1": round(df, 3),
            "r_in_pool": len(r_rows),
            "r_never_gated": sorted(r["ticket_id"] for r in r_rows if not r["detector_asked"]),
            "false_positive_ids": sorted(
                r["ticket_id"] for r in det_rows
                if r["detector_verdict"] is True and not r["gold_reciprocal"]),
            "false_negative_ids": sorted(
                r["ticket_id"] for r in det_rows
                if r["detector_verdict"] is False and r["gold_reciprocal"]),
        },
    }


def print_report(rep: dict) -> None:
    def block(title, s):
        print(f"\n{title}  (n={s['n']}, exact match {s['exact_match']}/{s['n']} = "
              f"{s['exact_match_rate']}, failed answers {s['failed']})")
        print(f"  {'reason':26} {'sup':>4} {'tp':>4} {'fp':>4} {'fn':>4} {'P':>6} {'R':>6} {'F1':>6}")
        for name, c in s["per_reason"].items():
            print(f"  {name:26} {c['support']:>4} {c['tp']:>4} {c['fp']:>4} {c['fn']:>4} "
                  f"{c['precision']:>6} {c['recall']:>6} {c['f1']:>6}")

    print("=" * 72)
    block("A. UNGATED — classifier on all tickets", rep["ungated"])
    block("A. UNGATED — reason-coded appeals only (no n, no r)", rep["ungated_reason_coded_appeals_only"])
    block("A. PRODUCTION GATE SIMULATED — tickets the gate would call", rep["production_gate_simulated"])
    print("\nGATE:", json.dumps(rep["gate"], indent=2))
    print("\nCONFUSION (gold -> predicted), ungated:")
    for k, v in rep["confusion_ungated"].items():
        print(f"  {v:>3}  {k}")
    print("\nRECIPROCAL CHECK:", json.dumps(rep["reciprocal_check"], indent=2))
    print("\nB. DETECTOR, LIVE GATE, POOL SLICE:", json.dumps(rep["detector_live_gate_pool"], indent=2))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", help="Labels file (required unless --rescore).")
    p.add_argument("--out", required=True, help="Output dir (MUST be gitignored).")
    p.add_argument("--max-calls", type=int, default=_DEFAULT_MAX_CALLS)
    p.add_argument("--dry-run", action="store_true", help="Mocked model, 0 real calls.")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--rescore", metavar="SAVED_JSON",
                   help="Re-score a saved run with current post-processing; 0 calls.")
    args = p.parse_args(argv)
    if not args.rescore and not args.labels:
        p.error("--labels is required unless --rescore is given")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.rescore:
        return rescore(args)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
