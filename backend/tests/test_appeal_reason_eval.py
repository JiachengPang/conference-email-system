"""Scoring helpers of scripts/appeal_reason_eval.py (pure; no model calls).

The eval's numbers are only as good as these functions, and a metric bug reads
exactly like a model result — so they get their own tests on synthetic data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_SCRIPT = _BACKEND_DIR / "scripts" / "appeal_reason_eval.py"


@pytest.fixture(scope="module")
def ev():
    sys.path.insert(0, str(_BACKEND_DIR / "scripts"))
    spec = importlib.util.spec_from_file_location("appeal_reason_eval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REASONS = ["a_r", "b_r", "c_r"]
CODES = {"a": "a_r", "b": "b_r", "c": "c_r"}


def test_gold_maps_codes_to_names_in_registry_order(ev):
    rec = {"is_reject_appeal": True, "appeal_reason_codes": ["c", "a"]}
    assert ev.gold_reasons(rec, CODES, REASONS) == ["a_r", "c_r"]


def test_gold_for_a_not_appeal_is_empty(ev):
    assert ev.gold_reasons({"is_reject_appeal": False, "appeal_reason_codes": []}, CODES, REASONS) == []


def test_gold_for_reciprocal_only_is_empty(ev):
    rec = {"is_reject_appeal": True, "appeal_reason_codes": [], "is_reciprocal_label": True}
    assert ev.gold_reasons(rec, CODES, REASONS) == []


def test_per_reason_counts_on_a_multilabel_case(ev):
    s = ev.score_multilabel([(["a_r", "b_r"], ["a_r", "c_r"])], REASONS)
    per = s["per_reason"]
    assert (per["a_r"]["tp"], per["a_r"]["fp"], per["a_r"]["fn"]) == (1, 0, 0)
    assert (per["b_r"]["tp"], per["b_r"]["fp"], per["b_r"]["fn"]) == (0, 0, 1)
    assert (per["c_r"]["tp"], per["c_r"]["fp"], per["c_r"]["fn"]) == (0, 1, 0)
    assert per["b_r"]["support"] == 1 and per["c_r"]["support"] == 0
    assert s["exact_match"] == 0


def test_exact_match_counts_empty_equals_empty(ev):
    s = ev.score_multilabel([([], []), (["a_r"], ["a_r"]), (["a_r"], [])], REASONS)
    assert s["exact_match"] == 2
    assert s["exact_match_rate"] == round(2 / 3, 3)


def test_a_failed_answer_is_never_an_exact_match_even_against_empty_gold(ev):
    """None (failed) and [] (answered NONE) must not collapse (D59)."""
    s = ev.score_multilabel([([], None)], REASONS)
    assert s["exact_match"] == 0
    assert s["failed"] == 1


def test_a_failed_answer_counts_as_a_miss_for_gold_reasons(ev):
    s = ev.score_multilabel([(["b_r"], None)], REASONS)
    assert s["per_reason"]["b_r"]["fn"] == 1
    assert s["per_reason"]["b_r"]["fp"] == 0


def test_prf_values(ev):
    s = ev.score_multilabel(
        [(["a_r"], ["a_r"]), (["a_r"], []), ([], ["a_r"]), (["a_r"], ["a_r"])], REASONS
    )
    a = s["per_reason"]["a_r"]
    assert (a["tp"], a["fp"], a["fn"]) == (2, 1, 1)
    assert a["precision"] == round(2 / 3, 3)
    assert a["recall"] == round(2 / 3, 3)


def test_confusion_pairs_substitution_under_and_over_assignment(ev):
    c = ev.confusion_pairs([
        (["c_r"], ["b_r"]),      # substitution
        (["a_r"], []),           # under-assignment
        ([], ["b_r"]),           # over-assignment
        (["a_r"], ["a_r"]),      # correct — no pair
        (["a_r"], None),         # failed — excluded
    ])
    assert c == {("c_r", "b_r"): 1, ("a_r", ev.NONE_LABEL): 1, (ev.NONE_LABEL, "b_r"): 1}


def _saved_run(tmp_path, *, sha, dry_run=False):
    import json

    rows = [{
        "ticket_id": "1", "gold_codes": ["c"], "gold_is_appeal": True,
        "gold_reciprocal": False, "gold_reasons": ["reviewer_misunderstanding"],
        "input_shape": "single", "requester_messages": 1, "intent": "review_decision_appeal",
        "detector_asked": False, "detector_verdict": None, "gate_would_call": True,
        "answer_ungated": ["reviewer_misunderstanding", "general_dissatisfaction"],
        "answer_production": ["reviewer_misunderstanding", "general_dissatisfaction"],
    }]
    path = tmp_path / "saved.json"
    path.write_text(json.dumps({"report": {"manifest": {
        "classifier_prompt_sha256": sha, "dry_run": dry_run}}, "rows": rows}))
    return path


def _current_sha():
    import hashlib

    from app.pipeline import appeal_reason_classifier as arc

    return hashlib.sha256(arc._SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def test_rescore_applies_the_d78_rule_to_saved_answers(ev, tmp_path):
    import json

    src = _saved_run(tmp_path, sha=_current_sha())
    out = tmp_path / "out"
    assert ev.main(["--rescore", str(src), "--out", str(out)]) == 0
    saved = json.loads((out / "appeal_reason_eval_rescored_d78.json").read_text())
    assert saved["rows"][0]["answer_ungated"] == ["reviewer_misunderstanding"]
    assert saved["report"]["manifest"]["tickets_changed_by_rule"] == ["1"]
    assert saved["report"]["ungated"]["exact_match"] == 1


@pytest.mark.parametrize("sha, dry_run", [("0" * 64, False), (None, True)])
def test_rescore_refuses_another_prompt_or_a_dry_run(ev, tmp_path, sha, dry_run):
    src = _saved_run(tmp_path, sha=sha or _current_sha(), dry_run=dry_run)
    with pytest.raises(SystemExit):
        ev.main(["--rescore", str(src), "--out", str(tmp_path / "out")])


def test_build_input_single_message_uses_the_body_path(ev):
    rec = {"subject": "S", "thread": []}

    def view(_r):
        return [{"body": "only", "created_at": "2025-09-22T10:05:18Z"}], 3

    d = ev.build_input(rec, view, lambda *a, **k: None, 16000)
    assert d["subject"] == "S" and d["body"] == "only"
    assert "thread_transcript" not in d
    assert d["_shape"] == "single" and d["_shown"] == 1


def test_build_input_several_messages_uses_production_build_transcript(ev):
    from app.pipeline.thread_transcript import build_transcript

    def view(_r):
        return [
            {"body": "FIRST", "created_at": "2025-09-22T10:00:00Z"},
            {"body": "SECOND", "created_at": "2025-09-23T10:00:00Z"},
        ], 1

    d = ev.build_input({"subject": "S"}, view, build_transcript, 16000)
    assert d["_shape"] == "thread"
    assert d["thread_transcript"].index("FIRST") < d["thread_transcript"].index("SECOND")
    assert d["body"] == "SECOND", "latest requester turn is the anchor"
