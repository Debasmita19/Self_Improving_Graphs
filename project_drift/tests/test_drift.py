#!/usr/bin/env python
"""
test_drift.py
~~~~~~~~~~~~~
Lightweight test suite for the project_drift adaptation layer.

Tests are split into two tiers:

* **Unit tests** (schema, config, window manager, reporter, staleness)
  — no heavy dependencies, always runnable.
* **Integration tests** (detector with synthetic numpy embeddings)
  — require ``alibi-detect`` and ``torch`` but do *not* download any
  embedding model.  The embedder is bypassed entirely by calling
  ``fit_from_embeddings`` / ``test_from_embeddings``.

Run::

    python -m project_drift.tests.test_drift
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from datetime import datetime, timezone

import numpy as np

from project_drift.config import DriftConfig
from project_drift.schema import RuntimeEvent
from project_drift.window_manager import WindowManager
from project_drift.reporter import DriftReport
from project_drift.staleness import (
    should_downweight_archive_branch,
    compute_archive_staleness_weight,
)

passed = 0
failed = 0
skipped = 0


def _run(name: str, fn, *, skip_if: bool = False, skip_reason: str = "") -> None:
    global passed, failed, skipped
    if skip_if:
        print(f"  [SKIP]  {name}  ({skip_reason})")
        skipped += 1
        return
    try:
        fn()
        print(f"  [PASS]  {name}")
        passed += 1
    except Exception:
        print(f"  [FAIL]  {name}")
        traceback.print_exc()
        failed += 1


# --- Check optional deps once ---
_HAS_ALIBI = False
try:
    from alibi_detect.cd import MMDDrift  # noqa: F401
    import torch  # noqa: F401
    _HAS_ALIBI = True
except Exception:
    pass


# ===================================================================
# Unit: RuntimeEvent schema
# ===================================================================

def test_event_creation():
    e = RuntimeEvent(prompt_text="hello", retrieved_context="world")
    assert e.prompt_text == "hello"
    assert e.event_id  # auto-generated
    assert e.timestamp  # auto-generated

def test_event_combined_text():
    e = RuntimeEvent(prompt_text="p", retrieved_context="c")
    assert e.combined_text() == "p [SEP] c"
    e2 = RuntimeEvent(prompt_text="p")
    assert e2.combined_text() == "p"

def test_event_roundtrip():
    e = RuntimeEvent(
        prompt_text="abc",
        retrieved_context="ctx",
        metadata={"key": "val"},
        label="positive",
        archive_branch_id="branch-1",
    )
    d = e.to_dict()
    e2 = RuntimeEvent.from_dict(d)
    assert e2.prompt_text == "abc"
    assert e2.metadata["key"] == "val"
    assert e2.archive_branch_id == "branch-1"

def test_event_json_serialisable():
    e = RuntimeEvent(prompt_text="test")
    s = json.dumps(e.to_dict())
    assert isinstance(s, str)


# ===================================================================
# Unit: DriftConfig
# ===================================================================

def test_config_defaults():
    cfg = DriftConfig()
    assert cfg.p_val_threshold == 0.05
    assert cfg.reference_window_size == 200
    assert cfg.embedding_model_name == "all-MiniLM-L6-v2"


# ===================================================================
# Unit: WindowManager
# ===================================================================

def test_window_add_reference():
    wm = WindowManager()
    for i in range(5):
        wm.add_reference_event(RuntimeEvent(prompt_text=f"ref-{i}"))
    assert wm.n_reference == 5

def test_window_freeze():
    wm = WindowManager()
    wm.add_reference_event(RuntimeEvent(prompt_text="r"))
    wm.freeze_reference()
    raised = False
    try:
        wm.add_reference_event(RuntimeEvent(prompt_text="r2"))
    except RuntimeError:
        raised = True
    assert raised, "Should reject adds after freeze"

def test_window_auto_route():
    wm = WindowManager()
    wm.add_event(RuntimeEvent(prompt_text="goes-to-ref"))
    assert wm.n_reference == 1
    wm.freeze_reference()
    wm.add_event(RuntimeEvent(prompt_text="goes-to-cur"))
    assert wm.n_current == 1

def test_window_sliding():
    cfg = DriftConfig(reference_window_size=3, current_window_size=2)
    wm = WindowManager(cfg)
    for i in range(5):
        wm.add_reference_event(RuntimeEvent(prompt_text=f"r-{i}"))
    assert wm.n_reference == 3  # oldest dropped

def test_window_is_ready():
    wm = WindowManager()
    assert not wm.is_ready(min_reference=1, min_current=1)
    wm.add_reference_event(RuntimeEvent(prompt_text="r"))
    wm.freeze_reference()
    assert not wm.is_ready(min_reference=1, min_current=1)
    wm.add_current_event(RuntimeEvent(prompt_text="c"))
    assert wm.is_ready(min_reference=1, min_current=1)

def test_window_save_load():
    wm = WindowManager()
    for i in range(3):
        wm.add_reference_event(RuntimeEvent(prompt_text=f"r-{i}"))
    wm.freeze_reference()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ref.json")
        wm.save_reference(path)

        wm2 = WindowManager()
        wm2.load_reference(path, freeze=True)
        assert wm2.n_reference == 3
        assert wm2.is_frozen

def test_window_clear_current():
    wm = WindowManager()
    wm.freeze_reference()
    wm.add_current_event(RuntimeEvent(prompt_text="c"))
    assert wm.n_current == 1
    wm.clear_current()
    assert wm.n_current == 0


# ===================================================================
# Unit: DriftReport
# ===================================================================

def test_report_creation():
    r = DriftReport(drift_detected=True, drift_score=0.95, p_value=0.05)
    assert r.drift_detected is True
    assert r.drift_score == 0.95

def test_report_roundtrip():
    r = DriftReport(
        drift_detected=False, drift_score=0.3, p_value=0.7,
        n_reference=100, n_current=50, method="MMDDrift",
    )
    d = r.to_dict()
    r2 = DriftReport.from_dict(d)
    assert r2.method == "MMDDrift"
    assert r2.n_reference == 100

def test_report_summary_line():
    r = DriftReport(drift_detected=True, drift_score=0.99, p_value=0.01)
    line = r.summary_line()
    assert "DRIFT" in line
    assert "0.99" in line

def test_report_has_required_fields():
    r = DriftReport()
    required = [
        "drift_detected", "drift_score", "p_value", "threshold",
        "n_reference", "n_current", "method", "timestamp",
    ]
    d = r.to_dict()
    for key in required:
        assert key in d, f"Missing field: {key}"


# ===================================================================
# Unit: staleness hooks
# ===================================================================

def test_staleness_no_drift():
    r = DriftReport(drift_score=0.3)
    assert not should_downweight_archive_branch(r)
    assert compute_archive_staleness_weight(r) == 1.0

def test_staleness_strong_drift():
    r = DriftReport(drift_score=0.95)
    cfg = DriftConfig(staleness_score_threshold=0.80)
    assert should_downweight_archive_branch(r, cfg)
    w = compute_archive_staleness_weight(r, cfg)
    assert 0.0 < w < 1.0

def test_staleness_extreme_drift():
    r = DriftReport(drift_score=1.0)
    cfg = DriftConfig(staleness_score_threshold=0.80, staleness_min_weight=0.05)
    w = compute_archive_staleness_weight(r, cfg)
    assert abs(w - 0.05) < 1e-6

def test_staleness_weight_range():
    cfg = DriftConfig(staleness_score_threshold=0.80, staleness_min_weight=0.05)
    for score in [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]:
        r = DriftReport(drift_score=score)
        w = compute_archive_staleness_weight(r, cfg)
        assert cfg.staleness_min_weight <= w <= 1.0


# ===================================================================
# Integration: ContextDriftDetector (needs alibi-detect + torch)
# ===================================================================

def test_detector_no_drift():
    """Identical reference and current → expect no drift."""
    from project_drift.detector import ContextDriftDetector

    rng = np.random.RandomState(0)
    ref = rng.randn(80, 32).astype(np.float32)
    cur = rng.randn(40, 32).astype(np.float32)  # same distribution

    cfg = DriftConfig(p_val_threshold=0.05, n_permutations=100)
    det = ContextDriftDetector(cfg)
    det.fit_from_embeddings(ref)
    report = det.test_from_embeddings(cur)

    assert isinstance(report, DriftReport)
    assert report.method == "MMDDrift"
    assert report.n_reference == 80
    assert report.n_current == 40
    # With identical distributions, we expect no drift most of the time.
    # We don't hard-assert drift_detected==False because permutation tests
    # have a natural false-positive rate, but the score should be low.
    assert report.drift_score < 0.99

def test_detector_with_drift():
    """Shifted current distribution → expect drift detected."""
    from project_drift.detector import ContextDriftDetector

    rng = np.random.RandomState(1)
    ref = rng.randn(80, 32).astype(np.float32)
    cur = rng.randn(40, 32).astype(np.float32) + 3.0  # large shift

    cfg = DriftConfig(p_val_threshold=0.05, n_permutations=100)
    det = ContextDriftDetector(cfg)
    det.fit_from_embeddings(ref)
    report = det.test_from_embeddings(cur)

    assert report.drift_detected is True
    assert report.drift_score > 0.90

def test_detector_score_ordering():
    """Stronger shift should produce higher drift score."""
    from project_drift.detector import ContextDriftDetector

    rng = np.random.RandomState(2)
    ref = rng.randn(80, 32).astype(np.float32)
    cur_mild = rng.randn(40, 32).astype(np.float32) + 0.5
    cur_strong = rng.randn(40, 32).astype(np.float32) + 5.0

    cfg = DriftConfig(p_val_threshold=0.05, n_permutations=100)
    det = ContextDriftDetector(cfg)
    det.fit_from_embeddings(ref)

    r_mild = det.test_from_embeddings(cur_mild)
    r_strong = det.test_from_embeddings(cur_strong)

    assert r_strong.drift_score >= r_mild.drift_score


# ===================================================================
# Runner
# ===================================================================

def main() -> None:
    unit_tests = [
        ("Schema: event creation",              test_event_creation),
        ("Schema: combined_text",               test_event_combined_text),
        ("Schema: event round-trip",            test_event_roundtrip),
        ("Schema: JSON serialisable",           test_event_json_serialisable),
        ("Config: defaults",                    test_config_defaults),
        ("Window: add reference",               test_window_add_reference),
        ("Window: freeze",                      test_window_freeze),
        ("Window: auto-route",                  test_window_auto_route),
        ("Window: sliding eviction",            test_window_sliding),
        ("Window: is_ready",                    test_window_is_ready),
        ("Window: save/load reference",         test_window_save_load),
        ("Window: clear current",               test_window_clear_current),
        ("Report: creation",                    test_report_creation),
        ("Report: round-trip",                  test_report_roundtrip),
        ("Report: summary_line",                test_report_summary_line),
        ("Report: required fields",             test_report_has_required_fields),
        ("Staleness: no drift",                 test_staleness_no_drift),
        ("Staleness: strong drift",             test_staleness_strong_drift),
        ("Staleness: extreme drift -> min wt",   test_staleness_extreme_drift),
        ("Staleness: weight range",             test_staleness_weight_range),
    ]

    integration_tests = [
        ("Detector: no drift (synthetic)",      test_detector_no_drift),
        ("Detector: with drift (synthetic)",    test_detector_with_drift),
        ("Detector: score ordering",            test_detector_score_ordering),
    ]

    total = len(unit_tests) + len(integration_tests)
    print(f"\nRunning {total} tests ...\n")

    for name, fn in unit_tests:
        _run(name, fn)

    needs_alibi = not _HAS_ALIBI
    for name, fn in integration_tests:
        _run(
            name, fn,
            skip_if=needs_alibi,
            skip_reason="alibi-detect or torch not installed",
        )

    print(f"\n{'=' * 50}")
    print(f"  Results:  {passed} passed,  {failed} failed,  {skipped} skipped")
    print(f"{'=' * 50}\n")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
