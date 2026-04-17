"""
project_drift.staleness
~~~~~~~~~~~~~~~~~~~~~~~
Lightweight heuristic hooks that translate a ``DriftReport`` into
archive-management signals.

These are *interface stubs* for integration with the CSIG archive
pruning module (``csig/drift.py`` in a later stage).  The logic is
deliberately simple — a proper Bayesian or bandit-based approach can
replace these heuristics without changing the function signatures.
"""

from __future__ import annotations

from project_drift.config import DriftConfig
from project_drift.reporter import DriftReport


def should_downweight_archive_branch(
    report: DriftReport,
    config: DriftConfig | None = None,
) -> bool:
    """Return ``True`` if the drift signal is strong enough to consider
    the tested archive branch stale.

    Decision rule (v1 heuristic):
        ``drift_score >= staleness_score_threshold``

    where ``drift_score = 1 − p_value``.
    """
    cfg = config or DriftConfig()
    return report.drift_score >= cfg.staleness_score_threshold


def compute_archive_staleness_weight(
    report: DriftReport,
    config: DriftConfig | None = None,
) -> float:
    """Map a drift report to a ``[min_weight, 1.0]`` multiplier.

    When no drift is detected the weight is ``1.0`` (full confidence).
    As the drift score increases beyond the threshold, the weight
    decays linearly toward ``staleness_min_weight``.

    Returns
    -------
    float in ``[staleness_min_weight, 1.0]``
    """
    cfg = config or DriftConfig()

    if report.drift_score < cfg.staleness_score_threshold:
        return 1.0

    excess = report.drift_score - cfg.staleness_score_threshold
    max_excess = 1.0 - cfg.staleness_score_threshold
    if max_excess <= 0:
        return cfg.staleness_min_weight

    decay = excess / max_excess
    weight = 1.0 - decay * (1.0 - cfg.staleness_min_weight)
    return max(weight, cfg.staleness_min_weight)
