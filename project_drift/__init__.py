"""
project_drift — Drift detection for LLM-agent runtime context streams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adaptation layer built on top of Alibi Detect's ``MMDDrift`` detector.

Detects distribution shift between a *reference window* of historical
runtime contexts and a *current window* of recent contexts, producing
a standardised ``DriftReport`` that downstream modules (e.g. CSIG
archive pruning) can consume.
"""

from project_drift.config import DriftConfig
from project_drift.schema import RuntimeEvent
from project_drift.embedder import ContextEmbedder
from project_drift.window_manager import WindowManager
from project_drift.detector import ContextDriftDetector
from project_drift.reporter import DriftReport
from project_drift.staleness import (
    should_downweight_archive_branch,
    compute_archive_staleness_weight,
)

__all__ = [
    "DriftConfig",
    "RuntimeEvent",
    "ContextEmbedder",
    "WindowManager",
    "ContextDriftDetector",
    "DriftReport",
    "should_downweight_archive_branch",
    "compute_archive_staleness_weight",
]
