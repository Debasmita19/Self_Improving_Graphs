"""
project_drift.reporter
~~~~~~~~~~~~~~~~~~~~~~
Standardised drift report dataclass.

Designed so that downstream modules (e.g. archive pruning, staleness
hooks) can consume drift results without depending on alibi-detect
internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class DriftReport:
    """Result of a single drift test.

    Attributes
    ----------
    drift_detected:
        ``True`` if the test rejected the null hypothesis (no drift).
    drift_score:
        A scalar in ``[0, 1]`` summarising drift severity.
        Defined as ``1 − p_value`` so that higher = more evidence of drift.
    p_value:
        Raw p-value from the MMD permutation test.
    threshold:
        The significance level used for the decision.
    n_reference:
        Number of samples in the reference window.
    n_current:
        Number of samples in the current (test) window.
    method:
        Drift-detection method name (e.g. ``"MMDDrift"``).
    embedding_model:
        Name of the embedding model used.
    timestamp:
        ISO-8601 UTC string of when the test was run.
    notes:
        Free-form annotation (e.g. which archive branch was tested).
    extra:
        Any additional key-value pairs from the detector output.
    """

    drift_detected: bool = False
    drift_score: float = 0.0
    p_value: float = 1.0
    threshold: float = 0.05
    n_reference: int = 0
    n_current: int = 0
    method: str = ""
    embedding_model: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftReport":
        return cls(**d)

    def summary_line(self) -> str:
        """One-line human-readable summary for logging."""
        status = "DRIFT" if self.drift_detected else "no-drift"
        return (
            f"[{status}] score={self.drift_score:.4f}  "
            f"p={self.p_value:.4f}  thr={self.threshold}  "
            f"ref={self.n_reference}  cur={self.n_current}  "
            f"method={self.method}"
        )
