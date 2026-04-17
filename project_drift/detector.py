"""
project_drift.detector
~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around ``alibi_detect.cd.MMDDrift`` that accepts
``RuntimeEvent`` windows and returns a ``DriftReport``.

**Why MMD?**
Maximum Mean Discrepancy is a kernel two-sample test that compares
distributions without assuming a parametric form — ideal for
high-dimensional text embeddings where we cannot assume normality.
Alibi Detect's implementation adds permutation-based p-values and
supports GPU acceleration via PyTorch, which we already have as a
dependency through sentence-transformers.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from project_drift.config import DriftConfig
from project_drift.embedder import ContextEmbedder
from project_drift.reporter import DriftReport
from project_drift.schema import RuntimeEvent

logger = logging.getLogger(__name__)


class ContextDriftDetector:
    """End-to-end drift detector for LLM-agent runtime context streams.

    Typical usage::

        detector = ContextDriftDetector()
        detector.fit(reference_events)
        report = detector.test(current_events)
        print(report.summary_line())

    Parameters
    ----------
    config:
        Pipeline configuration.
    embedder:
        Pre-built ``ContextEmbedder``.  If ``None``, one is created
        from *config*.
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        embedder: Optional[ContextEmbedder] = None,
    ) -> None:
        self._cfg = config or DriftConfig()
        self._embedder = embedder or ContextEmbedder(self._cfg)
        self._detector = None  # alibi-detect MMDDrift instance
        self._ref_embeddings: Optional[np.ndarray] = None

    # -----------------------------------------------------------------
    # Fit (build the reference distribution)
    # -----------------------------------------------------------------

    def fit(self, reference_events: List[RuntimeEvent]) -> None:
        """Embed the reference events and initialise the MMD detector.

        Must be called before ``test()``.
        """
        if not reference_events:
            raise ValueError("reference_events must be non-empty.")

        self._ref_embeddings = self._embedder.embed_events(reference_events)
        logger.info(
            "Embedded %d reference events → shape %s",
            len(reference_events),
            self._ref_embeddings.shape,
        )
        self._init_mmd(self._ref_embeddings)

    def fit_from_embeddings(self, ref_embeddings: np.ndarray) -> None:
        """Initialise directly from pre-computed embeddings.

        Useful in tests or when embeddings are cached.
        """
        self._ref_embeddings = np.asarray(ref_embeddings, dtype=np.float32)
        self._init_mmd(self._ref_embeddings)

    def _init_mmd(self, x_ref: np.ndarray) -> None:
        try:
            from alibi_detect.cd import MMDDrift
        except ImportError as exc:
            raise ImportError(
                "alibi-detect is required for ContextDriftDetector.  "
                "Install it with:  pip install alibi-detect"
            ) from exc

        self._detector = MMDDrift(
            x_ref=x_ref,
            p_val=self._cfg.p_val_threshold,
            n_permutations=self._cfg.n_permutations,
            backend=self._cfg.mmd_backend,
            device=self._cfg.device,
        )
        logger.info(
            "MMDDrift detector initialised (p_val=%.3f, n_perm=%d, backend=%s).",
            self._cfg.p_val_threshold,
            self._cfg.n_permutations,
            self._cfg.mmd_backend,
        )

    # -----------------------------------------------------------------
    # Test (compare current window against reference)
    # -----------------------------------------------------------------

    def test(
        self,
        current_events: List[RuntimeEvent],
        *,
        notes: str = "",
    ) -> DriftReport:
        """Run the MMD two-sample test on the current window.

        Parameters
        ----------
        current_events:
            Recent events from the live stream.
        notes:
            Free-form annotation added to the report.

        Returns
        -------
        DriftReport
        """
        if self._detector is None:
            raise RuntimeError("Call fit() before test().")
        if not current_events:
            raise ValueError("current_events must be non-empty.")

        cur_embeddings = self._embedder.embed_events(current_events)
        return self.test_from_embeddings(
            cur_embeddings, notes=notes,
            _n_current=len(current_events),
        )

    def test_from_embeddings(
        self,
        cur_embeddings: np.ndarray,
        *,
        notes: str = "",
        _n_current: Optional[int] = None,
    ) -> DriftReport:
        """Run the test from pre-computed current embeddings."""
        if self._detector is None:
            raise RuntimeError("Call fit() / fit_from_embeddings() first.")

        cur_embeddings = np.asarray(cur_embeddings, dtype=np.float32)
        preds = self._detector.predict(cur_embeddings)

        data = preds["data"]
        p_value = float(data["p_val"])
        is_drift = bool(data["is_drift"])

        return DriftReport(
            drift_detected=is_drift,
            drift_score=round(1.0 - p_value, 6),
            p_value=round(p_value, 6),
            threshold=self._cfg.p_val_threshold,
            n_reference=len(self._ref_embeddings),
            n_current=_n_current or len(cur_embeddings),
            method="MMDDrift",
            embedding_model=self._cfg.embedding_model_name,
            notes=notes,
            extra={
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in data.items()
                if k not in {"is_drift", "p_val"}
            },
        )
