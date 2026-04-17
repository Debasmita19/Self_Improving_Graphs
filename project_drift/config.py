"""
project_drift.config
~~~~~~~~~~~~~~~~~~~~
Configuration dataclass for the drift-detection adaptation layer.

Centralises all tuneable parameters so that experiments are reproducible
and easy to adjust from a single location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DriftConfig:
    """All settings for the drift-detection pipeline.

    Attributes
    ----------
    embedding_model_name:
        HuggingFace sentence-transformers model identifier.
        ``all-MiniLM-L6-v2`` is small (~80 MB), fast, and produces
        384-dimensional embeddings — a good default for research demos.
    embedding_batch_size:
        Batch size passed to the sentence-transformer encoder.
    max_text_length:
        Texts longer than this (in characters) are truncated before
        embedding.  Keeps memory predictable.
    reference_window_size:
        Maximum number of events retained in the reference window.
    current_window_size:
        Maximum number of events retained in the current (test) window.
    p_val_threshold:
        Significance level for the MMD permutation test.  A p-value
        below this triggers a drift decision.
    n_permutations:
        Number of permutations used by the MMD test to estimate the
        null distribution.  Higher → more accurate p-value, slower.
    mmd_backend:
        Compute backend for ``alibi_detect.cd.MMDDrift``.
        ``"pytorch"`` is recommended when sentence-transformers is
        installed (which requires torch anyway).
    device:
        Torch device string for both embedding and MMD computation.
    staleness_score_threshold:
        Drift score (1 − p_value) above which an archive branch is
        considered stale by the default heuristic.
    staleness_min_weight:
        Floor weight returned by ``compute_archive_staleness_weight``
        so that no branch is fully zeroed out.
    """

    # --- embedding ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    max_text_length: int = 2048

    # --- windows ---
    reference_window_size: int = 200
    current_window_size: int = 50

    # --- MMD detector ---
    p_val_threshold: float = 0.05
    n_permutations: int = 500
    mmd_backend: str = "pytorch"
    device: Optional[str] = None

    # --- staleness heuristic ---
    staleness_score_threshold: float = 0.80
    staleness_min_weight: float = 0.05
