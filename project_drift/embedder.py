"""
project_drift.embedder
~~~~~~~~~~~~~~~~~~~~~~
Convert ``RuntimeEvent`` text into dense vectors.

Primary backend: ``sentence-transformers`` (wraps HuggingFace models).
This is the de-facto standard for short-text embeddings in NLP research,
widely cited, and produces normalised vectors suitable for MMD comparison.

The module exposes a thin ``ContextEmbedder`` class so that the rest of
the pipeline never imports sentence-transformers directly — making it
easy to swap in a different encoder later.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from project_drift.config import DriftConfig
from project_drift.schema import RuntimeEvent

logger = logging.getLogger(__name__)


class ContextEmbedder:
    """Encode runtime event text into fixed-size embedding vectors.

    Parameters
    ----------
    config:
        Pipeline configuration.  Only the embedding-related fields
        (``embedding_model_name``, ``embedding_batch_size``,
        ``max_text_length``, ``device``) are used.
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self._cfg = config or DriftConfig()
        self._model = None  # lazy-loaded

    # -----------------------------------------------------------------
    # Lazy model loading
    # -----------------------------------------------------------------

    def _load_model(self):
        """Import and instantiate the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for ContextEmbedder.  "
                "Install it with:  pip install sentence-transformers"
            ) from exc

        logger.info(
            "Loading embedding model '%s' ...", self._cfg.embedding_model_name
        )
        self._model = SentenceTransformer(
            self._cfg.embedding_model_name,
            device=self._cfg.device,
        )

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of raw strings.

        Returns
        -------
        np.ndarray of shape ``(len(texts), embedding_dim)``
            Row-normalised float32 vectors.
        """
        truncated = [
            t[: self._cfg.max_text_length] for t in texts
        ]
        vectors = self.model.encode(
            truncated,
            batch_size=self._cfg.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_events(self, events: Sequence[RuntimeEvent]) -> np.ndarray:
        """Embed the ``combined_text()`` of each event.

        This is the main entry-point used by the detector.
        """
        texts = [e.combined_text() for e in events]
        return self.embed_texts(texts)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()
