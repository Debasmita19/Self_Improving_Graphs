"""
project_drift.schema
~~~~~~~~~~~~~~~~~~~~
Dataclass for a single runtime event in the LLM-agent context stream.

Mirrors the serialisation conventions of ``csig.schemas`` (``to_dict`` /
``from_dict``) so that events can be logged alongside CSIG nodes/edges.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class RuntimeEvent:
    """One observation in the agent's runtime input/context stream.

    This is the atomic unit that the drift detector operates on.
    A sequence of ``RuntimeEvent`` objects forms a *window*.

    Attributes
    ----------
    event_id:
        Unique identifier (auto-generated if omitted).
    prompt_text:
        The user / task prompt presented to the agent.
    retrieved_context:
        Any RAG-retrieved or otherwise fetched context that
        accompanied the prompt.  Empty string if none.
    metadata:
        Arbitrary key-value pairs — task_type, subgroup, source,
        domain, archive_branch_id, etc.
    timestamp:
        ISO-8601 UTC string.  Auto-populated on creation.
    label:
        Optional ground-truth or downstream label; useful for
        later evaluation but not consumed by the drift detector.
    archive_branch_id:
        Shortcut for ``metadata["archive_branch_id"]``.
        Kept top-level for ergonomics in staleness-hook queries.
    """

    event_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:12]
    )
    prompt_text: str = ""
    retrieved_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    label: Optional[str] = None
    archive_branch_id: Optional[str] = None

    # -----------------------------------------------------------------
    # Derived text that the embedder will encode
    # -----------------------------------------------------------------

    def combined_text(self) -> str:
        """Concatenate prompt and retrieved context for embedding.

        Uses a lightweight separator so the model can distinguish
        the two segments, but no special tokens are injected.
        """
        parts = [self.prompt_text]
        if self.retrieved_context:
            parts.append(self.retrieved_context)
        return " [SEP] ".join(parts)

    # -----------------------------------------------------------------
    # Serialisation (matches csig convention)
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuntimeEvent":
        return cls(**d)
