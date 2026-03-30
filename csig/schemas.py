"""
csig.schemas
~~~~~~~~~~~~
Dataclass definitions for CSIG nodes, edges, and modification descriptors.

Every schema is JSON-serialisable via its .to_dict() / .from_dict() pair so
that the persistence layer never needs to reach into private state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Descriptor  (attached to every edge)
# ---------------------------------------------------------------------------

@dataclass
class DiffStats:
    """Lightweight summary of a code diff."""
    lines_added: int = 0
    lines_removed: int = 0
    files_changed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiffStats":
        return cls(**d)


@dataclass
class ModificationDescriptor:
    """Structured description of *what* changed between parent and child."""
    mod_types: List[str] = field(default_factory=list)
    modules_changed: List[str] = field(default_factory=list)
    scope: str = "localized"          # localized | multi_module | architectural
    rationale: str = ""
    diff_stats: DiffStats = field(default_factory=DiffStats)

    def __post_init__(self) -> None:
        valid_scopes = {"localized", "multi_module", "architectural"}
        if self.scope not in valid_scopes:
            raise ValueError(
                f"scope must be one of {valid_scopes}, got '{self.scope}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModificationDescriptor":
        d = dict(d)
        if "diff_stats" in d and isinstance(d["diff_stats"], dict):
            d["diff_stats"] = DiffStats.from_dict(d["diff_stats"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Node  (one per agent variant)
# ---------------------------------------------------------------------------

@dataclass
class CSIGNode:
    """A single agent variant in the self-improvement graph."""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    task_name: str = ""
    context_summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CSIGNode":
        return cls(**d)


# ---------------------------------------------------------------------------
# Edge  (parent → child improvement arc)
# ---------------------------------------------------------------------------

@dataclass
class CSIGEdge:
    """Directed edge representing one self-improvement step."""
    edge_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str = ""
    child_id: str = ""
    descriptor: ModificationDescriptor = field(
        default_factory=ModificationDescriptor
    )
    evaluation_context: Dict[str, Any] = field(default_factory=dict)
    performance_delta: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CSIGEdge":
        d = dict(d)
        if "descriptor" in d and isinstance(d["descriptor"], dict):
            d["descriptor"] = ModificationDescriptor.from_dict(d["descriptor"])
        return cls(**d)
