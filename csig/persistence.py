"""
csig.persistence
~~~~~~~~~~~~~~~~
Standalone serialisation helpers for nodes and edges.

Two formats are supported:

* **JSON** — a single file containing ``{"nodes": [...], "edges": [...]}``.
  Used by ``CSIGraph.save()`` / ``CSIGraph.load()`` for round-tripping.
* **JSONL** — one JSON object per line.  Useful for append-only logging where
  each self-improvement step is written as it happens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from csig.schemas import CSIGNode, CSIGEdge


# ------------------------------------------------------------------
# JSON  (batch)
# ------------------------------------------------------------------

def save_nodes_json(nodes: List[CSIGNode], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([n.to_dict() for n in nodes], indent=2),
        encoding="utf-8",
    )


def load_nodes_json(path: str) -> List[CSIGNode]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CSIGNode.from_dict(d) for d in raw]


def save_edges_json(edges: List[CSIGEdge], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([e.to_dict() for e in edges], indent=2),
        encoding="utf-8",
    )


def load_edges_json(path: str) -> List[CSIGEdge]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CSIGEdge.from_dict(d) for d in raw]


# ------------------------------------------------------------------
# JSONL  (streaming / append-only)
# ------------------------------------------------------------------

def _append_jsonl(obj: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def append_node_jsonl(node: CSIGNode, path: str) -> None:
    """Append a single node record to a JSONL file."""
    record = {"type": "node", "data": node.to_dict()}
    _append_jsonl(record, path)


def append_edge_jsonl(edge: CSIGEdge, path: str) -> None:
    """Append a single edge record to a JSONL file."""
    record = {"type": "edge", "data": edge.to_dict()}
    _append_jsonl(record, path)


def load_jsonl(path: str) -> Dict[str, list]:
    """Read a JSONL log and return ``{"nodes": [...], "edges": [...]}``."""
    nodes: list[CSIGNode] = []
    edges: list[CSIGEdge] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record["type"] == "node":
            nodes.append(CSIGNode.from_dict(record["data"]))
        elif record["type"] == "edge":
            edges.append(CSIGEdge.from_dict(record["data"]))
    return {"nodes": nodes, "edges": edges}
