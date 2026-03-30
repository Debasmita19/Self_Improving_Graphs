"""
csig.graph
~~~~~~~~~~
CSIGraph — the directed-acyclic archive of agent self-improvement history.

Wraps ``networkx.DiGraph`` and exposes a purpose-built API for adding nodes /
edges, traversal queries, mod-type filtering, and persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from csig.schemas import CSIGNode, CSIGEdge
from csig.taxonomy import ALL_MOD_TYPES


class CycleError(Exception):
    """Raised when an operation would introduce a cycle into the DAG."""


class CSIGraph:
    """Directed acyclic graph of agent variants and improvement edges."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, CSIGNode] = {}
        self._edges: Dict[str, CSIGEdge] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: CSIGNode) -> None:
        """Insert a node (agent variant) into the graph."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists")
        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id)

    def add_edge(self, edge: CSIGEdge) -> None:
        """Insert a directed edge (parent → child) and validate acyclicity."""
        if edge.parent_id not in self._nodes:
            raise ValueError(
                f"Parent node '{edge.parent_id}' not in graph"
            )
        if edge.child_id not in self._nodes:
            raise ValueError(
                f"Child node '{edge.child_id}' not in graph"
            )
        if edge.edge_id in self._edges:
            raise ValueError(f"Edge '{edge.edge_id}' already exists")

        self._graph.add_edge(edge.parent_id, edge.child_id, edge_id=edge.edge_id)

        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(edge.parent_id, edge.child_id)
            raise CycleError(
                f"Adding edge {edge.parent_id} -> {edge.child_id} "
                "would create a cycle"
            )

        self._edges[edge.edge_id] = edge

    # ------------------------------------------------------------------
    # Single-node lookups
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[CSIGNode]:
        return self._nodes.get(node_id)

    def get_parent(self, node_id: str) -> Optional[CSIGNode]:
        """Return the (single) parent recorded in the node schema."""
        node = self._nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self._nodes.get(node.parent_id)

    def get_children(self, node_id: str) -> List[CSIGNode]:
        """Return all immediate children of *node_id* in the DAG."""
        if node_id not in self._graph:
            return []
        return [
            self._nodes[cid]
            for cid in self._graph.successors(node_id)
            if cid in self._nodes
        ]

    # ------------------------------------------------------------------
    # Ancestry / descendancy
    # ------------------------------------------------------------------

    def get_ancestors(self, node_id: str) -> List[CSIGNode]:
        """All transitive ancestors (does not include the node itself)."""
        if node_id not in self._graph:
            return []
        return [
            self._nodes[aid]
            for aid in nx.ancestors(self._graph, node_id)
            if aid in self._nodes
        ]

    def get_descendants(self, node_id: str) -> List[CSIGNode]:
        """All transitive descendants (does not include the node itself)."""
        if node_id not in self._graph:
            return []
        return [
            self._nodes[did]
            for did in nx.descendants(self._graph, node_id)
            if did in self._nodes
        ]

    # ------------------------------------------------------------------
    # Edge queries
    # ------------------------------------------------------------------

    def get_edges_by_mod_type(self, mod_type: str) -> List[CSIGEdge]:
        """Return every edge whose descriptor contains *mod_type*."""
        return [
            e for e in self._edges.values()
            if mod_type in e.descriptor.mod_types
        ]

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

    def query_nodes_by_context(self, filters: Dict[str, Any]) -> List[CSIGNode]:
        """Return nodes whose *context_summary* matches all key-value pairs."""
        results: List[CSIGNode] = []
        for node in self._nodes.values():
            if all(
                node.context_summary.get(k) == v for k, v in filters.items()
            ):
                results.append(node)
        return results

    # ------------------------------------------------------------------
    # Persistence helpers  (delegated to csig.persistence for formats)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the full graph to a JSON file."""
        data = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "CSIGraph":
        """Deserialise a graph from a JSON file produced by .save()."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls()
        for nd in raw["nodes"]:
            graph.add_node(CSIGNode.from_dict(nd))
        for ed in raw["edges"]:
            graph.add_edge(CSIGEdge.from_dict(ed))
        return graph

    # ------------------------------------------------------------------
    # Summary / introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a concise statistical summary of the archive."""
        mod_type_counts: Dict[str, int] = {}
        for edge in self._edges.values():
            for mt in edge.descriptor.mod_types:
                mod_type_counts[mt] = mod_type_counts.get(mt, 0) + 1

        accepted = sum(1 for n in self._nodes.values() if n.accepted)

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "accepted_nodes": accepted,
            "rejected_nodes": len(self._nodes) - accepted,
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
            "mod_type_distribution": mod_type_counts,
        }

    @property
    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    @property
    def edge_ids(self) -> List[str]:
        return list(self._edges.keys())

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"CSIGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
        )
