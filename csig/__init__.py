"""
csig — Causal Self-Improvement Graphs (Stage 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Minimal foundation: DAG archive, schemas, taxonomy, diff parser,
rule-based classifier, and persistence.
"""

from csig.schemas import CSIGNode, CSIGEdge, ModificationDescriptor, DiffStats
from csig.graph import CSIGraph, CycleError
from csig.taxonomy import ALL_MOD_TYPES, is_valid_mod_type
from csig.diff_parser import (
    diff_strings,
    diff_files,
    diff_file_sets,
    aggregate_diff_stats,
    modules_from_results,
)
from csig.classifier import classify_mod_types
from csig.persistence import (
    save_nodes_json,
    load_nodes_json,
    save_edges_json,
    load_edges_json,
    append_node_jsonl,
    append_edge_jsonl,
    load_jsonl,
)

__all__ = [
    "CSIGNode",
    "CSIGEdge",
    "ModificationDescriptor",
    "DiffStats",
    "CSIGraph",
    "CycleError",
    "ALL_MOD_TYPES",
    "is_valid_mod_type",
    "diff_strings",
    "diff_files",
    "diff_file_sets",
    "aggregate_diff_stats",
    "modules_from_results",
    "classify_mod_types",
    "save_nodes_json",
    "load_nodes_json",
    "save_edges_json",
    "load_edges_json",
    "append_node_jsonl",
    "append_edge_jsonl",
    "load_jsonl",
]
