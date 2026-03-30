#!/usr/bin/env python
"""
test_csig.py
~~~~~~~~~~~~
Automated tests for the Stage-1 CSIG foundation.

Covers:
  - Schema construction and round-trip serialisation
  - Taxonomy validation
  - Graph construction, DAG enforcement, and every query method
  - Diff parser
  - Rule-based classifier
  - Persistence (JSON and JSONL)

Run:
    python test_csig.py
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

from csig import (
    CSIGNode,
    CSIGEdge,
    CSIGraph,
    CycleError,
    ModificationDescriptor,
    DiffStats,
    ALL_MOD_TYPES,
    is_valid_mod_type,
    diff_strings,
    diff_file_sets,
    aggregate_diff_stats,
    modules_from_results,
    classify_mod_types,
    save_nodes_json,
    load_nodes_json,
    save_edges_json,
    load_edges_json,
    append_node_jsonl,
    append_edge_jsonl,
    load_jsonl,
)

passed = 0
failed = 0


def _run(name: str, fn) -> None:
    global passed, failed
    try:
        fn()
        print(f"  [PASS]  {name}")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL]  {name}")
        traceback.print_exc()
        failed += 1


# ======================================================================
# Schema tests
# ======================================================================

def test_node_roundtrip():
    n = CSIGNode(
        node_id="n1", parent_id=None, task_name="qa",
        context_summary={"dept": "cardiology"},
        metrics={"accuracy": 0.9}, accepted=True,
    )
    d = n.to_dict()
    n2 = CSIGNode.from_dict(d)
    assert n2.node_id == "n1"
    assert n2.context_summary["dept"] == "cardiology"
    assert n2.accepted is True


def test_edge_roundtrip():
    desc = ModificationDescriptor(
        mod_types=["prompt_template_change"],
        modules_changed=["prompts"],
        scope="localized",
        rationale="test",
        diff_stats=DiffStats(lines_added=5, lines_removed=2, files_changed=1),
    )
    e = CSIGEdge(
        edge_id="e1", parent_id="a", child_id="b",
        descriptor=desc,
        evaluation_context={"split": "dev"},
        performance_delta={"accuracy": 0.05},
    )
    d = e.to_dict()
    e2 = CSIGEdge.from_dict(d)
    assert e2.descriptor.mod_types == ["prompt_template_change"]
    assert e2.descriptor.diff_stats.lines_added == 5


def test_descriptor_invalid_scope():
    raised = False
    try:
        ModificationDescriptor(scope="galaxy")
    except ValueError:
        raised = True
    assert raised, "Should reject invalid scope"


# ======================================================================
# Taxonomy tests
# ======================================================================

def test_taxonomy_count():
    assert len(ALL_MOD_TYPES) == 10


def test_taxonomy_validation():
    assert is_valid_mod_type("retrieval_change")
    assert not is_valid_mod_type("made_up_change")


# ======================================================================
# Graph tests
# ======================================================================

def _make_graph() -> CSIGraph:
    """Build a small graph: v0 -> v1 -> v2, v0 -> v1b."""
    g = CSIGraph()
    g.add_node(CSIGNode(node_id="v0", parent_id=None, task_name="t",
                         context_summary={"department": "cardiology"},
                         metrics={"acc": 0.6}, accepted=True))
    g.add_node(CSIGNode(node_id="v1", parent_id="v0", task_name="t",
                         context_summary={"department": "cardiology"},
                         metrics={"acc": 0.7}, accepted=True))
    g.add_node(CSIGNode(node_id="v2", parent_id="v1", task_name="t",
                         context_summary={"department": "neurology"},
                         metrics={"acc": 0.75}, accepted=True))
    g.add_node(CSIGNode(node_id="v1b", parent_id="v0", task_name="t",
                         context_summary={"department": "cardiology"},
                         metrics={"acc": 0.55}, accepted=False))

    g.add_edge(CSIGEdge(
        edge_id="e01", parent_id="v0", child_id="v1",
        descriptor=ModificationDescriptor(mod_types=["prompt_template_change"]),
    ))
    g.add_edge(CSIGEdge(
        edge_id="e12", parent_id="v1", child_id="v2",
        descriptor=ModificationDescriptor(mod_types=["retrieval_change"]),
    ))
    g.add_edge(CSIGEdge(
        edge_id="e01b", parent_id="v0", child_id="v1b",
        descriptor=ModificationDescriptor(mod_types=["tool_selection_change"]),
    ))
    return g


def test_graph_add_duplicate_node():
    g = CSIGraph()
    g.add_node(CSIGNode(node_id="x"))
    raised = False
    try:
        g.add_node(CSIGNode(node_id="x"))
    except ValueError:
        raised = True
    assert raised


def test_graph_get_node():
    g = _make_graph()
    assert g.get_node("v0") is not None
    assert g.get_node("nonexistent") is None


def test_graph_get_parent():
    g = _make_graph()
    parent = g.get_parent("v1")
    assert parent is not None and parent.node_id == "v0"
    assert g.get_parent("v0") is None


def test_graph_get_children():
    g = _make_graph()
    children = g.get_children("v0")
    ids = {c.node_id for c in children}
    assert ids == {"v1", "v1b"}


def test_graph_ancestors():
    g = _make_graph()
    anc = g.get_ancestors("v2")
    ids = {a.node_id for a in anc}
    assert ids == {"v0", "v1"}


def test_graph_descendants():
    g = _make_graph()
    desc = g.get_descendants("v0")
    ids = {d.node_id for d in desc}
    assert ids == {"v1", "v2", "v1b"}


def test_graph_edges_by_mod_type():
    g = _make_graph()
    edges = g.get_edges_by_mod_type("prompt_template_change")
    assert len(edges) == 1 and edges[0].edge_id == "e01"


def test_graph_query_context():
    g = _make_graph()
    results = g.query_nodes_by_context({"department": "cardiology"})
    ids = {n.node_id for n in results}
    assert "v0" in ids and "v1" in ids and "v1b" in ids
    assert "v2" not in ids


def test_graph_cycle_detection():
    g = _make_graph()
    raised = False
    try:
        g.add_edge(CSIGEdge(
            edge_id="bad", parent_id="v2", child_id="v0",
            descriptor=ModificationDescriptor(mod_types=["retrieval_change"]),
        ))
    except CycleError:
        raised = True
    assert raised, "Graph should reject edges that create cycles"


def test_graph_summary():
    g = _make_graph()
    s = g.summary()
    assert s["total_nodes"] == 4
    assert s["total_edges"] == 3
    assert s["is_dag"] is True
    assert s["accepted_nodes"] == 3
    assert s["rejected_nodes"] == 1


# ======================================================================
# Persistence tests (JSON round-trip)
# ======================================================================

def test_graph_save_load():
    g = _make_graph()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "graph.json")
        g.save(path)
        g2 = CSIGraph.load(path)
        assert len(g2) == len(g)
        assert set(g2.node_ids) == set(g.node_ids)
        assert set(g2.edge_ids) == set(g.edge_ids)


def test_nodes_json_persistence():
    nodes = [
        CSIGNode(node_id="a", task_name="x"),
        CSIGNode(node_id="b", task_name="y"),
    ]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nodes.json")
        save_nodes_json(nodes, path)
        loaded = load_nodes_json(path)
        assert len(loaded) == 2
        assert loaded[0].node_id == "a"


def test_edges_json_persistence():
    edges = [
        CSIGEdge(edge_id="e1", parent_id="a", child_id="b"),
    ]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "edges.json")
        save_edges_json(edges, path)
        loaded = load_edges_json(path)
        assert len(loaded) == 1


def test_jsonl_persistence():
    n = CSIGNode(node_id="n1", task_name="t")
    e = CSIGEdge(edge_id="e1", parent_id="n1", child_id="n2")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "log.jsonl")
        append_node_jsonl(n, path)
        append_edge_jsonl(e, path)
        data = load_jsonl(path)
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1


# ======================================================================
# Diff parser tests
# ======================================================================

def test_diff_strings_basic():
    old = "line1\nline2\nline3\n"
    new = "line1\nline2_modified\nline3\nline4\n"
    result = diff_strings(old, new, "test.py")
    assert result.lines_added >= 1
    assert result.lines_removed >= 0


def test_diff_file_sets():
    old_files = {"a.py": "x = 1\n", "b.py": "y = 2\n"}
    new_files = {"a.py": "x = 10\n", "b.py": "y = 2\n", "c.py": "z = 3\n"}
    results = diff_file_sets(old_files, new_files)
    stats = aggregate_diff_stats(results)
    assert stats.files_changed >= 2  # a.py changed, c.py added
    modules = modules_from_results(results)
    assert len(modules) >= 1


# ======================================================================
# Classifier tests
# ======================================================================

def test_classifier_prompt():
    labels = classify_mod_types(filenames=["prompts/system_prompt.txt"])
    assert "prompt_template_change" in labels


def test_classifier_retrieval():
    labels = classify_mod_types(filenames=["retrieval/retriever.py"])
    assert "retrieval_change" in labels


def test_classifier_tool():
    labels = classify_mod_types(filenames=["tools/router.py"])
    assert "tool_selection_change" in labels


def test_classifier_memory():
    labels = classify_mod_types(keywords=["memory buffer"])
    assert "memory_update" in labels


def test_classifier_multi():
    labels = classify_mod_types(
        filenames=["prompts/main.txt", "retrieval/index.py"],
        rationale="Updated prompt and retrieval pipeline",
    )
    assert "prompt_template_change" in labels
    assert "retrieval_change" in labels


# ======================================================================
# Runner
# ======================================================================

def main() -> None:
    tests = [
        ("Schema: node round-trip",             test_node_roundtrip),
        ("Schema: edge round-trip",             test_edge_roundtrip),
        ("Schema: invalid scope rejected",      test_descriptor_invalid_scope),
        ("Taxonomy: 10 mod types",              test_taxonomy_count),
        ("Taxonomy: validation",                test_taxonomy_validation),
        ("Graph: duplicate node rejected",      test_graph_add_duplicate_node),
        ("Graph: get_node",                     test_graph_get_node),
        ("Graph: get_parent",                   test_graph_get_parent),
        ("Graph: get_children",                 test_graph_get_children),
        ("Graph: ancestors",                    test_graph_ancestors),
        ("Graph: descendants",                  test_graph_descendants),
        ("Graph: edges by mod_type",            test_graph_edges_by_mod_type),
        ("Graph: query by context",             test_graph_query_context),
        ("Graph: cycle detection",              test_graph_cycle_detection),
        ("Graph: summary",                      test_graph_summary),
        ("Persist: graph JSON round-trip",      test_graph_save_load),
        ("Persist: nodes JSON",                 test_nodes_json_persistence),
        ("Persist: edges JSON",                 test_edges_json_persistence),
        ("Persist: JSONL append & load",        test_jsonl_persistence),
        ("Diff: string comparison",             test_diff_strings_basic),
        ("Diff: file-set comparison",           test_diff_file_sets),
        ("Classifier: prompt files",            test_classifier_prompt),
        ("Classifier: retrieval files",         test_classifier_retrieval),
        ("Classifier: tool/router files",       test_classifier_tool),
        ("Classifier: memory keyword",          test_classifier_memory),
        ("Classifier: multi-signal",            test_classifier_multi),
    ]

    print(f"\nRunning {len(tests)} tests ...\n")
    for name, fn in tests:
        _run(name, fn)

    print(f"\n{'=' * 40}")
    print(f"  Results:  {passed} passed,  {failed} failed")
    print(f"{'=' * 40}\n")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
