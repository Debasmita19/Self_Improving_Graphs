#!/usr/bin/env python
"""
build_toy_csig.py
~~~~~~~~~~~~~~~~~
Synthetic demo: builds a small CSIG, saves it, reloads it, and runs queries.

Run:
    python build_toy_csig.py
"""

from __future__ import annotations

import json
import os
import pprint
import textwrap

from csig import (
    CSIGNode,
    CSIGEdge,
    CSIGraph,
    ModificationDescriptor,
    DiffStats,
    diff_strings,
    aggregate_diff_stats,
    modules_from_results,
    classify_mod_types,
)
from csig.taxonomy import (
    PROMPT_TEMPLATE_CHANGE,
    RETRIEVAL_CHANGE,
    TOOL_SELECTION_CHANGE,
    REASONING_STEP_CHANGE,
    ERROR_RETRY_LOGIC,
    SCHEMA_LINKING_CHANGE,
)

SAVE_PATH = os.path.join("output", "toy_csig.json")

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# -----------------------------------------------------------------------
# 1.  Define synthetic agent-variant nodes
# -----------------------------------------------------------------------

def build_nodes() -> list[CSIGNode]:
    root = CSIGNode(
        node_id="agent_v0",
        parent_id=None,
        task_name="ehr_query",
        context_summary={
            "department": "cardiology",
            "dataset": "mimic-iv",
            "run": 1,
        },
        metrics={"accuracy": 0.62, "execution_success": 0.70},
        accepted=True,
    )
    v1 = CSIGNode(
        node_id="agent_v1",
        parent_id="agent_v0",
        task_name="ehr_query",
        context_summary={
            "department": "cardiology",
            "dataset": "mimic-iv",
            "run": 2,
        },
        metrics={"accuracy": 0.71, "execution_success": 0.80},
        accepted=True,
    )
    v2 = CSIGNode(
        node_id="agent_v2",
        parent_id="agent_v1",
        task_name="ehr_query",
        context_summary={
            "department": "cardiology",
            "dataset": "mimic-iv",
            "run": 3,
        },
        metrics={"accuracy": 0.75, "execution_success": 0.85},
        accepted=True,
    )
    v1b = CSIGNode(
        node_id="agent_v1b",
        parent_id="agent_v0",
        task_name="ehr_query",
        context_summary={
            "department": "neurology",
            "dataset": "mimic-iv",
            "run": 2,
        },
        metrics={"accuracy": 0.58, "execution_success": 0.65},
        accepted=False,
    )
    v3 = CSIGNode(
        node_id="agent_v3",
        parent_id="agent_v2",
        task_name="ehr_query",
        context_summary={
            "department": "cardiology",
            "dataset": "mimic-iv",
            "run": 4,
        },
        metrics={"accuracy": 0.78, "execution_success": 0.88},
        accepted=True,
    )
    return [root, v1, v2, v1b, v3]


# -----------------------------------------------------------------------
# 2.  Build edges with descriptors
# -----------------------------------------------------------------------

OLD_PROMPT = textwrap.dedent("""\
    Given the patient record, generate an SQL query.
    Be concise.
""")
NEW_PROMPT = textwrap.dedent("""\
    Given the patient record, generate an SQL query.
    Think step by step.
    Always verify column names against the schema before writing the query.
    Be concise.
""")

OLD_RETRIEVER = textwrap.dedent("""\
    def retrieve(query):
        return vector_store.search(query, top_k=3)
""")
NEW_RETRIEVER = textwrap.dedent("""\
    def retrieve(query):
        results = vector_store.search(query, top_k=5)
        return rerank(results, query)
""")

OLD_TOOL_ROUTER = textwrap.dedent("""\
    def select_tool(query):
        return "sql_executor"
""")
NEW_TOOL_ROUTER = textwrap.dedent("""\
    def select_tool(query):
        if "schema" in query.lower():
            return "schema_inspector"
        return "sql_executor"
""")

OLD_SCHEMA_LINKER = textwrap.dedent("""\
    def link_schema(query, tables):
        return tables[:2]
""")
NEW_SCHEMA_LINKER = textwrap.dedent("""\
    def link_schema(query, tables):
        scored = [(t, score_relevance(query, t)) for t in tables]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, s in scored[:3] if s > 0.5]
""")


def build_edges() -> list[CSIGEdge]:
    # -- edge 0→1 : prompt change --
    diff_r1 = diff_strings(OLD_PROMPT, NEW_PROMPT, "prompts/ehr_prompt.txt")
    e1 = CSIGEdge(
        edge_id="e_v0_v1",
        parent_id="agent_v0",
        child_id="agent_v1",
        descriptor=ModificationDescriptor(
            mod_types=classify_mod_types(
                filenames=["prompts/ehr_prompt.txt"],
                rationale="Added chain-of-thought and schema verification",
            ),
            modules_changed=["prompts"],
            scope="localized",
            rationale="Added chain-of-thought and schema verification to prompt",
            diff_stats=DiffStats(
                lines_added=diff_r1.lines_added,
                lines_removed=diff_r1.lines_removed,
                files_changed=1,
            ),
        ),
        evaluation_context={"dataset": "mimic-iv", "split": "dev"},
        performance_delta={"accuracy": +0.09, "execution_success": +0.10},
    )

    # -- edge 1→2 : retrieval change --
    diff_r2 = diff_strings(OLD_RETRIEVER, NEW_RETRIEVER, "retrieval/retriever.py")
    e2 = CSIGEdge(
        edge_id="e_v1_v2",
        parent_id="agent_v1",
        child_id="agent_v2",
        descriptor=ModificationDescriptor(
            mod_types=classify_mod_types(
                filenames=["retrieval/retriever.py"],
                rationale="Increased top-k and added reranking",
            ),
            modules_changed=["retrieval"],
            scope="localized",
            rationale="Increased top-k from 3 to 5 and added reranking step",
            diff_stats=DiffStats(
                lines_added=diff_r2.lines_added,
                lines_removed=diff_r2.lines_removed,
                files_changed=1,
            ),
        ),
        evaluation_context={"dataset": "mimic-iv", "split": "dev"},
        performance_delta={"accuracy": +0.04, "execution_success": +0.05},
    )

    # -- edge 0→1b : tool selection change (rejected branch) --
    diff_r3 = diff_strings(OLD_TOOL_ROUTER, NEW_TOOL_ROUTER, "tools/router.py")
    e3 = CSIGEdge(
        edge_id="e_v0_v1b",
        parent_id="agent_v0",
        child_id="agent_v1b",
        descriptor=ModificationDescriptor(
            mod_types=classify_mod_types(
                filenames=["tools/router.py"],
                rationale="Added schema inspector routing",
            ),
            modules_changed=["tools"],
            scope="localized",
            rationale="Routed schema-related queries to schema_inspector tool",
            diff_stats=DiffStats(
                lines_added=diff_r3.lines_added,
                lines_removed=diff_r3.lines_removed,
                files_changed=1,
            ),
        ),
        evaluation_context={"dataset": "mimic-iv", "split": "dev"},
        performance_delta={"accuracy": -0.04, "execution_success": -0.05},
    )

    # -- edge 2→3 : schema linking change --
    diff_r4 = diff_strings(
        OLD_SCHEMA_LINKER, NEW_SCHEMA_LINKER, "schema/linker.py"
    )
    e4 = CSIGEdge(
        edge_id="e_v2_v3",
        parent_id="agent_v2",
        child_id="agent_v3",
        descriptor=ModificationDescriptor(
            mod_types=classify_mod_types(
                filenames=["schema/linker.py"],
                rationale="Replaced naive truncation with relevance scoring",
            ),
            modules_changed=["schema"],
            scope="localized",
            rationale="Schema linker now scores relevance and filters low-scoring tables",
            diff_stats=DiffStats(
                lines_added=diff_r4.lines_added,
                lines_removed=diff_r4.lines_removed,
                files_changed=1,
            ),
        ),
        evaluation_context={"dataset": "mimic-iv", "split": "dev"},
        performance_delta={"accuracy": +0.03, "execution_success": +0.03},
    )

    return [e1, e2, e3, e4]


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    _section("1. Building nodes and edges")
    nodes = build_nodes()
    edges = build_edges()
    print(f"  Created {len(nodes)} nodes and {len(edges)} edges")

    _section("2. Constructing CSIGraph")
    graph = CSIGraph()
    for n in nodes:
        graph.add_node(n)
        print(f"  + node  {n.node_id:<14}  (parent={n.parent_id})")
    for e in edges:
        graph.add_edge(e)
        mod_labels = ", ".join(e.descriptor.mod_types)
        print(f"  + edge  {e.edge_id:<14}  [{mod_labels}]")
    print(f"\n  Graph: {graph}")

    _section("3. Saving graph")
    graph.save(SAVE_PATH)
    print(f"  Saved to {SAVE_PATH}")

    _section("4. Reloading graph")
    loaded = CSIGraph.load(SAVE_PATH)
    print(f"  Loaded: {loaded}")

    _section("5. Summary")
    pprint.pprint(loaded.summary(), width=72)

    _section("6. Query: all prompt_template_change edges")
    prompt_edges = loaded.get_edges_by_mod_type(PROMPT_TEMPLATE_CHANGE)
    for e in prompt_edges:
        print(f"  edge {e.edge_id}: {e.parent_id} -> {e.child_id}")
        print(f"    rationale: {e.descriptor.rationale}")

    _section("7. Query: nodes where department=cardiology")
    cardio_nodes = loaded.query_nodes_by_context({"department": "cardiology"})
    for n in cardio_nodes:
        print(f"  {n.node_id}  accepted={n.accepted}  metrics={n.metrics}")

    _section("8. Query: ancestors of agent_v3")
    ancestors = loaded.get_ancestors("agent_v3")
    for a in ancestors:
        print(f"  {a.node_id}")

    _section("9. Query: descendants of agent_v0")
    descendants = loaded.get_descendants("agent_v0")
    for d in descendants:
        print(f"  {d.node_id}")

    _section("10. Query: children of agent_v0")
    children = loaded.get_children("agent_v0")
    for c in children:
        print(f"  {c.node_id}  accepted={c.accepted}")

    _section("11. DAG validation")
    s = loaded.summary()
    print(f"  is_dag = {s['is_dag']}")

    print("\n[OK] Toy CSIG demo complete.\n")


if __name__ == "__main__":
    main()
