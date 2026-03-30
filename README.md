# Changelog — Causal Self-Improvement Graphs (CSIG)

---

## Stage 1 — Minimal CSIG Foundation

**Date:** 2026-03-26

### Overview

Built the complete Stage 1 skeleton for the CSIG research project: a directed acyclic graph archive that represents agent variants as nodes and self-improvement edits as directed edges, with structured modification descriptors, persistence, and querying.

---

### Files Created

#### Core Library (`csig/`)

| File | Description |
|------|-------------|
| `csig/__init__.py` | Package initialiser; re-exports all public symbols for clean imports (`from csig import CSIGraph, CSIGNode, ...`) |
| `csig/schemas.py` | Dataclass definitions for `CSIGNode`, `CSIGEdge`, `ModificationDescriptor`, and `DiffStats` with full `to_dict()` / `from_dict()` round-trip serialisation |
| `csig/taxonomy.py` | 10-label modification-type taxonomy as module-level constants plus `ALL_MOD_TYPES` frozenset and `is_valid_mod_type()` validator |
| `csig/graph.py` | `CSIGraph` class backed by `networkx.DiGraph` with 12 methods: `add_node`, `add_edge`, `get_node`, `get_parent`, `get_children`, `get_ancestors`, `get_descendants`, `get_edges_by_mod_type`, `query_nodes_by_context`, `save`, `load`, `summary` |
| `csig/diff_parser.py` | Rule-based diff engine using `difflib.SequenceMatcher`; compares strings or file-sets and returns `DiffStats` plus inferred module names |
| `csig/classifier.py` | Keyword-rule classifier that infers modification types from filenames, module names, keywords, and rationale text (~30 substring rules) |
| `csig/persistence.py` | Standalone JSON (batch) and JSONL (streaming/append) save/load helpers for nodes and edges |

#### Scripts

| File | Description |
|------|-------------|
| `build_toy_csig.py` | Synthetic demo — constructs a 5-node / 4-edge DAG simulating an EHR agent improving over 4 iterations, saves/reloads the graph, and runs sample queries |
| `test_csig.py` | Automated test suite — 26 tests covering schemas, taxonomy, graph operations, cycle detection, persistence round-trips, diff parsing, and classifier accuracy |

#### Configuration

| File | Description |
|------|-------------|
| `requirements.txt` | Single dependency: `networkx>=3.0` |

#### Output (generated at runtime)

| File | Description |
|------|-------------|
| `output/toy_csig.json` | Serialised toy graph produced by `build_toy_csig.py` |

---

### Schemas Implemented

#### CSIGNode

| Field | Type | Purpose |
|-------|------|---------|
| `node_id` | `str` | Unique identifier (auto-generated 12-char hex if omitted) |
| `parent_id` | `str or None` | ID of the parent node (`None` for root) |
| `timestamp` | `str` | ISO-8601 UTC timestamp of creation |
| `task_name` | `str` | Name of the task this variant targets |
| `context_summary` | `dict` | Arbitrary key-value context (department, dataset, run, etc.) |
| `metrics` | `dict[str, float]` | Performance metrics (accuracy, execution_success, etc.) |
| `accepted` | `bool` | Whether this variant was accepted into the improvement lineage |

#### CSIGEdge

| Field | Type | Purpose |
|-------|------|---------|
| `edge_id` | `str` | Unique identifier |
| `parent_id` | `str` | Source node ID |
| `child_id` | `str` | Target node ID |
| `descriptor` | `ModificationDescriptor` | Structured description of what changed |
| `evaluation_context` | `dict` | Context under which the improvement was evaluated |
| `performance_delta` | `dict[str, float]` | Metric deltas (positive = improvement) |

#### ModificationDescriptor

| Field | Type | Purpose |
|-------|------|---------|
| `mod_types` | `list[str]` | Taxonomy labels from the 10-type system |
| `modules_changed` | `list[str]` | Names of changed modules/components |
| `scope` | `str` | One of `localized`, `multi_module`, `architectural` |
| `rationale` | `str` | Human-readable explanation of the change |
| `diff_stats` | `DiffStats` | Lines added, lines removed, files changed |

---

### Taxonomy (10 Modification Types)

| Label | Triggered By (keywords) |
|-------|------------------------|
| `retrieval_change` | retriev, rag, search |
| `prompt_template_change` | prompt, template |
| `tool_selection_change` | tool, router, dispatch |
| `reasoning_step_change` | reason, chain_of, cot |
| `memory_update` | memory, cache, buffer |
| `error_retry_logic` | retry, error, fallback |
| `verifier_change` | verif, check, valid |
| `schema_linking_change` | schema, link |
| `execution_guardrail_change` | guardrail, guard, sandbox |
| `decomposition_planning_change` | decompos, plan, subtask |

---

### Graph Methods Implemented

| Method | Description |
|--------|-------------|
| `add_node(node)` | Insert an agent variant; rejects duplicates |
| `add_edge(edge)` | Insert a directed edge; rejects missing endpoints and cycles |
| `get_node(node_id)` | Retrieve a node by ID |
| `get_parent(node_id)` | Retrieve the parent node recorded in the node schema |
| `get_children(node_id)` | List all immediate successors in the DAG |
| `get_ancestors(node_id)` | List all transitive ancestors |
| `get_descendants(node_id)` | List all transitive descendants |
| `get_edges_by_mod_type(mod_type)` | Filter edges by a taxonomy label |
| `query_nodes_by_context(filters)` | Filter nodes by key-value pairs in `context_summary` |
| `save(path)` | Serialise the full graph to JSON |
| `load(path)` | Deserialise a graph from JSON (class method) |
| `summary()` | Return statistics: node/edge counts, accepted/rejected, DAG check, mod-type distribution |

---

### Test Results

```
26 / 26 tests passed
```

| Category | Tests |
|----------|-------|
| Schema round-trips | 3 |
| Taxonomy validation | 2 |
| Graph operations | 9 |
| Persistence (JSON + JSONL) | 4 |
| Diff parser | 2 |
| Classifier | 5 |
| **Total** | **26** (pending: 0, failed: 0) |

---

### What Is NOT in Stage 1

The following are explicitly deferred to later stages:

- Causal inference / causal modelling
- Contextual bandits (UCB, Thompson sampling)
- Distribution drift detection
- Full EHRAgent / HyperAgents integration
- LLM-based descriptor generation
- Interactive visualisation

---

### Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all 26 tests
python test_csig.py

# Run the synthetic demo (builds, saves, reloads, queries)
python build_toy_csig.py
```
