# Causal Self-Improvement Graphs: Structured Credit Assignment for Open-Ended Agent Evolution Under Distribution Shift

**Target Venue:** NeurIPS 2026 (Main Track)
**Area:** Self-Improving Agents / Open-Ended Learning / Clinical AI

---

## 1. The Core Technical Problem (Not an Application Gap — an Algorithmic One)

The DGM → DGM-H → GEA lineage of self-improving agents shares a **fundamental architectural flaw**: the archive is a flat, unstructured collection of agent variants with no mechanism for understanding *why* a modification worked, *when* it will continue to work, or *for whom* it works best.

### What exactly is broken?

In DGM-H (HyperAgents, Zhang et al., Mar 2026), the self-improvement loop is:

```
1. Sample parent agent from archive
2. Meta-agent generates a code modification
3. Evaluate modified agent on benchmark → scalar score
4. If interesting, add to archive as stepping stone
5. Repeat
```

**Three deep limitations emerge:**

**(L1) No causal credit assignment for modifications.**
When agent_v42 outperforms agent_v41, the system records the *outcome* but not the *mechanism*. A single modification might change 15 lines of code touching retrieval logic, prompt structure, and tool selection simultaneously. The archive cannot distinguish which sub-component caused the gain. This means the meta-agent cannot learn *transferable modification strategies* — it just tries things and hopes.

The HyperAgents paper itself acknowledges this: *"the heuristic that downweights parents with many compilable children is unvalidated; alternative credit assignment and its impact on exploration/exploitation balance is unexplored."*

**(L2) Context-blind archive retrieval.**
Archive selection treats all stepping stones as equally relevant regardless of the *current evaluation context*. But in any domain with heterogeneous inputs (patient subgroups in EHR, code repository types in SWE-bench, etc.), a modification that helps for one context may hurt another. The archive has no mechanism for context-conditional selection.

GEA (Group-Evolving Agents, Zhu et al., Feb 2026) patches branch isolation via experience sharing but the archive remains context-blind — it still selects parents without conditioning on the target context.

**(L3) No temporal validity under distribution shift.**
Every archived agent is treated as permanently valid. But in non-stationary environments, older agents and the modifications that produced them may encode assumptions about a data distribution that no longer holds. There is no mechanism for detecting when an archived stepping stone has become a *misleading* stepping stone.

### Why this matters beyond any single application

These are not EHR-specific problems. They are fundamental limitations of population-based self-improvement algorithms operating in heterogeneous, non-stationary environments. The same issues apply to self-improving coding agents operating across diverse repositories, self-improving robotics agents deployed across different morphologies, or any real-world deployment of open-ended self-improvement.

**EHR is the ideal testbed** because it exposes all three limitations simultaneously: natural patient subgroup heterogeneity (L2), temporal distribution shift across years (L3), and multi-component clinical reasoning pipelines where credit assignment is genuinely difficult (L1).

---

## 2. Technical Contribution: Causal Self-Improvement Graphs (CSIG)

We propose replacing the flat archive in DGM-H with a **Causal Self-Improvement Graph (CSIG)** — a structured, queryable representation of the improvement landscape that enables principled credit assignment, context-conditional retrieval, and temporal validity tracking.

### 2.1 Formal Framework

**Definition 1 (Modification Descriptor).** When the meta-agent modifies agent $a_i$ to produce $a_j$, we extract a *modification descriptor* $\delta_{ij} \in \mathcal{D}$ that captures the semantic type of change. This is a structured representation obtained by:
1. Computing the code diff between $a_i$ and $a_j$
2. Using the FM to classify the diff into a taxonomy of modification types $\mathcal{T} = \{t_1, ..., t_K\}$ (e.g., "added retrieval augmentation", "modified prompt template", "introduced error retry logic", "changed tool selection heuristic")
3. Encoding additional metadata: which functional module was changed, the scope of the change (localized vs architectural), and the meta-agent's stated rationale

The taxonomy $\mathcal{T}$ is itself *part of the editable codebase*, so the hyperagent can refine its own modification classification over time (true metacognition over the improvement graph structure).

**Definition 2 (Causal Self-Improvement Graph).** A CSIG is a labeled DAG $G = (V, E, \phi, \psi)$ where:
- $V$ = set of agent variants (nodes)
- $E \subseteq V \times V$ = parent→child modification edges
- $\phi: E \rightarrow \mathcal{D}$ = modification descriptor labeling
- $\psi: E \times \mathcal{C} \times \mathcal{W} \rightarrow \mathbb{R}^d$ = contextual causal effect estimator, where $\mathcal{C}$ is the space of evaluation contexts and $\mathcal{W}$ is a temporal window index

### 2.2 Contribution 1: Causal Credit Assignment via Interventional Graph Analysis

**Problem:** Given that the archive DAG contains multiple instances of similar modification types (e.g., "added retrieval" appears in 12 different branches), can we estimate the *causal effect* of each modification type on downstream performance, controlling for confounders?

**Approach:** We formulate this as a causal inference problem on the archive DAG. The key insight is that the archive *already contains natural experiments* — the same modification type applied to different parent agents in different contexts creates a natural quasi-experiment.

**Algorithm: DAG-Grounded Causal Effect Estimation (DAGCE)**

```
Input: Archive DAG G, modification taxonomy T, performance records P
Output: Causal effect estimates θ_t for each modification type t ∈ T

1. For each modification type t ∈ T:
   a. Identify all edges e ∈ E where φ(e) involves type t
   b. For each such edge (a_i → a_j):
      - Compute performance delta: Δ_ij = perf(a_j) - perf(a_i)
      - Extract confounders: other modification types co-occurring in δ_ij,
        parent agent quality, evaluation context
   c. Estimate Average Treatment Effect (ATE) of modification type t:
      θ_t = E[Δ | do(t)] using inverse propensity weighting
      on the set of edges, treating modification co-occurrence as confounders
   d. Estimate Conditional ATE (CATE) conditioned on evaluation context:
      θ_t(c) = E[Δ | do(t), context=c]

2. Build a modification interaction model:
   For pairs (t_i, t_j), estimate interaction effects:
   θ_{t_i, t_j} = E[Δ | do(t_i, t_j)] - θ_{t_i} - θ_{t_j}
   (captures synergistic or antagonistic modification combinations)
```

**Why this is non-trivial:** Unlike standard causal inference where you have i.i.d. samples, the archive DAG has a specific graph structure where later agents are descendants of earlier ones. This introduces *path dependence* — the effect of a modification depends on what modifications preceded it. We handle this by conditioning on the *ancestor modification history* as part of the confounder set, and use doubly-robust estimation to handle potential model misspecification.

**Theoretical Result 1:** Under standard assumptions (overlap, consistency, no unmeasured confounders conditioned on ancestor history), the DAGCE estimator converges to the true CATE at rate $O(n^{-1/2})$ where $n$ is the number of edges involving modification type $t$. We derive finite-sample confidence intervals that account for the DAG dependency structure using a dependency-adjusted CLT.

### 2.3 Contribution 2: Context-Conditional Archive Selection as a Structured Bandit

**Problem:** Given a new evaluation context $c$ (e.g., a patient subgroup, a task type), which archived agent should we select as the parent for the next self-improvement step?

**Approach:** We formulate archive selection as a **contextual bandit over a combinatorial action space**, where the context is the evaluation setting and the arms are *modification strategies* (sequences of modification types to apply).

**Algorithm: CSIG-UCB (Contextual Self-Improvement Graph Upper Confidence Bound)**

```
Input: CSIG G, current context c, causal effect estimates θ
Output: Selected parent agent and suggested modification strategy

1. For each candidate parent a_i ∈ archive:
   a. Compute ancestor modification history h_i = ancestor_path(a_i, G)
   b. For each candidate modification type t ∈ T:
      - Predict expected improvement: μ̂(a_i, t, c) = θ_t(c) + θ_{h_i ∩ t}
        (causal effect of t in context c, adjusted for interaction with
        existing modification history)
      - Compute uncertainty: σ̂(a_i, t, c) from the DAGCE confidence interval
      - UCB score: U(a_i, t, c) = μ̂ + β · σ̂
   c. Select best modification type for this parent:
      t*_i = argmax_t U(a_i, t, c)
   d. Parent score: S(a_i) = U(a_i, t*_i, c)

2. Select parent: a* = argmax_{a_i} S(a_i)
3. Return (a*, t*_{a*}) — parent and recommended modification strategy
```

The meta-agent uses $t^*$ as a *suggestion* — it can still explore freely (maintaining open-endedness), but it has a principled starting point for directed improvement.

**Theoretical Result 2:** Under the assumption that causal effects are drawn from a linear contextual model with sub-Gaussian noise, CSIG-UCB achieves a dynamic regret bound of:

$$R_T = O\left(\sqrt{d \cdot K \cdot T \cdot \log T} + \Delta^{1/3} T^{2/3}\right)$$

where $d$ is the context dimension, $K$ is the number of modification types, $T$ is the number of self-improvement iterations, and $\Delta$ is the total variation budget of the environment (capturing distribution shift). The first term captures the cost of learning the causal effects; the second captures the cost of non-stationarity.

**Key insight:** This bound degrades gracefully with distribution shift — if $\Delta = 0$ (stationary), we recover the standard $\tilde{O}(\sqrt{T})$ rate. As $\Delta$ grows, the regret grows but remains sublinear as long as $\Delta = o(T)$.

### 2.4 Contribution 3: Temporal Validity via Drift-Adaptive Archive Pruning

**Problem:** Detect when archived agents and their associated modification strategies become invalid due to distribution shift, and update the CSIG accordingly.

**Algorithm: Adaptive CSIG Maintenance**

```
1. Maintain a sliding window W of recent evaluation results
2. For each modification type t, track the running CATE estimate θ_t^{(W)}
3. Compute drift statistic:
   D_t = |θ_t^{(W)} - θ_t^{(full)}| / σ̂_t
   (deviation of recent effect from historical average, normalized)

4. If D_t > threshold τ for modification type t:
   a. Flag all edges in CSIG labeled with type t as "drift-affected"
   b. Reduce sampling probability for agents in drift-affected branches
   c. Increase exploration budget for modifications that are NOT drift-affected
   d. Log the drift event with clinical context for interpretability

5. Periodically re-estimate all CATE values on the recent window only,
   creating a "temporal CSIG" that the meta-agent can query:
   "What modification strategies work NOW vs what worked HISTORICALLY?"
```

**Theoretical Result 3:** The drift detection mechanism has a false positive rate bounded by $\alpha$ (configurable) and a detection delay of $O(\sigma^2 / \delta^2)$ where $\delta$ is the true shift magnitude. Combined with CSIG-UCB, this yields a switching regret bound of $O(\sqrt{S \cdot T})$ where $S$ is the (unknown) number of stationary periods.

---

## 3. Why This is NeurIPS-Level Technical Contribution

| Dimension | What we contribute | What existed before |
|---|---|---|
| **Credit assignment** | Causal effect estimation on the archive DAG with CATE for modification types | DGM-H: none. GEA: experience sharing but no causal attribution |
| **Archive selection** | Contextual bandit formulation with UCB over modification strategies, with regret bounds | DGM-H: heuristic downweighting. GEA: random/uniform |
| **Non-stationarity** | Drift-adaptive archive pruning with detection guarantees and switching regret bounds | DGM/DGM-H/GEA: all assume stationary benchmark |
| **Theoretical** | Dynamic regret bound for contextual self-improvement under distribution shift | No existing regret analysis for any self-improving agent system |
| **Generality** | Algorithm applies to ANY population-based self-improving system, not just EHR | Contribution is to the self-improving agents field; EHR is the validation domain |

### Relationship to Concurrent Work

- **HyperAgents (Zhang et al., Mar 2026):** We extend DGM-H's flat archive to a structured causal graph. Orthogonal to their metacognitive contribution — CSIG can be combined with DGM-H's editable meta-agent.
- **GEA (Zhu et al., Feb 2026):** We address a different limitation. GEA adds experience sharing across branches (horizontal communication). We add causal structure within branches (vertical understanding). The two are complementary.
- **MemAgents Workshop (ICLR 2026):** Our CSIG is essentially a structured episodic memory for the self-improvement process, connecting to the workshop's theme of "temporal credit assignment across episodes."
- **ARTIST (ICML 2025 Workshop):** Self-improving transformer with RL and tool integration, but no population-based archive or causal analysis.

---

## 4. Experimental Design

### 4.1 Domain: Clinical EHR Reasoning (MIMIC-IV + eICU)

**Why this domain (beyond motivation):** It is the *hardest possible testbed* for our algorithm because it simultaneously stresses all three contributions:
- **Credit assignment stress:** Clinical reasoning pipelines have 5-8 interdependent components (retrieval, schema linking, code generation, execution, verification, medical knowledge integration). Modification credit is genuinely ambiguous.
- **Context heterogeneity:** Patient subgroups (by disease, demographics, acuity) require different reasoning strategies. A retrieval modification that helps for cardiology queries may hurt for nephrology.
- **Natural distribution shift:** MIMIC-IV spans 2008-2019. Treatment protocols, coding practices, and patient demographics shifted substantially. Temporal split evaluation is ecologically valid.

### 4.2 Concrete Experimental Setup

**Base Agent:** EHRAgent (Shi et al., EMNLP 2024) — code-generating LLM agent for multi-tabular EHR QA with tool use, medical knowledge integration, and long-term memory. We use the open-source implementation from `wshi83/EhrAgent`.

**Self-Improvement Framework:** DGM-H (from `facebookresearch/Hyperagents`) with our CSIG module replacing the flat archive.

**Datasets:**
- MIMIC-III + eICU (EHRSQL benchmark) — multi-table clinical QA, 4 complexity levels
- MIMIC-IV-Ext-22MCTS — 22M temporal clinical event sequences for next-event prediction
- MIMIC-IV structured data — mortality prediction, sepsis detection, ICU readmission

**Evaluation Protocol (5 experimental axes):**

**Experiment 1: Self-Improvement Efficiency (Sample Complexity)**
- Metric: improvement@k (following HyperAgents) — performance gain after k self-improvement iterations
- Compare: CSIG-DGM-H vs DGM-H vs GEA vs DGM vs static EHRAgent
- Hypothesis: CSIG's directed exploration achieves the same improvement@50 in ~20 iterations (2.5x more sample-efficient) due to causal credit assignment guiding the meta-agent

**Experiment 2: Context-Conditional Improvement**
- Split EHRSQL questions by clinical department (cardiology, pulmonology, nephrology, etc.)
- Metric: Per-department success rate after self-improvement
- Compare: CSIG-UCB (context-conditional selection) vs uniform archive selection
- Hypothesis: CSIG-UCB achieves higher minimum-department performance (Rawlsian fairness) because it learns department-specific modification strategies

**Experiment 3: Temporal Robustness Under Distribution Shift**
- Train self-improvement loop on MIMIC-IV 2008-2014 patient cohort
- Test on 2015-2019 cohort (natural temporal shift)
- Metric: Performance degradation ratio (2015-2019 vs 2008-2014)
- Compare: CSIG with drift detection vs DGM-H without (which will over-index on 2008-2014 patterns)
- Hypothesis: CSIG's drift-adaptive pruning maintains <10% degradation vs >25% for flat archive

**Experiment 4: Cross-Task Meta-Transfer**
- Pre-train CSIG on mortality prediction task
- Transfer the learned causal effect model to sepsis detection task (zero-shot meta-transfer)
- Metric: improvement@k on the new task with transferred vs fresh CSIG
- Hypothesis: Causal effect estimates for modification types (e.g., "adding temporal attention" helps for time-series tasks) transfer across clinical prediction tasks

**Experiment 5: Ablation and Theoretical Validation**
- Ablate each CSIG component: (a) causal credit assignment only, (b) context-conditional selection only, (c) drift detection only
- Validate regret bounds: plot empirical regret vs theoretical $O(\sqrt{T})$ and $O(\Delta^{1/3}T^{2/3})$ curves
- Validate drift detection: inject synthetic distribution shifts of known magnitude, measure detection delay vs theoretical prediction

### 4.3 Baselines

| Baseline | What it tests |
|---|---|
| Static EHRAgent (EMNLP 2024) | No self-improvement at all |
| TrustEHRAgent (2025) | Static agent with confidence estimation |
| MDAgents (NeurIPS 2024) | Multi-agent collaboration, no self-improvement |
| DGM-H (HyperAgents, 2026) | Self-improvement with flat archive |
| GEA (2026) | Self-improvement with experience sharing |
| CSIG w/o causal credit | Context-conditional + drift, but random credit |
| CSIG w/o context conditioning | Causal credit + drift, but uniform selection |
| CSIG w/o drift detection | Causal credit + context, but no temporal adaptation |
| **CSIG-DGM-H (Ours)** | Full framework |

---

## 5. Implementation Strategy

### Codebase

Build on two open-source repositories:
1. `facebookresearch/Hyperagents` — DGM-H loop, archive management, meta-agent scaffolding
2. `wshi83/EhrAgent` — Clinical EHR agent with MIMIC-III/eICU tooling, code generation, medical knowledge integration

The CSIG module is a **drop-in replacement** for the flat archive in `Hyperagents`. Core new code:
- `csig/graph.py` — DAG construction and querying
- `csig/credit.py` — DAGCE causal effect estimator
- `csig/bandit.py` — CSIG-UCB contextual selection
- `csig/drift.py` — Temporal drift detection and archive pruning
- `csig/descriptors.py` — Modification descriptor extraction (LLM-based diff classification)

### Compute Requirements

- DGM-H loop uses FM API calls (Claude/GPT-4), not GPU training → accessible to academic labs
- Each self-improvement iteration: ~$0.50-$2.00 in API costs (similar to EHRAgent's reported $0.17-$0.60 per query)
- 200 iterations × 5 experiment configurations × 3 seeds = ~3000 iterations total
- Estimated total API cost: $3,000-$6,000 (feasible for a research project)
- MIMIC-IV/eICU: Free via PhysioNet credentialing

---

## 6. Paper Structure (8 pages + appendix)

1. **Introduction** (1 page): Frame the archive management problem as a general limitation of self-improving agents. Motivate with clinical EHR as the hard case.
2. **Background** (0.75 pages): DGM, DGM-H, GEA. Formal notation.
3. **Causal Self-Improvement Graphs** (2.5 pages): Formal framework, DAGCE algorithm, CSIG-UCB algorithm, drift detection. Theorem statements.
4. **Theoretical Analysis** (1 page): Regret bounds, drift detection guarantees. Proof sketches (full proofs in appendix).
5. **Experiments** (2 pages): All 5 experiments with tables and figures.
6. **Discussion** (0.75 pages): Limitations, broader impact, safety considerations for self-improving clinical agents.

---

## 7. Anticipated Reviewer Concerns and Rebuttals

**Q: "Is this just applying bandits to archive selection?"**
A: No. The core novelty is the *causal effect estimation on the archive DAG* — using the graph structure of the self-improvement process as a natural experiment to learn which modification types are causally effective. The bandit formulation is built ON TOP of the causal estimates. Without the causal structure, you'd have a standard contextual bandit with noisy rewards; with it, you have a model-based bandit with structured prior knowledge from the improvement history.

**Q: "The causal assumptions (no unmeasured confounders) are strong."**
A: True. We provide two mitigations: (1) doubly-robust estimation that is consistent if either the propensity model or outcome model is correct, and (2) sensitivity analysis showing how violations of the no-unmeasured-confounders assumption affect our estimates. We also argue that the self-improvement setting is *more favorable* for causal inference than typical observational studies because the meta-agent's modification decisions are partially randomized (due to LLM stochasticity).

**Q: "How do you ensure the modification taxonomy is meaningful?"**
A: The taxonomy is bootstrapped from LLM-based diff classification and refined by the meta-agent over time. We validate it via a held-out human annotation study (10% of modifications classified by 3 annotators, measuring inter-rater agreement with the LLM-derived taxonomy). The taxonomy is also part of the editable codebase, so the hyperagent can refine it if it's not useful.

**Q: "Clinical safety — isn't self-modifying clinical AI dangerous?"**
A: We explicitly design CSIG with a safety constraint: no modification that reduces performance on ANY patient subgroup by more than ε below the current baseline is accepted into the archive. This is enforced at evaluation time and is orthogonal to the CSIG algorithm. We also provide the full audit trail (modification descriptors, causal effect estimates, drift logs) as an interpretability artifact.
