# LiteAgent — Comparison Table & Baseline Analysis

This document presents the system comparison table contrasting LiteAgent against state-of-the-art architectures, followed by an feasibility analysis of the closest baseline systems.

---

## 1. System Comparison Matrix

| Method | Routing | Dynamic Model Selection | KV Persistence | Multi-Agent | Edge | Heterogeneous Hardware | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RouteLLM** (2024) | Yes (ML Router) | Yes (Binary Selection) | No | No | No | No | Selects between strong/weak cloud models using human preference data. |
| **FrugalGPT** (2023) | Yes (Cascade) | Yes (Multi-Model) | No | No | No | No | Sequential fallback based on confidence thresholds. |
| **vLLM** (2023) | No | No | Partial (Prefix Cache) | No | No | No | Virtual memory paged KV-cache management for high throughput GPU serving. |
| **StreamingLLM** (2024)| No | No | Partial (Window Sink) | No | No | No | Attention-sink and sliding-window KV management for infinite context. |
| **LMCache** (2025) | No | No | Yes (Host/Remote) | No | No | No | Decoupled, engine-independent KV cache layer over tiered server storage. |
| **KVFlow** (2024) | No | No | Yes (Graph-Aware) | Yes (Scheduler) | No | No | Workflow-graph prefix caching for datacenter multi-agent setups. |
| **Mixture-of-Agents** (2024)| No | No | No | Yes (Layers) | No | No | Multi-model consensus layers to improve generation reasoning. |
| **LiteAgent (Ours)** | **Yes (Complexity)**| **Yes (Three-Tier)** | **Yes (Three-Tier)** | **Yes (4-role chain)**| **Yes (Pi 5)** | **Yes (Pi + PC Co-design)** | Co-designed complexity routing and KV-cache hierarchy for edge multi-agent workflows. |

---

## 2. Closest Baselines & Feasibility Analysis

Reviewers will naturally expect LiteAgent to be compared against prior routing and caching implementations. We identify three closest baselines.

> [!IMPORTANT]
> **None of the comparators below is the original system.** Each is a
> reimplemented approximation built to run on our hardware. We therefore make no
> claim of the form "LiteAgent outperforms RouteLLM" or "…outperforms vLLM."
> Every claim is scoped to the named approximation, and every results table must
> carry the `-approx` suffix and the caveat in its caption. The approximations
> are deliberately *architectural* comparators — they isolate a mechanism (learned
> routing signal, flat single-tier caching) rather than reproduce a tuned system,
> and an untuned approximation will generally understate the original's quality.

### Baseline 1: RouteLLM-style heuristic (`routellm_heuristic_approx`)
*   **Feasibility**: **Approximated — NOT the published system**
*   **What we actually implemented**: An independent keyword/feature-density linear scorer that routes between our Small/Medium/Large tiers. It reproduces RouteLLM's *architectural pattern* (a cheap classifier gating model selection), not its method.
*   **What differs from published RouteLLM**: RouteLLM trains its router on human preference data (Chatbot Arena) and calibrates against a cost/quality target. Ours is untrained and rule-based. **This comparator is therefore weaker than published RouteLLM and must not be presented as equivalent.**
*   **Why not the real thing**: The public codebase targets cloud API endpoints and its released routers are trained for a strong/weak *pair*, not our three local tiers; porting the trained checkpoints to this tier structure is out of scope for Phase 6.

### Baseline 2: Flat in-memory prefix cache (`vllm_prefix_cache_approx`)
*   **Feasibility**: **Approximated — NOT vLLM**
*   **What we actually implemented**: A single-tier, fixed-slot in-memory KV state cache with capacity exhaustion signalling (`EXPECTED_LIMIT_REACHED`), isolating the *absence of tiering* as the variable under test.
*   **What differs from published vLLM/LMCache**: No PagedAttention block allocator, no paged virtual memory, no continuous batching, no GPU kernel integration. This comparator isolates one property (flat vs. tiered residency); it is not a throughput comparison against vLLM.
*   **Why not the real thing**: vLLM's PagedAttention path depends on CUDA kernels unavailable on the Raspberry Pi 5 target, so it cannot run on the edge half of the testbed at all.

### Baseline 3: KVFlow (agent-aware caching)
*   **Feasibility**: **Not reproduced — related work only**
*   **Status**: We do **not** implement a KVFlow comparator. The Phase 6 "static workflow path" configuration tests *our own* static-routing ablation and is reported as such (`static_full`); it must not be labelled as KVFlow.
*   **Why**: Reproducing KVFlow's workflow-graph scheduler faithfully is a substantial engineering effort, and an unfaithful version would be a strawman on the one axis closest to our contribution. We instead position against it argumentatively in `novelty_matrix.md` and state the gap as a limitation.
