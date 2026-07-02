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
| **LiteAgent (Ours)** | **Yes (Complexity)**| **Yes (Three-Tier)** | **Yes (Three-Tier)** | **Yes (Planner/Critic)**| **Yes (Pi 5)** | **Yes (Pi + PC Co-design)** | Co-designed complexity routing and KV-cache hierarchy for edge multi-agent workflows. |

---

## 2. Closest Baselines & Feasibility Analysis

Reviewers will naturally expect LiteAgent to be compared against prior routing and caching implementations. We identify three closest baselines:

### Baseline 1: RouteLLM
*   **Feasibility**: **Fully Reimplementable**
*   **Approximation Plan (Phase 6)**: We can adapt RouteLLM's public codebase to route between our local Small (`llama3.2:1b`), Medium (`llama3.2:3b`), and Large (`llama3.1:8b`) models on our local network. We will use their classification thresholds (trained on preference datasets) as the baseline routing comparator for our custom complexity-aware router.

### Baseline 2: LMCache / vLLM Prefix Caching
*   **Feasibility**: **Partially Reimplementable**
*   **Approximation Plan (Phase 6)**: We cannot easily run full vLLM or LMCache on a Raspberry Pi 5 due to their heavy dependency on CUDA-specific libraries (like PagedAttention GPU kernels). However, we can approximate this baseline by running standard in-memory caching (retaining KV-cache parameters in host RAM without swapping or eviction tiering) on our local Python/Ollama runtime to contrast memory limits and eviction latencies.

### Baseline 3: KVFlow (Multi-Agent Caching Baseline)
*   **Feasibility**: **Related Work Only (Simulated Reimplementation)**
*   **Approximation Plan (Phase 6)**: KVFlow relies on orchestrating prefix caches across a cloud cluster based on static workflow graphs. Since we operate on a heterogeneous edge-workstation setup, we can only simulate KVFlow’s scheduling logic locally by comparing our dynamic routing and caching against a static workflow path (where agent calls are fixed to specific models without dynamic runtime routing).
