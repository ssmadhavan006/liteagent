# Phase 6 Baseline Inventory & Scope Confirmation

This document defines the baseline comparison systems to be constructed in **Phase 6** to evaluate LiteAgent against standard routing, caching, and static execution baselines, as outlined in the [Evaluation Plan](file:///d:/Coding/liteagent/docs/phase2/evaluation_plan.md).

---

## 1. Baseline Inventory

### 1. Static Full-Pipeline
*   **Description**: A static execution baseline representing standard multi-agent pipeline setups.
*   **Form**: Static mapping to the **Large tier** (`llama3.1:8b`). All four agents (Planner, Retriever, Executor, Critic) are always active.
*   **Cache Behavior**: Caching/persistence is disabled. Every context switch executes a fresh prefill from scratch.
*   **Log Identifier**: `"static_full"`
*   **Feature Flags**: `{"routing": false, "cache": false, "grpc": true}`

### 2. LiteAgent (Routing Only)
*   **Description**: Ablates the contribution of the three-tier cache manager.
*   **Form**: Toggled flag within the real LiteAgent execution pipeline. The Complexity Router and agent pruning are active.
*   **Cache Behavior**: Caching is disabled. Every routed task performs a fresh prompt prefill from scratch.
*   **Log Identifier**: `"routing_only"`
*   **Feature Flags**: `{"routing": true, "cache": false, "grpc": true}`

### 3. LiteAgent (Cache Only)
*   **Description**: Ablates the contribution of the complexity-aware router.
*   **Form**: Toggled flag within the real LiteAgent execution pipeline. Caching remains fully active.
*   **Routing Behavior**: Complexity routing is disabled. All tasks are statically mapped to the Large tier, and all four agents are always active.
*   **Log Identifier**: `"cache_only"`
*   **Feature Flags**: `{"routing": false, "cache": true, "grpc": true}`

### 4. RouteLLM-style Heuristic
*   **Description**: Compares LiteAgent's complexity router against a simplified RouteLLM-style heuristic.
*   **Form**: A standalone router module `src/liteagent/baselines/routellm_heuristic_approx.py`.
*   **Log Identifier**: `"routellm_heuristic"`
*   **Feature Flags**: `{"routing": true, "cache": false, "grpc": true}`
*   **Methodological Note**: This is a simple independent heuristic inspired by cost-aware routing literature, not a reimplementation of RouteLLM's trained router — included as a representative alternative-heuristic baseline, not a faithful reproduction. It is fully independent of LiteAgent's hand-tuned density features, calculating utility using a keyword-overlap density score against a list of 19 reasoning and technical indicators (e.g. `def`, `class`, `logic`, `solve`). It approximates RouteLLM's core decision principle (routing to the cheapest model predicted to satisfy a utility threshold), enabling evaluation of different architectural routing boundaries.
*   **Threshold Settings**: Calibrated at `threshold = 0.12`. Prompts scoring below `0.04` route to Small; between `0.04` and `0.12` route to Medium; above `0.12` route to Large.

### 5. Flat In-Memory Cache (vLLM-inspired)
*   **Description**: A single-tier prefix cache replicating standard in-memory KV-cache setups without secondary storage swapping.
*   **Form**: Standalone cache manager class `src/liteagent/baselines/vllm_prefix_cache_approx.py`.
*   **Log Identifier**: `"flat_cache"`
*   **Feature Flags**: `{"routing": false, "cache": true, "grpc": true}` (or customized for the specific run).
*   **Eviction Behavior**: Single-tier in-memory cache with **no eviction** to RAM/SSD storage.
*   **Phase 8 Slot Configuration**: While the dev smoke test used a tiny ceiling of `slots=1` to force immediate capacity failures, the real evaluation sweeps in Phase 8 will configure the Flat Cache baseline with a slot budget matching LiteAgent's active RAM cache limit (`max_ram_states = 5`) to measure capacity exhaustion under realistic design loads.
*   **Capacity Limit Classifications**:
    *   `EXPECTED_LIMIT_REACHED`: Cache slot allocation is exhausted under design load limits (expected behavior).
    *   `FAILURE_ALLOCATION`: Unexpected allocation error (runtime/implementation issue).
    *   `FAILURE_MEMORY_LIMIT`: Exceeds host memory bounds.

---

## 2. Metric and Parity Checking

All baselines hook into the exact same logging schema defined in Phase 5:
*   Logs are written to `experiments/operations.jsonl`.
*   An automated schema validation test will verify that all baseline runs write identical keys and types to log payloads to ensure perfect parity for evaluation sweeps.
