# Phase 6 Development Sanity Check Results

> [!WARNING]
> This document contains smoke-test results intended only to verify functional correctness. These measurements are not used in the paper's evaluation and must not be interpreted as benchmark results.

---

## 1. Dev Sanity Check Runs

We executed LiteAgent and all five baseline comparators on a workstation loopback setup using three representative tasks from the validation dataset:
*   **Task 2** (GSM8K): Low complexity (`Compute 15 + 23.`)
*   **Task 13** (HotpotQA): Medium complexity (`Compare the birth years of the directors of 'Inception' and 'Interstellar'...`)
*   **Task 26** (HumanEval): Medium/High complexity (`Write a python class implementing a red-black tree...`)

### Execution Trace Outcomes

| System Profile | Baseline Identifier | Task 2 (GSM8K) | Task 13 (HotpotQA) | Task 26 (HumanEval) | Caching Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LiteAgent (Full)** | `"liteagent"` | Routed/Executed: **Small** | Routed/Executed: **Medium** | Routed/Executed: **Medium** | persistent 3-tier active |
| **Static Full-Pipeline** | `"static_full"` | Routed/Executed: **Large** | Routed/Executed: **Large** | Routed/Executed: **Large** | disabled (full prefill) |
| **LiteAgent (Routing Only)** | `"routing_only"` | Routed/Executed: **Small** | Routed/Executed: **Medium** | Routed/Executed: **Medium** | disabled (full prefill) |
| **LiteAgent (Cache Only)** | `"cache_only"` | Routed/Executed: **Large** | Routed/Executed: **Large** | Routed/Executed: **Large** | persistent 3-tier active |
| **RouteLLM-style Heuristic** | `"routellm_heuristic"` | Routed/Executed: **Small** | Routed/Executed: **Medium** | Routed/Executed: **Medium** | disabled (full prefill) |
| **Flat In-Memory Cache (Slots=1)** | `"flat_cache"` | Routed/Executed: **Small** | **EXPECTED_LIMIT_REACHED** | (Not Executed) | prefix-only (no swap) |

---

## 2. Flat In-Memory Cache Failure Mode Verification

In-memory prefix cache depletion was stress-tested by setting `max_in_memory_slots = 1` and executing two sequential session requests:
1.  **Task 2** (`session_fc1`): Succeeded, occupying the single in-memory slot.
2.  **Task 13** (`session_fc2`): Attempted to save cache state. Since the slot ceiling was met and secondary storage swap was disabled, the cache manager threw a `RuntimeError` with the structured reason code **`EXPECTED_LIMIT_REACHED`** and logged it to the operation log. No unexpected implementation allocation (`FAILURE_ALLOCATION`) or memory crashes (`FAILURE_MEMORY_LIMIT`) occurred, confirming correct failure boundary handling.
