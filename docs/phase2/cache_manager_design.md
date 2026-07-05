# LiteAgent — Three-Tier KV-Cache Manager Design

This document outlines the architecture, tier allocations, serialization formats, eviction policies, and context restoration mechanisms of the LiteAgent Three-Tier KV-Cache Manager.

---

## 1. Context Serialization Primitive

Following our Phase 2 investigation, we bypass the high-level Ollama API for cache-critical paths and base our design on `llama-cpp-python` / `llama.cpp` context serialization APIs:

*   **Extraction (`save_state()`)**: Extracts the complete running model state (including token history, active context slots, and KV cache tensors) as a raw binary byte array (`bytes` object).
*   **Injection (`load_state(state_bytes)`)**: Direct, byte-level restoration of the execution state into a running model context. This completely bypasses the prefill phase, reducing latency from seconds to milliseconds.

---

## 2. Storage Tier Architecture

We partition the caching system across three distinct physical storage tiers:

```
+-------------------------------------------------------+
|                 Tier 1: Hot Cache                     |
|  - Location: Active VRAM (Workstation) / RAM (Pi 5)   |
|  - Speed: Instant (Active execution context)          |
+-------------------------------------------------------+
                           |  (save_state / load_state)
                           v
+-------------------------------------------------------+
|               Tier 2: Standby Cache                   |
|  - Location: System Host RAM (bytes dictionary)       |
|  - Speed: Sub-millisecond memory transfer             |
|  - Budget: 2.5 GB (Edge) / 16 GB (Workstation)        |
+-------------------------------------------------------+
                           |  (Disk read / write)
                           v
+-------------------------------------------------------+
|                Tier 3: Cold Cache                     |
|  - Location: NVMe PCIe SSD (Binary .bin files)        |
|  - Speed: ~30-100ms load time                         |
|  - Budget: 50 GB on local disk                        |
+-------------------------------------------------------+
```

### Capacity & Size Allocation

The serialized state size is determined by the context size $N_{ctx}$ and model dimensionality. For a 3B model (Medium tier) with $N_{ctx} = 2048$, the state size is approximately **60–80 MB**. 

*   **Tier 1 (Hot)**: Allocates exactly $1$ active execution slot per model instance to prevent Out-Of-Memory (OOM) faults on the 8GB Raspberry Pi 5.
*   **Tier 2 (Standby)**: Allocates up to 2.5 GB on the Pi 5 (holding ~30–40 saved states) and 16 GB on the Workstation (holding ~200–260 saved states).
*   **Tier 3 (Cold)**: Allocates up to 50 GB on the local SSD, allowing hundreds of conversation checkpoints to be persisted across system power cycles.

> [!IMPORTANT]
> **Operational Constraint: n_ctx Consistency**
> In `llama.cpp`, the memory layout of the saved state binary payload is tied directly to the `n_ctx` parameter used at model initialization. All sessions and agent interactions for a given model MUST use identical `n_ctx` parameters. Restoring a state file saved with `n_ctx=X` into a model instance initialized with `n_ctx=Y` (where $X \neq Y$) is an illegal boundary state and can result in silent cache memory alignment corruption or undefined logits.

---

## 3. Eviction Policy: Priority-Weighted LRU (PW-LRU)

To prevent cache thrashing during multi-agent context switching, we implement a **Priority-Weighted Least Recently Used (PW-LRU)** eviction heuristic.

### Heuristic Rationale
Traditional LRU evicts states based strictly on the time of last access. However, multi-agent workflows assign unequal importance to active agent states. For instance, the `Critic` agent state requires high retention since it performs iterative verification loops. Conversely, the `Retriever` fetches static contexts that are easily reloaded or recalculated if evicted. Assigning a static priority weight to each role guarantees that critical reasoning states are shielded from premature eviction without needing transition probability training data.

### Priority Weights Allocation
We assign a static priority weight $w_{agent} \in [0.0, 1.0]$ to each agent role:
*   **`Critic`**: Priority $1.0$ (High priority; protect during active review steps).
*   **`Executor`**: Priority $0.8$ (Medium-high priority).
*   **`Planner`**: Priority $0.5$ (Medium priority).
*   **`Retriever`**: Priority $0.3$ (Low priority; easily re-retrieved or re-computed).

### Mathematical Formulation
When Tier 2 (Standby RAM) reaches capacity, the cache manager computes a virtual recency score $V_s$ for each cached state:

$$V_s = t_{elapsed} \cdot (1.0 - w_{agent})$$

where:
*   $t_{elapsed}$ is the time elapsed since the state was last accessed.
*   $w_{agent}$ is the static priority weight of the agent role associated with the cached state.

**The state with the highest $V_s$ (longest idle time combined with the lowest role priority) is evicted to Tier 3 (Cold SSD).**

---

## 4. Context Restoration Workflow

When an agent execution request is dispatched:
1.  **Cache Key Generation**: A unique hash is computed based on the system prompt and conversation prefix:
    $$\text{Key} = \text{SHA256}(\text{model\_name} \ || \ \text{system\_prompt} \ || \ \text{conversation\_history\_prefix})$$
2.  **Tier Check**:
    *   **Hit in Tier 1 (Hot)**: Execute query immediately.
    *   **Hit in Tier 2 (Standby)**: Retrieve byte array from memory, call `load_state()`, and execute.
    *   **Hit in Tier 3 (Cold)**: Read binary file from NVMe SSD, copy to Standby memory, call `load_state()`, and execute.
    *   **Cache Miss**: Trigger full prefill, execute, and write back to Tier 1/2.
3.  **Post-Execution**: Save state using `save_state()`, update access timestamp, and run the PW-LRU eviction sequence if Standby RAM limit is exceeded.
