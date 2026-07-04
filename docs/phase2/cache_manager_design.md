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

---

## 3. Eviction Policy: Task-Affinity Aware LRU (TA-LRU)

To prevent cache thrashing during multi-agent context switching, we implement a **Task-Affinity Aware Least Recently Used (TA-LRU)** eviction heuristic.

### Heuristic Rationale
Traditional LRU evicts states based strictly on the time of last access. However, multi-agent workflows (Planner $\rightarrow$ Retriever $\rightarrow$ Executor $\rightarrow$ Critic) have highly predictable sequence transitions. For example, if the `Executor` agent has just completed, there is a very high probability that the `Critic` agent will be invoked next.

### Mathematical Formulation
When Tier 2 (Standby RAM) reaches capacity, the cache manager computes a virtual recency score $V_s$ for each cached state:

$$V_s = t_{elapsed} \cdot (1.0 - A[Agent_{current}, Agent_{cached}])$$

where:
*   $t_{elapsed}$ is the time elapsed since the state was last accessed.
*   $Agent_{current}$ is the agent role that is currently running.
*   $Agent_{cached}$ is the agent role associated with the cached state.
*   $A[A_i, A_j] \in [0.0, 1.0]$ is the transition affinity matrix defining the probability of transitioning from agent $A_i$ to agent $A_j$.

**The state with the highest $V_s$ (highest elapsed time modified by the lowest transition affinity) is evicted to Tier 3 (Cold SSD).**

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
3.  **Post-Execution**: Save state using `save_state()`, update access timestamp, and run the TA-LRU eviction sequence if Standby RAM limit is exceeded.
