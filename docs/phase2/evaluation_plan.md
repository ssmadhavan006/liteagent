# LiteAgent — Evaluation Plan

This document outlines the formal evaluation plan for LiteAgent, mapping core research hypotheses directly to empirical experiments, system metrics, and baseline comparisons.

---

## 1. Research Hypotheses

*   **H1 (Routing Effectiveness)**: Complexity-aware routing significantly reduces latency and energy consumption compared to static execution while maintaining comparable task quality.
*   **H2 (KV Cache Performance)**: A persistent three-tier KV cache reduces context restoration latency compared with flat in-memory caching.
*   **H3 (Co-Design Synergy)**: The combined routing + KV cache co-design provides greater overall system efficiency (e.g. latency vs. VRAM/RAM constraints) than either optimization implemented independently.
*   **H4 (Edge Feasibility)**: LiteAgent enables practical multi-agent inference on Raspberry Pi-class hardware with acceptable quality degradation relative to pure workstation execution.

---

## 2. Hypothesis-to-Experiment Mapping

| Hypothesis | Experiment ID | Target Baseline / Comparator | System Metrics | Task Quality Metrics | Experiment Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | Exp 1 (Routing Sweep) | Static Workstation Execution (`llama3.1:8b` only) & Static Edge Execution (`llama3.2:3b` only) | End-to-end Latency, Energy per Token | GSM8K EM, HotpotQA F1, HumanEval Pass@1 | Run full benchmark sweeps under various task complexity thresholds to measure latency/energy scaling against static model choices. |
| **H2** | Exp 2 (Cache Eviction) | Flat In-Memory Prefix Caching (vLLM approximation without RAM/SSD swapping) | TTFT, Context Restoration Latency, Peak VRAM | Cache Output Text Identity (Character-level Match) | Execute long multi-turn agent conversation chains. Benchmark time-to-first-token (TTFT) when loading from RAM/SSD compared to full prompt re-computation. Verify output losslessness by confirming character-by-character token identity between restored generation and fresh prefill generation. |
| **H3** | Exp 3 (Co-Design Stress Test) | Routing-Only (no persistent cache) & Cache-Only (static routing) | Latency, Peak RAM/VRAM, Energy | GSM8K EM, HotpotQA F1, HumanEval Pass@1 | Disable routing and cache components independently during multi-agent workflows to measure co-design synergy under memory constraints. |
| **H4** | Exp 4 (Edge Feasibility) | Native PC Workstation Inference (all tasks routed to PC) | End-to-end Latency, Peak VRAM, Energy | GSM8K EM, HotpotQA F1, HumanEval Pass@1 | Run full multi-agent benchmark on Raspberry Pi 5 with LiteAgent. Measure latency overhead, peak RAM footprint, and task quality relative to PC. |

---

## 3. Experiment Profiles

### Experiment 1: Complexity Routing Threshold Sweep (Targets H1)
*   **Goal**: Demonstrate that routing tasks by predicted complexity yields workstation-level accuracy at near-edge latency and energy costs.
*   **Setup**: Run evaluations on GSM8K, HotpotQA, and HumanEval. Sweep the complexity threshold parameter from $0.0$ (all tasks on Small edge tier) to $1.0$ (all tasks on Large workstation tier).
*   **Key Plot**: Cost/Energy vs. Task Quality curve.

### Experiment 2: Three-Tier Cache Latency Breakdown & Losslessness Verification (Targets H2)
*   **Goal**: Measure the context restoration speeds across cache hits, local RAM hits, SSD disk hits, and cold cache misses (re-computation), and verify the output-level losslessness of context restoration.
*   **Setup**:
    *   *Latency Setup*: Simulate multi-agent conversational switches. Enforce cache eviction from VRAM to RAM, and RAM to SSD. Measure TTFT when loading context from each cache tier.
    *   *Deterministic Cache Restoration Test*: Both the fresh-prefill generation and the restored-context generation must execute under identical, strict deterministic decoding configurations:
        *   `temperature = 0`
        *   `top_k = 1` (greedy sampling)
        *   `top_p` disabled
        *   Fixed RNG seed (`seed = 42`)
        *   Identical stop tokens and sampling configurations
    *   Compare the resulting generated tokens, generated text, and token count. Verify character-by-character text and token identity to confirm that cache restoration is mathematically lossless.
*   **Key Plot**: TTFT latency breakdown per cache tier, and a binary pass/fail verification table for restored output accuracy.

### Experiment 3: Co-Design Ablations (Targets H3)
*   **Goal**: Validate that routing and caching optimizations cooperate to handle resource bottlenecks.
*   **Setup**:
    *   *Configuration A*: Full LiteAgent.
    *   *Configuration B*: Router active, caching disabled (forces context re-computation on every step).
    *   *Configuration C*: Router disabled (static mapping to Large tier), caching active.
*   **Key Plot**: Throughput and VRAM utilization comparison charts.

### Experiment 4: Edge Resource Constraints (Targets H4)
*   **Goal**: Prove the feasibility of deploying agent chains on a Raspberry Pi 5.
*   **Setup**: Execute the multi-agent Planner-Retriever-Executor-Critic chain. Edge models (`llama3.2:1b` and `llama3.2:3b`) run on the Pi CPU, and Large model (`llama3.1:8b`) runs on the PC Workstation. Measure system memory (RAM) stability and energy draw on the Pi.
*   **Key Plot**: Time-series GPU/CPU memory and power allocation graph.
