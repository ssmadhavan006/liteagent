# LiteAgent — gRPC Schema & Network Bandwidth Analysis

This document outlines the gRPC message schemas used for communication between the Raspberry Pi 5 (Edge) and the PC Workstation. It includes a network bandwidth feasibility analysis justifying our design decisions.

---

## 1. Network Bandwidth Analysis

### The Problem of KV-Cache Network Transfer
A serialized state file for `llama3.1:8b` (Large Workstation model) at $N_{ctx} = 2048$ contains context parameters and KV tensors totaling approximately **180–220 MB**. 

Let's calculate the transmission time for a **200 MB** cache state over typical edge-workstation network links:
1.  **Fast Ethernet / Standard Wi-Fi (100 Mbps)**:
    $$\text{Latency}_{transfer} = \frac{200\text{ MB} \times 8\text{ bits/byte}}{100\text{ Mbps}} = 16.0\text{ seconds}$$
2.  **Gigabit Ethernet / Wi-Fi 6 (1 Gbps)**:
    $$\text{Latency}_{transfer} = \frac{200\text{ MB} \times 8\text{ bits/byte}}{1000\text{ Mbps}} = 1.6\text{ seconds}$$

On the RTX 5070 GPU, processing a prefill phase of 2000 tokens takes **under 0.3 seconds**. 
If the system transferred KV cache tensors over the network:
*   At 100 Mbps, network transfer is **53x slower** than re-computing the prefill.
*   At 1 Gbps, network transfer is **5x slower** than re-computing the prefill.

### Co-Design Decision
**KV cache state data is NEVER transmitted across the network.** 

*   **Local Partitioning**: Caches are bound to the device where the corresponding model execution occurs.
    *   Raspberry Pi 5 caches are persisted strictly on the Pi's local RAM/SSD (for `llama3.2:1b` and `llama3.2:3b` states).
    *   PC Workstation caches are persisted strictly on the Workstation's local RAM/SSD (for `llama3.1:8b` states).
*   **Coordination Protocol**: The Pi 5 dispatches tasks to the workstation by sending only the text prompt and a `session_id`. The workstation generates a cache key from the `session_id`, restores the context locally using its own three-tier cache manager, performs inference, saves the updated state locally, and returns only the text results and metadata.

---

## 2. Proto3 Schema Definition

```protobuf
syntax = "proto3";

package liteagent;

service WorkstationCoordinator {
  // Dispatches a high-complexity sub-task from Edge to Workstation
  rpc DispatchTask(TaskRequest) returns (TaskResponse);
}

message TaskRequest {
  string task_id = 1;
  string session_id = 2;       // Maps to local cache key on Workstation
  string agent_role = 3;       // Planner, Retriever, Executor, Critic
  string prompt = 4;           // Prompt content
  string system_prompt = 5;    // System instructions
  float temperature = 6;
  int32 max_tokens = 7;
}

message TaskResponse {
  string task_id = 1;
  string response_text = 2;
  int32 tokens_generated = 3;
  int32 prefill_tokens = 4;
  
  // Execution Latency Metrics
  double prefill_latency_ms = 5;
  double generation_latency_ms = 6;
  
  // Cache Tracking Metrics
  string cache_hit_tier = 7;   // "HOT", "STANDBY", "COLD", "MISS"
}
```
