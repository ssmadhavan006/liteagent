# LiteAgent — Novelty Matrix

This document provides a capability matrix to visually isolate the core innovations of LiteAgent compared to state-of-the-art LLM routing, KV-caching, and edge deployment systems.

## Capability Matrix

| Capability | RouteLLM | FrugalGPT | vLLM / PagedAttention | StreamingLLM | LiteAgent (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Complexity Routing** | ✓ | ✓ | ✗ | ✗ | **✓** |
| **Persistent KV Cache** | ✗ | ✗ | Partial | ✓ | **✓** |
| **Multi-Agent Support** | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Raspberry Pi Execution** | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Edge + Workstation Co-design** | ✗ | ✗ | ✗ | ✗ | **✓** |

## Novelty Analysis

As shown in the matrix above, while prior work individually optimizes routing (e.g. RouteLLM, FrugalGPT) or local context memory (e.g. vLLM, StreamingLLM), they do so for single-session cloud APIs or homogeneous high-end GPU cluster instances. 

LiteAgent's primary scientific novelty is the **co-design of complexity-aware routing and a three-tier persistent KV cache hierarchy explicitly tailored for multi-agent LLM workloads running across heterogeneous edge-workstation hardware.**
