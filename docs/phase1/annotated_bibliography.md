# LiteAgent — Annotated Bibliography

This document lists the 22 most relevant papers forming the academic foundation for LiteAgent, divided into four key clusters: Complexity-Aware Routing, KV-Cache Management, Edge Serving, and Multi-Agent Orchestration.

---

## Cluster 1: Complexity-Aware Routing & LLM Cascades

### RouteLLM: Learning to Route LLMs with Preference Data
- **Authors:** Wenshuo Ong, Yannis Katsis, Alon Albalak, Ryan Chi, Matthew Sobocinski, Doug Burger, and Chi Wang
- **Venue/Year:** COLM 2024
- **Link:** [https://arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665)
- **Cluster:** routing
- **Summary (paraphrased, 2-4 sentences):** This work presents RouteLLM, a framework designed to train lightweight router models that dynamically allocate incoming queries to either a strong or weak LLM. The routers are trained using human preference comparison datasets, learning to predict which model is sufficient for a specific task. By making classification decisions at inference time, it reduces operational costs by over 50% while maintaining performance close to a monolithic strong model.
- **Relevance to LiteAgent:** Serves as the primary baseline for the routing component. LiteAgent adapts the router to dynamically partition agent tasks between the edge device and workstation.

### FrugalGPT: How to Use Large Language Models More Efficiently and Cheaply
- **Authors:** Lingjiao Chen, Matei Zaharia, and James Zou
- **Venue/Year:** arXiv 2023
- **Link:** [https://arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176)
- **Cluster:** routing
- **Summary (paraphrased, 2-4 sentences):** FrugalGPT introduces cascading pipelines where queries are sequentially evaluated by increasingly capable and costly LLMs. Each step computes a prediction confidence, and if the response meets a quality threshold, the execution terminates early to save cost. The paper demonstrates that dynamic cascades can save up to 98% in API costs while matching or exceeding the accuracy of single-model calls.
- **Relevance to LiteAgent:** Justifies the multi-tiered model cascade logic. LiteAgent extends this concept into a collaborative edge-workstation cascade environment.

### CAS-Spec: Cascade Adaptive Self-Speculative Decoding
- **Authors:** Zilu Guo, Jiacheng Feng, and colleagues
- **Venue/Year:** arXiv 2024
- **Link:** [https://arxiv.org/abs/2406.01235](https://arxiv.org/abs/2406.01235)
- **Cluster:** routing
- **Summary (paraphrased, 2-4 sentences):** CAS-Spec introduces a dynamic self-speculative decoding framework where multi-level draft models are created from a single target model. It applies a Dynamic Tree Cascade (DyTC) algorithm to adaptively select drafting lengths and exit thresholds based on live token acceptance rates. This approach avoids the cost of maintaining separate draft model weights while boosting generation speed dynamically.
- **Relevance to LiteAgent:** Informs how adaptive routing can adjust model execution steps based on execution-time feedback.

### Speculative Decoding with a Cascading Hierarchy
- **Authors:** Google Research
- **Venue/Year:** ICLR 2024
- **Link:** [https://openreview.net/forum?id=Yw8rfxIbtO](https://openreview.net/forum?id=Yw8rfxIbtO)
- **Cluster:** routing
- **Summary (paraphrased, 2-4 sentences):** This paper introduces a speculative decoding framework utilizing a cascading hierarchy of draft models. Instead of a single static draft model, the system uses multiple draft models of varying sizes to propose tokens sequentially, verifying them against the largest target model. This multi-level prediction accelerates inference on complex tasks while minimizing verification overhead.
- **Relevance to LiteAgent:** Illustrates the potential of utilizing hierarchical model pools for progressive complexity verification.

### Large Language Model Cascades
- **Authors:** David Dohan, Winnie Zhou, and colleagues
- **Venue/Year:** arXiv 2022
- **Link:** [https://arxiv.org/abs/2205.10063](https://arxiv.org/abs/2205.10063)
- **Cluster:** routing
- **Summary (paraphrased, 2-4 sentences):** This work outlines a formal probabilistic framework for representing cascades of LLMs, drawing analogies to programming languages. It shows how chaining, routing, and checking LLM outputs in sequential chains improves logical reasoning consistency. The author maps diverse prompting methods under a unified probabilistic programming model.
- **Relevance to LiteAgent:** Provides the formal foundation for task-routing pipelines and verification cascades across agents.

---

## Cluster 2: KV-Cache Management & Memory Systems

### Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors:** Woosung Kwon, Zhuohan Li, Siyuan Zhuang, Shiyi Cao, Joseph E. Gonzalez, Davide Annovazzi, and Ion Stoica
- **Venue/Year:** SOSP 2023
- **Link:** [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **Cluster:** kv-cache
- **Summary (paraphrased, 2-4 sentences):** This seminal paper introduces PagedAttention, which partitions the KV cache of LLMs into non-contiguous physical pages similar to OS virtual memory. By managing logical tokens through a block table mapping, it eliminates internal and external memory fragmentation. This architecture drastically increases serving capacity and throughput, especially for parallel generation requests.
- **Relevance to LiteAgent:** Serves as a baseline memory management paradigm. LiteAgent extends block-level caching to support persistent, cross-invocation agent states.

### LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference
- **Authors:** LMCache Team
- **Venue/Year:** arXiv 2025
- **Link:** [https://arxiv.org/abs/2510.09665](https://arxiv.org/abs/2510.09665)
- **Cluster:** kv-cache
- **Summary (paraphrased, 2-4 sentences):** LMCache decouples the KV cache from standard GPU limits by creating a unified cache layer across a tiered storage hierarchy (CPU RAM, local disk, and remote storage). It supports engine-agnostic cache sharing across multiple model instances, avoiding redundant prefilling computations. The layer integrates seamlessly into popular engines like vLLM.
- **Relevance to LiteAgent:** Key comparative baseline for tiered KV storage. LiteAgent builds on this logic by optimizing the caching system specifically for edge-workstation split deployments.

### Efficient Streaming Language Models with Attention Sinks
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Frédo Durand
- **Venue/Year:** ICLR 2024
- **Link:** [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- **Cluster:** kv-cache
- **Summary (paraphrased, 2-4 sentences):** This paper identifies "attention sinks," where the first few tokens of a sequence collect a disproportionate amount of attention mass in LLMs. Based on this, the authors introduce StreamingLLM, which retains initial tokens alongside a sliding window of recent tokens in the KV cache. This simple eviction policy enables stable infinite-sequence generation without retraining.
- **Relevance to LiteAgent:** Informs the cache-eviction component of LiteAgent. The sliding window and attention-sink concepts are integrated into the edge device's memory-constrained cache manager.

### H2O: Heavy-Hitter Oracle for Efficient Generative LLaMA Inference
- **Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, and Zhangyang Wang
- **Venue/Year:** NeurIPS 2023
- **Link:** [https://arxiv.org/abs/2306.14048](https://arxiv.org/abs/2306.14048)
- **Cluster:** kv-cache
- **Summary (paraphrased, 2-4 sentences):** H2O proposes a dynamic KV-cache eviction policy based on the observation that attention scores are concentrated on a sparse set of "heavy hitter" tokens. By dynamically tracking and retaining these high-influence tokens alongside recent context, the system reduces the KV-cache size by up to 5x with minimal accuracy loss.
- **Relevance to LiteAgent:** Provides the foundation for the local cache manager's eviction heuristics, particularly when pruning context on the memory-constrained Raspberry Pi.

### CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving
- **Authors:** Linsong Guo and colleagues
- **Venue/Year:** SIGCOMM 2024
- **Link:** [https://arxiv.org/abs/2310.07240](https://arxiv.org/abs/2310.07240)
- **Cluster:** kv-cache
- **Summary (paraphrased, 2-4 sentences):** CacheGen presents a specialized codec for compressing and streaming the KV cache to accelerate context loading. It leverages token-wise distribution characteristics to compress KV tensors into compact bitstreams, adaptively adjusting compression ratios to match target network bandwidth. This significantly reduces network loading overhead in distributed LLM architectures.
- **Relevance to LiteAgent:** Informs the edge-workstation gRPC communication layer on how to stream KV caches compactly over local network links.

### InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
- **Authors:** USENIX OSDI 2024
- **Link:** [https://www.usenix.org/conference/osdi24/presentation/infinigen](https://www.usenix.org/conference/osdi24/presentation/infinigen)
- **Cluster:** kv-cache
- **Summary (paraphrased, 2-4 sentences):** InfiniGen proposes an offloading framework to handle long-context LLM inference by dynamically moving KV cache data between GPU memory and host memory. It uses an SVD-based importance predictor to prefetch only the most essential KV cache blocks before generation steps, minimizing PCIe bandwidth bottlenecks.
- **Relevance to LiteAgent:** Inspires our three-tier cache manager's offloading logic (VRAM to local RAM to disk/cold storage).

### FlexGen: High-Throughput Generation of Large Language Models on a Single GPU
- **Authors:** Ying Sheng, Lianmin Zheng, Binhang Yuan, and colleagues
- **Venue/Year:** ICML 2023
- **Link:** [https://arxiv.org/abs/2303.06865](https://arxiv.org/abs/2303.06865)
- **Cluster:** kv-cache
- **Summary (paraphrased, 2-4 sentences):** FlexGen addresses LLM serving under resource constraints by offloading weights, activations, and KV cache parameters across GPU, CPU, and disk. It formulates a linear programming problem to optimize throughput given memory hardware boundaries, yielding high-throughput execution for offline batch processing.
- **Relevance to LiteAgent:** Highlights baseline offloading constraints and informs design trade-offs when pushing caches to local disks.

---

## Cluster 3: Edge LLM Deployment

### llama.cpp: Inference of LLaMA model in pure C/C++
- **Authors:** Georgi Gerganov and the llama.cpp Contributors
- **Venue/Year:** GitHub Project
- **Link:** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Cluster:** edge-serving
- **Summary (paraphrased, 2-4 sentences):** llama.cpp is an open-source inference engine written in pure C/C++ that enables high-performance execution of LLMs on diverse CPU and GPU consumer hardware. It focuses on optimizing execution on ARM CPUs (like the Apple Silicon M-series and Raspberry Pi) using SIMD vector instructions and custom quantizations (GGUF format). It has democratized local LLM deployments by bypassing heavy Python dependencies.
- **Relevance to LiteAgent:** Serves as the underlying execution engine on the Raspberry Pi 5. LiteAgent interfaces with it via Ollama/llama.cpp APIs to execute local small/medium models.

### MLC-LLM: Machine Learning Compilation for Large Language Models
- **Authors:** MLC Team
- **Venue/Year:** GitHub / MLC Project
- **Link:** [https://github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)
- **Cluster:** edge-serving
- **Summary (paraphrased, 2-4 sentences):** MLC-LLM is a compiler-based deployment engine designed to run LLMs natively on diverse hardware backends, including mobile, web, and edge backends. It utilizes TVM Unity compilation to generate optimized shaders and execution kernels, allowing unified acceleration across Apple GPUs, Vulkan, WebGPU, and ARM CPUs.
- **Relevance to LiteAgent:** Provides comparative context on edge execution engines.

### LLM-Viewer: A Profiling Framework for LLMs on Edge Devices
- **Authors:** arXiv 2024
- **Link:** [https://arxiv.org/abs/2402.10803](https://arxiv.org/abs/2402.10803)
- **Cluster:** edge-serving
- **Summary (paraphrased, 2-4 sentences):** LLM-Viewer presents a profiling and prediction framework to analyze LLM deployment constraints on edge devices. It models execution latency, memory bottlenecks, and energy consumption across different model sizes and quantizations. It provides developers with analytical tools to optimize hardware selection and compile targets.
- **Relevance to LiteAgent:** Validates our focus on edge resource bottlenecks and reinforces the need for dynamic task routing.

### EdgeMoE: Fast Mixture-of-Experts Inference on Heterogeneous Edge Devices
- **Authors:** arXiv 2023
- **Link:** [https://arxiv.org/abs/2308.14352](https://arxiv.org/abs/2308.14352)
- **Cluster:** edge-serving
- **Summary (paraphrased, 2-4 sentences):** EdgeMoE optimizes Mixture-of-Experts (MoE) execution on memory-constrained edge hardware. It dynamically partitions active expert weights across memory tiers and overlaps parameter loading with active layer execution to minimize bottlenecks.
- **Relevance to LiteAgent:** Highlights parallel offloading techniques that are relevant to cache loading.

### MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use
- **Authors:** Zichao Li and colleagues
- **Venue/Year:** arXiv 2024
- **Link:** [https://arxiv.org/abs/2402.14905](https://arxiv.org/abs/2402.14905)
- **Cluster:** edge-serving
- **Summary (paraphrased, 2-4 sentences):** MobileLLM focuses on structural changes to sub-billion parameter models (e.g. block sharing, deeper structures) to optimize performance on mobile and edge devices. It shows that model depth is more critical than width for on-device reasoning accuracy.
- **Relevance to LiteAgent:** Reinforces the feasibility of the Small model tier (`llama3.2:1b`) for local edge execution.

---

## Cluster 4: Multi-Agent LLM Systems

### AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
- **Authors:** Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang
- **Venue/Year:** COLM 2024
- **Link:** [https://arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155)
- **Cluster:** multi-agent
- **Summary (paraphrased, 2-4 sentences):** AutoGen is a multi-agent framework that orchestrates complex task completion through conversational loops among customizable, cooperative agents. The framework supports human-in-the-loop, tool usage, and code execution environments. It models collaborative tasks as multi-agent conversations, simplifying complex coordination patterns.
- **Relevance to LiteAgent:** Defines the multi-agent system context. LiteAgent targets optimizations specifically for these conversational coordination patterns.

### MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework
- **Authors:** Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber
- **Venue/Year:** ICLR 2024
- **Link:** [https://arxiv.org/abs/2308.00352](https://arxiv.org/abs/2308.00352)
- **Cluster:** multi-agent
- **Summary (paraphrased, 2-4 sentences):** MetaGPT implements standardized operating procedures (SOPs) within a multi-agent framework to model software engineering workflows. By assigning distinct professional roles (PM, Coder, Architect) and enforcing structured output formats (documents, schemas), it ensures reliable, asynchronous agent coordination.
- **Relevance to LiteAgent:** Informs how structured coordination patterns can be parsed to optimize caching and routing.

### CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society
- **Authors:** Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem
- **Venue/Year:** NeurIPS 2023
- **Link:** [https://arxiv.org/abs/2303.17760](https://arxiv.org/abs/2303.17760)
- **Cluster:** multi-agent
- **Summary (paraphrased, 2-4 sentences):** CAMEL introduces a cooperative agent role-playing framework that allows two agents (e.g. user and assistant) to collaborate dynamically to solve a task. The work outlines prompt engineering mechanisms to prevent agent loops and maintain progress toward target goals.
- **Relevance to LiteAgent:** Serves as a reference multi-agent logic framework.

### Mixture-of-Agents Enhances Large Language Model Capabilities
- **Authors:** Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou
- **Venue/Year:** ICLR 2025 (Spotlight)
- **Link:** [https://arxiv.org/abs/2406.04692](https://arxiv.org/abs/2406.04692)
- **Cluster:** multi-agent
- **Summary (paraphrased, 2-4 sentences):** Mixture-of-Agents (MoA) introduces a layered model structure where multiple LLM agents generate draft responses in parallel, which are then passed as context to subsequent layers of agents for consolidation. The collective intelligence of smaller, diverse models outperforms monolithic models like GPT-4 on key benchmarks.
- **Relevance to LiteAgent:** Important baseline candidate for multi-agent execution. LiteAgent optimizes the caching overhead generated by these multi-layer, multi-model agent structures.

### KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows
- **Authors:** arXiv 2024
- **Link:** [https://arxiv.org/abs/2407.19502](https://arxiv.org/abs/2407.19502)
- **Cluster:** multi-agent
- **Summary (paraphrased, 2-4 sentences):** KVFlow optimizes KV-cache utilization in multi-agent workflows by abstracting agent schedules as an "Agent Step Graph." It schedules and prefetches cache blocks based on predicted execution paths, significantly reducing prefill latency in complex collaborative agent chains.
- **Relevance to LiteAgent:** Extremely relevant context. LiteAgent co-designs this workflow prefetching approach with edge-workstation complexity routing.
