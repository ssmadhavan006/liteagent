# LiteAgent — Complexity Router Design

This document details the design specifications of the LiteAgent Complexity Router, outlining feature extraction, complexity taxonomy, agent pruning mappings, and the tunable threshold configuration.

---

## 1. Feature Extraction Pipeline

To ensure the routing step does not introduce a latency bottleneck, the router utilizes a sub-millisecond hybrid feature extraction pipeline rather than a large LLM-based evaluator. The pipeline extracts a feature vector $X_i$ from the incoming prompt:

1.  **Context Length ($x_{len}$)**: Character count and approximate token count of the prompt. High length indicates multi-document contexts (HotpotQA) requiring greater reasoning capacity.
2.  **Structural Code Heuristics ($x_{code}$)**: Binary flags and counts of syntax keywords (`def`, `import`, `class`, `fn`, `let`, `return`, `{}`) indicating code generation (HumanEval).
3.  **Mathematical Keyword Density ($x_{math}$)**: Density of numeric characters, arithmetic operators (`+`, `-`, `*`, `/`, `^`), and math-related terms (`sum`, `average`, `ratio`, `percentage`, `total`) indicating calculation requirements (GSM8K).
4.  **Semantic Density ($x_{semantic}$)**: Frequency of multi-hop logical operators (`because`, `therefore`, `however`, `since`, `either/or`) indicating complex logical reasoning.

### Scoring Model
The features are fed into a lightweight logistic regression classifier (pre-trained on benchmark training subsets) to compute the task complexity score:
$$S_c = \sigma(W \cdot X_i + b) \in [0.0, 1.0]$$
where $\sigma$ is the sigmoid function, $W$ is the weight vector, and $b$ is the bias.

---

## 2. Complexity Taxonomy & Agent Pruning

We define three complexity tiers mapped to model execution locations and agent prunings:

| Complexity Tier | Condition | Target Model (Hardware) | Active Agents | Pruned Agents | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low** | $S_c < \theta_{low}$ | `llama3.2:1b` (Raspberry Pi 5) | Executor OR Retriever | Planner, Critic | Simple queries (e.g., direct lookup or single arithmetic steps) bypass planning and review to minimize edge execution latency. |
| **Medium** | $\theta_{low} \leq S_c < \theta_{high}$ | `llama3.2:3b` (Raspberry Pi 5) | Planner, Retriever, Executor | Critic | Multi-step queries requiring decomposition but not strict review. Pruning the Critic avoids iterative feedback loops on the Pi CPU. |
| **High** | $S_c \geq \theta_{high}$ | `llama3.1:8b` (Workstation via gRPC) | Planner, Retriever, Executor, Critic | None | Complex coding, multi-hop reasoning, or high-uncertainty tasks execute the full agent chain with verification loops. |

---

## 3. Tunable Routing Threshold Mapping

To coordinate with **Experiment 1 (Routing Sweep)** in `evaluation_plan.md`, the router's thresholds are controlled by a single master parameter $\Theta \in [0.0, 1.0]$. The thresholds $\theta_{low}$ and $\theta_{high}$ are dynamically derived from $\Theta$:

$$\theta_{low} = 0.5 \cdot \Theta$$
$$\theta_{high} = 0.5 + 0.5 \cdot \Theta$$

### Sweep Boundaries
*   **Minimum Sweep ($\Theta = 0.0$)**:
    *   $\theta_{low} = 0.0$ and $\theta_{high} = 0.5$.
    *   This minimizes threshold boundaries, routing more tasks to Medium/High tiers to prioritize task accuracy.
*   **Maximum Sweep ($\Theta = 1.0$)**:
    *   $\theta_{low} = 0.5$ and $\theta_{high} = 1.0$.
    *   This maximizes the threshold boundaries, forcing most tasks to run locally on the Pi 5 (Small/Medium tiers) to prioritize energy conservation and low latency.
