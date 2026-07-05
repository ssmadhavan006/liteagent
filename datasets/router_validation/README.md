# LiteAgent Router Sanity Validation Dataset

This directory contains a manually curated set of 80 validation prompts used to verify that the LiteAgent Complexity Router behaves deterministically and correctly maps tasks according to their complexity.

To prevent evaluation contamination during parameter optimization, the dataset is partitioned into two disjoint subsets:
*   **Calibration/Train Set (IDs 1–50)**: Used by our calibration scripts to perform a grid sweep and find optimal scorer weights and bias.
*   **Held-Out Test Set (IDs 51–80)**: Used strictly as a blind, uncontaminated set to evaluate and report final routing accuracy.

> [!NOTE]
> This dataset is a sanity check validation set for verifying router thresholds and feature normalization bounds. It is not presented as a statistically rigorous evaluation (which is handled in Phase 8).

---

## Labeling Criteria

Prompts are labeled as `Low`, `Medium`, or `High` complexity according to the following systems-level and reasoning criteria:

### 1. Low Complexity
*   **Definition**: Tasks that require simple retrieval, single-step lookup, or direct arithmetic computation.
*   **Characteristics**:
    *   Prompt length under 150 characters.
    *   No complex coding syntax (no structural definition requests).
    *   Single-step mathematical queries (e.g. basic addition/subtraction).
*   **Expected Model Mapping**: `Small` tier (`llama3.2:1b`).
*   **Agent Pruning**: Planner and Critic are pruned.

### 2. Medium Complexity
*   **Definition**: Tasks that require structured decomposition (Planner) or multi-step logic but do not demand formal logical review (Critic).
*   **Characteristics**:
    *   Prompt length between 150 and 800 characters.
    *   Basic multi-step math word problems (e.g. percentage calculations, simple algebra).
    *   Standard algorithmic code generation (e.g. sorting a list, reversing strings).
*   **Expected Model Mapping**: `Medium` tier (`llama3.2:3b`).
*   **Agent Pruning**: Critic is pruned.

### 3. High Complexity
*   **Definition**: Tasks demanding full multi-hop reasoning, complex code generation with edge cases, or strict validation requirements.
*   **Characteristics**:
    *   Prompt length exceeding 800 characters or containing multi-document contexts.
    *   Advanced algorithm implementations (e.g. graph search, tree balancing, custom class structures).
    *   Multi-stage mathematical reasoning with complex word constraints.
*   **Expected Model Mapping**: `Large` tier (`llama3.1:8b`).
*   **Agent Pruning**: No pruning (full Planner $\rightarrow$ Retriever $\rightarrow$ Executor $\rightarrow$ Critic chain).
