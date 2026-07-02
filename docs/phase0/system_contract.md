# LiteAgent — System Contract & Agent Definitions

This document defines the formal system contract for **LiteAgent**, outlining input/output schemas, agent roles, and evaluation metrics for the system.

---

## 1. Agent Definitions

LiteAgent utilizes a specialized set of four agents designed to address the target reasoning, retrieval, and code generation benchmarks:

1. **Planner (Decomposition Agent)**
   * **Responsibility**: Decomposes complex user queries into sequential, execution-ready sub-tasks, and schedules dependencies between them.
   * **Expected Input**: Raw task prompt, system state, and benchmark context.
   * **Expected Output**: A structured plan containing a list of sub-tasks, dependencies, and target roles.
   * **Benchmark Utility**: Acts as the initial orchestrator for multi-step reasoning in **GSM8K** and multi-hop questions in **HotpotQA**.

2. **Retriever (Information Fetching Agent)**
   * **Responsibility**: Searches, filters, and formats external knowledge or context documents relevant to a given query.
   * **Expected Input**: Search query / keywords, document corpora (or context arrays).
   * **Expected Output**: A subset of relevance-ranked passages or document snippets.
   * **Benchmark Utility**: Vital for finding evidence in **HotpotQA** and gathering external background facts.

3. **Executor (Code & Logic Generation Agent)**
   * **Responsibility**: Generates code implementations, executes logic, or computes intermediate results.
   * **Expected Input**: A code prompt, context variables, or function definitions.
   * **Expected Output**: Executable Python code or the stdout/stderr return values from safe execution.
   * **Benchmark Utility**: Serves as the primary engine for code synthesis in **HumanEval** and math execution/scratchpad calculation in **GSM8K**.

4. **Critic (Evaluation & Refinement Agent)**
   * **Responsibility**: Evaluates the correctness of executor outputs, checkmates logic errors, and triggers revision loops if verification fails.
   * **Expected Input**: Sub-task plan, generation output, ground truth formats, or execution feedback.
   * **Expected Output**: Approval status (boolean) and a feedback correction prompt if rejected.
   * **Benchmark Utility**: Validates intermediate steps in multi-step math reasoning (**GSM8K**) and syntactical correct execution in **HumanEval**.

---

## 2. Input / Output Schema Contract

### 2.1 Task Input Schema

All tasks entering the LiteAgent system must conform to the following schema structure:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LiteAgentTaskInput",
  "type": "object",
  "properties": {
    "task_id": { "type": "string" },
    "benchmark": { "type": "string", "enum": ["gsm8k", "hotpotqa", "humaneval"] },
    "prompt": { "type": "string" },
    "context": {
      "type": "object",
      "properties": {
        "documents": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Used by HotpotQA to pass candidate context paragraphs."
        },
        "entry_point": {
          "type": "string",
          "description": "Used by HumanEval to define the target function name."
        }
      }
    }
  },
  "required": ["task_id", "benchmark", "prompt"]
}
```

### 2.2 System Output Schema

The system must output a structured JSON response tailored to the benchmark type:

#### GSM8K Output
```json
{
  "task_id": "gsm8k-42",
  "predicted_answer": "14",
  "reasoning_steps": [
    "Step 1: Compute X = 2 * 10 = 20.",
    "Step 2: Subtract Y = 20 - 6 = 14."
  ],
  "agent_trace": [
    {"agent": "Planner", "model": "llama3.2:1b", "action": "decompose"},
    {"agent": "Executor", "model": "llama3.1:8b", "action": "calculate"},
    {"agent": "Critic", "model": "llama3.2:1b", "action": "verify"}
  ]
}
```

#### HotpotQA Output
```json
{
  "task_id": "hotpotqa-109",
  "predicted_answer": "The Battle of Hastings",
  "supporting_facts": [
    ["Document A", 0],
    ["Document B", 3]
  ],
  "agent_trace": [
    {"agent": "Retriever", "model": "llama3.1:8b", "action": "retrieve"},
    {"agent": "Executor", "model": "qwen2.5:14b", "action": "answer"}
  ]
}
```

#### HumanEval Output
```json
{
  "task_id": "humaneval-12",
  "generated_code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
  "agent_trace": [
    {"agent": "Executor", "model": "qwen2.5:14b", "action": "generate"}
  ]
}
```

---

## 3. Metric Definitions

### 3.1 System Performance Metrics

To evaluate routing and KV-cache co-design trade-offs, we define the following system metrics:

1. **End-to-End Latency ($L_{e2e}$)**
   * *Definition*: The wall-clock time elapsed from receiving the task input to writing the final output response.
   * *Unit*: Seconds ($s$) or Milliseconds ($ms$).
   * *Measurement Method*: System timestamp delta (`time.perf_counter()`).

2. **Time to First Token (TTFT)**
   * *Definition*: The duration between the initiation of the first model inference call for a task and the generation of its first output token.
   * *Unit*: Milliseconds ($ms$).
   * *Measurement Method*: Inline hook on the model client callback.

3. **Peak RAM / VRAM Footprint ($M_{peak}$)**
   * *Definition*: The maximum memory allocation observed on system memory (RAM) and GPU memory (VRAM) during execution.
   * *Unit*: Megabytes ($MB$) or Gigabytes ($GB$).
   * *Measurement Method*: Querying `/proc/meminfo` (Linux), `nvidia-smi` / PyTorch memory tracking, or Windows process performance APIs.

4. **Energy per Token ($E_{token}$)**
   * *Definition*: Total electrical energy consumed by the edge/workstation board during inference divided by the total tokens generated.
   * *Unit*: Joules per token ($J/tok$).
   * *Measurement Method*: For Raspberry Pi 5, an inline USB-C power meter or smart plug logger. For Workstation, software-level GPU/CPU TDP querying (e.g. `nvidia-smi` power consumption fields).

### 3.2 Task Quality Metrics

To ensure optimization does not compromise task-solving accuracy:

1. **GSM8K: Exact Match (EM)**
   * *Definition*: The percentage of tasks where the final numerical value extracted from `predicted_answer` exactly matches the ground-truth integer.
   * *Metric Standard*: Accuracy percentage.

2. **HotpotQA: F1 Score & Exact Match**
   * *Exact Match (EM)*: Binarized check on whether the normalized generated string matches the ground-truth span.
   * *F1 Score*: Harmonic mean of token-level precision and recall between the predicted answer and ground truth.

3. **HumanEval: Pass@k**
   * *Definition*: Probability that at least one of the $k$ generated code samples passes the unit tests. For standard single-pass runs, we report **Pass@1**.
   * *Metric Standard*: Pass rate percentage calculated over unit test execution.
