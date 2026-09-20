# Phase 7 Metric Definitions

This document defines the mathematical formulas, normalization rules, and output schemas used by the evaluation harness.

---

## 1. Quality Metrics

### GSM8K (Exact Match)
*   **Formula**:
    $$\text{GSM8K Score} = \begin{cases} 
      1.0 & \text{if } \text{normalize}(\text{predicted\_num}) = \text{normalize}(\text{ground\_truth\_num}) \\ 
      0.0 & \text{otherwise} 
    \end{cases}$$
*   **Normalization**: Strips commas, whitespace, and trailing periods.
*   **Ground Truth Extraction**: Matches OpenAI's standard format (splits at `####` and extracts the trailing value).
*   **Prediction Extraction**: Strips punctuation and searches for the last numeric token.

### HotpotQA (Exact Match & Token-level F1)
*   **Exact Match (EM)**:
    $$\text{EM} = \begin{cases} 
      1.0 & \text{if } \text{normalize}(\text{pred\_text}) = \text{normalize}(\text{ref\_text}) \\ 
      0.0 & \text{otherwise} 
    \end{cases}$$
*   **F1 Score**:
    $$\text{Precision} = \frac{|\text{pred\_tokens} \cap \text{ref\_tokens}|}{|\text{pred\_tokens}|}$$
    $$\text{Recall} = \frac{|\text{pred\_tokens} \cap \text{ref\_tokens}|}{|\text{ref\_tokens}|}$$
    $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
*   **Normalization**: Normalizes text by lowercasing, stripping punctuation, removing articles (`a`, `an`, `the`), and trimming whitespace.

### HumanEval (Single-Sample Pass@1)
*   **Formula**:
    $$\text{HumanEval Score} = \begin{cases} 
      1.0 & \text{if } \text{sandbox\_execution}(\text{solution} + \text{assert\_tests}) \text{ succeeds} \\ 
      0.0 & \text{otherwise (runtime error, timeout, or assertion failure)} 
    \end{cases}$$
*   **Single-Sample pass@1**: Since only one completion is generated per coding task during sweeps, we compute a binary pass/fail score (1.0 or 0.0) per task.

---

## 2. Record Schema Specification

Every completed task evaluation record is appended to `experiments/evaluation_results.jsonl` matching this versioned schema:

```json
{
  "schema_version": 1,
  "task_id": "humaneval_012",
  "dataset_index": 12,
  "dataset": "humaneval",
  "sample_seed": 42,
  "baseline": "liteagent",
  "metrics": {
    "quality_score": 1.0,
    "quality_score_extra": {
      "em": 1.0,
      "f1": 1.0
    },
    "latency_ms": 1420.5,
    "energy_joules": 28.5,
    "energy_samples": 14,
    "prefill_tokens": 120,
    "tokens_generated": 15,
    "cache_hit_tier": "STANDBY",
    "routed_tier": "Medium",
    "executed_tier": "Medium",
    "fallback_occurred": false
  }
}
```

If a task fails during execution or throws a sandbox violation/timeout, a record is added to `experiments/failed_evals.jsonl`:

```json
{
  "schema_version": 1,
  "task_id": "humaneval_012",
  "dataset_index": 12,
  "dataset": "humaneval",
  "failure_category": "FAILURE_TIMEOUT",
  "timeout_occurred": true,
  "generated_code": "def infinite_loop():\n    while True:\n        pass",
  "stderr": "subprocess timeout expired"
}
```
