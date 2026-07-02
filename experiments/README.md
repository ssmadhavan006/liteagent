# LiteAgent — Experiment Logging Directory

This directory stores the results, logs, and metadata of all benchmark and evaluation runs starting in Phase 8.

## Logging Strategy: CSV + Git

To ensure maximum reproducibility, zero external dependencies, and offline execution, LiteAgent uses a local structured **CSV + Git** tracking convention. Every evaluation run appends a row to a local master CSV file and is tied to a specific Git commit hash.

### Logging Location
All runs must be logged to:
*   **`experiments/runs.csv`** (Master database of runs)
*   **`experiments/logs/<run_id>/`** (Sub-directory containing raw JSON outputs, traces, and prompt logs for analysis)

---

## Master CSV Schema (`runs.csv`)

The master log file contains the following columns:

| Column | Description | Data Type | Example |
| :--- | :--- | :--- | :--- |
| `run_id` | Unique UUID or timestamp-based ID for the run | String | `run_20260702_204510` |
| `timestamp` | ISO-8601 timestamp when the run was recorded | String (ISO) | `2026-07-02T20:45:10+05:30` |
| `git_commit` | Exact 7-character Git commit hash of the codebase | String | `30cb310` |
| `benchmark` | Target benchmark name | Enum | `gsm8k` \| `hotpotqa` \| `humaneval` |
| `routing_policy` | Routing algorithm name (e.g., threshold-based, oracle, static) | String | `complexity_threshold_v1` |
| `small_count` | Number of tasks routed to the Small tier (`llama3.2:1b`) | Integer | `450` |
| `medium_count` | Number of tasks routed to the Medium tier (`llama3.1:8b`) | Integer | `320` |
| `large_count` | Number of tasks routed to the Large tier (`qwen2.5:14b`) | Integer | `230` |
| `accuracy` | Benchmark task-quality metric (EM, F1, or Pass@1) | Float (0.0–1.0) | `0.785` |
| `avg_latency_ms` | Mean end-to-end latency per task | Float | `842.5` |
| `avg_ttft_ms` | Mean Time To First Token | Float | `45.2` |
| `peak_ram_mb` | Peak system RAM observed during run | Float | `14200.0` |
| `peak_vram_mb` | Peak GPU memory (VRAM) observed during run | Float | `8450.0` |
| `notes` | Brief notes about the experiment run | String | `Initial baseline run with default cache sizes.` |

---

## Log Directory Convention

For each `run_id`, a sub-folder should be created under `experiments/logs/<run_id>/` containing:
1.  **`config.json`**: A dump of the system configuration, cache sizes, routing parameters, and model versions.
2.  **`traces.jsonl`**: A line-by-line JSONL file containing the trace of inputs, outputs, router decisions, agent-calling paths, and latencies for each individual task.
