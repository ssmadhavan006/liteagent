# Phase 7 Harness Design

This document describes the architectural design, security boundaries, and system metrics collection mechanisms of the evaluation harness.

---

## 1. Harness Execution Flow

The evaluation harness operates as a decoupled orchestration layer:
1.  **Start GPU Energy Polling**: Spawns a background thread on the workstation to query GPU power draw.
2.  **Inference Dispatch**: Invokes the `TaskDispatcher` (or a baseline runner) to execute the task.
3.  **Stop GPU Energy Polling**: Signals the thread to stop immediately after task completion.
4.  **Integrate Energy**: Computes total energy consumed (Joules).
5.  **Evaluate Answer Quality**: Runs the scoring metrics (Exact Match, F1, or sandboxed execution) on the outputs.
6.  **Emit Log**: Writes a unified JSON record satisfying the parity schema.

---

## 2. Workstation GPU Energy Calculation

### Polling Methodology
*   **Command**: `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits`
*   **Polling Interval ($\Delta t$)**: 100 ms (0.1 seconds)
*   **Calculation Formula**:
    $$E = \sum_{i=1}^{N} P_i \cdot \Delta t$$
    where $P_i$ is the sampled GPU power draw (Watts) at step $i$, and $E$ is the total energy consumed in Joules.

### Limitations & Caveats
*   **Scope**: Measures GPU power only (VRAM and GPU cores). Host CPU, system DRAM, SSD, motherboard, and cooling fans are not included.
*   **Spike Resolution**: A 100 ms sampling rate may miss short transient power spikes.
*   **Overhead**: Continuous background calls to `nvidia-smi` introduce minor CPU/process overhead (~1-2%).

---

## 3. HumanEval Subprocess Sandboxing

HumanEval requires executing model-generated code. We employ a layered subprocess isolation strategy:
1.  **Process Isolation**: Executes code inside a separate python subprocess.
2.  **Clean Working Directory**: Runs inside a randomized temporary directory outside the project folder. No project files are accessible.
3.  **Minimal Environment**: Strips all environment variables except essential OS variables (e.g. `SYSTEMROOT`, `PATH`, `PATHEXT`, `COMSPEC`, `PYTHONPATH`).
4.  **Interpreter Isolation**: Launches the subprocess with `python -E -I -S` to ignore environment overrides and prevent loading custom site packages.
5.  **Hard Timeout**: Enforces a 3.0s subprocess timeout.
6.  **OS Resource Limits**: Binds CPU and memory allocation limits via `resource.setrlimit` on Unix (Linux/macOS).
7.  **Secondary Safe Import Hook**: Replaces `builtins.__import__` inside the script to block importation of unauthorized modules (such as `os`, `sys`, `subprocess`, `socket`, `urllib`, `shutil`) while permitting standard helper libraries (`math`, `typing`, `collections`, `re`, `string`, `datetime`, `itertools`, `functools`, `heapq`, `array`, `bisect`).
8.  **Secondary Safe Open Hook**: Monkeypatches `builtins.open` to raise a `PermissionError` on any file read/write attempt.

> [!WARNING]
> **Honesty Disclaimer**
> This implementation provides process isolation and resource limits but is not intended to be a security-hardened sandbox comparable to container or VM isolation.

---

## 4. Raspberry Pi 5 Energy Profiling Blocker

> [!IMPORTANT]
> **Pi Energy Profiling: Blocker Status**
> Since we lack access to the physical Raspberry Pi 5 device, its energy-per-token profiling method remains an open blocker. We will log this blocker explicitly in the harness documentation.
