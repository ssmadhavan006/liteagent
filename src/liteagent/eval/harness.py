import os
import time
import json
import threading
import subprocess
from typing import Any, Dict

# Metrics
from liteagent.config import WORKSTATION_N_GPU_LAYERS
from liteagent.eval.energy import build_energy_monitor
from liteagent.eval.fewshot import build_prefix, shot_count
from liteagent.eval.metrics.gsm8k_metric import score_gsm8k
from liteagent.eval.metrics.hotpotqa_metric import score_hotpotqa
from liteagent.eval.metrics.humaneval_metric import score_humaneval

# Output-format contract, issued identically to every system under test.
#
# Without it, answer extraction falls back to "last number in the text", which
# picks up an intermediate value whenever generation is truncated mid-reasoning
# and scores a correct solve as a failure. It must be identical across
# configurations: if only the agent chain were told the expected format, it
# would score higher for a reason unrelated to routing or caching.
BENCHMARK_SYSTEM_PROMPTS = {
    "gsm8k": (
        "Solve the problem step by step. "
        "Finish with the final numeric answer on its own last line, "
        "written exactly as: #### <answer>"
    ),
    "hotpotqa": (
        "Answer using only the supplied context. "
        "Reply with the shortest exact answer span and nothing else."
    ),
    "humaneval": (
        "Complete the Python function. "
        "Reply with code only - no prose, no explanation, no markdown fences."
    ),
}

# Chain-of-thought on GSM8K routinely needs more than 100 tokens; truncating
# mid-derivation is scored as a wrong answer.
BENCHMARK_MAX_TOKENS = {
    "gsm8k": 400,
    "hotpotqa": 128,
    "humaneval": 512,
}


class EvaluationHarness:
    """
    Orchestrates the evaluation of a dataset split against a baseline runner.
    Appends versioned results and failure trace records.
    """
    def __init__(
        self,
        baseline_name: str,
        results_path: str = "experiments/evaluation_results.jsonl",
        failed_path: str = "experiments/failed_evals.jsonl",
        sample_seed: int = 42,
        few_shot: bool = True,
        energy_source: str = "auto",
        meter_csv: str = None
    ):
        self.baseline_name = baseline_name
        self.results_path = results_path
        self.failed_path = failed_path
        self.sample_seed = sample_seed
        self.few_shot = few_shot
        # Reports None when nothing can measure the work honestly; see energy.py.
        self.power_tracker = build_energy_monitor(
            source=energy_source,
            gpu_layers=WORKSTATION_N_GPU_LAYERS,
            meter_csv=meter_csv,
        )
        self._file_lock = threading.Lock()

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.failed_path), exist_ok=True)

        self.completed_tasks = self.get_completed_tasks()

    def _append_jsonl(self, filepath: str, record: dict):
        with self._file_lock:
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                import sys
                print(f"Warning: Failed to write to {filepath}: {e}", file=sys.stderr)

    def get_completed_tasks(self) -> set[str]:
        """
        Reads results_path and returns the set of completed task IDs for the current baseline.
        """
        completed = set()
        if os.path.exists(self.results_path):
            with open(self.results_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if rec.get("baseline") == self.baseline_name:
                                completed.add(rec.get("task_id"))
                        except Exception:
                            pass
        return completed

    def format_hotpotqa_prompt(self, item: Dict[str, Any]) -> str:
        paragraphs = []
        for title, sentences in item.get("context", []):
            joined_sentences = "".join(sentences)
            paragraphs.append(f"[{title}]: {joined_sentences}")
        context_str = "\n".join(paragraphs)
        question = item.get("question", "")
        return f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer in a short span:"

    def evaluate_task(
        self,
        dataset: str,
        task_id: str,
        dataset_index: int,
        runner: Any,
        task_item: Dict[str, Any],
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Executes a single benchmark task while polling GPU energy.
        """
        if task_id in self.completed_tasks:
            print(f"    Task {task_id} already completed for baseline {self.baseline_name}. Skipping...")
            return {"success": True, "skipped": True}

        if max_tokens is None:
            max_tokens = BENCHMARK_MAX_TOKENS.get(dataset, 256)

        # Determine prompt text and the structured context the agent chain consumes
        task_context: Dict[str, Any] = {}
        if dataset == "gsm8k":
            prompt = task_item.get("question", "")
        elif dataset == "hotpotqa":
            prompt = self.format_hotpotqa_prompt(task_item)
            # The chain retrieves from `documents` instead of re-reading the
            # full context already inlined into `prompt` for single-shot runners.
            task_context["documents"] = [
                (title, "".join(sentences)) for title, sentences in task_item.get("context", [])
            ]
            task_context["question"] = task_item.get("question", "")
        elif dataset == "humaneval":
            prompt = task_item.get("prompt", "")
            task_context["entry_point"] = task_item.get("entry_point", "")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Start energy tracker
        self.power_tracker.start()
        start_time = time.time()

        # The few-shot prefix is applied here, once, so every configuration
        # receives a byte-identical prompt for the same task.
        prefix = build_prefix(dataset, self.few_shot)
        if prefix:
            prompt = prefix + prompt
            # The agent chain reads `question` instead of the pre-inlined
            # prompt on document benchmarks, so it needs the same prefix or the
            # two paths would not be receiving the same protocol.
            if task_context.get("question"):
                task_context["question"] = prefix + task_context["question"]

        # Run inference
        task_dict = {
            "prompt": prompt,
            "id": task_id,
            "benchmark": dataset,
            "context": task_context,
        }
        session_id = f"session_{task_id}_{self.baseline_name}"

        try:
            res = runner.execute_task(
                task=task_dict,
                session_id=session_id,
                system_prompt=BENCHMARK_SYSTEM_PROMPTS.get(dataset, ""),
                max_tokens=max_tokens
            )
            latency_ms = (time.time() - start_time) * 1000.0
            energy_joules, energy_samples = self.power_tracker.stop()

            response = res.get("response_text", "")
            prefill_tokens = res.get("prefill_tokens", 0)
            tokens_gen = res.get("tokens_generated", 0)
            cache_hit_tier = res.get("cache_hit_tier", "NONE")
            routed_tier = res.get("routed_tier", "NONE")
            executed_tier = res.get("executed_tier", "NONE")
            fallback_occurred = res.get("fallback_occurred", False)
            agent_trace = res.get("agent_trace", [])
            active_agents = res.get("active_agents", [])
            pruned_agents = res.get("pruned_agents", [])
            model_calls = res.get("model_calls", 1)
            revisions = res.get("revisions", 0)

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            energy_joules, energy_samples = self.power_tracker.stop()

            err_str = str(e)
            if "EXPECTED_LIMIT_REACHED" in err_str:
                self._log_failure(
                    task_id=task_id,
                    dataset_index=dataset_index,
                    dataset=dataset,
                    category="EXPECTED_LIMIT_REACHED",
                    code="",
                    stderr=err_str,
                    timeout=False
                )
                return {"success": False, "reason": "EXPECTED_LIMIT_REACHED"}
            else:
                self._log_failure(
                    task_id=task_id,
                    dataset_index=dataset_index,
                    dataset=dataset,
                    category="FAILURE_ALLOCATION",
                    code="",
                    stderr=err_str,
                    timeout=False
                )
                raise e

        # Calculate dataset-specific quality metrics
        quality_score = 0.0
        quality_extra = {}

        if dataset == "gsm8k":
            ref = task_item.get("answer", "")
            quality_score = score_gsm8k(response, ref)
            quality_extra = {"em": quality_score}

        elif dataset == "hotpotqa":
            ref = task_item.get("answer", "")
            scores = score_hotpotqa(response, ref)
            quality_score = scores["f1"]
            quality_extra = scores

        elif dataset == "humaneval":
            test_code = task_item.get("test", "")
            # The prompt carries the signature, imports and docstring; without
            # it a body-only completion is not a runnable program.
            eval_res = score_humaneval(
                response,
                test_code,
                timeout=3.0,
                prompt=task_item.get("prompt", ""),
                entry_point=task_item.get("entry_point", ""),
            )
            quality_score = eval_res["pass_status"]
            quality_extra = {
                "success": eval_res["success"],
                "failure_category": eval_res["failure_category"]
            }

            # If sandbox execution failed, preserve artifacts to failed log
            if not eval_res["success"]:
                self._log_failure(
                    task_id=task_id,
                    dataset_index=dataset_index,
                    dataset=dataset,
                    category=eval_res["failure_category"] or "FAILURE_RUNTIME_ERROR",
                    code=response,
                    stderr=eval_res["stderr"],
                    timeout=eval_res["timeout_occurred"]
                )

        # Build final versioned schema evaluation record
        record = {
          "schema_version": 2,
          "task_id": task_id,
          "dataset_index": dataset_index,
          "dataset": dataset,
          "sample_seed": self.sample_seed,
          "baseline": self.baseline_name,
          "protocol": {
            "few_shot": self.few_shot,
            "shots": shot_count(dataset) if self.few_shot else 0,
            "max_tokens": max_tokens,
          },
          "metrics": {
            "quality_score": quality_score,
            "quality_score_extra": quality_extra,
            "latency_ms": latency_ms,
            # None (not 0.0) when no backend could measure this run, so an
            # unmeasured task cannot be averaged in as a real zero.
            "energy_joules": energy_joules,
            "energy_samples": energy_samples,
            "energy_source": self.power_tracker.name if energy_joules is not None else None,
            "prefill_tokens": prefill_tokens,
            "tokens_generated": tokens_gen,
            "cache_hit_tier": cache_hit_tier,
            "routed_tier": routed_tier,
            "executed_tier": executed_tier,
            "fallback_occurred": fallback_occurred,
            "model_calls": model_calls,
            "revisions": revisions
          },
          "agents": {
            "active_agents": active_agents,
            "pruned_agents": pruned_agents,
            "agent_trace": agent_trace
          }
        }

        # Append to results file
        self._append_jsonl(self.results_path, record)

        return {"success": True, "record": record}

    def _log_failure(
        self,
        task_id: str,
        dataset_index: int,
        dataset: str,
        category: str,
        code: str,
        stderr: str,
        timeout: bool
    ):
        """
        Helper to write a structured failure record for debugging.
        """
        record = {
          "schema_version": 1,
          "task_id": task_id,
          "dataset_index": dataset_index,
          "dataset": dataset,
          "failure_category": category,
          "timeout_occurred": timeout,
          "generated_code": code,
          "stderr": stderr
        }
        self._append_jsonl(self.failed_path, record)
