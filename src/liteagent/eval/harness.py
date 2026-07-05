import os
import time
import json
import threading
import subprocess
from typing import Any, Dict, List

# Metrics
from liteagent.eval.metrics.gsm8k_metric import score_gsm8k
from liteagent.eval.metrics.hotpotqa_metric import score_hotpotqa
from liteagent.eval.metrics.humaneval_metric import score_humaneval

class GPUPowerTracker:
    """
    Background thread to poll GPU power draw via nvidia-smi every 100ms.
    Integrates samples to compute total energy in Joules.
    """
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.power_samples = []
        self._stop_event = threading.Event()
        self._thread = None
        
    def _poll(self):
        while not self._stop_event.is_set():
            start_time = time.time()
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=0.08
                )
                val = float(res.stdout.strip())
                self.power_samples.append(val)
            except Exception:
                pass
            
            elapsed = time.time() - start_time
            sleep_time = max(0.001, self.interval - elapsed)
            self._stop_event.wait(sleep_time)
            
    def start(self):
        self.power_samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        
    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if not self.power_samples:
            return 0.0
        # Energy (Joules) = Sum of Power (Watts) * delta_t (seconds)
        energy = sum(self.power_samples) * self.interval
        return energy

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
        sample_seed: int = 42
    ):
        self.baseline_name = baseline_name
        self.results_path = results_path
        self.failed_path = failed_path
        self.sample_seed = sample_seed
        self.power_tracker = GPUPowerTracker()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.failed_path), exist_ok=True)

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
        task_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes a single benchmark task while polling GPU energy.
        """
        # Determine prompt text
        if dataset == "gsm8k":
            prompt = task_item.get("question", "")
        elif dataset == "hotpotqa":
            prompt = self.format_hotpotqa_prompt(task_item)
        elif dataset == "humaneval":
            prompt = task_item.get("prompt", "")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        # Start energy tracker
        self.power_tracker.start()
        start_time = time.time()
        
        # Run inference
        task_dict = {"prompt": prompt, "id": task_id}
        session_id = f"session_{task_id}_{self.baseline_name}"
        
        try:
            res = runner.execute_task(
                task=task_dict,
                session_id=session_id,
                system_prompt=""
            )
            latency_ms = (time.time() - start_time) * 1000.0
            energy_joules = self.power_tracker.stop()
            
            response = res.get("response_text", "")
            prefill_tokens = res.get("prefill_tokens", 0)
            tokens_gen = res.get("tokens_generated", 0)
            cache_hit_tier = res.get("cache_hit_tier", "NONE")
            routed_tier = res.get("routed_tier", "NONE")
            executed_tier = res.get("executed_tier", "NONE")
            fallback_occurred = res.get("fallback_occurred", False)
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            energy_joules = self.power_tracker.stop()
            
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
            eval_res = score_humaneval(response, test_code, timeout=3.0)
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
          "schema_version": 1,
          "task_id": task_id,
          "dataset_index": dataset_index,
          "dataset": dataset,
          "sample_seed": self.sample_seed,
          "baseline": self.baseline_name,
          "metrics": {
            "quality_score": quality_score,
            "quality_score_extra": quality_extra,
            "latency_ms": latency_ms,
            "energy_joules": energy_joules,
            "prefill_tokens": prefill_tokens,
            "tokens_generated": tokens_gen,
            "cache_hit_tier": cache_hit_tier,
            "routed_tier": routed_tier,
            "executed_tier": executed_tier,
            "fallback_occurred": fallback_occurred
          }
        }
        
        # Append to results file
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
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
        with open(self.failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
