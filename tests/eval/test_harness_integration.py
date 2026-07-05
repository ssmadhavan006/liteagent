import os
import json
import pytest
import tempfile
from liteagent.eval.harness import EvaluationHarness

class MockRunner:
    def __init__(self, response: str, fails_compile: bool = False):
        self.response = response
        self.fails_compile = fails_compile

    def execute_task(self, task: dict, session_id: str, system_prompt: str) -> dict:
        if self.fails_compile:
            raise RuntimeError("Mock runner compilation failure")
        return {
            "response_text": self.response,
            "prefill_tokens": 10,
            "tokens_generated": 5,
            "cache_hit_tier": "STANDBY",
            "routed_tier": "Medium",
            "executed_tier": "Medium",
            "fallback_occurred": False
        }

def test_harness_gsm8k_integration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_file = os.path.join(tmp_dir, "results.jsonl")
        fail_file = os.path.join(tmp_dir, "failed.jsonl")
        
        harness = EvaluationHarness(
            baseline_name="mock_baseline",
            results_path=res_file,
            failed_path=fail_file
        )
        
        runner = MockRunner(response="The answer is #### 42")
        task_item = {
            "question": "What is 40 + 2?",
            "answer": "#### 42"
        }
        
        res = harness.evaluate_task(
            dataset="gsm8k",
            task_id="gsm8k_001",
            dataset_index=0,
            runner=runner,
            task_item=task_item
        )
        
        assert res["success"] is True
        assert os.path.exists(res_file)
        
        # Read the logged record
        with open(res_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["schema_version"] == 1
            assert record["task_id"] == "gsm8k_001"
            assert record["baseline"] == "mock_baseline"
            assert record["dataset"] == "gsm8k"
            assert record["metrics"]["quality_score"] == 1.0
            assert isinstance(record["metrics"]["energy_joules"], float)

def test_harness_humaneval_sandbox_failure_integration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_file = os.path.join(tmp_dir, "results.jsonl")
        fail_file = os.path.join(tmp_dir, "failed.jsonl")
        
        harness = EvaluationHarness(
            baseline_name="mock_humaneval",
            results_path=res_file,
            failed_path=fail_file
        )
        
        # This code fails assertion
        runner = MockRunner(response="def add(a, b):\n    return a - b")
        task_item = {
            "prompt": "def add(a, b):\n",
            "test": "assert add(2, 3) == 5"
        }
        
        res = harness.evaluate_task(
            dataset="humaneval",
            task_id="he_001",
            dataset_index=5,
            runner=runner,
            task_item=task_item
        )
        
        assert res["success"] is True
        assert res["record"]["metrics"]["quality_score"] == 0.0
        
        # Assert both files exist
        assert os.path.exists(res_file)
        assert os.path.exists(fail_file)
        
        # Check failed log contains details
        with open(fail_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            fail_record = json.loads(lines[0])
            assert fail_record["schema_version"] == 1
            assert fail_record["task_id"] == "he_001"
            assert fail_record["failure_category"] == "FAILURE_ASSERTION"
            assert "add(a, b)" in fail_record["generated_code"]
