import os
import shutil
import json
import pytest
from liteagent.cache import KVCacheManager
from liteagent.network.dispatch import TaskDispatcher
from liteagent.baselines.static_full_pipeline import StaticFullPipelineRunner
from liteagent.baselines.ablation_configs import configure_routing_only, configure_cache_only
from liteagent.baselines.routellm_heuristic_approx import RouteLLMHeuristicDispatcher

SSD_TEST_DIR = "tests/baselines/temp_parity_ssd"
LOG_TEST_DIR = "tests/baselines/temp_parity_logs"

class DummyClient:
    def dispatch_task(self, request_id, task_id, session_id, agent_role, prompt, system_prompt, temperature, max_tokens, cache_disabled):
        return {
            "response_text": "large_res",
            "tokens_generated": 5,
            "prefill_tokens": 10,
            "prefill_latency_ms": 100.0,
            "generation_latency_ms": 200.0,
            "cache_hit_tier": "MISS",
            "protocol_version": 1,
            "request_id": request_id,
            "server_receive_ts": 1000.0,
            "server_start_compute_ts": 1001.0,
            "server_end_compute_ts": 1003.0,
            "server_serialize_duration_ms": 10.0
        }

@pytest.fixture(autouse=True)
def setup_and_teardown():
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    yield
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

def validate_log_schema(log_file: str, expected_baseline: str):
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # 1. Check required top-level keys
            assert "timestamp" in data and isinstance(data["timestamp"], str)
            assert "request_id" in data and isinstance(data["request_id"], str)
            assert "component" in data and isinstance(data["component"], str)
            assert "event" in data and isinstance(data["event"], str)
            assert "baseline" in data and isinstance(data["baseline"], str)
            assert "feature_flags" in data and isinstance(data["feature_flags"], dict)

            # 2. Check baseline value matches
            assert data["baseline"] == expected_baseline

            # 3. Check feature_flags keys and types
            ff = data["feature_flags"]
            assert "routing" in ff and isinstance(ff["routing"], bool)
            assert "cache" in ff and isinstance(ff["cache"], bool)
            assert "grpc" in ff and isinstance(ff["grpc"], bool)

def test_instrumentation_parity_all_baselines():
    edge_cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    client = DummyClient()
    task = {"id": 101, "prompt": "def greet():\n  pass\n" * 15, "benchmark": "HumanEval"}

    # 1. LiteAgent (Full)
    la_dispatcher = TaskDispatcher("config/router_config.yaml", edge_cm, client, LOG_TEST_DIR)
    la_dispatcher.baseline_name = "liteagent"
    la_dispatcher.execute_task(task, "s-la", "sys")
    validate_log_schema(os.path.join(LOG_TEST_DIR, "operations.jsonl"), "liteagent")

    # Reset log file for next test
    os.remove(os.path.join(LOG_TEST_DIR, "operations.jsonl"))

    # 2. Static Full-Pipeline
    sf_runner = StaticFullPipelineRunner(client, LOG_TEST_DIR)
    # Patch dummy manager to avoid directory collision in parallel runs
    sf_runner.dispatcher.edge_cache_manager = edge_cm
    sf_runner.execute_task(task, "s-sf", "sys")
    validate_log_schema(os.path.join(LOG_TEST_DIR, "operations.jsonl"), "static_full")

    # Reset
    os.remove(os.path.join(LOG_TEST_DIR, "operations.jsonl"))

    # 3. LiteAgent (Routing Only)
    ro_dispatcher = TaskDispatcher("config/router_config.yaml", edge_cm, client, LOG_TEST_DIR)
    configure_routing_only(ro_dispatcher)
    ro_dispatcher.execute_task(task, "s-ro", "sys")
    validate_log_schema(os.path.join(LOG_TEST_DIR, "operations.jsonl"), "routing_only")

    # Reset
    os.remove(os.path.join(LOG_TEST_DIR, "operations.jsonl"))

    # 4. LiteAgent (Cache Only)
    co_dispatcher = TaskDispatcher("config/router_config.yaml", edge_cm, client, LOG_TEST_DIR)
    configure_cache_only(co_dispatcher)
    co_dispatcher.execute_task(task, "s-co", "sys")
    validate_log_schema(os.path.join(LOG_TEST_DIR, "operations.jsonl"), "cache_only")

    # Reset
    os.remove(os.path.join(LOG_TEST_DIR, "operations.jsonl"))

    # 5. RouteLLM Heuristic
    rl_dispatcher = RouteLLMHeuristicDispatcher(client, edge_cm, LOG_TEST_DIR, threshold=0.5)
    rl_dispatcher.execute_task(task, "s-rl", "sys")
    validate_log_schema(os.path.join(LOG_TEST_DIR, "operations.jsonl"), "routellm_heuristic")
