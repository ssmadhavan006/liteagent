import os
import shutil
import pytest
from liteagent.cache import KVCacheManager
from liteagent.network.dispatch import TaskDispatcher
from liteagent.baselines.ablation_configs import configure_routing_only, configure_cache_only

SSD_TEST_DIR = "tests/baselines/temp_ssd"
LOG_TEST_DIR = "tests/baselines/temp_logs"

class DummyClient:
    def __init__(self):
        self.dispatched = []

    def dispatch_task(self, request_id, task_id, session_id, agent_role, prompt, system_prompt, temperature, max_tokens, cache_disabled):
        self.dispatched.append({
            "request_id": request_id,
            "task_id": task_id,
            "session_id": session_id,
            "agent_role": agent_role,
            "prompt": prompt,
            "cache_disabled": cache_disabled
        })
        return {
            "response_text": "mock_large_res",
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

def test_routing_only_ablation():
    edge_cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    client = DummyClient()
    
    dispatcher = TaskDispatcher(
        router_config_path="config/router_config.yaml",
        edge_cache_manager=edge_cm,
        workstation_client=client,
        log_dir=LOG_TEST_DIR
    )
    configure_routing_only(dispatcher)
    
    # We pass a highly complex prompt to guarantee Large-tier routing
    task = {
        "id": 101,
        "prompt": (
            "def calculate(x):\n"
            "    # math and code logic\n"
            "    if x > 0 and x < 10:\n"
            "        return x * x\n"
            "    return 0\n"
        ) * 15,
        "benchmark": "HumanEval"
    }
    
    res = dispatcher.execute_task(task, "session-ro", "system-ro")
    
    assert dispatcher.baseline_name == "routing_only"
    assert dispatcher.routing_disabled is False
    assert dispatcher.cache_disabled is True
    
    # Verify that gRPC dispatch request was called with cache_disabled=True
    assert len(client.dispatched) == 1
    assert client.dispatched[0]["cache_disabled"] is True
    assert res["response_text"] == "mock_large_res"

def test_cache_only_ablation():
    edge_cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    client = DummyClient()
    
    dispatcher = TaskDispatcher(
        router_config_path="config/router_config.yaml",
        edge_cache_manager=edge_cm,
        workstation_client=client,
        log_dir=LOG_TEST_DIR
    )
    configure_cache_only(dispatcher)
    
    # Even with a very simple low-complexity prompt, it must route to Large workstation tier
    task = {
        "id": 102,
        "prompt": "Hello",
        "benchmark": "HotpotQA"
    }
    
    res = dispatcher.execute_task(task, "session-co", "system-co")
    
    assert dispatcher.baseline_name == "cache_only"
    assert dispatcher.routing_disabled is True
    assert dispatcher.cache_disabled is False
    
    # Verify that gRPC dispatch request was called with cache_disabled=False
    assert len(client.dispatched) == 1
    assert client.dispatched[0]["cache_disabled"] is False
    assert res["response_text"] == "mock_large_res"
