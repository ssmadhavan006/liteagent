import os
import shutil
import pytest
import json
from liteagent.baselines.vllm_prefix_cache_approx import VLLMPrefixCacheApprox

SSD_TEST_DIR = "tests/baselines/temp_vllm_ssd"
LOG_TEST_DIR = "tests/baselines/temp_vllm_logs"

class MockLlamaState:
    def __init__(self, data: bytes):
        self.llama_state = data
        self.llama_state_size = len(data)

class MockLlama:
    def __init__(self):
        self.loaded_state = None
        self.save_count = 0

    def save_state(self) -> MockLlamaState:
        self.save_count += 1
        return MockLlamaState(f"state_{self.save_count}".encode())

    def load_state(self, state):
        self.loaded_state = state

@pytest.fixture(autouse=True)
def setup_and_teardown():
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    yield
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

def test_flat_cache_capacity_limit():
    # Instantiate with max 2 slots
    cache_mgr = VLLMPrefixCacheApprox(max_in_memory_slots=2, log_dir=LOG_TEST_DIR)
    llama = MockLlama()
    
    # Save session 1
    cache_mgr.save_cache("session-1", "Planner", llama, "mock-1b", 512, "hash1")
    assert len(cache_mgr.active_slots) == 1
    
    # Save session 2
    cache_mgr.save_cache("session-2", "Executor", llama, "mock-1b", 512, "hash2")
    assert len(cache_mgr.active_slots) == 2
    
    # Attempt to save session 3 -> should raise EXPECTED_LIMIT_REACHED
    with pytest.raises(RuntimeError) as exc_info:
        cache_mgr.save_cache("session-3", "Critic", llama, "mock-1b", 512, "hash3")
        
    assert "EXPECTED_LIMIT_REACHED" in str(exc_info.value)
    
    # Verify that EXPECTED_LIMIT_REACHED event was written to the operations log
    log_file = os.path.join(LOG_TEST_DIR, "cache_operations.jsonl")
    assert os.path.exists(log_file)
    
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            logs.append(json.loads(line))
            
    limit_logs = [log for log in logs if log["event"] == "EXPECTED_LIMIT_REACHED"]
    assert len(limit_logs) == 1
    assert limit_logs[0]["session_key"] == "session-3"
    assert limit_logs[0]["reason"] == "Max in-memory prefix slots exhausted."
