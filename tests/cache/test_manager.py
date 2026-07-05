import os
import shutil
import pytest
import json
import time
from liteagent.cache.manager import KVCacheManager, CacheRestoreError

SSD_TEST_DIR = "tests/cache/temp_manager_ssd"
LOG_TEST_DIR = "tests/cache/temp_manager_logs"

class MockLlamaState:
    def __init__(self, data: bytes):
        self.llama_state = data
        self.llama_state_size = len(data)
        self.input_ids = []
        self.scores = []
        self.n_tokens = 0
        self.seed = 42

class MockLlama:
    def __init__(self, state_bytes: bytes = b"dummy_llama_cpp_state"):
        self.state_bytes = state_bytes
        self.loaded_state = None

    def save_state(self) -> MockLlamaState:
        return MockLlamaState(self.state_bytes)

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

def test_cache_key_computation():
    cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    key1 = cm.compute_cache_key("model1", "system1", ["prefix1", "prefix2"])
    key2 = cm.compute_cache_key("model1", "system1", ["prefix1", "prefix2"])
    key3 = cm.compute_cache_key("model1", "system1", ["prefix1", "different"])
    
    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 64 # SHA256 hex string length

def test_save_and_load_hit_flow():
    cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    llama = MockLlama(b"state_content_123")
    
    # Save cache
    cm.save_cache("session1", "Planner", llama, "llama3.2:1b", 512, "hash1")
    
    # Verify standby storage
    assert "session1" in cm.storage.standby_cache
    
    # Manually clear hot_state to force standby hit
    cm.hot_state = None
    
    # Reset mock and load
    llama_fresh = MockLlama()
    status = cm.load_cache("session1", llama_fresh, "llama3.2:1b", 512, "hash1")
    
    assert status == "STANDBY"
    assert llama_fresh.loaded_state is not None
    assert llama_fresh.loaded_state.llama_state == b"state_content_123"

def test_pru_eviction_promotion_flow():
    # max_ram_states = 2
    cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    
    # Initialize mock model states
    llama_critic = MockLlama(b"critic_context_bytes")
    llama_retriever = MockLlama(b"retriever_context_bytes")
    llama_planner = MockLlama(b"planner_context_bytes")
    
    # Save two states to fill RAM capacity
    cm.save_cache("key_critic", "Critic", llama_critic, "llama3.2:1b", 512, "h_crit")
    cm.save_cache("key_retriever", "Retriever", llama_retriever, "llama3.2:1b", 512, "h_ret")
    
    assert len(cm.storage.standby_cache) == 2
    
    # Stagger last_accessed times explicitly to ensure positive elapsed time
    # Set last_accessed for both to 10 seconds ago
    now = time.time()
    cm.metadata_store["key_critic"].last_accessed = now - 10.0
    cm.metadata_store["key_retriever"].last_accessed = now - 10.0
    
    # Save third state: triggers eviction
    # Critic Vs = 10 * (1 - 1.0) = 0.0
    # Retriever Vs = 10 * (1 - 0.3) = 7.0
    # Retriever should be evicted to cold SSD!
    cm.save_cache("key_planner", "Planner", llama_planner, "llama3.2:1b", 512, "h_plan")
    
    assert len(cm.storage.standby_cache) == 2
    assert "key_retriever" not in cm.storage.standby_cache
    assert "key_critic" in cm.storage.standby_cache
    assert "key_planner" in cm.storage.standby_cache
    
    # Verify cold file exists for the victim (retriever)
    cold_bin = os.path.join(SSD_TEST_DIR, "key_retriever.bin")
    cold_json = os.path.join(SSD_TEST_DIR, "key_retriever.json")
    assert os.path.exists(cold_bin)
    assert os.path.exists(cold_json)
    
    # Force load of key_planner to change its last_accessed to now,
    # ensuring key_planner has Vs = 0 and key_planner is NOT immediately evicted
    # on the next step. Let's make key_critic elapsed time larger
    cm.metadata_store["key_critic"].last_accessed = now - 20.0
    cm.metadata_store["key_planner"].last_accessed = now - 10.0
    
    # Load key_retriever from COLD SSD
    llama_fresh = MockLlama()
    # Force loading from SSD by clearing hot and standby caches for this key
    cm.hot_state = None
    cm.storage.delete_from_standby("key_retriever")
    
    status = cm.load_cache("key_retriever", llama_fresh, "llama3.2:1b", 512, "h_ret")
    assert status == "COLD"
    assert llama_fresh.loaded_state.llama_state == b"retriever_context_bytes"
    
    # Checking structured log output
    log_file = os.path.join(LOG_TEST_DIR, "cache_operations.jsonl")
    assert os.path.exists(log_file)
    
    events = []
    with open(log_file, "r") as f:
        for line in f:
            events.append(json.loads(line))
            
    # Assert eviction log attributes
    evicts = [e for e in events if e["event"] == "EVICT"]
    assert len(evicts) == 2
    assert evicts[0]["session_key"] == "key_retriever"
    assert evicts[0]["agent_role"] == "Retriever"
    assert evicts[0]["destination"] == "Cold SSD"
    
    assert evicts[1]["session_key"] == "key_planner"
    assert evicts[1]["agent_role"] == "Planner"
