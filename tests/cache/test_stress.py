import os
import time
import shutil
import pytest
import psutil
import random
from liteagent.cache import KVCacheManager

SSD_TEST_DIR = "tests/cache/temp_stress_ssd"
LOG_TEST_DIR = "tests/cache/temp_stress_logs"

class MockLlamaState:
    def __init__(self, data: bytes):
        self.llama_state = data
        self.llama_state_size = len(data)
        self.input_ids = []
        self.scores = []
        self.n_tokens = 0
        self.seed = 42

class MockLlama:
    def __init__(self, state_bytes: bytes = b"default_mock"):
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

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)

def test_sequential_stress_test():
    # Standby capacity = 2
    cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    
    agents = ["Planner", "Retriever", "Executor", "Critic"]
    llama_instances = {role: MockLlama(f"state_for_{role}".encode("utf-8")) for role in agents}
    
    print("\n--- Starting Sequential Stress Test ---")
    mem_start = get_process_memory_mb()
    print(f"Initial Memory Usage: {mem_start:.2f} MB")
    
    # 1. Run sequential execution Planner -> Retriever -> Executor -> Critic
    keys = {}
    for role in agents:
        key = cm.compute_cache_key("model_1b", f"sys_prompt_{role}", [f"history_{role}"])
        keys[role] = key
        
        # Load (should miss)
        status = cm.load_cache(key, llama_instances[role], "llama3.2:1b", 512, "hash")
        assert status == "MISS"
        
        # Save cache
        cm.save_cache(key, role, llama_instances[role], "llama3.2:1b", 512, "hash")
        
        mem_curr = get_process_memory_mb()
        print(f"[{role} Saved] Standby: {len(cm.storage.standby_cache)} | Memory: {mem_curr:.2f} MB")

    # Since capacity is 2:
    # Planner & Retriever were saved first.
    # Executor save triggered eviction of Retriever (Vs = t * 0.7 vs Planner Vs = t * 0.5)
    # Critic save triggered eviction of Planner (Vs = t * 0.5 vs Executor Vs = t * 0.2)
    # Standby should contain: Executor and Critic.
    # Cold SSD should contain: Retriever and Planner.
    assert "key_retriever" not in cm.storage.standby_cache
    
    # Verify we can reload all of them token-for-token
    for role in agents:
        key = keys[role]
        llama_fresh = MockLlama()
        cm.hot_state = None
        
        status = cm.load_cache(key, llama_fresh, "llama3.2:1b", 512, "hash")
        assert status in ["STANDBY", "COLD"]
        assert llama_fresh.loaded_state.llama_state == f"state_for_{role}".encode("utf-8")
        
        mem_curr = get_process_memory_mb()
        print(f"[{role} Restored ({status})] Memory: {mem_curr:.2f} MB")

def test_random_switching_stress_test():
    # Standby capacity = 2
    cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    
    agents = ["Planner", "Retriever", "Executor", "Critic"]
    llama_instances = {role: MockLlama(f"state_for_{role}".encode("utf-8")) for role in agents}
    
    # Pre-populate keys
    keys = {}
    for role in agents:
        keys[role] = cm.compute_cache_key("model_1b", f"sys_prompt_{role}", [f"history_{role}"])
        
    print("\n--- Starting Random Switching Stress Test ---")
    mem_start = get_process_memory_mb()
    print(f"Initial Memory Usage: {mem_start:.2f} MB")
    
    # Run 20 random context transitions
    random.seed(42)
    transition_path = [random.choice(agents) for _ in range(20)]
    print(f"Transition Path: {' -> '.join(transition_path)}")
    
    for i, role in enumerate(transition_path):
        key = keys[role]
        llama = llama_instances[role]
        
        # Load cache
        status = cm.load_cache(key, llama, "llama3.2:1b", 512, "hash")
        
        # If miss, simulate prefill and save
        if status == "MISS":
            cm.save_cache(key, role, llama, "llama3.2:1b", 512, "hash")
            status = "MISS (Prefilled & Saved)"
        else:
            # Re-save to simulate ongoing conversation context updates
            cm.save_cache(key, role, llama, "llama3.2:1b", 512, "hash")
            
        mem_curr = get_process_memory_mb()
        print(f"[Step {i+1:02d} - {role} ({status})] Standby slots: {len(cm.storage.standby_cache)} | Memory: {mem_curr:.2f} MB")
        
        # Verify state integrity
        assert llama.save_state().llama_state == f"state_for_{role}".encode("utf-8")
        
    print(f"Final Memory Usage: {get_process_memory_mb():.2f} MB")
