import pytest
from liteagent.baselines.ablation_configs import configure_routing_only, configure_cache_only
from liteagent.baselines.vllm_prefix_cache_approx import VLLMPrefixCacheApprox

class DummyDispatcher:
    def __init__(self):
        self.routing_disabled = False
        self.cache_disabled = False
        self.baseline_name = ""

class DummyLlamaInstance:
    def save_state(self):
        return b"dummy_state_bytes"

    def load_state(self, state):
        pass

def test_ablation_configs():
    disp = DummyDispatcher()
    configure_routing_only(disp)
    assert disp.routing_disabled is False
    assert disp.cache_disabled is True
    assert disp.baseline_name == "routing_only"

    configure_cache_only(disp)
    assert disp.routing_disabled is True
    assert disp.cache_disabled is False
    assert disp.baseline_name == "cache_only"

def test_vllm_prefix_cache_approx():
    cache = VLLMPrefixCacheApprox(max_in_memory_slots=2, log_dir=None)
    llama = DummyLlamaInstance()

    # Save 2 slots
    cache.save_cache("session1", "Planner", llama, "llama3.2:1b", 512, "h1")
    cache.save_cache("session2", "Executor", llama, "llama3.2:1b", 512, "h2")

    assert cache.load_cache("session1", llama, "llama3.2:1b", 512, "h1") == "HOT"
    assert cache.load_cache("session3", llama, "llama3.2:1b", 512, "h3") == "MISS"

    # Slot overflow raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        cache.save_cache("session3", "Critic", llama, "llama3.2:1b", 512, "h3")
    assert "EXPECTED_LIMIT_REACHED" in str(exc_info.value)
