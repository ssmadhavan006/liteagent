import time
from liteagent.cache.tiers import CacheStateMetadata
from liteagent.cache.eviction import compute_eviction_score, select_eviction_victim

def test_compute_eviction_score():
    # If t_elapsed is 10.0 seconds
    # Critic weight is 1.0 -> Vs = 10.0 * (1.0 - 1.0) = 0.0
    assert compute_eviction_score(0.0, 10.0, "Critic") == 0.0

    # Retriever weight is 0.3 -> Vs = 10.0 * (1.0 - 0.3) = 7.0
    assert abs(compute_eviction_score(0.0, 10.0, "Retriever") - 7.0) < 1e-6

    # Planner weight is 0.5 -> Vs = 10.0 * (1.0 - 0.5) = 5.0
    assert abs(compute_eviction_score(0.0, 10.0, "Planner") - 5.0) < 1e-6

    # Executor weight is 0.8 -> Vs = 10.0 * (1.0 - 0.8) = 2.0
    assert abs(compute_eviction_score(0.0, 10.0, "Executor") - 2.0) < 1e-6

    # Unknown agent -> weight is 0.0 -> Vs = 10.0 * 1.0 = 10.0
    assert abs(compute_eviction_score(0.0, 10.0, "Unknown") - 10.0) < 1e-6

def test_select_eviction_victim():
    # Setup metadata store
    store = {}
    current = time.time()

    # Case 1: Critic accessed 100 seconds ago vs Retriever accessed 50 seconds ago
    # Critic Vs = 100 * (1.0 - 1.0) = 0
    # Retriever Vs = 50 * (1.0 - 0.3) = 35
    # Retriever should be the victim despite being accessed MORE recently than Critic!
    store["critic_key"] = CacheStateMetadata(
        session_key="critic_key",
        agent_role="Critic",
        model_tag="llama3.2:1b",
        ctx_size=512,
        prompt_hash="",
        state_size_bytes=100
    )
    store["critic_key"].last_accessed = current - 100.0

    store["retriever_key"] = CacheStateMetadata(
        session_key="retriever_key",
        agent_role="Retriever",
        model_tag="llama3.2:1b",
        ctx_size=512,
        prompt_hash="",
        state_size_bytes=100
    )
    store["retriever_key"].last_accessed = current - 50.0

    victim, score = select_eviction_victim(store, ["critic_key", "retriever_key"])
    assert victim == "retriever_key"
    assert abs(score - 35.0) < 0.5 # account for small time delta during test execution
