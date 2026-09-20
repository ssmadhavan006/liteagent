import time
from liteagent.cache.tiers import CacheStateMetadata

ROLE_PRIORITIES = {
    "Critic": 1.0,
    "Executor": 0.8,
    "Planner": 0.5,
    "Retriever": 0.3
}

def compute_eviction_score(last_accessed: float, current_time: float, agent_role: str) -> float:
    """
    Computes virtual recency score Vs = t_elapsed * (1.0 - w_agent)
    """
    t_elapsed = max(0.0, current_time - last_accessed)
    w_agent = ROLE_PRIORITIES.get(agent_role, 0.0)
    return t_elapsed * (1.0 - w_agent)

def select_eviction_victim(
    metadata_store: dict[str, CacheStateMetadata],
    active_keys: list[str]
) -> tuple[str | None, float]:
    """
    Selects the key from active_keys that has the highest PW-LRU eviction score.
    Returns (victim_key, score).
    """
    if not active_keys:
        return None, -1.0

    current_time = time.time()
    highest_score = -1.0
    victim_key = None

    for key in active_keys:
        meta = metadata_store.get(key)
        if not meta:
            # If metadata is missing, prioritize this key for eviction
            score = float("inf")
        else:
            score = compute_eviction_score(meta.last_accessed, current_time, meta.agent_role)

        if score > highest_score:
            highest_score = score
            victim_key = key

    if victim_key is None and active_keys:
        victim_key = active_keys[0]
        highest_score = 0.0

    return victim_key, highest_score

