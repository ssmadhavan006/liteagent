from liteagent.router.pruning import map_tier_and_pruning

def test_pruning_low_complexity():
    # score = 0.15 (< 0.25) -> Low Complexity
    tier, location, active, pruned = map_tier_and_pruning(0.15, 0.25, 0.75)
    assert tier == "Small"
    assert location == "Edge"
    assert active == ["Executor"]
    assert set(pruned) == {"Planner", "Retriever", "Critic"}

    # HotpotQA supplies documents, so retrieval is added alongside the Executor.
    tier, location, active, pruned = map_tier_and_pruning(0.15, 0.25, 0.75, benchmark="HotpotQA")
    assert tier == "Small"
    assert active == ["Retriever", "Executor"]
    assert set(pruned) == {"Planner", "Critic"}

def test_pruning_medium_complexity():
    # score = 0.50 (between 0.25 and 0.75) -> Medium Complexity
    tier, location, active, pruned = map_tier_and_pruning(0.50, 0.25, 0.75)
    assert tier == "Medium"
    assert location == "Edge"
    assert active == ["Planner", "Executor"]
    assert set(pruned) == {"Retriever", "Critic"}

    tier, location, active, pruned = map_tier_and_pruning(0.50, 0.25, 0.75, benchmark="hotpotqa")
    assert active == ["Planner", "Retriever", "Executor"]
    assert pruned == ["Critic"]

def test_pruning_high_complexity():
    # score = 0.80 (>= 0.75) -> High Complexity
    tier, location, active, pruned = map_tier_and_pruning(0.80, 0.25, 0.75)
    assert tier == "Large"
    assert location == "Workstation"
    assert active == ["Planner", "Executor", "Critic"]
    assert pruned == ["Retriever"]

    tier, location, active, pruned = map_tier_and_pruning(0.80, 0.25, 0.75, benchmark="hotpotqa")
    assert active == ["Planner", "Retriever", "Executor", "Critic"]
    assert pruned == []

def test_active_agents_preserve_canonical_chain_order():
    for benchmark in (None, "gsm8k", "hotpotqa", "humaneval"):
        for score in (0.1, 0.5, 0.9):
            _, _, active, pruned = map_tier_and_pruning(score, 0.25, 0.75, benchmark)
            canonical = ["Planner", "Retriever", "Executor", "Critic"]
            assert active == [a for a in canonical if a in active]
            assert set(active) | set(pruned) == set(canonical)
            assert not set(active) & set(pruned)
