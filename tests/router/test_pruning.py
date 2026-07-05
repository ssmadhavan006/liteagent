from liteagent.router.pruning import map_tier_and_pruning

def test_pruning_low_complexity():
    # theta = 0.50 -> low_threshold = 0.25, high_threshold = 0.75
    # score = 0.15 (< 0.25) -> Low Complexity
    tier, location, active, pruned = map_tier_and_pruning(0.15, 0.50)
    assert tier == "Small"
    assert location == "Edge"
    assert active == ["Executor"]
    assert set(pruned) == {"Planner", "Retriever", "Critic"}
    
    # HotpotQA benchmark check
    tier, location, active, pruned = map_tier_and_pruning(0.15, 0.50, benchmark="HotpotQA")
    assert tier == "Small"
    assert active == ["Retriever"]
    assert set(pruned) == {"Planner", "Executor", "Critic"}

def test_pruning_medium_complexity():
    # score = 0.50 (between 0.25 and 0.75) -> Medium Complexity
    tier, location, active, pruned = map_tier_and_pruning(0.50, 0.50)
    assert tier == "Medium"
    assert location == "Edge"
    assert active == ["Planner", "Retriever", "Executor"]
    assert pruned == ["Critic"]

def test_pruning_high_complexity():
    # score = 0.80 (>= 0.75) -> High Complexity
    tier, location, active, pruned = map_tier_and_pruning(0.80, 0.50)
    assert tier == "Large"
    assert location == "Workstation"
    assert active == ["Planner", "Retriever", "Executor", "Critic"]
    assert pruned == []
