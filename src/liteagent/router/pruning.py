def map_tier_and_pruning(score: float, theta_low: float, theta_high: float, benchmark: str = None) -> tuple[str, str, list[str], list[str]]:
    """
    Maps complexity score Sc and thresholds to model tiers and pruned agents.
    Returns (model_tier, execution_location, active_agents, pruned_agents)
    """
    all_agents = ["Planner", "Retriever", "Executor", "Critic"]
    
    if score < theta_low:
        model_tier = "Small"
        execution_location = "Edge"
        # Determine whether lookup or execution is needed
        if benchmark == "HotpotQA":
            active_agents = ["Retriever"]
        else:
            active_agents = ["Executor"]
        pruned_agents = [agent for agent in all_agents if agent not in active_agents]
        
    elif theta_low <= score < theta_high:
        model_tier = "Medium"
        execution_location = "Edge"
        active_agents = ["Planner", "Retriever", "Executor"]
        pruned_agents = ["Critic"]
        
    else:
        model_tier = "Large"
        execution_location = "Workstation"
        active_agents = ["Planner", "Retriever", "Executor", "Critic"]
        pruned_agents = []
        
    return model_tier, execution_location, active_agents, pruned_agents
