ALL_AGENTS = ["Planner", "Retriever", "Executor", "Critic"]

# Benchmarks that ship candidate documents and therefore need a retrieval turn.
RETRIEVAL_BENCHMARKS = {"hotpotqa"}


def map_tier_and_pruning(score: float, theta_low: float, theta_high: float, benchmark: str = None) -> tuple[str, str, list[str], list[str]]:
    """
    Maps complexity score Sc and thresholds to model tiers and pruned agents.
    Returns (model_tier, execution_location, active_agents, pruned_agents)

    The Executor is never pruned: it is the only agent that produces an answer,
    so removing it would leave the chain with nothing to score.
    """
    needs_retrieval = (benchmark or "").lower() in RETRIEVAL_BENCHMARKS

    if score < theta_low:
        model_tier = "Small"
        execution_location = "Edge"
        active_agents = ["Executor"]

    elif theta_low <= score < theta_high:
        model_tier = "Medium"
        execution_location = "Edge"
        active_agents = ["Planner", "Executor"]

    else:
        model_tier = "Large"
        execution_location = "Workstation"
        active_agents = ["Planner", "Executor", "Critic"]

    if needs_retrieval:
        active_agents.append("Retriever")

    active_agents = [agent for agent in ALL_AGENTS if agent in active_agents]
    pruned_agents = [agent for agent in ALL_AGENTS if agent not in active_agents]

    return model_tier, execution_location, active_agents, pruned_agents
