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
        # The Critic is excluded from the default chain on measured evidence.
        # Its verdicts do not discriminate correct from incorrect answers
        # (Youden's J = -0.098 at 1B, +0.092 at 3B), so the revisions it triggers
        # are close to random edits applied to already-correct answers. On 25
        # GSM8K tasks it won 0 and lost 4 against an otherwise identical chain,
        # taking accuracy from 0.640 to 0.480 while raising median latency from
        # 16.5s to 72.8s. See docs/phase9/.
        # It remains implemented and is still used by the cascade configuration,
        # where rejections drive escalation rather than revision.
        active_agents = ["Planner", "Executor"]

    if needs_retrieval:
        active_agents.append("Retriever")

    active_agents = [agent for agent in ALL_AGENTS if agent in active_agents]
    pruned_agents = [agent for agent in ALL_AGENTS if agent not in active_agents]

    return model_tier, execution_location, active_agents, pruned_agents
