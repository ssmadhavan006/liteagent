import os

from liteagent.agents.orchestrator import AgentOrchestrator
from liteagent.cache import KVCacheManager
from liteagent.network.dispatch import TaskDispatcher


class LiteAgentRunner:
    """
    Full LiteAgent system: complexity routing + three-tier KV cache + agent chain.

    This is the Configuration A runner for the co-design ablation (H3). The
    routing-only and cache-only configurations are the same object with one
    half disabled, so the two halves are never compared across different
    implementations.
    """

    def __init__(
        self,
        workstation_client=None,
        router_config_path: str = "config/router_config.yaml",
        ssd_dir: str = "cache_ssd",
        max_ram_states: int = 5,
        log_dir: str = "experiments",
        max_revisions: int = 1,
        baseline_name: str = "liteagent",
    ):
        self.cache_manager = KVCacheManager(
            max_ram_states=max_ram_states,
            ssd_dir=ssd_dir,
            log_dir=log_dir,
        )
        self.dispatcher = TaskDispatcher(
            router_config_path=router_config_path,
            edge_cache_manager=self.cache_manager,
            workstation_client=workstation_client,
            log_dir=log_dir,
        )
        self.dispatcher.baseline_name = baseline_name
        self.orchestrator = AgentOrchestrator(
            dispatcher=self.dispatcher,
            router_config_path=router_config_path,
            max_revisions=max_revisions,
            log_dir=log_dir,
        )

    def execute_task(self, task, session_id, system_prompt="", temperature=0.0, max_tokens=256):
        return self.orchestrator.execute_task(
            task=task,
            session_id=session_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def build_routing_only_runner(**kwargs) -> LiteAgentRunner:
    """Router and agent pruning active; context caching disabled."""
    runner = LiteAgentRunner(baseline_name="routing_only", **kwargs)
    runner.dispatcher.routing_disabled = False
    runner.dispatcher.cache_disabled = True
    return runner


def build_fixed_tier_runner(tier: str, **kwargs) -> LiteAgentRunner:
    """
    Pins every task to one tier, with the full agent chain and caching active.

    `always-medium` is the comparator that matters: on the capability-labelled
    set it solves 81.9% of solvable tasks at 43% of always-large's cost, so any
    routing policy has to beat that trade rather than merely beat always-large.
    """
    if tier not in ("Small", "Medium", "Large"):
        raise ValueError(f"Unknown tier: {tier}")
    runner = LiteAgentRunner(baseline_name=f"always_{tier.lower()}", **kwargs)
    runner.dispatcher.routing_disabled = True
    runner.dispatcher.cache_disabled = False
    runner.orchestrator.force_tier = tier
    runner.orchestrator.force_agents = ["Planner", "Retriever", "Executor", "Critic"]
    return runner


def build_cascade_runner(**kwargs) -> LiteAgentRunner:
    """
    Starts at the smallest tier and escalates when the Critic rejects.

    Replaces a-priori tier prediction, which does not generalise
    (docs/phase9/router_capability_analysis.md), with observed failure. Solve
    rate is then bounded by the Critic's specificity and cost by its
    sensitivity, neither of which requires predicting difficulty.
    """
    runner = LiteAgentRunner(baseline_name="cascade", **kwargs)
    runner.dispatcher.routing_disabled = True
    runner.dispatcher.cache_disabled = False
    runner.orchestrator.force_tier = "Small"
    runner.orchestrator.force_agents = ["Planner", "Retriever", "Executor", "Critic"]
    runner.orchestrator.escalate_on_reject = True
    return runner


def build_cache_only_runner(**kwargs) -> LiteAgentRunner:
    """
    Caching active; routing disabled.

    Every task is pinned to the Large tier with the full agent chain, which is
    what "no routing" means for a multi-agent workload: no tier selection and
    no pruning.
    """
    runner = LiteAgentRunner(baseline_name="cache_only", **kwargs)
    runner.dispatcher.routing_disabled = True
    runner.dispatcher.cache_disabled = False
    runner.orchestrator.force_tier = "Large"
    runner.orchestrator.force_agents = ["Planner", "Retriever", "Executor", "Critic"]
    return runner
