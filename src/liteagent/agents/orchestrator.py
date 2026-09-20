import time
import uuid
import logging

from liteagent.agents.messages import Blackboard, DRAFT, CRITIQUE
from liteagent.agents.roles import AGENT_REGISTRY, CHAIN_ORDER, ExecutorAgent
from liteagent.router.router import route_task

logger = logging.getLogger("liteagent.agents.orchestrator")


class AgentOrchestrator:
    """
    Runs the Planner -> Retriever -> Executor -> Critic chain for one task.

    Routing happens once per task: the complexity score picks a model tier and
    the set of active agents. Pruned agents are genuinely never invoked, so the
    cost saved is a real saved model call rather than an accounting artifact.

    Each agent gets its own cache session key, which is what gives the
    role-priority eviction policy distinct entries to choose between.
    """

    def __init__(
        self,
        dispatcher,
        router_config_path: str = "config/router_config.yaml",
        max_revisions: int = 1,
        log_dir: str = "experiments",
    ):
        self.dispatcher = dispatcher
        self.router_config_path = router_config_path
        self.max_revisions = max_revisions
        self.log_dir = log_dir
        # Set by the routing-disabled ablation to pin tier/agent selection.
        self.force_tier = None
        self.force_agents = None

    def _build_blackboard(self, task: dict) -> Blackboard:
        context = task.get("context") or {}
        documents = context.get("documents") or []
        # Accept both [(title, body)] and pre-joined strings.
        normalised = []
        for doc in documents:
            if isinstance(doc, (list, tuple)) and len(doc) == 2:
                title, body = doc
                normalised.append((str(title), body if isinstance(body, str) else "".join(body)))
            else:
                normalised.append((str(doc)[:40], str(doc)))

        # When documents are supplied the Retriever provides the evidence, so the
        # chain works from the bare question rather than the pre-inlined context.
        prompt = context.get("question") if normalised else None

        return Blackboard(
            task_id=str(task.get("id", "task")),
            benchmark=task.get("benchmark", ""),
            prompt=prompt or task.get("prompt", ""),
            documents=normalised,
            entry_point=context.get("entry_point"),
        )

    def execute_task(
        self,
        task: dict,
        session_id: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> dict:
        chain_start = time.time()
        request_id = str(uuid.uuid4())

        if self.force_tier or self.force_agents:
            # Routing-disabled ablation: no tier selection and no pruning.
            tier = self.force_tier or "Large"
            active = list(self.force_agents or CHAIN_ORDER)
            pruned = [r for r in CHAIN_ORDER if r not in active]
            route = {"model_tier": tier, "active_agents": active, "pruned_agents": pruned}
        else:
            route = route_task(task, self.router_config_path, log_dir=self.log_dir)
            tier = route["model_tier"]
            active = route["active_agents"]
            pruned = route["pruned_agents"]

        bb = self._build_blackboard(task)
        trace: list[dict] = []
        totals = {"prefill_tokens": 0, "tokens_generated": 0, "model_calls": 0}
        cache_tiers: list[str] = []

        def run_agent(agent, revision_pass: bool = False) -> bool:
            if not agent.should_run(bb):
                trace.append({
                    "agent": agent.role,
                    "skipped": True,
                    "reason": "not_applicable",
                })
                return False

            step_tokens = min(agent.max_tokens, max_tokens) if agent.role == "Executor" else agent.max_tokens
            step_start = time.time()
            try:
                res = self.dispatcher.execute_agent_step(
                    task_id=bb.task_id,
                    prompt=agent.build_prompt(bb),
                    system_prompt=agent.system_prompt(bb),
                    agent_role=agent.role,
                    session_key=f"{session_id}::{agent.role}",
                    tier=tier,
                    request_id=request_id,
                    temperature=temperature,
                    max_tokens=step_tokens,
                )
            except Exception as exc:
                logger.warning("Agent step %s failed: %s", agent.role, exc)
                trace.append({
                    "agent": agent.role,
                    "skipped": True,
                    "reason": "step_failed",
                    "error": str(exc),
                })
                # Only the Executor is load-bearing; the rest degrade silently.
                if agent.role == "Executor":
                    raise
                return False

            agent.consume(res.get("response_text", ""), bb)

            totals["prefill_tokens"] += res.get("prefill_tokens", 0)
            totals["tokens_generated"] += res.get("tokens_generated", 0)
            totals["model_calls"] += 1
            cache_tiers.append(res.get("cache_hit_tier", "NONE"))

            trace.append({
                "agent": agent.role,
                "model_tier": res.get("executed_tier", tier),
                "revision_pass": revision_pass,
                "latency_ms": round((time.time() - step_start) * 1000.0, 2),
                "prefill_tokens": res.get("prefill_tokens", 0),
                "tokens_generated": res.get("tokens_generated", 0),
                "cache_hit_tier": res.get("cache_hit_tier", "NONE"),
                "fallback_occurred": res.get("fallback_occurred", False),
            })
            return True

        agents = {role: AGENT_REGISTRY[role]() for role in CHAIN_ORDER if role in active}
        if "Executor" in agents:
            # system_prompt carries the harness output contract, issued
            # identically to the single-shot baselines.
            agents["Executor"] = ExecutorAgent(
                max_tokens=max_tokens, output_contract=system_prompt
            )

        for role in CHAIN_ORDER:
            if role in agents:
                run_agent(agents[role])

        # Bounded revision loop: only when an active Critic actually rejected.
        revisions = 0
        while revisions < self.max_revisions and "Critic" in agents and "Executor" in agents:
            critique = bb.latest(CRITIQUE)
            if critique is None or critique.metadata.get("approved", True):
                break
            revisions += 1
            if not run_agent(agents["Executor"], revision_pass=True):
                break
            run_agent(agents["Critic"], revision_pass=True)

        final_answer = bb.content_of(DRAFT)
        if not final_answer:
            raise RuntimeError("Agent chain produced no Executor draft")

        return {
            "response_text": final_answer,
            "tokens_generated": totals["tokens_generated"],
            "prefill_tokens": totals["prefill_tokens"],
            "latency_ms": round((time.time() - chain_start) * 1000.0, 2),
            "cache_hit_tier": cache_tiers[0] if cache_tiers else "NONE",
            "cache_hit_tiers": cache_tiers,
            "routed_tier": tier,
            "executed_tier": tier,
            "fallback_occurred": any(t.get("fallback_occurred") for t in trace),
            "request_id": request_id,
            "agent_trace": trace,
            "active_agents": active,
            "pruned_agents": pruned,
            "model_calls": totals["model_calls"],
            "revisions": revisions,
            "message_transcript": bb.transcript(),
        }
