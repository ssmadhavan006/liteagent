import time
import uuid
import logging

from liteagent.agents.messages import Blackboard, DRAFT, CRITIQUE
from liteagent.agents.roles import AGENT_REGISTRY, CHAIN_ORDER, ExecutorAgent
from liteagent.router.router import route_task

logger = logging.getLogger("liteagent.agents.orchestrator")

# Cheapest to most capable. Escalation walks up this ladder.
TIER_LADDER = ["Small", "Medium", "Large"]


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
        # Cascade mode: on a Critic rejection, retry on the next larger tier
        # instead of asking the same model again. Difficulty is not predictable
        # from the prompt (docs/phase9/router_capability_analysis.md), so the
        # cascade observes failure rather than trying to anticipate it.
        self.escalate_on_reject = False
        self.max_escalations = 2

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
        prompt = prompt or task.get("prompt", "")

        # The harness hands the prompt over with the few-shot prefix already
        # attached. Split it back out so each agent can emit it first and the
        # cache has a span that repeats across tasks.
        shared_prefix = context.get("few_shot_prefix", "") or ""
        if shared_prefix and prompt.startswith(shared_prefix):
            prompt = prompt[len(shared_prefix):]

        return Blackboard(
            task_id=str(task.get("id", "task")),
            benchmark=task.get("benchmark", ""),
            prompt=prompt,
            documents=normalised,
            entry_point=context.get("entry_point"),
            shared_prefix=shared_prefix,
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
            # Reusable across every task for this role: the role's system prompt
            # plus the benchmark's few-shot exemplars. Caching this prefix is what
            # gives the KV hierarchy something to restore.
            agent_system = agent.system_prompt(bb)
            reusable_prefix = f"{agent_system}\n{bb.shared_prefix}" if bb.shared_prefix else None
            try:
                res = self.dispatcher.execute_agent_step(
                    task_id=bb.task_id,
                    prompt=agent.build_prompt(bb),
                    system_prompt=agent_system,
                    cache_prefix=reusable_prefix,
                    agent_role=agent.role,
                    # Tier is part of the key: KV state is model-specific, so an
                    # escalated step must not collide with the smaller model's
                    # entry for the same role.
                    session_key=f"{session_id}::{tier}::{agent.role}",
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

            totals.setdefault("ttft_ms", []).append(res.get("ttft_ms", 0.0))
            totals["cached_prefix_tokens"] = totals.get("cached_prefix_tokens", 0) + \
                res.get("cached_prefix_tokens", 0)

            trace.append({
                "agent": agent.role,
                "model_tier": res.get("executed_tier", tier),
                "revision_pass": revision_pass,
                "latency_ms": round((time.time() - step_start) * 1000.0, 2),
                "ttft_ms": round(res.get("ttft_ms", 0.0), 2),
                "prefill_tokens": res.get("prefill_tokens", 0),
                "cached_prefix_tokens": res.get("cached_prefix_tokens", 0),
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

        # A Critic rejection either retries on the same tier (revision) or moves
        # up the ladder (cascade). `tier` is read by run_agent from this scope,
        # so reassigning it here redirects subsequent steps to the new tier.
        revisions = 0
        escalations = 0
        while "Critic" in agents and "Executor" in agents:
            critique = bb.latest(CRITIQUE)
            if critique is None or critique.metadata.get("approved", True):
                break

            if self.escalate_on_reject:
                position = TIER_LADDER.index(tier) if tier in TIER_LADDER else len(TIER_LADDER) - 1
                if position + 1 >= len(TIER_LADDER) or escalations >= self.max_escalations:
                    break
                tier = TIER_LADDER[position + 1]
                escalations += 1
                # Fresh attempt: the larger model should not be anchored to the
                # smaller model's rejected answer.
                agents["Executor"].include_feedback = False
            else:
                if revisions >= self.max_revisions:
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
            # TTFT of the task is the first turn's: that is when the user would
            # see output, and it is the span a restored context shortens.
            "ttft_ms": round(totals.get("ttft_ms", [0.0])[0], 2) if totals.get("ttft_ms") else 0.0,
            "cached_prefix_tokens": totals.get("cached_prefix_tokens", 0),
            "routed_tier": route.get("model_tier", tier),
            "executed_tier": tier,
            "fallback_occurred": any(t.get("fallback_occurred") for t in trace),
            "request_id": request_id,
            "agent_trace": trace,
            "active_agents": active,
            "pruned_agents": pruned,
            "model_calls": totals["model_calls"],
            "revisions": revisions,
            "escalations": escalations,
            "message_transcript": bb.transcript(),
        }
