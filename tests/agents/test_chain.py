import pytest

from liteagent.agents.messages import Blackboard, PLAN, EVIDENCE, DRAFT, CRITIQUE
from liteagent.agents.orchestrator import AgentOrchestrator
from liteagent.agents.roles import (
    CriticAgent,
    ExecutorAgent,
    PlannerAgent,
    RetrieverAgent,
)
from liteagent.router.pruning import map_tier_and_pruning


class RecordingDispatcher:
    """Stands in for TaskDispatcher, recording every agent turn it is asked to run."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def execute_agent_step(self, task_id, prompt, system_prompt, agent_role,
                           session_key, tier, request_id=None,
                           temperature=0.0, max_tokens=128, cache_prefix=None):
        self.calls.append({
            "agent_role": agent_role,
            "session_key": session_key,
            "tier": tier,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "cache_prefix": cache_prefix,
        })
        default = {"Planner": "1. Understand\n2. Compute\n3. Check",
                   "Retriever": "",
                   "Executor": "42",
                   "Critic": "APPROVE"}
        text = self.responses.get(agent_role, default.get(agent_role, ""))
        if callable(text):
            text = text(len([c for c in self.calls if c["agent_role"] == agent_role]))
        return {
            "response_text": text,
            "prefill_tokens": 10,
            "tokens_generated": 5,
            "cache_hit_tier": "MISS",
            "executed_tier": tier,
            "fallback_occurred": False,
        }

    def roles_called(self):
        return [c["agent_role"] for c in self.calls]


def make_orchestrator(dispatcher, **kwargs):
    return AgentOrchestrator(
        dispatcher=dispatcher,
        router_config_path="config/router_config.yaml",
        log_dir=None,
        **kwargs,
    )


# --- message passing primitives -------------------------------------------------

def test_blackboard_routes_messages_by_type_and_recipient():
    bb = Blackboard(task_id="t1", benchmark="gsm8k", prompt="2+2?")
    bb.post("Planner", "Executor", PLAN, "1. add")
    bb.post("Executor", "Critic", DRAFT, "4")

    assert bb.content_of(PLAN) == "1. add"
    assert bb.content_of(DRAFT) == "4"
    assert [m.msg_type for m in bb.inbox("Executor")] == [PLAN]
    assert [m.msg_type for m in bb.inbox("Critic")] == [DRAFT]
    assert [m.seq for m in bb.messages] == [1, 2]


def test_planner_parses_numbered_steps_into_plan_message():
    bb = Blackboard(task_id="t1", benchmark="gsm8k", prompt="q")
    PlannerAgent().consume("1. First step\n2) Second step\n3. Third step", bb)

    plan = bb.latest(PLAN)
    assert plan.recipient == "Executor"
    assert plan.metadata["step_count"] == 3
    assert "First step" in plan.content


def test_executor_prompt_carries_plan_and_evidence_forward():
    bb = Blackboard(task_id="t1", benchmark="hotpotqa", prompt="Who directed it?")
    bb.post("Planner", "Executor", PLAN, "1. find director")
    bb.post("Retriever", "Executor", EVIDENCE, "[Inception]: Nolan directed it.")

    prompt = ExecutorAgent().build_prompt(bb)
    assert "1. find director" in prompt
    assert "Nolan directed it." in prompt


def test_retriever_falls_back_to_lexical_overlap_when_titles_unrecognised():
    docs = [("Inception", "A film directed by Christopher Nolan."),
            ("Baking", "How to bake sourdough bread.")]
    bb = Blackboard(task_id="t1", benchmark="hotpotqa",
                    prompt="Who directed Inception?", documents=docs)

    RetrieverAgent().consume("I could not find anything useful.", bb)

    evidence = bb.latest(EVIDENCE)
    assert evidence.metadata["fallback_used"] is True
    assert "Inception" in evidence.content


def test_critic_defaults_to_approval_on_unparseable_verdict():
    bb = Blackboard(task_id="t1", benchmark="gsm8k", prompt="q")
    bb.post("Executor", "Critic", DRAFT, "42")
    CriticAgent().consume("hmm, unclear", bb)

    assert bb.latest(CRITIQUE).metadata["approved"] is True


# --- pruning is real ------------------------------------------------------------

def test_executor_is_never_pruned_at_any_tier():
    for score in (0.0, 0.5, 1.0):
        _, _, active, pruned = map_tier_and_pruning(score, 0.35, 0.75, "gsm8k")
        assert "Executor" in active
        assert "Executor" not in pruned


def test_retriever_only_active_for_document_benchmarks():
    _, _, active_hotpot, _ = map_tier_and_pruning(0.9, 0.35, 0.75, "hotpotqa")
    _, _, active_gsm, pruned_gsm = map_tier_and_pruning(0.9, 0.35, 0.75, "gsm8k")

    assert "Retriever" in active_hotpot
    assert "Retriever" not in active_gsm
    assert "Retriever" in pruned_gsm


def test_pruned_agents_are_never_invoked():
    """A pruned agent must cost zero model calls, not just be absent from a label."""
    dispatcher = RecordingDispatcher()
    orch = make_orchestrator(dispatcher)

    # A trivial prompt routes Small -> only the Executor survives pruning.
    res = orch.execute_task(
        task={"id": "t1", "prompt": "2+2", "benchmark": "gsm8k"},
        session_id="s1",
    )

    assert res["active_agents"] == ["Executor"]
    assert dispatcher.roles_called() == ["Executor"]
    for pruned_role in res["pruned_agents"]:
        assert pruned_role not in dispatcher.roles_called()


# --- chain execution ------------------------------------------------------------

def test_full_chain_assigns_distinct_cache_session_key_per_agent():
    """Per-role keys are what give the role-priority eviction policy distinct entries."""
    dispatcher = RecordingDispatcher()
    orch = make_orchestrator(dispatcher)

    long_prompt = (
        "Write a Python class implementing a red-black tree with insert, delete, "
        "and rebalance operations, then prove the height invariant holds and "
        "analyse the complexity of each operation in detail. Explain why."
    )
    res = orch.execute_task(
        task={"id": "t2", "prompt": long_prompt, "benchmark": "humaneval"},
        session_id="s2",
    )

    assert res["routed_tier"] == "Large"
    keys = [c["session_key"] for c in dispatcher.calls]
    assert len(keys) == len(set(keys)), "each agent turn needs its own cache key"
    assert all(k.startswith("s2::") for k in keys)


def test_critic_rejection_triggers_bounded_revision():
    dispatcher = RecordingDispatcher(responses={
        "Critic": lambda n: "REVISE: the answer is wrong",
        "Executor": lambda n: f"draft-{n}",
    })
    orch = make_orchestrator(dispatcher, max_revisions=1)
    # The Critic is no longer routed by default, so this exercises the
    # revision path explicitly rather than relying on tier mapping.
    orch.force_tier = "Large"
    orch.force_agents = ["Executor", "Critic"]

    long_prompt = (
        "Write a Python class implementing a red-black tree with insert, delete, "
        "and rebalance operations, then prove the height invariant holds and "
        "analyse the complexity of each operation in detail. Explain why."
    )
    res = orch.execute_task(
        task={"id": "t3", "prompt": long_prompt, "benchmark": "humaneval"},
        session_id="s3",
    )

    executor_calls = [c for c in dispatcher.calls if c["agent_role"] == "Executor"]
    assert res["revisions"] == 1
    assert len(executor_calls) == 2, "one initial draft plus exactly one revision"
    assert "Reviewer feedback" in executor_calls[1]["prompt"]
    assert res["response_text"] == "draft-2"


def test_chain_reports_trace_and_aggregated_costs():
    dispatcher = RecordingDispatcher()
    orch = make_orchestrator(dispatcher)

    long_prompt = (
        "Write a Python class implementing a red-black tree with insert, delete, "
        "and rebalance operations, then prove the height invariant holds and "
        "analyse the complexity of each operation in detail. Explain why."
    )
    res = orch.execute_task(
        task={"id": "t4", "prompt": long_prompt, "benchmark": "humaneval"},
        session_id="s4",
    )

    executed = [t for t in res["agent_trace"] if not t.get("skipped")]
    assert len(executed) == res["model_calls"] == len(dispatcher.calls)
    assert res["prefill_tokens"] == 10 * res["model_calls"]
    assert res["tokens_generated"] == 5 * res["model_calls"]
    assert [t["agent"] for t in executed] == dispatcher.roles_called()


def test_non_executor_step_failure_degrades_instead_of_aborting():
    class FlakyDispatcher(RecordingDispatcher):
        def execute_agent_step(self, **kwargs):
            if kwargs["agent_role"] == "Planner":
                raise RuntimeError("planner unavailable")
            return super().execute_agent_step(**kwargs)

    dispatcher = FlakyDispatcher()
    orch = make_orchestrator(dispatcher)

    long_prompt = (
        "Write a Python class implementing a red-black tree with insert, delete, "
        "and rebalance operations, then prove the height invariant holds and "
        "analyse the complexity of each operation in detail. Explain why."
    )
    res = orch.execute_task(
        task={"id": "t5", "prompt": long_prompt, "benchmark": "humaneval"},
        session_id="s5",
    )

    assert res["response_text"] == "42"
    failed = [t for t in res["agent_trace"] if t.get("reason") == "step_failed"]
    assert [t["agent"] for t in failed] == ["Planner"]


def test_routing_disabled_ablation_pins_tier_and_runs_every_agent():
    """Cache-only ablation: no tier selection, no pruning, chain still real."""
    dispatcher = RecordingDispatcher()
    orch = make_orchestrator(dispatcher)
    orch.force_tier = "Large"
    orch.force_agents = ["Planner", "Retriever", "Executor", "Critic"]

    res = orch.execute_task(
        task={"id": "t7", "prompt": "2+2", "benchmark": "gsm8k"},
        session_id="s7",
    )

    assert res["routed_tier"] == "Large"
    assert res["pruned_agents"] == []
    # Retriever self-skips with no documents, but was not pruned by routing.
    assert dispatcher.roles_called() == ["Planner", "Executor", "Critic"]
    skipped = [t for t in res["agent_trace"] if t.get("reason") == "not_applicable"]
    assert [t["agent"] for t in skipped] == ["Retriever"]


def test_cascade_escalates_tier_on_rejection():
    """Observed failure drives tier choice, replacing a-priori prediction."""
    calls = {"n": 0}

    def critic_then_accept(_n):
        # Reject at Small and Medium, accept once escalated to Large.
        calls["n"] += 1
        return "APPROVE" if calls["n"] >= 3 else "REVISE: wrong"

    dispatcher = RecordingDispatcher(responses={"Critic": critic_then_accept})
    orch = make_orchestrator(dispatcher)
    orch.force_tier = "Small"
    orch.force_agents = ["Executor", "Critic"]
    orch.escalate_on_reject = True

    res = orch.execute_task(
        task={"id": "c1", "prompt": "2+2", "benchmark": "gsm8k"},
        session_id="sc1",
    )

    tiers = [c["tier"] for c in dispatcher.calls if c["agent_role"] == "Executor"]
    assert tiers == ["Small", "Medium", "Large"]
    assert res["escalations"] == 2
    assert res["executed_tier"] == "Large"


def test_cascade_stops_at_top_tier():
    dispatcher = RecordingDispatcher(responses={"Critic": lambda n: "REVISE: still wrong"})
    orch = make_orchestrator(dispatcher)
    orch.force_tier = "Small"
    orch.force_agents = ["Executor", "Critic"]
    orch.escalate_on_reject = True

    res = orch.execute_task(
        task={"id": "c2", "prompt": "2+2", "benchmark": "gsm8k"},
        session_id="sc2",
    )
    assert res["escalations"] == 2, "must not escalate past the largest tier"
    assert res["executed_tier"] == "Large"


def test_escalated_attempt_is_not_anchored_to_rejected_answer():
    """The larger model retries the task, it does not patch a wrong answer."""
    calls = {"n": 0}

    def critic(_n):
        calls["n"] += 1
        return "APPROVE" if calls["n"] >= 2 else "REVISE: wrong"

    dispatcher = RecordingDispatcher(responses={"Critic": critic})
    orch = make_orchestrator(dispatcher)
    orch.force_tier = "Small"
    orch.force_agents = ["Executor", "Critic"]
    orch.escalate_on_reject = True

    orch.execute_task(task={"id": "c3", "prompt": "2+2", "benchmark": "gsm8k"},
                      session_id="sc3")

    escalated = [c for c in dispatcher.calls if c["agent_role"] == "Executor"][1]
    assert "Reviewer feedback" not in escalated["prompt"]


def test_escalated_step_uses_a_distinct_cache_key():
    """KV state is model-specific; tiers must not share a cache entry."""
    calls = {"n": 0}

    def critic(_n):
        calls["n"] += 1
        return "APPROVE" if calls["n"] >= 2 else "REVISE: wrong"

    dispatcher = RecordingDispatcher(responses={"Critic": critic})
    orch = make_orchestrator(dispatcher)
    orch.force_tier = "Small"
    orch.force_agents = ["Executor", "Critic"]
    orch.escalate_on_reject = True

    orch.execute_task(task={"id": "c4", "prompt": "2+2", "benchmark": "gsm8k"},
                      session_id="sc4")

    exec_keys = [c["session_key"] for c in dispatcher.calls if c["agent_role"] == "Executor"]
    assert len(set(exec_keys)) == len(exec_keys)
    assert "Small" in exec_keys[0] and "Medium" in exec_keys[1]


def test_revision_mode_stays_on_tier_when_escalation_disabled():
    dispatcher = RecordingDispatcher(responses={"Critic": lambda n: "REVISE: wrong"})
    orch = make_orchestrator(dispatcher, max_revisions=1)
    orch.force_tier = "Medium"
    orch.force_agents = ["Executor", "Critic"]
    orch.escalate_on_reject = False

    res = orch.execute_task(
        task={"id": "c5", "prompt": "2+2", "benchmark": "gsm8k"},
        session_id="sc5",
    )
    assert res["escalations"] == 0
    assert res["revisions"] == 1
    assert {c["tier"] for c in dispatcher.calls} == {"Medium"}


def test_executor_failure_propagates():
    class BrokenExecutor(RecordingDispatcher):
        def execute_agent_step(self, **kwargs):
            if kwargs["agent_role"] == "Executor":
                raise RuntimeError("executor unavailable")
            return super().execute_agent_step(**kwargs)

    orch = make_orchestrator(BrokenExecutor())
    with pytest.raises(RuntimeError, match="executor unavailable"):
        orch.execute_task(
            task={"id": "t6", "prompt": "2+2", "benchmark": "gsm8k"},
            session_id="s6",
        )


# --- execution-gated escalation (H1') -------------------------------------------

HE_PROMPT = ('def add_one(xs):\n    """Add one.\n'
             '    >>> add_one([1, 2])\n    [2, 3]\n    """\n')


def _exec_orch(dispatcher, **kw):
    orch = make_orchestrator(dispatcher, **kw)
    orch.force_tier = "Small"
    orch.force_agents = ["Executor"]      # no Critic: execution is the verifier
    orch.escalate_on_reject = True
    orch.verifier = "execution"
    return orch


def _he_task():
    return {"id": "h1", "prompt": HE_PROMPT, "benchmark": "humaneval",
            "context": {"entry_point": "add_one"}}


def test_execution_verifier_escalates_on_wrong_code():
    """A draft that fails the prompt's own examples must move up a tier."""
    wrong, right = "    return [x - 1 for x in xs]", "    return [x + 1 for x in xs]"
    calls = {"n": 0}

    def executor(_n):
        calls["n"] += 1
        return wrong if calls["n"] == 1 else right

    dispatcher = RecordingDispatcher(responses={"Executor": executor})
    res = _exec_orch(dispatcher).execute_task(task=_he_task(), session_id="e1")

    tiers = [c["tier"] for c in dispatcher.calls if c["agent_role"] == "Executor"]
    assert tiers == ["Small", "Medium"], "wrong code should have escalated once"
    assert res["escalations"] == 1
    assert res["verdict_sources"][0] == "execution"


def test_execution_verifier_accepts_correct_code_without_escalating():
    dispatcher = RecordingDispatcher(responses={"Executor": lambda n: "    return [x + 1 for x in xs]"})
    res = _exec_orch(dispatcher).execute_task(task=_he_task(), session_id="e2")

    assert res["escalations"] == 0, "correct code must not be escalated"
    assert res["verdict_sources"] == ["execution"]


def test_execution_verdict_costs_no_model_call():
    """Unlike the Critic, verification is free: no Critic turn is dispatched."""
    dispatcher = RecordingDispatcher(responses={"Executor": lambda n: "    return [x + 1 for x in xs]"})
    _exec_orch(dispatcher).execute_task(task=_he_task(), session_id="e3")

    assert "Critic" not in dispatcher.roles_called()


def test_unverifiable_prompt_does_not_count_as_approval():
    """Absence of examples is not evidence of correctness."""
    prompt = 'def f(x):\n    """No examples."""\n'
    dispatcher = RecordingDispatcher(responses={"Executor": lambda n: "    return x"})
    orch = _exec_orch(dispatcher)
    res = orch.execute_task(
        task={"id": "h2", "prompt": prompt, "benchmark": "humaneval",
              "context": {"entry_point": "f"}},
        session_id="e4",
    )
    # No Critic is present either, so the chain stops rather than escalating
    # blindly - but the verdict must not be recorded as an execution pass.
    assert res["verdict_sources"][0] != "execution"
    assert res["escalations"] == 0


def test_execution_verifier_unused_for_non_code_benchmarks():
    dispatcher = RecordingDispatcher(responses={"Executor": lambda n: "42"})
    orch = _exec_orch(dispatcher)
    res = orch.execute_task(
        task={"id": "g1", "prompt": "2+2", "benchmark": "gsm8k"}, session_id="e5")
    assert "execution" not in res["verdict_sources"]
