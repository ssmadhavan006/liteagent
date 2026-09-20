import re

from liteagent.agents.messages import Blackboard, PLAN, EVIDENCE, DRAFT, CRITIQUE


class Agent:
    """
    Base class for a chain participant.

    Each agent turns the blackboard into one prompt, consumes one model
    response, and posts one typed message back. Agents hold no state between
    calls; everything shared lives on the blackboard.
    """

    role = "Agent"
    max_tokens = 128

    def system_prompt(self, bb: Blackboard) -> str:
        raise NotImplementedError

    def build_prompt(self, bb: Blackboard) -> str:
        raise NotImplementedError

    def consume(self, raw: str, bb: Blackboard) -> None:
        raise NotImplementedError

    def should_run(self, bb: Blackboard) -> bool:
        return True


class PlannerAgent(Agent):
    role = "Planner"
    max_tokens = 160

    def system_prompt(self, bb: Blackboard) -> str:
        return (
            "You are the Planner. Break the task into at most three short, concrete steps. "
            "Write one step per line, numbered. Do not solve the task."
        )

    def build_prompt(self, bb: Blackboard) -> str:
        return f"Task:\n{bb.prompt}\n\nSteps:"

    def consume(self, raw: str, bb: Blackboard) -> None:
        steps = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^\(?(\d+)[.)\]]?\s+(.*)$", line)
            if match:
                steps.append(match.group(2).strip())
            elif line.startswith(("-", "*")):
                steps.append(line.lstrip("-* ").strip())
            if len(steps) >= 3:
                break

        if not steps:
            steps = [s.strip() for s in raw.strip().split(".") if s.strip()][:3]

        content = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        bb.post(self.role, "Executor", PLAN, content, step_count=len(steps))


class RetrieverAgent(Agent):
    role = "Retriever"
    max_tokens = 96

    def should_run(self, bb: Blackboard) -> bool:
        # Retrieval is only meaningful when candidate documents were supplied.
        return bool(bb.documents)

    def system_prompt(self, bb: Blackboard) -> str:
        return (
            "You are the Retriever. Choose the documents needed to answer the question. "
            "Reply with only their titles, one per line. Do not answer the question."
        )

    def build_prompt(self, bb: Blackboard) -> str:
        titles = "\n".join(f"- {title}" for title, _ in bb.documents)
        plan = bb.content_of(PLAN)
        plan_block = f"\nPlan:\n{plan}\n" if plan else ""
        return (
            f"Question:\n{bb.prompt}\n{plan_block}\n"
            f"Available documents:\n{titles}\n\nMost relevant titles:"
        )

    def consume(self, raw: str, bb: Blackboard) -> None:
        selected = self._match_titles(raw, bb)
        if not selected:
            selected = self._keyword_fallback(bb)

        evidence = "\n\n".join(f"[{title}]: {body}" for title, body in selected)
        bb.post(
            self.role,
            "Executor",
            EVIDENCE,
            evidence,
            selected_titles=[t for t, _ in selected],
            fallback_used=not self._match_titles(raw, bb),
        )

    def _match_titles(self, raw: str, bb: Blackboard) -> list[tuple[str, str]]:
        lowered = raw.lower()
        matched = [(title, body) for title, body in bb.documents if title.lower() in lowered]
        return matched[:2]

    def _keyword_fallback(self, bb: Blackboard) -> list[tuple[str, str]]:
        """If the model named nothing recognisable, fall back to lexical overlap."""
        question_terms = set(re.findall(r"\w+", bb.prompt.lower()))
        scored = []
        for title, body in bb.documents:
            body_terms = set(re.findall(r"\w+", f"{title} {body}".lower()))
            scored.append((len(question_terms & body_terms), title, body))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(title, body) for _, title, body in scored[:2]]


class ExecutorAgent(Agent):
    role = "Executor"

    def __init__(self, max_tokens: int = 256, output_contract: str = ""):
        self.max_tokens = max_tokens
        # Supplied by the harness and identical for every system under test, so
        # the chain gains no scoring advantage from knowing the answer format.
        self.output_contract = output_contract

    def system_prompt(self, bb: Blackboard) -> str:
        if self.output_contract:
            return f"You are the Executor. {self.output_contract}"
        # Standalone use outside the harness: fall back to an equivalent contract.
        if bb.benchmark == "humaneval":
            return (
                "You are the Executor. Complete the Python function. "
                "Reply with code only, no prose and no markdown fences."
            )
        if bb.benchmark == "gsm8k":
            return (
                "You are the Executor. Solve the problem step by step, "
                "then give the final numeric answer on its own last line after '#### '."
            )
        return (
            "You are the Executor. Answer using only the supplied evidence. "
            "Reply with the shortest exact answer span, no explanation."
        )

    def build_prompt(self, bb: Blackboard) -> str:
        sections = []
        evidence = bb.content_of(EVIDENCE)
        if evidence:
            sections.append(f"Evidence:\n{evidence}")
        plan = bb.content_of(PLAN)
        if plan:
            sections.append(f"Plan:\n{plan}")
        sections.append(f"Task:\n{bb.prompt}")

        critique = bb.latest(CRITIQUE)
        if critique is not None and not critique.metadata.get("approved", True):
            previous = bb.content_of(DRAFT)
            sections.append(
                f"Your previous attempt:\n{previous}\n\n"
                f"Reviewer feedback:\n{critique.content}\n\nProduce a corrected answer."
            )

        return "\n\n".join(sections) + "\n\nAnswer:"

    def consume(self, raw: str, bb: Blackboard) -> None:
        revision = bb.latest(CRITIQUE) is not None
        bb.post(self.role, "Critic", DRAFT, raw.strip(), revision=revision)


class CriticAgent(Agent):
    role = "Critic"
    max_tokens = 96

    def system_prompt(self, bb: Blackboard) -> str:
        return (
            "You are the Critic. Check the answer against the task. "
            "Reply with exactly 'APPROVE' if it is correct and well-formed, "
            "otherwise reply 'REVISE:' followed by one sentence saying what is wrong."
        )

    def build_prompt(self, bb: Blackboard) -> str:
        return (
            f"Task:\n{bb.prompt}\n\n"
            f"Proposed answer:\n{bb.content_of(DRAFT)}\n\nVerdict:"
        )

    def consume(self, raw: str, bb: Blackboard) -> None:
        text = raw.strip()
        upper = text.upper()
        # Default to approval: an unparseable verdict must not discard a good draft.
        approved = "REVISE" not in upper or upper.startswith("APPROVE")
        feedback = ""
        if not approved:
            _, _, tail = text.partition(":")
            feedback = (tail or text).strip()

        bb.post(
            self.role,
            "Executor",
            CRITIQUE,
            feedback or "APPROVE",
            approved=approved,
        )


AGENT_REGISTRY = {
    "Planner": PlannerAgent,
    "Retriever": RetrieverAgent,
    "Executor": ExecutorAgent,
    "Critic": CriticAgent,
}

CHAIN_ORDER = ["Planner", "Retriever", "Executor", "Critic"]
