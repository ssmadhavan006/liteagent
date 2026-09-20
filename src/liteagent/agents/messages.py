import time
from dataclasses import dataclass, field


PLAN = "PLAN"
EVIDENCE = "EVIDENCE"
DRAFT = "DRAFT"
CRITIQUE = "CRITIQUE"
REVISION = "REVISION"


@dataclass
class AgentMessage:
    """A single typed message passed from one agent to another."""
    sender: str
    recipient: str
    msg_type: str
    content: str
    seq: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "msg_type": self.msg_type,
            "content": self.content,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class Blackboard:
    """
    Shared state for one task's agent chain.

    Agents never call each other directly: a producer posts a typed message and
    the consumer reads it back by type. This keeps the chain a dataflow graph,
    so a pruned agent simply leaves its message absent and downstream agents
    degrade instead of breaking.
    """

    def __init__(
        self,
        task_id: str,
        benchmark: str,
        prompt: str,
        documents: list[tuple[str, str]] | None = None,
        entry_point: str | None = None,
    ):
        self.task_id = task_id
        self.benchmark = (benchmark or "").lower()
        self.prompt = prompt
        self.documents = documents or []
        self.entry_point = entry_point
        self.messages: list[AgentMessage] = []
        self._seq = 0

    def post(self, sender: str, recipient: str, msg_type: str, content: str, **metadata) -> AgentMessage:
        self._seq += 1
        msg = AgentMessage(
            sender=sender,
            recipient=recipient,
            msg_type=msg_type,
            content=content,
            seq=self._seq,
            metadata=metadata,
        )
        self.messages.append(msg)
        return msg

    def latest(self, msg_type: str) -> AgentMessage | None:
        for msg in reversed(self.messages):
            if msg.msg_type == msg_type:
                return msg
        return None

    def inbox(self, recipient: str) -> list[AgentMessage]:
        return [m for m in self.messages if m.recipient == recipient]

    def content_of(self, msg_type: str, default: str = "") -> str:
        msg = self.latest(msg_type)
        return msg.content if msg else default

    def transcript(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]
