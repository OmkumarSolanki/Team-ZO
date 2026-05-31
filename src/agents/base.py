from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import weave
import wandb


class AgentCapability(str, Enum):
    RESEARCH = "research"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    CODE = "code"
    REVIEW = "review"
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"


@dataclass
class AgentCard:
    """A2A-style agent card describing capabilities and metadata."""
    name: str
    description: str
    capabilities: list[AgentCapability]
    version: str = "1.0"
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "version": self.version,
            "metadata": self.metadata,
        }


class BaseAgent(weave.Model):
    """Base agent with Weave tracing and W&B logging built in."""
    name: str
    system_prompt: str
    card: AgentCard = None
    model_name: str = "gpt-4o-mini"

    def model_post_init(self, __context: Any) -> None:
        if self.card is None:
            self.card = AgentCard(
                name=self.name,
                description=f"Agent: {self.name}",
                capabilities=[],
            )

    @weave.op()
    def invoke(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement invoke()")

    @weave.op()
    def handle_message(self, message: "A2AMessage") -> "A2AMessage":
        from src.protocols.a2a import A2AMessage, MessageType
        result = self.invoke(message.payload.get("task", ""), message.payload)
        return A2AMessage(
            sender=self.card.agent_id,
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            payload=result,
            correlation_id=message.message_id,
        )

    def log_to_wandb(self, metrics: dict[str, Any]):
        wandb.log({f"{self.name}/{k}": v for k, v in metrics.items()})
