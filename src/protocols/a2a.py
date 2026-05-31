"""Agent-to-Agent (A2A) protocol implementation.

Defines the message format and communication patterns
for inter-agent communication, traced by Weave.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import weave
import wandb


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    DELEGATE = "delegate"
    BROADCAST = "broadcast"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class A2AMessage:
    sender: str
    receiver: str
    message_type: MessageType
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


@dataclass
class TaskEnvelope:
    """Wraps a task as it flows through the multi-agent pipeline."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    messages: list[A2AMessage] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    def add_trace(self, agent_name: str, action: str):
        entry = f"[{datetime.now(timezone.utc).isoformat()}] {agent_name}: {action}"
        self.trace.append(entry)

    def to_wandb_table_row(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "num_messages": len(self.messages),
            "trace_steps": len(self.trace),
            "trace_log": "\n".join(self.trace),
        }


class A2ARouter:
    """Routes messages between agents, logging everything to Weave + W&B."""

    def __init__(self):
        self._message_log: list[A2AMessage] = []
        self._task_table = wandb.Table(
            columns=["task_id", "description", "status", "num_messages", "trace_steps", "trace_log"]
        )

    @weave.op()
    def send(self, message: A2AMessage, registry) -> A2AMessage | None:
        self._message_log.append(message)
        wandb.log({
            "a2a/messages_sent": len(self._message_log),
            "a2a/message_type": message.message_type.value,
        })

        target_agent = registry.get(message.receiver)
        if target_agent is None:
            return A2AMessage(
                sender="router",
                receiver=message.sender,
                message_type=MessageType.ERROR,
                payload={"error": f"Agent {message.receiver} not found"},
                correlation_id=message.message_id,
            )

        return target_agent.handle_message(message)

    @weave.op()
    def broadcast(self, message: A2AMessage, registry) -> list[A2AMessage]:
        responses = []
        for agent_info in registry.list_all():
            if agent_info["agent_id"] != message.sender:
                msg = A2AMessage(
                    sender=message.sender,
                    receiver=agent_info["agent_id"],
                    message_type=MessageType.BROADCAST,
                    payload=message.payload,
                    correlation_id=message.message_id,
                )
                resp = self.send(msg, registry)
                if resp:
                    responses.append(resp)
        return responses

    def log_task(self, envelope: TaskEnvelope):
        self._task_table.add_data(*envelope.to_wandb_table_row().values())
        wandb.log({"a2a/task_pipeline": self._task_table})

    def get_message_log(self) -> list[dict]:
        return [m.to_dict() for m in self._message_log]
