from __future__ import annotations

from typing import Any

import weave

from .base import AgentCapability, BaseAgent


class AgentRegistry:
    """Central registry for agent discovery (A2A pattern)."""

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> str:
        agent_id = agent.card.agent_id
        self._agents[agent_id] = agent
        return agent_id

    def discover(self, capability: AgentCapability) -> list[BaseAgent]:
        return [
            agent for agent in self._agents.values()
            if capability in agent.card.capabilities
        ]

    def get(self, agent_id: str) -> BaseAgent | None:
        return self._agents.get(agent_id)

    def list_all(self) -> list[dict[str, Any]]:
        return [agent.card.to_dict() for agent in self._agents.values()]

    @weave.op()
    def route_task(self, task: str, required_capability: AgentCapability) -> BaseAgent | None:
        candidates = self.discover(required_capability)
        if not candidates:
            return None
        return candidates[0]
