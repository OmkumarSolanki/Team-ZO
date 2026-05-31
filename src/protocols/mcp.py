"""Model Context Protocol (MCP) integration layer.

Provides tool definitions and context management that agents
can use to interact with external services, all traced via Weave.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import weave


@dataclass
class MCPTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class MCPToolServer:
    """Serves tools to agents via MCP-style interface."""

    def __init__(self):
        self._tools: dict[str, MCPTool] = {}

    def register_tool(self, tool: MCPTool):
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tools.values()]

    @weave.op()
    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        return tool.handler(**arguments)

    def get_tool_names(self) -> list[str]:
        return list(self._tools.keys())
