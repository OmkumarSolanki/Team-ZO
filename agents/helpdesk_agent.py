"""Help Desk Agent — owns the WFM through MCP tools.

Handles search, validation, close-out, and new-fault posting.
All tool calls go through the MCP server and are Weave-traced.
"""
from __future__ import annotations

import json
from typing import Any

import weave
from openai import OpenAI

from config import MODEL
from mcp_servers.wfm_server import MCP_TOOLS, TOOL_DISPATCH


class HelpDeskAgent(weave.Model):
    model_name: str = MODEL

    @weave.op()
    def predict(self, task: str, context: dict | None = None) -> dict:
        return self.execute(task, context)

    @weave.op()
    def execute(self, task: str, context: dict | None = None) -> dict:
        """Execute a help desk task using MCP tools."""
        client = OpenAI()

        system = """You are a help desk agent for a water utility WFM system.
You have access to tools to search, get, validate, and close work requests.
Use the tools to fulfill the task. Always validate before closing a request."""

        messages = [{"role": "system", "content": system}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context: {json.dumps(context, default=str)}",
            })

        messages.append({"role": "user", "content": task})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=MCP_TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )

        tool_results = []
        max_turns = 5
        turn = 0

        while turn < max_turns:
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                result = self._call_tool(fn_name, fn_args)
                tool_results.append({"tool": fn_name, "args": fn_args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=MCP_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            turn += 1

        final_message = response.choices[0].message.content or ""

        return {
            "response": final_message,
            "tool_calls": tool_results,
            "status": "success",
        }

    @weave.op()
    def search(self, text: str) -> list[dict]:
        """Direct search shortcut."""
        from mcp_servers.wfm_server import search_requests
        return search_requests(text)

    @weave.op()
    def close(self, request_id: int, codes: dict) -> dict:
        """Direct close shortcut."""
        from mcp_servers.wfm_server import close_request
        return close_request(request_id, codes)

    @weave.op()
    def validate(self, payload: dict, mode: str = "close_out") -> dict:
        """Direct validate shortcut."""
        from mcp_servers.wfm_server import validate_request
        return validate_request(payload, mode)

    @weave.op()
    def _call_tool(self, name: str, args: dict) -> Any:
        handler = TOOL_DISPATCH.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        return handler(args)
