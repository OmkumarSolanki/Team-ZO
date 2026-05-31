"""Specialist agents that plug into the orchestrator.

Each agent is a weave.Model with full tracing, uses Weave-managed prompts,
and communicates via A2A protocol.
"""
from __future__ import annotations

import json
from typing import Any

import weave
from openai import OpenAI

from .base import AgentCapability, AgentCard, BaseAgent


class LLMAgent(BaseAgent):
    """Agent that calls an LLM with Weave-traced prompts."""
    temperature: float = 0.7

    @weave.op()
    def invoke(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        client = OpenAI()

        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context from other agents:\n{json.dumps(context, indent=2, default=str)}",
            })

        messages.append({"role": "user", "content": task})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

        result = response.choices[0].message.content
        usage = response.usage

        self.log_to_wandb({
            "tokens_prompt": usage.prompt_tokens,
            "tokens_completion": usage.completion_tokens,
            "tokens_total": usage.total_tokens,
        })

        return {
            "result": result,
            "model": self.model_name,
            "tokens_used": usage.total_tokens,
            "status": "success",
        }


def create_research_agent() -> LLMAgent:
    return LLMAgent(
        name="researcher",
        system_prompt="""You are a research agent. Given a topic or question,
provide thorough, well-sourced analysis. Structure your response with:
- Key findings (bullet points)
- Supporting evidence
- Confidence level (high/medium/low)
- Suggested follow-up questions""",
        card=AgentCard(
            name="researcher",
            description="Deep research and information gathering",
            capabilities=[AgentCapability.RESEARCH, AgentCapability.SUMMARIZE],
        ),
    )


def create_analyzer_agent() -> LLMAgent:
    return LLMAgent(
        name="analyzer",
        system_prompt="""You are an analysis agent. Given data or information,
perform deep analysis including:
- Pattern identification
- Strengths and weaknesses
- Quantitative metrics where applicable
- Actionable recommendations""",
        card=AgentCard(
            name="analyzer",
            description="Data and information analysis",
            capabilities=[AgentCapability.ANALYZE, AgentCapability.REVIEW],
        ),
    )


def create_planner_agent() -> LLMAgent:
    return LLMAgent(
        name="planner",
        system_prompt="""You are a planning agent. Break down complex tasks into
actionable subtasks. Output structured plans with:
- Subtasks with priorities
- Dependencies between tasks
- Resource requirements
- Timeline estimates""",
        card=AgentCard(
            name="planner",
            description="Task decomposition and planning",
            capabilities=[AgentCapability.PLAN],
        ),
        temperature=0.3,
    )


def create_validator_agent() -> LLMAgent:
    return LLMAgent(
        name="validator",
        system_prompt="""You are a validation agent. Review outputs for:
- Factual accuracy
- Logical consistency
- Completeness relative to the original task
- Potential issues or gaps
Return a validation report with pass/fail and specific findings.""",
        card=AgentCard(
            name="validator",
            description="Output validation and quality assurance",
            capabilities=[AgentCapability.VALIDATE, AgentCapability.REVIEW],
        ),
        temperature=0.1,
    )


def create_all_agents() -> list[LLMAgent]:
    return [
        create_research_agent(),
        create_analyzer_agent(),
        create_planner_agent(),
        create_validator_agent(),
    ]
