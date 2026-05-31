"""Weave-native scorers for evaluating multi-agent outputs.

These scorers run as part of weave.Evaluation to produce
metrics visible in the Weave dashboard.
"""
from __future__ import annotations

import json
from typing import Any

import weave
from openai import OpenAI


@weave.op()
def relevance_scorer(output: dict, expected: dict) -> dict:
    """Scores whether the agent output is relevant to the task."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Score relevance from 0.0 to 1.0. Return JSON: {\"score\": float, \"reason\": str}"},
            {"role": "user", "content": f"Task: {expected.get('task', '')}\nOutput: {output.get('result', '')}\nScore relevance:"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


@weave.op()
def completeness_scorer(output: dict, expected: dict) -> dict:
    """Scores whether the output fully addresses all parts of the task."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Score completeness from 0.0 to 1.0. Return JSON: {\"score\": float, \"missing\": list[str]}"},
            {"role": "user", "content": f"Task: {expected.get('task', '')}\nExpected elements: {expected.get('elements', [])}\nOutput: {output.get('result', '')}\nScore completeness:"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


@weave.op()
def orchestration_scorer(output: dict, expected: dict) -> dict:
    """Scores the quality of multi-agent orchestration."""
    trace = output.get("trace", [])
    agents_used = set()
    for entry in trace:
        parts = entry.split("] ")
        if len(parts) > 1:
            agent_name = parts[1].split(":")[0].strip()
            agents_used.add(agent_name)

    num_agents = len(agents_used)
    num_steps = len(trace)
    has_delegation = any("delegate" in str(e).lower() for e in trace)
    has_validation = any("validat" in str(e).lower() for e in trace)

    score = min(1.0, (
        (0.3 if num_agents >= 2 else 0.1 * num_agents) +
        (0.2 if num_steps >= 3 else 0.1) +
        (0.25 if has_delegation else 0.0) +
        (0.25 if has_validation else 0.0)
    ))

    return {
        "score": score,
        "agents_used": list(agents_used),
        "num_steps": num_steps,
        "has_delegation": has_delegation,
        "has_validation": has_validation,
    }


@weave.op()
def latency_scorer(output: dict, expected: dict) -> dict:
    """Scores based on execution time."""
    elapsed = output.get("elapsed_seconds", 0)
    if elapsed <= 5:
        score = 1.0
    elif elapsed <= 15:
        score = 0.7
    elif elapsed <= 30:
        score = 0.4
    else:
        score = 0.2
    return {"score": score, "elapsed_seconds": elapsed}
