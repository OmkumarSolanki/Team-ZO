"""Weave evaluation pipeline for the multi-agent system.

Creates datasets, runs evaluations, and logs results
to both Weave and W&B for full observability.
"""
from __future__ import annotations

from typing import Any

import wandb
import weave

from .scorers import (
    completeness_scorer,
    latency_scorer,
    orchestration_scorer,
    relevance_scorer,
)


def create_eval_dataset(examples: list[dict[str, Any]]) -> weave.Dataset:
    """Create a Weave dataset for evaluation."""
    return weave.Dataset(name="agent-eval-set", rows=examples)


async def run_evaluation(
    model: weave.Model,
    dataset: weave.Dataset,
    scorers: list | None = None,
) -> dict:
    """Run a full Weave evaluation and log summary to W&B."""
    if scorers is None:
        scorers = [
            relevance_scorer,
            completeness_scorer,
            orchestration_scorer,
            latency_scorer,
        ]

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=scorers,
    )

    results = await evaluation.evaluate(model)

    wandb.log({
        "eval/results": results,
    })

    return results


DEFAULT_EVAL_EXAMPLES = [
    {
        "task": "Research the latest trends in multi-agent AI systems",
        "elements": ["agent orchestration", "communication protocols", "real-world applications"],
    },
    {
        "task": "Analyze the pros and cons of microservices architecture",
        "elements": ["scalability", "complexity", "deployment", "monitoring"],
    },
    {
        "task": "Create a project plan for building a chatbot",
        "elements": ["requirements", "timeline", "tech stack", "testing strategy"],
    },
]
