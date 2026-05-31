"""Team-ZO: Multi-Agent Orchestration System

Entry point for the AGI Hackathon project.
Uses LangGraph + A2A protocol with full W&B + Weave observability.
"""
import asyncio
import sys

import wandb

from src.config import init_all
from src.agents.specialist import create_all_agents
from src.evaluation.pipeline import (
    DEFAULT_EVAL_EXAMPLES,
    create_eval_dataset,
    run_evaluation,
)
from src.orchestrator import MultiAgentOrchestrator
from src.prompts import publish_all_prompts


def build_orchestrator() -> MultiAgentOrchestrator:
    orchestrator = MultiAgentOrchestrator()
    agents = create_all_agents()
    for agent in agents:
        agent_id = orchestrator.register_agent(agent)
        print(f"  Registered: {agent.name} ({agent_id})")
    return orchestrator


def run_task(orchestrator: MultiAgentOrchestrator, task: str) -> dict:
    print(f"\nTask: {task}")
    print("-" * 60)
    result = orchestrator.predict(task)
    print(f"Status: {result['status']}")
    print(f"Time: {result['elapsed_seconds']:.2f}s")
    print(f"Steps: {len(result['trace'])}")
    for step in result["trace"]:
        print(f"  {step}")
    return result


async def run_eval(orchestrator: MultiAgentOrchestrator):
    print("\nRunning Weave evaluation...")
    print("=" * 60)
    dataset = create_eval_dataset(DEFAULT_EVAL_EXAMPLES)
    results = await run_evaluation(orchestrator, dataset)
    print(f"Evaluation complete: {results}")
    return results


def main():
    print("=" * 60)
    print("Team-ZO Multi-Agent Orchestration System")
    print("AGI House Hackathon - May 31, 2026")
    print("=" * 60)

    run = init_all(
        run_name="team-zo-orchestration",
        config={
            "architecture": "langgraph+a2a",
            "agents": ["researcher", "analyzer", "planner", "validator"],
            "protocols": ["a2a", "mcp"],
            "frameworks": ["langchain", "llama-index"],
        },
    )

    print("\nPublishing prompts to Weave...")
    publish_all_prompts()

    print("\nBuilding orchestrator...")
    orchestrator = build_orchestrator()

    wandb.log({
        "system/registered_agents": len(orchestrator.registry.list_all()),
        "system/agent_cards": orchestrator.registry.list_all(),
    })

    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        asyncio.run(run_eval(orchestrator))
    elif len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        run_task(orchestrator, task)
    else:
        task = "Research and analyze the current state of multi-agent AI systems, including orchestration patterns, communication protocols, and real-world applications."
        run_task(orchestrator, task)

    run.finish()
    print("\nDone! Check W&B and Weave dashboards for full traces.")


if __name__ == "__main__":
    main()
