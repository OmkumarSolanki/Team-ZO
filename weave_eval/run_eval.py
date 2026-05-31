"""Run Weave evaluation on the structuring agent.

Usage:
  python weave_eval/run_eval.py                  # close-out eval, 20 samples
  python weave_eval/run_eval.py --mode new_fault # new-fault eval
  python weave_eval/run_eval.py --limit 50       # more samples
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
import weave

from agents.structuring_agent import StructuringAgent
from config import WANDB_ENTITY, WANDB_PROJECT
from weave_eval.build_dataset import get_weave_dataset
from weave_eval.scorers import PriorityJudge, enum_match


class StructuringModel(weave.Model):
    """Wrapper that matches Weave evaluation interface."""
    agent: StructuringAgent

    @weave.op()
    def predict(self, transcript: str, **kwargs) -> dict:
        return self.agent.structure(transcript)


async def run(mode: str = "close_out", limit: int = 20):
    weave.init(WANDB_PROJECT)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"eval-{mode}-{limit}",
        tags=["eval", mode],
    )

    agent = StructuringAgent(mode=mode)
    model = StructuringModel(agent=agent)

    dataset = get_weave_dataset(mode=mode, limit=limit)
    print(f"Dataset: {len(dataset.rows)} examples ({mode})")

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[enum_match, PriorityJudge()],
    )

    results = await evaluation.evaluate(model)
    print(f"\nResults: {results}")

    wandb.log({"eval/results_summary": str(results)})
    run.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="close_out", choices=["close_out", "new_fault"])
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(run(args.mode, args.limit))
