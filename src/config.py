import os
import wandb
import weave
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "agi-hackathon")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def init_all(run_name: str = None, config: dict = None):
    weave.init(WANDB_PROJECT)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
        tags=["multi-agent", "a2a", "hackathon"],
    )
    return run


def init_wandb(run_name: str = None, config: dict = None):
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
    )


def init_weave():
    weave.init(WANDB_PROJECT)
