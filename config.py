import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "agi-hackathon")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "zuansah-munggaran-massachusetts-institute-of-technology")
WANDB_INFERENCE_URL = "https://api.inference.wandb.ai/v1"
WANDB_FULL_PROJECT = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

MODEL = os.getenv("STRUCTURING_MODEL", "OpenPipe/Qwen3-14B-Instruct")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "OpenPipe/Qwen3-14B-Instruct")


def get_client() -> OpenAI:
    """Get an OpenAI client pointing at the W&B inference API."""
    return OpenAI(
        base_url=WANDB_INFERENCE_URL,
        api_key=WANDB_API_KEY,
        project=WANDB_FULL_PROJECT,
    )
