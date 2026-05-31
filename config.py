import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("STRUCTURING_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "agi-hackathon")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
