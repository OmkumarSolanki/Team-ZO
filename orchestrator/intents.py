"""Intent classification for the orchestrator."""
from __future__ import annotations

from enum import Enum

import weave
from openai import OpenAI

from config import MODEL


class Intent(str, Enum):
    CLOSE_OUT = "close_out"
    NEW_FAULT = "new_fault"
    STATUS_CHECK = "status_check"
    UNKNOWN = "unknown"


@weave.op()
def classify_intent(transcript: str) -> Intent:
    """Classify the technician's intent from their transcript."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """Classify the intent of this field technician's report.
Return EXACTLY one of: close_out, new_fault, status_check

- close_out: Technician is reporting completion of a job (fixed, repaired, replaced, isolated, done, finished, completed)
- new_fault: Technician is reporting a new problem (found a leak, spotted damage, new issue)
- status_check: Technician is asking about a job's status (what's the status, any updates, where's the job at)

Return ONLY the intent string, nothing else.""",
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip().lower()
    try:
        return Intent(text)
    except ValueError:
        return Intent.UNKNOWN
