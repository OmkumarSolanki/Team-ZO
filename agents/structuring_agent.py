"""Structuring Agent — the core of the system.

Maps messy technician speech into the controlled vocabulary
of WFM enum codes, with per-field confidence and clarification signals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import weave

from config import MODEL, get_client

CODES_PATH = Path(__file__).parent.parent / "data" / "codes.json"


def _load_codes() -> dict:
    with open(str(CODES_PATH)) as f:
        return json.load(f)


def _build_system_prompt(mode: str = "close_out") -> str:
    codes = _load_codes()

    problem_list = "\n".join(f"  {c['code']}: {c['label']}" for c in codes["problem"])
    cause_list = "\n".join(f"  {c['code']}: {c['label']}" for c in codes["cause"])
    rectify_list = "\n".join(f"  {c['code']}: {c['label']}" for c in codes["rectify"])
    method_list = "\n".join(f"  {c['code']}: {c['label']}" for c in codes["method"])

    if mode == "close_out":
        return f"""You are a structuring agent for a water utility field service system.
Your job is to map a technician's spoken report into structured close-out codes.

You MUST only use codes from these lists:

PROBLEM_CODE (what was wrong):
{problem_list}

CAUSE_CODE (why it happened):
{cause_list}

RECTIFY_CODE (what was done to fix it):
{rectify_list}

METHOD_CODE (how it was fixed):
{method_list}

Return a JSON object with EXACTLY this structure:
{{
  "problem_code": "<code>",
  "cause_code": "<code>",
  "rectify_code": "<code>",
  "method_code": "<code>",
  "req_status": "COMPLETE",
  "field_confidence": {{
    "problem_code": <0.0-1.0>,
    "cause_code": <0.0-1.0>,
    "rectify_code": <0.0-1.0>,
    "method_code": <0.0-1.0>
  }},
  "needs_clarification": null or {{
    "field": "<field_name>",
    "question": "<question to ask the technician>",
    "options": ["<option1>", "<option2>", ...]
  }}
}}

Rules:
- Every code MUST come from the lists above. Never invent codes.
- Set confidence < 0.6 if the transcript is ambiguous for that field.
- If ANY field has confidence < 0.5, set needs_clarification with a specific question.
- Be precise: "leaking" -> P_LKG, "broken" -> P_BKN, "replaced" -> R_RPL, etc.
- Return ONLY valid JSON, no other text."""

    return f"""You are a structuring agent for a water utility field service system.
Your job is to classify a new fault report into structured fields.

Valid REQ_CLASS values:
  HYDRANT_MAINTENANCE, INSPECT, LEAK, MANHOLE_MAINTENANCE,
  METER_FRAME_LEAK, NO_WATER, OTHER, OVERFLOW_SHAFT, PATHTAP_FAULT,
  RESTORE_SUPPLY_AFTER_RESTRICTION, SEWER_MANHOLE_OVERFLOWING,
  SEWER_SHAFT_HOLDING, START_DAY, WATER_QUALITY_DIRTY_WATER

Valid PRIORITY values (SLA windows):
  P1_URG_1H_4H (urgent, 1-4 hours)
  P2_HIGH_6H_1D (high, 6 hours to 1 day)
  P3_MED_2D_5D (medium, 2-5 days)
  P4_PLAN_10D (planned, 10 days)
  P8_PLAN_12M (planned, 12 months)
  P10_PLAN_7D (planned, 7 days)

Valid SEVERITY: LOW, MEDIUM, HIGH

Valid USER_DEF21 (location type):
  DRIVEWAY, FOOTPATH, FOOTPATH_CONCRETE, FOOTPATH_GRASS,
  PARK_RESERVE, ROAD, YARD

Return a JSON object:
{{
  "req_class": "<value>",
  "priority": "<value>",
  "severity": "<value>",
  "user_def21": "<value>",
  "cust_prob_descr": "<cleaned description>",
  "field_confidence": {{
    "req_class": <0.0-1.0>,
    "priority": <0.0-1.0>,
    "severity": <0.0-1.0>,
    "user_def21": <0.0-1.0>
  }},
  "needs_clarification": null or {{
    "field": "<field_name>",
    "question": "<question>",
    "options": ["<opt1>", "<opt2>"]
  }}
}}

Rules:
- Gushing/flooding/burst = P1_URG + HIGH severity
- Steady flow/visible leak = P2_HIGH or P3_MED + MEDIUM severity
- Seeping/damp = P3_MED or P4_PLAN + LOW/MEDIUM severity
- If severity and priority seem contradictory, flag needs_clarification.
- Return ONLY valid JSON."""


class StructuringAgent(weave.Model):
    model_name: str = MODEL
    mode: str = "close_out"

    @weave.op()
    def predict(self, transcript: str, context: dict | None = None) -> dict:
        return self.structure(transcript, context)

    @weave.op()
    def structure(self, transcript: str, context: dict | None = None) -> dict:
        client = get_client()
        system_prompt = _build_system_prompt(self.mode)

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Additional context: {json.dumps(context, default=str)}",
            })

        messages.append({
            "role": "user",
            "content": f"Technician report:\n{transcript}",
        })

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        result["_tokens"] = response.usage.total_tokens
        result["_model"] = self.model_name

        return result
