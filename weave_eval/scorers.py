"""Weave scorers for evaluating the multi-agent system.

1. enum_match — exact match on SLA-bearing fields
2. PriorityJudge — LLM-as-judge for priority justification
3. consistency_guard — live guardrail blocking bad combos
4. latency_scorer — execution time
"""
from __future__ import annotations

import json

import weave
from openai import OpenAI

from config import JUDGE_MODEL


@weave.op()
def enum_match(output: dict, expected: dict) -> dict:
    """Exact-match on the structured fields."""
    if "problem_code" in expected:
        fields = ["problem_code", "cause_code", "rectify_code", "method_code"]
    else:
        fields = ["req_class", "priority", "severity", "user_def21"]

    hits = {}
    for f in fields:
        out_val = str(output.get(f, "")).upper()
        exp_val = str(expected.get(f, "")).upper()
        hits[f] = out_val == exp_val

    return {
        "all_correct": all(hits.values()),
        "fields_correct": sum(hits.values()),
        "fields_total": len(hits),
        "priority_correct": hits.get("priority", hits.get("problem_code", False)),
        **hits,
    }


class PriorityJudge(weave.Scorer):
    model_id: str = JUDGE_MODEL

    @weave.op()
    def score(self, output: dict, expected: dict) -> dict:
        """LLM-as-judge: is the priority justified by the transcript?"""
        client = OpenAI()
        transcript = expected.get("transcript", expected.get("cust_prob_descr", ""))
        assigned_priority = output.get("priority", output.get("problem_code", ""))

        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": """You are evaluating whether a priority/code assignment is justified by the technician's report.
Return JSON: {"justified": true/false, "reason": "<brief explanation>"}""",
                },
                {
                    "role": "user",
                    "content": f"Transcript: {transcript}\nAssigned: {assigned_priority}\nIs this justified?",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)


@weave.op()
def consistency_guard(output: dict) -> dict:
    """Guardrail that blocks impossible field combinations."""
    issues = []

    if output.get("priority", "").startswith("P1") and output.get("severity") == "LOW":
        issues.append("P1 urgent but LOW severity")

    if output.get("req_status") == "COMPLETE":
        for code in ["problem_code", "cause_code", "rectify_code", "method_code"]:
            if not output.get(code):
                issues.append(f"COMPLETE status but missing {code}")

    return {
        "safe": len(issues) == 0,
        "issue": issues[0] if issues else None,
        "all_issues": issues,
    }


@weave.op()
def latency_scorer(output: dict, expected: dict) -> dict:
    """Score based on execution time."""
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
