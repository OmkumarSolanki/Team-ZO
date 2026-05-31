"""Weave Evaluation — Full multi-agent pipeline: Vera → GEMI → Hade.

Each agent is a separate weave.Model, so all three appear by name in Weave traces:
  - Vera: Voice Agent (STT simulation)
  - GEMI: Structuring Agent (maps transcript to PCRM codes)
  - Hade: HelpDesk Agent (validates, corrects, approves)

Hade's correction loop: validates GEMI's output against the enum schema.
If codes are invalid, Hade sends feedback back to GEMI for retry (up to 2x).

Input: task_text.COMBINED_TEXT (simulation script)
Ground truth: close_out_PCRM codes

Usage:
  python weave_eval/run_eval.py              # 20 rows
  python weave_eval/run_eval.py --limit 50   # 50 rows
  python weave_eval/run_eval.py --all        # all 130 rows
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import weave

from config import MODEL, WANDB_FULL_PROJECT

DB_PATH = Path(__file__).parent.parent / "data" / "wfm.sqlite"
CODES_PATH = Path(__file__).parent.parent / "data" / "codes.json"


# ─── Dataset ─────────────────────────────────────────────────────────────────

def build_eval_dataset(limit: int | None = None) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    cur = conn.execute("""
        SELECT c.TASK_ID, c.REQUEST_ID,
               c.PROBLEM_CODE, c.CAUSE_CODE, c.RECTIFY_CODE, c.METHOD_CODE,
               t.COMBINED_TEXT
        FROM close_out c
        JOIN task_text t ON c.TASK_ID = t.TASK_ID
        WHERE c.PROBLEM_CODE IS NOT NULL
          AND c.CAUSE_CODE IS NOT NULL
          AND c.RECTIFY_CODE IS NOT NULL
          AND c.METHOD_CODE IS NOT NULL
          AND t.COMBINED_TEXT IS NOT NULL
          AND LENGTH(t.COMBINED_TEXT) > 50
        ORDER BY c.TASK_ID
    """)

    rows = []
    for r in cur.fetchall():
        r = dict(r)
        rows.append({
            "task_id": r["TASK_ID"],
            "request_id": r["REQUEST_ID"],
            "transcript": r["COMBINED_TEXT"],
            "expected": {
                "problem_code": r["PROBLEM_CODE"],
                "cause_code": r["CAUSE_CODE"],
                "rectify_code": r["RECTIFY_CODE"],
                "method_code": r["METHOD_CODE"],
            },
        })

    conn.close()
    if limit:
        rows = rows[:limit]
    return rows


def _load_valid_codes() -> dict[str, list[str]]:
    with open(str(CODES_PATH)) as f:
        codes = json.load(f)
    return {
        "problem_code": [c["code"] for c in codes["problem"]],
        "cause_code": [c["code"] for c in codes["cause"]],
        "rectify_code": [c["code"] for c in codes["rectify"]],
        "method_code": [c["code"] for c in codes["method"]],
    }


# ─── Pipeline Model using actual agent instances ─────────────────────────────

class MultiAgentPipeline(weave.Model):
    """Full orchestrated pipeline calling Vera, GEMI, and Hade as separate agents.

    Each agent is a weave.Model instance — their predict/structure/validate_and_correct
    calls appear as separate named traces in Weave.
    """
    name: str = "Pipeline (Vera → GEMI → Hade)"
    model_name: str = MODEL
    max_corrections: int = 2

    @weave.op
    def predict(self, transcript: str, **kwargs) -> dict:
        from agents.helpdesk_agent import Hade
        from agents.structuring_agent import GEMI
        from agents.voice_agent import Vera

        vera = Vera()
        gemi = GEMI(mode="close_out")
        hade = Hade()

        # Step 1: Vera receives the technician's report (shows as Vera.predict in Weave)
        vera_output = vera.predict(text=transcript)

        # Step 2: GEMI structures into codes (shows as GEMI.structure in Weave)
        gemi_output = gemi.structure(vera_output["transcript"])

        # Step 3: Hade validates and corrects (shows as Hade.validate_and_correct in Weave)
        final_output = hade.validate_and_correct(
            vera_output["transcript"],
            gemi_output,
            gemi_agent=gemi,
            max_corrections=self.max_corrections,
        )

        return final_output


# ─── Scorers ─────────────────────────────────────────────────────────────────

@weave.op
def pcrm_exact_match(expected: dict, output: dict) -> dict[str, bool]:
    """Compares final output vs close_out_PCRM ground truth."""
    p = str(output.get("problem_code", "")).upper() == str(expected.get("problem_code", "")).upper()
    c = str(output.get("cause_code", "")).upper() == str(expected.get("cause_code", "")).upper()
    r = str(output.get("rectify_code", "")).upper() == str(expected.get("rectify_code", "")).upper()
    m = str(output.get("method_code", "")).upper() == str(expected.get("method_code", "")).upper()

    return {
        "problem_correct": p,
        "cause_correct": c,
        "rectify_correct": r,
        "method_correct": m,
        "all_correct": p and c and r and m,
    }


class HadeJudge(weave.Scorer):
    """Hade as Judge — reports whether all codes are valid enums."""
    name: str = "Hade (Judge)"

    @weave.op
    def score(self, output: dict) -> dict:
        valid_codes = _load_valid_codes()

        all_valid = True
        for field, valid_list in valid_codes.items():
            value = output.get(field, "")
            if not value or value not in valid_list:
                all_valid = False

        return {
            "hade_approved": output.get("_hade_approved", False),
            "all_codes_valid": all_valid,
            "corrections_needed": output.get("_hade_corrections", 0),
        }


class GEMIConfidence(weave.Scorer):
    """GEMI's self-reported confidence scores."""
    name: str = "GEMI (Confidence)"

    @weave.op
    def score(self, output: dict) -> dict:
        conf = output.get("field_confidence", {})
        values = list(conf.values()) if conf else [0]
        return {
            "avg_confidence": sum(values) / len(values),
            "min_confidence": min(values),
            "needs_clarification": output.get("needs_clarification") is not None,
        }


# ─── Run ─────────────────────────────────────────────────────────────────────

def run_eval(limit: int = 20):
    rows = build_eval_dataset(limit=limit)

    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  Multi-Agent Evaluation: Vera → GEMI → Hade                     ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Vera:  Voice Agent — receives simulation script                 ║")
    print(f"║  GEMI:  Structuring Agent — maps to PCRM codes                   ║")
    print(f"║  Hade:  HelpDesk Agent — validates, corrects, approves            ║")
    print(f"║                                                                  ║")
    print(f"║  Hade correction loop: up to 2 retries if codes invalid           ║")
    print(f"║  Ground truth: close_out_PCRM sheet                              ║")
    print(f"║  Model: {MODEL:<52} ║")
    print(f"║  Examples: {len(rows):<49} ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print()

    dataset = weave.Dataset(name="wfm-pcrm-eval", rows=rows)
    model = MultiAgentPipeline()

    evaluation = weave.Evaluation(
        name="multi-agent-pcrm-eval",
        dataset=dataset,
        scorers=[
            pcrm_exact_match,
            HadeJudge(),
            GEMIConfidence(),
        ],
    )

    results = asyncio.run(evaluation.evaluate(model))

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    pcrm = results.get("pcrm_exact_match", {})
    print(f"  Problem code:  {pcrm.get('problem_correct', {}).get('true_fraction', 0):.0%}")
    print(f"  Cause code:    {pcrm.get('cause_correct', {}).get('true_fraction', 0):.0%}")
    print(f"  Rectify code:  {pcrm.get('rectify_correct', {}).get('true_fraction', 0):.0%}")
    print(f"  Method code:   {pcrm.get('method_correct', {}).get('true_fraction', 0):.0%}")
    print(f"  All 4 correct: {pcrm.get('all_correct', {}).get('true_fraction', 0):.0%}")

    hade = results.get("Hade (Judge)", results.get("Hade-Judge", {}))
    print(f"\n  Hade approved:    {hade.get('hade_approved', {}).get('true_fraction', 0):.0%}")
    print(f"  All codes valid:  {hade.get('all_codes_valid', {}).get('true_fraction', 0):.0%}")
    print(f"  Avg corrections:  {hade.get('corrections_needed', {}).get('mean', 0):.1f}")

    gemi = results.get("GEMI (Confidence)", results.get("GEMI-Confidence", {}))
    print(f"\n  GEMI confidence:  {gemi.get('avg_confidence', {}).get('mean', 0):.2f}")
    print(f"  Latency:          {results.get('model_latency', {}).get('mean', 0):.2f}s")

    print(f"\n📊 View: https://wandb.ai/{WANDB_FULL_PROJECT}/weave/evaluations")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-agent PCRM evaluation")
    parser.add_argument("--limit", type=int, default=20, help="Number of examples")
    parser.add_argument("--all", action="store_true", help="Run on all 130 examples")
    args = parser.parse_args()

    weave.init(WANDB_FULL_PROJECT)

    limit = None if args.all else args.limit
    run_eval(limit=limit)

    print("\n✅ Done! Check Weave Evaluations page.")
