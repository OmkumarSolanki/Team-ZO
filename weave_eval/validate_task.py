"""Per-task validation: run a single real task through the pipeline and compare to ground truth.

Usage:
  python weave_eval/validate_task.py 388967              # single task
  python weave_eval/validate_task.py 388967 389271 388763 # multiple tasks
  python weave_eval/validate_task.py --all --limit 20     # batch validation
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
import weave

from config import WANDB_ENTITY, WANDB_FULL_PROJECT, WANDB_PROJECT

DB_PATH = Path(__file__).parent.parent / "data" / "wfm.sqlite"
CODES_PATH = Path(__file__).parent.parent / "data" / "codes.json"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _load_codes():
    with open(str(CODES_PATH)) as f:
        return json.load(f)


def load_task(task_id: int) -> dict | None:
    """Load all ground truth data for a given task_id."""
    conn = _get_conn()
    cur = conn.execute("""
        SELECT r.REQUEST_ID, c.TASK_ID,
               r.CUST_PROB_DESCR, r.REQ_CLASS, r.PRIORITY, r.SEVERITY, r.USER_DEF21,
               r.PLACE_ID, r.REQ_STATUS,
               c.PROBLEM_CODE, c.CAUSE_CODE, c.RECTIFY_CODE, c.METHOD_CODE,
               t.COMBINED_TEXT
        FROM request r
        JOIN close_out c ON r.REQUEST_ID = c.REQUEST_ID
        JOIN task_text t ON c.TASK_ID = t.TASK_ID
        WHERE c.TASK_ID = ?
    """, [task_id])
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    row = dict(row)

    cur2 = conn.execute("""
        SELECT CONTRIBUTING_FACTOR, PROBABLE_CAUSE, PIPE_DEPTH
        FROM water_break WHERE REQUEST_ID = ? AND TASK_ID = ?
    """, [row["REQUEST_ID"], task_id])
    wb_row = cur2.fetchone()

    conn.close()

    codes = _load_codes()
    code_labels = {}
    for category in ["problem", "cause", "rectify", "method"]:
        for c in codes[category]:
            code_labels[c["code"]] = c["label"]

    return {
        "task_id": row["TASK_ID"],
        "request_id": row["REQUEST_ID"],
        "customer_report": row["CUST_PROB_DESCR"],
        "technician_transcript": row["COMBINED_TEXT"],
        "classification": {
            "req_class": row["REQ_CLASS"],
            "priority": row["PRIORITY"],
            "severity": row["SEVERITY"],
            "user_def21": row["USER_DEF21"],
        },
        "expected_codes": {
            "problem_code": row["PROBLEM_CODE"],
            "cause_code": row["CAUSE_CODE"],
            "rectify_code": row["RECTIFY_CODE"],
            "method_code": row["METHOD_CODE"],
        },
        "expected_labels": {
            "problem": code_labels.get(row["PROBLEM_CODE"], "?"),
            "cause": code_labels.get(row["CAUSE_CODE"], "?"),
            "rectify": code_labels.get(row["RECTIFY_CODE"], "?"),
            "method": code_labels.get(row["METHOD_CODE"], "?"),
        },
        "water_break": dict(wb_row) if wb_row else None,
    }


def list_available_tasks(limit: int = 20) -> list[int]:
    conn = _get_conn()
    cur = conn.execute("""
        SELECT c.TASK_ID FROM close_out c
        JOIN task_text t ON c.TASK_ID = t.TASK_ID
        JOIN request r ON c.REQUEST_ID = r.REQUEST_ID
        WHERE c.PROBLEM_CODE IS NOT NULL
          AND c.CAUSE_CODE IS NOT NULL
          AND c.RECTIFY_CODE IS NOT NULL
          AND c.METHOD_CODE IS NOT NULL
          AND t.COMBINED_TEXT IS NOT NULL
          AND r.CUST_PROB_DESCR IS NOT NULL
        LIMIT ?
    """, [limit])
    task_ids = [r[0] for r in cur.fetchall()]
    conn.close()
    return task_ids


@weave.op()
def validate_single_task(task_id: int) -> dict:
    """Run full validation for one task: voice → structuring → compare to ground truth."""
    from agents.structuring_agent import GEMI
    from agents.voice_agent import Vera
    from weave_eval.scorers import consistency_guard, enum_match

    task_data = load_task(task_id)
    if not task_data:
        return {"error": f"Task {task_id} not found"}

    start = time.time()
    voice = Vera()
    structuring = GEMI(mode="close_out")

    # Step 1: Voice agent receives the technician transcript
    voice_result = voice.predict(text=task_data["technician_transcript"])

    # Step 2: Structuring agent maps transcript to codes
    agent_output = structuring.structure(
        voice_result["transcript"],
        context={"request_id": task_data["request_id"]},
    )

    elapsed = time.time() - start

    # Step 3: Compare agent output to ground truth
    expected = task_data["expected_codes"]
    match_result = enum_match(agent_output, expected)

    # Step 4: Run guardrail
    guard_result = consistency_guard(agent_output)

    # Step 5: Per-field comparison
    field_comparison = {}
    codes = _load_codes()
    code_labels = {}
    for category in ["problem", "cause", "rectify", "method"]:
        for c in codes[category]:
            code_labels[c["code"]] = c["label"]

    for field in ["problem_code", "cause_code", "rectify_code", "method_code"]:
        agent_val = agent_output.get(field, "")
        expected_val = expected.get(field, "")
        confidence = agent_output.get("field_confidence", {}).get(field, 0)
        field_comparison[field] = {
            "agent": agent_val,
            "agent_label": code_labels.get(agent_val, "?"),
            "expected": expected_val,
            "expected_label": code_labels.get(expected_val, "?"),
            "match": str(agent_val).upper() == str(expected_val).upper(),
            "confidence": confidence,
        }

    result = {
        "task_id": task_id,
        "request_id": task_data["request_id"],
        "customer_report": task_data["customer_report"],
        "technician_transcript": task_data["technician_transcript"][:300],
        "voice_output": voice_result,
        "agent_codes": {
            "problem_code": agent_output.get("problem_code"),
            "cause_code": agent_output.get("cause_code"),
            "rectify_code": agent_output.get("rectify_code"),
            "method_code": agent_output.get("method_code"),
        },
        "expected_codes": expected,
        "expected_labels": task_data["expected_labels"],
        "field_comparison": field_comparison,
        "all_correct": match_result["all_correct"],
        "fields_correct": match_result["fields_correct"],
        "fields_total": match_result["fields_total"],
        "guardrail": guard_result,
        "needs_clarification": agent_output.get("needs_clarification"),
        "elapsed_seconds": elapsed,
        "model": agent_output.get("_model", "unknown"),
        "tokens": agent_output.get("_tokens", 0),
    }

    return result


def print_validation_result(result: dict):
    """Pretty-print a validation result."""
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return

    print(f"\n{'='*80}")
    print(f"TASK {result['task_id']} | REQUEST {result['request_id']}")
    print(f"{'='*80}")
    print(f"\n  Customer report:  {result['customer_report'][:120]}")
    print(f"  Tech transcript:  {result['technician_transcript'][:120]}...")
    print(f"  Voice received:   {result['voice_output'].get('source', '?')}")

    print(f"\n  {'Field':<15} {'Agent':<10} {'Expected':<10} {'Match':<6} {'Conf':<6} {'Agent Label':<25} {'Expected Label':<25}")
    print(f"  {'-'*97}")
    for field, comp in result["field_comparison"].items():
        match_icon = "YES" if comp["match"] else "NO"
        print(f"  {field:<15} {comp['agent']:<10} {comp['expected']:<10} {match_icon:<6} {comp['confidence']:<6.2f} {comp['agent_label']:<25} {comp['expected_label']:<25}")

    score = result["fields_correct"]
    total = result["fields_total"]
    print(f"\n  Score: {score}/{total} fields correct {'ALL MATCH' if result['all_correct'] else 'MISMATCH'}")
    print(f"  Guardrail: {'SAFE' if result['guardrail']['safe'] else 'BLOCKED: ' + str(result['guardrail'].get('issue'))}")

    if result.get("needs_clarification"):
        print(f"  Clarification needed: {result['needs_clarification']}")

    print(f"  Time: {result['elapsed_seconds']:.2f}s | Tokens: {result['tokens']} | Model: {result['model']}")


def run_batch_validation(task_ids: list[int]) -> dict:
    """Run validation for multiple tasks and aggregate results."""
    results = []
    total_correct = 0
    total_fields = 0
    field_accuracy = {"problem_code": 0, "cause_code": 0, "rectify_code": 0, "method_code": 0}
    field_total = {"problem_code": 0, "cause_code": 0, "rectify_code": 0, "method_code": 0}

    for task_id in task_ids:
        print(f"\nValidating task {task_id}...")
        result = validate_single_task(task_id)
        results.append(result)
        print_validation_result(result)

        if not result.get("error"):
            total_correct += result["fields_correct"]
            total_fields += result["fields_total"]
            for field, comp in result["field_comparison"].items():
                field_total[field] += 1
                if comp["match"]:
                    field_accuracy[field] += 1

    # Aggregate
    overall_accuracy = total_correct / total_fields if total_fields > 0 else 0
    per_field = {f: field_accuracy[f] / field_total[f] if field_total[f] > 0 else 0 for f in field_accuracy}
    all_correct_count = sum(1 for r in results if r.get("all_correct"))

    summary = {
        "total_tasks": len(task_ids),
        "overall_field_accuracy": overall_accuracy,
        "all_fields_correct_count": all_correct_count,
        "all_fields_correct_rate": all_correct_count / len(task_ids) if task_ids else 0,
        "per_field_accuracy": per_field,
        "total_tokens": sum(r.get("tokens", 0) for r in results),
        "avg_time": sum(r.get("elapsed_seconds", 0) for r in results) / len(results) if results else 0,
    }

    # Log to W&B
    wandb.log({
        "validation/overall_accuracy": overall_accuracy,
        "validation/all_correct_rate": summary["all_fields_correct_rate"],
        "validation/problem_code_accuracy": per_field["problem_code"],
        "validation/cause_code_accuracy": per_field["cause_code"],
        "validation/rectify_code_accuracy": per_field["rectify_code"],
        "validation/method_code_accuracy": per_field["method_code"],
        "validation/total_tasks": len(task_ids),
        "validation/avg_time_seconds": summary["avg_time"],
    })

    # Log results table
    table = wandb.Table(columns=[
        "task_id", "request_id", "transcript",
        "P_agent", "P_expected", "P_match",
        "C_agent", "C_expected", "C_match",
        "R_agent", "R_expected", "R_match",
        "M_agent", "M_expected", "M_match",
        "all_correct", "time_s",
    ])
    for r in results:
        if r.get("error"):
            continue
        fc = r["field_comparison"]
        table.add_data(
            r["task_id"], r["request_id"], r["technician_transcript"][:100],
            fc["problem_code"]["agent"], fc["problem_code"]["expected"], fc["problem_code"]["match"],
            fc["cause_code"]["agent"], fc["cause_code"]["expected"], fc["cause_code"]["match"],
            fc["rectify_code"]["agent"], fc["rectify_code"]["expected"], fc["rectify_code"]["match"],
            fc["method_code"]["agent"], fc["method_code"]["expected"], fc["method_code"]["match"],
            r["all_correct"], round(r["elapsed_seconds"], 2),
        )
    wandb.log({"validation/results_table": table})

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate agent against real task data")
    parser.add_argument("task_ids", nargs="*", type=int, help="Task IDs to validate")
    parser.add_argument("--all", action="store_true", help="Validate all available tasks")
    parser.add_argument("--limit", type=int, default=10, help="Limit for --all mode")
    args = parser.parse_args()

    weave.init(WANDB_FULL_PROJECT)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"validation-{int(time.time())}",
        tags=["validation", "ground-truth"],
    )

    if args.all:
        task_ids = list_available_tasks(args.limit)
    elif args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = list_available_tasks(5)

    print(f"Validating {len(task_ids)} tasks against ground truth...")
    summary = run_batch_validation(task_ids)

    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Tasks validated:        {summary['total_tasks']}")
    print(f"  Overall field accuracy: {summary['overall_field_accuracy']:.1%}")
    print(f"  All-correct rate:       {summary['all_fields_correct_rate']:.1%}")
    print(f"  Per-field accuracy:")
    for field, acc in summary["per_field_accuracy"].items():
        print(f"    {field:<15} {acc:.1%}")
    print(f"  Total tokens:           {summary['total_tokens']}")
    print(f"  Avg time per task:      {summary['avg_time']:.2f}s")

    run.finish()
    print(f"\nResults logged to W&B + Weave!")


if __name__ == "__main__":
    main()
