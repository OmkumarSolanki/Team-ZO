"""Build comprehensive Weave evaluation datasets from the xlsx data.

Joins all sheets to produce richly labelled eval sets:
- close_out: CUST_PROB_DESCR + COMBINED_TEXT → PCRM codes (ground truth)
- new_fault: CUST_PROB_DESCR → REQ_CLASS, PRIORITY, SEVERITY, USER_DEF21
- water_break: adds CONTRIBUTING_FACTOR + PROBABLE_CAUSE for deeper validation

All datasets are published to Weave and logged as W&B artifacts.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import wandb
import weave

DB_PATH = Path(__file__).parent.parent / "data" / "wfm.sqlite"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def build_closeout_dataset() -> list[dict]:
    """Build eval dataset for close-out structuring.

    Joins request + close_out_PCRM + task_text to get:
    - Input: COMBINED_TEXT (technician's own words) or CUST_PROB_DESCR (customer call)
    - Ground truth: PROBLEM_CODE, CAUSE_CODE, RECTIFY_CODE, METHOD_CODE
    """
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT r.REQUEST_ID, r.CUST_PROB_DESCR, r.REQ_CLASS, r.PRIORITY,
               r.SEVERITY, r.USER_DEF21, r.PLACE_ID,
               c.PROBLEM_CODE, c.CAUSE_CODE, c.RECTIFY_CODE, c.METHOD_CODE,
               c.TASK_ID,
               t.COMBINED_TEXT
        FROM request r
        JOIN close_out c ON r.REQUEST_ID = c.REQUEST_ID
        LEFT JOIN task_text t ON c.TASK_ID = t.TASK_ID
        WHERE r.CUST_PROB_DESCR IS NOT NULL
          AND c.PROBLEM_CODE IS NOT NULL
          AND c.CAUSE_CODE IS NOT NULL
          AND c.RECTIFY_CODE IS NOT NULL
          AND c.METHOD_CODE IS NOT NULL
    """)

    rows = []
    for r in cur.fetchall():
        r = dict(r)
        combined = r["COMBINED_TEXT"] or ""
        cust_descr = r["CUST_PROB_DESCR"] or ""
        transcript = combined if combined else cust_descr

        rows.append({
            "request_id": r["REQUEST_ID"],
            "task_id": r["TASK_ID"],
            "transcript": transcript,
            "cust_prob_descr": cust_descr,
            "combined_text": combined,
            "req_class": r["REQ_CLASS"],
            "priority": r["PRIORITY"],
            "severity": r["SEVERITY"],
            "user_def21": r["USER_DEF21"],
            "place_id": r["PLACE_ID"],
            "expected": {
                "problem_code": r["PROBLEM_CODE"],
                "cause_code": r["CAUSE_CODE"],
                "rectify_code": r["RECTIFY_CODE"],
                "method_code": r["METHOD_CODE"],
            },
        })

    conn.close()
    return rows


def build_newfault_dataset() -> list[dict]:
    """Build eval dataset for new-fault classification.

    Input: CUST_PROB_DESCR (the customer's report)
    Ground truth: REQ_CLASS, PRIORITY, SEVERITY, USER_DEF21
    """
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT REQUEST_ID, CUST_PROB_DESCR, REQ_CLASS, PRIORITY,
               SEVERITY, USER_DEF21, PLACE_ID
        FROM request
        WHERE CUST_PROB_DESCR IS NOT NULL
          AND REQ_CLASS IS NOT NULL
          AND PRIORITY IS NOT NULL
    """)

    rows = []
    for r in cur.fetchall():
        r = dict(r)
        rows.append({
            "request_id": r["REQUEST_ID"],
            "transcript": r["CUST_PROB_DESCR"],
            "cust_prob_descr": r["CUST_PROB_DESCR"],
            "place_id": r["PLACE_ID"],
            "expected": {
                "req_class": r["REQ_CLASS"],
                "priority": r["PRIORITY"],
                "severity": r["SEVERITY"] or "MEDIUM",
                "user_def21": r["USER_DEF21"] or "FOOTPATH_GRASS",
            },
        })

    conn.close()
    return rows


def get_weave_dataset(mode: str = "close_out", limit: int | None = None) -> weave.Dataset:
    if mode == "close_out":
        rows = build_closeout_dataset()
    else:
        rows = build_newfault_dataset()

    if limit:
        rows = rows[:limit]

    return weave.Dataset(name=f"wfm-{mode}-eval", rows=rows)


def publish_datasets_to_wandb():
    """Publish all datasets to W&B as artifacts and Weave datasets."""
    closeout_rows = build_closeout_dataset()
    newfault_rows = build_newfault_dataset()

    print(f"Close-out dataset: {len(closeout_rows)} labelled examples")
    print(f"New-fault dataset: {len(newfault_rows)} labelled examples")

    closeout_ds = weave.Dataset(name="wfm-closeout-eval", rows=closeout_rows)
    newfault_ds = weave.Dataset(name="wfm-newfault-eval", rows=newfault_rows)

    weave.publish(closeout_ds, name="wfm-closeout-eval")
    weave.publish(newfault_ds, name="wfm-newfault-eval")
    print("Published both datasets to Weave")

    artifact = wandb.Artifact("wfm-eval-data", type="dataset")
    artifact.add_file(str(DB_PATH), name="wfm.sqlite")
    artifact.add_file(
        str(Path(__file__).parent.parent / "data" / "codes.json"),
        name="codes.json",
    )
    artifact.add_file(
        str(Path(__file__).parent.parent / "schema" / "request_schema.json"),
        name="request_schema.json",
    )

    table_closeout = wandb.Table(
        columns=["request_id", "transcript", "problem_code", "cause_code",
                 "rectify_code", "method_code", "req_class", "priority"]
    )
    for row in closeout_rows:
        exp = row["expected"]
        table_closeout.add_data(
            row["request_id"],
            row["transcript"][:200],
            exp["problem_code"], exp["cause_code"],
            exp["rectify_code"], exp["method_code"],
            row.get("req_class", ""), row.get("priority", ""),
        )

    table_newfault = wandb.Table(
        columns=["request_id", "transcript", "req_class", "priority",
                 "severity", "user_def21"]
    )
    for row in newfault_rows:
        exp = row["expected"]
        table_newfault.add_data(
            row["request_id"],
            row["transcript"][:200],
            exp["req_class"], exp["priority"],
            exp["severity"], exp["user_def21"],
        )

    wandb.log({
        "datasets/closeout_examples": table_closeout,
        "datasets/newfault_examples": table_newfault,
        "datasets/closeout_count": len(closeout_rows),
        "datasets/newfault_count": len(newfault_rows),
    })

    wandb.log_artifact(artifact)
    print("Published artifact + tables to W&B")

    return closeout_rows, newfault_rows


def get_data_summary() -> dict:
    """Get a summary of the dataset for display."""
    conn = _get_conn()

    summary = {}

    cur = conn.execute("SELECT COUNT(*) FROM request")
    summary["total_requests"] = cur.fetchone()[0]

    cur = conn.execute("SELECT COUNT(*) FROM close_out WHERE PROBLEM_CODE IS NOT NULL")
    summary["total_closeouts_with_codes"] = cur.fetchone()[0]

    cur = conn.execute("SELECT COUNT(*) FROM task_text WHERE COMBINED_TEXT IS NOT NULL")
    summary["total_task_texts"] = cur.fetchone()[0]

    for col in ["REQ_CLASS", "PRIORITY", "SEVERITY", "USER_DEF21"]:
        cur = conn.execute(
            f"SELECT {col}, COUNT(*) FROM request WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY COUNT(*) DESC"
        )
        summary[f"dist_{col.lower()}"] = {r[0]: r[1] for r in cur.fetchall()}

    for col in ["PROBLEM_CODE", "CAUSE_CODE", "RECTIFY_CODE", "METHOD_CODE"]:
        cur = conn.execute(
            f"SELECT {col}, COUNT(*) FROM close_out WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY COUNT(*) DESC"
        )
        summary[f"dist_{col.lower()}"] = {r[0]: r[1] for r in cur.fetchall()}

    conn.close()
    return summary
