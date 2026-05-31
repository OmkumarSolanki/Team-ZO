"""Build Weave evaluation dataset from the xlsx data.

Pairs CUST_PROB_DESCR (input) with structured fields (expected output),
using both the request sheet and close_out_PCRM for ground truth labels.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import weave

DB_PATH = Path(__file__).parent.parent / "data" / "wfm.sqlite"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def build_closeout_dataset() -> list[dict]:
    """Build eval dataset for close-out structuring from real data."""
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT r.REQUEST_ID, r.CUST_PROB_DESCR, r.REQ_CLASS, r.PRIORITY,
               r.SEVERITY, r.USER_DEF21,
               c.PROBLEM_CODE, c.CAUSE_CODE, c.RECTIFY_CODE, c.METHOD_CODE,
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
        transcript = r["COMBINED_TEXT"] or r["CUST_PROB_DESCR"]
        rows.append({
            "transcript": transcript,
            "cust_prob_descr": r["CUST_PROB_DESCR"],
            "task": transcript,
            "expected": {
                "problem_code": r["PROBLEM_CODE"],
                "cause_code": r["CAUSE_CODE"],
                "rectify_code": r["RECTIFY_CODE"],
                "method_code": r["METHOD_CODE"],
            },
            "request_id": r["REQUEST_ID"],
        })

    conn.close()
    return rows


def build_newfault_dataset() -> list[dict]:
    """Build eval dataset for new-fault classification."""
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT REQUEST_ID, CUST_PROB_DESCR, REQ_CLASS, PRIORITY,
               SEVERITY, USER_DEF21
        FROM request
        WHERE CUST_PROB_DESCR IS NOT NULL
          AND REQ_CLASS IS NOT NULL
          AND PRIORITY IS NOT NULL
    """)

    rows = []
    for r in cur.fetchall():
        r = dict(r)
        rows.append({
            "transcript": r["CUST_PROB_DESCR"],
            "cust_prob_descr": r["CUST_PROB_DESCR"],
            "task": r["CUST_PROB_DESCR"],
            "expected": {
                "req_class": r["REQ_CLASS"],
                "priority": r["PRIORITY"],
                "severity": r["SEVERITY"] or "MEDIUM",
                "user_def21": r["USER_DEF21"] or "FOOTPATH_GRASS",
            },
            "request_id": r["REQUEST_ID"],
        })

    conn.close()
    return rows


def get_weave_dataset(mode: str = "close_out", limit: int | None = None) -> weave.Dataset:
    """Get a Weave Dataset for evaluation."""
    if mode == "close_out":
        rows = build_closeout_dataset()
    else:
        rows = build_newfault_dataset()

    if limit:
        rows = rows[:limit]

    return weave.Dataset(name=f"wfm-{mode}-eval", rows=rows)
