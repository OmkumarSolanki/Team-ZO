"""WFM (Work Force Management) MCP Server.

Exposes search, get, validate, close, and post operations
over the SQLite WFM database. Every tool call is traced by Weave.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import weave

DB_PATH = Path(__file__).parent.parent / "data" / "wfm.sqlite"
CODES_PATH = Path(__file__).parent.parent / "data" / "codes.json"
SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "request_schema.json"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _load_codes() -> dict:
    with open(str(CODES_PATH)) as f:
        return json.load(f)


def _load_schema() -> dict:
    with open(str(SCHEMA_PATH)) as f:
        return json.load(f)


@weave.op()
def search_requests(text: str) -> list[dict]:
    """Search open/active requests by free text (address, description, ID)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT REQUEST_ID, REQ_STATUS, REQ_CLASS, PRIORITY, SEVERITY,
               CUST_PROB_DESCR, PLACE_ID, ADDRESS_ID, USER_DEF21
        FROM request
        WHERE CUST_PROB_DESCR LIKE ? OR PLACE_ID LIKE ? OR CAST(ADDRESS_ID AS TEXT) LIKE ?
              OR CAST(REQUEST_ID AS TEXT) = ?
        LIMIT 10
        """,
        [f"%{text}%", f"%{text}%", f"%{text}%", text],
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@weave.op()
def get_request(request_id: int) -> dict | None:
    """Get a single request by ID with full details."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM request WHERE REQUEST_ID = ?", [request_id])
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


@weave.op()
def validate_request(payload: dict, mode: str = "close_out") -> dict:
    """Validate a payload against the WFM schema and business rules."""
    schema = _load_schema()
    codes = _load_codes()
    errors = []

    mode_schema = schema.get(mode, {})
    required = mode_schema.get("required", [])
    fields = mode_schema.get("fields", {})

    for field in required:
        if not payload.get(field):
            errors.append(f"Missing required field: {field}")

    for field, spec in fields.items():
        value = payload.get(field)
        if value is None:
            continue

        if spec["type"] == "enum":
            if "source" in spec:
                source_key = spec["source"].split(".")
                valid_values = [c["code"] for c in codes.get(source_key[1], [])]
            else:
                valid_values = spec["values"]
            if value not in valid_values:
                errors.append(f"Invalid value for {field}: '{value}'. Valid: {valid_values}")

    for rule in schema.get("business_rules", []):
        if "P1_URG" in rule.get("check", ""):
            if payload.get("priority") == "P1_URG_1H_4H" and payload.get("severity") == "LOW":
                errors.append(rule["rule"])
        if "COMPLETE requires" in rule.get("rule", ""):
            if payload.get("req_status") == "COMPLETE":
                for code_field in ["problem_code", "cause_code", "rectify_code", "method_code"]:
                    if not payload.get(code_field):
                        errors.append(f"{rule['rule']}: missing {code_field}")

    return {"ok": len(errors) == 0, "errors": errors}


@weave.op()
def close_request(request_id: int, codes: dict) -> dict:
    """Close a request with the given PCRM codes."""
    validation = validate_request({**codes, "request_id": request_id, "req_status": "COMPLETE"}, mode="close_out")
    if not validation["ok"]:
        return {"status": "error", "errors": validation["errors"]}

    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("SELECT REQUEST_ID FROM request WHERE REQUEST_ID = ?", [request_id])
    if not cur.fetchone():
        conn.close()
        return {"status": "error", "errors": [f"Request {request_id} not found"]}

    cur.execute(
        "UPDATE request SET REQ_STATUS = 'COMPLETE', PROBLEM_CODE = ? WHERE REQUEST_ID = ?",
        [codes.get("problem_code"), request_id],
    )

    cur.execute(
        """INSERT OR REPLACE INTO close_out
           (REQUEST_ID, TASK_ID, PROBLEM_CODE, CAUSE_CODE, RECTIFY_CODE, METHOD_CODE)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            request_id, 0,
            codes.get("problem_code"), codes.get("cause_code"),
            codes.get("rectify_code"), codes.get("method_code"),
        ],
    )

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "request_id": request_id,
        "new_status": "COMPLETE",
        "codes_applied": codes,
    }


@weave.op()
def post_request(payload: dict) -> dict:
    """Create a new fault request."""
    validation = validate_request(payload, mode="new_fault")
    if not validation["ok"]:
        return {"status": "error", "errors": validation["errors"]}

    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("SELECT MAX(REQUEST_ID) FROM request")
    max_id = cur.fetchone()[0] or 500000
    new_id = max_id + 1

    cur.execute(
        """INSERT INTO request
           (REQUEST_ID, REQ_STATUS, REQ_CLASS, PRIORITY, SEVERITY, CUST_PROB_DESCR, USER_DEF21)
           VALUES (?, 'OPEN', ?, ?, ?, ?, ?)""",
        [
            new_id,
            payload.get("req_class"),
            payload.get("priority"),
            payload.get("severity"),
            payload.get("cust_prob_descr"),
            payload.get("user_def21"),
        ],
    )

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "request_id": new_id,
        "new_status": "OPEN",
    }


MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_requests",
            "description": "Search WFM requests by free text (address, description, or ID)",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Search text"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_request",
            "description": "Get full details of a specific request by ID",
            "parameters": {
                "type": "object",
                "properties": {"request_id": {"type": "integer"}},
                "required": ["request_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_request",
            "description": "Validate a close-out or new-fault payload against WFM schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "mode": {"type": "string", "enum": ["close_out", "new_fault"]},
                },
                "required": ["payload", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_request",
            "description": "Close a request with PCRM codes (problem, cause, rectify, method)",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "integer"},
                    "codes": {"type": "object"},
                },
                "required": ["request_id", "codes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_request",
            "description": "Create a new fault request in the WFM system",
            "parameters": {
                "type": "object",
                "properties": {"payload": {"type": "object"}},
                "required": ["payload"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_requests": lambda args: search_requests(args["text"]),
    "get_request": lambda args: get_request(args["request_id"]),
    "validate_request": lambda args: validate_request(args["payload"], args.get("mode", "close_out")),
    "close_request": lambda args: close_request(args["request_id"], args["codes"]),
    "post_request": lambda args: post_request(args["payload"]),
}
