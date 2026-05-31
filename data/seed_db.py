"""Seed SQLite database from the xlsx export.

Loads request, task, close_out_PCRM, and task_text sheets
into a local SQLite DB for the WFM MCP server.
"""
import json
import sqlite3
from pathlib import Path

import openpyxl

XLSX_PATH = Path(__file__).parent.parent.parent / "Example_dataset_water_maintenance.xlsx"
DB_PATH = Path(__file__).parent / "wfm.sqlite"
CODES_PATH = Path(__file__).parent / "codes.json"
SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "request_schema.json"

REQUEST_COLS = [
    "REQUEST_ID", "REQ_STATUS", "REQ_CLASS", "PRIORITY", "SEVERITY",
    "CUST_PROB_DESCR", "PLACE_ID", "ADDRESS_ID", "USER_DEF21",
    "PROBLEM_CODE", "REQUEST_DTTM", "PERSON_ID_OWNER", "CROSS_REFERENCE",
]

CLOSE_OUT_COLS = [
    "REQUEST_ID", "TASK_ID", "PROBLEM_CODE", "CAUSE_CODE",
    "RECTIFY_CODE", "METHOD_CODE",
]

TASK_TEXT_COLS = ["TASK_ID", "COMBINED_TEXT"]


def load_sheet(wb, sheet_name, columns):
    ws = wb[sheet_name]
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    col_indices = {}
    for col in columns:
        if col in headers:
            col_indices[col] = headers.index(col)

    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        vals = list(row)
        row_dict = {}
        for col, idx in col_indices.items():
            row_dict[col] = vals[idx] if idx < len(vals) else None
        rows.append(row_dict)
    return rows


def seed_database():
    wb = openpyxl.load_workbook(str(XLSX_PATH), read_only=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE request (
            REQUEST_ID INTEGER PRIMARY KEY,
            REQ_STATUS TEXT,
            REQ_CLASS TEXT,
            PRIORITY TEXT,
            SEVERITY TEXT,
            CUST_PROB_DESCR TEXT,
            PLACE_ID TEXT,
            ADDRESS_ID INTEGER,
            USER_DEF21 TEXT,
            PROBLEM_CODE TEXT,
            REQUEST_DTTM TEXT,
            PERSON_ID_OWNER TEXT,
            CROSS_REFERENCE TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE close_out (
            REQUEST_ID INTEGER,
            TASK_ID INTEGER,
            PROBLEM_CODE TEXT,
            CAUSE_CODE TEXT,
            RECTIFY_CODE TEXT,
            METHOD_CODE TEXT,
            PRIMARY KEY (REQUEST_ID, TASK_ID)
        )
    """)

    cur.execute("""
        CREATE TABLE task_text (
            TASK_ID INTEGER PRIMARY KEY,
            COMBINED_TEXT TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE water_break (
            REQUEST_ID INTEGER,
            TASK_ID INTEGER,
            CONTRIBUTING_FACTOR TEXT,
            PROBABLE_CAUSE TEXT,
            PIPE_DEPTH REAL,
            PRIMARY KEY (REQUEST_ID, TASK_ID)
        )
    """)

    requests = load_sheet(wb, "request", REQUEST_COLS)
    for r in requests:
        if r.get("REQUEST_ID"):
            cur.execute(
                "INSERT OR IGNORE INTO request VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [r.get(c) for c in REQUEST_COLS],
            )

    close_outs = load_sheet(wb, "close_out_PCRM", CLOSE_OUT_COLS)
    for c in close_outs:
        if c.get("REQUEST_ID") and c.get("TASK_ID"):
            cur.execute(
                "INSERT OR IGNORE INTO close_out VALUES (?,?,?,?,?,?)",
                [c.get(col) for col in CLOSE_OUT_COLS],
            )

    task_texts = load_sheet(wb, "task_text(aggregated)", TASK_TEXT_COLS)
    for t in task_texts:
        if t.get("TASK_ID"):
            cur.execute(
                "INSERT OR IGNORE INTO task_text VALUES (?,?)",
                [t.get("TASK_ID"), t.get("COMBINED_TEXT")],
            )

    water_break_cols = ["REQUEST_ID", "TASK_ID", "CONTRIBUTING_FACTOR", "PROBABLE_CAUSE", "PIPE_DEPTH"]
    water_breaks = load_sheet(wb, "close_out_water_break", water_break_cols)
    for w in water_breaks:
        if w.get("REQUEST_ID") and w.get("TASK_ID"):
            cur.execute(
                "INSERT OR IGNORE INTO water_break VALUES (?,?,?,?,?)",
                [w.get(col) for col in water_break_cols],
            )

    conn.commit()

    counts = {}
    for table in ["request", "close_out", "task_text", "water_break"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cur.fetchone()[0]

    conn.close()
    wb.close()

    print(f"Database seeded at {DB_PATH}")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")

    return counts


def generate_codes_json():
    wb = openpyxl.load_workbook(str(XLSX_PATH), read_only=True)
    ws = wb["Codes"]

    codes = {"problem": [], "cause": [], "rectify": [], "method": []}
    seen = {"problem": set(), "cause": set(), "rectify": set(), "method": set()}

    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[0] and row[1] and row[1] not in seen["problem"]:
            codes["problem"].append({"label": str(row[0]), "code": str(row[1])})
            seen["problem"].add(row[1])
        if row[3] and row[4] and row[4] not in seen["cause"]:
            codes["cause"].append({"label": str(row[3]), "code": str(row[4])})
            seen["cause"].add(row[4])
        if row[6] and row[7] and row[7] not in seen["rectify"]:
            codes["rectify"].append({"label": str(row[6]), "code": str(row[7])})
            seen["rectify"].add(row[7])
        if row[9] and row[10] and row[10] not in seen["method"]:
            codes["method"].append({"label": str(row[9]), "code": str(row[10])})
            seen["method"].add(row[10])

    wb.close()

    with open(str(CODES_PATH), "w") as f:
        json.dump(codes, f, indent=2)

    print(f"Codes written to {CODES_PATH}")
    for category, items in codes.items():
        print(f"  {category}: {len(items)} codes")

    return codes


def generate_request_schema():
    schema = {
        "close_out": {
            "required": ["request_id", "problem_code", "cause_code", "rectify_code", "method_code"],
            "fields": {
                "request_id": {"type": "integer"},
                "problem_code": {"type": "enum", "source": "codes.problem"},
                "cause_code": {"type": "enum", "source": "codes.cause"},
                "rectify_code": {"type": "enum", "source": "codes.rectify"},
                "method_code": {"type": "enum", "source": "codes.method"},
                "req_status": {"type": "enum", "values": ["COMPLETE"]},
            },
        },
        "new_fault": {
            "required": ["req_class", "priority", "severity", "cust_prob_descr"],
            "fields": {
                "req_class": {
                    "type": "enum",
                    "values": [
                        "HYDRANT_MAINTENANCE", "INSPECT", "LEAK",
                        "MANHOLE_MAINTENANCE", "METER_FRAME_LEAK", "NO_WATER",
                        "OTHER", "OVERFLOW_SHAFT", "PATHTAP_FAULT",
                        "RESTORE_SUPPLY_AFTER_RESTRICTION",
                        "SEWER_MANHOLE_OVERFLOWING", "SEWER_SHAFT_HOLDING",
                        "START_DAY", "WATER_QUALITY_DIRTY_WATER",
                    ],
                },
                "priority": {
                    "type": "enum",
                    "values": [
                        "P1_URG_1H_4H", "P2_HIGH_6H_1D", "P3_MED_2D_5D",
                        "P4_PLAN_10D", "P8_PLAN_12M", "P10_PLAN_7D",
                    ],
                },
                "severity": {
                    "type": "enum",
                    "values": ["LOW", "MEDIUM", "HIGH"],
                },
                "user_def21": {
                    "type": "enum",
                    "values": [
                        "DRIVEWAY", "FOOTPATH", "FOOTPATH_CONCRETE",
                        "FOOTPATH_GRASS", "PARK_RESERVE", "ROAD", "YARD",
                    ],
                },
                "cust_prob_descr": {"type": "text"},
            },
        },
        "business_rules": [
            {"rule": "P1_URG requires severity HIGH", "check": "if priority == 'P1_URG_1H_4H' then severity must be 'HIGH'"},
            {"rule": "COMPLETE requires all PCRM codes", "check": "if req_status == 'COMPLETE' then problem_code, cause_code, rectify_code, method_code must all be set"},
        ],
    }

    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(SCHEMA_PATH), "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Schema written to {SCHEMA_PATH}")
    return schema


if __name__ == "__main__":
    seed_database()
    generate_codes_json()
    generate_request_schema()
