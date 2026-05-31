"""Mainline — Web Frontend.

Multi-page app:
  /           — Dashboard (stats, pending tickets, completed history)
  /chat?id=X  — Voice close-out for a specific ticket
  /chat       — Free-form voice close-out
  /turn       — AJAX conversation endpoint

Continuous voice via browser Web Speech API.
Agent auto-speaks via SpeechSynthesis.

Run: python frontend/server.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
import weave
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
from starlette.concurrency import run_in_threadpool

from config import WANDB_ENTITY, WANDB_FULL_PROJECT, WANDB_PROJECT
from agents.structuring_agent import GEMI
from agents.helpdesk_agent import Hade
from agents.voice_agent import Vera
from orchestrator.intents import classify_intent, Intent

gemi = GEMI(mode="close_out")
hade = Hade()
vera = Vera()

DB_PATH = Path(__file__).parent.parent / "data" / "wfm.sqlite"
CODES_PATH = Path(__file__).parent.parent / "data" / "codes.json"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _load_codes():
    with open(str(CODES_PATH)) as f:
        return json.load(f)


def _code_label(code: str) -> str:
    if not code:
        return "—"
    codes = _load_codes()
    for category in codes.values():
        for item in category:
            if item["code"] == code:
                return item["label"]
    return code


def _get_stats():
    conn = _get_conn()
    cur = conn.execute("SELECT REQ_STATUS, COUNT(*) as cnt FROM request GROUP BY REQ_STATUS")
    stats = {r["REQ_STATUS"]: r["cnt"] for r in cur.fetchall()}
    cur2 = conn.execute("SELECT COUNT(DISTINCT REQ_CLASS) as classes FROM request")
    classes = cur2.fetchone()["classes"]
    conn.close()
    return {
        "pending": stats.get("FIELDCOMPLETE", 0),
        "completed": stats.get("COMPLETE", 0),
        "cancelled": stats.get("CANCELED", 0),
        "total": sum(stats.values()),
        "job_types": classes,
    }


def _get_pending_jobs():
    conn = _get_conn()
    cur = conn.execute("""
        SELECT REQUEST_ID, REQ_STATUS, REQ_CLASS, PRIORITY, SEVERITY,
               CUST_PROB_DESCR, PLACE_ID, USER_DEF21
        FROM request WHERE REQ_STATUS = 'FIELDCOMPLETE'
        ORDER BY
          CASE WHEN PRIORITY LIKE 'P1%' THEN 1
               WHEN PRIORITY LIKE 'P2%' THEN 2
               WHEN PRIORITY LIKE 'P3%' THEN 3
               ELSE 4 END,
          REQUEST_ID DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _get_completed_jobs():
    conn = _get_conn()
    cur = conn.execute("""
        SELECT r.REQUEST_ID, r.REQ_CLASS, r.PRIORITY, r.CUST_PROB_DESCR,
               r.PLACE_ID, r.USER_DEF21,
               c.PROBLEM_CODE as P, c.CAUSE_CODE as C,
               c.RECTIFY_CODE as R, c.METHOD_CODE as M
        FROM request r
        LEFT JOIN close_out c ON r.REQUEST_ID = c.REQUEST_ID
        WHERE r.REQ_STATUS = 'COMPLETE'
        ORDER BY r.REQUEST_ID DESC LIMIT 50
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _get_job(request_id: int):
    conn = _get_conn()
    cur = conn.execute("SELECT * FROM request WHERE REQUEST_ID = ?", [request_id])
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def _priority_label(p: str) -> str:
    if not p:
        return "—"
    mapping = {
        "P1_URG_1H_4H": "P1 Urgent",
        "P2_HIGH_6H_1D": "P2 High",
        "P3_MED_2D_5D": "P3 Medium",
        "P4_PLAN_10D": "P4 Planned",
        "P8_PLAN_12M": "P8 Scheduled",
        "P10_PLAN_7D": "P10 Planned",
    }
    return mapping.get(p, p)


def _priority_class(p: str) -> str:
    if not p:
        return ""
    if "P1" in p:
        return "priority-urgent"
    if "P2" in p:
        return "priority-high"
    if "P3" in p:
        return "priority-medium"
    return "priority-low"


FIELD_LABELS = {
    "problem_code": "what the problem was",
    "cause_code": "what caused it",
    "rectify_code": "how you resolved it",
    "method_code": "the method or equipment you used",
}


def _build_followup(result: dict, state: dict) -> str:
    """Build a progressive human-worded follow-up question."""
    confi = result.get("field_confidence", {})
    missing = [f for f in FIELD_LABELS if not result.get(f) or confi.get(f, 0) < 0.6]

    if result.get("needs_clarification", {}) and result["needs_clarification"].get("question"):
        return result["needs_clarification"]["question"]

    if not missing:
        return "Can you tell me a bit more about what happened?"

    if len(missing) == 1:
        return f"Nearly there — can you tell me {FIELD_LABELS[missing[0]]}?"
    elif len(missing) == 2:
        return f"Thanks for that. Can you also tell me {FIELD_LABELS[missing[0]]} and {FIELD_LABELS[missing[1]]}?"
    else:
        parts = [FIELD_LABELS[f] for f in missing[:3]]
        return f"I need a few more details — {', '.join(parts[:-1])}, and {parts[-1]}?"


@weave.op(name="Vera-Voice-Intake")
def stage_vera(message: str) -> dict:
    """Vera: Captures technician speech and converts to text."""
    return vera.predict(text=message)


@weave.op(name="LangGraph-Orchestrator")
def stage_orchestrator(message: str, complaint: str | None, transcript: str) -> dict:
    """LangGraph: Routes intent, builds context for structuring."""
    full_transcript = (transcript or "") + " " + message
    context_text = full_transcript
    if complaint:
        context_text = f"Reported problem: {complaint}\n\nTechnician close-out: {full_transcript}"
    return {
        "context_text": context_text,
        "full_transcript": full_transcript,
        "route": "structure → validate → close",
    }


@weave.op(name="GEMI-Structuring")
def stage_gemi(context_text: str) -> dict:
    """GEMI: Maps technician speech to PCRM work codes via LLM."""
    return gemi.structure(context_text)


@weave.op(name="Hade-Validation")
def stage_hade(context_text: str, gemi_output: dict) -> dict:
    """Hade: Validates codes, cross-checks semantics, corrects errors."""
    return hade.validate_and_correct(context_text, gemi_output, gemi_agent=gemi)


@weave.op(name="WFM-Close-Out")
def stage_wfm_close(request_id: int, codes: dict) -> dict:
    """WFM: Writes validated close-out codes to the work management system."""
    return hade.close(request_id, codes)


@weave.op(name="Mainline-Pipeline")
def pipeline_turn(message: str, request_id: int | None, complaint: str | None, transcript: str) -> dict:
    """Full agent pipeline — each stage is a separate traced operation in Weave.

    Vera → LangGraph → GEMI → Hade → WFM
    """
    # Stage 1: Vera — voice/text intake
    vera_output = stage_vera(message)
    processed_text = vera_output.get("transcript", message)

    # Stage 2: LangGraph Orchestrator — route and build context
    orch_output = stage_orchestrator(processed_text, complaint, transcript)
    context_text = orch_output["context_text"]
    full_transcript = orch_output["full_transcript"]

    # Stage 3: GEMI — structure into PCRM codes
    gemi_output = stage_gemi(context_text)

    if gemi_output.get("needs_clarification"):
        return {
            "stage": "clarify",
            "gemi_output": gemi_output,
            "transcript": full_transcript,
        }

    # Stage 4: Hade — validate and correct
    hade_output = stage_hade(context_text, gemi_output)
    codes = {
        "problem_code": hade_output.get("problem_code"),
        "cause_code": hade_output.get("cause_code"),
        "rectify_code": hade_output.get("rectify_code"),
        "method_code": hade_output.get("method_code"),
    }

    # Stage 5: WFM — close the job
    if request_id:
        stage_wfm_close(request_id, codes)

    return {
        "stage": "confirm",
        "codes": codes,
        "transcript": full_transcript,
        "hade_approved": hade_output.get("_hade_approved", True),
        "agent_chain": "Vera → LangGraph → GEMI → Hade → WFM",
    }


@weave.op(name="LangGraph-Job-Resolution")
def stage_resolve_job(message: str) -> dict:
    """LangGraph: Classify intent and resolve job from WFM."""
    intent = classify_intent(message)
    results = hade.search(message) if intent in (Intent.CLOSE_OUT, Intent.NEW_FAULT) else []
    return {
        "intent": intent.value,
        "results": results,
        "resolved": bool(results),
    }


def _advance(state: dict, message: str) -> dict:
    """Advance the conversation state.

    Progressive clarification strategy:
    - Track how many confident fields we have (0-4)
    - Only escalate after 3+ stalled rounds (user answered but no new info gained)
    - Safety cap at 10 turns to avoid infinite loops
    - Ask targeted questions about specific missing fields
    """
    state["convo"].append(["you", message])
    state["transcript"] = (state.get("transcript") or "") + " " + message
    state["turn"] = state.get("turn", 0) + 1

    if not state.get("request_id"):
        resolution = stage_resolve_job(message)
        if resolution["intent"] not in (Intent.CLOSE_OUT.value, Intent.NEW_FAULT.value):
            state["convo"].append(["agent", "I can help you close out a job or report a new fault. Which would you like to do?"])
            return state

        if resolution["results"]:
            state["request_id"] = resolution["results"][0]["REQUEST_ID"]
            state["complaint"] = resolution["results"][0].get("CUST_PROB_DESCR", "")
            state["convo"].append(["agent", f"Found job #{state['request_id']} — reported as: \"{state['complaint']}\". What did you find on site, and how did you fix it?"])
        else:
            state["convo"].append(["agent", "Tell me about the job — what was the problem, what caused it, and how did you fix it?"])
        return state

    # Run the full traced pipeline: Vera → GEMI → Hade → WFM
    pipeline_result = pipeline_turn(
        message=message,
        request_id=state.get("request_id"),
        complaint=state.get("complaint"),
        transcript=state.get("transcript", ""),
    )

    if pipeline_result["stage"] == "clarify":
        result = pipeline_result["gemi_output"]
        filled = sum(1 for f in ["problem_code", "cause_code", "rectify_code", "method_code"]
                     if result.get(f) and result.get("field_confidence", {}).get(f, 0) >= 0.6)
        best = state.get("best", 0)
        if filled > best:
            state["best"] = filled
            state["stall"] = 0
        else:
            state["stall"] = state.get("stall", 0) + 1

        if state.get("stall", 0) >= 3 or state["turn"] > 10:
            state["convo"].append(["agent",
                "I've asked a few times but I'm still missing key details to close this out. "
                "I'll pass this to the team so they can follow up with you directly."])
            state["outcome"] = "ESCALATE"
            return state

        followup = _build_followup(result, state)
        state["convo"].append(["agent", followup])
        return state

    # Confirmed — job closed
    codes = pipeline_result["codes"]
    confirm_msg = (
        f"All set — closed job #{state.get('request_id', 'new')}. "
        f"Problem: {_code_label(codes['problem_code'])}, "
        f"Cause: {_code_label(codes['cause_code'])}, "
        f"Rectify: {_code_label(codes['rectify_code'])}, "
        f"Method: {_code_label(codes['method_code'])}."
    )
    state["convo"].append(["agent", confirm_msg])
    state["outcome"] = "CONFIRM"
    state["codes"] = codes
    return state


# ─── Routes ────────────────────────────────────────────────────────────────────

async def homepage(request: Request):
    return HTMLResponse(_render_dashboard())


async def chat_page(request: Request):
    request_id = request.query_params.get("id")
    job = None
    if request_id:
        job = _get_job(int(request_id))
    return HTMLResponse(_render_chat(job))


async def turn_endpoint(request: Request):
    body = await request.json()
    state = body.get("state", {"convo": [], "transcript": "", "turn": 0})
    message = body.get("message", "")
    if not message.strip():
        return JSONResponse({"state": state})
    updated = await run_in_threadpool(_advance, state, message)
    return JSONResponse({"state": updated})


async def api_jobs(request: Request):
    status = request.query_params.get("status", "pending")
    if status == "completed":
        return JSONResponse({"jobs": _get_completed_jobs()})
    return JSONResponse({"jobs": _get_pending_jobs()})


# ─── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #F8FAFC;
  --surface: #FFFFFF;
  --panel: #F1F5F9;
  --ink: #0F172A;
  --secondary: #475569;
  --muted: #94A3B8;
  --accent: #2563EB;
  --accent-hover: #1D4ED8;
  --accent-light: #EFF6FF;
  --success: #059669;
  --success-bg: #ECFDF5;
  --warning: #D97706;
  --warning-bg: #FFFBEB;
  --danger: #DC2626;
  --danger-bg: #FEF2F2;
  --border: #E2E8F0;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
  --radius: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: var(--ink); min-height: 100vh; line-height: 1.6; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Layout */
.layout { display: flex; min-height: 100vh; }
.sidebar {
  width: 260px; background: var(--ink); color: white;
  padding: 24px 16px; display: flex; flex-direction: column;
  position: fixed; top: 0; left: 0; bottom: 0; z-index: 100;
}
.sidebar .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 4px; }
.sidebar .subtitle { font-size: 0.7rem; color: var(--muted); margin-bottom: 36px; text-transform: uppercase; letter-spacing: 0.08em; }
.sidebar nav { display: flex; flex-direction: column; gap: 2px; flex: 1; }
.sidebar nav a {
  display: flex; align-items: center; gap: 12px; padding: 11px 14px;
  border-radius: var(--radius); color: rgba(255,255,255,0.6); font-size: 0.88rem;
  font-weight: 500; transition: all 0.15s;
}
.sidebar nav a:hover { background: rgba(255,255,255,0.08); color: white; text-decoration: none; }
.sidebar nav a.active { background: var(--accent); color: white; }
.sidebar nav a .icon { font-size: 1rem; width: 20px; text-align: center; opacity: 0.8; }
.sidebar nav a .badge {
  margin-left: auto; background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.9);
  font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600;
}
.sidebar nav a.active .badge { background: rgba(255,255,255,0.25); }
.sidebar .footer { margin-top: auto; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.1); }
.sidebar .footer a { font-size: 0.78rem; color: rgba(255,255,255,0.5); display: block; padding: 6px 0; }
.sidebar .footer a:hover { color: white; text-decoration: none; }

.main { margin-left: 260px; flex: 1; padding: 32px 40px; }
.page-header { margin-bottom: 28px; }
.page-header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
.page-header p { color: var(--secondary); font-size: 0.9rem; margin-top: 4px; }

/* Stats */
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
.stat-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: 20px; box-shadow: var(--shadow-sm);
}
.stat-card .stat-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 8px; }
.stat-card .stat-value { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
.stat-card .stat-sub { font-size: 0.78rem; color: var(--secondary); margin-top: 4px; }
.stat-card.urgent .stat-value { color: var(--danger); }
.stat-card.pending .stat-value { color: var(--warning); }
.stat-card.complete .stat-value { color: var(--success); }

/* Tabs */
.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.section-header h2 { font-size: 1.1rem; font-weight: 600; }
.tabs { display: flex; gap: 4px; background: var(--panel); padding: 4px; border-radius: var(--radius); }
.tab {
  padding: 7px 14px; border-radius: 6px; font-size: 0.82rem; font-weight: 500;
  cursor: pointer; border: none; background: none; color: var(--secondary); transition: all 0.15s;
}
.tab:hover { color: var(--ink); }
.tab.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow-sm); }

/* Table */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); box-shadow: var(--shadow-sm); overflow: hidden; }
table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
thead th {
  text-align: left; padding: 12px 16px; font-weight: 600; font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
  border-bottom: 1px solid var(--border); background: var(--panel);
}
tbody tr { border-bottom: 1px solid var(--border); transition: background 0.1s; }
tbody tr:hover { background: var(--accent-light); }
tbody tr:last-child { border-bottom: none; }
tbody td { padding: 14px 16px; vertical-align: middle; }
.clickable { cursor: pointer; }

/* Badges */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600;
}
.badge-pending { background: var(--warning-bg); color: var(--warning); }
.badge-complete { background: var(--success-bg); color: var(--success); }
.badge-cancelled { background: var(--panel); color: var(--muted); }
.priority-urgent { background: var(--danger-bg); color: var(--danger); }
.priority-high { background: #FFF7ED; color: #C2410C; }
.priority-medium { background: var(--accent-light); color: var(--accent); }
.priority-low { background: var(--panel); color: var(--secondary); }
.badge-type { background: var(--panel); color: var(--secondary); border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; }

/* Buttons */
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 14px; border-radius: var(--radius); font-size: 0.82rem;
  font-weight: 500; cursor: pointer; border: none; transition: all 0.15s; text-decoration: none;
}
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: var(--accent-hover); text-decoration: none; }
.btn-ghost { background: none; border: 1px solid var(--border); color: var(--secondary); }
.btn-ghost:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }

.empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; opacity: 0.5; }
.empty-state h3 { font-size: 1rem; color: var(--ink); margin-bottom: 4px; }
.empty-state p { font-size: 0.85rem; }

/* Agent Workflow */
.workflow-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 24px 28px; box-shadow: var(--shadow-sm); margin-bottom: 28px; }
.workflow-card h3 { font-size: 0.92rem; font-weight: 700; margin-bottom: 18px; }
.workflow-pipeline { display: flex; align-items: center; gap: 0; justify-content: center; flex-wrap: wrap; }
.workflow-node {
  display: flex; flex-direction: column; align-items: center; gap: 6px;
  padding: 14px 18px; border-radius: var(--radius-lg); background: var(--panel);
  border: 1px solid var(--border); min-width: 120px; text-align: center;
  transition: all 0.2s;
}
.workflow-node:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: var(--shadow-md); }
.workflow-node .node-icon { font-size: 1.4rem; }
.workflow-node .node-name { font-size: 0.82rem; font-weight: 700; color: var(--ink); }
.workflow-node .node-role { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.workflow-node.vera { border-left: 3px solid #7C3AED; }
.workflow-node.gemi { border-left: 3px solid #2563EB; }
.workflow-node.hade { border-left: 3px solid #059669; }
.workflow-node.wfm { border-left: 3px solid #D97706; }
.workflow-node.orch { border-left: 3px solid #DC2626; }
.workflow-arrow { color: var(--muted); font-size: 1.2rem; padding: 0 8px; }
.workflow-sub { font-size: 0.72rem; color: var(--secondary); margin-top: 14px; text-align: center; line-height: 1.7; }
.workflow-sub code { background: var(--panel); padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; }

.code-group { display: flex; gap: 4px; flex-wrap: wrap; }
.code-pill { background: var(--panel); border-radius: 4px; padding: 2px 7px; font-size: 0.72rem; font-family: 'SF Mono', monospace; color: var(--secondary); white-space: nowrap; }

.desc-text { color: var(--secondary); max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Chat */
.chat-layout { display: flex; gap: 24px; height: calc(100vh - 140px); }
.chat-main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.chat-sidebar-panel { width: 300px; flex-shrink: 0; }

.job-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: 20px; box-shadow: var(--shadow-sm);
}
.job-card .job-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.job-card .job-card-header h3 { font-size: 0.95rem; font-weight: 700; }
.job-card .field { margin-bottom: 14px; }
.job-card .field-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; margin-bottom: 3px; }
.job-card .field-value { font-size: 0.88rem; color: var(--ink); }
.job-card .divider { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

.chat-box {
  flex: 1; overflow-y: auto; padding: 24px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-xl) var(--radius-xl) 0 0;
  display: flex; flex-direction: column; gap: 16px;
}
.bubble {
  max-width: 75%; padding: 12px 18px; border-radius: 18px;
  font-size: 0.9rem; line-height: 1.6; position: relative;
  animation: fadeIn 0.2s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
.bubble.agent {
  align-self: flex-start; background: var(--panel); color: var(--ink);
  border-bottom-left-radius: 4px;
}
.bubble.you {
  align-self: flex-end; background: var(--accent); color: white;
  border-bottom-right-radius: 4px;
}
.bubble .replay-btn {
  position: absolute; top: 6px; right: 10px; background: none; border: none;
  cursor: pointer; font-size: 0.7rem; opacity: 0; transition: opacity 0.15s;
}
.bubble:hover .replay-btn { opacity: 0.7; }
.bubble .replay-btn:hover { opacity: 1; }
.bubble.agent .replay-btn { color: var(--secondary); }

.input-bar {
  display: flex; gap: 10px; align-items: center;
  padding: 16px 20px; background: var(--surface);
  border: 1px solid var(--border); border-top: none;
  border-radius: 0 0 var(--radius-xl) var(--radius-xl);
}
.input-bar textarea {
  flex: 1; resize: none; border: 1px solid var(--border); border-radius: 24px;
  padding: 12px 20px; font-size: 0.9rem; font-family: inherit;
  min-height: 44px; max-height: 100px; outline: none; line-height: 1.4;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.input-bar textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
.btn-mic {
  width: 44px; height: 44px; border-radius: 50%; border: 2px solid var(--border);
  background: var(--surface); color: var(--secondary); font-size: 1.1rem;
  cursor: pointer; transition: all 0.15s; display: flex; align-items: center; justify-content: center;
}
.btn-mic:hover { border-color: var(--accent); color: var(--accent); }
.btn-mic.active { border-color: var(--danger); background: var(--danger); color: white; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
.btn-send {
  width: 44px; height: 44px; border-radius: 50%; border: none;
  background: var(--accent); color: white; font-size: 1.1rem;
  cursor: pointer; transition: all 0.15s; display: flex; align-items: center; justify-content: center;
}
.btn-send:hover { background: var(--accent-hover); transform: scale(1.05); }
.btn-send:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

.outcome-banner {
  padding: 16px 20px; border-radius: var(--radius-lg); margin-top: 12px;
  display: flex; align-items: center; gap: 12px; font-weight: 500; font-size: 0.9rem;
}
.outcome-banner.confirm { background: var(--success-bg); color: var(--success); border: 1px solid #A7F3D0; }
.outcome-banner.escalate { background: var(--warning-bg); color: var(--warning); border: 1px solid #FDE68A; }
.outcome-banner a { font-weight: 600; }

.status-line { font-size: 0.78rem; color: var(--muted); text-align: center; padding: 8px; min-height: 28px; }
"""


def _render_dashboard():
    pending = _get_pending_jobs()
    completed = _get_completed_jobs()
    stats = _get_stats()

    urgent_count = sum(1 for j in pending if "P1" in (j.get("PRIORITY") or ""))

    pending_rows = ""
    for job in pending:
        p_cls = _priority_class(job.get("PRIORITY"))
        pending_rows += f"""
        <tr class="clickable" onclick="window.location='/chat?id={job['REQUEST_ID']}'">
          <td><strong>#{job['REQUEST_ID']}</strong></td>
          <td><span class="badge-type">{job.get('REQ_CLASS','—')}</span></td>
          <td><span class="badge {p_cls}">{_priority_label(job.get('PRIORITY'))}</span></td>
          <td><div class="desc-text">{(job.get('CUST_PROB_DESCR') or '—')[:90]}</div></td>
          <td>{job.get('USER_DEF21','—').replace('_',' ').title() if job.get('USER_DEF21') else '—'}</td>
          <td><a href="/chat?id={job['REQUEST_ID']}" class="btn btn-primary">Close out &#8250;</a></td>
        </tr>"""

    completed_rows = ""
    for job in completed[:30]:
        codes_html = ""
        if job.get("P"):
            codes_html = f"""<div class="code-group">
              <span class="code-pill" title="{_code_label(job['P'])}">{job['P']}</span>
              <span class="code-pill" title="{_code_label(job.get('C'))}">{job.get('C','')}</span>
              <span class="code-pill" title="{_code_label(job.get('R'))}">{job.get('R','')}</span>
              <span class="code-pill" title="{_code_label(job.get('M'))}">{job.get('M','')}</span>
            </div>"""
        completed_rows += f"""
        <tr>
          <td><strong>#{job['REQUEST_ID']}</strong></td>
          <td><span class="badge-type">{job.get('REQ_CLASS','—')}</span></td>
          <td><div class="desc-text">{(job.get('CUST_PROB_DESCR') or '—')[:70]}</div></td>
          <td>{codes_html or '<span style="color:var(--muted)">—</span>'}</td>
          <td><span class="badge badge-complete">Complete</span></td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mainline — Dashboard</title><style>{CSS}</style></head><body>
<div class="layout">
  {_sidebar_html('dashboard')}
  <div class="main">
    <div class="page-header">
      <h1>Dashboard</h1>
      <p>A direct line from the crew's voice to the work record</p>
    </div>

    <div class="stats-grid">
      <div class="stat-card urgent">
        <div class="stat-label">Urgent (P1)</div>
        <div class="stat-value">{urgent_count}</div>
        <div class="stat-sub">Requires immediate close-out</div>
      </div>
      <div class="stat-card pending">
        <div class="stat-label">Awaiting Close-out</div>
        <div class="stat-value">{stats['pending']}</div>
        <div class="stat-sub">Field complete, need codes</div>
      </div>
      <div class="stat-card complete">
        <div class="stat-label">Completed</div>
        <div class="stat-value">{stats['completed']}</div>
        <div class="stat-sub">Successfully closed</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Job Types</div>
        <div class="stat-value">{stats['job_types']}</div>
        <div class="stat-sub">Across {stats['total']} total records</div>
      </div>
    </div>

    <div class="workflow-card">
      <h3>Agent Pipeline</h3>
      <div class="workflow-pipeline">
        <div class="workflow-node vera">
          <div class="node-icon">&#127908;</div>
          <div class="node-name">Vera</div>
          <div class="node-role">Voice Agent</div>
        </div>
        <div class="workflow-arrow">&#8594;</div>
        <div class="workflow-node orch">
          <div class="node-icon">&#9881;</div>
          <div class="node-name">LangGraph</div>
          <div class="node-role">Orchestrator</div>
        </div>
        <div class="workflow-arrow">&#8594;</div>
        <div class="workflow-node gemi">
          <div class="node-icon">&#9889;</div>
          <div class="node-name">GEMI</div>
          <div class="node-role">Structuring</div>
        </div>
        <div class="workflow-arrow">&#8594;</div>
        <div class="workflow-node hade">
          <div class="node-icon">&#9989;</div>
          <div class="node-name">Hade</div>
          <div class="node-role">Validation</div>
        </div>
        <div class="workflow-arrow">&#8594;</div>
        <div class="workflow-node wfm">
          <div class="node-icon">&#128451;</div>
          <div class="node-name">WFM</div>
          <div class="node-role">Work System</div>
        </div>
      </div>
      <div class="workflow-sub">
        <code>Vera</code> captures speech &rarr; <code>LangGraph</code> routes intent &rarr; <code>GEMI</code> maps to PCRM codes &rarr; <code>Hade</code> validates &amp; corrects &rarr; <code>WFM</code> closes the job<br>
        All agents traced via <strong>W&amp;B Weave</strong> &middot; Model: <code>Qwen3-14B</code> via W&amp;B Inference
      </div>
    </div>

    <div class="section-header">
      <h2>Tickets</h2>
      <div class="tabs">
        <button class="tab active" onclick="showTab('pending')">Pending ({len(pending)})</button>
        <button class="tab" onclick="showTab('completed')">Completed ({len(completed)})</button>
      </div>
    </div>

    <div id="tab-pending" class="card">
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>ID</th><th>Type</th><th>Priority</th><th>Description</th><th>Location</th><th></th></tr></thead>
          <tbody>{pending_rows if pending_rows else '<tr><td colspan="6"><div class="empty-state"><div class="icon">&#9989;</div><h3>All caught up!</h3><p>No pending close-outs</p></div></td></tr>'}</tbody>
        </table>
      </div>
    </div>

    <div id="tab-completed" class="card" style="display:none">
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>ID</th><th>Type</th><th>Description</th><th>PCRM Codes</th><th>Status</th></tr></thead>
          <tbody>{completed_rows if completed_rows else '<tr><td colspan="5"><div class="empty-state"><div class="icon">&#128196;</div><h3>No completed jobs</h3></div></td></tr>'}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>
<script>
function showTab(tab) {{
  document.getElementById('tab-pending').style.display = tab==='pending'?'block':'none';
  document.getElementById('tab-completed').style.display = tab==='completed'?'block':'none';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
}}
</script>
</body></html>"""


def _sidebar_html(active: str):
    pending_count = len(_get_pending_jobs())
    return f"""
  <div class="sidebar">
    <div class="brand">Mainline</div>
    <div class="subtitle">Voice Close-Out Platform</div>
    <nav>
      <a href="/" class="{'active' if active=='dashboard' else ''}">
        <span class="icon">&#9776;</span> Dashboard
        <span class="badge">{pending_count}</span>
      </a>
      <a href="/chat" class="{'active' if active=='chat' else ''}">
        <span class="icon">&#127908;</span> Voice Close-Out
      </a>
    </nav>
    <div class="footer">
      <a href="https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/weave" target="_blank">Weave Traces &#8599;</a>
      <a href="https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}" target="_blank">W&B Metrics &#8599;</a>
    </div>
  </div>"""


def _render_chat(job: dict | None):
    job_sidebar = ""
    initial_state = '{ "convo": [], "transcript": "", "turn": 0 }'
    greeting = "G'day! I'm your field service voice agent. Are you closing out a job, or reporting a new fault? Click the mic and tell me what you did."

    if job:
        initial_state = json.dumps({
            "convo": [],
            "transcript": "",
            "turn": 0,
            "request_id": job["REQUEST_ID"],
            "complaint": job.get("CUST_PROB_DESCR", ""),
        })
        greeting = f"G'day! I can see job #{job['REQUEST_ID']} — reported as: \\&quot;{(job.get('CUST_PROB_DESCR') or '').replace(chr(34), '')}\\&quot;. What did you find on site, and how did you fix it?"
        job_sidebar = f"""
    <div class="chat-sidebar-panel">
      <div class="job-card">
        <div class="job-card-header">
          <h3>Job #{job['REQUEST_ID']}</h3>
          <span class="badge badge-pending">Pending</span>
        </div>
        <div class="field"><div class="field-label">Type</div><div class="field-value">{job.get('REQ_CLASS','—')}</div></div>
        <div class="field"><div class="field-label">Priority</div><div class="field-value"><span class="badge {_priority_class(job.get('PRIORITY'))}">{_priority_label(job.get('PRIORITY'))}</span></div></div>
        <hr class="divider">
        <div class="field"><div class="field-label">Reported Problem</div><div class="field-value">{job.get('CUST_PROB_DESCR','—')}</div></div>
        <div class="field"><div class="field-label">Location Type</div><div class="field-value">{(job.get('USER_DEF21') or '—').replace('_',' ').title()}</div></div>
        <div class="field"><div class="field-label">Coordinates</div><div class="field-value" style="font-size:0.78rem;color:var(--muted)">{job.get('PLACE_ID','—')}</div></div>
      </div>
    </div>"""

    # Escape greeting for JS
    greeting_escaped = greeting.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{'Close Out #' + str(job['REQUEST_ID']) if job else 'Voice Close-Out'} — Mainline</title><style>{CSS}</style></head><body>
<div class="layout">
  {_sidebar_html('chat')}
  <div class="main">
    <div class="page-header">
      <h1>{'Close Out Job #' + str(job['REQUEST_ID']) if job else 'Voice Close-Out'}</h1>
      <p>Speak naturally — the mic stays on until you click stop. Each recording starts fresh.</p>
    </div>

    <div class="chat-layout">
      <div class="chat-main">
        <div class="chat-box" id="chatBox"></div>
        <div class="input-bar" id="inputBar">
          <textarea id="userInput" rows="1" placeholder="Describe what you did on site..."></textarea>
          <button class="btn-mic" id="micBtn" title="Click to start speaking">&#127908;</button>
          <button class="btn-send" id="sendBtn" title="Send">&#10148;</button>
        </div>
        <div class="status-line" id="status"></div>
        <div id="outcome"></div>
      </div>
      {job_sidebar}
    </div>
  </div>
</div>

<script>
const chatBox = document.getElementById('chatBox');
const input = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const statusEl = document.getElementById('status');
const outcomeEl = document.getElementById('outcome');
const inputBar = document.getElementById('inputBar');

let state = {initial_state};
let speaking = false;
let recognition = null;
let listening = false;
let finalTranscript = '';

function initSpeechRecognition() {{
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {{ statusEl.textContent = 'Speech not supported — type your response instead'; return; }}

  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-AU';

  recognition.onresult = (e) => {{
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {{
      if (e.results[i].isFinal) {{
        finalTranscript += e.results[i][0].transcript + ' ';
      }} else {{
        interim += e.results[i][0].transcript;
      }}
    }}
    input.value = finalTranscript + interim;
    autoResize();
  }};

  recognition.onend = () => {{
    if (listening) recognition.start();
  }};

  recognition.onerror = (e) => {{
    if (e.error === 'no-speech') return;
    stopListening();
  }};
}}

function startListening() {{
  if (!recognition) return;
  // RESET transcript each time mic is clicked — fresh start
  finalTranscript = '';
  input.value = '';
  listening = true;
  recognition.start();
  micBtn.classList.add('active');
  micBtn.innerHTML = '&#9632;';
  micBtn.title = 'Click to stop';
  statusEl.textContent = 'Listening... speak naturally, click stop when done';
}}

function stopListening() {{
  listening = false;
  if (recognition) recognition.stop();
  micBtn.classList.remove('active');
  micBtn.innerHTML = '&#127908;';
  micBtn.title = 'Click to start speaking';
  statusEl.textContent = '';
  const text = input.value.trim();
  if (text) {{
    sendMessage(text);
    input.value = '';
    finalTranscript = '';
  }}
}}

micBtn.addEventListener('click', () => {{
  if (listening) stopListening(); else startListening();
}});

sendBtn.addEventListener('click', () => {{
  const text = input.value.trim();
  if (text) {{ sendMessage(text); input.value = ''; finalTranscript = ''; }}
}});

input.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    const text = input.value.trim();
    if (text) {{ sendMessage(text); input.value = ''; finalTranscript = ''; }}
  }}
}});

function autoResize() {{
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 100) + 'px';
}}
input.addEventListener('input', autoResize);

async function sendMessage(text) {{
  addBubble('you', text);
  statusEl.textContent = 'Processing...';
  sendBtn.disabled = true;

  try {{
    const resp = await fetch('/turn', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ state, message: text }})
    }});
    const data = await resp.json();
    state = data.state;
    renderNewMessages();
  }} catch (err) {{
    addBubble('agent', 'Connection error. Please try again.');
  }}
  sendBtn.disabled = false;
  statusEl.textContent = '';
}}

let renderedCount = 0;
function renderNewMessages() {{
  while (renderedCount < state.convo.length) {{
    const [role, text] = state.convo[renderedCount];
    if (role === 'agent') {{
      addBubble('agent', text);
      speakText(text);
    }}
    renderedCount++;
  }}

  if (state.outcome === 'CONFIRM') {{
    outcomeEl.innerHTML = '<div class="outcome-banner confirm">&#9989; Job closed successfully &mdash; <a href="/">Back to dashboard</a></div>';
    inputBar.style.display = 'none';
  }} else if (state.outcome === 'ESCALATE') {{
    outcomeEl.innerHTML = '<div class="outcome-banner escalate">&#9888;&#65039; Escalated to supervisor &mdash; <a href="/">Back to dashboard</a></div>';
    inputBar.style.display = 'none';
  }}

  chatBox.scrollTop = chatBox.scrollHeight;
}}

function addBubble(role, text) {{
  const div = document.createElement('div');
  div.className = 'bubble ' + role;
  const span = document.createElement('span');
  span.textContent = text;
  div.appendChild(span);
  if (role === 'agent') {{
    const btn = document.createElement('button');
    btn.className = 'replay-btn';
    btn.textContent = '\\uD83D\\uDD0A';
    btn.onclick = (e) => {{ e.stopPropagation(); speakText(text); }};
    div.appendChild(btn);
  }}
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}}

function speakText(text) {{
  if (speaking) speechSynthesis.cancel();
  // Pause mic while agent speaks so it doesn't record TTS output
  let wasListening = listening;
  if (listening) {{
    listening = false;
    if (recognition) recognition.stop();
    micBtn.classList.remove('active');
    micBtn.innerHTML = '&#127908;';
    statusEl.textContent = 'Agent speaking...';
  }}
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.onstart = () => {{ speaking = true; }};
  utter.onend = () => {{
    speaking = false;
    // Resume mic if it was active before agent spoke
    if (wasListening) {{
      finalTranscript = input.value;
      listening = true;
      recognition.start();
      micBtn.classList.add('active');
      micBtn.innerHTML = '&#9632;';
      statusEl.textContent = 'Listening... speak naturally, click stop when done';
    }} else {{
      statusEl.textContent = '';
    }}
  }};
  speechSynthesis.speak(utter);
}}

// Init
initSpeechRecognition();
const greetingText = `{greeting_escaped}`;
addBubble('agent', greetingText);
renderedCount = 0;
speakText(greetingText);
</script>
</body></html>"""


# ─── App ───────────────────────────────────────────────────────────────────────

app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/chat", chat_page),
        Route("/turn", turn_endpoint, methods=["POST"]),
        Route("/api/jobs", api_jobs),
    ],
)


if __name__ == "__main__":
    weave.init(WANDB_FULL_PROJECT)
    print(f"\n  Mainline — http://127.0.0.1:8000")
    print(f"  A direct line from the crew's voice to the work record")
    print(f"  Weave: https://wandb.ai/{WANDB_FULL_PROJECT}/weave\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
