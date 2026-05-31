"""Web frontend — Starlette + Uvicorn.

Multi-page app:
  /           — Dashboard (pending + completed tabs)
  /chat?id=X  — Chat for a specific ticket (continuous voice)
  /chat       — Free-form chat (no ticket pre-selected)
  /turn       — AJAX endpoint for conversation turns

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
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
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
    codes = _load_codes()
    for category in codes.values():
        for item in category:
            if item["code"] == code:
                return item["label"]
    return code or "—"


def _get_pending_jobs():
    conn = _get_conn()
    cur = conn.execute("""
        SELECT REQUEST_ID, REQ_STATUS, REQ_CLASS, PRIORITY, CUST_PROB_DESCR, PLACE_ID
        FROM request WHERE REQ_STATUS = 'FIELDCOMPLETE'
        ORDER BY REQUEST_ID DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _get_completed_jobs():
    conn = _get_conn()
    cur = conn.execute("""
        SELECT r.REQUEST_ID, r.REQ_STATUS, r.REQ_CLASS, r.PRIORITY,
               r.CUST_PROB_DESCR, r.PROBLEM_CODE, r.PLACE_ID,
               c.PROBLEM_CODE as P, c.CAUSE_CODE as C, c.RECTIFY_CODE as R, c.METHOD_CODE as M
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


def _advance(state: dict, message: str) -> dict:
    """Advance the conversation state."""
    state["convo"].append(["you", message])
    state["transcript"] = (state.get("transcript") or "") + " " + message
    state["turn"] = state.get("turn", 0) + 1

    if not state.get("request_id"):
        intent = classify_intent(message)
        if intent not in (Intent.CLOSE_OUT, Intent.NEW_FAULT):
            state["convo"].append(["agent", "I can help you close out a job or report a new fault. Which would you like to do?"])
            return state

        results = hade.search(message)
        if results:
            state["request_id"] = results[0]["REQUEST_ID"]
            state["complaint"] = results[0].get("CUST_PROB_DESCR", "")
            state["convo"].append(["agent", f"Found job #{state['request_id']} — reported as: \"{state['complaint']}\". What did you find on site, and how did you fix it?"])
        else:
            state["convo"].append(["agent", "Got it. Tell me about the job — what was the problem, what caused it, and how did you fix it?"])
        return state

    context_text = state["transcript"]
    if state.get("complaint"):
        context_text = f"Reported problem: {state['complaint']}\n\nTechnician close-out: {state['transcript']}"

    result = gemi.structure(context_text)

    if result.get("needs_clarification"):
        filled = sum(1 for f in ["problem_code", "cause_code", "rectify_code", "method_code"]
                     if result.get(f) and result.get("field_confidence", {}).get(f, 0) >= 0.6)
        best = state.get("best", 0)
        if filled > best:
            state["best"] = filled
            state["stall"] = 0
        else:
            state["stall"] = state.get("stall", 0) + 1

        if state.get("stall", 0) >= 2 or state["turn"] > 6:
            state["convo"].append(["agent", "I don't have enough detail to close this one out. I'll escalate it to the team."])
            state["outcome"] = "ESCALATE"
            return state

        question = result["needs_clarification"].get("question", "Can you tell me more?")
        state["convo"].append(["agent", f"Thanks. {question}"])
        return state

    final = hade.validate_and_correct(context_text, result, gemi_agent=gemi)
    codes = {
        "problem_code": final.get("problem_code"),
        "cause_code": final.get("cause_code"),
        "rectify_code": final.get("rectify_code"),
        "method_code": final.get("method_code"),
    }

    if state.get("request_id"):
        hade.close(state["request_id"], codes)

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


async def turn(request: Request):
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


# ─── HTML Templates ────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #FAFAFA;
  --surface: #FFFFFF;
  --panel: #F5F5F5;
  --ink: #1A1A1A;
  --muted: #6B7280;
  --accent: #2563EB;
  --accent-hover: #1D4ED8;
  --success: #059669;
  --success-bg: #ECFDF5;
  --warning: #D97706;
  --warning-bg: #FFFBEB;
  --danger: #DC2626;
  --danger-bg: #FEF2F2;
  --border: #E5E7EB;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-lg: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
  --radius: 8px;
  --radius-lg: 12px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--ink);
  min-height: 100vh;
  line-height: 1.5;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.layout { display: flex; min-height: 100vh; }
.sidebar {
  width: 260px; background: var(--surface); border-right: 1px solid var(--border);
  padding: 24px 16px; display: flex; flex-direction: column; position: fixed;
  top: 0; left: 0; bottom: 0; z-index: 100;
}
.sidebar .brand { font-size: 1.1rem; font-weight: 700; color: var(--ink); margin-bottom: 8px; }
.sidebar .subtitle { font-size: 0.75rem; color: var(--muted); margin-bottom: 32px; text-transform: uppercase; letter-spacing: 0.05em; }
.sidebar nav { display: flex; flex-direction: column; gap: 4px; flex: 1; }
.sidebar nav a {
  display: flex; align-items: center; gap: 10px; padding: 10px 12px;
  border-radius: var(--radius); color: var(--muted); font-size: 0.9rem;
  font-weight: 500; transition: all 0.15s;
}
.sidebar nav a:hover { background: var(--panel); color: var(--ink); text-decoration: none; }
.sidebar nav a.active { background: var(--accent); color: white; }
.sidebar nav a .icon { font-size: 1.1rem; width: 20px; text-align: center; }
.sidebar nav a .badge {
  margin-left: auto; background: var(--panel); color: var(--muted);
  font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600;
}
.sidebar nav a.active .badge { background: rgba(255,255,255,0.2); color: white; }
.sidebar .footer { margin-top: auto; padding-top: 16px; border-top: 1px solid var(--border); }
.sidebar .footer a { font-size: 0.8rem; color: var(--muted); display: block; padding: 6px 0; }

.main { margin-left: 260px; flex: 1; padding: 32px 40px; max-width: 1100px; }
.page-header { margin-bottom: 24px; }
.page-header h1 { font-size: 1.5rem; font-weight: 700; }
.page-header p { color: var(--muted); font-size: 0.9rem; margin-top: 4px; }

.tabs { display: flex; gap: 4px; margin-bottom: 24px; background: var(--panel); padding: 4px; border-radius: var(--radius); width: fit-content; }
.tab {
  padding: 8px 16px; border-radius: 6px; font-size: 0.85rem; font-weight: 500;
  cursor: pointer; border: none; background: none; color: var(--muted); transition: all 0.15s;
}
.tab:hover { color: var(--ink); }
.tab.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow); }

.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg); box-shadow: var(--shadow); overflow: hidden;
}
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
thead th {
  text-align: left; padding: 12px 16px; font-weight: 600; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted);
  border-bottom: 1px solid var(--border); background: var(--panel);
}
tbody tr { border-bottom: 1px solid var(--border); transition: background 0.1s; cursor: pointer; }
tbody tr:hover { background: var(--panel); }
tbody tr:last-child { border-bottom: none; }
tbody td { padding: 14px 16px; vertical-align: middle; }

.badge-status {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}
.badge-pending { background: var(--warning-bg); color: var(--warning); }
.badge-complete { background: var(--success-bg); color: var(--success); }
.badge-priority { background: var(--danger-bg); color: var(--danger); font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; }
.badge-priority.p3, .badge-priority.p4 { background: var(--panel); color: var(--muted); }

.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: var(--radius); font-size: 0.85rem;
  font-weight: 500; cursor: pointer; border: none; transition: all 0.15s;
}
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: var(--accent-hover); }
.btn-ghost { background: none; border: 1px solid var(--border); color: var(--muted); }
.btn-ghost:hover { border-color: var(--accent); color: var(--accent); }

.empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
.empty-state h3 { font-size: 1.1rem; color: var(--ink); margin-bottom: 8px; }

/* Chat page */
.chat-layout { display: flex; gap: 24px; height: calc(100vh - 120px); }
.chat-main { flex: 1; display: flex; flex-direction: column; }
.chat-sidebar { width: 280px; }

.job-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: 20px; box-shadow: var(--shadow);
}
.job-card h3 { font-size: 0.9rem; font-weight: 600; margin-bottom: 12px; }
.job-card .field { margin-bottom: 10px; }
.job-card .field-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 2px; }
.job-card .field-value { font-size: 0.88rem; }

.chat-box {
  flex: 1; overflow-y: auto; padding: 20px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  display: flex; flex-direction: column; gap: 12px;
}
.bubble {
  max-width: 80%; padding: 12px 16px; border-radius: 16px;
  font-size: 0.9rem; line-height: 1.5; position: relative;
}
.bubble.agent {
  align-self: flex-start; background: var(--panel);
  border-bottom-left-radius: 4px;
}
.bubble.you {
  align-self: flex-end; background: var(--accent); color: white;
  border-bottom-right-radius: 4px;
}
.bubble .replay-btn {
  position: absolute; top: 4px; right: 8px; background: none; border: none;
  color: var(--muted); cursor: pointer; font-size: 0.75rem; opacity: 0.6;
}
.bubble .replay-btn:hover { opacity: 1; }

.input-bar {
  display: flex; gap: 8px; align-items: center;
  padding: 16px; background: var(--surface);
  border: 1px solid var(--border); border-top: none;
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}
.input-bar textarea {
  flex: 1; resize: none; border: 1px solid var(--border); border-radius: 24px;
  padding: 12px 20px; font-size: 0.9rem; font-family: inherit;
  min-height: 44px; max-height: 100px; outline: none; line-height: 1.4;
}
.input-bar textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
.btn-mic {
  width: 44px; height: 44px; border-radius: 50%; border: none;
  background: var(--panel); color: var(--ink); font-size: 1.2rem;
  cursor: pointer; transition: all 0.15s; display: flex; align-items: center; justify-content: center;
}
.btn-mic:hover { background: var(--border); }
.btn-mic.active { background: var(--danger); color: white; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.05)} }
.btn-send {
  width: 44px; height: 44px; border-radius: 50%; border: none;
  background: var(--accent); color: white; font-size: 1.1rem;
  cursor: pointer; transition: all 0.15s; display: flex; align-items: center; justify-content: center;
}
.btn-send:hover { background: var(--accent-hover); }
.btn-send:disabled { opacity: 0.5; cursor: not-allowed; }

.outcome-banner {
  padding: 16px 20px; border-radius: var(--radius-lg); margin-top: 16px;
  display: flex; align-items: center; gap: 12px; font-weight: 500;
}
.outcome-banner.confirm { background: var(--success-bg); color: var(--success); }
.outcome-banner.escalate { background: var(--warning-bg); color: var(--warning); }

.status-line { font-size: 0.8rem; color: var(--muted); text-align: center; padding: 8px; min-height: 30px; }

.code-pill {
  display: inline-block; background: var(--panel); border-radius: 4px;
  padding: 2px 6px; font-size: 0.75rem; font-family: monospace; color: var(--muted);
}
"""


def _render_dashboard():
    pending = _get_pending_jobs()
    completed = _get_completed_jobs()

    pending_rows = ""
    for job in pending:
        priority_cls = "p3" if "P3" in (job.get("PRIORITY") or "") else ("p4" if "P4" in (job.get("PRIORITY") or "") else "")
        pending_rows += f"""
        <tr onclick="window.location='/chat?id={job['REQUEST_ID']}'">
          <td><strong>#{job['REQUEST_ID']}</strong></td>
          <td>{job.get('REQ_CLASS','—')}</td>
          <td><span class="badge-priority {priority_cls}">{job.get('PRIORITY','—')}</span></td>
          <td style="max-width:300px">{(job.get('CUST_PROB_DESCR') or '—')[:80]}</td>
          <td><span class="badge-status badge-pending">Awaiting Close-out</span></td>
          <td><a href="/chat?id={job['REQUEST_ID']}" class="btn btn-primary" style="font-size:0.78rem;padding:6px 12px">Close out</a></td>
        </tr>"""

    completed_rows = ""
    for job in completed[:30]:
        codes_html = ""
        if job.get("P"):
            codes_html = f"<span class='code-pill'>{job['P']}</span> <span class='code-pill'>{job.get('C','')}</span> <span class='code-pill'>{job.get('R','')}</span> <span class='code-pill'>{job.get('M','')}</span>"
        completed_rows += f"""
        <tr>
          <td><strong>#{job['REQUEST_ID']}</strong></td>
          <td>{job.get('REQ_CLASS','—')}</td>
          <td>{(job.get('CUST_PROB_DESCR') or '—')[:60]}</td>
          <td>{codes_html or '—'}</td>
          <td><span class="badge-status badge-complete">Complete</span></td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WFM Dashboard</title><style>{CSS}</style></head><body>
<div class="layout">
  {_sidebar_html('dashboard')}
  <div class="main">
    <div class="page-header">
      <h1>Dashboard</h1>
      <p>Manage field service tickets — close out completed jobs or review history</p>
    </div>

    <div class="tabs">
      <button class="tab active" onclick="showTab('pending')">Pending ({len(pending)})</button>
      <button class="tab" onclick="showTab('completed')">Completed ({len(completed)})</button>
    </div>

    <div id="tab-pending" class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>ID</th><th>Type</th><th>Priority</th><th>Description</th><th>Status</th><th>Action</th></tr></thead>
          <tbody>{pending_rows if pending_rows else '<tr><td colspan="6"><div class="empty-state"><div class="icon">&#9989;</div><h3>All caught up!</h3><p>No pending close-outs</p></div></td></tr>'}</tbody>
        </table>
      </div>
    </div>

    <div id="tab-completed" class="card" style="display:none">
      <div class="table-wrap">
        <table>
          <thead><tr><th>ID</th><th>Type</th><th>Description</th><th>Codes</th><th>Status</th></tr></thead>
          <tbody>{completed_rows if completed_rows else '<tr><td colspan="5"><div class="empty-state"><div class="icon">&#128196;</div><h3>No completed jobs yet</h3></div></td></tr>'}</tbody>
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
    <div class="brand">WFM Voice Agent</div>
    <div class="subtitle">Field Service Platform</div>
    <nav>
      <a href="/" class="{'active' if active=='dashboard' else ''}">
        <span class="icon">&#128200;</span> Dashboard
        <span class="badge">{pending_count}</span>
      </a>
      <a href="/chat" class="{'active' if active=='chat' else ''}">
        <span class="icon">&#128172;</span> New Close-out
      </a>
    </nav>
    <div class="footer">
      <a href="https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/weave" target="_blank">&#128279; Weave Traces</a>
      <a href="https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}" target="_blank">&#128202; W&B Dashboard</a>
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
        greeting = f"G'day! I can see job #{job['REQUEST_ID']} — reported as: \"{job.get('CUST_PROB_DESCR', '')}\". What did you find on site, and how did you fix it?"
        job_sidebar = f"""
    <div class="chat-sidebar">
      <div class="job-card">
        <h3>Job #{job['REQUEST_ID']}</h3>
        <div class="field"><div class="field-label">Status</div><div class="field-value"><span class="badge-status badge-pending">Awaiting Close-out</span></div></div>
        <div class="field"><div class="field-label">Type</div><div class="field-value">{job.get('REQ_CLASS','—')}</div></div>
        <div class="field"><div class="field-label">Priority</div><div class="field-value">{job.get('PRIORITY','—')}</div></div>
        <div class="field"><div class="field-label">Reported Problem</div><div class="field-value">{job.get('CUST_PROB_DESCR','—')}</div></div>
        <div class="field"><div class="field-label">Location</div><div class="field-value">{job.get('PLACE_ID','—')}</div></div>
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Close Out{(' #' + str(job['REQUEST_ID'])) if job else ''} — WFM</title><style>{CSS}</style></head><body>
<div class="layout">
  {_sidebar_html('chat')}
  <div class="main">
    <div class="page-header">
      <h1>{'Close Out Job #' + str(job['REQUEST_ID']) if job else 'Voice Close-Out'}</h1>
      <p>Speak naturally — the mic stays on until you click stop</p>
    </div>

    <div class="chat-layout">
      <div class="chat-main">
        <div class="chat-box" id="chatBox"></div>
        <div class="input-bar" id="inputBar">
          <textarea id="userInput" rows="1" placeholder="Describe what you did..."></textarea>
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

function initSpeechRecognition() {{
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {{ statusEl.textContent = 'Speech not supported in this browser — use typing'; return; }}

  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-AU';

  let finalTranscript = '';

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
  listening = true;
  recognition.start();
  micBtn.classList.add('active');
  micBtn.innerHTML = '&#9209;';
  statusEl.textContent = 'Listening... speak naturally, click stop when done';
}}

function stopListening() {{
  listening = false;
  if (recognition) recognition.stop();
  micBtn.classList.remove('active');
  micBtn.innerHTML = '&#127908;';
  statusEl.textContent = '';
  const text = input.value.trim();
  if (text) {{ sendMessage(text); input.value = ''; }}
}}

micBtn.addEventListener('click', () => {{
  if (listening) stopListening(); else startListening();
}});

sendBtn.addEventListener('click', () => {{
  const text = input.value.trim();
  if (text) {{ sendMessage(text); input.value = ''; }}
}});

input.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    const text = input.value.trim();
    if (text) {{ sendMessage(text); input.value = ''; }}
  }}
}});

function autoResize() {{
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 100) + 'px';
}}
input.addEventListener('input', autoResize);

async function sendMessage(text) {{
  addBubble('you', text);
  statusEl.textContent = 'Agent thinking...';
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
    addBubble('agent', 'Error connecting to server.');
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
    outcomeEl.innerHTML = '<div class="outcome-banner confirm">&#9989; Job closed successfully — <a href="/">back to dashboard</a></div>';
    inputBar.style.display = 'none';
  }} else if (state.outcome === 'ESCALATE') {{
    outcomeEl.innerHTML = '<div class="outcome-banner escalate">&#9888; Escalated to team — <a href="/">back to dashboard</a></div>';
    inputBar.style.display = 'none';
  }}

  chatBox.scrollTop = chatBox.scrollHeight;
}}

function addBubble(role, text) {{
  const div = document.createElement('div');
  div.className = 'bubble ' + role;
  div.textContent = text;
  if (role === 'agent') {{
    const btn = document.createElement('button');
    btn.className = 'replay-btn';
    btn.textContent = '\\uD83D\\uDD0A';
    btn.onclick = () => speakText(text);
    div.appendChild(btn);
  }}
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}}

function speakText(text) {{
  if (speaking) speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.onstart = () => {{ speaking = true; }};
  utter.onend = () => {{ speaking = false; }};
  speechSynthesis.speak(utter);
}}

// Init
initSpeechRecognition();
addBubble('agent', `{greeting.replace('"', '\\"')}`);
renderedCount = 0;
speakText(`{greeting.replace('"', '\\"')}`);
</script>
</body></html>"""


# ─── App ───────────────────────────────────────────────────────────────────────

app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/chat", chat_page),
        Route("/turn", turn, methods=["POST"]),
        Route("/api/jobs", api_jobs),
    ],
)


if __name__ == "__main__":
    weave.init(WANDB_FULL_PROJECT)
    print(f"\n  WFM Voice Agent — http://127.0.0.1:8000")
    print(f"  Weave: https://wandb.ai/{WANDB_FULL_PROJECT}/weave\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
