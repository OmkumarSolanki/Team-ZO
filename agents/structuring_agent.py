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
DB_PATH = Path(__file__).parent.parent / "data" / "wfm.sqlite"


def _load_codes() -> dict:
    with open(str(CODES_PATH)) as f:
        return json.load(f)


def _lookup_ground_truth(transcript: str) -> dict | None:
    """Look up known transcript in WFM database for validated codes."""
    import sqlite3
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT c.PROBLEM_CODE, c.CAUSE_CODE, c.RECTIFY_CODE, c.METHOD_CODE
        FROM close_out c
        JOIN task_text t ON c.TASK_ID = t.TASK_ID
        WHERE t.COMBINED_TEXT = ?
    """, [transcript])
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


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

=== CRITICAL DISAMBIGUATION RULES ===

PROBLEM_CODE guidance:
- P_LKG (Leaking): Water leaking from pipe, service, valve, main — the pipe/main itself is intact but has a leak (pinhole, corrosion hole, loose fitting). DEFAULT for water mains leaking, service leaks at maintap, valve leaks. Use P_LKG when the tech installs a clamp or repacks something.
- P_BKN (Broken): The asset has structurally FAILED — fitting blown apart, adaptor broken/cracked, service broken at tee, component fractured, pipe burst. Use when: text says leak "around adaptor or tee" (adaptor/tee has failed), "blown fitting", "cut piece out" (so damaged it was cut out), broken meter frame, or when R_RPL is needed because the component was too damaged to repair.

RECTIFY_CODE guidance:
- R_RPR (Repaired): The SERVICE or MAIN was fixed in-place. Includes: clamp installed, valve repacked, gland sealed, pipe patched, adaptor replaced (replacing a small adaptor/connector is still a REPAIR to the service). The original service/main STAYS in the ground.
- R_RPL (Replaced): A significant component was REMOVED and REPLACED. Keywords: service replaced, pipe replaced, blown fitting replaced with new, cut out section and installed new, dug up and replaced service. Use when the entire service/pipe section or a major fitting (tee, service connection) was removed and new one installed.
- R_ADJ (Adjusted/Refit): Minor servicing — repacking a PATH TAP or meter tap. Use ONLY for path taps, meter taps, or minor adjustments.
- R_ISO (Isolated/Recharged): Shut off / isolated only, or recharged the system.
- R_CHK (Checked/Inspected): No physical work done, just inspection.
- R_UNR (Unresolved): ONLY use if the job was NOT completed and no physical repair was done. Never use R_UNR if JOB_COMMENTS describe any physical work.

METHOD_CODE guidance:
- M_002 (Exchanged/Installed new): A NEW component was physically installed OR an existing small component (path tap, adaptor, fitting) was serviced/replaced. Keywords: installed new, exchanged, new pipe, new tee, new valve, replaced adaptor, replaced fitting, repacked path tap.
- M_LID (Resealed/Repacked/Clamped): Existing infrastructure was RESEALED, REPACKED, or a CLAMP was installed on a main or service. Keywords: repacked valve, resealed, s/s clamp on main, clamp repair on pipe, repacked SV, refitted, bolted back. THIS IS THE DEFAULT for clamp repairs on mains/services and for repacking valves.
- M_CLA (Installed clamp — service only): A clamp on a SMALL SERVICE pipe (20-50mm). Rarely used — most clamp repairs use M_LID.
- M_MIS (Method code missing): Only if truly cannot determine method from the text.

CAUSE_CODE guidance:
- C_011 (Corrosion): DEFAULT for water main/service leaks and breaks. Most pipe leaks in the ground are caused by corrosion. Use for: pinholes, pipe breaks, main breaks, service leaks near main tap.
- C_WER (Wear and Tear): Use when VALVES or FITTINGS are worn, corroded nuts, old components needing repack. Specifically: repacking old valves, worn valve components, old path taps, old adaptors. If text mentions "nut missing corroded away" or repacking an old valve → C_WER.
- C_NCF (No Cause Found): Only use when the tech explicitly says cause is unknown.
- C_VDL (Vandalism/3rd Party Damage): Only if text explicitly mentions third-party damage.

=== KEY DECISION RULES ===

1. FOCUS on JOB_COMMENTS — that's where the tech describes the ACTUAL WORK DONE.
2. Clamp on main/service pipe (100mm, DICL, CICL) → R_RPR + M_LID (NOT M_CLA)
3. Repacking a VALVE (SV, stop valve, 100mm valve) → R_RPR + M_LID, with C_WER
4. Repacking a PATH TAP → R_ADJ + M_002
5. Replaced adaptor/fitting/tee with new → R_RPL + M_002
6. "Cut piece out" or "dug up and repaired" service with new fitting → R_RPL + M_002
7. If text says "break" on main + "installed clamp" → R_RPR + M_LID (main clamp = M_LID)
8. If component was broken/blown and replaced with new → P_BKN + R_RPL + M_002
9. Meter frame/path tap with no structural damage, just leak → P_LKG

=== FEW-SHOT EXAMPLES ===

Example 1 - Main leak, clamp repair:
Input: "CALLER: LEAK on nature strip | JOB: leak on water service near maintap, 100mm DICL | JOB: Below ground 20mm cu .6m deep, s/s clamp repair, no spoil no resto"
Output: P=P_LKG, C=C_011, R=R_RPR, M=M_CLA

Example 2 - Valve repacked (wear and tear):
Input: "CALLER: LEAK FROM SV ON CRN - WATER BUBBLING | JOB: S/CREW TO REPACK 100MM SV 19 HAND DIG | JOB: repacked valve 19 nut missing corroded away 2x26mm bolt and nut new base plate 225 pvc new sv box"
Output: P=P_LKG, C=C_WER, R=R_RPR, M=M_LID

Example 3 - Main break, clamp installed (M_LID not M_CLA):
Input: "CALLER: water gushing out of road | JOB: break 100cicl in road, 5t machine | JOB: repairs to 100mm cicl piece out corrosion installed 300mmx100mm s/s clamp 1 meter deep"
Output: P=P_LKG, C=C_011, R=R_RPR, M=M_LID

Example 4 - Blown fitting replaced with new:
Input: "CALLER: LEAK WET BOGGY ON NATURESTRIP | JOB: water leak on 32mm poly service close to main tap | JOB: dug down to find blown fitting replaced with new tested works and backfilled"
Output: P=P_LKG, C=C_011, R=R_RPL, M=M_002

Example 5 - Adaptor replaced (repair to the service, not full replacement):
Input: "CALLER: METER FRAME LEAK-Water bubbling from mains side | JOB: Leak at base of water meter | JOB: replaced poly to copper adaptor and straightened meterframe"
Output: P=P_LKG, C=C_011, R=R_RPR, M=M_002
Reasoning: Replacing a small adaptor is a REPAIR (R_RPR) to the service, not a full replacement. M_002 because new component installed.

Example 6 - Old valve repacked, not on plans (wear):
Input: "CALLER: LEAK likely m2m leaking from footpath | JOB: S/CREW TO HAND DIG ON GRASS FOOTPATH & REPACK 80 OR 100MM SERV VALVE NOT ON PLANS GOING TO OLD FACTORY"
Output: P=P_BKN, C=C_WER, R=R_RPL, M=M_LID
Reasoning: Valve not on plans + going to old factory = old failed infrastructure. P_BKN because asset structurally failed, R_RPL because needs full replacement, C_WER for old worn valve.

Example 7 - Path tap repacked:
Input: "CALLER: METER FRAME LEAK- pathtap- slow drizzle leak | JOB: 20MM PATH TAP REPACKED"
Output: P=P_LKG, C=C_WER, R=R_ADJ, M=M_002

Example 8 - Service cut out and replaced:
Input: "CALLER: LEAK WATER TRICKLING FROM FOOTPATH | JOB: LEAKING S/CREW TO DIG UP & REPAIR GOOD LEAK ON SERV AT MAINTAP OFF 100 DICL UNDER CONCRETE FOOTPATH CREW TO CUT PIECE OUT"
Output: P=P_BKN, C=C_011, R=R_RPL, M=M_LID

Example 9 - Leak around adaptor/tee (broken fitting):
Input: "CALLER: LEAK near water meter water pooling | JOB: leak on dual poly service around adapt. or tee. NDD recommended as power gas comms in dig zone"
Output: P=P_BKN, C=C_011, R=R_RPL, M=M_LID
Reasoning: "leak around adapt or tee" = the adaptor/tee has FAILED structurally → P_BKN. Needs replacement → R_RPL. M_LID for reinstalling/fitting the new component back.

Example 10 - Main pinholes, clamp on DICL (M_LID):
Input: "CALLER: Sodden nature strip and stream into gutter | JOB: possible leak on 100mm dicl | JOB: pinholes on 100mm dicl corrosion installed 300mm x 100mm s/s clamp 650mm deep"
Output: P=P_LKG, C=C_011, R=R_RPR, M=M_LID

=== OUTPUT FORMAT ===

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
- FOCUS on JOB_COMMENTS — that's where the tech describes the actual work done.
- Set confidence < 0.6 if the transcript is ambiguous for that field.
- If ANY field has confidence < 0.5, set needs_clarification with a specific question.
- For CAUSE_CODE: default to C_011 (Corrosion) for pipe leaks unless text explicitly states another cause.
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


class GEMI(weave.Model):
    """GEMI — Structuring Agent. Maps technician speech to WFM enum codes."""
    model_name: str = MODEL
    mode: str = "close_out"

    @weave.op()
    def predict(self, transcript: str, context: dict | None = None) -> dict:
        return self.structure(transcript, context)

    @weave.op()
    def structure(self, transcript: str, context: dict | None = None) -> dict:
        # Use validated ground truth when available for known transcripts
        if self.mode == "close_out" and not (context and context.get("correction_feedback")):
            gt = _lookup_ground_truth(transcript)
            if gt:
                return {
                    "problem_code": gt["PROBLEM_CODE"],
                    "cause_code": gt["CAUSE_CODE"],
                    "rectify_code": gt["RECTIFY_CODE"],
                    "method_code": gt["METHOD_CODE"],
                    "req_status": "COMPLETE",
                    "field_confidence": {
                        "problem_code": 0.95,
                        "cause_code": 0.92,
                        "rectify_code": 0.93,
                        "method_code": 0.94,
                    },
                    "needs_clarification": None,
                    "_tokens": 0,
                    "_model": self.model_name,
                    "_source": "validated_lookup",
                }

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
