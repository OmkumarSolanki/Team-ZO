"""Help Desk Agent — owns the WFM through MCP tools.

Handles search, validation, close-out, and new-fault posting.
All tool calls go through the MCP server and are Weave-traced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import weave

from config import MODEL, get_client
from mcp_servers.wfm_server import MCP_TOOLS, TOOL_DISPATCH

CODES_PATH = Path(__file__).parent.parent / "data" / "codes.json"


def _load_valid_codes() -> dict[str, list[str]]:
    with open(str(CODES_PATH)) as f:
        codes = json.load(f)
    return {
        "problem_code": [c["code"] for c in codes["problem"]],
        "cause_code": [c["code"] for c in codes["cause"]],
        "rectify_code": [c["code"] for c in codes["rectify"]],
        "method_code": [c["code"] for c in codes["method"]],
    }


class Hade(weave.Model):
    """Hade — HelpDesk Agent. Validates, guardrails, and posts to WFM system."""
    model_name: str = MODEL

    @weave.op()
    def predict(self, task: str, context: dict | None = None) -> dict:
        return self.execute(task, context)

    @weave.op()
    def execute(self, task: str, context: dict | None = None) -> dict:
        """Execute a help desk task using MCP tools."""
        client = get_client()

        system = """You are a help desk agent for a water utility WFM system.
You have access to tools to search, get, validate, and close work requests.
Use the tools to fulfill the task. Always validate before closing a request."""

        messages = [{"role": "system", "content": system}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context: {json.dumps(context, default=str)}",
            })

        messages.append({"role": "user", "content": task})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=MCP_TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )

        tool_results = []
        max_turns = 5
        turn = 0

        while turn < max_turns:
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                result = self._call_tool(fn_name, fn_args)
                tool_results.append({"tool": fn_name, "args": fn_args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=MCP_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            turn += 1

        final_message = response.choices[0].message.content or ""

        return {
            "response": final_message,
            "tool_calls": tool_results,
            "status": "success",
        }

    @weave.op()
    def search(self, text: str) -> list[dict]:
        """Direct search shortcut."""
        from mcp_servers.wfm_server import search_requests
        return search_requests(text)

    @weave.op()
    def close(self, request_id: int, codes: dict) -> dict:
        """Direct close shortcut."""
        from mcp_servers.wfm_server import close_request
        return close_request(request_id, codes)

    @weave.op()
    def validate(self, payload: dict, mode: str = "close_out") -> dict:
        """Direct validate shortcut."""
        from mcp_servers.wfm_server import validate_request
        return validate_request(payload, mode)

    @weave.op()
    def validate_and_correct(self, transcript: str, gemi_output: dict, gemi_agent=None, max_corrections: int = 2) -> dict:
        """Validate GEMI output and cross-check with independent analysis.

        Two-stage validation:
        1. Schema check — are all codes valid enums?
        2. Semantic check — does the code combination make sense for this transcript?
        If either fails, send targeted correction feedback to GEMI.
        """
        valid_codes = _load_valid_codes()
        corrections = 0
        current_output = gemi_output

        while corrections < max_corrections:
            issues = []

            # Stage 1: Schema validity
            for field, valid_list in valid_codes.items():
                value = current_output.get(field, "")
                if value and value not in valid_list:
                    issues.append(f"{field}='{value}' is not valid. Valid: {valid_list}")

            for field in ["problem_code", "cause_code", "rectify_code", "method_code"]:
                if not current_output.get(field):
                    issues.append(f"{field} is missing")

            # Stage 2: Semantic cross-check (skip if source is validated lookup)
            if not issues and current_output.get("_source") != "validated_lookup":
                semantic_issues = self._semantic_check(transcript, current_output)
                issues.extend(semantic_issues)

            if not issues:
                current_output["_hade_approved"] = True
                current_output["_hade_corrections"] = corrections
                current_output["_agent_chain"] = "Vera → GEMI → Hade ✓"
                return current_output

            corrections += 1
            feedback = f"Hade review found issues: {'; '.join(issues)}. Re-read the JOB_COMMENTS carefully and correct."
            if gemi_agent:
                current_output = gemi_agent.structure(
                    transcript,
                    context={"correction_feedback": feedback},
                )
            else:
                break

        current_output["_hade_approved"] = corrections <= max_corrections
        current_output["_hade_corrections"] = corrections
        current_output["_agent_chain"] = f"Vera → GEMI → Hade ({corrections} corrections)"
        return current_output

    @weave.op()
    def _semantic_check(self, transcript: str, output: dict) -> list[str]:
        """Independent semantic verification of code combinations."""
        issues = []
        text = transcript.upper()
        p = output.get("problem_code", "")
        c = output.get("cause_code", "")
        r = output.get("rectify_code", "")
        m = output.get("method_code", "")

        # Extract JOB_COMMENTS specifically (that's where work description is)
        job_text = ""
        for part in text.split("|"):
            if "JOB_COMMENT" in part:
                job_text += part + " "

        # Rule: Clamp on MAIN pipe (clamp mentioned in same JOB_COMMENT as 100mm+) → M_LID
        # But clamp on small service (20mm, 25mm copper) → M_CLA is correct
        if m == "M_CLA":
            clamp_on_main = False
            for part in text.split("|"):
                part_up = part.upper()
                if "CLAMP" in part_up and any(kw in part_up for kw in ["100MM", "150MM", "200MM", "DICL", "CICL"]):
                    clamp_on_main = True
            if clamp_on_main:
                issues.append("Clamp installed on a MAIN pipe (100mm+/DICL/CICL) → M_LID (not M_CLA). M_CLA is only for small service clamps.")

        # Rule: Repacking a valve (SV, stop valve) → C_WER, R_RPR, M_LID
        if any(kw in job_text for kw in ["REPACK", "REPACKED"]):
            if "PATH TAP" in text or "PATHTAP" in text:
                if r != "R_ADJ":
                    issues.append("Repacking a PATH TAP → R_ADJ. Path tap repack is minor adjustment.")
                if m != "M_002":
                    issues.append("Path tap repacked → M_002.")
            elif any(kw in job_text for kw in [" SV ", "VALVE", "STOP VALVE"]):
                if c != "C_WER":
                    issues.append("Repacking a valve indicates wear and tear → C_WER.")
                if m != "M_LID":
                    issues.append("Repacking a valve → M_LID.")

        # Rule: "blown fitting replaced with new" → R_RPL + M_002
        if any(kw in job_text for kw in ["BLOWN FITTING REPLACED", "REPLACED WITH NEW", "REPLACED 32MM", "REPLACED FITTING"]):
            if r != "R_RPL":
                issues.append("Blown/old fitting replaced with new → R_RPL.")
            if m != "M_002":
                issues.append("Fitting replaced with new → M_002.")

        # Rule: "replaced adaptor" or "replaced poly to copper adaptor" → R_RPR + M_002
        # Replacing a small adaptor/connector is a REPAIR to the service, not full replacement
        if any(kw in job_text for kw in ["REPLACED POLY TO COPPER", "REPLACED ADAPTOR", "REPLACED ADAPTER"]):
            if r != "R_RPR":
                issues.append("Replacing an adaptor is a repair to the service → R_RPR (not R_RPL).")
            if m != "M_002":
                issues.append("Installing new adaptor → M_002.")

        # Rule: "CUT PIECE OUT" on service → P_BKN, R_RPL
        if any(kw in job_text for kw in ["CUT PIECE OUT", "CUT PEICE OUT"]):
            if "CLAMP" not in job_text:
                if p != "P_BKN":
                    issues.append("'Cut piece out' indicates broken asset → P_BKN.")
                if r != "R_RPL":
                    issues.append("'Cut piece out' = replacement needed → R_RPL.")

        # Rule: leak on service "around adapt or tee" with NDD → P_BKN, R_RPL, M_LID
        if "ADAPT" in job_text and ("TEE" in job_text or "T " in job_text):
            if "NDD" in text or "DIG" in job_text:
                if p != "P_BKN":
                    issues.append("Leak around adaptor/tee on service suggests broken fitting → P_BKN.")
                if r != "R_RPL":
                    issues.append("Broken adaptor/tee needs replacement → R_RPL.")

        # Rule: Don't use R_UNR or M_004 unless explicitly unresolved or no-repair
        if r == "R_UNR" and any(kw in job_text for kw in ["REPAIR", "REPLACED", "REPACK", "CLAMP", "INSTALL", "DIG"]):
            issues.append("Work was described in JOB_COMMENTS — R_UNR (Unresolved) is wrong. Choose R_RPR or R_RPL based on the work done.")
        if m == "M_004" and any(kw in job_text for kw in ["REPAIR", "REPLACED", "REPACK", "CLAMP", "INSTALL"]):
            issues.append("Work method is described — M_004 is wrong. Choose M_LID, M_002, or M_CLA based on the method.")

        return issues

    @weave.op()
    def _call_tool(self, name: str, args: dict) -> Any:
        handler = TOOL_DISPATCH.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        return handler(args)
