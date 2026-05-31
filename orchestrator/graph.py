"""Orchestrator state machine using LangGraph.

Coordinates 3 agents:
  - Vera (Voice Agent): STT transcription
  - GEMI (Structuring Agent): Maps transcript to PCRM codes
  - Hade (HelpDesk Agent): Validates, guardrails, corrects, and posts to WFM

Routes by intent, manages the clarification/correction loop,
and coordinates all agents via A2A-style messaging.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

import wandb
import weave
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.helpdesk_agent import Hade
from agents.structuring_agent import GEMI
from agents.voice_agent import Vera
from orchestrator.intents import Intent, classify_intent
from weave_eval.scorers import consistency_guard


class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    transcript: str
    intent: str
    request_id: int | None
    structured_payload: dict
    clarification_count: int
    clarification_response: str
    validation_result: dict
    final_result: dict
    trace: list[str]
    status: str


# Agents
vera = Vera()
gemi_closeout = GEMI(mode="close_out")
gemi_newfault = GEMI(mode="new_fault")
hade = Hade()


@weave.op()
def classify_node(state: OrchestratorState) -> dict:
    transcript = state["transcript"]
    intent = classify_intent(transcript)
    trace = state.get("trace", [])
    trace.append(f"Orchestrator: Classified intent as {intent.value}")
    wandb.log({"orchestrator/intent": intent.value})
    return {"intent": intent.value, "trace": trace, "status": "classified"}


@weave.op()
def search_request_node(state: OrchestratorState) -> dict:
    transcript = state["transcript"]
    trace = state.get("trace", [])

    results = hade.search(transcript)
    trace.append(f"Hade: Searched WFM, found {len(results)} matching requests")

    request_id = None
    if results:
        request_id = results[0]["REQUEST_ID"]
        trace.append(f"Hade: Resolved to REQUEST_ID {request_id}")

    wandb.log({"orchestrator/search_results": len(results)})
    return {"request_id": request_id, "trace": trace}


@weave.op()
def structure_node(state: OrchestratorState) -> dict:
    transcript = state["transcript"]
    intent = state["intent"]
    trace = state.get("trace", [])
    clarification_resp = state.get("clarification_response", "")

    if intent == Intent.CLOSE_OUT.value:
        agent = gemi_closeout
    else:
        agent = gemi_newfault

    context = {}
    if state.get("request_id"):
        context["request_id"] = state["request_id"]
    if clarification_resp:
        context["clarification_answer"] = clarification_resp
        context["previous_payload"] = state.get("structured_payload", {})

    result = agent.structure(transcript, context if context else None)
    trace.append(f"GEMI: Mapped transcript to codes (model={result.get('_model', 'unknown')})")

    low_confidence = {
        k: v for k, v in result.get("field_confidence", {}).items() if v < 0.6
    }
    if low_confidence:
        trace.append(f"GEMI: Low confidence on {list(low_confidence.keys())}")

    wandb.log({
        "orchestrator/structuring_tokens": result.get("_tokens", 0),
        "orchestrator/low_confidence_fields": len(low_confidence),
    })

    return {"structured_payload": result, "trace": trace, "status": "structured"}


@weave.op()
def clarification_node(state: OrchestratorState) -> dict:
    payload = state["structured_payload"]
    trace = state.get("trace", [])
    count = state.get("clarification_count", 0)

    clarify = payload.get("needs_clarification")
    if clarify:
        question = clarify.get("question", "Can you provide more details?")
        options = clarify.get("options", [])
        voice_result = vera.ask_clarification(question, options)
        trace.append(f"Vera: Asked clarification — {question}")
        trace.append(f"Orchestrator: Clarification loop iteration {count + 1}")
    else:
        trace.append("Orchestrator: No clarification needed")

    return {"clarification_count": count + 1, "trace": trace}


@weave.op()
def guardrail_node(state: OrchestratorState) -> dict:
    payload = state["structured_payload"]
    trace = state.get("trace", [])

    guard_result = consistency_guard(payload)
    trace.append(f"Hade: Guardrail {'PASSED' if guard_result['safe'] else 'BLOCKED — ' + str(guard_result.get('issue'))}")

    wandb.log({"orchestrator/guardrail_passed": guard_result["safe"]})

    return {
        "validation_result": guard_result,
        "trace": trace,
        "status": "guardrail_checked",
    }


@weave.op()
def validate_and_post_node(state: OrchestratorState) -> dict:
    payload = state["structured_payload"]
    intent = state["intent"]
    trace = state.get("trace", [])

    if intent == Intent.CLOSE_OUT.value:
        request_id = state.get("request_id")
        if not request_id:
            trace.append("Hade: ERROR — No request_id resolved, cannot close")
            return {"final_result": {"status": "error", "reason": "no request_id"}, "trace": trace, "status": "error"}

        codes = {
            "request_id": request_id,
            "problem_code": payload.get("problem_code"),
            "cause_code": payload.get("cause_code"),
            "rectify_code": payload.get("rectify_code"),
            "method_code": payload.get("method_code"),
        }

        val = hade.validate(codes, "close_out")
        trace.append(f"Hade: Validation {'passed' if val['ok'] else 'FAILED: ' + str(val['errors'])}")

        if val["ok"]:
            result = hade.close(request_id, codes)
            trace.append(f"Hade: Closed request {request_id} → {result.get('new_status')}")
        else:
            result = {"status": "validation_failed", "errors": val["errors"]}
            trace.append("Hade: Close aborted due to validation errors")

    else:
        result = {"status": "new_fault_posted", "payload": payload}
        trace.append("Hade: New fault request posted")

    wandb.log({"orchestrator/final_status": result.get("status")})

    return {"final_result": result, "trace": trace, "status": "posted"}


@weave.op()
def confirm_node(state: OrchestratorState) -> dict:
    result = state.get("final_result", {})
    trace = state.get("trace", [])

    confirmation = vera.confirm_result(result)
    trace.append(f"Vera: Read back — {confirmation}")

    return {"trace": trace, "status": "confirmed"}


def needs_clarification(state: OrchestratorState) -> str:
    payload = state.get("structured_payload", {})
    count = state.get("clarification_count", 0)

    if payload.get("needs_clarification") and count < 2:
        return "clarify"
    return "guardrail"


def guardrail_decision(state: OrchestratorState) -> str:
    val = state.get("validation_result", {})
    if val.get("safe", True):
        return "post"
    return "clarify"


def intent_router(state: OrchestratorState) -> str:
    intent = state.get("intent")
    if intent == Intent.CLOSE_OUT.value:
        return "search"
    if intent == Intent.NEW_FAULT.value:
        return "structure"
    return "structure"


def build_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify_node)
    graph.add_node("search", search_request_node)
    graph.add_node("structure", structure_node)
    graph.add_node("clarify", clarification_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("post", validate_and_post_node)
    graph.add_node("confirm", confirm_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", intent_router, {
        "search": "search",
        "structure": "structure",
    })
    graph.add_edge("search", "structure")

    graph.add_conditional_edges("structure", needs_clarification, {
        "clarify": "clarify",
        "guardrail": "guardrail",
    })

    graph.add_edge("clarify", "structure")

    graph.add_conditional_edges("guardrail", guardrail_decision, {
        "post": "post",
        "clarify": "clarify",
    })

    graph.add_edge("post", "confirm")
    graph.add_edge("confirm", END)

    return graph
