"""Field Ops Orchestrator — Team ZO

Voice-driven multi-agent system for water maintenance field service.
A technician speaks a job report and it lands as a structured WFM record.

Usage:
  python app.py                                    # interactive mode
  python app.py "Finished the job at Collaroy..."  # direct input
  python app.py --eval                             # run Weave evaluation
"""
from __future__ import annotations

import sys
import time

import wandb
import weave

from config import MODEL, WANDB_ENTITY, WANDB_PROJECT
from orchestrator.graph import build_graph, OrchestratorState


def init():
    weave.init(WANDB_PROJECT)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"field-ops-{int(time.time())}",
        config={
            "model": MODEL,
            "architecture": "langgraph-a2a-mcp",
            "agents": ["voice", "structuring", "helpdesk"],
            "protocols": ["a2a", "mcp"],
        },
        tags=["field-ops", "multi-agent", "hackathon"],
    )
    return run


def process_report(transcript: str, clarification_response: str = "") -> dict:
    """Process a technician's report through the full pipeline."""
    graph = build_graph()
    app = graph.compile()

    initial_state: OrchestratorState = {
        "messages": [],
        "transcript": transcript,
        "intent": "",
        "request_id": None,
        "structured_payload": {},
        "clarification_count": 0,
        "clarification_response": clarification_response,
        "validation_result": {},
        "final_result": {},
        "trace": [],
        "status": "started",
    }

    start = time.time()
    final_state = {}
    for step in app.stream(initial_state):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        final_state.update(node_output)

        if "trace" in node_output:
            latest = node_output["trace"][-1] if node_output["trace"] else ""
            print(f"  [{node_name}] {latest}")

    elapsed = time.time() - start
    final_state["elapsed_seconds"] = elapsed

    wandb.log({
        "run/elapsed_seconds": elapsed,
        "run/status": final_state.get("status", "unknown"),
        "run/intent": final_state.get("intent", "unknown"),
        "run/trace_steps": len(final_state.get("trace", [])),
    })

    return final_state


def interactive_mode():
    """Interactive REPL for testing the system."""
    print("=" * 60)
    print("Field Ops Orchestrator — Team ZO")
    print("Type a technician report or 'quit' to exit")
    print("=" * 60)

    while True:
        print()
        transcript = input("Tech report > ").strip()
        if transcript.lower() in ("quit", "exit", "q"):
            break
        if not transcript:
            continue

        result = process_report(transcript)
        print(f"\nStatus: {result.get('status')}")
        print(f"Intent: {result.get('intent')}")
        print(f"Time: {result.get('elapsed_seconds', 0):.2f}s")

        payload = result.get("structured_payload", {})
        if payload.get("needs_clarification"):
            clarify = payload["needs_clarification"]
            print(f"\nClarification needed: {clarify['question']}")
            if clarify.get("options"):
                print(f"Options: {', '.join(clarify['options'])}")
            answer = input("Answer > ").strip()
            if answer:
                result = process_report(transcript, clarification_response=answer)
                print(f"\nAfter clarification — Status: {result.get('status')}")

        final = result.get("final_result", {})
        if final:
            print(f"\nFinal result: {final}")

        print("\nTrace:")
        for step in result.get("trace", []):
            print(f"  {step}")


def main():
    run = init()

    if "--eval" in sys.argv:
        import asyncio
        from weave_eval.run_eval import run as run_eval
        asyncio.run(run_eval())
    elif len(sys.argv) > 1 and sys.argv[1] != "--eval":
        transcript = " ".join(sys.argv[1:])
        print(f"\nProcessing: {transcript}\n")
        result = process_report(transcript)
        print(f"\nFinal: {result.get('final_result', {})}")
        print(f"Elapsed: {result.get('elapsed_seconds', 0):.2f}s")
    else:
        interactive_mode()

    run.finish()
    print("\nDone! Check Weave traces at:")
    print(f"  https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/weave")


if __name__ == "__main__":
    main()
