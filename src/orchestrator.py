"""Multi-agent orchestrator using LangGraph + A2A protocol.

This is the main harness that coordinates agents through
a state machine, with every step traced by Weave and logged to W&B.
"""
from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

import wandb
import weave
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agents import AgentRegistry, BaseAgent
from src.protocols.a2a import A2AMessage, A2ARouter, MessageType, TaskEnvelope, TaskStatus


class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    current_agent: str
    results: dict[str, Any]
    trace: list[str]
    status: str
    iteration: int


class MultiAgentOrchestrator(weave.Model):
    """LangGraph-based orchestrator with A2A routing and Weave tracing."""
    name: str = "orchestrator"
    registry: Any = None
    router: Any = None
    max_iterations: int = 10

    def model_post_init(self, __context: Any) -> None:
        if self.registry is None:
            self.registry = AgentRegistry()
        if self.router is None:
            self.router = A2ARouter()

    def register_agent(self, agent: BaseAgent) -> str:
        return self.registry.register(agent)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(OrchestratorState)

        graph.add_node("planner", self._plan_step)
        graph.add_node("executor", self._execute_step)
        graph.add_node("validator", self._validate_step)
        graph.add_node("synthesizer", self._synthesize_step)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "validator")
        graph.add_conditional_edges(
            "validator",
            self._should_continue,
            {"continue": "executor", "synthesize": "synthesizer"},
        )
        graph.add_edge("synthesizer", END)

        return graph

    @weave.op()
    def _plan_step(self, state: OrchestratorState) -> dict:
        task = state["task"]
        available = self.registry.list_all()
        trace = state.get("trace", [])
        trace.append(f"Planner: Analyzing task and selecting agents from {len(available)} available")

        wandb.log({"orchestrator/step": "plan", "orchestrator/available_agents": len(available)})

        return {
            "current_agent": "planner",
            "trace": trace,
            "status": "planning",
            "results": state.get("results", {}),
        }

    @weave.op()
    def _execute_step(self, state: OrchestratorState) -> dict:
        task = state["task"]
        trace = state.get("trace", [])
        results = state.get("results", {})
        iteration = state.get("iteration", 0)

        agents = self.registry.list_all()
        for agent_info in agents:
            agent = self.registry.get(agent_info["agent_id"])
            if agent is None:
                continue

            msg = A2AMessage(
                sender="orchestrator",
                receiver=agent_info["agent_id"],
                message_type=MessageType.REQUEST,
                payload={"task": task, "context": results},
            )
            response = self.router.send(msg, self.registry)
            if response and response.message_type != MessageType.ERROR:
                results[agent_info["name"]] = response.payload
                trace.append(f"{agent_info['name']}: Completed task execution")

        wandb.log({
            "orchestrator/step": "execute",
            "orchestrator/iteration": iteration,
            "orchestrator/agents_invoked": len(agents),
        })

        return {
            "results": results,
            "trace": trace,
            "iteration": iteration + 1,
            "status": "executing",
        }

    @weave.op()
    def _validate_step(self, state: OrchestratorState) -> dict:
        results = state.get("results", {})
        trace = state.get("trace", [])

        has_results = len(results) > 0
        all_successful = all(
            isinstance(v, dict) and v.get("status") != "error"
            for v in results.values()
        )

        trace.append(f"Validator: {'Passed' if has_results and all_successful else 'Needs retry'}")
        wandb.log({"orchestrator/step": "validate", "orchestrator/valid": has_results and all_successful})

        return {
            "trace": trace,
            "status": "validated" if all_successful else "needs_retry",
        }

    def _should_continue(self, state: OrchestratorState) -> str:
        if state.get("iteration", 0) >= self.max_iterations:
            return "synthesize"
        if state.get("status") == "needs_retry":
            return "continue"
        return "synthesize"

    @weave.op()
    def _synthesize_step(self, state: OrchestratorState) -> dict:
        results = state.get("results", {})
        trace = state.get("trace", [])
        trace.append(f"Synthesizer: Combining results from {len(results)} agents")

        wandb.log({
            "orchestrator/step": "synthesize",
            "orchestrator/total_agents": len(results),
            "orchestrator/total_steps": len(trace),
        })

        return {
            "results": results,
            "trace": trace,
            "status": "completed",
        }

    @weave.op()
    def predict(self, task: str) -> dict[str, Any]:
        """Main entry point — called by Weave evaluations."""
        start = time.time()

        graph = self._build_graph()
        app = graph.compile()

        initial_state: OrchestratorState = {
            "messages": [],
            "task": task,
            "current_agent": "",
            "results": {},
            "trace": [],
            "status": "pending",
            "iteration": 0,
        }

        final_state = None
        for state in app.stream(initial_state):
            final_state = state

        elapsed = time.time() - start

        last_value = list(final_state.values())[0] if final_state else {}

        envelope = TaskEnvelope(
            description=task,
            status=TaskStatus.COMPLETED,
            trace=last_value.get("trace", []),
            results=last_value.get("results", {}),
        )
        self.router.log_task(envelope)

        wandb.log({
            "orchestrator/elapsed_seconds": elapsed,
            "orchestrator/status": "completed",
        })

        return {
            "result": last_value.get("results", {}),
            "trace": last_value.get("trace", []),
            "elapsed_seconds": elapsed,
            "status": "completed",
        }
