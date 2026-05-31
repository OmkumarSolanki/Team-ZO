"""Streamlit Web App — Two-Way Voice Agent + Orchestrator Pipeline.

Architecture:
  Voice Agent (STT) → Orchestrator (LangGraph) → Structuring Agent (GEMI) → WFM System
  All traced in Weave, all metrics logged to W&B.

Run with: streamlit run webapp.py
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="WFM Voice Agent — Field Service",
    page_icon="🔧",
    layout="wide",
)

CODES_PATH = Path(__file__).parent / "data" / "codes.json"
SCHEMA_PATH = Path(__file__).parent / "schema" / "request_schema.json"


# ─── Cached resources ────────────────────────────────────────────────────────

@st.cache_resource
def init_services():
    import wandb
    import weave
    from config import WANDB_ENTITY, WANDB_FULL_PROJECT, WANDB_PROJECT
    weave.init(WANDB_FULL_PROJECT)
    return WANDB_ENTITY, WANDB_PROJECT, WANDB_FULL_PROJECT


@st.cache_resource
def get_wandb_run():
    import wandb
    from config import WANDB_ENTITY, WANDB_PROJECT
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"voice-session-{int(time.time())}",
        tags=["webapp", "voice-agent", "orchestrator"],
        reinit=True,
    )
    return run


@st.cache_resource
def load_whisper_model():
    import whisper
    return whisper.load_model("base")


@st.cache_resource
def build_orchestrator_graph():
    """Build the LangGraph orchestrator (compiled once)."""
    from orchestrator.graph import build_graph
    graph = build_graph()
    return graph.compile()


@st.cache_data
def load_codes():
    with open(str(CODES_PATH)) as f:
        return json.load(f)


@st.cache_data
def load_schema():
    with open(str(SCHEMA_PATH)) as f:
        return json.load(f)


# ─── Domain system prompt ────────────────────────────────────────────────────

def get_domain_system_prompt() -> str:
    codes = load_codes()
    schema = load_schema()

    problem_list = "\n".join(f"  - {c['code']}: {c['label']}" for c in codes["problem"])
    cause_list = "\n".join(f"  - {c['code']}: {c['label']}" for c in codes["cause"])
    rectify_list = "\n".join(f"  - {c['code']}: {c['label']}" for c in codes["rectify"])
    method_list = "\n".join(f"  - {c['code']}: {c['label']}" for c in codes["method"])

    req_classes = schema["new_fault"]["fields"]["req_class"]["values"]
    priorities = schema["new_fault"]["fields"]["priority"]["values"]

    return f"""You are a conversational voice agent for a water utility field service (WFM) system.
You help field technicians close out completed jobs and report new faults via natural conversation.

=== YOUR ROLE ===
You are the front-end voice interface. After collecting enough info from the technician,
you will hand off to the Orchestrator which routes to the Structuring Agent (GEMI) for code mapping.

=== WORKFLOW ===
1. Greet the tech, ask what they need (close-out or new fault)
2. Collect the relevant details through conversation
3. For CLOSE-OUT: ask about the problem, cause, what they did to fix it, and how
4. For NEW FAULT: ask about location, severity, what they observed
5. When you have enough info, produce a FINAL TRANSCRIPT summary that will be sent to the orchestrator

=== DOMAIN KNOWLEDGE ===

Water utility maintenance. Technicians fix:
- Water mains (100-300mm DICL/CICL pipes)
- Service pipes (20-50mm poly/copper)
- Meter frames, hydrants, valves, main taps

CLOSE-OUT needs 4 codes (PCRM):
- PROBLEM: {problem_list}
- CAUSE: {cause_list}
- RECTIFY: {rectify_list}
- METHOD: {method_list}

NEW FAULT needs: REQ_CLASS ({', '.join(req_classes[:5])}...), PRIORITY ({', '.join(priorities)}), SEVERITY, location

=== KEY DISAMBIGUATION QUESTIONS ===
- "Did you repair the existing pipe or replace it with new?" (R_RPR vs R_RPL)
- "How did you fix it — clamp, new section, or reseal?" (M_CLA vs M_002 vs M_LID)
- "Was it natural corrosion or third-party damage?" (C_011 vs C_VDL)
- "Is water gushing/flooding (P1) or a steady leak (P2) or just seeping (P3)?"

=== CONVERSATION RULES ===
- Keep responses SHORT (1-2 sentences)
- Ask ONE question at a time
- Use plain language, not code names
- If the tech mentions a job/request number, note it
- When you have enough info, say "Let me process that through the system" and include:

```json
{{"ready": true, "transcript_summary": "<complete summary of what happened>", "request_id": <number or null>}}
```

The transcript_summary should be a natural language description including:
- What the problem was
- What caused it
- What they did to fix it
- How they fixed it (method/tools used)
- Any job/request numbers mentioned"""


# ─── Utility functions ───────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using local Whisper."""
    import wandb

    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        result = model.transcribe(
            f.name,
            language="en",
            initial_prompt="Water maintenance field report. Technical terms: main tap, service pipe, DICL, CICL, poly, copper, clamp, repack, break, leak, meter frame, hydrant, footpath, nature strip, gland, resealed.",
        )

    transcript = result["text"].strip()
    run = get_wandb_run()
    wandb.log({
        "voice/transcription_length": len(transcript),
        "voice/segments": len(result.get("segments", [])),
    })
    return transcript


def get_agent_response(messages: list[dict]) -> str:
    """Conversational agent response via W&B inference API (traced by Weave)."""
    import wandb
    from config import MODEL, get_client

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )

    tokens = response.usage.total_tokens if response.usage else 0
    run = get_wandb_run()
    wandb.log({
        "conversation/turn": len(messages) - 1,
        "conversation/tokens_this_turn": tokens,
    })

    return response.choices[0].message.content


def extract_ready_signal(response: str) -> dict | None:
    """Check if agent says it's ready to send to orchestrator."""
    if "```json" in response:
        try:
            json_str = response.split("```json")[1].split("```")[0].strip()
            data = json.loads(json_str)
            if data.get("ready"):
                return data
        except (json.JSONDecodeError, IndexError):
            pass
    return None


def text_to_speech(text: str) -> bytes | None:
    """TTS using macOS `say` when available; gracefully degrades on Linux/Windows."""
    import subprocess
    import shutil

    if not shutil.which("say"):
        return None

    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        try:
            subprocess.run(["say", "-o", f.name, text], check=True, capture_output=True)
            with open(f.name, "rb") as audio_file:
                return audio_file.read()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


def run_orchestrator(transcript: str, request_id: int | None = None) -> dict:
    """Run the full LangGraph orchestrator pipeline.

    This is the key integration: Voice → Orchestrator → GEMI → WFM.
    Every node is @weave.op() traced, visible in the Weave dashboard.
    """
    import wandb

    graph = build_orchestrator_graph()

    initial_state = {
        "messages": [],
        "transcript": transcript,
        "intent": "",
        "request_id": request_id,
        "structured_payload": {},
        "clarification_count": 0,
        "clarification_response": "",
        "validation_result": {},
        "final_result": {},
        "trace": [],
        "status": "started",
    }

    # Run the graph — this triggers:
    # classify_node → search_request_node → structure_node → guardrail_node → validate_and_post_node → confirm_node
    # All are @weave.op() decorated = full trace tree in Weave
    result = graph.invoke(initial_state)

    # Log orchestration result to W&B
    run = get_wandb_run()
    wandb.log({
        "orchestrator/intent": result.get("intent", "unknown"),
        "orchestrator/status": result.get("status", "unknown"),
        "orchestrator/nodes_executed": len(result.get("trace", [])),
        "orchestrator/request_id": result.get("request_id"),
    })

    return result


def get_code_label(code: str) -> str:
    codes = load_codes()
    for category in codes.values():
        for item in category:
            if item["code"] == code:
                return item["label"]
    return code or "?"


# ─── Main app ───────────────────────────────────────────────────────────────

def main():
    entity, project, full_project = init_services()
    _ = get_wandb_run()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.system_prompt = get_domain_system_prompt()
        st.session_state.turn_count = 0
        st.session_state.orchestrator_result = None
        st.session_state.pipeline_running = False

    # Header
    col_title, col_links = st.columns([3, 1])
    with col_title:
        st.title("🔧 WFM Voice Agent + Orchestrator")
        st.caption("Voice → Orchestrator → GEMI Structuring → WFM System | All traced in Weave")
    with col_links:
        st.markdown(f"[📊 W&B](https://wandb.ai/{entity}/{project})")
        st.markdown(f"[🔍 Weave](https://wandb.ai/{entity}/{project}/weave)")

    st.divider()

    # Layout
    col_chat, col_pipeline = st.columns([2, 1.3])

    # ─── Right column: Pipeline status ──────────────────────────────────────
    with col_pipeline:
        st.markdown("### 🔀 Pipeline")

        # Architecture diagram
        st.markdown("""
        ```
        🎤 Voice Agent (Whisper)
              ↓ transcript
        🤖 Orchestrator (LangGraph)
              ↓ classify intent
        🔍 HelpDesk Agent (search WFM)
              ↓ request context
        🧠 GEMI Structuring Agent
              ↓ PCRM codes
        🛡️ Guardrail Check
              ↓ validated
        📤 WFM System (MCP)
              ↓ posted
        ✅ Confirmation
        ```
        """)

        st.divider()

        # Show orchestrator result if available
        if st.session_state.orchestrator_result:
            result = st.session_state.orchestrator_result
            st.success("Pipeline complete!")

            # Show trace
            with st.expander("📜 Execution Trace", expanded=True):
                for step in result.get("trace", []):
                    if "ERROR" in step or "FAILED" in step:
                        st.error(step)
                    elif "Low confidence" in step or "Clarification" in step:
                        st.warning(step)
                    else:
                        st.caption(f"→ {step}")

            # Show structured codes
            payload = result.get("structured_payload", {})
            if payload:
                st.markdown("#### 🏷️ Codes (from GEMI)")
                for field in ["problem_code", "cause_code", "rectify_code", "method_code"]:
                    val = payload.get(field, "?")
                    conf = payload.get("field_confidence", {}).get(field, 0)
                    label = get_code_label(val)
                    icon = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.6 else "🔴"
                    st.markdown(f"{icon} **{field}**: `{val}` — {label} ({conf:.0%})")

            # Show final result
            final = result.get("final_result", {})
            if final:
                st.markdown("#### 📤 WFM Result")
                if final.get("new_status") == "COMPLETE":
                    st.success(f"Request #{final.get('request_id')} → COMPLETE")
                elif final.get("status") == "error":
                    st.error(f"Error: {final.get('reason') or final.get('errors')}")
                else:
                    st.info(f"Status: {final.get('status')}")

            st.markdown(f"[🔗 View full trace in Weave →](https://wandb.ai/{entity}/{project}/weave)")

        else:
            st.info("Waiting for conversation to complete...")
            st.caption(f"Turn {st.session_state.turn_count}")

        st.divider()
        st.markdown("### 🎛️ Controls")
        input_mode = st.radio("Input:", ["🎤 Voice", "⌨️ Text"], index=0)
        auto_speak = st.checkbox("🔊 Speak responses", value=True)

        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.turn_count = 0
            st.session_state.orchestrator_result = None
            st.session_state.pipeline_running = False
            st.rerun()

    # ─── Left column: Chat ──────────────────────────────────────────────────
    with col_chat:
        st.markdown("### 💬 Conversation")

        chat_container = st.container(height=480)
        with chat_container:
            # Initial greeting
            if not st.session_state.messages:
                with st.chat_message("assistant", avatar="🔧"):
                    greeting = "G'day! I'm your field service voice agent. Are you closing out a job, or reporting a new fault?"
                    st.markdown(greeting)
                    if auto_speak:
                        audio = text_to_speech(greeting)
                        if audio:
                            st.audio(audio, format="audio/aiff", autoplay=True)

            # Message history
            for msg in st.session_state.messages:
                avatar = "👷" if msg["role"] == "user" else "🔧"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("audio"):
                        st.audio(msg["audio"], format="audio/aiff")

        # Input
        st.divider()

        if st.session_state.orchestrator_result:
            st.success("✅ Pipeline completed. Click 'New Session' to start again.")
        elif input_mode == "🎤 Voice":
            audio_data = st.audio_input("🎤 Speak to the agent", key=f"voice_{st.session_state.turn_count}")
            if audio_data:
                with st.spinner("Transcribing with Whisper..."):
                    user_text = transcribe_audio(audio_data.getvalue())
                if user_text:
                    st.info(f"**You said:** {user_text}")
                    _process_turn(user_text, auto_speak, entity, project)
        else:
            user_text = st.chat_input("Type your message...")
            if user_text:
                _process_turn(user_text, auto_speak, entity, project)


def _process_turn(user_text: str, auto_speak: bool, entity: str, project: str):
    """Process one conversation turn. If ready, trigger the orchestrator."""
    import wandb

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.turn_count += 1

    # Get conversational agent response
    full_messages = [
        {"role": "system", "content": st.session_state.system_prompt}
    ] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.spinner("Agent thinking..."):
        response = get_agent_response(full_messages)

    # Check if agent is ready to hand off to orchestrator
    ready_signal = extract_ready_signal(response)

    # Clean response for display
    display_response = response
    if "```json" in display_response:
        display_response = display_response.split("```json")[0].strip()

    # TTS
    audio_bytes = None
    if auto_speak and display_response:
        audio_bytes = text_to_speech(display_response)

    msg_data = {"role": "assistant", "content": display_response}
    if audio_bytes:
        msg_data["audio"] = audio_bytes
    st.session_state.messages.append(msg_data)

    # If ready, run the orchestrator pipeline
    if ready_signal:
        transcript_summary = ready_signal.get("transcript_summary", user_text)
        request_id = ready_signal.get("request_id")

        # Try to extract request_id from conversation if not provided
        if not request_id:
            for msg in st.session_state.messages:
                matches = re.findall(r'\b(\d{5,7})\b', msg.get("content", ""))
                if matches:
                    request_id = int(matches[0])
                    break

        # Run the orchestrator — this is the full pipeline traced in Weave
        with st.spinner("🔀 Running orchestrator pipeline (classify → search → GEMI → guardrail → post)..."):
            orch_result = run_orchestrator(transcript_summary, request_id)

        st.session_state.orchestrator_result = orch_result

        # Build confirmation message
        final = orch_result.get("final_result", {})
        payload = orch_result.get("structured_payload", {})

        if final.get("new_status") == "COMPLETE":
            codes = final.get("codes_applied", payload)
            confirm_msg = (
                f"✅ **Done!** Request #{final.get('request_id')} is now **COMPLETE**.\n\n"
                f"Codes applied:\n"
                f"- Problem: `{codes.get('problem_code')}` ({get_code_label(codes.get('problem_code'))})\n"
                f"- Cause: `{codes.get('cause_code')}` ({get_code_label(codes.get('cause_code'))})\n"
                f"- Rectify: `{codes.get('rectify_code')}` ({get_code_label(codes.get('rectify_code'))})\n"
                f"- Method: `{codes.get('method_code')}` ({get_code_label(codes.get('method_code'))})\n\n"
                f"[View trace in Weave →](https://wandb.ai/{entity}/{project}/weave)"
            )
        elif final.get("status") == "new_fault_posted":
            confirm_msg = f"✅ New fault request created. [View in Weave →](https://wandb.ai/{entity}/{project}/weave)"
        elif final.get("status") == "error":
            confirm_msg = f"❌ Pipeline error: {final.get('reason') or final.get('errors')}"
        else:
            confirm_msg = (
                f"Pipeline completed with status: `{orch_result.get('status')}`\n\n"
                f"Structured codes: P={payload.get('problem_code')} C={payload.get('cause_code')} "
                f"R={payload.get('rectify_code')} M={payload.get('method_code')}\n\n"
                f"[View trace in Weave →](https://wandb.ai/{entity}/{project}/weave)"
            )

        st.session_state.messages.append({"role": "assistant", "content": confirm_msg})

        # Log full session to W&B
        run = get_wandb_run()
        table = wandb.Table(columns=["turn", "role", "content"])
        for i, msg in enumerate(st.session_state.messages):
            table.add_data(i, msg["role"], msg["content"][:500])
        wandb.log({
            "session/conversation": table,
            "session/total_turns": st.session_state.turn_count,
            "session/final_status": final.get("status") or final.get("new_status", "unknown"),
        })

    st.rerun()


if __name__ == "__main__":
    main()
