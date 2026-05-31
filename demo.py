"""Hackathon Demo — Live voice-driven field service orchestrator.

Usage:
  python demo.py                    # Full voice demo (record → classify → structure → close)
  python demo.py --text "report"    # Text mode (skip recording)
  python demo.py --duration 15      # Record for 15 seconds
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import wandb
import weave

from config import WANDB_ENTITY, WANDB_FULL_PROJECT, WANDB_PROJECT


def run_demo(text: str = None, duration: float = 15.0, audio_path: str = None):
    """Run the full voice-driven demo pipeline."""
    from agents.structuring_agent import GEMI
    from agents.voice_agent import Vera
    from orchestrator.intents import classify_intent
    from weave_eval.scorers import consistency_guard

    voice = Vera()
    structuring = GEMI(mode="close_out")

    print("=" * 70)
    print("  WATER MAINTENANCE FIELD SERVICE — VOICE ORCHESTRATOR DEMO")
    print("=" * 70)

    # Step 1: Get the technician's report
    print("\n┌─ STEP 1: VOICE INPUT ─────────────────────────────────────────────┐")
    start = time.time()

    if text:
        print(f"  [Text mode] Input: {text[:100]}...")
        voice_result = voice.predict(text=text)
    elif audio_path:
        print(f"  [File mode] Transcribing: {audio_path}")
        voice_result = voice.predict(audio_path=audio_path)
    else:
        voice_result = voice.record_until_silence(
            max_duration=duration,
            prompt="Speak your field report now (e.g., 'I just finished job at 14 Smith St, it was a leaking main tap, repacked the gland'):",
        )

    transcript = voice_result.get("transcript", "")
    print(f"\n  📝 Transcript: \"{transcript}\"")
    dur = voice_result.get('duration_seconds') or voice_result.get('duration') or '?'
    print(f"  Source: {voice_result.get('source', '?')} | Duration: {dur}s")
    print("└──────────────────────────────────────────────────────────────────┘")

    if not transcript:
        print("\n❌ No transcript received. Aborting.")
        return

    # Step 2: Classify intent
    print("\n┌─ STEP 2: INTENT CLASSIFICATION ───────────────────────────────────┐")
    intent = classify_intent(transcript)
    print(f"  🎯 Intent: {intent.value}")
    print("└──────────────────────────────────────────────────────────────────┘")

    # Step 3: Structure into codes
    print("\n┌─ STEP 3: STRUCTURING AGENT ───────────────────────────────────────┐")
    result = structuring.structure(transcript)
    print(f"  Problem:  {result.get('problem_code', '?')}")
    print(f"  Cause:    {result.get('cause_code', '?')}")
    print(f"  Rectify:  {result.get('rectify_code', '?')}")
    print(f"  Method:   {result.get('method_code', '?')}")
    print(f"\n  Confidence: {result.get('field_confidence', {})}")

    if result.get("needs_clarification"):
        clar = result["needs_clarification"]
        print(f"\n  ⚠️  Clarification needed: {clar.get('question')}")
        if not text and not audio_path:
            # Record clarification response
            clar_result = voice.ask_clarification(
                clar["question"], clar.get("options", [])
            )
            if clar_result.get("response"):
                print(f"  📝 Clarification response: \"{clar_result['response']}\"")
                # Re-run structuring with clarification
                result = structuring.structure(
                    transcript,
                    context={"clarification_answer": clar_result["response"], "previous_payload": result},
                )
                print(f"\n  [Re-structured with clarification]")
                print(f"  Problem:  {result.get('problem_code', '?')}")
                print(f"  Cause:    {result.get('cause_code', '?')}")
                print(f"  Rectify:  {result.get('rectify_code', '?')}")
                print(f"  Method:   {result.get('method_code', '?')}")
    print("└──────────────────────────────────────────────────────────────────┘")

    # Step 4: Guardrail check
    print("\n┌─ STEP 4: GUARDRAIL ───────────────────────────────────────────────┐")
    guard = consistency_guard(result)
    if guard["safe"]:
        print("  ✅ Guardrail PASSED — payload is consistent")
    else:
        print(f"  ❌ Guardrail BLOCKED: {guard.get('issue')}")
    print("└──────────────────────────────────────────────────────────────────┘")

    # Step 5: Confirmation
    elapsed = time.time() - start
    print("\n┌─ STEP 5: CONFIRMATION ────────────────────────────────────────────┐")
    confirmation = voice.confirm_result({
        "new_status": "COMPLETE",
        "request_id": "DEMO",
        "codes_applied": {
            "problem_code": result.get("problem_code"),
            "cause_code": result.get("cause_code"),
            "rectify_code": result.get("rectify_code"),
            "method_code": result.get("method_code"),
        },
    })
    print(f"\n  ⏱️  Total elapsed: {elapsed:.2f}s")
    print(f"  🔗 Tokens used: {result.get('_tokens', 0)}")
    print(f"  🧠 Model: {result.get('_model', 'unknown')}")
    print("└──────────────────────────────────────────────────────────────────┘")

    # Log to W&B
    wandb.log({
        "demo/transcript_length": len(transcript),
        "demo/intent": intent.value,
        "demo/elapsed_seconds": elapsed,
        "demo/tokens": result.get("_tokens", 0),
        "demo/guardrail_passed": guard["safe"],
        "demo/source": voice_result.get("source", "unknown"),
    })

    return {
        "transcript": transcript,
        "intent": intent.value,
        "codes": {
            "problem_code": result.get("problem_code"),
            "cause_code": result.get("cause_code"),
            "rectify_code": result.get("rectify_code"),
            "method_code": result.get("method_code"),
        },
        "confidence": result.get("field_confidence", {}),
        "guardrail": guard,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Voice-driven field service demo")
    parser.add_argument("--text", type=str, help="Use text input instead of microphone")
    parser.add_argument("--audio", type=str, help="Path to audio file to transcribe")
    parser.add_argument("--duration", type=float, default=15.0, help="Max recording duration (seconds)")
    args = parser.parse_args()

    weave.init(WANDB_FULL_PROJECT)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"demo-{int(time.time())}",
        tags=["demo", "voice"],
    )

    result = run_demo(text=args.text, duration=args.duration, audio_path=args.audio)

    run.finish()
    print("\n✅ Demo complete — results logged to W&B + Weave!")


if __name__ == "__main__":
    main()
