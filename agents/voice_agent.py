"""Voice Agent — speech in, speech out.

Handles STT transcription and TTS read-back.
The orchestrator can re-invoke it for follow-up questions.
"""
from __future__ import annotations

from typing import Any

import weave
from openai import OpenAI


class VoiceAgent(weave.Model):
    model_name: str = "whisper-1"
    tts_model: str = "tts-1"

    @weave.op()
    def predict(self, audio_path: str = None, text: str = None, context: dict | None = None) -> dict:
        if audio_path:
            return self.transcribe(audio_path)
        if text:
            return {"transcript": text, "source": "text_input"}
        return {"error": "No input provided"}

    @weave.op()
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio to text using Whisper."""
        client = OpenAI()
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=self.model_name,
                file=f,
                response_format="verbose_json",
            )
        return {
            "transcript": result.text,
            "language": getattr(result, "language", "en"),
            "source": "whisper",
        }

    @weave.op()
    def speak(self, text: str, output_path: str = "response.mp3") -> dict:
        """Convert text to speech for read-back."""
        client = OpenAI()
        response = client.audio.speech.create(
            model=self.tts_model,
            voice="nova",
            input=text,
        )
        response.stream_to_file(output_path)
        return {"audio_path": output_path, "text": text}

    @weave.op()
    def ask_clarification(self, question: str, options: list[str] | None = None) -> dict:
        """Ask a follow-up question (text mode for hackathon)."""
        prompt = question
        if options:
            prompt += "\nOptions: " + ", ".join(options)
        return {
            "question": question,
            "options": options,
            "awaiting_response": True,
        }

    @weave.op()
    def confirm_result(self, result: dict) -> str:
        """Generate a human-readable confirmation of the result."""
        if result.get("new_status") == "COMPLETE":
            codes = result.get("codes_applied", {})
            return (
                f"Job {result['request_id']} has been closed. "
                f"Problem: {codes.get('problem_code', 'N/A')}, "
                f"Cause: {codes.get('cause_code', 'N/A')}, "
                f"Rectification: {codes.get('rectify_code', 'N/A')}, "
                f"Method: {codes.get('method_code', 'N/A')}."
            )
        return f"Request processed: {result.get('status', 'unknown')}"
