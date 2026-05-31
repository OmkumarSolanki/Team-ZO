"""Voice Agent — speech in, speech out.

Handles STT transcription (local Whisper) and TTS read-back.
Supports live microphone recording for the hackathon demo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import weave
import whisper

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


class Vera(weave.Model):
    """Vera — Voice Agent. Handles STT (Whisper) and TTS for field technicians."""
    model_name: str = "whisper-1"
    tts_model: str = "tts-1"
    sample_rate: int = 16000

    @weave.op()
    def predict(self, audio_path: str = None, text: str = None, context: dict | None = None) -> dict:
        if audio_path:
            return self.transcribe(audio_path)
        if text:
            return {"transcript": text, "source": "text_input"}
        return {"error": "No input provided"}

    @weave.op()
    def record(self, duration: float = 10.0, prompt: str = None) -> dict:
        """Record audio from microphone and transcribe it."""
        if prompt:
            print(f"\n🎤 {prompt}")
        else:
            print(f"\n🎤 Recording for {duration}s... Speak now!")

        print("   [Press Ctrl+C to stop early]")

        try:
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            frames_recorded = len(audio) if 'audio' in dir() else 0
            if frames_recorded == 0:
                return {"error": "Recording cancelled"}

        audio = np.squeeze(audio)
        if np.max(np.abs(audio)) < 0.01:
            print("   ⚠️  Very quiet recording — may not transcribe well")

        audio_path = str(RECORDINGS_DIR / "latest_recording.wav")
        sf.write(audio_path, audio, self.sample_rate)
        print(f"   ✓ Saved {len(audio)/self.sample_rate:.1f}s of audio")

        result = self.transcribe(audio_path)
        result["audio_path"] = audio_path
        result["duration_seconds"] = len(audio) / self.sample_rate
        return result

    @weave.op()
    def record_until_silence(self, max_duration: float = 30.0, silence_threshold: float = 0.01, silence_duration: float = 2.0, prompt: str = None) -> dict:
        """Record until silence is detected (more natural for field reports)."""
        if prompt:
            print(f"\n🎤 {prompt}")
        else:
            print("\n🎤 Listening... (will stop after 2s of silence)")

        block_size = int(self.sample_rate * 0.5)
        blocks = []
        silence_blocks = 0
        silence_limit = int(silence_duration / 0.5)
        max_blocks = int(max_duration / 0.5)
        started_speaking = False

        for i in range(max_blocks):
            block = sd.rec(block_size, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            blocks.append(np.squeeze(block))

            rms = np.sqrt(np.mean(block ** 2))
            if rms > silence_threshold:
                started_speaking = True
                silence_blocks = 0
            elif started_speaking:
                silence_blocks += 1
                if silence_blocks >= silence_limit:
                    print("   ✓ Silence detected, stopping.")
                    break

        audio = np.concatenate(blocks)
        audio_path = str(RECORDINGS_DIR / "latest_recording.wav")
        sf.write(audio_path, audio, self.sample_rate)
        print(f"   ✓ Recorded {len(audio)/self.sample_rate:.1f}s of audio")

        result = self.transcribe(audio_path)
        result["audio_path"] = audio_path
        result["duration_seconds"] = len(audio) / self.sample_rate
        return result

    @weave.op()
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio to text using local Whisper model."""
        model = _get_whisper_model()
        result = model.transcribe(
            audio_path,
            language="en",
            initial_prompt="Water maintenance field report. Technical terms: main tap, service pipe, DICL, CICL, poly, copper, clamp, repack, break, leak, meter frame, hydrant, footpath, nature strip.",
        )
        return {
            "transcript": result["text"].strip(),
            "language": result.get("language", "en"),
            "duration": None,
            "source": "whisper_local",
            "segments": len(result.get("segments", [])),
        }

    @weave.op()
    def speak(self, text: str, output_path: str = None) -> dict:
        """Convert text to speech (uses macOS say as fallback)."""
        if output_path is None:
            output_path = str(RECORDINGS_DIR / "response.wav")

        # Use macOS built-in TTS (no API key needed)
        import subprocess
        subprocess.run(
            ["say", "-o", output_path, "--data-format=LEF32@22050", text],
            check=True,
        )
        return {"audio_path": output_path, "text": text}

    @weave.op()
    def ask_clarification(self, question: str, options: list[str] | None = None) -> dict:
        """Ask a follow-up question via voice (records response)."""
        prompt = f"Question: {question}"
        if options:
            prompt += "\nOptions: " + ", ".join(options)

        print(f"\n📋 Clarification needed: {question}")
        if options:
            for i, opt in enumerate(options, 1):
                print(f"   {i}. {opt}")

        print("\n🎤 Please respond (10s max)...")
        result = self.record(duration=10.0)

        return {
            "question": question,
            "options": options,
            "response": result.get("transcript", ""),
            "source": result.get("source", "unknown"),
        }

    @weave.op()
    def confirm_result(self, result: dict) -> str:
        """Generate a human-readable confirmation and speak it."""
        if result.get("new_status") == "COMPLETE":
            codes = result.get("codes_applied", {})
            text = (
                f"Job {result['request_id']} has been closed successfully. "
                f"Problem code: {codes.get('problem_code', 'N/A')}, "
                f"Cause: {codes.get('cause_code', 'N/A')}, "
                f"Rectification: {codes.get('rectify_code', 'N/A')}, "
                f"Method: {codes.get('method_code', 'N/A')}."
            )
        elif result.get("status") == "new_fault_posted":
            text = f"New fault request created: {result.get('request_id', 'pending')}."
        else:
            text = f"Request processed with status: {result.get('status', 'unknown')}"

        print(f"\n📢 {text}")
        return text
