# Deploy (Hugging Face Spaces — free)

Deploys the Streamlit app (`webapp.py`). The browser records audio; Whisper
transcribes server-side. Free CPU Basic (2 vCPU / 16 GB RAM) is enough.

## Files already prepared
- `requirements.txt` — includes streamlit, openai-whisper, torch, soundfile, sounddevice
- `packages.txt` — `ffmpeg` + `libportaudio2` (system libs)
- `data/wfm.sqlite` — seeded DB ships with the app (runtime writes are **not** persisted)

## Steps
1. Create a new Space → SDK: **Streamlit** → Hardware: **CPU basic (free)**.
2. Push this repo to the Space (or connect the GitHub repo).
3. In **Settings → Variables and secrets**, add:
   - `WANDB_API_KEY` — **required** (used for the LLM inference call + Weave logging)
   - `WANDB_ENTITY` — set to **your own** W&B entity (the code default points elsewhere)
   - `WANDB_PROJECT` — optional (default `agi-hackathon`)
4. Ensure the Space's `README.md` starts with this header so it runs `webapp.py`:

```yaml
---
title: Field Ops Orchestrator
sdk: streamlit
app_file: webapp.py
pinned: false
---
```

## Notes
- No server microphone exists in the cloud → `demo.py` mic mode won't run there; use the web app's in-browser recorder.
- SQLite resets on redeploy/restart. Use a persistent volume or external DB for durable close-outs.
