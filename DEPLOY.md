# Deploy (Hugging Face Spaces — free)

Deploys the Streamlit app (`webapp.py`). The browser records audio; Whisper
transcribes server-side. Free CPU Basic (2 vCPU / 16 GB RAM) is enough.

## Files already prepared
- `Dockerfile` — runs `streamlit run webapp.py` on port 7860 (HF Docker SDK)
- `requirements.txt` — includes streamlit, openai-whisper, torch, soundfile, sounddevice
- `packages.txt` — unused under Docker (system libs are installed in the Dockerfile)
- `data/wfm.sqlite` — seeded DB ships with the app (runtime writes are **not** persisted)

## Steps
1. Create a new Space → SDK: **Docker** (Blank) → Hardware: **CPU basic (free)**.
2. Push this repo to the Space (`git push hf main --force`).
3. In **Settings → Variables and secrets**, add:
   - `WANDB_API_KEY` — **required** (used for the LLM inference call + Weave logging)
   - `WANDB_ENTITY` — set to **your own** W&B entity (the code default points elsewhere)
   - `WANDB_PROJECT` — optional (default `agi-hackathon`)
4. Ensure the Space's `README.md` starts with this header:

```yaml
---
title: Field Ops Orchestrator
emoji: 🔧
sdk: docker
app_port: 7860
pinned: false
---
```

## Notes
- No server microphone exists in the cloud → `demo.py` mic mode won't run there; use the web app's in-browser recorder.
- SQLite resets on redeploy/restart. Use a persistent volume or external DB for durable close-outs.
