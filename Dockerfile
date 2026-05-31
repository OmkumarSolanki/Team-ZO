FROM python:3.10-slim

# System libs: ffmpeg (Whisper) + libportaudio2 (sounddevice import)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as non-root uid 1000 with a writable HOME
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["streamlit", "run", "webapp.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
