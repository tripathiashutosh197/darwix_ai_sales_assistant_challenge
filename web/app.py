"""
web/app.py
-----------
FastAPI web interface for the Empathy Engine.

Endpoints:
  GET  /           → Web UI
  POST /synthesize → JSON { text } → audio file + full metadata
  GET  /audio/{filename} → serve generated audio files
  GET  /health     → status check

Run:
  uvicorn web.app:app --reload --port 8000
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Empathy Engine", version="2.0.0")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import EmpathyPipeline
        _pipeline = EmpathyPipeline(output_dir=OUTPUT_DIR)
    return _pipeline


# ── Request / Response models ─────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str


class SynthesizeResponse(BaseModel):
    emotion:        str
    secondary:      str | None
    emotion_vector: dict          # full 15-dim scores
    intensity:      float
    rate_percent:   float
    pitch_st:       float
    pitch_hz:       float         # Hz in 175-225 band
    volume_db:      float
    emphasis:       str
    audio_url:      str
    latency_ms:     int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is required and cannot be empty.")
    try:
        pipeline = get_pipeline()
        result   = pipeline.run(req.text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    audio_url = f"/audio/{os.path.basename(result.audio_path)}"

    return SynthesizeResponse(
        emotion        = result.emotion,
        secondary      = result.secondary,
        emotion_vector = result.emotion_vector,
        intensity      = result.intensity,
        rate_percent   = result.rate_percent,
        pitch_st       = result.pitch_st,
        pitch_hz       = result.pitch_hz,
        volume_db      = result.volume_db,
        emphasis       = result.emphasis,
        audio_url      = audio_url,
        latency_ms     = result.latency_ms,
    )


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(path, media_type=media_type)


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}
