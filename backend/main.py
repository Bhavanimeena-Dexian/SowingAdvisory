import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Ensure 'static/tts' directory exists for serving audio files
STATIC_DIR = "static/tts"
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Sowing Advisory API with GPT-4o, STT & TTS")

# Root route for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to Sowing Advisory API!"}

# Import & Register API Routes
from routes.chat import chat_router
from routes.query import query_router
from routes.stt import stt_router
from routes.tts import tts_router  # Ensure this file exists and has tts_router

app.include_router(chat_router, prefix="/chat")
app.include_router(query_router, prefix="/query")
app.include_router(stt_router, prefix="/stt")
app.include_router(tts_router, prefix="/tts")

# Mount static files for TTS audio file serving
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
