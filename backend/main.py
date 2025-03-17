from fastapi import FastAPI
from routes.chat import chat_router
from routes.query import query_router
from routes.stt import stt_router  # Speech-to-Text
from routes.tts import tts_router  # Text-to-Speech

# Initialize FastAPI app
app = FastAPI(title="Sowing Advisory API with GPT-4o, STT & TTS")

# Register API routes
app.include_router(chat_router, prefix="/chat")
app.include_router(query_router, prefix="/query")
app.include_router(stt_router, prefix="/stt")
app.include_router(tts_router, prefix="/tts")
