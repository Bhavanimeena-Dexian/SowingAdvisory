from fastapi import APIRouter
from services.tts_service import generate_speech

tts_router = APIRouter()

@tts_router.post("/text-to-speech/")
async def text_to_speech(text: str):
    """
    Converts text into speech using gTTS.
    """
    return generate_speech(text)
