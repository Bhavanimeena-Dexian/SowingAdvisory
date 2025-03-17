from fastapi import APIRouter, UploadFile, File, HTTPException
from services.stt_service import convert_speech_to_text

stt_router = APIRouter()

@stt_router.post("/speech-to-text/")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Converts an uploaded audio file to text using WhisperX.
    """
    return convert_speech_to_text(audio)
