import os
import uuid
from gtts import gTTS
from fastapi import HTTPException

# Setting up directories
STATIC_AUDIO_DIR = "static"
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)

def generate_speech(text):
    """
    Converts text to speech using Google Text-to-Speech (gTTS).
    """
    unique_filename = f"tts_{uuid.uuid4().hex}.mp3"
    audio_output_path = os.path.join(STATIC_AUDIO_DIR, unique_filename)

    try:
        tts = gTTS(text=text, lang="en")
        tts.save(audio_output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech conversion failed: {e}")

    return {"audio_url": f"http://127.0.0.1:8000/static/{unique_filename}", "text": text}
